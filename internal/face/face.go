package face

import (
	"encoding/json"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"surveillance-go/internal/config"

	"gocv.io/x/gocv"
)

type ResultFace struct {
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
	Left       int     `json:"left"`
	Top        int     `json:"top"`
	Right      int     `json:"right"`
	Bottom     int     `json:"bottom"`
}

type LastResult struct {
	UpdatedAt         int64        `json:"updated_at"`
	FrameIndex        int64        `json:"frame_index"`
	BroadcastFrameSeq int64        `json:"broadcast_frame_seq"`
	ImageWidth        int          `json:"image_width"`
	ImageHeight       int          `json:"image_height"`
	Faces             []ResultFace `json:"faces"`
}

type Status struct {
	Enabled            bool       `json:"enabled"`
	Available          bool       `json:"available"`
	Initializing       bool       `json:"initializing"`
	Backend            string     `json:"backend"`
	Message            string     `json:"message"`
	KnownFacesCount    int        `json:"known_faces_count"`
	DetectEveryNFrames int        `json:"detect_every_n_frames"`
	MatchThreshold     float64    `json:"match_threshold"`
	MaxFaces           int        `json:"max_faces"`
	Result             LastResult `json:"result"`
}

type knownFaceSample struct {
	Name      string
	Path      string
	Signature string
	Mat       gocv.Mat
}

type cacheEntry struct {
	Path      string `json:"path"`
	Signature string `json:"signature"`
}

type faceCache struct {
	Version int          `json:"version"`
	Entries []cacheEntry `json:"entries"`
}

const faceCacheVersion = 1

type detectJob struct {
	jpegBuffer []byte
	frameSeq   int64
	result     chan bool
}

type Service struct {
	mu sync.RWMutex

	cfg config.Config

	status       Status
	frameCounter int64

	processInFlight bool
	classifier      gocv.CascadeClassifier
	samples         []knownFaceSample
	knownFaceCache  map[string]knownFaceSample
	cascadePath     string
	tracks          []faceTrack
	detectQueue     chan *detectJob
	workersDone     sync.WaitGroup
}

type candidateFace struct {
	face ResultFace
	rect image.Rectangle
}

type faceTrack struct {
	face        ResultFace
	rect        image.Rectangle
	streak      int
	missedCount int
}

func New(cfg config.Config) *Service {
	message := "disabled"
	if cfg.FaceRecognitionEnabled {
		message = "not initialized"
	}
	return &Service{
		cfg: cfg,
		status: Status{
			Enabled:            cfg.FaceRecognitionEnabled,
			Available:          false,
			Initializing:       false,
			Backend:            "opencv-haar-template",
			Message:            message,
			KnownFacesCount:    0,
			DetectEveryNFrames: cfg.FaceRecognitionDetectEveryNFrames,
			MatchThreshold:     cfg.FaceRecognitionMatchThreshold,
			MaxFaces:           cfg.FaceRecognitionMaxFaces,
			Result:             LastResult{Faces: []ResultFace{}},
		},
		detectQueue: make(chan *detectJob, 4),
	}
}

func (s *Service) Init() {
	s.mu.RLock()
	enabled := s.status.Enabled
	s.mu.RUnlock()

	// Start 2 concurrent detection workers.
	numWorkers := 2
	for i := 0; i < numWorkers; i++ {
		s.workersDone.Add(1)
		go s.detectionWorker()
	}

	if !enabled {
		return
	}
	s.loadModels()
}

// detectionWorker processes frames from the detection queue.
func (s *Service) detectionWorker() {
	defer s.workersDone.Done()
	for job := range s.detectQueue {
		s.processFrameSync(job.jpegBuffer, job.frameSeq)
		job.result <- true
	}
}

// ProcessFrame queues a frame for asynchronous detection.
// Returns true if queued, false if queue is full (frame dropped).
func (s *Service) ProcessFrame(jpegBuffer []byte, frameSeq int64) bool {
	select {
	case s.detectQueue <- &detectJob{
		jpegBuffer: jpegBuffer,
		frameSeq:   frameSeq,
		result:     make(chan bool, 1),
	}:
		return true
	default:
		// Queue full, drop frame.
		return false
	}
}

// processFrameSync performs synchronous face detection on a frame.
func (s *Service) processFrameSync(jpegBuffer []byte, frameSeq int64) {
	s.mu.Lock()
	s.frameCounter++
	if !s.status.Enabled || !s.status.Available || s.status.Initializing || s.processInFlight {
		s.mu.Unlock()
		return
	}
	if s.status.DetectEveryNFrames > 1 && s.frameCounter%int64(s.status.DetectEveryNFrames) != 0 {
		s.mu.Unlock()
		return
	}

	s.processInFlight = true
	maxFaces := s.status.MaxFaces
	matchThreshold := s.status.MatchThreshold
	classifier := &s.classifier
	samples := cloneSamples(s.samples)
	s.mu.Unlock()

	defer func() {
		closeSamples(samples)
		s.mu.Lock()
		s.processInFlight = false
		s.mu.Unlock()
	}()

	img, err := gocv.IMDecode(jpegBuffer, gocv.IMReadColor)
	if err != nil || img.Empty() {
		return
	}
	defer img.Close()

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	minDim := min(gray.Cols(), gray.Rows())
	minFace := max(36, int(float64(minDim)*0.10))
	maxFace := int(float64(minDim) * 0.80)
	boxes := classifier.DetectMultiScaleWithParams(
		gray,
		1.10,
		7,
		0,
		image.Pt(minFace, minFace),
		image.Pt(maxFace, maxFace),
	)
	if maxFaces > 0 && len(boxes) > maxFaces {
		boxes = boxes[:maxFaces]
	}

	candidates := make([]candidateFace, 0, len(boxes))
	for _, box := range boxes {
		if !isPlausibleFaceRect(box, gray.Cols(), gray.Rows()) {
			continue
		}
		normalized, ok := normalizedFaceFromGray(gray, box)
		if !ok {
			continue
		}

		name := "Unknown"
		confidence := 0.0
		if len(samples) > 0 {
			bestName, bestScore := bestSampleMatch(normalized, samples)
			confidence = roundTo3(bestScore)
			if bestScore >= matchThreshold {
				name = bestName
			}
		}
		normalized.Close()

		face := ResultFace{
			Name:       name,
			Confidence: confidence,
			Left:       max(0, box.Min.X),
			Top:        max(0, box.Min.Y),
			Right:      min(img.Cols()-1, box.Max.X),
			Bottom:     min(img.Rows()-1, box.Max.Y),
		}
		candidates = append(candidates, candidateFace{face: face, rect: box})
	}

	s.mu.Lock()
	faces := s.updateTracks(candidates)
	s.status.Result.FrameIndex = s.frameCounter
	s.status.Result.BroadcastFrameSeq = frameSeq
	s.status.Result.UpdatedAt = time.Now().UnixMilli()
	s.status.Result.ImageWidth = img.Cols()
	s.status.Result.ImageHeight = img.Rows()
	s.status.Result.Faces = faces
	s.mu.Unlock()
}

func (s *Service) SetEnabled(enabled bool) {
	s.mu.Lock()
	s.status.Enabled = enabled
	if !enabled {
		s.status.Message = "disabled"
		s.status.Result = LastResult{Faces: []ResultFace{}}
		s.mu.Unlock()
		return
	}
	if s.status.Available {
		s.status.Message = fmt.Sprintf("ready (%d known face sample(s))", s.status.KnownFacesCount)
		s.mu.Unlock()
		return
	}
	s.status.Message = "initializing"
	s.status.Result = LastResult{Faces: []ResultFace{}}
	s.mu.Unlock()

	go s.loadModels()
}

func (s *Service) GetStatus() Status {
	s.mu.RLock()
	defer s.mu.RUnlock()
	st := s.status
	st.Result.Faces = append([]ResultFace(nil), st.Result.Faces...)
	if st.Result.Faces == nil {
		st.Result.Faces = []ResultFace{}
	}
	return st
}

func (s *Service) Shutdown() {
	close(s.detectQueue)
	s.workersDone.Wait()

	s.mu.Lock()
	defer s.mu.Unlock()
	_ = s.classifier.Close()
	s.classifier = gocv.NewCascadeClassifier()
	closeSamples(s.samples)
	s.samples = nil
	s.knownFaceCache = nil
	s.tracks = nil
	s.status.Available = false
	if s.status.Enabled {
		s.status.Message = "stopped"
	} else {
		s.status.Message = "disabled"
	}
}

func (s *Service) loadModels() {
	s.mu.Lock()
	if s.status.Initializing {
		s.mu.Unlock()
		return
	}
	s.status.Initializing = true
	s.status.Available = false
	s.status.Message = "loading OpenCV classifier"
	currentCache := s.knownFaceCache
	s.mu.Unlock()

	log.Printf("face: loading known faces from %s", s.cfg.FaceRecognitionKnownFacesDir)

	cascadePath := resolveCascadePath(s.cfg.FaceRecognitionCascadePath)
	if cascadePath == "" {
		s.finishInitError("could not find haarcascade_frontalface_default.xml")
		return
	}

	classifier := gocv.NewCascadeClassifier()
	if !classifier.Load(cascadePath) {
		_ = classifier.Close()
		s.finishInitError("failed to load cascade file: " + cascadePath)
		return
	}
	log.Printf("face: classifier loaded from %s", cascadePath)

	persistedCache, err := loadFaceCache(s.cfg.FaceRecognitionKnownFacesDir)
	if err != nil {
		log.Printf("face: warning - failed to load persisted cache: %v", err)
	}
	if persistedCache == nil {
		persistedCache = make(map[string]cacheEntry)
	}

	samples, cache, err := loadKnownFaceSamples(classifier, s.cfg.FaceRecognitionKnownFacesDir, currentCache, persistedCache)
	if err != nil {
		_ = classifier.Close()
		s.finishInitError(err.Error())
		return
	}
	if err := saveFaceCache(s.cfg.FaceRecognitionKnownFacesDir, samples); err != nil {
		log.Printf("face: warning - failed to save cache: %v", err)
	}
	log.Printf("face: loaded %d known face sample(s)", len(samples))

	s.mu.Lock()
	_ = s.classifier.Close()
	s.classifier = classifier
	s.samples = samples
	s.knownFaceCache = cache
	s.tracks = nil
	s.cascadePath = cascadePath
	s.status.KnownFacesCount = len(samples)
	s.status.Initializing = false
	s.status.Available = true
	if !s.status.Enabled {
		s.status.Message = "disabled"
	} else {
		s.status.Message = fmt.Sprintf("ready (%d known face sample(s))", len(samples))
	}
	s.mu.Unlock()
}

func (s *Service) finishInitError(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status.Initializing = false
	s.status.Available = false
	s.status.Message = "initialization failed: " + message
}

func loadKnownFaceSamples(classifier gocv.CascadeClassifier, knownFacesDir string, previousCache map[string]knownFaceSample, persistedCache map[string]cacheEntry) ([]knownFaceSample, map[string]knownFaceSample, error) {
	log.Printf("face: scanning known faces directory %s", knownFacesDir)
	entries, err := os.ReadDir(knownFacesDir)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("face: known faces directory does not exist: %s", knownFacesDir)
			return nil, nil, nil
		}
		return nil, nil, fmt.Errorf("failed to read known faces dir: %w", err)
	}

	persons := make([]string, 0)
	for _, entry := range entries {
		if entry.IsDir() {
			persons = append(persons, entry.Name())
		}
	}
	sort.Strings(persons)

	samples := make([]knownFaceSample, 0)
	updatedCache := make(map[string]knownFaceSample)
	for _, person := range persons {
		personDir := filepath.Join(knownFacesDir, person)
		log.Printf("face: loading known face samples for %s", person)
		images, err := os.ReadDir(personDir)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read known face person dir %s: %w", personDir, err)
		}
		sort.Slice(images, func(i, j int) bool {
			return images[i].Name() < images[j].Name()
		})
		loadedForPerson := 0
		for _, imgEntry := range images {
			if imgEntry.IsDir() {
				continue
			}
			name := strings.ToLower(imgEntry.Name())
			if !strings.HasSuffix(name, ".jpg") && !strings.HasSuffix(name, ".jpeg") && !strings.HasSuffix(name, ".png") {
				continue
			}
			imgPath := filepath.Join(personDir, imgEntry.Name())
			info, err := imgEntry.Info()
			if err != nil {
				return nil, nil, fmt.Errorf("failed to stat known face image %s: %w", imgPath, err)
			}
			signature := fmt.Sprintf("%d:%d", info.Size(), info.ModTime().UnixNano())
			if previousCache != nil {
				if cached, ok := previousCache[imgPath]; ok && cached.Signature == signature {
					log.Printf("face: cache hit for %s", imgPath)
					samples = append(samples, cached)
					updatedCache[imgPath] = cached
					loadedForPerson++
					continue
				}
			}
			if persistedCache != nil {
				if cached, ok := persistedCache[imgPath]; ok && cached.Signature == signature {
					log.Printf("face: persisted cache hit (skipping reload) for %s", imgPath)
					continue
				}
			}
			log.Printf("face: reading known face image %s", imgPath)
			img := gocv.IMRead(imgPath, gocv.IMReadColor)
			if img.Empty() {
				log.Printf("face: skipping unreadable image %s", imgPath)
				img.Close()
				continue
			}

			gray := gocv.NewMat()
			gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
			img.Close()

			faces := classifier.DetectMultiScale(gray)
			if len(faces) == 0 {
				gray.Close()
				continue
			}

			sort.Slice(faces, func(i, j int) bool {
				ai := faces[i].Dx() * faces[i].Dy()
				aJ := faces[j].Dx() * faces[j].Dy()
				return ai > aJ
			})

			faceMat, ok := normalizedFaceFromGray(gray, faces[0])
			gray.Close()
			if !ok {
				log.Printf("face: no usable face found in %s", imgPath)
				continue
			}

			sample := knownFaceSample{Name: person, Path: imgPath, Signature: signature, Mat: faceMat}
			samples = append(samples, sample)
			updatedCache[imgPath] = sample
			loadedForPerson++
		}
		log.Printf("face: loaded %d sample(s) for %s", loadedForPerson, person)
	}
	if previousCache != nil {
		for path, sample := range previousCache {
			if _, ok := updatedCache[path]; !ok {
				sample.Mat.Close()
			}
		}
	}
	log.Printf("face: finished loading known faces, total samples=%d", len(samples))
	return samples, updatedCache, nil
}

func normalizedFaceFromGray(gray gocv.Mat, rect image.Rectangle) (gocv.Mat, bool) {
	if gray.Empty() {
		return gocv.NewMat(), false
	}
	bounds := image.Rect(0, 0, gray.Cols(), gray.Rows())
	box := rect.Intersect(bounds)
	if box.Empty() {
		return gocv.NewMat(), false
	}

	region := gray.Region(box)
	defer region.Close()

	resized := gocv.NewMat()
	gocv.Resize(region, &resized, image.Pt(128, 128), 0, 0, gocv.InterpolationLinear)
	if resized.Empty() {
		resized.Close()
		return gocv.NewMat(), false
	}

	equalized := gocv.NewMat()
	gocv.EqualizeHist(resized, &equalized)
	resized.Close()
	if equalized.Empty() {
		equalized.Close()
		return gocv.NewMat(), false
	}
	return equalized, true
}

func bestSampleMatch(face gocv.Mat, samples []knownFaceSample) (string, float64) {
	bestName := "Unknown"
	bestScore := 0.0
	for _, sample := range samples {
		distance := gocv.NormWithMats(face, sample.Mat, gocv.NormL2)
		score := l2DistanceToConfidence(distance, face.Rows(), face.Cols())
		if score > bestScore {
			bestScore = score
			bestName = sample.Name
		}
	}
	return bestName, bestScore
}

func cloneSamples(in []knownFaceSample) []knownFaceSample {
	if len(in) == 0 {
		return nil
	}
	out := make([]knownFaceSample, 0, len(in))
	for _, sample := range in {
		out = append(out, knownFaceSample{Name: sample.Name, Path: sample.Path, Signature: sample.Signature, Mat: sample.Mat.Clone()})
	}
	return out
}

func closeSamples(samples []knownFaceSample) {
	for _, sample := range samples {
		sample.Mat.Close()
	}
}

func isPlausibleFaceRect(rect image.Rectangle, frameW, frameH int) bool {
	w := rect.Dx()
	h := rect.Dy()
	if w <= 0 || h <= 0 {
		return false
	}
	area := w * h
	if area < 1400 {
		return false
	}
	aspect := float64(w) / float64(h)
	if aspect < 0.70 || aspect > 1.45 {
		return false
	}
	borderMarginX := int(float64(frameW) * 0.01)
	borderMarginY := int(float64(frameH) * 0.01)
	if rect.Min.X <= borderMarginX || rect.Min.Y <= borderMarginY || rect.Max.X >= frameW-borderMarginX || rect.Max.Y >= frameH-borderMarginY {
		return false
	}
	return true
}

func (s *Service) updateTracks(candidates []candidateFace) []ResultFace {
	for i := range s.tracks {
		s.tracks[i].missedCount++
	}

	usedTrack := make([]bool, len(s.tracks))
	for _, cand := range candidates {
		bestIdx := -1
		bestScore := -1.0
		for i := range s.tracks {
			if usedTrack[i] {
				continue
			}
			iou := rectIOU(cand.rect, s.tracks[i].rect)
			if iou <= 0.10 {
				continue
			}
			if iou > bestScore {
				bestScore = iou
				bestIdx = i
			}
		}

		if bestIdx >= 0 {
			usedTrack[bestIdx] = true
			s.tracks[bestIdx].rect = cand.rect
			s.tracks[bestIdx].face = cand.face
			s.tracks[bestIdx].streak++
			s.tracks[bestIdx].missedCount = 0
			continue
		}

		s.tracks = append(s.tracks, faceTrack{
			face:        cand.face,
			rect:        cand.rect,
			streak:      1,
			missedCount: 0,
		})
		usedTrack = append(usedTrack, true)
	}

	liveTracks := make([]faceTrack, 0, len(s.tracks))
	for _, t := range s.tracks {
		if t.missedCount <= 2 {
			liveTracks = append(liveTracks, t)
		}
	}
	s.tracks = liveTracks

	stable := make([]ResultFace, 0, len(s.tracks))
	for _, t := range s.tracks {
		if t.missedCount == 0 && t.streak >= s.cfg.FaceRecognitionMinConsecutiveFrames {
			stable = append(stable, t.face)
		}
	}
	return stable
}

func rectIOU(a, b image.Rectangle) float64 {
	intersection := a.Intersect(b)
	if intersection.Empty() {
		return 0
	}
	interArea := intersection.Dx() * intersection.Dy()
	unionArea := (a.Dx() * a.Dy()) + (b.Dx() * b.Dy()) - interArea
	if unionArea <= 0 {
		return 0
	}
	return float64(interArea) / float64(unionArea)
}

func resolveCascadePath(overridePath string) string {
	candidate := strings.TrimSpace(overridePath)
	if candidate != "" {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}

	candidates := []string{
		"/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
		"/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
		"/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
		"/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
	}
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

func l2DistanceToConfidence(distance float64, rows, cols int) float64 {
	if rows <= 0 || cols <= 0 {
		return 0
	}
	maxDistance := float64(max(rows, cols) * 255)
	if maxDistance <= 0 {
		return 0
	}
	normalizedDistance := distance / maxDistance
	score := 1.0 - normalizedDistance
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	return score
}

func roundTo3(v float64) float64 {
	return math.Round(v*1000) / 1000
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func saveFaceCache(knownFacesDir string, samples []knownFaceSample) error {
	cacheFile := filepath.Join(knownFacesDir, ".face_sample_cache.json")
	entries := make([]cacheEntry, 0, len(samples))
	for _, sample := range samples {
		entries = append(entries, cacheEntry{
			Path:      sample.Path,
			Signature: sample.Signature,
		})
	}
	cache := faceCache{
		Version: faceCacheVersion,
		Entries: entries,
	}
	data, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal cache: %w", err)
	}
	if err := os.WriteFile(cacheFile, data, 0o644); err != nil {
		return fmt.Errorf("failed to write cache file: %w", err)
	}
	log.Printf("face: saved cache with %d entries to %s", len(entries), cacheFile)
	return nil
}

func loadFaceCache(knownFacesDir string) (map[string]cacheEntry, error) {
	cacheFile := filepath.Join(knownFacesDir, ".face_sample_cache.json")
	data, err := os.ReadFile(cacheFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read cache file: %w", err)
	}
	var cache faceCache
	if err := json.Unmarshal(data, &cache); err != nil {
		return nil, fmt.Errorf("failed to unmarshal cache: %w", err)
	}
	if cache.Version != faceCacheVersion {
		log.Printf("face: cache version mismatch (got %d, expected %d); will regenerate", cache.Version, faceCacheVersion)
		return nil, nil
	}
	entries := make(map[string]cacheEntry)
	for _, entry := range cache.Entries {
		entries[entry.Path] = entry
	}
	log.Printf("face: loaded cache with %d entries from %s", len(entries), cacheFile)
	return entries, nil
}
