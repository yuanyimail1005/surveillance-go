package face

import (
	"fmt"
	"image"
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
	Name string
	Mat  gocv.Mat
}

type Service struct {
	mu sync.RWMutex

	cfg config.Config

	status       Status
	frameCounter int64

	processInFlight bool
	classifier      gocv.CascadeClassifier
	samples         []knownFaceSample
	cascadePath     string
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
	}
}

func (s *Service) Init() {
	s.mu.RLock()
	enabled := s.status.Enabled
	s.mu.RUnlock()
	if !enabled {
		return
	}
	s.loadModels()
}

func (s *Service) ProcessFrame(jpegBuffer []byte, frameSeq int64) {
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

	boxes := classifier.DetectMultiScale(gray)
	if maxFaces > 0 && len(boxes) > maxFaces {
		boxes = boxes[:maxFaces]
	}

	faces := make([]ResultFace, 0, len(boxes))
	for _, box := range boxes {
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

		faces = append(faces, ResultFace{
			Name:       name,
			Confidence: confidence,
			Left:       max(0, box.Min.X),
			Top:        max(0, box.Min.Y),
			Right:      min(img.Cols()-1, box.Max.X),
			Bottom:     min(img.Rows()-1, box.Max.Y),
		})
	}

	s.mu.Lock()
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
	st.Result.Faces = append([]ResultFace(nil), s.status.Result.Faces...)
	return st
}

func (s *Service) Shutdown() {
	s.mu.Lock()
	defer s.mu.Unlock()
	_ = s.classifier.Close()
	s.classifier = gocv.NewCascadeClassifier()
	closeSamples(s.samples)
	s.samples = nil
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
	s.mu.Unlock()

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

	samples, err := loadKnownFaceSamples(classifier, s.cfg.FaceRecognitionKnownFacesDir)
	if err != nil {
		_ = classifier.Close()
		s.finishInitError(err.Error())
		return
	}

	s.mu.Lock()
	_ = s.classifier.Close()
	s.classifier = classifier
	closeSamples(s.samples)
	s.samples = samples
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

func loadKnownFaceSamples(classifier gocv.CascadeClassifier, knownFacesDir string) ([]knownFaceSample, error) {
	entries, err := os.ReadDir(knownFacesDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read known faces dir: %w", err)
	}

	persons := make([]string, 0)
	for _, entry := range entries {
		if entry.IsDir() {
			persons = append(persons, entry.Name())
		}
	}
	sort.Strings(persons)

	samples := make([]knownFaceSample, 0)
	for _, person := range persons {
		personDir := filepath.Join(knownFacesDir, person)
		images, _ := os.ReadDir(personDir)
		for _, imgEntry := range images {
			if imgEntry.IsDir() {
				continue
			}
			name := strings.ToLower(imgEntry.Name())
			if !strings.HasSuffix(name, ".jpg") && !strings.HasSuffix(name, ".jpeg") && !strings.HasSuffix(name, ".png") {
				continue
			}
			imgPath := filepath.Join(personDir, imgEntry.Name())
			img := gocv.IMRead(imgPath, gocv.IMReadColor)
			if img.Empty() {
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
				aj := faces[j].Dx() * faces[j].Dy()
				return ai > aj
			})

			faceMat, ok := normalizedFaceFromGray(gray, faces[0])
			gray.Close()
			if !ok {
				continue
			}

			samples = append(samples, knownFaceSample{Name: person, Mat: faceMat})
		}
	}
	return samples, nil
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
		out = append(out, knownFaceSample{Name: sample.Name, Mat: sample.Mat.Clone()})
	}
	return out
}

func closeSamples(samples []knownFaceSample) {
	for _, sample := range samples {
		sample.Mat.Close()
	}
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
