package face

import (
	"sync"
	"time"

	"surveillance-go/internal/config"
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

type Service struct {
	mu           sync.RWMutex
	status       Status
	frameCounter int64
}

func New(cfg config.Config) *Service {
	message := "disabled"
	if cfg.FaceRecognitionEnabled {
		message = "unavailable in Go build (API compatible stub)"
	}
	return &Service{
		status: Status{
			Enabled:            cfg.FaceRecognitionEnabled,
			Available:          false,
			Initializing:       false,
			Backend:            "go-face-stub",
			Message:            message,
			KnownFacesCount:    0,
			DetectEveryNFrames: cfg.FaceRecognitionDetectEveryNFrames,
			MatchThreshold:     cfg.FaceRecognitionMatchThreshold,
			MaxFaces:           cfg.FaceRecognitionMaxFaces,
			Result:             LastResult{Faces: []ResultFace{}},
		},
	}
}

func (s *Service) Init() {}

func (s *Service) ProcessFrame(_ []byte, frameSeq int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.frameCounter++
	s.status.Result.FrameIndex = s.frameCounter
	s.status.Result.BroadcastFrameSeq = frameSeq
	s.status.Result.UpdatedAt = time.Now().UnixMilli()
}

func (s *Service) SetEnabled(enabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status.Enabled = enabled
	if enabled {
		s.status.Message = "unavailable in Go build (API compatible stub)"
	} else {
		s.status.Message = "disabled"
	}
	s.status.Result = LastResult{Faces: []ResultFace{}}
}

func (s *Service) GetStatus() Status {
	s.mu.RLock()
	defer s.mu.RUnlock()
	st := s.status
	st.Result.Faces = append([]ResultFace(nil), s.status.Result.Faces...)
	return st
}

func (s *Service) Shutdown() {}
