package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/gorilla/websocket"

	"surveillance-go/internal/config"
	"surveillance-go/internal/face"
	"surveillance-go/internal/media"
)

type server struct {
	media    *media.Manager
	face     *face.Service
	upgrader websocket.Upgrader
}

func main() {
	cfg := config.Load()
	if _, err := os.Stat(cfg.SSLCertPath); err != nil {
		log.Fatalf("missing SSL cert: %s", cfg.SSLCertPath)
	}
	if _, err := os.Stat(cfg.SSLKeyPath); err != nil {
		log.Fatalf("missing SSL key: %s", cfg.SSLKeyPath)
	}

	mediaMgr := media.New(cfg)
	faceSvc := face.New(cfg)
	faceSvc.Init()

	mediaMgr.SetVideoFrameHook(func(buf []byte, seq int64) {
		if mediaMgr == nil {
			return
		}
		if len(buf) == 0 {
			return
		}
		faceSvc.ProcessFrame(buf, seq)
		st := faceSvc.GetStatus()
		mediaMgr.BroadcastVideoJSON(map[string]any{
			"type":                  "face_data",
			"enabled":               st.Enabled,
			"available":             st.Available,
			"initializing":          st.Initializing,
			"backend":               st.Backend,
			"message":               st.Message,
			"known_faces_count":     st.KnownFacesCount,
			"detect_every_n_frames": st.DetectEveryNFrames,
			"match_threshold":       st.MatchThreshold,
			"max_faces":             st.MaxFaces,
			"result":                st.Result,
		})
	})

	s := &server{
		media:    mediaMgr,
		face:     faceSvc,
		upgrader: websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/status", s.handleStatus)
	mux.HandleFunc("/camera_settings", s.handleCameraSettings)
	mux.HandleFunc("/server_audio_devices", s.handleServerAudioDevices)
	mux.HandleFunc("/server_audio_devices/select", s.handleServerAudioSelect)
	mux.HandleFunc("/speaker_volume", s.handleSpeakerVolume)
	mux.HandleFunc("/face_status", s.handleFaceStatus)
	mux.HandleFunc("/face_settings", s.handleFaceSettings)
	mux.HandleFunc("/video_feed", s.handleVideoWS)
	mux.HandleFunc("/audio_feed", s.handleAudioWS)
	mux.HandleFunc("/ws/talk", s.handleTalkWS)

	publicDir := filepath.Join(mustWd(), "public")
	mux.Handle("/", http.FileServer(http.Dir(publicDir)))

	httpServer := &http.Server{
		Addr:      fmt.Sprintf("%s:%d", cfg.ServerHost, cfg.ServerPort),
		Handler:   logMiddleware(mux),
		TLSConfig: &tls.Config{MinVersion: tls.VersionTLS12},
	}

	go func() {
		log.Printf("Server: https://%s:%d", cfg.ServerHost, cfg.ServerPort)
		if err := httpServer.ListenAndServeTLS(cfg.SSLCertPath, cfg.SSLKeyPath); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server failed: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(ctx)
	mediaMgr.Shutdown()
	faceSvc.Shutdown()
}

func (s *server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, s.media.GetStatus())
}

func (s *server) handleCameraSettings(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, s.media.GetCameraSettingsPayload())
	case http.MethodPost:
		var req media.CameraSettingsRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		payload, code, err := s.media.ApplyCameraSettings(req)
		if err != nil {
			writeError(w, code, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, payload)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *server) handleServerAudioDevices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	status := map[string]any{
		"status":      "ok",
		"microphones": s.media.ListCaptureDevices(),
		"speakers":    s.media.ListOutputSinks(),
	}
	selectedMic, selectedSpeaker := s.media.GetSelectedAudioDevices()
	status["selected_microphone"] = selectedMic
	status["selected_speaker"] = selectedSpeaker
	writeJSON(w, http.StatusOK, status)
}

func (s *server) handleServerAudioSelect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var body struct {
		Microphone string `json:"microphone"`
		Speaker    string `json:"speaker"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	writeJSON(w, http.StatusOK, s.media.SelectServerAudioDevices(body.Microphone, body.Speaker))
}

func (s *server) handleSpeakerVolume(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		volume, ok := s.media.GetSpeakerVolume()
		if !ok {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"status": "error", "message": "Could not read speaker volume via pactl", "available": false})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "available": true, "volume": volume, "control": "pactl"})
	case http.MethodPost:
		var body struct {
			Volume int `json:"volume"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		if body.Volume < 0 {
			body.Volume = 0
		}
		if body.Volume > 100 {
			body.Volume = 100
		}
		if !s.media.SetSpeakerVolume(body.Volume) {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"status": "error", "message": "Could not set speaker volume via pactl"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "volume": body.Volume, "control": "pactl"})
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *server) handleFaceStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, s.face.GetStatus())
}

func (s *server) handleFaceSettings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var body struct {
		Enabled *bool `json:"enabled"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if body.Enabled == nil {
		writeError(w, http.StatusBadRequest, "Provide \"enabled\" (boolean) in request body")
		return
	}
	s.face.SetEnabled(*body.Enabled)
	resp := s.face.GetStatus()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":                "ok",
		"enabled":               resp.Enabled,
		"available":             resp.Available,
		"initializing":          resp.Initializing,
		"backend":               resp.Backend,
		"message":               resp.Message,
		"known_faces_count":     resp.KnownFacesCount,
		"detect_every_n_frames": resp.DetectEveryNFrames,
		"match_threshold":       resp.MatchThreshold,
		"max_faces":             resp.MaxFaces,
		"result":                resp.Result,
	})
}

func (s *server) handleVideoWS(w http.ResponseWriter, r *http.Request) {
	s.handleSubWS(w, r, true)
}

func (s *server) handleAudioWS(w http.ResponseWriter, r *http.Request) {
	s.handleSubWS(w, r, false)
}

func (s *server) handleSubWS(w http.ResponseWriter, r *http.Request, video bool) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	client := media.NewWSClient(conn)
	if video {
		s.media.SubscribeVideo(client)
	} else {
		s.media.SubscribeAudio(client)
	}
	defer func() {
		if video {
			s.media.UnsubscribeVideo(client)
		} else {
			s.media.UnsubscribeAudio(client)
		}
		_ = client.Close()
	}()
	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
	}
}

func (s *server) handleTalkWS(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()
	for {
		msgType, payload, err := conn.ReadMessage()
		if err != nil {
			return
		}
		if msgType != websocket.BinaryMessage {
			continue
		}
		s.media.WriteTalkback(payload)
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"status": "error", "message": msg})
}

func writeJSON(w http.ResponseWriter, code int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(payload)
}

func mustWd() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return wd
}

func logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/video_feed") || strings.HasPrefix(r.URL.Path, "/audio_feed") || strings.HasPrefix(r.URL.Path, "/ws/talk") {
			next.ServeHTTP(w, r)
			return
		}
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond).String())
	})
}
