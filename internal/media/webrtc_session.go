package media

import (
	"errors"
	"log"
	"strings"
	"sync"

	"github.com/pion/webrtc/v4"

	"surveillance-go/internal/config"
)

type WebRTCSignalRequest struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
}

type WebRTCSignalResponse struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
}

type WebRTCSession struct {
	peer          *webrtc.PeerConnection
	videoSub      *VideoTrackSubscriber
	audioSub      *AudioTrackSubscriber
	videoAttached bool
	audioAttached bool
	closeOnce     sync.Once
	mu            sync.Mutex
}

func NewWebRTCSession(webrtcConfig webrtc.Configuration, mediaPortMin uint16, mediaPortMax uint16) (*WebRTCSession, error) {
	settingEngine := webrtc.SettingEngine{}
	if err := settingEngine.SetEphemeralUDPPortRange(mediaPortMin, mediaPortMax); err != nil {
		return nil, err
	}

	api := webrtc.NewAPI(webrtc.WithSettingEngine(settingEngine))
	peer, err := api.NewPeerConnection(webrtcConfig)
	if err != nil {
		return nil, err
	}
	return &WebRTCSession{peer: peer}, nil
}

func BuildWebRTCConfiguration(cfg config.Config) webrtc.Configuration {
	iceServers := make([]webrtc.ICEServer, 0, len(cfg.WebRTCICEServers))
	for _, serverURL := range cfg.WebRTCICEServers {
		iceServer := webrtc.ICEServer{URLs: []string{serverURL}}
		if strings.HasPrefix(serverURL, "turn:") || strings.HasPrefix(serverURL, "turns:") {
			if cfg.WebRTCTurnUsername != "" {
				iceServer.Username = cfg.WebRTCTurnUsername
			}
			if cfg.WebRTCTurnPassword != "" {
				iceServer.Credential = cfg.WebRTCTurnPassword
			}
		}
		iceServers = append(iceServers, iceServer)
	}
	return webrtc.Configuration{ICEServers: iceServers}
}

func (s *WebRTCSession) Peer() *webrtc.PeerConnection {
	return s.peer
}

func (s *WebRTCSession) BindManager(manager *Manager) error {
	manager.mu.RLock()
	fps := manager.cameraSettings.FPS
	sampleRate := manager.cfg.SampleRate
	micChannels := manager.cfg.MicChannels
	manager.mu.RUnlock()

	videoSub, err := NewVideoTrackSubscriber(fps)
	if err != nil {
		log.Printf("webrtc session: failed to create video subscriber: %v", err)
		return err
	}
	audioSub, err := NewAudioTrackSubscriber(sampleRate, micChannels)
	if err != nil {
		_ = videoSub.Close()
		log.Printf("webrtc session: failed to create audio subscriber: %v", err)
		return err
	}
	log.Printf("webrtc session: subscribers created (fps=%d, sampleRate=%d, micChannels=%d)", fps, sampleRate, micChannels)

	s.videoSub = videoSub
	s.audioSub = audioSub

	videoSender, err := s.peer.AddTrack(videoSub.Track())
	if err != nil {
		_ = audioSub.Close()
		_ = videoSub.Close()
		log.Printf("webrtc session: failed to add video track: %v", err)
		return err
	}
	go drainRTCP(videoSender)

	audioSender, err := s.peer.AddTrack(audioSub.Track())
	if err != nil {
		_ = audioSub.Close()
		_ = videoSub.Close()
		log.Printf("webrtc session: failed to add audio track: %v", err)
		return err
	}
	go drainRTCP(audioSender)
	log.Printf("webrtc session: audio/video tracks added")

	manager.SubscribeVideo(videoSub)
	s.videoAttached = true
	manager.SubscribeAudio(audioSub)
	s.audioAttached = true
	log.Printf("webrtc session: manager subscriptions attached")

	s.peer.OnDataChannel(func(dc *webrtc.DataChannel) {
		switch dc.Label() {
		case "face-data":
			s.videoSub.SetFaceChannel(dc)
		case "talk-binary":
			dc.OnMessage(func(msg webrtc.DataChannelMessage) {
				if len(msg.Data) == 0 {
					return
				}
				manager.WriteTalkback(append([]byte(nil), msg.Data...))
			})
		}
	})

	s.peer.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		// Treat Disconnected as transient; closing here can prematurely end media tracks.
		if state == webrtc.PeerConnectionStateClosed || state == webrtc.PeerConnectionStateFailed {
			s.Close(manager)
		}
	})

	return nil
}

func (s *WebRTCSession) Answer(req WebRTCSignalRequest) (WebRTCSignalResponse, error) {
	if req.SDP == "" || req.Type == "" {
		return WebRTCSignalResponse{}, errors.New("missing sdp or type")
	}
	offer := webrtc.SessionDescription{Type: webrtc.NewSDPType(req.Type), SDP: req.SDP}
	if err := s.peer.SetRemoteDescription(offer); err != nil {
		return WebRTCSignalResponse{}, err
	}
	answer, err := s.peer.CreateAnswer(nil)
	if err != nil {
		return WebRTCSignalResponse{}, err
	}
	gatherComplete := webrtc.GatheringCompletePromise(s.peer)
	if err := s.peer.SetLocalDescription(answer); err != nil {
		return WebRTCSignalResponse{}, err
	}
	<-gatherComplete
	local := s.peer.LocalDescription()
	if local == nil {
		return WebRTCSignalResponse{}, errors.New("missing local description")
	}
	return WebRTCSignalResponse{SDP: local.SDP, Type: local.Type.String()}, nil
}

func (s *WebRTCSession) Close(manager *Manager) {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		videoSub := s.videoSub
		audioSub := s.audioSub
		videoAttached := s.videoAttached
		audioAttached := s.audioAttached
		peer := s.peer
		s.videoAttached = false
		s.audioAttached = false
		s.videoSub = nil
		s.audioSub = nil
		s.mu.Unlock()

		if videoAttached && videoSub != nil {
			manager.UnsubscribeVideo(videoSub)
		}
		if audioAttached && audioSub != nil {
			manager.UnsubscribeAudio(audioSub)
		}
		if videoSub != nil {
			_ = videoSub.Close()
		}
		if audioSub != nil {
			_ = audioSub.Close()
		}
		if peer != nil {
			_ = peer.Close()
		}
	})
}

func drainRTCP(sender *webrtc.RTPSender) {
	buf := make([]byte, 1500)
	for {
		if _, _, err := sender.Read(buf); err != nil {
			return
		}
	}
}
