package media

import (
	"bytes"
	"io"
	"os/exec"
	"strconv"
	"sync"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/pion/webrtc/v4/pkg/media/ivfreader"
	"github.com/pion/webrtc/v4/pkg/media/oggreader"
)

type VideoTrackSubscriber struct {
	track       *webrtc.TrackLocalStaticSample
	faceChannel *webrtc.DataChannel
	stdin       io.WriteCloser
	cmd         *exec.Cmd
	mu          sync.RWMutex
	closeOnce   sync.Once
	done        chan struct{}
}

func NewVideoTrackSubscriber(fps int) (*VideoTrackSubscriber, error) {
	if fps <= 0 {
		fps = 25
	}
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8, ClockRate: 90000},
		"video",
		"surveillance-video",
	)
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(
		"ffmpeg",
		"-loglevel", "error",
		"-fflags", "nobuffer",
		"-f", "mjpeg",
		"-framerate", strconv.Itoa(fps),
		"-i", "pipe:0",
		"-an",
		"-c:v", "libvpx",
		"-deadline", "realtime",
		"-cpu-used", "5",
		"-g", strconv.Itoa(fps),
		"-f", "ivf",
		"pipe:1",
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, err
	}

	subscriber := &VideoTrackSubscriber{
		track:     track,
		stdin:     stdin,
		cmd:       cmd,
		done:      make(chan struct{}),
	}
	go subscriber.pipeIVF(stdout, fps)
	return subscriber, nil
}

func (s *VideoTrackSubscriber) Track() *webrtc.TrackLocalStaticSample {
	return s.track
}

func (s *VideoTrackSubscriber) SetFaceChannel(dc *webrtc.DataChannel) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.faceChannel = dc
}

func (s *VideoTrackSubscriber) WriteBinary(payload []byte) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.stdin == nil {
		return nil
	}
	_, err := s.stdin.Write(payload)
	return err
}

func (s *VideoTrackSubscriber) WriteText(payload []byte) error {
	if !bytes.Contains(payload, []byte(`"type"`)) {
		return nil
	}
	// Forward both frame metadata and face data through the data channel
	if !bytes.Contains(payload, []byte(`"type":"face_data"`)) && !bytes.Contains(payload, []byte(`"type":"frame_meta"`)) {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.faceChannel == nil || s.faceChannel.ReadyState() != webrtc.DataChannelStateOpen {
		return nil
	}
	return s.faceChannel.SendText(string(payload))
}

func (s *VideoTrackSubscriber) Close() error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		stdin := s.stdin
		cmd := s.cmd
		s.stdin = nil
		s.cmd = nil
		s.faceChannel = nil
		s.mu.Unlock()
		if stdin != nil {
			_ = stdin.Close()
		}
		if cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		}
		<-s.done
	})
	return nil
}

func (s *VideoTrackSubscriber) pipeIVF(r io.Reader, fallbackFPS int) {
	defer close(s.done)
	reader, header, err := ivfreader.NewWith(r)
	if err != nil {
		return
	}
	defaultDuration := time.Second / time.Duration(maxInt(fallbackFPS, 1))
	if header.TimebaseNumerator > 0 && header.TimebaseDenominator > 0 {
		defaultDuration = time.Duration(float64(time.Second) * float64(header.TimebaseNumerator) / float64(header.TimebaseDenominator))
	}
	var lastTimestamp uint64
	for {
		frame, frameHeader, err := reader.ParseNextFrame()
		if err != nil {
			return
		}
		duration := defaultDuration
		if lastTimestamp != 0 && header.TimebaseNumerator > 0 && header.TimebaseDenominator > 0 {
			delta := frameHeader.Timestamp - lastTimestamp
			if delta > 0 {
				duration = time.Duration(float64(time.Second) * float64(delta) * float64(header.TimebaseNumerator) / float64(header.TimebaseDenominator))
			}
		}
		lastTimestamp = frameHeader.Timestamp
		if duration <= 0 {
			duration = defaultDuration
		}
		_ = s.track.WriteSample(media.Sample{Data: frame, Duration: duration})
	}
}

type AudioTrackSubscriber struct {
	track     *webrtc.TrackLocalStaticSample
	stdin     io.WriteCloser
	cmd       *exec.Cmd
	mu        sync.RWMutex
	closeOnce sync.Once
	done      chan struct{}
}

func NewAudioTrackSubscriber(sampleRate, channels int) (*AudioTrackSubscriber, error) {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if channels <= 0 {
		channels = 1
	}
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: uint16(channels)},
		"audio",
		"surveillance-audio",
	)
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(
		"ffmpeg",
		"-loglevel", "error",
		"-f", "s16le",
		"-ar", strconv.Itoa(sampleRate),
		"-ac", strconv.Itoa(channels),
		"-i", "pipe:0",
		"-vn",
		"-c:a", "libopus",
		"-application", "lowdelay",
		"-frame_duration", "20",
		"-f", "ogg",
		"pipe:1",
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, err
	}

	subscriber := &AudioTrackSubscriber{
		track: track,
		stdin: stdin,
		cmd:   cmd,
		done:  make(chan struct{}),
	}
	go subscriber.pipeOgg(stdout)
	return subscriber, nil
}

func (s *AudioTrackSubscriber) Track() *webrtc.TrackLocalStaticSample {
	return s.track
}

func (s *AudioTrackSubscriber) WriteBinary(payload []byte) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.stdin == nil {
		return nil
	}
	_, err := s.stdin.Write(payload)
	return err
}

func (s *AudioTrackSubscriber) WriteText([]byte) error {
	return nil
}

func (s *AudioTrackSubscriber) Close() error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		stdin := s.stdin
		cmd := s.cmd
		s.stdin = nil
		s.cmd = nil
		s.mu.Unlock()
		if stdin != nil {
			_ = stdin.Close()
		}
		if cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
		}
		<-s.done
	})
	return nil
}

func (s *AudioTrackSubscriber) pipeOgg(r io.Reader) {
	defer close(s.done)
	reader, _, err := oggreader.NewWith(r)
	if err != nil {
		return
	}
	const opusClockRate = 48000
	defaultDuration := 20 * time.Millisecond
	var lastGranule uint64
	for {
		payload, pageHeader, err := reader.ParseNextPage()
		if err != nil {
			return
		}
		duration := defaultDuration
		if lastGranule != 0 && pageHeader.GranulePosition > lastGranule {
			delta := pageHeader.GranulePosition - lastGranule
			duration = time.Duration(float64(time.Second) * float64(delta) / opusClockRate)
		}
		lastGranule = pageHeader.GranulePosition
		if duration <= 0 {
			duration = defaultDuration
		}
		_ = s.track.WriteSample(media.Sample{Data: payload, Duration: duration})
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}