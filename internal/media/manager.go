package media

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"surveillance-go/internal/config"
)

type cameraSettings struct {
	Width  int
	Height int
	FPS    int
}

type Manager struct {
	cfg config.Config

	mu                     sync.RWMutex
	cameraSettings         cameraSettings
	cameraDevicePreference string
	activeCameraDevice     string
	pulseSinkName          string
	pulseCaptureSourceName string

	pulseStartedByApp bool
	pulseLocalEnv     map[string]string

	cameraCmd        *exec.Cmd
	audioCaptureCmd  *exec.Cmd
	audioPlaybackCmd *exec.Cmd
	audioPlaybackIn  io.WriteCloser

	videoSubscribers map[*WSClient]struct{}
	audioSubscribers map[*WSClient]struct{}

	videoBuffer    []byte
	frameSeq       int64
	videoFrameHook func([]byte, int64)

	shuttingDown bool
}

type DeviceOption struct {
	Path                 string       `json:"path"`
	Name                 string       `json:"name"`
	Supported            bool         `json:"supported"`
	Reason               string       `json:"reason"`
	SupportedResolutions []Resolution `json:"supported_resolutions"`
}

type Resolution struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

func New(cfg config.Config) *Manager {
	return &Manager{
		cfg:                    cfg,
		cameraSettings:         cameraSettings{Width: cfg.CameraWidth, Height: cfg.CameraHeight, FPS: cfg.CameraFPS},
		cameraDevicePreference: cfg.CameraDevice,
		pulseSinkName:          cfg.PulseSinkName,
		pulseCaptureSourceName: cfg.PulseCaptureSourceName,
		videoSubscribers:       make(map[*WSClient]struct{}),
		audioSubscribers:       make(map[*WSClient]struct{}),
		videoBuffer:            make([]byte, 0),
	}
}

func (m *Manager) SetVideoFrameHook(hook func([]byte, int64)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.videoFrameHook = hook
}

func (m *Manager) BroadcastVideoJSON(payload any) {
	body, err := json.Marshal(payload)
	if err != nil {
		return
	}
	m.mu.RLock()
	clients := make([]*WSClient, 0, len(m.videoSubscribers))
	for c := range m.videoSubscribers {
		clients = append(clients, c)
	}
	m.mu.RUnlock()

	for _, c := range clients {
		_ = c.WriteText(body)
	}
}

func (m *Manager) SubscribeVideo(c *WSClient) {
	m.mu.Lock()
	m.videoSubscribers[c] = struct{}{}
	startNeeded := m.cameraCmd == nil
	m.mu.Unlock()

	if startNeeded {
		_ = m.startCamera()
	}
}

func (m *Manager) UnsubscribeVideo(c *WSClient) {
	m.mu.Lock()
	delete(m.videoSubscribers, c)
	stopNeeded := len(m.videoSubscribers) == 0
	m.mu.Unlock()

	if stopNeeded {
		m.stopCamera()
	}
}

func (m *Manager) SubscribeAudio(c *WSClient) {
	m.mu.Lock()
	m.audioSubscribers[c] = struct{}{}
	startNeeded := m.audioCaptureCmd == nil
	m.mu.Unlock()

	if startNeeded {
		_ = m.startAudioCapture()
	}
}

func (m *Manager) UnsubscribeAudio(c *WSClient) {
	m.mu.Lock()
	delete(m.audioSubscribers, c)
	stopNeeded := len(m.audioSubscribers) == 0
	m.mu.Unlock()

	if stopNeeded {
		m.stopAudioCapture()
	}
}

func (m *Manager) WriteTalkback(monoChunk []byte) bool {
	m.mu.RLock()
	cmd := m.audioPlaybackCmd
	stdin := m.audioPlaybackIn
	gain := m.cfg.TalkbackPlaybackGain
	m.mu.RUnlock()

	if cmd == nil || stdin == nil {
		if err := m.startAudioPlayback(); err != nil {
			return false
		}
		m.mu.RLock()
		stdin = m.audioPlaybackIn
		m.mu.RUnlock()
		if stdin == nil {
			return false
		}
	}

	stereo := convertMonoToStereo(monoChunk, gain)
	_, err := stdin.Write(stereo)
	return err == nil
}

func (m *Manager) GetStatus() map[string]any {
	m.mu.RLock()
	defer m.mu.RUnlock()

	camDev := m.activeCameraDevice
	if camDev == "" {
		camDev = m.cameraDevicePreference
	}
	if camDev == "" {
		camDev = m.cfg.CameraDevice
	}

	return map[string]any{
		"camera":                   m.cameraCmd != nil,
		"audio":                    m.audioCaptureCmd != nil,
		"queue_size":               0,
		"camera_device":            camDev,
		"camera_source_type":       getCameraSourceType(camDev),
		"camera_device_preference": m.cameraDevicePreference,
		"camera_width":             m.cameraSettings.Width,
		"camera_height":            m.cameraSettings.Height,
		"camera_fps":               m.cameraSettings.FPS,
	}
}

func (m *Manager) GetCameraSettingsPayload() map[string]any {
	m.mu.RLock()
	selected := m.activeCameraDevice
	if selected == "" {
		selected = m.cameraDevicePreference
	}
	if selected == "" {
		selected = m.cfg.CameraDevice
	}
	width := m.cameraSettings.Width
	height := m.cameraSettings.Height
	fps := m.cameraSettings.FPS
	cameraDevice := m.activeCameraDevice
	m.mu.RUnlock()

	allowed := make([]Resolution, 0, len(m.cfg.VideoAllowedResolutions))
	for _, pair := range m.cfg.VideoAllowedResolutions {
		allowed = append(allowed, Resolution{Width: pair[0], Height: pair[1]})
	}

	return map[string]any{
		"status":                   "ok",
		"width":                    width,
		"height":                   height,
		"fps":                      fps,
		"camera_device":            cameraDevice,
		"selected_camera_device":   selected,
		"camera_source_type":       getCameraSourceType(selected),
		"available_camera_devices": m.ListCameraDeviceOptions(selected),
		"supported_resolutions":    defaultSupportedResolutions(m.cfg),
		"allowed_resolutions":      allowed,
		"fps_range":                map[string]int{"min": m.cfg.VideoMinFPS, "max": m.cfg.VideoMaxFPS},
	}
}

type CameraSettingsRequest struct {
	Width        int    `json:"width"`
	Height       int    `json:"height"`
	FPS          int    `json:"fps"`
	CameraDevice string `json:"camera_device"`
}

func (m *Manager) ApplyCameraSettings(req CameraSettingsRequest) (map[string]any, int, error) {
	if req.Width <= 0 || req.Height <= 0 || req.FPS <= 0 {
		return nil, 400, errors.New("width, height and fps must be integers")
	}

	validRes := false
	for _, p := range m.cfg.VideoAllowedResolutions {
		if p[0] == req.Width && p[1] == req.Height {
			validRes = true
			break
		}
	}
	if !validRes {
		return nil, 400, fmt.Errorf("unsupported resolution %dx%d", req.Width, req.Height)
	}
	if req.FPS < m.cfg.VideoMinFPS || req.FPS > m.cfg.VideoMaxFPS {
		return nil, 400, fmt.Errorf("fps must be between %d and %d", m.cfg.VideoMinFPS, m.cfg.VideoMaxFPS)
	}

	m.mu.Lock()
	if strings.TrimSpace(req.CameraDevice) != "" {
		m.cameraDevicePreference = strings.TrimSpace(req.CameraDevice)
	}
	m.cameraSettings = cameraSettings{Width: req.Width, Height: req.Height, FPS: req.FPS}
	m.mu.Unlock()

	if err := m.startCamera(); err != nil {
		return nil, 500, errors.New("failed to restart camera process with the requested settings")
	}
	return m.GetCameraSettingsPayload(), 200, nil
}

func (m *Manager) SelectServerAudioDevices(microphone, speaker string) map[string]any {
	m.mu.Lock()
	if strings.TrimSpace(microphone) != "" {
		m.pulseCaptureSourceName = strings.TrimSpace(microphone)
	}
	if strings.TrimSpace(speaker) != "" {
		m.pulseSinkName = strings.TrimSpace(speaker)
	}
	hasAudioSubs := len(m.audioSubscribers) > 0
	hasPlayback := m.audioPlaybackCmd != nil
	m.mu.Unlock()

	if hasAudioSubs {
		_ = m.startAudioCapture()
	} else {
		m.stopAudioCapture()
	}
	if hasPlayback {
		_ = m.startAudioPlayback()
	}

	m.mu.RLock()
	selectedMic := m.pulseCaptureSourceName
	selectedSpeaker := m.pulseSinkName
	m.mu.RUnlock()
	return map[string]any{
		"status":              "ok",
		"selected_microphone": selectedMic,
		"selected_speaker":    selectedSpeaker,
	}
}

func (m *Manager) GetSelectedAudioDevices() (string, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.pulseCaptureSourceName, m.pulseSinkName
}

func (m *Manager) Shutdown() {
	m.mu.Lock()
	m.shuttingDown = true
	m.mu.Unlock()

	m.stopCamera()
	m.stopAudioCapture()
	m.stopAudioPlayback()
	m.stopPulseAudioIfStartedByApp()
}

func (m *Manager) ListCaptureDevices() []map[string]any {
	if !m.ensurePulseAudioReady() {
		return []map[string]any{{"id": "@DEFAULT_SOURCE@", "name": "Default microphone", "kind": "default"}}
	}
	out, err := m.runPulseCmd("pactl list short sources")
	devices := []map[string]any{{"id": "@DEFAULT_SOURCE@", "name": "Default microphone", "kind": "default"}}
	if err != nil {
		return devices
	}
	for _, line := range strings.Split(out, "\n") {
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		name := parts[1]
		if strings.HasSuffix(name, ".monitor") {
			continue
		}
		devices = append(devices, map[string]any{"id": name, "name": name, "kind": "pulseaudio-source"})
	}
	return devices
}

func (m *Manager) ListOutputSinks() []map[string]any {
	if !m.ensurePulseAudioReady() {
		return []map[string]any{{"id": "@DEFAULT_SINK@", "name": "Default speaker", "kind": "default"}}
	}
	out, err := m.runPulseCmd("pactl list short sinks")
	sinks := []map[string]any{{"id": "@DEFAULT_SINK@", "name": "Default speaker", "kind": "default"}}
	if err != nil {
		return sinks
	}
	for _, line := range strings.Split(out, "\n") {
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		name := parts[1]
		sinks = append(sinks, map[string]any{"id": name, "name": name, "kind": "pulseaudio-sink"})
	}
	return sinks
}

func (m *Manager) GetSpeakerVolume() (int, bool) {
	if !m.ensurePulseAudioReady() {
		return 0, false
	}
	m.mu.RLock()
	sink := m.pulseSinkName
	if sink == "" {
		sink = "@DEFAULT_SINK@"
	}
	m.mu.RUnlock()
	out, err := m.runPulseCmd(fmt.Sprintf("pactl get-sink-volume %s", shellEscape(sink)))
	if err != nil {
		return 0, false
	}
	re := regexp.MustCompile(`(\d{1,3})%`)
	matches := re.FindStringSubmatch(out)
	if len(matches) != 2 {
		return 0, false
	}
	v, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0, false
	}
	return v, true
}

func (m *Manager) SetSpeakerVolume(volume int) bool {
	if !m.ensurePulseAudioReady() {
		return false
	}
	m.mu.RLock()
	sink := m.pulseSinkName
	if sink == "" {
		sink = "@DEFAULT_SINK@"
	}
	m.mu.RUnlock()
	_, err := m.runPulseCmd(fmt.Sprintf("pactl set-sink-volume %s %d%%", shellEscape(sink), volume))
	return err == nil
}

func (m *Manager) startCamera() error {
	m.mu.Lock()
	width := m.cameraSettings.Width
	height := m.cameraSettings.Height
	fps := m.cameraSettings.FPS
	preferred := m.cameraDevicePreference
	if preferred == "" {
		preferred = m.activeCameraDevice
	}
	if preferred == "" {
		preferred = m.cfg.CameraDevice
	}
	m.mu.Unlock()

	resolved := m.resolveCameraDevice(preferred)
	if resolved == "" {
		return errors.New("no supported camera device found")
	}

	m.stopCamera()

	var cmd *exec.Cmd
	if idx, ok := parseRpicamPath(resolved); ok {
		cmd = exec.Command("rpicam-vid",
			"--camera", idx,
			"--codec", "mjpeg",
			"--width", strconv.Itoa(width),
			"--height", strconv.Itoa(height),
			"--framerate", strconv.Itoa(fps),
			"--timeout", "0",
			"--nopreview",
			"--output", "-",
		)
	} else {
		cmd = exec.Command("ffmpeg",
			"-loglevel", "error",
			"-f", "v4l2",
			"-input_format", "mjpeg",
			"-video_size", fmt.Sprintf("%dx%d", width, height),
			"-framerate", strconv.Itoa(fps),
			"-i", resolved,
			"-vcodec", "copy",
			"-f", "mjpeg",
			"pipe:1",
		)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return err
	}

	m.mu.Lock()
	m.cameraCmd = cmd
	m.activeCameraDevice = resolved
	m.videoBuffer = make([]byte, 0)
	hook := m.videoFrameHook
	m.mu.Unlock()

	go func(localCmd *exec.Cmd, r io.Reader, frameHook func([]byte, int64)) {
		buf := make([]byte, 64*1024)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				m.handleCameraBytes(buf[:n], frameHook)
			}
			if err != nil {
				break
			}
		}
		_ = localCmd.Wait()
		m.mu.Lock()
		isCurrent := m.cameraCmd == localCmd
		hasSubs := len(m.videoSubscribers) > 0
		shuttingDown := m.shuttingDown
		if isCurrent {
			m.cameraCmd = nil
		}
		m.mu.Unlock()
		if isCurrent && !shuttingDown && hasSubs {
			time.AfterFunc(300*time.Millisecond, func() { _ = m.startCamera() })
		}
	}(cmd, stdout, hook)

	return nil
}

func (m *Manager) handleCameraBytes(chunk []byte, frameHook func([]byte, int64)) {
	m.mu.Lock()
	m.videoBuffer = append(m.videoBuffer, chunk...)
	buffer := m.videoBuffer
	latest := []byte(nil)

	for {
		start := bytes.Index(buffer, []byte{0xff, 0xd8})
		if start == -1 {
			buffer = buffer[:0]
			break
		}
		end := bytes.Index(buffer[start+2:], []byte{0xff, 0xd9})
		if end == -1 {
			buffer = append([]byte(nil), buffer[start:]...)
			break
		}
		end = start + 2 + end
		jpg := append([]byte(nil), buffer[start:end+2]...)
		buffer = buffer[end+2:]
		latest = jpg
	}
	m.videoBuffer = buffer
	if len(latest) == 0 {
		m.mu.Unlock()
		return
	}
	m.frameSeq++
	seq := m.frameSeq
	clients := make([]*WSClient, 0, len(m.videoSubscribers))
	for c := range m.videoSubscribers {
		clients = append(clients, c)
	}
	m.mu.Unlock()

	if frameHook != nil {
		frameHook(latest, seq)
	}
	m.BroadcastVideoJSON(map[string]any{"type": "frame_meta", "broadcast_frame_seq": seq})
	for _, c := range clients {
		_ = c.WriteBinary(latest)
	}
}

func (m *Manager) stopCamera() {
	m.mu.Lock()
	cmd := m.cameraCmd
	m.cameraCmd = nil
	m.videoBuffer = make([]byte, 0)
	m.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Signal(syscall.SIGTERM)
	}
}

func (m *Manager) startAudioCapture() error {
	if !m.ensurePulseAudioReady() {
		return errors.New("pulseaudio unavailable")
	}
	m.stopAudioCapture()

	m.mu.RLock()
	source := m.pulseCaptureSourceName
	if source == "" {
		source = "@DEFAULT_SOURCE@"
	}
	env := m.pulseEnvForChild()
	m.mu.RUnlock()

	cmd := exec.Command("parec",
		"--device", source,
		"--format=s16le",
		"--rate", strconv.Itoa(m.cfg.SampleRate),
		"--channels", strconv.Itoa(m.cfg.MicChannels),
		"--raw",
	)
	cmd.Env = env
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return err
	}

	m.mu.Lock()
	m.audioCaptureCmd = cmd
	m.mu.Unlock()

	go func(localCmd *exec.Cmd, r io.Reader) {
		buf := make([]byte, 64*1024)
		for {
			n, err := r.Read(buf)
			if n > 0 {
				data := append([]byte(nil), buf[:n]...)
				m.broadcastAudio(data)
			}
			if err != nil {
				break
			}
		}
		_ = localCmd.Wait()
		m.mu.Lock()
		isCurrent := m.audioCaptureCmd == localCmd
		hasSubs := len(m.audioSubscribers) > 0
		shuttingDown := m.shuttingDown
		if isCurrent {
			m.audioCaptureCmd = nil
		}
		m.mu.Unlock()
		if isCurrent && !shuttingDown && hasSubs {
			time.AfterFunc(300*time.Millisecond, func() { _ = m.startAudioCapture() })
		}
	}(cmd, stdout)

	return nil
}

func (m *Manager) broadcastAudio(chunk []byte) {
	m.mu.RLock()
	clients := make([]*WSClient, 0, len(m.audioSubscribers))
	for c := range m.audioSubscribers {
		clients = append(clients, c)
	}
	m.mu.RUnlock()

	for _, c := range clients {
		_ = c.WriteBinary(chunk)
	}
}

func (m *Manager) stopAudioCapture() {
	m.mu.Lock()
	cmd := m.audioCaptureCmd
	m.audioCaptureCmd = nil
	m.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Signal(syscall.SIGTERM)
	}
}

func (m *Manager) startAudioPlayback() error {
	if !m.ensurePulseAudioReady() {
		return errors.New("pulseaudio unavailable")
	}
	m.stopAudioPlayback()

	m.mu.RLock()
	sink := m.pulseSinkName
	if sink == "" {
		sink = "@DEFAULT_SINK@"
	}
	env := m.pulseEnvForChild()
	m.mu.RUnlock()

	cmd := exec.Command("pacat",
		"--playback",
		"--raw",
		"--format=s16le",
		"--rate", strconv.Itoa(m.cfg.SampleRate),
		"--channels", strconv.Itoa(m.cfg.SpeakerChannels),
		"--device", sink,
		"--stream-name", "surveillance-speaker",
	)
	cmd.Env = env
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return err
	}

	m.mu.Lock()
	m.audioPlaybackCmd = cmd
	m.audioPlaybackIn = stdin
	m.mu.Unlock()

	go func(localCmd *exec.Cmd, in io.WriteCloser) {
		_ = localCmd.Wait()
		_ = in.Close()
		m.mu.Lock()
		if m.audioPlaybackCmd == localCmd {
			m.audioPlaybackCmd = nil
			m.audioPlaybackIn = nil
		}
		m.mu.Unlock()
	}(cmd, stdin)

	return nil
}

func (m *Manager) stopAudioPlayback() {
	m.mu.Lock()
	cmd := m.audioPlaybackCmd
	in := m.audioPlaybackIn
	m.audioPlaybackCmd = nil
	m.audioPlaybackIn = nil
	m.mu.Unlock()
	if in != nil {
		_ = in.Close()
	}
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Signal(syscall.SIGTERM)
	}
}

func convertMonoToStereo(mono []byte, gain float64) []byte {
	samples := len(mono) / 2
	out := make([]byte, samples*4)
	for i := 0; i < samples; i++ {
		s := int16(binaryLEToInt16(mono[i*2], mono[i*2+1]))
		v := int(float64(s) * gain)
		if v > 32767 {
			v = 32767
		}
		if v < -32768 {
			v = -32768
		}
		writeInt16LE(out, i*4, int16(v))
		writeInt16LE(out, i*4+2, int16(v))
	}
	return out
}

func binaryLEToInt16(lo, hi byte) int16 {
	return int16(uint16(lo) | uint16(hi)<<8)
}

func writeInt16LE(dst []byte, offset int, v int16) {
	u := uint16(v)
	dst[offset] = byte(u & 0xff)
	dst[offset+1] = byte((u >> 8) & 0xff)
}

func (m *Manager) ensurePulseAudioReady() bool {
	if _, err := runCmd("pactl info", nil); err == nil {
		m.mu.Lock()
		m.pulseLocalEnv = nil
		m.mu.Unlock()
		return true
	}
	if !hasCmd("pulseaudio") {
		return false
	}
	localEnv := map[string]string{
		"XDG_RUNTIME_DIR": "/tmp/pulse-runtime",
		"PULSE_SERVER":    "",
		"PULSE_COOKIE":    "",
	}
	_, _ = runCmd("mkdir -p /tmp/pulse-runtime && chmod 700 /tmp/pulse-runtime", localEnv)
	_, _ = runCmd("pulseaudio --start --daemonize=true --exit-idle-time=-1", localEnv)
	if _, err := runCmd("pactl info", localEnv); err != nil {
		return false
	}
	m.mu.Lock()
	m.pulseLocalEnv = localEnv
	m.pulseStartedByApp = true
	m.mu.Unlock()
	return true
}

func (m *Manager) stopPulseAudioIfStartedByApp() {
	m.mu.RLock()
	started := m.pulseStartedByApp
	env := m.pulseLocalEnv
	m.mu.RUnlock()
	if !started {
		return
	}
	_, _ = runCmd("pulseaudio --kill", env)
	m.mu.Lock()
	m.pulseStartedByApp = false
	m.mu.Unlock()
}

func (m *Manager) pulseEnvForChild() []string {
	m.mu.RLock()
	local := m.pulseLocalEnv
	m.mu.RUnlock()
	if local == nil {
		return os.Environ()
	}
	env := make(map[string]string)
	for _, e := range os.Environ() {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) == 2 {
			env[parts[0]] = parts[1]
		}
	}
	for k, v := range local {
		env[k] = v
	}
	out := make([]string, 0, len(env))
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	return out
}

func (m *Manager) runPulseCmd(cmd string) (string, error) {
	m.mu.RLock()
	env := m.pulseLocalEnv
	m.mu.RUnlock()
	return runCmd(cmd, env)
}

func (m *Manager) resolveCameraDevice(preferred string) string {
	if idx, ok := parseRpicamPath(preferred); ok {
		if m.isSupportedRpicamCamera("rpicam://" + idx) {
			return preferred
		}
	} else if preferred != "" && m.isSupportedV4L2Camera(preferred) {
		return preferred
	}

	for _, device := range m.listVideoNodes() {
		if m.isSupportedV4L2Camera(device) {
			return device
		}
	}

	for _, csi := range m.listRpicamCameraOptions() {
		if csi.Supported {
			return csi.Path
		}
	}
	return ""
}

func (m *Manager) ListCameraDeviceOptions(selected string) []DeviceOption {
	seen := map[string]struct{}{}
	options := make([]DeviceOption, 0)

	candidates := []string{selected}
	if selected == "" {
		m.mu.RLock()
		candidates[0] = m.cameraDevicePreference
		m.mu.RUnlock()
	}
	if candidates[0] == "" {
		candidates[0] = m.cfg.CameraDevice
	}
	candidates = append(candidates, m.listVideoNodes()...)

	for _, dev := range candidates {
		if dev == "" {
			continue
		}
		if _, ok := seen[dev]; ok {
			continue
		}
		seen[dev] = struct{}{}
		if m.isSupportedV4L2Camera(dev) {
			options = append(options, DeviceOption{
				Path:                 dev,
				Name:                 dev,
				Supported:            true,
				Reason:               "video capture with mjpeg support",
				SupportedResolutions: defaultSupportedResolutions(m.cfg),
			})
		}
	}

	for _, c := range m.listRpicamCameraOptions() {
		if c.Supported {
			options = append(options, c)
		}
	}

	return options
}

func (m *Manager) listVideoNodes() []string {
	entries, err := os.ReadDir("/dev")
	if err != nil {
		return nil
	}
	out := make([]string, 0)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "video") {
			if _, err := strconv.Atoi(strings.TrimPrefix(e.Name(), "video")); err == nil {
				out = append(out, filepath.Join("/dev", e.Name()))
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func (m *Manager) isSupportedV4L2Camera(devicePath string) bool {
	if _, err := os.Stat(devicePath); err != nil {
		return false
	}
	if !hasCmd("v4l2-ctl") {
		return true
	}
	caps, err := runCmd(fmt.Sprintf("v4l2-ctl -d %s --all", shellEscape(devicePath)), nil)
	if err != nil {
		return false
	}
	low := strings.ToLower(caps)
	if strings.Contains(low, "metadata capture") && !strings.Contains(low, "video capture") {
		return false
	}
	if !strings.Contains(low, "video capture") {
		return false
	}
	formats, err := runCmd(fmt.Sprintf("v4l2-ctl -d %s --list-formats-ext", shellEscape(devicePath)), nil)
	if err != nil {
		return false
	}
	fl := strings.ToLower(formats)
	return strings.Contains(fl, "mjpg") || strings.Contains(fl, "motion-jpeg")
}

func (m *Manager) listRpicamCameraOptions() []DeviceOption {
	if !hasCmd("rpicam-hello") {
		return nil
	}
	out, err := runCmd("rpicam-hello --list-cameras", nil)
	if err != nil {
		return nil
	}
	re := regexp.MustCompile(`^(\d+)\s*:\s*(.+)$`)
	options := make([]DeviceOption, 0)
	for _, raw := range strings.Split(out, "\n") {
		line := strings.TrimSpace(raw)
		matches := re.FindStringSubmatch(line)
		if len(matches) == 3 {
			idx := matches[1]
			options = append(options, DeviceOption{
				Path:                 "rpicam://" + idx,
				Name:                 "CSI Camera " + idx + ": " + matches[2],
				Supported:            true,
				Reason:               "CSI camera via rpicam-vid mjpeg stream",
				SupportedResolutions: defaultSupportedResolutions(m.cfg),
			})
		}
	}
	return options
}

func (m *Manager) isSupportedRpicamCamera(cameraPath string) bool {
	if !hasCmd("rpicam-vid") {
		return false
	}
	_, ok := parseRpicamPath(cameraPath)
	return ok
}

func parseRpicamPath(path string) (string, bool) {
	re := regexp.MustCompile(`^rpicam://(\d+)$`)
	m := re.FindStringSubmatch(strings.TrimSpace(path))
	if len(m) != 2 {
		return "", false
	}
	return m[1], true
}

func getCameraSourceType(cameraPath string) string {
	if _, ok := parseRpicamPath(cameraPath); ok {
		return "CSI"
	}
	if strings.HasPrefix(cameraPath, "/dev/video") {
		return "V4L2"
	}
	return "Unknown"
}

func defaultSupportedResolutions(cfg config.Config) []Resolution {
	out := make([]Resolution, 0, len(cfg.VideoAllowedResolutions))
	seen := map[string]struct{}{}
	for _, pair := range cfg.VideoAllowedResolutions {
		key := fmt.Sprintf("%dx%d", pair[0], pair[1])
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, Resolution{Width: pair[0], Height: pair[1]})
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Width*out[i].Height < out[j].Width*out[j].Height
	})
	return out
}

func hasCmd(cmd string) bool {
	_, err := exec.LookPath(cmd)
	return err == nil
}

func runCmd(command string, envOverrides map[string]string) (string, error) {
	cmd := exec.Command("sh", "-lc", command)
	if envOverrides != nil {
		env := make(map[string]string)
		for _, e := range os.Environ() {
			parts := strings.SplitN(e, "=", 2)
			if len(parts) == 2 {
				env[parts[0]] = parts[1]
			}
		}
		for k, v := range envOverrides {
			env[k] = v
		}
		cmd.Env = make([]string, 0, len(env))
		for k, v := range env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), err
	}
	return string(out), nil
}

func shellEscape(v string) string {
	return "'" + strings.ReplaceAll(v, "'", "'\\''") + "'"
}
