package config

import (
	"os"
	"path/filepath"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	ServerHost  string
	ServerPort  int
	SSLCertPath string
	SSLKeyPath  string

	CameraDevice string
	CameraWidth  int
	CameraHeight int
	CameraFPS    int

	SampleRate             int
	MicChannels            int
	SpeakerChannels        int
	PulseSinkName          string
	PulseCaptureSourceName string
	TalkbackPlaybackGain   float64

	VideoAllowedResolutions [][2]int
	VideoMinFPS             int
	VideoMaxFPS             int

	FaceRecognitionEnabled            bool
	FaceRecognitionKnownFacesDir      string
	FaceRecognitionDetectEveryNFrames int
	FaceRecognitionMatchThreshold     float64
	FaceRecognitionMaxFaces           int
	FaceRecognitionCascadePath        string
}

func Load() Config {
	_ = godotenv.Load()

	wd, _ := os.Getwd()
	resolve := func(value string) string {
		if filepath.IsAbs(value) {
			return value
		}
		if wd == "" {
			return value
		}
		return filepath.Clean(filepath.Join(wd, value))
	}

	cascadePath := ""
	if rawCascade := getenv("FACE_RECOGNITION_CASCADE_PATH", ""); rawCascade != "" {
		cascadePath = resolve(rawCascade)
	}

	cfg := Config{
		ServerHost:  getenv("SERVER_HOST", "0.0.0.0"),
		ServerPort:  getenvInt("SERVER_PORT", 5000),
		SSLCertPath: resolve(getenv("SSL_CERT_PATH", "./cert.pem")),
		SSLKeyPath:  resolve(getenv("SSL_KEY_PATH", "./key.pem")),

		CameraDevice: getenv("CAMERA_DEVICE", "/dev/video0"),
		CameraWidth:  getenvInt("CAMERA_WIDTH", 1920),
		CameraHeight: getenvInt("CAMERA_HEIGHT", 1080),
		CameraFPS:    getenvInt("CAMERA_FPS", 25),

		SampleRate:             getenvInt("SAMPLE_RATE", 48000),
		MicChannels:            getenvInt("MIC_CHANNELS", 1),
		SpeakerChannels:        getenvInt("SPEAKER_CHANNELS", 2),
		PulseSinkName:          getenv("PULSE_SINK_NAME", "@DEFAULT_SINK@"),
		PulseCaptureSourceName: getenv("PULSE_CAPTURE_SOURCE_NAME", "@DEFAULT_SOURCE@"),
		TalkbackPlaybackGain:   getenvFloat("TALKBACK_PLAYBACK_GAIN", 5.0),

		VideoAllowedResolutions: [][2]int{{640, 480}, {1280, 720}, {1920, 1080}, {2560, 1440}},
		VideoMinFPS:             1,
		VideoMaxFPS:             60,

		FaceRecognitionEnabled:            getenvBool("FACE_RECOGNITION_ENABLED", false),
		FaceRecognitionKnownFacesDir:      resolve(getenv("FACE_RECOGNITION_KNOWN_FACES_DIR", "./known_faces")),
		FaceRecognitionDetectEveryNFrames: getenvIntAllowUnset("FACE_RECOGNITION_DETECT_EVERY_N_FRAMES", 0),
		FaceRecognitionMatchThreshold:     getenvFloat("FACE_RECOGNITION_MATCH_THRESHOLD", 0.6),
		FaceRecognitionMaxFaces:           getenvInt("FACE_RECOGNITION_MAX_FACES", 8),
		FaceRecognitionCascadePath:        cascadePath,
	}

	if cfg.FaceRecognitionDetectEveryNFrames <= 0 {
		detectEvery := cfg.CameraFPS / 2
		if detectEvery < 1 {
			detectEvery = 1
		}
		cfg.FaceRecognitionDetectEveryNFrames = detectEvery
	}

	return cfg
}

func getenv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	if v, ok := os.LookupEnv(key); ok {
		if p, err := strconv.Atoi(v); err == nil {
			return p
		}
	}
	return fallback
}

func getenvIntAllowUnset(key string, fallback int) int {
	v, ok := os.LookupEnv(key)
	if !ok || v == "" {
		return fallback
	}
	p, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return p
}

func getenvFloat(key string, fallback float64) float64 {
	if v, ok := os.LookupEnv(key); ok {
		if p, err := strconv.ParseFloat(v, 64); err == nil {
			return p
		}
	}
	return fallback
}

func getenvBool(key string, fallback bool) bool {
	if v, ok := os.LookupEnv(key); ok {
		parsed, err := strconv.ParseBool(v)
		if err == nil {
			return parsed
		}
	}
	return fallback
}
