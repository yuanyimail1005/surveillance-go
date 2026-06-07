# surveillance-go

Go implementation of the Raspberry Pi surveillance system with browser-based live video/audio, two-way talkback, camera and audio controls, snapshot capture, and video recording.

This project preserves the same UI and API/WebSocket contract as the Node implementation in `/home/eric/surveillance-node`, so the frontend behavior remains the same.

## Features

- Live MJPEG video streaming over WebSocket
- Live microphone audio streaming from server to browser
- Two-way talkback from browser microphone to server speakers
- Camera resolution, FPS, and device selection controls
- Server microphone/speaker selection
- Speaker volume control
- Snapshot and browser-side recording in the existing UI
- HTTPS/WSS transport
- Face-recognition API compatibility endpoints (`/face_status`, `/face_settings`)

## Important Note About Face Recognition

The Go version currently ships an API-compatible face-recognition stub backend (`go-face-stub`) so the UI toggles and status panel continue working.

- Endpoints are present and return valid payloads.
- The Go service does not yet perform server-side ML face detection/recognition.
- Existing UI overlays remain functional, but no recognized faces are produced by the backend.

## Prerequisites

- Linux host (Raspberry Pi OS / Debian Bookworm recommended)
- Go 1.22+
- `ffmpeg`
- `pulseaudio`, `pulseaudio-utils`
- `v4l2-ctl`
- camera device (`/dev/video*` and/or `rpicam-*` tools on Raspberry Pi)

## Configuration

Copy env template:

```bash
cd /home/eric/surveillance-go
cp .env.example .env
```

Generate a self-signed certificate if needed:

```bash
cd /home/eric/surveillance-go
chmod +x scripts/gen-cert.sh
./scripts/gen-cert.sh
```

## Run Locally

```bash
cd /home/eric/surveillance-go
go mod tidy
go run ./cmd/server
```

Open:

```text
https://<host-or-pi-ip>:5000
```

## API Summary

HTTP:

- `GET /status`
- `GET /camera_settings`
- `POST /camera_settings`
- `GET /server_audio_devices`
- `POST /server_audio_devices/select`
- `GET /speaker_volume`
- `POST /speaker_volume`
- `GET /face_status`
- `POST /face_settings`

WebSocket:

- `/video_feed`
- `/audio_feed`
- `/ws/talk`

## Docker

```bash
cd /home/eric/surveillance-go
sudo docker compose build
sudo docker compose up -d
```
