FROM golang:1.22-bookworm AS build

WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=$(go env GOARCH) go build -o /out/surveillance-go ./cmd/server

FROM debian:bookworm-slim
WORKDIR /app

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      ffmpeg \
      pulseaudio \
      pulseaudio-utils \
      v4l-utils; \
    arch="$(dpkg --print-architecture)"; \
    if [ "$arch" = "arm64" ] || [ "$arch" = "armhf" ]; then \
      mkdir -p /etc/apt/keyrings; \
      curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/raspberrypi-archive-keyring.gpg; \
      echo "deb [signed-by=/etc/apt/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.com/debian/ bookworm main" \
        > /etc/apt/sources.list.d/raspi.list; \
      apt-get update; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rpicam-apps; \
    fi; \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /out/surveillance-go /app/surveillance-go
COPY public /app/public
COPY .env.example /app/.env.example

EXPOSE 5000
CMD ["/app/surveillance-go"]
