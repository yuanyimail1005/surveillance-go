# Build stage: Debian Trixie ships OpenCV 4.10 (same as host)
FROM golang:1.25-trixie AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      libopencv-dev \
      libopencv-contrib-dev; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN go mod vendor && \
    rm -f \
      vendor/gocv.io/x/gocv/aruco.go \
      vendor/gocv.io/x/gocv/aruco.cpp \
      vendor/gocv.io/x/gocv/aruco.h \
      vendor/gocv.io/x/gocv/aruco_dictionaries.go && \
    CGO_ENABLED=1 GOOS=linux GOARCH=$(go env GOARCH) go build -mod=vendor -o /out/surveillance-go ./cmd/server

# Runtime stage: Debian Trixie for matching OpenCV runtime libs
FROM debian:trixie-slim
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      pulseaudio \
      pulseaudio-utils \
      v4l-utils \
      libopencv-dev \
      libopencv-contrib-dev; \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /out/surveillance-go /app/surveillance-go
COPY public /app/public
COPY .env.example /app/.env.example

EXPOSE 5000
CMD ["/app/surveillance-go"]

COPY --from=build /out/surveillance-go /app/surveillance-go
COPY public /app/public
COPY .env.example /app/.env.example

EXPOSE 5000
CMD ["/app/surveillance-go"]
