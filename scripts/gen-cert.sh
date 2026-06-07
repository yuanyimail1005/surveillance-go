#!/usr/bin/env sh
set -eu

openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem \
  -out cert.pem \
  -days 3650 \
  -nodes \
  -subj '/CN=localhost'

echo 'Generated cert.pem and key.pem'
