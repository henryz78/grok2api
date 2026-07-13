#!/bin/sh
set -eu

umask 077

mkdir -p /app/data /run/grok2api
chown grok2api:grok2api /app/data /run/grok2api

config_source="${GROK2API_CONFIG_SOURCE:-/run/grok2api/config.yaml}"

if [ -f "${config_source}" ]; then
  cp "${config_source}" /app/config.yaml
else
  rm -f /app/config.yaml
  echo "config file not mounted; using defaults with GROK2API_* environment overrides" >&2
fi

if [ -f /app/config.yaml ]; then
  chown grok2api:grok2api /app/config.yaml
  chmod 0600 /app/config.yaml
fi

exec su-exec grok2api:grok2api "$@"

