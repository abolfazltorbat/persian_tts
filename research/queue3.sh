#!/usr/bin/env bash
# queue 3 became queue4.sh (started early, 10-31 ~07:00); this wrapper only waits
# for tts-q4 so the judges chained on tts-q3 fire after the real work is done.
while systemctl --user is-active --quiet tts-q4; do sleep 30; done
echo "=== queue3 (wrapper) done $(date +%T)"
