#!/usr/bin/env bash
# Start a systemd unit for a command once another unit has finished.
#   research/chain.sh <wait-unit> <new-unit> <log> <command...>
set -u
wait_unit="$1"; new="$2"; log="$3"; shift 3
# a unit that has not started yet is "inactive" too: first wait for it to
# appear (or for its own chain to be gone), then for it to finish
while ! systemctl --user is-active --quiet "$wait_unit" && pgrep -f "chain.sh [^ ]+ $wait_unit " >/dev/null; do sleep 30; done
while systemctl --user is-active --quiet "$wait_unit"; do sleep 30; done
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
systemd-run --user --unit="$new" --collect -p MemoryMax=6G -p MemorySwapMax=1G \
  --working-directory="$PWD" -p StandardOutput=append:"$PWD/$log" -p StandardError=append:"$PWD/$log" "$@"
echo "started $new after $wait_unit"
