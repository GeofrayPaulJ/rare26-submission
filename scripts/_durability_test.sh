#!/usr/bin/env bash
# Dummy long-running job for the tmux/container durability test.
# Writes a monotonically increasing tick to a log so we can tell, after a
# container restart, whether the process survived, resumed, or died.
set -u
LOG=/workspace/RARE26/logs/durability_test.log
mkdir -p "$(dirname "$LOG")"
i=0
while true; do
    i=$((i + 1))
    echo "tick $i pid=$$ $(date -u +%H:%M:%S)" >> "$LOG"
    sleep 1
done
