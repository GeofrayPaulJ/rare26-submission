#!/bin/bash
# Installed as /opt/nvidia/entrypoint.d/99-rare26-sweep.sh inside the
# Prometheus container (that install is a one-line copy; this file is the
# version-controlled source of truth for what it does).
#
# WHY ENTRYPOINT.D AND NOT .bashrc. NVIDIA's own entrypoint
# (/opt/nvidia/nvidia_entrypoint.sh) sources every *.sh in this directory, in
# alpha order, on every container start -- confirmed by reading that script,
# not assumed: it globs "${_SCRIPT_DIR}/entrypoint.d"/*.sh, sources each one,
# then execs bash (interactive) or the given CMD. .bashrc only runs for an
# interactive login shell, which `docker start` on a detached container never
# creates -- that is the exact non-firing failure mode this hook exists to
# avoid.
#
# WHY BACKGROUNDED. This file is SOURCED (not executed) into the entrypoint's
# own shell, which is PID 1 in the container. If this blocked or replaced that
# shell, the entrypoint would never reach its final `exec bash` and the
# container would exit the instant the sweep driver did. The subshell + `&`
# backgrounds run_sweep.sh so sourcing returns immediately; `disown` detaches
# it so the parent shell exiting later does not signal it.
#
# WHY THIS CAN BE A ONE-LINER (x3, as of 2026-08-05). run_sweep.sh,
# run_chain.sh, and run_pinned_chain.sh each contain their own complete
# idempotency logic (halt/complete/stage sentinels, PID-based liveness,
# stale-tmux cleanup) -- see each script's own header. This hook's only job
# is to make sure ALL THREE get a chance to run on every container start; it
# duplicates none of their reasoning.
#
# run_chain.sh was added 2026-08-05 alongside the overnight chain
# (scripts/40_overnight_chain.py, tmux session rare26_chain). run_pinned_chain.sh
# was added the same day for the decisive isolation arm
# (scripts/44_pinned_arm_chain.py, tmux session rare26_pinned_chain, item 5 of
# the A4-reproducibility remediation). Before run_chain.sh existed, only
# run_sweep.sh (session rare26_sweep) was revived here -- a container restart
# would bring the GPU training driver back but silently NOT resume whatever
# was orchestrating around it, since nothing else knew that session existed.
# All three are safe to invoke unconditionally on every start: each checks
# its own PID file before doing anything, so a call here when the driver is
# already alive (e.g. the container was never actually down) is a fast no-op.
(
    bash /workspace/RARE26/scripts/run_sweep.sh \
        >> /workspace/RARE26/logs/entrypoint_launch.log 2>&1
) &
disown 2>/dev/null || true
(
    bash /workspace/RARE26/scripts/run_chain.sh \
        >> /workspace/RARE26/logs/entrypoint_launch.log 2>&1
) &
disown 2>/dev/null || true
(
    bash /workspace/RARE26/scripts/run_pinned_chain.sh \
        >> /workspace/RARE26/logs/entrypoint_launch.log 2>&1
) &
disown 2>/dev/null || true
