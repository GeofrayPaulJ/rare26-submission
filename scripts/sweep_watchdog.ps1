# Host-side watchdog for the RARE26 magnitude sweep.
#
# WHY THIS EXISTS. tmux does not survive a container restart -- it lives in the
# container's PID namespace, so `docker restart` takes the tmux server and every
# process under it. That was tested rather than assumed: the session vanished,
# `tmux ls` reported no server, and the dummy job's log stopped dead. tmux
# therefore buys survival of CLIENT disconnect (SSH, closing a terminal) and
# nothing more.
#
# Real durability needs three things together, and this script is the second:
#   1. the container's `unless-stopped` restart policy, so Docker brings the
#      container back;
#   2. this watchdog, so something re-enters the container and restarts the
#      sweep once it is back;
#   3. the sweep's own resumability, so a restart re-derives its plan from the
#      parquets on disk and loses at most the unit that was in flight.
#
# It deliberately will NOT restart a sweep that halted on a broken invariant or
# one that finished -- scripts/run_sweep.sh checks both sentinels. A watchdog
# that restarts a run into the same wall forever is worse than no watchdog.

param(
    [string]$ContainerName = "Prometheus",
    [int]$IntervalSeconds  = 60,
    [string]$LogPath       = "D:\RARE26\logs\watchdog.log",
    [string]$PidPath       = "D:\RARE26\logs\watchdog.pid",
    [string]$HeartbeatPath = "D:\RARE26\logs\watchdog_heartbeat.txt"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

# Publish liveness as a PID and a heartbeat, never as a process-name match.
# Asking Windows "is a pwsh running whose CommandLine contains sweep_watchdog"
# matches the very command asking the question -- the filter string lands in the
# asking process's own command line. That produced a confident false positive
# here (a watchdog reported running when none was), which is the same trap that
# `pgrep -f 23_sweep.py` set on the container side. A PID plus a timestamp that
# advances is checkable without ambiguity.
Set-Content -Path $PidPath -Value $PID

function Write-Log([string]$Message) {
    # UTC, and genuinely UTC. The container logs in UTC, so a host-side log
    # stamping LOCAL time with a "Z" suffix would put the two five and a half
    # hours apart while claiming they agree -- precisely the confusion nobody
    # needs while working out what happened during an overnight failure.
    $line = "[{0}] {1}" -f (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"), $Message
    Add-Content -Path $LogPath -Value $line
}

Write-Log "watchdog started (container=$ContainerName, interval=${IntervalSeconds}s)"

while ($true) {
    # Heartbeat first, every poll. The log only records events, so without this
    # a quiet watchdog and a dead one look identical from outside.
    try {
        Set-Content -Path $HeartbeatPath `
            -Value ((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))
    } catch { }

    try {
        # --- is EACH driver finished or deliberately halted? ---
        # Checked independently: the sweep (23_sweep.py, session rare26_sweep)
        # and the chain (40_overnight_chain.py, session rare26_chain) are two
        # separate long-running things on two separate schedules. 2026-08-05:
        # this watchdog used to stand down the instant the SWEEP alone reached
        # sweep_stage_complete -- which meant it went completely dark the
        # moment the chain's real overnight work (hours of checkpointed
        # retraining) was just getting started, because nothing here knew the
        # chain existed. Both must be terminal before this loop exits.
        $sentinel = docker exec $ContainerName bash -c `
            "test -f /workspace/RARE26/logs/sweep_halt.json && echo SWEEP_HALTED; test -f /workspace/RARE26/logs/sweep_stage_complete && echo SWEEP_COMPLETE; test -f /workspace/RARE26/logs/chain_complete && echo CHAIN_COMPLETE; ls /workspace/RARE26/logs/chain_stage*_HALT >/dev/null 2>&1 && echo CHAIN_HALTED; test -f /workspace/RARE26/logs/pinned_chain_done && echo PINNED_COMPLETE; test -f /workspace/RARE26/logs/a4_loco_chain_done && echo A4LOCO_COMPLETE; test -f /workspace/RARE26/logs/swa_chain_report_done && echo SWA_COMPLETE; test -f /workspace/RARE26/logs/pauc_chain_report_done && echo PAUC_COMPLETE; test -f /workspace/RARE26/logs/swaloco_chain_report_done && echo SWALOCO_COMPLETE; test -f /workspace/RARE26/logs/ema_chain_report_done && echo EMA_COMPLETE; test -f /workspace/RARE26/logs/pauc_p1_chain_report_done && echo PAUCP1_COMPLETE; test -f /workspace/RARE26/logs/g3ckpt_chain_report_done && echo G3CKPT_COMPLETE; test -f /workspace/RARE26/logs/paucloco_chain_report_done && echo PAUCLOCO_COMPLETE; test -f /workspace/RARE26/logs/g3pauc_full_chain_done && echo G3PAUC_COMPLETE; test -f /workspace/RARE26/logs/deploy_a4_full_chain_done && echo DEPLOYA4FULL_COMPLETE" 2>$null

        $sweepTerminal = ($sentinel -match "SWEEP_HALTED") -or ($sentinel -match "SWEEP_COMPLETE")
        $chainTerminal = ($sentinel -match "CHAIN_HALTED") -or ($sentinel -match "CHAIN_COMPLETE")
        $pinnedTerminal = ($sentinel -match "PINNED_COMPLETE")
        $a4LocoTerminal = ($sentinel -match "A4LOCO_COMPLETE")
        $swaTerminal = ($sentinel -match "SWA_COMPLETE")
        $paucTerminal = ($sentinel -match "PAUC_COMPLETE")
        $swalocoTerminal = ($sentinel -match "SWALOCO_COMPLETE")
        $emaTerminal = ($sentinel -match "EMA_COMPLETE")
        $paucp1Terminal = ($sentinel -match "PAUCP1_COMPLETE")
        $g3ckptTerminal = ($sentinel -match "G3CKPT_COMPLETE")
        $paucLocoTerminal = ($sentinel -match "PAUCLOCO_COMPLETE")
        # 2026-08-10 extension: g3pauc chain (stages 1/2/5/7 --
        # scripts/run_g3pauc_chain.sh). Matched against G3PAUC_COMPLETE,
        # distinct from any tag the earlier stages-1+2-only version of that
        # script used, precisely so a stale old sentinel can never be
        # mistaken for this longer chain being done.
        $g3paucTerminal = ($sentinel -match "G3PAUC_COMPLETE")
        # 2026-08-11 extension: deploy_a4_full chain (R1 -- A4-corrected
        # full-data deployment checkpoints, seeds 0-4, scripts/82_deploy_a4_full_chain.py
        # via run_deploy_a4_full_chain.sh). Same PID/sentinel discipline.
        $deployA4FullTerminal = ($sentinel -match "DEPLOYA4FULL_COMPLETE")

        if ($sweepTerminal -and $chainTerminal -and $pinnedTerminal -and $a4LocoTerminal -and $swaTerminal -and $paucTerminal -and $swalocoTerminal -and $emaTerminal -and $paucp1Terminal -and $g3ckptTerminal -and $paucLocoTerminal -and $g3paucTerminal -and $deployA4FullTerminal) {
            $flat = ($sentinel -join ",")
            Write-Log "all thirteen drivers terminal ($flat); watchdog standing down"
            break
        }

        # --- container up? (applies to both drivers) ---
        $running = docker inspect -f '{{.State.Running}}' $ContainerName 2>$null
        if ($running -ne "true") {
            Write-Log "container is not running; starting it"
            docker start $ContainerName | Out-Null
            Start-Sleep -Seconds 20
        }

        # --- sweep driver alive? (skip once sweep itself is terminal) ---
        # By PID, never by process name. `pgrep -f 23_sweep.py` matches the
        # `bash -c` wrapper this very command creates, because that wrapper's
        # command line contains the pattern -- so it answers "alive" forever and
        # the watchdog never relaunches anything. That was observed during the
        # restart test with zero driver processes actually running.
        if (-not $sweepTerminal) {
            $alive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/sweep.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($alive -notmatch "YES") {
                Write-Log "sweep driver not running; invoking run_sweep.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_sweep.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_sweep: $_" }
            }
        }

        # --- chain driver alive? (skip once chain itself is terminal) ---
        # Same PID-not-process-name discipline, against chain.pid /
        # 40_overnight_chain.py / run_chain.sh instead of the sweep's own.
        if (-not $chainTerminal) {
            $chainAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($chainAlive -notmatch "YES") {
                Write-Log "chain driver not running; invoking run_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_chain: $_" }
            }
        }

        # --- pinned-arm chain driver alive? (skip once terminal) ---
        # Same discipline again, against pinned_chain.pid /
        # 44_pinned_arm_chain.py / run_pinned_chain.sh. Added 2026-08-05
        # alongside the driver itself -- learned from the chain's own gap
        # (found and fixed the same day) not to let a third independent
        # driver go unwatched from the start this time.
        if (-not $pinnedTerminal) {
            $pinnedAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/pinned_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($pinnedAlive -notmatch "YES") {
                Write-Log "pinned-arm chain driver not running; invoking run_pinned_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_pinned_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_pinned_chain: $_" }
            }
        }

        # --- A4 LOCO (corrected-code) chain driver alive? (skip once terminal) ---
        # Same discipline again, against a4_loco_chain.pid /
        # 49_a4_loco_chain.py / run_a4_loco_chain.sh. Added 2026-08-07 for
        # JOB 1 (A4 LOCO under corrected code, ~4.2h/10 units) alongside the
        # driver itself, same day, so this job is never left unwatched.
        if (-not $a4LocoTerminal) {
            $a4LocoAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/a4_loco_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($a4LocoAlive -notmatch "YES") {
                Write-Log "A4 LOCO chain driver not running; invoking run_a4_loco_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_a4_loco_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_a4_loco_chain: $_" }
            }
        }

        # --- SWA/EMA arm + pAUC smoke (2026-08-07 overnight, tmux swa/pauc).
        # Same PID discipline. run_pauc_chain.sh self-guards on the swa
        # sentinel, so invoking both every cycle cannot start them in the
        # wrong order or double-book the GPU.
        if (-not $swaTerminal) {
            $swaAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/swa_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($swaAlive -notmatch "YES") {
                Write-Log "SWA chain driver not running; invoking run_swa_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_swa_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_swa_chain: $_" }
            }
        }
        if (-not $paucTerminal) {
            $paucAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/pauc_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($paucAlive -notmatch "YES") {
                Write-Log "pAUC chain driver not running; invoking run_pauc_chain.sh (self-guarded on swa)"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_pauc_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_pauc_chain: $_" }
            }
        }

        # --- 2026-08-07/08 overnight extension: swaloco (step 3, the
        # shipping gate) -> ema (step 4, full-data checkpoint) -> pauc_p1
        # (step 5, gated on pauc smoke). Same PID/sentinel discipline.
        if (-not $swalocoTerminal) {
            $swalocoAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/swaloco_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($swalocoAlive -notmatch "YES") {
                Write-Log "swaloco chain driver not running; invoking run_swaloco_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_swaloco_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_swaloco_chain: $_" }
            }
        }
        if (-not $emaTerminal) {
            $emaAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/ema_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($emaAlive -notmatch "YES") {
                Write-Log "ema chain driver not running; invoking run_ema_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_ema_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_ema_chain: $_" }
            }
        }
        if (-not $paucp1Terminal) {
            $paucp1Alive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/pauc_p1_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($paucp1Alive -notmatch "YES") {
                Write-Log "pauc_p1 chain driver not running; invoking run_pauc_p1_chain.sh (self-guarded on ema+smoke)"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_pauc_p1_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_pauc_p1_chain: $_" }
            }
        }

        # --- 2026-08-08 extension: g3ckpt (single-checkpoint retrain,
        # unblocks JOB D) -> paucloco (JOB C). Same PID/sentinel discipline.
        if (-not $g3ckptTerminal) {
            $g3ckptAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/g3ckpt_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($g3ckptAlive -notmatch "YES") {
                Write-Log "g3ckpt chain driver not running; invoking run_g3ckpt_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_g3ckpt_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_g3ckpt_chain: $_" }
            }
        }
        if (-not $paucLocoTerminal) {
            $paucLocoAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/paucloco_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($paucLocoAlive -notmatch "YES") {
                Write-Log "paucloco chain driver not running; invoking run_paucloco_chain.sh (self-guarded on g3ckpt)"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_paucloco_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_paucloco_chain: $_" }
            }
        }

        # --- 2026-08-10 extension: g3pauc chain (G3+pAUC pooled+LOCO, G3
        # full-data deploy checkpoint, tighten-margin LOCO seeds 5-9 for
        # A4-corrected+G3 -- scripts/run_g3pauc_chain.sh). Same PID/sentinel
        # discipline as every other block above. Independent of every other
        # driver here (no self-guard on a prerequisite sentinel needed).
        if (-not $g3paucTerminal) {
            $g3paucAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/g3pauc_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($g3paucAlive -notmatch "YES") {
                Write-Log "g3pauc chain driver not running; invoking run_g3pauc_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_g3pauc_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_g3pauc_chain: $_" }
            }
        }
        # --- deploy_a4_full chain driver alive? (skip once terminal) ---
        # Same PID/cmdline discipline as every block above, against
        # deploy_a4_full_chain.pid / 82_deploy_a4_full_chain.py /
        # run_deploy_a4_full_chain.sh. Independent of every other driver
        # here -- no self-guard on a prerequisite sentinel needed.
        if (-not $deployA4FullTerminal) {
            $deployA4FullAlive = docker exec $ContainerName bash -c `
                "p=`$(cat /workspace/RARE26/logs/deploy_a4_full_chain.pid 2>/dev/null); if [ -n `"`$p`" ] && kill -0 `$p 2>/dev/null; then echo YES; else echo NO; fi" 2>$null
            if ($deployA4FullAlive -notmatch "YES") {
                Write-Log "deploy_a4_full chain driver not running; invoking run_deploy_a4_full_chain.sh"
                docker exec $ContainerName bash /workspace/RARE26/scripts/run_deploy_a4_full_chain.sh 2>&1 |
                    ForEach-Object { Write-Log "  run_deploy_a4_full_chain: $_" }
            }
        }
    }
    catch {
        # A transient docker CLI failure must never kill the watchdog -- that
        # would silently remove the very durability this exists to provide.
        Write-Log "transient error (ignored): $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $IntervalSeconds
}

Write-Log "watchdog exiting"
