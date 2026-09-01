# Register the RARE26 sweep watchdog as a self-restarting Windows Scheduled Task.
#
# WHY THIS IS NOT REPO CODE. Everything else in this project's durability story
# lives inside the container and is restored by the container itself:
#
#   container dies / is restarted -> Docker's `unless-stopped` policy brings it
#       back, then /opt/nvidia/entrypoint.d/99-rare26-sweep.sh runs
#       scripts/run_sweep.sh and the driver resumes. No host help needed.
#
# The host-side watchdog covers the one case that chain CANNOT cover: the
# container is DOWN and stays down (Docker Desktop restarted, the engine was
# not running at boot, the restart policy did not fire). A stopped container
# cannot run its own entrypoint to revive itself -- something outside it has to
# issue `docker start`. That is the watchdog's job, and it is why the watchdog
# dying is a real gap even though the inner layers are self-healing.
#
# A process cannot durably restart itself after being killed; only something
# outside its lifetime can. On Windows that is Task Scheduler (or a service
# wrapper such as NSSM). Hence: host config, and hence this script is provided
# for YOU to run rather than executed automatically -- it makes a persistent,
# machine-level change outside the repository.
#
# WHAT THIS CREATES
#   Task name : RARE26SweepWatchdog
#   Triggers  : at system startup, AND every 5 minutes indefinitely
#   Action    : pwsh -NoProfile -File D:\RARE26\scripts\sweep_watchdog.ps1
#   Policy    : IgnoreNew -- if an instance is already running, do not start a
#               second one. This is what makes the 5-minute trigger a
#               "restart it if and only if it is dead" check rather than a
#               fork bomb.
#
# ON THE 5-MINUTE TRIGGER AND DELIBERATE STAND-DOWN. sweep_watchdog.ps1 exits
# on purpose when it sees sweep_halt.json or sweep_stage_complete ("standing
# down"). With this trigger it will be restarted 5 minutes later, immediately
# re-read those sentinels, and exit again -- a cheap no-op that adds two lines
# to watchdog.log per cycle. That is deliberate: it means clearing a halt
# sentinel is sufficient to bring the watchdog back on its own, with no manual
# relaunch. If the log noise is unwanted, delete the 5-minute trigger and keep
# only the startup trigger; you then get boot coverage but not death coverage.
#
# RUN AS ADMINISTRATOR:
#     pwsh -NoProfile -ExecutionPolicy Bypass -File D:\RARE26\scripts\install_watchdog_task.ps1
#
# To verify:      Get-ScheduledTask -TaskName RARE26SweepWatchdog
# To remove:      Unregister-ScheduledTask -TaskName RARE26SweepWatchdog -Confirm:$false

param(
    [string]$TaskName   = "RARE26SweepWatchdog",
    [string]$ScriptPath = "D:\RARE26\scripts\sweep_watchdog.ps1",
    [int]$RepeatMinutes = 5
)

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "Must run as Administrator (registering a machine-level task)."
    exit 1
}
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Watchdog script not found at $ScriptPath"
    exit 1
}

$pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = (Get-Command powershell).Source }

$action = New-ScheduledTaskAction -Execute $pwsh `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$ScriptPath`""

# Trigger 1: at boot, so a host reboot restores the outer safety net.
$atStartup = New-ScheduledTaskTrigger -AtStartup

# Trigger 2: a repeating liveness check. Combined with IgnoreNew below this
# starts the watchdog only when no instance is alive, which is precisely
# "restart it if it died" -- the gap that opened on 2026-08-03 when the
# watchdog was lost twice and nothing brought it back.
#
# -RepetitionDuration IS REQUIRED, NOT OPTIONAL, DESPITE THE CMDLET ACCEPTING
# ITS ABSENCE SILENTLY. First attempt omitted it: the task ran for a few
# cycles then VANISHED ENTIRELY from Task Scheduler -- `Get-ScheduledTask`
# returned nothing, no error, no event log entry. Omitting -RepetitionDuration
# does not mean "repeat forever"; it silently bounds the window, and once that
# elapses the trigger (and, empirically, the whole task) is treated as
# expired and dropped.
#
# [TimeSpan]::MaxValue was the obvious next attempt and is ALSO wrong, for a
# different reason: Task Scheduler's XML schema rejects it outright at
# registration --
#   "The task XML contains a value which is incorrectly formatted or out of
#    range. (13,42):Duration:P99999999DT23H59M59S"
# -- confirmed by hitting that exact error. Durations have to fit the
# scheduler's own range, and MaxValue does not.
#
# The actual fix is the standard idiom for "every N minutes, forever" in Task
# Scheduler: wrap a ONE-DAY repetition (schema-valid, nowhere near the range
# limit) inside a DAILY trigger. The daily trigger has no expiration of its
# own, so every day it re-arms a fresh 24h window of 5-minute repetition --
# net effect is continuous 5-minute polling with no end date, built entirely
# from bounded, schema-legal pieces instead of an unbounded one.
$dailyStart = (Get-Date).AddMinutes(1)
$repeating = New-ScheduledTaskTrigger -Daily -At $dailyStart
$repeating.Repetition = (New-ScheduledTaskTrigger -Once -At $dailyStart `
    -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 1)).Repetition

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)
# RestartCount/RestartInterval is a SECOND, independent restart path: if the
# watchdog process exits on its own (not merely "was never started"), Task
# Scheduler relaunches it within a minute, rather than waiting for the next
# 5-minute IgnoreNew tick. Belt and suspenders for the one job this task
# exists to do.

# SYSTEM so it runs with no user logged in -- an unattended run must survive
# the console session being signed out.
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" `
    -LogonType ServiceAccount -RunLevel Highest

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "removed existing task $TaskName"
}

Register-ScheduledTask -TaskName $TaskName -Action $action `
    -Trigger @($atStartup, $repeating) -Settings $settings `
    -Principal $principal `
    -Description "Restarts the RARE26 sweep watchdog if it is not running. See scripts/install_watchdog_task.ps1 for why this lives in Task Scheduler and not in the repo." | Out-Null

Write-Host "registered $TaskName"
Write-Host "  triggers : at startup, and every $RepeatMinutes min within each day (IgnoreNew)"
Write-Host "  action   : $pwsh -File $ScriptPath"
Write-Host "  runs as  : SYSTEM (survives user sign-out)"
Write-Host ""

# SELF-VERIFY rather than trust that "no exception" means "correctly
# persisted". The previous version of this script printed a clean success
# message for a task that silently vanished within the hour -- a print
# statement is not evidence. Read the registration BACK from Task Scheduler
# and check the fields that actually determine whether it survives.
$readback = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $readback) {
    Write-Error "VERIFICATION FAILED: $TaskName was not found immediately " `
        + "after Register-ScheduledTask returned. Do not trust this install."
    exit 1
}
$rep = $readback.Triggers | Where-Object { $_.Repetition -and $_.Repetition.Interval } |
    Select-Object -First 1 -ExpandProperty Repetition
if (-not $rep -or -not $rep.Duration) {
    Write-Error ("VERIFICATION FAILED: the repeating trigger has no " +
        "repetition duration ({0}). This is exactly the configuration that " +
        "caused the task to expire and disappear last time -- do not trust " +
        "this install." -f $rep.Duration)
    exit 1
}
Write-Host "verified: repetition interval=$($rep.Interval) duration=$($rep.Duration)"
Write-Host ("verified: readback confirms {0} trigger(s) registered" -f `
    $readback.Triggers.Count)
Write-Host ""
Write-Host "This confirms the task is CORRECTLY CONFIGURED right now. It does" `
    " not confirm it will still exist in an hour -- that failure mode gave" `
    " no error at registration time either. Re-check after this task has" `
    " had time to complete at least two repetition cycles:"
Write-Host "  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
