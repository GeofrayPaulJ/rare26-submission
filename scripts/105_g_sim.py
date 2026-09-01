"""G-SIM -- governor coverage simulation.

JOB B: reconstruct per-batch data-path + GPU cost curves for a 384-image
(12-batch, batch_size=32) case at k=5 with the h+v+2-rotations TTA
candidate (6 views -- the most expensive candidate this project's own
Stage 2 sweep ACCEPTED at k=5, `reports/stage2_tta_results.md`), on T4
and A10G separately.

Data path (decode/FOV/crop/two-stage-resize, CPU-bound, confirmed
vendor-independent -- reports/g2_scaling_fix.md addendum A1: GPU time is
~1% of wall time, T4-vs-A10G derate applies only to the GPU-bound
portion): ONE curve, used for both vendors.
  - b1-b5: job c2c00341's own measured T4 numbers, the correct case size
    (reports/d_rule_configuration_decision.md amendment,
    reports/g3_timeout_semantics.md CONSEQUENCE 2): 121.5, 56.6, 36.1,
    26.4, 23.9 seconds.
  - b6-b12: NOT measured (c2c00341's own report never quotes past b5) --
    extended using the GROWTH SLOPE fit to Probe C job ba39f6c6
    (reports/d_rule_configuration_decision.md): batch_time(b) ~= 1.86 +
    1.885*b for b>=8, i.e. +1.885 s/batch. Anchored to c2c00341's own
    b5=23.9s floor (its ABSOLUTE floor differs from ba39f6c6's 17.5s --
    two different jobs -- so only the SLOPE transfers, not the level).
    This is the "include the ItkStack leak's monotonic growth term" step
    the task explicitly asked for; c2c00341's own report only assumed
    the floor held flat past b5, which is optimistic given G1/G2's
    confirmed no-plateau finding.

GPU addition, per batch, ON TOP of the data-path curve (which already
embeds ONE forward pass's worth of GPU cost, k=1/1-view, per c2c00341's
own real numbers): incremental passes = (members x views) - 1.
  - T4 rate: 10.9375 ms/image/pass (job c2c00341's own GPU-only number,
    reports/r3_k5_rebuild.md: "2.1s / 192 images / 1 member").
  - A10G rate: 6.5 ms/image/pass (try-out job dbe59741's own number,
    reports/t2_runtime_projection.md: "1.3s / 200 images, k=1").

JOB A/C: governor trigger logic, reconstructed exactly from
submission/rare26_infer/predict.py as currently shipped (G6) and from
reports/g4_time_governor.md's own description of the first-shipped (G4)
logic (G4's code is not literally in the repo any more -- predict.py has
been overwritten through G5/G6 -- so G4 is reconstructed from that
report's own description, quoted in reports/g_sim.md).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/105_g_sim.py'
"""
from __future__ import annotations

import json

N = 384
BATCH = 32
N_BATCHES = N // BATCH  # 12, exact -- no partial final batch
BUDGET = 540.0          # TIME_BUDGET_SECONDS, unchanged G4->G6
HARD_STOP = 480.0       # introduced G5, unchanged G5->G6; ABSENT in G4

# --- JOB B: data-path curve (vendor-independent), seconds/batch ---
DATA_MEASURED = [121.5, 56.6, 36.1, 26.4, 23.9]           # b1-b5, job c2c00341, T4, real
GROWTH_SLOPE = 1.885                                        # s/batch, fit to job ba39f6c6 (post-floor)
DATA_PATH = list(DATA_MEASURED) + [
    DATA_MEASURED[-1] + GROWTH_SLOPE * (b - 5) for b in range(6, N_BATCHES + 1)
]
assert len(DATA_PATH) == N_BATCHES

# --- GPU rates ---
T4_MS_PER_IMAGE_PASS = 10.9375
A10G_MS_PER_IMAGE_PASS = 6.5
MEMBERS_FULL = 5
VIEWS = 6  # h+v+2 rotations, the most expensive Stage-2-ACCEPTED k=5 TTA candidate
PASSES_FULL = MEMBERS_FULL * VIEWS       # 30
PASSES_DEGRADED = 1 * VIEWS              # 6, after governor step 2 (ensemble -> 1 member)


def gpu_increment_seconds(rate_ms_per_pass: float, passes: int, batch_size: int = BATCH) -> float:
    """Incremental GPU cost added on top of the data-path curve, which
    already embeds ONE pass's GPU cost (k=1, 1 view)."""
    incremental_passes = passes - 1
    return batch_size * incremental_passes * rate_ms_per_pass / 1000.0


def build_curve(rate_ms_per_pass: float):
    full_inc = gpu_increment_seconds(rate_ms_per_pass, PASSES_FULL)
    degraded_inc = gpu_increment_seconds(rate_ms_per_pass, PASSES_DEGRADED)
    full = [d + full_inc for d in DATA_PATH]
    degraded = [d + degraded_inc for d in DATA_PATH]
    return full, degraded  # per-batch time IF that batch runs at full / degraded ensemble size


def simulate(rate_ms_per_pass: float, governor: str) -> dict:
    """governor: 'G4' (cumulative rate, no MIN_BATCHES, no hard stop) or
    'G6' (trailing-min(2,4), MIN_BATCHES=5, 480s hard stop)."""
    full_curve, degraded_curve = build_curve(rate_ms_per_pass)

    seen = 0
    active_members = MEMBERS_FULL
    t_start = 0.0
    elapsed_since_start = 0.0
    batch_wall_times: list = []
    batch_sizes: list = []
    trace = []
    stopped_at_batch = None
    stop_reason = None

    for bi in range(N_BATCHES):  # 0-indexed, matches predict.py's `bi`
        this_batch_time = (full_curve if active_members == MEMBERS_FULL else degraded_curve)[bi]
        elapsed_since_start += this_batch_time
        seen += BATCH
        batch_wall_times.append(this_batch_time)
        batch_sizes.append(BATCH)

        row = {"batch_index_0based": bi, "batch_time_s": round(this_batch_time, 3),
               "elapsed_since_start_s": round(elapsed_since_start, 3),
               "active_members": active_members, "seen": seen}

        if governor == "G6":
            if elapsed_since_start > HARD_STOP and seen < N:
                stopped_at_batch = bi
                stop_reason = f"HARD STOP: elapsed {elapsed_since_start:.1f}s > {HARD_STOP:.0f}s"
                row["action"] = stop_reason
                trace.append(row)
                break
            if len(batch_wall_times) < 5:  # MIN_BATCHES_FOR_PROJECTION
                row["action"] = "below MIN_BATCHES_FOR_PROJECTION, no check"
                trace.append(row)
                continue
            w2, s2 = batch_wall_times[-2:], batch_sizes[-2:]
            w4, s4 = batch_wall_times[-4:], batch_sizes[-4:]
            rate2 = sum(s2) / sum(w2)
            rate4 = sum(s4) / sum(w4)
            proj_short = elapsed_since_start + (N - seen) / rate2
            proj_long = elapsed_since_start + (N - seen) / rate4
            proj = min(proj_short, proj_long)
            row["proj_short"] = round(proj_short, 1)
            row["proj_long"] = round(proj_long, 1)
            row["proj_min"] = round(proj, 1)
            if proj > BUDGET and seen < N:
                if active_members > 1:
                    active_members = 1
                    row["action"] = f"DEGRADE step2: ensemble -> 1 member (proj {proj:.1f} > {BUDGET:.0f})"
                else:
                    stopped_at_batch = bi
                    stop_reason = f"DEGRADE step3: stop+fill (proj {proj:.1f} > {BUDGET:.0f}, already 1 member)"
                    row["action"] = stop_reason
                    trace.append(row)
                    break
            else:
                row["action"] = "within budget, no change"
            trace.append(row)

        elif governor == "G4":
            # No MIN_BATCHES gate, no hard stop, CUMULATIVE rate since loop start.
            rate = seen / elapsed_since_start
            proj = elapsed_since_start + (N - seen) / rate
            row["proj_cumulative"] = round(proj, 1)
            if proj > BUDGET and seen < N:
                if active_members > 1:
                    active_members = 1
                    row["action"] = f"DEGRADE step2: ensemble -> 1 member (proj {proj:.1f} > {BUDGET:.0f})"
                else:
                    stopped_at_batch = bi
                    stop_reason = f"DEGRADE step3: stop+fill (proj {proj:.1f} > {BUDGET:.0f}, already 1 member)"
                    row["action"] = stop_reason
                    trace.append(row)
                    break
            else:
                row["action"] = "within budget, no change"
            trace.append(row)
        else:
            raise ValueError(governor)

    images_scored = seen if stopped_at_batch is None else stopped_at_batch * BATCH + (
        BATCH if stopped_at_batch is not None and trace[-1].get("seen", 0) == seen else 0
    )
    # `seen` already reflects the batch that triggered the stop (it IS scored -- the
    # governor stops scoring FUTURE batches, the triggering batch's own images are real).
    images_scored = seen
    coverage = images_scored / N

    return {
        "governor": governor,
        "gpu_vendor": "T4" if rate_ms_per_pass == T4_MS_PER_IMAGE_PASS else "A10G",
        "stopped_at_batch_0based": stopped_at_batch,
        "stop_reason": stop_reason,
        "images_scored": images_scored,
        "coverage_fraction": round(coverage, 4),
        "final_elapsed_s": round(elapsed_since_start, 1),
        "platform_600s_kill": elapsed_since_start > 600 and stopped_at_batch is None,
        "trace": trace,
    }


def main() -> int:
    results = []
    for vendor_rate in (T4_MS_PER_IMAGE_PASS, A10G_MS_PER_IMAGE_PASS):
        for gov in ("G4", "G6"):
            results.append(simulate(vendor_rate, gov))

    # No-governor case (the ACTUAL 8 August configuration): k=1, no TTA, no governor at
    # all -- total wall time is just the data path (already embeds k=1's GPU cost).
    no_gov_total = sum(DATA_PATH)
    results.append({
        "governor": "NONE (actual 8 Aug config: k=1, no TTA, predates G4 by 4 days)",
        "gpu_vendor": "n/a (k=1 GPU cost already embedded in data path)",
        "total_wall_time_s": round(no_gov_total, 1),
        "platform_600s_kill": no_gov_total > 600,
        "outcome": "COMPLETE (100% coverage, real file written)" if no_gov_total <= 600
                   else "KILLED (0% coverage -- NO FILE written at all, per reports/g3_timeout_semantics.md)",
    })

    with open("reports/g_sim_raw.json", "w") as fh:
        json.dump(results, fh, indent=2)

    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "trace"}, indent=2))
        print()

    print("wrote reports/g_sim_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
