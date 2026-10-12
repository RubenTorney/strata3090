# RTX 3090 + Ryzen 5 5600X: where the host's per-layer time goes (2026-10-12)

Question: a `perf record` of a 2,500-token decode showed the 6 pool threads idle two thirds of the time (63 % of samples in
`ExpertPool::worker` spinning, 10 % in `Verifier::run` spinning, ~23 % in expert kernels). Is the CPU share of a verify window a
per-layer latency chain (wake-ups, barriers, flag hops) that could be shortened, or kernel time?

Answer, measured: it is kernel time. The pool's synchronisation costs about 0.4 us per row phase (two phases per layer), the
host's plan/quantization/job building about 5 us per layer, and the rows themselves 80 us per missed expert on all six
threads. The idle two thirds are the part of the window where the GPU is the bottleneck (the host waits for the GPU to reach
the next layer, 11-13 ms of a 18-21 ms window). Nothing cheap was found; details below.

## Machine and build

RTX 3090 24 GB (sm_86, driver 595.99.02, CUDA 13.2), Ryzen 5 5600X (6 cores, AVX2), 62 GiB DDR4, Ubuntu 24.04, kernel 7.0.
Branch `dev/latency` on `opt/3090`, built in its own `build/` (Release, sm_86). Model Qwen3.8-Flash-Next GSQ-RCO IQ3_S,
context 262,144, K/V int8, MTP drafter, the retained config `bench/results/2026-10-12-rtx3090-max/config_rtx3090_max.json`
(`--pcie-frac 0.35 --spec 4 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --adapt-async 1`,
`STRATA_PCIE_BRANCH=1 STRATA_FETCH_BLOCKS=64 STRATA_GR_FAST=1 STRATA_IQ_ZEN3=1`). Pool: 5 workers on cores 1-5 + the host
thread on core 0 draining too (6 threads). Driver: `tools/strata_ab_linux.py speed --no-deterministic --max-tokens 256`, a
fresh engine per arm, 2 warm-ups, 12 cold prompts (4 for the profile arms).

## The measurement switches

- `STRATA_DECODE_TIMING=1` (existing): per request, the window's split and the per-layer host split `[plan actq jobs CPU]`.
- New in this branch, measurement only, default off:
  - `STRATA_POOL_STAMPS=1`: in every row phase of `run_split_multi(_native)`, clock stamps at the host's publish, each thread's
    first claim and last `done`, the host's own drain end, the host seeing `done == n`, and the re-park; reported per
    request as `strata decode pool stamps:` (gate/up and down phases separately, averaged over the phases since the last
    report, so a request's prefill phases are in the average too).
  - the `strata decode host chain` line (always with `STRATA_DECODE_TIMING=1`): `ms_plan` split into `begin_layer`, the plan,
    the publish of flag A and the PCIe fetch submit; the pool's gate/up phase, the host's intermediate quantization and the
    down phase per window (decode windows only).
- `STRATA_VERIFY_PROFILE=1` (existing): the GPU stage stamps, including the GPU's waits for the plan (`waitA`), the PCIe
  share (`waitB`) and the CPU rows (`waitCPU`). It slows the engine (108.4 against 118.0 tok/s on the same 4 prompts).

Files: `profile2_stamps_lines.txt`, `profile2_stampsgpu_lines.txt` (the engine's lines for the 4 prompts, after 2 warm-ups),
`profile2.jsonl`, `profile.jsonl`/`profile.txt` (a first run where the env syntax left the stamps off; its timing lines agree).

## Where a layer's host time goes (prompt 3 of the `stamps` arm: 109 windows, 2.35 tokens/window, 20.8 ms/window)

Per window: GPU-reach wait 11.40 ms + per-layer host 6.97 ms (plan 0.10, activation quantization 0.09, job building 0.03,
pool 6.73) + draft 1.32 + commit 0.01. Per layer-window: 1.85 missed experts on the CPU (2.63 entries), 0.52 over PCIe,
22.9 VRAM hits.

Per layer (45 layers), host side:

| Step | us per layer | Note |
|---|---:|---|
| `begin_layer` (source) | 0.04 | |
| the GPU plan (distinct experts, VRAM/PCIe/CPU split, the plan tables) | 2.1 | |
| publish flag A + submit the PCIe fetch | 0.04 | the fetch is a GPU copy kernel on the branch stream (`STRATA_PCIE_BRANCH=1`) |
| activation quantization (per token) | 2.0 | |
| job building | 0.7 | |
| gate/up phase: park-wait + publish to first claim | 0.2 | all 6 threads take a task in every phase (18 tasks, 3 per thread) |
| gate/up rows | 100 (54 us per expert) | last `done` - mean thread finish: 7 us (prefill phases in the average) |
| last `done` to the host seeing it + re-park | 0.2 | |
| intermediate quantization (host, serial) | 0.5 | |
| down phase: sync as above | 0.4 | |
| down rows | 48 (26 us per expert) | tail 4 us |
| **total** | **~155** | of which rows ~148 |

(The rows per layer are the per-window phase times `gate/up 4.52 + down 2.18 ms` over 45 layers; the sync and tail numbers
are from the stamps, averaged over this request's 3,999 phases including its prefill.)

So one missed expert costs ~80 us of six-thread time: gate/up 1.41 MB of IQ3_S at ~26 GB/s over 6 threads (4.3 GB/s per
thread, the i-quant codebook decode rate noted earlier), down 0.74 MB of IQ4_NL at ~30 GB/s. The sync of a phase (wake-up,
done visibility, re-park) is 0.4 us; the workers never sleep during a request (1-2 phases per request woke a sleeper: the
first after the idle gap). The tail imbalance of a phase (slowest thread against the mean) is 5-7 %.

GPU side (the `stampsgpu` arm, prompt 2: 122 windows, 21.2 ms/window with the profiler on): `waitCPU` 1.38 + 0.38 = 1.76 ms per
window (8 %) is the GPU idling for the CPU rows after it has done its VRAM hits and PCIe share; `waitB` 2.13 + 0.60 = 2.73 ms
(13 %) is its wait for the PCIe share before that; `waitA` 0.42 ms for the plan. The CPU rows and the PCIe fetch run at the
same time, so the GPU's exposed wait for the misses is their max, 4.5 ms per window (21 %); the rest of the window is GPU
kernels (13-14 ms) and the draft.

## What that rules out

- Cutting a wake-up hop or a barrier: a whole phase's sync is 0.4 us, 0.8 us per layer, 0.04 ms per window (0.2 %).
- Finer row splitting / all cores on every miss: already the case (18 equal row ranges across all the layer's missed experts
  over 6 threads; a range may span two experts).
- Starting an expert's down rows before the gate/up phase ends (one fused phase): saves at most the gate/up tail, ~7 us per
  layer, 0.3 ms of six-thread time per window, of which only the exposed share (`waitCPU`/pool ~ 25 %) would show: <= 0.4 %.
- The host computing rows instead of spinning: it already drains 3 of the 18 tasks per phase.
- The host's serial pieces (plan 2 us, activation quantization 2 us, intermediate quantization 0.5 us): 5 us per layer, 0.2 ms
  per window of which the exposed share is ~0.05 ms (0.25 %).

The only lever left on the CPU side is the per-row kernel (the codebook decode rate), which earlier rounds measured at ~1 %
for a faster kernel, and the number of misses (VRAM). The larger exposed wait is the PCIe share (`waitB`), not the CPU.

## Speed (one round, 12 cold prompts, 256 tokens, median tok/s; the baseline is the same series)

`--pool-tasks N` is the existing option for the row-range count per phase (default 0 = 3 per thread = 18 here): finer ranges
cut the tail imbalance, coarser ones the claims. Same rows and the same per-row arithmetic in every split (only the range
boundaries move).

Series 1 (`speed_tasks.txt/.jsonl`, engines in this order):

| Arm | tok/s | Note |
|---|---:|---|
| `ref` | 121.0 | the first engine of the series |
| `pt36` (`--pool-tasks 36`) | 122.8 | |
| `pt12` (`--pool-tasks 12`) | 122.8 | |

Series 2 (`speed_tasks2.txt/.jsonl`, the order reversed to check the first-engine effect):

| Arm | tok/s | paired ratio |
|---|---:|---:|
| `pt36` (first engine) | 122.8 | |
| `ref` | 122.8 | 0.999, 4/12 prompts faster |

So the +1.5 % of series 1 was the first engine of a series running slow (seen before in `2026-10-12-rtx3090-max`, `speed1`),
not the task count: `--pool-tasks 36` and `--pool-tasks 12` are within noise of the default. The baseline of this build
(122.8 in series 2) matches the retained config's 122.2-123.7.

Nothing from this round is recommended for the config. The measurement switches stay in the code, off by default.

## Not tested

- A fused gate/up + down phase (per-expert completion counters): not implemented, the measured ceiling is under 0.5 %.
- `--pool-tasks` under the deterministic parity check: the same rows and the same per-row arithmetic in every split, only the
  boundaries move; parity was not run because the speed arms did not reach the 1 % bar.
- HIP, SYCL, Windows; long contexts; the PCIe wait (`waitB`), which is the bigger exposed wait and was tuned in earlier rounds.
