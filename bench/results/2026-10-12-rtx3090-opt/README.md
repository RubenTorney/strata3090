# RTX 3090 + Ryzen 5 5600X, Linux: optimisation round on 0.1.43 + the async arena tier

Branch `opt/3090` (main = `perf/async-adapt-arena` merged with upstream v0.1.43). Measured 2026-10-11/12 on one PC.

## Machine

- RTX 3090 24 GB (GA102, sm_86, 82 SMs, 2 copy engines, 6 MB L2), PCIe 4.0 x16, driver 595.99.02 (open kernel modules),
  CUDA 13.2.2, the card also drives the desktop
- Ryzen 5 5600X (6 cores / 12 threads, AVX2, no AVX-512, no AVX-VNNI), 62 GiB DDR4, THP in `madvise` mode
- Ubuntu 24.04, kernel 7.0. The 47 GiB expert arena is page-locked whole (`cudaHostRegister PORTABLE ok`) and sits in
  transparent huge pages (48.8 GB `AnonHugePages` of a 51.5 GB RSS, checked on a running engine)
- Qwen3.8-Flash-Next GSQ-RCO IQ3_S (gate/up: 14 layers IQ3_XXS, 20 IQ2_S, 10 IQ3_S, 1 IQ4_XS; down: 39 IQ4_NL, 9 Q2_0),
  `--max-context 262144 --kv int8 --kv-grow`, MTP drafter, draft vocabulary `fr`

## Protocol

`strata_ab2_linux.py` (this folder): a fresh engine per arm through `serve/server.py` of the v0.1.42 install (its
`.venv`; the server's solo-path protocol is unchanged in 0.1.43), 2 warm-ups, then the 12 cold prompts once each with fixed
seeds (temperature 0.7, 256 tokens). Decode tok/s is the engine's own log line; the arm's number is the median over the 12.
Every arm starts from the calibrated config (`--pcie-frac 0.35 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80
--adapt-decay 0.97 --kv-grow`) plus `--adapt-async 1`.

**Noise.** The same arm (`async43`) measured 121.2, 118.3 and 112.9 tok/s in three batches an hour apart. Prompt by prompt,
where the sampled text was identical the speed agreed within 1-2 %; the spread comes from prompts whose text diverged
(the CPU/GPU split of the missed experts is timing-dependent and rounds differently), with a different acceptance rate
and a different expert set. So a 12-prompt median carries about +-4 % of "text luck" on this build; a change is judged
against the baseline of the SAME batch, by the paired per-prompt ratio and the prompts-faster count.

## Where a window's time goes (0.1.43, `--adapt-async 1`, `STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`, prompt 1)

26.3 ms per window of 3.3 tokens = GPU-reach wait 13.9 + per-layer host (CPU experts) 9.5 + draft 1.6. The GPU side
(profiler on, so a little inflated): GDN layers 13.8 ms of kernels (qkv gemv 1.4, out-proj 1.0, shared expert 0.9, z 0.8,
router+plan 0.7, the four hyper-connection reads 3.1 at ~30 % of the memory roofline, VRAM experts 2.8), QSA layers 4.9,
head 0.9 (676 MB of Q8_0 vocabulary per window); waits: the PCIe read of the share (`waitB`) 3.1 + 1.0, the CPU rows
(`waitCPU`) 2.0 + 0.6. Per layer-window: 2.4 experts on the CPU (3.3 entries) at ~79 us each with 6 threads, 0.8 read
over PCIe at ~24 GB/s (link rate), 30 VRAM hits; decode hit rate 90-94 %.

The CPU expert kernels are bound by the i-quant codebook lookups (~5 GB/s per core, the engine's own note), not by DRAM:
SMT workers did not help (below). A missed expert serving one token of the window goes through ggml's single-token dot
(`STRATA_IQ_MT_MIN=2`, #152); on this AMD CPU the gathered grid decode is off (`cpu_gather_fast` wants GenuineIntel +
AVX-VNNI), so `STRATA_IQ3S_MT1` is inert here.

## Results (median decode tok/s over the 12 cold prompts; ratio = paired per-prompt median against the batch's baseline)

Batch 0 (`cold_baseline.jsonl`): the 0.1.43 baseline.

| Arm | tok/s | Note |
|---|---:|---|
| `async43` (`--adapt-async 1`) | **121.2** | the number to beat (0.1.42 build: 117.3) |
| `b43` (blocking tier, upstream's `STRATA_ADAPT_OVERLAP` on by default on Linux) | 83.4 | 0.1.42's blocking tier was 102.7 here: upstream's new default looks like a regression on this machine; not isolated further (our path does not use it) |

Batch 1 (`cold_sys1.jsonl`), under a system-settings bundle (sudo): `nvidia-smi -pm 1`, `-lgc 2100`, `-lmc 9751`
(the card then ran at 1920 MHz / 9501 MHz in P2 under load), CPU governor `performance` (amd-pstate-epp, was
`powersave` + EPP `balance_performance`). All reverted afterwards (`-rgc`, `-rmc`, `-pm 0`, governor back); none persist
across a reboot.

| Arm | tok/s | ratio | faster | Kept |
|---|---:|---:|---:|---|
| `async43` under the bundle | 118.3 | (vs batch 0: 0.976) | | no: the bundle is -2.4 %, reverted |
| `pm_dma` (`--pcie-mode dma`: the copy engine stages the share) | 84.0 | 0.750 | 0/12 | no |
| `pm_direct` (`--pcie-mode direct`: the kernel reads the mapped arena) | 113.1 | 0.962 | 1/12 | no |
| `pf50` (`--pcie-frac 0.5`, serial fetch) | 115.4 | 0.966 | 0/12 | no |
| `sw40` (`--adapt-swaps 40`) | 115.2 | 0.952 | 2/12 | no |

Batch 2 (`cold_cpu1.jsonl`), settings reverted:

| Arm | tok/s | ratio | faster | Kept |
|---|---:|---:|---:|---|
| `async43` | 112.9 | 1.000 | | baseline |
| `mt1` (`STRATA_IQ_MT_MIN=1`: the multi-token CPU kernels for lone tokens too; changes a lone token's rounding) | 114.9 | 1.013 | 7/12 | no (within noise) |
| `gather` (`STRATA_IQ256_GATHER=1`: vpgather grid decode, same bits) | 113.8 | 0.994 | 2/12 | no |
| `pw11` (`--pool-workers 11`: SMT siblings as workers) | 111.7 | 0.952 | 2/12 | no |

Batch 3 (`cold_branch1.jsonl`), build with `STRATA_PCIE_BRANCH` (the PCIe share's flag wait, fetch and grouped experts on
a side stream beside the VRAM hits; `=0` restores the serial order):

| Arm | tok/s | ratio | faster | Kept |
|---|---:|---:|---:|---|
| `broff` (serial, as before) | 117.1 | 1.000 | | baseline |
| `br` (branch, share 0.35) | 116.1 | 1.003 | 6/12 | - |
| `br50` (branch, `--pcie-frac 0.5`) | 112.9 | 0.998 | 4/12 | - |
| `br60` (branch, `--pcie-frac 0.6`) | 114.4 | 0.975 | 1/12 | no |
| `spec5` (`--spec 5`) | 115.9 | 0.964 | 1/12 | no |

Batch 4 (`cold_fetch1.jsonl`), build with `STRATA_FETCH_BLOCKS` (the PCIe share's fetch kernel on 64 blocks of 256
threads, four 16-byte loads per thread per step, instead of 384 blocks of one load each; `=0` restores the old launch;
a plain copy, the same bytes):

| Arm | tok/s | ratio | faster | Kept |
|---|---:|---:|---:|---|
| `old` (`STRATA_PCIE_BRANCH=0 STRATA_FETCH_BLOCKS=0`: as upstream) | 117.6 | 1.000 | | baseline |
| `fb64` (branch + 64-block fetch, share 0.35) | 118.7 | 1.010 | 12/12 | see batch 5 (second round) |
| `fb64_50` (+ `--pcie-frac 0.5`) | 113.3 | 0.998 | 6/12 | no |
| `fb16` (`STRATA_FETCH_BLOCKS=16`) | 114.9 | 1.004 | 6/12 | no |

Batch 5 (`cold_batch5.jsonl`, round 2; `old` and `fb64` paired over both rounds, n = 24):

| Arm | tok/s | ratio | faster | Kept |
|---|---:|---:|---:|---|
| `old` | 115.1 | 1.000 | | baseline |
| `fb64` (branch + 64-block fetch) | 117.8 | 1.013 (both rounds: **1.011**) | 9/12 (both rounds: 10/12 by mean) | consistent but under the 2 % bar: committed opt-in |
| `mg3` (`--adapt-min-gain 3`) | 116.1 | 1.000 | 6/12 | no |
| `ab32_50` (`STRATA_ADAPT_ASYNC_BATCH=32 --pcie-frac 0.5`: the round's copies in batches) | 112.2 | 0.952 | 1/12 | no |
| `grv3` (`STRATA_GR_V3=1`: hyper-connection read v3, a different fp32 sum order) | 115.5 | 1.019 | 8/12 | see batch 6 |
| `qfuse` (`STRATA_QFUSE=1`) | 95.4 | 0.827 | 1/12 | no |
| `vr300` (`--vram-reserve-mib 300`) | - | - | - | no: `verify: instantiate: out of memory` at start (the verify graphs need the reserve) |
| `sw120` | not run (the batch stopped at `vr300`) | | | |

(continued below as the batches land)

Batch 6 (`cold_batch6.txt`, round 3):

| Arm | tok/s | Kept |
|---|---:|---|
| `old` | 118.0 | baseline |
| `oldgrv3` (`STRATA_GR_V3=1` on the serial path) | 116.1 | no |
| `fb64grv3` (branch + 64-block fetch + `STRATA_GR_V3=1`) | 119.3 | within noise |

**Parity not shown.** `det_parity.jsonl`: greedy answers of `det_broff` and `det_br` differ on 6 of 6 prompts, but two
fresh engines of `det_broff` also differ from each other on 6 of 6, so the check does not separate the branch from the
engine's own run-to-run variation. For that reason `STRATA_PCIE_BRANCH` and `STRATA_FETCH_BLOCKS` are committed
**opt-in** (unset = the upstream path); the arms above named `br`, `fb64`, `fb16` were measured on a build where they
were on by default (now: `STRATA_PCIE_BRANCH=1 STRATA_FETCH_BLOCKS=64`).
