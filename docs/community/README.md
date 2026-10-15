# Community configurations — local V0

An opt-in extension of the existing Strata 0.1.44 portal. It reuses Outfit, Strata's
CSS tokens, cards, tabs, buttons, light/dark themes, authentication and telemetry.
No frontend framework, external CDN, online account or new Python dependency.

## Start

Use the Python environment you already use for Strata, from this checkout:

```sh
python -m serve.server --engine strata --config /absolute/path/to/your-run-config.json \
  --host 127.0.0.1 --port 8080 --community-db /absolute/path/to/community.sqlite
```

Stop the existing model server first. Two model processes may not fit in RAM.
Open `http://127.0.0.1:8080/#community`. The feature is off without `--community-db`;
no database is created and the tab stays hidden. No engine source or defaults changed.
The SQLite file is private local data; keep it out of Git. Back it up while the server
is stopped. The GPU catalog is seeded automatically from `serve/data/community-gpus.json`.

A UI-only preview needs no model:

```sh
python -m serve.server --engine mock --port 18130 --community-db /tmp/community-demo.sqlite
```

Mock runs carry `simulated: true` and are not performance evidence.

## The short workflow

1. **Share my setup**: name, optional nickname and description. GPU(s), CPU, OS, RAM,
   loaded model, context, engine version, KV format and vision are captured automatically
   from the same telemetry as Your PC / Model & engine. Manual hardware corrections are opt-in.
   Tuning details are expandable. Blank tuning fields inherit this server's settings.
2. **Share file** downloads a JSON bundle. Review its nickname, description and hardware;
   it has no executable, model paths, environment, API keys, hostname or chat history.
   There is no upload. Sharing a bundle is anonymous when the nickname is empty.
3. **Import file** adds a library entry, without applying anything. Reimporting the same
   normalized bundle is idempotent. Imported numbers remain **reported / unverified**.
4. Browse the full-width grid or list; filter by GPU, single/mixed multi-GPU, OS, date, RAM capacity, personal rating or favorites. Mobile filters collapse behind one button. Open a card for its details and actions.
5. **Test & compare** measures current settings and the selected preset with the same
   short protocol. New chat/API mutations receive HTTP 409 while the job owns the engine.
   Metrics and job progress remain readable. Cancel stops generation and restores the original
   settings; a restart already in progress must finish first.
6. **Use for this session** applies the tuning through an engine restart. **Restore previous
   settings** returns to the pre-adoption arguments. The private run config is never rewritten;
   restarting the server also returns to that file's settings.

**Share link** embeds the sanitized bundle in a portable URL. It opens an explicit import
preview; it never applies settings or starts a benchmark. Recipients use their own Strata
Community origin (adjust host/port if needed). Large configurations use JSON files instead.
URLs contain the shared metadata, so review names/descriptions before forwarding them.
No hosted link service is implied.

Favorites and stars are private to this SQLite library, not global community scores.
Cards show 0 votes or 1 local vote and the personal score; no aggregate community votes
are fabricated. A future shared backend is needed for independent voter aggregation.
Cards show dated local paired gains, not a forecast against today's settings. Imported
results remain unverified; configurations without a local pair show "Gain not measured".
The benchmark measures a short workload, not every context length.
A shared benchmark never becomes a local measurement merely by being imported.

## UI scope

The current UI focuses on discrete GPU and mixed multi-GPU machines. Existing Mac,
unified-memory and cluster profiles remain in SQLite and the interchange schema for
compatibility, but are hidden from the main grid and removed from the creation form.
No existing records are deleted. Unknown memory types are not guessed.

## Interchange compatibility: unified memory and clusters

`rig.gpus` has one entry per card, including heterogeneous GPUs and each card's VRAM.
NVIDIA/AMD telemetry readers describe the GPUs selected by the existing server, not an
inventory of every installed GPU. Unsupported detection stays Unknown. Users can correct
metadata before sharing. RAM capacity comes from OS-visible memory by default, which can
be lower than installed capacity (62.7 GiB visible on the 64 GiB test machine).
RAM type uses unprivileged Linux EDAC when available, otherwise Unknown or a user-declared
value retained locally. DDR3/4/5, LPDDR variants, Mixed and Other are accepted.

Mac mini, Mac Studio, DGX Spark and generic workstations/servers have platform and memory
architecture fields. Unified memory is not duplicated as dedicated VRAM. Reference entries
include Mac mini M4/M4 Pro and GB10, with vendor-specified unified-memory options.

A cluster has up to 16 `rig.nodes` groups, each with its own count (1–64), platform, GPU list,
RAM capacity/type and memory architecture. `interconnect` is descriptive text. No addresses,
SSH credentials or remote commands are accepted. Two 128 GiB nodes are shown as two nodes,
not a promise of a 256 GiB allocation. Mixed nodes are supported in the data format.

**V0 does not orchestrate clusters or implement a Mac inference backend.** The benchmark
runs only on the already configured Strata server. A cluster profile is a reference, and its
local test uses the current server's devices/layer split. Replica/parallel-engine tests are
refused. Single-engine multi-GPU use retains the trusted device selection; actual multi-GPU,
Mac and DGX execution needs validation on that hardware. Catalog membership is not backend
compatibility certification.

## What is tested

Allowed numeric tuning flags: `--pcie-frac`, `--spec`, `--spec-min-p`, `--adapt-every`,
`--adapt-swaps`, `--adapt-decay`, `--prompt-cache`. All other engine settings, including
model, quantization, context size, device selection, layer split, executable, paths and
environment, stay those of the local server. This is a tuning-preset format, not a portable
installer or a complete machine config. A different model name cannot be applied.

Protocol `strata-community-code-v1`:

- One fixed Python-code prompt, greedy sampling, seed 42, thinking off, 128 output tokens maximum.
- Baseline: fresh engine, then the same prompt again. Candidate: fresh engine, then same prompt again.
- Original arguments restored in `finally`; a failed restoration is prominently reported.
- Cold means fresh engine/conversation cache, **not** flushed OS/SSD caches.
- TTFT: time from local request preparation to first nonempty streamed text event; excludes engine loading.
- Decode tok/s: engine-reported completed-request throughput, or null when unavailable.
- RAM/VRAM: peak of available system/device telemetry snapshots during generation. The original sampler
  refreshes roughly every second; the job reads it every 200 ms. These are sampled peaks, not exact
  allocator peaks, not process RSS, and not startup-memory peaks. Unified-memory readings may overlap;
  never add RAM + GPU memory. Per-device peaks are retained where available.
- Output-token counts, prompt-token counts and answer hashes are saved. Differences in speed do not
  establish equal output quality. One short run is not statistically significant or a long-context test.
- Paired local percentage differences appear only for the same protocol/prompt-token count and real runs.
  Shared figures are displayed as reference values, with no extrapolated speed promise for another rig.

## Implementation and safety

`serve/community.py`: strict bundle validation, SQLite schema version 1, per-operation connections,
parameterized SQL, numeric whitelist, job admission, restart/restore and metrics.
`serve/server.py`: opt-in CLI and authenticated JSON endpoints under `/community/` with the existing
same-origin policy. Body limit 128 KiB; no remote fetching. No shell execution is introduced.
The restart uses only the original trusted process specification, replacing whitelisted numeric arguments.
The opt-in server serializes mutating API requests; the feature-disabled admission path is unchanged.

Tables: `configs`, `runs`, `gpus`, `local_meta`. JSON documents preserve extensible descriptive metadata;
IDs and fingerprints support deduplication. Imported origins and local runs are separate. For a future
online service, keep bundles/protocols versioned and add explicit publish consent, authenticated write
access, server-side rate limits and provenance; do not treat today's personal stars as a global rating.

## Validation

```sh
python -m unittest serve.test_community serve.test_runconfig serve.test_security serve.test_lifecycle serve.test_server -q
```

The first full regression pass completed **359 tests** (before the final cluster-only additions).
The final focused suite passed **12 tests**, covering heterogeneous rigs, unified-memory node groups, rejected nested clusters,
forbidden remote-address fields and concurrent preference updates.

Real smoke test on 2026-10-15: RTX 3090 24 GiB, Ryzen 5 5600X, 64 GiB DDR4,
Qwen3.8-Flash-Next IQ3_S, upstream 0.1.44 binary, configured context 262,144.
One four-request run compared 6 versus 2 conversation-cache slots. Prompt: 33 tokens;
outputs: 128 tokens each; all four answer hashes identical. Raw numbers are in
`validation-rtx3090.json`. This validates the workflow, **not** a recommendation to reduce cache slots.
Per-device peak fields were added afterwards and covered with synthetic telemetry tests, not another GPU run.

Browser checks exercise save, export/download, reimport, favorites, stars, filters and the comparison.
An imported-entry rendering bug discovered during QA was fixed; screenshots show the corrected interface.
Desktop/mobile and light/dark visual checks use the built-in browser. No CUDA/HIP/SYCL kernel changed;
no backend rebuild was needed. Mac, DGX Spark and physical multi-GPU/cluster execution were not tested.

## Design

`imagegen-concept.png` is the requested ImageGen exploration from this task. Its illustrated sample
names, stars and settings are not seeded data or measured results. The implementation follows the
existing Strata design system instead of replacing it. All new UI is scoped to the Community view.

## UX validation update — 15 October 2026

Automatic capture and safe metadata, numeric JSON deduplication, save/import/export, personal rating, benchmark/restore and API security are covered by the focused suite. Browser QA exercises grid/list, mobile filters, setup capture, portable-link confirmation and a simulated four-request benchmark with restoration. The live GPU benchmark was not repeated for this UI change.
