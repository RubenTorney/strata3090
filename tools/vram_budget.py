"""tools/vram_budget.py - how many experts the VRAM holds at a given context length, K/V format and reserve.

On a 24 GB card the expert cache and the K/V cache share the same VRAM: with `--kv-grow` the K/V takes VRAM only for
the cells the requests have used, and the expert cache gives slots up as it grows ("the expert cache gave 55 slots for
it" in the engine log).  A longer context therefore costs experts.  This prints that trade from files on disk:

    python tools/vram_budget.py --pack PACK_DIR --from-log strata-iq3_s.log [--profile data/expert-profile.bin]
                                [--vram-reserve-mib 700] [--kv int8|k8v4|q4_0]
                                [--contexts 32768,65536,131072,262144]

What is derived (pure arithmetic): the bytes of one expert per layer (the pack's native_experts.txt), how many of the
profile's pairs fit in a byte budget (the engine's own walk down the profile, blobs rounded up to 256 bytes), and the K/V
bytes per cell of each format (12 full-attention layers, 2 K/V heads of 256: int8 1,056 B per layer, k8v4 832 B, q4_0 576 B).

What is only known when the engine runs: the free VRAM it sees after the model is loaded, and what the prompt path and
the verify windows take.  Give them as `--free-gib` (the "expert cache auto: X GiB free" line of a start-up) and
`--other-mib`, or let `--from-log` read the last such line and work `--other-mib` out from it (free - reserve - draft head
- the slots it chose).  Without either, the absolute counts are not printed, only the experts each context costs.

Not modelled: KV streaming (`--kv-resident`: the K/V then lives in RAM), a layer split, `--no-kv-grow` (the whole K/V is then
taken at start: use the table's cost at --max-context), the drafter's own K/V.  The K/V bytes are the formula times
KV_OVERHEAD, calibrated on one engine log (int8: 1.68 GiB at 131,072 cells, which it matches; 3.35 GiB at 262,144, which it
undershoots by 2%); the other formats are the formula alone.  On that log the table's cost at 131,072 cells (7,684 experts
left) is the engine's own figure (7,684 of 8,441 slots hold experts).  The output is an estimate, not a measurement.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_profile as mp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GIB, MIB = 1 << 30, 1 << 20
N_QSA_LAYERS = 12
N_HEAD_KV, HEAD_DIM = 2, 256
KV_Q8_GROUP, QK4_0, Q4_0_BLOCK_BYTES = 64, 32, 18
KV_OVERHEAD = 1.06          # page tables, the indexer's rows and the RoPE tables, on top of the pools (see the header)
KV_ELASTIC_INIT = 16384     # cells the elastic K/V holds in VRAM from the start (the engine's g_kv_elastic_init)


def kv_bytes_per_cell(kv):
    """K/V bytes of one cell (one token) over all full-attention layers, before KV_OVERHEAD."""
    q4_head = HEAD_DIM // QK4_0 * Q4_0_BLOCK_BYTES
    if kv == "int8":
        per_layer = N_HEAD_KV * HEAD_DIM * 2 + N_HEAD_KV * (HEAD_DIM // KV_Q8_GROUP) * 2 * 2
    elif kv == "k8v4":
        per_layer = N_HEAD_KV * HEAD_DIM + N_HEAD_KV * (HEAD_DIM // KV_Q8_GROUP) * 2 + N_HEAD_KV * q4_head
    elif kv == "q4_0":
        per_layer = N_HEAD_KV * q4_head * 2
    else:
        raise ValueError(f"unknown --kv {kv!r} (int8, k8v4, q4_0)")
    return per_layer * N_QSA_LAYERS


def kv_bytes(kv, cells):
    """K/V bytes for `cells` cells, including the overhead calibrated on the int8 log."""
    return int(kv_bytes_per_cell(kv) * cells * KV_OVERHEAD)


def read_native_blobs(path):
    """Expert blob bytes per layer from a pack's native_experts.txt (columns: layer gu_type d_type offset blob_bytes ...)."""
    blobs = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        blobs[int(f[0])] = int(f[4])
    if sorted(blobs) != list(range(mp.N_LAYER)):
        raise SystemExit(f"{path}: expected the blobs of layers 0..{mp.N_LAYER - 1}, found {len(blobs)} layers")
    return [blobs[i] for i in range(mp.N_LAYER)]


def slots_footprint(blob):
    return (blob + 255) // 256 * 256


def pairs_in_budget(ranked, blobs, budget_bytes):
    """How many of the ranked pairs fit in `budget_bytes`: the engine's walk down the profile (blobs rounded up to 256)."""
    used = 0
    for n, (layer, _e) in enumerate(ranked):
        b = slots_footprint(blobs[layer])
        if used + b > budget_bytes:
            return n
        used += b
    return len(ranked)


def parse_log_anchor(text):
    """(free_gib, reserve_mib, draft_mib, slots) of the LAST `expert cache auto:` line of an engine log, or None."""
    rx = re.compile(r"expert cache auto: ([0-9.]+) GiB free, (\d+) MiB reserved \(\+(\d+) MiB for the draft head\) -> (\d+) slots")
    last = None
    for m in rx.finditer(text):
        last = m
    return None if last is None else (float(last.group(1)), int(last.group(2)), int(last.group(3)), int(last.group(4)))


def other_mib(anchor, max_blob):
    """What the start took besides the reserve, the draft head and the slots: the prompt path's buffers and the verify
    windows (not printed by the engine, so worked out from the line)."""
    free_gib, reserve, draft, slots = anchor
    return free_gib * 1024 - reserve - draft - slots * max_blob / MIB


def experts_held(ranked, blobs, free_gib, reserve_mib, draft_mib, other):
    """Pairs the cache holds when the engine starts with `free_gib` free, before any K/V growth."""
    budget = free_gib * GIB - (reserve_mib + draft_mib + other) * MIB
    return pairs_in_budget(ranked, blobs, max(int(budget), 0))


def context_table(ranked, blobs, held, kv, contexts):
    """[(cells, kv bytes beyond the initial 16,384 cells, experts given up, experts left)] for each context length.
    The experts given up are the coldest resident ones; their size is taken from the resident set's tail."""
    rows = []
    base = kv_bytes(kv, KV_ELASTIC_INIT)
    for cells in contexts:
        extra = max(kv_bytes(kv, cells) - base, 0)
        left, freed = held, 0
        while left > 0 and freed < extra:
            left -= 1
            freed += slots_footprint(blobs[ranked[left][0]])
        rows.append((cells, extra, held - left, left))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, help="the pack directory (reads native_experts.txt)")
    ap.add_argument("--profile", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--from-log", help="an engine log: the last 'expert cache auto:' line anchors the free VRAM")
    ap.add_argument("--free-gib", type=float, help="the 'GiB free' figure of the engine's start-up line")
    ap.add_argument("--other-mib", type=float, help="the prompt path's buffers and verify windows (default: from the log)")
    ap.add_argument("--vram-reserve-mib", type=int, default=None, help="the reserve to plan with (default 700; the log's own)")
    ap.add_argument("--draft-mib", type=int, default=None, help="the draft head's MiB (default: the log's, else 98)")
    ap.add_argument("--kv", choices=("int8", "k8v4", "q4_0"), default="int8")
    ap.add_argument("--contexts", default="32768,65536,131072,196608,262144")
    a = ap.parse_args()

    blobs = read_native_blobs(Path(a.pack) / "native_experts.txt")
    ranked = mp.read_profile(a.profile)
    contexts = [int(x) for x in a.contexts.split(",") if x.strip()]
    print(f"expert blobs: {min(blobs) / MIB:.2f}-{max(blobs) / MIB:.2f} MiB per expert "
          f"(mean {sum(blobs) / len(blobs) / MIB:.2f}); the profile ranks {len(ranked)} pairs")
    print(f"K/V {a.kv}: {kv_bytes_per_cell(a.kv):,} B per cell x {KV_OVERHEAD} = "
          f"{kv_bytes(a.kv, 1):,} B with the overhead; "
          f"{kv_bytes(a.kv, 262144) / GIB:.2f} GiB at 262,144 cells")

    anchor = parse_log_anchor(Path(a.from_log).read_text(encoding="utf-8", errors="replace")) if a.from_log else None
    if a.from_log and anchor is None:
        print(f"{a.from_log}: no 'expert cache auto:' line; give --free-gib and --other-mib", file=sys.stderr)
    free = a.free_gib if a.free_gib is not None else (anchor[0] if anchor else None)
    reserve = a.vram_reserve_mib if a.vram_reserve_mib is not None else (anchor[1] if anchor else 700)
    draft = a.draft_mib if a.draft_mib is not None else (anchor[2] if anchor else 98)
    other = a.other_mib if a.other_mib is not None else (other_mib(anchor, max(slots_footprint(b) for b in blobs)) if anchor else None)
    if free is None or other is None:
        held = None
        print("no anchor for the free VRAM: absolute counts not printed (pass --from-log, or --free-gib and --other-mib)")
    else:
        held = experts_held(ranked, blobs, free, reserve, draft, other)
        print(f"start: {free:.2f} GiB free, {reserve} MiB reserve, {draft} MiB draft head, {other:.0f} MiB prompt path/windows "
              f"-> {held} experts in VRAM ({held / len(ranked):.0%} of the model's) before the K/V grows")
        if anchor:
            print(f"  (the log's own start: {anchor[3]} auto slots of the largest blob; this walk, at the log's reserve, "
                  f"gives {experts_held(ranked, blobs, anchor[0], anchor[1], anchor[2], other_mib(anchor, max(slots_footprint(b) for b in blobs)))})")
    print()
    print(f"{'context':>9} {'K/V grown':>10} {'experts given up':>17} {'experts left':>13}")
    for cells, extra, gave, left in context_table(ranked, blobs, held if held is not None else len(ranked), a.kv, contexts):
        print(f"{cells:>9} {extra / GIB:>8.2f} G {gave:>17} " + (f"{left:>13}" if held is not None else f"{'-':>13}"))
    if held is not None:
        print(f"\nEach 100 MiB of --vram-reserve-mib is ~{int(100 * MIB / (sum(blobs) / len(blobs)))} experts.")
    print("An estimate from the formula and one log, not a measurement: compare with the engine's own 'K/V grown to N cells "
          "... the expert cache gave S slots' lines.")


if __name__ == "__main__":
    main()
