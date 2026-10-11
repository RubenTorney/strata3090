"""tools/profile_personal.py - an expert-cache profile from YOUR OWN prompts and conversations, mixed with the shipped one.

The shipped data/expert-profile.bin ranks the experts by a generic corpus.  This ranks them by what you ask the model:
it reads your prompts (a folder, JSONL files), has the engine decode an answer to each and record the routed experts
(`--dump-routing`, the way tools/collect_profile_traces.py does), and writes a profile `--expert-profile` reads.

    python tools/profile_personal.py CONFIG.json SOURCE [SOURCE ...] --out PATH [--base-weight 0.3] [--dry-run]

CONFIG.json is the run config setup wrote (strata-*.json).  Each SOURCE is a folder or a file:
    .jsonl / .json   one record per line (or a list), any of: {"p"|"prompt"|"text": "..."}; {"messages": [{"role",
                     "content"}...]} (OpenAI or Anthropic content blocks); one message per line {"role","content"};
                     a Claude Code transcript line {"type": "user"|"assistant", "message": {...}}.  Lines of one file
                     with a "sessionId" are one conversation; single messages of one file without it are one too.
    .txt / .md       the whole file is one user prompt.
A conversation gives one prompt per user turn: the conversation up to that turn.  `--max-prompts N` (default 40) keeps N of
them, evenly spread.  `--system-prompt FILE` puts your agent client's system prompt in front of every one (its decode then
runs in that context; it costs a prefill of its length per prompt).  The server keeps no log of prompt text, so there is
none to read: export your client's conversations, or write the prompts to a folder.

The ranking: every prompt's trace becomes a distribution (counts / total) and each prompt counts the same; the pairs are
ranked by it.  With `--base-weight W` (0-1, default 0.3 when a base exists) that ranking is blended with the base
profile's by rank: score = (1 - W) * your percentile + W * the base's, so topics you did not write about keep a place
instead of falling to the tail.  `--base-weight 0` or `--no-base` ranks by your traces alone.  `--learned FILE[:W]` adds a
profile the engine saved with `"expert_profile_save"` (docs/DETAILS.md) as one more ranking at weight W (default 1).
The pairs nobody ranked follow in make_profile.py's fill order.

`--holdout F` (default 0.2, needs 5+ prompts) builds the profile from the other prompts first and prints how much of the
held-out prompts' routing the top `--top` pairs of the base and of the new profile cover.  That is an offline coverage, not
the engine's hit rate, and it is what the decode of YOUR prompts routes, not of the prompts the engine will meet.

The engine runs once per prompt (20-60 s to start, ~10 s to decode; 40 prompts take ~30-45 minutes) and needs the VRAM:
STOP the server first.  `--dry-run` reads the inputs, tokenizes, and prints the plan without starting anything.  A prompt whose
trace already exists in OUTDIR (default: <out>.traces) is not run again.  `--traces-only` skips the engine and builds from
the traces in OUTDIR.  Not measured on a machine yet: see docs/DETAILS.md, "A personal expert profile".
"""
import argparse
import json
import os
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_profile as mp  # noqa: E402

N_PAIRS = mp.N_LAYER * mp.N_EXPERT
DEFAULT_TOP = 8441          # pairs the 24 GB card's cache holds at the default reserve (docs/DETAILS.md; see vram_budget.py)


# ---------------------------------------------------------------------------------------------------- pure: ranking

def distribution(freq):
    """A trace's counts as shares of its total ({} when empty)."""
    total = float(sum(freq.values()))
    return {p: c / total for p, c in freq.items()} if total > 0 else {}


def ranking_from_traces(traces, weights=None):
    """Pairs most-wanted first from several traces ({pair: count}), each trace counting `weight` (default 1) of one."""
    score = defaultdict(float)
    for i, freq in enumerate(traces):
        w = 1.0 if weights is None else weights[i]
        for p, s in distribution(freq).items():
            score[p] += w * s
    return [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def percentiles(ranked, n_total=N_PAIRS):
    """{pair: score in (0, 1]} by rank (rank 0 -> 1.0); a pair the list does not hold gets the mean of the tail it leaves
    (the pairs it does not rank are tied, not last)."""
    n = len(ranked)
    out = {p: 1.0 - i / n_total for i, p in enumerate(ranked)}
    out[None] = 1.0 - (n + n_total) / 2.0 / n_total if n < n_total else 0.0   # the unranked pairs' shared score
    return out


def blend_rankings(sources, n_expert=mp.N_EXPERT):
    """One ranked (layer, expert) list from [(weight, ranked list)]: the weighted sum of each list's rank percentile.
    Ties break by (layer, expert); the pairs no list ranks are appended in make_profile.py's fill order."""
    n_total = mp.N_LAYER * n_expert
    sources = [(w, r) for w, r in sources if w > 0 and r]
    tables = [(w, percentiles(r, n_total)) for w, r in sources]
    pairs = set()
    for _, r in sources:
        pairs.update(r)
    score = {}
    for p in pairs:
        score[p] = sum(w * t.get(p, t[None]) for w, t in tables)
    ranked = [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    ranked += [(layer, e) for e in range(n_expert) for layer in range(mp.N_LAYER) if (layer, e) not in seen]
    return ranked


def personal_ranking(trace_freqs, base_ranked, base_weight, learned=(), n_expert=mp.N_EXPERT):
    """The profile's ranked list: your traces (weight 1 - base_weight), the base (base_weight), each `learned` (list, weight)."""
    mine = ranking_from_traces(trace_freqs)
    sources = []
    if mine:
        sources.append((1.0 - base_weight if base_ranked else 1.0, mine))
    if base_ranked and base_weight > 0:
        sources.append((base_weight, list(base_ranked)))
    sources += [(w, list(r)) for r, w in learned]
    if not sources:
        raise ValueError("nothing to rank: no traces, no base, no learned profile")
    return blend_rankings(sources, n_expert)


def coverage(ranked, freq, top):
    """The share of a trace's routing (by count) that falls on the first `top` ranked pairs."""
    total = float(sum(freq.values()))
    if total <= 0:
        return 0.0
    keep = set(ranked[:top])
    return sum(c for p, c in freq.items() if p in keep) / total


def split_holdout(items, fraction):
    """(train, held-out): every k-th item is held out so the split is spread, not the tail; none below 5 items."""
    if fraction <= 0 or len(items) < 5:
        return list(items), []
    k = max(int(round(1.0 / fraction)), 2)
    held = [x for i, x in enumerate(items) if i % k == k - 1]
    return [x for i, x in enumerate(items) if i % k != k - 1], held


def pooled(freqs):
    out = defaultdict(int)
    for f in freqs:
        for p, c in f.items():
            out[p] += c
    return out


# ---------------------------------------------------------------------------------------------------- pure: inputs

def _text(content):
    """The text of an OpenAI string / Anthropic block-list content."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict):
            t = b.get("type")
            if t in (None, "text") and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif t == "tool_use":
                parts.append("<tool_call>\n" + json.dumps({"name": b.get("name"), "arguments": b.get("input")},
                                                           ensure_ascii=False) + "\n</tool_call>")
            elif t == "tool_result":
                parts.append("<tool_response>\n" + _text(b.get("content")) + "\n</tool_response>")
    return "\n".join(parts)


def _message(d):
    """(role, text) of a record that is one message, else None."""
    if isinstance(d.get("message"), dict) and d.get("type") in ("user", "assistant", "system"):
        d = d["message"]
    role = d.get("role")
    if role in ("user", "assistant", "system", "tool") and "content" in d:
        text = _text(d["content"])
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments")
            text += "\n<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
        if role == "tool":
            role, text = "user", "<tool_response>\n" + text + "\n</tool_response>"
        return role, text
    return None


def conversations_from_records(records):
    """Conversations ([(role, text)]) from parsed JSON records of ONE file; see the module doc for the shapes."""
    convs, loose, by_session = [], [], defaultdict(list)
    for d in records:
        if not isinstance(d, dict):
            continue
        if isinstance(d.get("messages"), list):
            msgs = [m for m in (_message(x) for x in d["messages"] if isinstance(x, dict)) if m]
            if msgs:
                convs.append(msgs)
            continue
        for key in ("p", "prompt", "text"):
            if isinstance(d.get(key), str) and d[key].strip():
                convs.append([("user", d[key])])
                break
        else:
            m = _message(d)
            if m:
                (by_session[d["sessionId"]] if d.get("sessionId") else loose).append(m)
    convs += list(by_session.values())
    if loose:
        convs.append(loose)
    return convs


def load_conversations(path):
    """Conversations of a file or a folder (recursive), in sorted file order.  Raises FileNotFoundError / ValueError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path}: no such file or folder")
    files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    convs = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix in (".txt", ".md"):
            t = f.read_text(encoding="utf-8", errors="replace")
            if t.strip():
                convs.append([("user", t)])
        elif suffix in (".jsonl", ".json"):
            raw = f.read_text(encoding="utf-8-sig", errors="replace")
            try:
                data = json.loads(raw)
                records = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                records = []
                for n, line in enumerate(raw.splitlines(), 1):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise ValueError(f"{f}:{n}: not JSON ({e.msg})")
            convs += conversations_from_records(records)
    return convs


def prompt_points(convs):
    """One candidate per user turn: the conversation up to and including it (system turns of the source are dropped)."""
    out = []
    for conv in convs:
        conv = [(r, t) for r, t in conv if r != "system" and t.strip()]
        for i, (r, _) in enumerate(conv):
            if r == "user":
                out.append(conv[:i + 1])
    return out


def pick_evenly(items, n):
    """n items spread evenly over the list, in order (all of them when there are fewer)."""
    if n <= 0 or len(items) <= n:
        return list(items)
    return [items[(i * len(items)) // n + (len(items) // n) // 2] for i in range(n)]


def render_prompt(conv, system=None, max_turns=None):
    """The chat-template text that ends where the model answers (the empty think block, as collect_profile_traces.py)."""
    if max_turns is not None and len(conv) > max_turns:
        conv = conv[-max_turns:]
        while conv and conv[0][0] != "user":
            conv = conv[1:]
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    for role, text in conv:
        s += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ---------------------------------------------------------------------------------------------------- the engine

def engine_command(cfg, tokens_file, max_new, trace):
    import collect_profile_traces as C
    args = C.drop(C.drop(C.drop(cfg["args"], "--serve"), "--batch", True), "--expert-profile-save", True)
    return [cfg["exe"]] + args + ["--tokens-file", str(tokens_file), "--max-new", str(max_new), "--greedy",
                                  "--dump-routing", str(trace)]


def engine_env(cfg):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[key] = os.pathsep.join(dirs + ([env[key]] if env.get(key) else []))
    return env


def server_listening(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def find_config(arg):
    if arg:
        return Path(arg)
    found = sorted(Path.cwd().glob("strata-*.json"))
    if len(found) != 1:
        raise SystemExit("name the config: " + (", ".join(map(str, found)) if found else "no strata-*.json here"))
    return found[0]


# ---------------------------------------------------------------------------------------------------- main

def parse_learned(spec):
    path, w = spec, 1.0
    head, sep, tail = spec.rpartition(":")
    if sep:
        try:
            path, w = head, float(tail)
        except ValueError:
            pass
    return path, w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="the run config setup wrote (strata-*.json)")
    ap.add_argument("sources", nargs="*", help="folders or files of your prompts/conversations")
    ap.add_argument("--out", help="the profile to write (required unless --dry-run)")
    ap.add_argument("--outdir", help="where the traces go (default: <out>.traces)")
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--base-weight", type=float, default=0.3, help="the base's share of the blend, 0-1 (default 0.3)")
    ap.add_argument("--learned", action="append", default=[], metavar="FILE[:W]",
                    help="a profile saved by the engine (expert_profile_save), as another ranking at weight W")
    ap.add_argument("--system-prompt", help="a file: your client's system prompt, put in front of every prompt")
    ap.add_argument("--max-prompts", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=250, help="tokens decoded per prompt (default 250)")
    ap.add_argument("--max-turns", type=int, default=12, help="turns of a conversation kept, most recent (default 12)")
    ap.add_argument("--max-prompt-tokens", type=int, default=8192, help="skip a prompt longer than this (default 8192)")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP, help="pairs the cache holds, for the coverage lines")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--traces-only", action="store_true", help="no engine: build from the traces already in OUTDIR")
    ap.add_argument("--force", action="store_true", help="run although something listens on the config's port")
    ap.add_argument("--dry-run", action="store_true", help="validate the inputs and print the plan; start nothing")
    a = ap.parse_args()
    if not 0.0 <= a.base_weight <= 1.0:
        raise SystemExit("--base-weight is a share between 0 and 1")
    if not a.dry_run and not a.out:
        raise SystemExit("--out PATH is required (the profile to write)")
    out = Path(a.out) if a.out else Path("expert-profile-personal.bin")
    outdir = Path(a.outdir) if a.outdir else out.with_name(out.name + ".traces")
    base_ranked = [] if a.no_base else mp.read_profile(a.base)
    learned = []
    for spec in a.learned:
        path, w = parse_learned(spec)
        learned.append((mp.read_profile(path), w))

    traces = []
    if a.traces_only:
        files = sorted(outdir.glob("p*.trace"))
        if not files:
            raise SystemExit(f"{outdir}: no p*.trace files")
        traces = [(f.name, mp.read_trace(f)) for f in files]
        plan = None
    else:
        if not a.sources:
            raise SystemExit("name at least one SOURCE (a folder or a file of prompts), or use --traces-only")
        convs = []
        for s in a.sources:
            try:
                convs += load_conversations(s)
            except (FileNotFoundError, ValueError) as e:
                raise SystemExit(str(e))
        points = prompt_points(convs)
        if not points:
            raise SystemExit("no user prompt found in the sources")
        chosen = pick_evenly(points, a.max_prompts)
        system = Path(a.system_prompt).read_text(encoding="utf-8") if a.system_prompt else None
        cfg_path = find_config(a.config)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        import collect_profile_traces as C
        tok = C.load_tokenizer(cfg["tokenizer"])
        sys_ids = tok.encode(f"<|im_start|>system\n{system}<|im_end|>\n", parse_special=True) if system else []
        plan = []
        for i, conv in enumerate(chosen):      # the system prompt is tokenized once: it is the same in front of all
            ids = sys_ids + tok.encode(render_prompt(conv, None, a.max_turns), parse_special=True)
            plan.append((i, ids))
        too_long = [i for i, ids in plan if len(ids) - len(sys_ids) > a.max_prompt_tokens]
        plan = [(i, ids) for i, ids in plan if i not in too_long]
        n_tok = [len(ids) for _, ids in plan]
        done = sum(1 for i, _ in plan if (outdir / f"p{i:03d}.trace").exists() and (outdir / f"p{i:03d}.trace").stat().st_size)
        print(f"{len(convs)} conversations, {len(points)} user turns, {len(chosen)} chosen, {len(too_long)} over "
              f"--max-prompt-tokens skipped -> {len(plan)} prompts to run ({done} already traced)")
        if n_tok:
            print(f"prompt tokens: min {min(n_tok)}, median {sorted(n_tok)[len(n_tok) // 2]}, max {max(n_tok)}")
        left = len(plan) - done
        print(f"engine runs: {left} x (start-up 20-60 s + prefill + decode {a.max_new} tokens): roughly "
              f"{left * 35 // 60}-{left * 80 // 60} minutes; the GPU must be free (stop the server)")
        print(f"traces -> {outdir}; profile -> {out}; ranking: yours {1 - a.base_weight if base_ranked else 1:.2f}"
              f"{f', base {a.base_weight:.2f}' if base_ranked else ''}"
              f"{''.join(f', learned {Path(p).name} {w}' for p, w in map(parse_learned, a.learned))}")
        if plan:
            tf = outdir / "p000.tok"
            print("first run: " + " ".join(engine_command(cfg, tf, a.max_new, outdir / "p000.trace")))
        if a.dry_run:
            print("dry run: nothing started, nothing written")
            return
        if not a.force and server_listening(cfg.get("port", 8080)):
            raise SystemExit(f"something listens on 127.0.0.1:{cfg.get('port', 8080)}: stop the server first (or --force)")
        outdir.mkdir(parents=True, exist_ok=True)
        env = engine_env(cfg)
        for i, ids in plan:
            trace = outdir / f"p{i:03d}.trace"
            if not (trace.exists() and trace.stat().st_size):
                tf = outdir / f"p{i:03d}.tok"
                tf.write_text(",".join(str(t) for t in ids))
                print(f"[{i + 1}/{len(plan)}] {len(ids)} prompt tokens", flush=True)
                r = subprocess.run(engine_command(cfg, tf, a.max_new, trace), cwd=cfg.get("cwd") or None, env=env,
                                   capture_output=True, text=True)
                if r.returncode != 0 or not trace.exists():
                    print(f"  failed (rc {r.returncode}): {r.stderr[-300:]}", file=sys.stderr)
                    sys.exit(1)
        traces = [(f"p{i:03d}.trace", mp.read_trace(outdir / f"p{i:03d}.trace")) for i, _ in plan]

    freqs = [f for _, f in traces if f]
    if a.dry_run:
        print(f"dry run: {len(freqs)} traces found in {outdir}; nothing written")
        return
    if not freqs:
        raise SystemExit("every trace is empty")
    train, held = split_holdout(freqs, a.holdout)
    if held and base_ranked:
        trial = personal_ranking(train, base_ranked, a.base_weight, learned)
        h = pooled(held)
        print(f"held out {len(held)} of {len(freqs)} prompts: the top {a.top} pairs cover "
              f"{coverage(base_ranked, h, a.top):.1%} of their routing with the base, "
              f"{coverage(trial, h, a.top):.1%} with a profile from the other {len(train)} (offline coverage, not the hit rate)")
    ranked = personal_ranking(freqs, base_ranked, a.base_weight, learned)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp.write_profile(out, ranked)
    assert mp.read_profile(out) == ranked, "the profile did not survive the round trip"
    allf = pooled(freqs)
    msg = f"wrote {out}: {len(ranked)} ranked pairs from {len(freqs)} traces"
    if base_ranked:
        msg += (f"; {len(set(ranked[:a.top]) & set(base_ranked[:a.top]))} of the top {a.top} are the base's "
                f"(in-sample coverage of your traces: {coverage(base_ranked, allf, a.top):.1%} -> {coverage(ranked, allf, a.top):.1%})")
    print(msg)
    print(f"use it: set \"--expert-profile\" in your config to {out.resolve()} and restart the server")


if __name__ == "__main__":
    main()
