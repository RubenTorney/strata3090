"""tools/profile_personal.py - an expert-cache profile from YOUR OWN prompts and conversations, mixed with the shipped one.

The shipped data/expert-profile.bin ranks the experts by a generic corpus.  This ranks them by what you ask the model:
it reads your prompts (a folder, JSONL files), has the engine decode an answer to each and record the routed experts
(`--dump-routing`, the way tools/collect_profile_traces.py does), and writes a profile `--expert-profile` reads.

    python tools/profile_personal.py CONFIG.json SOURCE [SOURCE ...] --out PATH [--base-weight 0.3] [--dry-run]

CONFIG.json is the run config setup wrote (strata-*.json).  Each SOURCE is a folder or a file:
    .jsonl / .json   one record per line (or a list), any of: {"p"|"prompt"|"text": "..."}; {"messages": [{"role",
                     "content"}...]} (OpenAI, or Anthropic content blocks), with optional "system" and "tools"; one
                     message per line {"role","content"}; a Claude Code transcript line {"type": "user"|"assistant",
                     "message": {...}}.  Lines of one file with a "sessionId" are one conversation; single messages of
                     one file without it are one too.  A bad line is an error naming file and line (`--skip-bad`
                     skips and counts them).
    .txt / .md       the whole file is one user prompt.
A conversation gives one prompt per user message (a tool result is not one unless `--tool-turns`): the conversation up to
that message.  `--max-prompts N` (default 40) keeps N of them, evenly spread.  `--system-prompt FILE` (and `--tools FILE`,
a JSON list of the client's tool definitions) go in front of every prompt, as your client sends them.  The server keeps no
log of prompt text, so there is none to read: export your client's conversations, or write the prompts to a folder.

The prompt is rendered and tokenized with the SERVER's own code (serve/frontend.py: its chat template, its conversion of
OpenAI and Anthropic messages, tool blocks, merged tool results, reasoning-effort kwargs, and the rule that a control
token written inside a message stays text), so it is the prompt the server would send.  Thinking follows the server's
default (the config's "reasoning_effort", else the template's); `--think` / `--no-think` override it.  Not modelled:
"effort_position": "end" (a setting that moves the effort text), images, a client that sends other fields.

The ranking: every prompt's trace becomes a distribution (counts / total), each prompt counting the same.  The base
profile (and each `--learned FILE`, a profile the engine saved with "expert_profile_save") is a ranking without counts, so
it is turned into shares by the engine's own coverage curve (the held share of the routed mass H(f) = 1 - (1 - f)^3,
src/program/generate.cpp; `--base-curve-b`).  The profile ranks by the weighted sum of the shares, the weights
(yours 1 - W, the base W, each learned) scaled to add up to 1.  So a pair you routed once cannot outrank the base's hottest
pair, and a pair you route often can; nothing is promised for topics you did not write about beyond that the base's share
still counts for them.  `--base-weight 0` or `--no-base` ranks by your traces alone; with `--base-weight 0` the base's order
is what follows your pairs, with `--no-base` the fill order.  `--n-expert` defaults to the base's.

`--holdout F` (default 0.2, needs 5+ conversations) builds the profile from the other conversations first and prints how
much of the held-out ones' routing the top `--top` pairs of the base and of the new profile cover.  That is an offline
coverage of the model's own answers, not the engine's hit rate.  The trace holds every position the engine decoded,
including draft tokens a speculative window then rejected.

The engine runs once per prompt (20-60 s to start, ~10 s to decode; 40 prompts take ~30-45 minutes) and needs the VRAM:
STOP the server first (the tool refuses while something listens on the config's port).  A run that fails or exceeds
`--timeout` is killed with its process group, counted, and skipped; more than `--max-failed` of them (a share) is an error and
no profile is written.  `--dry-run` reads the inputs, tokenizes, and prints the plan without starting anything.  Traces go
to OUTDIR (default <out>.traces) as pNNN.trace (written to .part and renamed only on success) beside pNNN.tok, which holds
the prompt as token ids (reversible text: decode them to read the prompt); a prompt is run again unless its trace exists
AND its .tok equals the new token ids.  `--traces-only` skips the engine.  Not measured on a machine yet: see
docs/DETAILS.md, "A personal expert profile".
"""
import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import make_profile as mp  # noqa: E402

DEFAULT_TOP = 8441          # pairs the 24 GB card's cache holds at the default reserve (docs/DETAILS.md; see vram_budget.py)
COVER_B = 3.0               # the engine's coverage curve H(f) = 1 - (1 - f)^b (src/program/generate.cpp)


# ---------------------------------------------------------------------------------------------------- pure: ranking

def distribution(freq):
    """A trace's counts as shares of its total ({} when empty)."""
    total = float(sum(freq.values()))
    return {p: c / total for p, c in freq.items()} if total > 0 else {}


def mean_distribution(traces):
    """The shares of several traces, each counting the same (they add up to 1)."""
    score = defaultdict(float)
    n = sum(1 for t in traces if distribution(t))
    for freq in traces:
        for p, s in distribution(freq).items():
            score[p] += s / n
    return dict(score)


def rank_shares(ranked, n_total, b=COVER_B):
    """{pair: share} for a ranking without counts: rank r holds the increment of the coverage curve
    H(f) = 1 - (1 - f)^b, f = r / n_total, so any prefix of N pairs holds H(N / n_total) and the shares add up to 1."""
    return {p: (1.0 - r / n_total) ** b - (1.0 - (r + 1) / n_total) ** b for r, p in enumerate(ranked)}


def personal_ranking(trace_freqs, base_ranked, base_weight, learned=(), n_expert=mp.N_EXPERT, b=COVER_B):
    """The profile's ranked list from [{pair: count}] traces, the base ranking, its weight and [(ranked, weight)] more.

    score(pair) = sum of weight * share, the weights (yours 1 - base_weight, the base's, each learned) scaled to add up
    to 1.  With base_weight 0 the base's order follows the scored pairs (then the fill order for what it lacks)."""
    n_total = mp.N_LAYER * n_expert
    mine = mean_distribution(trace_freqs)
    parts = []                                   # (weight, {pair: share})
    if mine:
        parts.append((1.0 - base_weight if base_ranked else 1.0, mine))
    if base_ranked and base_weight > 0:
        parts.append((base_weight, rank_shares(base_ranked, n_total, b)))
    parts += [(w, rank_shares(r, n_total, b)) for r, w in learned if w > 0 and r]
    if not parts:
        raise ValueError("nothing to rank: no traces, no base, no learned profile")
    total_w = sum(w for w, _ in parts)
    score = defaultdict(float)
    for w, shares in parts:
        for p, s in shares.items():
            score[p] += w / total_w * s
    ranked = [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    tail = [p for p in (base_ranked if base_weight <= 0 else ()) if p not in seen]
    seen.update(tail)
    ranked += tail
    ranked += [(layer, e) for e in range(n_expert) for layer in range(mp.N_LAYER) if (layer, e) not in seen]
    return ranked


def coverage(ranked, freq, top):
    """The share of a trace's routing (by count) that falls on the first `top` ranked pairs."""
    total = float(sum(freq.values()))
    if total <= 0:
        return 0.0
    keep = set(ranked[:top])
    return sum(c for p, c in freq.items() if p in keep) / total


def split_holdout(groups, fraction):
    """[bool held-out] for a list of group ids (one per trace): WHOLE groups (conversations) are held out, every k-th
    group in order of first appearance; nothing below 5 groups."""
    order = list(dict.fromkeys(groups))
    if fraction <= 0 or len(order) < 5:
        return [False] * len(groups)
    k = max(int(round(1.0 / fraction)), 2)
    held = {g for i, g in enumerate(order) if i % k == k - 1}
    return [g in held for g in groups]


def pooled(freqs):
    out = defaultdict(int)
    for f in freqs:
        for p, c in f.items():
            out[p] += c
    return out


def read_trace_checked(path, n_expert=mp.N_EXPERT):
    """make_profile.read_trace, but a trace whose last record is cut short (or that has stray bytes) is a ValueError
    naming the file, not a struct.error or a silently short count."""
    blob = Path(path).read_bytes()
    off = 0
    try:
        while off + 8 <= len(blob):
            _layer, k = struct.unpack_from("<ii", blob, off)
            if k < 0 or k > 1024:
                raise struct.error("bad record")
            off += 8 + 8 * k
    except struct.error:
        off = -1
    if off != len(blob):
        raise ValueError(f"{path}: truncated or damaged trace (a run that was cut off): delete it")
    return mp.read_trace(path, n_expert)


def profile_n_expert(path):
    """The experts per layer in a profile's header, or None if the file is not a profile."""
    try:
        head = Path(path).read_bytes()[:24]
    except OSError:
        return None
    if len(head) < 24 or head[:4] != mp.MAGIC:
        return None
    return struct.unpack_from("<5I", head, 4)[2]


# ---------------------------------------------------------------------------------------------------- pure: inputs

TOOL_BLOCKS = ("tool_use", "tool_result", "thinking")


def _blocks(content):
    """The content as a list of blocks, or the string: a dict is one block, None is empty."""
    if content is None:
        return ""
    if isinstance(content, dict):
        return [content]
    if isinstance(content, (str, list)):
        return content
    raise ValueError(f"content is a {type(content).__name__}, not text or a list of blocks")


def _is_tool_result(msg):
    c = msg["content"]
    return msg["role"] == "tool" or (isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
                                     and not any(isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
                                                 for b in c))


def _message(d):
    """{"role", "content", ...} of a record that is one message, else None (a summary line, a metadata line)."""
    if isinstance(d.get("message"), dict) and d.get("type") in ("user", "assistant", "system"):
        d = d["message"]
    role = d.get("role")
    if role not in ("user", "assistant", "system", "tool", "developer") or ("content" not in d and not d.get("tool_calls")):
        return None
    m = {"role": "system" if role == "developer" else role, "content": _blocks(d.get("content"))}
    for k in ("tool_calls", "reasoning_content", "tool_call_id", "name"):
        if d.get(k) is not None:
            m[k] = d[k]
    return m


def _style(msgs):
    """"anthropic" when a message carries tool_use / tool_result / thinking blocks, else "openai"."""
    for m in msgs:
        if isinstance(m["content"], list) and any(isinstance(b, dict) and b.get("type") in TOOL_BLOCKS for b in m["content"]):
            return "anthropic"
    return "openai"


def _conversation(msgs, system=None, tools=None):
    sys_msgs = [m["content"] if isinstance(m["content"], str) else "\n".join(
        b.get("text", "") for b in m["content"] if isinstance(b, dict)) for m in msgs if m["role"] == "system"]
    return {"messages": [m for m in msgs if m["role"] != "system"], "system": system or ("\n\n".join(sys_msgs) or None),
            "tools": tools}


def conversations_from_records(records, name="records", bad=None):
    """Conversations ({"messages", "system", "tools"}) from [(line number, parsed JSON)] of ONE file; see the module doc.
    A malformed record raises ValueError("<name>:<line>: ..."), or with `bad` (a list) is appended to it and skipped."""
    convs, loose, by_session = [], [], defaultdict(list)
    for n, d in records:
        where = f"{name}:{n}"
        if not isinstance(d, dict):
            if bad is None:
                raise ValueError(f"{where}: a record must be a JSON object")
            bad.append(f"{where}: a record must be a JSON object")
            continue
        try:
            if "messages" in d:
                if not isinstance(d["messages"], list) or not all(isinstance(x, dict) for x in d["messages"]):
                    raise ValueError("messages must be a list of objects")
                msgs = [m for m in (_message(x) for x in d["messages"]) if m]
                tools = d.get("tools") if isinstance(d.get("tools"), list) else None
                sysp = d.get("system") if isinstance(d.get("system"), str) else None
                if msgs:
                    convs.append(_conversation(msgs, sysp, tools))
                continue
            for key in ("p", "prompt", "text"):
                if key in d:
                    if not isinstance(d[key], str):
                        raise ValueError(f"{key} must be a string")
                    if d[key].strip():
                        convs.append(_conversation([{"role": "user", "content": d[key]}]))
                    break
            else:
                m = _message(d)
                if m:
                    (by_session[d["sessionId"]] if d.get("sessionId") else loose).append(m)
        except ValueError as e:
            if bad is None:
                raise ValueError(f"{where}: {e}")
            bad.append(f"{where}: {e}")
    convs += [_conversation(v) for v in by_session.values()]
    if loose:
        convs.append(_conversation(loose))
    return convs


def load_conversations(path, skip_bad=False, bad=None):
    """Conversations of a file or a folder (recursive), in sorted file order.  Raises FileNotFoundError / ValueError;
    with skip_bad a bad line is appended to `bad` (a list) and skipped instead."""
    bad = [] if bad is None else bad
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
                convs.append(_conversation([{"role": "user", "content": t}]))
        elif suffix in (".jsonl", ".json"):
            raw = f.read_text(encoding="utf-8-sig", errors="replace")
            records = []
            try:
                data = json.loads(raw)
                records = list(enumerate(data if isinstance(data, list) else [data], 1))
            except json.JSONDecodeError:
                for n, line in enumerate(raw.splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        records.append((n, json.loads(line)))
                    except json.JSONDecodeError as e:
                        if not skip_bad:
                            raise ValueError(f"{f}:{n}: not JSON ({e.msg})")
                        bad.append(f"{f}:{n}")
            convs += conversations_from_records(records, str(f), bad if skip_bad else None)
    return convs


def prompt_points(convs, tool_turns=False):
    """[(conversation index, request)] one per user message: the conversation up to and including it.  A tool result
    (a "tool" message, or a user message with only tool_result blocks) is one only with `tool_turns`.  The request is
    {"messages", "system", "tools", "style"} in the shape serve/frontend.py's converters read."""
    out = []
    for ci, conv in enumerate(convs):
        msgs = [m for m in conv["messages"] if m["role"] in ("user", "assistant", "tool")]
        style = _style(msgs)
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                continue
            tool = _is_tool_result(m)
            if (m["role"] == "user" and not tool) or (tool_turns and tool):
                if m["content"] in ("", []):
                    continue
                out.append((ci, {"messages": msgs[:i + 1], "system": conv["system"], "tools": conv["tools"], "style": style}))
    return out


def pick_evenly(items, n):
    """n items spread evenly over the list, in order (all of them when there are fewer)."""
    if n <= 0 or len(items) <= n:
        return list(items)
    return [items[(i * len(items)) // n + (len(items) // n) // 2] for i in range(n)]


def trim_turns(msgs, max_turns):
    """The last `max_turns` messages, from a plain user message on (a tool result is never the first)."""
    if max_turns is None or len(msgs) <= max_turns:
        return msgs
    cut = msgs[-max_turns:]
    while len(cut) > 1 and not (cut[0]["role"] == "user" and not _is_tool_result(cut[0])):
        cut = cut[1:]
    return cut


# ---------------------------------------------------------------------------------------------------- the server's code

class Renderer:
    """serve/frontend.py's rendering, as the server's Service.encode_prompt does it (effort_position "end" aside)."""

    def __init__(self, tokenizer_dir, cfg, think=None):
        import serve.frontend as F
        self.F = F
        tdir = Path(tokenizer_dir)
        tpl = tdir / "chat_template.jinja"
        self.template = F.ChatTemplate(tpl if tpl.exists() else ROOT / "serve" / "chat_template.jinja")
        import collect_profile_traces as C
        self.tok = C.load_tokenizer(tdir)
        self.literals = F.literal_tags(getattr(self.tok, "control_tokens", ()))
        # the server's default: the config's reasoning_effort for a request that gives none, else the template's own
        self.kwargs = F.effort_kwargs(cfg.get("reasoning_effort"))
        if think is False:
            self.kwargs = {"enable_thinking": False}
        elif think is True:
            self.kwargs = {k: v for k, v in self.kwargs.items() if k != "enable_thinking"}

    def convert(self, req, system=None, tools=None, max_turns=None):
        F = self.F
        msgs = trim_turns(req["messages"], max_turns)
        sysp = system if system is not None else req["system"]
        body = {"messages": msgs, "tools": tools if tools is not None else req["tools"]}
        if req["style"] == "anthropic":
            if sysp:
                body["system"] = sysp
            return F.anthropic_to_messages(body)[:2]
        if sysp:
            body["messages"] = [{"role": "system", "content": sysp}] + msgs
        return F.openai_to_messages(body)[:2]

    def encode(self, req, system=None, tools=None, max_turns=None):
        F = self.F
        messages, tpl_tools = self.convert(req, system, tools, max_turns)
        marked, marked_tools, changed = F.mark_think_literals(messages, tpl_tools, self.literals)
        prompt = self.template.render(marked, tools=marked_tools, **self.kwargs)
        if not changed:
            return self.tok.encode(prompt, parse_special=True)
        prompt, plain = F.unmark_think_literals(prompt, self.literals)
        return self.tok.encode(prompt, parse_special=True, plain=plain)


# ---------------------------------------------------------------------------------------------------- the engine

STRIP_FLAGS = (("--serve", False), ("--batch", True), ("--expert-profile-save", True),
               ("--expert-profile-save-every", True), ("--vision", False))


def drop(args, flag, with_value=False):
    """`args` without every `flag` (and its value)."""
    args = list(args)
    while flag in args:
        i = args.index(flag)
        del args[i:i + (2 if with_value else 1)]
    return args


def engine_args(cfg):
    """The engine's arguments as the server builds them (serve/server.py engine_args), minus what a one-shot trace
    run does not use: --serve, --batch, the profile save, --vision (the trace is text)."""
    try:
        import serve.server as S
        args = S.engine_args(cfg)
    except Exception:                                   # the server's module is not importable: the config's own
        args = list(cfg["args"])
    for flag, valued in STRIP_FLAGS:
        args = drop(args, flag, valued)
    return args


def engine_env(cfg):
    """The engine's environment as the server builds it (serve/server.py child_env): the GPU numbering, the config's env,
    the library directories."""
    try:
        import serve.server as S
        return S.child_env(cfg)
    except Exception:
        env = dict(os.environ)
        gpu = cfg.get("gpu")
        if gpu not in (None, ""):
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x).strip() for x in (gpu if isinstance(gpu, (list, tuple)) else str(gpu).split(",")))
        env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
        dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
        if dirs:
            key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
            env[key] = os.pathsep.join(dirs + ([env[key]] if env.get(key) else []))
        return env


def resolve_exe(cfg, config_path):
    """The engine binary: a relative path is the config's cwd's (else the config's folder)."""
    exe = Path(cfg["exe"])
    if exe.is_absolute() or (os.sep not in cfg["exe"] and "/" not in cfg["exe"]):
        return str(exe)
    base = Path(cfg["cwd"]) if cfg.get("cwd") else Path(config_path).resolve().parent
    if not base.is_absolute():
        base = Path(config_path).resolve().parent / base
    return str((base / exe).resolve())


def engine_command(cfg, config_path, tokens_file, max_new, trace):
    return [resolve_exe(cfg, config_path)] + engine_args(cfg) + [
        "--tokens-file", str(tokens_file), "--max-new", str(max_new), "--greedy", "--dump-routing", str(trace)]


def arg_value(args, flag):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def listening(cfg):
    """'host:port' when something answers on the config's port, on 127.0.0.1 or on the config's host, else None."""
    hosts = ["127.0.0.1"]
    h = cfg.get("host")
    if isinstance(h, str) and h and h not in ("0.0.0.0", "::", "127.0.0.1"):
        hosts.append(h)
    for host in hosts:
        try:
            with socket.create_connection((host, int(cfg.get("port", 8080))), timeout=0.5):
                return f"{host}:{cfg.get('port', 8080)}"
        except OSError:
            pass
    return None


def kill_group(p):
    try:
        if os.name == "nt":
            p.kill()
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_engine(cmd, cwd, env, timeout):
    """Runs the engine in its own process group; (returncode, stderr tail).  A run past `timeout` seconds is killed with
    its group (no matching by name) and returns (None, ...)."""
    kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         errors="replace", **kw)
    try:
        _out, err = p.communicate(timeout=timeout)
        return p.returncode, err[-600:]
    except subprocess.TimeoutExpired:
        kill_group(p)
        _out, err = p.communicate()
        return None, err[-600:]
    except BaseException:
        kill_group(p)
        raise


# ---------------------------------------------------------------------------------------------------- main

def parse_learned(spec):
    path, w = spec, 0.5
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
    ap.add_argument("--base-curve-b", type=float, default=COVER_B, help="b of the coverage curve that gives ranks their share")
    ap.add_argument("--learned", action="append", default=[], metavar="FILE[:W]",
                    help="a profile saved by the engine (expert_profile_save), weight W beside yours+base = 1 (default 0.5)")
    ap.add_argument("--n-expert", type=int, help="experts per layer (default: the base's, else 512)")
    ap.add_argument("--system-prompt", help="a file: your client's system prompt, put in front of every prompt")
    ap.add_argument("--tools", help="a JSON file: your client's tool definitions (OpenAI or Anthropic), sent with every prompt")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                    help="--think / --no-think override the server's default (the config's reasoning_effort, else the template's)")
    ap.add_argument("--tool-turns", action="store_true", help="a tool result is a prompt point too")
    ap.add_argument("--skip-bad", action="store_true", help="skip and count unreadable input lines instead of stopping")
    ap.add_argument("--max-prompts", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=250, help="tokens decoded per prompt (default 250)")
    ap.add_argument("--max-turns", type=int, default=12, help="messages of a conversation kept, most recent (default 12)")
    ap.add_argument("--max-prompt-tokens", type=int, default=20000,
                    help="skip a prompt longer than this, system prompt included (default 20000)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds an engine run may take (default 900)")
    ap.add_argument("--max-failed", type=float, default=0.2, help="share of runs that may fail before it is an error")
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
    config_path = Path(a.config)
    if config_path.is_dir() or not config_path.is_file():
        raise SystemExit(f"{a.config}: the first argument is the run config (strata-*.json), a file")
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        cfg["exe"], cfg["args"], cfg["tokenizer"]
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"{a.config}: not a run config ({e})")
    out = Path(a.out if a.out else "expert-profile-personal.bin").resolve()
    outdir = Path(a.outdir).resolve() if a.outdir else out.with_name(out.name + ".traces")

    ne = a.n_expert or (profile_n_expert(a.base) if not a.no_base else None) or mp.N_EXPERT
    base_ranked = [] if a.no_base else mp.read_profile(a.base, ne)
    learned = []
    for spec in a.learned:
        path, w = parse_learned(spec)
        learned.append((mp.read_profile(path, ne), w))

    items = []                                   # (trace name, conversation id, {pair: count})
    failed = 0
    try:
        if a.traces_only:
            files = sorted(outdir.glob("p*.trace"))
            if not files:
                raise SystemExit(f"{outdir}: no p*.trace files")
            try:
                groups = json.loads((outdir / "plan.json").read_text())
            except (OSError, ValueError):
                groups = {}
            for f in files:
                items.append((f.name, groups.get(f.name, f.name), read_trace_checked(f, ne)))
        else:
            if not a.sources:
                raise SystemExit("name at least one SOURCE (a folder or a file of prompts), or use --traces-only")
            convs, bad = [], []
            for s in a.sources:
                try:
                    convs += load_conversations(s, a.skip_bad, bad)
                except (FileNotFoundError, ValueError) as e:
                    raise SystemExit(str(e))
            points = prompt_points(convs, a.tool_turns)
            if not points:
                raise SystemExit("no user prompt found in the sources")
            chosen = pick_evenly(points, a.max_prompts)
            system = Path(a.system_prompt).read_text(encoding="utf-8") if a.system_prompt else None
            tools = json.loads(Path(a.tools).read_text(encoding="utf-8-sig")) if a.tools else None
            try:
                rend = Renderer(cfg["tokenizer"], cfg, a.think)
            except ImportError as e:
                raise SystemExit(f"the server's renderer needs its Python packages ({e}); use the Strata venv")
            max_ctx = arg_value(cfg["args"], "--max-context")
            max_ctx = int(max_ctx) if max_ctx and max_ctx.isdigit() else None
            plan, skipped = [], defaultdict(int)
            for i, (ci, req) in enumerate(chosen):
                try:
                    ids = rend.encode(req, system, tools, a.max_turns)
                except ValueError as e:                 # the template or the converter refused these messages
                    skipped[f"refused by the server's renderer ({e})"] += 1
                    continue
                if len(ids) > a.max_prompt_tokens:
                    skipped["over --max-prompt-tokens"] += 1
                elif max_ctx and len(ids) + a.max_new > max_ctx:
                    skipped[f"prompt + --max-new over the config's --max-context {max_ctx}"] += 1
                else:
                    plan.append((i, ci, ids))
            n_tok = [len(ids) for _, _, ids in plan]
            done = 0
            for i, _, ids in plan:
                tf, tr = outdir / f"p{i:03d}.tok", outdir / f"p{i:03d}.trace"
                if tr.exists() and tr.stat().st_size and tf.exists() and tf.read_text() == ",".join(map(str, ids)):
                    done += 1
            print(f"{len(convs)} conversations, {len(points)} prompt points, {len(chosen)} chosen -> {len(plan)} to run "
                  f"({done} already traced)" + (f"; {len(bad)} unreadable lines skipped" if bad else ""))
            for why, n in skipped.items():
                print(f"  skipped {n}: {why}")
            if n_tok:
                print(f"prompt tokens (system prompt and tools included): min {min(n_tok)}, median "
                      f"{sorted(n_tok)[len(n_tok) // 2]}, max {max(n_tok)}; thinking: {rend.kwargs or 'the template default'}")
            left = len(plan) - done
            print(f"engine runs: {left} x (start-up 20-60 s + prefill + decode {a.max_new} tokens): roughly "
                  f"{left * 35 // 60}-{left * 80 // 60} minutes; the GPU must be free (stop the server)")
            mix = f"yours {1 - a.base_weight:.2f}, base {a.base_weight:.2f}" if base_ranked else "yours alone"
            print(f"traces -> {outdir}; profile -> {out}; ranking: {mix}"
                  f"{''.join(f', learned {Path(p).name} {w}' for p, w in map(parse_learned, a.learned))} (scaled to add up to 1)")
            if plan:
                print("first run: " + " ".join(engine_command(cfg, config_path, outdir / "p000.run", a.max_new,
                                                                outdir / "p000.trace.part")))
            if a.dry_run:
                print("dry run: nothing started, nothing written")
                return
            if not plan:
                raise SystemExit("no prompt is left to run")
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "plan.json").write_text(json.dumps({f"p{i:03d}.trace": f"c{ci}" for i, ci, _ in plan}))
            env, cwd = engine_env(cfg), cfg.get("cwd") or None
            for i, ci, ids in plan:
                trace, part = outdir / f"p{i:03d}.trace", outdir / f"p{i:03d}.trace.part"
                tf, run = outdir / f"p{i:03d}.tok", outdir / f"p{i:03d}.run"
                text = ",".join(map(str, ids))
                if trace.exists() and trace.stat().st_size and tf.exists() and tf.read_text() == text:
                    continue
                busy = None if a.force else listening(cfg)
                if busy:
                    raise SystemExit(f"something listens on {busy}: stop the server first (or --force)")
                trace.unlink(missing_ok=True)
                tf.unlink(missing_ok=True)
                run.write_text(text)
                print(f"[{i + 1}/{len(chosen)}] {len(ids)} prompt tokens", flush=True)
                try:
                    rc, tail = run_engine(engine_command(cfg, config_path, run, a.max_new, part), cwd, env, a.timeout)
                    ok = rc == 0 and part.exists() and part.stat().st_size > 0
                    if ok:
                        os.replace(part, trace)
                        tf.write_text(text)
                finally:
                    part.unlink(missing_ok=True)
                    run.unlink(missing_ok=True)
                if not ok:
                    failed += 1
                    print(f"  failed ({'timeout' if rc is None else 'rc %s' % rc}); the engine said:\n    "
                          + tail.strip().replace("\n", "\n    "), file=sys.stderr)
            if failed > a.max_failed * len(plan):
                raise SystemExit(f"{failed} of {len(plan)} runs failed: no profile written (finished traces kept in {outdir})")
            for i, ci, ids in plan:
                f = outdir / f"p{i:03d}.trace"
                if f.exists():
                    items.append((f.name, f"c{ci}", read_trace_checked(f, ne)))
            if failed:
                print(f"{failed} of {len(plan)} runs failed and were skipped")
    except ValueError as e:
        raise SystemExit(str(e))

    items = [it for it in items if it[2]]
    if a.dry_run:
        print(f"dry run: {len(items)} traces found in {outdir}; nothing written")
        return
    if not items:
        raise SystemExit("every trace is empty")
    freqs = [f for _, _, f in items]
    held = split_holdout([g for _, g, _ in items], a.holdout)
    train = [f for f, h in zip(freqs, held) if not h]
    test = [f for f, h in zip(freqs, held) if h]
    if test and base_ranked:
        trial = personal_ranking(train, base_ranked, a.base_weight, learned, ne, a.base_curve_b)
        h = pooled(test)
        print(f"held out {len(test)} of {len(freqs)} traces (whole conversations): the top {a.top} pairs cover "
              f"{coverage(base_ranked, h, a.top):.1%} of their routing with the base, "
              f"{coverage(trial, h, a.top):.1%} with a profile from the other {len(train)} (offline coverage, not the hit rate)")
    ranked = personal_ranking(freqs, base_ranked, a.base_weight, learned, ne, a.base_curve_b)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp.write_profile(out, ranked, ne)
    assert mp.read_profile(out, ne) == ranked, "the profile did not survive the round trip"
    allf = pooled(freqs)
    msg = f"wrote {out}: {len(ranked)} ranked pairs from {len(freqs)} traces"
    if base_ranked:
        msg += (f"; {len(set(ranked[:a.top]) & set(base_ranked[:a.top]))} of the top {a.top} are the base's "
                f"(in-sample coverage of your traces: {coverage(base_ranked, allf, a.top):.1%} -> {coverage(ranked, allf, a.top):.1%})")
    print(msg)
    print(f"use it: set \"--expert-profile\" in your config to {out} and restart the server")


if __name__ == "__main__":
    main()
