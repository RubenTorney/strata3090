"""strata_ab_linux.py - a valid A/B of two engine builds or settings on one PC (Linux), with the engine's deterministic mode.

Why: sampled text at temperature 0.7 diverges between runs, and a diverged prompt changes the work (acceptance, expert set), so
a 12-prompt median moved 121.2 / 118.3 / 112.9 tok/s for the same arm.  Under STRATA_DETERMINISTIC=1 (docs/DETAILS.md,
"Run-to-run determinism") with the cache size and the prompt chunk fixed, greedy text is the same in every fresh engine of one
build, so (1) "bit-identical" can be checked end to end, and (2) speed is compared on the same tokens.

  probe                   start one engine, print the expert-cache slots and the prompt chunk it chose (to pin them)
  parity                  --engines N fresh engines per arm, greedy: the answers' hashes per prompt must agree inside an arm
                          (else "arm not self-consistent" and nothing more is run), then across arms
  speed                   one fresh engine per arm and round, greedy: if the answers' hashes agree across arms, the paired
                          per-prompt decode ratio against the first arm, its median and the prompts-faster count; a prompt
                          whose text differs is marked "text diverged" and left out of the ratio

The server returns text, not token ids: a hash is taken over the answer text, the finish reason and the completion token count.
Equal hashes mean "the same text over N tokens", not a proof of bit-identical logits.  An empty answer is an error.

BEFORE A RUN: stop the live server.  One engine needs about 52 GB of RAM and most of the GPU; two do not fit in 62 GiB, and the
script refuses to start an engine when MemAvailable is below --min-free-ram-gb (default 50) or the free VRAM is far below what
`probe` saw, rather than risk the OOM killer.

Arms: --arm 'NAME;exe=/path/to/strata;arg=--adapt-swaps 40;env=STRATA_X=1' (repeat ; the first arm is the baseline).  Without
--arm there is one arm, "base".  Every engine is a fresh `serve/server.py --engine strata` started by this script in its own
process group and stopped through that group (and the processes under it): nothing is ever found or stopped by name.

Needs no GPU for --help / the unit tests (tools/test_strata_ab_linux.py).  It starts engines: run it only when the GPU is yours.
Environment: STRATA_DIR (install dir with serve/server.py and .venv), STRATA_AB_CONFIG (base run config), STRATA_DEV_EXE.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import statistics
import ctypes
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# the engine's per-request log line (generate.cpp, "strata serve: prompt N tokens = R reused + P read in ..."); the same
# fields serve/server.py's ENGINE_REQUEST reads
LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+)(?: of \d+)? read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in "
                  r"(\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
CACHE_LINE = re.compile(r"expert cache (\d+) slots")
CHUNK_LINE = re.compile(r"prompt chunk auto: (\d+) tokens")
PIN_LINE = re.compile(r"cudaHostRegister|registered for the GPU|pin by need|page-locked")
STRIP_KEYS = ("expert_profile_save", "expert_profile_save_every", "slot_save_path", "idle_unload_s", "lazy_load", "parallel",
              "replicas", "api_key", "before_load", "vram_elastic", "mcp_servers", "mcp")
STRIP_FLAGS = {"--vram-elastic": False, "--expert-profile-save": True, "--slot-save-path": True}
NOTE = re.compile(r"STRATA_DETERMINISTIC|\bERR\b|WARNING|tail skip|STRATA_ROUTE_TAIL_SKIP|adaptive tier|--adapt-async|pcie_frac")

COLD = [
    "Explique le fonctionnement d'une pompe à chaleur en 6 phrases.",
    "Write a Python class implementing an LRU cache with O(1) get and put, with a short usage example.",
    "Describe the main causes of the fall of the Western Roman Empire.",
    "Donne une recette détaillée de ratatouille pour 4 personnes.",
    "Explain how mRNA vaccines work, step by step.",
    "Écris une fonction JavaScript qui fait un debounce d'une autre fonction, avec des commentaires en français.",
    "Write a short project status email announcing a two-week delay and its reasons.",
    "Présente les principales caractéristiques géographiques de l'île de La Réunion.",
    "Explain the difference between permutations and combinations with three worked examples.",
    "Write a SQL query that returns the top 3 customers by revenue per country, and explain it.",
    "Résume la pensée de Descartes sur le doute méthodique.",
    "Compare TCP and UDP and give two use cases for each.",
]
WARM = ["Bonjour, présente-toi en une phrase.", "Say hello in one short sentence."]


# ---------------------------------------------------------------- pure parts (unit-tested without an engine)

def parse_engine_line(line: str) -> dict | None:
    """The engine's request line -> its numbers, or None for any other line."""
    m = LINE.search(line)
    if not m:
        return None
    return dict(prompt_tokens=int(m[1]), reused=int(m[2]), read=int(m[3]), prefill_ms=int(m[4]), prefill_tps=float(m[5]),
                gen=int(m[6]), decode_ms=int(m[7]), decode_tps=float(m[8]),
                accepted=int(m[9]) if m[9] else 0, drafted=int(m[10]) if m[10] else 0)


def parse_startup(text: str) -> dict:
    """What a log says about the start-up choices: the expert-cache slots and the auto prompt chunk (the last of each)."""
    out: dict = {}
    for m in CACHE_LINE.finditer(text):
        out["expert_cache"] = int(m[1])
    for m in CHUNK_LINE.finditer(text):
        out["prefill_chunk"] = int(m[1])
    out["pin"] = [l.strip()[:200] for l in text.splitlines() if PIN_LINE.search(l)]
    return out


def answer_hash(text: str, completion_tokens: int | None, finish: str | None = None, reasoning: str = "") -> str:
    """The server gives text, not token ids: the text, the reasoning text, the finish reason and the token count stand for the
    answer ("the same text over N tokens").  An empty answer is an error, not a match."""
    if not text and not reasoning:
        raise ValueError("empty answer")
    h = hashlib.sha256()
    for part in (text, reasoning, finish or ""):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"%d" % (-1 if completion_tokens is None else completion_tokens))
    return h.hexdigest()[:16]


def pin_mismatch(startups: list[dict]) -> bool:
    """The engines' log lines about the registered / page-locked arena differ (a different pinned prefix is a different split)."""
    return len({tuple(s.get("pin", [])) for s in startups}) > 1


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def self_consistency(hashes_by_engine: list[list[str]]) -> list[int]:
    """Prompt indexes whose hash is not the same in every engine of one arm (empty: the arm is self-consistent)."""
    if not hashes_by_engine:
        return []
    n = min(len(h) for h in hashes_by_engine)
    return [i for i in range(n) if len({h[i] for h in hashes_by_engine}) != 1]


def cross_arm(ref: list[str], other: list[str]) -> list[int]:
    """Prompt indexes where an arm's (representative) hashes differ from the baseline's."""
    return [i for i in range(min(len(ref), len(other))) if ref[i] != other[i]]


def paired_speed(base: dict[int, dict], arm: dict[int, dict]) -> dict:
    """Per prompt index: {'hash': h, 'tps': [..]} for the baseline and an arm.  A prompt whose hashes differ (or that has no
    decode number on either side) is 'text diverged' (or 'no speed') and is left out of the ratio.  Returns the ratios
    (arm / baseline, median tok/s of each side's repeats), their median, the prompts-faster count and the excluded prompts."""
    ratios: dict[int, float] = {}
    diverged: list[int] = []
    nospeed: list[int] = []
    mixed: list[int] = []
    for i in sorted(set(base) & set(arm)):
        b, a = base[i], arm[i]
        if b["hash"].startswith("*mixed*") or a["hash"].startswith("*mixed*"):
            mixed.append(i)       # repeats of one arm disagree: not self-consistent, never paired
            continue
        if b["hash"] != a["hash"]:
            diverged.append(i)
            continue
        if not b["tps"] or not a["tps"] or median(b["tps"]) <= 0:
            nospeed.append(i)
            continue
        ratios[i] = median(a["tps"]) / median(b["tps"])
    vals = list(ratios.values())
    return dict(ratios=ratios, median=median(vals) if vals else float("nan"), faster=sum(v > 1.0 for v in vals), n=len(vals),
                diverged=diverged, nospeed=nospeed, mixed=mixed)


def merge_hash(per_round: list[tuple[str, float | None]]) -> dict:
    """One arm's repeats of one prompt -> {'hash': h or '*mixed*' when the repeats differ, 'tps': [...]}."""
    hs = {h for h, _ in per_round}
    return dict(hash=hs.pop() if len(hs) == 1 else "*mixed*:" + ",".join(sorted(hs)),
                tps=[t for _, t in per_round if t is not None])


def parse_arm(spec: str) -> dict:
    """'NAME;exe=PATH;arg=--flag value;env=K=V' -> {'name','exe','args':{flag: value|None},'env':{}}."""
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    if not parts or "=" in parts[0]:
        raise ValueError(f"arm {spec!r}: the first field is its name")
    arm: dict = dict(name=parts[0], exe=None, args={}, env={})
    for p in parts[1:]:
        key, _, val = p.partition("=")
        if key == "exe":
            arm["exe"] = val
        elif key == "arg":
            toks = shlex.split(val)
            if not toks or not toks[0].startswith("-") or len(toks) > 2:
                raise ValueError(f"arm {parts[0]}: arg= takes '--flag' or '--flag value', got {val!r}")
            arm["args"][toks[0]] = toks[1] if len(toks) == 2 else None
        elif key == "env":
            k, eq, v = val.partition("=")
            if not eq or not k:
                raise ValueError(f"arm {parts[0]}: env= takes K=V, got {val!r}")
            arm["env"][k] = v
        else:
            raise ValueError(f"arm {parts[0]}: unknown field {key!r}")
    return arm


def apply_args(args: list, overrides: dict) -> list:
    """A config's engine argument list with flags set or added; a value of None is a flag without a value."""
    a = list(args)
    for flag, val in overrides.items():
        if flag in a:
            i = a.index(flag)
            if val is None:
                continue
            if i + 1 < len(a) and not str(a[i + 1]).startswith("--"):
                a[i + 1] = val
            else:
                a.insert(i + 1, val)
        else:
            a += [flag] if val is None else [flag, val]
    return a


def strip_flags(args: list) -> list:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in STRIP_FLAGS:
            i += 2 if STRIP_FLAGS[a] else 1
            continue
        out.append(a)
        i += 1
    return out


def arm_conflicts(arm: dict, deterministic: bool) -> list[str]:
    """Settings of an arm that bring back what the deterministic mode removes."""
    if not deterministic:
        return []
    bad = []
    for flag in ("--suffix-draft", "--lookup-chain"):
        if arm["args"].get(flag) not in (None, "0"):
            bad.append(f"{flag} {arm['args'][flag]}")
    if arm["args"].get("--expert-cache") == "auto":
        bad.append("--expert-cache auto")
    if arm["args"].get("--prefill") == "auto":
        bad.append("--prefill auto")
    env = arm["env"]
    for k, v in (("STRATA_DETERMINISTIC", None), ("STRATA_PREFILL_CPU_SHARE", "auto"), ("STRATA_ADAPT_ASYNC_WAIT", "0"),
                 ("STRATA_ADAPT_NOWAIT", "1"), ("STRATA_ADAPT_PICK_INLINE", "0"), ("STRATA_FS_SLOTS", None)):
        if k in env and (v is None or env[k] == v) and not (k == "STRATA_DETERMINISTIC" and env[k] == "1"):
            bad.append(f"{k}={env[k]}")
    return bad


def build_config(base: dict, arm: dict, det: dict, exe: str | None, log: str, port: int) -> dict:
    """The run config of one engine: the base config, the deterministic bundle, then the arm's own settings (the arm wins)."""
    cfg = copy.deepcopy(base)
    for k in STRIP_KEYS:     # server features that write files, unload the model, open the network or start other programs
        cfg.pop(k, None)
    cfg["host"] = "127.0.0.1"
    cfg["args"] = strip_flags(cfg.get("args", []))
    if exe or arm.get("exe"):
        cfg["exe"] = arm.get("exe") or exe
    cfg["args"] = apply_args(apply_args(cfg.get("args", []), det.get("args", {})), arm.get("args", {}))
    cfg["env"] = {**cfg.get("env", {}), **det.get("env", {}), **arm.get("env", {})}
    cfg["log"] = log
    cfg["port"] = port
    return cfg


def det_bundle(cache_slots: int | None, prefill_chunk: int | None, pcie_frac: str | None, tail_skip: str, deterministic: bool) -> dict:
    """The settings that make a fresh engine repeat itself (docs/DETAILS.md, "Run-to-run determinism")."""
    args: dict = {}
    env: dict = {}
    if deterministic:
        env["STRATA_DETERMINISTIC"] = "1"
        env["STRATA_ROUTE_TAIL_SKIP"] = tail_skip
        env["STRATA_PREFILL_CPU_SHARE"] = "0"       # unset = auto below 1024 tokens (timed means)
        args["--suffix-draft"] = "0"
        args["--lookup-chain"] = "0"
    if cache_slots:
        args["--expert-cache"] = str(cache_slots)
    if prefill_chunk:
        args["--prefill"] = str(prefill_chunk)
    if pcie_frac:
        args["--pcie-frac"] = pcie_frac
    return dict(args=args, env=env)


def descendants(pid: int, table: dict[int, int]) -> list[int]:
    """PIDs below `pid` in a {pid: ppid} table."""
    kids: dict[int, list[int]] = {}
    for p, pp in table.items():
        kids.setdefault(pp, []).append(p)
    out, stack = [], [pid]
    while stack:
        for k in kids.get(stack.pop(), []):
            out.append(k)
            stack.append(k)
    return out


# ---------------------------------------------------------------- processes and HTTP

def proc_table() -> dict[int, int]:
    table: dict[int, int] = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text()
                table[int(d.name)] = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    return table


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_until(what: str, ok, timeout: float, step: float = 1.0, abort=None) -> bool:
    """Poll `ok()` until true or `timeout` seconds (or `abort()` is true); every wait in this script goes through here."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if ok():
            return True
        if abort is not None and abort():
            break
        time.sleep(step)
    print(f"  timed out after {timeout:.0f} s waiting for {what}", flush=True)
    return False


def free_ram_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def free_vram_mib() -> int | None:
    """Free VRAM of GPU 0 from nvidia-smi (a query, no CUDA context), or None when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def starttime(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _die_with_parent() -> None:
    """In the child, before exec: SIGKILL when this script dies, so a killed driver cannot orphan an engine."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)    # PR_SET_PDEATHSIG
    except Exception:
        pass


def install_signal_handlers() -> None:
    def bye(signum, _frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, bye)


class Engine:
    """One fresh `server.py --engine strata`, in its own process group, stopped through it."""

    def __init__(self, ctx: argparse.Namespace, tag: str, cfg: dict):
        self.ctx, self.tag = ctx, tag
        self.proc = None
        work = Path(ctx.workdir)
        work.mkdir(parents=True, exist_ok=True)
        self.log = work / f"ab_{tag}.log"
        self.out = work / f"ab_{tag}.out"
        self.log.write_bytes(b"")
        cfg_path = work / f"ab_{tag}.json"
        cfg["log"] = str(self.log)
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        self.port = cfg["port"]
        if not wait_until(f"port {self.port} to be free", lambda: not port_in_use(self.port), ctx.wait_port):
            raise SystemExit(f"port {self.port} is in use (another server or engine): not starting; stop it or choose --port")
        if ctx.min_free_ram_gb and not wait_until(f"{ctx.min_free_ram_gb} GB of free RAM", lambda: free_ram_gb() >= ctx.min_free_ram_gb,
                                                  ctx.wait_ram):
            raise SystemExit(f"MemAvailable {free_ram_gb():.1f} GB < {ctx.min_free_ram_gb} GB: another engine is running? Stop the live "
                             f"server first (two engines do not fit); not starting")
        check_vram(ctx)
        py = Path(ctx.install_dir) / ".venv" / "bin" / "python"
        print(f"[{tag}] start, free RAM {free_ram_gb():.1f} GB", flush=True)
        outf = open(self.out, "wb")
        self.proc = subprocess.Popen([str(py if py.exists() else sys.executable), "serve/server.py", "--engine", "strata",
                                      "--config", str(cfg_path), "--port", str(self.port)], cwd=ctx.install_dir,
                                     stdout=outf, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_die_with_parent)
        outf.close()
        self.pgid = self.proc.pid       # start_new_session: the group id is the child's pid
        self.tree: dict[int, str | None] = {}
        try:
            t0 = time.monotonic()

            def up() -> bool:
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=5).read()
                except Exception:
                    return False
                if self.proc.poll() is not None:     # something else answers on our port
                    raise SystemExit(f"[{tag}] our server exited but port {self.port} answers: a foreign listener; stopping")
                return True
            if not wait_until("the server to answer", up, ctx.wait_start, 3.0, abort=lambda: self.proc.poll() is not None):
                tail = self.out.read_bytes()[-1500:].decode("utf-8", "replace")
                raise SystemExit(f"[{tag}] server did not start\n{tail}")
            print(f"[{tag}] up after {time.monotonic() - t0:.0f} s", flush=True)
            for p in WARM:
                self.ask(p, max_tokens=48)
        except BaseException:
            self.stop(quiet=True)
            raise

    def ask(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, seed: int | None = None) -> dict:
        req = {"model": "strata", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
               "temperature": temperature, "top_p": 1.0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        if seed:
            req["seed"] = seed
        start = self.log.stat().st_size
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(req).encode(),
                                   {"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=self.ctx.request_timeout) as resp:
            body = json.loads(resp.read())
        msg = body["choices"][0]["message"]
        text = msg.get("content") or ""
        n = (body.get("usage") or {}).get("completion_tokens")
        try:
            h = answer_hash(text, n, body["choices"][0].get("finish_reason"), msg.get("reasoning_content") or "")
        except ValueError:
            raise SystemExit(f"[{self.tag}] empty answer to {prompt[:40]!r}: the engine failed or the prompt was refused")
        row: dict = dict(text_len=len(text), hash=h, notes=[])
        got: dict | None = None

        def line_there() -> bool:          # the engine writes its line just after the reply: wait for it, bounded
            nonlocal got
            for line in self.log.read_bytes()[start:].decode("utf-8", "replace").splitlines():
                got = parse_engine_line(line) or got
            return got is not None
        wait_until("the engine's request line", line_there, 10.0, 0.25)
        if got:
            row.update(got)
        for line in self.log.read_bytes()[start:].decode("utf-8", "replace").splitlines():
            if NOTE.search(line):
                row["notes"].append(line.strip()[:300])
        return row

    def startup(self) -> dict:
        return parse_startup(self.log.read_bytes().decode("utf-8", "replace"))

    def stop(self, quiet: bool = False) -> None:
        """SIGTERM then SIGKILL to this engine's process group only; each wait is bounded.  Raises when a process that was under
        the server is still alive afterwards (a 50 GB engine must not be left behind)."""
        if self.proc is None:
            return
        for pid in descendants(self.proc.pid, proc_table()) + [self.proc.pid]:
            self.tree.setdefault(pid, starttime(pid))

        def alive() -> list[int]:
            return [p for p, st in self.tree.items() if st is not None and starttime(p) == st
                    and not (p == self.proc.pid and self.proc.poll() is not None)]
        for sig, wait in ((signal.SIGTERM, 40.0), (signal.SIGKILL, 20.0)):
            try:
                os.killpg(self.pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            if wait_until("the engine to exit", lambda: self.proc.poll() is not None and not alive(), wait):
                break
        left = alive()
        wait_until(f"port {self.port} to close", lambda: not port_in_use(self.port), 30.0)
        if left:
            raise RuntimeError(f"[{self.tag}] processes {left} are still alive after SIGKILL to the group: stop them by hand "
                               f"(they may hold the GPU and ~50 GB of RAM)")
        self.proc = None


def check_vram(ctx: argparse.Namespace) -> None:
    """Refuse when the free VRAM is far below what `probe` saw (another program is on the card)."""
    ref = None
    try:
        ref = json.loads((Path(ctx.workdir) / "probe.json").read_text()).get("free_vram_mib")
    except (OSError, ValueError):
        pass
    now = free_vram_mib()
    need = ctx.min_free_vram_mib or (int(ref * 0.9) if ref else 0)
    if now is not None and need and now < need:
        raise SystemExit(f"free VRAM {now} MiB < {need} MiB (probe saw {ref}): another program is using the GPU; not starting")


# ---------------------------------------------------------------- commands

def load_base(ctx: argparse.Namespace) -> dict:
    p = Path(ctx.config)
    if not p.is_file():
        raise SystemExit(f"base config {p} not found: --config / STRATA_AB_CONFIG")
    return json.loads(p.read_text(encoding="utf-8"))


def arms_of(ctx: argparse.Namespace) -> list[dict]:
    arms = [parse_arm(s) for s in (ctx.arm or ["base"])]
    for a in arms:
        if a["exe"]:
            a["exe"] = os.path.abspath(a["exe"])
        bad = arm_conflicts(a, not ctx.no_deterministic)
        if bad:
            raise SystemExit(f"arm {a['name']} brings back what the deterministic mode removes: {', '.join(bad)}")
    return arms


def prompts_of(ctx: argparse.Namespace) -> list[str]:
    if ctx.prompts_file:
        rows = [json.loads(l) for l in open(ctx.prompts_file, encoding="utf-8") if l.strip()]
        ps = [r if isinstance(r, str) else r["prompt"] for r in rows]
    else:
        ps = COLD
    return ps[: ctx.n_prompts] if ctx.n_prompts else ps


def make_cfg(ctx: argparse.Namespace, base: dict, arm: dict) -> dict:
    det = det_bundle(ctx.cache_slots, ctx.prefill_chunk, ctx.pcie_frac, ctx.tail_skip, not ctx.no_deterministic)
    return build_config(base, arm, det, ctx.exe, "", ctx.port)


def warn_unpinned(ctx: argparse.Namespace) -> None:
    if ctx.no_deterministic:
        print("WARNING: --no-deterministic: nothing is pinned; hashes may differ run to run", flush=True)
        return
    if not ctx.cache_slots or not ctx.prefill_chunk:
        print("WARNING: --cache-slots / --prefill-chunk not given: the expert cache and the prompt chunk follow the free VRAM at "
              "start (run `probe`, then pass both), so engines may differ", flush=True)


def run_arm_once(ctx, base, arm, tag, prompts, jsonl) -> tuple[list[dict], dict]:
    eng = Engine(ctx, tag, make_cfg(ctx, base, arm))      # a failed start stops itself
    try:
        rows = []
        for i, p in enumerate(prompts):
            r = eng.ask(p, max_tokens=ctx.max_tokens)
            r.update(arm=arm["name"], tag=tag, prompt_i=i)
            rows.append(r)
            if jsonl:
                with open(jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows, eng.startup()
    finally:
        eng.stop()


def cmd_probe(ctx) -> int:
    base = load_base(ctx)
    arm = arms_of(ctx)[0]
    vram = free_vram_mib()
    eng = Engine(ctx, "probe", make_cfg(ctx, base, arm))
    try:
        s = eng.startup()
    finally:
        eng.stop()
    s["free_vram_mib"] = vram
    (Path(ctx.workdir) / "probe.json").write_text(json.dumps(s), encoding="utf-8")
    print(json.dumps(s))
    print("pin them with: --cache-slots %s --prefill-chunk %s   (take a little less if the desktop's VRAM use varies)"
          % (s.get("expert_cache", "?"), s.get("prefill_chunk", "?")))
    return 0


def cmd_parity(ctx) -> int:
    base, arms, prompts = load_base(ctx), arms_of(ctx), prompts_of(ctx)
    warn_unpinned(ctx)
    ref: list[str] | None = None
    ref_name = ""
    worst = 0
    pins: list[dict] = []
    for arm in arms:
        runs = []
        for k in range(ctx.engines):
            rows, st = run_arm_once(ctx, base, arm, f"{arm['name']}_{k}", prompts, ctx.out)
            pins.append(st)
            runs.append([r["hash"] for r in rows])
            print(f"  {arm['name']} engine {k}: {' '.join(h[:6] for h in runs[-1])}", flush=True)
        bad = self_consistency(runs)
        if bad:
            print(f"arm not self-consistent: {arm['name']}: {len(bad)}/{len(prompts)} prompts differ between {ctx.engines} fresh "
                  f"engines (prompts {bad}); nothing more is run. The nondeterminism is not pinned by these settings "
                  f"(see docs/DETAILS.md, Run-to-run determinism).", flush=True)
            return 2
        if ctx.engines < 2:
            print(f"{arm['name']}: self-consistency NOT CHECKED (one engine; use --engines 2 or more)", flush=True)
        else:
            print(f"{arm['name']}: self-consistent over {ctx.engines} engines ({len(prompts)}/{len(prompts)} prompts)", flush=True)
        if pin_mismatch(pins):
            print("WARNING: the engines' arena registration lines differ (pinned bytes): " + str(pins), flush=True)
        if ref is None:
            ref, ref_name = runs[0], arm["name"]
            continue
        diff = cross_arm(ref, runs[0])
        print(f"{arm['name']} vs {ref_name}: the same text over the same token count on {len(prompts) - len(diff)}/{len(prompts)} "
              f"prompts" + (f"; differ: {diff}" if diff else ""), flush=True)
        worst = max(worst, 1 if diff else 0)
    return worst


def cmd_speed(ctx) -> int:
    base, arms, prompts = load_base(ctx), arms_of(ctx), prompts_of(ctx)
    warn_unpinned(ctx)
    if ctx.rounds % len(arms):
        print(f"WARNING: --rounds {ctx.rounds} is not a multiple of the {len(arms)} arms: the arm order is not balanced", flush=True)
    reps: dict[str, dict[int, list]] = {a["name"]: {} for a in arms}
    pins: list[dict] = []
    for rnd in range(ctx.rounds):
        order = arms[rnd % len(arms):] + arms[: rnd % len(arms)]     # the arm order rotates between rounds
        for arm in order:
            rows, st = run_arm_once(ctx, base, arm, f"{arm['name']}_r{rnd}", prompts, ctx.out)
            pins.append(st)
            for r in rows:
                reps[arm["name"]].setdefault(r["prompt_i"], []).append((r["hash"], r.get("decode_tps")))
            tps = [r["decode_tps"] for r in rows if r.get("decode_tps")]
            print(f"  round {rnd} {arm['name']}: median {median(tps):.1f} tok/s over {len(tps)} prompts", flush=True)
    if pin_mismatch(pins):
        print("WARNING: the engines' arena registration lines differ (pinned bytes): " + str(pins), flush=True)
    merged = {n: {i: merge_hash(v) for i, v in d.items()} for n, d in reps.items()}
    for a in arms:
        allv = [median(m["tps"]) for m in merged[a["name"]].values() if m["tps"]]
        print(f"{a['name']}: median {median(allv):.1f} tok/s over all {len(allv)} prompts (not paired)")
    b = arms[0]["name"]
    rc = 0
    for arm in arms[1:]:
        res = paired_speed(merged[b], merged[arm["name"]])
        print(f"\n{arm['name']} vs {b}: ", end="")
        if res["n"]:
            print(f"paired ratio median {res['median']:.3f}, prompts faster {res['faster']}/{res['n']}")
        else:
            print("no prompt with identical text: no ratio")
        if res["n"] * 2 < len(prompts):
            print(f"  WARNING: only {res['n']}/{len(prompts)} prompts are comparable: a ratio over a minority is not the arm's speed")
        if res["diverged"]:
            print(f"  text diverged (excluded): prompts {res['diverged']} - the arms did different work there. Excluding them "
                  f"selects the prompts where the change did not alter the text, which can bias the ratio")
            rc = max(rc, 1)
        if res["mixed"]:
            print(f"  arm not self-consistent (repeats differ, excluded): prompts {res['mixed']}")
            rc = 2
        if res["nospeed"]:
            print(f"  no decode line (excluded): prompts {res['nospeed']}")
        for i, v in sorted(res["ratios"].items()):
            print(f"    prompt {i:2d}: {v:.3f}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("command", choices=["probe", "parity", "speed"])
    ap.add_argument("--install-dir", default=os.environ.get("STRATA_DIR", ""), help="install dir with serve/server.py and .venv "
                    "(env STRATA_DIR)")
    ap.add_argument("--config", default=os.environ.get("STRATA_AB_CONFIG", ""), help="base run config JSON "
                    "(env STRATA_AB_CONFIG; default <install-dir>/strata-<model>.json must be given)")
    ap.add_argument("--exe", default=os.environ.get("STRATA_DEV_EXE"), help="engine exe for every arm without its own exe= "
                    "(env STRATA_DEV_EXE; default: the config's)")
    ap.add_argument("--arm", action="append", help="NAME;exe=..;arg=--flag value;env=K=V (repeat; first = baseline)")
    ap.add_argument("--workdir", default="ab_work", help="configs, logs and server output (default ./ab_work)")
    ap.add_argument("--out", default="", help="append one JSON line per answer here")
    ap.add_argument("--port", type=int, default=18080, help="server port (default 18080, not the 8080 a live server uses)")
    ap.add_argument("--engines", type=int, default=3, help="parity: fresh engines per arm")
    ap.add_argument("--rounds", type=int, default=1, help="speed: fresh engines per arm (arm order rotates)")
    ap.add_argument("--n-prompts", type=int, default=0, help="use the first N prompts (default all 12)")
    ap.add_argument("--prompts-file", default="", help="JSONL of prompts (strings or {prompt})")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--cache-slots", type=int, default=0, help="--expert-cache N for every engine (from `probe`)")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="--prefill N for every engine (from `probe`)")
    ap.add_argument("--pcie-frac", default="", help="--pcie-frac for every engine (default: the config's)")
    ap.add_argument("--tail-skip", default="7", help="STRATA_ROUTE_TAIL_SKIP for every engine (7 is the CUDA default when it applies)")
    ap.add_argument("--no-deterministic", action="store_true", help="do not set STRATA_DETERMINISTIC and its flags")
    ap.add_argument("--min-free-ram-gb", type=float, default=50.0, help="refuse to start an engine below this MemAvailable (0: off)")
    ap.add_argument("--min-free-vram-mib", type=int, default=0, help="refuse below this free VRAM (default 90%% of what probe saw)")
    ap.add_argument("--wait-port", type=float, default=120.0, help="seconds to wait for the port to be free")
    ap.add_argument("--wait-ram", type=float, default=120.0)
    ap.add_argument("--wait-start", type=float, default=600.0, help="seconds for the server to answer after start")
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ctx = ap.parse_args(argv)
    for k in ("install_dir", "workdir", "config", "exe", "out", "prompts_file"):
        if getattr(ctx, k):
            setattr(ctx, k, os.path.abspath(getattr(ctx, k)))     # the server runs with cwd = install dir
    if not ctx.install_dir or not (Path(ctx.install_dir) / "serve" / "server.py").is_file():
        ap.error("--install-dir (or STRATA_DIR) must hold serve/server.py")
    if not ctx.config:
        ap.error("--config (or STRATA_AB_CONFIG) is the base run config")
    install_signal_handlers()
    return {"probe": cmd_probe, "parity": cmd_parity, "speed": cmd_speed}[ctx.command](ctx)


if __name__ == "__main__":
    sys.exit(main())
