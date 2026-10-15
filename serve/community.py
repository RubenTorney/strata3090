"""Opt-in, local-only community library. No remote code, paths or environment in bundles."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LIMIT = 128 * 1024
PROTOCOL = "strata-community-code-v1"
PROMPT = "Write a Python function merge_intervals(intervals) with type hints and three assert examples. Return code only."
# Only tuning flags with numeric/enum arguments. No paths, commands, model loading or env overrides.
KNOBS = {
    "--pcie-frac": (0, 1, False), "--spec": (0, 8, True), "--spec-min-p": (0, 1, False),
    "--adapt-every": (0, 100, True), "--adapt-swaps": (0, 256, True),
    "--adapt-decay": (0, 1, False), "--prompt-cache": (0, 6, True),
}
SCHEMA = {k: {"min": v[0], "max": v[1], "integer": v[2]} for k, v in KNOBS.items()}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text(v, name, maximum=120):
    if not isinstance(v, str) or len(v) > maximum or any(ord(c) < 32 for c in v):
        raise ValueError(f"Invalid {name}")
    return v.strip()


def number(v, name, low=0, high=1e9):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f"Invalid {name}")
    return v


def settings(value):
    if not isinstance(value, dict) or set(value) - KNOBS.keys():
        raise ValueError("Only documented tuning settings are accepted; no paths, commands or environment")
    out = {}
    for k, v in value.items():
        low, high, integer = KNOBS[k]
        number(v, k, low, high)
        if integer and int(v) != v:
            raise ValueError(f"{k} requires an integer")
        out[k] = int(v) if integer else float(v)
    return out


def from_args(args):
    found = {}
    for i, x in enumerate(args[:-1]):
        if x in KNOBS:
            try:
                found[x] = float(args[i + 1])
            except ValueError:
                pass
    return settings(found)


def replace_args(args, values):
    out, i = [], 0
    while i < len(args):
        if args[i] in values:
            i += 2
        else:
            out.append(args[i]); i += 1
    for k, v in values.items():
        out.extend([k, str(v)])
    return out


RAM_TYPES = ("Unknown", "DDR3", "DDR4", "DDR5", "LPDDR4", "LPDDR4X", "LPDDR5", "LPDDR5X", "Mixed", "Other")


PLATFORMS = ("PC / workstation", "Mac mini", "Mac Studio", "DGX Spark", "Server", "Other")
MEMORY_ARCH = ("Unknown", "Discrete", "Unified", "Mixed")


def validate_rig(rig, nested=False):
    if not isinstance(rig, dict) or set(rig) - {"gpu", "gpus", "os", "ram_gib", "ram_type", "vram_gib", "cpu", "platform", "memory_architecture", "nodes", "interconnect"}:
        raise ValueError("Invalid rig")
    out = {k: text(rig.get(k, "Unknown"), k) for k in ("os", "cpu")}
    out["platform"] = rig.get("platform", "PC / workstation")
    out["memory_architecture"] = rig.get("memory_architecture", "Unknown")
    if out["platform"] not in PLATFORMS or out["memory_architecture"] not in MEMORY_ARCH:
        raise ValueError("Invalid platform or memory architecture")
    out["interconnect"] = text(rig.get("interconnect", "Unknown"), "interconnect", 80)
    nodes = rig.get("nodes", [])
    if not isinstance(nodes, list) or len(nodes) > 16 or (nested and nodes):
        raise ValueError("At most 16 node groups; no nested clusters")
    out["nodes"] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Invalid node group")
        count = node.get("count", 1)
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError("A node group has 1–64 machines")
        spec = validate_rig({k:v for k,v in node.items() if k != "count"}, nested=True)
        out["nodes"].append({**spec, "count": count})
    out["ram_type"] = rig.get("ram_type", "Unknown")
    if out["ram_type"] not in RAM_TYPES:
        raise ValueError("Unknown RAM type")
    out["ram_gib"] = None if rig.get("ram_gib") is None else number(rig["ram_gib"], "RAM", 0, 8192)
    cards = rig.get("gpus")
    if cards is None:
        # Earlier V0 files: keep them readable, without guessing a split from a concatenated name.
        cards = [{"name": rig.get("gpu", "Unknown"), "vram_gib": rig.get("vram_gib")}]
    if not isinstance(cards, list) or not 1 <= len(cards) <= 16:
        raise ValueError("A rig needs 1–16 GPU entries (one entry per physical card)")
    out["gpus"] = []
    for card in cards:
        if not isinstance(card, dict) or set(card) - {"name", "vram_gib"}:
            raise ValueError("Invalid GPU entry")
        out["gpus"].append({"name": text(card.get("name", "Unknown"), "GPU"),
                            "vram_gib": None if card.get("vram_gib") is None else number(card["vram_gib"], "VRAM", 0, 1024)})
    out["gpu"] = " + ".join(g["name"] for g in out["gpus"])
    out["vram_gib"] = sum(g["vram_gib"] for g in out["gpus"]) if all(g["vram_gib"] is not None for g in out["gpus"]) else None
    return out


def ram_type():
    # Linux EDAC may expose this without elevated privileges. Never guess from CPU generation.
    types = set()
    for file in Path("/sys/devices/system/edac/mc").glob("mc*/dimm*/dimm_mem_type"):
        try:
            value = file.read_text().strip().upper()
            if value in RAM_TYPES:
                types.add(value)
        except OSError:
            pass
    return next(iter(types)) if len(types) == 1 else "Mixed" if types else "Unknown"


def bundle(raw):
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise ValueError("Expected a Strata community bundle (schema 1)")
    allowed = {"schema", "title", "author", "description", "published", "model", "rig", "settings", "results", "runtime"}
    if set(raw) - allowed:
        raise ValueError("Unknown bundle fields; export a community bundle, not a private run config")
    rig = validate_rig(raw.get("rig", {}))
    out = {"schema": 1, "title": text(raw.get("title", ""), "title", 80),
           "author": text(raw.get("author", ""), "author", 40) or "Anonymous",
           "description": text(raw.get("description", ""), "description", 500),
           "model": text(raw.get("model", ""), "model"), "published": text(raw.get("published", now()), "date", 40),
           "settings": settings(raw.get("settings", {})), "rig": rig}
    if not out["title"] or not out["model"]:
        raise ValueError("Title and model are required")
    try:
        datetime.fromisoformat(out["published"].replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Invalid publication date") from None
    if "runtime" in raw:
        out["runtime"] = validate_runtime(raw["runtime"])
    results = raw.get("results", [])
    if not isinstance(results, list) or len(results) > 8:
        raise ValueError("At most eight reported measurements")
    out["results"] = [validate_result(r) for r in results]
    return out


def validate_runtime(raw):
    numeric = {"max_context", "expert_slots", "pool_workers", "cores", "threads"}
    strings = {"version", "kv"}
    if not isinstance(raw, dict) or set(raw) - numeric - strings - {"images"}:
        raise ValueError("Invalid runtime metadata")
    out = {}
    for key, value in raw.items():
        if key in numeric:
            out[key] = int(number(value, key, 0, 100000000))
        elif key in strings:
            out[key] = text(value, key, 80)
        else:
            if type(value) is not bool:
                raise ValueError("Invalid vision flag")
            out[key] = value
    return out


def validate_result(r):
    keys = {"protocol", "phase", "ttft_s", "decode_tok_s", "ram_peak_gib", "vram_peak_gib", "output_tokens",
            "prompt_tokens", "duration_s", "answer_sha256", "simulated", "gpu_peaks"}
    if not isinstance(r, dict) or set(r) - keys or r.get("protocol") != PROTOCOL or r.get("phase") not in ("cold", "warm"):
        raise ValueError("Unsupported measurement protocol")
    out = {"protocol": PROTOCOL, "phase": r["phase"], "simulated": r.get("simulated", False)}
    if not isinstance(out["simulated"], bool):
        raise ValueError("Invalid simulated flag")
    for k in keys - {"protocol", "phase", "simulated", "answer_sha256", "gpu_peaks"}:
        out[k] = None if r.get(k) is None else number(r[k], k)
    digest = r.get("answer_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid answer digest")
    out["answer_sha256"] = digest
    if "gpu_peaks" in r:
        peaks = r["gpu_peaks"]
        if not isinstance(peaks, list) or len(peaks) > 16:
            raise ValueError("Invalid per-GPU peaks")
        out["gpu_peaks"] = []
        for peak in peaks:
            if not isinstance(peak, dict) or set(peak) != {"index", "peak_gib"}:
                raise ValueError("Invalid per-GPU peak")
            index = number(peak["index"], "GPU index", 0, 255)
            if int(index) != index:
                raise ValueError("Invalid GPU index")
            out["gpu_peaks"].append({"index": int(index), "peak_gib": number(peak["peak_gib"], "GPU peak", 0, 1024)})
    return out


class Library:
    def __init__(self, path):
        self.path = Path(path).expanduser().absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported community database version")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS configs(id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL,
                  document TEXT NOT NULL, origin TEXT NOT NULL, added TEXT NOT NULL,
                  favorite INTEGER NOT NULL DEFAULT 0, rating INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, config_id TEXT NOT NULL,
                  created TEXT NOT NULL, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS gpus(id TEXT PRIMARY KEY, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS local_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                PRAGMA user_version=1;
            """)
            for g in json.loads(Path(__file__).with_name("data").joinpath("community-gpus.json").read_text()):
                db.execute("INSERT OR REPLACE INTO gpus VALUES(?,?)", (g["id"], json.dumps(g)))

    def db(self):
        # Each operation owns a connection; no request thread shares one.
        from contextlib import contextmanager
        @contextmanager
        def connect():
            c = sqlite3.connect(self.path, timeout=10)
            c.row_factory = sqlite3.Row
            try:
                with c:
                    yield c
            finally:
                c.close()
        return connect()

    def add(self, raw, origin="imported"):
        doc = bundle(raw)
        encoded = json.dumps(doc, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.db() as db:
            # JS JSON round trips turn 24.0 into 24. Compare numeric values, not just serialization.
            # Also accepts libraries created before this normalization without rewriting saved profiles.
            db.execute("BEGIN IMMEDIATE")
            for existing in db.execute("SELECT id,document FROM configs ORDER BY added ASC"):
                if json.loads(existing["document"]) == doc:
                    return existing["id"]
            db.execute("INSERT OR IGNORE INTO configs(id,fingerprint,document,origin,added) VALUES(?,?,?,?,?)",
                       (uuid.uuid4().hex, digest, encoded, origin, now()))
            return db.execute("SELECT id FROM configs WHERE fingerprint=?", (digest,)).fetchone()[0]

    def get(self, cid):
        with self.db() as db:
            row = db.execute("SELECT * FROM configs WHERE id=?", (cid,)).fetchone()
        if not row:
            raise ValueError("Configuration not found")
        return {**json.loads(row["document"]), **{k: row[k] for k in ("id", "origin", "favorite", "rating", "added")}}

    def list(self):
        with self.db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM configs ORDER BY added DESC, id")]
        return [self.get(i) for i in ids]

    def export(self, cid):
        row = self.get(cid)
        return {k: v for k, v in row.items() if k not in ("id", "origin", "favorite", "rating", "added")}

    def preference(self, cid, favorite=None, rating=None):
        self.get(cid)
        if (favorite is not None and type(favorite) is not bool) or (rating is not None and (type(rating) is not int or not 0 <= rating <= 5)):
            raise ValueError("Favorite must be boolean; personal rating must be 0–5")
        with self.db() as db:
            db.execute("UPDATE configs SET favorite=COALESCE(?,favorite),rating=COALESCE(?,rating) WHERE id=?", (favorite, rating, cid))

    def runs(self):
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT document FROM runs ORDER BY created DESC LIMIT 100")]

    def save_run(self, run):
        with self.db() as db:
            db.execute("INSERT INTO runs VALUES(?,?,?,?)", (run["id"], run["config_id"], run["created"], json.dumps(run)))

    def gpus(self):
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT document FROM gpus ORDER BY id")]


class Community:
    def __init__(self, svc, path):
        self.svc, self.library = svc, Library(path)
        self.gate = threading.Lock()  # admission for ALL mutating API calls, only while feature enabled
        self.lock = threading.Lock()
        self.job = {"state": "idle"}
        self.cancel = threading.Event()
        self.previous_spawn = None

    def rig(self):
        m = self.svc.metrics()
        h, s = m["hardware"], m["hardware_static"]
        cards = []
        for index, reader in getattr(getattr(self.svc, "telemetry", None), "gpus", []):
            values = reader.read() if reader.ok() else {}
            cards.append({"name": reader.name() or "Unknown", "vram_gib": values.get("mem_total", 0) / 2**30 or None})
        if not cards:
            cards = [{"name": s.get("gpu_name") or "Unknown", "vram_gib": h.get("gpu_mem_total", 0) / 2**30 or None}]
        with self.library.db() as db:
            declared = db.execute("SELECT value FROM local_meta WHERE key='ram_type'").fetchone()
        return validate_rig({"gpus": cards, "cpu": s.get("cpu_name") or "Unknown", "os": "macOS" if platform.system() == "Darwin" else platform.system(),
                             "ram_gib": h.get("ram_total", 0) / 2**30 or None,
                             "ram_type": declared[0] if declared else ram_type()})

    def runtime(self):
        metrics = self.svc.metrics()
        engine, hardware = metrics.get("engine", {}), metrics.get("hardware_static", {})
        values = {k: engine[k] for k in ("version", "max_context", "kv", "images", "expert_slots", "pool_workers") if engine.get(k) is not None}
        values.update({k: hardware[k] for k in ("cores", "threads") if hardware.get(k) is not None})
        return validate_runtime(values)

    def current(self):
        spawn = getattr(self.svc.engine, "spawn", None)
        return from_args(spawn[1]) if spawn else {}

    def state(self):
        with self.lock:
            job = copy.deepcopy(self.job)
        return {"configs": self.library.list(), "runs": self.library.runs(), "gpus": self.library.gpus(),
                "rig": self.rig(), "runtime": self.runtime(), "model": self.svc.model, "platforms": PLATFORMS, "memory_architectures": MEMORY_ARCH, "settings": self.current(), "knobs": SCHEMA,
                "job": job, "can_restore": self.previous_spawn is not None,
                "simulated": not hasattr(self.svc.engine, "spawn"), "protocol": PROTOCOL, "ram_types": RAM_TYPES}

    def dispatch(self, action, req):
        if not isinstance(req, dict):
            raise ValueError("Expected an object")
        if action == "import":
            return {"id": self.library.add(req)}
        if action == "save":
            doc = {"schema": 1, "title": req.get("title"), "author": req.get("author", ""),
                   "description": req.get("description", ""), "model": self.svc.model, "runtime": self.runtime(),
                   "rig": validate_rig(req.get("rig", self.rig())), "settings": settings(req.get("settings", self.current()))}
            cid = self.library.add(doc, "local")
            with self.library.db() as db:
                db.execute("INSERT OR REPLACE INTO local_meta VALUES('ram_type',?)", (doc["rig"]["ram_type"],))
            return {"id": cid}
        if action == "preference":
            self.library.preference(req.get("id"), req.get("favorite"), req.get("rating"))
            return {"ok": True}
        if action == "export":
            doc = self.library.export(req.get("id"))
            # Explicitly attach only a local completed run of THIS configuration.
            if req.get("run_id"):
                run = next((r for r in self.library.runs() if r["id"] == req["run_id"] and r["config_id"] == req["id"]), None)
                if not run:
                    raise ValueError("Local run not found")
                doc["results"] = run["candidate"]
                doc["rig"] = run["rig"]
                doc["settings"] = run["effective_settings"]
                if "runtime" in run:
                    doc["runtime"] = run["runtime"]
                else:
                    doc.pop("runtime", None)
            doc["author"] = text(req.get("author", ""), "author", 40) or "Anonymous"
            return doc
        if action == "cancel":
            with self.lock:
                if self.job.get("action") != "benchmark" or self.job["state"] != "running":
                    raise ValueError("No running benchmark to cancel")
            self.cancel.set()
            return {"ok": True}
        if action in ("benchmark", "adopt", "restore"):
            self.start(action, req.get("id"))
            return {"ok": True}
        raise ValueError("Unknown community action")

    def start(self, action, cid):
        selected = self.library.get(cid) if action != "restore" else None
        if selected and selected["model"] != self.svc.model:
            raise ValueError("This bundle targets a different model; it cannot be applied")
        if getattr(self.svc.engine, "batch", 0) or hasattr(self.svc.engine, "replica_view"):
            raise ValueError("V0 benchmarks support one sequential engine only")
        if action == "restore" and self.previous_spawn is None:
            raise ValueError("No previous settings to restore")
        if not self.gate.acquire(blocking=False):
            raise ValueError("The server is busy. Try again after the current request.")
        with self.svc.status_lock:
            busy = self.svc.status.get("busy") or self.svc.status.get("queued") or self.svc.live_reqs
        if busy:
            self.gate.release()
            raise ValueError("The server is busy")
        self.cancel.clear()
        with self.lock:
            self.job = {"state": "running", "phase": "Preparing", "action": action, "config_id": cid}
        threading.Thread(target=self._worker, args=(action, selected), daemon=True).start()

    def phase(self, message):
        with self.lock:
            self.job["phase"] = message

    def restart(self, spawn):
        if spawn is None:  # mock engine: integration tests, never a performance claim
            return
        # Retain the trusted executable, paths and env; only argv numbers are varied.
        with self.svc.fifo:
            self.svc.engine.spawn = copy.deepcopy(spawn)
            self.svc.engine.restart(tries=1)

    def _worker(self, action, selected):
        original = copy.deepcopy(getattr(self.svc.engine, "spawn", None))
        original_idle = self.svc.idle_unload_s
        self.svc.idle_unload_s = 0
        success = False
        try:
            candidate = copy.deepcopy(original)
            if candidate is not None and selected:
                candidate = list(candidate)
                candidate[1] = replace_args(candidate[1], selected["settings"])
                candidate = tuple(candidate)
            if action == "restore":
                self.restart(self.previous_spawn)
                self.previous_spawn = None
            elif action == "adopt":
                self.phase("Applying settings (engine restart)")
                self.restart(candidate)
                if original is not None and self.previous_spawn is None:
                    self.previous_spawn = original
            else:
                run = {"id": uuid.uuid4().hex, "config_id": selected["id"], "created": now(), "rig": self.rig(),
                       "model": self.svc.model, "runtime": self.runtime(), "protocol": PROTOCOL, "baseline_settings": self.current(),
                       "effective_settings": from_args(candidate[1]) if candidate else selected["settings"],
                       "baseline": [], "candidate": []}
                for arm, spawn in (("baseline", original), ("candidate", candidate)):
                    if self.cancel.is_set():
                        raise ValueError("Cancelled")
                    self.phase(f"{arm}: restarting engine")
                    self.restart(spawn)
                    for phase in ("cold", "warm"):
                        self.phase(f"{arm}: {phase} prompt")
                        run[arm].append(self.measure(phase))
                self.library.save_run(run)
                with self.lock:
                    self.job["run_id"] = run["id"]
            success = True
        except Exception as exc:
            # Don't expose engine log paths or secrets in shareable records / UI.
            with self.lock:
                self.job["error"] = "Cancelled" if self.cancel.is_set() else f"Operation failed ({type(exc).__name__}); see server log."
            print(f"[community] {action} failed: {exc}", flush=True)
        finally:
            if action == "benchmark" or not success:
                self.phase("Restoring original settings")
                try:
                    self.restart(original)
                except Exception as exc:
                    success = False
                    with self.lock:
                        self.job["error"] = "Restore failed. Stop testing and restart the server with your original run config."
                    print(f"[community] restore failed: {exc}", flush=True)
            self.svc.idle_unload_s = original_idle
            with self.lock:
                self.job["state"] = "done" if success else "failed"
                self.job["phase"] = "Original settings restored" if action == "benchmark" and success else action
            self.gate.release()

    def measure(self, phase):
        started, first, done = time.perf_counter(), None, None
        peaks = {"ram_peak_gib": None, "vram_peak_gib": None}
        gpu_peaks = {}
        stop = threading.Event()
        def sample():
            while not stop.is_set():
                h = self.svc.metrics()["hardware"]
                for name, key in (("ram_peak_gib", "ram_used"), ("vram_peak_gib", "gpu_mem_used")):
                    value = h.get(key)
                    if value is not None:
                        peaks[name] = max(peaks[name] or 0, value / 2**30)
                values = h.get("gpus") or [{"index": getattr(self.svc, "gpu_index", 0), "mem_used": h.get("gpu_mem_used")}]
                for gpu in values:
                    if gpu.get("mem_used") is not None:
                        index = gpu["index"]
                        gpu_peaks[index] = max(gpu_peaks.get(index, 0), gpu["mem_used"] / 2**30)
                stop.wait(.2)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        timer = threading.Timer(300, self.cancel.set)
        timer.start()
        answer = hashlib.sha256()
        try:
            req = {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": 42,
                   "reasoning_budget_tokens": 0, "experimental_speed_projection": False}
            ids, thinking, cap = self.svc.prepare([{"role": "user", "content": PROMPT}], [],
                                                  {"enable_thinking": False}, 128, req=req, cancel=self.cancel)
            for kind, value in self.svc.run(ids, thinking, [], cap, req, self.cancel):
                if kind == "event" and getattr(value, "text", None):
                    if first is None:
                        first = time.perf_counter()
                    answer.update(value.text.encode())
                elif kind == "done":
                    done = value
            if self.cancel.is_set() or done is None or first is None:
                raise ValueError("No complete answer or benchmark cancelled")
            stop.set(); sampler.join(timeout=2)
            last = self.svc.history[-1] if self.svc.history else {}
            return {"protocol": PROTOCOL, "phase": phase, "ttft_s": first - started,
                    "decode_tok_s": last.get("decode_tok_s"), "output_tokens": done["completion_tokens"],
                    "prompt_tokens": len(ids), "duration_s": time.perf_counter() - started, **peaks,
                    "gpu_peaks": [{"index": i, "peak_gib": value} for i, value in sorted(gpu_peaks.items())],
                    "answer_sha256": answer.hexdigest(), "simulated": not hasattr(self.svc.engine, "spawn")}
        finally:
            timer.cancel(); stop.set(); sampler.join(timeout=2)
