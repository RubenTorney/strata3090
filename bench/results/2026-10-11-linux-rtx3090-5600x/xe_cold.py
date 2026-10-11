"""The 12 cold prompts of strata_ab2.py against XeStrata's packaged engine (run inside the container)."""
import copy, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path
CFG = json.loads(Path("/root/.config/xestrata/xestrata-iq3_s.json").read_text())
PORT = 8095
TUNED = ["--kv-grow", "--pcie-frac", "0.35", "--adapt-every", "1", "--adapt-swaps", "80", "--adapt-decay", "0.97"]
COLD = json.loads(Path("/data/cold_prompts.json").read_text())
WARM = ["Bonjour, présente-toi en une phrase.", "Say hello in one short sentence."]
LINE = re.compile(r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")

def ask(p, seed, max_tokens, log):
    start = log.stat().st_size
    req = {"model": "strata", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0.7,
           "top_p": 0.8, "stream": False, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(r, timeout=300).read()
    time.sleep(0.5)
    m = [LINE.search(l) for l in log.read_bytes()[start:].decode("utf-8", "replace").splitlines()]
    m = [x for x in m if x]
    return float(m[-1][3]) if m else None

for name in sys.argv[1].split(","):
    cfg = copy.deepcopy(CFG)
    if name == "tuned":
        a = cfg["args"]; a[a.index("--spec-min-p") + 1] = "0.70"; a += TUNED
    log = Path(f"/data/xe_{name}.log"); log.write_bytes(b""); cfg["log"] = str(log)
    Path("/data/xe-ab.json").write_text(json.dumps(cfg, indent=1))
    proc = subprocess.Popen(["python3", "/usr/share/xestrata/serve/server.py", "--engine", "strata", "--config", "/data/xe-ab.json",
                             "--port", str(PORT)], stdout=open(f"/data/xe_{name}.out", "wb"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read(); break
            except Exception:
                if proc.poll() is not None or time.time() - t0 > 600:
                    raise SystemExit(f"{name}: server did not start\n" + Path(f"/data/xe_{name}.out").read_bytes()[-1500:].decode("utf-8", "replace"))
                time.sleep(3)
        for i, p in enumerate(WARM):
            ask(p, 7 + i, 48, log)
        vals = [ask(p, 3000 + i, 256, log) for i, p in enumerate(COLD)]
        Path(f"/data/xe_{name}.json").write_text(json.dumps(vals))
        v = sorted(x for x in vals if x)
        print(f"xestrata {name}: median {v[len(v) // 2]:.1f} tok/s (min {v[0]:.1f}, max {v[-1]:.1f}) {vals}", flush=True)
    finally:
        os.killpg(proc.pid, signal.SIGTERM); time.sleep(8)
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        subprocess.run(["pkill", "-KILL", "-x", "strata"]); time.sleep(5)
