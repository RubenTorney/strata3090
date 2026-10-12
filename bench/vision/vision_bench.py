#!/usr/bin/env python3
"""vision_bench.py - the screenshot-to-HTML benchmark: one screenshot of a web page goes to the model with the request
to rebuild it as a single HTML file; the answer's HTML, a rendering of it and the timings are kept.

  python vision_bench.py CONFIG.json IMAGE.png OUTDIR [--exe ENGINE] [--image-tokens 300,1024] [--max-tokens 6000]
                         [--port 18081] [--install-dir ~/Documents/strata] [--tag NAME] [--think]

For each --image-tokens value it starts a fresh server (serve/server.py of the install, a copy of CONFIG with
vision.max_tokens set, its own port and log), sends the request once at temperature 0, and writes in OUTDIR:
  <tag>_<N>.html        the HTML taken from the answer
  <tag>_<N>.png         that HTML rendered by headless Firefox at the screenshot's size (if firefox is there)
  <tag>_<N>.answer.txt  the raw answer
  stats.jsonl           one line per run: image tokens, encode time, prompt and decode speed, sizes
The live server (port 8080) must be stopped first: two engines do not fit in this PC's RAM.
"""
import argparse, base64, json, os, re, shutil, signal, socket, struct, subprocess, sys, time, urllib.request
from pathlib import Path

PROMPT = ("Voici la capture d'écran d'une page web. Refais cette page en UN SEUL fichier HTML autonome "
          "(CSS dans une balise <style>, aucun fichier externe, aucune image externe). Reproduis fidèlement la mise en "
          "page, les couleurs, les tailles de texte et le contenu textuel visible. Réponds uniquement avec le code HTML "
          "complet, de <!DOCTYPE html> à </html>.")

RE_REQ = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
RE_HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)%")
RE_VIS = re.compile(r"strata-vision: .*")


def png_size(p: Path):
    b = p.read_bytes()[:24]
    return struct.unpack(">II", b[16:24])


def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def extract_html(text: str) -> str:
    m = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i = text.lower().find("<!doctype")
    if i < 0:
        i = text.lower().find("<html")
    j = text.lower().rfind("</html>")
    if i >= 0:
        text = text[i:(j + 7) if j > i else None]
    return text.strip() + "\n"


def run_one(a, cfg: dict, ntok: int, out: Path) -> dict:
    name = f"{a.tag}_{ntok}"
    c = json.loads(json.dumps(cfg))
    c["vision"]["max_tokens"] = ntok
    c["port"] = a.port
    c["log"] = str(out / f"{name}.engine.log")
    if a.exe:
        c["exe"] = a.exe
    for k in ("expert_profile_save", "slot_save_path", "idle_unload_s", "lazy_load", "api_key", "before_load"):
        c.pop(k, None)
    cpath = out / f"{name}.config.json"
    cpath.write_text(json.dumps(c, indent=1))
    Path(c["log"]).unlink(missing_ok=True)
    if not port_free(a.port):
        sys.exit(f"port {a.port} is in use")
    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    inst = Path(a.install_dir).expanduser().resolve()
    srv_out = open(out / f"{name}.server.out", "w")
    srv = subprocess.Popen([str(inst / ".venv/bin/python"), str(inst / "serve/server.py"), "--engine", "strata",
                            "--config", str(cpath), "--port", str(a.port)], cwd=str(inst), env=env,
                           stdout=srv_out, stderr=subprocess.STDOUT, start_new_session=True)
    row = {"run": name, "image_tokens_max": ntok, "image": str(a.image), "exe": c["exe"], "env": a.env}
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                sys.exit(f"{name}: the server exited (see {name}.server.out)")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=2).read()
                break
            except Exception:
                if time.time() - t0 > 600:
                    sys.exit(f"{name}: the server did not start in 600 s")
                time.sleep(2)
        row["start_s"] = round(time.time() - t0, 1)
        b64 = base64.b64encode(a.image.read_bytes()).decode()
        req = {"messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                   {"type": "text", "text": PROMPT}]}],
               "max_tokens": a.max_tokens, "temperature": 0}
        if not a.think:
            req["chat_template_kwargs"] = {"enable_thinking": False}
        t1 = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(req).encode(),
            {"content-type": "application/json"}), timeout=a.timeout))
        row["wall_s"] = round(time.time() - t1, 2)
        msg = r["choices"][0]["message"]
        text = msg.get("content") or ""
        row["finish_reason"] = r["choices"][0].get("finish_reason")
        row["usage"] = r.get("usage")
        row["reasoning_chars"] = len(msg.get("reasoning_content") or "")
        (out / f"{name}.answer.txt").write_text(text)
        html = extract_html(text)
        (out / f"{name}.html").write_text(html)
        row["html_bytes"] = len(html.encode())
        row["html_complete"] = "</html>" in html.lower()
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=60)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass
        srv_out.close()
    log = Path(c["log"]).read_text(errors="replace") if Path(c["log"]).exists() else ""
    log += (out / f"{name}.server.out").read_text(errors="replace")
    m = None
    for m in RE_REQ.finditer(log):
        pass
    if m:
        g = m.groups()
        row.update(prompt_tokens=int(g[0]), prompt_read_ms=int(g[3]), prompt_tok_s=float(g[4]),
                   gen_tokens=int(g[5]), gen_ms=int(g[6]), decode_tok_s=float(g[7]),
                   drafts_accepted=int(g[8]), drafts=int(g[9]))
        row["encode_and_queue_s"] = round(row["wall_s"] - (int(g[3]) + int(g[6])) / 1000, 2)
    h = None
    for h in RE_HIT.finditer(log):
        pass
    if h:
        row["decode_hit_rate_pct"] = float(h.group(1))
    row["vision_lines"] = [x.group(0)[:200] for x in RE_VIS.finditer(log)][-6:]
    if shutil.which("firefox") and row.get("html_bytes"):
        w, hh = png_size(a.image)
        prof = out / "ffprofile"
        prof.mkdir(exist_ok=True)
        png = out / f"{name}.png"
        subprocess.run(["firefox", "--headless", "--no-remote", "--profile", str(prof), f"--window-size={w},{hh}",
                        "--screenshot", str(png), (out / f"{name}.html").resolve().as_uri()],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        row["render"] = str(png) if png.exists() else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--exe")
    ap.add_argument("--env", action="append", default=[], help="K=V for the engine (repeat)")
    ap.add_argument("--image-tokens", default="300,1024")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--install-dir", default="~/Documents/strata")
    ap.add_argument("--tag", default="shot")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--think", action="store_true", help="leave the model's reasoning on")
    a = ap.parse_args()
    a.image = a.image.resolve()
    out = a.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())
    if "vision" not in cfg:
        sys.exit("the config has no vision section (run setup with --vision cpu or gpu)")
    for n in [int(x) for x in a.image_tokens.split(",")]:
        row = run_one(a, cfg, n, out)
        with (out / "stats.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in ("vision_lines", "usage", "env")},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
