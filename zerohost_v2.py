"""ZeroHost vLLM v2: single GPU vs data parallel vs tensor parallel on Kaggle's 2x T4.

One session runs every mode in MODES back to back on the same machine, benchmarks each one
through the load-balancing proxy with `vllm bench serve`, writes comparison.{json,md}, then
serves SERVE_MODE publicly through the ngrok static domain.

  single : one vLLM on GPU 0                          (v1 baseline, 16 slots)
  dp     : one vLLM per GPU, proxy load-balances       (2 x 16 slots)
  tp     : one vLLM split across both GPUs, TP=2       (32 slots)

Secrets (NGROK_AUTHTOKEN, NGROK_DOMAIN, VLLM_API_KEY) come from secrets.json in your private
<kaggle-username>/zerohost-secrets dataset (created by setup.py).
Outputs in DATA_DIR: requests.jsonl, metrics.jsonl, bench_<mode>_c<N>.json, comparison.*, vllm_*.log
"""
import asyncio
import glob
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
SERVED_NAME = "qwen2.5-7b-instruct"
VLLM_SPEC = "vllm==0.30.0"  # verified on Kaggle T4, driver 580 / CUDA 13.0
PROXY_PORT = 8000
DATA_DIR = "/kaggle/working/zerohost"
MODES = ["single", "tp", "dp"]  # dp last: it keeps running as the served mode, no reload
SERVE_MODE = "dp"
ENGINES = {  # per mode: one entry per vLLM process
    "single": [{"gpus": "0", "port": 8001, "tp": 1, "max_num_seqs": 16}],
    "dp": [{"gpus": "0", "port": 8001, "tp": 1, "max_num_seqs": 16},
           {"gpus": "1", "port": 8002, "tp": 1, "max_num_seqs": 16}],
    "tp": [{"gpus": "0,1", "port": 8001, "tp": 2, "max_num_seqs": 32}],  # 32 slots, same as dp
}
BENCH_CONCURRENCY = [1, 4, 8, 16, 32]
BENCH_INPUT, BENCH_OUTPUT = 512, 128
BENCH_TIMEOUT_S = 10 * 60
STARTUP_TIMEOUT_S = 20 * 60
CHECK_INTERVAL_S = 30
METRICS_INTERVAL_S = 15
RUN_HOURS = 11.75  # exit before Kaggle's 12h kill so /kaggle/working is saved as output

# shared with the proxy thread; the main thread swaps these when switching modes
STATE = {"mode": None, "upstreams": []}


def log(msg):
    print(f"[zerohost {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_jsonl(name, record, lock=threading.Lock()):
    with lock, open(f"{DATA_DIR}/{name}", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def install():
    log(f"installing {VLLM_SPEC} and pyngrok")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", VLLM_SPEC, "pyngrok"], check=True)


def load_secrets():
    # CLI pushes drop attached Kaggle Secrets, so they come from the private zerohost-secrets dataset
    paths = glob.glob("/kaggle/input/**/secrets.json", recursive=True)
    if not paths:
        raise FileNotFoundError("secrets.json not found; attach the private zerohost-secrets dataset")
    with open(paths[0]) as f:
        return json.load(f)


def diagnostics():
    for cmd in (["nvidia-smi"], ["nvidia-smi", "topo", "-m"], ["df", "-h", "/dev/shm", "/kaggle/working"]):
        subprocess.run(cmd)


# ---------------------------------------------------------------- engines

def engine_log(mode, cfg):
    return f"{DATA_DIR}/vllm_{mode}_{cfg['port']}.log"


def start_engine(mode, cfg, api_key):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--host", "127.0.0.1",
        "--port", str(cfg["port"]),
        "--dtype", "half",  # T4 (sm75) has no bf16
        "--quantization", "awq",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--tensor-parallel-size", str(cfg["tp"]),
        "--served-model-name", SERVED_NAME,
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cfg["gpus"], "VLLM_API_KEY": api_key}
    log(f"[{mode}] starting vLLM on GPU {cfg['gpus']} :{cfg['port']} (tp={cfg['tp']}, seqs={cfg['max_num_seqs']})")
    logfile = open(engine_log(mode, cfg), "a")
    return subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def gpu_memory_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def stop_engines(procs):
    """Kill each engine's whole process group (the EngineCore child holds the GPU), then wait for VRAM."""
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 60
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(2)
    for proc in procs:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    deadline = time.time() + 60
    while time.time() < deadline and any(m > 500 for m in gpu_memory_mib()):
        time.sleep(2)
    log(f"engines stopped, GPU memory now {gpu_memory_mib()} MiB")


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def tail(path, n=40):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def wait_until_healthy(mode, procs, cfgs):
    deadline = time.time() + STARTUP_TIMEOUT_S
    pending = list(zip(procs, cfgs))
    while pending and time.time() < deadline:
        for proc, cfg in list(pending):
            if proc.poll() is not None:
                raise RuntimeError(f"[{mode}] vLLM :{cfg['port']} exited ({proc.returncode}):\n"
                                   f"{tail(engine_log(mode, cfg))}")
            if healthy(cfg["port"]):
                pending.remove((proc, cfg))
        time.sleep(10)
    if pending:
        raise TimeoutError(f"[{mode}] not healthy after {STARTUP_TIMEOUT_S}s:\n"
                           f"{tail(engine_log(mode, pending[0][1]))}")
    log(f"[{mode}] all engines healthy")


def start_mode(mode, api_key):
    cfgs = ENGINES[mode]
    procs = [start_engine(mode, cfg, api_key) for cfg in cfgs]
    try:
        wait_until_healthy(mode, procs, cfgs)
    except Exception:
        stop_engines(procs)
        raise
    STATE["upstreams"] = [{"port": c["port"], "inflight": 0, "down_until": 0.0} for c in cfgs]
    STATE["mode"] = mode
    return procs


# ---------------------------------------------------------------- benchmark + comparison

def run_benchmarks(mode, api_key):
    """Load-test through the proxy so every mode pays the same proxy cost and dp exercises the balancer."""
    env = {**os.environ, "OPENAI_API_KEY": api_key}
    for c in BENCH_CONCURRENCY:
        name = f"bench_{mode}_c{c}.json"
        try:
            result = subprocess.run([
                sys.executable, "-c", "from vllm.entrypoints.cli.main import main; main()", "bench", "serve",
                "--base-url", f"http://127.0.0.1:{PROXY_PORT}",
                "--model", SERVED_NAME, "--tokenizer", MODEL,
                "--dataset-name", "random", "--random-input-len", str(BENCH_INPUT),
                "--random-output-len", str(BENCH_OUTPUT), "--ignore-eos",
                "--num-prompts", str(max(32, 6 * c)), "--max-concurrency", str(c),
                "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
                "--save-result", "--result-dir", DATA_DIR, "--result-filename", name,
            ], env=env, capture_output=True, text=True, timeout=BENCH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log(f"[{mode}] bench c={c} timed out, skipping the rest of this mode")
            return
        if result.returncode != 0:
            log(f"[{mode}] bench c={c} failed:\n{(result.stdout + result.stderr)[-1500:]}")
            return
        r = json.load(open(f"{DATA_DIR}/{name}"))
        log(f"[{mode}] c={c:>2}  out {r['output_throughput']:7.1f} tok/s  req {r['request_throughput']:5.2f}/s  "
            f"TTFT p50 {r['median_ttft_ms']:7.0f} p95 {r['p95_ttft_ms']:7.0f} ms  "
            f"ITL p50 {r['median_itl_ms']:5.1f} ms")


COMPARE_FIELDS = ["output_throughput", "request_throughput", "median_ttft_ms", "p95_ttft_ms",
                  "median_itl_ms", "p95_itl_ms", "median_tpot_ms", "median_e2el_ms"]


def write_comparison():
    rows = []
    for mode in MODES:
        for c in BENCH_CONCURRENCY:
            path = f"{DATA_DIR}/bench_{mode}_c{c}.json"
            if os.path.exists(path):
                r = json.load(open(path))
                rows.append({"mode": mode, "concurrency": c, **{k: r.get(k) for k in COMPARE_FIELDS}})
    with open(f"{DATA_DIR}/comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    header = "| mode | conc | out tok/s | req/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | ITL p95 ms | TPOT p50 ms | E2E p50 ms |"
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        vals = [r[k] or 0 for k in COMPARE_FIELDS]
        lines.append(f"| {r['mode']} | {r['concurrency']} | {vals[0]:.1f} | {vals[1]:.2f} | {vals[2]:.0f} | "
                     f"{vals[3]:.0f} | {vals[4]:.1f} | {vals[5]:.1f} | {vals[6]:.1f} | {vals[7]:.0f} |")
    with open(f"{DATA_DIR}/comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        log(f"COMPARE {line}")


# ---------------------------------------------------------------- metrics

def sample_metrics():
    """Every METRICS_INTERVAL_S: vLLM Prometheus counters per engine port, plus both GPUs."""
    while True:
        record = {"ts": time.time(), "mode": STATE["mode"]}
        for up in list(STATE["upstreams"]):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{up['port']}/metrics", timeout=5) as r:
                    for line in r.read().decode().splitlines():
                        if line.startswith("vllm:") and "_bucket{" not in line and "_created{" not in line:
                            name, value = line.rsplit(" ", 1)
                            record[f"p{up['port']}:{name}"] = float(value)
            except Exception as e:
                record[f"p{up['port']}:error"] = str(e)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            for line in out.strip().splitlines():
                i, util, mem, temp, power = (x.strip() for x in line.split(","))
                record.update({f"gpu{i}_util_pct": float(util), f"gpu{i}_mem_mib": float(mem),
                               f"gpu{i}_temp_c": float(temp), f"gpu{i}_power_w": float(power)})
        except Exception as e:
            record["gpu_error"] = str(e)
        append_jsonl("metrics.jsonl", record)
        time.sleep(METRICS_INTERVAL_S)


# ---------------------------------------------------------------- load-balancing proxy

def summarize_response(raw, content_type):
    """Pull output text, usage and finish_reason out of a JSON or SSE completion response."""
    if "text/event-stream" in content_type:
        text, usage, finish = [], None, None
        for line in raw.decode(errors="replace").splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                text.append((choice.get("delta") or {}).get("content") or choice.get("text") or "")
                finish = choice.get("finish_reason") or finish
        return {"output": "".join(text), "usage": usage, "finish_reason": finish}
    body = json.loads(raw)
    choice = (body.get("choices") or [{}])[0]
    output = (choice.get("message") or {}).get("content") or choice.get("text")
    return {"output": output, "usage": body.get("usage"), "finish_reason": choice.get("finish_reason")}


_rr = itertools.count()


def pick_upstream(exclude=()):
    """Least outstanding requests among healthy upstreams; round-robin breaks ties."""
    ups = [u for u in STATE["upstreams"] if u["port"] not in exclude]
    live = [u for u in ups if u["down_until"] < time.time()] or ups
    if not live:
        return None
    least = min(u["inflight"] for u in live)
    tied = [u for u in live if u["inflight"] == least]
    return tied[next(_rr) % len(tied)]


async def serve_proxy(api_key):
    import aiohttp
    from aiohttp import web

    # forwarded-for headers would make vLLM log the client's public IP
    drop_headers = {"host", "content-length", "transfer-encoding", "connection",
                    "x-forwarded-for", "x-real-ip", "forwarded"}
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def handle(request):
        if request.path != "/health" and request.headers.get("Authorization") != f"Bearer {api_key}":
            return web.json_response({"error": "unauthorized"}, status=401)
        if request.path == "/admin/files":
            return web.json_response(sorted(os.listdir(DATA_DIR)))
        if request.path.startswith("/admin/files/"):
            path = os.path.join(DATA_DIR, os.path.basename(request.path))
            return web.FileResponse(path) if os.path.isfile(path) else web.Response(status=404)

        body = await request.read()
        trace = {"id": str(uuid.uuid4()), "ts": time.time(), "mode": STATE["mode"], "method": request.method,
                 "path": request.path_qs, "client_ip": request.headers.get("X-Forwarded-For", request.remote)}
        try:
            trace["request"] = json.loads(body) if body else None
        except ValueError:
            trace["request"] = body.decode(errors="replace")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in drop_headers}
        t0 = time.perf_counter()
        chunks, tried, content_type = [], [], ""
        try:
            while True:
                up = pick_upstream(exclude=tried)
                if up is None:
                    trace["status"] = 503
                    return web.json_response({"error": "no healthy upstream"}, status=503)
                tried.append(up["port"])
                trace["upstream"] = up["port"]
                up["inflight"] += 1
                try:
                    async with session.request(request.method, f"http://127.0.0.1:{up['port']}{request.path_qs}",
                                               headers=headers, data=body) as resp_up:
                        trace["status"] = resp_up.status
                        resp = web.StreamResponse(status=resp_up.status, headers={
                            k: v for k, v in resp_up.headers.items() if k.lower() not in drop_headers})
                        await resp.prepare(request)
                        async for chunk in resp_up.content.iter_any():
                            if not chunks:
                                trace["ttfb_s"] = round(time.perf_counter() - t0, 4)
                            chunks.append(chunk)
                            await resp.write(chunk)
                        await resp.write_eof()
                        content_type = resp_up.headers.get("Content-Type", "")
                        return resp
                except aiohttp.ClientConnectorError:
                    up["down_until"] = time.time() + 10  # connection refused: fail over to the next engine
                    trace["failovers"] = trace.get("failovers", 0) + 1
                finally:
                    up["inflight"] -= 1
        except Exception as e:
            trace["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            trace["latency_s"] = round(time.perf_counter() - t0, 4)
            if chunks and request.path.startswith("/v1/") and "error" not in trace:
                try:
                    trace.update(summarize_response(b"".join(chunks), content_type))
                    tokens = (trace.get("usage") or {}).get("completion_tokens")
                    if tokens:
                        trace["tokens_per_s"] = round(tokens / trace["latency_s"], 2)
                except Exception as e:
                    trace["parse_error"] = str(e)
            append_jsonl("requests.jsonl", trace)

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PROXY_PORT).start()
    log(f"load-balancing proxy on :{PROXY_PORT}")
    await asyncio.Event().wait()


def start_daemon(target, *args):
    threading.Thread(target=target, args=args, daemon=True).start()


# ---------------------------------------------------------------- tunnel + supervisor

def open_tunnel(secrets, attempts=10):
    from pyngrok import conf, ngrok

    conf.get_default().auth_token = secrets["NGROK_AUTHTOKEN"]
    # retries cover a previous session still holding the static domain during a redeploy
    for attempt in range(1, attempts + 1):
        try:
            tunnel = ngrok.connect(PROXY_PORT, domain=secrets["NGROK_DOMAIN"])
            log(f"PUBLIC_URL {tunnel.public_url}/v1")
            return tunnel
        except Exception as e:
            log(f"tunnel attempt {attempt}/{attempts} failed: {e}")
            if attempt == attempts:
                raise
            time.sleep(30)


def supervise(mode, procs, secrets, deadline):
    from pyngrok import ngrok

    cfgs = ENGINES[mode]
    while time.time() < deadline:
        time.sleep(CHECK_INTERVAL_S)
        for i, (proc, cfg) in enumerate(zip(procs, cfgs)):
            if proc.poll() is not None:
                log(f"[{mode}] vLLM :{cfg['port']} died ({proc.returncode}), restarting:\n"
                    f"{tail(engine_log(mode, cfg), 20)}")
                procs[i] = start_engine(mode, cfg, secrets["VLLM_API_KEY"])  # proxy fails over meanwhile
                wait_until_healthy(mode, [procs[i]], [cfg])
        try:
            if not ngrok.get_tunnels():
                log("tunnel missing, reconnecting")
                open_tunnel(secrets)
        except Exception as e:
            log(f"tunnel check failed: {e}")
    log(f"RUN_HOURS={RUN_HOURS} reached, shutting down so outputs are saved")
    ngrok.kill()
    stop_engines(procs)


def main():
    deadline = time.time() + RUN_HOURS * 3600
    secrets = load_secrets()  # fail fast before the long install if secrets aren't attached
    api_key = secrets["VLLM_API_KEY"]
    os.makedirs(DATA_DIR, exist_ok=True)
    diagnostics()
    install()
    start_daemon(lambda: asyncio.run(serve_proxy(api_key)))
    start_daemon(sample_metrics)

    procs = []
    for mode in MODES:
        try:
            procs = start_mode(mode, api_key)
            run_benchmarks(mode, api_key)
        except Exception as e:
            log(f"[{mode}] failed, moving on: {e}")
            procs = []
        if mode != MODES[-1] or mode != SERVE_MODE:
            STATE.update(mode=None, upstreams=[])  # proxy answers 503 while no mode is up
            stop_engines(procs)
            procs = []
    write_comparison()

    if not procs:
        procs = start_mode(SERVE_MODE, api_key)
    log(f"serving mode: {STATE['mode']}")
    open_tunnel(secrets)
    supervise(STATE["mode"], procs, secrets, deadline)  # blocks so the Kaggle session stays alive


if __name__ == "__main__":
    main()
