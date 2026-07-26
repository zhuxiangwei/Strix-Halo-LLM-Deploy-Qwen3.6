#!/usr/bin/env python3
"""benchlib.py — Minimal benchmark library for Strix Halo LLM testing.

Design principle: SIMPLE > CLEVER. No over-abstraction.
Test scripts are themselves test code — don't make them need their own tests.

Provides:
- kill_all_llama: pkill -9 + verify dead
- wait_clean: kill + verify + wait memory — LOOPS FOREVER until clean
- LlamaServer: start/stop (minimal wrapper around subprocess)
- Prompt generation, SSE parsing, CSV output
"""

import subprocess
import time
import json
import csv
import os
import sys
import signal
import urllib.request
import urllib.error

# ── Constants ──────────────────────────────────────────────────
CHARS_PER_TOKEN = 3.6
DEFAULT_CTX = 262144  # parallel=1 × 262144 per slot
DEFAULT_BATCH = 4096
DEFAULT_UBATCH = 256  # both models unified at 256 for stability
DEFAULT_THREADS = 8
DEFAULT_REASONING_BUDGET = 32768
DEFAULT_CACHE_RAM_278 = 49152
DEFAULT_CACHE_RAM_358 = 49152
DEFAULT_SLOT_PROMPT_SIMILARITY = 0.8
DEFAULT_SERVER_PORT = 12345
DEFAULT_SERVER_READY_TIMEOUT = 180
DEFAULT_REQUEST_TIMEOUT = 7200


# ── Kill + Clean + Wait ────────────────────────────────────────

def _get_mem_available_gb():
    """Read MemAvailable from /proc/meminfo, return GB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return -1


def _get_mem_total_gb():
    """Read MemTotal from /proc/meminfo, return GB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 32  # fallback


def _pgrep_llama():
    """Return list of llama-server PIDs (excluding zombies), empty if none."""
    try:
        r = subprocess.run(["pgrep", "-f", "llama-server"],
                           capture_output=True, text=True, timeout=5)
        if r.stdout.strip():
            pids = [int(x) for x in r.stdout.strip().splitlines()]
            # Filter out zombies (State: Z) - they can't be killed and will
            # only disappear when parent reaps them
            alive = []
            for p in pids:
                try:
                    with open(f"/proc/{p}/status") as sf:
                        for line in sf:
                            if line.startswith("State:"):
                                if "Z" not in line:
                                    alive.append(p)
                                break
                except (FileNotFoundError, PermissionError):
                    pass  # process gone, skip
            return alive
    except Exception:
        pass
    return []


def _port_in_use(port=DEFAULT_SERVER_PORT):
    """Check if port has a live llama-server."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


def kill_all_llama():
    """pkill -9 all llama-server, verify dead. Returns True if clean."""
    print("  [KILL] Killing all llama-server...")
    for attempt in range(5):
        try:
            subprocess.run(["pkill", "-9", "-f", "llama-server"],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        time.sleep(2)
        pids = _pgrep_llama()
        if not pids:
            break
        print(f"  [KILL] Attempt {attempt+1}: still alive PIDs={pids}")
    pids = _pgrep_llama()
    if pids:
        print(f"  [KILL] WARNING: PIDs still alive: {pids}")
        return False
    # Wait for port to be freed
    for _ in range(10):
        if not _port_in_use():
            break
        time.sleep(2)
    print("  [KILL] All llama-server dead, port free")
    return True


def wait_clean(target_gb=None):
    """Kill + clean + wait. LOOPS FOREVER until conditions met.
    
    - Repeatedly kill residual llama-server processes
    - Wait for available memory to reach target (default 75% of total)
    - Does NOT return until clean — this is intentional
    """
    if target_gb is None:
        target_gb = _get_mem_total_gb() * 0.75
    
    round_num = 0
    while True:
        round_num += 1
        # 1. Kill any residual processes
        kill_all_llama()
        
        # 2. Check memory
        avail = _get_mem_available_gb()
        port_ok = not _port_in_use()
        pids = _pgrep_llama()
        
        print(f"  [WAIT] Round {round_num}: avail={avail:.1f}GB / target={target_gb:.1f}GB, "
              f"port_free={port_ok}, pids={pids}")
        
        if avail >= target_gb and port_ok and not pids:
            print(f"  [WAIT] Clean! avail={avail:.1f}GB >= {target_gb:.1f}GB")
            return True
        
        # Not clean yet — wait and loop again
        print(f"  [WAIT] Not clean yet, waiting 10s and retrying...")
        time.sleep(10)


# ── Lock ───────────────────────────────────────────────────────

def acquire_lock(lock_name="bench"):
    """PID lock to prevent duplicate instances."""
    lock_file = f"/tmp/bench_{lock_name}.lock"
    if os.path.exists(lock_file):
        try:
            with open(lock_file) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            print(f"  [LOCK] ERROR: Another bench running (PID {old_pid})")
            print(f"  [LOCK] If stale: rm {lock_file}")
            sys.exit(1)
        except (OSError, ProcessLookupError, ValueError):
            print(f"  [LOCK] Stale lock (PID {old_pid} dead), stealing...")
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))
    print(f"  [LOCK] Acquired: {lock_file}")


def release_lock(lock_name="bench"):
    lock_file = f"/tmp/bench_{lock_name}.lock"
    try:
        os.remove(lock_file)
    except OSError:
        pass


# NOTE: Router service should be manually disabled before benchmarks.
def disable_router_service():
    """Stop and disable llm-router service."""
    print("  [SERVICE] Stopping and disabling llama-router...")
    try:
        subprocess.run(["systemctl", "--user", "stop", "llama-router.service"],
                       capture_output=True, timeout=30)
        subprocess.run(["systemctl", "--user", "disable", "llama-router.service"],
                       capture_output=True, timeout=30)
    except Exception as e:
        print(f"  [SERVICE] Warning: {e}")
    time.sleep(2)


# No enable_router_service() — per user rule, re-enable manually after all tests.


# ── LlamaServer ────────────────────────────────────────────────

class LlamaServer:
    """Minimal llama-server wrapper. start() calls wait_clean() first."""

    def __init__(self, model_path, alias, base_dir, api_key="",
                 ubatch=DEFAULT_UBATCH, cache_type_k="f16", cache_type_v="f16",
                 ctx=DEFAULT_CTX, batch=DEFAULT_BATCH, threads=DEFAULT_THREADS,
                 port=DEFAULT_SERVER_PORT, parallel=1,
                 cache_ram=None, slot_prompt_similarity=DEFAULT_SLOT_PROMPT_SIMILARITY,
                 mmproj_path=None, cache_reuse=None,
                 spec_draft_n_max=2,
                 ready_timeout=DEFAULT_SERVER_READY_TIMEOUT):
        self.model_path = model_path
        self.alias = alias
        self.base_dir = base_dir
        self.api_key = api_key
        self.ubatch = ubatch
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.ctx = ctx
        self.batch = batch
        self.threads = threads
        self.port = port
        self.parallel = parallel
        self.cache_ram = cache_ram
        self.slot_prompt_similarity = slot_prompt_similarity
        self.mmproj_path = mmproj_path
        self.cache_reuse = cache_reuse
        self.spec_draft_n_max = spec_draft_n_max
        self.ready_timeout = ready_timeout
        self.pid = None

    def start(self):
        """Start llama-server. Calls wait_clean() first. Returns True on success."""
        # Ensure absolutely clean before starting
        wait_clean()

        cmd = [
            os.path.join(self.base_dir, "llama.cpp/build" + os.environ.get("LLM_BUILD_SUFFIX", "") + "/bin/llama-server"),
            "--model", self.model_path,
            "--n-gpu-layers", "99",
            "--ctx-size", str(self.ctx),
            "--batch-size", str(self.batch),
            "--ubatch-size", str(self.ubatch),
            "--threads", str(self.threads),
            "--flash-attn", "auto",
            "--parallel", str(self.parallel),
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(self.spec_draft_n_max),
            "--mlock",
            "--numa", "distribute",
            "--reasoning-budget", str(DEFAULT_REASONING_BUDGET),
            "--reasoning-preserve",
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
            "--kv-unified",
            "--cache-ram", str(self.cache_ram or DEFAULT_CACHE_RAM_278),
            "--slot-prompt-similarity", str(self.slot_prompt_similarity),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--alias", self.alias,
            "--timeout", "3600",
            "--metrics",
        ]

        # Optional: mmproj for vision models
        if self.mmproj_path:
            cmd.extend(["--mmproj", self.mmproj_path])

        # Optional: KV cache reuse threshold
        if self.cache_reuse is not None:
            cmd.extend(["--cache-reuse", str(self.cache_reuse)])

        log_path = os.path.join(self.base_dir, "test/server_bench.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_fh = open(log_path, "w")

        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        self.pid = proc.pid
        kv_desc = f"KV={self.cache_type_k}/{self.cache_type_v}" if self.cache_type_k != "f16" else "F16 KV"
        print(f"  [SERVER] Started PID={self.pid}, UB={self.ubatch}, {kv_desc}")

        # Wait for ready
        t0 = time.time()
        while time.time() - t0 < self.ready_timeout:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("status") == "ok":
                        print(f"  [SERVER] Ready in {time.time()-t0:.1f}s")
                        return True
            except Exception:
                pass
            time.sleep(2)

        print(f"  [SERVER] FAILED to start within {self.ready_timeout}s")
        self.stop()
        return False

    def stop(self):
        """Kill llama-server, verify dead, wait clean."""
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except Exception:
                pass
            time.sleep(1)
        # Force kill ALL llama-server (not just our PID)
        kill_all_llama()
        self.pid = None
        # Wait until truly clean
        wait_clean()




# ── Prompt Generation ──────────────────────────────────────────

def load_prompt_data(script_path):
    data_file = os.path.join(os.path.dirname(os.path.abspath(script_path)), "test_bench_data.txt")
    with open(data_file, "r", encoding="utf-8") as f:
        return f.read()


def make_prompt(text, target_tokens):
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    truncated = text[:target_chars]
    actual_tokens_est = len(truncated) / CHARS_PER_TOKEN
    prompt = f"{truncated}\n\n---\nSummarize the above content in 2-3 sentences."
    return prompt, int(actual_tokens_est)


# ── SSE Benchmark ──────────────────────────────────────────────

def run_test(server, prompt, prompt_tokens_est, alias,
             request_timeout=DEFAULT_REQUEST_TIMEOUT):
    """Send chat completion, parse SSE, return metrics dict.
    
    ALL performance metrics come from the API `timings` object (server-side).
    No client-side measurement for prefill/gen speed — API timings are more
    stable and exclude network latency.
    
    Timings object fields used:
        prompt_n           — prompt tokens processed
        prompt_ms          — prefill time (ms)
        prompt_per_second  — prefill speed (tok/s)
        predicted_n        — tokens generated (includes thinking)
        predicted_ms       — generation time (ms)
        predicted_per_second — gen speed (tok/s)
        draft_n            — MTP draft tokens
        draft_n_accepted   — MTP accepted tokens
    """
    api_base = f"http://127.0.0.1:{server.port}"
    payload = json.dumps({
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {server.api_key}",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        f"{api_base}/chat/completions", data=payload, headers=headers, method="POST",
    )

    t0 = time.time()
    last_timings = None

    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            buf = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    # Extract timings from SSE chunk
                    # (timings chunk may have empty choices array, extract before skip)
                    timings = data.get("timings")
                    if timings:
                        last_timings = timings
    except urllib.error.URLError as e:
        return {"error": f"URL Error: {e}", "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"error": f"Exception: {e}", "elapsed_s": round(time.time() - t0, 2)}

    elapsed = time.time() - t0

    # ── All metrics from API timings ───────────────────────────
    result = {"elapsed_s": round(elapsed, 2)}

    if last_timings:
        # Prompt / prefill
        prompt_n = last_timings.get("prompt_n", 0)
        prompt_ms = last_timings.get("prompt_ms", 0)
        prompt_per_s = last_timings.get("prompt_per_second")
        result["prompt_tokens"] = prompt_n
        result["prefill_ms"] = round(prompt_ms, 1)
        result["prefill_tps"] = round(prompt_per_s, 1) if prompt_per_s else None
        result["ttft_s"] = round(prompt_ms / 1000, 3) if prompt_ms else None

        # Generation
        predicted_n = last_timings.get("predicted_n", 0)
        predicted_ms = last_timings.get("predicted_ms", 0)
        predicted_per_s = last_timings.get("predicted_per_second")
        result["completion_tokens"] = predicted_n
        result["gen_ms"] = round(predicted_ms, 1)
        result["gen_tps"] = round(predicted_per_s, 1) if predicted_per_s else None

        # MTP
        draft_n = last_timings.get("draft_n")
        draft_accepted = last_timings.get("draft_n_accepted")
        if draft_n is not None and draft_accepted is not None:
            result["mtp_draft"] = draft_n
            result["mtp_accept"] = draft_accepted
            if draft_n > 0:
                result["mtp_rate"] = round(draft_accepted / draft_n * 100, 1)
    else:
        # Fallback: no timings received
        result["prompt_tokens"] = prompt_tokens_est
        result["prefill_tps"] = None
        result["gen_tps"] = None
        result["ttft_s"] = None

    return result


# ── CSV ────────────────────────────────────────────────────────

CSV_FIELDS = ["test", "prompt_tokens", "completion_tokens",
              "prefill_ms", "prefill_tps", "ttft_s",
              "gen_ms", "gen_tps",
              "mtp_rate", "mtp_draft", "mtp_accept",
              "elapsed_s", "error"]


def save_csv(results, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def print_summary(results):
    print(f"\n{'=' * 90}")
    print("  Benchmark Complete (API timings)")
    print(f"{'=' * 90}")
    print(f"  {'Test':<10} {'PTok':>7} {'CTok':>7} {'TTFT':>9} "
          f"{'Prefill':>10} {'Gen':>10} {'MTP':>6} {'Total':>9}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*10} {'─'*6} {'─'*9}")
    for r in results:
        if "error" in r and r.get("test"):
            print(f"  {r['test']:<10} {'ERROR':>7} {r.get('error', '')[:50]}")
        else:
            pt = r.get("prompt_tokens", "?")
            ct = r.get("completion_tokens", "?")
            ttft = f"{r.get('ttft_s', '?')}s" if r.get('ttft_s') else "?"
            pf = f"{r.get('prefill_tps', '?')} t/s" if r.get('prefill_tps') else "?"
            gf = f"{r.get('gen_tps', '?')} t/s" if r.get('gen_tps') else "?"
            mtp = f"{r.get('mtp_rate', '?')}%" if r.get('mtp_rate') else "—"
            el = f"{r.get('elapsed_s', '?')}s"
            print(f"  {r.get('test','?'):<10} {str(pt):>7} {str(ct):>7} {ttft:>9} "
                  f"{pf:>10} {gf:>10} {mtp:>6} {el:>9}")
