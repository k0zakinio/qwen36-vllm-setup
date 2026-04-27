#!/usr/bin/env python3
"""Live monitor for the vLLM server via its Prometheus /metrics endpoint.

Polls /metrics on a fixed interval and shows per-poll spec-decode acceptance,
prompt/generation TPS, prefix cache hit rate, KV usage, queue depth, and
recent TTFT/ITL — all derived from monotonic counters, so prefill ticks
appear immediately and idle gaps don't drop samples. Stdlib-only — run with
system python3, no venv needed.

  Usage: ./watch-vllm.py [--url URL] [--interval SECONDS] [--window N]

  --url       metrics endpoint (default: http://localhost:<PORT>/metrics,
              reading PORT from config.env)
  --interval  poll interval in seconds (default: 2.0)
  --window    rolling window in samples (default: 30 = 60s @ 2s)
"""
import argparse
import collections
import os
import re
import sys
import time
import urllib.error
import urllib.request


def default_url() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.join(here, "config.env")
    port = "8000"
    if os.path.isfile(cfg):
        with open(cfg) as f:
            for line in f:
                m = re.match(r"\s*PORT\s*=\s*(.+?)\s*$", line)
                if m:
                    port = m.group(1).strip().strip('"').strip("'")
                    break
    return f"http://localhost:{port}/metrics"


METRIC_RE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_metrics(text: str) -> dict:
    """Parse Prometheus text format into {(name, frozenset(label_pairs)): float}."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        try:
            val = float(m.group(3))
        except ValueError:
            continue
        labels = frozenset(LABEL_RE.findall(m.group(2) or ""))
        out[(m.group(1), labels)] = val
    return out


def get(metrics: dict, name: str, **filter_labels) -> float | None:
    """Sum metric values matching `name` and (optionally) all `filter_labels`."""
    total, found = 0.0, False
    for (n, lbls), v in metrics.items():
        if n != name:
            continue
        if filter_labels:
            d = dict(lbls)
            if not all(d.get(k) == v2 for k, v2 in filter_labels.items()):
                continue
        total += v
        found = True
    return total if found else None


def snapshot(metrics: dict) -> dict:
    """Pull just what we render."""
    g = lambda *a, **kw: get(metrics, *a, **kw) or 0.0
    return {
        "_t": time.time(),
        "prompt_tokens": g("vllm:prompt_tokens_total"),
        "gen_tokens": g("vllm:generation_tokens_total"),
        "drafts": g("vllm:spec_decode_num_drafts_total"),
        "draft_tokens": g("vllm:spec_decode_num_draft_tokens_total"),
        "accepted": g("vllm:spec_decode_num_accepted_tokens_total"),
        "accepted_p0": g("vllm:spec_decode_num_accepted_tokens_per_pos_total", position="0"),
        "accepted_p1": g("vllm:spec_decode_num_accepted_tokens_per_pos_total", position="1"),
        "accepted_p2": g("vllm:spec_decode_num_accepted_tokens_per_pos_total", position="2"),
        "prefix_queries": g("vllm:prefix_cache_queries_total"),
        "prefix_hits": g("vllm:prefix_cache_hits_total"),
        "preemptions": g("vllm:num_preemptions_total"),
        "completed": g("vllm:request_success_total"),
        "ttft_sum": g("vllm:time_to_first_token_seconds_sum"),
        "ttft_count": g("vllm:time_to_first_token_seconds_count"),
        "itl_sum": g("vllm:inter_token_latency_seconds_sum"),
        "itl_count": g("vllm:inter_token_latency_seconds_count"),
        "running": g("vllm:num_requests_running"),
        "waiting": g("vllm:num_requests_waiting"),
        "kv_pct": g("vllm:kv_cache_usage_perc") * 100.0,
        "engine_awake": g("vllm:engine_sleep_state", sleep_state="awake"),
    }


def diff_row(a: dict, b: dict) -> dict:
    """Per-poll deltas + instantaneous gauges from b."""
    dt = b["_t"] - a["_t"]
    if dt <= 0:
        dt = 1e-9
    d = lambda k: b[k] - a[k]
    d_drafts = d("drafts")
    d_dtok = d("draft_tokens")
    d_pq = d("prefix_queries")
    d_ttft_n = d("ttft_count")
    d_itl_n = d("itl_count")
    return {
        "t": time.strftime("%H:%M:%S", time.localtime(b["_t"])),
        "prompt_tps": d("prompt_tokens") / dt,
        "gen_tps": d("gen_tokens") / dt,
        "running": int(b["running"]),
        "waiting": int(b["waiting"]),
        "kv": b["kv_pct"],
        "mean_accept": (d("accepted") / d_drafts + 1.0) if d_drafts > 0 else None,
        "p0": (d("accepted_p0") / d_drafts) if d_drafts > 0 else None,
        "p1": (d("accepted_p1") / d_drafts) if d_drafts > 0 else None,
        "p2": (d("accepted_p2") / d_drafts) if d_drafts > 0 else None,
        "accept_rate": (100.0 * d("accepted") / d_dtok) if d_dtok > 0 else None,
        "prefix_hit": (100.0 * d("prefix_hits") / d_pq) if d_pq > 0 else None,
        "ttft": (d("ttft_sum") / d_ttft_n) if d_ttft_n > 0 else None,
        "itl": (d("itl_sum") / d_itl_n) if d_itl_n > 0 else None,
        "preempt": int(d("preemptions")),
        "completed": int(d("completed")),
    }


def color(s: str, code: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m"


def fmt_ms(v: float | None, w: int = 4) -> str:
    return f"{v*1000:>{w}.0f}ms" if v is not None else " " * (w - 2) + " --  "


def fmt_pct(v: float | None, w: int = 4) -> str:
    return f"{v:>{w}.0f}%" if v is not None else " " * (w - 2) + "-- "


def draw(args, samples, last_err):
    parts = ["\x1b[H\x1b[J"]
    span = args.window * args.interval
    hdr = (f"vLLM live monitor — interval {args.interval:.1f}s, window {args.window} "
           f"({span:.0f}s)   now: {time.strftime('%H:%M:%S')}   Ctrl-C to quit")
    parts.append(color(hdr, "1;36") + "\n")
    parts.append(color(f"metrics: {args.url}", "2") + "\n")
    if last_err:
        parts.append(color(f"  last error: {last_err}", "31") + "\n")
    parts.append("\n")

    if len(samples) < 2:
        parts.append(color("  (collecting first two samples…)\n", "2"))
        sys.stdout.write("".join(parts))
        sys.stdout.flush()
        return

    rows = [diff_row(samples[i - 1], samples[i]) for i in range(1, len(samples))]

    parts.append(color("── Spec-decode (per poll) ─────────────────────────────────", "1") + "\n")
    parts.append(f"{'time':>8}  {'mean':>5}  {'pos1':>5}  {'pos2':>5}  {'pos3':>5}  {'accept%':>7}\n")
    spec_rows = [r for r in rows if r["mean_accept"] is not None]
    if spec_rows:
        for r in spec_rows[-12:]:
            mean, rate = r["mean_accept"], r["accept_rate"] or 0.0
            mean_c = "32" if mean >= 3.2 else "33" if mean >= 2.6 else "31"
            rate_c = "32" if rate >= 80 else "33" if rate >= 60 else "31"
            parts.append(
                f"{r['t']:>8}  {color(f'{mean:>5.2f}', mean_c)}  "
                f"{r['p0']:>5.2f}  {r['p1']:>5.2f}  {r['p2']:>5.2f}  "
                f"{color(f'{rate:>6.1f}%', rate_c)}\n"
            )
        n = len(spec_rows)
        avg = lambda k: sum(r[k] for r in spec_rows) / n
        parts.append(color(
            f"{'avg':>8}  {avg('mean_accept'):>5.2f}  "
            f"{avg('p0'):>5.2f}  {avg('p1'):>5.2f}  {avg('p2'):>5.2f}  "
            f"{avg('accept_rate'):>6.1f}%\n",
            "1;36",
        ))
    else:
        parts.append(color("  (no decode steps in window)\n", "2"))

    parts.append("\n" + color("── Throughput / cache (per poll) ──────────────────────────", "1") + "\n")
    parts.append(
        f"{'time':>8}  {'gen_tps':>7}  {'prompt_tps':>10}  "
        f"{'run':>3}  {'wait':>4}  {'kv%':>5}  {'pfx':>5}  {'ttft':>6}  {'itl':>6}\n"
    )
    for r in rows[-12:]:
        gen, prompt = r["gen_tps"], r["prompt_tps"]
        gen_c = "32" if gen >= 60 else "33" if gen >= 20 else "31" if gen > 0.5 else "2"
        prompt_c = "32" if prompt >= 1000 else "33" if prompt >= 200 else "0" if prompt > 1.0 else "2"
        kv_c = "31" if r["kv"] >= 80 else "33" if r["kv"] >= 50 else "32"
        parts.append(
            f"{r['t']:>8}  {color(f'{gen:>7.1f}', gen_c)}  {color(f'{prompt:>10.1f}', prompt_c)}  "
            f"{r['running']:>3d}  {r['waiting']:>4d}  "
            f"{color(f'{r['kv']:>4.1f}%', kv_c)}  "
            f"{fmt_pct(r['prefix_hit']):>5}  {fmt_ms(r['ttft']):>6}  {fmt_ms(r['itl']):>6}\n"
        )

    gens = [r["gen_tps"] for r in rows]
    prompts = [r["prompt_tps"] for r in rows]
    active_gen = [g for g in gens if g > 1.0]
    active_prompt = [p for p in prompts if p > 1.0]
    peak_gen, peak_prompt = max(gens), max(prompts)
    avg_gen = sum(active_gen) / len(active_gen) if active_gen else 0.0
    avg_prompt = sum(active_prompt) / len(active_prompt) if active_prompt else 0.0
    completed = sum(r["completed"] for r in rows)
    preempt = sum(r["preempt"] for r in rows)
    ttft_rows = [r["ttft"] for r in rows if r["ttft"] is not None]
    itl_rows = [r["itl"] for r in rows if r["itl"] is not None]
    ttft_avg = (sum(ttft_rows) / len(ttft_rows)) if ttft_rows else None
    itl_avg = (sum(itl_rows) / len(itl_rows)) if itl_rows else None
    parts.append(color(
        f"\npeak gen_tps:    {peak_gen:>7.1f}   avg gen_tps    (active): {avg_gen:>7.1f}   "
        f"gen-active:    {len(active_gen):>3d}/{len(rows)} polls\n"
        f"peak prompt_tps: {peak_prompt:>7.1f}   avg prompt_tps (active): {avg_prompt:>7.1f}   "
        f"prompt-active: {len(active_prompt):>3d}/{len(rows)} polls\n"
        f"completed: {completed}   preemptions: {preempt}   "
        f"ttft (window avg): {fmt_ms(ttft_avg).strip()}   itl (window avg): {fmt_ms(itl_avg).strip()}\n",
        "1;36",
    ))

    sys.stdout.write("".join(parts))
    sys.stdout.flush()


def fetch(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser(description="Live vLLM monitor (Prometheus-based)")
    p.add_argument("--url", default=default_url(), help="metrics endpoint URL")
    p.add_argument("--interval", type=float, default=2.0, help="poll interval (s)")
    p.add_argument("--window", type=int, default=30, help="rolling window (samples)")
    args = p.parse_args()

    samples = collections.deque(maxlen=args.window)
    last_err = None
    try:
        while True:
            t0 = time.time()
            try:
                text = fetch(args.url, timeout=max(1.0, args.interval * 0.8))
                samples.append(snapshot(parse_metrics(text)))
                last_err = None
            except (urllib.error.URLError, urllib.error.HTTPError,
                    ConnectionError, TimeoutError, OSError) as e:
                last_err = f"{type(e).__name__}: {e}"
            draw(args, samples, last_err)
            elapsed = time.time() - t0
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
