#!/usr/bin/env python3
"""Run a small OpenAI-compatible prompt-cache telemetry probe.

The probe sends a stable, cacheable prefix followed by tiny changing suffixes and
writes each raw provider response as JSONL. The existing verifier consumes the
result directly:

  go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/openai.jsonl

No API key is a skip, not a failure: without provider-authored cached_tokens
telemetry, a vCache/OpenAI savings claim is not evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


SCHEMA = "fak-vcache-openai-probe/1"
DEFAULT_MODEL = "gpt-4o-mini"
SKIP = 77


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def build_anchor(words: int) -> str:
    words = max(words, 1)
    chunks = []
    for i in range(words):
        chunks.append(f"vcache_anchor_{i % 97:02d}")
    return (
        "Stable vCache probe prefix. Keep this text byte-identical across all "
        "requests so provider prompt caching can only depend on the changing "
        "suffix below.\n\n" + " ".join(chunks)
    )


def chat_payload(model: str, anchor: str, suffix: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": anchor},
            {"role": "user", "content": f"Reply exactly: {suffix}"},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def responses_payload(model: str, anchor: str, suffix: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": anchor}]},
            {"role": "user", "content": [{"type": "input_text", "text": f"Reply exactly: {suffix}"}]},
        ],
        "temperature": 0,
        "max_output_tokens": max_tokens,
    }


def endpoint_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if endpoint == "responses":
        return f"{base}/v1/responses"
    return f"{base}/v1/chat/completions"


def post_json(url: str, api_key: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return json.loads(body.decode("utf-8"))


def cached_tokens(usage: dict | None) -> int:
    if not isinstance(usage, dict):
        return 0
    for details_key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, dict):
            try:
                return int(details.get("cached_tokens") or 0)
            except (TypeError, ValueError):
                return 0
    try:
        return int(usage.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def prompt_tokens(usage: dict | None) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in ("input_tokens", "prompt_tokens"):
        try:
            value = int(usage.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            return value
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="JSONL output path")
    p.add_argument("--endpoint", choices=("chat", "responses"), default="chat")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    p.add_argument("--requests", type=int, default=4)
    p.add_argument("--anchor-words", type=int, default=1600,
                   help="roughly cacheable stable prefix size; keep >=1024 tokens in practice")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between requests")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true", help="print plan JSON; do not call the provider")
    return p.parse_args(argv)


def run(argv: list[str], *, post=post_json, env=os.environ, stderr=sys.stderr, stdout=sys.stdout) -> int:
    args = parse_args(argv)
    if args.requests <= 0:
        print("vcache_openai_probe: --requests must be positive", file=stderr)
        return 2

    api_key = (env.get(args.api_key_env) or "").strip()
    anchor = build_anchor(args.anchor_words)
    url = endpoint_url(args.base_url, args.endpoint)
    out = Path(args.out)
    plan = {
        "schema": SCHEMA,
        "endpoint": args.endpoint,
        "url": url,
        "model": args.model,
        "requests": args.requests,
        "anchor_words": args.anchor_words,
        "out": str(out),
        "api_key_env": args.api_key_env,
        "api_key_present": bool(api_key),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2), file=stdout)
        return 0
    if not api_key:
        print(
            f"vcache_openai_probe: skipping live probe; {args.api_key_env} is not set",
            file=stderr,
        )
        return SKIP

    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    payload_builder = responses_payload if args.endpoint == "responses" else chat_payload
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(args.requests):
            label = f"vcache-openai-sibling-{i + 1}"
            payload = payload_builder(args.model, anchor, label, args.max_tokens)
            start = time.perf_counter()
            response = post(url, api_key, payload, args.timeout)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            row = dict(response)
            row.update({
                "schema": SCHEMA,
                "captured_utc": utc_now(),
                "endpoint": args.endpoint,
                "request_index": i + 1,
                "label": label,
                "duration_ms": elapsed_ms,
            })
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            f.flush()
            rows.append(row)
            if args.sleep > 0 and i + 1 < args.requests:
                time.sleep(args.sleep)

    total_prompt = sum(prompt_tokens(r.get("usage")) for r in rows)
    total_cached = sum(cached_tokens(r.get("usage")) for r in rows)
    print(
        f"wrote {len(rows)} rows to {out}; prompt_tokens={total_prompt} cached_tokens={total_cached}",
        file=stderr,
    )
    if total_cached == 0:
        print("provider returned cached_tokens=0 for every row; verifier should REFUTE", file=stderr)
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
