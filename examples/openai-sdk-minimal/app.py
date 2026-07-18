#!/usr/bin/env python3
"""The universal "set one base URL" recipe, as running code.

Point the official OpenAI SDK at a `fak serve` gateway and you get two things for
one line of change (`base_url`):

  1. Your app is unchanged. The same OpenAI-SDK call works; fak proxies it to
     whatever upstream you configured (`fak serve --base-url …`).
  2. Every tool call the model proposes crosses a governed syscall boundary. This
     script makes that boundary visible by asking fak to *adjudicate* two proposed
     tool calls WITHOUT running them (`POST /v1/fak/adjudicate` — a pre-execution
     verdict only: no dispatch, no engine), and printing the verdict.

Run it against a local gateway (see README.md):

    fak serve --addr 127.0.0.1:8080          # in another terminal
    pip install -r requirements.txt
    python app.py
"""
import json
import os
import urllib.request

from openai import OpenAI

BASE = os.environ.get("FAK_BASE_URL", "http://127.0.0.1:8080/v1")
# fak only requires a key if it was started with --require-key-env; otherwise any
# value is ignored. So a placeholder is fine for the local, keyless demo.
KEY = os.environ.get("FAK_GATEWAY_KEY", "local-demo-no-key-required")
MODEL = os.environ.get("FAK_MODEL", "qwen2.5:1.5b")

# The gateway origin (drop the trailing /v1) so we can reach /healthz and /v1/fak/*.
ORIGIN = BASE.rsplit("/v1", 1)[0]


def adjudicate(tool: str, arguments: dict) -> dict:
    """Ask fak for the pre-execution verdict on a proposed tool call."""
    body = json.dumps({"tool": tool, "arguments": arguments}).encode()
    req = urllib.request.Request(
        f"{ORIGIN}/v1/fak/adjudicate",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["verdict"]


def main() -> None:
    # --- 1) Your OpenAI-SDK app, unchanged except for base_url ------------------
    client = OpenAI(base_url=BASE, api_key=KEY)
    try:
        chat = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say OK."}],
        )
        print(f"model reply: {chat.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001 — the demo works even with no upstream wired
        print(f"(chat.completions needs an upstream — `fak serve --base-url …`; skipped: {e})")

    # --- 2) The syscall boundary, made visible ----------------------------------
    # An allow-listed read vs a denied exec, adjudicated but NOT executed. Under the
    # built-in default floor, read_* is allowed and an un-allow-listed exec is denied.
    print("\nproposed tool call            ->  fak verdict")
    print("-" * 52)
    for tool, args in [
        ("read_file", {"path": "README.md"}),
        ("Bash", {"command": "git push origin main"}),
    ]:
        v = adjudicate(tool, args)
        line = f"verdict={v['kind']}"
        if v.get("reason"):
            line += f" reason={v['reason']}"
        if v.get("disposition"):
            line += f" disposition={v['disposition']}"
        print(f"  {tool:<26}->  {line}")


if __name__ == "__main__":
    main()
