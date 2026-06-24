#!/usr/bin/env python
"""local_shim.py — a tiny OpenAI-compatible serving seam for the dogfood loop.

This is the off-path Python serving seam the dogfood launchers
(`scripts/dogfood-claude.{sh,ps1}`) put BEHIND the kernel when no ollama / no
large local server is available: it loads one HuggingFace causal-LM with
`transformers` and answers the two OpenAI-compatible routes `fak serve
--provider openai --base-url http://127.0.0.1:<port>/v1` drives —

    GET  /v1/models             readiness + the served model id (the launcher polls this)
    POST /v1/chat/completions   one buffered (non-streaming) chat turn

It is deliberately the SMALLEST thing that satisfies fak's planner contract
(`internal/agent/chat.go`: `choices[0].message.content`, `finish_reason`, and a
`usage` block). fak lifts any tool call out of the assistant TEXT itself
(`normalizeCompletionToolCalls`), so the shim never has to emit structured
`tool_calls` — it just generates text and reports tokens. The kernel in front of
it is what adjudicates; this is only the engine.

Architecturally it is an off-path oracle/serving seam (see
`internal/architest/architest_test.go` `oracleSeamFiles`): reachable only from the
dogfood scripts / off-path commands, never from the binary's decision path.

Device: CUDA fp16 when `torch.cuda.is_available()`, else CPU fp32 — the same
auto-detect the Windows dogfood doc promises. Force either with
`FAK_SHIM_DEVICE=cuda|cpu`. Small models (SmolLM2-135M) keep a CPU turn at
seconds; a 1.5B+ on CPU is minutes (point `--model` at it only with a GPU).

Usage:
  python experiments/agent-live/local_shim.py --model HuggingFaceTB/SmolLM2-135M-Instruct --port 8190
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Module state, set once in main() before the listener binds, so a /v1/models
# readiness probe only succeeds after the weights are resident.
TOKENIZER = None
MODEL = None
MODEL_ID = ""
DEVICE = "cpu"


def pick_device():
    """CUDA fp16 if available else CPU fp32; FAK_SHIM_DEVICE forces either."""
    forced = os.environ.get("FAK_SHIM_DEVICE", "").strip().lower()
    if forced in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda"
    if forced in ("cpu",):
        return "cpu"
    if forced in ("cuda", "gpu") and not torch.cuda.is_available():
        sys.stderr.write("[shim] FAK_SHIM_DEVICE=cuda but no CUDA device; falling back to CPU\n")
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_id):
    """Load the tokenizer + causal LM resident on the chosen device."""
    global TOKENIZER, MODEL, MODEL_ID, DEVICE
    DEVICE = pick_device()
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    sys.stderr.write(f"[shim] loading {model_id} on {DEVICE} ({dtype})\n")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    MODEL = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    MODEL.to(DEVICE)
    MODEL.eval()
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    MODEL_ID = model_id
    sys.stderr.write(f"[shim] ready: {model_id}\n")


def coerce_messages(messages):
    """Flatten fak's chat messages to the {role, content} pairs a chat template
    accepts. fak may send role=tool turns and assistant turns carrying only
    tool_calls; map a tool result to a user turn and keep text content, so the
    template never trips on a role/shape it does not know."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # OpenAI typed content parts -> concatenated text.
            content = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        content = content or ""
        if role == "tool":
            role = "user"
        elif role not in ("system", "user", "assistant"):
            role = "user"
        if not content and role == "assistant":
            # An assistant turn that was pure tool_calls leaves nothing for the
            # template; skip it rather than emit an empty assistant turn.
            continue
        out.append({"role": role, "content": content})
    if not out:
        out = [{"role": "user", "content": ""}]
    return out


def render_prompt(messages):
    """Apply the model's chat template; fall back to a plain concatenation when a
    model ships no template."""
    msgs = coerce_messages(messages)
    try:
        return TOKENIZER.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:  # noqa: BLE001 - any template failure -> manual render
        sys.stderr.write(f"[shim] chat template failed ({exc}); plain concat\n")
        return "".join(f"{m['role']}: {m['content']}\n" for m in msgs) + "assistant:"


def truncate_at_stop(text, stops):
    """Honor OpenAI `stop`: cut at the earliest stop string the model emitted."""
    cut = len(text)
    for s in stops or []:
        if not s:
            continue
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def generate(req):
    """One buffered chat turn -> (text, prompt_tokens, completion_tokens, hit_cap)."""
    prompt = render_prompt(req.get("messages", []))
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    max_new = req.get("max_tokens") or 512
    max_new = max(1, min(int(max_new), 4096))
    temperature = req.get("temperature", 0.0)
    do_sample = temperature is not None and float(temperature) > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        pad_token_id=TOKENIZER.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        if req.get("top_p") is not None:
            gen_kwargs["top_p"] = float(req["top_p"])

    with torch.inference_mode():
        out = MODEL.generate(**inputs, **gen_kwargs)
    new_ids = out[0][prompt_tokens:]
    completion_tokens = int(new_ids.shape[0])
    text = TOKENIZER.decode(new_ids, skip_special_tokens=True)

    stops = req.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    text = truncate_at_stop(text, stops)
    hit_cap = completion_tokens >= max_new
    return text, prompt_tokens, completion_tokens, hit_cap


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # quiet the default per-request stderr spam
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local-shim"}],
            })
        elif self.path.rstrip("/") in ("/healthz", "/health"):
            self._send(200, {"ok": True, "model": MODEL_ID, "device": DEVICE})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": {"message": f"bad json: {exc}"}})
            return
        try:
            text, p_tok, c_tok, hit_cap = generate(req)
        except Exception as exc:  # noqa: BLE001 - surface a generation fault as a 500
            sys.stderr.write(f"[shim] generation error: {exc}\n")
            self._send(500, {"error": {"message": f"generation failed: {exc}"}})
            return
        self._send(200, {
            "id": f"chatcmpl-shim-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model") or MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if hit_cap else "stop",
            }],
            "usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
            },
        })


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible transformers shim for the fak dogfood loop")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    load_model(args.model)  # block until weights are resident, THEN bind
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(f"[shim] serving {args.model} on http://{args.host}:{args.port}/v1\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
