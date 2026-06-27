---
title: "Serving the strongest model on a MacBook (pure fak) + benchmarking it agentically from a second laptop"
description: "Design note: what the pure fak stack can actually serve on Apple Silicon today, the purity-vs-speed frontier, and how to drive an agentic A/B against it over the LAN from another laptop. Grounded in shipped fak code + on-device M3 Pro numbers, not aspirations."
date: 2026-06-24
---

# MacBook serve + cross-laptop agentic bench — the option, grounded

> The goal: serve the most powerful model we can on a MacBook with the **pure fak
> stack**, and benchmark its **agentic** performance from a second laptop. This note
> separates what is shipped and witnessed from what is an open lane, picks the model,
> and gives the cross-laptop harness. Every number is cited to an on-device artifact.

## The crux first

There are **two different "pure fak stack" answers**, and they trade purity against
speed on a sharp frontier:

1. **fak's OWN in-kernel forward pass** (the truest "pure fak"): one Go binary,
   `fak serve --gguf model.gguf`, fak's own `internal/model` forward over GGUF
   weights, no llama.cpp / vLLM / MLX anywhere in the path. **This runs on Apple
   Silicon today — but pure-Go on the CPU** (no Metal in the default build), so it is
   ~0.5× llama.cpp on a 7B and ~0.12× on a 27B (numbers below).
2. **fak as the gateway/kernel in front of a Mac-native engine**: `fak serve
   --base-url http://localhost:11434/v1` fronting Ollama / llama.cpp-Metal. You get
   real Metal GPU speed, and fak still contributes its actual product — the
   capability-lock + adjudication + per-agent KV-reuse + audit floor. This is the
   posture fak's own docs call "gateway-first" (`docs/cli-reference.md:12`).

The honest tension: **fak's own forward is the purest but slowest; the gateway path is
fast but the inference isn't fak's.** A third, in-between option exists and is the
interesting one for a *demo of fak itself*: fak's own forward with the **opt-in Metal
prefill** backend (`-tags fakmetal`), which accelerates the prefill GEMMs on the GPU
while decode stays on CPU NEON. It is real code, not aspirational, but it is
prefill-only and not in the shipped binary.

## What is actually shipped on Mac (witnessed, dated)

fak's in-kernel engine loads GGUF natively (`internal/ggufload`) and maps these
architectures: `qwen2/qwen2moe/qwen3/qwen3moe`, `qwen35/qwen35moe` (the Qwen3.5/3.6
Gated-DeltaNet hybrid), `glm-dsa/glm_moe_dsa` (GLM-5.2), `gemma2/3/4`, and Llama
(`internal/ggufload/gguf_tensor_canonical.go:200,286,334`). The chat-capable families
**witnessed running end-to-end through fak's own forward on an M3 Pro** are Qwen2.5
(1.5B/7B), Qwen3.6-27B, and GLM-5.2 (dense path).

fak's own pure-Go **CPU** decode on M3 Pro (Qwen2.5 Q8), from `BENCHMARK-AUTHORITY.md`
at the repo root (single source of truth — read, don't hardcode):

| Model | fak own-forward decode | llama.cpp Metal | ratio | usable for an agent loop? |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B Q8 | **38.1 tok/s** | 68.7 | 0.55× | yes — snappy |
| Qwen2.5-7B Q8 | **8.7 tok/s** | 17.6 (Metal) | 0.50× | yes — usable, the realistic ceiling |
| Qwen3.6-27B q4_k | **0.9 tok/s** | 7.29 (Metal) | 0.12× | no — a correctness witness, not an agent |

So on **fak's own forward**, the strongest model that is still *agentically usable*
(you need responsiveness across many turns) is **Qwen2.5-7B-Instruct Q8** at ~8.7
tok/s. The 27B runs and is coherent (`FAK-NATIVE-QWEN35-RESULTS.md` — first two
greedy tokens match llama.cpp b9707 on the real 15 GB GGUF, peaking ~25.8 GB RSS on a
36 GB M3 Pro) but at 0.9 tok/s it is a "the architecture runs in our own kernel"
proof, not something you'd run an agent against.

The Metal acceleration lane (`-tags fakmetal`, `internal/metalgemm/metalgemm.go:16`
linking `-framework Metal -framework MetalPerformanceShaders`) is real but
**prefill-only** — it routes the seven per-layer projection GEMMs to the GPU
(`internal/model/metal_prefill.go`) and leaves decode on CPU NEON. It is opt-in and
not in the shipped binary; full-GPU-resident own-forward decode parity with
llama.cpp-Metal is an **open lane**, not a shipped capability. There is **no MLX path
at all** (zero `mlx` references in any fak code; explicitly rejected as a bar).

Do **not** read the new `cmd/glmdsatput` binary as a Mac serving path. It measures
fak's native `glm_moe_dsa` (GLM-5.2) decode on a **CUDA** device with **synthetic
weights, a reduced layer count, and dense-FFN (no MoE experts)** — its own header calls
it an optimistic lower-bound on per-token device cost and explicitly **not** a 753B
serving number (`cmd/glmdsatput/main.go:1-19`). It is a GPU kernel witness, not a
MacBook option; the 753B native serve remains the labeled wall.

The fastest *on-device* fleet number — **5 agents × 200 turns of a 7B in 8.2 min on an
M3 Pro** (`FLEET-5X200-7B-10MIN-RESULTS.md`) — used **llama.cpp's Metal forward**
fronted by fak's reuse/batching, NOT fak's own forward. The same row openly fences
that fak's own forward would be ~22–51 min. Respect that fence in any claim.

## The serving command (from source, `cmd/fak/serve.go`)

`fak serve` binds a free address and serves both the OpenAI `/v1/chat/completions` and
Anthropic `/v1/messages` wires:

```bash
# On the MacBook — fak's OWN forward, Qwen2.5-7B, reachable on the LAN, auth required:
export FAK_KEY=$(openssl rand -hex 16)
go run ./cmd/fak serve \
  --addr 0.0.0.0:8080 \
  --gguf hf://Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q8_0.gguf \
  --require-key-env FAK_KEY \
  --policy examples/customer-support-readonly-policy.json
```

- `--addr 0.0.0.0:8080` makes it LAN-reachable (default is `127.0.0.1:8080`;
  `serve.go:31`). The second laptop hits `http://<macbook-ip>:8080/v1`.
- `--gguf` loads the weights into the **in-kernel** engine, so `/v1/chat/completions`
  serves fak's own forward with the GGUF's embedded tokenizer — no `--base-url`, no
  upstream (`serve.go:49-50,112-119`). `hf://` auto-fetches+caches (`serve.go:65`).
- `--require-key-env` enforces a bearer token and **refuses to start network-facing if
  the env var is empty** (`serve.go:139`) — a real interlock; don't expose it on the
  LAN without it.
- For the **gateway-in-front-of-Ollama** path instead, drop `--gguf` and add
  `--base-url http://localhost:11434/v1 --model qwen2.5:7b`.
- There is **no `metal` backend wired into `serve`** today — `--backend` accepts
  `cuda` (needs a `-tags cuda` build + a GPU). On a Mac, `serve` = pure-Go CPU
  forward. The Metal lane lives in `cmd/fakchat`/`cmd/modelbench` under `-tags
  fakmetal`.

## Benchmarking agentic performance from the second laptop

fak gives you **more cross-laptop than the first draft of this note credited**: not just
turn/token/cost instrumentation but a real, dataset-scored SWE-bench Verified resolve-rate
driven against the Mac's `fak serve`. The honest gap is now narrow and specific — the
final resolve *grade* wants a Docker host, so the loop doesn't yet close on two laptops
**alone**. That residual is the part worth building.

What exists and works over the LAN:

- **`fak agent --base-url http://<macbook-ip>:8080/v1 --model qwen2.5-7b ...`** drives
  a real turn-count A/B against any OpenAI-compatible endpoint (`cmd/fak/main.go:452`).
  Per arm it measures **Turns** (the headline), ToolCalls, ToolErrors,
  Prompt/CompletionTokens, plus fak's own kernel counters (Repairs / VDSOHits / Denies
  / Quarantines). Its task-success bit **is real but single-task**: a booking
  "succeeded" iff the model calls `book` and the mock tool returns a result containing
  "confirmation" without "error" (`internal/agent/loop.go:283-285`). So it's a genuine
  goal-reached signal for **one canned airline-support task**, not a generic grader. It
  also tracks two safety bits: InjectionInContext and DestructiveExecuted.
- **`fak swebench run --agent fleet --gateway <macbook-ip>:8080 --model qwen2.5-7b`** is
  the dataset-scored, end-to-end agentic-quality runner — and it is **shipped**, not a
  thing to write. The fleet runner drives a real read/edit coding loop over
  `/v1/chat/completions` against a running `fak serve` (`internal/swebench/fleet.go:17`),
  clones each repo at its `base_commit`, and emits a real per-instance `git diff`
  predictions file in the exact shape the **official SWE-bench Verified harness** applies
  (`cmd/fak/swebench.go:68-74`). `fak swebench eval --predictions …` then folds the
  real `RESOLVED N / Total (pass rate)` — actual task completion against the canonical
  500-instance set, not a substring stub. **The cross-laptop split:** the second laptop
  drives the MacBook's `fak serve` and collects predictions on the Mac with **no Docker**
  (the runbook is explicit: "runs now, on this Mac, with no GPU / Docker / network" for
  the cost/cache/turns metrics — `docs/benchmarks/SWEBENCH-RESULTS.md`); the **resolve
  grade** is Docker-gated, so `fak swebench eval` either runs it locally or prints the
  exact `python -m swebench.harness.run_evaluation …` to run on whatever box has Docker
  (`docs/benchmarks/SWEBENCH-PURE-KERNEL-RUNBOOK.md:34,100-104`). So the *quality* number
  exists today; only the final grade wants a Linux/Docker host.
- **The economics bench family** (`fanbench`, `fleetbench`, `turntax`, `sessionbench`,
  `radixbench`) measures turn-tax, fan-out, fleet KV-reuse, and cache-hit economics —
  all **measured on the real kernel**, model-independent or model-driven. These price
  what fak's reuse/adjudication buys, which is fak's actual story.
- **`webbench-run`** executes web-agent tasks against a model API and measures real
  token usage — **but its success signal is stubbed** (`navigateToPage`/`executeAction`
  are placeholders; "success" is a substring match for "done"/"complete" or
  `turn==turns`). Treat its token numbers as real and its success rate as not yet real.

What is **missing** is narrower than "a dataset-scored agentic runner" — that ships as
`fak swebench`. The genuine residual is a **Docker-free, Mac-hostable grade** so the
whole loop closes on *two laptops alone* with no Linux box in the picture. Today the
second laptop drives the Mac's `fak serve` and collects real predictions on-device, but
the canonical resolve-rate grade is Docker-gated — so "quality per token" cross-laptop
currently means "Mac serves + collects, a Docker host grades." Two builds would close it
fully: (a) a lightweight in-process grade for the slice of SWE-bench instances whose
tests run without a container (or a τ-bench-style tool-environment eval that needs no
Docker at all), reporting `RESOLVED` next to fak's turn/token/cost counters; and (b)
folding `fak agent`'s and `fak swebench`'s kernel counters (Repairs / VDSOHits / Denies
/ Quarantines) into the same report so you read **quality per token** and **quality per
turn** in one place, not one or the other.

## Recommendation

**Serve Qwen2.5-7B-Instruct Q8 on the MacBook via `fak serve --gguf` (fak's own
forward, pure-Go CPU), bound to `0.0.0.0:8080` with `--require-key-env`.** It is the
strongest model that stays agentically responsive (~8.7 tok/s) while keeping the path
100% fak — the honest "pure fak stack" answer. Use the 27B only as a
"our-own-kernel-runs-the-architecture" showpiece, never for an agent loop. If raw speed
for a *demo* matters more than path-purity, run fak's own forward with `-tags fakmetal`
(prefill on Metal) or front Ollama/llama.cpp-Metal with `fak serve --base-url` — and
label which forward produced the number, the way `BENCHMARK-AUTHORITY.md` does.

**For the cross-laptop agentic benchmark:** you have two shipped paths from the second
laptop today — `fak agent --base-url http://<macbook-ip>:8080/v1` for the
turn/token/safety A/B, and `fak swebench run --agent fleet --gateway <macbook-ip>:8080`
for a **real, dataset-scored** SWE-bench Verified resolve-rate (the Mac serves and
collects predictions Docker-free; the official grade runs on whatever box has Docker).
So "serve the strongest model on a MacBook" can already mean strongest *at getting
agentic work done*, not just at emitting tokens. The one genuinely new piece worth
writing is the **Docker-free Mac-hostable grade** — a container-free resolve slice or a
τ-bench-style tool-environment eval — so the entire quality-per-token loop closes on the
two laptops alone, with fak's kernel counters folded into the same report.
