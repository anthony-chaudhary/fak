# Getting started with fak

This is the install-and-run front door. The dense pitch is in [`README.md`](README.md);
this page gets you from a clean checkout to a running kernel — and to serving a model
behind it — with copy-pasteable commands that were run on a clean build before being
written down.

`fak` is **one Go binary**. There are four things you can do with it, in rising order
of setup cost:

| Tier | What you get | Setup | Downloads |
|---|---|---|---|
| **0 — Try the kernel** | Run/measure the adjudication boundary offline | `go build` | none |
| **1 — Front a real model** | Put the kernel in front of a model you serve elsewhere (Ollama / vLLM / llama.cpp / a cloud provider) | + a running OpenAI-compatible server | a chat model |
| **2 — The fused in-kernel model** | The pure-Go SmolLM2 forward pass the kernel owns | + (real weights) Python export | ~135M params |
| **2b — Expert: Qwen3.6 in-kernel** | Run Qwen3.6-27B through fak's own GGUF->Q8 Gated-DeltaNet path | local GGUF (tokenizer optional — embedded by default) | ~15 GB GGUF, ~26 GB RSS |

If you just want to **serve a useful model with fak in front of it**, you want **Tier 1**.
Tier 2's in-kernel model is a *reference forward pass* proven bit-for-bit against
HuggingFace — not a chat-quality serving engine (see the honest caveat in §4).

> **Prefer not to install anything?** Run these tiers in a hosted cloud notebook — a free
> Colab/Kaggle T4 for Tiers 0–1, a neocloud GPU for Tier 2. See
> [`notebooks/`](../notebooks/README.md).

> **Operator's local-testing default (2026-06-19).** When testing fak *locally*,
> default to **Tier 2 — the fused in-kernel model with real weights** (`fak serve
> --gguf …`), not the Tier 1 proxy and not the synthetic checkpoint. fak's thesis
> is that the model runs inside the kernel address space; local testing should
> exercise that path. The code already agrees — `--engine` defaults to `inkernel`
> (not the offline mock). Reach for **Tier 1** only when you already have a model
> server you want to put fak in front of, and reach for the **synthetic
> checkpoint** (`fak serve --engine inkernel` with no `--gguf` / `FAK_MODEL_DIR`)
> only for explicit wire/API / dispatch-path testing where the model output is
> irrelevant. The biggest model currently exercisable on the in-kernel path on a
> 36 GB M3 Pro is `Qwen3.6-27B.q4_k_m` (≈15 GB GGUF, ≈26 GB RSS with KV); see §4c.

---

## 0. Prerequisites

- **Go 1.26+.** `fak/go.mod` declares `go 1.26`. With Go's default `GOTOOLCHAIN=auto`,
  an older `go` will download the right toolchain automatically on first build (needs
  network once); otherwise install Go 1.26 from <https://go.dev/dl/>. Check with
  `go version`.
- **That's all for Tiers 0 and 2-synthetic** — no GPU, no API key, no network.
- **Tier 1** additionally needs any OpenAI-compatible model server (e.g. Ollama).
- **Tier 2 with real weights** additionally needs **Python 3.10+**; the fetch script
  (§4b) creates a venv and installs `torch`/`transformers` for you.

---

## 1. Get the binary

`fak` is one self-contained, static binary. Pick the path that fits you:

**Adopter — no clone, no Go.** Download the prebuilt binary for your platform from the
[latest release](https://github.com/anthony-chaudhary/fak/releases/latest):

| How | Command |
|---|---|
| **One-liner** (Linux/macOS; checksum-verified) | `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh \| sh` |
| **Manual download** | grab `fak_<version>_<os>_<arch>.tar.gz` (`.zip` on Windows), `tar -xzf` it, move `fak` onto your `PATH` |
| **Docker** (production) | `docker build -t fak https://github.com/anthony-chaudhary/fak.git` then `docker run --rm -p 8080:8080 fak serve --addr 0.0.0.0:8080 …` |

The installer honors `FAK_VERSION` (pin a version) and `FAK_INSTALL_DIR` (default
`/usr/local/bin`, else `~/.local/bin`). Published targets: `linux_amd64`,
`darwin_amd64`, `darwin_arm64`, `windows_amd64`.

**Install with Go.** The module path `github.com/anthony-chaudhary/fak` is the repository
root, so it installs directly:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # -> $(go env GOBIN) / $GOPATH/bin
```

**Contributor — build from the clone:**

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak   (Windows: build with -o fak.exe — see the Windows note)
./fak help
```

> **Windows note.** `go build`/`go vet`/`go run` work natively. Running the *test
> suite* (`go test ./...`) can hit an OS Application-Control policy that blocks the
> freshly-compiled test binaries — that's an OS quirk, not a code failure, and it does
> **not** affect using `fak`. If you need the suite on Windows, run it under WSL with
> `go test ./...`. **On Windows, build with `go build -o fak.exe ./cmd/fak`** — the explicit
> `-o fak` (no extension) leaves a literal `fak` file that cmd.exe / PowerShell cannot launch
> by name (Go only auto-appends `.exe` when you *omit* `-o`; git-bash can still run the
> extensionless binary via its exec bit). Then type the binary as `.\fak.exe` (or `fak` if it's
> on your `PATH`) wherever this guide writes `./fak`.

---

## 2. Tier 0 — try the kernel (zero downloads, ~2 min)

Everything here is offline and deterministic. Run from inside `fak/` (the commands
find `testdata/` relative to the working directory, and write their report files —
`report.json`, `agent-report.json`, … — into the current directory).

**Replay a tool-call trace through the kernel:**

```bash
./fak run --trace testdata/tau2/tau2-smoke.json
```

```
[ 0] get_user_details             verdict=ALLOW     by=monitor   status=OK
[ 1] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[ 2] get_reservation_details      verdict=ALLOW     by=vdso      status=OK
...
summary: submits=12 vdso_hits=6 engine_calls=6 denies=0 transforms=0 quarantines=0
```

`by=vdso` is a call served from the local tool fast-path (no engine call); `by=monitor`
went through to the engine.

**See the capability floor refuse a call (structural, model-independent):**

```bash
./fak preflight --tool create_user --args '{"_positional":["alice"]}'
# verdict=DENY reason=DEFAULT_DENY by=monitor      <- not on the allow-list => fail-closed

./fak preflight --tool get_user_details --args '{}'
# verdict=ALLOW ...                                <- on the allow-list
```

> **cmd.exe note.** The single-quoted `--args '{...}'` works in git-bash and PowerShell but
> **not** cmd.exe, which passes the quotes through literally. On cmd.exe, drop the single quotes
> and escape the inner double quotes (`--args "{""_positional"":[""alice""]}"`) — or simply run
> these examples from git-bash / PowerShell, where the shown syntax works unchanged.

**The headline cost gate and the injection A/B:**

```bash
./fak bench  --suite tau2-smoke      # in-process adjudication p50 vs spawned-hook p50
./fak agent  --offline               # the prompt-injection A/B on the deterministic planner
```

**Inspect / author the deployable capability floor:**

```bash
./fak policy --dump > floor.json     # the built-in default as an editable manifest
# edit floor.json, then:
./fak policy --check floor.json      # validate it (closed refusal vocabulary)
# load it on any verb with: --policy floor.json
```

See [`POLICY.md`](POLICY.md) for the manifest schema.

---

## 3. Tier 1 — put fak in front of a real model (the practical serving path)

`fak serve` is an **OpenAI-compatible gateway that adjudicates tool calls**. You serve a
model with any OpenAI-compatible server; `fak serve --base-url` points at it. On every
`/v1/chat/completions`, fak calls your upstream model, then **denies / repairs /
quarantines the tool calls it proposes at the boundary**, and returns only the admitted
ones (with a `fak` extension describing each decision). fak never executes your tools —
your client does, on the survivors.

Example with [Ollama](https://ollama.com):

```bash
ollama serve &                       # OpenAI-compatible on :11434
until curl -sf http://localhost:11434/api/tags >/dev/null; do sleep 1; done  # wait for it to bind
ollama pull qwen2.5:1.5b

# fak serve runs in the FOREGROUND — Ctrl-C to stop. Run the client calls below
# from a SECOND terminal. To background it: bash -> append ' &' (stop with 'kill %1');
# Windows -> start it in its own window with Start-Process (PowerShell) or `start` (cmd)
# — '&'/'kill %1' are bash-only — then curl from a second terminal.
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b
```

Confirm it's up (from another terminal):

```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"qwen2.5:1.5b","ok":true}   <- engine=inkernel is the
#   dispatch engine for the /v1/fak/* routes — a SEPARATE axis from --base-url. Your
#   Tier-1 upstream model is reached only via /v1/chat/completions, so this is expected.
```

The same `--base-url` swap works for vLLM, a llama.cpp server, or a cloud provider
(`--provider openai|anthropic|gemini|xai`, `--api-key-env YOUR_ENV_VAR`). Point any
OpenAI client at `http://127.0.0.1:8080/v1`.

Routes the gateway exposes:

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | the adjudicating proxy described above |
| `GET /healthz` | unauthenticated liveness (`{"...","ok":true}`) |
| `GET /v1/models` | advertises the served model id |
| `POST /v1/fak/syscall` | run one adjudicated tool call through the kernel directly |
| `POST /v1/fak/adjudicate` | get the verdict for a call without dispatching it |
| `GET /v1/fak/changes`, `POST /v1/fak/revoke` | the cross-agent "what changed" feed / refute a poisoned witness |
| `POST /mcp` | MCP-over-HTTP (`fak serve --stdio` serves MCP over stdin/stdout) |
| `GET /metrics` | Prometheus exposition for gateway HTTP latency/status, verdict counters, kernel counters, inflight requests, build labels, and vDSO hit ratio |
| `GET /debug/vars` | authenticated expvar-style JSON snapshot of gateway config/uptime, runtime memory/goroutines, kernel counters, and completed HTTP/operation metric rows |

> The `/v1/fak/*` routes dispatch to the bound `--engine` (default `mock`, or the
> in-kernel model in Tier 2) — a **separate axis** from `--base-url`. Your upstream
> model is reached only through `/v1/chat/completions`.
>
> `fak serve` also writes one JSON access-log event per HTTP request to its log sink
> (`event=gateway_http_request`, route, status, duration, bytes, and `trace_id`).
> It honors an incoming `X-Trace-Id`; when absent, it mints one, returns it in the
> `X-Trace-Id` response header, and threads it into gateway kernel operations, so
> scrape metrics, per-request logs, per-operation verdict logs
> (`event=gateway_operation`), and kernel events can be correlated without exposing
> request bodies, arguments, or result content.
> `GET /debug/vars` gives operators the same live process view as JSON for break-glass
> checks and one-off probes; it follows the gateway auth policy just like `/metrics`.

Two gateway behaviors to know before you wire a real client to Tier 1:

- **Client sampling params are honored.** The gateway forwards the inbound
  `max_tokens`/`temperature`/`top_p`/`stop` to the upstream model per request (both the
  OpenAI `/v1/chat/completions` and the Anthropic `/v1/messages` wires); an omitted field
  falls through to the planner default, so a client that asks for a long completion is no
  longer hard-capped — the old 1024-token truncation is fixed.
- **SSE is buffered, not token-streaming.** When a client sends `stream:true`, the
  gateway adjudicates the **whole** upstream turn first, then re-serializes the
  finished result as a well-formed SSE event sequence. The wire is identical to a real
  stream (a client parses it the same way), but partial tokens are never emitted — the
  stream carries the already-adjudicated turn, not live decode.
- **Auth.** `--require-key-env VAR` accepts the secret over **either** the
  `Authorization: Bearer <tok>` header (OpenAI/fak-native clients) **or** the
  `x-api-key: <tok>` header that Claude Code and the Anthropic SDKs send.

Harden it for real use:

```bash
./fak serve --addr 0.0.0.0:8080 --base-url … --model … \
  --policy floor.json \               # enforce a reviewable allow-list
  --require-key-env FAK_TOKEN         # require Authorization: Bearer $FAK_TOKEN
```

---

## 4. Tier 2 — run the fused in-kernel model

The kernel can dispatch an allowed tool call to a **real pure-Go SmolLM2 forward pass it
owns** (`--engine inkernel`), decoding over a kernel-owned KV cache. This is the deepest
fusion — the model runs inside the kernel address space — and it's reachable via
`/v1/fak/syscall`.

### 4a. Synthetic weights — instant, zero download

By default `--engine inkernel` runs a small **deterministic synthetic checkpoint**, so the
decode path works with no model export:

```bash
./fak serve --addr 127.0.0.1:8137 --engine inkernel --model smollm2-inkernel &
# stop it later with:  kill %1  (bash)  /  Stop-Process  (PowerShell)  /  Ctrl-C if foreground.
# Windows has no '&'/'kill %1': run it in its own window via Start-Process / `start` instead.

curl -s http://127.0.0.1:8137/healthz
# {"engine":"inkernel","model":"smollm2-inkernel","ok":true}

# the fak-native wire key is "arguments" (NOT "args" — an unknown key is silently dropped):
curl -s -X POST http://127.0.0.1:8137/v1/fak/syscall \
  -H 'Content-Type: application/json' \
  -d '{"tool":"read_file","arguments":{"path":"notes.txt"}}'
# {"verdict":{"kind":"ALLOW","by":"monitor"},
#  "result":{"status":"OK",
#    "content":"{\"tool\":\"read_file\",\"engine\":\"inkernel\",\"model\":\"smollm2-inkernel\",\"generated_tokens\":[125,125,...,125]}",
#    "meta":{"engine":"inkernel","ifc_taint":"trusted","input_tokens":"29","output_tokens":"16"}}}
```

This exercises the **real** in-kernel prefill+decode loop over the kernel-owned KV cache.
The *weights* are random-init synthetic, so the tokens are meaningless — it proves the
dispatch+decode path, not output quality.

### 4b. Real SmolLM2-135M weights — one command

The fused model loads a checkpoint exported from HuggingFace (`config.json` +
`manifest.json` + `weights.f32`). One script does the whole export:

```bash
# from fak/ :
./scripts/fetch-model.sh                       # macOS/Linux/WSL/git-bash
#   - or on Windows PowerShell:
#   ./scripts/fetch-model.ps1

# on success the script prints the exact two lines to run — copy them:
export FAK_MODEL_DIR="$PWD/internal/model/.cache/smollm2-135m"
./fak serve --addr 127.0.0.1:8137 --engine inkernel --model smollm2-135m
```

`FAK_MODEL_DIR` is what actually selects the real weights; `--model` is just the id
advertised on `/v1/models` and `/healthz` (a free-form label).

The script creates a Python venv, installs `torch`/`transformers`/`numpy` (CPU is enough),
downloads `HuggingFaceTB/SmolLM2-135M-Instruct`, and runs
`internal/model/export_oracle.py` into `internal/model/.cache/smollm2-135m`
(git-ignored; regenerable). Preview without doing the work:

```bash
./scripts/fetch-model.sh --check               # report Python + what it would export
FAK_EXPORT_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct ./scripts/fetch-model.sh   # a different model
```

Point any verb that uses the engine at the real weights with `FAK_MODEL_DIR`; if the load
fails the engine falls back to the synthetic checkpoint rather than wedging.

### 4c. Expert smoke: Qwen3.6-27B on pure fak

For the Qwen3.6 goal lane, `cmd/fakchat` can run the real local
`Qwen3.6-27B.q4_k_m.gguf` through fak's own in-kernel Gated-DeltaNet path. This does
not use `fak serve`, llama.cpp, Ollama, or an OpenAI-compatible upstream.

```bash
go run ./cmd/fakchat \
  -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  -tok ~/.cache/fak-models/tokenizers/qwen3.6 \
  -p "Say OK." \
  -n 1
```

On the witnessed M3 Pro run this loaded the model in about 75 s, peaked at about
25.8 GB RSS, prefilling 22 tokens at about 0.5 tok/s and decoding one cached token at
about 0.1 tok/s. The first greedy token is `<think>`, matching llama.cpp for the same
ChatML prompt. Treat this as a runnability/debug smoke; the current speed bar and the
remaining broader logit-oracle work are tracked in `QWEN36-PARITY-RESULTS.md` and
`FAK-NATIVE-QWEN35-RESULTS.md`.

### 4d. In-kernel CHAT through `fak serve` (both OpenAI + Anthropic wires)

`fak serve` can serve the in-kernel model as a **real chat backend** — not just the
byte-tokenized `/v1/fak/syscall` dispatch demo. With `--gguf` and **no** `--base-url`
(a separate `--tokenizer` is optional — the GGUF's embedded tokenizer is used when
omitted), the gateway routes BOTH `/v1/chat/completions` (OpenAI wire) AND
`/v1/messages` (Anthropic wire) through the in-kernel model via `internal/tokenizer`
+ the `cmd/fakchat` ChatML→Prefill→Step recipe (factored into `agent.InKernelPlanner`).
This is the "test fak locally with the model up" path — fak's own engine as the chat
backend, no llama-server/Ollama proxy:

```bash
FAK_Q4K=1 ./fak serve --addr 127.0.0.1:8137 \
  --gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  --tokenizer ~/.cache/fak-models/tokenizers/qwen3.6 \
  --model qwen3.6-27b-q4k
# then from another terminal — both work, same model:
curl -s localhost:8137/v1/chat/completions -d '{"model":"x","messages":[{"role":"user","content":"Say OK."}]}'
curl -s localhost:8137/v1/messages        -d '{"model":"x","max_tokens":48,"messages":[{"role":"user","content":"Say OK."}]}'
```

Witnessed on M3 Pro / Qwen3.6-27B q4_k_m: `/v1/chat/completions` returns
`<think>\n\n</think>\n\nOK`; `/v1/messages` returns a live reasoning trace. Decode
depth/sampling default to a greedy 256-token turn (`FAK_INKERNEL_MAX_TOKENS` /
`FAK_INKERNEL_TEMP` / `FAK_INKERNEL_SEED` override). The planner emits **text** today
(no structured tool-call emission yet); the gateway's adjudication layer still runs on
whatever the caller proposed. `--base-url` (Tier 1 proxy) wins if both are set.

> **Honest caveat (why Tier 2 is not a production chat server).** The
> `fak serve --engine inkernel` SmolLM2 path is proven correct at the *tensor* layer
> against a HuggingFace oracle, and `/v1/fak/syscall` feeds it a bounded **byte-level**
> prompt. `cmd/fakchat` is a separate command-line harness for tokenizer-backed local
> model experiments, including the Qwen3.6 smoke above. These paths make model state
> first-class kernel-owned state; they are not production serving engines. For practical
> chat-quality serving, use **Tier 1**. (This matches the scope in [`CLAIMS.md`](CLAIMS.md)
> and the README's honesty ledger.)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `go: go.mod requires go >= 1.26` | Install Go 1.26 (<https://go.dev/dl/>) or ensure `GOTOOLCHAIN=auto` (the default) with network so it self-fetches. |
| `An Application Control policy has blocked this file` during `go test` (Windows) | OS quirk on test binaries only — run the suite under WSL via `./test.ps1`; the binary itself is unaffected. |
| `fak run`: `no such file testdata/...` | Run from inside `fak/` (traces resolve relative to the working dir), or pass an absolute `--trace`. |
| `fetch-model.sh`: `need python3` | Install Python 3.10+ or set `PYTHON=/path/to/python`. |
| `fetch-model`: offline / can't reach HuggingFace | The export needs network for the first download; the script forces `HF_HUB_OFFLINE=0`. Re-run once online; the HF cache makes repeats offline-safe. |
| `address already in use` on `fak serve` | Pick another `--addr` port. |

## Where to go next

- [`docs/fak/tutorial.md`](../docs/fak/tutorial.md) — **the guided first session**: a
  step-by-step walk through Tiers 0–2 with the real, captured output of every command
  (the friendliest on-ramp if this reference felt dense).
- [`DOGFOOD-CLAUDE.md`](DOGFOOD-CLAUDE.md) — **use it as a product**: one command spins up
  a local model behind the kernel as a native Anthropic `/v1/messages` server and points
  the real Claude Code CLI at it (`./scripts/dogfood-claude.sh`, or `.\scripts\dogfood-claude.ps1`
  on Windows — no ollama, CPU-friendly). Live turns on your own box; witnessed on macOS + Windows.
- [`POLICY.md`](POLICY.md) — the deployable capability floor (the adopter's front door).
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how a new idea bakes in as a package + one registration.
- [`LIVE-RESULTS.md`](docs/benchmarks/LIVE-RESULTS.md) — the live prompt-injection A/B on real models.
- [`CLAIMS.md`](CLAIMS.md) — every capability tagged `[SHIPPED]` / `[SIMULATED]` / `[STUB]`.
