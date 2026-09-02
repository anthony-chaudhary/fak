# Getting started with fak

*This page owns one job: getting a working `fak` binary onto this machine and proving it
runs. It is for a reader who has already decided to try fak and now needs commands and
their checkable results — not the pitch ([README](README.md)), not a route map
([START-HERE](START-HERE.md)), not a concept course ([LEARNING-PATH](LEARNING-PATH.md)).
It is also the one root page that prints the verbatim `fak agent --offline` expected
output; the others link here rather than repeating it.*

**Audience:** new builders choosing an install, proof, or production onboarding path.

Choose one path by the result you need. Each path names its prerequisite and first
checkable result; the default is **Install → Prove offline** before production setup.

| Path | Choose it when | Prerequisite | Start and check |
|---|---|---|---|
| **Install** | You need the `fak` binary on this machine. | Go 1.26+ with `GOTOOLCHAIN=auto`, or a released artifact. | [Get the binary](#1-get-the-binary), then run `fak version`. |
| **Prove offline** *(default after install)* | You want to verify the managed-agent and policy boundary without a key, model, or GPU. | An installed `fak` binary; no model download. | [Run Tier 0](#2-tier-0--try-the-kernel-zero-downloads-2-min), then confirm the task completes while poisoned and destructive actions are blocked. |
| **Run in production** | You are connecting a real agent or client to a durable model path. | An installed binary plus the credentials, model server, or GGUF required by your selected backend. | Use `fak guard` for one existing agent or [configure `fak serve`](#3-tier-1--put-fak-in-front-of-a-real-model-the-practical-serving-path) for a shared endpoint; verify the documented health request before adding clients. |

**Next action:** choose the one row matching your result, verify its prerequisite, and
run that row's linked first check. If this is your first installation, use the default
**Install → Prove offline** sequence.

## Route context

- **Mode:** offline proof uses a deterministic planner; production uses `fak guard` or
  `fak serve` with a real backend. Offline success does not claim live-model quality or latency.
- **Generation:** this is the current `gen/now` builder route. Versioned and research
  procedures remain in their linked benchmark or notes pages rather than defining this path.
- **Lifecycle:** commands here target the current release and repository tip; release-artifact
  verification is documented in [Get the binary](#1-get-the-binary).
- **Support boundary:** backend, accelerator, credential, and operating-system prerequisites
  vary by production choice. The selected section is authoritative for those requirements;
  unsupported or unavailable prerequisites are not silently replaced by the offline proof.

The [guided first session](docs/fak/tutorial.md) expands the default path with captured
output. The [governed-agent quickstart](docs/fak/governed-agent-quickstart.md) is the
fastest route to a running, kernel-governed agent — offline, floor + audit + a visible
DENY — in under 10 minutes. [`README.md`](README.md) is the concise product and mode front
door; this page is the builder's install-and-run reference.

## Already running an agent? Two commands

These are the shortest path from "I already run Claude Code (or any agent)" to "every tool
call it makes is adjudicated". Neither needs the rest of this page:

```bash
fak manage claude                        # your real model, behind the kernel — the one-command proxy
fak manage --gguf qwen2.5:7b -- claude   # a local GGUF model in-kernel — no API key, no network
```

`fak manage` starts the gateway in-process, injects the base URL into the child only (your
shell is untouched), and prints what it allowed vs blocked on exit. The per-harness recipes
are in [`docs/integrations/claude.md`](docs/integrations/claude.md); the fuller versions of
both bullets are under [Where to go next](#where-to-go-next).

---

## 0. Prerequisites

- **Go 1.26+.** `fak/go.mod` declares `go 1.26`. With Go's default `GOTOOLCHAIN=auto`,
  an older `go` will download the right toolchain automatically on first build (needs
  network once); otherwise install Go 1.26 from <https://go.dev/dl/>. Check with
  `go version`.
- **That's all for Tiers 0 and 2-synthetic**: no GPU, no API key, no network.
- **Tier 1** additionally needs any OpenAI-compatible model server (e.g. Ollama).
- **Tier 2 with real weights** additionally needs **Python 3.10+**; the fetch script
  creates a venv and installs `torch`/`transformers` for you. Tier 2 is the
  kernel-developer path and lives on its own page —
  [Tier 2: run the fused in-kernel model](docs/fak/in-kernel-model.md).

---

## 1. Get the binary

`fak` is one self-contained, static binary. Pick the path that fits you:

**Adopter (no clone, no Go).** Download the prebuilt binary for your platform from the
[latest release](https://github.com/anthony-chaudhary/fak/releases/latest):

| How | Command |
|---|---|
| **One-liner** (Linux/macOS; checksum-verified) | `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh \| sh` |
| **Manual download** | grab `fak_<version>_<os>_<arch>.tar.gz` (`.zip` on Windows), `tar -xzf` it, move `fak` onto your `PATH` |
| **Docker** (production) | `docker build -t fak https://github.com/anthony-chaudhary/fak.git` then `docker run --rm -p 8080:8080 fak serve --addr 0.0.0.0:8080 …` |

The installer honors `FAK_VERSION` (pin a version) and `FAK_INSTALL_DIR` (default
`/usr/local/bin`, else `~/.local/bin`). Published targets: `linux_amd64`,
`linux_arm64`, `darwin_amd64`, `darwin_arm64`, `windows_amd64`.

**Verify provenance (optional — stronger than the checksum).** `install.sh` already
checks each download's SHA-256 against the release's `SHA256SUMS`. A checksum only proves
the file matches a *published number* — one a tamperer who rewrote the release could rewrite
too. For a machine-verifiable guarantee that the binary was **built by this repository's CI
from a specific commit**, every release archive **and** the aggregate `SHA256SUMS`
`install.sh` anchors on carry a [SLSA build-provenance attestation](https://github.com/anthony-chaudhary/fak/attestations).
Verify a downloaded asset with the GitHub CLI (the `gh attestation` command set):

```bash
gh attestation verify fak_<version>_<os>_<arch>.tar.gz --repo anthony-chaudhary/fak
gh attestation verify SHA256SUMS                       --repo anthony-chaudhary/fak
```

A successful run names the attested build workflow and the source commit it was built from;
a tampered or unattested asset fails closed. This is the supply-chain check a downstream agent
or CI wants before trusting a `@latest` binary it did not build itself (#1372).

**Install with Go.** The module path `github.com/anthony-chaudhary/fak` is the repository
root, so it installs directly:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # -> $(go env GOBIN) / $GOPATH/bin
```

**Contributor (build from the clone):**

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak   (Windows: build with -o fak.exe — see the Windows note)
./fak help
```

> **Windows note.** `go build`/`go vet`/`go run` work natively. Running the *test
> suite* (`go test ./...`) can hit an OS Application-Control policy that blocks the
> freshly-compiled test binaries. That's an OS quirk, not a code failure, and it does
> **not** affect using `fak`. If you need the suite on Windows, run it under WSL with
> `go test ./...`. **On Windows, build with `go build -o fak.exe ./cmd/fak`.** The explicit
> `-o fak` (no extension) leaves a literal `fak` file that cmd.exe / PowerShell cannot launch
> by name (Go only auto-appends `.exe` when you *omit* `-o`; git-bash can still run the
> extensionless binary via its exec bit). Then type the binary as `.\fak.exe` (or `fak` if it's
> on your `PATH`) wherever this guide writes `./fak`.

---

## 2. Tier 0 — try the kernel (zero downloads, ~2 min)

Everything here is offline and deterministic.

**You do not need a clone for the default proof.** `./fak agent --offline` below runs from
any directory, so the `go install` and one-line-installer paths reach it too. Only the
commands that name a path under `testdata/` — such as the optional `fak run --trace` replay —
must run from inside the clone, or be given an absolute path.

Each command writes its report file (`report.json`, `agent-report.json`) into the current
working directory; pass `--out <path>` to put it elsewhere.

**Run the default offline proof:**

```bash
./fak agent --offline
```

**Expected output** — this block is the one copy carried by a root page; [`README.md`](README.md)
and [`START-HERE.md`](START-HERE.md) link here instead of reprinting it. It is abridged; the
run is deterministic, so you should see the same verdict lines, and the full capture is in
[the tutorial](docs/fak/tutorial.md):

```
metric                        now(base)          fak
--------------------------   ----------   ----------
injection in context                YES           no
destructive op executed             YES           no
task completed (booked)             YES          YES

HEADLINE
  poisoned result blocked   : YES
  destructive op prevented  : YES

report written: agent-report.json
```

The proof passes when `task completed (booked)` is `YES` in both arms while
`poisoned result blocked` and `destructive op prevented` are both `YES` for fak.
This deterministic planner exercises the managed-agent and policy boundary; it does
not measure live-model quality or latency.

For trace replay, adjudication latency, and individual capability-floor checks, continue
with the optional diagnostics below or use the full [reproduction packet](docs/repro-packet.md).

**Optional trace replay:**

```bash
./fak run --trace testdata/tau2/tau2-smoke.json
```

A successful replay ends with a `summary:` of submit, local-hit, engine-call, and verdict counts.

**Optional adjudication latency check:**

```bash
./fak bench --suite tau2-smoke
```

**See the capability floor refuse a call (structural, model-independent):**

```bash
./fak preflight --tool create_user --args '{"_positional":["alice"]}'
# verdict=DENY reason=DEFAULT_DENY by=monitor      <- not on the allow-list => fail-closed

./fak preflight --tool get_user_details --args '{}'
# verdict=ALLOW ...                                <- on the allow-list
```

> **Windows shell note.** The single-quoted `--args '{...}'` works unchanged in git-bash and`n> PowerShell 7. Windows PowerShell 5.1 strips the inner JSON quotes at the native-process boundary;`n> use `--args '{\"_positional\":[\"alice\"]}'` there. On cmd.exe, use`n> `--args "{""_positional"":[""alice""]}"`.

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

The syscall executor used by `fak serve`, trace replay, and `fak guard` defaults to
`mock`: a cheap deterministic tool-result path, not synthetic token generation. Use
`--engine inkernel` only when you intentionally want fak-native model execution.
Explicit GGUF and real-model paths remain fak-native; fak does not silently switch them
to llama.cpp.

`fak serve` is an **OpenAI-compatible gateway that adjudicates tool calls**. You serve a
model with any OpenAI-compatible server; `fak serve --base-url` points at it. On every
`/v1/chat/completions`, fak calls your upstream model, then **denies / repairs /
quarantines the tool calls it proposes at the boundary**, and returns only the admitted
ones (with a `fak` extension describing each decision). fak never executes your tools.
Your client does, on the survivors.

Example with [Ollama](https://ollama.com):

```bash
ollama serve &                       # OpenAI-compatible on :11434
until curl -sf http://localhost:11434/api/tags >/dev/null; do sleep 1; done  # wait for it to bind
ollama pull qwen2.5:1.5b

# fak serve runs in the FOREGROUND (Ctrl-C to stop). Run the client calls below
# from a SECOND terminal. To background it: bash -> append ' &' (stop with 'kill %1');
# Windows -> start it in its own window with Start-Process (PowerShell) or `start` (cmd),
# since '&'/'kill %1' are bash-only. Then curl from a second terminal.
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b
```

Confirm it's up (from another terminal):

```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"qwen2.5:1.5b","ok":true}   <- engine=inkernel is the
#   dispatch engine for the /v1/fak/* routes, a SEPARATE axis from --base-url. Your
#   Tier-1 upstream model is reached only via /v1/chat/completions, so this is expected.
```

The same `--base-url` swap works for vLLM, a llama.cpp server, or a cloud provider
(`--provider openai|anthropic|gemini|xai`, `--api-key-env YOUR_ENV_VAR`). Point any
OpenAI client at `http://127.0.0.1:8080/v1`.

Routes the gateway exposes:

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | the adjudicating proxy described above (OpenAI wire) |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | the same adjudicating proxy on the Anthropic wire + its token counter |
| `POST /v1/embeddings`, `POST /v1/moderations` | OpenAI-compatible embeddings / moderations passthrough |
| `GET /healthz` | unauthenticated liveness (`{"...","ok":true}`) |
| `GET /v1/models` | advertises the served model id |
| `POST /v1/fak/syscall` | run one adjudicated tool call through the kernel directly |
| `POST /v1/fak/adjudicate` | get the verdict for a call without dispatching it |
| `POST /v1/fak/admit` | admit a tool *result* through the quarantine/IFC gate without a call |
| `GET /v1/fak/changes`, `POST /v1/fak/revoke` | the cross-agent "what changed" feed / refute a poisoned witness |
| `GET /v1/fak/events` | drain the durable, hash-chained audit journal after a `?since=` cursor (404 unless `FAK_AUDIT_JOURNAL` is set) |
| `POST /v1/fak/context/change` | record a safe requester-initiated mutation (e.g. a recall-page tombstone) |
| `POST /v1/fak/policy/reload` | reload the configured policy manifest in place |
| `POST /v1/fak/trace/reset` | reset the per-trace IFC taint state |
| `POST /mcp` | MCP-over-HTTP (`fak serve --stdio` serves MCP over stdin/stdout) |
| `GET /metrics` | Prometheus exposition for gateway HTTP latency/status, verdict counters, kernel counters, inflight requests, build labels, and vDSO hit ratio |
| `GET /debug/vars` | authenticated expvar-style JSON snapshot of gateway config/uptime, runtime memory/goroutines, kernel counters, and completed HTTP/operation metric rows |

> The `/v1/fak/*` routes dispatch to the bound `--engine` (default `inkernel` — the
> Tier-2 fused model, on its synthetic checkpoint until you load real weights; see
> [Tier 2](docs/fak/in-kernel-model.md)), a **separate axis** from `--base-url`. Your upstream
> model is reached only through `/v1/chat/completions`.

> `fak serve` also writes one JSON access-log event per HTTP request to its log sink.
> The `event=gateway_http_request` line carries route and status, duration and bytes, plus `trace_id`.
> It honors an incoming `X-Trace-Id`; when absent, it mints one, returns it in the
> `X-Trace-Id` response header, and threads it into gateway kernel operations. The id
> ties together scrape metrics, per-request logs, per-operation verdict logs
> (`event=gateway_operation`), and kernel events. They can all be correlated without exposing
> request bodies, arguments, or result content.

> `GET /debug/vars` gives operators the same live process view as JSON for break-glass
> checks and one-off probes; it follows the gateway auth policy just like `/metrics`.

Two gateway behaviors to know before you wire a real client to Tier 1:

- **Client sampling params are honored.** The gateway forwards the inbound
  `max_tokens`/`temperature`/`top_p`/`stop` to the upstream model per request (both the
  OpenAI `/v1/chat/completions` and the Anthropic `/v1/messages` wires). An omitted field
  falls through to the planner default, so a client that asks for a long completion is no
  longer hard-capped; the old 1024-token truncation is fixed.
- **SSE is buffered rather than token-streaming.** When a client sends `stream:true`, the
  gateway adjudicates the **whole** upstream turn first, then re-serializes the
  finished result as a well-formed SSE event sequence. The wire is identical to a real
  stream (a client parses it the same way), but partial tokens are never emitted. The
  stream carries the already-adjudicated turn rather than live decode. This is a
  consequence of whole-turn adjudication, not a missing feature — a tool call cannot
  be allowed/denied/repaired until its arguments fully arrive (see the honest-scope
  note in `POLICY.md`). Expect full-turn latency, not token-by-token streaming.
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

## 4. Tier 2 — run the fused in-kernel model *(kernel developers)*

**The newcomer route ends above.** Tiers 0 and 1 are everything you need to install `fak`,
prove the boundary offline, and put it in front of the model you already run. Tier 2 is for
people working on `fak`'s **own** inference kernel — it is optional, and skipping it costs you
nothing.

The kernel can dispatch an allowed tool call to a **real pure-Go SmolLM2 forward pass it
owns** (`--engine inkernel`), decoding over a kernel-owned KV cache. This is the deepest
fusion: the model runs inside the kernel address space, and it's reachable via
`/v1/fak/syscall`.

**→ [Tier 2: run the fused in-kernel model](docs/fak/in-kernel-model.md)** — the synthetic
checkpoint and raw `/v1/fak/syscall` JSON (was §4a), the one-command SmolLM2-135M export
(§4b), the Qwen3.6-27B GGUF smoke through `cmd/fakchat` (§4c), in-kernel chat through
`fak serve --gguf` on both the OpenAI and Anthropic wires (§4d), and the honest caveat on
why Tier 2 is not a production chat server.

Otherwise, continue to [Where to go next](#where-to-go-next).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `go: go.mod requires go >= 1.26` | Install Go 1.26 (<https://go.dev/dl/>) or ensure `GOTOOLCHAIN=auto` (the default) with network so it self-fetches. |
| `An Application Control policy has blocked this file` during `go test` (Windows) | OS quirk on test binaries only — run the suite under WSL via `./test.ps1`; the binary itself is unaffected. |
| `fak run`: `no such file testdata/...` | Run from inside `fak/` (traces resolve relative to the working dir), or pass an absolute `--trace`. |
| `address already in use` on `fak serve` | Pick another `--addr` port. |

That table is **build-time**. If the build succeeded but your first *run* misbehaved —
`fak: command not found`, canned replies because `--base-url` was empty, an upstream `401`,
a tool call refused by the default-deny floor, or the serve port already taken — go to
[**It didn't work: the five most likely fak first-run failures**](docs/adoption/troubleshooting-first-run.md),
which gives each one a symptom, a cause, and a one-line fix.

Tier 2's own symptoms (`fetch-model.sh` needing Python, or offline) are on
[the Tier 2 page](docs/fak/in-kernel-model.md#troubleshooting).

## Where to go next

- **`fak guard --gguf <model> -- claude`: local model, one command.** Run Claude Code (or any OpenAI-compatible agent) with a local GGUF model behind the kernel — no API key, no network, no second terminal. The model loads in-kernel, the kernel adjudicates every tool call, and your data never leaves your box. Example: `fak guard --gguf qwen2.5:7b -- claude` (downloads on first run, ~5 GB cached). Small-model agentic quality is a ramp; for frontier-quality coding, `fak guard -- claude` (proxy to Anthropic) is still the default. See [`docs/integrations/claude.md`](docs/integrations/claude.md). For the runbook that measures local vs frontier coding on a minimal CPU-runnable fixture, see [`docs/benchmarks/LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md`](docs/benchmarks/LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md).
- **`fak guard -- claude`: the one-command proxy front door.** Run the Claude Code (or any agent) you already use, with the kernel adjudicating every tool call it proposes. It starts the gateway in-process, injects the base URL into the child only (your shell is untouched), proxies your real Anthropic key + prompt cache through in passthrough mode, and prints what it allowed vs blocked on exit. No script, no second terminal, any OS. Embedded secure floor (`fak guard --dump-policy` to see it). See [`docs/integrations/claude.md`](docs/integrations/claude.md).
- [`docs/fak/tutorial.md`](docs/fak/tutorial.md): **the guided first session**. It walks
  step by step through Tiers 0–2 with the real, captured output of every command
  (the friendliest on-ramp if this reference felt dense).
- [`DOGFOOD-CLAUDE.md`](DOGFOOD-CLAUDE.md): **use it as a product**. One command spins up
  a local model behind the kernel as a native Anthropic `/v1/messages` server and points
  the real Claude Code CLI at it (`./scripts/dogfood-claude.sh`, or `.\scripts\dogfood-claude.ps1`
  on Windows; no ollama, CPU-friendly). Live turns on your own box; witnessed on macOS + Windows.
- [`POLICY.md`](POLICY.md): the deployable capability floor (the adopter's front door).
- [`ARCHITECTURE.md`](ARCHITECTURE.md): how a new idea bakes in as a package + one registration.
- [`LIVE-RESULTS.md`](docs/benchmarks/LIVE-RESULTS.md): the live prompt-injection A/B on real models.
- [`CLAIMS.md`](CLAIMS.md): every capability tagged `[SHIPPED]` / `[SIMULATED]` / `[STUB]`.
