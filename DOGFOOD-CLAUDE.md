# Dogfood: Claude Code on our own local kernel-fronted model

> One command spins up a local model, puts the **fak kernel** in front of it as a
> **native Anthropic Messages server**, and points the **real Claude Code CLI** at
> it. We use the product the way an adopter would — live turns, our own server, our
> own harness — and every tool call Claude proposes is adjudicated by the kernel
> before Claude ever sees it.

```
 ┌─────────────┐   POST /v1/messages   ┌────────────────────────┐   /v1/chat/...   ┌──────────────┐
 │ Claude Code │ ───────────────────▶  │  fak serve (the kernel) │ ───────────────▶ │  local model │
 │ (the harness)│ ◀──── SSE stream ───  │  adjudicates every tool │ ◀────────────── │ ollama / shim│
 └─────────────┘                        └────────────────────────┘                  └──────────────┘
        ▲ ANTHROPIC_BASE_URL=http://127.0.0.1:8080            kernel drops / repairs proposed tool calls
        │ CLAUDE_CONFIG_DIR chosen by tools/fleet_accounts.py (the account switcher)
```

## First-class: Claude Code on the REAL Claude API, through fak

The headline experience is not a toy local model — it's **us, in Claude Code, on the
real Anthropic API, with the kernel adjudicating our actual coding turns.** fak sits
as a transparent hop in front of `api.anthropic.com`:

```
 ┌─────────────┐  POST /v1/messages  ┌────────────────────────┐  /v1/messages  ┌──────────────────┐
 │ Claude Code │ ─────────────────▶  │ fak serve (the kernel)  │ ─────────────▶ │ api.anthropic.com │
 │ (the harness)│ ◀──── SSE stream ─  │ adjudicates every tool  │ ◀──────────── │  (real Claude)    │
 └─────────────┘                      └────────────────────────┘                └──────────────────┘
   --provider anthropic --base-url https://api.anthropic.com      your own key + cache_control pass through
```

```bash
# macOS / Linux
FAK_DOGFOOD_BACKEND=anthropic ./scripts/dogfood-claude.sh --probe "say pong"
FAK_DOGFOOD_BACKEND=anthropic ./scripts/dogfood-claude.sh           # interactive, real models
```
```powershell
# Windows
$env:FAK_DOGFOOD_BACKEND='anthropic'; .\scripts\dogfood-claude.ps1 --probe "say pong"
```

Or wire it by hand — `fak serve` already speaks both wires natively:

```bash
fak serve --provider anthropic --base-url https://api.anthropic.com
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude   # your normal claude, now kernel-adjudicated
```

**Why this is transparent (the two design facts that make it first-class):**

1. **Your own key and real model tiers flow through.** The kernel only adjudicates
   the *response's* proposed tool calls — it never mutates the request. So the
   `anthropic` backend does **not** pin a placeholder key or remap model tiers: Claude
   Code keeps using `claude-opus-4-8` (etc.) and its own credential. The inbound
   credential is forwarded to the upstream (`anthropicInboundKey` →
   `WithUpstreamAPIKey`) under the scheme the token itself implies — a plain key as
   `x-api-key`, a Claude Pro/Max **subscription** OAuth token (`sk-ant-oat…`) as
   `Authorization: Bearer` + `anthropic-beta: oauth-2025-04-20` (the only scheme the
   API accepts it under). fak holds no second secret. So a **subscription is the default**
   for `fak guard -- claude` when no `ANTHROPIC_API_KEY` is set: fak sources the OAuth
   token (`CLAUDE_CODE_OAUTH_TOKEN` → `<config>/.oauth-token` → `~/.claude/.credentials.json`)
   and sends it upstream as a bearer token. Set `ANTHROPIC_API_KEY` to use API billing
   instead; `--anthropic-oauth` forces the subscription path.

2. **Prompt caching survives byte-for-byte.** Anthropic prompt caching is a *byte-exact
   prefix hash*. fak's canonical transcript is lossy (it flattens `system` blocks and
   re-marshals tool schemas), which would miss the cache on every turn and re-bill the
   full ~5K-token prefix. In passthrough mode the gateway forwards the **original
   request bytes unchanged** (`AnthropicMessagesRequest.Raw` → `WithRawRequestBody`),
   so the client's `cache_control` breakpoints land intact upstream and the
   `cache_read_input_tokens` hit is reported straight back to Claude Code. Without
   this, a kernel-in-the-middle is a silent ~10× cost regression on daily use.

Witnessed end-to-end by `TestAnthropicMessagesPassthroughPreservesCacheAndAdjudicates`
(`internal/gateway`): the upstream receives the inbound bytes verbatim, the client's
key is forwarded, a denied tool call is still stripped, and the upstream cache-read
count reaches the client.

> The rest of this doc covers the **local-model** path — point fak at ollama/a shim/a
> large local OpenAI-compatible server instead of the real API. That path proves the
> wire and the kernel boundary without burning API tokens; the real-API path above is
> the same kernel, same `/v1/messages` front door, with the upstream swapped.

## The one command

```bash
./scripts/dogfood-claude.sh                  # interactive Claude Code on the local model
./scripts/dogfood-claude.sh --probe "hi"     # ONE headless live turn (witnessable), then exit
./scripts/dogfood-claude.sh --smoke          # curl the wire end-to-end (no model needed), then exit
./scripts/dogfood-claude.sh --print-env      # the export lines for your own `claude` invocation
./scripts/dogfood-claude.sh --list-accounts  # the account switcher roster
./scripts/dogfood-claude.sh --install        # symlink `fak`, `fak-dogfood`, and `fak-qwen36-claude` onto PATH
```

### Run it from anywhere (one line)

```bash
./scripts/dogfood-claude.sh --install        # one-time: installs PATH launchers
fak-dogfood --smoke                           # then, from ANY directory:
fak-dogfood --probe "hi"                      #   one witnessable live turn
fak-dogfood                                   #   interactive
fak-qwen36-claude --probe "hi"                #   Qwen3.6 local preset
fak serve --help                              #   repo CLI from PATH
```

`--install` is idempotent and picks the first writable dir among `~/.local/bin`,
`/opt/homebrew/bin`, `/usr/local/bin` (override with `FAK_DOGFOOD_BINDIR`). The
launchers are symlinks; the script resolves them back to the repo, so they always run
the in-tree code. It also builds `tools/.bin/fak` and symlinks it as `fak`, so
manual commands like `fak serve --help` work after install.

**It cannot damage your normal `claude`.** Every wiring env var
(`ANTHROPIC_BASE_URL`, `CLAUDE_CONFIG_DIR`, the model tiers) is exported only into
the child `claude` process the script spawns — it never touches the parent shell, so
a `claude` in another terminal is unaffected. `CLAUDE_CONFIG_DIR` points at the
isolated `~/.claude-faklocal` account, never the default `~/.claude`; the script
writes to neither your shell rc nor `settings.json`. Verified: a probe run leaves
`~/.claude/settings.json` byte-identical and the parent shell's `ANTHROPIC_BASE_URL`
unset.

It builds `fak`, ensures a local model is being served (ollama by default — Metal,
tool-capable; `FAK_DOGFOOD_BACKEND=shim` uses the in-tree `local_shim.py` instead),
starts `fak serve` in front of it, resolves the `.claude` config dir through the
**account switcher** (`tools/fleet_accounts.py`, defaulting to an isolated
`.claude-faklocal` dogfood account), exports the Claude Code wiring, and launches
Claude Code (or runs a single headless turn for `--probe`). It tears the kernel down
on exit.

> **One canonical resolve path.** Every front door (this launcher, `launch_goal_detached.ps1`,
> `issue_dispatch`) picks its account through the switcher's single subcommand,
> `fleet_accounts.py resolve` — one call returns the `config_dir`, the long-lived
> `oauth_token`, and the model tier in a flat record (pin a tag with `--account`, take the
> isolated dogfood account with `--faklocal-ok`). No front door re-implements the
> roster+route+token dance.

> **Fanning out across accounts?** `fleet_accounts.py wave --count N` hands each lane a
> *distinct* account, so a burst doesn't pile onto one rate-limit pool (a single-account
> `resolve` returns the same account N times in a burst). It's a per-account rate-limit
> load balancer — operator fleet plumbing, not a kernel feature.

Knobs: `FAK_DOGFOOD_PORT` (8080), `FAK_DOGFOOD_MODEL` (override; default = the
**largest installed** ollama model, auto-upgraded to `FAK_DOGFOOD_FALLBACK_MODEL`
when the box has only tiny <=3B models), `FAK_DOGFOOD_FALLBACK_MODEL`
(`qwen2.5-coder:7b`), `FAK_DOGFOOD_BACKEND` (ollama|shim|openai),
`FAK_DOGFOOD_BASE_URL` (OpenAI-compatible upstream for `openai`, e.g. a
`llama-server` `/v1` URL), `FAK_DOGFOOD_TIMEOUT_S` (planner/write timeout; default
300s, or 900s for `openai`), `FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON` (optional JSON
merged into upstream requests), `FAK_DOGFOOD_PRESET` (`qwen36-local`),
`FAK_DOGFOOD_CLAUDE_DEBUG` (default `api`; set `0`/`off` to disable),
`FAK_DOGFOOD_CLAUDE_DEBUG_FILE` (optional Claude debug file),
`FAK_DOGFOOD_ACCOUNT` (switcher tag), `FAK_DOGFOOD_POLICY`
(capability-floor manifest to enforce).

> **Pick a model that can actually drive the agentic loop.** A tiny model
> (`qwen2.5:1.5b` and similar ≤3B) intermittently emits malformed/raw tool calls
> under Claude Code's large multi-tool prompt — the script warns when it resolves
> one. The kernel now lifts text-form `<tool_call>` blocks back onto the
> adjudication path (so they no longer render raw or bypass the kernel), but for a
> genuinely usable session use a 7B+ tool model, e.g.
> `FAK_DOGFOOD_MODEL=qwen2.5-coder:7b fak-dogfood`. The launcher also fails loud if
> the port is already held by a prior kernel (so you never silently attach to a
> stale one), and `/healthz` now reports a `"planner"` field — `"proxy"` (a live
> `--base-url` upstream), `"inkernel"` (a `--gguf` fused model), or `"mock"` (the
> scripted offline fallback) — so a probe can tell a real backend from the mock.

### Large local OpenAI-compatible model

For a large local model already served by `llama-server`, LM Studio, vLLM, or SGLang,
point the dogfood launcher at that endpoint instead of Ollama. The focused Qwen3.6
Claude Code playbook is
[`docs/qwen36-claude-dogfood-playbook.md`](docs/qwen36-claude-dogfood-playbook.md):

```bash
fak-qwen36-claude --probe "Reply with exactly the word: pong"
```

The `fak-qwen36-claude` launcher is the `qwen36-local` preset: backend `openai`,
base URL `http://127.0.0.1:8131/v1`, model
`lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M`, Qwen extra body, and Claude
`--debug api`. The explicit equivalent is:

```bash
FAK_DOGFOOD_BACKEND=openai \
FAK_DOGFOOD_BASE_URL=http://127.0.0.1:8131/v1 \
FAK_DOGFOOD_MODEL=lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M \
FAK_DOGFOOD_TIMEOUT_S=900 \
FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON='{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}' \
fak-dogfood --probe "Reply with exactly the word: pong"
```

If `FAK_DOGFOOD_MODEL` is omitted on a generic OpenAI-compatible run, the launcher
reads the first id from `$FAK_DOGFOOD_BASE_URL/models`.

The launcher raises `API_TIMEOUT_MS` for the child Claude Code process from
`FAK_DOGFOOD_TIMEOUT_S`, and the gateway emits Anthropic SSE `ping` events while a
slow local upstream is still generating. That keeps full Claude Code prompts alive
on large local models whose first token can take minutes.

For request debugging, `--probe` writes Claude stderr/debug to `/tmp/fak-claude.log`
and the gateway writes `/tmp/fak-serve.log`. The Grafana stack provisions a
dedicated **FAK Dogfood Slow Requests** dashboard for `/v1/messages` latency,
in-flight requests, status mix, and kernel activity.

## The capability floor (what's allowed, what's denied)

With **no** policy the kernel default-denies *every* tool, so Claude Code can do
nothing. The launcher therefore loads a sensible default floor,
`examples/dogfood-claude-policy.json` (override with `FAK_DOGFOOD_POLICY=<path>`, or
`FAK_DOGFOOD_POLICY=none` to run the raw fail-closed kernel). The floor **allows the
standard Claude Code tool set** — `Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`,
`Task`, … — so interactive sessions work, while the kernel still adjudicates *every*
call. The interesting part is what it refuses:

| Try this in the session | Verdict | Why |
|---|---|---|
| `ls`, `cat`, `grep`, `git commit` | ✅ allowed | everyday dev work |
| `rm -rf …`, `rm -f …` | ⛔ `POLICY_BLOCK` | destructive removal, denied by **argument value** |
| `go build -o ../x`, `> ../x`, `cp .. ../x` | ⛔ `POLICY_BLOCK` | write that **escapes the repo** into a sibling tree — see [`docs/repo-guard.md`](docs/repo-guard.md) |
| `sudo …` | ⛔ `POLICY_BLOCK` | privilege escalation |
| `git push …` | ⛔ `POLICY_BLOCK` | the agent can commit but not publish |
| `curl … \| sh`, `:(){ :\|:& };:`, `dd if=… of=/dev/sd…` | ⛔ `POLICY_BLOCK` | RCE pipe / fork bomb / disk wipe |
| `Edit`/`Write` into `.git/`, `.ssh/`, `internal/kernel/`, `VERSION` | ⛔ `SELF_MODIFY` | can't rewrite the kernel or secrets |
| any tool the floor never named | ⛔ `DEFAULT_DENY` | fail-closed |

The deny is on the **argument**, not just the tool name: `Bash` is allowed, but
`Bash{command:"rm -rf /"}` is refused *before the shell sees it*. Check any call
without launching a session:

```bash
fak preflight --tool Bash --args '{"command":"rm -rf /tmp/x"}' \
  --policy examples/dogfood-claude-policy.json   # => verdict=DENY reason=POLICY_BLOCK
fak policy --check examples/dogfood-claude-policy.json   # eyeball the whole floor
```

## Windows (PowerShell): one command, no ollama

The bash script above is macOS/Linux-shaped (ollama+Metal default, `/opt/homebrew`,
`$HOME` POSIX paths). On Windows use the native twin — same flow, no ollama needed:

```powershell
.\scripts\dogfood-claude.ps1 --smoke                 # curl the wire end-to-end (no model), then exit
.\scripts\dogfood-claude.ps1 --probe "say pong"      # ONE headless live turn (witnessable), then exit
.\scripts\dogfood-claude.ps1                          # interactive Claude Code on the local model
.\scripts\dogfood-claude.ps1 --print-env             # the $env: lines for your own `claude` invocation
.\scripts\dogfood-claude.ps1 --list-accounts         # the account switcher roster
.\scripts\dogfood-claude.ps1 --install               # copy fak.exe + a fak-dogfood.cmd shim onto PATH, then exit
```

### Run it from anywhere (Windows)

```powershell
.\scripts\dogfood-claude.ps1 --install   # one-time: copies fak.exe + writes fak-dogfood.cmd
fak-dogfood --smoke                        # then, from ANY directory:
fak-dogfood --probe "say pong"             #   one witnessable live turn
fak serve --help                           #   repo CLI from PATH
```

Windows symlinks need elevation/dev-mode, so `--install` **copies** the built `fak.exe`
and writes a `fak-dogfood.cmd` shim (which re-invokes the in-tree script) into
`%USERPROFILE%\bin` (override with `FAK_DOGFOOD_BINDIR`). It prints a `setx PATH` hint if
that dir isn't on PATH yet. Re-run `--install` to refresh the `fak.exe` copy after a
rebuild (the shim always runs the current in-tree script).

What differs (so it works on a CPU-only Windows box out of the box):

- **Backend = the in-tree transformers `shim`** (no ollama dependency). It loads a
  cached HF model via `python local_shim.py` and serves it OpenAI-compatible; the
  kernel fronts it exactly as it fronts ollama.
- **Model defaults to `HuggingFaceTB/SmolLM2-135M-Instruct`** — *the* knob that makes
  CPU serving usable. Measured on this 16-core Windows host (transformers, fp32, CPU):
  a ~2.4K-token prefill takes **3.6 s** on SmolLM2-135M vs **304 s** on Qwen-1.5B
  (~85× — fp32 CPU attention is the wall, not threading; torch already used all 16
  cores). The proven-on-Mac probe was fast only because ollama's Metal GPU + its
  2048-token default context did the prefill; a CPU shim feeds the *full* prompt, so
  the small model is what keeps a turn at seconds.
- **GPU when present.** The shim auto-detects CUDA: it loads in fp16 on the GPU when
  `torch.cuda.is_available()`, otherwise fp32 on CPU (force either with
  `FAK_SHIM_DEVICE=cuda|cpu`). With a GPU a turn lands in seconds regardless of prompt
  size; the CPU numbers above are the floor for a GPU-less host.
- **Port auto-bump**: if `:8080` (or the shim port) is busy — common on a shared
  fleet box — it walks to the next free port instead of failing.
- **Windows paths + PowerShell process management**; `CLAUDE_CONFIG_DIR` is the
  Windows `%USERPROFILE%\.claude-faklocal`, resolved through the same account switcher.

Same isolation guarantee: every wiring var is set only on the child `claude`
PowerShell spawns, never your profile; the kernel + shim are killed on exit.

Knobs (env): `FAK_DOGFOOD_MODEL` (point at e.g. `Qwen/Qwen2.5-1.5B-Instruct` for
stronger output — but expect minutes/turn on CPU), `FAK_DOGFOOD_PORT` (8080),
`FAK_DOGFOOD_SHIM_PORT` (8190), `FAK_DOGFOOD_BACKEND` (shim|ollama), `FAK_PYTHON`
(python), `FAK_DOGFOOD_ACCOUNT` (switcher tag).

**Timeouts are pre-raised for you.** Claude Code sends a large prompt (its full system
prompt + every tool schema, ~5–6K tokens), so a turn can exceed `fak serve`'s default
60 s planner / 90 s HTTP `WriteTimeout` — on CPU it gets cut off with a 502. The launcher
therefore defaults both `FAK_PLANNER_TIMEOUT_S` and `FAK_HTTP_WRITE_TIMEOUT_S` to
**300 s** (override either before launching; `0` disables the write timeout). On the GPU
a turn is well under that ceiling; on CPU the raised ceiling is what lets it finish.

### Witnessed live on Windows

`scripts/dogfood-claude.ps1 --probe` → `experiments/agent-live/dogfood-claude-probe-win.json`:

```json
{"type":"result","subtype":"success","is_error":false,"num_turns":1,
 "stop_reason":"end_turn","terminal_reason":"completed","duration_ms":35622,
 "modelUsage":{"HuggingFaceTB/SmolLM2-135M-Instruct":{"inputTokens":5816,"outputTokens":256}}}
```

The real Claude Code CLI (v2.1.181) completed a turn against the local kernel-fronted
model in ~36 s on this Windows host — no ollama, the in-tree shim auto-fronted the model
on the box's GPU (fp16 CUDA); a GPU-less host runs the identical path on fp32 CPU with
the pre-raised timeouts. The 135M model's *answer* is weak — that is the model-quality
caveat below, not a wire/kernel issue.

## What this added (and what it reused)

**New, additive code — `fak serve` now also speaks the Anthropic Messages API natively:**

| Route | Behavior |
|---|---|
| `POST /v1/messages` | adjudication proxy on the Anthropic wire — buffered JSON, or **SSE** when `"stream":true` |
| `POST /v1/messages/count_tokens` | cheap tokenizer-free input-token estimate |

These run **alongside** the existing `/v1/chat/completions`, both backed by the same
planner + kernel. `--provider` still selects the *upstream* wire (the local model is
OpenAI-compatible); the Anthropic surface is the *downstream* wire we serve. The
implementation is the structural inverse of the already-tested client adapter:

- `internal/agent/anthropic_server.go` — decodes an inbound `/v1/messages` request
  (system-as-string-or-blocks, `tool_use`/`tool_result` blocks, parallel tool results)
  into the canonical transcript, and renders a `Completion` back into Anthropic
  content blocks + `stop_reason`. The buffered upstream completion is re-serialized as
  a well-formed SSE event sequence (no true token streaming needed — Claude Code parses
  it identically, and the `tool_use` ids survive the round trip).
- `internal/gateway/messages.go` — the HTTP + SSE handlers, reusing the shared
  `adjudicateProposed` filter (extracted from `handleChatCompletions`, so both
  protocols run the *identical* kernel boundary).

**Reused (≈95% of the agent setup was already here):**
- the Anthropic wire types + client adapter (`internal/agent/adapters.go`),
- the gateway's planner, kernel, and per-call adjudication (`internal/gateway/`),
- the local-model serving patterns (`fak serve`, `experiments/agent-live/local_shim.py`),
- the account switcher (`tools/fleet_accounts.py`, via `CLAUDE_CONFIG_DIR`).

## Required items — each proven by a witness

| # | Required item | Witness | Status |
|---|---|---|---|
| 1 | `fak` serves a native Anthropic `/v1/messages` endpoint | `internal/gateway/messages.go`; `TestAnthropicMessagesFiltersAndRepairs` | ✅ |
| 2 | Streaming (`stream:true`) emits the full Anthropic SSE sequence | `TestAnthropicMessagesSSE` (message_start…message_stop, `input_json_delta`) | ✅ |
| 3 | Tool-use `id`s survive the round trip (results match back) | `TestDecodeAnthropicMessagesRequest`, `TestAnthropicMessagesFiltersAndRepairs` | ✅ |
| 4 | The kernel adjudicates every proposed call on the Anthropic wire | shared `adjudicateProposed`; deny dropped + transform repaired in the test | ✅ |
| 5 | `count_tokens` answered (optional, but implemented) | `TestAnthropicCountTokens` | ✅ |
| 6 | A **real local model** completes a live turn through `/v1/messages` | curl → qwen2.5:1.5b emitted a `get_weather` `tool_use` (see below) | ✅ |
| 7 | The **real Claude Code CLI** completes a live turn against the kernel | `experiments/agent-live/dogfood-claude-probe.json` — includes the Qwen3.6 witness `result:"pong"` and `duration_ms:218024` | ✅ |
| 8 | One command spins the whole stack up | `scripts/dogfood-claude.sh` (`run`/`probe`/`smoke`/`print-env`/`list-accounts`) | ✅ |
| 9 | Aligned with the account switcher | default `.claude-faklocal` account shows in `fleet_accounts.py list`; `--account <tag>` resolves any worker | ✅ |
| 10 | Build + vet clean, suite green for the touched packages | `go build/vet ./...`; `go test ./internal/agent ./internal/gateway` | ✅ |

### Captured live evidence

`--smoke` (against a live `fak serve`):

```
POST /v1/messages → {"type":"message","role":"assistant",
  "content":[{"type":"tool_use","id":"call_0","name":"get_user_details",...}],
  "stop_reason":"tool_use","usage":{"input_tokens":0,"output_tokens":24}}
stream:true → event: message_start / content_block_start / content_block_delta /
              content_block_stop / message_delta / message_stop
```

Real model (qwen2.5:1.5b via ollama) through `/v1/messages`:

```
user "What is the weather in Paris? Use the tool." →
  tool_use get_weather {"city":"Paris"}   (adjudicated ALLOW, stop_reason tool_use)
```

Real Claude Code CLI (`--probe`), `dogfood-claude-probe.json`:

```json
{"type":"result","subtype":"success","is_error":false,"num_turns":1,
 "stop_reason":"end_turn","modelUsage":{"qwen2.5:1.5b":{"inputTokens":2048,"outputTokens":11}}}
```

Large local model (`lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M` via `llama-server`
on `http://127.0.0.1:8131/v1`) through the real Claude Code CLI:

```json
{"type":"result","subtype":"success","is_error":false,"num_turns":1,
 "result":"pong","duration_ms":218024,
 "modelUsage":{"lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M":{"inputTokens":25638,"outputTokens":24}}}
```

## Honest caveats

- **Model quality.** The default `qwen2.5:1.5b` proves the *wire and the kernel
  boundary*, not Claude-grade reasoning — its answers are weak. Point
  `FAK_DOGFOOD_MODEL` at a larger tool-capable model (e.g. `qwen2.5-coder:7b`,
  `llama3.1:8b`) for real work; the plumbing is identical.
- **Sampling params.** The gateway forwards the client's
  `max_tokens`/`temperature`/`top_p`/`stop` to the upstream model per request; an
  omitted field falls through to the planner default. Long Claude Code responses are no
  longer truncated at 1024 tokens (#62 closed).
- **Auth.** `--require-key-env` accepts the secret over **either** the
  `Authorization: Bearer <tok>` header (OpenAI/fak-native clients) **or** the
  `x-api-key: <tok>` header that Claude Code and the Anthropic SDKs send — both
  compared against the same secret in constant time, so an authenticated
  (non-loopback) gateway serves Claude Code directly. The dogfood still runs
  loopback with no key (the default), which needs none.
- **Token-streaming (real API) / synthesized (local model).** When fak fronts the
  **real Anthropic API** (`--provider anthropic`, the `fak guard -- claude` default),
  `/v1/messages` now relays a **true live token stream**: text and thinking deltas reach
  Claude Code the instant the model emits them (first token tracks the model, not the
  whole turn), so the prompt-cache hit's fast prefill is *felt* instead of buffered away —
  while every `tool_use` block is still held for kernel adjudication before the client
  sees it (denied calls dropped, repaired args rewritten, survivors renumbered
  contiguously). Witnessed live end-to-end (`claude-haiku-4-5` through fak): a warm cache
  read of 23,428 tokens cut time-to-first-byte 1,129 ms → 760 ms, and a longer turn's
  first token landed 323 ms before the turn finished. When fak fronts a **local
  OpenAI-compatible model** on this wire, the SSE is still synthesized from the finished,
  already-adjudicated turn (that upstream's true-streaming rung is the
  `/v1/chat/completions` proxy). Witness:
  `TestAnthropicMessagesPassthroughStreamsLiveAndAdjudicates`.
