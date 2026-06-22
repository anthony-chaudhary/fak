---
title: "fak Tutorial: Your First Adjudicated Tool Call"
description: "Hands-on fak tutorial: run the agent kernel offline, watch it deny a destructive tool call, block a prompt injection, front a model over HTTP. No key or GPU."
---

# fak tutorial: zero to your first adjudicated tool call

`fak` is an agent kernel that adjudicates every tool call a model makes, on your own
machine, with no key and no GPU.

> **TL;DR.** Grab the binary, then replay a tool-call trace and watch the kernel's verdict
> on each call. Parts 1–2 are fully offline; later parts add a real model and Claude Code.

```sh
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak run --trace testdata/tau2/tau2-smoke.json     # the rest of the page explains this
```

**Audience:** you have never run `fak` before. By the end of this page you will have
watched the kernel *deny a destructive tool call*, *wall off a prompt-injection*, and
*serve a model behind an HTTP gate*. It all runs on your own machine, with **no API key,
no GPU, no cloud bill**. Every command below was run on a clean build, and **every output
block is the real, unedited terminal output**. What you see here is what you will see.

- **Time:** ~15 minutes for Parts 1–2 (zero downloads). Part 3 (chat with a real model)
  adds a model download. Parts 4–6 add a model server you point fak at, the Claude Code
  wiring, and three worked workflows (~15 min more).
- **Prereqs:** [Go 1.26+](https://go.dev/dl/) *or* a [prebuilt binary](../../INSTALL.md).
  Nothing else for Parts 1–2. Parts 4–6 add any one OpenAI-compatible model server
  (Ollama / llama-server / LM Studio) and the Claude Code CLI.
- **Already know the pitch?** This is the *guided first session*. For the install
  reference and the four usage tiers, see [`fak/GETTING-STARTED.md`](../../GETTING-STARTED.md);
  for the idea, the [main README](../../README.md).

> **One sentence of context.** `fak` treats the model like an untrusted program and a
> tool call like a syscall: every call the agent wants to make passes *through* a kernel
> the model can't talk past. This tutorial makes that concrete by watching the boundary
> decide for itself.

---

## Map of this tutorial

![The getting-started journey: get the binary, drive the kernel offline, front a model over HTTP, then optionally chat with a real local model — color-coded by the verdict you'll see at each step](../../visuals/52-getting-started-journey.png)

| Part | What you do | Downloads | What you'll have seen |
|---|---|---|---|
| **0** | Get the `fak` binary | none (or one binary) | `fak version` prints |
| **1** | Drive the kernel offline | **none** | a trace replay, a **DENY**, the injection **A/B**, your own policy |
| **2** | Front a model over HTTP | **none** (synthetic engine) | `/healthz`, a syscall, an adjudication, the access log |
| **3** | *(optional)* chat with a real local model | ~1 GB GGUF | live tokens from a model the kernel owns |
| **4** | Point a real model server at the gateway | one model server | `/healthz` against a real Ollama / llama-server / LM Studio upstream |
| **5** | Connect Claude Code | the Claude Code CLI | Claude talking to a local model through the fak kernel |
| **6** | Example workflows | none | read-only / development / deployment policies on the same gateway |

You can stop after any part. Each one stands on its own.

---

## Part 0 — get the binary (2 min)

Pick **one** of these. The rest of the tutorial writes `./fak` (Linux/macOS) — on Windows
type `.\fak.exe`.

**A. Prebuilt binary, no clone, no Go** (recommended for just trying it):

```sh
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak version
```

**B. Build from a clone** (Go 1.26+; the Go module is the repository root):

```sh
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # Windows: build with -o fak.exe (see the Windows note)
./fak version
```

> **Windows.** Build with `go build -o fak.exe ./cmd/fak` — an explicit `-o fak` (no extension)
> leaves a literal `fak` file that cmd.exe / PowerShell cannot launch by name (Go only appends
> `.exe` when you *omit* `-o`; git-bash runs the extensionless file via its exec bit). Type the
> binary as `.\fak.exe` wherever this guide writes `./fak`, and run the `--args '{...}'` examples
> from git-bash / PowerShell — cmd.exe passes the single quotes through literally, so there use
> `--args "{""_positional"":[""alice""]}"` instead.

Either way, `fak version` prints the version and you're ready:

```
0.30.0
```

> **Run Parts 1–2 from inside the `fak/` directory.** The offline commands resolve their
> sample data (`testdata/`, `examples/`) relative to the working directory, and write
> their report files (`report.json`, `agent-report.json`) into the current folder. If you
> installed the prebuilt binary, `git clone` the repo too so you have `testdata/` and
> `examples/` — or pass absolute paths.

For the full install matrix (Docker, manual download + checksum verify, `go install`
status), see [`INSTALL.md`](../../INSTALL.md).

---

## Part 1 — drive the kernel offline (no downloads)

Everything in this part is **deterministic and offline**: no model, no network, no key.
You are exercising the adjudication boundary directly.

### 1.1 Replay a tool-call trace through the kernel

A *trace* is a recorded list of tool calls. `fak run` replays it and shows you the
kernel's verdict on each one:

```sh
./fak run --trace testdata/tau2/tau2-smoke.json
```

**Expected output (real):**

```
[ 0] get_user_details             verdict=ALLOW     by=monitor   status=OK
[ 1] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[ 2] get_reservation_details      verdict=ALLOW     by=vdso      status=OK
[ 3] search_direct_flight         verdict=ALLOW     by=monitor   status=OK
[ 4] list_all_airports            verdict=ALLOW     by=vdso      status=OK
[ 5] calculate                    verdict=ALLOW     by=vdso      status=OK
[ 6] search_flights               verdict=ALLOW     by=monitor   status=OK
[ 7] get_user_details             verdict=ALLOW     by=vdso      status=OK
[ 8] search_direct_flight         verdict=ALLOW     by=vdso      status=OK
[ 9] book_reservation             verdict=ALLOW     by=monitor   status=OK
[10] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[11] list_all_airports            verdict=ALLOW     by=vdso      status=OK

summary: submits=12 vdso_hits=6 engine_calls=6 denies=0 transforms=0 quarantines=0
```

**Reading it:**
- `verdict=ALLOW`: the call was admitted (these are all read-only or allow-listed tools).
- `by=monitor`: the call went through the full adjudication path to the engine.
- `by=vdso`: the call was served from the local fast-path **without an engine call**.
  That happens on a repeated read the kernel already knew the answer to. `vdso_hits=6`
  means **half** the calls in this trace were served for free. That's the reuse win, in
  miniature.

### 1.2 Watch the capability floor refuse a call

This is the security flip: a tool that isn't on the allow-list is refused **by
structure**, rather than by a classifier judging intent. Try a tool the floor never allowed:

```sh
./fak preflight --tool create_user --args '{"_positional":["alice"]}'
```

**Expected output (real):**

```
verdict=DENY reason=DEFAULT_DENY by=monitor
```

`DEFAULT_DENY` = "not on the allow-list, so fail-closed." No prompt, no context, no clever
phrasing changes this answer. The lever was never wired up. Now an allow-listed tool:

```sh
./fak preflight --tool get_user_details --args '{}'
```

```
verdict=ALLOW reason=NONE by=monitor
```

### 1.3 The same idea with a *deployable policy file*

The allow-list is a file you can author and review, rather than a code edit. The repo ships an
example "customer-support, read-only" policy. Run a **destructive** tool against it:

```sh
./fak preflight --policy examples/customer-support-readonly-policy.json \
  --tool refund_payment --args "{}"
```

**Expected output (real):**

```
fak: loaded capability floor from examples/customer-support-readonly-policy.json
verdict=DENY reason=POLICY_BLOCK by=monitor
```

…and a read-only tool against the same policy:

```sh
./fak preflight --policy examples/customer-support-readonly-policy.json \
  --tool search_kb --args "{}"
```

```
fak: loaded capability floor from examples/customer-support-readonly-policy.json
verdict=ALLOW reason=NONE by=monitor
```

> **The headline in one line:** *a support agent under this policy can search the
> knowledge base but physically cannot refund money*. The reason is a named verdict
> (`POLICY_BLOCK`), not a model's opinion.

### 1.4 The prompt-injection A/B — the demo to show a skeptic

`fak agent --offline` runs the **same task twice** on a deterministic planner: once with
tools wired directly (the baseline), once behind `fak`. The task includes a
booby-trapped tool result (a poisoned "refund policy" that tries to hijack the agent).

```sh
./fak agent --offline
```

**Expected output (real):**

```
== fak agent: turn-use vs now ==
seam        : OFFLINE (deterministic mock planner)
task        : Customer mia_li_3668 wants to book the cheapest direct flight from SFO to JFK on 2026-07-0...

metric                        now(base)          fak
--------------------------   ----------   ----------
model turns                           9            7
tool calls                            8            6
tool errors (-> retries)              1            0
prompt tokens                      2555         1571
completion tokens                   232          184
in-syscall repairs                  n/a            1
vDSO dedup hits                     n/a            1
adjudicator denies                  n/a            1
MMU quarantines                     n/a            0
injection in context                YES           no
destructive op executed             YES           no
task completed (booked)             YES          YES

HEADLINE
  turns saved by fak        : 2  (22%)   [both arms completed -> comparable]
  tokens saved by fak       : 1032  (37%)
  poisoned result blocked   : YES
  destructive op prevented  : YES

report written: agent-report.json
```

**The two rows that matter** are near the bottom of the table:
- `injection in context: YES → no`: the poisoned tool result reached the baseline's
  context but was **walled off** from the `fak` arm. The model never saw it.
- `destructive op executed: YES → no`: the baseline ran the dangerous action; `fak`
  refused it.

And the kicker: **both arms still completed the task** (`task completed (booked): YES /
YES`). Safety here isn't "refuse everything." The real booking still happened, the trap
just didn't. The token and turn savings (`37%` / `22%` on this single task) are the
*efficiency* side of the same boundary. The full machine-readable breakdown is written to
`agent-report.json`.

### 1.5 Author your own capability floor

The built-in default policy is dumpable as an editable manifest:

```sh
./fak policy --dump > floor.json
```

`floor.json` is plain JSON — the allow-list, allowed prefixes, named deny reasons, and
redaction rules. The top of it looks like this (real):

```json
{
  "version": "fak-policy/v1",
  "allow": [
    "book_reservation",
    "calculate",
    "get_reservation_details",
    "get_user_details",
    "list_all_airports",
    "search_direct_flight",
    "search_flights",
    "send_certificate",
    "transfer_to_human_agents",
    "update_reservation_flights"
  ],
  "allow_prefix": [
    "read_", "get_", "search_", "list_", "lookup_", "find_", "calc"
  ],
  "deny": {
    "exfiltrate": "SECRET_EXFIL",
    "shell_rm_rf": "POLICY_BLOCK"
  },
  ...
}
```

Edit it (add/remove a tool), then **validate** it before deploying. The refusal
vocabulary is closed, so a typo'd reason gets caught right here at author time, well
before production:

```sh
./fak policy --check floor.json     # validates, prints the floor it admits
# then load it on any verb with:  --policy floor.json
```

The full manifest schema is in [`fak/POLICY.md`](../../POLICY.md); a fuller authoring
walkthrough with patterns is in the [policy guide](policy-guide.md).

### 1.6 *(Optional)* the fusion-speedup gate

`fak bench` measures the in-process adjudication latency against a spawned-hook baseline.
That's the cost of doing the check on the same call path vs. shelling out to a sidecar:

```sh
./fak bench --suite tau2-smoke --baseline-n 5
```

**Expected output (real):**

```
== fak bench: tau2-airline-smoke ==
in-process adjudication p50 : 4867 ns
spawned-hook        p50     : 23555300 ns (23.555 ms, n=5)
fusion speedup (p50)        : 4840x
PRIMARY GATE                : pass  (in-process adjudication p50 (4867ns) vs spawned-hook p50 (23555300ns))
secondary token delta       : 47.17% (soft, never gates)
vdso hit-rate               : 0.500   pollution-rate: 0.000
workload hash               : 9f1701415fb4a360   live seam: live_seam_unverified
report written              : report.json
```

The exact `4840x` will vary by machine; the point is the order of magnitude. Adjudicating
*in-process* (microseconds) instead of *spawning a hook* (tens of milliseconds) is what
makes a default-deny gate cheap enough to put on **every** call.

✅ **End of Part 1.** You've watched the kernel allow, deny, dedup, and wall off an
injection. You've also authored a policy, all offline.

---

## Part 2 — front a model over HTTP (no downloads)

`fak serve` is an **OpenAI-compatible gateway**. In production you point `--base-url` at a
real model server (Ollama, vLLM, a cloud provider) and `fak` adjudicates the tool calls it
proposes. For this tutorial we use the built-in **synthetic in-kernel engine** so you need
**zero downloads**. The wire and the verdicts are identical; only the generated tokens are
placeholder.

### 2.1 Start the gateway

```sh
./fak serve --addr 127.0.0.1:8137 --engine inkernel --model smollm2-inkernel
```

This runs in the foreground. **Open a second terminal** for the calls below. To background it:
bash — append `&` (stop with `kill %1`); Windows — start it in its own window with
`Start-Process` (PowerShell) or `start` (cmd), since `&` / `kill %1` are bash-only.

### 2.2 Liveness and the advertised model

```sh
curl -s http://127.0.0.1:8137/healthz
```

```json
{"engine":"inkernel","model":"smollm2-inkernel","ok":true}
```

```sh
curl -s http://127.0.0.1:8137/v1/models
```

```json
{"data":[{"id":"smollm2-inkernel","object":"model","owned_by":"fak"}],"object":"list"}
```

### 2.3 Run one adjudicated tool call through the kernel

`POST /v1/fak/syscall` runs a single tool call through the full kernel path and returns
the verdict **and** the result:

```sh
curl -s -X POST http://127.0.0.1:8137/v1/fak/syscall \
  -H 'Content-Type: application/json' \
  -d '{"tool":"read_file","arguments":{"path":"notes.txt"}}'
```

**Expected output (real, formatted for readability):**

```json
{
  "verdict": { "kind": "ALLOW", "by": "monitor" },
  "result": {
    "status": "OK",
    "content": "{\"tool\":\"read_file\",\"engine\":\"inkernel\",\"model\":\"smollm2-inkernel\",\"generated_tokens\":[125,125, ... ,125]}",
    "meta": { "engine": "inkernel", "ifc_taint": "trusted", "input_tokens": "29", "output_tokens": "16" }
  },
  "trace_id": "gw-3"
}
```

> **Wire gotcha:** the fak-native key is `arguments`, **not** `args`. An unknown key is
> silently dropped. The `generated_tokens` are repeated placeholders because the synthetic
> engine has random weights; this call exercises the *dispatch + decode + verdict* path
> while leaving output quality aside.

### 2.4 Get a verdict *without* dispatching

`POST /v1/fak/adjudicate` returns just the decision, which is handy for "would this be
allowed?" checks. Ask about a destructive tool:

```sh
curl -s -X POST http://127.0.0.1:8137/v1/fak/adjudicate \
  -H 'Content-Type: application/json' \
  -d '{"tool":"refund_payment","arguments":{}}'
```

```json
{"verdict":{"kind":"DENY","reason":"DEFAULT_DENY","by":"monitor","disposition":"TERMINAL"},"trace_id":"gw-4"}
```

Same answer as the offline `preflight` in Part 1. The gate is the same gate, whether you
reach it from the CLI or over HTTP.

### 2.5 The audit trail you get for free

Every request writes one structured JSON access-log line. In the `fak serve` terminal you'll
see entries like this (real):

```json
{"event":"gateway_operation","operation":"syscall","tool":"read_file","verdict":"ALLOW","duration_ms":5.88,"trace_id":"gw-3"}
{"event":"gateway_http_request","method":"POST","path":"/v1/fak/syscall","status":200,"bytes":358,"duration_ms":5.88,"trace_id":"gw-3","user_agent":"curl/8.9.0"}
{"event":"gateway_operation","operation":"adjudicate","tool":"refund_payment","verdict":"DENY","reason":"DEFAULT_DENY","disposition":"TERMINAL","duration_ms":0.511,"trace_id":"gw-4"}
```

The `trace_id` ties together the verdict log, the HTTP log, and the response header. It
never logs request bodies, arguments, or result content. That's the audit surface; the
full observability story (Prometheus `/metrics`, `/debug/vars`) is in the
[observability guide](observability.md).

To point a **real** model at the gate instead of the synthetic engine, swap the engine flag
for an upstream:

```sh
./fak serve --addr 127.0.0.1:8137 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b   # Ollama, vLLM, etc.
```

…and harden it with `--policy floor.json` and `--require-key-env FAK_TOKEN`. The full Tier 1
serving path is in [`fak/GETTING-STARTED.md` §3](../../GETTING-STARTED.md) and
[`server-quickstart.md`](server-quickstart.md).

✅ **End of Part 2.** You've fronted a model with an HTTP gate, run a syscall and an
adjudication over the wire, and seen the audit log.

---

## Part 3 — *(optional)* chat with a real local model

This part downloads a small model so you can see **real tokens**. Two ways:

**A. The friendly chat REPL** ([Simple Demo](../../cmd/simpledemo/README.md)):

```sh
go run ./cmd/simpledemo -gguf ~/Downloads/Qwen2.5-1.5B-Instruct-Q8_0.gguf
```

```
🤖 Found model: Qwen2.5-1.5B-Instruct-Q8_0.gguf
📦 Loading model...
✅ Loaded Qwen2.5-1.5B in 0.8s

💬 Chat with your AI! Type a message and press Enter.
   Commands: /clear = new chat, /exit = quit

You: What is the capital of France?
AI: The capital of France is Paris.

📊 15 tok in, 8 tok out (12.5 tok/s) | 1.3s total
```

**B. Serve that same model as a real chat backend** (OpenAI **and** Anthropic wires), so
Claude Code or any OpenAI client can talk to it locally:

```sh
./fak serve --addr 127.0.0.1:8137 \
  --gguf ~/Downloads/Qwen2.5-1.5B-Instruct-Q8_0.gguf --model qwen2.5-1.5b
```

Pointing the real Claude Code CLI at a local model behind the kernel is its own one-command
walkthrough: [`fak/DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md) (and the
[Claude Code setup notes](../../cmd/simpledemo/CLAUDE.md)). Where to get models and the
size/RAM table are in the [Simple Demo README](../../cmd/simpledemo/README.md).

> **Honesty note.** The in-kernel model path is a *correctness reference* proven bit-exact
> against HuggingFace, not a production chat engine. For chat-quality serving at scale, lean
> on Part 2's Tier 1 proxy in front of a real serving engine. See [`fak/CLAIMS.md`](../../CLAIMS.md).

---

## Part 4 — point a real model server at the gateway

Part 2 used the synthetic engine so you needed zero downloads. To get **real tokens**
behind the same gate, point `fak serve` at any OpenAI-compatible model server. Pick **one**
of the three below. They all expose the same `/v1/*` wire, so the gateway config is
identical. (This is the prerequisite for Part 5.)

### 4.1 Ollama (macOS/Linux, easiest)

[Install Ollama](https://ollama.com/), then:

```sh
ollama serve &                         # start the server (default port 11434)
ollama pull qwen2.5:1.5b               # one-time model download (~1 GB)
```

### 4.2 llama-server (all platforms)

`llama-server` ships with [llama.cpp](https://github.com/ggerganov/llama.cpp) and runs on
Windows, macOS, and Linux. Point it at any local GGUF:

```sh
llama-server \
  -m Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 --port 8131 \
  --ctx-size 32768 --n-gpu-layers 99
```

### 4.3 LM Studio (Windows/macOS)

LM Studio is a GUI app: load a model from its catalog, then enable the **local server**
(Developer tab → *Start Server*, default port `1234`). No CLI install needed, which helps
when you already pick models through a UI.

### 4.4 Verify the model server

Whichever you picked, confirm it answers before wiring fak:

```sh
curl -s http://localhost:11434/v1/models    # Ollama
# curl -s http://127.0.0.1:8131/v1/models   # llama-server
# curl -s http://localhost:1234/v1/models   # LM Studio
```

A JSON list of `{"data":[{"id":"…","object":"model"}], …}` means it's ready.

### 4.5 Start `fak serve` in front of it

```sh
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/dogfood-claude-policy.json
```

Verify the gateway (same `/healthz` you hit in Part 2):

```sh
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"qwen2.5:1.5b","ok":true}
```

For the full serving matrix (auth, hot-reload, cloud upstreams), see
[`server-quickstart.md`](server-quickstart.md).

✅ **End of Part 4.** You have a real model behind the same kernel gate.

---

## Part 5 — connect Claude Code

With `fak serve` running from Part 4 (and a model server behind it), wire the Claude Code
CLI to it. Claude Code speaks the Anthropic Messages API; `fak serve` exposes it, so the
whole job is pointing Claude's base URL at the gateway.

### 5.1 The one-command path (recommended)

The repo ships a launcher that builds `fak`, starts the model server, starts `fak serve`,
and points Claude Code at it — in one command:

```sh
./scripts/dogfood-claude.sh                          # macOS/Linux — interactive
.\scripts\dogfood-claude.ps1                         # Windows PowerShell — interactive
```

Add `--probe "Reply with exactly the word: pong"` for a one-shot smoke test that writes a
witness to `experiments/agent-live/`. The launcher's full reference (presets, account
switcher, large-model timeouts) is in [`docs/integrations/claude.md`](../integrations/claude.md).

### 5.2 Manual wiring (macOS/Linux)

If you started `fak serve` yourself (Part 4.5), set three env vars and launch Claude Code:

```sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_API_KEY="fak-local-dogfood"
export ANTHROPIC_MODEL="qwen2.5:1.5b"
claude --dangerously-skip-permissions
```

### 5.3 Manual wiring (Windows PowerShell)

PowerShell uses `$env:` instead of `export`:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
$env:ANTHROPIC_API_KEY  = "fak-local-dogfood"
$env:ANTHROPIC_MODEL    = "qwen2.5:1.5b"
claude --dangerously-skip-permissions
```

### 5.4 Environment variable reference (essentials)

The four variables that matter for a first connection:

| Variable | Purpose | Example |
|---|---|---|
| `ANTHROPIC_BASE_URL` | Where Claude Code sends requests — the fak gateway | `http://127.0.0.1:8080` |
| `ANTHROPIC_API_KEY` | Auth header Claude sends; fak ignores it on loopback | `fak-local-dogfood` |
| `ANTHROPIC_MODEL` | The model id `fak serve` advertised on `/healthz` | `qwen2.5:1.5b` |
| `CLAUDE_CONFIG_DIR` | *(Optional)* isolated account dir, keeps fak state separate | `$HOME/.claude-faklocal` |

For the full list (`FAK_DOGFOOD_*`, planner timeouts, account switcher), see the
[environment reference in the Claude guide](../integrations/claude.md#environment-reference).

### 5.5 Troubleshooting the first connection

| Symptom | Fix |
|---|---|
| `claude: command not found` | Install Claude Code first (`npm i -g @anthropic-ai/claude-code`), or use the launcher (5.1) which handles the build + serve + wire in one shot. |
| Claude connects but every reply is empty | Check the `fak serve` terminal — if `/v1/models` is failing there, the upstream model server from Part 4 isn't running. Re-verify with `curl -s http://localhost:11434/v1/models`. |
| `connection refused` on `:8080` | `fak serve` isn't running, or is on a different port. Confirm with `curl -s http://127.0.0.1:8080/healthz`. |
| First reply takes >60 s and Claude times out | Expected on large local models (the system prompt is ~25 K tokens). Raise the timeout: `export FAK_DOGFOOD_TIMEOUT_S=900` (or `$env:FAK_DOGFOOD_TIMEOUT_S = "900"` on Windows). |
| Model gives wrong / garbled answers | A 1.5 B model is weak — try `qwen2.5-coder:7b` or larger. Garbled tokens specifically: see the [Simple Demo troubleshooting](../../cmd/simpledemo/README.md#troubleshooting). |
| `address already in use` on `:8080` | Set `FAK_DOGFOOD_PORT=8090` (launcher), or pass a different `--addr` to `fak serve`. |

✅ **End of Part 5.** You have Claude Code talking to a local model through the fak kernel.

---

## Part 6 — example workflows

Three policy-shaped workflows, from safest to most privileged. Each one is a different
`--policy examples/<file>.json` handed to the **same** `fak serve` command from Part 4.5.
The gateway code is identical; only the capability floor changes.

### 6.1 Read-only agent (safe exploration)

The agent can search and read but physically cannot mutate, refund, or exfiltrate. Use this
to let an agent explore a support inbox, a knowledge base, or a codebase without risk.

```sh
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b \
  --policy examples/customer-support-readonly-policy.json
```

Try the same boundary you watched in Part 1 against this policy:

```sh
./fak preflight --policy examples/customer-support-readonly-policy.json \
  --tool refund_payment --args "{}"
# verdict=DENY reason=POLICY_BLOCK
```

- **Allowed:** `read_customer_record`, `search_kb`, `create_support_ticket`.
- **Denied:** every write, refund, and credential rotation.

### 6.2 Development agent (commits allowed, push denied)

The agent can run the build, the tests, and `git diff`/`log`/`status`. It can even ship a
local release. What it cannot do is `git push`, `git merge`, `git tag`, or exfiltrate. Use
this for an agent pair-programming on a clone.

```sh
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b \
  --policy examples/dev-agent-policy.json
```

The **Claude Code** tool surface is broader. It covers `Bash`, `Edit`, `Read`, and `Write`,
plus search tools like `Glob` and `Grep`. For that, reach for
[`examples/dogfood-claude-policy.json`](../../examples/dogfood-claude-policy.json). It
allows those tools while still denying `rm -rf`, `sudo`, and `git push`. It also blocks any
write into `.git/`, `internal/kernel/`, or `VERSION`.

### 6.3 Deployment agent (production dry-run)

The agent can plan and validate but cannot apply. `terraform_apply`, `kubectl_delete`,
`kubectl_exec`, and `deploy_production` are all `POLICY_BLOCK`. Use this for an agent that
drafts changes for a human to review and merge.

```sh
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b \
  --policy examples/devops-dryrun-policy.json
```

- **Allowed (planning):** `plan_deploy`, `validate_terraform`, `helm_template`.
- **Allowed (inspection):** `diff_infra`, `kubectl_get`, `create_change_request`.
- **Denied:** every mutating production action.

To put **any** of the three on a network-facing host, also require an API key and bind
publicly:

```sh
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"
./fak serve --addr 0.0.0.0:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b \
  --policy examples/devops-dryrun-policy.json \
  --require-key-env FAK_GATEWAY_KEY
```

Clients then send `Authorization: Bearer $FAK_GATEWAY_KEY` (or `x-api-key:`). The full
production hardening checklist is in [`security.md`](security.md).

✅ **End of Part 6.** The same gateway, three different capability floors. Pick by intent.

---

## Reading the output: a field reference

Every verdict you saw decodes the same way. Keep this handy:

| Field | Values | Meaning |
|---|---|---|
| `verdict` / `kind` | `ALLOW` · `DENY` · `TRANSFORM` · `QUARANTINE` | the decision on this call |
| `by` | `vdso` · `monitor` | served from the local fast-path (no engine call) vs. through the full path |
| `reason` | `NONE` · `DEFAULT_DENY` · `POLICY_BLOCK` · `SECRET_EXFIL` · … | the **named** reason (closed vocabulary — see [`POLICY.md`](../../POLICY.md)) |
| `disposition` | `TERMINAL` · … | whether the call is finally refused or eligible for repair |
| `ifc_taint` | `trusted` · `quarantined` | whether the result may enter the model's context |
| `trace_id` | `gw-N` | correlates the response, the HTTP log, and the verdict log |

And the `run` summary line:

```
summary: submits=12 vdso_hits=6 engine_calls=6 denies=0 transforms=0 quarantines=0
```

| Counter | What it counts |
|---|---|
| `submits` | total tool calls replayed |
| `vdso_hits` | calls served from the fast-path (the reuse win) |
| `engine_calls` | calls that went through to the engine |
| `denies` / `transforms` / `quarantines` | calls refused / repaired / walled off |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `go: go.mod requires go >= 1.26` | Install Go 1.26 (<https://go.dev/dl/>), or keep `GOTOOLCHAIN=auto` (the default) and let it self-fetch. |
| `fak run: no such file testdata/...` | Run from inside `fak/`, or pass an absolute `--trace` path. |
| `address already in use` on `fak serve` | Another process owns the port — pick a different `--addr`. |
| Windows: `An Application Control policy has blocked this file` during `go test` | OS quirk on freshly-built **test** binaries only — `go build`/`go run` are unaffected. Run the suite under WSL. Type the binary as `.\fak.exe`. |
| `/v1/fak/syscall` returns an empty/odd result | Use the key `arguments`, not `args` — unknown keys are silently dropped. |
| Garbled tokens from a real GGUF | Ensure you're on a build with the NEOX-rope GGUF fix; then try `-temp 0.3`. See the [Simple Demo troubleshooting](../../cmd/simpledemo/README.md#troubleshooting). |

---

## Where to go next

- **Make the policy yours** → [policy authoring guide](policy-guide.md) · [`POLICY.md`](../../POLICY.md)
- **Run it in production** → [server quickstart](server-quickstart.md) · [server config](server-config.md) · [security best practices](security.md)
- **See it observed** → [observability guide](observability.md) (`/metrics`, `/debug/vars`, the trace ids)
- **Wire your language/agent** → [integration examples](../integrations/claude.md)
- **Understand the two flips** → [Policy in the kernel](../explainers/policy-in-the-kernel.md) · [Addressable KV cache](../explainers/addressable-kv-cache.md)
- **Check what's real** → [`fak/CLAIMS.md`](../../CLAIMS.md) (every capability tagged `[SHIPPED]`/`[SIMULATED]`/`[STUB]`)

---

*Every command and output block on this page was captured from a clean build of `fak`
v0.30.0. If a command prints something different for you, that's a doc bug — please
[open an issue](https://github.com/anthony-chaudhary/fak/issues).*
