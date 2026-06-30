# Dogfood: the real Claude Code CLI on your own kernel-fronted model

**Use fak as a *product*.** One command spins up a local model, puts the **fak kernel**
in front of it as a **native Anthropic `/v1/messages` server**, and points the **real
Claude Code CLI** at it. You drive the product the way an adopter would — live turns, your
own server, your own harness — and every tool call Claude proposes is **adjudicated by the
kernel before Claude ever sees it** (denied calls dropped, repaired args rewritten).

```
 ┌─────────────┐   POST /v1/messages   ┌────────────────────────┐   /v1/chat/...   ┌──────────────┐
 │ Claude Code │ ───────────────────▶  │ fak serve (the kernel)  │ ───────────────▶ │  local model │
 │ (the harness)│ ◀──── SSE stream ───  │ adjudicates every tool  │ ◀────────────── │ ollama / shim│
 └─────────────┘                        └────────────────────────┘                  └──────────────┘
        ▲ ANTHROPIC_BASE_URL=http://127.0.0.1:8080            kernel drops / repairs proposed tool calls
        │ CLAUDE_CONFIG_DIR = an isolated .claude-faklocal account (your real ~/.claude is untouched)
```

This is the most concrete "fak is a real thing" demonstration for a Claude-using adopter:
the *actual* Claude Code CLI, talking to *your* kernel, over the *real* Anthropic wire — no
ollama required, CPU-friendly on the default model.

> **This directory wraps the shipped scripts; it does not reimplement them.** The launchers
> already live under the repo-level `scripts/` directory for macOS/Linux and Windows, and
> the full design + caveats are in [`../../DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md).
> The thin wrappers here
> ([`run.sh`](run.sh) / [`run.ps1`](run.ps1)) invoke those scripts and print "what to look
> for" hints, in the adoption shape every other example carries.

## Prerequisites

- The **real Claude Code CLI** installed and on `PATH` (`claude --version`). This is the
  one thing you bring; the script handles the rest.
- **Go** (to build `fak`) and one local-model backend, picked automatically:
  - **macOS/Linux:** [ollama](https://ollama.com) if installed, else the in-tree Python
    transformers shim (`python3`). With neither, the script prints an actionable hint.
  - **Windows:** the in-tree transformers shim — **no ollama needed**, CPU-friendly by
    default (model `HuggingFaceTB/SmolLM2-135M-Instruct`, the knob that keeps a CPU turn at
    seconds).

You do **not** need an Anthropic API key or any cloud spend for the default local path — the
kernel fronts a model on your own box.

## Run it

```bash
# macOS / Linux
./examples/dogfood-claude/run.sh --smoke           # curl the wire end-to-end (no model), then exit
./examples/dogfood-claude/run.sh --probe "say pong" # ONE headless live Claude Code turn, then exit
./examples/dogfood-claude/run.sh                    # interactive Claude Code on the local model
```

```powershell
# Windows (PowerShell)
.\examples\dogfood-claude\run.ps1 --smoke            # curl the wire end-to-end (no model), then exit
.\examples\dogfood-claude\run.ps1 --probe "say pong" # ONE headless live Claude Code turn, then exit
.\examples\dogfood-claude\run.ps1                    # interactive Claude Code on the local model
```

Each wrapper is a few lines: it prints what to look for, then `exec`s the shipped launcher
with your flags. Every flag the launcher takes (`--smoke`, `--probe`, `--print-env`,
`--list-accounts`, `--install`) passes straight through. The launcher tears down the kernel
(and the shim/ollama if it started one) on exit.
Expected runtime: the no-model `--smoke` path completes in seconds; `--probe` and
interactive runs depend on the selected model and can take tens of seconds.

**It cannot damage your normal `claude`.** Every wiring env var (`ANTHROPIC_BASE_URL`,
`CLAUDE_CONFIG_DIR`, the model tiers) is exported only into the child `claude` process the
script spawns — it never touches the parent shell or `~/.claude/settings.json`.
`CLAUDE_CONFIG_DIR` points at an isolated `.claude-faklocal` account, never your default
`~/.claude`.

## What the operator sees per turn

The kernel adjudicates **every** tool call Claude proposes, against a capability floor. The
default floor is [`../dogfood-claude-policy.json`](../dogfood-claude-policy.json) — it
**allows** the standard Claude Code tool set (`Bash`, `Read`, `Edit`, `Write`, `Glob`,
`Grep`, `Task`, …) so a session is usable, while still refusing dangerous calls **by
argument value, before the shell sees them**:

| Ask Claude to do this in the session | What the kernel does | Verdict |
|---|---|---|
| `ls`, `cat`, `grep`, `git commit` | runs — everyday dev work | ✅ ALLOW |
| `rm -rf …`, `sudo …`, `git push …` | refused on the **argument**, before dispatch | ⛔ `POLICY_BLOCK` |
| `curl … \| sh`, a fork bomb, `dd … of=/dev/sd…` | refused — RCE pipe / fork bomb / disk wipe | ⛔ `POLICY_BLOCK` |
| `Edit`/`Write` into `.git/`, `.ssh/`, `internal/kernel/`, `VERSION` | refused — can't rewrite the kernel or secrets | ⛔ `SELF_MODIFY` |
| any tool the floor never named | refused — fail-closed | ⛔ `DEFAULT_DENY` |

The deny is on the **argument**, not just the tool name: `Bash` is allowed, but
`Bash{command:"rm -rf /"}` is refused *before the shell sees it*. Try it live — ask Claude to
run `rm -rf /tmp/x`, `sudo ...`, or `git push` and watch the kernel refuse while `ls`/`cat`
run. Check any call **without** launching a session:

```bash
fak preflight --tool Bash --args '{"command":"rm -rf /tmp/x"}' \
  --policy examples/dogfood-claude-policy.json    # => verdict=DENY reason=POLICY_BLOCK
```

Where to look while a session runs:

- **The fak serve log** — `/tmp/fak-serve.log` (macOS/Linux). The kernel logs each
  `/v1/messages` turn and the adjudication verdict on every proposed tool call: ALLOW,
  the `POLICY_BLOCK`/`DEFAULT_DENY`/`SELF_MODIFY` refusals above, a `TRANSFORM` when a
  `redact_fields` value is rewritten before dispatch, and a quarantine event when a flagged
  tool result is held out of the context window.
- **Claude Code's own view** — a denied call simply never reaches Claude as an executable
  tool; a repaired call arrives with its arguments already rewritten. The harness sees a
  clean, adjudicated stream.

A captured live turn is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## Variations

The wrappers pass through to the launcher, so every documented backend works here. From
[`../../DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md):

| Goal | How | Notes |
|---|---|---|
| Front the **real Claude API** through fak | `FAK_DOGFOOD_BACKEND=anthropic ./run.sh --probe "say pong"` | your own key + real model tiers flow through; `cache_control` survives byte-for-byte |
| **fak's own in-kernel decode** (no shim, no proxy) | `./run.sh --kernel --smoke` (or `-Kernel` on Windows) | `fak serve --gguf` answers on the Anthropic wire; asserts `/healthz planner=inkernel` |
| A **larger local model** | `FAK_DOGFOOD_MODEL=qwen2.5-coder:7b ./run.sh` | a 7B+ tool model is what makes an *interactive* session usable |
| A **large OpenAI-compatible** server (llama-server, LM Studio, vLLM) | `fak-qwen36-claude --probe "say pong"` | the `qwen36-local` preset; see `docs/qwen36-claude-dogfood-playbook.md` |

## Witnessed

The real Claude Code CLI has completed live turns against the local kernel on **both
platforms**:

- **macOS** — `scripts/dogfood-claude.sh --probe` →
  `experiments/agent-live/dogfood-claude-probe.json` (a Qwen3.6-27B turn returned
  `result:"pong"`).
- **Windows** — `scripts/dogfood-claude.ps1 --probe` →
  `experiments/agent-live/dogfood-claude-probe-win.json`. The CLI (v2.1.181) completed a
  turn against the local kernel-fronted model in ~36 s on a 16-core Windows host — no
  ollama, the in-tree shim auto-fronted the model on the box's GPU (fp16 CUDA); a GPU-less
  host runs the identical path on fp32 CPU with the pre-raised timeouts.

See [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) for the captured turn.

## Honest scope — what this proves and what it does not

- **UNVERIFIED in this README's CI:** the *full* interactive live run is **not** exercised
  here — it needs the Claude Code CLI installed plus a live model serve, which is heavy and
  long (a full turn prefills Claude Code's ~5–6K-token system + tool prompt). The wire, the
  kernel boundary, and the adjudication are the witnessed parts (the `--smoke` and `--probe`
  paths above, and the captured turns in `EXAMPLE-OUTPUT.md`). Run the live interactive
  session on your own box.
- **Model quality is not Claude quality.** The CPU-friendly default models (SmolLM2-135M on
  Windows, qwen2.5:1.5b via ollama) prove the *wire and the kernel boundary*, not
  Claude-grade reasoning. Point `FAK_DOGFOOD_MODEL` at a 7B+ tool model, or use the
  `anthropic` backend, for real work — the plumbing is identical.
- **In-kernel CPU forward is too slow for a *full* interactive turn.** The `--kernel`
  (`--gguf`) backend proves fak's own pure-Go decode on the `--smoke` path in seconds, but a
  full Claude Code turn on a CPU is GPU territory — see the honest caveat in
  [`../../DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md).

## Files

| file | what it is |
|---|---|
| [`run.sh`](run.sh) | thin wrapper around the repo-level macOS/Linux launcher, prints "what to look for" hints, passes flags through |
| [`run.ps1`](run.ps1) | thin wrapper around the repo-level Windows launcher, same hints + passthrough |
| [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) | a captured live Claude Code turn through the kernel |
| [`../dogfood-claude-policy.json`](../dogfood-claude-policy.json) | the capability floor the kernel enforces |

Related: [`../../DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md) (full design + caveats and
the repo-level launchers); [`adjudication-demo/`](../adjudication-demo/README.md) puts the
same kernel + floor in front of a Python OpenAI client instead of the Claude Code CLI.
