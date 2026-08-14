---
title: "Adopt fak in an existing repo: a 10-minute migration checklist"
description: "Put fak in front of an agent project you already run in 10 minutes: repoint one base URL, load a starter policy, wire the DOS trust gate, watch a verdict fire."
slug: adopt-fak-in-your-repo
keywords:
  - adopt fak
  - migrate to fak
  - put fak in front of an existing agent
  - repoint base URL
  - fak manage
  - dos init hooks
  - 10-minute migration checklist
date: 2026-07-03
---

# Adopt fak in an existing repo: a 10-minute checklist

You already have a working agent — Claude Code, an OpenAI-SDK loop, Cursor, an MCP
client. This is the checklist for putting fak in front of it in about ten minutes,
without rewriting the agent. You repoint one base URL, load a starter capability
floor, and wire the DOS trust gate into your runtime so a false "done" is refused
from git evidence, not just noticed.

Dimension G — Integration recipes of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
The mechanical reference behind each step is the
[migration guide](../fak/migration-guide.md) and the
[adopter playbook](../integrations/adopter-playbook.md); this page is the ordered
10-minute version.

## What does not change

Read this first, because it is most of the reassurance:

- **Your model and your keys stay put.** fak fronts the engine you already run
  (Ollama, vLLM, llama-server, or a cloud API); it is not a replacement token
  engine. On the Claude path `fak manage` uses your logged-in Pro/Max
  **subscription** by default — no API key needed.
- **Your prompts, tool definitions, and agent loop stay put.** fak returns only the
  admitted (or repaired) tool calls; your existing loop still executes them.
- **Your IDE and harness stay put.** `fak manage` injects the base URL into the
  **child process only**, so your shell, your `settings.json`, and any other agent
  in another terminal are untouched.

The single change is where one client points.

## The 10-minute checklist

### 1. Get the binary (about 2 minutes)

One static Go binary — two `golang.org/x` modules, no Python, no CUDA toolchain:

```
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Or download a release binary from the
[getting-started guide](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md#1-get-the-binary).
Confirm it is on your PATH:

```
fak --version
```

### 2. Put fak in front, one command (about 2 minutes)

For a Claude Code agent, the whole gateway is one verb. No second terminal, no
config-file edit:

```
fak manage claude
```

`fak manage` starts an in-process gateway on a private loopback port, loads a secure
capability floor embedded in the binary, injects `ANTHROPIC_BASE_URL` into the child
only, and proxies to the real Anthropic API on your subscription. Every tool call
your agent proposes crosses the floor first.

Wrap a different agent by naming it after `--` and switching the provider:

```
fak manage --provider openai -- codex       # an OpenAI-compatible coding agent
fak manage --provider openai -- opencode
```

If your client reads an environment variable instead of launching under a wrapper,
run a standalone gateway and repoint the one base URL — this is the migration in
its purest form:

```
fak serve --addr 127.0.0.1:8080 --provider openai \
  --base-url http://127.0.0.1:11434/v1 \     # your existing model server
  --model qwen2.5-coder:7b --policy policy.json
```

```
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"   # OpenAI-wire clients
# or, for an Anthropic SDK, point at the origin (the SDK appends /v1 itself):
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
```

### 3. Choose a starter policy (about 2 minutes)

With no policy the kernel default-denies every tool, so pick a reviewable starting
floor and adapt it. Print the built-in guard floor, or dump the default for the
serve path:

```
fak manage --dump-policy > policy.json       # the embedded guard floor
# or
fak policy --dump > policy.json              # the default serve floor
```

Edit `policy.json` to allow the tools your framework registers, then validate it
before it gates a run:

```
fak policy --check policy.json
```

Shipped starting points to copy from live in
[examples/](https://github.com/anthony-chaudhary/fak/tree/main/examples):
`dev-agent-policy.json`, `customer-support-readonly-policy.json`,
`research-agent-policy.json`. Load your floor with `--policy policy.json` on either
`fak manage` or `fak serve`. Honest scope: the floor bounds which tools run by tool
name; it does not yet filter the arguments of an allow-listed tool, so keep
irreversible operations off the allow-list rather than allow-listing broadly. The
full discussion is in
[POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).

### 4. Verify a verdict appears (about 2 minutes)

You do not need a live model to prove the boundary. Check one call against your
policy with no server at all:

```
fak preflight --policy policy.json --tool refund_payment --args "{}"
# verdict=DENY reason=DEFAULT_DENY
fak preflight --policy policy.json --tool search_kb --args "{}"
# verdict=ALLOW
```

The refusal is a decision, not an error. When you run a real turn under
`fak manage`, the same verdicts show up in the exit summary it prints when the agent
exits:

```
fak manage: 131 kernel decision(s) — 121 allowed, 5 denied, 2 repaired, 0 quarantined, 3 deferred
  blocked: POLICY_BLOCK     x4
  blocked: SELF_MODIFY      x1
```

That line is the verified verdict: a tool call your agent proposed crossed the
capability floor and the kernel recorded its decision.

### 5. Wire the DOS trust gate into your runtime (about 2 minutes)

The last step closes the trust loop. DOS refuses a false "done" from git evidence
rather than the agent's word. Run this from the agent repo you actually work in so
DOS can detect the runtime marker (`.claude/`, `.cursor/`, `.codex/`, ...):

```
dos init --hooks auto .
```

Honest note: run from a throwaway directory with no runtime marker, `--hooks auto`
refuses with a way forward — it lists the hosts it found and tells you to name one
explicitly (`dos init --hooks <host>`). That refusal-with-a-next-step is the DOS
house style; it is not a failure.

Now make the gate fire. After a commit, ask DOS to audit the claim against the diff
it can see:

```
dos commit-audit --workspace . <sha>
# OK — diff-witnessed        (the claimed change is present in the commit)
# or a non-zero verdict when the "done" is not backed by the diff
```

A commit that claims work it did not do audits non-`OK`, and the exit code is
non-zero — the gate firing. That is the whole point: the verdict in step 4 governs
what a tool call may do, and the DOS gate here governs whether a reported "done" is
believed.

## Current limits (honest)

- **KV poison-eviction is a no-op on a subscription or proxy seat, by design.** The
  model lives upstream, so there is no local KV prefix to drop. A quarantined tool
  result is still paged out before the model reads it; the in-kernel evictor is the
  local-model (`--gguf`) path.
- **Anthropic-wire streaming is buffered.** On the `/v1/messages` path the SSE
  stream is synthesized from a fully-adjudicated turn, so time-to-first-token
  equals full-generation time there. Live streaming is shipped on the
  OpenAI-compatible content wire; the Anthropic-wire rung is next.
- **The durable audit journal is opt-in.** Set `FAK_AUDIT_JOURNAL=/path/audit.jsonl`
  for the hash-chained, tamper-evident decision trail; the in-memory exit summary is
  always on.
- **`--hooks auto` needs a real agent repo**, per step 5 — it cannot be exercised
  from an empty scratch directory.

## Go deeper

- [Migrating to fak](../fak/migration-guide.md) — the per-framework base-URL
  reference (OpenAI SDK, LangChain, AutoGen, llama.cpp) behind step 2.
- [Adopter playbook](../integrations/adopter-playbook.md) — the production
  bare-serve path, the manual MCP server, and the CI embed.
- [Claude Code integration guide](../integrations/claude.md) — the full Claude path:
  architecture, denial table, cloud providers, observability, the subscription
  proof.
- [Concept card](concept-card.md) — the one-page pitch and the 60-second proof
  command.
- [POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) ·
  [README](../../README.md) — the capability floor schema and the front door.
