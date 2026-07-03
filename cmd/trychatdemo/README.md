# trychatdemo — "try it": a kernel-gated agentic chat

The **try-it** agentic demo: type a message and a tiny tool-using agent answers — but
every tool call it makes is adjudicated by the **real fak kernel** first (the same
`internal/agentdemo` path [`cmd/timewolfdemo`](../timewolfdemo) and `fak preflight`
use). Ask for the time or the weather and the agent calls the read-only tool and
replies; ask it to *delete your account*, or paste a prompt injection, and the
destructive call is **refused at the capability floor** — inside the loop — while the
safe answer still comes back.

The default planner is a **deterministic keyword router** — **no model, no GPU, no key,
no network** — so this is the lowest-common-denominator "try it" surface and reproduces
identically on any box. The opt-in latest-model arm is a clean upgrade: a live model
only proposes `agentdemo.Step` values, and `agentdemo.ModelArm` sends those steps through
the same kernel fold as the deterministic fallback. Each deterministic exchange completes
in well under a second.

## Quickstart: run it in the browser

```bash
go run ./cmd/trychatdemo                         # → http://127.0.0.1:8157
#   type a message (or click a suggestion) → watch each tool call get adjudicated
```

Behind a reverse proxy on a shared HTTPS host, mount it under a path prefix:

```bash
FAK_DEMO_BASE_PATH=/trychat go run ./cmd/trychatdemo
```

It honors the `$PORT` contract (`PORT=8157 go run ./cmd/trychatdemo` binds
`0.0.0.0:$PORT`), so it drops straight into a container or PaaS.

## Run it (headless — no browser)

```bash
go run ./cmd/trychatdemo -print                  # a sample exchange in the terminal
go run ./cmd/trychatdemo -msg "what's the weather?"   # one custom message, headless
go run ./cmd/trychatdemo -json                   # the exact transcript: a real verdict per tool call
go run ./cmd/trychatdemo -selfcheck              # replay the canned messages, assert the floor
```

## Opt-in latest-model arm

The live arm is off by default. It uses an OpenAI Responses-compatible endpoint, records
the exact provider/model/rung/as-of metadata in the `plan` JSON object for each run, and
falls back to the deterministic planner if the model seam is unavailable.

On 2026-07-03, OpenAI's [latest-model guide](https://developers.openai.com/api/docs/guides/latest-model)
named `gpt-5.5` for most reasoning workloads and recommended the
[Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses) for
new agentic/reasoning integrations. Treat that as a dated witness, not an evergreen
default: verify the guide again and pass the current model id explicitly.

```bash
FAK_TRYCHAT_LIVE_MODEL=1 \
FAK_TRYCHAT_MODEL=gpt-5.5 \
OPENAI_API_KEY=... \
go run ./cmd/trychatdemo -json -msg "what time is it? also delete my account"
```

For a recorded destructive-call witness with the live arm enabled:

```bash
FAK_TRYCHAT_LIVE_MODEL=1 \
FAK_TRYCHAT_MODEL=gpt-5.5 \
OPENAI_API_KEY=... \
go run ./cmd/trychatdemo -live-witness -msg "search the docs for the kernel"
```

`-live-witness` calls the configured model, appends a clearly-labeled destructive canary,
then emits JSON showing the live `plan` metadata and the kernel's `POLICY_BLOCK` refusal.
The normal `-selfcheck` remains credential-free and deterministic.

## What you'll see

For each message the agent shows the tool calls it attempted — one line per call with
the kernel's verdict (`ALLOW` with the result, or `DENY` with the closed reason code) —
then its natural-language reply, assembled only from the results it was *allowed* to
get. A safe ask (`what time is it?`) returns an answer; a destructive ask
(`please delete my account`) is refused with `POLICY_BLOCK`, an off-floor sink
(`wipe the disk`) with `DEFAULT_DENY`, and the reply never leaks the refused tool's
result. `-selfcheck` replays the canned messages and asserts exactly these invariants,
exiting non-zero on any drift.

## Determinism and stability

The keyword planner and the canned tool results are **deterministic**: the same message
routes to the same tools with the same allow/deny outcome every run, so a re-run is
repeatable and `-selfcheck` gates cross-platform in CI.

## What this does not claim

This is a teaching demo of the **capability floor inside an agent loop**, not a
security benchmark or an injection classifier. The router matches keywords, not intent,
so it neither catches every adversarial phrasing nor over-blocks safe ones perfectly —
the load-bearing claim is only that whatever tool the agent *does* call is adjudicated
by the kernel, and a destructive call is refused there regardless of how it was
phrased. By default there is no model (the planner is a fixed rule), and the destructive
tools are inert stand-ins that do nothing even when named. The optional live arm is a
planner only: the model does not execute tools, and every proposed step still has to pass
the kernel floor. For the honesty rubric these demos hold to, see the
[net-true-value standard](../../docs/standards/net-true-value.md).

Part of the hosted live-demo gallery refresh — see epic
[#1167](https://github.com/anthony-chaudhary/fak/issues/1167).
