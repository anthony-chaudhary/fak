---
title: "fak guarded sessions from Slack"
description: "Two shipped, public Slack surfaces: watch every fak manage session as a durable run-card in a Slack channel, and chat with a fak serve-hosted model from a channel. Your own workspace bot, no lab identifiers, nothing baked into source."
---

# fak guarded sessions from Slack

> **Audience.** Anyone who runs `fak manage` or `fak serve` and has (or can create) a
> Slack workspace bot — by the end you'll be able to watch every guarded session as a
> run-card in a channel and chat with a kernel-hosted model from Slack.

Two things fak can do with a Slack channel today, both public and both invokable with
your own workspace bot:

1. **Watch your guarded sessions.** Every `fak manage` run posts a durable run-card to a
   Slack channel — who started, which agent and provider, the pid, and where the audit
   journal lives — so a long session is visible from your phone, not just the terminal it
   started in.
2. **Chat with a kernel-hosted model.** `fak chatrelay` bridges one Slack channel to a
   `fak serve` endpoint, so a model you serve (a local GGUF on CPU, GLM-5.2, whatever you
   loaded) answers in-thread like any other chatbot.

Neither bakes a token, a channel id, or a host into source. You point them at your own
Slack app at runtime, so the public tree stays clear of the secret-needle scan. This is
the *public* Slack story; the *private lab control* bridge that drives remote GPU boxes is
a separate, non-public transport — see [lab-dev-loop.md](lab-dev-loop.md).

## One-time setup: a workspace bot

Both surfaces need one Slack bot token (`xoxb-…`) with `chat:write`, plus
`conversations.history` for the chat relay. Put it — and the channel ids — in a
gitignored `.env.slack.local` at your repo root (or export the env vars):

```
FAK_GUARD_SESSIONS_TOKEN=xoxb-your-bot-token
FAK_GUARD_SESSIONS_CHANNEL=C0XXXXSESSIONS
FAK_CHATRELAY_TOKEN=xoxb-your-bot-token
FAK_CHATRELAY_CHANNEL=C0XXXXCHAT
```

`.env.slack.local` is already gitignored. Nothing here is ever committed; the resolver
walks up from the working directory to find it. If you only set one token, both surfaces
fall back to the shared scoreboard bot token when their own is unset — one workspace bot
can serve everything.

Invite the bot to each channel (`/invite @your-bot`). Channel ids are the `C0…` strings
from a channel's *View channel details*, not the `#name`.

## Watch every guard session from Slack

Nothing to turn on: when a `FAK_GUARD_SESSIONS_TOKEN` is resolvable, `fak manage` enqueues
a session run-card at startup and drains it to Slack in the background.

```bash
fak manage claude
```

The startup report tells you whether the card was queued:

```
fak manage: slack thread — queued root in C0XXXXSESSIONS (nonce=…)
```

The card carries the `session_thread_id`, `trace_id`, the agent name, the provider, the
pid, the UTC start time, and the audit-journal reference — enough to find the run again
and to correlate it with the guard's own audit trail. If no token is resolvable, the line
reads `slack thread — unavailable` and the guard runs exactly as before; the Slack post is
additive, never a gate on your session.

Delivery is durable, not fire-and-forget. The card goes through the
[transactional outbox](../../internal/slackoutbox/doc.go): a local JSONL spool, one
serialized drainer, nonce idempotency (a restart never double-posts), and `Retry-After`
backoff for Slack's rate limits. A card enqueued while Slack is unreachable is posted when
it comes back, rather than lost. Every outbound body is needle-scanned before send, so an
internal path or a token that slipped into a field is refused at the fence rather than
leaked into the channel.

Point sessions at a specific channel per-run without touching `.env.slack.local`:

```bash
FAK_GUARD_SESSIONS_CHANNEL=C0XXXXOPS fak manage claude
```

## Chat with a kernel-hosted model from Slack

`fak chatrelay` polls one Slack channel, forwards each new human message to a served
OpenAI-compatible `/v1/chat/completions` endpoint, and posts the reply in-thread. It is a
generic chatbot front end — no shell, no commands, no lab identifiers — so it stays on the
public side of the boundary.

The end-to-end "a served model, usable from Slack" path:

```bash
# 1. serve a model on the in-kernel forward (CPU here; the fast pure-forward path)
FAK_Q4K=1 fak serve --gguf <model-shard1.gguf> --addr 127.0.0.1:8080

# 2. bridge a Slack channel to it (token + channel from .env.slack.local, or flags)
fak chatrelay --endpoint http://127.0.0.1:8080 --channel C0XXXXCHAT --model glm-5.2

# or run the bridge itself under the guard's capability floor:
fak manage fak chatrelay --endpoint http://127.0.0.1:8080 --channel C0XXXXCHAT
```

Check the wiring before a live run — `--dry-run` prints the resolved config (with the
token redacted to its last four chars) and exits without connecting:

```bash
fak chatrelay --dry-run
```

Useful flags (full set in the [CLI reference](../cli-reference.md)):

- `--mention <@U07BOT>` — only answer messages that address the bot, and strip the mention
  from the prompt. Empty answers every human message in the channel.
- `--system "<prompt>"` — a system prompt prepended to every turn.
- `--prime=false` — also answer the latest backlog on start (default skips it and only
  answers messages posted after launch).
- `--once` — one poll then exit, for a smoke test; `--interval 3s` — poll cadence.
- `--api-key-env VAR` — send a bearer token to a `--require-key-env` serve.

The relay skips its own posts and any `bot_id` message, so it never talks to itself.

## Where the model runs

The chat relay is a thin bridge; the model runs wherever you pointed `--endpoint`. Serve a
local GGUF for offline or privacy-bound chat, or point at any OpenAI-compatible endpoint.
The honest fence is the same as the rest of fak: a small local model is a quality ramp,
not a frontier coder — use `--gguf` for private work and a stronger endpoint for the best
reasoning.

## The boundary, stated plainly

- **Public (this doc):** your own workspace bot, your own channels, your own served model.
  Tokens and channel ids live only in a gitignored env file. The guard session-thread is
  an *outbound* status post; the chat relay reads channel text only as chatbot input and
  never as a command — there is no shell verb and no command router on this path.
- **Private (not here):** the lab control bridge that drives remote GPU boxes speaks a lab
  protocol carrying a host, a channel, and a token — none of which enter this repo. That
  loop lives in `fak-private`; see [lab-dev-loop.md](lab-dev-loop.md) for its shape.

## Related

- [lab-dev-loop.md](lab-dev-loop.md) — develop fak on a remote box, driven from the private
  Slack bridge.
- [../cli-reference.md](../cli-reference.md) — `fak chatrelay`, `fak scoreboard post`,
  `fak blockers post`, `fak slack health`.
- [../../internal/slackoutbox/doc.go](../../internal/slackoutbox/doc.go) — the durable
  delivery outbox behind the guard session run-card.
