---
title: "The Slack control plane: one durable wire out, one authorized door in"
description: "Design note + SOTA survey for epic #2259 (children #2261-#2268): durable delivery of every fak-native message to Slack (outbox, run-cards) and authorized control of the fleet from Slack (closed chatops verbs, approvals), built on the dgx-bridge/fakrpc transport invariants."
---

# The Slack control plane — one durable wire out, one authorized door in

Spine for epic [#2259](https://github.com/anthony-chaudhary/fak/issues/2259)
(children [#2261](https://github.com/anthony-chaudhary/fak/issues/2261)–[#2268](https://github.com/anthony-chaudhary/fak/issues/2268)).
Operator goal, 2026-07-02: *"Lay foundation for slack control … durable and flexible tooling
for fak native messages and updates to slack as well as control from slack."*

## What exists at HEAD

- **Out:** fourteen registered surfaces (`cmd/fak/slack.go`) posting through two parallel
  hand-rolled clients — `internal/scoreboard.Client` (no `thread_ts`, no `chat.update`, no
  429 handling) and `internal/chatrelay.HTTPSlack`. Delivery is fire-and-forget and
  fail-open: a failed post is *lost*; the watchdog family (#1425/#1855,
  `fak slack health`'s `OK|INCOMPLETE|AUTH_FAIL|STALE`) witnesses the silence after the
  fact but cannot recover a message. The session budget/stop observer (#743/#761 seam in
  `internal/session`) fires once — a missed fire is gone.
- **In:** `internal/chatrelay` — polls `conversations.history`, relays text to a served
  model, deliberately **not** control (no shell, no commands). No command router, no
  authorization, no idempotency ledger, no audit journal.
- **Contract:** `internal/fakrpc` (#930) — the proven text-bridge discipline: detach slow
  work, fresh worker, results out-of-band, nonce-spool idempotency, FAKRES framing.
- **Boundary:** `docs/gpu-server-private-boundary.md` — the *lab* control bridge is
  private. This plane is fak-native chatops: closed verbs binding to in-process fak
  surfaces, no shell verb ever, no lab identifiers in source. Leaf names respect the
  commit gate's private-only path patterns.
- **Plane split:** #2208 owns the intent levers (chatops verbs bind to them), #2209 owns
  rendering parity (run-cards reuse its shared-item fold). This epic owns the wire and
  the door.

## The dgx-bridge / fakrpc learnings, as design rules

1. **Detach slow work from the transport** — ack now, run detached, post completion
   out-of-band (fakrpc invariant 1).
2. **Results out-of-band** — run state and files are the source of truth; never parse
   channel history for results (invariant 3: a transcript floods with stale echoes).
3. **Nonce idempotency as spool state** — a restarted consumer never re-executes; keyed by
   nonce, not message replay (invariant 4; "nonce every bridge poll").
4. **Single-writer serialized drain** — concurrent readback loses the tail (the dgxbridge
   flakiness lesson).
5. **Leak fence outbound** — an internal hostname in a nightrun ledger note tripped
   PUBLIC_LEAK; every outgoing body is needle-scanned before send.
6. **Capacity refusals are structured** — REFUSE_AT_CAP posts its reason token; a chat
   command never steals a seat or a lease.

## SOTA survey (researched 2026-07-02; per-claim sources)

### Rate limits — the 2025 change, verified

Slack's 2025-05-29 change moved `conversations.history`/`conversations.replies` to Tier 1
(1 req/min, 15 objects/page) for **commercially distributed non-Marketplace apps**;
enforcement for existing installs was pushed to 2026-03-03. The 2025-06-03 clarification
**exempts internal customer-built apps**, which keep Tier 3 (~50 req/min, 1000/page) — so
fak's polling bridge stays legal, merely inferior. The same ToS update prohibits using
Slack data to train LLMs. Sources:
[changelog 2025-05-29](https://docs.slack.dev/changelog/2025/05/29/rate-limit-changes-for-non-marketplace-apps/),
[clarification 2025-06-03](https://docs.slack.dev/changelog/2025/06/03/rate-limits-clarity/),
[FAQ](https://api.slack.com/changelog/2025-05-terms-rate-limit-update-and-faq),
[rate-limits doc](https://docs.slack.dev/apis/web-api/rate-limits/).

### Socket Mode

Everything arrives over the WebSocket — Events API subscriptions, slash commands, and
Block Kit `block_actions` — with **zero public ingress** and no signing-secret machinery
(pre-authenticated socket; the `xapp-` app-level token can only open the socket, never
post). `apps.connections.open` → one-time WSS URL; `hello` carries
`approximate_connection_time`; connections refresh every few hours by design; cap 10
simultaneous connections; every envelope acked by `envelope_id` (~3 s for user-facing);
delivery is at-least-once with an undocumented retry schedule — dedupe on `event_id`.
Socket Mode apps cannot be Marketplace-listed: it is explicitly the internal-app
transport. Sources:
[Using Socket Mode](https://docs.slack.dev/apis/events-api/using-socket-mode/),
[no-SDK implementation guide](https://api.slack.com/apis/connections/socket-implement),
[apps.connections.open](https://api.slack.com/methods/apps.connections.open).
Known SDK failure modes to design around: unhandled `too_many_websockets`
([node-slack-sdk #1654](https://github.com/slackapi/node-slack-sdk/issues/1654)),
open/close 429 loops after days of uptime
([java-slack-sdk #1256](https://github.com/slackapi/java-slack-sdk/issues/1256)).

### Outbound durability

`chat.postMessage` is special-tier: ~1 msg/s per channel with burst tolerance (untouched
by the 2025 re-tiering); 429s carry `Retry-After` in seconds, scoped per-method-per-
workspace. **No server-side idempotency exists** (the
[2018 feature request](https://github.com/slackapi/slack-api-specs/issues/10) was never
implemented; method docs warn a retried `internal_error` may have half-succeeded) —
mitigation is client-side: an idempotency key in message `metadata` plus a history check
on ambiguous failure. `chat.update` is Tier 3; the shipping pattern for live status cards
is post-placeholder → debounce updates ≥500 ms → coalesce to latest state; ≤50
blocks/message, overflow to the thread. The delivery frame is the transactional outbox:
local spool, one relay drainer, at-least-once + idempotent consumer, DLQ. Sources:
[chat.postMessage](https://docs.slack.dev/reference/methods/chat.postMessage/),
[rate limits](https://docs.slack.dev/apis/web-api/rate-limits/),
[blocks reference](https://docs.slack.dev/reference/block-kit/blocks/),
[AWS transactional outbox](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html),
[event-driven.io delivery guarantees](https://event-driven.io/en/outbox_inbox_patterns_and_delivery_guarantees_explained/).
No published end-to-end Slack-outbox-CLI reference was found; the composition
(outbox + Retry-After + per-channel pacing + update-coalescing) is synthesized.

### How shipping products do agent control from Slack

- **Claude / Claude Code in Slack** (research preview 2025-12-08; Claude Tag beta
  2026-06-23): `@mention` + natural language, thread-per-task, status cards with buttons,
  per-user identity linking, channel-level restriction as an access layer, org audit view.
  [docs](https://code.claude.com/docs/en/slack),
  [Claude Tag](https://support.claude.com/en/articles/15594475-what-is-claude-tag).
- **Devin (Cognition):** `@Devin` + NL plus a small closed set of in-thread keyword verbs
  (`!ask`, `mute`, `sleep`, `archive`); configurable plan-approval gate; session-per-thread;
  per-user account linking. [docs](https://docs.devin.ai/integrations/slack).
- **GitHub Copilot coding agent:** `@GitHub` + NL with structured params
  (`repo=… branch=…`); review happens in the PR; actions run under the invoker's linked
  GitHub permissions. [docs](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/integrate-cloud-agent-with-slack).
- **OpenAI Codex:** `@Codex` + NL, replies link to the cloud task; enterprise admins can
  force link-only replies. [docs](https://developers.openai.com/codex/integrations/slack).
- **HumanLayer:** the cleanest button-gated tool-approval reference
  (`require_approval` → Approve/Deny-with-feedback Block Kit), but the SDK is deprecated
  (issues-only repo since mid-2025) — pattern, not dependency.
  [repo](https://github.com/humanlayer/humanlayer).
- **LangGraph HITL:** `interrupt()` + checkpointer + resume-by-thread-id; notable that
  LangChain **moved off Slack** to a dedicated Agent Inbox because approvals drown in
  channel noise — the argument for our single dedicated control channel + `pending` verb.
  [HITL docs](https://docs.langchain.com/oss/python/langchain/human-in-the-loop).

Cross-product patterns: mention+NL beats slash commands; thread = session universally;
approvals are Block Kit buttons where the vendor owns the loop; authorization is
identity-linking so actions carry the invoker's permissions; small closed keyword verbs
are the proven lightweight control channel. fak's variant is deliberately narrower: the
closed verb set IS the grammar (no NL execution path) — see security below.

### Go client landscape

Go stdlib has no WebSocket client. `golang.org/x/net/websocket` is retired in all but
name (its own docs redirect away). `gorilla/websocket` was archived 2022, revived 2023,
still soliciting maintainers. **`coder/websocket`** (ex-`nhooyr.io/websocket`, transferred
2024) is the 2026 consensus for new code — context-first, safe concurrent writes.
`slack-go/slack` (v0.26.x, active, no v1 promise, gorilla-based) buys typed structs at
the cost of a large dependency surface. Slack maintains an official no-SDK protocol
guide; a hand-rolled Socket Mode client is ~300 lines with `coder/websocket` as the sole
new dependency. Sources: [golang/go #33215](https://github.com/golang/go/issues/33215),
[coder/websocket](https://github.com/coder/websocket),
[slack-go](https://github.com/slack-go/slack),
[socket-implement guide](https://api.slack.com/apis/connections/socket-implement).
The transport decision is #2267's deliverable; polling stays the fallback behind the
existing `SlackClient` seam.

### Security posture

Gate on immutable Slack **user IDs** (`Uxxxx`), never display names; restrict the bot to
its one invited control channel; ignore `bot_id` messages (loop fence); dedupe on
`event_id`/`ts`. Indirect prompt injection through channel text is a demonstrated Slack
attack class (the Aug-2024 Slack-AI exfiltration disclosure —
[PromptArmor](https://www.promptarmor.com/resources/data-exfiltration-from-slack-ai-via-indirect-prompt-injection),
[Simon Willison](https://simonwillison.net/2024/Aug/20/data-exfiltration-from-slack-ai/));
the governing rule is the
[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) — deny one
leg per execution path. fak's fence: **channel text is never instructions** — a mention
either parses as a closed verb or is refused; free-form text reaches an agent only as
data inside an already-authorized task. Plus: `halt` kill-switch verb, per-user/per-verb
rate limits, append-only audit journal keyed to Slack `ts` (tamper-evident pointer back
to the Slack record). Signing-secret verification is HTTP-mode-only — irrelevant under
Socket Mode ([Slack security docs](https://docs.slack.dev/security)).

## The spine

| # | Issue | Leaf | Ships |
|---|---|---|---|
| C1 | [#2261](https://github.com/anthony-chaudhary/fak/issues/2261) | `internal/slackwire` | one transport: post/update/history/auth, 429/Retry-After, typed errors; scoreboard+chatrelay migrate |
| C2 | [#2262](https://github.com/anthony-chaudhary/fak/issues/2262) | `internal/slackoutbox` | durable enqueue→drain: JSONL spool, nonce idempotency, backoff + update-coalescing, dead-letter into health, leak fence |
| C3 | [#2263](https://github.com/anthony-chaudhary/fak/issues/2263) | run-cards | thread-per-run, `chat.update` in place, final edit carries the witness |
| C4 | [#2264](https://github.com/anthony-chaudhary/fak/issues/2264) | `internal/chatops` | inbound door v0: closed verbs, `FAK_CHATOPS_ADMINS` fail-closed, ts-nonce done-ledger, JSONL audit, structured refusals, `halt` |
| C5 | [#2265](https://github.com/anthony-chaudhary/fak/issues/2265) | detached exec | act verbs through guarded dispatch: ack now, witnessed completion, stall escalation |
| C6 | [#2266](https://github.com/anthony-chaudhary/fak/issues/2266) | approvals | reply-keyword HITL v0 (nonce, TTL, audit); Block Kit v1 blocked on C7 |
| C7 | [#2267](https://github.com/anthony-chaudhary/fak/issues/2267) | transport spike | Socket Mode vs polling; websocket-dep decision |
| C8 | [#2268](https://github.com/anthony-chaudhary/fak/issues/2268) | surface registration | slackSurfaces row, health rungs, docs front door, devindex/cli-reference |

Dependencies: C1 → {C2, C7}; C2 → {C3, C5}; C4 → {C5, C6}; C7 → C6-v1. C8 rides each ship.

## Honest fences

- **Not yet, by construction:** nothing in this note is shipped; every child carries its
  own witnessed DoD. The epic's end-to-end DoD (outage-surviving post; Slack-driven
  dispatch with witnessed completion; refused-then-approved gated verb) is the claim gate.
- **Never:** an arbitrary-shell verb; lab-box control in the public tree; NL command
  parsing; per-metric channel routing (#1003 decided); a message broker.
- **At-least-once, not exactly-once:** Slack offers no server-side idempotency; the
  outbox + metadata-key + history-check discipline is the honest contract, and the one
  crash window (between post and record) is documented rather than denied.
