// Package slackoutbox is the durable Slack outbox (#2262, epic #2259): every fak-native
// message survives crashes, 429s, and token drift by being ENQUEUED as a local JSONL
// append first and POSTED by a single serialized drainer second — the transactional-outbox
// pattern sized for a one-box fleet.
//
// Before this leaf every outbound surface was fire-and-forget and fail-open: a feeder
// that could not post (missing secret, revoked token, 429, network cut) exited 0 and the
// message was *lost*, not delayed; the watchdog family (#1425/#1855) witnessed the
// silence after the fact but could recover nothing. Here, Enqueue returns once the row is
// durable on disk, and delivery is the drainer's problem:
//
//   - per-channel FIFO through internal/slackwire (the ONE transport), pacing ≤1 msg/s
//     per channel — chat.postMessage's special tier, verified in the design note
//     (docs/notes/SLACK-CONTROL-FOUNDATION-2026-07-02.md);
//   - nonce idempotency as SPOOL STATE (the fakrpc #930 discipline): a nonce in a
//     terminal state is never re-sent, and the nonce rides in message metadata
//     (slackwire.PostMessageIdem) so the one at-least-once window — a crash between
//     post and record — is closed by probing recent History for the nonce before any
//     re-send. Slack has no server-side idempotency; this is the honest client-side
//     contract, documented rather than denied;
//   - one drainer at a time (internal/flock on drain.lock — the dgx-bridge readback
//     lesson: concurrent drainers lose the tail);
//   - update coalescing: queued chat.update rows for the same card collapse to the
//     newest state before send (superseded rows are recorded, not silently dropped);
//   - bounded retries: transient failures back off to the NEXT drain pass (the wire
//     already honors Retry-After within a call); after MaxAttempts a row goes DEAD and
//     surfaces in `fak slack health` — never silently dropped;
//   - a leak fence before every send (hooks.ScanOutboundText): a PUBLIC_LEAK needle or
//     SECRET_SHAPE hit refuses the row with the finding as its structured reason,
//     terminally — a refused body must be re-authored, never retried into posting.
//
// Tier: foundation (1) — imports slackwire(1), hooks(1), flock(1) and stdlib; off the
// hot path. The `fak slack outbox` verbs and the health rung live in cmd/fak.
package slackoutbox
