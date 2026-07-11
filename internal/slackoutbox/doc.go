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
//   - ephemeral reaping (reap.go): posted messages in the dgx-bridge channels (the
//     FAK_SLACK_EPHEMERAL_CHANNELS allowlist, or a row that opted in via DeleteAfterS) are
//     chat.delete'd once they go IDLE past their TTL (default 30m), measured from last
//     activity so a live card is not culled mid-run. One reap pass runs at the tail of every
//     Drain (and on demand via `fak slack outbox reap`); a message already gone is recorded
//     reaped idempotently, a transient delete failure retries next drain. This clears the
//     channel noise the outbox otherwise accretes without touching a channel nobody opted in;
//   - a leak fence before every send (hooks.ScanOutboundText): a PUBLIC_LEAK needle or
//     SECRET_SHAPE hit refuses the row with the finding as its structured reason,
//     terminally — a refused body must be re-authored, never retried into posting;
//   - assertive compaction (compact.go): the append-only spool and state files are folded
//     down by default — the drainer runs one seal-quiesce-rewrite pass when it is due, and
//     `fak slack outbox compact` forces one. A row still owed a delivery, and a DEAD row an
//     operator may retry, is NEVER dropped ("never silently dropped" still holds for
//     everything owed or actionable); only a SETTLED row past its retention window is
//     forgotten — a posted row after it is old enough that no deferred producer will probe
//     its ts again, a superseded/refused row shortly after — and the per-drain heartbeat
//     storm collapses to one line. This is the deliberate retention exception to the
//     "terminal states are terminal forever" reading above.
//
// Tier: foundation (1) — imports slackwire(1), hooks(1), flock(1) and stdlib; off the
// hot path. The `fak slack outbox` verbs and the health rung live in cmd/fak.
package slackoutbox
