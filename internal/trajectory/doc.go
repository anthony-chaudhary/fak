// Package trajectory is fak's TRAJECTORY DATA PLANE — the typed, exportable record
// of what an agent actually did, turn by turn, that application-layer optimizers
// build trajectory/memory/cache/planner analyses ON TOP of.
//
// Tier: composer (3) — see internal/architest. It folds the kernel's existing
// abi.Event stream (the abi.Emitter seam) into per-trace Turn rows and may stamp a
// reference query embedding via internal/simhash (tier 1). It imports only abi and
// simhash; it is off the hot path and opt-in.
//
// WHY A SEPARATE PLANE FROM THE JOURNAL. internal/journal already gives a durable,
// hash-chained, tamper-evident DECISION ledger — the regulated-audit surface. Its
// rows are decision-shaped: a verdict over an args/result DIGEST. That is exactly
// right for "prove what the kernel decided" and exactly wrong for "find the bad
// trajectories": a digest is opaque to similarity, the rows are not grouped into a
// turn, and the query text never lands in the ledger. trajectory is the
// complementary ANALYSIS surface: it keeps the human-meaningful query, the
// working-set/verdict/cost shape of each turn, and — optionally — a deterministic
// query embedding, so a downstream skill can cluster turns, flag near-duplicate or
// outlier queries, and propose memory gardening. fak ships the data and the seam;
// the semantic layer is built on top (see internal/trajhook and `fak traj`).
//
// HOW IT BINDS WITHOUT TOUCHING THE FROZEN ABI. A Recorder is an abi.Emitter. The
// kernel already fans every lifecycle transition to registered emitters, and
// abi.Event carries an OPEN Fields map plus the call's OPEN Meta — so the producer
// (the gateway/agent loop) stamps the query text and per-turn token/byte cost into
// Fields/Meta, and the Recorder folds them into a Turn. No ABI field is added; the
// recorder reads only what is already there, defaulting cleanly when a field is
// absent.
//
// ENABLEMENT mirrors the journal: off by default. trajectory.Enable() registers ONE
// Recorder against the frozen ABI and returns it (idempotent); Default is the
// process-global instance a front door reads. Unset and never Enabled => inert.
package trajectory
