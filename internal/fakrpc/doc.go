// Package fakrpc is the pure, transport-neutral core of disaggregated agent-RPC over a
// text-only control bridge (#930): the request envelope a caller spools to a resident
// worker, and the FAKRES nonce/sha frame the worker wraps its result in.
//
// The model runs on a remote GPU box; the control plane runs locally; the only thing
// that crosses the (lossy, text-only) bridge is a one-line pointer. Files do the work,
// the bridge is a doorbell. This package owns neither the bridge nor the worker — it is
// the data contract both sides agree on, so it imports nothing internal and never shells
// out. The resident worker (cmd/fakrpcd) and the pluggable transports build ON it.
//
// # The four transport invariants this contract is shaped by
//
//  1. Detach the slow work from the transport. A model/agent turn outlives any single
//     control-channel command; the worker runs it detached so the launching command
//     returns immediately and the decode runs free.
//  2. Deliver to a fresh worker, not a reused/aged one. An idle control session can
//     silently stop accepting work while still appearing live; address a fresh one per
//     burst (or a long-lived resident daemon).
//  3. Read results out-of-band of the transcript. A replay/transcript channel floods
//     with stale echoes under load; read results from a separate append-only channel or
//     the spool file — never the transcript. (DecodeFrame scans past leading echoes to
//     the framed body, and the Nonce defeats a stale tail.)
//  4. Result is a file first, a notification second. The worker writes out/<nonce>.result
//     and emits ONE "done" line. Idempotency is spool state (done/<nonce> ⇒ skip), keyed
//     by Request.Nonce — not message replay.
//
// # The FAKRES frame
//
// A result is wrapped so a truncated or corrupt transfer is REJECTED, not trusted:
//
//	<<<FAKRES nonce=<n> rc=<rc> sha=<sha256-hex of body> len=<bytes>>>>
//	<body: exactly len bytes>
//	<<<ENDFAKRES nonce=<n>>>
//
// The body is carried raw and read BY LENGTH (the frame is self-describing), so a body
// that itself contains newlines or even the sentinel strings round-trips intact; the
// len catches a split/short transfer and the sha catches a corrupt one. This is the same
// frame the GLM GPU witness runner already emits (tools/dgx_witness_run.sh), so this
// package is the canonical Go reference implementation of a wire the fleet already speaks.
//
// Tier: foundation (1) — see internal/architest. This package may import only packages
// whose tier is <= 1; an upward import fails the architest gate. See AGENTS.md and
// internal/architest for the layering contract.
package fakrpc
