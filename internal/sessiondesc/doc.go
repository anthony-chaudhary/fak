// Package sessiondesc is the session-descriptor join schema
// (fak.session.descriptor.v1, issue #2214, epic #2209): ONE record that binds
// the four identity spaces a fak session lives in but which today never join —
//
//  1. the gateway DRIVE state (trace id, run state, lineage, rev — the
//     /v1/fak/sessions snapshot and the JSON --log sink's trace_id),
//  2. the cross-host leaseref descriptor (refs/fak/locks/session-<id>,
//     internal/leaseref: id, host, pcb_state),
//  3. the harness identity (which agent — claude/codex/opencode/... — and
//     which account/rotation identity, internal/harnessprofile +
//     internal/fleetaccounts),
//  4. the transcript/census namespace (reserved: bound by the #2213
//     cross-agent census, absent until then).
//
// Every rollup surface (`fak fleet`, tools/fleet_top.py, `fak rollup`,
// the #2215 sidecar pane, the #1203 fleet fold) re-invents an ad-hoc join over
// these spaces; this package is the one schema they can all read instead.
//
// DESIGN RULES (load-bearing):
//
//   - DATA-ONLY. Pure types + a pure Fold over caller-supplied, already-parsed
//     inputs. No I/O, no git, no HTTP, no clock — the callers own their
//     sources; this package owns only the join. Stdlib-only, imports nothing
//     internal (the same mirror-type pattern as gatewayusageledger, so tier-1
//     layering holds and no consumer drags the gateway onto its import path).
//
//   - CLOSED ABSENCE VOCABULARY. Every key space a descriptor could not bind
//     states WHY with one of three tokens (ABSENT_NOT_OBSERVED /
//     ABSENT_SOURCE_UNAVAILABLE / ABSENT_NO_BINDING) — absence is a typed
//     answer, never a nil a reader guesses about.
//
//   - EXACT JOIN ONLY. Two rows merge iff their session ids are byte-equal.
//     No fuzzy matching, no mtime heuristics — two different sessions must
//     never fold into one descriptor (the collision test pins this).
//
//   - NO SELF-REPORT. A descriptor carries identity and observation pointers,
//     never progress claims — the same no-`claimed`-field discipline as the
//     DOS status digest. Progress belongs to verified ledgers, not to an
//     identity record.
package sessiondesc
