// Package knobcensus is the KNOB CENSUS — issue #2210, epic #2208
// (meaningful-control plane). It generalizes #2199's context-knob counter to the
// whole user-facing surface and adds a verdict from a closed two-token
// vocabulary, so the two ratchets read ONE inventory.
//
// # The boundary it makes executable
//
// The operator goal (2026-07-01): "Users [get] more control for things they
// should meaningfully control … the management of context and cache is super
// automatic by default." The census draws that boundary per knob:
//
//   - INTENT — the knob encodes intent the system cannot infer
//     (which/what/how-much/consent). Disposition: promote (epic #2208): it must
//     have a full, consistent route on every surface.
//   - HOUSEKEEPING — the knob is derivable from telemetry (residency, warmth,
//     TTLs, breakpoints, eviction, hygiene). Disposition: automate (#2198
//     doctrine): the default path must not require it; it survives as an
//     operator/debug override.
//
// A knob misfiled on either side is a defect the census surfaces.
//
// # What it walks (first checkable step)
//
//   - #2199's context-knob inventory, folded in verbatim as HOUSEKEEPING (context
//     and cache management is the automatic-by-default domain). This CONSUMES the
//     R1 counter rather than re-deriving it — there is no second context count.
//   - cmd/fak flag registrations + FAK_* env lookups whose name gates user
//     behavior (guard/session/account/model/fleet), classified INTENT or
//     HOUSEKEEPING by a deliberately narrow, name-only heuristic (the same
//     philosophy as #2199's isContextFlagName). A name that gates nothing
//     user-visible — plumbing, output format, paths, ids — is excluded.
//
// dos.toml keys that gate user behavior are named in the issue as part of the
// eventual surface; this first data-only pass covers flags + env, the step the
// issue calls out. Broadening the walk and landing the CI ratchet (HOUSEKEEPING
// count non-increasing) are deferred to after two stable runs, per the issue.
//
// # Deterministic, data-only for now
//
// Scan is deterministic — the same tree marshals to byte-identical bytes, the
// census witness. It is surfaced by `fak index knobs` (table + --json), a query,
// not a gate. The enforcing ratchet lands in a follow-on once the emit is stable.
package knobcensus
