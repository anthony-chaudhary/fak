// Package ctxknobs is the MANUAL-OVERLAY COUNTER — R1 of the zero-knob
// automatic-context epic (#2199, epic #2198; spine
// docs/notes/CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md).
//
// # The doctrine it witnesses
//
// "Nobody manages the context window." A user- or agent-facing instruction,
// habit, flag, or skill whose ONLY purpose is context management is a defect
// (doctrine L5). The knobs may survive as operator/debug surfaces; the DEFAULT
// path must never require one. This package enumerates the surviving overlays,
// classifies each, and ratchets the DEFECT count so a new one cannot land
// silently.
//
// # What it walks
//
//   - cmd/fak flag registrations and env lookups whose name touches
//     context/cache/session budgets → classified operator-debug (an operator
//     surface off the default path — fine per L5).
//   - .claude/skills/*/SKILL.md whose PURPOSE is managing the context window or
//     memory store → classified user-required (the defect: a skill an agent or
//     human must run to keep context healthy). The memory-compact skill is the
//     canonical one.
//
// Harness-prompt warnings ("don't read large files, it will overflow your
// context", the A1 overlay) live in the EXTERNAL agent harness, not in this
// repo, so they are not enumerable here; they are R6's scalp, tracked
// separately. This counter is honest about what it can and cannot see.
//
// # Why flags are operator-debug and skills are the defect
//
// A static walk cannot tell whether the DEFAULT path forces a flag — that is a
// runtime property (R5 auto-envelope territory). So the counter takes the
// conservative, doctrine-faithful line: a flag/env is an operator surface
// (fine), and the enforced ratchet is over the clearly user-facing overlays —
// the skills/instructions whose reason for existing is context management.
//
// # The ratchet (architest/pythongate style)
//
// The set of user-required knob keys at HEAD is frozen in baseline.go. The
// count may only go DOWN. A new user-required overlay whose key is not in the
// baseline is refused by TestNoNewUserRequiredKnobs (which runs under
// `make ci`), naming the offending file:line — until the baseline is updated in
// the SAME commit. Like pythongate, the gate does not ban context knobs; it
// bans NEW user-required ones.
package ctxknobs
