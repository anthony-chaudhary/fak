package sessionreset

import "github.com/anthony-chaudhary/fak/internal/ctxplan"

// relaypolicy.go — issue #1901 (epic #1860, Track I — Pointer-only carryover): the STRICT
// contributor set a RELAY reset carries.
//
// THE PROBLEM. The default BuildSeed folds every registered contributor, including
// verbatim_tail (the last N turns, verbatim) and the model recap (model_distill, a
// model-written "where we are"). Those are exactly the transcript-shaped, GROWING pieces a
// human keeps across a one-off reset. A relay, by contrast, rotates the same goal across an
// unbounded number of legs — so its carryover must be O(1) DURABLE POINTERS, not a recap
// that grows a little more every leg until it is itself the thing that needs compacting.
//
// THE POLICY. A relay seed folds ONLY pointer-class contributors:
//
//   - objective_pin — the standing goal's addressable identity (id + content digest). The
//     pin is O(1) and stable across legs; it is the anchor #1583 already made checkable via
//     CarryObjective. The relay host mints/carries it (PinObjective / RepinObjective) and
//     passes it in, so the relay seed names exactly which objective this leg continues.
//   - warm_prefix   — a DESCRIPTOR (token count + digest) for replaying the stable system
//     prefix from the vCache prefix-DAG. It carries no preamble bytes, only a pointer, and
//     does not grow with the transcript.
//
// It deliberately EXCLUDES verbatim_tail and model_distill (the two the issue names) AND
// task_distill and durability_facts — the latter two lift transcript text and grow with the
// session, so they are not O(1) pointers. The result: a relay leg's seed is bounded by the
// pin and the prefix descriptor, never by how long the prior leg ran.
//
// This does NOT change the default (non-relay) seed policy: BuildSeed still folds the whole
// registry. BuildRelaySeed is a SEPARATE entry a relay driver opts into; a caller that never
// relays sees no behavior change.

// RelayContributorNames is the closed set of contributor Names a relay seed is allowed to
// carry, in the order they are folded. It is the audit surface for #1901: a relay seed's
// Parts must be a SUBSET of this set — any other contributor firing means a transcript-shaped
// or growing part leaked into what must stay pointer-only. Returned as a fresh slice so a
// caller cannot mutate the policy.
func RelayContributorNames() []string {
	return []string{"objective_pin", "warm_prefix"}
}

// relayPolicyContributors returns the ordered pointer-class contributor set a relay reset
// folds: the objective pin (injected — it is opt-in and never lives in the global registry)
// followed by the warm-prefix descriptor. Neither lifts verbatim transcript bytes nor grows
// with the session, so the folded seed is O(1) in the drained transcript's length.
func relayPolicyContributors(pin ctxplan.ObjectivePin) []Contributor {
	return []Contributor{
		NewObjectivePinContributor(pin), // objective_pin — Order 22; declines on a zero pin
		warmPrefix{},                    // warm_prefix   — Order 0; a durable pointer, not bytes
	}
}

// BuildRelaySeed folds ONLY the strict pointer-only relay contributor set over in and returns
// the carryover a relay leg is seeded with. Unlike BuildSeed it does NOT consult the global
// registry, so a peer that registered a transcript-carrying contributor cannot leak it into a
// relay leg's seed — the relay policy is a closed set by construction. pin is the objective
// the relay carries forward (mint the first with PinObjective, chain successors with
// RepinObjective); a zero pin makes the objective_pin line decline, leaving only the
// warm-prefix pointer. The fold order, sort, and render match BuildSeed exactly.
func BuildRelaySeed(pin ctxplan.ObjectivePin, in Input) Seed {
	return buildSeedFrom(relayPolicyContributors(pin), in)
}
