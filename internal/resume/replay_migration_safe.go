package resume

import "sort"

// This file answers ONE pure question at the mid-stream token-replay boundary:
// when a live turn's worker dies, may we warm-continue it by replaying the tokens
// already delivered to a fresh worker, or would that replay silently corrupt the
// output? Some in-flight requests carry generation state that a token replay CANNOT
// reproduce, so replaying their tokens onto a new worker re-emits wrong output
// instead of failing honestly. This predicate names those traits and refuses the
// replay for them, fail-closed.
//
// The borrowed safety rule (INSPIRE, clean-room from NVIDIA Dynamo's
// migration.rs @ea89e8bd, Apache-2.0) hard-disables replay migration whenever the
// request used:
//
//   - constrained / guided (structured) decode: the constrained-grammar FSM would
//     restart from its schema root on the new worker, so replaying tokens taken
//     mid-constraint yields invalid or double-constrained output; and
//   - n>1 multi-sequence sampling: per-sequence sampler state is not reconstructable
//     from the delivered token stream alone.
//
// Everything else — a plain single-sequence free-form turn — replays as intended.
// This is the first checkable step of epic #3352 / issue #5278: a deterministic
// predicate with a typed refusal reason, no clock, no network, no model.

// ReplayShape is the generation shape of an in-flight request, read from the body
// the client sent — exactly the traits that decide whether a token replay can
// faithfully reconstruct the request's mid-stream state. It intentionally holds no
// transcript, no tokens, and no identity: only the shape.
type ReplayShape struct {
	// Structured is true when the request drove its output through a constrained
	// grammar or guided decode (an OpenAI response_format / the native guided-decode
	// carriers). Such a request cannot be replay-migrated: the grammar FSM restarts
	// from its schema root on a fresh worker.
	Structured bool
	// Sequences is the number of parallel output sequences the request asked for
	// (the sampling n). A plain request is exactly 1. Any value above 1 is
	// multi-sequence and cannot be replay-migrated (per-sequence sampler state is
	// not reconstructable from delivered tokens). A value at or below 0 is a
	// degenerate shape the caller could not read as a positive count.
	Sequences int
	// ShapeKnown reports whether the caller could actually read this request's
	// generation shape. When false the shape is unknown and the predicate refuses
	// fail-closed — an unread shape is treated as non-replayable, never as safe.
	// The zero-value ReplayShape (ShapeKnown false) therefore refuses by default.
	ShapeKnown bool
}

// The closed reason vocabulary a refusal carries, so an observability sink can
// record WHY a replay migration was refused without exposing any request content.
const (
	// ReplayReasonStructuredDecode: the request used constrained / guided (structured)
	// decode, whose grammar FSM state cannot be rebuilt from delivered tokens.
	ReplayReasonStructuredDecode = "structured_decode_active"
	// ReplayReasonMultiSequence: the request asked for n>1 parallel sequences, whose
	// per-sequence sampler state cannot be rebuilt from the delivered token stream.
	ReplayReasonMultiSequence = "multi_sequence_requested"
	// ReplayReasonUnknownShape: the request's generation shape could not be read (or
	// read as a degenerate non-positive sequence count), so the replay refuses
	// fail-closed rather than migrating on a guess.
	ReplayReasonUnknownShape = "unknown_generation_shape"
)

// ReplayVerdict is the typed answer: may this request be warm-continued by replaying
// its delivered tokens, and if not, the closed reasons why. ReplaySafe is true only
// for a fully-read plain single-sequence free-form request; otherwise Reasons lists
// every non-replayable trait found, in a fixed order, and ReplaySafe is false.
type ReplayVerdict struct {
	// ReplaySafe reports whether a token replay may faithfully reconstruct the
	// request's mid-stream state. True only when Reasons is empty.
	ReplaySafe bool `json:"replay_safe"`
	// Reasons is the closed set of non-replayable traits, sorted and deduplicated,
	// empty when ReplaySafe is true. A request that trips several traits at once
	// carries all of them, so the caller sees every reason its request was refused.
	Reasons []string `json:"reasons,omitempty"`
}

// ReplayMigrationSafe is THE deterministic predicate: same shape in, same verdict
// out — no clock, no I/O, no model. It refuses fail-closed on any non-replayable
// trait (unread shape, degenerate sequence count, structured decode, or n>1) and
// allows a token replay only for a fully-read plain single-sequence free-form turn.
// It is total: every ReplayShape yields a defined verdict, and the zero value
// refuses (unknown shape).
func ReplayMigrationSafe(shape ReplayShape) ReplayVerdict {
	reasons := map[string]struct{}{}

	if !shape.ShapeKnown || shape.Sequences <= 0 {
		// The shape could not be read, or read as a degenerate non-positive
		// sequence count: refuse fail-closed, never migrate on a guess.
		reasons[ReplayReasonUnknownShape] = struct{}{}
	}
	if shape.Structured {
		reasons[ReplayReasonStructuredDecode] = struct{}{}
	}
	if shape.Sequences > 1 {
		reasons[ReplayReasonMultiSequence] = struct{}{}
	}

	if len(reasons) == 0 {
		return ReplayVerdict{ReplaySafe: true}
	}
	out := make([]string, 0, len(reasons))
	for r := range reasons {
		out = append(out, r)
	}
	sort.Strings(out)
	return ReplayVerdict{ReplaySafe: false, Reasons: out}
}
