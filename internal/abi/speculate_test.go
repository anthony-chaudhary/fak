package abi

import (
	"context"
	"testing"
)

// roRef is a tiny helper: an inline Ref carrying the given bytes (the derived-arg
// payload a pattern produces). Inline keeps the test resolver-free.
func roRef(s string) Ref { return Ref{Kind: RefInline, Inline: []byte(s)} }

// readPattern builds a PASTE-style pattern that, after a "list" call, predicts a
// read-only "read" whose args are DERIVED from the prior output (the id the list
// returned). The Meta stamps it effect-free so the default-deny gate admits it.
func readPattern() SpecPattern {
	return SpecPattern{
		Signature:   "list",
		PredictTool: "read",
		SuccessProb: 0.9,
		Meta:        map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
		DeriveArgs: func(prior []*Result) (Ref, bool) {
			if len(prior) == 0 || prior[0] == nil {
				return Ref{}, false // args not derivable from prior outputs
			}
			return roRef("read:" + string(prior[0].Payload.Inline)), true
		},
	}
}

// TestPredictMatchCommit is the issue's first mandated test: predict -> match ->
// commit. The speculator predicts the next call, a provisional effect is staged
// under its epoch, the model's authoritative emission MATCHES the prediction, and
// Resolve commits the effect (OutcomeCommitted) so it lands in the durable ledger.
func TestPredictMatchCommit(t *testing.T) {
	spec := NewSpeculator(0.5)
	spec.Learn(readPattern())

	prior := []*Result{{Payload: roRef("file-42")}}
	pred := spec.Predict("list", prior, 7 /*parentEpoch*/)
	if pred == nil {
		t.Fatal("expected a speculative prediction, got nil")
	}
	if !pred.Spec.Speculative || pred.Spec.Epoch == 0 {
		t.Fatalf("prediction must be stamped speculative with a non-zero epoch, got %+v", pred.Spec)
	}
	if pred.Spec.ParentEpoch != 7 {
		t.Fatalf("prediction must branch from the parent epoch 7, got %d", pred.Spec.ParentEpoch)
	}
	if pred.Tool != "read" || string(pred.Args.Inline) != "read:file-42" {
		t.Fatalf("prediction tool/args wrong: tool=%q args=%q", pred.Tool, string(pred.Args.Inline))
	}

	// Run the speculation: a provisional effect lands in the sink under the epoch.
	sink := NewBufferSink()
	sink.Stage(pred.Spec.Epoch, roRef("provisional-read-result"))
	if got := sink.Committed(); len(got) != 0 {
		t.Fatalf("provisional effect must NOT be committed before resolution, got %d", len(got))
	}

	// The model authoritatively emits the SAME call -> MATCH -> commit.
	authoritative := &ToolCall{Tool: "read", Args: roRef("read:file-42")}
	if !PredictionMatches(pred, authoritative) {
		t.Fatal("identical predicted/authoritative call must match")
	}
	out, err := Resolve(context.Background(), []ProvisionalSink{sink}, 0, pred, authoritative)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if out != OutcomeCommitted {
		t.Fatalf("a matched prediction must resolve OutcomeCommitted, got %v", out)
	}
	committed := sink.Committed()
	if len(committed) != 1 || string(committed[0].Inline) != "provisional-read-result" {
		t.Fatalf("matched prediction must commit the provisional effect, got %v", committed)
	}
	if sink.PendingEpochs() != 0 {
		t.Fatalf("no epoch should remain pending after commit, got %d", sink.PendingEpochs())
	}
}

// TestPredictMissSquash is the issue's second mandated test: predict -> miss ->
// squash. The speculator predicts, a provisional effect is staged, the model's
// authoritative emission DIFFERS, and Resolve squashes the effect (OutcomeSquashed)
// so the buffer is left with NO trace of the speculative branch — "squash actually
// undoes the effect".
func TestPredictMissSquash(t *testing.T) {
	spec := NewSpeculator(0.5)
	spec.Learn(readPattern())

	prior := []*Result{{Payload: roRef("file-42")}}
	pred := spec.Predict("list", prior, 0)
	if pred == nil {
		t.Fatal("expected a speculative prediction, got nil")
	}

	sink := NewBufferSink()
	sink.Stage(pred.Spec.Epoch, roRef("provisional-read-result"))

	// The model authoritatively emits a DIFFERENT call -> MISS -> squash.
	// Same tool, different args is still a miss (the provisional result was computed
	// for the predicted args).
	authoritative := &ToolCall{Tool: "read", Args: roRef("read:file-99")}
	if PredictionMatches(pred, authoritative) {
		t.Fatal("a call with different derived args must NOT match")
	}
	out, err := Resolve(context.Background(), []ProvisionalSink{sink}, 0, pred, authoritative)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if out != OutcomeSquashed {
		t.Fatalf("a missed prediction must resolve OutcomeSquashed, got %v", out)
	}
	if got := sink.Committed(); len(got) != 0 {
		t.Fatalf("squash must leave NO committed effect, got %v", got)
	}
	if sink.PendingEpochs() != 0 {
		t.Fatalf("squash must clear the provisional scratch, %d epochs still pending", sink.PendingEpochs())
	}
}

// TestSpeculatorDefaultOff pins the safety floor: a zero-value (disabled)
// speculator, and a nil one, predict NOTHING — so a kernel that never opts in is
// byte-for-byte the v0.1 no-op (every call ordinary, no epoch ever issued).
func TestSpeculatorDefaultOff(t *testing.T) {
	var zero Speculator // Enabled == false
	zero.Learn(readPattern())
	if got := zero.Predict("list", []*Result{{Payload: roRef("file-42")}}, 0); got != nil {
		t.Fatalf("a disabled speculator must predict nil, got %+v", got)
	}
	var nilSpec *Speculator
	if got := nilSpec.Predict("list", nil, 0); got != nil {
		t.Fatalf("a nil speculator must predict nil, got %+v", got)
	}
}

// TestSpeculationDefaultDenyOnEffects pins THE LAW: a mutating / destructive /
// unstamped prediction is NEVER issued, even when a pattern matches and the args
// are derivable. Only a provably effect-free call clears the gate.
func TestSpeculationDefaultDenyOnEffects(t *testing.T) {
	derive := func(prior []*Result) (Ref, bool) { return roRef("x"), true }

	cases := []struct {
		name    string
		pattern SpecPattern
		wantNil bool
	}{
		{
			name: "write-shaped tool refused",
			pattern: SpecPattern{Signature: "sig", PredictTool: "delete_file", SuccessProb: 1,
				Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}, DeriveArgs: derive},
			wantNil: true,
		},
		{
			name: "explicit destructive refused",
			pattern: SpecPattern{Signature: "sig", PredictTool: "fetch", SuccessProb: 1,
				Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true", "destructive": "true"}, DeriveArgs: derive},
			wantNil: true,
		},
		{
			name: "missing read-only hint refused",
			pattern: SpecPattern{Signature: "sig", PredictTool: "fetch", SuccessProb: 1,
				Meta: map[string]string{"idempotentHint": "true"}, DeriveArgs: derive},
			wantNil: true,
		},
		{
			name: "missing idempotent hint refused",
			pattern: SpecPattern{Signature: "sig", PredictTool: "fetch", SuccessProb: 1,
				Meta: map[string]string{"readOnlyHint": "true"}, DeriveArgs: derive},
			wantNil: true,
		},
		{
			name: "effect-free read admitted",
			pattern: SpecPattern{Signature: "sig", PredictTool: "fetch", SuccessProb: 1,
				Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}, DeriveArgs: derive},
			wantNil: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec := NewSpeculator(0)
			spec.Learn(tc.pattern)
			got := spec.Predict("sig", nil, 0)
			if tc.wantNil && got != nil {
				t.Fatalf("effectful prediction must be refused (nil), got %+v", got)
			}
			if !tc.wantNil && got == nil {
				t.Fatal("effect-free prediction must be admitted, got nil")
			}
		})
	}
}

// TestPredictResistsUnderivableArgs pins the resist-speculation case: when the
// symbolic DeriveArgs cannot build the args from prior outputs (the freely-
// generated-arg case), the pattern declines and no speculation is issued.
func TestPredictResistsUnderivableArgs(t *testing.T) {
	spec := NewSpeculator(0)
	spec.Learn(SpecPattern{
		Signature: "list", PredictTool: "read", SuccessProb: 1,
		Meta:       map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
		DeriveArgs: func(prior []*Result) (Ref, bool) { return Ref{}, false }, // never derivable
	})
	if got := spec.Predict("list", []*Result{{Payload: roRef("file-42")}}, 0); got != nil {
		t.Fatalf("an un-derivable arg pattern must decline, got %+v", got)
	}
}

// TestPredictBelowProbFloorDeclines pins the economics floor: a pattern whose
// empirical success probability is below MinProb is not worth the slack and is not
// issued.
func TestPredictBelowProbFloorDeclines(t *testing.T) {
	spec := NewSpeculator(0.8)
	p := readPattern()
	p.SuccessProb = 0.5 // below the 0.8 floor
	spec.Learn(p)
	if got := spec.Predict("list", []*Result{{Payload: roRef("file-42")}}, 0); got != nil {
		t.Fatalf("a sub-floor prediction must decline, got %+v", got)
	}
}

// TestPredictPicksHighestProbability pins pattern selection: among several matching
// patterns the speculator issues the one with the highest empirical success
// probability (the best bet for the slack spent).
func TestPredictPicksHighestProbability(t *testing.T) {
	spec := NewSpeculator(0)
	lo := readPattern()
	lo.PredictTool, lo.SuccessProb = "read_lo", 0.3
	hi := readPattern()
	hi.PredictTool, hi.SuccessProb = "read_hi", 0.95
	spec.Learn(lo)
	spec.Learn(hi)
	got := spec.Predict("list", []*Result{{Payload: roRef("file-42")}}, 0)
	if got == nil || got.Tool != "read_hi" {
		t.Fatalf("must pick the highest-probability matching pattern, got %+v", got)
	}
}

// TestBufferSinkRetractsOnly pins the BufferSink contract directly: Promote makes
// exactly the promoted epoch's effects durable; Rollback retracts an epoch and a
// promote of a different epoch is unaffected (per-epoch isolation).
func TestBufferSinkRetractsOnly(t *testing.T) {
	sink := NewBufferSink()
	sink.Stage(1, roRef("a"))
	sink.Stage(1, roRef("b"))
	sink.Stage(2, roRef("c"))

	if sink.PendingEpochs() != 2 {
		t.Fatalf("two epochs staged, got %d pending", sink.PendingEpochs())
	}

	// Squash epoch 2: it retracts, leaving no trace; epoch 1 is untouched.
	if err := sink.Rollback(context.Background(), 0, 2); err != nil {
		t.Fatalf("Rollback: %v", err)
	}
	// Commit epoch 1: both its effects become durable, in stage order.
	if err := sink.Promote(context.Background(), 0, 1); err != nil {
		t.Fatalf("Promote: %v", err)
	}
	got := sink.Committed()
	if len(got) != 2 || string(got[0].Inline) != "a" || string(got[1].Inline) != "b" {
		t.Fatalf("only epoch 1's effects, in stage order, must be committed; got %v", got)
	}
	if sink.PendingEpochs() != 0 {
		t.Fatalf("both epochs resolved, got %d pending", sink.PendingEpochs())
	}
}

// TestPredictionMatchesNilIsMiss pins the fail-closed match rule: a nil prediction
// or a nil authoritative call never matches (an absent prediction can never claim a
// hit, so it always squashes).
func TestPredictionMatchesNilIsMiss(t *testing.T) {
	c := &ToolCall{Tool: "read", Args: roRef("x")}
	if PredictionMatches(nil, c) {
		t.Fatal("nil prediction must not match")
	}
	if PredictionMatches(c, nil) {
		t.Fatal("nil authoritative call must not match")
	}
	if !PredictionMatches(c, &ToolCall{Tool: "read", Args: roRef("x")}) {
		t.Fatal("identical calls must match")
	}
}
