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

// TestObserveClosesEmpiricalAlphaLoop is the #4201 witness: a pattern issued off an
// OPTIMISTIC declared prior keeps being predicted only until enough real outcomes
// are observed; once the MEASURED hit-rate falls below the floor the gate FLIPS and
// the pattern stops being issued. The speculator self-corrects to observed traffic
// instead of trusting the declared α — the article's "measure α, tune to it"
// discipline made executable. The whole loop runs end-to-end through
// ResolveAndObserve: predict -> stage -> resolve+observe.
func TestObserveClosesEmpiricalAlphaLoop(t *testing.T) {
	spec := NewSpeculator(0.5) // MinProb floor = 0.5
	spec.MinTrials = 10        // warm up after 10 observed outcomes
	p := readPattern()
	p.SuccessProb = 0.95 // OPTIMISTIC declared prior, well above the floor
	spec.Learn(p)

	prior := []*Result{{Payload: roRef("file-42")}}

	// Before any outcome: only the optimistic declared prior exists, so it is issued.
	if spec.Predict("list", prior, 0) == nil {
		t.Fatal("with only the optimistic declared prior, the pattern must be issued")
	}
	if _, warm := spec.MeasuredProb("list", "read"); warm {
		t.Fatal("no outcomes observed yet — the pattern must not be warm")
	}

	// Drive outcomes at a TRUE hit-rate of 1-in-5 (0.2), well below the 0.5 floor.
	// Every 5th authoritative call confirms (commit); the rest refute (squash).
	hit := &ToolCall{Tool: "read", Args: roRef("read:file-42")}  // byte-identical => match
	miss := &ToolCall{Tool: "read", Args: roRef("read:file-99")} // different args => miss

	const trials = 40
	commits := 0
	for i := 0; i < trials; i++ {
		pred := spec.Predict("list", prior, 0)
		if pred == nil {
			// Once the measured rate crosses below the floor the gate flips and Predict
			// stops issuing — but the dispatcher was still speculating this pattern, so we
			// keep FEEDING outcomes to it (reconsidered, not forgotten). Reconstruct the
			// predicted call so the trial is still attributed to (list, read).
			pred = &ToolCall{Tool: "read", Args: roRef("read:file-42"),
				Spec: SpeculationContext{Speculative: true, Epoch: 999}}
		}
		auth := miss
		if i%5 == 0 {
			auth = hit
		}
		sink := NewBufferSink()
		sink.Stage(pred.Spec.Epoch, roRef("provisional-read"))
		out, err := spec.ResolveAndObserve(context.Background(), []ProvisionalSink{sink}, 0, "list", pred, auth)
		if err != nil {
			t.Fatalf("ResolveAndObserve[%d]: %v", i, err)
		}
		if auth == hit {
			if out != OutcomeCommitted {
				t.Fatalf("trial %d: a matching authoritative call must commit, got %v", i, out)
			}
			if got := sink.Committed(); len(got) != 1 {
				t.Fatalf("trial %d: a committed speculation must leave its effect durable, got %v", i, got)
			}
			commits++
		} else if out != OutcomeSquashed {
			t.Fatalf("trial %d: a refuted speculation must squash, got %v", i, out)
		}
	}

	prob, warm := spec.MeasuredProb("list", "read")
	if !warm {
		t.Fatalf("after %d outcomes the pattern must be warm", trials)
	}
	if prob >= 0.5 {
		t.Fatalf("measured hit-rate must reflect the ~0.2 true rate, got %.3f", prob)
	}
	if commits != 8 {
		t.Fatalf("expected 8 commits at a 1-in-5 true rate over %d trials, got %d", trials, commits)
	}

	// THE WITNESS: the gate has flipped. Despite the declared 0.95 prior, the MEASURED
	// ~0.2 rate is below the 0.5 floor, so the pattern is no longer issued — the loop
	// self-corrected to observed traffic instead of trusting the optimistic prior.
	if got := spec.Predict("list", prior, 0); got != nil {
		t.Fatalf("measured α below the floor must flip the gate: expected nil, got %+v", got)
	}
}

// TestMeasuredProbPromotesConservativePattern is the inverse witness: a high
// MEASURED hit-rate must be able to PROMOTE a pattern whose declared prior sat below
// the floor and would never be issued on the prior alone. The loop corrects in BOTH
// directions — it does not merely demote optimists.
func TestMeasuredProbPromotesConservativePattern(t *testing.T) {
	spec := NewSpeculator(0.5)
	spec.MinTrials = 10
	p := readPattern()
	p.SuccessProb = 0.3 // conservative declared prior, BELOW the 0.5 floor
	spec.Learn(p)

	prior := []*Result{{Payload: roRef("file-42")}}

	// On the declared prior alone the pattern is refused.
	if got := spec.Predict("list", prior, 0); got != nil {
		t.Fatalf("a sub-floor declared prior must decline before observation, got %+v", got)
	}

	// Observe a high true hit-rate: every authoritative call confirms the prediction.
	pred := &ToolCall{Tool: "read", Args: roRef("read:file-42")}
	for i := 0; i < 20; i++ {
		spec.Observe("list", pred, pred)
	}

	prob, warm := spec.MeasuredProb("list", "read")
	if !warm || prob <= 0.5 {
		t.Fatalf("after 20 confirmed outcomes the measured rate must clear the floor, got prob=%.3f warm=%v", prob, warm)
	}
	// THE WITNESS: the measured rate promotes the pattern the conservative prior blocked.
	if got := spec.Predict("list", prior, 0); got == nil {
		t.Fatal("a measured hit-rate above the floor must PROMOTE the pattern past its sub-floor declared prior")
	}
}

// TestObserveWarmupTrustsDeclaredPrior pins the warmup floor: below MinTrials the
// sample is too small to trust, so a run of early MISSES must NOT evict a pattern
// whose declared prior clears the floor. The measured rate governs only once warm.
func TestObserveWarmupTrustsDeclaredPrior(t *testing.T) {
	spec := NewSpeculator(0.5)
	spec.MinTrials = 20
	p := readPattern()
	p.SuccessProb = 0.9
	spec.Learn(p)

	prior := []*Result{{Payload: roRef("file-42")}}
	pred := &ToolCall{Tool: "read", Args: roRef("read:file-42")}
	miss := &ToolCall{Tool: "read", Args: roRef("read:file-99")}

	// 5 straight misses — but that is below the 20-trial warmup floor.
	for i := 0; i < 5; i++ {
		spec.Observe("list", pred, miss)
	}
	if _, warm := spec.MeasuredProb("list", "read"); warm {
		t.Fatal("5 outcomes is below the 20-trial floor — must not be warm")
	}
	// Still issued: the declared 0.9 prior governs until warmup.
	if got := spec.Predict("list", prior, 0); got == nil {
		t.Fatal("below the warmup floor the declared prior must still govern; expected an issued prediction")
	}
}

// TestObserveDisabledIsNoop pins the default-off invariant for the feedback path: a
// disabled or nil speculator, and a nil predicted call, record NOTHING (a kernel
// that never speculates measures nothing) and never panic.
func TestObserveDisabledIsNoop(t *testing.T) {
	var zero Speculator // disabled (Enabled == false)
	pred := &ToolCall{Tool: "read", Args: roRef("read:file-42")}
	zero.Observe("list", pred, pred) // must be a no-op, not a panic
	if prob, warm := zero.MeasuredProb("list", "read"); prob != 0 || warm {
		t.Fatalf("a disabled speculator observes nothing, got prob=%.3f warm=%v", prob, warm)
	}

	var nilSpec *Speculator
	nilSpec.Observe("list", pred, pred) // nil receiver must not panic

	// A nil predicted call records no trial even on an enabled speculator.
	spec := NewSpeculator(0)
	spec.Observe("list", nil, pred)
	if _, warm := spec.MeasuredProb("list", "read"); warm {
		t.Fatal("a nil predicted call must record no trial")
	}
}
