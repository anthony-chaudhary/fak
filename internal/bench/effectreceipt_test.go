package bench

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runOneArm captures the pre-state and drives a single arm — the two-call shape the
// contract requires, so a test can capture over one subject set and then run over a
// different one (the scope-widening path).
func runOneArm(t *testing.T, ctx context.Context, env *SyntheticEnv, arm EffectArm) EffectReceipt {
	t.Helper()
	rc, before := CaptureSubjects(env, arm.Subjects)
	if before.Stage != StageBefore || !before.Observed {
		t.Fatalf("capture stage = %q observed=%v, want %s observed", before.Stage, before.Observed, StageBefore)
	}
	return RunEffectArm(ctx, rc, arm, env)
}

func nopMeasure(context.Context) error { return nil }

// TestEffectReceipt is the re-runnable witness for issue #5683. One deterministic
// sequence demonstrates all three halves the proof artifacts name on the same run:
//
//   - the ACCEPTED case — arm 1 applies, is observed effective, measures,
//     compensates, and is independently observed back at the captured state;
//   - the FAILED-RESTORATION refusal — arm 2's cleanup REPORTS SUCCESS while the
//     post-action readback proves the subject never came back; and
//   - the CONTAMINATION refusal — arm 3 is turned away before it observes anything,
//     because the subject it needs is exactly the one arm 2 leaked.
//
// The sequence is pinned to a committed golden (the scrubbed, re-derivable receipt).
func TestEffectReceipt(t *testing.T) {
	env := DefaultEffectEnv()
	got := RunEffectSequence(context.Background(), DefaultEffectArms(env), env)

	if got.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("provenance = %q, want %q", got.Provenance.Kind, ProvenanceSimulated)
	}
	if got.Verdict != VerdictEffectFlagged {
		t.Fatalf("verdict = %q, want %q", got.Verdict, VerdictEffectFlagged)
	}
	if len(got.Arms) != 3 {
		t.Fatalf("arms = %d, want 3", len(got.Arms))
	}

	// --- the accepted case ---
	base := got.Arms[0]
	if !base.Success() || base.Outcome != OutcomeRestored {
		t.Errorf("baseline arm = %q success=%v, want %q success", base.Outcome, base.Success(), OutcomeRestored)
	}
	if !base.StateVerifiedClean || !base.MeasurementRan || !base.CompensationAccepted {
		t.Errorf("baseline arm clean/measured/compensated = %v/%v/%v, want all true",
			base.StateVerifiedClean, base.MeasurementRan, base.CompensationAccepted)
	}

	// --- the failed-restoration case: primary success must NOT hide it ---
	leaky := got.Arms[1]
	if leaky.Outcome != OutcomeRestoreFailed || leaky.Success() {
		t.Errorf("leaky arm = %q success=%v, want %q non-success", leaky.Outcome, leaky.Success(), OutcomeRestoreFailed)
	}
	if !leaky.MeasurementRan {
		t.Errorf("leaky arm did not run its measurement; the point of the case is that the PRIMARY operation succeeded")
	}
	if !leaky.CompensationAccepted {
		t.Errorf("leaky arm compensation_accepted = false; the point of the case is that cleanup CLAIMED success")
	}
	if leaky.StateVerifiedClean {
		t.Errorf("leaky arm reported verified-clean state while its subject never came back")
	}

	// --- the contamination refusal ---
	succ := got.Arms[2]
	if succ.Outcome != OutcomePredecessorUnclean || succ.Success() {
		t.Errorf("successor arm = %q, want %q", succ.Outcome, OutcomePredecessorUnclean)
	}
	if succ.MeasurementRan {
		t.Errorf("successor arm ran its measurement after being refused")
	}
	for _, st := range succ.Stages {
		if st.Refusal != OutcomePredecessorUnclean {
			t.Errorf("successor stage %s refusal = %q, want %q", st.Stage, st.Refusal, OutcomePredecessorUnclean)
		}
		if len(st.Subjects) != 0 {
			t.Errorf("successor stage %s observed %d subject(s); a refused arm must touch nothing",
				st.Stage, len(st.Subjects))
		}
	}

	// Every arm records all five typed stages, in order, all bound to one receipt.
	wantStages := []string{StageBefore, StageRequested, StageEffective, StageRestoreAttempted, StageRestored}
	for _, arm := range got.Arms {
		if len(arm.Stages) != len(wantStages) {
			t.Fatalf("arm %q has %d stages, want %d", arm.Arm, len(arm.Stages), len(wantStages))
		}
		for i, want := range wantStages {
			if arm.Stages[i].Stage != want {
				t.Errorf("arm %q stage[%d] = %q, want %q", arm.Arm, i, arm.Stages[i].Stage, want)
			}
			if arm.Stages[i].Binding != arm.Binding {
				t.Errorf("arm %q stage %s binding = %q, want the arm's binding %q — all stages must bind to ONE receipt",
					arm.Arm, want, arm.Stages[i].Binding, arm.Binding)
			}
		}
		// Acceptance stages are never marked independently observed; readback
		// stages always are.
		for _, st := range arm.Stages {
			observed := st.Stage != StageRequested && st.Stage != StageRestoreAttempted
			if st.Observed != observed {
				t.Errorf("arm %q stage %s independently_observed = %v, want %v",
					arm.Arm, st.Stage, st.Observed, observed)
			}
		}
	}

	// Undeclared ambient state is preserved: no stage ever reached a subject the
	// capture did not observe.
	if v := env.Snapshot()[AmbientSubject]; v != "ambient-untouched" {
		t.Errorf("ambient subject %s = %q, want it untouched; the transaction widened past its receipt",
			AmbientSubject, v)
	}

	// The exported convenience re-derives the same run byte-for-byte (determinism:
	// no clock, no randomness, no map iteration in output).
	gotJSON, err := got.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	defJSON, err := DefaultEffectSequence().JSON()
	if err != nil {
		t.Fatalf("DefaultEffectSequence JSON: %v", err)
	}
	if !bytes.Equal(gotJSON, defJSON) {
		t.Errorf("DefaultEffectSequence() drifted from the inline run — the receipt is not deterministic")
	}

	golden := filepath.Join("testdata", "effect_receipt.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(gotJSON, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(gotJSON, "\n")) {
		t.Errorf("receipt drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestEffectReceiptRestorationPaths is the synthetic-adapter matrix the contract
// names: restoration is attempted and verified after success, after a benchmark
// failure, and after cancellation; a cleanup failure is caught; and ambient state
// the arm never declared is preserved on every one of those paths.
func TestEffectReceiptRestorationPaths(t *testing.T) {
	cases := []struct {
		name        string
		arm         func(env *SyntheticEnv, cancel context.CancelFunc) EffectArm
		inject      func(env *SyntheticEnv)
		cancelled   bool
		wantOutcome string
		wantClean   bool
		wantMeasure bool
	}{
		{
			name:        "success restores",
			arm:         func(*SyntheticEnv, context.CancelFunc) EffectArm { return effectArm(nopMeasure) },
			wantOutcome: OutcomeRestored, wantClean: true, wantMeasure: true,
		},
		{
			name: "benchmark failure still restores",
			arm: func(*SyntheticEnv, context.CancelFunc) EffectArm {
				return effectArm(func(context.Context) error { return errors.New("benchmark body failed") })
			},
			wantOutcome: OutcomeMeasurementFailed, wantClean: true, wantMeasure: true,
		},
		{
			name: "cancellation still restores",
			arm: func(_ *SyntheticEnv, cancel context.CancelFunc) EffectArm {
				return effectArm(func(ctx context.Context) error { cancel(); return ctx.Err() })
			},
			cancelled:   true,
			wantOutcome: OutcomeCancelled, wantClean: true, wantMeasure: true,
		},
		{
			name:        "cleanup failure is caught by the post-action readback",
			arm:         func(*SyntheticEnv, context.CancelFunc) EffectArm { return effectArm(nopMeasure) },
			inject:      func(env *SyntheticEnv) { env.Inject("bench.threads", FaultInjectRestoreIgnored) },
			wantOutcome: OutcomeRestoreFailed, wantClean: false, wantMeasure: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := DefaultEffectEnv()
			if tc.inject != nil {
				tc.inject(env)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			r := runOneArm(t, ctx, env, tc.arm(env, cancel))
			if r.Outcome != tc.wantOutcome {
				t.Errorf("outcome = %q, want %q (finding: %s)", r.Outcome, tc.wantOutcome, r.Finding)
			}
			if r.StateVerifiedClean != tc.wantClean {
				t.Errorf("state_verified_clean = %v, want %v", r.StateVerifiedClean, tc.wantClean)
			}
			if r.MeasurementRan != tc.wantMeasure {
				t.Errorf("measurement_ran = %v, want %v", r.MeasurementRan, tc.wantMeasure)
			}
			if (r.Outcome == OutcomeRestored) != r.Success() {
				t.Errorf("Success()=%v disagrees with outcome %q", r.Success(), r.Outcome)
			}

			// Compensation is ATTEMPTED on every path — success, failure, and
			// cancellation alike — and the readback stage always runs.
			attempt := effectStage(t, r, StageRestoreAttempted)
			if len(attempt.Subjects) == 0 {
				t.Errorf("no compensation was attempted on the %q path", tc.name)
			}
			if len(effectStage(t, r, StageRestored).Subjects) == 0 {
				t.Errorf("no independent post-action readback ran on the %q path", tc.name)
			}

			// Ambient state the arm never declared is untouched on every path.
			if v := env.Snapshot()[AmbientSubject]; v != "ambient-untouched" {
				t.Errorf("ambient subject leaked on the %q path: %q", tc.name, v)
			}
		})
	}
}

// TestEffectReceiptTypedRefusals exercises EVERY typed outcome the contract uses, so
// none of them is decorative: each is reachable from a real adapter fault and each
// keeps Success() false. Ineffective application, failed restoration, and an
// unreadable post-state in particular must survive a primary operation that
// succeeded.
func TestEffectReceiptTypedRefusals(t *testing.T) {
	cases := []struct {
		name        string
		fault       string
		on          string
		wantOutcome string
		wantClean   bool
	}{
		{"apply rejected", FaultInjectApplyRejects, "bench.threads", OutcomeApplyRejected, true},
		{"accepted but ineffective", FaultInjectApplyIgnored, "bench.threads", OutcomeIneffective, true},
		{"pre-state unreadable", FaultInjectReadBefore, "bench.threads", OutcomePreStateUnreadable, false},
		// A subject that becomes unreadable the moment it is mutated is UNKNOWN, not
		// merely ineffective — and unknown must never collapse into clean.
		{"unreadable once mutated", FaultInjectReadAfterApply, "bench.threads", OutcomePostStateUnreadable, false},
		{"compensation rejected", FaultInjectRestoreRejects, "bench.threads", OutcomeRestoreFailed, false},
		{"compensation ignored", FaultInjectRestoreIgnored, "bench.threads", OutcomeRestoreFailed, false},
		{"post-state unreadable", FaultInjectReadAfterRestore, "bench.threads", OutcomePostStateUnreadable, false},
	}

	seen := map[string]bool{OutcomeRestored: true, OutcomeMeasurementFailed: true, OutcomeCancelled: true}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := DefaultEffectEnv().Inject(tc.on, tc.fault)
			r := runOneArm(t, context.Background(), env, effectArm(nopMeasure))
			seen[r.Outcome] = true
			if r.Outcome != tc.wantOutcome {
				t.Errorf("outcome = %q, want %q (finding: %s)", r.Outcome, tc.wantOutcome, r.Finding)
			}
			if r.Success() {
				t.Errorf("outcome %q reported Success(); only %q may", r.Outcome, OutcomeRestored)
			}
			if r.StateVerifiedClean != tc.wantClean {
				t.Errorf("state_verified_clean = %v, want %v", r.StateVerifiedClean, tc.wantClean)
			}
			if strings.TrimSpace(r.Finding) == "" {
				t.Errorf("outcome %q carries no finding — a refusal must name its reason", r.Outcome)
			}
			// A pre-state refusal must never have mutated anything.
			if tc.wantOutcome == OutcomePreStateUnreadable && env.Snapshot()["bench.run_label"] != "baseline" {
				t.Errorf("a pre-state refusal mutated a subject: %q", env.Snapshot()["bench.run_label"])
			}
		})
	}

	// Scope widening: capture over ONE subject, then hand the arm a wider set. The
	// refusal must land before a single Apply.
	t.Run("scope widening", func(t *testing.T) {
		env := DefaultEffectEnv()
		narrow := []SubjectMutation{{Subject: "bench.threads", Requested: "1"}}
		rc, _ := CaptureSubjects(env, narrow)
		r := RunEffectArm(context.Background(), rc, effectArm(nopMeasure), env)
		seen[r.Outcome] = true
		if r.Outcome != OutcomeSubjectSetWidened || r.Success() {
			t.Fatalf("outcome = %q, want %q", r.Outcome, OutcomeSubjectSetWidened)
		}
		if r.MeasurementRan {
			t.Errorf("a widened arm ran its measurement")
		}
		before := DefaultEffectEnv().Snapshot()
		for k, want := range before {
			if got := env.Snapshot()[k]; got != want {
				t.Errorf("subject %q = %q after a widening refusal, want the untouched %q", k, got, want)
			}
		}
	})

	// A nil receipt observed nothing, so ANY mutation widens it.
	t.Run("no receipt at all", func(t *testing.T) {
		env := DefaultEffectEnv()
		r := RunEffectArm(context.Background(), nil, effectArm(nopMeasure), env)
		if r.Outcome != OutcomeSubjectSetWidened {
			t.Errorf("nil receipt outcome = %q, want %q", r.Outcome, OutcomeSubjectSetWidened)
		}
	})

	seen[OutcomePredecessorUnclean] = DefaultEffectSequence().Arms[2].Outcome == OutcomePredecessorUnclean

	// Every constant in the closed outcome vocabulary is actually reachable.
	for _, o := range []string{
		OutcomeRestored, OutcomeMeasurementFailed, OutcomeCancelled, OutcomeApplyRejected,
		OutcomeIneffective, OutcomeRestoreFailed, OutcomePostStateUnreadable,
		OutcomePreStateUnreadable, OutcomeSubjectSetWidened, OutcomePredecessorUnclean,
	} {
		if !seen[o] {
			t.Errorf("outcome %q is declared but no fixture reaches it — the vocabulary is decorative", o)
		}
	}
}

// TestEffectReceiptSemanticEquivalence proves the declared restoration rule is
// load-bearing rather than vacuous. The volatile subject genuinely does NOT come
// back byte-for-byte, so declaring it `exact` must flag failed restoration, while
// declaring it `semantic` with the matching normalizer restores. Same adapter, same
// values — only the declaration differs.
func TestEffectReceiptSemanticEquivalence(t *testing.T) {
	semantic := SubjectMutation{
		Subject: "bench.run_label", Requested: "arm-under-test",
		Equivalence: EquivalenceSemantic, Normalizer: "drop-volatile-write-marker",
		Normalize: StripVolatileMarker,
	}
	exact := SubjectMutation{Subject: "bench.run_label", Requested: "arm-under-test", Equivalence: EquivalenceExact}

	env := DefaultEffectEnv()
	r := runOneArm(t, context.Background(), env, EffectArm{Name: "semantic", Subjects: []SubjectMutation{semantic}, Measure: nopMeasure})
	if r.Outcome != OutcomeRestored {
		t.Errorf("semantic subject outcome = %q, want %q (finding: %s)", r.Outcome, OutcomeRestored, r.Finding)
	}
	if got := env.Snapshot()["bench.run_label"]; got == "baseline" {
		t.Fatalf("fixture drift: the volatile subject came back byte-for-byte (%q), so the semantic rule proves nothing", got)
	}

	env2 := DefaultEffectEnv()
	r2 := runOneArm(t, context.Background(), env2, EffectArm{Name: "exact", Subjects: []SubjectMutation{exact}, Measure: nopMeasure})
	if r2.Outcome != OutcomeRestoreFailed {
		t.Errorf("exact rule over a dynamic subject = %q, want %q — the rule is not load-bearing", r2.Outcome, OutcomeRestoreFailed)
	}

	// The policy the receipt PUBLISHES matches what was declared, per subject.
	if len(r.Policy) != 1 || r.Policy[0].Rule != EquivalenceSemantic || r.Policy[0].Normalizer != "drop-volatile-write-marker" {
		t.Errorf("published policy = %+v, want the declared semantic rule and its normalizer label", r.Policy)
	}
	if len(r2.Policy) != 1 || r2.Policy[0].Rule != EquivalenceExact || r2.Policy[0].Normalizer != "identity" {
		t.Errorf("published policy = %+v, want the declared exact rule", r2.Policy)
	}
}

// TestEffectReceiptScrubbedAndOpaque pins the two disclosure rules: the emitted
// receipt carries no raw environmental value and no adapter error text (only
// subject-salted digests and the typed fault vocabulary), and the subject-set
// receipt's identity is never published — the artifact proves the stages agree on
// scope through the non-reversible binding alone.
func TestEffectReceiptScrubbedAndOpaque(t *testing.T) {
	env := DefaultEffectEnv().Inject("bench.threads", FaultInjectRestoreRejects)
	rc, _ := CaptureSubjects(env, DefaultEffectSubjects())
	r := RunEffectArm(context.Background(), effectAssert(t, rc), effectArm(nopMeasure), env)

	seq := SequenceReceipt{Schema: "effect-receipt.v1", Arms: []EffectReceipt{r}}
	js, err := seq.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}

	// No raw value, requested or captured, and no adapter error text.
	for _, secret := range []string{"baseline", "arm-under-test", "ambient-untouched", "refused the compensating change"} {
		if bytes.Contains(js, []byte(secret)) {
			t.Errorf("receipt leaked %q — it is not publishable without disclosing the environment", secret)
		}
	}
	// The opaque receipt identity never appears.
	if rc.id == "" {
		t.Fatal("receipt has no identity to keep opaque")
	}
	if bytes.Contains(js, []byte(rc.id)) {
		t.Errorf("receipt published its own identity %q; only the subject-set binding may appear", rc.id)
	}
	// The binding does appear, and identically on every stage.
	if !bytes.Contains(js, []byte(rc.Binding())) {
		t.Errorf("receipt does not carry the subject-set binding at all")
	}

	// The compensation rejection surfaces as a TYPED fault, not as free text.
	attempt := effectStage(t, r, StageRestoreAttempted)
	found := false
	for _, st := range attempt.Subjects {
		if st.Subject == "bench.threads" {
			found = st.Fault == FaultRestoreRejected && st.Request == "rejected"
		}
	}
	if !found {
		t.Errorf("rejected compensation did not surface as the typed fault %q: %+v", FaultRestoreRejected, attempt.Subjects)
	}
	if r.CompensationAccepted {
		t.Errorf("compensation_accepted = true after the adapter rejected the compensating change")
	}
}

// TestEffectReceiptRequestIsNotEffect pins the distinction the contract turns on:
// an ACCEPTED request is not an observed effect. The adapter accepts the change and
// silently does nothing; the REQUESTED stage records acceptance, and only the
// independent EFFECTIVE readback catches that nothing happened.
func TestEffectReceiptRequestIsNotEffect(t *testing.T) {
	env := DefaultEffectEnv().Inject("bench.threads", FaultInjectApplyIgnored)
	r := runOneArm(t, context.Background(), env, effectArm(nopMeasure))

	req := effectStage(t, r, StageRequested)
	if req.Observed {
		t.Errorf("REQUESTED is marked independently observed; it is acceptance only")
	}
	for _, st := range req.Subjects {
		if st.Subject == "bench.threads" && st.Request != "accepted" {
			t.Errorf("REQUESTED for bench.threads = %q, want accepted (the adapter DID accept it)", st.Request)
		}
	}

	eff := effectStage(t, r, StageEffective)
	if !eff.Observed {
		t.Errorf("EFFECTIVE is not marked independently observed")
	}
	caught := false
	for _, st := range eff.Subjects {
		if st.Subject == "bench.threads" && st.Effect == "ineffective" {
			caught = true
		}
	}
	if !caught {
		t.Errorf("the independent readback did not catch the accepted-but-ineffective change: %+v", eff.Subjects)
	}
	if r.MeasurementRan {
		t.Errorf("the measurement ran over state that was never actually mutated")
	}
}

// effectArm builds the standard two-subject arm used across these tests.
func effectArm(measure func(context.Context) error) EffectArm {
	return EffectArm{Name: "arm", Subjects: DefaultEffectSubjects(), Measure: measure}
}

func effectAssert(t *testing.T, rc *SubjectReceipt) *SubjectReceipt {
	t.Helper()
	if rc == nil {
		t.Fatal("capture returned no receipt")
	}
	return rc
}

func effectStage(t *testing.T, r EffectReceipt, name string) StageRecord {
	t.Helper()
	for _, st := range r.Stages {
		if st.Stage == name {
			return st
		}
	}
	t.Fatalf("arm %q has no %s stage", r.Arm, name)
	return StageRecord{}
}
