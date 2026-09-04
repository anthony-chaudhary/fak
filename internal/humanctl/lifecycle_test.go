package humanctl

import (
	"strings"
	"testing"
	"time"
)

// TestHumanControlLifecycleStepByStep verifies the progressive lifecycle rungs of an instruction envelope.
func TestHumanControlLifecycleStepByStep(t *testing.T) {
	now := time.Date(2026, 9, 4, 10, 0, 0, 0, time.UTC)

	// Step 1: Draft creation (must have no addressee and empty outcome).
	draft := Envelope{
		Instruction: Instruction{Verb: FlagConcern, Text: "suspicious loop behavior"},
		Delivery:    DeliveryDraft,
		Lifetime:    Lifetime{Duration: DurationTurn},
	}
	if err := draft.ValidateAt(now); err != nil {
		t.Fatalf("draft validation failed: %v", err)
	}

	// Draft with an addressee must fail closed.
	invalidDraft := draft
	invalidDraft.Addressee = Addressee{Kind: AddresseeTurn, Cardinality: CardinalityOne, IDs: []string{"turn-1"}}
	if err := invalidDraft.ValidateAt(now); err == nil {
		t.Fatal("expected draft with addressee to fail validation")
	}

	// Draft with an outcome must fail closed.
	invalidDraftOutcome := draft
	invalidDraftOutcome.Outcome = Outcome{Receipt: ReceiptUnacknowledged, Admission: AdmissionPending, Effect: EffectUnobserved}
	if err := invalidDraftOutcome.ValidateAt(now); err == nil {
		t.Fatal("expected draft with outcome to fail validation")
	}

	// Step 2: Enqueue / dispatch to a session.
	env := Envelope{
		Instruction: Instruction{Verb: Redirect, Target: "internal/humanctl", Reason: "advance maturity"},
		Delivery:    DeliveryImmediate,
		Addressee:   Addressee{Kind: AddresseeSession, Cardinality: CardinalityOne, IDs: []string{"session-42"}},
		Lifetime:    Lifetime{Duration: DurationSession},
		Outcome:     Outcome{Receipt: ReceiptUnacknowledged, Admission: AdmissionPending, Effect: EffectUnobserved},
	}
	if err := env.ValidateAt(now); err != nil {
		t.Fatalf("dispatched envelope validation failed: %v", err)
	}

	// Step 3: Transport receipt acknowledged.
	env.Outcome.Receipt = ReceiptAcknowledged
	if err := env.ValidateAt(now); err != nil {
		t.Fatalf("acknowledged receipt envelope validation failed: %v", err)
	}

	// Step 4: Admitted by executor.
	env.Outcome.Admission = AdmissionAccepted
	env.Outcome.Effect = EffectPending
	if err := env.ValidateAt(now); err != nil {
		t.Fatalf("admitted envelope validation failed: %v", err)
	}

	// Step 5: Effect observed with required witness.
	env.Outcome.Effect = EffectObserved
	env.Outcome.EffectWitness = "sha256:abcd1234ef5678"
	if err := env.ValidateAt(now); err != nil {
		t.Fatalf("completed envelope validation failed: %v", err)
	}

	// Guard: observed effect without witness must fail closed.
	noWitness := env
	noWitness.Outcome.EffectWitness = ""
	if err := noWitness.ValidateAt(now); err == nil {
		t.Fatal("expected observed effect without witness to fail validation")
	}

	// Guard: rejected admission cannot claim effect.
	rejected := env
	rejected.Outcome.Admission = AdmissionRejected
	rejected.Outcome.Effect = EffectPending
	rejected.Outcome.EffectWitness = ""
	if err := rejected.ValidateAt(now); err == nil {
		t.Fatal("expected rejected admission with pending effect to fail validation")
	}
}

// TestPauseResumeLifecycle verifies human lifecycle controls for suspending and resuming execution.
func TestPauseResumeLifecycle(t *testing.T) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)

	// Check pause verb properties in catalog.
	pauseDef, ok := Lookup("pause")
	if !ok {
		t.Fatal("lookup for pause failed")
	}
	if pauseDef.Verb != Pause {
		t.Fatalf("expected verb %v, got %v", Pause, pauseDef.Verb)
	}
	if pauseDef.Family != Lifecycle {
		t.Fatalf("expected family %v, got %v", Lifecycle, pauseDef.Family)
	}
	if pauseDef.CanCompose {
		t.Fatal("pause must not be composable with subsequent commands")
	}
	if pauseDef.Terminal {
		t.Fatal("pause is a suspension, not a terminal termination")
	}

	// Pause instruction alone validates cleanly.
	pauseInst := Instruction{Verb: Pause, Reason: "human review required"}
	if err := pauseInst.Validate(); err != nil {
		t.Fatalf("pause instruction validation failed: %v", err)
	}

	// Compose with pause alone succeeds.
	prog, err := Compose(pauseInst)
	if err != nil {
		t.Fatalf("compose pause failed: %v", err)
	}
	if len(prog) != 1 {
		t.Fatalf("expected 1 instruction, got %d", len(prog))
	}

	// Compose with pause followed by another instruction must fail closed.
	_, err = Compose(pauseInst, Instruction{Verb: Continue})
	if err == nil {
		t.Fatal("expected compose with instruction after pause to fail")
	}
	if !strings.Contains(err.Error(), "must be last") {
		t.Fatalf("expected 'must be last' error, got: %v", err)
	}

	// Check resume verb properties in catalog.
	resumeDef, ok := Lookup("resume")
	if !ok {
		t.Fatal("lookup for resume failed")
	}
	if resumeDef.Verb != Resume || resumeDef.Family != Lifecycle {
		t.Fatalf("unexpected resume definition: %#v", resumeDef)
	}
	if resumeDef.CanCompose {
		t.Fatal("resume must not be composable with subsequent commands")
	}

	// Resume instruction alone validates cleanly.
	resumeInst := Instruction{Verb: Resume, Reason: "human review approved"}
	if err := resumeInst.Validate(); err != nil {
		t.Fatalf("resume instruction validation failed: %v", err)
	}

	// Compose with resume alone succeeds.
	if _, err := Compose(resumeInst); err != nil {
		t.Fatalf("compose resume failed: %v", err)
	}

	// Compose with instruction after resume must fail closed.
	if _, err := Compose(resumeInst, Instruction{Verb: Continue}); err == nil {
		t.Fatal("expected compose with instruction after resume to fail")
	}

	// Session token wire mapping for lifecycle tokens.
	tokenCases := []struct {
		token    string
		wantVerb Verb
		valid    bool
	}{
		{"PAUSE_SESSION", Pause, true},
		{"CONTINUE", Continue, true},
		{"STOP_SESSION", Stop, true},
		{"END_TURN", EndTurn, true},
		{"UNKNOWN_TOKEN", "", false},
		{"RESUME", "", false}, // wire mapping only supports declared tokens
	}
	for _, tc := range tokenCases {
		inst, ok := InstructionFromSessionDecision(tc.token)
		if ok != tc.valid {
			t.Fatalf("InstructionFromSessionDecision(%q) ok = %v; want %v", tc.token, ok, tc.valid)
		}
		if tc.valid && inst.Verb != tc.wantVerb {
			t.Fatalf("InstructionFromSessionDecision(%q) verb = %v; want %v", tc.token, inst.Verb, tc.wantVerb)
		}
	}

	// Envelope containing Pause instruction with until_expiry lifetime.
	pauseEnv := Envelope{
		Instruction: pauseInst,
		Delivery:    DeliveryImmediate,
		Addressee:   Addressee{Kind: AddresseeSession, Cardinality: CardinalityOne, IDs: []string{"sess-99"}},
		Lifetime:    Lifetime{Duration: DurationUntilExpiry, ExpiresAt: now.Add(10 * time.Minute)},
		Outcome:     Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionAccepted, Effect: EffectPending},
	}
	if err := pauseEnv.ValidateAt(now); err != nil {
		t.Fatalf("pause envelope validation failed: %v", err)
	}

	// When now is past expiry, validation fails closed.
	if err := pauseEnv.ValidateAt(now.Add(15 * time.Minute)); err == nil {
		t.Fatal("expected expired pause envelope to fail validation")
	}
}

// TestHumanControlOverride verifies human instructions that override or redirect agent behavior.
func TestHumanControlOverride(t *testing.T) {
	// Redirection override requires a target.
	redirWithoutTarget := Instruction{Verb: Redirect, Reason: "change course"}
	if err := redirWithoutTarget.Validate(); err == nil {
		t.Fatal("expected redirect without target to fail validation")
	}

	redirWithTarget := Instruction{
		Verb:   Redirect,
		Target: "internal/safepath",
		Reason: "avoid forbidden subsystem",
		Text:   "switch focus immediately",
	}
	if err := redirWithTarget.Validate(); err != nil {
		t.Fatalf("valid redirect failed: %v", err)
	}

	// Check strength override: default vs explicit.
	if redirWithTarget.EffectiveStrength() != StrengthMedium {
		t.Fatalf("expected default strength %v, got %v", StrengthMedium, redirWithTarget.EffectiveStrength())
	}

	overriddenStrength := redirWithTarget
	overriddenStrength.Strength = StrengthAbsolute
	if overriddenStrength.EffectiveStrength() != StrengthAbsolute {
		t.Fatalf("expected overridden strength %v, got %v", StrengthAbsolute, overriddenStrength.EffectiveStrength())
	}

	// Multi-step override composition: Reject -> Avoid -> Redirect -> Verify.
	overrideProgram, err := Compose(
		Instruction{Verb: Reject, Reason: "produced diff contains unintended edits"},
		Instruction{Verb: Avoid, Target: "rm -rf", Reason: "destructive command"},
		Instruction{Verb: Redirect, Target: "internal/humanctl", Reason: "focus here"},
		Instruction{Verb: Verify, Target: "go vet checks", Reason: "witness correctness"},
	)
	if err != nil {
		t.Fatalf("multi-step override compose failed: %v", err)
	}
	if len(overrideProgram) != 4 {
		t.Fatalf("expected 4 instructions in override program, got %d", len(overrideProgram))
	}

	// Terminal override: Stop terminates and cannot be followed.
	stopDef, ok := Lookup("stop")
	if !ok || !stopDef.Terminal {
		t.Fatal("stop must be defined as terminal")
	}
	stopInst := Instruction{Verb: Stop, Reason: "operator abort"}
	if err := stopInst.Validate(); err != nil {
		t.Fatalf("stop instruction validation failed: %v", err)
	}

	// Stop at end of composition is valid.
	termProg, err := Compose(
		Instruction{Verb: Reject, Reason: "unrecoverable error"},
		stopInst,
	)
	if err != nil {
		t.Fatalf("terminal program compose failed: %v", err)
	}
	if len(termProg) != 2 {
		t.Fatalf("expected 2 instructions, got %d", len(termProg))
	}

	// Stop before another instruction fails closed.
	if _, err := Compose(stopInst, Instruction{Verb: Continue}); err == nil {
		t.Fatal("expected compose with instruction after stop to fail")
	}
}

// TestAddresseeScopingConstraints verifies target constraints across execution tiers.
func TestAddresseeScopingConstraints(t *testing.T) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	base := Envelope{
		Instruction: Instruction{Verb: Continue},
		Delivery:    DeliveryNextTurn,
		Lifetime:    Lifetime{Duration: DurationTurn},
		Outcome:     Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionPending, Effect: EffectUnobserved},
	}

	// Valid single addressee.
	single := base
	single.Addressee = Addressee{Kind: AddresseeSubagent, Cardinality: CardinalityOne, IDs: []string{"worker-1"}}
	if err := single.ValidateAt(now); err != nil {
		t.Fatalf("valid single addressee failed: %v", err)
	}

	// Single addressee with multiple IDs fails closed.
	singleMulti := base
	singleMulti.Addressee = Addressee{Kind: AddresseeSubagent, Cardinality: CardinalityOne, IDs: []string{"worker-1", "worker-2"}}
	if err := singleMulti.ValidateAt(now); err == nil {
		t.Fatal("expected single addressee with 2 IDs to fail")
	}

	// Valid cohort addressee (>= 2 IDs).
	cohort := base
	cohort.Addressee = Addressee{Kind: AddresseeCohort, Cardinality: CardinalityMany, IDs: []string{"worker-1", "worker-2"}}
	if err := cohort.ValidateAt(now); err != nil {
		t.Fatalf("valid cohort failed: %v", err)
	}

	// Cohort with 1 ID fails closed.
	cohortOne := base
	cohortOne.Addressee = Addressee{Kind: AddresseeCohort, Cardinality: CardinalityMany, IDs: []string{"worker-1"}}
	if err := cohortOne.ValidateAt(now); err == nil {
		t.Fatal("expected cohort with 1 ID to fail")
	}

	// Valid fleet addressee (0 IDs).
	fleet := base
	fleet.Addressee = Addressee{Kind: AddresseeFleet, Cardinality: CardinalityAll}
	if err := fleet.ValidateAt(now); err != nil {
		t.Fatalf("valid fleet failed: %v", err)
	}

	// Fleet with IDs fails closed.
	fleetWithIDs := base
	fleetWithIDs.Addressee = Addressee{Kind: AddresseeFleet, Cardinality: CardinalityAll, IDs: []string{"worker-1"}}
	if err := fleetWithIDs.ValidateAt(now); err == nil {
		t.Fatal("expected fleet with IDs to fail")
	}

	// Blank ID fails closed.
	blankID := base
	blankID.Addressee = Addressee{Kind: AddresseeSession, Cardinality: CardinalityOne, IDs: []string{"   "}}
	if err := blankID.ValidateAt(now); err == nil {
		t.Fatal("expected blank ID to fail")
	}
}

// BenchmarkHumanCtl measures end-to-end performance of lookup, instruction validation, compose, and envelope checks.
func BenchmarkHumanCtl(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		def, ok := Lookup("redirect")
		if !ok {
			b.Fatal("lookup failed")
		}
		inst := Instruction{
			Verb:     def.Verb,
			Strength: StrengthHigh,
			Target:   "internal/humanctl",
			Reason:   "benchmark execution",
		}
		if err := inst.Validate(); err != nil {
			b.Fatal(err)
		}
		prog, err := Compose(inst, Instruction{Verb: Verify, Target: "benchmark witness"})
		if err != nil {
			b.Fatal(err)
		}
		env := Envelope{
			Instruction: prog[0],
			Delivery:    DeliveryImmediate,
			Addressee:   Addressee{Kind: AddresseeSession, Cardinality: CardinalityOne, IDs: []string{"bench-session"}},
			Lifetime:    Lifetime{Duration: DurationSession},
			Outcome:     Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionAccepted, Effect: EffectPending},
		}
		if err := env.ValidateAt(now); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCompose measures multi-instruction composition validation throughput.
func BenchmarkCompose(b *testing.B) {
	insts := []Instruction{
		{Verb: Reject, Reason: "benchmark reject"},
		{Verb: Avoid, Target: "danger_zone", Reason: "safety"},
		{Verb: Redirect, Target: "safe_zone", Reason: "steer"},
		{Verb: Verify, Target: "check_results"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Compose(insts...); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEnvelopeValidate measures envelope validation throughput at fixed evaluation timestamp.
func BenchmarkEnvelopeValidate(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	env := Envelope{
		Instruction: Instruction{Verb: Verify, Target: "perf witness"},
		Delivery:    DeliveryNextSafePoint,
		Addressee:   Addressee{Kind: AddresseeSubagent, Cardinality: CardinalityOne, IDs: []string{"worker-bench"}},
		Lifetime:    Lifetime{Duration: DurationUntilExpiry, ExpiresAt: now.Add(time.Hour)},
		Outcome:     Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionAccepted, Effect: EffectObserved, EffectWitness: "observed-proof"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := env.ValidateAt(now); err != nil {
			b.Fatal(err)
		}
	}
}
