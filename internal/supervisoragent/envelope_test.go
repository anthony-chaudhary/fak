package supervisoragent

import (
	"errors"
	"reflect"
	"testing"
)

// presentInput is a fully-witnessed SupervisorInput: every surface present
// (some deliberately empty — a present empty list is NOT an absent witness).
func presentInput() SupervisorInput {
	return SupervisorInput{
		Liveness:    Seen(Liveness{RunID: "RID-1", Class: "moving", Commits: 3}),
		Workers:     Seen([]WorkerVerdict{{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent", State: WorkerHealthy}}),
		Escalations: Seen([]Escalation{}),
		Leases:      Seen([]Lease{}),
	}
}

// fullWidths grants every action class its demanded width — the widest earned
// envelope expressible.
func fullWidths(t *testing.T) map[ActionKind]int {
	t.Helper()
	out := map[ActionKind]int{}
	for _, k := range []ActionKind{ActionSpawn, ActionReplace, ActionRedispatch, ActionWiden, ActionEscalate, ActionHold} {
		d, ok := DemandFor(k)
		if !ok {
			t.Fatalf("DemandFor(%q) missing from the ladder", k)
		}
		out[k] = d
	}
	return out
}

// TestEnvelopeModeSoakSwitch pins the warn->enforce soak switch (mirror of
// #3517's operator triage gate): unset/warn is the soak default, enforce is
// explicit, and a MALFORMED token resolves to enforce — a typo can only
// tighten the envelope, never widen it (fail-closed).
func TestEnvelopeModeSoakSwitch(t *testing.T) {
	cases := []struct {
		value string
		want  EnvelopeMode
	}{
		{"", ModeWarn},
		{"warn", ModeWarn},
		{" WARN ", ModeWarn},
		{"enforce", ModeEnforce},
		{"Enforce", ModeEnforce},
		// Malformed tokens fail closed: unrecognized never means "off".
		{"off", ModeEnforce},
		{"0", ModeEnforce},
		{"disabled", ModeEnforce},
		{"warn,enforce", ModeEnforce},
	}
	for _, c := range cases {
		if got := ModeFromValue(c.value); got != c.want {
			t.Errorf("ModeFromValue(%q) = %q, want %q", c.value, got, c.want)
		}
	}
}

// TestEnvelopeLadderCoversVocabulary pins the pre-declared ladder as data:
// every one of the six action classes has a demand row; the two narrowing
// verbs (hold, escalate) demand width 0 — they stay inside EVERY envelope,
// including the collapsed fail-closed one — and every widening verb demands
// at least width 1, so a zero-width (or missing) earned entry never licenses
// an unattended widening action.
func TestEnvelopeLadderCoversVocabulary(t *testing.T) {
	zero := map[ActionKind]bool{ActionHold: true, ActionEscalate: true}
	for _, k := range []ActionKind{ActionSpawn, ActionReplace, ActionRedispatch, ActionWiden, ActionEscalate, ActionHold} {
		d, ok := DemandFor(k)
		if !ok {
			t.Fatalf("ladder is missing action class %q", k)
		}
		if zero[k] && d != 0 {
			t.Errorf("narrowing verb %q demands width %d, want 0", k, d)
		}
		if !zero[k] && d < 1 {
			t.Errorf("widening verb %q demands width %d, want >= 1", k, d)
		}
	}
	if _, ok := DemandFor(ActionKind("exfiltrate")); ok {
		t.Errorf("ladder claims a demand for an out-of-vocabulary class")
	}
}

// TestActionAcrossRatchetThreshold is the DoD witness: it drives ONE action
// class across the ratchet threshold. Below the earned width the action is
// PROPOSED (enforce: confirmation required, no admission verb runs); at and
// above the earned width it executes unattended and leaves its witnessed
// artifact.
func TestActionAcrossRatchetThreshold(t *testing.T) {
	in := presentInput()
	act := SpawnAction{Issue: "4480", Lane: "supervisoragent"}
	demand, ok := DemandFor(ActionSpawn)
	if !ok {
		t.Fatal("DemandFor(spawn) missing")
	}

	t.Run("below the earned width: proposed, admission never reached", func(t *testing.T) {
		env := Envelope{Widths: map[ActionKind]int{ActionSpawn: demand - 1}}
		d := Authorize(in, env, ModeEnforce, act)
		if d.Auth != AuthConfirm || d.Reason != ReasonSubEnvelope {
			t.Fatalf("verdict = %+v, want confirm/%s", d, ReasonSubEnvelope)
		}
		if d.Demand != demand || d.Earned != demand-1 {
			t.Errorf("verdict widths = demand %d earned %d, want %d/%d", d.Demand, d.Earned, demand, demand-1)
		}
		v := &fakeVerbs{}
		_, eff, err := LowerInEnvelope(act, v, in, env, ModeEnforce)
		if !errors.Is(err, ErrConfirmRequired) {
			t.Fatalf("LowerInEnvelope error = %v, want ErrConfirmRequired", err)
		}
		if len(v.calls) != 0 {
			t.Errorf("a sub-envelope action reached admission verbs: %v", v.calls)
		}
		if eff.Lease != nil || eff.Admit != nil || eff.Packet != nil {
			t.Errorf("a proposed action carries an artifact: %+v", eff)
		}
	})

	for name, width := range map[string]int{"at the earned width": demand, "above the earned width": demand + 1} {
		t.Run(name+": executes unattended with a witnessed artifact", func(t *testing.T) {
			env := Envelope{Widths: map[ActionKind]int{ActionSpawn: width}}
			d := Authorize(in, env, ModeEnforce, act)
			if d.Auth != AuthUnattended || d.Reason != ReasonSuperEnvelope {
				t.Fatalf("verdict = %+v, want unattended/%s", d, ReasonSuperEnvelope)
			}
			v := &fakeVerbs{}
			_, eff, err := LowerInEnvelope(act, v, in, env, ModeEnforce)
			if err != nil {
				t.Fatalf("LowerInEnvelope error: %v", err)
			}
			if !reflect.DeepEqual(v.calls, []string{"arbitrate(supervisoragent,[])"}) {
				t.Errorf("admission calls = %v, want the lane arbitration", v.calls)
			}
			if eff.Lease == nil {
				t.Errorf("unattended spawn left no witnessed lease row: %+v", eff)
			}
		})
	}
}

// TestWarnModeRecordsButAllows pins the soak semantics: in warn mode a
// sub-envelope action still executes, but the verdict RECORDS the would-be
// confirmation (Finding) so the soak leaves a typed trail. An in-envelope
// action records no finding.
func TestWarnModeRecordsButAllows(t *testing.T) {
	in := presentInput()
	act := SpawnAction{Issue: "4480", Lane: "supervisoragent"}
	demand, _ := DemandFor(ActionSpawn)

	env := Envelope{Widths: map[ActionKind]int{ActionSpawn: demand - 1}}
	d, eff, err := LowerInEnvelope(act, &fakeVerbs{}, in, env, ModeWarn)
	if err != nil {
		t.Fatalf("warn-mode sub-envelope action refused: %v", err)
	}
	if d.Auth != AuthUnattended || d.Reason != ReasonSubEnvelope || d.Finding != FindingWouldConfirm {
		t.Errorf("warn verdict = %+v, want allowed with %s recorded", d, FindingWouldConfirm)
	}
	if eff.Lease == nil {
		t.Errorf("warn-mode action left no artifact: %+v", eff)
	}

	wide := Envelope{Widths: map[ActionKind]int{ActionSpawn: demand}}
	d2 := Authorize(in, wide, ModeWarn, act)
	if d2.Finding != "" {
		t.Errorf("in-envelope verdict carries a finding: %+v", d2)
	}
}

// TestFailClosedOnRefusedRow is the DoD fail-closed witness: an observed
// witness_refused row collapses the envelope in BOTH modes — the widening
// verbs escalate (never act, never merely confirm), no admission verb runs,
// and the soak switch cannot soften it. The two narrowing verbs (hold,
// escalate) stay executable: the mandated fail-closed response IS escalation,
// so the escape hatch can never deadlock.
func TestFailClosedOnRefusedRow(t *testing.T) {
	in := presentInput()
	env := Envelope{Widths: fullWidths(t), WitnessRefused: true}
	widening := []SupervisorAction{
		SpawnAction{Issue: "4480", Lane: "supervisoragent"},
		ReplaceAction{RunID: "RID-dead", Issue: "4480", Lane: "supervisoragent"},
		RedispatchAction{Issue: "4480", Lane: "supervisoragent"},
		WidenAction{Lane: "supervisoragent", Tree: []string{"internal/supervisoragent/**"}},
	}
	for _, mode := range []EnvelopeMode{ModeWarn, ModeEnforce} {
		for _, act := range widening {
			d := Authorize(in, env, mode, act)
			if d.Auth != AuthEscalate || d.Reason != ReasonRefused {
				t.Errorf("mode %s %s: verdict = %+v, want escalate/%s", mode, act.Kind(), d, ReasonRefused)
			}
			v := &fakeVerbs{}
			_, eff, err := LowerInEnvelope(act, v, in, env, mode)
			if !errors.Is(err, ErrEnvelopeFailClosed) {
				t.Errorf("mode %s %s: error = %v, want ErrEnvelopeFailClosed", mode, act.Kind(), err)
			}
			if len(v.calls) != 0 {
				t.Errorf("mode %s %s: fail-closed action reached admission verbs: %v", mode, act.Kind(), v.calls)
			}
			if eff.Lease != nil || eff.Admit != nil || eff.Packet != nil {
				t.Errorf("mode %s %s: fail-closed action carries an artifact: %+v", mode, act.Kind(), eff)
			}
		}
		// The escape hatch: escalate and hold remain unattended under fail-closed.
		esc := EscalateAction{RunID: "RID-1", Issue: "4480", Class: "stall", Severity: "operator", ReasonCode: "REQUIRE_WITNESS"}
		v := &fakeVerbs{}
		d, eff, err := LowerInEnvelope(esc, v, in, env, mode)
		if err != nil || d.Auth != AuthUnattended || d.Reason != ReasonNarrowing {
			t.Errorf("mode %s: escalate under fail-closed = (%+v, %v), want unattended/%s", mode, d, err, ReasonNarrowing)
		}
		if eff.Packet == nil {
			t.Errorf("mode %s: fail-closed escalation left no packet", mode)
		}
		if d2, _, err := LowerInEnvelope(HoldAction{}, &fakeVerbs{}, in, env, mode); err != nil || d2.Auth != AuthUnattended {
			t.Errorf("mode %s: hold under fail-closed = (%+v, %v), want unattended", mode, d2, err)
		}
	}
}

// TestFailClosedOnAbsentSurface pins the other fail-closed leg: an ABSENT
// input surface (a witness that could not be read) forces escalation for every
// widening verb in both modes, even at full earned width. A present-but-empty
// surface does not — green presence and green absence stay distinct.
func TestFailClosedOnAbsentSurface(t *testing.T) {
	env := Envelope{Widths: fullWidths(t)}
	absent := presentInput()
	absent.Workers = Absent[[]WorkerVerdict]()
	act := SpawnAction{Issue: "4480", Lane: "supervisoragent"}

	for _, mode := range []EnvelopeMode{ModeWarn, ModeEnforce} {
		d := Authorize(absent, env, mode, act)
		if d.Auth != AuthEscalate || d.Reason != ReasonSurfaceAbsent {
			t.Errorf("mode %s: verdict on absent surface = %+v, want escalate/%s", mode, d, ReasonSurfaceAbsent)
		}
		v := &fakeVerbs{}
		if _, _, err := LowerInEnvelope(act, v, absent, env, mode); !errors.Is(err, ErrEnvelopeFailClosed) {
			t.Errorf("mode %s: error = %v, want ErrEnvelopeFailClosed", mode, err)
		}
		if len(v.calls) != 0 {
			t.Errorf("mode %s: absent-surface action reached admission verbs: %v", mode, v.calls)
		}
	}

	if d := Authorize(presentInput(), env, ModeEnforce, act); d.Auth != AuthUnattended {
		t.Errorf("present-but-empty surfaces escalated: %+v", d)
	}
}

// TestEnvelopeFailClosedOnMalformed pins the malformed edges: a nil or
// out-of-vocabulary action escalates before any verb runs, and a missing (or
// nil-map) earned-width entry reads as width 0 — an unconfigured envelope
// never defaults open.
func TestEnvelopeFailClosedOnMalformed(t *testing.T) {
	in := presentInput()
	env := Envelope{Widths: fullWidths(t)}

	for name, act := range map[string]SupervisorAction{"nil": nil, "rogue": rogueAction{}} {
		d := Authorize(in, env, ModeWarn, act)
		if d.Auth != AuthEscalate || d.Reason != ReasonOutOfVocabulary {
			t.Errorf("%s action verdict = %+v, want escalate/%s", name, d, ReasonOutOfVocabulary)
		}
		v := &fakeVerbs{}
		if _, _, err := LowerInEnvelope(act, v, in, env, ModeWarn); !errors.Is(err, ErrEnvelopeFailClosed) {
			t.Errorf("%s action error = %v, want ErrEnvelopeFailClosed", name, err)
		}
		if len(v.calls) != 0 {
			t.Errorf("%s action reached admission verbs: %v", name, v.calls)
		}
	}

	empty := Envelope{} // nil widths map: nothing earned
	for _, act := range []SupervisorAction{
		SpawnAction{Issue: "4480", Lane: "supervisoragent"},
		WidenAction{Lane: "supervisoragent"},
	} {
		if d := Authorize(in, empty, ModeEnforce, act); d.Auth != AuthConfirm || d.Earned != 0 {
			t.Errorf("unconfigured envelope verdict for %s = %+v, want confirm at earned 0", act.Kind(), d)
		}
	}
}

// TestKeepBitVerdictFromCounterSeries is the DoD fixture: the keep-or-revert
// verdict computed from a synthesized babysitting-counter series (touches per
// witnessed shipped unit, sampled at soak checkpoints against the pre-seat
// baseline). A rise reverts the seat; an unmeasured baseline or a soak with no
// measured checkpoint can never testify keep (fail-closed).
func TestKeepBitVerdictFromCounterSeries(t *testing.T) {
	measured := func(v float64) KeepSample { return KeepSample{Measured: true, Value: v} }
	cases := []struct {
		name     string
		baseline KeepSample
		soak     []KeepSample
		wantBit  KeepBit
		wantWhy  string
	}{
		{
			name:     "touches fell across the soak: keep",
			baseline: measured(3.0),
			soak:     []KeepSample{measured(2.5), measured(2.0), measured(1.5)},
			wantBit:  KeepSeat, wantWhy: KeepReasonHeld,
		},
		{
			name:     "flat is not a rise: keep",
			baseline: measured(2.0),
			soak:     []KeepSample{measured(2.0)},
			wantBit:  KeepSeat, wantWhy: KeepReasonHeld,
		},
		{
			name:     "a mid-soak rise reverts even if the tail recovers",
			baseline: measured(2.0),
			soak:     []KeepSample{measured(1.0), measured(2.5), measured(1.0)},
			wantBit:  RevertSeat, wantWhy: KeepReasonRose,
		},
		{
			name:     "unmeasured baseline can never testify keep",
			baseline: KeepSample{},
			soak:     []KeepSample{measured(0.5)},
			wantBit:  RevertSeat, wantWhy: KeepReasonBaselineUnmeasured,
		},
		{
			name:     "empty soak series can never testify keep",
			baseline: measured(2.0),
			soak:     nil,
			wantBit:  RevertSeat, wantWhy: KeepReasonSoakUnmeasured,
		},
		{
			name:     "all-unmeasured soak can never testify keep",
			baseline: measured(2.0),
			soak:     []KeepSample{{}, {}},
			wantBit:  RevertSeat, wantWhy: KeepReasonSoakUnmeasured,
		},
		{
			name:     "an unmeasured checkpoint's value is never read",
			baseline: measured(2.0),
			soak:     []KeepSample{{Measured: false, Value: 99}, measured(1.5)},
			wantBit:  KeepSeat, wantWhy: KeepReasonHeld,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := KeepBitVerdict(c.baseline, c.soak)
			if got.Bit != c.wantBit || got.Reason != c.wantWhy {
				t.Errorf("verdict = %+v, want %s/%s", got, c.wantBit, c.wantWhy)
			}
		})
	}
	// The rise fixture also carries its evidence: baseline and observed peak.
	rise := KeepBitVerdict(measured(2.0), []KeepSample{measured(2.5)})
	if rise.Baseline != 2.0 || rise.Peak != 2.5 {
		t.Errorf("rise verdict evidence = %+v, want baseline 2.0 / peak 2.5", rise)
	}
}

// TestKeepBitPreservedAcrossRatchet pins the keep-bit's independence from the
// ratchet: the verdict is a pure function of the counter series — driving
// actions through the envelope in warn mode, flipping the soak switch to
// enforce, and driving fail-closed refusals neither mutates the series nor
// changes the verdict. The ratchet gates ACTIONS; only the counter moves the
// keep-bit.
func TestKeepBitPreservedAcrossRatchet(t *testing.T) {
	baseline := KeepSample{Measured: true, Value: 3.0}
	soak := []KeepSample{
		{Measured: true, Value: 2.5},
		{Measured: false},
		{Measured: true, Value: 2.0},
	}
	soakCopy := append([]KeepSample(nil), soak...)

	before := KeepBitVerdict(baseline, soak)
	if before.Bit != KeepSeat {
		t.Fatalf("fixture verdict = %+v, want keep", before)
	}

	in := presentInput()
	act := SpawnAction{Issue: "4480", Lane: "supervisoragent"}
	demand, _ := DemandFor(ActionSpawn)
	narrow := Envelope{Widths: map[ActionKind]int{ActionSpawn: demand - 1}}
	closed := Envelope{Widths: fullWidths(t), WitnessRefused: true}

	if _, _, err := LowerInEnvelope(act, &fakeVerbs{}, in, narrow, ModeWarn); err != nil {
		t.Fatalf("warn-mode soak action failed: %v", err)
	}
	if _, _, err := LowerInEnvelope(act, &fakeVerbs{}, in, narrow, ModeEnforce); !errors.Is(err, ErrConfirmRequired) {
		t.Fatalf("enforce-mode action error = %v, want ErrConfirmRequired", err)
	}
	if _, _, err := LowerInEnvelope(act, &fakeVerbs{}, in, closed, ModeEnforce); !errors.Is(err, ErrEnvelopeFailClosed) {
		t.Fatalf("fail-closed action error = %v, want ErrEnvelopeFailClosed", err)
	}

	after := KeepBitVerdict(baseline, soak)
	if !reflect.DeepEqual(before, after) {
		t.Errorf("keep-bit moved across the ratchet: before %+v, after %+v", before, after)
	}
	if !reflect.DeepEqual(soak, soakCopy) {
		t.Errorf("gating mutated the counter series: %+v", soak)
	}
}
