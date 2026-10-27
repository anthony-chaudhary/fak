package supervisoragent

import (
	"reflect"
	"testing"
)

// TestDecideReasonVocabularyClosed pins the closed reason vocabulary: exactly
// these tokens, distinct, non-empty, with their wire spellings. Adding or
// renaming a token is a deliberate, test-breaking act.
func TestDecideReasonVocabularyClosed(t *testing.T) {
	wantTokens := map[Reason]string{
		ReasonWitnessAbsent:     "witness_absent",
		ReasonEnvelopeRefused:   "envelope_refused",
		ReasonHoldGreen:         "hold_green",
		ReasonSpawnAdmitted:     "spawn_admitted",
		ReasonReplaceStaleLease: "replace_stale_lease",
		ReasonReplanDrift:       "replan_drift",
		ReasonWidenCapacity:     "widen_capacity",
		ReasonEscalateResidual:  "escalate_residual",
	}
	if len(wantTokens) != 8 {
		t.Fatalf("vocabulary has %d tokens; want 8", len(wantTokens))
	}
	for r, tok := range wantTokens {
		if string(r) == "" {
			t.Error("empty reason token in the closed vocabulary")
		}
		if string(r) != tok {
			t.Errorf("reason token spells %q; want %q", string(r), tok)
		}
	}
}

// withhold* / with* build table inputs from presentInput()'s fully-witnessed
// green baseline (one healthy worker RID-1/4480/supervisoragent, moving
// liveness, no open packets, no leases).
func withholdLiveness(in SupervisorInput) SupervisorInput {
	in.Liveness = Absent[Liveness]()
	return in
}
func withholdWorkers(in SupervisorInput) SupervisorInput {
	in.Workers = Absent[[]WorkerVerdict]()
	return in
}
func withholdEscalations(in SupervisorInput) SupervisorInput {
	in.Escalations = Absent[[]Escalation]()
	return in
}
func withholdLeases(in SupervisorInput) SupervisorInput { in.Leases = Absent[[]Lease](); return in }

func withEmptyWorkers(in SupervisorInput) SupervisorInput {
	in.Workers = Seen([]WorkerVerdict{})
	return in
}

func withWorker(in SupervisorInput, w WorkerVerdict) SupervisorInput {
	vals := append(append([]WorkerVerdict{}, in.Workers.Value...), w)
	in.Workers = Seen(vals)
	return in
}

func withWorkerState(in SupervisorInput, runID string, st WorkerState) SupervisorInput {
	vals := make([]WorkerVerdict, len(in.Workers.Value))
	copy(vals, in.Workers.Value)
	for i := range vals {
		if vals[i].RunID == runID {
			vals[i].State = st
		}
	}
	in.Workers = Seen(vals)
	return in
}

func withLiveness(in SupervisorInput, runID, class string) SupervisorInput {
	l := in.Liveness.Value
	l.RunID, l.Class = runID, class
	in.Liveness = Seen(l)
	return in
}

func withStalledLiveness(in SupervisorInput) SupervisorInput {
	return withLiveness(in, in.Liveness.Value.RunID, stalledClass)
}

var absentHead = EscalateAction{Class: "witness_absent", Severity: "operator", ReasonCode: "witness_absent"}

// TestDecideTable covers the deterministic-first ladder end to end: absent
// witness -> escalate on EVERY verb path (fail-closed), envelope refusal
// (confirm keeps the proposal tagged refused; a collapsed envelope replaces it
// with the narrowing escalate verb), green fleet -> hold, and each positive
// witness->verb mapping. Every case also re-runs Decide to pin determinism.
func TestDecideTable(t *testing.T) {
	full := Envelope{Widths: fullWidths(t)}
	zero := Envelope{Widths: map[ActionKind]int{}}
	collapsed := Envelope{Widths: fullWidths(t), WitnessRefused: true}

	deadWorker := func() SupervisorInput { return withWorkerState(presentInput(), "RID-1", WorkerDead) }
	staleWorker := func() SupervisorInput { return withWorkerState(presentInput(), "RID-1", WorkerStale) }
	blockedWorker := func() SupervisorInput {
		return withWorker(presentInput(), WorkerVerdict{RunID: "RID-2", Issue: "9001", Lane: "ops", State: WorkerBlocked})
	}
	doneOwnerStalled := func() SupervisorInput {
		return withStalledLiveness(withWorkerState(presentInput(), "RID-1", WorkerDone))
	}

	cases := []struct {
		name       string
		in         SupervisorInput
		env        Envelope
		mode       EnvelopeMode
		wantVerb   ActionKind
		wantReason Reason
		wantAction SupervisorAction
	}{
		// Rung 1: any withheld witness escalates, before every other rung.
		{"absent_liveness_escalates", withholdLiveness(presentInput()), full, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},
		{"absent_workers_escalates", withholdWorkers(presentInput()), full, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},
		{"absent_escalations_escalates", withholdEscalations(presentInput()), full, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},
		{"absent_leases_escalates", withholdLeases(presentInput()), full, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},
		{"absent_in_warn_mode_still_escalates", withholdLeases(presentInput()), full, ModeWarn, ActionEscalate, ReasonWitnessAbsent, absentHead},
		// Rung 1 kills every verb path that would otherwise fire.
		{"absent_kills_replace_path", withholdLeases(deadWorker()), full, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},
		{"absent_kills_replan_path", withholdEscalations(doneOwnerStalled()), full, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},
		{"absent_kills_escalate_path", withholdLiveness(blockedWorker()), full, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},
		{"absent_kills_hold_path", withholdWorkers(presentInput()), zero, ModeEnforce, ActionEscalate, ReasonWitnessAbsent, absentHead},

		// Rung 2: the envelope gates the proposed verb.
		{"enforce_sub_envelope_refuses_proposal", deadWorker(), zero, ModeEnforce, ActionReplace, ReasonEnvelopeRefused, ReplaceAction{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent"}},
		{"witness_refused_collapses_to_escalate", deadWorker(), collapsed, ModeEnforce, ActionEscalate, ReasonEnvelopeRefused,
			EscalateAction{Class: "envelope_refused", Severity: "operator", ReasonCode: "WITNESS_REFUSED"}},
		{"warn_sub_envelope_executes_with_positive_reason", deadWorker(), zero, ModeWarn, ActionReplace, ReasonReplaceStaleLease,
			ReplaceAction{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent"}},
		{"enforce_zero_widths_hold_stays_inside", presentInput(), zero, ModeEnforce, ActionHold, ReasonHoldGreen, HoldAction{}},

		// Rung 3: a green fleet holds.
		{"green_fleet_holds", presentInput(), full, ModeEnforce, ActionHold, ReasonHoldGreen, HoldAction{}},
		{"empty_census_is_not_pressure", withEmptyWorkers(presentInput()), full, ModeEnforce, ActionHold, ReasonHoldGreen, HoldAction{}},

		// Rung 4: the strongest typed witness maps to its verb.
		{"dead_worker_replaces", deadWorker(), full, ModeEnforce, ActionReplace, ReasonReplaceStaleLease,
			ReplaceAction{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent"}},
		{"stale_worker_replaces", staleWorker(), full, ModeEnforce, ActionReplace, ReasonReplaceStaleLease,
			ReplaceAction{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent"}},
		{"blocked_worker_escalates", blockedWorker(), full, ModeEnforce, ActionEscalate, ReasonEscalateResidual,
			EscalateAction{RunID: "RID-2", Issue: "9001", Class: "worker_blocked", Severity: "operator", ReasonCode: "worker_blocked"}},
		{"stalled_done_owner_replans", doneOwnerStalled(), full, ModeEnforce, ActionRedispatch, ReasonReplanDrift,
			RedispatchAction{Issue: "4480", Lane: "supervisoragent"}},
		{"stalled_active_owner_escalates", withStalledLiveness(presentInput()), full, ModeEnforce, ActionEscalate, ReasonEscalateResidual,
			EscalateAction{RunID: "RID-1", Class: "run_stalled", Severity: "operator", ReasonCode: "run_stalled"}},
		{"stalled_unowned_run_escalates", withLiveness(presentInput(), "RID-9", stalledClass), full, ModeEnforce, ActionEscalate, ReasonEscalateResidual,
			EscalateAction{RunID: "RID-9", Class: "run_stalled", Severity: "operator", ReasonCode: "run_stalled"}},
		{"strongest_witness_wins", withLiveness(withWorker(withWorkerState(presentInput(), "RID-1", WorkerDead),
			WorkerVerdict{RunID: "RID-9", Issue: "9999", Lane: "batch", State: WorkerDone}), "RID-9", stalledClass),
			full, ModeEnforce, ActionReplace, ReasonReplaceStaleLease,
			ReplaceAction{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, gotReason := Decide(tc.in, tc.env, tc.mode)
			again, againReason := Decide(tc.in, tc.env, tc.mode)
			if !reflect.DeepEqual(got, again) || gotReason != againReason {
				t.Fatalf("Decide is not deterministic: (%#v, %v) vs (%#v, %v)", got, gotReason, again, againReason)
			}
			if got.Kind() != tc.wantVerb {
				t.Errorf("verb = %v; want %v", got.Kind(), tc.wantVerb)
			}
			if gotReason != tc.wantReason {
				t.Errorf("reason = %v; want %v", gotReason, tc.wantReason)
			}
			if tc.wantAction != nil && !reflect.DeepEqual(got, tc.wantAction) {
				t.Errorf("action = %#v; want %#v", got, tc.wantAction)
			}
		})
	}
}
