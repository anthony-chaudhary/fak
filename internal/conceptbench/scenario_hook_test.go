package conceptbench

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// mustJSON marshals v to a compact JSON string for use as a model's emitted
// handoff artifact. A marshal failure is a test bug, not a graded outcome.
func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal handoff fixture: %v", err)
	}
	return string(b)
}

// TestHookTasksEndInCleanStop pins the #2737 scope: >=2 tasks, each with a
// prompt, and the corpus exercises BOTH sanctioned clean-stop shapes (a
// next-steps handoff and a no-next-step-reason handoff).
func TestHookTasksEndInCleanStop(t *testing.T) {
	tasks := HookTasks()
	if len(tasks) < 2 {
		t.Fatalf("HookTasks() = %d tasks, want >=2 per the #2737 scope", len(tasks))
	}
	seen := map[string]bool{}
	var sawNextStep, sawNoNextStep bool
	for _, task := range tasks {
		if task.Name == "" || task.Prompt == "" {
			t.Errorf("task %q has an empty name or prompt", task.Name)
		}
		if seen[task.Name] {
			t.Errorf("duplicate task name %q", task.Name)
		}
		seen[task.Name] = true
		if task.NoNextStep {
			sawNoNextStep = true
		} else {
			sawNextStep = true
		}
	}
	if !sawNextStep || !sawNoNextStep {
		t.Errorf("corpus must exercise both clean-stop shapes: next_steps=%v no_next_step=%v", sawNextStep, sawNoNextStep)
	}
}

// TestValidHandoffPassesRealReferee is the positive half of the #2737 acceptance
// witness: the canonical valid handoff — in both clean-stop shapes — clears the
// real taskmgr.ReviewHandoff schema referee and the clean-stop gate.
func TestValidHandoffPassesRealReferee(t *testing.T) {
	for _, withNext := range []bool{true, false} {
		h := ValidHandoff("task_2737", withNext)
		// Sanity: the reference shape must satisfy the real in-tree referee, or
		// the fixture — not the model — is wrong.
		if rev := taskmgr.ReviewHandoff(h); !rev.OK {
			t.Fatalf("ValidHandoff(withNextStep=%v) refused by real referee: %v", withNext, rev.Reasons)
		}
		task := HookTasks()[0]
		if withNext == false {
			task = HookTasks()[1]
		}
		row := GradeHook(task, HookEmission{HandoffJSON: mustJSON(t, h), CleanStop: true})
		if row.Outcome != HookPass || !row.Pass {
			t.Errorf("withNextStep=%v: outcome=%s pass=%v, want a clean pass (evidence: %s)", withNext, row.Outcome, row.Pass, row.Evidence)
		}
		if !row.GateAllows {
			t.Errorf("withNextStep=%v: gate_allows=false, want the clean-stop gate to pass (evidence: %s)", withNext, row.Evidence)
		}
		if row.Score != 1 {
			t.Errorf("withNextStep=%v: score=%v, want 1", withNext, row.Score)
		}
		if row.WitnessSource != WitnessHandoffSchema {
			t.Errorf("witness_source=%q, want %q — the row must name its referee", row.WitnessSource, WitnessHandoffSchema)
		}
		if row.FailureMode != "" {
			t.Errorf("withNextStep=%v: failure_mode=%q on a pass, want empty", withNext, row.FailureMode)
		}
	}
}

// TestGradeHookFailureModes is the negative half of the #2737 acceptance
// witness: a missing handoff, a malformed-JSON handoff, and a missing-witness
// handoff (>=2 malformed cases) EACH fail with the right, distinct mode — never
// collapsed into one generic fail. Missing-witness is graded through the real
// taskmgr.ReviewHandoff referee.
func TestGradeHookFailureModes(t *testing.T) {
	// A schema-valid handoff, then mutated per case to isolate one defect.
	valid := ValidHandoff("task_2737", true)

	missingWitness := valid
	missingWitness.Task.Witness = nil // drops the witnessed-done field only

	unverifiedWitness := valid
	unverifiedWitness.Task.Witness = &taskmgr.WitnessRecord{VerifiedState: taskmgr.VerifiedRefused, Source: "package-test"}

	noCurrentState := valid
	noCurrentState.CurrentState = "" // a schema defect that is NOT a missing witness

	cases := []struct {
		name    string
		em      HookEmission
		outcome HookOutcome
	}{
		{
			name:    "missing_handoff_empty",
			em:      HookEmission{HandoffJSON: "", CleanStop: true},
			outcome: HookMissingHandoff,
		},
		{
			name:    "missing_handoff_whitespace",
			em:      HookEmission{HandoffJSON: "   \n\t ", CleanStop: true},
			outcome: HookMissingHandoff,
		},
		{
			name:    "malformed_json_truncated",
			em:      HookEmission{HandoffJSON: `{"schema":"fak.task-handoff.v1","task":`, CleanStop: true},
			outcome: HookMalformedJSON,
		},
		{
			name:    "malformed_json_prose",
			em:      HookEmission{HandoffJSON: "done, everything is committed and green", CleanStop: true},
			outcome: HookMalformedJSON,
		},
		{
			name:    "missing_witness_nil",
			em:      HookEmission{HandoffJSON: mustJSON(t, missingWitness), CleanStop: true},
			outcome: HookMissingWitness,
		},
		{
			name:    "missing_witness_unverified",
			em:      HookEmission{HandoffJSON: mustJSON(t, unverifiedWitness), CleanStop: true},
			outcome: HookMissingWitness,
		},
		{
			name:    "schema_invalid_no_current_state",
			em:      HookEmission{HandoffJSON: mustJSON(t, noCurrentState), CleanStop: true},
			outcome: HookSchemaInvalid,
		},
		{
			// A schema-valid handoff but the turn was NOT a clean stop: the gate
			// cannot pass, and it is not a missing witness — the distinct
			// catch-all, never mislabeled.
			name:    "schema_valid_but_not_clean_stop",
			em:      HookEmission{HandoffJSON: mustJSON(t, valid), CleanStop: false},
			outcome: HookSchemaInvalid,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			row := GradeHook(HookTasks()[0], tc.em)
			if row.Outcome != tc.outcome {
				t.Errorf("outcome=%s, want %s (evidence: %s)", row.Outcome, tc.outcome, row.Evidence)
			}
			if row.Pass {
				t.Errorf("pass=true on a failure mode (evidence: %s)", row.Evidence)
			}
			if row.Score != 0 {
				t.Errorf("score=%v, want 0 on a failure", row.Score)
			}
			if row.FailureMode != string(tc.outcome) {
				t.Errorf("failure_mode=%q, want %q — the row must record the specific mode", row.FailureMode, tc.outcome)
			}
			if row.GateAllows {
				t.Errorf("gate_allows=true on a failure mode — the clean-stop gate must hold (evidence: %s)", row.Evidence)
			}
			if row.WitnessSource != WitnessHandoffSchema {
				t.Errorf("witness_source=%q, want %q", row.WitnessSource, WitnessHandoffSchema)
			}
			if row.Evidence == "" {
				t.Error("empty evidence — the referee's reading must be auditable")
			}
		})
	}
}

// TestGradeHookMatchesGuardGate pins the consumer relationship with the real
// guard: GradeHook's accept/refuse decision is the SAME schema referee the
// clean-stop gate binds (taskmgr.ReviewHandoff), so a handoff the grader passes
// is one guard_stophook.go would allow the stop for, and one it fails is one the
// gate would hold open.
func TestGradeHookMatchesGuardGate(t *testing.T) {
	for _, tc := range []struct {
		name string
		h    taskmgr.Handoff
	}{
		{"valid_next_step", ValidHandoff("task_gate", true)},
		{"valid_no_next_step", ValidHandoff("task_gate", false)},
		{"missing_witness", func() taskmgr.Handoff {
			h := ValidHandoff("task_gate", true)
			h.Task.Witness = nil
			return h
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gateOK := taskmgr.ReviewHandoff(tc.h).OK
			row := GradeHook(HookTasks()[0], HookEmission{HandoffJSON: mustJSON(t, tc.h), CleanStop: true})
			if row.GateAllows != gateOK {
				t.Errorf("gate_allows=%v but the real referee OK=%v — the grader must track the gate", row.GateAllows, gateOK)
			}
			if row.Pass != gateOK {
				t.Errorf("pass=%v but the real referee OK=%v", row.Pass, gateOK)
			}
		})
	}
}
