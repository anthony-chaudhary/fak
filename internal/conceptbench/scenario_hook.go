// scenario_hook.go — the guard hook-protocol scenario + grader (#2737, epic
// #2721 concept #5). `fak guard` injects a Stop-gate harness protocol into the
// wrapped agent's session (cmd/fak/guard_stophook.go): at a clean stop the agent
// must have written a valid `fak.task-handoff.v1` JSON to `FAK_TASK_HANDOFF_FILE`
// — witnessed-done task state, a current-state line, and either 1-2 next steps
// or an explicit no-next-step reason — or the gate holds the stop open. Weaker
// models plausibly emit nothing, malformed JSON, or a handoff missing the
// witnessed-done field. This scenario measures that compliance.
//
// The whole point of this scenario is that the model's artifact is RAW TEXT: the
// bytes it wrote to the handoff file, exactly as the real gate reads them
// (guard_stophook.go readAndReviewGuardTaskHandoff — os.ReadFile → json.Unmarshal
// → taskmgr.ReviewHandoff). Grading from the raw bytes (never a pre-parsed
// struct) is what lets the row tell the #2737 failure modes apart, which a nil
// *Handoff would collapse into one:
//
//   - missing_handoff — nothing written at the clean stop (empty artifact).
//   - malformed_json  — bytes that are not valid JSON (the exact repair the gate
//     asks for; a nil *Handoff cannot distinguish this from an empty file).
//   - missing_witness — parses and is a handoff, but lacks the witnessed-done
//     field (task.witness / verified_state) the gate requires.
//   - schema_invalid  — parses but the schema referee refuses it for some OTHER
//     reason (bad schema id, no current_state, no next-step-or-reason) — kept
//     distinct so a non-witness defect is never mislabeled missing_witness.
//
// A pass is graded through the REAL in-tree schema referee, taskmgr.ReviewHandoff
// — the identical gate cmd/fak/guard_stophook.go binds — not a recording: the
// row's WitnessSource is fak.task-handoff.v1, and GateAllows mirrors the
// clean-stop gate's own allow decision (exit 0), so "the clean-stop gate would
// pass" is asserted by the same code path the guard runs.
package conceptbench

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// HookOutcome is the bucket for one hook-protocol episode: a clean pass, or
// exactly one of the #2737 named failure modes.
type HookOutcome string

const (
	HookPass           HookOutcome = "pass"
	HookMissingHandoff HookOutcome = "missing_handoff"
	HookMalformedJSON  HookOutcome = "malformed_json"
	HookMissingWitness HookOutcome = "missing_witness"
	HookSchemaInvalid  HookOutcome = "schema_invalid"
)

// Pass reports whether the episode cleared the clean-stop gate. It is all-or-
// nothing: the #2737 DoD requires a schema-valid handoff with a witnessed-done
// field AND either next-steps or a no-next-step reason (exactly ReviewHandoff.OK),
// so any failure mode is a fail.
func (o HookOutcome) Pass() bool { return o == HookPass }

// Score folds the bucket to a per-row score: a gate-clearing handoff is 1, any
// failure mode is 0 (no partial credit — a stop the gate would hold open is not
// partially compliant).
func (o HookOutcome) Score() float64 {
	if o == HookPass {
		return 1
	}
	return 0
}

// HookTask is one task engineered to end in a clean stop, requiring the model to
// write a valid fak.task-handoff.v1 handoff. NoNextStep marks a task whose
// correct handoff carries a no_next_step_reason instead of 1-2 next steps, so
// the corpus exercises BOTH sanctioned clean-stop shapes the gate accepts.
type HookTask struct {
	Name       string
	Prompt     string // the instruction naming the finished work and the required handoff
	NoNextStep bool   // true = the correct handoff ends with a no-next-step reason, not next steps
}

// HookTasks returns the committed task set (>=2 per the #2737 scope): a task
// whose clean stop leaves obvious follow-up work (the handoff carries 1-2 next
// steps) and a task that is genuinely complete with nothing to hand on (the
// handoff carries a no-next-step reason) — both shapes the gate must pass.
func HookTasks() []HookTask {
	return []HookTask{
		{
			Name:       "next_steps",
			Prompt:     "You finished wiring the report column and verified it with the package test. Write a valid fak.task-handoff.v1 record to $FAK_TASK_HANDOFF_FILE (witnessed-done task, current state, and 1-2 next steps for the obvious follow-up) before you stop.",
			NoNextStep: false,
		},
		{
			Name:       "no_next_step",
			Prompt:     "You finished a self-contained one-line doc fix and verified it; there is no reasonable follow-up. Write a valid fak.task-handoff.v1 record to $FAK_TASK_HANDOFF_FILE (witnessed-done task, current state, and an explicit no_next_step_reason) before you stop.",
			NoNextStep: true,
		},
	}
}

// HookEmission is the model's emitted artifact for one episode: the RAW bytes it
// wrote to $FAK_TASK_HANDOFF_FILE, exactly as the gate reads them (never a
// pre-parsed struct — the raw text is what distinguishes missing from malformed
// from schema-invalid). CleanStop records that the turn ended in a clean stop
// (the condition under which the guard runs the handoff gate at all).
type HookEmission struct {
	HandoffJSON string // the raw text written to $FAK_TASK_HANDOFF_FILE ("" = nothing written)
	CleanStop   bool   // the turn ended in a clean stop (the gate only runs here)
}

// HookRow is the scenario's graded row for one (task, emission) episode. It
// names every rung the gate reads (parsed, the schema referee's verdict/reasons,
// whether the gate would allow the stop) and the SPECIFIC failure mode, so a
// localizable signal survives — never collapsed to a bare pass/fail, the #2737
// DoD's recorded distinction.
type HookRow struct {
	Task          string
	CleanStop     bool
	Parsed        bool     // the raw bytes unmarshalled into a taskmgr.Handoff
	ReviewOK      bool     // the schema referee (taskmgr.ReviewHandoff) accepted it
	GateAllows    bool     // the clean-stop gate would allow the stop (exit 0)
	Verdict       string   // the referee's verdict ("ready" | "not_applicable" | "refused")
	Reasons       []string // the referee's refusal reasons, for audit
	FailureMode   string   // "" on pass; one of the named modes otherwise
	Outcome       HookOutcome
	Score         float64
	Pass          bool
	WitnessSource string // always fak.task-handoff.v1 (taskmgr.ReviewHandoff)
	Evidence      string
}

// witnessReasons are the ReviewHandoff refusal reasons that mean the handoff is
// missing its witnessed-done field — the #2737 "missing witness field" mode. A
// nil task.witness reads MISSING_COMPLETION_WITNESS; a witness present but not
// verified_done reads COMPLETION_NOT_VERIFIED. Both are the witnessed-done rung.
var witnessReasons = map[string]bool{
	"MISSING_COMPLETION_WITNESS": true,
	"COMPLETION_NOT_VERIFIED":    true,
}

func hasWitnessReason(reasons []string) bool {
	for _, r := range reasons {
		if witnessReasons[r] {
			return true
		}
	}
	return false
}

// GradeHook grades one episode: read the model's raw handoff artifact exactly as
// the guard does (empty → missing; not-JSON → malformed; else unmarshal and run
// the real taskmgr.ReviewHandoff schema referee), then bucket the FIRST failing
// rung in the precedence the gate refuses: no artifact → not JSON → (parsed) the
// witnessed-done field, then any other schema defect. A test fixture that
// isolates one failure lands in exactly its mode. A pass requires the referee to
// accept the handoff AND the turn to be a clean stop (the gate runs only there).
func GradeHook(task HookTask, em HookEmission) HookRow {
	row := HookRow{Task: task.Name, CleanStop: em.CleanStop, WitnessSource: WitnessHandoffSchema}
	raw := strings.TrimSpace(em.HandoffJSON)

	switch {
	case raw == "":
		row.Outcome = HookMissingHandoff
	default:
		var h taskmgr.Handoff
		if err := json.Unmarshal([]byte(raw), &h); err != nil {
			row.Outcome = HookMalformedJSON
			row.Reasons = []string{"MALFORMED_JSON: " + err.Error()}
		} else {
			row.Parsed = true
			rev := taskmgr.ReviewHandoff(h)
			row.ReviewOK = rev.OK
			row.Verdict = rev.Verdict
			row.Reasons = rev.Reasons
			switch {
			case rev.OK && em.CleanStop:
				row.Outcome = HookPass
			case hasWitnessReason(rev.Reasons):
				row.Outcome = HookMissingWitness
			default:
				// A schema-valid handoff on a NON-clean stop, or any other
				// schema defect: both refuse the gate but are not a missing
				// witness — keep them in the distinct catch-all mode.
				row.Outcome = HookSchemaInvalid
			}
		}
	}

	// The gate allows the stop iff the referee accepted the handoff and the turn
	// was a clean stop — the exact exit-0 condition guard_stophook.go computes.
	row.GateAllows = row.ReviewOK && em.CleanStop
	row.Pass = row.Outcome.Pass()
	row.Score = row.Outcome.Score()
	if row.Outcome != HookPass {
		row.FailureMode = string(row.Outcome)
	}
	row.Evidence = fmt.Sprintf("task=%s clean_stop=%v parsed=%v review_ok=%v gate_allows=%v verdict=%s reasons=[%s] outcome=%s",
		task.Name, em.CleanStop, row.Parsed, row.ReviewOK, row.GateAllows, row.Verdict, strings.Join(row.Reasons, ","), row.Outcome)
	return row
}

// ValidHandoff returns a canonical schema-valid fak.task-handoff.v1 record for
// taskID — the reference shape a passing episode emits. withNextStep chooses the
// clean-stop shape: a single concrete next step, or a no-next-step reason. It is
// the fixture builder for this scenario's tests (and a machine-checkable example
// of a handoff the real taskmgr.ReviewHandoff referee accepts).
func ValidHandoff(taskID string, withNextStep bool) taskmgr.Handoff {
	h := taskmgr.Handoff{
		Schema: taskmgr.SchemaHandoff,
		Task: taskmgr.HandoffTask{
			TaskID:  taskID,
			Title:   "conceptbench hook-protocol episode",
			State:   taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{VerifiedState: taskmgr.VerifiedDone, Source: "package-test", Verdict: "green"},
		},
		CurrentState: "The change is committed on main and the package test is green.",
	}
	if withNextStep {
		h.NextSteps = []taskmgr.HandoffNextStep{{
			Key:    taskID + "-followup",
			Title:  "Fold the new column into the leaderboard rollup",
			Body:   "The report column landed; wire it into the per-model rollup and add a golden-row assertion.",
			Reason: "The column is graded per-episode but not yet summarized per model.",
		}}
	} else {
		h.NoNextStepReason = "Self-contained one-line fix; verified and complete, with no reasonable follow-up."
	}
	return h
}
