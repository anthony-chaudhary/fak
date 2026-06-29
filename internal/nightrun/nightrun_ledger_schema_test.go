package nightrun

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// nightrun_ledger_schema_test.go covers the ledger schema/enum reconciliation (#1140):
// the widened Outcome vocabulary that admits the real bridge-fold outcomes
// (passed/degraded), the typed RecordRow builder a bridge witness appends THROUGH, and
// the ValidateLedger CI gate that rejects an off-enum outcome or an unregistered task_id.

// TestBridgeOutcomesAreInVocabulary pins that the outcomes the hand-folded GLM/DGX
// witnesses actually wrote (passed/degraded) are now first-class members of the closed
// vocabulary — so the tool can stamp them, not just the run-loop set.
func TestBridgeOutcomesAreInVocabulary(t *testing.T) {
	for _, o := range []Outcome{
		OutcomeCollected, OutcomeFailed, OutcomeTimeout, OutcomeDryRun, OutcomeSkipped,
		OutcomePassed, OutcomeDegraded,
	} {
		if !IsValidOutcome(o) {
			t.Errorf("%q must be a valid ledger outcome", o)
		}
	}
	if IsValidOutcome(Outcome("running-not-performant")) {
		t.Error("a free-text status must NOT be a valid outcome (that is the #1140 defect)")
	}
	// passed is a collected-class datum (it met its bar); degraded ran but missed its bar,
	// so it must NOT mark a datum fresh.
	if !CollectedOutcome(OutcomePassed) {
		t.Error("passed must count as a collected datum (it met its bar)")
	}
	if CollectedOutcome(OutcomeDegraded) {
		t.Error("degraded must NOT count as collected (it ran but missed its bar — a re-measure is still owed)")
	}
}

// TestRecordRowBuildsSchemaValidRow pins that a bridge-fold payload becomes a schema-valid
// row through RecordRow: the schema tag, date, and generated_at are stamped, the Value
// defaults from the registered Task, and a passed/degraded outcome is accepted.
func TestRecordRowBuildsSchemaValidRow(t *testing.T) {
	now := mustTime(t, "2026-06-28T11:58:00Z")
	registered := Task{ID: "witness-glm52-cuda-llamacpp-comparison", Value: ValueFrontier}
	in := RecordInput{
		TaskID:   registered.ID,
		Box:      "box-sm80-dc",
		Command:  "llama-bench -m glm.gguf -ngl 99",
		Outcome:  OutcomePassed,
		Number:   "0.174 tok/s",
		Note:     "bridge-folded GLM-5.2 decode witness",
		Duration: 271750 * time.Millisecond,
	}
	row, err := RecordRow(in, registered, now)
	if err != nil {
		t.Fatalf("RecordRow on a valid passed datum: %v", err)
	}
	if row.Schema != CollectSchema {
		t.Errorf("schema = %q, want %q", row.Schema, CollectSchema)
	}
	if row.Date != "2026-06-28" || row.GeneratedAt != "2026-06-28T11:58:00Z" {
		t.Errorf("date/generated_at not stamped from now: %+v", row)
	}
	if row.Value != string(ValueFrontier) {
		t.Errorf("value must default from the registered Task, got %q", row.Value)
	}
	if row.Outcome != string(OutcomePassed) || row.Number != "0.174 tok/s" || row.Note == "" {
		t.Errorf("observed fields not carried through: %+v", row)
	}
	// The built row round-trips through the ledger reader (it parses back as one row).
	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatal(err)
	}
	if got := ParseLedger(line + "\n"); len(got) != 1 || got[0].TaskID != registered.ID {
		t.Errorf("the recorded row must parse back as one ledger row, got %+v", got)
	}
}

// TestRecordRowRejectsOffSchemaOutcome pins that an off-vocabulary outcome (a free-text
// bridge status like the historical "running-not-performant") is refused by the builder,
// so a typo can never reach the ledger through the tool.
func TestRecordRowRejectsOffSchemaOutcome(t *testing.T) {
	now := mustTime(t, "2026-06-28T00:00:00Z")
	_, err := RecordRow(RecordInput{TaskID: "witness-x", Outcome: Outcome("running-not-performant")}, Task{}, now)
	if err == nil {
		t.Fatal("RecordRow must reject an off-schema outcome")
	}
	var off *OffSchemaError
	if !errors.As(err, &off) {
		t.Errorf("want an *OffSchemaError, got %T: %v", err, err)
	}
	// An empty task id is also refused.
	if _, err := RecordRow(RecordInput{Outcome: OutcomePassed}, Task{}, now); err == nil {
		t.Error("RecordRow must reject a row with no task_id")
	}
}

// TestValidateLedgerFlagsOffSchemaAndUnregistered pins the CI validator: it flags a row
// whose outcome is off the enum AND a row whose task_id is not a registered backlog Task,
// and passes a clean ledger. These are exactly the #1140 defects (passed/degraded before
// the enum admitted them, and the unregistered nightrun-saturation-tick row).
func TestValidateLedgerFlagsOffSchemaAndUnregistered(t *testing.T) {
	registered := TaskIDSet([]Task{{ID: "bench-ok"}, {ID: "witness-known"}})
	rows := []CollectRow{
		{TaskID: "bench-ok", Outcome: string(OutcomeCollected)},               // clean
		{TaskID: "witness-known", Outcome: string(OutcomePassed)},             // clean (bridge passed, registered)
		{TaskID: "witness-known", Outcome: "running-not-performant"},          // off-schema outcome
		{TaskID: "nightrun-saturation-tick", Outcome: string(OutcomeSkipped)}, // unregistered task id
	}
	defects := ValidateLedger(rows, registered)
	if len(defects) != 2 {
		t.Fatalf("want exactly 2 defects (off-schema outcome + unregistered id), got %d: %+v", len(defects), defects)
	}
	var sawOffSchema, sawUnregistered bool
	for _, d := range defects {
		if d.Line == 3 && strings.Contains(d.Reason, "off-schema") {
			sawOffSchema = true
		}
		if d.Line == 4 && strings.Contains(d.Reason, "not a registered") {
			sawUnregistered = true
		}
	}
	if !sawOffSchema {
		t.Error("the off-schema outcome row (line 3) must be flagged")
	}
	if !sawUnregistered {
		t.Error("the unregistered task_id row (line 4) must be flagged")
	}
}

// TestNewBacklogRowRegistered pins that the #1138 frontier backlog row is registered, so
// once a CUDA box with weights+net is reachable it is a pickable task — and so the ledger
// validator accepts a row recorded against it.
func TestNewBacklogRowRegistered(t *testing.T) {
	var found Task
	for _, task := range witnessTasks() {
		if task.ID == "witness-glm52-cuda-llamacpp-comparison" {
			found = task
		}
	}
	if found.ID == "" {
		t.Fatal("the witness-glm52-cuda-llamacpp-comparison backlog row must be registered")
	}
	if found.Value != ValueFrontier || !found.Manual {
		t.Errorf("the new row must be a Manual frontier witness, got value=%q manual=%v", found.Value, found.Manual)
	}
	wantReq := map[Requirement]bool{ReqCUDA: true, ReqWeights: true, ReqNet: true}
	for _, r := range found.Requires {
		delete(wantReq, r)
	}
	if len(wantReq) != 0 {
		t.Errorf("the new row must require cuda+weights+net, missing %v", wantReq)
	}
	// On a CUDA box with weights+net it is feasible (a pickable, surfaced datum).
	caps := Capabilities{Box: "a100", GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	if ok, why := caps.Satisfies(found); !ok {
		t.Errorf("the new row must be feasible on a cuda+weights+net box, got infeasible: %q", why)
	}
}
