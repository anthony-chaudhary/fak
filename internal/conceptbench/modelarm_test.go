// modelarm_test.go — the #2731 acceptance witness: >=2 model ids drive through
// the registry to a transcript (recorded transports — no key/GPU), an unknown
// id refuses with the typed model_unknown class, and a walled-model fixture is
// CLASSIFIED via the shared sessionsignals vocabulary, never scored as a
// concept failure.
package conceptbench

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func armTask() Task {
	return Task{
		Schema:          TaskSchemaV1,
		ID:              "arm-smoke-1",
		Concept:         ConceptCommitStamp,
		Prompt:          "commit the fix with a witnessed stamp",
		FixtureRef:      "arm-smoke-1.fixture.json",
		ExpectedWitness: WitnessDosVerify,
		Difficulty:      DifficultyEasy,
	}
}

// recorded returns a replay Transport that answers every model with a canned
// output — the "recorded transcript" the issue's DoD accepts in place of a
// live key/GPU call.
func recorded(output, errText string) Transport {
	return func(model, prompt string) (string, string, error) {
		return output, errText, nil
	}
}

// TestRegistryDrivesTwoModelsToTranscript is the acceptance gate: two model
// ids — one gateway, one in-kernel serve — resolve through the SAME registry
// to transcripts carrying per-run {model, arm, source} provenance.
func TestRegistryDrivesTwoModelsToTranscript(t *testing.T) {
	r := NewRegistry()
	r.Bind(ArmGateway, recorded("gateway says: committed (fak gateway)", ""), true)
	r.Bind(ArmServe, recorded("serve says: committed (fak serve)", ""), true)

	task := armTask()
	want := []struct {
		model  string
		arm    ArmKind
		source string
	}{
		{"claude-sonnet-5", ArmGateway, "internal/gateway"},
		{"qwen3.6", ArmServe, "fak serve --gguf"},
	}
	for _, w := range want {
		tr, err := r.Drive(w.model, task)
		if err != nil {
			t.Fatalf("Drive(%s): %v", w.model, err)
		}
		if tr.Schema != TranscriptSchemaV1 {
			t.Errorf("Drive(%s) schema = %q, want %q", w.model, tr.Schema, TranscriptSchemaV1)
		}
		if tr.TaskID != task.ID || tr.Output == "" {
			t.Errorf("Drive(%s) = task %q output %q, want task %q with non-empty output", w.model, tr.TaskID, tr.Output, task.ID)
		}
		if tr.Provenance != (Provenance{Model: w.model, Arm: w.arm, Source: w.source}) {
			t.Errorf("Drive(%s) provenance = %+v, want {%s %s %s}", w.model, tr.Provenance, w.model, w.arm, w.source)
		}
		if !tr.Scoreable() {
			t.Errorf("Drive(%s) not scoreable (signal class %q), want scoreable", w.model, tr.SignalClass)
		}
		if !tr.Replay {
			t.Errorf("Drive(%s) Replay = false, want true (recorded transport must be labeled)", w.model)
		}
	}
}

// TestUnknownModelIsTypedModelUnknown pins the DoD's unknown-id rule: the
// refusal is the SAME typed class Layer-2 re-dispatch acts on
// (dispatchtick.ModelSwitchableReason), not free text.
func TestUnknownModelIsTypedModelUnknown(t *testing.T) {
	r := NewRegistry()
	_, err := r.Resolve("gpt-9-nano")
	var ae *ArmError
	if !errors.As(err, &ae) {
		t.Fatalf("Resolve(gpt-9-nano) err = %v, want *ArmError", err)
	}
	if ae.Class != "model_unknown" {
		t.Fatalf("Resolve(gpt-9-nano) class = %q, want the typed model_unknown (== dispatchtick.NoCommitModelUnknown)", ae.Class)
	}
	if !dispatchtick.ModelSwitchableReason(ae.Class) {
		t.Fatalf("ModelSwitchableReason(%q) = false, want true (36185e28 alignment)", ae.Class)
	}
}

// TestUnboundArmIsGated pins the live-call fence: a known model whose arm has
// no bound transport refuses arm_gated instead of silently calling out.
func TestUnboundArmIsGated(t *testing.T) {
	r := NewRegistry()
	_, err := r.Resolve("claude-opus-4-8")
	var ae *ArmError
	if !errors.As(err, &ae) || ae.Class != ArmErrGated {
		t.Fatalf("Resolve on unbound arm err = %v, want *ArmError{Class: arm_gated}", err)
	}
}

// TestWalledModelClassifiedNotScored is the walled-model fixture: a run whose
// error channel carries a usage-limit banner is recorded as its sessionsignals
// class (usage_cap), excluded from headline scoring, and never poisons or
// backs the report's result claim.
func TestWalledModelClassifiedNotScored(t *testing.T) {
	r := NewRegistry()
	r.Bind(ArmGateway, recorded("", "Fable 5 limit · resets 6pm (America/Los_Angeles)"), false)
	r.Bind(ArmServe, recorded("", "no access to model glm-5.2"), false)

	walled, err := r.Drive("fable", armTask())
	if err != nil {
		t.Fatalf("Drive(fable): %v", err)
	}
	if walled.SignalClass != dispatchtick.NoCommitUsageCap {
		t.Fatalf("walled SignalClass = %q, want usage_cap", walled.SignalClass)
	}
	if walled.Scoreable() {
		t.Fatal("walled transcript is scoreable, want classified-not-scored")
	}

	unentitled, err := r.Drive("glm-5.2", armTask())
	if err != nil {
		t.Fatalf("Drive(glm-5.2): %v", err)
	}
	if unentitled.SignalClass != dispatchtick.NoCommitModelUnknown {
		t.Fatalf("unentitled SignalClass = %q, want model_unknown", unentitled.SignalClass)
	}

	// Into the report: one measured row + the walled row. The walled row must
	// neither count as a concept failure nor refuse the honesty gate.
	measured := ReportRow{
		Concept: ConceptCommitStamp, Pass: true,
		WitnessSource: WitnessDosVerify, FidelityRate: 1.0, Evidence: "diff-witnessed",
	}
	measured.Model, measured.Arm, measured.ModelSource = "claude-sonnet-5", string(ArmGateway), "internal/gateway"
	walledRow := walled.StampProvenance(ReportRow{Concept: ConceptCommitStamp})

	if walledRow.Model != "fable" || walledRow.Arm != string(ArmGateway) || walledRow.ModelSource != "internal/gateway" {
		t.Fatalf("StampProvenance row = {%s %s %s}, want {fable gateway internal/gateway}", walledRow.Model, walledRow.Arm, walledRow.ModelSource)
	}
	rep := BuildReport("2026-07-11T00:00:00Z", []ReportRow{measured, walledRow})
	if !rep.ResultClaimAllowed {
		t.Fatalf("walled row refused the result claim (gate: %s), want classified-and-allowed", rep.HonestyGate.Reason)
	}
	if rep.HonestyGate.ClassifiedRows != 1 || rep.HonestyGate.HeadlineRows != 1 {
		t.Fatalf("gate = %+v, want 1 classified + 1 headline", rep.HonestyGate)
	}
	for _, roll := range rep.Rollup {
		if roll.Model != "fable" {
			continue
		}
		if roll.Total != 0 || roll.Pass != 0 {
			t.Fatalf("walled model rollup = %d/%d, want 0/0 (classified, not scored)", roll.Pass, roll.Total)
		}
		if roll.NoCommitReasons[dispatchtick.NoCommitUsageCap] != 1 {
			t.Fatalf("walled model reasons = %v, want usage_cap×1", roll.NoCommitReasons)
		}
	}
}
