package ultracodebench

import (
	"encoding/json"
	"os"
	"testing"
)

func TestActivationReceiptNeedsExplicitObservation(t *testing.T) {
	r, err := BeforeSpawn(BeforeSpawnInput{
		RunID: "run-1", ChildID: "worker-1", Harness: "codex",
		Requested: SettingAuto, Resolved: SettingOn, Injected: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Observable != ObservableUnknown || r.State() != ActivationUnknown {
		t.Fatalf("pre-spawn receipt inferred activation: %+v", r)
	}
	active, err := Acknowledge(r, ObservableActive, SourceExplicitAcknowledgement)
	if err != nil {
		t.Fatal(err)
	}
	if active.State() != ActivationActive || active.ObservationSource != SourceExplicitAcknowledgement {
		t.Fatalf("acknowledged receipt = %+v", active)
	}
	if _, err := Acknowledge(r, ObservableActive, "process_started"); err == nil {
		t.Fatal("process start was accepted as activation evidence")
	}
}

func TestActivationFixtureCoversRequiredStatesAndHarnesses(t *testing.T) {
	raw, err := os.ReadFile("testdata/activation_states.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema   string              `json:"schema"`
		Receipts []ActivationReceipt `json:"receipts"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != "fak.ultracode_activation_fixture.v1" {
		t.Fatalf("fixture schema=%q", fixture.Schema)
	}
	coverage, err := SummarizeActivation(fixture.Receipts)
	if err != nil {
		t.Fatal(err)
	}
	if coverage.Total != 6 || coverage.Active != 2 || coverage.Inactive != 2 || coverage.Degraded != 1 || coverage.Unknown != 1 || coverage.Verified != 4 {
		t.Fatalf("coverage=%+v", coverage)
	}
	harnesses := map[string]bool{}
	for _, row := range coverage.Children {
		harnesses[row.Harness] = true
	}
	if !harnesses["claude"] || !harnesses["codex"] {
		t.Fatalf("fixture harnesses=%v", harnesses)
	}
}

func TestActivationDecodeRejectsPrivateOrRawFields(t *testing.T) {
	for _, field := range []string{"path", "prompt", "account", "host", "raw_settings", "argv"} {
		raw := []byte(`{"schema":"fak.ultracode_activation.v1","run_id":"r","child_id":"c","harness":"codex","requested":"on","resolved":"on","injected":true,"observable":"unknown","` + field + `":"secret"}`)
		if _, err := DecodeActivation(raw); err == nil {
			t.Fatalf("private field %q accepted", field)
		}
	}
}

func TestActivationClassificationDistinguishesOffAndDegraded(t *testing.T) {
	off, err := BeforeSpawn(BeforeSpawnInput{RunID: "r", ChildID: "off", Harness: "claude", Requested: SettingOff, Resolved: SettingOff})
	if err != nil || off.State() != ActivationInactive {
		t.Fatalf("off=%+v err=%v", off, err)
	}
	degraded, err := BeforeSpawn(BeforeSpawnInput{RunID: "r", ChildID: "unsupported", Harness: "codex", Requested: SettingOn, Resolved: SettingOn, Degradations: []string{"harness_cannot_inject"}})
	if err != nil || degraded.State() != ActivationDegraded {
		t.Fatalf("degraded=%+v err=%v", degraded, err)
	}
}
