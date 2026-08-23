package session

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func readOnlyProjection() ProjectionManifest {
	return ProjectionManifest{
		Schema: ProjectionManifestSchema, Name: "read-only notification", CorpusSchema: CapabilityCorpusSchema,
		SupportedActions: []string{"observe", "replay"},
		Omissions: []ProjectionOmission{
			{Action: "input", TypedReason: "READ_ONLY_SURFACE", Handoff: "fak://session/full"},
			{Action: "decision", TypedReason: "INTERACTION_NOT_RENDERED", Handoff: "fak://session/full"},
			{Action: "move", TypedReason: "PLACEMENT_CONTROL_NOT_RENDERED", Handoff: "fak://session/full"},
		},
		FullClientHandoff: "fak://session/full",
	}
}

func TestProjectionManifestGoldenReadOnly(t *testing.T) {
	canonical := []string{"observe", "replay", "input", "decision", "move"}
	coverage, err := ValidateProjectionManifest(readOnlyProjection(), CapabilityCorpusSchema, canonical)
	if err != nil {
		t.Fatal(err)
	}

	text := coverage.RenderText()
	html := coverage.RenderHTML()
	for _, want := range []string{"Reduced projection: read-only notification", "observe: available", "input: unavailable (READ_ONLY_SURFACE)", "Full client: fak://session/full"} {
		if !strings.Contains(text, want) {
			t.Errorf("text render missing %q:\n%s", want, text)
		}
	}
	for _, want := range []string{"Reduced projection: read-only notification", `data-action="observe">available`, "READ_ONLY_SURFACE", `href="fak://session/full"`} {
		if !strings.Contains(html, want) {
			t.Errorf("HTML render missing %q:\n%s", want, html)
		}
	}
	wantLabels := map[string]string{"session_client_kind": "projection", "session_projection": "read-only notification", "session_projection_corpus": CapabilityCorpusSchema}
	if got := coverage.TelemetryLabels(); !reflect.DeepEqual(got, wantLabels) {
		t.Fatalf("telemetry labels = %#v, want %#v", got, wantLabels)
	}
}

func TestProjectionManifestFailsClosedOnCorpusDrift(t *testing.T) {
	manifest := readOnlyProjection()
	if _, err := ValidateProjectionManifest(manifest, "fak.session-capability-corpus.v2", []string{"observe"}); err == nil {
		t.Fatal("unknown corpus revision passed")
	}
	if _, err := ValidateProjectionManifest(manifest, CapabilityCorpusSchema, []string{"observe", "replay", "input", "decision", "move", "synthetic_new_action"}); err == nil || !strings.Contains(err.Error(), "synthetic_new_action") {
		t.Fatalf("new action error = %v", err)
	}
}

func TestProjectionManifestRequiresTypedOmissionAndHandoff(t *testing.T) {
	manifest := readOnlyProjection()
	manifest.Omissions[0].TypedReason = ""
	if _, err := ValidateProjectionManifest(manifest, CapabilityCorpusSchema, []string{"observe", "replay", "input", "decision", "move"}); err == nil {
		t.Fatal("untyped omission passed")
	}
	manifest = readOnlyProjection()
	manifest.FullClientHandoff = ""
	if _, err := ValidateProjectionManifest(manifest, CapabilityCorpusSchema, []string{"observe", "replay", "input", "decision", "move"}); err == nil {
		t.Fatal("missing full-client handoff passed")
	}
}

func ExampleValidateProjectionManifest() {
	manifest := readOnlyProjection()
	coverage, err := ValidateProjectionManifest(manifest, CapabilityCorpusSchema, []string{"observe", "replay", "input", "decision", "move"})
	if err != nil {
		panic(err)
	}
	fmt.Print(coverage.RenderText())
	// Output:
	// Reduced projection: read-only notification
	// Full client: fak://session/full
	// observe: available
	// replay: available
	// decision: unavailable (INTERACTION_NOT_RENDERED); handoff=fak://session/full
	// input: unavailable (READ_ONLY_SURFACE); handoff=fak://session/full
	// move: unavailable (PLACEMENT_CONTROL_NOT_RENDERED); handoff=fak://session/full
}
