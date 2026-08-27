package localappcert

import "testing"

func validMatrix() Matrix {
	e := Envelope{ID: "m3", Chip: "Apple M3 Pro", MemoryBytes: 36 << 30, MacOS: "26.6.2", Power: "AC", Thermal: "nominal", PackRevision: "pack@sha256:x", RuntimeRevision: "r1", Supported: true}
	for _, n := range RequiredScenarios {
		e.Scenarios = append(e.Scenarios, Scenario{Name: n, Status: StatusPass, Evidence: "witness/" + n + ".json", Receipt: &Receipt{Engine: "fak-native", Fallback: "none", Artifact: "sha256:model", RuntimeRevision: "r1"}})
	}
	return Matrix{Schema: Schema, GeneratedAt: "2026-08-27T00:00:00Z", Envelopes: []Envelope{e, {ID: "m1-8g", Chip: "Apple M1", MemoryBytes: 8 << 30, MacOS: "26.6.2", Power: "battery", Thermal: "nominal", PackRevision: "pack@sha256:x", RuntimeRevision: "r1", Supported: false, Reason: "INSUFFICIENT_UNIFIED_MEMORY"}}}
}
func TestValidateCompleteMatrix(t *testing.T) {
	if err := Validate(validMatrix()); err != nil {
		t.Fatal(err)
	}
}
func TestValidateRejectsMissingScenario(t *testing.T) {
	m := validMatrix()
	m.Envelopes[0].Scenarios = m.Envelopes[0].Scenarios[:len(m.Envelopes[0].Scenarios)-1]
	if err := Validate(m); err == nil {
		t.Fatal("accepted missing scenario")
	}
}
func TestValidateRejectsFallback(t *testing.T) {
	m := validMatrix()
	m.Envelopes[0].Scenarios[0].Receipt.Fallback = "llama.cpp"
	if err := Validate(m); err == nil {
		t.Fatal("accepted fallback")
	}
}
func TestValidateRejectsUnprovenUnsupported(t *testing.T) {
	m := validMatrix()
	m.Envelopes[1].Reason = ""
	if err := Validate(m); err == nil {
		t.Fatal("accepted untyped unsupported envelope")
	}
}
