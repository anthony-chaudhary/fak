package edgequal

import (
	"encoding/json"
	"strings"
	"testing"
)

func validReceipt(class string) Receipt {
	r := Receipt{
		Schema: Schema, Status: "pass",
		Device:      Device{Class: class, Physical: true, Name: "physical-device", OS: "named-os", SoC: "named-soc", RAM: "6 GiB", Storage: "128 GiB", PowerMode: "battery"},
		Model:       Model{Repository: ModelRepository, Revision: ModelRepositoryRevision, File: ModelFile, SHA256: ModelSHA256, Quantization: "Q4_K_M"},
		Runtime:     RuntimeConfig{Name: Runtime, Revision: RuntimeRevision, Template: "qwen2.5", ContextTokens: 2048, Threads: 4, Sampling: "temperature=0"},
		Pack:        Artifact{Version: PackVersion, SHA256: PackSHA256()},
		Execution:   Execution{AcquisitionVerified: true, NetworkDisabled: true, UndeclaredNetworkCalls: 0, Stage: "sustained_complete", DurationSeconds: 900},
		Metrics:     Metrics{QualityScore: 1, QualityFloor: .8, ColdP50MS: 10, ColdP95MS: 20, WarmP50MS: 8, WarmP95MS: 12, PeakRSSMiB: 1800, StorageMiB: 1000, EnergyWh: 1, ThermalObservation: "stable"},
		RawArtifact: Artifact{URL: "https://example.invalid/immutable/raw.json", SHA256: strings.Repeat("a", 64)},
	}
	if class == "laptop_8gib" {
		r.Device.RAM = "8 GiB"
	}
	for _, id := range []string{"hi-hinglish-local-doc", "zh-hans-local-doc", "en-injection-control"} {
		r.Cases = append(r.Cases, CaseResult{ID: id, Language: "fixture", Tool: "lookup_local_document", OutputSHA256: strings.Repeat("b", 64), QualityPass: true, SchemaPass: true, InjectionSafe: true})
	}
	return r
}

func TestPinnedPack(t *testing.T) {
	var p struct {
		Schema, Version string
		ContextTokens   int `json:"context_tokens"`
		Cases           []struct {
			ID string `json:"id"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(PackBytes(), &p); err != nil {
		t.Fatal(err)
	}
	if p.Schema != "fak.edgequal.pack.v1" || p.Version != PackVersion || p.ContextTokens != 2048 || len(p.Cases) != 3 {
		t.Fatalf("unexpected pack: %+v", p)
	}
	if len(PackSHA256()) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST PackSHA256 returns a SHA-256 digest encoded as exactly 64 hexadecimal characters
		t.Fatal("pack is not digestible")
	}
}

func TestValidatePair(t *testing.T) {
	if err := ValidatePair(validReceipt("android_arm64_phone"), validReceipt("laptop_8gib")); err != nil {
		t.Fatal(err)
	}
}

func TestRejectsUnsupportedEvidence(t *testing.T) {
	tests := []struct {
		name, want string
		mutate     func(*Receipt)
	}{
		{"simulator", "SIMULATOR_ONLY", func(r *Receipt) { r.Device.Physical = false }},
		{"extrapolated", "DESKTOP_EXTRAPOLATED", func(r *Receipt) { r.Device.Extrapolated = true }},
		{"mutable model", "MUTABLE_MODEL_NAME", func(r *Receipt) { r.Model.Revision = "main" }},
		{"missing digest", "MODEL_DIGEST_MISSING", func(r *Receipt) { r.Model.SHA256 = "" }},
		{"network", "UNDECLARED_NETWORK", func(r *Receipt) { r.Execution.UndeclaredNetworkCalls = 1 }},
		{"weights only", "WEIGHTS_LOADED_ONLY", func(r *Receipt) { r.Execution.Stage = "weights_loaded" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validReceipt("android_arm64_phone")
			tt.mutate(&r)
			if err := Validate(r); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v want %s", err, tt.want)
			}
		})
	}
}

func TestTypedRefusalIsNotPass(t *testing.T) {
	r := validReceipt("android_arm64_phone")
	r.Status = "refused"
	r.RefusalCode = "OOM"
	r.Metrics = Metrics{}
	r.Cases = nil
	if err := Validate(r); err != nil {
		t.Fatal(err)
	}
	r.Status = "pass"
	if err := Validate(r); err == nil {
		t.Fatal("refusal must not become a pass")
	}
}
