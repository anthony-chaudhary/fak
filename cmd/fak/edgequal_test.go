package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/edgequal"
)

func validTestReceipt(class string) edgequal.Receipt {
	r := edgequal.Receipt{
		Schema: edgequal.Schema, Status: "pass",
		Device:      edgequal.Device{Class: class, Physical: true, Name: "physical-device", OS: "named-os", SoC: "named-soc", RAM: "6 GiB", Storage: "128 GiB", PowerMode: "battery"},
		Model:       edgequal.Model{Repository: edgequal.ModelRepository, Revision: edgequal.ModelRepositoryRevision, File: edgequal.ModelFile, SHA256: edgequal.ModelSHA256, Quantization: "Q4_K_M"},
		Runtime:     edgequal.RuntimeConfig{Name: edgequal.Runtime, Revision: edgequal.RuntimeRevision, Template: "qwen2.5", ContextTokens: 2048, Threads: 4, Sampling: "temperature=0"},
		Pack:        edgequal.Artifact{Version: edgequal.PackVersion, SHA256: edgequal.PackSHA256()},
		Execution:   edgequal.Execution{AcquisitionVerified: true, NetworkDisabled: true, UndeclaredNetworkCalls: 0, Stage: "sustained_complete", DurationSeconds: 900},
		Metrics:     edgequal.Metrics{QualityScore: 1, QualityFloor: .8, ColdP50MS: 10, ColdP95MS: 20, WarmP50MS: 8, WarmP95MS: 12, PeakRSSMiB: 1800, StorageMiB: 1000, EnergyWh: 1, ThermalObservation: "stable"},
		RawArtifact: edgequal.Artifact{URL: "https://example.invalid/immutable/raw.json", SHA256: strings.Repeat("a", 64)},
	}
	if class == "laptop_8gib" {
		r.Device.RAM = "8 GiB"
	}
	for _, id := range []string{"hi-hinglish-local-doc", "zh-hans-local-doc", "en-injection-control"} {
		r.Cases = append(r.Cases, edgequal.CaseResult{ID: id, Language: "fixture", Tool: "lookup_local_document", OutputSHA256: strings.Repeat("b", 64), QualityPass: true, SchemaPass: true, InjectionSafe: true})
	}
	return r
}

func TestEdgequalCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Help / empty usage
	code := runEdgequal(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("runEdgequal with empty args = %d, want 2", code)
	}

	// Pack command
	stdout.Reset()
	stderr.Reset()
	code = runEdgequal(&stdout, &stderr, []string{"pack", "--sha256"})
	if code != 0 {
		t.Fatalf("runEdgequal pack --sha256 = %d, want 0", code)
	}
	if got := strings.TrimSpace(stdout.String()); got != edgequal.PackSHA256() {
		t.Fatalf("pack --sha256 = %q, want %q", got, edgequal.PackSHA256())
	}

	stdout.Reset()
	stderr.Reset()
	code = runEdgequal(&stdout, &stderr, []string{"pack"})
	if code != 0 {
		t.Fatalf("runEdgequal pack = %d, want 0", code)
	}
	var packData map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &packData); err != nil {
		t.Fatalf("pack output is not valid json: %v", err)
	}

	// Validate command
	dir := t.TempDir()
	phoneReceipt := validTestReceipt("android_arm64_phone")
	phoneBytes, err := json.Marshal(phoneReceipt)
	if err != nil {
		t.Fatal(err)
	}
	phonePath := filepath.Join(dir, "phone.json")
	if err := os.WriteFile(phonePath, phoneBytes, 0600); err != nil {
		t.Fatal(err)
	}

	stdout.Reset()
	stderr.Reset()
	code = runEdgequal(&stdout, &stderr, []string{"validate", "--receipt", phonePath})
	if code != 0 {
		t.Fatalf("runEdgequal validate = %d, want 0; stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "VALID: pass") {
		t.Fatalf("runEdgequal validate output = %q, want VALID: pass", stdout.String())
	}

	// Validate pair command
	laptopReceipt := validTestReceipt("laptop_8gib")
	laptopBytes, err := json.Marshal(laptopReceipt)
	if err != nil {
		t.Fatal(err)
	}
	laptopPath := filepath.Join(dir, "laptop.json")
	if err := os.WriteFile(laptopPath, laptopBytes, 0600); err != nil {
		t.Fatal(err)
	}

	stdout.Reset()
	stderr.Reset()
	code = runEdgequal(&stdout, &stderr, []string{"validate-pair", "--phone", phonePath, "--laptop", laptopPath})
	if code != 0 {
		t.Fatalf("runEdgequal validate-pair = %d, want 0; stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "VALID PAIR") {
		t.Fatalf("runEdgequal validate-pair output = %q, want VALID PAIR", stdout.String())
	}
}
