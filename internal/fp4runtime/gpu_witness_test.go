package fp4runtime

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestSanctionedGPUEvidence(t *testing.T) {
	if os.Getenv("FAK_FP4_GPU_EVIDENCE") != "1" {
		t.Skip("sanctioned GPU evidence is opt-in; run: " + SanctionedGPUEvidenceCommand)
	}
	runtimeID := requiredEnv(t, "FAK_FP4_RUNTIME_ID")
	runtimeVersion := requiredEnv(t, "FAK_FP4_RUNTIME_VERSION")
	runtimeFile := requiredEnv(t, "FAK_FP4_RUNTIME_FILE")
	expect := Outcome(requiredEnv(t, "FAK_FP4_EXPECT_OUTCOME"))
	if expect != OutcomeDelegate && expect != OutcomeRefuse {
		t.Fatalf("FAK_FP4_EXPECT_OUTCOME=%q, want delegate or refuse", expect)
	}

	probe := exec.Command("nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader")
	probeRaw, err := probe.Output()
	if err != nil {
		t.Fatalf("nvidia-smi hardware probe: %v", err)
	}
	line := strings.TrimSpace(strings.Split(string(probeRaw), "\n")[0])
	fields := strings.Split(line, ",")
	if len(fields) < 3 {
		t.Fatalf("unexpected nvidia-smi output %q", line)
	}
	device := strings.TrimSpace(fields[0])
	architecture, err := computeCapabilityArchitecture(strings.TrimSpace(fields[1]))
	if err != nil {
		t.Fatal(err)
	}
	driver := strings.TrimSpace(fields[2])

	runtimeDigest := fileSHA256(t, runtimeFile)
	artifactDigest := fileSHA256(t, filepath.Join("testdata", "nvfp4-artifact-v1.json"))
	recipeDigest := fileSHA256(t, filepath.Join("testdata", "nvfp4-recipe-v1.json"))
	accumulator := AccumulatorSemantics{
		ID:         AccumulatorFP32BF16RNE,
		Product:    "fp4-e2m1",
		Accumulate: "fp32",
		Output:     "bf16",
		Rounding:   "rne",
	}
	matrix := gpuEvidenceMatrix(runtimeID, runtimeVersion, accumulator)
	request := Request{
		Schema: SchemaV1,
		Artifact: Artifact{
			Pin:           Pin{ID: string(ArtifactNVIDIANVFP4), Version: "1.0", SHA256: artifactDigest},
			ElementFormat: "e2m1",
			ScaleFormat:   "ue4m3",
			BlockSize:     16,
		},
		Recipe:      Pin{ID: "runtime-probed-nvfp4-recipe", Version: "1", SHA256: recipeDigest},
		Runtime:     Pin{ID: runtimeID, Version: runtimeVersion, SHA256: runtimeDigest},
		GPU:         GPU{Vendor: "nvidia", Architecture: architecture, Device: device},
		Accumulator: accumulator,
	}
	evidencePayload := strings.Join([]string{
		string(probeRaw), runtimeID, runtimeVersion, runtimeDigest,
		artifactDigest, recipeDigest, string(accumulator.ID),
	}, "\n")
	evidenceSum := sha256.Sum256([]byte(evidencePayload))
	request.HardwareEvidence = &HardwareEvidence{
		Source:            "sanctioned-gpu:nvidia-smi+runtime-file",
		RunSHA256:         hex.EncodeToString(evidenceSum[:]),
		RuntimeSHA256:     runtimeDigest,
		Architecture:      architecture,
		AccumulatorID:     accumulator.ID,
		DeviceFingerprint: fmt.Sprintf("vendor=nvidia;arch=%s;device=%s;driver=%s", architecture, device, driver),
		Command:           SanctionedGPUEvidenceCommand,
	}

	got := Negotiate(request, matrix)
	t.Logf("FP4_GPU_EVIDENCE outcome=%s reason=%s profile=%s architecture=%s device=%q driver=%s evidence_sha256=%s; compatibility only, no quality or performance claim",
		got.Outcome, got.Reason, got.ProfileID, architecture, device, driver, request.HardwareEvidence.RunSHA256)
	if got.Outcome != expect {
		t.Fatalf("observed %s/%s, want outcome %s", got.Outcome, got.Reason, expect)
	}
	if got.Outcome == OutcomeDelegate && !got.Claims.Hardware.Observed {
		t.Fatalf("positive hardware result did not retain the independent evidence: %#v", got)
	}
}

func gpuEvidenceMatrix(runtimeID, runtimeVersion string, accumulator AccumulatorSemantics) Matrix {
	architectures := []ArchitectureSpec{
		{Vendor: "nvidia", ID: ArchitectureSM80, Class: "ampere"},
		{Vendor: "nvidia", ID: ArchitectureSM86, Class: "ampere"},
		{Vendor: "nvidia", ID: ArchitectureSM89, Class: "ada"},
		{Vendor: "nvidia", ID: ArchitectureSM90, Class: "hopper"},
		{Vendor: "nvidia", ID: ArchitectureSM100, Class: "blackwell"},
		{Vendor: "nvidia", ID: ArchitectureSM103, Class: "blackwell"},
		{Vendor: "nvidia", ID: ArchitectureSM120, Class: "blackwell"},
	}
	matrix := Matrix{
		Schema: MatrixSchemaV1,
		Artifacts: []ArtifactSpec{{
			ID: ArtifactNVIDIANVFP4, Version: "1.0",
			ElementFormat: "e2m1", ScaleFormat: "ue4m3", BlockSize: 16,
		}},
		Runtimes:      []RuntimeSpec{{ID: RuntimeID(runtimeID), Version: runtimeVersion}},
		Architectures: architectures,
		Accumulators:  []AccumulatorSemantics{accumulator},
	}
	for _, arch := range []ArchitectureID{ArchitectureSM100, ArchitectureSM103, ArchitectureSM120} {
		matrix.Profiles = append(matrix.Profiles, Profile{
			ID:         "sanctioned-nvfp4-" + ProfileID(arch),
			ArtifactID: ArtifactNVIDIANVFP4, ArtifactVersion: "1.0",
			RuntimeID: RuntimeID(runtimeID), RuntimeVersion: runtimeVersion,
			Architecture: arch, AccumulatorID: accumulator.ID,
			Mode:      ModeExternal,
			Authority: "runtime-installation-plus-device-probe",
		})
	}
	return matrix
}

func requiredEnv(t *testing.T, name string) string {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		t.Fatalf("%s is required", name)
	}
	return value
}

func fileSHA256(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(name)
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func computeCapabilityArchitecture(value string) (ArchitectureID, error) {
	parts := strings.SplitN(value, ".", 2)
	if len(parts) != 2 {
		return "", fmt.Errorf("compute capability %q is not major.minor", value)
	}
	major, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return "", fmt.Errorf("compute capability %q: %w", value, err)
	}
	minor, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil {
		return "", fmt.Errorf("compute capability %q: %w", value, err)
	}
	return ArchitectureID(fmt.Sprintf("sm_%d%d", major, minor)), nil
}
