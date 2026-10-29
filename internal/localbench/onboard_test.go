package localbench

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10s lane=default
func TestOnboard(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)

	args := []string{"onboard", "--machine-id", "test-rtx5090", "--gpu-model", "NVIDIA GeForce RTX 5090", "--vram-gib", "32"}
	var stdout, stderr bytes.Buffer
	if err := RunCLI(args, &stdout, &stderr); err != nil {
		t.Fatalf("first onboard run failed: %v\nstderr=%s", err, stderr.String())
	}

	specsPath := filepath.Join("experiments", "benchmark", "machines", "test-rtx5090", "specs.json")
	data, err := os.ReadFile(specsPath)
	if err != nil {
		t.Fatalf("specs.json not written: %v", err)
	}
	var specs MachineSpecs
	if err := json.Unmarshal(data, &specs); err != nil {
		t.Fatalf("specs.json does not parse: %v", err)
	}
	if specs.Schema != machineSpecsSchema || specs.MachineID != "test-rtx5090" {
		t.Fatalf("schema/machine_id = %q/%q, want %q/test-rtx5090", specs.Schema, specs.MachineID, machineSpecsSchema)
	}
	if len(specs.Hardware.GPU) != 1 {
		t.Fatalf("gpu entries = %d, want 1", len(specs.Hardware.GPU))
	}
	gpu := specs.Hardware.GPU[0]
	if !strings.Contains(strings.ToLower(gpu.Model), "rtx 5090") {
		t.Fatalf("gpu model round-trip = %q", gpu.Model)
	}
	if gpu.MemoryGB != 32 {
		t.Fatalf("vram round-trip = %v GiB, want 32", gpu.MemoryGB)
	}
	if gpu.ComputeCapability != "12.0" {
		t.Fatalf("arch round-trip = %q, want 12.0 (sm_120 default for RTX 5090)", gpu.ComputeCapability)
	}
	if specs.RegisteredAt == "" || specs.Hardware.CPU.Architecture == "" {
		t.Fatalf("required spec fields missing: %+v", specs)
	}
	firstRegisteredAt := specs.RegisteredAt

	receipts, err := filepath.Glob(filepath.Join("experiments", "benchmark", "machines", "test-rtx5090", "onboard-receipts", "onboard-receipt-*.json"))
	if err != nil || len(receipts) != 1 {
		t.Fatalf("receipts after first run = %v (err %v), want exactly 1", receipts, err)
	}
	r, err := readAndVerify(receipts[0])
	if err != nil {
		t.Fatalf("receipt did not verify: %v", err)
	}
	if r.Schema != receiptSchema || r.Benchmark != onboardBenchmark || r.ExitStatus != 0 {
		t.Fatalf("receipt fields = schema %q benchmark %q exit %d", r.Schema, r.Benchmark, r.ExitStatus)
	}
	if r.Engine != "sm_120" {
		t.Fatalf("receipt engine = %q, want sm_120", r.Engine)
	}
	if r.Integrity.SHA256 == "" || r.OutputSHA256 == "" {
		t.Fatalf("receipt integrity/output digests missing: %+v", r)
	}
	firstReceipt, err := os.ReadFile(receipts[0])
	if err != nil {
		t.Fatalf("read first receipt: %v", err)
	}

	// Idempotent re-run: prior receipt untouched byte-for-byte, new version-stamped
	// receipt appended, specs.json still valid with registered_at preserved.
	if err := RunCLI(args, &stdout, &stderr); err != nil {
		t.Fatalf("re-run failed: %v\nstderr=%s", err, stderr.String())
	}
	receipts, err = filepath.Glob(filepath.Join("experiments", "benchmark", "machines", "test-rtx5090", "onboard-receipts", "onboard-receipt-*.json"))
	if err != nil || len(receipts) != 2 {
		t.Fatalf("receipts after re-run = %v (err %v), want 2 (prior receipt preserved)", receipts, err)
	}
	again, err := os.ReadFile(receipts[0])
	if err != nil {
		t.Fatalf("read prior receipt after re-run: %v", err)
	}
	if !bytes.Equal(firstReceipt, again) {
		t.Fatalf("re-run corrupted the prior receipt")
	}
	for _, p := range receipts {
		if _, err := readAndVerify(p); err != nil {
			t.Fatalf("receipt %s does not verify after re-run: %v", p, err)
		}
	}
	data, err = os.ReadFile(specsPath)
	if err != nil {
		t.Fatalf("specs.json missing after re-run: %v", err)
	}
	if err := json.Unmarshal(data, &specs); err != nil {
		t.Fatalf("specs.json does not parse after re-run: %v", err)
	}
	if specs.RegisteredAt != firstRegisteredAt {
		t.Fatalf("registered_at changed on re-run: %q -> %q", firstRegisteredAt, specs.RegisteredAt)
	}

	// JSON summary mode must print valid JSON naming both paths.
	var jsOut bytes.Buffer
	if err := RunCLI(append(append([]string{}, args...), "--json"), &jsOut, &stderr); err != nil {
		t.Fatalf("json onboard run failed: %v\nstderr=%s", err, stderr.String())
	}
	var summary OnboardSummary
	if err := json.Unmarshal(jsOut.Bytes(), &summary); err != nil {
		t.Fatalf("--json output is not valid JSON: %v\n%s", err, jsOut.String())
	}
	if summary.MachineID != "test-rtx5090" || summary.Arch != "sm_120" || summary.VRAMGiB != 32 || summary.SpecsPath == "" || summary.ReceiptPath == "" {
		t.Fatalf("json summary = %+v", summary)
	}
}

func TestOnboardValidation(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	var stdout, stderr bytes.Buffer
	cases := [][]string{
		{"onboard"},                           // missing --machine-id
		{"onboard", "--machine-id", "Bad_ID"}, // slug violation
		{"onboard", "--machine-id", "ok", "--arch", "sm_12"},  // arch shape
		{"onboard", "--machine-id", "ok", "--vram-gib", "-1"}, // negative vram
	}
	for i, args := range cases {
		err := RunCLI(args, &stdout, &stderr)
		if err == nil {
			t.Fatalf("case %d (%v): expected error, got nil", i, args)
		}
	}
	if _, err := os.Stat(filepath.Join("experiments", "benchmark", "machines")); err == nil {
		t.Fatal("failed runs must not create registry directories")
	}
}
