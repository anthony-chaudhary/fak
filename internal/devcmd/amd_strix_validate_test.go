package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func TestRunAMDStrixValidate_UnknownSubkernel_JSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := []string{"--subkernels=invalid_kernel", "--ablate=none", "--json", "--timeout=2"}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("RunAMDStrixValidate exit code = %d, want 1", code)
	}

	outStr := stdout.String()
	errStr := stderr.String()

	if !strings.Contains(errStr, "invalid_kernel") {
		t.Errorf("stderr %q should mention invalid_kernel", errStr)
	}

	var receipt amdgpu.StrixValidationReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v\nstdout: %s", err, outStr)
	}

	if receipt.Verdict != "FAIL" {
		t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
	}
	if receipt.Verified {
		t.Errorf("receipt.Verified = true, want false")
	}

	foundErr := false
	for _, f := range receipt.Failures {
		if strings.Contains(f, "invalid_kernel") {
			foundErr = true
			break
		}
	}
	if !foundErr {
		t.Errorf("receipt.Failures %v does not contain invalid_kernel", receipt.Failures)
	}
}

func TestRunAMDStrixValidate_UnknownSubkernel_Human(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := []string{"--subkernels=invalid_kernel", "--ablate=none", "--timeout=2"}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("RunAMDStrixValidate exit code = %d, want 1", code)
	}

	errStr := stderr.String()
	if !strings.Contains(errStr, "invalid_kernel") {
		t.Errorf("stderr %q should mention invalid_kernel", errStr)
	}
}
