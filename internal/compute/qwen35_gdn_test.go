package compute

import (
	"errors"
	"strings"
	"testing"
)

func TestQwen35GDNTypedRefusalsCarryStableIdentity(t *testing.T) {
	if Qwen35GDNCUDAPath != "cuda/qwen35-gdn-ssm-decode-v1" {
		t.Fatalf("Qwen35GDNCUDAPath = %q", Qwen35GDNCUDAPath)
	}
	if Qwen35GDNParityCosineMin != 0.999 {
		t.Fatalf("Qwen35GDNParityCosineMin = %v, want 0.999", Qwen35GDNParityCosineMin)
	}

	var err error = &Qwen35GDNGeometryError{Operand: "geometry", Reason: "invalid head grouping"}
	var geometry *Qwen35GDNGeometryError
	if !errors.As(err, &geometry) || !strings.Contains(err.Error(), "invalid head grouping") {
		t.Fatalf("geometry refusal lost its type/detail: %T %v", err, err)
	}
	err = &Qwen35GDNKernelError{Stage: "recurrent-gated-norm", Code: 50001}
	var kernel *Qwen35GDNKernelError
	if !errors.As(err, &kernel) || !strings.Contains(err.Error(), "no CPU fallback") {
		t.Fatalf("kernel refusal lost its type/fail-closed detail: %T %v", err, err)
	}
}
