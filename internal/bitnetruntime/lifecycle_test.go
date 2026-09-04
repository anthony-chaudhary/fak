package bitnetruntime

import (
	"testing"
)

// Invariant: BitNet runtime delegation must enforce exact architecture and host feature compatibility.
// Guard: Admit rejects unsupported kernel architectures and mismatched CPU feature sets.

func TestBitnetRuntimeLifecycleValidation(t *testing.T) {
	t.Parallel()

	rt := Runtime{
		Name:    RuntimeName,
		Version: "1.0.1",
		Build:   "prod",
		Kernels: []Kernel{KernelTL2, KernelI2S},
	}
	host := Host{
		OS:       "linux",
		Arch:     "amd64",
		Features: []string{"avx2"},
	}
	model := Model{
		ID:                  "bitnet-1.58b",
		Alphabet:            AlphabetTernary,
		Kernel:              KernelTL2,
		BitsPerWeightStored: 1.67,
	}

	result := Admit(rt, host, model)
	if result.Outcome != OutcomeDelegate {
		t.Fatalf("expected OutcomeDelegate, got %s: %v", result.Outcome, result.Reasons)
	}

	// Unsupported host OS
	badHost := host
	badHost.OS = "unknown_os"
	if res := Admit(rt, badHost, model); res.Outcome != OutcomeUnsupported {
		t.Fatalf("expected unsupported on bad host OS, got %s", res.Outcome)
	}
}
