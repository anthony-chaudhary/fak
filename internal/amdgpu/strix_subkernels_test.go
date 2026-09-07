package amdgpu

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestFilterSubkernelSpecs_UnknownSelector(t *testing.T) {
	tests := []struct {
		name     string
		selected []string
		wantSub  string
		wantSub2 string
	}{
		{
			name:     "single unknown selector",
			selected: []string{"invalid_kernel"},
			wantSub:  "invalid_kernel",
		},
		{
			name:     "multiple unknown selectors",
			selected: []string{"bad_one", "bad_two"},
			wantSub:  "bad_one",
			wantSub2: "bad_two",
		},
		{
			name:     "mixed valid and unknown",
			selected: []string{"argmax", "unknown_kernel"},
			wantSub:  "unknown_kernel",
		},
		{
			name:     "empty string selector",
			selected: []string{""},
			wantSub:  "unknown subkernel selector",
		},
		{
			name:     "whitespace only selector",
			selected: []string{"   "},
			wantSub:  "unknown subkernel selector",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			specs, err := FilterSubkernelSpecs(tc.selected)
			if err == nil {
				t.Fatalf("expected error for selected=%v, got specs=%+v", tc.selected, specs)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.wantSub)
			}
			if tc.wantSub2 != "" && !strings.Contains(err.Error(), tc.wantSub2) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.wantSub2)
			}
		})
	}
}

func TestFilterSubkernelSpecs_ValidSelectors(t *testing.T) {
	t.Run("nil slice selects all default specs", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != len(DefaultSubkernelSpecs) {
			t.Errorf("got %d specs, want %d", len(specs), len(DefaultSubkernelSpecs))
		}
	})

	t.Run("empty slice selects all default specs", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs([]string{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != len(DefaultSubkernelSpecs) {
			t.Errorf("got %d specs, want %d", len(specs), len(DefaultSubkernelSpecs))
		}
	})

	t.Run("selector 'all' selects all specs", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs([]string{"all"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != len(DefaultSubkernelSpecs) {
			t.Errorf("got %d specs, want %d", len(specs), len(DefaultSubkernelSpecs))
		}
	})

	t.Run("case insensitive 'ALL'", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs([]string{"ALL"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != len(DefaultSubkernelSpecs) {
			t.Errorf("got %d specs, want %d", len(specs), len(DefaultSubkernelSpecs))
		}
	})

	t.Run("by category 'prefill'", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs([]string{"prefill"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != 1 {
			t.Fatalf("got %d specs, want 1", len(specs))
		}
		if specs[0].Name != "qwen35_sequence_prefill" {
			t.Errorf("got spec name %q, want qwen35_sequence_prefill", specs[0].Name)
		}
		if specs[0].Category != "prefill" {
			t.Errorf("got spec category %q, want prefill", specs[0].Category)
		}
	})

	t.Run("by spec name 'qwen35_sequence_prefill'", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs([]string{"qwen35_sequence_prefill"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != 1 {
			t.Fatalf("got %d specs, want 1", len(specs))
		}
		if specs[0].Name != "qwen35_sequence_prefill" {
			t.Errorf("got spec name %q, want qwen35_sequence_prefill", specs[0].Name)
		}
	})

	t.Run("by spec name 'argmax' with whitespace", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs([]string{"  argmax  "})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != 1 || specs[0].Name != "argmax" {
			t.Errorf("unexpected specs: %+v", specs)
		}
	})

	t.Run("deduplicates category and overlapping spec name", func(t *testing.T) {
		specs, err := FilterSubkernelSpecs([]string{"prefill", "qwen35_sequence_prefill"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != 1 {
			t.Errorf("got %d specs, want 1 (should deduplicate)", len(specs))
		}
	})
}

func TestRunSubkernelTests_FailFastOnInvalidSelectors(t *testing.T) {
	ctx := context.Background()
	// target is nil, proving it fails fast before dereferencing target or checking reachability
	results, err := RunSubkernelTests(ctx, nil, []string{"invalid_kernel"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if results != nil {
		t.Errorf("expected nil results, got %+v", results)
	}
	if !strings.Contains(err.Error(), "invalid_kernel") {
		t.Errorf("error %q should name 'invalid_kernel'", err.Error())
	}
}

func TestRunStrixValidation_UnknownSubkernel(t *testing.T) {
	defer func() {
		ClearPresenceCache()
	}()

	target := &StrixTarget{
		Mode:           "ssh",
		Host:           "test-strix-sim",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		LockupTimeout:  -1,
		LatencyMS:      1.0,
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}
	SavePresenceCache(target)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	opts := StrixValidationOpts{
		Host:          "test-strix-sim",
		RunSubkernels: true,
		Subkernels:    []string{"invalid_kernel"},
		RunAblations:  false,
		Command:       "fak-dev amd-strix-validate --subkernels=invalid_kernel",
	}

	receipt, _ := RunStrixValidation(ctx, opts)
	if receipt == nil {
		t.Fatal("expected non-nil receipt")
	}
	if receipt.Verdict != "FAIL" {
		t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
	}
	if receipt.Verified {
		t.Errorf("receipt.Verified = true, want false")
	}
	if len(receipt.Failures) == 0 {
		t.Fatal("expected at least one failure recorded on receipt")
	}

	foundSubkernelErr := false
	for _, f := range receipt.Failures {
		if strings.Contains(f, "invalid_kernel") {
			foundSubkernelErr = true
			break
		}
	}
	if !foundSubkernelErr {
		t.Errorf("receipt.Failures %v does not mention invalid_kernel", receipt.Failures)
	}

	// Validate() should succeed because receipt is a valid FAIL receipt
	if err := receipt.Validate(); err != nil {
		t.Errorf("receipt.Validate() failed: %v", err)
	}
}

func TestSubkernelExecution_SourceBindingMismatchFailsRemoteCommand(t *testing.T) {
	ctx := context.Background()
	ctx = WithSourceBinding(ctx, "0000000000000000000000000000000000000000", "")

	t.Setenv("FAK_STRIX_DIR", ".")
	target := &StrixTarget{
		Host:      "localhost",
		Mode:      "local",
		Reachable: true,
	}

	res := executeOneSubkernel(ctx, target, DefaultSubkernelSpecs[0])
	if res.Status != "FAIL" {
		t.Errorf("res.Status = %q, want FAIL on source binding mismatch", res.Status)
	}
	if !strings.Contains(res.Error, "source binding mismatch") {
		t.Errorf("res.Error %q should mention 'source binding mismatch'", res.Error)
	}
}
