package amdgpu

import (
	"context"
	"errors"
	"fmt"
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

func TestReceiptParityEventFromSubkernelPreservesRegisteredCosineArgmax(t *testing.T) {
	contract, ok := LookupSubkernelParityContract("q4k_matmul")
	if !ok {
		t.Fatal("q4k_matmul contract missing")
	}
	cosine, exact := 0.999, true
	event := StrixSubkernelParityEvent{
		Schema: StrixSubkernelParitySchema, Selector: contract.Selector, TestName: contract.TestName,
		OracleKind: contract.OracleKind, Engine: contract.Engine, DeviceObserved: true, CaseCount: 4, Passed: true,
		Observed: StrixSubkernelObservedMetrics{Cosine: &cosine, ArgmaxExact: &exact},
	}
	got := receiptParityEventFromSubkernel(event, contract)
	if err := got.Validate(); err != nil {
		t.Fatalf("converted event invalid: %v", err)
	}
	if err := validateReceiptEventContract(contract.Selector, got); err != nil {
		t.Fatalf("converted event lost registered contract: %v", err)
	}
	if got.OracleKind != StrixOracleCosineArgmax || got.Observed.CosineSimilarity == nil || got.Observed.ArgmaxExact == nil {
		t.Fatalf("converted event lost cosine+argmax evidence: %+v", got)
	}
}

func TestDefaultSubkernelParityContractsAreExhaustive(t *testing.T) {
	if len(DefaultSubkernelParityContracts) != len(DefaultSubkernelSpecs) {
		t.Fatalf("DefaultSubkernelParityContracts count = %d, want %d", len(DefaultSubkernelParityContracts), len(DefaultSubkernelSpecs))
	}

	seenSpecs := make(map[string]bool)
	for _, spec := range DefaultSubkernelSpecs {
		contract, ok := DefaultSubkernelParityContracts[spec.Name]
		if !ok {
			t.Errorf("DefaultSubkernelParityContracts missing contract for spec %q", spec.Name)
			continue
		}
		seenSpecs[spec.Name] = true

		if contract.Selector != spec.Name {
			t.Errorf("contract.Selector = %q, want %q", contract.Selector, spec.Name)
		}
		if contract.TestName == "" {
			t.Errorf("contract %q has empty TestName", spec.Name)
		}
		if contract.Engine == "" {
			t.Errorf("contract %q has empty Engine", spec.Name)
		}
		if !IsKnownOracleKind(contract.OracleKind) {
			t.Errorf("contract %q has unknown oracle kind %q", spec.Name, contract.OracleKind)
		}

		// Individual contract invariants based on production Vulkan tests
		switch spec.Name {
		case "argmax":
			if contract.OracleKind != OracleExactArgmax {
				t.Errorf("argmax OracleKind = %q, want %q", contract.OracleKind, OracleExactArgmax)
			}
			if !contract.Bounds.RequireArgmaxExact {
				t.Errorf("argmax must require RequireArgmaxExact=true")
			}
			if contract.Bounds.MinCosine != nil {
				t.Errorf("argmax must not declare a MinCosine bound")
			}
			if contract.Bounds.MaxAbsDelta != nil {
				t.Errorf("argmax must not declare a MaxAbsDelta bound")
			}
			if !contract.DeviceObserved {
				t.Errorf("argmax must be device_observed=true")
			}
			if contract.Engine != StrixVulkanEngine {
				t.Errorf("argmax Engine = %q, want %q", contract.Engine, StrixVulkanEngine)
			}

		case "matmul_f32", "matmul2_f32", "matmul3_f32":
			if contract.OracleKind != OracleCosineMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleCosineMaxAbs)
			}
			if contract.Bounds.MinCosine == nil || *contract.Bounds.MinCosine < 0.9999 {
				t.Errorf("%s MinCosine must be >= 0.9999", spec.Name)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 1e-2 {
				t.Errorf("%s MaxAbsDelta must be <= 1e-2", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "q8_matmul", "q8_matmul_wide", "q8_matmul_vocab":
			if contract.OracleKind != OracleCosineMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleCosineMaxAbs)
			}
			if contract.Bounds.MinCosine == nil || *contract.Bounds.MinCosine < 0.9999 {
				t.Errorf("%s MinCosine must be >= 0.9999", spec.Name)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 1e-3 {
				t.Errorf("%s MaxAbsDelta must be <= 1e-3", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "q4k_matmul", "q2k_matmul":
			if contract.OracleKind != OracleCosineArgmax {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleCosineArgmax)
			}
			if contract.Bounds.MinCosine == nil || *contract.Bounds.MinCosine < 0.995 {
				t.Errorf("%s MinCosine must be >= 0.995", spec.Name)
			}
			if !contract.Bounds.RequireArgmaxExact {
				t.Errorf("%s must require RequireArgmaxExact=true", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "rmsnorm", "swiglu":
			if contract.OracleKind != OracleMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleMaxAbs)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 1e-3 {
				t.Errorf("%s MaxAbsDelta must be <= 1e-3", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "rmsnorm_matmul", "rmsnorm_matmul2", "rmsnorm_matmul3":
			if contract.OracleKind != OracleCosineMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleCosineMaxAbs)
			}
			if contract.Bounds.MinCosine == nil || *contract.Bounds.MinCosine < 0.9999 {
				t.Errorf("%s MinCosine must be >= 0.9999", spec.Name)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 1e-2 {
				t.Errorf("%s MaxAbsDelta must be <= 1e-2", spec.Name)
			}
			if !contract.Bounds.RequireSourceMutationCheck || contract.Bounds.MaxSourceDelta == nil || *contract.Bounds.MaxSourceDelta > 0.0 {
				t.Errorf("%s must require source mutation delta <= 0", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "swiglu_matmul_add":
			if contract.OracleKind != OracleCosineMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleCosineMaxAbs)
			}
			if contract.Bounds.MinCosine == nil || *contract.Bounds.MinCosine < 0.9999 {
				t.Errorf("%s MinCosine must be >= 0.9999", spec.Name)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 1e-2 {
				t.Errorf("%s MaxAbsDelta must be <= 1e-2", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "rope":
			if contract.OracleKind != OracleMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleMaxAbs)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 1e-3 {
				t.Errorf("%s MaxAbsDelta must be <= 1e-3", spec.Name)
			}
			if !contract.Bounds.RequireSourceMutationCheck || contract.Bounds.MaxSourceDelta == nil || *contract.Bounds.MaxSourceDelta > 0.0 {
				t.Errorf("%s must require source mutation delta <= 0", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "attention":
			if contract.OracleKind != OracleCosineMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleCosineMaxAbs)
			}
			if contract.Bounds.MinCosine == nil || *contract.Bounds.MinCosine < 0.999 {
				t.Errorf("%s MinCosine must be >= 0.999", spec.Name)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 1e-2 {
				t.Errorf("%s MaxAbsDelta must be <= 1e-2", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "qwen35_gdn_decode":
			if contract.OracleKind != OracleStateContinuity {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleStateContinuity)
			}
			if contract.TestName != "TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace" {
				t.Errorf("%s TestName = %q, want TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace", spec.Name, contract.TestName)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 3e-4 {
				t.Errorf("%s MaxAbsDelta must be <= 3e-4", spec.Name)
			}
			if !contract.Bounds.RequireStateIdentity {
				t.Errorf("%s must require RequireStateIdentity=true", spec.Name)
			}
			if !contract.Bounds.RequireFinite {
				t.Errorf("%s must require RequireFinite=true", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "qwen35_gdn_preprojected":
			if contract.OracleKind != OracleStateContinuity {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleStateContinuity)
			}
			if contract.TestName != "TestVulkanQwen35GDNPreprojectedParityAndStateContinuity" {
				t.Errorf("%s TestName = %q, want TestVulkanQwen35GDNPreprojectedParityAndStateContinuity", spec.Name, contract.TestName)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 2e-4 {
				t.Errorf("%s MaxAbsDelta must be <= 2e-4", spec.Name)
			}
			if !contract.Bounds.RequireStateIdentity {
				t.Errorf("%s must require RequireStateIdentity=true", spec.Name)
			}
			if !contract.Bounds.RequireFinite {
				t.Errorf("%s must require RequireFinite=true", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "qwen35_sequence_prefill":
			if contract.OracleKind != OracleMaxAbs {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleMaxAbs)
			}
			// Must map to device parity test, NOT the geometry sibling
			if contract.TestName != "TestVulkanQwen35SequenceQuantizedPanelsMatchCPU" {
				t.Errorf("%s TestName = %q, want TestVulkanQwen35SequenceQuantizedPanelsMatchCPU (not geometry sibling)", spec.Name, contract.TestName)
			}
			if strings.Contains(contract.TestName, "Geometry") {
				t.Errorf("%s must not map to geometry sibling test", spec.Name)
			}
			if contract.Bounds.MaxAbsDelta == nil || *contract.Bounds.MaxAbsDelta > 2e-3 {
				t.Errorf("%s MaxAbsDelta must be <= 2e-3", spec.Name)
			}
			if !contract.Bounds.RequireFinite {
				t.Errorf("%s must require RequireFinite=true", spec.Name)
			}
			if !contract.DeviceObserved || contract.Engine != StrixVulkanEngine {
				t.Errorf("%s must declare engine=%q and device_observed=true", spec.Name, StrixVulkanEngine)
			}

		case "f16_kv_contiguize":
			if contract.OracleKind != OracleHostContract {
				t.Errorf("%s OracleKind = %q, want %q", spec.Name, contract.OracleKind, OracleHostContract)
			}
			if contract.TestName != "TestRADVContiguizeShader" {
				t.Errorf("%s TestName = %q, want TestRADVContiguizeShader", spec.Name, contract.TestName)
			}
			if contract.DeviceObserved {
				t.Errorf("f16_kv_contiguize must declare device_observed=false (cannot earn physical parity)")
			}
			if contract.Engine == StrixVulkanEngine {
				t.Errorf("f16_kv_contiguize must not claim engine=%q", StrixVulkanEngine)
			}
		}
	}

	// Reverse check: every contract belongs to DefaultSubkernelSpecs
	for name := range DefaultSubkernelParityContracts {
		if !seenSpecs[name] {
			t.Errorf("contract %q in DefaultSubkernelParityContracts has no matching entry in DefaultSubkernelSpecs", name)
		}
	}
}

func TestParseStrixSubkernelParity(t *testing.T) {
	t.Run("ValidEvents", func(t *testing.T) {
		validArgmax := `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":3,"passed":true,"observed":{"argmax_exact":true}}`
		ev, err := ParseStrixSubkernelParity(validArgmax, "argmax")
		if err != nil {
			t.Fatalf("unexpected error parsing valid argmax: %v", err)
		}
		if ev.Selector != "argmax" || !ev.Passed || ev.Observed.ArgmaxExact == nil || !*ev.Observed.ArgmaxExact {
			t.Errorf("unexpected event content: %+v", ev)
		}

		validMatMul := fmt.Sprintf(
			"=== RUN   TestVulkanMatMulApprox\n    vulkan_test.go:539: {\"schema\":\"%s\",\"selector\":\"matmul_f32\",\"test_name\":\"TestVulkanMatMulApprox\",\"oracle_kind\":\"cosine_max_abs\",\"engine\":\"fak-native/vulkan\",\"device_observed\":true,\"case_count\":1,\"passed\":true,\"observed\":{\"cosine\":0.999995,\"max_abs_delta\":0.005}}\n--- PASS: TestVulkanMatMulApprox (0.02s)\nPASS\n",
			StrixSubkernelParitySchema,
		)
		ev, err = ParseStrixSubkernelParity(validMatMul, "matmul_f32")
		if err != nil {
			t.Fatalf("unexpected error parsing valid matmul: %v", err)
		}
		if ev.Observed.Cosine == nil || *ev.Observed.Cosine < 0.9999 {
			t.Errorf("expected cosine >= 0.9999, got %v", ev.Observed.Cosine)
		}

		validQ4K := `{"schema":"fak.strix.subkernel-parity/v1","selector":"q4k_matmul","test_name":"TestVulkanQ4KMatMulMatchesCPUReference","oracle_kind":"cosine_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.998,"argmax_exact":true}}`
		ev, err = ParseStrixSubkernelParity(validQ4K, "q4k_matmul")
		if err != nil {
			t.Fatalf("unexpected error parsing valid q4k: %v", err)
		}
		if ev.Observed.ArgmaxExact == nil || !*ev.Observed.ArgmaxExact {
			t.Errorf("expected argmax_exact=true")
		}

		validGDN := `{"schema":"fak.strix.subkernel-parity/v1","selector":"qwen35_gdn_decode","test_name":"TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace","oracle_kind":"state_continuity","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"max_abs_delta":0.00015,"state_identity":true,"finite_output":true}}`
		ev, err = ParseStrixSubkernelParity(validGDN, "qwen35_gdn_decode")
		if err != nil {
			t.Fatalf("unexpected error parsing valid GDN: %v", err)
		}

		validPrefill := `{"schema":"fak.strix.subkernel-parity/v1","selector":"qwen35_sequence_prefill","test_name":"TestVulkanQwen35SequenceQuantizedPanelsMatchCPU","oracle_kind":"max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":4,"passed":true,"observed":{"max_abs_delta":0.001,"finite_output":true}}`
		ev, err = ParseStrixSubkernelParity(validPrefill, "qwen35_sequence_prefill")
		if err != nil {
			t.Fatalf("unexpected error parsing valid prefill: %v", err)
		}

		validHost := `{"schema":"fak.strix.subkernel-parity/v1","selector":"f16_kv_contiguize","test_name":"TestRADVContiguizeShader","oracle_kind":"host_contract","engine":"fak-native/host","device_observed":false,"case_count":6,"passed":true,"observed":{}}`
		ev, err = ParseStrixSubkernelParity(validHost, "f16_kv_contiguize")
		if err != nil {
			t.Fatalf("unexpected error parsing valid host contract: %v", err)
		}
		if ev.DeviceObserved {
			t.Errorf("host contract must have device_observed=false")
		}

		// Self-resolved contract without explicit expectedSelector argument
		ev, err = ParseStrixSubkernelParity(validArgmax)
		if err != nil {
			t.Fatalf("unexpected error parsing without explicit selector: %v", err)
		}
		if ev.Selector != "argmax" {
			t.Errorf("expected selector argmax, got %q", ev.Selector)
		}

		// LookupSubkernelParityContract case insensitivity
		if _, ok := LookupSubkernelParityContract("ARGMAX"); !ok {
			t.Errorf("expected LookupSubkernelParityContract to match 'ARGMAX' case-insensitively")
		}
		if _, ok := LookupSubkernelParityContract("  q4k_matmul  "); !ok {
			t.Errorf("expected LookupSubkernelParityContract to trim whitespace")
		}
		if _, ok := LookupSubkernelParityContract("nonexistent_selector"); ok {
			t.Errorf("expected LookupSubkernelParityContract to return false for nonexistent selector")
		}
	})

	t.Run("AbsentEventFailsClosed", func(t *testing.T) {
		for _, out := range []string{
			"",
			"--- PASS: TestVulkanArgmaxExact (0.01s)\nPASS\n",
			"=== RUN   TestVulkanMatMulApprox\n--- PASS: TestVulkanMatMulApprox (0.01s)\nok github.com/anthony-chaudhary/fak/internal/compute 0.05s\n",
			"execution completed successfully with 0 errors\n",
		} {
			_, err := ParseStrixSubkernelParity(out, "argmax")
			if err == nil {
				t.Errorf("expected error for absent event in %q, got nil", out)
			}
			if !errors.Is(err, ErrSubkernelParityAbsent) {
				t.Errorf("expected ErrSubkernelParityAbsent, got %v", err)
			}
		}
	})

	t.Run("DuplicateEventsRejected", func(t *testing.T) {
		single := `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"argmax_exact":true}}`
		out := single + "\n" + single + "\n"
		_, err := ParseStrixSubkernelParity(out, "argmax")
		if err == nil {
			t.Fatal("expected error for duplicate events, got nil")
		}
		if !errors.Is(err, ErrSubkernelParityDuplicate) {
			t.Errorf("expected ErrSubkernelParityDuplicate, got %v", err)
		}
	})

	t.Run("MalformedEventsRejected", func(t *testing.T) {
		tests := []struct {
			name string
			json string
		}{
			{"broken syntax", `{"schema":"fak.strix.subkernel-parity/v1", "selector":`},
			{"wrong schema", `{"schema":"fak.strix.subkernel-parity/v2","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true}`},
			{"empty selector", `{"schema":"fak.strix.subkernel-parity/v1","selector":"","test_name":"TestVulkanArgmaxExact","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true}`},
			{"empty test_name", `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true}`},
			{"empty oracle_kind", `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true}`},
			{"zero case count", `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":0,"passed":true}`},
			{"negative case count", `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":-1,"passed":true}`},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				_, err := ParseStrixSubkernelParity(tc.json, "argmax")
				if err == nil {
					t.Fatalf("expected malformed error for %q, got nil", tc.name)
				}
				if !errors.Is(err, ErrSubkernelParityMalformed) {
					t.Errorf("expected ErrSubkernelParityMalformed, got %v", err)
				}
			})
		}
	})

	t.Run("NonFiniteMetricsRejected", func(t *testing.T) {
		tests := []struct {
			name string
			json string
		}{
			{"NaN string in observed", `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":"NaN","max_abs_delta":0.001}}`},
			{"+Inf string in observed", `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.9999,"max_abs_delta":"+Inf"}}`},
			{"-Inf string in observed", `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":"-Inf","max_abs_delta":0.001}}`},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				_, err := ParseStrixSubkernelParity(tc.json, "matmul_f32")
				if err == nil {
					t.Fatalf("expected non-finite error for %q, got nil", tc.name)
				}
				if !errors.Is(err, ErrSubkernelParityNonFinite) {
					t.Errorf("expected ErrSubkernelParityNonFinite, got %v", err)
				}
			})
		}
	})

	t.Run("WrongEngineOrDeviceRejected", func(t *testing.T) {
		tests := []struct {
			name     string
			selector string
			json     string
		}{
			{
				name:     "device subkernel claims device_observed=false",
				selector: "matmul_f32",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":false,"case_count":1,"passed":true,"observed":{"cosine":0.99995,"max_abs_delta":0.005}}`,
			},
			{
				name:     "device subkernel claims llama.cpp engine",
				selector: "matmul_f32",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"llama.cpp","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.99995,"max_abs_delta":0.005}}`,
			},
			{
				name:     "device subkernel claims cpu engine",
				selector: "matmul_f32",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"cpu","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.99995,"max_abs_delta":0.005}}`,
			},
			{
				name:     "host contract claims device_observed=true",
				selector: "f16_kv_contiguize",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"f16_kv_contiguize","test_name":"TestRADVContiguizeShader","oracle_kind":"host_contract","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{}}`,
			},
			{
				name:     "host contract claims vulkan engine",
				selector: "f16_kv_contiguize",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"f16_kv_contiguize","test_name":"TestRADVContiguizeShader","oracle_kind":"host_contract","engine":"fak-native/vulkan","device_observed":false,"case_count":1,"passed":true,"observed":{}}`,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				_, err := ParseStrixSubkernelParity(tc.json, tc.selector)
				if err == nil {
					t.Fatalf("expected wrong engine/device error for %q, got nil", tc.name)
				}
				if !errors.Is(err, ErrSubkernelParityWrongEngine) {
					t.Errorf("expected ErrSubkernelParityWrongEngine, got %v", err)
				}
			})
		}
	})

	t.Run("SelectorOrTestMismatchRejected", func(t *testing.T) {
		tests := []struct {
			name     string
			selector string
			json     string
		}{
			{
				name:     "selector mismatch",
				selector: "argmax",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.99995,"max_abs_delta":0.005}}`,
			},
			{
				name:     "test_name geometry sibling mismatch",
				selector: "qwen35_sequence_prefill",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"qwen35_sequence_prefill","test_name":"TestVulkanQwen35SequenceGeometryValidation","oracle_kind":"max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"max_abs_delta":0.001,"finite_output":true}}`,
			},
			{
				name:     "oracle_kind mismatch",
				selector: "argmax",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"argmax_exact":true}}`,
			},
			{
				name:     "unknown selector in event without explicit selector",
				selector: "",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"unknown_selector","test_name":"TestFoo","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"argmax_exact":true}}`,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				_, err := ParseStrixSubkernelParity(tc.json, tc.selector)
				if err == nil {
					t.Fatalf("expected mismatch error for %q, got nil", tc.name)
				}
				if !errors.Is(err, ErrSubkernelParityMismatch) {
					t.Errorf("expected ErrSubkernelParityMismatch, got %v", err)
				}
			})
		}
	})

	t.Run("OutOfBoundsMetricsRejected", func(t *testing.T) {
		tests := []struct {
			name     string
			selector string
			json     string
		}{
			{
				name:     "passed declared false",
				selector: "matmul_f32",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":false,"observed":{"cosine":0.99995,"max_abs_delta":0.005}}`,
			},
			{
				name:     "cosine below minimum bound",
				selector: "matmul_f32",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.9998,"max_abs_delta":0.005}}`,
			},
			{
				name:     "max_abs above maximum bound",
				selector: "matmul_f32",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"matmul_f32","test_name":"TestVulkanMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.99995,"max_abs_delta":0.02}}`,
			},
			{
				name:     "rmsnorm_matmul mutated source tensor",
				selector: "rmsnorm_matmul",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"rmsnorm_matmul","test_name":"TestVulkanRMSNormMatMulApprox","oracle_kind":"cosine_max_abs","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.99995,"max_abs_delta":0.005,"max_source_delta":0.001}}`,
			},
			{
				name:     "q4k argmax_exact false",
				selector: "q4k_matmul",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"q4k_matmul","test_name":"TestVulkanQ4KMatMulMatchesCPUReference","oracle_kind":"cosine_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"cosine":0.998,"argmax_exact":false}}`,
			},
			{
				name:     "qwen35_gdn_decode state_identity false",
				selector: "qwen35_gdn_decode",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"qwen35_gdn_decode","test_name":"TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace","oracle_kind":"state_continuity","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"max_abs_delta":0.00015,"state_identity":false,"finite_output":true}}`,
			},
			{
				name:     "qwen35_gdn_decode finite_output false",
				selector: "qwen35_gdn_decode",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"qwen35_gdn_decode","test_name":"TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace","oracle_kind":"state_continuity","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"max_abs_delta":0.00015,"state_identity":true,"finite_output":false}}`,
			},
			{
				name:     "qwen35_gdn_decode max_abs above bound",
				selector: "qwen35_gdn_decode",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"qwen35_gdn_decode","test_name":"TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace","oracle_kind":"state_continuity","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"max_abs_delta":0.0004,"state_identity":true,"finite_output":true}}`,
			},
			{
				name:     "argmax carrying fabricated cosine",
				selector: "argmax",
				json:     `{"schema":"fak.strix.subkernel-parity/v1","selector":"argmax","test_name":"TestVulkanArgmaxExact","oracle_kind":"exact_argmax","engine":"fak-native/vulkan","device_observed":true,"case_count":1,"passed":true,"observed":{"argmax_exact":true,"cosine":0.9999}}`,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				_, err := ParseStrixSubkernelParity(tc.json, tc.selector)
				if err == nil {
					t.Fatalf("expected out-of-bounds error for %q, got nil", tc.name)
				}
				if !errors.Is(err, ErrSubkernelParityOutOfBounds) {
					t.Errorf("expected ErrSubkernelParityOutOfBounds, got %v", err)
				}
			})
		}
	})
}
