package amdgpu

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestStrixValidationReceiptValidate(t *testing.T) {
	target := StrixTarget{
		Mode:           "ssh",
		Host:           "strix1",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}

	receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
	receipt.Schema = StrixValidationSchemaV1
	receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
		Name:       "argmax",
		Status:     "PASS",
		DurationUS: 320,
		Iterations: 1,
		Parity: StrixParityVerdict{
			ReferenceGEMV: "CPU reference (argmax)",
			Passed:        true,
			ArgmaxExact:   true,
		},
	})
	receipt.Ablations = append(receipt.Ablations, StrixAblationResult{
		Dimension: "target",
		Feature:   "cpu_vs_vulkan_gpu",
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 77800,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 428,
		},
		Speedup:   181.7,
		LiftRatio: 181.7,
		Verdict:   "VERIFIED_LIFT",
	})

	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest failed: %v", err)
	}
	if !strings.HasPrefix(digest, "sha256:") {
		t.Errorf("digest %q missing sha256 prefix", digest)
	}
	receipt.Digest = digest

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt should validate: %v", err)
	}

	// Assert empty digest fails closed in Validate
	receipt.Digest = ""
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "receipt digest is required") {
		t.Errorf("expected 'receipt digest is required' error for empty digest, got: %v", err)
	}

	// Assert empty digest fails closed in EvaluateReceipt (both verified and unverified)
	reg := NewStrixCandidateRegistry()
	receipt.Verified = false
	if _, err := reg.EvaluateReceipt(receipt); err == nil || !strings.Contains(err.Error(), "receipt missing required digest") {
		t.Errorf("expected 'receipt missing required digest' error for empty digest, got: %v", err)
	}

	// Restore digest for subsequent checks
	receipt.Digest = digest

	// Corrupt verdict and check validation
	receipt.Verdict = "UNKNOWN_VERDICT"
	if err := receipt.Validate(); err == nil {
		t.Error("expected error for invalid verdict")
	}

	// Revert verdict, mark target unreachable, expect failure
	receipt.Verdict = "PASS"
	receipt.Target.Reachable = false
	if err := receipt.Validate(); err == nil {
		t.Error("expected error when PASS but target unreachable")
	}

	// Corrupt digest and check validation failure
	receipt.Target.Reachable = true
	receipt.Digest = "sha256:corrupted"
	if err := receipt.Validate(); err == nil {
		t.Error("expected error for corrupted digest")
	}
}

func testStrixTarget() StrixTarget {
	return StrixTarget{
		Mode:           "ssh",
		Host:           "strix1",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}
}

func sealReceipt(t *testing.T, r *StrixValidationReceipt) {
	t.Helper()
	if r.Schema == StrixValidationSchemaV2 {
		r.Provenance.GitTip = testTip
		r.Provenance.SourceArchiveSHA256 = testHash
		r.Provenance.BinarySHA256 = testHash
		r.Provenance.ShaderBundleSHA256 = testHash
		r.Provenance.BuildCommandSHA256 = testHash
		r.Provenance.EngineIdentity = "fak-native/vulkan"
		r.Provenance.CleanupObserved = true
		r.SelectedCount = len(r.Subkernels)
		r.ExecutedCount = len(r.Subkernels)
		r.SelectedSubkernels = len(r.Subkernels)
		r.ExecutedSubkernels = len(r.Subkernels)
		r.SelectedAblations = len(r.Ablations)
		r.ExecutedAblations = len(r.Ablations)
		for i := range r.Subkernels {
			r.Subkernels[i].Evidence = validStrixExecutionEvidence()
			r.Subkernels[i].Evidence.DeviceIdentity = r.Target.GPUName + "|" + r.Target.TargetISA
		}
		for i := range r.Ablations {
			r.Ablations[i].Evidence = validStrixExecutionEvidence()
			r.Ablations[i].Evidence.DeviceIdentity = r.Target.GPUName + "|" + r.Target.TargetISA
			if r.Ablations[i].BaselineArm.Samples == 0 {
				r.Ablations[i].BaselineArm.Samples = 1
			}
			if r.Ablations[i].CandidateArm.Samples == 0 {
				r.Ablations[i].CandidateArm.Samples = 1
			}
		}
		r.Provenance.ExecutionManifestSHA256 = executionManifestDigest(r)
		r.Verified = true
		hostOnly := len(r.AllParityEvents()) > 0
		for _, ev := range r.AllParityEvents() {
			if ev.OracleKind != StrixOracleHostContract {
				hostOnly = false
			}
		}
		if hostOnly {
			r.Verdict = "SKIPPED"
			r.Verified = false
		}
	}
	digest, err := r.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest failed: %v", err)
	}
	r.Digest = digest
}

func TestStrixValidationReceipt_TypedParity_AcceptSupportedOracles(t *testing.T) {
	target := testStrixTarget()

	// 1. exact_argmax
	t.Run("exact_argmax", func(t *testing.T) {
		receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
		ev := NewExactArgmaxParityEvent("fak-native/vulkan", 128, true, true)
		if err := ev.Validate(); err != nil {
			t.Fatalf("exact_argmax event should validate: %v", err)
		}
		if !ev.PhysicalParityCredit() {
			t.Fatalf("exact_argmax event must earn physical parity credit")
		}

		receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
			Name:         "argmax",
			Status:       "PASS",
			DurationUS:   300,
			Iterations:   1,
			ParityEvents: []StrixParityEvent{ev},
		})
		sealReceipt(t, receipt)
		if err := receipt.Validate(); err != nil {
			t.Fatalf("receipt with exact_argmax should validate: %v", err)
		}
		if !receipt.CreditEligible() {
			t.Fatalf("receipt with exact_argmax should be credit eligible")
		}
	})

	// 2. cosine_max_abs
	t.Run("cosine_max_abs", func(t *testing.T) {
		receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
		ev := NewCosineMaxAbsParityEvent("fak-native/vulkan", 256, true, 0.999995, 0.999900, 0.0012, 0.01)
		if err := ev.Validate(); err != nil {
			t.Fatalf("cosine_max_abs event should validate: %v", err)
		}
		if !ev.PhysicalParityCredit() {
			t.Fatalf("cosine_max_abs event must earn physical parity credit")
		}

		receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
			Name:         "matmul_f32",
			Status:       "PASS",
			DurationUS:   450,
			Iterations:   1,
			ParityEvents: []StrixParityEvent{ev},
		})
		sealReceipt(t, receipt)
		if err := receipt.Validate(); err != nil {
			t.Fatalf("receipt with cosine_max_abs should validate: %v", err)
		}
		if !receipt.CreditEligible() {
			t.Fatalf("receipt with cosine_max_abs should be credit eligible")
		}
	})

	// 3. max_abs
	t.Run("max_abs", func(t *testing.T) {
		receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
		ev := NewMaxAbsParityEvent("fak-native/vulkan", 64, true, 0.0004, 0.001)
		if err := ev.Validate(); err != nil {
			t.Fatalf("max_abs event should validate: %v", err)
		}
		if !ev.PhysicalParityCredit() {
			t.Fatalf("max_abs event must earn physical parity credit")
		}

		receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
			Name:         "rmsnorm",
			Status:       "PASS",
			DurationUS:   120,
			Iterations:   1,
			ParityEvents: []StrixParityEvent{ev},
		})
		sealReceipt(t, receipt)
		if err := receipt.Validate(); err != nil {
			t.Fatalf("receipt with max_abs should validate: %v", err)
		}
		if !receipt.CreditEligible() {
			t.Fatalf("receipt with max_abs should be credit eligible")
		}
	})

	// 4. state_continuity
	t.Run("state_continuity", func(t *testing.T) {
		receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
		state := true
		finite := true
		maxAbs := 0.0001
		maxAbsBound := 0.0003
		ev := StrixParityEvent{
			OracleKind: StrixOracleStateContinuity, CaseCount: 16, DeviceObserved: true,
			Engine: "fak-native/vulkan", Passed: true,
			Observed: StrixParityMetrics{MaxAbsoluteDelta: &maxAbs, StateIdentity: &state, FiniteOutput: &finite},
			Bounds:   StrixParityBounds{MaxAbsDelta: &maxAbsBound, MaxAbsComparison: "<=", StateIdentity: &state, FiniteOutput: &finite},
		}
		if err := ev.Validate(); err != nil {
			t.Fatalf("state_continuity event should validate: %v", err)
		}
		if !ev.PhysicalParityCredit() {
			t.Fatalf("state_continuity event must earn physical parity credit")
		}

		receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
			Name:         "qwen35_gdn_decode",
			Status:       "PASS",
			DurationUS:   800,
			Iterations:   1,
			ParityEvents: []StrixParityEvent{ev},
		})
		sealReceipt(t, receipt)
		if err := receipt.Validate(); err != nil {
			t.Fatalf("receipt with state_continuity should validate: %v", err)
		}
		if !receipt.CreditEligible() {
			t.Fatalf("receipt with state_continuity should be credit eligible")
		}
	})

	// 5. host_contract
	t.Run("host_contract", func(t *testing.T) {
		receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
		ev := NewHostContractParityEvent("fak-native/host", 1, "vk_device_properties_abi", true)
		if err := ev.Validate(); err != nil {
			t.Fatalf("host_contract event should validate: %v", err)
		}
		// Host contracts never earn physical-device parity credit
		if ev.PhysicalParityCredit() {
			t.Fatalf("host_contract event must NOT earn physical parity credit")
		}

		receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
			Name:         "host_abi_check",
			Status:       "PASS",
			DurationUS:   50,
			Iterations:   1,
			ParityEvents: []StrixParityEvent{ev},
		})
		sealReceipt(t, receipt)
		if err := receipt.Validate(); err != nil {
			t.Fatalf("receipt with host_contract should validate: %v", err)
		}
		// Receipt with ONLY host_contract must not be eligible for physical parity credit
		if receipt.CreditEligible() {
			t.Fatalf("receipt with ONLY host_contract must NOT earn physical parity credit")
		}
	})
}

func TestStrixValidationReceipt_TypedParity_RejectUnsupportedOracleKind(t *testing.T) {
	for _, badKind := range []StrixOracleKind{"unknown", "magic_oracle", "untyped", ""} {
		ev := StrixParityEvent{
			OracleKind:     badKind,
			CaseCount:      10,
			DeviceObserved: true,
			Engine:         "fak-native/vulkan",
			Passed:         true,
		}
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "unsupported or unknown oracle kind") {
			t.Errorf("expected error for bad kind %q, got: %v", badKind, err)
		}
	}
}

func TestStrixValidationReceipt_TypedParity_RejectMissingMetrics(t *testing.T) {
	t.Run("exact_argmax_missing_metric", func(t *testing.T) {
		ev := StrixParityEvent{
			OracleKind:     StrixOracleExactArgmax,
			CaseCount:      10,
			DeviceObserved: true,
			Engine:         "fak-native/vulkan",
			Passed:         true,
			Observed:       StrixParityMetrics{}, // missing ArgmaxExact
			Bounds:         StrixParityBounds{Comparison: "=="},
		}
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires observed argmax_exact") {
			t.Errorf("expected missing argmax_exact error, got: %v", err)
		}
	})

	t.Run("cosine_max_abs_missing_metrics", func(t *testing.T) {
		ev := StrixParityEvent{
			OracleKind:     StrixOracleCosineMaxAbs,
			CaseCount:      10,
			DeviceObserved: true,
			Engine:         "fak-native/vulkan",
			Passed:         true,
			Observed: StrixParityMetrics{
				CosineSimilarity: Float64Ptr(0.9999),
				// missing MaxAbsoluteDelta
			},
			Bounds: StrixParityBounds{
				MinCosine:   Float64Ptr(0.999),
				MaxAbsDelta: Float64Ptr(0.01),
			},
		}
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires observed max_absolute_delta") {
			t.Errorf("expected missing max_absolute_delta error, got: %v", err)
		}

		// Missing MinCosine bound
		ev.Observed.MaxAbsoluteDelta = Float64Ptr(0.001)
		ev.Bounds.MinCosine = nil
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires min_cosine bound") {
			t.Errorf("expected missing min_cosine bound error, got: %v", err)
		}

		// Missing MaxAbsDelta bound
		ev.Bounds.MinCosine = Float64Ptr(0.999)
		ev.Bounds.MaxAbsDelta = nil
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires max_abs_delta bound") {
			t.Errorf("expected missing max_abs_delta bound error, got: %v", err)
		}
	})

	t.Run("max_abs_missing_metrics", func(t *testing.T) {
		ev := StrixParityEvent{
			OracleKind:     StrixOracleMaxAbs,
			CaseCount:      10,
			DeviceObserved: true,
			Engine:         "fak-native/vulkan",
			Passed:         true,
			Observed:       StrixParityMetrics{}, // missing MaxAbsoluteDelta
			Bounds:         StrixParityBounds{MaxAbsDelta: Float64Ptr(0.01)},
		}
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires observed max_absolute_delta") {
			t.Errorf("expected missing max_absolute_delta error, got: %v", err)
		}

		ev.Observed.MaxAbsoluteDelta = Float64Ptr(0.001)
		ev.Bounds.MaxAbsDelta = nil
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires max_abs_delta bound") {
			t.Errorf("expected missing max_abs_delta bound error, got: %v", err)
		}
	})

	t.Run("state_continuity_missing_metrics", func(t *testing.T) {
		ev := StrixParityEvent{
			OracleKind:     StrixOracleStateContinuity,
			CaseCount:      10,
			DeviceObserved: true,
			Engine:         "fak-native/vulkan",
			Passed:         true,
			Observed:       StrixParityMetrics{}, // all nil
		}
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires observed state continuity metric") {
			t.Errorf("expected missing state metric error, got: %v", err)
		}

		ev.Observed.StateCosine = Float64Ptr(0.9999)
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires min_state_cosine bound") {
			t.Errorf("expected missing min_state_cosine bound error, got: %v", err)
		}
	})

	t.Run("host_contract_missing_metrics", func(t *testing.T) {
		ev := StrixParityEvent{
			OracleKind:     StrixOracleHostContract,
			CaseCount:      1,
			DeviceObserved: false,
			Engine:         "fak-native/host",
			Passed:         true,
			Observed:       StrixParityMetrics{}, // missing ContractHolds
		}
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires observed contract_holds") {
			t.Errorf("expected missing contract_holds error, got: %v", err)
		}
	})

	t.Run("missing_engine_or_case_count", func(t *testing.T) {
		ev := NewExactArgmaxParityEvent("", 10, true, true)
		if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "requires non-empty engine") {
			t.Errorf("expected missing engine error, got: %v", err)
		}

		ev2 := NewExactArgmaxParityEvent("fak-native/vulkan", 0, true, true)
		if err := ev2.Validate(); err == nil || !strings.Contains(err.Error(), "requires positive case_count") {
			t.Errorf("expected non-positive case_count error, got: %v", err)
		}

		ev3 := NewExactArgmaxParityEvent("fak-native/vulkan", -5, true, true)
		if err := ev3.Validate(); err == nil || !strings.Contains(err.Error(), "requires positive case_count") {
			t.Errorf("expected negative case_count error, got: %v", err)
		}
	})
}

func TestStrixValidationReceipt_TypedParity_RejectNonFiniteNaNInf(t *testing.T) {
	tests := []struct {
		name string
		ev   StrixParityEvent
	}{
		{
			name: "observed_cosine_nan",
			ev: StrixParityEvent{
				OracleKind:     StrixOracleCosineMaxAbs,
				CaseCount:      10,
				DeviceObserved: true,
				Engine:         "fak-native/vulkan",
				Observed:       StrixParityMetrics{CosineSimilarity: Float64Ptr(math.NaN()), MaxAbsoluteDelta: Float64Ptr(0.01)},
				Bounds:         StrixParityBounds{MinCosine: Float64Ptr(0.999), MaxAbsDelta: Float64Ptr(0.05)},
			},
		},
		{
			name: "observed_max_abs_inf",
			ev: StrixParityEvent{
				OracleKind:     StrixOracleCosineMaxAbs,
				CaseCount:      10,
				DeviceObserved: true,
				Engine:         "fak-native/vulkan",
				Observed:       StrixParityMetrics{CosineSimilarity: Float64Ptr(0.9999), MaxAbsoluteDelta: Float64Ptr(math.Inf(1))},
				Bounds:         StrixParityBounds{MinCosine: Float64Ptr(0.999), MaxAbsDelta: Float64Ptr(0.05)},
			},
		},
		{
			name: "bound_min_cosine_nan",
			ev: StrixParityEvent{
				OracleKind:     StrixOracleCosineMaxAbs,
				CaseCount:      10,
				DeviceObserved: true,
				Engine:         "fak-native/vulkan",
				Observed:       StrixParityMetrics{CosineSimilarity: Float64Ptr(0.9999), MaxAbsoluteDelta: Float64Ptr(0.01)},
				Bounds:         StrixParityBounds{MinCosine: Float64Ptr(math.NaN()), MaxAbsDelta: Float64Ptr(0.05)},
			},
		},
		{
			name: "bound_max_abs_neginf",
			ev: StrixParityEvent{
				OracleKind:     StrixOracleCosineMaxAbs,
				CaseCount:      10,
				DeviceObserved: true,
				Engine:         "fak-native/vulkan",
				Observed:       StrixParityMetrics{CosineSimilarity: Float64Ptr(0.9999), MaxAbsoluteDelta: Float64Ptr(0.01)},
				Bounds:         StrixParityBounds{MinCosine: Float64Ptr(0.999), MaxAbsDelta: Float64Ptr(math.Inf(-1))},
			},
		},
		{
			name: "observed_state_cosine_nan",
			ev: StrixParityEvent{
				OracleKind:     StrixOracleStateContinuity,
				CaseCount:      10,
				DeviceObserved: true,
				Engine:         "fak-native/vulkan",
				Observed:       StrixParityMetrics{StateCosine: Float64Ptr(math.NaN())},
				Bounds:         StrixParityBounds{MinStateCosine: Float64Ptr(0.999)},
			},
		},
		{
			name: "observed_state_delta_inf",
			ev: StrixParityEvent{
				OracleKind:     StrixOracleStateContinuity,
				CaseCount:      10,
				DeviceObserved: true,
				Engine:         "fak-native/vulkan",
				Observed:       StrixParityMetrics{StateDelta: Float64Ptr(math.Inf(1))},
				Bounds:         StrixParityBounds{MaxStateDelta: Float64Ptr(0.01)},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.ev.Validate()
			if err == nil || !strings.Contains(err.Error(), "cannot be NaN or Inf") {
				t.Errorf("expected NaN/Inf rejection, got: %v", err)
			}
		})
	}
}

func TestStrixValidationReceipt_TypedParity_RejectWrongComparisons(t *testing.T) {
	// exact_argmax wrong comparison
	evExact := NewExactArgmaxParityEvent("fak-native/vulkan", 10, true, true)
	evExact.Bounds.Comparison = "<="
	if err := evExact.Validate(); err == nil || !strings.Contains(err.Error(), "wrong comparison") {
		t.Errorf("expected wrong comparison error for exact_argmax, got: %v", err)
	}

	// cosine_max_abs inverted cosine comparison
	evCos := NewCosineMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.9999, 0.999, 0.001, 0.01)
	evCos.Bounds.CosineComparison = "<="
	if err := evCos.Validate(); err == nil || !strings.Contains(err.Error(), "wrong comparison") {
		t.Errorf("expected wrong comparison error for cosine in cosine_max_abs, got: %v", err)
	}

	// cosine_max_abs inverted max_abs comparison
	evCos2 := NewCosineMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.9999, 0.999, 0.001, 0.01)
	evCos2.Bounds.MaxAbsComparison = ">="
	if err := evCos2.Validate(); err == nil || !strings.Contains(err.Error(), "wrong comparison") {
		t.Errorf("expected wrong comparison error for max_abs in cosine_max_abs, got: %v", err)
	}

	// max_abs wrong comparison
	evMaxAbs := NewMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.001, 0.01)
	evMaxAbs.Bounds.Comparison = ">="
	if err := evMaxAbs.Validate(); err == nil || !strings.Contains(err.Error(), "wrong comparison") {
		t.Errorf("expected wrong comparison error for max_abs, got: %v", err)
	}

	// state_continuity wrong comparison
	evState := NewStateContinuityParityEvent("fak-native/vulkan", 10, true, 0.9999, 0.999, 0.001, 0.01)
	evState.Bounds.StateCosineComparison = "<=" // inverted for state cosine
	if err := evState.Validate(); err == nil || !strings.Contains(err.Error(), "wrong comparison") {
		t.Errorf("expected wrong comparison error for state_cosine, got: %v", err)
	}

	evStateDelta := NewStateContinuityParityEvent("fak-native/vulkan", 10, true, 0.9999, 0.999, 0.001, 0.01)
	evStateDelta.Bounds.StateDeltaComparison = ">=" // inverted for state delta
	if err := evStateDelta.Validate(); err == nil || !strings.Contains(err.Error(), "wrong comparison") {
		t.Errorf("expected wrong comparison error for state_delta, got: %v", err)
	}

	// host_contract wrong comparison
	evHost := NewHostContractParityEvent("fak-native/host", 1, "test_contract", true)
	evHost.Bounds.Comparison = "!="
	if err := evHost.Validate(); err == nil || !strings.Contains(err.Error(), "wrong comparison") {
		t.Errorf("expected wrong comparison error for host_contract, got: %v", err)
	}
}

func TestStrixValidationReceipt_TypedParity_RejectLoosenedBounds(t *testing.T) {
	// cosine_max_abs: min_cosine below 0.90 is a loosened bound
	evCosLoose := NewCosineMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.5, 0.4, 0.001, 0.01)
	if err := evCosLoose.Validate(); err == nil || !strings.Contains(err.Error(), "loosened bounds") {
		t.Errorf("expected loosened bounds error for min_cosine=0.4, got: %v", err)
	}

	// cosine_max_abs: max_abs_delta negative or unbounded
	evCosLooseDelta := NewCosineMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.9999, 0.999, 0.001, -0.01)
	if err := evCosLooseDelta.Validate(); err == nil || !strings.Contains(err.Error(), "loosened bounds") {
		t.Errorf("expected loosened bounds error for negative max_abs_delta, got: %v", err)
	}

	// max_abs: max_abs_delta > 1.0 (unacceptable loose bound for normalized kernels)
	evMaxAbsLoose := NewMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.01, 5.0)
	if err := evMaxAbsLoose.Validate(); err == nil || !strings.Contains(err.Error(), "loosened bounds") {
		t.Errorf("expected loosened bounds error for max_abs_delta=5.0, got: %v", err)
	}

	// exact_argmax: exact_match = false (loosening exact identity)
	evExactLoose := NewExactArgmaxParityEvent("fak-native/vulkan", 10, true, true)
	evExactLoose.Bounds.ExactMatch = BoolPtr(false)
	if err := evExactLoose.Validate(); err == nil || !strings.Contains(err.Error(), "loosened bounds") {
		t.Errorf("expected loosened bounds error for exact_match=false, got: %v", err)
	}

	// state_continuity: min_state_cosine < 0.90
	evStateLoose := NewStateContinuityParityEvent("fak-native/vulkan", 10, true, 0.95, 0.50, 0.001, 0.01)
	if err := evStateLoose.Validate(); err == nil || !strings.Contains(err.Error(), "loosened bounds") {
		t.Errorf("expected loosened bounds error for min_state_cosine=0.50, got: %v", err)
	}
}

func TestStrixValidationReceipt_TypedParity_RejectFalsePasses(t *testing.T) {
	// exact_argmax: Passed=true but ArgmaxExact=false
	evExactFalse := NewExactArgmaxParityEvent("fak-native/vulkan", 10, true, false)
	evExactFalse.Passed = true // false pass!
	if err := evExactFalse.Validate(); err == nil || !strings.Contains(err.Error(), "false pass") {
		t.Errorf("expected false pass error for exact_argmax, got: %v", err)
	}

	// cosine_max_abs: Passed=true but cosine fails bound
	evCosFalse := NewCosineMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.98, 0.999, 0.001, 0.01)
	evCosFalse.Passed = true // false pass!
	if err := evCosFalse.Validate(); err == nil || !strings.Contains(err.Error(), "false pass") {
		t.Errorf("expected false pass error for cosine < min_cosine, got: %v", err)
	}

	// cosine_max_abs: Passed=true but max_abs fails bound
	evCosFalseDelta := NewCosineMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.9999, 0.999, 0.05, 0.01)
	evCosFalseDelta.Passed = true // false pass!
	if err := evCosFalseDelta.Validate(); err == nil || !strings.Contains(err.Error(), "false pass") {
		t.Errorf("expected false pass error for max_abs > bound, got: %v", err)
	}

	// max_abs: Passed=true but delta fails bound
	evMaxAbsFalse := NewMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.05, 0.01)
	evMaxAbsFalse.Passed = true // false pass!
	if err := evMaxAbsFalse.Validate(); err == nil || !strings.Contains(err.Error(), "false pass") {
		t.Errorf("expected false pass error for max_abs, got: %v", err)
	}

	// state_continuity: Passed=true but state delta fails bound
	evStateFalse := NewStateContinuityParityEvent("fak-native/vulkan", 10, true, 0.9999, 0.999, 0.05, 0.01)
	evStateFalse.Passed = true // false pass!
	if err := evStateFalse.Validate(); err == nil || !strings.Contains(err.Error(), "false pass") {
		t.Errorf("expected false pass error for state delta, got: %v", err)
	}

	// host_contract: Passed=true but ContractHolds=false
	evHostFalse := NewHostContractParityEvent("fak-native/host", 1, "test_contract", false)
	evHostFalse.Passed = true // false pass!
	if err := evHostFalse.Validate(); err == nil || !strings.Contains(err.Error(), "false pass") {
		t.Errorf("expected false pass error for host_contract, got: %v", err)
	}
}

func TestStrixValidationReceipt_TypedParity_RejectExactCosineConflation(t *testing.T) {
	// exact_argmax with observed CosineSimilarity
	evExact := NewExactArgmaxParityEvent("fak-native/vulkan", 10, true, true)
	evExact.Observed.CosineSimilarity = Float64Ptr(1.0)
	if err := evExact.Validate(); err == nil || !strings.Contains(err.Error(), "exact/cosine conflation") {
		t.Errorf("expected exact/cosine conflation error, got: %v", err)
	}

	// exact_argmax with bound MinCosine
	evExact2 := NewExactArgmaxParityEvent("fak-native/vulkan", 10, true, true)
	evExact2.Bounds.MinCosine = Float64Ptr(0.999)
	if err := evExact2.Validate(); err == nil || !strings.Contains(err.Error(), "exact/cosine conflation") {
		t.Errorf("expected exact/cosine conflation error, got: %v", err)
	}

	// max_abs with observed CosineSimilarity
	evMaxAbs := NewMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.001, 0.01)
	evMaxAbs.Observed.CosineSimilarity = Float64Ptr(0.999)
	if err := evMaxAbs.Validate(); err == nil || !strings.Contains(err.Error(), "exact/cosine conflation") {
		t.Errorf("expected exact/cosine conflation error for max_abs, got: %v", err)
	}

	// host_contract with observed CosineSimilarity
	evHost := NewHostContractParityEvent("fak-native/host", 1, "test_contract", true)
	evHost.Observed.CosineSimilarity = Float64Ptr(1.0)
	if err := evHost.Validate(); err == nil || !strings.Contains(err.Error(), "exact/cosine conflation") {
		t.Errorf("expected exact/cosine conflation error for host_contract, got: %v", err)
	}

	// Legacy fields remain readable but cannot synthesize current typed credit.
	sk := StrixSubkernelResult{
		Name:       "conflated_subkernel",
		Status:     "PASS",
		DurationUS: 100,
		Iterations: 1,
		Parity: StrixParityVerdict{
			Passed:                true,
			ArgmaxExact:           true,
			LogitCosineSimilarity: 0.9999,
		},
	}
	events := sk.AllParityEvents()
	if len(events) != 0 {
		t.Fatalf("legacy fields synthesized %d current typed event(s)", len(events))
	}
}

func TestStrixValidationReceipt_TypedParity_HostContractNonCredit(t *testing.T) {
	// Host contract claiming device_observed = true must fail validation
	ev := StrixParityEvent{
		OracleKind:     StrixOracleHostContract,
		CaseCount:      1,
		DeviceObserved: true, // Non-physical contract claiming physical device
		Engine:         "fak-native/host",
		Passed:         true,
		Observed:       StrixParityMetrics{ContractHolds: BoolPtr(true)},
		Bounds:         StrixParityBounds{Comparison: "=="},
	}
	if err := ev.Validate(); err == nil || !strings.Contains(err.Error(), "host_contract cannot claim device_observed=true") {
		t.Errorf("expected error for host_contract claiming device_observed=true, got: %v", err)
	}

	// Valid host contract: PhysicalParityCredit is false
	evValid := NewHostContractParityEvent("fak-native/host", 1, "test_contract", true)
	if evValid.PhysicalParityCredit() {
		t.Errorf("host contract must never earn physical parity credit")
	}

	// Receipt with only host_contract: CreditEligible is false
	target := testStrixTarget()
	receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
	receipt.ParityEvents = append(receipt.ParityEvents, evValid)
	sealReceipt(t, receipt)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt should validate: %v", err)
	}
	if receipt.CreditEligible() {
		t.Errorf("receipt containing only host_contract must not be CreditEligible")
	}
}

func TestStrixValidationReceipt_TypedParity_V1HistoricalNonCredit(t *testing.T) {
	target := testStrixTarget()
	receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
	receipt.Schema = StrixValidationSchemaV1 // historical v1
	receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
		Name:       "argmax",
		Status:     "PASS",
		DurationUS: 300,
		Iterations: 1,
		Parity: StrixParityVerdict{
			Passed:      true,
			ArgmaxExact: true,
		},
	})
	sealReceipt(t, receipt)

	// Historical v1 remains integrity-readable
	if err := receipt.Validate(); err != nil {
		t.Fatalf("v1 receipt should be integrity-readable: %v", err)
	}
	// Historical v1 is non-credit
	if receipt.CreditEligible() {
		t.Errorf("historical v1 receipt must NOT be credit eligible")
	}
	if receipt.PhysicalParityCredit() {
		t.Errorf("historical v1 receipt must NOT have physical parity credit")
	}
}

func TestStrixValidationReceiptRejectsSlowParityMatch(t *testing.T) {
	receipt := validStrixReceipt(t)
	receipt.Ablations = []StrixAblationResult{{
		Dimension: "decode", Feature: "slow_candidate",
		BaselineArm:  StrixArmResult{Name: "baseline", LatencyUS: 100, Samples: 3},
		CandidateArm: StrixArmResult{Name: "candidate", LatencyUS: 200, Samples: 3},
		Speedup:      0.5, LiftRatio: 0.5, CosineParity: 1,
		Evidence: validStrixExecutionEvidence(), Verdict: "PARITY_MATCH",
	}}
	receipt.SelectedAblations, receipt.ExecutedAblations = 1, 1
	receipt.Provenance.ExecutionManifestSHA256 = executionManifestDigest(receipt)
	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatal(err)
	}
	receipt.Digest = digest
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "parity outside") {
		t.Fatalf("slow PARITY_MATCH earned credit: %v", err)
	}
}

func TestCreditEligibleRequiresFullReceiptValidation(t *testing.T) {
	receipt := NewStrixValidationReceipt(testStrixTarget(), "HEAD", testTip, "fak validate --strix")
	receipt.Verified = true
	receipt.ParityEvents = []StrixParityEvent{NewExactArgmaxParityEvent("fak-native/vulkan", 1, true, true)}
	if receipt.CreditEligible() {
		t.Fatal("minimal PASS receipt without digest/provenance/execution evidence earned credit")
	}
}

func TestStrixValidationReceipt_TypedParity_CannotBeResealedWithWeakEvidence(t *testing.T) {
	target := testStrixTarget()
	receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")

	// Create a subkernel with false-pass parity event
	ev := NewExactArgmaxParityEvent("fak-native/vulkan", 10, true, false)
	ev.Passed = true // false pass!
	receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
		Name:         "tampered_argmax",
		Status:       "PASS",
		DurationUS:   300,
		Iterations:   1,
		ParityEvents: []StrixParityEvent{ev},
	})

	// Reseal the receipt around the false pass
	sealReceipt(t, receipt)

	// Validation must fail closed despite valid digest
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "false pass") {
		t.Fatalf("resealed receipt around false pass must fail closed, got: %v", err)
	}

	// Now try resealing with loosened bounds
	evLoose := NewCosineMaxAbsParityEvent("fak-native/vulkan", 10, true, 0.5, 0.4, 0.001, 0.01)
	receipt.Subkernels[0].ParityEvents = []StrixParityEvent{evLoose}
	sealReceipt(t, receipt)

	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "loosened bounds") {
		t.Fatalf("resealed receipt around loosened bounds must fail closed, got: %v", err)
	}
}
