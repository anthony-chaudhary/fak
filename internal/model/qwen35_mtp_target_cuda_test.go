package model

import (
	"errors"
	"testing"
)

// namedMTPTargetBackend re-labels an injected device backend so target
// admission is exercised for each backend name with identical capabilities.
type namedMTPTargetBackend struct {
	*mtpRawHiddenPartialBackend
	name string
}

func (b *namedMTPTargetBackend) Name() string { return b.name }

// namedSequenceOnlyTargetBackend lacks raw-hidden capture: it models a resident
// backend that has sequence prefill, all-row logits and prefix replay only.
type namedSequenceOnlyTargetBackend struct {
	*partialCommitTestBackend
	name string
}

func (b *namedSequenceOnlyTargetBackend) Name() string { return b.name }

// TestQwen35MTPTargetAdmitsCUDAOnlyWithResidentCapabilities pins target
// capture admission: a CUDA-named backend is admitted with the same four
// resident capabilities Vulkan needs, a backend lacking raw-hidden capture
// refuses with the typed downgrade, and an unknown backend name stays refused.
//
// fak-test:runtime fast est=50ms lane=default
func TestQwen35MTPTargetAdmitsCUDAOnlyWithResidentCapabilities(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	full := func(t *testing.T, name string) *Session {
		t.Helper()
		base := &partialCommitTestBackend{devicePanelTestBackend: newDevicePanelTestBackend(m)}
		backend := &namedMTPTargetBackend{mtpRawHiddenPartialBackend: &mtpRawHiddenPartialBackend{partialCommitTestBackend: base, nonfiniteRawRow: -1}, name: name}
		target, err := m.NewBackendSessionChecked(backend)
		if err != nil {
			t.Fatal(err)
		}
		base.target = target
		base.devicePanelTestBackend.target = target
		t.Cleanup(target.Close)
		return target
	}
	for _, name := range []string{"cuda", "CUDA", "vulkan"} {
		t.Run("admits "+name, func(t *testing.T) {
			if err := validateQwen35MTPVulkanTarget(full(t, name)); err != nil {
				t.Fatalf("%s target with every resident capability refused: %v", name, err)
			}
		})
	}
	t.Run("refuses unknown backend", func(t *testing.T) {
		assertQwen35MTPUnsupported(t, validateQwen35MTPVulkanTarget(full(t, "rocm")))
	})
	t.Run("refuses cuda without raw hidden", func(t *testing.T) {
		base := &partialCommitTestBackend{devicePanelTestBackend: newDevicePanelTestBackend(m)}
		backend := &namedSequenceOnlyTargetBackend{partialCommitTestBackend: base, name: "cuda"}
		target, err := m.NewBackendSessionChecked(backend)
		if err != nil {
			t.Fatal(err)
		}
		base.target = target
		base.devicePanelTestBackend.target = target
		t.Cleanup(target.Close)
		err = validateQwen35MTPVulkanTarget(target)
		var unsupported *Qwen35MTPSpecDecodeUnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("cuda target without raw hidden err=%v, want typed refusal", err)
		}
	})
}

// TestEvaluateQwen38MTPEligibilityAdmitsCUDAQ4K pins the backend enum: the
// retained Q4_K MTP layout is eligible on CUDA exactly as on Metal once the
// caller reports the backend ready, and an unnamed backend still downgrades.
//
// fak-test:runtime fast est=50ms lane=default
func TestEvaluateQwen38MTPEligibilityAdmitsCUDAQ4K(t *testing.T) {
	q4, _ := qwen38MTPQ4KTestModels(t)
	input := Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact: true,
		MTPBackendReady:   true,
		Backend:           Qwen38MTPBackendCUDA,
		Model:             q4,
		Greedy:            true,
		Depth:             3,
		FreshSession:      true,
		MemoryHeadroomOK:  true,
		OperatorEnabled:   true,
	}
	got := EvaluateQwen38MTPEligibility(input)
	if !got.Eligible || got.Backend != Qwen38MTPBackendCUDA || got.MTPTensorFormat != Qwen38MTPFormatQ4K {
		t.Fatalf("CUDA Q4_K receipt=%+v, want eligible cuda q4_k", got)
	}
	input.MTPBackendReady = false
	if got := EvaluateQwen38MTPEligibility(input); got.Eligible || got.DowngradeReason != Qwen38MTPBackendUnsupported {
		t.Fatalf("CUDA backend not ready receipt=%+v, want backend_unsupported", got)
	}
	input.MTPBackendReady = true
	input.Backend = ""
	if got := EvaluateQwen38MTPEligibility(input); got.Eligible || got.DowngradeReason != Qwen38MTPBackendUnsupported {
		t.Fatalf("unnamed backend Q4_K receipt=%+v, want backend_unsupported", got)
	}
}
