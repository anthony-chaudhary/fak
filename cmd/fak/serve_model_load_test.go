package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// backendAutoProbeFake is an opaque registered-backend sentinel for the auto
// probe tests: it makes no device claim of its own and is only ever read through
// serveCUDAProbe. It deliberately does NOT implement
// compute.CUDADeviceReceiptProvider, so the production probe fails closed on it.
type backendAutoProbeFake struct {
	compute.Backend
	name string
}

func (b *backendAutoProbeFake) Name() string { return b.name }

// backendAutoReceiptFake implements the receipt provider with a real receipt the
// production probe reads, so covered/uncovered cases exercise the production
// seam rather than an injected stub.
type backendAutoReceiptFake struct {
	compute.Backend
	name    string
	receipt compute.CUDADeviceReceipt
}

func (b *backendAutoReceiptFake) Name() string                                 { return b.name }
func (b *backendAutoReceiptFake) CUDADeviceReceipt() compute.CUDADeviceReceipt { return b.receipt }

// backendAutoLookup is the injected registry: a nil entry reads as unregistered.
func backendAutoLookup(cuda, vulkan compute.Backend) serveBackendLookup {
	return func(name string) (compute.Backend, error) {
		switch name {
		case "cuda":
			if cuda != nil {
				return cuda, nil
			}
		case "vulkan":
			if vulkan != nil {
				return vulkan, nil
			}
		}
		return nil, fmt.Errorf("backend %q is not registered", name)
	}
}

// assertCUDASkip checks the typed refusal: a *cudaAutoRefusalError with the
// wanted closed-vocabulary reason (never prose). wantCause is a substring the
// preserved underlying failure must carry; "" means the skip was a clean
// classification with no underlying failure.
func assertCUDASkip(t *testing.T, err error, want serveCUDASkipReason, wantCause string) {
	t.Helper()
	var typed *cudaAutoRefusalError
	if !errors.As(err, &typed) {
		t.Fatalf("cuda refusal = %v (%T), want *cudaAutoRefusalError", err, err)
	}
	if typed.Reason != want {
		t.Fatalf("cuda refusal reason = %q, want %q", typed.Reason, want)
	}
	if wantCause == "" {
		if typed.Cause != nil {
			t.Fatalf("cuda refusal cause = %v, want none", typed.Cause)
		}
		return
	}
	if typed.Cause == nil || !strings.Contains(typed.Cause.Error(), wantCause) {
		t.Fatalf("cuda refusal cause = %v, want one containing %q", typed.Cause, wantCause)
	}
}

// fak-test:runtime fast est=3ms lane=default
func TestBackendAutoCoveredArchSelectsCUDA(t *testing.T) {
	vulkan := &backendAutoReceiptFake{name: "vulkan"}
	for _, goos := range []string{"linux", "windows"} {
		for _, coverage := range []compute.CUDACoverage{compute.CUDANative, compute.CUDAJITFromPTX} {
			cuda := &backendAutoReceiptFake{name: "cuda", receipt: compute.CUDADeviceReceipt{Backend: "cuda", Coverage: coverage}}
			sel, err := resolveServeBackendSelection("auto", "", goos, backendAutoLookup(cuda, vulkan), probeServeCUDAArchCoverage)
			if err != nil {
				t.Fatalf("goos=%s coverage=%s: resolve: %v", goos, coverage, err)
			}
			if sel.name != "cuda" || sel.backend != compute.Backend(cuda) {
				t.Fatalf("goos=%s coverage=%s: selected %q, want cuda", goos, coverage, sel.name)
			}
			if sel.cudaRefusal != nil {
				t.Fatalf("goos=%s coverage=%s: refusal = %v, want none after cuda selection", goos, coverage, sel.cudaRefusal)
			}
		}
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestBackendAutoUncoveredArchFallsThrough(t *testing.T) {
	uncovered := func(compute.Backend) (compute.CUDACoverage, error) { return compute.CUDAUncovered, nil }
	cuda := &backendAutoProbeFake{name: "cuda"}
	vulkan := &backendAutoProbeFake{name: "vulkan"}

	sel, err := resolveServeBackendSelection("auto", "", "linux", backendAutoLookup(cuda, vulkan), uncovered)
	if err != nil {
		t.Fatalf("uncovered arch must fall through, not fail: %v", err)
	}
	if sel.name != "vulkan" || sel.backend != compute.Backend(vulkan) {
		t.Fatalf("uncovered arch selected %q, want vulkan fallthrough", sel.name)
	}
	assertCUDASkip(t, sel.cudaRefusal, cudaSkipArchUncovered, "")

	sel, err = resolveServeBackendSelection("auto", "", "windows", backendAutoLookup(cuda, nil), uncovered)
	if err != nil {
		t.Fatalf("uncovered arch must fall through, not fail: %v", err)
	}
	if sel.name != serveBackendCPU || sel.backend != nil || sel.source != "default" {
		t.Fatalf("uncovered arch without vulkan selected %q (source=%q), want cpu default", sel.name, sel.source)
	}
	assertCUDASkip(t, sel.cudaRefusal, cudaSkipArchUncovered, "")
}

// fak-test:runtime fast est=2ms lane=default
func TestBackendAutoUnregisteredFallsThrough(t *testing.T) {
	probe := func(compute.Backend) (compute.CUDACoverage, error) {
		t.Fatal("probe ran for an unregistered cuda backend")
		return compute.CUDAUncovered, nil
	}
	vulkan := &backendAutoProbeFake{name: "vulkan"}

	sel, err := resolveServeBackendSelection("", "", "linux", backendAutoLookup(nil, vulkan), probe)
	if err != nil {
		t.Fatalf("unregistered cuda must fall through, not fail: %v", err)
	}
	if sel.name != "vulkan" || sel.backend != compute.Backend(vulkan) {
		t.Fatalf("unregistered cuda selected %q, want vulkan fallthrough", sel.name)
	}
	assertCUDASkip(t, sel.cudaRefusal, cudaSkipNotRegistered, "not registered")

	sel, err = resolveServeBackendSelection("auto", "", "windows", backendAutoLookup(nil, nil), probe)
	if err != nil {
		t.Fatalf("unregistered cuda must fall through, not fail: %v", err)
	}
	if sel.name != serveBackendCPU || sel.backend != nil {
		t.Fatalf("unregistered cuda without vulkan selected %q, want cpu", sel.name)
	}
	assertCUDASkip(t, sel.cudaRefusal, cudaSkipNotRegistered, "not registered")
}

// fak-test:runtime fast est=2ms lane=default
func TestBackendAutoProbeErrorTypedRefusal(t *testing.T) {
	vulkan := &backendAutoProbeFake{name: "vulkan"}

	// A registered backend that emits no typed device receipt fails the
	// production probe closed: no detected arch, no coverage claim.
	receiptless := &backendAutoProbeFake{name: "cuda"}
	sel, err := resolveServeBackendSelection("auto", "", "linux", backendAutoLookup(receiptless, vulkan), probeServeCUDAArchCoverage)
	if err != nil {
		t.Fatalf("probe error must fall through, not fail: %v", err)
	}
	if sel.name != "vulkan" || sel.backend != compute.Backend(vulkan) {
		t.Fatalf("probe error selected %q, want vulkan fallthrough", sel.name)
	}
	assertCUDASkip(t, sel.cudaRefusal, cudaSkipProbeError, "no typed CUDA device receipt")

	// An injected probe failure keeps its typed reason AND its cause.
	probeErr := errors.New("device receipt read failed")
	cuda := &backendAutoProbeFake{name: "cuda"}
	sel, err = resolveServeBackendSelection("auto", "", "windows", backendAutoLookup(cuda, vulkan),
		func(compute.Backend) (compute.CUDACoverage, error) { return compute.CUDAUncovered, probeErr })
	if err != nil {
		t.Fatalf("probe error must fall through, not fail: %v", err)
	}
	if sel.name != "vulkan" {
		t.Fatalf("probe error selected %q, want vulkan fallthrough", sel.name)
	}
	assertCUDASkip(t, sel.cudaRefusal, cudaSkipProbeError, "device receipt read failed")
	if !errors.Is(sel.cudaRefusal, probeErr) {
		t.Fatalf("cuda refusal %v does not wrap cause %v", sel.cudaRefusal, probeErr)
	}
}
