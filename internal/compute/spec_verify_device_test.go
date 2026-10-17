package compute

import (
	"errors"
	"testing"
)

type hostReferenceSpecVerifyBackend struct{ Backend }

func (b hostReferenceSpecVerifyBackend) Name() string { return "fake-device" }

func (b hostReferenceSpecVerifyBackend) Caps() Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = true
	return caps
}

func (hostReferenceSpecVerifyBackend) SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	return specVerifyOnHost(q, k, v, out, qLen, kvLen, nH, nHkv, d)
}

func (hostReferenceSpecVerifyBackend) specVerifyRunsOnHost() {}

func specVerifyFallbackCount() uint64 {
	for _, c := range HostFallbackCounts() {
		if c.Stage == HostFallbackStageSpecVerify {
			return c.Count
		}
	}
	return 0
}

func specVerifyOperands(be Backend) (q, k, v Tensor) {
	var s lcg = 9
	qLen, kvLen, nH, nHkv, d := 2, 4, 2, 1, 8
	return NewF32(be, []int{qLen, nH, d}, randVec(&s, qLen*nH*d)),
		NewF32(be, []int{kvLen, nHkv, d}, randVec(&s, kvLen*nHkv*d)),
		NewF32(be, []int{kvLen, nHkv, d}, randVec(&s, kvLen*nHkv*d))
}

// fak-test:runtime fast est=2ms lane=default
func TestSpecVerifyDeviceHostReferenceCountsAndDeviceOnlyRefuses(t *testing.T) {
	dev := hostReferenceSpecVerifyBackend{Backend: Default()}
	q, k, v := specVerifyOperands(dev)

	before := specVerifyFallbackCount()
	var out Tensor
	if err := SpecVerifyAttention(&q, &k, &v, &out, 2, 4, 2, 1, 8); err != nil {
		t.Fatalf("portable SpecVerifyAttention: %v", err)
	}
	if got := specVerifyFallbackCount(); got != before+1 {
		t.Fatalf("portable host reference counted %d, want %d", got, before+1)
	}

	var refused Tensor
	err := SpecVerifyAttentionDeviceOnly(&q, &k, &v, &refused, 2, 4, 2, 1, 8)
	if !errors.Is(err, ErrSpecVerifyDeviceOnly) {
		t.Fatalf("device-only err=%v, want ErrSpecVerifyDeviceOnly", err)
	}
	if refused.Backend() != nil {
		t.Fatal("device-only refusal still produced an output")
	}
	if got := specVerifyFallbackCount(); got != before+2 {
		t.Fatalf("device-only refusal counted %d, want %d", got, before+2)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpecVerifyDeviceCPUBackendIsNotAFallback(t *testing.T) {
	q, k, v := specVerifyOperands(Default())
	before := specVerifyFallbackCount()
	var out Tensor
	if err := SpecVerifyAttentionDeviceOnly(&q, &k, &v, &out, 2, 4, 2, 1, 8); err != nil {
		t.Fatalf("CPU-owned operands refused under device-only: %v", err)
	}
	if got := specVerifyFallbackCount(); got != before {
		t.Fatalf("CPU backend's own verify counted as a host fallback (%d -> %d)", before, got)
	}
}
