package model

// v41_head_readback_test.go — the witness for the V4.1 head-projection readback
// host-fallback counter (hostFallbackV41HeadReadback, kv.go). stepV41 always
// returns host logits, so every V4.1 decode step on a DeviceMemory backend reads
// the head back to the host (one ~517 KB D2H per token) and must count exactly
// one v41_head_readback fallback. A host session (no device backend) reads
// nothing back and must count zero. The counter is observation-only: the device
// and host arms must return identical logits for the same token.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV41HeadReadbackCountsHostFallback(t *testing.T) {
	m := v41ReducedModel(t)
	if !m.Cfg.IsDeepSeekV41() {
		t.Fatal("fixture is not a V4.1 model")
	}
	be := deviceCapsBackend{compute.Pick("cpu-ref")}
	if !be.Caps().DeviceMemory {
		t.Fatal("deviceCapsBackend did not force DeviceMemory")
	}

	// Assertion 4: the stage is declared in compute.HostFallbackCounts(). Calling
	// the helper at all t.Fatalf's if the stage is not declared, so this line is
	// the declaration witness.
	_ = hostFallbackCount(t, hostFallbackV41HeadReadback)

	// Assertion 1: V4.1 + DeviceMemory backend + Step(id) -> v41_head_readback
	// increases by exactly 1 and the returned logits are non-nil.
	devStep := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
	before := hostFallbackCount(t, hostFallbackV41HeadReadback)
	devLogits := devStep.Step(1)
	if devLogits == nil {
		t.Fatal("device V4.1 Step returned nil logits")
	}
	if got := hostFallbackCount(t, hostFallbackV41HeadReadback) - before; got != 1 {
		t.Fatalf("device V4.1 Step v41_head_readback +%d, want +1", got)
	}

	// Assertion 2: V4.1 + DeviceMemory backend + StepGreedy(id) -> +1, next == 0,
	// logits non-nil (the V4.1 branch ignores greedy and returns host logits).
	devGreedy := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
	before = hostFallbackCount(t, hostFallbackV41HeadReadback)
	next, greedyLogits := devGreedy.StepGreedy(1)
	if next != 0 {
		t.Fatalf("device V4.1 StepGreedy next=%d, want 0 with host logits", next)
	}
	if greedyLogits == nil {
		t.Fatal("device V4.1 StepGreedy returned nil logits")
	}
	if got := hostFallbackCount(t, hostFallbackV41HeadReadback) - before; got != 1 {
		t.Fatalf("device V4.1 StepGreedy v41_head_readback +%d, want +1", got)
	}

	// Assertion 3: negative control. A host session (no backend) must count zero;
	// this fails if the DeviceMemory guard is dropped.
	hostStep := &Session{M: m}
	before = hostFallbackCount(t, hostFallbackV41HeadReadback)
	hostLogits := hostStep.Step(1)
	if hostLogits == nil {
		t.Fatal("host V4.1 Step returned nil logits")
	}
	if got := hostFallbackCount(t, hostFallbackV41HeadReadback) - before; got != 0 {
		t.Fatalf("host V4.1 Step v41_head_readback +%d, want +0 (DeviceMemory guard dropped?)", got)
	}

	// Assertion 5: parity. The counter is observation-only; the device and host
	// arms must produce the same logits for the same token.
	assertSameF32(t, "v41 device vs host Step logits", devLogits, hostLogits)
}
