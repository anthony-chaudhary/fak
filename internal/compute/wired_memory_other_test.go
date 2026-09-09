//go:build !darwin || !cgo

package compute

import (
	"errors"
	"testing"
)

func TestWiredMemoryOther_Stubs(t *testing.T) {
	if s := WireMethodNone.String(); s != "none" {
		t.Fatalf("WireMethodNone.String() = %q, want %q", s, "none")
	}
	if s := WireMethodMachVMWire.String(); s != "mach_vm_wire" {
		t.Fatalf("WireMethodMachVMWire.String() = %q, want %q", s, "mach_vm_wire")
	}
	if s := WireMethodMLock.String(); s != "mlock" {
		t.Fatalf("WireMethodMLock.String() = %q, want %q", s, "mlock")
	}
	if s := WireMethod(999).String(); s != "unknown" {
		t.Fatalf("WireMethod(999).String() = %q, want %q", s, "unknown")
	}

	limits := DarwinWorkingSetLimits()
	if limits.RecommendedMaxWorkingSet != 0 || limits.MaxBufferLength != 0 || limits.HasUnifiedMemory || limits.CurrentAllocatedSize != 0 {
		t.Fatalf("unexpected non-zero limits on non-darwin/non-cgo: %+v", limits)
	}
	if rec := RecommendedMaxWorkingSetSize(); rec != 0 {
		t.Fatalf("RecommendedMaxWorkingSetSize() = %d, want 0", rec)
	}

	_, err := WireMemory(0x10000, 4096)
	if !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("WireMemory err = %v, want %v", err, ErrWiredMemoryUnavailable)
	}

	err = UnwireMemory(0x10000, 4096, WireMethodMLock)
	if !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("UnwireMemory err = %v, want %v", err, ErrWiredMemoryUnavailable)
	}

	_, _, err = IsMemoryWired(0x10000)
	if !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("IsMemoryWired err = %v, want %v", err, ErrWiredMemoryUnavailable)
	}

	_, teardownSpan, err := WireModelSpan(0x10000, 4096)
	if !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("WireModelSpan err = %v, want %v", err, ErrWiredMemoryUnavailable)
	}
	teardownSpan()

	_, teardownBuf, err := WireModelBuffer([]byte("test"))
	if !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("WireModelBuffer err = %v, want %v", err, ErrWiredMemoryUnavailable)
	}
	teardownBuf()
}
