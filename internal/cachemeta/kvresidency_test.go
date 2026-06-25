package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestFromKVResidencyMapsTheTrichotomy proves the abi seam's typed residency outcome
// lowers onto the SAME LookupVerdict trichotomy an in-stream kv_transfer entry does:
// OK -> HIT, MISS -> restore_miss, FAULT -> residency_fault. This is the
// "KVTransferVerdict-shaped" bridge #638 calls for — a remote restore is never a
// silent recompute.
func TestFromKVResidencyMapsTheTrichotomy(t *testing.T) {
	ok := FromKVResidency(abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: "span-A", Positions: 4, BytesMoved: 256})
	if ok.Kind != LookupHit || !ok.CanServe() {
		t.Fatalf("OK -> %+v, want a servable HIT", ok)
	}

	miss := FromKVResidency(abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: "span-A"})
	if miss.Kind != LookupMiss || miss.Reason != ReasonRestoreMiss {
		t.Fatalf("MISS -> kind=%v reason=%v, want miss/restore_miss", miss.Kind, miss.Reason)
	}

	fault := FromKVResidency(abi.KVResidency{Outcome: abi.KVResidencyFault, Digest: "span-A", Reason: "page-in-EIO"})
	if fault.Kind != LookupFault || fault.Reason != ReasonResidencyFault {
		t.Fatalf("FAULT -> kind=%v reason=%v, want fault/residency_fault", fault.Kind, fault.Reason)
	}
}

// TestFromKVResidencyZeroValueFailsClosed pins that an unset outcome (the zero value)
// is a MISS, never an accidental HIT — a half-built residency result cannot be
// mistaken for a successful transfer.
func TestFromKVResidencyZeroValueFailsClosed(t *testing.T) {
	v := FromKVResidency(abi.KVResidency{})
	if v.Kind != LookupMiss || v.Reason != ReasonAbsent {
		t.Fatalf("zero-value residency -> kind=%v reason=%v, want miss/absent (fail-closed)", v.Kind, v.Reason)
	}
	if v.CanServe() {
		t.Fatalf("zero-value residency must not be servable")
	}
}
