package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestMooncakeEventKindDirection pins the §2.2 event->Direction mapping the issue
// specifies: prefill->decode = migrate, distributed lookup = route, remote
// materialization = restore, and an unknown kind fails closed to the empty direction.
func TestMooncakeEventKindDirection(t *testing.T) {
	cases := []struct {
		kind MooncakeEventKind
		want KVTransferDirection
	}{
		{MooncakeMigrate, KVMigrate},
		{MooncakeRoute, KVRoute},
		{MooncakeRestore, KVRestore},
		{MooncakeEventKind("bogus"), ""},
	}
	for _, c := range cases {
		if got := c.kind.Direction(); got != c.want {
			t.Fatalf("Direction(%q) = %q, want %q", c.kind, got, c.want)
		}
	}
}

// TestFromMooncakeTransferMapsEntry witnesses that a healthy Transfer Engine event
// lowers onto a kv_transfer-plane Entry that populates the §4 external-system
// requirements: stable digest, from/to residency tiers, OK outcome, and BytesMoved.
func TestFromMooncakeTransferMapsEntry(t *testing.T) {
	e := FromMooncakeTransfer(MooncakeTransferEvent{
		Kind:       MooncakeMigrate,
		Backend:    "rdma",
		SpanDigest: "span-abc",
		Tokens:     512,
		ModelID:    "m1",
		FromTier:   TierHBM,
		ToTier:     TierRemote,
		Owner:      "decode-node-7",
		Outcome:    KVTransferOK,
		BytesMoved: 4096,
	})

	if e.Plane != PlaneKVTransfer {
		t.Fatalf("plane = %q, want %q", e.Plane, PlaneKVTransfer)
	}
	if e.ID.Digest != "span-abc" {
		t.Fatalf("digest = %q, want span-abc", e.ID.Digest)
	}
	if e.ID.MediaType != MediaKVSpan {
		t.Fatalf("media = %q, want %q", e.ID.MediaType, MediaKVSpan)
	}
	if e.Residency.Tier != TierRemote {
		t.Fatalf("to tier = %q, want %q", e.Residency.Tier, TierRemote)
	}
	if e.Metrics.BytesTransferred != 4096 {
		t.Fatalf("bytes = %d, want 4096", e.Metrics.BytesTransferred)
	}
	if e.Labels["direction"] != string(KVMigrate) {
		t.Fatalf("direction label = %q, want %q", e.Labels["direction"], KVMigrate)
	}
	if e.Labels["from_tier"] != string(TierHBM) || e.Labels["to_tier"] != string(TierRemote) {
		t.Fatalf("tier labels = %q/%q", e.Labels["from_tier"], e.Labels["to_tier"])
	}
	if e.Labels["mooncake_event"] != string(MooncakeMigrate) {
		t.Fatalf("mooncake_event label = %q", e.Labels["mooncake_event"])
	}
	if e.Labels["backend"] != "rdma" {
		t.Fatalf("backend label = %q", e.Labels["backend"])
	}
	if e.Labels["outcome"] != string(KVTransferOK) {
		t.Fatalf("outcome label = %q", e.Labels["outcome"])
	}
	if e.Coherence.InvalidationMode != InvalidationExternalRefutation {
		t.Fatalf("coherence = %q, want external refutation", e.Coherence.InvalidationMode)
	}
}

// TestMooncakeTransferVerdictTrichotomy witnesses the ordinary lookup mapping shared
// with KVTransferVerdict: OK routes serve (Hit), a missed distributed lookup is a
// typed restore MISS, and an unidentified event never backs a hit.
func TestMooncakeTransferVerdictTrichotomy(t *testing.T) {
	ok := MooncakeTransferVerdict(MooncakeTransferEvent{
		Kind: MooncakeRoute, SpanDigest: "s1", ToTier: TierRemote, Outcome: KVTransferOK,
	})
	if !ok.CanServe() {
		t.Fatalf("OK route: CanServe=false, verdict=%+v", ok)
	}

	missed := MooncakeTransferVerdict(MooncakeTransferEvent{
		Kind: MooncakeRoute, SpanDigest: "s1", Outcome: KVTransferMissed,
	})
	if missed.Kind != LookupMiss || missed.Reason != ReasonRestoreMiss {
		t.Fatalf("missed lookup: got %s/%s, want miss/restore_miss", missed.Kind, missed.Reason)
	}

	unnamed := MooncakeTransferVerdict(MooncakeTransferEvent{Kind: MooncakeRestore, Outcome: KVTransferOK})
	if unnamed.Kind != LookupMiss || unnamed.CanServe() {
		t.Fatalf("unidentified event served: %+v", unnamed)
	}

	unknownKind := MooncakeTransferVerdict(MooncakeTransferEvent{Kind: "bogus", SpanDigest: "s1"})
	if unknownKind.Kind != LookupMiss {
		t.Fatalf("unknown kind: got %s, want miss", unknownKind.Kind)
	}
}

// TestMooncakeTransferFaultPath witnesses the fault path both ways: a plain fault is a
// typed residency_fault FAULT (never a silent recompute), and a fault inside a DSA
// path escalates to QUARANTINE with the entry itself quarantined (§1.2).
func TestMooncakeTransferFaultPath(t *testing.T) {
	plain := MooncakeTransferVerdict(MooncakeTransferEvent{
		Kind: MooncakeRestore, SpanDigest: "s2", ToTier: TierRemote,
		Outcome: KVTransferFault, FaultReason: "rdma_timeout",
	})
	if plain.Kind != LookupFault || plain.Reason != ReasonResidencyFault {
		t.Fatalf("plain fault: got %s/%s, want fault/residency_fault", plain.Kind, plain.Reason)
	}
	if plain.CanServe() {
		t.Fatalf("faulted transfer served")
	}

	dsa := MooncakeTransferVerdict(MooncakeTransferEvent{
		Kind: MooncakeRestore, SpanDigest: "s2", ToTier: TierRemote,
		Outcome: KVTransferFault, FaultReason: "rdma_timeout", DSAPath: true,
	})
	if dsa.Kind != LookupQuarantine || dsa.Reason != ReasonResidencyFault {
		t.Fatalf("dsa fault: got %s/%s, want quarantine/residency_fault", dsa.Kind, dsa.Reason)
	}
	if dsa.Entry.Security.Taint != abi.TaintQuarantined || dsa.Entry.Security.AdmissionVerdict != AdmissionQuarantine {
		t.Fatalf("dsa fault entry not quarantined: taint=%q admit=%q",
			dsa.Entry.Security.Taint, dsa.Entry.Security.AdmissionVerdict)
	}
	if dsa.Entry.Labels["fault_reason"] != "rdma_timeout" {
		t.Fatalf("fault_reason label = %q", dsa.Entry.Labels["fault_reason"])
	}
}

// TestMooncakeResetTargets witnesses the adapter responding to fak's
// ExternalInvalidationDirective (§3.2, §4.5): a directive naming a valid span becomes
// a precise exact-span reset, while a span-less directive becomes a whole-store reset
// fallback rather than silently resetting nothing.
func TestMooncakeResetTargets(t *testing.T) {
	if got := MooncakeResetTargets(nil); got != nil {
		t.Fatalf("nil dirs = %+v, want nil", got)
	}

	dirs := []ExternalInvalidationDirective{
		{
			Kind:  ExternalInvalidateKVSpan,
			Entry: EntryID{Digest: "span-xyz", MediaType: MediaKVSpan, Length: 128, Unit: UnitPositions},
			Reason: "poisoned_kv",
		},
		{
			Kind:   ExternalInvalidateKVSpan,
			Entry:  EntryID{}, // no span identity -> whole-store reset fallback
			Reason: "coarse_reset",
		},
	}
	got := MooncakeResetTargets(dirs)
	if len(got) != 2 {
		t.Fatalf("got %d targets, want 2", len(got))
	}
	if got[0].WholeReset || got[0].SpanDigest != "span-xyz" || got[0].Length != 128 {
		t.Fatalf("exact target wrong: %+v", got[0])
	}
	if got[0].Reason != "poisoned_kv" {
		t.Fatalf("exact target reason = %q", got[0].Reason)
	}
	if !got[1].WholeReset || got[1].SpanDigest != "" {
		t.Fatalf("fallback target not whole-reset: %+v", got[1])
	}
}
