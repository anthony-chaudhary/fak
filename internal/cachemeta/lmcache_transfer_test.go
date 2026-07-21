package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestLMCEventKindDirection pins the §2.3 event->Direction mapping the issue
// specifies: append (offload) = KVOffload, lookup (restore/re-materialize) = KVRestore,
// and an unknown kind fails closed to the empty direction.
func TestLMCEventKindDirection(t *testing.T) {
	cases := []struct {
		kind LMCEventKind
		want KVTransferDirection
	}{
		{LMCAppend, KVOffload},
		{LMCLookup, KVRestore},
		{LMCEventKind("bogus"), ""},
	}
	for _, c := range cases {
		if got := c.kind.Direction(); got != c.want {
			t.Fatalf("Direction(%q) = %q, want %q", c.kind, got, c.want)
		}
	}
}

// TestFromLMCAppendMapsOffload witnesses that a healthy lmcache.append() event lowers
// onto a kv_transfer-plane Entry that populates the §4 external-system requirements:
// stable digest, from/to residency tiers, OK outcome, BytesMoved, and the KVOffload
// direction the append maps to.
func TestFromLMCAppendMapsOffload(t *testing.T) {
	e := FromLMCTransfer(LMCTransferEvent{
		Kind:       LMCAppend,
		Backend:    "cpu",
		SpanDigest: "span-abc",
		Tokens:     512,
		ModelID:    "m1",
		FromTier:   TierHBM,
		ToTier:     TierDRAM,
		Owner:      "prefill-node-3",
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
	if e.Residency.Tier != TierDRAM {
		t.Fatalf("to tier = %q, want %q", e.Residency.Tier, TierDRAM)
	}
	if e.Metrics.BytesTransferred != 4096 {
		t.Fatalf("bytes = %d, want 4096", e.Metrics.BytesTransferred)
	}
	if e.Labels["direction"] != string(KVOffload) {
		t.Fatalf("direction label = %q, want %q", e.Labels["direction"], KVOffload)
	}
	if e.Labels["from_tier"] != string(TierHBM) || e.Labels["to_tier"] != string(TierDRAM) {
		t.Fatalf("tier labels = %q/%q", e.Labels["from_tier"], e.Labels["to_tier"])
	}
	if e.Labels["lmc_event"] != string(LMCAppend) {
		t.Fatalf("lmc_event label = %q", e.Labels["lmc_event"])
	}
	if e.Labels["backend"] != "cpu" {
		t.Fatalf("backend label = %q", e.Labels["backend"])
	}
	if e.Labels["outcome"] != string(KVTransferOK) {
		t.Fatalf("outcome label = %q", e.Labels["outcome"])
	}
	if e.Coherence.InvalidationMode != InvalidationExternalRefutation {
		t.Fatalf("coherence = %q, want external refutation", e.Coherence.InvalidationMode)
	}
}

// TestFromLMCLookupMapsRestore witnesses that a successful lmcache.lookup() +
// re-materialization lowers onto the KVRestore direction, restoring the span to a serving
// tier.
func TestFromLMCLookupMapsRestore(t *testing.T) {
	e := FromLMCTransfer(LMCTransferEvent{
		Kind:       LMCLookup,
		Backend:    "disk",
		SpanDigest: "span-def",
		FromTier:   TierDRAM,
		ToTier:     TierHBM,
		Outcome:    KVTransferOK,
	})
	if e.Labels["direction"] != string(KVRestore) {
		t.Fatalf("direction label = %q, want %q", e.Labels["direction"], KVRestore)
	}
	if e.Residency.Tier != TierHBM {
		t.Fatalf("restore to tier = %q, want %q", e.Residency.Tier, TierHBM)
	}
	if e.Labels["lmc_event"] != string(LMCLookup) {
		t.Fatalf("lmc_event label = %q", e.Labels["lmc_event"])
	}
}

// TestLMCTransferVerdictTrichotomy witnesses the ordinary lookup mapping shared with
// KVTransferVerdict: an OK restore serves (Hit), a lookup miss is a typed restore MISS
// (may trigger recompute), and an unidentified event never backs a hit.
func TestLMCTransferVerdictTrichotomy(t *testing.T) {
	ok := LMCTransferVerdict(LMCTransferEvent{
		Kind: LMCLookup, SpanDigest: "s1", ToTier: TierHBM, Outcome: KVTransferOK,
	})
	if !ok.CanServe() {
		t.Fatalf("OK restore: CanServe=false, verdict=%+v", ok)
	}

	missed := LMCTransferVerdict(LMCTransferEvent{
		Kind: LMCLookup, SpanDigest: "s1", Outcome: KVTransferMissed,
	})
	if missed.Kind != LookupMiss || missed.Reason != ReasonRestoreMiss {
		t.Fatalf("missed lookup: got %s/%s, want miss/restore_miss", missed.Kind, missed.Reason)
	}

	unnamed := LMCTransferVerdict(LMCTransferEvent{Kind: LMCLookup, Outcome: KVTransferOK})
	if unnamed.Kind != LookupMiss || unnamed.CanServe() {
		t.Fatalf("unidentified event served: %+v", unnamed)
	}

	unknownKind := LMCTransferVerdict(LMCTransferEvent{Kind: "bogus", SpanDigest: "s1"})
	if unknownKind.Kind != LookupMiss {
		t.Fatalf("unknown kind: got %s, want miss", unknownKind.Kind)
	}
}

// TestLMCTransferFaultPath witnesses the fault path both ways: a plain transfer error
// is a typed residency_fault FAULT (never a silent recompute), and a fault inside a DSA
// path escalates to QUARANTINE with the entry itself quarantined (§1.2).
func TestLMCTransferFaultPath(t *testing.T) {
	plain := LMCTransferVerdict(LMCTransferEvent{
		Kind: LMCLookup, SpanDigest: "s2", ToTier: TierHBM,
		Outcome: KVTransferFault, FaultReason: "page_in_eio",
	})
	if plain.Kind != LookupFault || plain.Reason != ReasonResidencyFault {
		t.Fatalf("plain fault: got %s/%s, want fault/residency_fault", plain.Kind, plain.Reason)
	}
	if plain.CanServe() {
		t.Fatalf("faulted transfer served")
	}

	dsa := LMCTransferVerdict(LMCTransferEvent{
		Kind: LMCLookup, SpanDigest: "s2", ToTier: TierHBM,
		Outcome: KVTransferFault, FaultReason: "page_in_eio", DSAPath: true,
	})
	if dsa.Kind != LookupQuarantine || dsa.Reason != ReasonResidencyFault {
		t.Fatalf("dsa fault: got %s/%s, want quarantine/residency_fault", dsa.Kind, dsa.Reason)
	}
	if dsa.Entry.Security.Taint != abi.TaintQuarantined || dsa.Entry.Security.AdmissionVerdict != AdmissionQuarantine {
		t.Fatalf("dsa fault entry not quarantined: taint=%q admit=%q",
			dsa.Entry.Security.Taint, dsa.Entry.Security.AdmissionVerdict)
	}
	if dsa.Entry.Labels["fault_reason"] != "page_in_eio" {
		t.Fatalf("fault_reason label = %q", dsa.Entry.Labels["fault_reason"])
	}
}

// TestLMCResetTargets witnesses the adapter responding to fak's
// ExternalInvalidationDirective (§3.2, §4.5): a directive naming a valid span becomes a
// precise exact-span reset, while a span-less directive becomes a whole-store reset
// fallback rather than silently resetting nothing.
func TestLMCResetTargets(t *testing.T) {
	if got := LMCResetTargets(nil); got != nil {
		t.Fatalf("nil dirs = %+v, want nil", got)
	}

	dirs := []ExternalInvalidationDirective{
		{
			Kind:   ExternalInvalidateKVSpan,
			Entry:  EntryID{Digest: "span-xyz", MediaType: MediaKVSpan, Length: 128, Unit: UnitPositions},
			Reason: "poisoned_kv",
		},
		{
			Kind:   ExternalInvalidateKVSpan,
			Entry:  EntryID{}, // no span identity -> whole-store reset fallback
			Reason: "coarse_reset",
		},
	}
	got := LMCResetTargets(dirs)
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
