package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestFromKVTransferRecordsResidencyTransition(t *testing.T) {
	e := FromKVTransfer(KVTransfer{
		Direction:   KVOffload,
		SpanDigest:  "span-abc",
		Tokens:      2048,
		ModelID:     "m",
		TokenizerID: "tok",
		FromTier:    TierHBM,
		ToTier:      TierDRAM,
		Owner:       "kvmmu",
		BytesMoved:  1 << 24,
	})
	if e.Plane != PlaneKVTransfer || e.ID.MediaType != MediaKVSpan {
		t.Fatalf("bad kv_transfer identity: %+v", e)
	}
	if e.Residency.Tier != TierDRAM || e.Residency.Owner != "kvmmu" {
		t.Fatalf("residency must record where the span now lives: %+v", e.Residency)
	}
	if e.Labels["direction"] != "offload" || e.Labels["outcome"] != "ok" ||
		e.Labels["from_tier"] != "hbm" || e.Labels["to_tier"] != "dram" {
		t.Fatalf("transition labels missing: %+v", e.Labels)
	}
	if e.Derivation.ModelID != "m" || e.Derivation.PositionMode != PositionPrefixAligned {
		t.Fatalf("KV transfer must bind model + position mode: %+v", e.Derivation)
	}
	if v := KVTransferVerdict(e); v.Kind != LookupHit {
		t.Fatalf("ok transfer should be a HIT, got %s", v.Kind)
	}
}

func TestFromKVTransferCarriesTrustDescriptor(t *testing.T) {
	e := FromKVTransfer(KVTransfer{
		Direction:        KVMigrate,
		SpanDigest:       "span-trust",
		Tokens:           4,
		SerializerID:     "fak-paged-kv-f32-v1",
		ToTier:           TierRemote,
		Lease:            "lease-1",
		SecuritySet:      true,
		Taint:            abi.TaintQuarantined,
		Scope:            abi.ScopeAgent,
		AdmissionVerdict: AdmissionQuarantine,
		AdmittedBy:       "admission-gate",
	})
	if e.Derivation.SerializerID != "fak-paged-kv-f32-v1" {
		t.Fatalf("serializer id = %q", e.Derivation.SerializerID)
	}
	if e.Security.Taint != abi.TaintQuarantined ||
		e.Security.Scope != abi.ScopeAgent ||
		e.Security.AdmissionVerdict != AdmissionQuarantine ||
		e.Security.AdmittedBy != "admission-gate" {
		t.Fatalf("trust descriptor not carried: %+v", e.Security)
	}
	if e.Residency.Lease != "lease-1" {
		t.Fatalf("lease not carried: %+v", e.Residency)
	}
}

// §2.2 parity: failure to restore/load KV is a typed miss or fault, never silent
// recompute.
func TestKVTransferRestoreFaultIsTypedNotSilent(t *testing.T) {
	faulted := FromKVTransfer(KVTransfer{Direction: KVRestore, Outcome: KVTransferFault, FaultReason: "page-in-EIO"})
	if v := KVTransferVerdict(faulted); v.Kind != LookupFault || v.Reason != ReasonResidencyFault {
		t.Fatalf("restore fault must be FAULT(residency_fault), got %+v", v)
	}
	missed := FromKVTransfer(KVTransfer{Direction: KVRestore, Outcome: KVTransferMissed})
	if v := KVTransferVerdict(missed); v.Kind != LookupMiss || v.Reason != ReasonRestoreMiss {
		t.Fatalf("restore miss must be MISS(restore_miss), got %+v", v)
	}
}

func TestKVTransferVerdictRefusesUnknownOutcome(t *testing.T) {
	e := FromKVTransfer(KVTransfer{Direction: KVRestore, Outcome: KVTransferOK})
	e.Labels["outcome"] = "unknown"
	if v := KVTransferVerdict(e); v.Kind != LookupMiss || v.Reason != ReasonAbsent {
		t.Fatalf("unknown outcome should MISS(absent), got %+v", v)
	}
}

func TestKVRouteAndMigrateDirectionsSupported(t *testing.T) {
	for _, d := range []KVTransferDirection{KVRoute, KVMigrate} {
		e := FromKVTransfer(KVTransfer{Direction: d, ToTier: TierRemote, Outcome: KVTransferOK})
		if e.Labels["direction"] != string(d) {
			t.Fatalf("direction %s not recorded", d)
		}
		if v := KVTransferVerdict(e); v.Kind != LookupHit {
			t.Fatalf("%s ok should HIT, got %s", d, v.Kind)
		}
	}
}
