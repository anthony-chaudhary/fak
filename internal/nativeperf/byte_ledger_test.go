package nativeperf

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestReconcilePhaseBytes_ExactConservation(t *testing.T) {
	// Weights: 4 GiB, Quant: 256 MiB, KV: 512 MiB, Recurrent: 0, Activation: 128 MiB, Copies: 0
	logical := LogicalByteBreakdown{
		Weights:             ByteValue{Bytes: 4 * 1024 * 1024 * 1024, Provenance: ProvenanceDeclared, SourceDetail: "Qwen3.8-7B-INT4-weights"},
		QuantMetadata:       ByteValue{Bytes: 256 * 1024 * 1024, Provenance: ProvenanceDeclared, SourceDetail: "group-scales-and-zeros"},
		KVCache:             ByteValue{Bytes: 512 * 1024 * 1024, Provenance: ProvenanceEstimated, SourceDetail: "p32-t64-dim-calc"},
		ActivationWorkspace: ByteValue{Bytes: 128 * 1024 * 1024, Provenance: ProvenanceEstimated, SourceDetail: "intermediate-activations"},
	}

	totLogical := logical.TotalBytes()
	if totLogical != (4*1024+256+512+128)*1024*1024 {
		t.Fatalf("unexpected logical total: %d", totLogical)
	}

	// Physical HBM reads match exactly, writes match activations + KV updates
	tiers := []PhysicalTierTraffic{
		{
			Tier:       TierHBM,
			ReadBytes:  ByteValue{Bytes: (4*1024+256)*1024*1024 + 512*1024*1024, Provenance: ProvenanceMeasured, SourceDetail: "ncu-dram-read-bytes"},
			WriteBytes: ByteValue{Bytes: 128 * 1024 * 1024, Provenance: ProvenanceMeasured, SourceDetail: "ncu-dram-write-bytes"},
		},
		{
			Tier:       TierL2,
			ReadBytes:  ByteValue{Bytes: (4*1024 + 256 + 512) * 1024 * 1024, Provenance: ProvenanceMeasured, SourceDetail: "l2-read-hit-plus-miss"},
			WriteBytes: ByteValue{Bytes: 128 * 1024 * 1024, Provenance: ProvenanceMeasured, SourceDetail: "l2-write-bytes"},
			HitBytes:   ByteValue{Bytes: 1024 * 1024 * 1024, Provenance: ProvenanceMeasured},
			MissBytes:  ByteValue{Bytes: (4*1024 + 256 + 512 - 1024) * 1024 * 1024, Provenance: ProvenanceMeasured},
		},
	}

	ledger, err := ReconcilePhaseBytes(PhaseDecode, logical, tiers, TierHBM, 5.0)
	if err != nil {
		t.Fatalf("ReconcilePhaseBytes failed: %v", err)
	}

	if !ledger.Reconciled {
		t.Errorf("expected reconciled=true, got false, notes: %v", ledger.Notes)
	}
	if ledger.TotalLogicalBytes != totLogical {
		t.Errorf("TotalLogicalBytes = %d, want %d", ledger.TotalLogicalBytes, totLogical)
	}
	if ledger.AmplificationFactor != 1.0 {
		t.Errorf("AmplificationFactor = %f, want 1.0", ledger.AmplificationFactor)
	}
	if ledger.ResidualUnknownBytes != 0 {
		t.Errorf("ResidualUnknownBytes = %d, want 0", ledger.ResidualUnknownBytes)
	}
	if ledger.UnexplainedPercent != 0.0 {
		t.Errorf("UnexplainedPercent = %f, want 0.0", ledger.UnexplainedPercent)
	}

	// Verify L2 hit rate calculation
	var l2Tier *PhysicalTierTraffic
	for i := range ledger.PhysicalTiers {
		if ledger.PhysicalTiers[i].Tier == TierL2 {
			l2Tier = &ledger.PhysicalTiers[i]
			break
		}
	}
	if l2Tier == nil || l2Tier.HitRate == nil {
		t.Fatalf("expected L2 tier with derived hit rate")
	}
	expectedHitRate := float64(1024*1024*1024) / float64((4*1024+256+512)*1024*1024)
	if math.Abs(*l2Tier.HitRate-expectedHitRate) > 1e-6 {
		t.Errorf("L2 hit rate = %f, want %f", *l2Tier.HitRate, expectedHitRate)
	}
}

func TestReconcilePhaseBytes_AmplificationAttribution(t *testing.T) {
	// Logical demand = 1,000,000 bytes
	logical := LogicalByteBreakdown{
		Weights: ByteValue{Bytes: 1_000_000, Provenance: ProvenanceDeclared, SourceDetail: "model-weights"},
	}

	// Due to 128-byte cache line granularity, uncoalesced strides cause 1,500,000 bytes transferred
	tiers := []PhysicalTierTraffic{
		{
			Tier:       TierHBM,
			ReadBytes:  ByteValue{Bytes: 1_500_000, Provenance: ProvenanceMeasured, SourceDetail: "dram-read"},
			WriteBytes: ByteValue{Bytes: 0, Provenance: ProvenanceMeasured},
		},
	}

	// With tolerance 10%, 50% amplification should flag unreconciled
	ledger, err := ReconcilePhaseBytes(PhasePrefill, logical, tiers, TierHBM, 10.0)
	if err != nil {
		t.Fatalf("ReconcilePhaseBytes failed: %v", err)
	}

	if ledger.AmplificationFactor != 1.5 {
		t.Errorf("AmplificationFactor = %f, want 1.5", ledger.AmplificationFactor)
	}
	if ledger.ResidualUnknownBytes != 500_000 {
		t.Errorf("ResidualUnknownBytes = %d, want 500000", ledger.ResidualUnknownBytes)
	}
	if ledger.UnexplainedPercent != 50.0 {
		t.Errorf("UnexplainedPercent = %f, want 50.0", ledger.UnexplainedPercent)
	}
	if ledger.Reconciled {
		t.Errorf("expected reconciled=false due to exceeding tolerance")
	}
	if len(ledger.Notes) == 0 || !strings.Contains(ledger.Notes[0], "exceeds tolerance threshold") {
		t.Errorf("expected tolerance warning note, got: %v", ledger.Notes)
	}
}

func TestBuildByteReconciliationReceipt_MultiPhaseConservation(t *testing.T) {
	prefillLogical := LogicalByteBreakdown{
		Weights:             ByteValue{Bytes: 2_000_000_000, Provenance: ProvenanceDeclared},
		ActivationWorkspace: ByteValue{Bytes: 500_000_000, Provenance: ProvenanceEstimated},
	}
	prefillTiers := []PhysicalTierTraffic{
		{
			Tier:       TierVRAM,
			ReadBytes:  ByteValue{Bytes: 2_000_000_000, Provenance: ProvenanceMeasured},
			WriteBytes: ByteValue{Bytes: 500_000_000, Provenance: ProvenanceMeasured},
		},
		{
			Tier:       TierPCIe,
			ReadBytes:  ByteValue{Bytes: 100_000_000, Provenance: ProvenanceMeasured},
			WriteBytes: ByteValue{Bytes: 0, Provenance: ProvenanceMeasured},
		},
	}
	prefillLedger, err := ReconcilePhaseBytes(PhasePrefill, prefillLogical, prefillTiers, TierVRAM, 5.0)
	if err != nil {
		t.Fatalf("prefill ledger failed: %v", err)
	}

	decodeLogical := LogicalByteBreakdown{
		Weights: ByteValue{Bytes: 2_000_000_000, Provenance: ProvenanceDeclared},
		KVCache: ByteValue{Bytes: 100_000_000, Provenance: ProvenanceEstimated},
	}
	decodeTiers := []PhysicalTierTraffic{
		{
			Tier:       TierVRAM,
			ReadBytes:  ByteValue{Bytes: 2_000_000_000, Provenance: ProvenanceMeasured},
			WriteBytes: ByteValue{Bytes: 100_000_000, Provenance: ProvenanceMeasured},
		},
	}
	decodeLedger, err := ReconcilePhaseBytes(PhaseDecode, decodeLogical, decodeTiers, TierVRAM, 5.0)
	if err != nil {
		t.Fatalf("decode ledger failed: %v", err)
	}

	receipt, err := BuildByteReconciliationReceipt(
		"fak-native",
		"metal",
		"qwen38_metal",
		false,
		[]PhaseByteLedger{prefillLedger, decodeLedger},
		10.0,
	)
	if err != nil {
		t.Fatalf("BuildByteReconciliationReceipt failed: %v", err)
	}

	if err := receipt.Validate(10.0); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	if !receipt.ConservationPassed {
		t.Errorf("expected conservation_passed=true")
	}

	// Total logical: (2000+500) + (2000+100) = 4600M bytes
	wantLogical := uint64(4_600_000_000)
	if receipt.TotalLogicalBytes != wantLogical {
		t.Errorf("TotalLogicalBytes = %d, want %d", receipt.TotalLogicalBytes, wantLogical)
	}

	// Total physical: (2500M + 100M pcie) + 2100M = 4700M bytes
	wantPhysical := uint64(4_700_000_000)
	if receipt.TotalPhysicalBytes != wantPhysical {
		t.Errorf("TotalPhysicalBytes = %d, want %d", receipt.TotalPhysicalBytes, wantPhysical)
	}

	// Global amplification: 4700 / 4600 = ~1.0217
	wantAmp := float64(wantPhysical) / float64(wantLogical)
	if math.Abs(receipt.GlobalAmplification-wantAmp) > 1e-4 {
		t.Errorf("GlobalAmplification = %f, want %f", receipt.GlobalAmplification, wantAmp)
	}

	// Check JSON serialization roundtrip
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var roundtrip ByteReconciliationReceipt
	if err := json.Unmarshal(data, &roundtrip); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if roundtrip.Schema != ByteLedgerSchema {
		t.Errorf("roundtrip schema = %q, want %q", roundtrip.Schema, ByteLedgerSchema)
	}
	if len(roundtrip.Phases) != 2 {
		t.Errorf("roundtrip phases count = %d, want 2", len(roundtrip.Phases))
	}
}

func TestBuildByteReconciliationReceipt_FailsClosedOnFallbackOrNonNative(t *testing.T) {
	dummyPhase := PhaseByteLedger{
		Phase:              PhaseDecode,
		PrimaryTier:        TierHBM,
		TotalLogicalBytes:  100,
		TotalPhysicalBytes: 100,
		Reconciled:         true,
	}

	// Non-native engine must fail
	if _, err := BuildByteReconciliationReceipt("llama.cpp", "metal", "qwen", false, []PhaseByteLedger{dummyPhase}, 5.0); err == nil {
		t.Fatal("expected error for non-native engine")
	}

	// Fallback active must fail
	if _, err := BuildByteReconciliationReceipt("fak-native", "metal", "qwen", true, []PhaseByteLedger{dummyPhase}, 5.0); err == nil {
		t.Fatal("expected error for fallback-active execution")
	}

	// Empty phases must fail
	if _, err := BuildByteReconciliationReceipt("fak-native", "metal", "qwen", false, nil, 5.0); err == nil {
		t.Fatal("expected error for empty phases")
	}
}

func TestReconcilePhaseBytes_ZeroDemandBoundary(t *testing.T) {
	// Zero logical and zero physical should result in 1.0 amplification
	logical := LogicalByteBreakdown{}
	tiers := []PhysicalTierTraffic{
		{Tier: TierHostDRAM, ReadBytes: ByteValue{Bytes: 0}, WriteBytes: ByteValue{Bytes: 0}},
	}

	ledger, err := ReconcilePhaseBytes(PhaseQueueAdmission, logical, tiers, TierHostDRAM, 0)
	if err != nil {
		t.Fatalf("ReconcilePhaseBytes failed: %v", err)
	}

	if ledger.AmplificationFactor != 1.0 {
		t.Errorf("AmplificationFactor for 0/0 = %f, want 1.0", ledger.AmplificationFactor)
	}
	if ledger.ResidualUnknownBytes != 0 {
		t.Errorf("ResidualUnknownBytes for 0/0 = %d, want 0", ledger.ResidualUnknownBytes)
	}

	// Zero logical with positive physical traffic is +Inf amplification
	tiersWithTraffic := []PhysicalTierTraffic{
		{Tier: TierHostDRAM, ReadBytes: ByteValue{Bytes: 5000}, WriteBytes: ByteValue{Bytes: 0}},
	}
	ledgerInf, err := ReconcilePhaseBytes(PhaseQueueAdmission, logical, tiersWithTraffic, TierHostDRAM, 0)
	if err != nil {
		t.Fatalf("ReconcilePhaseBytes failed: %v", err)
	}
	if !math.IsInf(ledgerInf.AmplificationFactor, 1) {
		t.Errorf("AmplificationFactor for positive/0 = %f, want +Inf", ledgerInf.AmplificationFactor)
	}
}
