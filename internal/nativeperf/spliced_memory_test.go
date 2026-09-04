package nativeperf

import (
	"encoding/json"
	"testing"
)

func makeCoherentRecord(secName, passID string, passIdx int, runID, devID string, startNS, endNS int64, alloc, freed, peak uint64) VendorMemoryRecord {
	rec := VendorMemoryRecord{
		SectionName:      secName,
		PassID:           passID,
		PassIndex:        passIdx,
		RunID:            runID,
		DeviceID:         devID,
		StartTimestampNS: startNS,
		EndTimestampNS:   endNS,
		AllocatedBytes:   alloc,
		FreedBytes:       freed,
		PeakBytes:        peak,
	}
	rec.Stamp = MemorySectionStamp{
		SectionID: secName + "-stamp",
		PassID:    passID,
		PassIndex: passIdx,
		RunID:     runID,
		DeviceID:  devID,
		Hash:      ComputeSectionStamp(rec),
	}
	return rec
}

func TestSplicedMemory_Coherent(t *testing.T) {
	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 2,
		Sections: []VendorMemoryRecord{
			makeCoherentRecord("alloc-weights", "pass-0", 0, "run-42", "gpu-0", 1000, 2000, 10000, 2000, 8000),
			makeCoherentRecord("kv-cache", "pass-0", 0, "run-42", "gpu-0", 2000, 3000, 5000, 1000, 4000),
			makeCoherentRecord("alloc-weights", "pass-1", 1, "run-42", "gpu-0", 3500, 4500, 10000, 2000, 8000),
			makeCoherentRecord("kv-cache", "pass-1", 1, "run-42", "gpu-0", 4500, 5500, 5000, 1000, 4000),
		},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictCoherent {
		t.Fatalf("expected verdict %s, got %s (reasons: %v)", MemoryVerdictCoherent, res.Verdict, res.Reasons)
	}
	if len(res.Reasons) != 0 {
		t.Fatalf("expected zero reasons, got %v", res.Reasons)
	}
	if res.PassCount != 2 {
		t.Fatalf("expected 2 passes, got %d", res.PassCount)
	}
}

func TestSplicedMemory_SplicedTamperedHash(t *testing.T) {
	rec := makeCoherentRecord("alloc", "pass-0", 0, "run-42", "gpu-0", 1000, 2000, 10000, 1000, 9000)
	// Tamper with bytes after stamping
	rec.AllocatedBytes = 9999999

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 1,
		Sections:    []VendorMemoryRecord{rec},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_SplicedCrossRun(t *testing.T) {
	// Section from run-99 spliced into run-42 evidence
	rec := makeCoherentRecord("alloc", "pass-0", 0, "run-99", "gpu-0", 1000, 2000, 10000, 1000, 9000)

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 1,
		Sections:    []VendorMemoryRecord{rec},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s for cross-run splice, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_SplicedCrossDevice(t *testing.T) {
	rec := makeCoherentRecord("alloc", "pass-0", 0, "run-42", "gpu-1", 1000, 2000, 10000, 1000, 9000)

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 1,
		Sections:    []VendorMemoryRecord{rec},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s for cross-device splice, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_SplicedContradictoryPassID(t *testing.T) {
	// Same pass index (0) carries contradictory pass IDs ("pass-A" and "pass-B")
	rec1 := makeCoherentRecord("alloc-weights", "pass-A", 0, "run-42", "gpu-0", 1000, 2000, 10000, 1000, 9000)
	rec2 := makeCoherentRecord("kv-cache", "pass-B", 0, "run-42", "gpu-0", 2000, 3000, 5000, 500, 4500)

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 1,
		Sections:    []VendorMemoryRecord{rec1, rec2},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s for contradictory pass ID, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_SplicedTemporalInversion(t *testing.T) {
	// End timestamp precedes start timestamp
	rec := makeCoherentRecord("alloc", "pass-0", 0, "run-42", "gpu-0", 5000, 2000, 10000, 1000, 9000)

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 1,
		Sections:    []VendorMemoryRecord{rec},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s for temporal inversion, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_SplicedCrossPassRegression(t *testing.T) {
	// Pass 1 starts at 1500 ns, but Pass 0 ended at 2000 ns
	rec0 := makeCoherentRecord("alloc", "pass-0", 0, "run-42", "gpu-0", 1000, 2000, 10000, 1000, 9000)
	rec1 := makeCoherentRecord("alloc", "pass-1", 1, "run-42", "gpu-0", 1500, 2500, 10000, 1000, 9000)

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 2,
		Sections:    []VendorMemoryRecord{rec0, rec1},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s for cross-pass temporal regression, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_SplicedPassSequenceReversal(t *testing.T) {
	// Pass 1 listed before Pass 0
	rec1 := makeCoherentRecord("alloc", "pass-1", 1, "run-42", "gpu-0", 2500, 3500, 10000, 1000, 9000)
	rec0 := makeCoherentRecord("alloc", "pass-0", 0, "run-42", "gpu-0", 1000, 2000, 10000, 1000, 9000)

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 2,
		Sections:    []VendorMemoryRecord{rec1, rec0},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s for out-of-order pass stream, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_SplicedReusedStamp(t *testing.T) {
	// Pass 1 duplicates pass 0's stamp hash
	rec0 := makeCoherentRecord("alloc", "pass-0", 0, "run-42", "gpu-0", 1000, 2000, 10000, 1000, 9000)
	rec1 := makeCoherentRecord("alloc", "pass-1", 1, "run-42", "gpu-0", 2100, 3000, 10000, 1000, 9000)
	rec1.Stamp.Hash = rec0.Stamp.Hash // force duplicate

	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-42",
		DeviceID:    "gpu-0",
		TotalPasses: 2,
		Sections:    []VendorMemoryRecord{rec0, rec1},
	}

	res := CheckVendorMemoryEvidence(ev)
	if res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected verdict %s for reused stamp hash, got %s", MemoryVerdictSpliced, res.Verdict)
	}
}

func TestSplicedMemory_Unknown(t *testing.T) {
	// 1. Nil
	resNil := CheckVendorMemoryEvidence(nil)
	if resNil.Verdict != MemoryVerdictUnknown {
		t.Fatalf("expected UNKNOWN for nil")
	}

	// 2. Zero sections
	evEmpty := &VendorMemoryEvidence{
		Schema:   VendorMemoryEvidenceSchema,
		Vendor:   "nvidia",
		Sections: []VendorMemoryRecord{},
	}
	resEmpty := CheckVendorMemoryEvidence(evEmpty)
	if resEmpty.Verdict != MemoryVerdictUnknown {
		t.Fatalf("expected UNKNOWN for empty sections")
	}

	// 3. No vendor
	evNoVendor := &VendorMemoryEvidence{
		Schema:   VendorMemoryEvidenceSchema,
		Sections: []VendorMemoryRecord{{SectionName: "alloc"}},
	}
	resNoVendor := CheckVendorMemoryEvidence(evNoVendor)
	if resNoVendor.Verdict != MemoryVerdictUnknown {
		t.Fatalf("expected UNKNOWN for missing vendor")
	}

	// 4. Sections lack stamps
	evNoStamps := &VendorMemoryEvidence{
		Schema:   VendorMemoryEvidenceSchema,
		Vendor:   "nvidia",
		Sections: []VendorMemoryRecord{{SectionName: "alloc", PassIndex: 0, PassID: "p0"}},
	}
	resNoStamps := CheckVendorMemoryEvidence(evNoStamps)
	if resNoStamps.Verdict != MemoryVerdictUnknown {
		t.Fatalf("expected UNKNOWN when stamps are absent")
	}
}

func TestSplicedMemory_JSONRoundtrip(t *testing.T) {
	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "apple",
		RunID:       "run-metal-1",
		DeviceID:    "m3-max",
		TotalPasses: 1,
		Sections: []VendorMemoryRecord{
			makeCoherentRecord("metal-heaps", "pass-0", 0, "run-metal-1", "m3-max", 1000, 2000, 2048, 512, 1536),
		},
	}

	data, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	res := ParseAndCheckVendorMemoryEvidence(data)
	if res.Verdict != MemoryVerdictCoherent {
		t.Fatalf("expected COHERENT, got %s (reasons: %v)", res.Verdict, res.Reasons)
	}

	// Test unparseable JSON
	badRes := ParseAndCheckVendorMemoryEvidence([]byte("{bad-json"))
	if badRes.Verdict != MemoryVerdictUnknown {
		t.Fatalf("expected UNKNOWN for bad JSON")
	}
}

func TestSplicedMemory_VerifySectionStamp_DirectFidelity(t *testing.T) {
	rec := BindVendorMemoryRecord("weights", "pass-0", 0, "run-1", "gpu-0", 1000, 2000, 4096, 1024, 3072)
	if !VerifySectionStamp(rec) {
		t.Fatalf("expected VerifySectionStamp to pass on pristine record")
	}

	// 1. Missing hash
	noHash := rec
	noHash.Stamp.Hash = ""
	if VerifySectionStamp(noHash) {
		t.Fatalf("expected failure for missing hash")
	}

	// 2. Mismatched PassID
	badPass := rec
	badPass.Stamp.PassID = "pass-different"
	if VerifySectionStamp(badPass) {
		t.Fatalf("expected failure for mismatched pass_id")
	}

	// 3. Mismatched PassIndex
	badIdx := rec
	badIdx.Stamp.PassIndex = 99
	if VerifySectionStamp(badIdx) {
		t.Fatalf("expected failure for mismatched pass_index")
	}

	// 4. Mismatched RunID
	badRun := rec
	badRun.Stamp.RunID = "run-different"
	if VerifySectionStamp(badRun) {
		t.Fatalf("expected failure for mismatched run_id")
	}

	// 5. Mismatched DeviceID
	badDev := rec
	badDev.Stamp.DeviceID = "gpu-different"
	if VerifySectionStamp(badDev) {
		t.Fatalf("expected failure for mismatched device_id")
	}

	// 6. Mutated content
	mutated := rec
	mutated.AllocatedBytes = 999999
	if VerifySectionStamp(mutated) {
		t.Fatalf("expected failure for mutated payload")
	}
}

func TestSplicedMemory_MarshalVendorMemoryEvidence(t *testing.T) {
	ev := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		RunID:       "run-1",
		TotalPasses: 1,
		Sections: []VendorMemoryRecord{
			BindVendorMemoryRecord("alloc", "p0", 0, "run-1", "gpu-0", 100, 200, 1000, 100, 900),
		},
	}

	data, err := MarshalVendorMemoryEvidence(ev)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	parsed, err := ParseVendorMemoryEvidence(data)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if parsed.Vendor != ev.Vendor || len(parsed.Sections) != 1 {
		t.Fatalf("mismatched parsed evidence: %+v", parsed)
	}

	// Nil evidence error check
	if _, err := MarshalVendorMemoryEvidence(nil); err == nil {
		t.Fatalf("expected error marshaling nil evidence")
	}

	// Empty bytes error check
	if _, err := ParseVendorMemoryEvidence(nil); err == nil {
		t.Fatalf("expected error parsing empty bytes")
	}
}

func TestSplicedMemory_AdditionalSplicedAndUnknownConditions(t *testing.T) {
	// 1. Invalid schema -> UNKNOWN
	badSchema := &VendorMemoryEvidence{
		Schema:   "unrecognized-schema",
		Vendor:   "nvidia",
		Sections: []VendorMemoryRecord{BindVendorMemoryRecord("alloc", "p0", 0, "run-1", "gpu-0", 100, 200, 1000, 100, 900)},
	}
	if res := CheckVendorMemoryEvidence(badSchema); res.Verdict != MemoryVerdictUnknown {
		t.Fatalf("expected UNKNOWN for bad schema, got %s", res.Verdict)
	}

	// 2. Empty section name -> SPLICED
	emptySecName := &VendorMemoryEvidence{
		Schema: VendorMemoryEvidenceSchema,
		Vendor: "nvidia",
		Sections: []VendorMemoryRecord{
			BindVendorMemoryRecord("", "p0", 0, "run-1", "gpu-0", 100, 200, 1000, 100, 900),
		},
	}
	if res := CheckVendorMemoryEvidence(emptySecName); res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected SPLICED for empty section name, got %s", res.Verdict)
	}

	// 3. Empty pass ID -> SPLICED
	emptyPassID := &VendorMemoryEvidence{
		Schema: VendorMemoryEvidenceSchema,
		Vendor: "nvidia",
		Sections: []VendorMemoryRecord{
			BindVendorMemoryRecord("alloc", "", 0, "run-1", "gpu-0", 100, 200, 1000, 100, 900),
		},
	}
	if res := CheckVendorMemoryEvidence(emptyPassID); res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected SPLICED for empty pass ID, got %s", res.Verdict)
	}

	// 4. Negative pass index -> SPLICED
	negPassIdx := &VendorMemoryEvidence{
		Schema: VendorMemoryEvidenceSchema,
		Vendor: "nvidia",
		Sections: []VendorMemoryRecord{
			BindVendorMemoryRecord("alloc", "p0", -1, "run-1", "gpu-0", 100, 200, 1000, 100, 900),
		},
	}
	if res := CheckVendorMemoryEvidence(negPassIdx); res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected SPLICED for negative pass index, got %s", res.Verdict)
	}

	// 5. Negative timestamp -> SPLICED
	negTimestamp := &VendorMemoryEvidence{
		Schema: VendorMemoryEvidenceSchema,
		Vendor: "nvidia",
		Sections: []VendorMemoryRecord{
			BindVendorMemoryRecord("alloc", "p0", 0, "run-1", "gpu-0", -10, 200, 1000, 100, 900),
		},
	}
	if res := CheckVendorMemoryEvidence(negTimestamp); res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected SPLICED for negative timestamp, got %s", res.Verdict)
	}

	// 6. Stamp device ID mismatch -> SPLICED
	devMismatchRec := BindVendorMemoryRecord("alloc", "p0", 0, "run-1", "gpu-0", 100, 200, 1000, 100, 900)
	devMismatchRec.Stamp.DeviceID = "gpu-99"
	stampDevMismatch := &VendorMemoryEvidence{
		Schema:   VendorMemoryEvidenceSchema,
		Vendor:   "nvidia",
		Sections: []VendorMemoryRecord{devMismatchRec},
	}
	if res := CheckVendorMemoryEvidence(stampDevMismatch); res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected SPLICED for stamp device_id mismatch, got %s", res.Verdict)
	}

	// 7. Total passes exceeded -> SPLICED
	totalExceeded := &VendorMemoryEvidence{
		Schema:      VendorMemoryEvidenceSchema,
		Vendor:      "nvidia",
		TotalPasses: 1, // claims 1 pass, but sections contain pass 0 and pass 1
		Sections: []VendorMemoryRecord{
			BindVendorMemoryRecord("alloc", "p0", 0, "run-1", "gpu-0", 100, 200, 1000, 100, 900),
			BindVendorMemoryRecord("alloc", "p1", 1, "run-1", "gpu-0", 300, 400, 1000, 100, 900),
		},
	}
	if res := CheckVendorMemoryEvidence(totalExceeded); res.Verdict != MemoryVerdictSpliced {
		t.Fatalf("expected SPLICED for total passes exceeded, got %s", res.Verdict)
	}
}
