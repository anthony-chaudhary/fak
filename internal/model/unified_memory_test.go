package model

import "testing"

func TestMeasureUnifiedMemoryCountsTransfers(t *testing.T) {
	r, err := MeasureUnifiedMemory(UnifiedMemorySample{StorageMode: "shared", ResourceBytes: 1000, CPUWriteBytes: 100, GPUReadBytes: 800, GPUWriteBytes: 100, PageFaultBytes: 200, SLCResidentBytes: 500, CommandNanoseconds: 100, AcceptedTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native-metal" || r.SharedCapacityBytes != 1000 || r.EffectiveTransferBytes != 1200 || r.PageFaultBytes != 200 || r.SLCResidencyRatio != .5 || r.EffectiveGBps != 12 || r.BytesPerAccepted != 120 {
		t.Fatalf("receipt=%+v", r)
	}
}
func TestMeasureUnifiedMemoryRejectsCapacityMyth(t *testing.T) {
	if _, err := MeasureUnifiedMemory(UnifiedMemorySample{StorageMode: "shared", ResourceBytes: 1, SLCResidentBytes: 2, CommandNanoseconds: 1}); err == nil {
		t.Fatal("impossible residency accepted")
	}
}
