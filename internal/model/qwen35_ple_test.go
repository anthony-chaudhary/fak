package model

import (
	"testing"
)

func TestPLEConfig_DefaultInvariants(t *testing.T) {
	cfg := DefaultPLEConfig()
	if cfg.NumHeads != 16 {
		t.Errorf("NumHeads = %d, want 16", cfg.NumHeads)
	}
	if cfg.HeadDim != 160 {
		t.Errorf("HeadDim = %d, want 160", cfg.HeadDim)
	}
	if cfg.VocabPerHead != 20000000 {
		t.Errorf("VocabPerHead = %d, want 20000000", cfg.VocabPerHead)
	}
	if cfg.RowSizeBytes != 130 {
		t.Errorf("RowSizeBytes = %d, want 130", cfg.RowSizeBytes)
	}
	if !cfg.HostMapped {
		t.Error("HostMapped must be true by default")
	}
	if cfg.PinEngramRAM {
		t.Error("PinEngramRAM must be false by default")
	}
	if cfg.MaxResidentRSS != 2*1024*1024*1024 {
		t.Errorf("MaxResidentRSS = %d, want 2 GiB", cfg.MaxResidentRSS)
	}
	if cfg.TargetPrefillTokS < 300.0 {
		t.Errorf("TargetPrefillTokS = %.2f, want >= 300.0", cfg.TargetPrefillTokS)
	}
}

func TestComputePLERowOffset(t *testing.T) {
	// Head 0, Token 5 => row 5, offset 5 * 130 = 650
	off0 := ComputePLERowOffset(5, 0, 20000000, 130)
	if off0 != 650 {
		t.Errorf("off0 = %d, want 650", off0)
	}

	// Head 1, Token 5 => row 20000005, offset = 20000005 * 130
	off1 := ComputePLERowOffset(5, 1, 20000000, 130)
	expectedOff1 := int64(20000005) * 130
	if off1 != expectedOff1 {
		t.Errorf("off1 = %d, want %d", off1, expectedOff1)
	}
}

func TestPrefetchPLEUBatchRows(t *testing.T) {
	cfg := DefaultPLEConfig()
	cfg.NumHeads = 4
	cfg.VocabPerHead = 1000

	tokens := []int{10, 20, 30}
	rows := PrefetchPLEUBatchRows(tokens, cfg)

	// 3 tokens * 4 heads = 12 unique rows
	if len(rows) != 12 {
		t.Fatalf("expected 12 rows, got %d", len(rows))
	}
}

func TestValidatePLEMemoryCarveout(t *testing.T) {
	// Valid host-mapped with <= 2.0 GiB RSS
	if err := ValidatePLEMemoryCarveout(true, false, 1024*1024*1024); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Violation: not host-mapped
	if err := ValidatePLEMemoryCarveout(false, false, 1024*1024*1024); err == nil {
		t.Fatal("expected error when hostMapped is false")
	}

	// Violation: uploaded to GTT
	if err := ValidatePLEMemoryCarveout(true, true, 1024*1024*1024); err == nil {
		t.Fatal("expected error when gttAllocated is true")
	}

	// Violation: RSS > 2.0 GiB
	if err := ValidatePLEMemoryCarveout(true, false, int64(3)*1024*1024*1024); err == nil {
		t.Fatal("expected error when RSS exceeds 2.0 GiB")
	}
}

func TestValidatePLEPrefillThroughput(t *testing.T) {
	if err := ValidatePLEPrefillThroughput(352.0); err != nil {
		t.Fatalf("expected 352 tok/s to pass, got %v", err)
	}
	if err := ValidatePLEPrefillThroughput(250.0); err == nil {
		t.Fatal("expected 250 tok/s to fail (< 300 tok/s floor)")
	}
}
