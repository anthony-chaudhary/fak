package slotstream

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// SlotStreamingReceipt schema for Issue #11105.
type SlotStreamingReceipt struct {
	Schema                 string  `json:"schema"`
	Issue                  int     `json:"issue"`
	Role                   string  `json:"role"`
	Engine                 string  `json:"engine"`
	ModelArchitecture      string  `json:"model_architecture"`
	VRAMBudgetBytes        int64   `json:"vram_budget_bytes"`
	DenseTrunkBytes        int64   `json:"dense_trunk_bytes"`
	SlotsPerLayer          int     `json:"slots_per_layer"`
	NumLayers              int     `json:"num_layers"`
	ExpertChunkSize        int     `json:"expert_chunk_size_bytes"`
	SustainedStreamingGBs  float64 `json:"sustained_streaming_gb_s"`
	AchievedDecodeTokS     float64 `json:"achieved_decode_tok_s"`
	MinBandwidthThreshold  float64 `json:"min_bandwidth_threshold_gb_s"`
	MinDecodeRateThreshold float64 `json:"min_decode_rate_threshold_tok_s"`
	VRAMEnclosedProven     bool    `json:"vram_enclosed_proven"`
}

func TestSlotStreamPoolBudget(t *testing.T) {
	// 40 MoE layers with 16 slots per layer (16 * 2.76 MB = ~44.2 MB per layer, ~1.77 GB total slots)
	pool, err := NewMultiTensorSlotPool(40, 16)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}

	if pool.TotalVRAMBytes > DefaultVRAMBudgetBytes {
		t.Fatalf("pool total VRAM %d > 24 GB budget", pool.TotalVRAMBytes)
	}

	// Dense trunk (3.8 GB) + 40 * 16 * 2.76 MB (~1.85 GB) = ~5.65 GB total device residency
	if pool.TotalVRAMBytes > 6*1024*1024*1024 {
		t.Fatalf("unexpected high VRAM allocation: %d bytes", pool.TotalVRAMBytes)
	}
}

// TestSlotStreamRtx5090CapLift pins the cap lift: a 32 GB card budgets its
// full reported VRAM instead of the 24 GiB fallback on the expert-slot path,
// while a 24 GB device and an unknown-VRAM device keep today's exact behavior.
func TestSlotStreamRtx5090CapLift(t *testing.T) {
	const gib = int64(1) << 30

	// Budget derivation: unknown VRAM falls back to the 24 GiB cap; a known
	// device budgets its full reported VRAM (the 24 GiB default implies zero
	// headroom, so a 24 GB device is byte-identical to pre-lift behavior).
	if got := VRAMBudgetForDevice(0); got != DefaultVRAMBudgetBytes {
		t.Fatalf("VRAMBudgetForDevice(0) = %d, want fallback %d", got, DefaultVRAMBudgetBytes)
	}
	if got := VRAMBudgetForDevice(-1); got != DefaultVRAMBudgetBytes {
		t.Fatalf("VRAMBudgetForDevice(-1) = %d, want fallback %d", got, DefaultVRAMBudgetBytes)
	}
	if got := VRAMBudgetForDevice(24 * gib); got != DefaultVRAMBudgetBytes {
		t.Fatalf("VRAMBudgetForDevice(24 GiB) = %d, want %d (24 GB unchanged)", got, DefaultVRAMBudgetBytes)
	}
	if got := VRAMBudgetForDevice(32 * gib); got != 32*gib {
		t.Fatalf("VRAMBudgetForDevice(32 GiB) = %d, want %d", got, 32*gib)
	}

	// Geometry that fits a 32 GB card but exceeded the old 24 GiB cap:
	// 160 layers x 54 slots x ~2.76 MB + 3.8 GB trunk ~= 26.8 GiB.
	const layers, slots = 160, 54

	pool, err := NewMultiTensorSlotPoolForDevice(layers, slots, 32*gib)
	if err != nil {
		t.Fatalf("32 GB card must no longer be refused on the expert-slot path: %v", err)
	}
	if pool.TotalVRAMBytes > 32*gib {
		t.Fatalf("pool total VRAM %d exceeds the 32 GiB device budget", pool.TotalVRAMBytes)
	}

	// The expert-slot path must actually run on the lifted pool.
	engine := NewStreamEngine(pool)
	hits, misses, _, err := engine.StreamPicks(0, []int{0, 1, 2}, 1)
	if err != nil || hits != 0 || misses != 3 {
		t.Fatalf("StreamPicks on 32 GB pool: hits=%d misses=%d err=%v, want 0/3/nil", hits, misses, err)
	}

	// Same geometry under the 24 GiB fallback (unknown VRAM): still refused.
	if _, err := NewMultiTensorSlotPoolForDevice(layers, slots, 0); err == nil {
		t.Fatalf("unknown-VRAM device must keep the 24 GiB fallback refusal")
	}
	// Same geometry on a 24 GB device: unchanged refusal.
	if _, err := NewMultiTensorSlotPoolForDevice(layers, slots, 24*gib); err == nil {
		t.Fatalf("24 GB device behavior changed: large geometry now accepted")
	}
	// Legacy constructor: unchanged 24 GiB cap.
	if _, err := NewMultiTensorSlotPool(layers, slots); err == nil {
		t.Fatalf("NewMultiTensorSlotPool cap changed: large geometry now accepted")
	}

	// A geometry that fit before the lift still fits on a 24 GB device with
	// identical sizing (no behavior change for 24 GB devices).
	small, err := NewMultiTensorSlotPool(40, 16)
	if err != nil {
		t.Fatalf("small geometry regressed on the fallback budget: %v", err)
	}
	smallDev, err := NewMultiTensorSlotPoolForDevice(40, 16, 24*gib)
	if err != nil {
		t.Fatalf("small geometry on explicit 24 GB device: %v", err)
	}
	if small.TotalVRAMBytes != smallDev.TotalVRAMBytes {
		t.Fatalf("24 GB sizing drifted: legacy %d vs device-sized %d", small.TotalVRAMBytes, smallDev.TotalVRAMBytes)
	}
}

func TestSlotStreamEngineHitAndEviction(t *testing.T) {
	pool, err := NewMultiTensorSlotPool(2, 4)
	if err != nil {
		t.Fatal(err)
	}
	engine := NewStreamEngine(pool)

	// Turn 1: experts 0, 1, 2, 3 (all misses, loaded)
	hits, misses, _, err := engine.StreamPicks(0, []int{0, 1, 2, 3}, 100)
	if err != nil {
		t.Fatal(err)
	}
	if hits != 0 || misses != 4 {
		t.Fatalf("expected 0 hits, 4 misses, got hits=%d misses=%d", hits, misses)
	}

	// Turn 2: experts 1, 2 (both hits)
	hits, misses, _, err = engine.StreamPicks(0, []int{1, 2}, 200)
	if err != nil {
		t.Fatal(err)
	}
	if hits != 2 || misses != 0 {
		t.Fatalf("expected 2 hits, 0 misses, got hits=%d misses=%d", hits, misses)
	}

	// Turn 3: expert 4 replaces LRU slot (expert 0, accessed at 100)
	hits, misses, _, err = engine.StreamPicks(0, []int{4}, 300)
	if err != nil {
		t.Fatal(err)
	}
	if hits != 0 || misses != 1 {
		t.Fatalf("expected 0 hits, 1 miss, got hits=%d misses=%d", hits, misses)
	}
}

func TestSlotStreamStreamingBandwidthAndDecodeRate(t *testing.T) {
	const numLayers = 40
	const slotsPerLayer = 16
	const topK = 8

	pool, err := NewMultiTensorSlotPool(numLayers, slotsPerLayer)
	if err != nil {
		t.Fatal(err)
	}
	engine := NewStreamEngine(pool)

	// Simulate streaming 100 decode tokens across 40 layers
	const numTokens = 100
	var totalBytesStreamed int64
	var totalDuration time.Duration

	now := time.Now().UnixNano()
	for tok := 0; tok < numTokens; tok++ {
		start := time.Now()
		for l := 0; l < numLayers; l++ {
			// Top-8 picks per layer
			picks := []int{
				(tok*8 + 0) % 288,
				(tok*8 + 1) % 288,
				(tok*8 + 2) % 288,
				(tok*8 + 3) % 288,
				(tok*8 + 4) % 288,
				(tok*8 + 5) % 288,
				(tok*8 + 6) % 288,
				(tok*8 + 7) % 288,
			}
			_, misses, _, err := engine.StreamPicks(l, picks, now)
			if err != nil {
				t.Fatal(err)
			}
			totalBytesStreamed += int64(misses) * ExpertChunkBytes
		}
		totalDuration += time.Since(start)
	}

	gbStreamed := float64(totalBytesStreamed) / (1024 * 1024 * 1024)
	bandwidthGBs := gbStreamed / totalDuration.Seconds()
	tokS := float64(numTokens) / totalDuration.Seconds()

	// Assert sustained bandwidth >= 12 GB/s
	if bandwidthGBs < 12.0 {
		t.Fatalf("sustained streaming bandwidth %.2f GB/s < 12.0 GB/s", bandwidthGBs)
	}

	// Assert decode rate >= 15 tok/s
	if tokS < 15.0 {
		t.Fatalf("achieved decode rate %.2f tok/s < 15.0 tok/s", tokS)
	}

	// The tracked historical hardware receipt must never be a write target of an
	// ordinary package test: this test only exercises SIMULATED StreamPicks and
	// its timings are host-dependent. Capture the tracked path and its bytes so
	// the regression assertion below can prove the test leaves them untouched.
	trackedReceiptPath := filepath.Join("..", "..", "..", "docs", "_witnesses", "issue-11105-slot-streaming", "receipt.json")
	trackedBefore, trackedBeforeErr := os.ReadFile(trackedReceiptPath)

	// Build a SIMULATION receipt. Provenance is explicit: engine is marked as
	// simulated (not a physical `fak-native` claim) and VRAMEnclosedProven stays
	// false so simulation cannot masquerade as hardware evidence.
	receipt := SlotStreamingReceipt{
		Schema:                 "fak-slot-streaming-simulated/v1",
		Issue:                  11105,
		Role:                   "simulation",
		Engine:                 "simulated",
		ModelArchitecture:      "SIMULATED (not physical hardware evidence)",
		VRAMBudgetBytes:        DefaultVRAMBudgetBytes,
		DenseTrunkBytes:        DefaultDenseTrunkBytes,
		SlotsPerLayer:          slotsPerLayer,
		NumLayers:              numLayers,
		ExpertChunkSize:        ExpertChunkBytes,
		SustainedStreamingGBs:  bandwidthGBs,
		AchievedDecodeTokS:     tokS,
		MinBandwidthThreshold:  12.0,
		MinDecodeRateThreshold: 15.0,
		VRAMEnclosedProven:     false,
	}

	receiptPath := filepath.Join(t.TempDir(), "receipt.json")
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	if err := os.WriteFile(receiptPath, data, 0644); err != nil {
		t.Fatalf("failed to write receipt: %v", err)
	}
	if err := verifySimulationReceiptProvenance(receipt); err != nil {
		t.Fatalf("simulation receipt provenance is dishonest: %v", err)
	}

	// Optional explicit persistence: only when the operator asks for it via the
	// environment. It still lands under a temp/out dir, never the tracked path.
	if outDir := os.Getenv("FAK_SLOTSTREAM_WRITE_RECEIPT"); outDir != "" {
		persistDir := filepath.Join(outDir, "issue-12809-slot-streaming-simulated")
		if err := os.MkdirAll(persistDir, 0755); err != nil {
			t.Fatalf("failed to create receipt out dir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(persistDir, "receipt.json"), data, 0644); err != nil {
			t.Fatalf("failed to persist receipt: %v", err)
		}
	}

	// Regression: the tracked hardware-evidence receipt must be unchanged by
	// this simulated test run (this is the fak#12809 invariant).
	trackedAfter, trackedAfterErr := os.ReadFile(trackedReceiptPath)
	if trackedBeforeErr == nil && trackedAfterErr != nil {
		t.Fatalf("test removed tracked receipt %s: %v", trackedReceiptPath, trackedAfterErr)
	}
	if trackedBeforeErr == nil && trackedAfterErr == nil && string(trackedBefore) != string(trackedAfter) {
		t.Fatalf("test overwrote tracked receipt %s with simulation data", trackedReceiptPath)
	}
}

// verifySimulationReceiptProvenance asserts a simulated receipt never asserts
// physical hardware execution or a physical engine identity.
func verifySimulationReceiptProvenance(r SlotStreamingReceipt) error {
	if r.Role != "simulation" {
		return fmt.Errorf("role %q must be %q for a simulated receipt", r.Role, "simulation")
	}
	if r.Engine == "fak-native" {
		return fmt.Errorf("simulated receipt must not claim physical engine %q", r.Engine)
	}
	if r.VRAMEnclosedProven {
		return fmt.Errorf("simulated receipt must not assert VRAMEnclosedProven")
	}
	return nil
}
