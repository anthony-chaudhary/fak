package fitplan

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// DS V4.1 Flash-like header: the dense core and routed experts as the planner reads them.
const (
	flashDenseBytes  int64 = 3_230_237_376
	flashRoutedBytes int64 = 196_765_286_400
)

func flashLikeInput() Input {
	active := flashRoutedBytes * 6 / 256
	in := Input{
		Model: Model{
			PayloadWeightBytes:        flashDenseBytes + flashRoutedBytes,
			StreamBytesPerToken:       flashDenseBytes + active,
			RoutedExpertBytes:         flashRoutedBytes,
			ActiveRoutedBytesPerToken: active,
			NumExperts:                256,
			ExpertsUsed:               6,
			MaxContextTokens:          131072,
		},
		Weights:     compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: flashDenseBytes + flashRoutedBytes, Scope: compute.MemoryScopeDevice}},
		Hardware:    Hardware{Topology: TopologyDiscrete, PoolUsableBytes: 16 * gib, HostPoolUsableBytes: 64 * gib, Peaks: testPeaks()},
		Workload:    DefaultWorkload(),
		Calibration: GenericEmpirical(),
	}
	in.Candidates = BuildCandidates(testTiers(), 4096)
	return in
}

// wirePlacement is the fit_plan placement block as a consumer reads it off the JSON wire.
type wirePlacement struct {
	Status    string          `json:"status"`
	Refusal   json.RawMessage `json:"refusal"`
	Placement *struct {
		Strategy string `json:"strategy"`
		Rows     []struct {
			Segment string `json:"segment"`
			Pool    string `json:"pool"`
			Bytes   int64  `json:"bytes"`
		} `json:"rows"`
		MaxContextTokens int     `json:"max_context_tokens"`
		MaxBatch         int     `json:"max_batch"`
		DecodeTokS       float64 `json:"decode_tok_s"`
		Tier             string  `json:"tier"`
	} `json:"placement"`
}

func decodeWire(t *testing.T, v any) wirePlacement {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var w wirePlacement
	if err := json.Unmarshal(raw, &w); err != nil {
		t.Fatal(err)
	}
	return w
}

var poolRank = map[string]int{"device": 0, "host_ram": 1, "nvme_mmap": 2}

// assertDenseCoreFirst checks the closed placement contract: status tiered with no refusal,
// the dense core whole on the device, the routed experts filling device -> host -> NVMe in
// order, every byte placed, and decode/context/batch stated under a known evidence tier.
func assertDenseCoreFirst(t *testing.T, w wirePlacement) {
	t.Helper()
	if w.Status != "tiered" || len(w.Refusal) != 0 || w.Placement == nil {
		t.Fatalf("status %q refusal %s placement %v, want a tiered placement and no refusal", w.Status, w.Refusal, w.Placement)
	}
	pl := w.Placement
	if pl.Strategy != "dense_core_first" || len(pl.Rows) == 0 {
		t.Fatalf("placement %+v, want dense_core_first rows", pl)
	}
	if r := pl.Rows[0]; r.Segment != "dense_core" || r.Pool != "device" || r.Bytes != flashDenseBytes {
		t.Fatalf("rows[0] %+v, want the whole dense core on device", r)
	}
	var total int64
	ring, last := false, -1
	for _, r := range pl.Rows[1:] {
		rank, ok := poolRank[r.Pool]
		if r.Segment != "routed_experts" || !ok || rank < last || r.Bytes <= 0 {
			t.Fatalf("row %+v out of the device -> host_ram -> nvme_mmap order", r)
		}
		ring = ring || r.Pool == "device"
		last = rank
		total += r.Bytes
	}
	if !ring || total != flashRoutedBytes {
		t.Fatalf("expert ring on device %v, routed placed %d, want a ring and %d", ring, total, flashRoutedBytes)
	}
	if !(pl.DecodeTokS > 0) || pl.MaxContextTokens <= 0 || pl.MaxBatch <= 0 {
		t.Fatalf("decode %v ctx %d batch %d, want all stated", pl.DecodeTokS, pl.MaxContextTokens, pl.MaxBatch)
	}
	if _, ok := map[string]bool{"sol": true, "empirical": true, "calibrated": true, "silicon": true}[pl.Tier]; !ok {
		t.Fatalf("tier %q outside the evidence-tier vocabulary", pl.Tier)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestOversizeMoEGetsDenseCoreFirstPlacement(t *testing.T) {
	in := flashLikeInput()
	assertDenseCoreFirst(t, decodeWire(t, Plan(in)))

	search := Search(SearchInput{
		Model: in.Model, Weights: in.Weights, Hardware: in.Hardware, Calibration: in.Calibration,
		Tiers: testTiers(), ResolvedContext: 4096, CurrentKV: "f32",
		Workload: SearchWorkload{PromptTokens: 2048, OutputTokens: 512, Concurrency: 1},
	})
	assertDenseCoreFirst(t, decodeWire(t, search))

	// Only an operator-pinned all-resident placement over its budget still refuses, typed.
	in.Pins.Placement = PlacementDeviceResident
	pinned := Plan(in)
	if pinned.Status != StatusRefused || pinned.Placement != nil || !errors.Is(refusalErr(pinned), ErrWeightsExceedPool) {
		t.Fatalf("pinned: status %s placement %v refusal %+v, want weights_exceed_pool", pinned.Status, pinned.Placement, pinned.Refusal)
	}
}
