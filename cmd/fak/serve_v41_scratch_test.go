package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serve_v41_scratch_test.go — witness for fak#13300: the V4.1 native forward's
// forward-scoped scratch + layer-scoped expert cache high-water must be charged
// ONCE in the serve memory plan, derived from the header config, before the model
// loads. Before this the buffers were charged to no ledger, so the warmup's RSS
// exceeded the declared plan and the kernel OOM-killed the process (#13288).
//
// The tests below pin the exact config-derived byte sum (tiny geometry), the
// cache-on vs cache-off delta, the non-V4.1 no-op, the overflow fail-closed
// clamp, and the plan-level charge (a V4.1 header's plan carries exactly one
// v41-forward-transient-scratch row with the expected bytes).

// v41TinyConfig is the smallest coherent V4.1 geometry the estimator admits, so
// the expected byte sum is small enough to state by hand in the assertion.
func v41TinyConfig() fakmodel.Config {
	return fakmodel.Config{
		ModelType:           "deepseek41",
		HiddenSize:          8,
		MoEIntermediateSize: 4,
		NumHeads:            2,
		HeadDim:             4,
		OLoraRank:           3,
		OGroups:             2,
	}
}

// v41TinyConfigExpectedBytes is the exact element count for v41TinyConfig:
//
//	woA  = OLoraRank*NumHeads*HeadDim = 3*2*4    = 24
//	woB  = Hidden*(OLoraRank*OGroups) = 8*(3*2)  = 48
//	expW13 = 2*MoEIntermediate*Hidden = 2*4*8   = 64
//	expW2  = Hidden*MoEIntermediate   = 8*4     = 32
//	mhc  = v41MHCMixWidth*4*Hidden     = 24*4*8  = 768
//	                                          elems = 936
//	bytes = 936*4 = 3744
const v41TinyConfigExpectedBytes = int64(936 * 4)

func TestServeV41ScratchEstimatorTinyGeometryExact(t *testing.T) {
	cfg := v41TinyConfig()
	if got := cfg.V41TransientResidentBytes(0); got != v41TinyConfigExpectedBytes {
		t.Fatalf("V41TransientResidentBytes(cache off) = %d, want %d", got, v41TinyConfigExpectedBytes)
	}
	// The cache is a SEPARATE, additive charge (a disabled cache must not be
	// double-counted, and an enabled one must add exactly its bound).
	const cache = int64(1 << 20)
	if got, want := cfg.V41TransientResidentBytes(cache), v41TinyConfigExpectedBytes+cache; got != want {
		t.Fatalf("V41TransientResidentBytes(cache on) = %d, want %d (tiny sum + cache)", got, want)
	}
}

// TestServeV41ScratchNonV41IsInert pins the fail-open contract: a non-V4.1 config
// (or one without the V4.1 MoE geometry) reports nothing, so every other serve's
// plan is byte-for-byte unchanged.
func TestServeV41ScratchNonV41IsInert(t *testing.T) {
	cases := map[string]fakmodel.Config{
		"llama":       {ModelType: "llama", HiddenSize: 4096, MoEIntermediateSize: 14336},
		"no moe":      {ModelType: "deepseek41", HiddenSize: 5120},
		"empty":       {},
		"hidden zero": {ModelType: "deepseek41", MoEIntermediateSize: 2304},
	}
	for name, cfg := range cases {
		if got := cfg.V41TransientResidentBytes(1 << 30); got != 0 {
			t.Errorf("%s: V41TransientResidentBytes = %d, want 0 (inert)", name, got)
		}
	}
}

// TestServeV41ScratchOverflowFailsClosed pins that an unrepresentable geometry
// saturates to math.MaxInt64 rather than wrapping to a falsely-fitting small
// number: an overflow can only make the plan refuse MORE.
func TestServeV41ScratchOverflowFailsClosed(t *testing.T) {
	cfg := fakmodel.Config{
		ModelType:           "deepseek41",
		HiddenSize:          math.MaxInt32,
		MoEIntermediateSize: math.MaxInt32,
		NumHeads:            math.MaxInt32,
		HeadDim:             math.MaxInt32,
		OLoraRank:           math.MaxInt32,
		OGroups:             math.MaxInt32,
	}
	if got := cfg.V41TransientResidentBytes(0); got != math.MaxInt64 {
		t.Fatalf("overflowing geometry = %d, want math.MaxInt64 (fail closed)", got)
	}
}

// TestServeV41ScratchPublishedGeometryDecomposes pins the estimator against the
// PINNED published V4.1 config (testdata) and re-derives the same sum
// independently: the estimator must equal the hand-computed per-buffer sum at the
// real geometry, not merely be non-zero.
func TestServeV41ScratchPublishedGeometryDecomposes(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "internal", "model", "testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatalf("read pinned published V4.1 config: %v", err)
	}
	var cfg fakmodel.Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("parse pinned published V4.1 config: %v", err)
	}
	if !cfg.IsDeepSeekV41() {
		t.Fatalf("pinned config is not a V4.1 identity: model_type=%q", cfg.ModelType)
	}
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	elems := int64(cfg.OLoraRank)*int64(cfg.NumHeads)*int64(cfg.HeadDim) +
		int64(H)*int64(cfg.OLoraRank)*int64(cfg.OGroups) +
		2*int64(I)*int64(H) +
		int64(H)*int64(I) +
		int64(24)*4*int64(H)
	want := elems * 4
	if got := cfg.V41TransientResidentBytes(0); got != want {
		t.Fatalf("published geometry estimate = %d, want independently-derived %d (elems=%d)", got, want, elems)
	}
	// Sanity: the published per-buffer sum is the ballpark the #13288 comment
	// names (~416 MiB for wo_a+wo_b; the estimate is the whole transient).
	if want <= 0 || want > 1<<30 {
		t.Fatalf("published transient %d bytes is outside the plausible (0, 1 GiB] band", want)
	}
}

// TestServeV41ScratchPlanChargesRowOnce pins the integration: a plan built from a
// V4.1 header through the shared appendServeGGUFDevicePlan carries exactly ONE
// host-scoped v41-forward-transient-scratch scratchpad row with the config-derived
// bytes, and the row participates in the plan's host total.
func TestServeV41ScratchPlanChargesRowOnce(t *testing.T) {
	cfg := v41TinyConfig()
	want := cfg.V41TransientResidentBytes(0)
	// The base row is DEVICE-scoped (empty Scope defaults to device per the
	// compute.MemoryDemand contract), so the host total isolates the transient
	// row the charge adds. A host-scoped base row proves additivity instead.
	base := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 4096, Detail: "base", Scope: compute.MemoryScopeHost}}
	plan := appendServeV41TransientCharge(base, cfg, 0)

	var rows int
	var charged int64
	for _, d := range plan {
		if d.Detail == "v41-forward-transient-scratch" {
			rows++
			charged = d.Bytes
			if d.Class != compute.MemoryScratchpad {
				t.Errorf("transient row class = %q, want %q", d.Class, compute.MemoryScratchpad)
			}
			if d.Scope != compute.MemoryScopeHost {
				t.Errorf("transient row scope = %q, want host (anonymous RAM)", d.Scope)
			}
		}
	}
	if rows != 1 {
		t.Fatalf("plan carries %d transient rows, want exactly 1", rows)
	}
	if charged != want {
		t.Fatalf("transient row bytes = %d, want %d", charged, want)
	}
	if got := plan.HostTotal(); got != int64(4096)+want {
		t.Fatalf("plan host total = %d, want base+transient = %d", got, int64(4096)+want)
	}
}

// TestServeV41ScratchPlanInertForNonV41 pins the plan-level no-op: a non-V4.1
// config adds no row and leaves the plan host total unchanged.
func TestServeV41ScratchPlanInertForNonV41(t *testing.T) {
	base := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 4096, Detail: "base"}}
	plan := appendServeV41TransientCharge(base, fakmodel.Config{ModelType: "llama", HiddenSize: 4096}, 0)
	if len(plan) != len(base) {
		t.Fatalf("non-V4.1 plan gained a row: len=%d want=%d", len(plan), len(base))
	}
	if got := plan.HostTotal(); got != base.HostTotal() {
		t.Fatalf("non-V4.1 host total changed: %d -> %d", base.HostTotal(), got)
	}
}
