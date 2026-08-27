// Package main tests for coalescebench. Everything here is resource-free and
// deterministic: no model file, no clock, no filesystem, no subprocess — the point under
// test IS the determinism (same flags → identical table) plus the exact §2 roofline
// arithmetic and the synthetic router's shape/skew contract.
package main

import (
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
)

// smallCfg is a tiny, fast config that still exercises every code path (multiple layers,
// multiple batch sizes including the B=1 baseline, a capacity small enough to evict).
func smallCfg() benchConfig {
	return benchConfig{
		Seed: 7, Steps: 4, Layers: 3, Experts: 16, TopK: 2,
		Skew: 1.0, ExpertMiB: 1.0, NonRoutedGiB: 0.01,
		BWSSDGiB: 1.0, BWRAMGiB: 100.0, ActiveGFLOP: 1.0, GFLOPS: 1000.0,
		CacheGiB: 0.01, // 10.24 MiB → 10 one-MiB groups: forces eviction traffic
		Batches:  []int{1, 2, 4},
	}
}

// TestRunBenchDeterministic is the DoD determinism gate: the same config must render the
// byte-identical table, twice, with no time or global-PRNG dependence.
func TestRunBenchDeterministic(t *testing.T) {
	cfg := smallCfg()
	first, err := runBench(cfg, io.Discard)
	if err != nil {
		t.Fatalf("runBench: %v", err)
	}
	second, err := runBench(cfg, io.Discard)
	if err != nil {
		t.Fatalf("runBench (second run): %v", err)
	}
	if first != second {
		t.Fatalf("runBench is not deterministic:\n--- first ---\n%s\n--- second ---\n%s", first, second)
	}
	// Structural sanity on the artifact: every table row carries the PROJECTED label,
	// and the two §2.1 baselines + the regime-transition line are present.
	for _, line := range strings.Split(first, "\n") {
		if strings.HasPrefix(line, "| ") && !strings.HasPrefix(line, "| B ") && !strings.HasSuffix(line, "| PROJECTED |") {
			t.Fatalf("table row missing PROJECTED label: %q", line)
		}
	}
	for _, want := range []string{"regime transition:", "un-coalesced shared-SSD floor", "×vs-uncoalesced", "×vs-1agent", "PROJECTED", "#5251"} {
		if !strings.Contains(first, want) {
			t.Fatalf("output missing %q:\n%s", want, first)
		}
	}
}

// TestArtifactEmitsAnalyticalSimulationEvidence captures the actual shared
// benchcli JSON emitter. The nested block is the mechanical fence that keeps
// this deterministic PROJECTED roofline from being read as achieved hardware
// throughput, while projected_markdown proves the old headline artifact is
// carried byte-for-byte beside it.
func TestArtifactEmitsAnalyticalSimulationEvidence(t *testing.T) {
	cfg := smallCfg()
	run, err := runSweep(cfg, io.Discard)
	if err != nil {
		t.Fatalf("runSweep: %v", err)
	}
	markdown := render(cfg, run.Capacity, run.Rows, run.Base)

	// Freeze the generic benchcli envelope so this proof only observes the
	// coalescebench evidence contract, not the host clock/checkout/node.
	t.Setenv("FAK_BENCH_UTC", "2026-08-27T00:00:00Z")
	t.Setenv("FAK_BENCH_COMMIT", "0123456789abcdef")
	t.Setenv("FAK_BENCH_NODE", "coalescebench-test")
	t.Setenv("FAK_BENCH_RUN_ID", "coalescebench-evidence-test")
	path := filepath.Join(t.TempDir(), "coalescebench.json")
	if err := writeArtifact(path, cfg, run, markdown, 12*time.Millisecond); err != nil {
		t.Fatalf("writeArtifact: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if !strings.Contains(string(raw), `"simulation_evidence": {`) {
		t.Fatalf("emitted artifact has no simulation_evidence block:\n%s", raw)
	}

	art, ok := benchcli.DecodeArtifact(raw)
	if !ok {
		t.Fatalf("benchcli.DecodeArtifact rejected emitted artifact:\n%s", raw)
	}
	if art.SimulationEvidence == nil {
		t.Fatalf("decoded artifact lost simulation_evidence:\n%s", raw)
	}
	ev := art.SimulationEvidence
	if ev.EvidenceType != benchcli.EvidenceAnalyticalBound || ev.ClaimCeiling != benchcli.ClaimBottleneckOnly {
		t.Fatalf("evidence promotion fence = %q/%q, want %q/%q",
			ev.EvidenceType, ev.ClaimCeiling, benchcli.EvidenceAnalyticalBound, benchcli.ClaimBottleneckOnly)
	}
	configJSON, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal expected config: %v", err)
	}
	const wantRevision = "0123456789abcdef"
	wantConfig := benchcli.SHA256Digest(configJSON)
	wantWorkload := benchcli.SHA256Digest(append([]byte(wantRevision+"\n"), configJSON...))
	if ev.Engine.Revision != wantRevision || ev.Engine.ConfigDigest != wantConfig || ev.Workload.Digest != wantWorkload {
		t.Fatalf("evidence identity = revision %q config %q workload %q, want revision %q config %q workload %q",
			ev.Engine.Revision, ev.Engine.ConfigDigest, ev.Workload.Digest, wantRevision, wantConfig, wantWorkload)
	}
	if ev.Replay.Seed == nil || *ev.Replay.Seed != cfg.Seed || ev.Replay.Repetitions != 1 ||
		ev.Replay.IndependentStreams != 1 || ev.Replay.Stochastic {
		t.Fatalf("replay identity = %+v, want one deterministic fixed-seed fixture with seed=%d", ev.Replay, cfg.Seed)
	}
	if ev.ValidityEnvelope.Dimensions["seed"] != "7" {
		t.Fatalf("validity seed = %q, want 7", ev.ValidityEnvelope.Dimensions["seed"])
	}
	hasCrossSeedFence := false
	for _, excluded := range ev.ExcludedEffects {
		if strings.Contains(excluded, "cross-seed uncertainty") {
			hasCrossSeedFence = true
			break
		}
	}
	if !hasCrossSeedFence {
		t.Fatalf("excluded effects do not fence cross-seed uncertainty: %v", ev.ExcludedEffects)
	}
	if ev.Cost.HostWallTimeMS != 12 || ev.Cost.HostCPUTimeMS != 0 || ev.Cost.Bytes <= 0 {
		t.Fatalf("simulator cost = %+v, want wall=12ms cpu=0-unsampled bytes>0", ev.Cost)
	}
	if err := benchcli.ValidateBenchmarkArtifact(art); err != nil {
		t.Fatalf("emitted evidence does not validate: %v", err)
	}

	var report struct {
		Label   string                 `json:"label"`
		Results coalesceArtifactResult `json:"results"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("decode coalesce report: %v", err)
	}
	if report.Label != "PROJECTED" || report.Results.ProjectedMarkdown != markdown {
		t.Fatalf("artifact changed legacy headline semantics: label=%q markdown_equal=%v",
			report.Label, report.Results.ProjectedMarkdown == markdown)
	}
	for _, fence := range []string{"NOT a served measurement", "PROJECTED", "un-coalesced shared-SSD floor", "×vs-1agent"} {
		if !strings.Contains(report.Results.ProjectedMarkdown, fence) {
			t.Fatalf("artifact projected_markdown lost %q:\n%s", fence, report.Results.ProjectedMarkdown)
		}
	}
}

// TestArtifactRejectsUnknownEngineRevision proves the analytical receipt cannot
// escape a Git checkout (or another stamped build-identity source) by silently
// substituting the generic lineage placeholder for the simulator code identity.
func TestArtifactRejectsUnknownEngineRevision(t *testing.T) {
	cfg := smallCfg()
	run, err := runSweep(cfg, io.Discard)
	if err != nil {
		t.Fatalf("runSweep: %v", err)
	}
	markdown := render(cfg, run.Capacity, run.Rows, run.Base)
	t.Setenv("FAK_BENCH_COMMIT", "unknown")
	path := filepath.Join(t.TempDir(), "must-not-exist.json")
	err = writeArtifact(path, cfg, run, markdown, time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "engine revision unresolved") {
		t.Fatalf("writeArtifact error = %v, want unresolved engine revision refusal", err)
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Fatalf("writeArtifact created %s despite unresolved revision (stat error %v)", path, statErr)
	}
}

// TestRunBenchSeedChangesOutput proves the seed actually reaches the router: a different
// seed must produce a different table (else the "determinism" would be vacuous).
func TestRunBenchSeedChangesOutput(t *testing.T) {
	a := smallCfg()
	b := smallCfg()
	b.Seed = 8
	outA, err := runBench(a, io.Discard)
	if err != nil {
		t.Fatalf("runBench(seed=7): %v", err)
	}
	outB, err := runBench(b, io.Discard)
	if err != nil {
		t.Fatalf("runBench(seed=8): %v", err)
	}
	if outA == outB {
		t.Fatalf("different seeds produced the identical table — seed is not wired into the router")
	}
}

func TestSynthRoutesShapeAndDeterminism(t *testing.T) {
	cfg := smallCfg()
	const agents = 3
	steps, stats := synthRoutes(cfg, agents)

	if want := cfg.Steps * cfg.Layers; len(steps) != want {
		t.Fatalf("len(steps) = %d, want steps*layers = %d", len(steps), want)
	}
	if stats.entries != cfg.Steps*cfg.Layers {
		t.Fatalf("union entries = %d, want %d", stats.entries, cfg.Steps*cfg.Layers)
	}
	for i, s := range steps {
		if wantLayer := i % cfg.Layers; s.Layer != wantLayer {
			t.Fatalf("steps[%d].Layer = %d, want %d", i, s.Layer, wantLayer)
		}
		if len(s.PerAgent) != agents {
			t.Fatalf("steps[%d] has %d agents, want %d", i, len(s.PerAgent), agents)
		}
		for a, sel := range s.PerAgent {
			if len(sel) != cfg.TopK {
				t.Fatalf("steps[%d].PerAgent[%d] has %d experts, want K=%d", i, a, len(sel), cfg.TopK)
			}
			seen := map[int]bool{}
			for _, e := range sel {
				if e < 0 || e >= cfg.Experts {
					t.Fatalf("steps[%d].PerAgent[%d] expert %d out of range [0,%d)", i, a, e, cfg.Experts)
				}
				if seen[e] {
					t.Fatalf("steps[%d].PerAgent[%d] selected expert %d twice (top-K must be distinct)", i, a, e)
				}
				seen[e] = true
			}
		}
	}

	// Same seed → identical routes.
	again, againStats := synthRoutes(cfg, agents)
	if againStats != stats {
		t.Fatalf("union stats not deterministic: %+v vs %+v", stats, againStats)
	}
	for i := range steps {
		for a := range steps[i].PerAgent {
			for j := range steps[i].PerAgent[a] {
				if steps[i].PerAgent[a][j] != again[i].PerAgent[a][j] {
					t.Fatalf("routes not deterministic at step %d agent %d slot %d", i, a, j)
				}
			}
		}
	}

	// Different seed → different routes somewhere.
	cfg2 := cfg
	cfg2.Seed = cfg.Seed + 1
	other, _ := synthRoutes(cfg2, agents)
	same := true
	for i := range steps {
		for a := range steps[i].PerAgent {
			for j := range steps[i].PerAgent[a] {
				if steps[i].PerAgent[a][j] != other[i].PerAgent[a][j] {
					same = false
				}
			}
		}
	}
	if same {
		t.Fatalf("seed change produced identical routes")
	}
}

func TestUnionSize(t *testing.T) {
	stamp := make([]int, 8)
	if got := unionSize([][]int{{0, 1}, {1, 2}, {2, 3}}, stamp, 1); got != 4 {
		t.Fatalf("unionSize overlap = %d, want 4", got)
	}
	if got := unionSize([][]int{{4, 5}, {6, 7}}, stamp, 2); got != 4 {
		t.Fatalf("unionSize disjoint = %d, want 4", got)
	}
	if got := unionSize([][]int{{3, 4}, {3, 4}, {3, 4}}, stamp, 3); got != 2 {
		t.Fatalf("unionSize full overlap = %d, want 2", got)
	}
}

func TestSampleTopKDistinctInRange(t *testing.T) {
	r := newRNG(42)
	w := layerWeights(1, 32, 1.5)[0]
	// k is the requested top-k; the invariant is that sampleTopK returns exactly k
	// distinct experts, so assert the relation len(sel)==k rather than freezing a
	// magic count that would only churn if k itself changed.
	const k = 5
	for trial := 0; trial < 50; trial++ {
		sel := sampleTopK(r, w, k)
		if len(sel) != k {
			t.Fatalf("trial %d: got %d experts, want %d", trial, len(sel), k)
		}
		seen := map[int]bool{}
		for _, e := range sel {
			if e < 0 || e >= 32 {
				t.Fatalf("trial %d: expert %d out of range", trial, e)
			}
			if seen[e] {
				t.Fatalf("trial %d: duplicate expert %d", trial, e)
			}
			seen[e] = true
		}
	}
}

// TestSkewConcentratesUnion is the DoD "U(B) growth is inspectable" knob check: at the
// same seed, a strongly Zipf-skewed router must coalesce harder (smaller mean per-layer
// union) than the uniform router.
func TestSkewConcentratesUnion(t *testing.T) {
	base := smallCfg()
	base.Experts = 64
	base.TopK = 4
	base.Steps = 8
	base.Layers = 2

	uniform := base
	uniform.Skew = 0
	skewed := base
	skewed.Skew = 2.0

	const agents = 8
	_, uStats := synthRoutes(uniform, agents)
	_, sStats := synthRoutes(skewed, agents)
	if !(sStats.uMean() < uStats.uMean()) {
		t.Fatalf("skew=2 union mean %.2f not below uniform union mean %.2f — the skew knob is not concentrating routes",
			sStats.uMean(), uStats.uMean())
	}
	// Both are bounded by min(B*K, N).
	if maxU := float64(agents * base.TopK); uStats.uMean() > maxU || sStats.uMean() > maxU {
		t.Fatalf("union mean exceeds B*K=%v: uniform %.2f skewed %.2f", maxU, uStats.uMean(), sStats.uMean())
	}
}

// TestComputeRowRoofline pins the §2 arithmetic exactly:
//
//	SSD_term = distinct/step · e / (B · BW_ssd), RAM_term = (NR + U·L·e)/BW_ram,
//	FLOP_term = active/FLOPS, t = max, net = B/t, C = B·K/U.
func TestComputeRowRoofline(t *testing.T) {
	cfg := benchConfig{
		Steps: 1, Layers: 4, Experts: 16, TopK: 2,
		ExpertMiB: 1.0, NonRoutedGiB: 1.0,
		BWSSDGiB: 1.0, BWRAMGiB: 10.0, ActiveGFLOP: 10.0, GFLOPS: 100.0,
	}
	u := unionStats{total: 12, entries: 4} // U = 3 distinct experts per (layer,step)
	r := computeRow(cfg, 2, u, 64)         // 64 simulated page-in groups per decode step

	wantSSD := 64.0 * (1.0 / 1024.0) / 2.0 // = 0.03125 s
	wantRAM := (1.0 + 12.0/1024.0) / 10.0  // = 0.101171875 s
	wantFLOP := 0.1                        // 10/100
	wantNet := 2.0 / wantRAM               // RAM binds
	approx := func(got, want float64, name string) {
		t.Helper()
		if math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("%s = %.15g, want %.15g", name, got, want)
		}
	}
	approx(r.UMean, 3.0, "UMean")
	approx(r.C, 4.0/3.0, "C")
	approx(r.SSDTerm, wantSSD, "SSDTerm")
	approx(r.RAMTerm, wantRAM, "RAMTerm")
	approx(r.FLOPTerm, wantFLOP, "FLOPTerm")
	approx(r.NetToks, wantNet, "NetToks")
	if r.Binding != "RAM" {
		t.Fatalf("Binding = %q, want RAM", r.Binding)
	}
}

// TestUncoalescedFloor pins the shared-SSD un-coalesced baseline the #5245 landing review
// asked for: BW_ssd/(L·K·e), constant in B. smallCfg: 1 GiB/s over 3·2·1 MiB = 1024/6 tok/s.
func TestUncoalescedFloor(t *testing.T) {
	got := uncoalescedFloor(smallCfg())
	want := 1024.0 / 6.0
	if math.Abs(got-want) > 1e-12*want {
		t.Fatalf("uncoalescedFloor = %.15g, want %.15g", got, want)
	}
}

func TestRegimeTransition(t *testing.T) {
	rows := []row{
		{B: 1, SSDTerm: 2.0, RAMTerm: 1.0},
		{B: 4, SSDTerm: 1.5, RAMTerm: 1.0},
		{B: 16, SSDTerm: 0.5, RAMTerm: 1.0},
		{B: 64, SSDTerm: 0.1, RAMTerm: 1.0},
	}
	star, ok := regimeTransition(rows)
	if !ok || star.B != 16 {
		t.Fatalf("regimeTransition = (%v, %v), want B*=16", star.B, ok)
	}
	if _, ok := regimeTransition(rows[:2]); ok {
		t.Fatalf("regimeTransition found a transition where SSD_term never drops below RAM_term")
	}
	if _, ok := regimeTransition(nil); ok {
		t.Fatalf("regimeTransition on no rows must report none")
	}
}

// TestCapacityGroups pins the cache-budget → whole-group conversion and its fail-closed
// too-small edge.
func TestCapacityGroups(t *testing.T) {
	cfg := smallCfg() // 0.01 GiB budget over 1 MiB groups = 10.24 → 10 groups
	got, err := cfg.capacityGroups()
	if err != nil || got != 10 {
		t.Fatalf("capacityGroups = (%d, %v), want (10, nil)", got, err)
	}
	cfg.ExpertMiB = 64 // budget now smaller than one group
	if _, err := cfg.capacityGroups(); err == nil {
		t.Fatalf("capacityGroups accepted a cache smaller than one expert group")
	}
}
