// Command hotsetcheck measures the decisive unmeasured quantity of the MoE expert-cache program
// (fak-private#cluster-bringup-rtx6000-b300-readiness-20261104 P2-4): the ROUTED-EXPERT CACHE HIT
// RATE. It runs a routed-expert trace — synthetic Zipf/uniform fixtures by default, or a recorded
// trace file with --trace — through the learner-wired residency loop
// (model.ExpertHotSetAdvisor -> ctxmmu.ExpertResidencyPolicy) and its plain-LRU baseline, and
// reports hit rate, pin stability (hysteresis engaged, no thrash), and transient-ring behavior.
//
// Every number is [SW-VERIFIED] on fixtures: no weights, no kernels, no clock, no hardware. A
// receipt this command emits must never be read as a throughput or hardware claim; the physical
// rung belongs to the moe-expert-cache-benchmarks suite's Halo witness.
//
// Usage:
//
//	hotsetcheck [--zipf 1.2] [--tokens 200] [--layers 40] [--experts 384] [--topk 6]
//	            [--hot-set-size 96] [--expert-bytes 1048576] [--hot-bytes N] [--transient-bytes N]
//	            [--seed 1] [--turn-tokens 50] [--sweep-every 20] [--sweep-width 64]
//	            [--learn-fraction 0.25] [--trace file.json] [--json]
//
// The trace file format (--trace) is {"schema":"fak.hotsetcheck-trace/v1","events":[{"layer":L,
// "expert":E}, ...]}; negative coordinates are refused, not silently dropped into a real slot.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const traceFileSchema = "fak.hotsetcheck-trace/v1"

// traceFileEvent is one recorded routed-expert touch.
type traceFileEvent struct {
	Layer  int `json:"layer"`
	Expert int `json:"expert"`
}

// traceFile is the on-disk envelope for --trace input.
type traceFile struct {
	Schema string           `json:"schema"`
	Events []traceFileEvent `json:"events"`
}

func main() {
	var (
		zipf      float64
		tokens    int
		layers    int
		experts   int
		topk      int
		hotSize   int
		expBytes  int64
		hotB      int64
		transB    int64
		seed      int64
		turnTok   int
		sweepEv   int
		sweepW    int
		learnFrac float64
		trace     string
		asJSON    bool
	)
	flag.Float64Var(&zipf, "zipf", 1.2, "Zipf skew exponent; 0 selects uniform over all experts")
	flag.IntVar(&tokens, "tokens", 0, "simulated decode steps (default 200)")
	flag.IntVar(&layers, "layers", 0, "routed layers (default: V4.1 shape)")
	flag.IntVar(&experts, "experts", 0, "routed experts per layer (default: V4.1 shape)")
	flag.IntVar(&topk, "topk", 0, "experts routed per layer (default: V4.1 shape)")
	flag.IntVar(&hotSize, "hot-set-size", 0, "hot-range width the Zipf mass concentrates on (default: experts/4 when skewed)")
	flag.Int64Var(&expBytes, "expert-bytes", 0, "resident bytes per (layer, expert) (default 1 MiB)")
	flag.Int64Var(&hotB, "hot-bytes", 0, "decode hot-set ring budget in bytes (default 512 x expert-bytes)")
	flag.Int64Var(&transB, "transient-bytes", 0, "prefill transient ring budget in bytes (default 128 x expert-bytes)")
	flag.Int64Var(&seed, "seed", 1, "PRNG seed; the same seed reproduces the same trace")
	flag.IntVar(&turnTok, "turn-tokens", 0, "measured tokens between learner re-commits (default 50)")
	flag.IntVar(&sweepEv, "sweep-every", 0, "inject one prefill sweep every N measured tokens (default 20)")
	flag.IntVar(&sweepW, "sweep-width", 0, "distinct experts per prefill sweep (default 64)")
	flag.Float64Var(&learnFrac, "learn-fraction", 0, "share of tokens spent learning before the measured window (default 0.25)")
	flag.StringVar(&trace, "trace", "", "recorded trace file (fak.hotsetcheck-trace/v1 JSON); replaces the synthetic trace")
	flag.BoolVar(&asJSON, "json", false, "emit the machine-readable receipt instead of the report")
	flag.Parse()

	if zipf < 0 {
		fatalf("--zipf must be >= 0 (0 selects uniform), got %g", zipf)
	}
	opts := Options{
		Tokens:         tokens,
		Seed:           seed,
		HotSetSize:     hotSize,
		ZipfExponent:   zipf,
		ExpertBytes:    expBytes,
		HotBytes:       hotB,
		TransientBytes: transB,
		LearnFraction:  learnFrac,
		TurnTokens:     turnTok,
		SweepEvery:     sweepEv,
		SweepWidth:     sweepW,
		TraceFile:      trace,
	}
	// Resolve the shape FIRST: the hot-range default below reads its expert count, and a zero
	// shape would silently fall through to the generator's uniform arm.
	shape := model.V41ExpertCacheShapeDefault()
	if layers > 0 {
		shape.Layers = layers
	}
	if experts > 0 {
		shape.Experts = experts
	}
	if topk > 0 {
		shape.TopK = topk
	}
	opts.Shape = shape
	// Default the hot range to a quarter of the expert space when skewed but unspecified: a
	// bounded hot set is what makes the fixture a working-set study rather than pure noise.
	if opts.HotSetSize <= 0 && opts.ZipfExponent > 0 {
		opts.HotSetSize = opts.Shape.Experts / 4
	}

	rec, err := runFixture(opts)
	if err != nil {
		fatalf("hotsetcheck: %v", err)
	}
	if asJSON {
		out, err := json.MarshalIndent(rec, "", "  ")
		if err != nil {
			fatalf("hotsetcheck: marshal receipt: %v", err)
		}
		fmt.Println(string(out))
		return
	}
	printReport(rec)
}

// loadTraceFile reads a recorded routed-expert trace (fak.hotsetcheck-trace/v1). A wrong schema or
// an empty event list is refused; negative coordinates are refused too — a malformed touch must
// not silently land on a real (layer, expert) slot.
func loadTraceFile(path string) ([]model.ExpertCacheTraceEvent, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read trace %s: %w", path, err)
	}
	var tf traceFile
	if err := json.Unmarshal(data, &tf); err != nil {
		return nil, fmt.Errorf("parse trace %s: %w", path, err)
	}
	if tf.Schema != traceFileSchema {
		return nil, fmt.Errorf("trace %s schema %q, want %q", path, tf.Schema, traceFileSchema)
	}
	if len(tf.Events) == 0 {
		return nil, fmt.Errorf("trace %s carries no events", path)
	}
	evs := make([]model.ExpertCacheTraceEvent, 0, len(tf.Events))
	for i, e := range tf.Events {
		if e.Layer < 0 || e.Expert < 0 {
			return nil, fmt.Errorf("trace %s event %d has negative coordinates (%d,%d)", path, i, e.Layer, e.Expert)
		}
		evs = append(evs, model.ExpertCacheTraceEvent{Layer: e.Layer, Expert: e.Expert})
	}
	return evs, nil
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "hotsetcheck: "+format+"\n", args...)
	os.Exit(1)
}

// printReport renders the human-readable witness: one line per arm, the pin-stability line, and the
// transient-ring line, each carrying its own counters so no rate hides its numerator.
func printReport(rec receipt) {
	base, learn := rec.Arms[0], rec.Arms[1]
	fmt.Printf("hotsetcheck [SW-VERIFIED] fixture=%s shape=%dx%d/top%d tokens=%d seed=%d\n",
		rec.Fixture.Kind, rec.Fixture.Layers, rec.Fixture.Experts, rec.Fixture.TopK,
		rec.Fixture.Tokens, rec.Fixture.Seed)
	printArm(base)
	printArm(learn)
	fmt.Printf("  pin_stability        learn_swaps=%d over %d turns (hysteresis_engaged=%v)\n",
		learn.LearnSwaps, learn.LearnTurns, learn.LearnSwaps == 0 && learn.FinalHotSetSize > 0)
	fmt.Printf("  transient_ring       hits=%d evicts=%d entries_end=%d | sizing_advisory recommended=%d configured=%d (observed=%d hot_set=%d avg=%d cap=%d)\n",
		learn.TransientHits, learn.TransientEvictions, learn.TransientEntries,
		rec.TransientSizing.Recommended, rec.TransientSizing.Configured,
		rec.TransientSizing.Observed, rec.TransientSizing.HotSet,
		rec.TransientSizing.AvgExpertBytes, rec.TransientSizing.CapBytes)
	fmt.Printf("  delta_hit_rate       %+0.4f (learner-wired minus baseline-lru)\n", rec.DeltaHitRate)
}

func printArm(a armReceipt) {
	fmt.Printf("  %-17s hit_rate=%.4f hits=%d misses=%d (resident=%d transient=%d demand_miss=%d bypass=%d refused=%d) hot_evict=%d trans_evict=%d loads=%d touches=%d\n",
		a.Name, a.HitRate, a.ResidentHits+a.TransientHits, a.DemandMisses,
		a.ResidentHits, a.TransientHits, a.DemandMisses, a.DecodeBypasses, a.ColdMisses,
		a.HotEvictions, a.TransientEvictions, a.BackingLoads, a.TotalTouches)
}
