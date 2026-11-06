package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// replay.go — the fixture driver behind hotsetcheck: it runs a routed-expert trace through the
// SAME machinery the serve path will use (ctxmmu.ExpertResidencyPolicy + model.ExpertHotSetAdvisor)
// and reports the decisive unmeasured quantity — the routed-expert cache hit rate — alongside pin
// stability (hysteresis engaged, no thrash) and transient-ring behavior.
//
// Every number here is [SW-VERIFIED] on synthetic fixtures: no weights, no kernels, no clock. The
// backing store is a counting fake whose only job is to prove which touches moved bytes; nothing
// in this file may be read as a hardware or throughput claim.
//
// Two arms race the identical event stream:
//
//	baseline-lru  — ExpertResidencyPolicy with no advisor: plain-LRU hot-set admission.
//	learner-wired — the same policy wired to an ExpertHotSetAdvisor: warm-started from the
//	               learner's committed hot-set, cold-miss admission gated by the learner's
//	               displacement band, Learn() re-committed at every turn boundary.
//
// The stream is split into a LEARNING phase (observed into the learner only, no policy interaction,
// so both arms start the measured window with empty rings) and a MEASURED phase, in which every
// decode touch is observed into the learner before it is decided, a deterministic prefill sweep is
// injected every sweepEvery tokens to exercise the transient ring, and the learner re-commits at
// every turnTokens tokens. Steady state actuates through the admission gate alone: warm start is
// one-shot, so the learned pins protect the ring instead of churning it.

// ReceiptSchema tags the machine-readable output.
const ReceiptSchema = "fak.hotsetcheck/v1"

// Options parameterizes a fixture run; the zero value takes the pinned defaults below.
type Options struct {
	// Shape is the routed geometry; zero fields take the V4.1 defaults (40 layers, 384 experts,
	// top-6) via model.V41ExpertCacheShapeDefault().
	Shape model.V41ExpertCacheShape
	// Seed governs the synthetic trace; the same seed reproduces the same trace.
	Seed int64
	// Tokens is the total simulated decode steps; each contributes Layers*TopK events.
	Tokens int
	// HotSetSize bounds the Zipf mass (the generator's hot-range width); <= 0 or >= Experts means
	// no hot set (uniform).
	HotSetSize int
	// ZipfExponent is the skew: 0 => uniform over all experts; > 0 => Zipf-like mass on the hot set.
	ZipfExponent float64
	// ExpertBytes is the resident bytes charged per (layer, expert).
	ExpertBytes int64
	// HotBytes / TransientBytes are the two ring budgets; <= 0 takes the defaults below.
	HotBytes       int64
	TransientBytes int64
	// LearnFraction is the share of tokens spent learning before the measured window; <= 0 or >= 1
	// takes 0.25.
	LearnFraction float64
	// TurnTokens bounds how many measured tokens pass between learner re-commits; <= 0 takes 50.
	TurnTokens int
	// SweepEvery injects one prefill sweep every N measured tokens; 0 takes 20, a negative value
	// disables sweeps (the decode-gate isolation arm).
	SweepEvery int
	// SweepWidth is the distinct experts one prefill sweep touches; <= 0 takes 64.
	SweepWidth int
	// TraceFile, when non-empty, replaces the synthetic trace with a recorded event stream
	// (traceFileFormat); the shape fields above then describe only the report identity.
	TraceFile string
}

// Defaults: a working set far larger than the hot ring, so both arms live in the eviction regime
// where a residency decision actually matters.
const (
	defaultExpertBytes    int64 = 1 << 20 // 1 MiB per (layer, expert)
	defaultHotSlots       int   = 512     // hot ring holds 512 entries
	defaultTransientSlots int   = 128     // transient ring holds 128 entries
	defaultLearnFraction        = 0.25
	defaultTurnTokens           = 50
	defaultSweepEvery           = 20
	defaultSweepWidth           = 64
)

func (o Options) resolved() Options {
	if o.Shape == (model.V41ExpertCacheShape{}) {
		o.Shape = model.V41ExpertCacheShapeDefault()
	}
	if o.Tokens <= 0 {
		o.Tokens = 200
	}
	if o.ExpertBytes <= 0 {
		o.ExpertBytes = defaultExpertBytes
	}
	if o.HotBytes <= 0 {
		o.HotBytes = int64(defaultHotSlots) * o.ExpertBytes
	}
	if o.TransientBytes <= 0 {
		o.TransientBytes = int64(defaultTransientSlots) * o.ExpertBytes
	}
	if o.LearnFraction <= 0 || o.LearnFraction >= 1 {
		o.LearnFraction = defaultLearnFraction
	}
	if o.TurnTokens <= 0 {
		o.TurnTokens = defaultTurnTokens
	}
	// SweepEvery: 0 takes the default cadence, a NEGATIVE value disables sweeps entirely (tests
	// that isolate the decode gate from the transient ring).
	switch {
	case o.SweepEvery == 0:
		o.SweepEvery = defaultSweepEvery
	case o.SweepEvery < 0:
		o.SweepEvery = -1
	}
	if o.SweepWidth <= 0 {
		o.SweepWidth = defaultSweepWidth
	}
	return o
}

// fixtureKind names the trace for the receipt: "trace-file" or the synthetic descriptor.
func (o Options) fixtureKind() string {
	if o.TraceFile != "" {
		return "trace-file"
	}
	if o.ZipfExponent > 0 {
		return fmt.Sprintf("zipf(s=%.4g)", o.ZipfExponent)
	}
	return "uniform"
}

// armCounters is one arm's raw outcome: the policy's own accounting plus the driver-side totals.
// Hits + Misses == TotalTouches by construction (every touch is served exactly once), asserted in
// runArm so a drifted counter fails the run instead of printing a phantom rate.
type armCounters struct {
	Stats           ctxmmu.ExpertResidencyStats
	TotalTouches    int
	BackingLoads    int
	LearnTurns      int
	LearnSwaps      int
	FinalHotSetSize int
	ObservedUnits   int
}

func (c armCounters) hits() int64 { return c.Stats.ResidentHits + c.Stats.TransientHits }

// misses is the demand-miss count: every touch served from backing because neither ring held it
// (the policy's own DemandMisses; ColdMisses counts backing REFUSALS, which serve nothing and are
// a different event — the driver errors on them instead of tallying them).
func (c armCounters) misses() int64 { return c.Stats.DemandMisses }

// hitRate is the routed-expert cache hit rate: the share of touches served without a backing read.
func (c armCounters) hitRate() float64 {
	total := c.hits() + c.misses()
	if total == 0 {
		return 0
	}
	return float64(c.hits()) / float64(total)
}

// receipt is the machine-readable output (--json). Measurement is ALWAYS "simulated": this driver
// has no hardware in the loop, so the rung is structural, mirroring the L1 receipt's discipline.
type receipt struct {
	Schema       string       `json:"schema"`
	Measurement  string       `json:"measurement"`
	Fixture      fixtureInfo  `json:"fixture"`
	Arms         []armReceipt `json:"arms"`
	DeltaHitRate float64      `json:"delta_hit_rate"`
	// TransientSizing is the advisory output of ctxmmu.SizeTransientRing over the learner arm's
	// observed breadth: what the learned working set WOULD budget, next to what the run actually
	// configured. Advisory only — the operator picks the final TransientBytes.
	TransientSizing transientSizing `json:"transient_sizing"`
}

type fixtureInfo struct {
	Kind           string  `json:"kind"`
	Seed           int64   `json:"seed,omitempty"`
	Tokens         int     `json:"tokens"`
	Layers         int     `json:"layers"`
	Experts        int     `json:"experts"`
	TopK           int     `json:"top_k"`
	HotSetSize     int     `json:"hot_set_size,omitempty"`
	ZipfExponent   float64 `json:"zipf_exponent,omitempty"`
	ExpertBytes    int64   `json:"expert_bytes"`
	HotBytes       int64   `json:"hot_bytes"`
	TransientBytes int64   `json:"transient_bytes"`
	LearnTokens    int     `json:"learn_tokens"`
	TurnTokens     int     `json:"turn_tokens"`
	SweepEvery     int     `json:"sweep_every"`
	SweepWidth     int     `json:"sweep_width"`
}

type armReceipt struct {
	Name          string `json:"name"`
	ResidentHits  int64  `json:"resident_hits"`
	TransientHits int64  `json:"transient_hits"`
	// DemandMisses is the hit-rate denominator partner (touches served from backing);
	// DecodeBypasses is the advisor-gated subset of it. ColdMisses counts backing refusals.
	DemandMisses       int64 `json:"demand_misses"`
	DecodeBypasses     int64 `json:"decode_bypasses"`
	ColdMisses         int64 `json:"cold_misses"`
	Promotions         int64 `json:"promotions"`
	HotEvictions       int64 `json:"hot_evictions"`
	TransientEvictions int64 `json:"transient_evictions"`
	// ResidentEntries / TransientEntries are the rings' end-of-run occupancy (the policy's own
	// accounting), so a reader can check hits/evicts against what the rings actually held.
	ResidentEntries  int     `json:"resident_entries"`
	TransientEntries int     `json:"transient_entries"`
	TotalTouches     int     `json:"total_touches"`
	HitRate          float64 `json:"hit_rate"`
	BackingLoads     int     `json:"backing_loads"`
	LearnTurns       int     `json:"learn_turns,omitempty"`
	LearnSwaps       int     `json:"learn_swaps,omitempty"`
	FinalHotSetSize  int     `json:"final_hot_set_size,omitempty"`
	ObservedUnits    int     `json:"observed_units,omitempty"`
}

type transientSizing struct {
	Recommended    int64 `json:"recommended"`
	Configured     int64 `json:"configured"`
	Observed       int   `json:"observed"`
	HotSet         int   `json:"hot_set"`
	AvgExpertBytes int64 `json:"avg_expert_bytes"`
	CapBytes       int64 `json:"cap_bytes"`
}

// fakeBacking is the counting ResidencyBacking: every Load moves ExpertBytes, so a backing load
// count IS a miss count, and "no payload re-read" stays a count assertion.
type fakeBacking struct {
	bytes int64
	loads int
}

func (b *fakeBacking) Load(layer, expert int) (ctxmmu.ExpertDescriptor, bool) {
	b.loads++
	return ctxmmu.ExpertDescriptor{Layer: layer, Expert: expert, Bytes: b.bytes}, true
}

// buildTrace produces the shared event stream: the synthetic trace (or the recorded one) as a
// flat (layer, expert) sequence.
func buildTrace(opts Options) ([]model.ExpertCacheTraceEvent, error) {
	if opts.TraceFile != "" {
		evs, err := loadTraceFile(opts.TraceFile)
		if err != nil {
			return nil, err
		}
		return evs, nil
	}
	trace, err := model.GenerateV41ExpertCacheTrace(model.ExpertCacheTraceOptions{
		Shape:        opts.Shape,
		Seed:         opts.Seed,
		Tokens:       opts.Tokens,
		HotSetSize:   opts.HotSetSize,
		ZipfExponent: opts.ZipfExponent,
		ExpertBytes:  opts.ExpertBytes,
	})
	if err != nil {
		return nil, err
	}
	return trace.Events, nil
}

// perTokenEvents is the events one decode step contributes at the resolved shape.
func perTokenEvents(o Options) int { return o.Shape.Layers * o.Shape.TopK }

// learnSplit returns how many leading tokens are the observation-only learning phase: the
// LearnFraction share of the whole stream, clamped to [0, len/perToken] and to at least one full
// token when the stream holds one. Both arms split identically, so the measured window is the same
// for every arm.
func learnSplit(events []model.ExpertCacheTraceEvent, o Options) int {
	perToken := perTokenEvents(o)
	n := len(events) / perToken
	learn := int(float64(n) * o.LearnFraction)
	if learn < 1 && n >= 1 {
		learn = 1
	}
	if learn > n {
		learn = n
	}
	return learn
}

// runArm replays events through one arm's policy and returns its counters. withLearner selects
// the learner-wired arm; both arms see the identical stream and budgets.
func runArm(events []model.ExpertCacheTraceEvent, opts Options, withLearner bool) (armCounters, error) {
	o := opts.resolved()
	shape := o.Shape
	perToken := shape.Layers * shape.TopK

	var (
		backing *fakeBacking
		policy  *ctxmmu.ExpertResidencyPolicy
		advisor *model.ExpertHotSetAdvisor
		err     error
	)
	backing = &fakeBacking{bytes: o.ExpertBytes}
	if withLearner {
		// The learner's slot budget matches the ring's entry capacity: the learned hot-set and the
		// ring select the same number of units.
		slots := int(o.HotBytes / o.ExpertBytes)
		if slots < 1 {
			slots = 1
		}
		advisor = model.NewExpertHotSetAdvisor(slots, model.DefaultExpertHotSetHysteresis)
		policy, err = ctxmmu.NewExpertResidencyPolicyWithAdvisor(
			ctxmmu.ExpertResidencyOptions{HotBytes: o.HotBytes, TransientBytes: o.TransientBytes},
			backing, advisor)
	} else {
		policy, err = ctxmmu.NewExpertResidencyPolicy(
			ctxmmu.ExpertResidencyOptions{HotBytes: o.HotBytes, TransientBytes: o.TransientBytes},
			backing)
	}
	if err != nil {
		return armCounters{}, err
	}

	// Learning phase: observation only, so both arms enter the measured window with empty rings.
	learnTokens := learnSplit(events, o)
	measured := events[learnTokens*perToken:]
	if withLearner {
		for _, ev := range events[:learnTokens*perToken] {
			advisor.Observe(ev.Layer, ev.Expert)
		}
		advisor.Learn()
		policy.WarmStartFromAdvisor()
	}

	var c armCounters
	sweep := 0
	for ti := 0; ti < len(measured); ti += perToken {
		// tok is the MEASURED token index (ti is the event offset): sweeps and turn boundaries
		// are cadenced in tokens, never events.
		tok := ti / perToken
		// Turn boundary: the learner re-commits its hot-set under the hysteresis band. Zero swaps
		// on a stable workload is the no-thrash witness; the count is the pin-stability metric.
		if withLearner && tok > 0 && tok%o.TurnTokens == 0 {
			c.LearnTurns++
			c.LearnSwaps += len(advisor.Learn())
		}
		// Prefill sweep: a broad one-shot union exercising the transient ring, injected at a
		// deterministic offset so both arms see the same sweep. Disabled when SweepEvery < 0.
		if o.SweepEvery > 0 && tok%o.SweepEvery == 0 {
			limit := o.SweepWidth
			if limit > shape.Experts {
				limit = shape.Experts
			}
			layer := sweep % shape.Layers
			for i := 0; i < limit; i++ {
				expert := (sweep*o.SweepWidth + i) % shape.Experts
				if withLearner {
					advisor.Observe(layer, expert)
				}
				if _, ok := policy.Prefill(layer, expert); !ok {
					return armCounters{}, fmt.Errorf("hotsetcheck: prefill (%d,%d) refused by backing", layer, expert)
				}
				c.TotalTouches++
			}
			sweep++
		}
		// Decode touches: observed into the learner first (the live path folds every staged
		// projection into the turn histogram), then decided by the policy.
		for _, ev := range measured[ti : ti+perToken] {
			if withLearner {
				advisor.Observe(ev.Layer, ev.Expert)
			}
			if _, ok := policy.Decode(ev.Layer, ev.Expert); !ok {
				return armCounters{}, fmt.Errorf("hotsetcheck: decode (%d,%d) refused by backing", ev.Layer, ev.Expert)
			}
			c.TotalTouches++
		}
	}

	c.Stats = policy.Stats()
	c.BackingLoads = backing.loads
	if withLearner {
		c.FinalHotSetSize = len(advisor.HotSet())
		c.ObservedUnits = advisor.ObservedCount()
	}
	// Closed contract: every touch is served exactly once, by exactly one tier or a bypass.
	if got, want := c.hits()+c.misses(), int64(c.TotalTouches); got != want {
		return armCounters{}, fmt.Errorf("hotsetcheck: arm tally %d+%d != %d touches", c.hits(), c.misses(), c.TotalTouches)
	}
	return c, nil
}

// runFixture drives both arms over one shared stream and assembles the receipt.
func runFixture(opts Options) (receipt, error) {
	o := opts.resolved()
	events, err := buildTrace(o)
	if err != nil {
		return receipt{}, err
	}
	base, err := runArm(events, o, false)
	if err != nil {
		return receipt{}, fmt.Errorf("hotsetcheck: baseline arm: %w", err)
	}
	learn, err := runArm(events, o, true)
	if err != nil {
		return receipt{}, fmt.Errorf("hotsetcheck: learner arm: %w", err)
	}
	learnTokens := learnSplit(events, o)
	return receipt{
		Schema:      ReceiptSchema,
		Measurement: "simulated",
		Fixture: fixtureInfo{
			Kind:           o.fixtureKind(),
			Seed:           o.Seed,
			Tokens:         o.Tokens,
			Layers:         o.Shape.Layers,
			Experts:        o.Shape.Experts,
			TopK:           o.Shape.TopK,
			HotSetSize:     o.HotSetSize,
			ZipfExponent:   o.ZipfExponent,
			ExpertBytes:    o.ExpertBytes,
			HotBytes:       o.HotBytes,
			TransientBytes: o.TransientBytes,
			LearnTokens:    learnTokens,
			TurnTokens:     o.TurnTokens,
			SweepEvery:     o.SweepEvery,
			SweepWidth:     o.SweepWidth,
		},
		Arms: []armReceipt{
			armReceiptOf("baseline-lru", base),
			armReceiptOf("learner-wired", learn),
		},
		DeltaHitRate: learn.hitRate() - base.hitRate(),
		TransientSizing: transientSizing{
			Recommended:    ctxmmu.SizeTransientRing(learn.ObservedUnits, learn.FinalHotSetSize, o.ExpertBytes, o.TransientBytes),
			Configured:     o.TransientBytes,
			Observed:       learn.ObservedUnits,
			HotSet:         learn.FinalHotSetSize,
			AvgExpertBytes: o.ExpertBytes,
			CapBytes:       o.TransientBytes,
		},
	}, nil
}

func armReceiptOf(name string, c armCounters) armReceipt {
	return armReceipt{
		Name:               name,
		ResidentHits:       c.Stats.ResidentHits,
		TransientHits:      c.Stats.TransientHits,
		DemandMisses:       c.Stats.DemandMisses,
		DecodeBypasses:     c.Stats.DecodeBypasses,
		ColdMisses:         c.Stats.ColdMisses,
		Promotions:         c.Stats.Promotions,
		HotEvictions:       c.Stats.HotEvictions,
		TransientEvictions: c.Stats.TransientEvicts,
		ResidentEntries:    c.Stats.ResidentEntries,
		TransientEntries:   c.Stats.TransientEntries,
		TotalTouches:       c.TotalTouches,
		HitRate:            c.hitRate(),
		BackingLoads:       c.BackingLoads,
		LearnTurns:         c.LearnTurns,
		LearnSwaps:         c.LearnSwaps,
		FinalHotSetSize:    c.FinalHotSetSize,
		ObservedUnits:      c.ObservedUnits,
	}
}
