package model

// expert_cache_bench.go — the SIMULATED routed-expert cache driver (issue #12990).
//
// One narrow question, no hardware in the loop: given a deterministic synthetic routing trace at
// the pinned V4.1 shape, how many routed-expert touches does the SHIPPED expert-ring residency
// decision serve from the resident tier (hit) versus stream from the next tier (miss), as the ring
// byte budget grows? It is the L1 "simulated" rung of the moe-expert-cache-benchmarks suite: a
// receipt emitted here is structurally marked ExpertCacheMeasurementSimulated and can never be
// mistaken for a hardware measurement.
//
// The replay drives the SHIPPED ring, not a local cache model: the residency policy is
// polymodel.Pool's, bound per-weight by pagedRing, and re-implementing an eviction loop here would
// measure a policy that does not ship. Every reported counter comes from the ring's OWN
// bookkeeping — HitCount = r.hit, MissCount = r.pageIn, Refusals = r.refused (with
// r.lookups == r.hit + r.pageIn + r.refused), BytesReadNVMe = the miss events' bytes
// (cross-checked against r.pageInBytes), BytesReadDRAM = the hit events' bytes (tallied per event
// against the ring's hit transition; the ring counts hits, not the bytes those hits read),
// BytesRefused = the refused events' bytes (the three-way split totals the trace's bytes exactly),
// PeakResidentBytes = r.peakUsed().
//
// Layer-major trace consequence: a routed expert is keyed by (layer, expert), so before any
// cross-token repeat can hit, the ring must hold a resident entry for each layer. A budget below
// ~Layers*TopK*ExpertBytes measures an almost-pure cold-stream regime regardless of skew; that is
// the shipped ring's decision and is deliberately not corrected here.
//
// Scope fence: no real weights, no kernels, no new ring policy, no CLI verb, no hardware throughput
// claim. The mk callback returns a 1x1 f32 tensor because the residency DECISION is what is
// measured, never the GEMM; throughput/latency fields stay zero (unmeasured), since fabricating
// them would be the simulated-as-hardware conflation the measurement rung exists to prevent.

import (
	"fmt"
	"math/rand"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// V41ExpertCacheShape is the pinned routed geometry the driver drives: Layers (the resident-set
// height a layer-major trace must span before any reuse can hit), Experts (the id space), and TopK
// (picks per layer). MoeInter is echoed into the receipt so the measurement is stamped at the
// V4.1 expert width. HiddenSize is carried for receipt identity only — it is NOT read by the
// trace, the replay, or any counter, so no hidden-size-dependent claim may be read out of a
// receipt this driver emits.
type V41ExpertCacheShape struct {
	Layers     int
	Experts    int
	TopK       int
	MoeInter   int
	HiddenSize int
}

// V41ExpertCacheShapeDefault returns the pinned V4.1 routed geometry (40 layers, 384 routed
// experts, top-6, 2304 expert intermediate width, 5120 hidden). The MoE axes are reused from the
// shipped V4.1 router constants so a geometry change there cannot leave this driver measuring a
// stale shape; the 40/5120 decoder axes are the pinned DeepSeekV41AttentionGeometry values the
// shipped admission check asserts.
func V41ExpertCacheShapeDefault() V41ExpertCacheShape {
	return V41ExpertCacheShape{
		Layers:     40, // DeepSeekV41AttentionGeometry.NumLayers, asserted by V4.1 admission
		Experts:    V41RouterExperts,
		TopK:       V41RouterTopK,
		MoeInter:   V41RouterMoEWidth,
		HiddenSize: 5120, // DeepSeekV41AttentionGeometry.HiddenSize, asserted by V4.1 admission
	}
}

// ExpertCacheTraceOptions parameterizes the deterministic synthetic routing trace; same options
// yield a byte-identical trace (see GenerateV41ExpertCacheTrace's determinism contract).
type ExpertCacheTraceOptions struct {
	// Shape is the routed geometry the trace is drawn at; the zero value takes V41ExpertCacheShapeDefault().
	Shape V41ExpertCacheShape
	// Seed is the PRNG seed; the same seed reproduces the same trace.
	Seed int64
	// Tokens is the number of decode steps simulated; each contributes Shape.Layers * Shape.TopK events.
	Tokens int
	// HotSetSize is the number of "hot" experts (indices 0..HotSetSize-1) skewed traffic concentrates
	// on; <= 0 or >= Experts means no hot set. Only meaningful when ZipfExponent > 0.
	HotSetSize int
	// ZipfExponent is the skew: 0 => uniform over all experts; > 0 => a Zipf-like mass on the hot
	// set, so a smaller hot set or larger exponent tightens the working set and raises the hit rate.
	ZipfExponent float64
	// ExpertBytes is the resident bytes charged per (layer, expert) weight; the ring budgets on it
	// and the trace records it per event so byte accounting survives a future per-expert size.
	ExpertBytes int64
}

// ExpertCacheTraceEvent is one routed-expert touch: the (layer, expert) pair the router selected
// and the resident bytes its staged weight accounts.
type ExpertCacheTraceEvent struct {
	Layer       int
	Expert      int
	WeightBytes int64
}

// ExpertCacheTrace is a deterministic routing trace at a shape. Plain value (no maps, no
// pointers) so two traces generated from the same options compare equal with reflect.DeepEqual.
type ExpertCacheTrace struct {
	Shape  V41ExpertCacheShape
	Seed   int64
	Tokens int
	Events []ExpertCacheTraceEvent
}

// GenerateV41ExpertCacheTrace builds the synthetic routing trace.
//
// Determinism contract: the trace is a pure function of (Shape, Seed, Tokens, HotSetSize,
// ZipfExponent, ExpertBytes) — no clock, no global RNG, no map iteration; the only randomness is
// math/rand seeded from Seed, mirroring GenerateExpertReplaySyntheticTrace (expert_replay.go).
//
// Shape accuracy: for each token every layer routes exactly TopK DISTINCT experts in
// [0, Experts), deduplicated by linear probing ((e+1) % Experts), so a token never routes the same
// expert twice in one layer even when the skew draws a repeat.
func GenerateV41ExpertCacheTrace(opts ExpertCacheTraceOptions) (ExpertCacheTrace, error) {
	if opts.Shape == (V41ExpertCacheShape{}) {
		opts.Shape = V41ExpertCacheShapeDefault()
	}
	shape := opts.Shape
	if shape.Layers <= 0 || shape.Experts <= 0 || shape.TopK <= 0 {
		return ExpertCacheTrace{}, fmt.Errorf("expert cache trace: non-positive shape %+v", shape)
	}
	if shape.TopK > shape.Experts {
		return ExpertCacheTrace{}, fmt.Errorf("expert cache trace: top_k %d exceeds experts %d", shape.TopK, shape.Experts)
	}
	if opts.Tokens <= 0 {
		return ExpertCacheTrace{}, fmt.Errorf("expert cache trace: tokens must be positive, got %d", opts.Tokens)
	}
	if opts.ExpertBytes <= 0 {
		return ExpertCacheTrace{}, fmt.Errorf("expert cache trace: expert bytes must be positive, got %d", opts.ExpertBytes)
	}

	// The hot set is the low expert indices; clamping to Experts keeps the generator total rather
	// than silently truncating. A hot set covering the whole range is exactly the uniform case.
	hotSet := opts.HotSetSize
	if hotSet <= 0 || hotSet > shape.Experts {
		hotSet = shape.Experts
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	// Zipf over [0, hotSet-1] when skewed; nil in the uniform case. rand.NewZipf panics on invalid
	// params, so only construct it for a real hot-set regime.
	var zipf *rand.Zipf
	if opts.ZipfExponent > 0 && hotSet < shape.Experts {
		zipf = rand.NewZipf(rng, opts.ZipfExponent, 1, uint64(hotSet-1))
	}
	// Shares rng so the seed governs the uniform case too, keeping the draw branch single-sourced.
	uniform := rng

	events := make([]ExpertCacheTraceEvent, 0, opts.Tokens*shape.Layers*shape.TopK)
	// dedupRange bounds the duplicate probe walk: at least TopK, because a layer must route TopK
	// DISTINCT experts and a hot set narrower than TopK cannot supply them. Probing across the FULL
	// expert range would spray marginal picks uniformly over the cold set and wash out the
	// concentration the skew exists to create; probing only within a hot set narrower than TopK
	// would spin forever. Clamping to TopK keeps the walk bounded and picks distinct.
	dedupRange := shape.Experts
	if zipf != nil {
		dedupRange = hotSet
		if dedupRange < shape.TopK {
			dedupRange = shape.TopK
		}
	}
	for token := 0; token < opts.Tokens; token++ {
		for layer := 0; layer < shape.Layers; layer++ {
			seen := make(map[int]bool, shape.TopK)
			for pick := 0; pick < shape.TopK; pick++ {
				var expert int
				if zipf != nil {
					expert = int(zipf.Uint64())
				} else {
					expert = uniform.Intn(shape.Experts)
				}
				for seen[expert] {
					expert = (expert + 1) % dedupRange
				}
				seen[expert] = true
				events = append(events, ExpertCacheTraceEvent{
					Layer:       layer,
					Expert:      expert,
					WeightBytes: opts.ExpertBytes,
				})
			}
		}
	}
	return ExpertCacheTrace{Shape: shape, Seed: opts.Seed, Tokens: opts.Tokens, Events: events}, nil
}

// ExpertCacheReplayResult is the raw outcome of replaying a trace through the shipped ring: pure
// counters, no derived rate or throughput. Every field comes from the pagedRing's own
// bookkeeping (see the file header), so the result audits against the ring, not this driver.
type ExpertCacheReplayResult struct {
	// RingByteCap is the budget the replay ran under.
	RingByteCap int64
	// HitCount is r.hit  -  touches served from the resident tier.
	HitCount int64
	// MissCount is r.pageIn  -  touches that admitted a cold weight.
	MissCount int64
	// Evictions is r.evict  -  LRU page-outs made to stay within budget.
	Evictions int64
	// Refusals is r.refused  -  touches the budget could not admit at all.
	Refusals int64
	// Lookups is r.lookups; HitCount+MissCount+Refusals == Lookups.
	Lookups int64
	// BytesReadDRAM is the resident-tier bytes served on a hit: this event's own WeightBytes summed
	// over the events the ring served resident. The ring counts hits, not the bytes those hits
	// read, so this side is tallied here against the ring's hit counter; it is NOT a ring counter.
	// For a single-size trace it equals HitCount * ExpertBytes.
	BytesReadDRAM int64
	// BytesReadNVMe is the streamed-tier bytes a miss moved: r.pageInBytes, the ring's own upload-byte tally.
	BytesReadNVMe int64
	// BytesRefused is the bytes of events the budget could not admit at all (r.refused): served by
	// neither tier, so booked here rather than silently dropped. The replay enforces
	// BytesReadDRAM + BytesReadNVMe + BytesRefused == total trace bytes; a non-zero value means the
	// budget is too small to admit individual weights, and the receipt must not present the tier
	// split as complete.
	BytesRefused int64
	// PeakResidentBytes is r.peakUsed()  -  the high-water mark the ring held,
	// never an invented footprint.
	PeakResidentBytes int64
}

// ReplayExpertCacheTrace drives a trace through the shipped ring residency decision and folds
// the ring's counters into a result.
//
// Each event is staged under the canonical routed-expert tensor name
// (expertName(layer, expert, "gate_proj.weight")), so the ring key carries the real (layer,
// expert) identity the shipped HAL uses. The mk callback returns a 1x1 f32 tensor: the residency
// decision is measured, not the math. ringByteCap <= 0 is refused: a ring with no budget admits
// nothing, and the resulting 100% refusal rate would be a meaningless receipt.
func ReplayExpertCacheTrace(trace ExpertCacheTrace, ringByteCap int64) (ExpertCacheReplayResult, error) {
	if ringByteCap <= 0 {
		return ExpertCacheReplayResult{}, fmt.Errorf("expert cache replay: ring byte cap must be positive, got %d", ringByteCap)
	}
	if len(trace.Events) == 0 {
		return ExpertCacheReplayResult{}, fmt.Errorf("expert cache replay: empty trace")
	}
	ring := newPagedRing(nil, ringByteCap)
	// Per-tier bytes are tallied per event from THIS access's own footprint, so a trace with
	// per-expert sizes attributes the right bytes to each hit.
	var dramBytes, nvmeBytes, refusedBytes, totalBytes int64
	var hits, misses, refusals int64
	for _, ev := range trace.Events {
		if ev.WeightBytes <= 0 {
			return ExpertCacheReplayResult{}, fmt.Errorf("expert cache replay: event weight bytes must be positive, got %d", ev.WeightBytes)
		}
		totalBytes += ev.WeightBytes
		name := expertName(ev.Layer, ev.Expert, "gate_proj.weight")
		beforeHit, beforePageIn, beforeRefused := ring.hit, ring.pageIn, ring.refused
		ring.stage(name, func() compute.Tensor {
			return compute.NewF32(compute.Default(), []int{1, 1}, []float32{1})
		}, compute.F32, ev.WeightBytes, false)
		// Attribute this event's bytes to the tier the ring actually chose for it; a refusal is
		// served by NO tier, so its bytes are booked separately rather than dropped.
		switch {
		case ring.hit > beforeHit:
			hits++
			dramBytes += ev.WeightBytes
		case ring.pageIn > beforePageIn:
			misses++
			nvmeBytes += ev.WeightBytes
		case ring.refused > beforeRefused:
			refusals++
			refusedBytes += ev.WeightBytes
		default:
			return ExpertCacheReplayResult{}, fmt.Errorf("expert cache replay: event for layer %d expert %d changed no ring counter", ev.Layer, ev.Expert)
		}
	}
	if int64(ring.hit) != hits || int64(ring.pageIn) != misses || int64(ring.refused) != refusals {
		return ExpertCacheReplayResult{}, fmt.Errorf("expert cache replay: ring counters (%d hits, %d page-ins, %d refused) diverged from per-event tally (%d/%d/%d)", ring.hit, ring.pageIn, ring.refused, hits, misses, refusals)
	}
	if got, want := dramBytes+nvmeBytes+refusedBytes, totalBytes; got != want {
		// Every event is booked to exactly one tier, so this holds for ANY per-event sizes.
		return ExpertCacheReplayResult{}, fmt.Errorf("expert cache replay: tier bytes %d+%d+%d=%d != trace total %d", dramBytes, nvmeBytes, refusedBytes, got, want)
	}
	// Cross-check the streamed tier against the ring's OWN byte counter (pageInBytes): both sides
	// accrue from the same per-event WeightBytes, so this catches a missed or double-counted
	// transition, not a wrong WeightBytes.
	if nvmeBytes != ring.pageInBytes {
		return ExpertCacheReplayResult{}, fmt.Errorf("expert cache replay: streamed bytes %d != ring pageInBytes %d", nvmeBytes, ring.pageInBytes)
	}

	return ExpertCacheReplayResult{
		RingByteCap: ringByteCap,
		HitCount:    hits,
		MissCount:   misses,
		Evictions:   int64(ring.evict),
		Refusals:    refusals,
		Lookups:     int64(ring.lookups),
		// All three tiers derive from the ring's own outcome counters; the ring's pageInBytes is
		// cross-checked against the streamed total above.
		BytesReadDRAM:     dramBytes,
		BytesReadNVMe:     nvmeBytes,
		BytesRefused:      refusedBytes,
		PeakResidentBytes: ring.peakUsed(),
	}, nil
}

// BuildExpertCacheReceipt folds a replay result and a shape into the locked L1 receipt.
// Measurement is ALWAYS ExpertCacheMeasurementSimulated: this driver has no hardware in the loop,
// so the rung is structural, not a caller choice.
//
// Throughput and latency stay zero on purpose: no GEMM and no clock, so prefill/decode
// tokens-per-sec and TTFT are unmeasured; the L1 schema carries the fields for the hardware leaf
// that can actually time a forward pass. Writing a non-zero number here would fabricate a
// throughput claim the measurement rung forbids.
//
// Every per-tier byte figure is folded straight from the replay result, refused bytes included:
// dropping them would leave bytes_read_dram + bytes_read_nvme short of the trace total with no
// structural signal. bytes_refused is the signal, and ExpertCacheTierSplitComplete(bytes_refused)
// is the derived completeness rung.
func BuildExpertCacheReceipt(shape V41ExpertCacheShape, res ExpertCacheReplayResult) ExpertCacheReceipt {
	rate, known := ExpertCacheHitRate(res.HitCount, res.MissCount)
	return ExpertCacheReceipt{
		Schema:            ExpertCacheReceiptSchema,
		Measurement:       ExpertCacheMeasurementSimulated,
		ModelID:           DeepSeekV41FlashModelID,
		Arch:              "deepseek_v41_text",
		Precision:         "fp4",
		Quant:             "native-fp4",
		Layers:            shape.Layers,
		Experts:           shape.Experts,
		TopK:              shape.TopK,
		MoeInter:          shape.MoeInter,
		RingByteCap:       res.RingByteCap,
		HitCount:          res.HitCount,
		MissCount:         res.MissCount,
		HitRate:           rate,
		HitRateKnown:      known,
		BytesReadDRAM:     res.BytesReadDRAM,
		BytesReadNVMe:     res.BytesReadNVMe,
		BytesRefused:      res.BytesRefused,
		PrefillToksPerSec: 0, // unmeasured: no clock, no GEMM (see doc above)
		DecodeToksPerSec:  0, // unmeasured
		TTFTMillis:        0, // unmeasured
		PeakMemoryBytes:   res.PeakResidentBytes,
	}
}
