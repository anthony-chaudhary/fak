package model

// expert_cache_bench_test.go  -  witnesses for the SIMULATED routed-expert cache
// driver (issue #12990). Every test name carries ExpertCacheBench so the issue
// witness regex `-run 'ExpertCacheBench|SyntheticTrace'` selects the suite, and
// the determinism witness carries SyntheticTrace explicitly.
//
// The assertions are the driver's honesty contract, not its performance: a
// trace is reproducible, shape-accurate, and replay hit rate is monotone in the
// ring budget; a receipt is field-locked, simulated-rung, and seed-live.

import (
	"encoding/json"
	"reflect"
	"testing"
)

func benchShape(t *testing.T) V41ExpertCacheShape {
	t.Helper()
	shape := V41ExpertCacheShapeDefault()
	if shape.Layers != 40 || shape.Experts != V41RouterExperts || shape.TopK != V41RouterTopK ||
		shape.MoeInter != V41RouterMoEWidth || shape.HiddenSize != 5120 {
		t.Fatalf("default shape drifted: %+v", shape)
	}
	return shape
}

func benchTrace(t *testing.T, opts ExpertCacheTraceOptions) ExpertCacheTrace {
	t.Helper()
	trace, err := GenerateV41ExpertCacheTrace(opts)
	if err != nil {
		t.Fatalf("GenerateV41ExpertCacheTrace: %v", err)
	}
	return trace
}

// TestExpertCacheBenchSyntheticTraceDeterministic pins the determinism
// contract: the same options produce byte-identical traces, and a different
// seed does not.
func TestExpertCacheBenchSyntheticTraceDeterministic(t *testing.T) {
	opts := ExpertCacheTraceOptions{
		Shape:        benchShape(t),
		Seed:         42,
		Tokens:       8,
		HotSetSize:   32,
		ZipfExponent: 1.25,
		ExpertBytes:  64,
	}
	a := benchTrace(t, opts)
	b := benchTrace(t, opts)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same options produced different traces")
	}
	if len(a.Events) != len(b.Events) {
		t.Fatalf("event count differs: %d != %d", len(a.Events), len(b.Events))
	}

	other := opts
	other.Seed = opts.Seed + 1
	c := benchTrace(t, other)
	if reflect.DeepEqual(a, c) {
		t.Fatal("different seed produced an identical trace; the seed is not live")
	}
}

// TestExpertCacheBenchTraceShapeAccurate pins that a trace routes every layer of
// every token to exactly TopK DISTINCT in-range experts, at the pinned shape.
func TestExpertCacheBenchTraceShapeAccurate(t *testing.T) {
	shape := benchShape(t)
	trace := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 7, Tokens: 5, HotSetSize: 24, ZipfExponent: 1.5, ExpertBytes: 64,
	})
	want := trace.Tokens * shape.Layers * shape.TopK
	if len(trace.Events) != want {
		t.Fatalf("event count %d != tokens*layers*topk %d", len(trace.Events), want)
	}

	perToken := shape.Layers * shape.TopK
	for tok := 0; tok < trace.Tokens; tok++ {
		for layer := 0; layer < shape.Layers; layer++ {
			seen := map[int]bool{}
			base := tok*perToken + layer*shape.TopK
			for i := 0; i < shape.TopK; i++ {
				ev := trace.Events[base+i]
				if ev.Layer != layer {
					t.Fatalf("event %d layer %d != %d", base+i, ev.Layer, layer)
				}
				if ev.Expert < 0 || ev.Expert >= shape.Experts {
					t.Fatalf("expert %d out of range [0,%d)", ev.Expert, shape.Experts)
				}
				if seen[ev.Expert] {
					t.Fatalf("token %d layer %d routed expert %d twice", tok, layer, ev.Expert)
				}
				seen[ev.Expert] = true
			}
		}
	}
	if shape.Layers != 40 || shape.Experts != 384 || shape.TopK != 6 || shape.MoeInter != 2304 || shape.HiddenSize != 5120 {
		t.Fatalf("shape is not the pinned V4.1 geometry: %+v", shape)
	}
}

// TestExpertCacheBenchReplayMonotoneInRingCap pins the core property: for a
// fixed trace, hit rate never falls as the ring budget grows, and once the
// budget holds the trace's whole touched working set the rate saturates (a
// larger budget cannot change it).
func TestExpertCacheBenchReplayMonotoneInRingCap(t *testing.T) {
	shape := benchShape(t)
	const expertBytes = 64
	trace := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 99, Tokens: 6, HotSetSize: 12, ZipfExponent: 1.5, ExpertBytes: expertBytes,
	})

	// Every distinct (layer, expert) the trace touches, times the per-event
	// bytes: a budget at or above this can admit the whole working set, so the
	// rate must saturate there. (It is far below Layers*Experts because a
	// handful of tokens touch only a few experts per layer.)
	distinct := int64(shape.Layers * shape.TopK * trace.Tokens)
	workingSet := distinct * expertBytes
	caps := []int64{
		expertBytes,                     // one expert
		expertBytes * int64(shape.TopK), // one layer's top-k
		expertBytes * int64(shape.TopK) * 2,
		workingSet,     // the whole touched working set
		workingSet * 2, // twice it: must not change the saturated rate
	}

	prev := -1.0
	var rates []float64
	for _, cap := range caps {
		res, err := ReplayExpertCacheTrace(trace, cap)
		if err != nil {
			t.Fatalf("ReplayExpertCacheTrace(cap=%d): %v", cap, err)
		}
		if res.Lookups != int64(len(trace.Events)) {
			t.Fatalf("cap %d: lookups %d != events %d", cap, res.Lookups, len(trace.Events))
		}
		if res.HitCount+res.MissCount+res.Refusals != res.Lookups {
			t.Fatalf("cap %d: hit+miss+refused %d != lookups %d", cap, res.HitCount+res.MissCount+res.Refusals, res.Lookups)
		}
		if res.PeakResidentBytes > res.RingByteCap {
			t.Fatalf("cap %d: peak resident %d exceeds the ring cap", cap, res.PeakResidentBytes)
		}
		if cap < workingSet && res.Evictions == 0 {
			t.Fatalf("cap %d below the working set %d must evict to enforce the bound; got 0 evictions", cap, workingSet)
		}
		rate, known := ExpertCacheHitRate(res.HitCount, res.MissCount)
		if !known {
			t.Fatalf("cap %d: hit rate unknown for a non-empty trace", cap)
		}
		if rate+tinyEpsilon < prev {
			t.Fatalf("hit rate fell as budget grew: cap %d rate %v < previous %v", cap, rate, prev)
		}
		prev = rate
		rates = append(rates, rate)
	}
	if rates[len(rates)-2] < rates[len(rates)-1]-tinyEpsilon {
		t.Fatalf("whole-working-set rate %v < double-budget rate %v; not saturated", rates[len(rates)-2], rates[len(rates)-1])
	}
	if rates[0] >= rates[len(rates)-1]-tinyEpsilon {
		t.Fatalf("one-expert budget rate %v equals the saturated rate %v; the budget is inert", rates[0], rates[len(rates)-1])
	}
	// The sweep must actually observe a STRICT improvement, not a single 0 -> x
	// step that lets every smaller cap collapse to zero. At least one adjacent
	// pair must show a strict rise.
	strictRise := false
	for i := 1; i < len(rates); i++ {
		if rates[i] > rates[i-1]+tinyEpsilon {
			strictRise = true
		}
	}
	if !strictRise {
		t.Fatalf("no strict hit-rate rise across caps %v; the monotone sweep is vacuous", rates)
	}
}

// TestExpertCacheBenchRefusalAccounting pins the tier-byte identity in the one
// regime that breaks a naive DRAM+NVMe total: a budget too small to admit a
// single weight. Every event is then refused and its bytes are booked to
// BytesRefused, so DRAM+NVMe+Refused still totals the trace and DRAM+NVMe alone
// does not. This is the case the whole-working-set budget test cannot reach.
func TestExpertCacheBenchRefusalAccounting(t *testing.T) {
	shape := benchShape(t)
	const expertBytes = 64
	trace := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 23, Tokens: 3, HotSetSize: 8, ZipfExponent: 1.5, ExpertBytes: expertBytes,
	})
	// One byte below a single expert's footprint: every admit is ErrTooLarge.
	res, err := ReplayExpertCacheTrace(trace, expertBytes-1)
	if err != nil {
		t.Fatalf("ReplayExpertCacheTrace: %v", err)
	}
	total := int64(len(trace.Events)) * expertBytes
	if res.Refusals != int64(len(trace.Events)) {
		t.Fatalf("refusals %d != events %d at sub-expert budget", res.Refusals, len(trace.Events))
	}
	if res.BytesReadDRAM != 0 || res.BytesReadNVMe != 0 {
		t.Fatalf("no tier can serve a refused event: dram=%d nvme=%d", res.BytesReadDRAM, res.BytesReadNVMe)
	}
	if res.BytesReadDRAM+res.BytesReadNVMe+res.BytesRefused != total {
		t.Fatalf("tier bytes %d+%d+%d != trace total %d",
			res.BytesReadDRAM, res.BytesReadNVMe, res.BytesRefused, total)
	}
	if res.BytesRefused != total {
		t.Fatalf("refused bytes %d != trace total %d", res.BytesRefused, total)
	}
}

// TestExpertCacheBenchSkewConcentratesTraffic pins that the hot set actually
// concentrates traffic, rather than merely changing it: with a skew, the hot
// experts' share of touches must exceed their share of the id space by a wide
// margin. Without this, the skew test would compare two near-uniform traces.
func TestExpertCacheBenchSkewConcentratesTraffic(t *testing.T) {
	shape := benchShape(t)
	const expertBytes = 64
	const hotSet = 8
	trace := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 11, Tokens: 40, HotSetSize: hotSet, ZipfExponent: 1.5, ExpertBytes: expertBytes,
	})
	var hot, cold int
	for _, ev := range trace.Events {
		if ev.Expert < hotSet {
			hot++
		} else {
			cold++
		}
	}
	share := float64(hot) / float64(hot+cold)
	idShare := float64(hotSet) / float64(shape.Experts)
	if share < 3*idShare {
		t.Fatalf("hot set holds %.3f of touches but %.3f of the id space; not concentrated", share, idShare)
	}
	if cold != 0 {
		// Dedup walks within the hot set, so a skewed trace never emits a cold
		// expert; that is the direct observable of a real concentration.
		t.Fatalf("skewed trace emitted %d cold-expert touches; dedup leaked outside the hot set", cold)
	}
}

const tinyEpsilon = 1e-9

// TestExpertCacheBenchHotSetBelowTopKLeaks documents the one regime where a
// skewed trace is NOT confined to its hot set: a hot set narrower than TopK
// cannot supply TopK distinct picks, so the dedup walk spills into adjacent ids
// up to the clamp max(hotSet, TopK). Distinctness and the clamp bound are
// preserved; confinement to the hot set is not, and this test pins that
// honestly rather than letting the spill read as a bug.
func TestExpertCacheBenchHotSetBelowTopKLeaks(t *testing.T) {
	shape := benchShape(t)
	const hotSet = 2 // < TopK (6)
	trace := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 13, Tokens: 4, HotSetSize: hotSet, ZipfExponent: 1.5, ExpertBytes: 64,
	})
	var spilled int
	for _, ev := range trace.Events {
		if ev.Expert >= hotSet {
			if ev.Expert >= shape.TopK {
				t.Fatalf("expert %d spilled past the dedup clamp TopK=%d", ev.Expert, shape.TopK)
			}
			spilled++
		}
	}
	if spilled == 0 {
		t.Fatal("hot set narrower than TopK should spill past it to reach TopK distinct picks")
	}
	// Distinctness still holds in every layer, which is the invariant that must
	// never break.
	perToken := shape.Layers * shape.TopK
	for tok := 0; tok < trace.Tokens; tok++ {
		for layer := 0; layer < shape.Layers; layer++ {
			seen := map[int]bool{}
			base := tok*perToken + layer*shape.TopK
			for i := 0; i < shape.TopK; i++ {
				ev := trace.Events[base+i]
				if seen[ev.Expert] {
					t.Fatalf("expert %d routed twice in token %d layer %d", ev.Expert, tok, layer)
				}
				seen[ev.Expert] = true
			}
		}
	}
}

// TestExpertCacheBenchMixedSizeTraceReconciles pins that the exported replay
// accepts a trace whose events carry DIFFERENT footprints (the generator only
// emits uniform ones, but the API does not require it) and still reconciles the
// three-way tier split against the true per-event total, not a uniform guess.
func TestExpertCacheBenchMixedSizeTraceReconciles(t *testing.T) {
	shape := V41ExpertCacheShape{Layers: 2, Experts: 8, TopK: 2, MoeInter: 2304, HiddenSize: 5120}
	sizes := []int64{64, 128, 96, 64}
	trace := ExpertCacheTrace{Shape: shape, Seed: 1, Tokens: 2, Events: []ExpertCacheTraceEvent{
		{Layer: 0, Expert: 0, WeightBytes: sizes[0]},
		{Layer: 0, Expert: 1, WeightBytes: sizes[1]},
		{Layer: 1, Expert: 0, WeightBytes: sizes[2]},
		{Layer: 1, Expert: 1, WeightBytes: sizes[3]},
	}}
	var total int64
	for _, s := range sizes {
		total += s
	}
	res, err := ReplayExpertCacheTrace(trace, 100) // admits 64/96, refuses 128
	if err != nil {
		t.Fatalf("ReplayExpertCacheTrace on a mixed-size trace: %v", err)
	}
	if res.BytesReadDRAM+res.BytesReadNVMe+res.BytesRefused != total {
		t.Fatalf("tier bytes %d+%d+%d != true total %d",
			res.BytesReadDRAM, res.BytesReadNVMe, res.BytesRefused, total)
	}
}

// TestExpertCacheBenchBytesReconcile pins the per-tier byte accounting against
// the ring's own counters in the ADMITTING regime (zero refusals): DRAM bytes +
// NVMe bytes == the trace's total bytes, and each tier's bytes are its event
// count times the per-event footprint. The refusal regime is a separate witness
// (TestExpertCacheBenchRefusalAccounting) because a refused event is served by
// neither tier.
func TestExpertCacheBenchBytesReconcile(t *testing.T) {
	shape := benchShape(t)
	const expertBytes = 64
	trace := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 17, Tokens: 12, HotSetSize: 16, ZipfExponent: 1.5, ExpertBytes: expertBytes,
	})
	// Budget for the whole touched working set: the trace is layer-major, so a
	// hit needs every layer's entry resident before a cross-token reuse can land.
	distinct := int64(shape.Layers * shape.TopK * trace.Tokens)
	res, err := ReplayExpertCacheTrace(trace, distinct*expertBytes)
	if err != nil {
		t.Fatalf("ReplayExpertCacheTrace: %v", err)
	}
	if res.HitCount == 0 {
		t.Fatalf("expected some hits at this budget; counters %+v", res)
	}
	total := int64(len(trace.Events)) * expertBytes
	if res.Refusals != 0 {
		t.Fatalf("this budget should admit every weight; got %d refusals", res.Refusals)
	}
	if res.BytesReadDRAM+res.BytesReadNVMe+res.BytesRefused != total {
		t.Fatalf("tier bytes %d+%d+%d != trace total %d", res.BytesReadDRAM, res.BytesReadNVMe, res.BytesRefused, total)
	}
	if res.BytesReadDRAM != res.HitCount*expertBytes {
		t.Fatalf("DRAM bytes %d != hits %d * %d", res.BytesReadDRAM, res.HitCount, expertBytes)
	}
	if res.BytesReadNVMe != res.MissCount*expertBytes {
		t.Fatalf("NVMe bytes %d != misses %d * %d (no refusals at this budget)", res.BytesReadNVMe, res.MissCount, expertBytes)
	}
	if res.PeakResidentBytes > res.RingByteCap {
		t.Fatalf("peak resident %d exceeds ring cap %d", res.PeakResidentBytes, res.RingByteCap)
	}
}

// TestExpertCacheBenchReceiptFieldLocked pins the L1 receipt contract against
// the shared schema: exact key set, simulated rung, and reconciling counters.
func TestExpertCacheBenchReceiptFieldLocked(t *testing.T) {
	shape := benchShape(t)
	trace := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 3, Tokens: 4, HotSetSize: 8, ZipfExponent: 1.25, ExpertBytes: 64,
	})
	res, err := ReplayExpertCacheTrace(trace, 64*int64(shape.TopK))
	if err != nil {
		t.Fatalf("ReplayExpertCacheTrace: %v", err)
	}
	receipt := BuildExpertCacheReceipt(shape, res)
	if receipt.Measurement != ExpertCacheMeasurementSimulated {
		t.Fatalf("measurement %q != simulated", receipt.Measurement)
	}
	if receipt.Schema != ExpertCacheReceiptSchema {
		t.Fatalf("schema %q != %q", receipt.Schema, ExpertCacheReceiptSchema)
	}
	if !receipt.HitRateKnown {
		t.Fatal("hit_rate_known false for a non-empty replay")
	}
	if receipt.HitCount+receipt.MissCount != int64(len(trace.Events)) {
		t.Fatalf("hit+miss %d != events %d", receipt.HitCount+receipt.MissCount, len(trace.Events))
	}
	if receipt.PrefillToksPerSec != 0 || receipt.DecodeToksPerSec != 0 || receipt.TTFTMillis != 0 {
		t.Fatalf("throughput fields must be unmeasured zero: %v/%v/%v",
			receipt.PrefillToksPerSec, receipt.DecodeToksPerSec, receipt.TTFTMillis)
	}

	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	want := map[string]bool{}
	for _, k := range ExpertCacheRequiredFields() {
		want[k] = true
	}
	if len(got) != len(want) {
		t.Fatalf("receipt has %d json keys, schema requires %d", len(got), len(want))
	}
	for k := range got {
		if !want[k] {
			t.Fatalf("receipt carries unexpected json key %q", k)
		}
	}
	for k := range want {
		if _, ok := got[k]; !ok {
			t.Fatalf("receipt missing required json key %q", k)
		}
	}
}

// TestExpertCacheBenchReceiptSameSeedIdentical pins that the seed flows through
// trace and receipt: the same seed reproduces an identical trace and identical
// marshal bytes, while a different seed draws a different trace (proving the
// seed is live) and a different receipt.
func TestExpertCacheBenchReceiptSameSeedIdentical(t *testing.T) {
	shape := benchShape(t)
	const expertBytes = 64
	// HotSetSize must be at least TopK for the skew to leave the PRNG any room:
	// at hotSet < TopK every layer deterministically exhausts the hot set and
	// spills to adjacent ids, so all seeds draw the SAME experts. 16 gives the
	// Zipf draw real freedom, which is what makes the seed observable.
	opts := ExpertCacheTraceOptions{
		Shape: shape, Seed: 1234, Tokens: 10, HotSetSize: 16, ZipfExponent: 1.5, ExpertBytes: expertBytes,
	}
	// A budget wide enough to hold several experts per layer lets hits actually
	// happen, so hit counts are sensitive to which experts each seed draws and
	// distinct traces cannot collapse to equal counters.
	ringCap := int64(expertBytes) * 240

	traceFor := func(seed int64) ExpertCacheTrace {
		o := opts
		o.Seed = seed
		return benchTrace(t, o)
	}
	receiptFor := func(tr ExpertCacheTrace) []byte {
		res, err := ReplayExpertCacheTrace(tr, ringCap)
		if err != nil {
			t.Fatalf("ReplayExpertCacheTrace: %v", err)
		}
		b, err := json.Marshal(BuildExpertCacheReceipt(shape, res))
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}
		return b
	}

	a := traceFor(opts.Seed)
	a2 := traceFor(opts.Seed)
	if !reflect.DeepEqual(a, a2) {
		t.Fatal("same seed produced different traces")
	}
	if !reflect.DeepEqual(receiptFor(a), receiptFor(a2)) {
		t.Fatal("same seed produced different receipts")
	}

	b := traceFor(opts.Seed + 1)
	if reflect.DeepEqual(a, b) {
		t.Fatal("different seed produced an identical trace; the seed is inert")
	}
	if reflect.DeepEqual(receiptFor(a), receiptFor(b)) {
		t.Fatal("different seed produced an identical receipt; the seed is inert")
	}
}

// TestExpertCacheBenchSkewRaisesHitRate pins that concentration helps: at a
// fixed moderate budget, a tighter hot set yields a hit rate at least as high
// as a flat distribution.
func TestExpertCacheBenchSkewRaisesHitRate(t *testing.T) {
	shape := benchShape(t)
	const expertBytes = 64
	ringCap := expertBytes * int64(shape.TopK) * 8

	skewed := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 5, Tokens: 30, HotSetSize: 8, ZipfExponent: 1.5, ExpertBytes: expertBytes,
	})
	flat := benchTrace(t, ExpertCacheTraceOptions{
		Shape: shape, Seed: 5, Tokens: 30, HotSetSize: shape.Experts, ZipfExponent: 0, ExpertBytes: expertBytes,
	})

	rate := func(trace ExpertCacheTrace) float64 {
		res, err := ReplayExpertCacheTrace(trace, ringCap)
		if err != nil {
			t.Fatalf("ReplayExpertCacheTrace: %v", err)
		}
		r, _ := ExpertCacheHitRate(res.HitCount, res.MissCount)
		return r
	}
	skewedRate, flatRate := rate(skewed), rate(flat)
	if skewedRate+tinyEpsilon < flatRate {
		t.Fatalf("skewed hot set hit rate %v < flat rate %v; concentration must not hurt", skewedRate, flatRate)
	}
}
