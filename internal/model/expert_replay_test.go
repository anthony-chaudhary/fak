package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestExpertReplayCapturesObservedQuantizedRoutes is the end-to-end spine for #4233:
// the real MoE router emits its selected experts, the recorder sizes the actual resident
// Q8 payloads, and the existing KV Belady oracle scores pagedRing's LRU policy.
func TestExpertReplayCapturesObservedQuantizedRoutes(t *testing.T) {
	cfg := moeCfgForTest()
	m := NewSyntheticMoE(cfg)
	m.Quantize()
	weightBytes := m.ExpertResidentWeightBytes(0, 0)
	if weightBytes <= 0 {
		t.Fatal("quantized synthetic expert has no resident Q8 footprint")
	}
	recorder := NewExpertTraceRecorder(m, "observed-router-fixture", weightBytes*2)
	m.SetExpertRouteObserver(recorder.Observer)
	if !m.ExpertRouteObserverSet() {
		t.Fatal("expert route observer was not installed")
	}
	for token := 0; token < 8; token++ {
		x := make([]float32, cfg.HiddenSize)
		for i := range x {
			x[i] = float32(((token+1)*(i+3))%17-8) / 8
		}
		for layer := 0; layer < cfg.NumLayers; layer++ {
			route(m, layer, x, f32Kernel{m})
		}
	}
	m.SetExpertRouteObserver(nil)

	trace := recorder.Snapshot()
	wantEvents := 8 * cfg.NumLayers * cfg.NumExpertsPerTok
	if len(trace.Events) != wantEvents || trace.UnsizedTouches != 0 {
		t.Fatalf("observed trace events=%d unsized=%d, want %d/0", len(trace.Events), trace.UnsizedTouches, wantEvents)
	}
	for i, event := range trace.Events {
		if event.WeightBytes != m.ExpertResidentWeightBytes(event.Layer, event.Expert) {
			t.Fatalf("event %d weight_bytes=%d, want model resident footprint", i, event.WeightBytes)
		}
	}
	report, err := ReplayExpertAccessTrace(trace)
	if err != nil {
		t.Fatalf("ReplayExpertAccessTrace: %v", err)
	}
	if report.PagedRingLRU.Policy != compute.KVEvictLRU {
		t.Fatalf("pagedRing policy=%v, want LRU", report.PagedRingLRU.Policy)
	}
	if !report.Oracle.Exact {
		t.Fatal("small observed fixture unexpectedly used the approximate Belady fallback")
	}
	// THE OBSERVED ANSWER on this deterministic routed fixture: at a two-expert
	// resident budget, pagedRing LRU recovers none of the hits the exact offline
	// oracle preserves (GoodDecisionRatio=0), even though observed recency has a
	// positive next-use correlation over the layer-sequential null. Temporal locality
	// exists here, but it is not sufficient evidence that plain LRU is a good policy.
	if report.PagedRingLRU.GoodDecisionRatio != 0 {
		t.Fatalf("observed LRU GoodDecisionRatio=%v, want the adjudicated fixture answer 0", report.PagedRingLRU.GoodDecisionRatio)
	}
	if report.Locality.RecencyNextUseCorrelation <= report.Locality.LayerSequentialNullCorrelation {
		t.Fatalf("observed locality correlation=%v, want > layer-sequential null %v",
			report.Locality.RecencyNextUseCorrelation, report.Locality.LayerSequentialNullCorrelation)
	}
	t.Logf("observed expert routing: events=%d resident_bytes/expert=%d LRU_good_decision_ratio=%.4f locality_corr=%.4f layer_null=%.4f",
		len(trace.Events), weightBytes, report.PagedRingLRU.GoodDecisionRatio,
		report.Locality.RecencyNextUseCorrelation, report.Locality.LayerSequentialNullCorrelation)
}

func TestExpertReplayExposesLRURegretAgainstBelady(t *testing.T) {
	const bytes = int64(144)
	trace := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: "lru-regret-witness", Source: "deterministic-unit",
		BudgetBytes: 2 * bytes,
		Events: []ExpertAccessTraceEvent{
			{Layer: 0, Expert: 0, WeightBytes: bytes},
			{Layer: 0, Expert: 1, WeightBytes: bytes},
			{Layer: 0, Expert: 2, WeightBytes: bytes},
			{Layer: 0, Expert: 0, WeightBytes: bytes},
			{Layer: 0, Expert: 1, WeightBytes: bytes},
			{Layer: 0, Expert: 2, WeightBytes: bytes},
		},
	}
	// Even a caller requesting only a comparison policy still receives the mandatory
	// pagedRing-LRU row: the issue's primary gauge cannot disappear by option choice.
	report, err := ReplayExpertAccessTrace(trace, compute.KVEvictCostAware)
	if err != nil {
		t.Fatalf("ReplayExpertAccessTrace: %v", err)
	}
	if report.Oracle.HitTokens != 2*int(bytes) {
		t.Fatalf("Belady hit bytes=%d, want %d", report.Oracle.HitTokens, 2*bytes)
	}
	if report.PagedRingLRU.HitTokens != 0 || report.PagedRingLRU.GoodDecisionRatio != 0 {
		t.Fatalf("LRU hits=%d ratio=%v, want 0/0", report.PagedRingLRU.HitTokens, report.PagedRingLRU.GoodDecisionRatio)
	}
	// Drive the actual pagedRing policy over the same composite expert identities and
	// byte budget. Its realized hit bytes must agree with the reused replay simulator;
	// this catches a future victim-order drift between polymodel.Pool and KVEvictLRU.
	ring := newPagedRing(nil, 2*bytes)
	x := compute.NewF32(compute.Default(), []int{1}, []float32{1})
	for _, event := range trace.Events {
		name := "layer-" + itoa(event.Layer) + "-expert-" + itoa(event.Expert)
		if got := ring.matMul(name, []int{1, 1}, []float32{1}, x, event.WeightBytes, false); len(got) != 1 || got[0] != 1 {
			t.Fatalf("pagedRing %s result=%v, want [1]", name, got)
		}
	}
	if got := ring.hit * int(bytes); got != report.PagedRingLRU.HitTokens {
		t.Fatalf("real pagedRing hit bytes=%d, replay LRU=%d", got, report.PagedRingLRU.HitTokens)
	}
	if ring.pageIn != len(trace.Events) || ring.evict != len(trace.Events)-2 {
		t.Fatalf("real pagedRing pageIn/evict=%d/%d, want %d/%d", ring.pageIn, ring.evict, len(trace.Events), len(trace.Events)-2)
	}
}

func TestGenerateExpertReplaySyntheticTraceDeterministic(t *testing.T) {
	opts := ExpertReplaySyntheticOptions{Seed: 4233, Tokens: 8, Layers: 2, Experts: 4, TopK: 2}
	first := GenerateExpertReplaySyntheticTrace(opts)
	second := GenerateExpertReplaySyntheticTrace(opts)
	if !reflect.DeepEqual(first, second) {
		t.Fatal("synthetic expert replay trace changed for an identical seed")
	}
	report, err := ReplayExpertAccessTrace(first)
	if err != nil {
		t.Fatalf("ReplayExpertAccessTrace(synthetic): %v", err)
	}
	if report.Locality.Samples == 0 || report.Locality.LayerSequentialNullSamples == 0 {
		t.Fatalf("locality samples observed/null=%d/%d, want both nonzero", report.Locality.Samples, report.Locality.LayerSequentialNullSamples)
	}
	for name, value := range map[string]float64{
		"observed": report.Locality.RecencyNextUseCorrelation,
		"null":     report.Locality.LayerSequentialNullCorrelation,
	} {
		if value < -1 || value > 1 {
			t.Fatalf("%s locality correlation=%v outside [-1,1]", name, value)
		}
	}
}

func TestExpertTraceRefusesF32ExpansionAndSizeDrift(t *testing.T) {
	m := NewSyntheticMoE(moeCfgForTest())
	recorder := NewExpertTraceRecorder(m, "f32-only", 1024)
	recorder.Observer(0, []int{0, 1})
	trace := recorder.Snapshot()
	if len(trace.Events) != 0 || trace.UnsizedTouches != 2 {
		t.Fatalf("f32-only trace events=%d unsized=%d, want 0/2", len(trace.Events), trace.UnsizedTouches)
	}
	if _, err := ReplayExpertAccessTrace(trace); err == nil {
		t.Fatal("replay accepted a trace with no quantized resident-byte events")
	}

	trace.Events = []ExpertAccessTraceEvent{
		{Layer: 0, Expert: 0, WeightBytes: 144},
		{Layer: 0, Expert: 0, WeightBytes: 288},
	}
	if _, err := ReplayExpertAccessTrace(trace); err == nil {
		t.Fatal("replay accepted changing resident bytes for one expert identity")
	}
}
