package fitplan

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// searchInput is the T04 golden fixture: an 8B-class dense model on a discrete
// 16 GiB pool, generic-empirical calibration, f32+q8 tiers, 4k prompt / 512
// output / target concurrency 2, no SLO, no pins, boot context 32768.
func searchInput(avail int64) SearchInput {
	return SearchInput{
		Model:           denseModel(),
		Weights:         denseWeights(),
		Hardware:        Hardware{Topology: TopologyDiscrete, PoolUsableBytes: avail, Peaks: testPeaks()},
		Calibration:     GenericEmpirical(),
		Tiers:           testTiers(),
		ResolvedContext: 32768,
		CurrentKV:       "f32",
		Workload:        SearchWorkload{PromptTokens: 4096, OutputTokens: 512, Concurrency: 2},
	}
}

// closedFormRate is the hand arithmetic for the per-seq decode rate at batch b
// of the golden fixture: min(bandwidth roofline, FLOP roofline) at the
// generic-empirical efficiencies.
func closedFormRate(b int, kvRead int64) float64 {
	const (
		bwEff = 0.55 * 256e9
		fEff  = 0.5 * 29.7e12
	)
	stream := float64(4 << 30) // rho = 1 exactly for the swept weights
	bytes := stream + float64(b)*float64(kvRead)
	flopsPerSeq := 2*7e9 + 4*(32*32*128)*4096
	rateBW := bwEff / bytes
	rateF := fEff / (float64(b) * flopsPerSeq)
	if rateBW < rateF {
		return rateBW
	}
	return rateF
}

// fak-test:runtime fast est=5ms lane=default
func TestFitAutoGoldenSearch(t *testing.T) {
	rep := Search(searchInput(16 * gib))
	if rep.Status != StatusFits || rep.Chosen == nil {
		t.Fatalf("status %s chosen %+v refusal %+v", rep.Status, rep.Chosen, rep.Refusal)
	}
	cscQ8 := testCSC(compute.KVPrecisionQ8)
	kvRead := attendedKVBytes(cscQ8, 4096)
	room := subSat(subSat(16*gib, 4*gib), testScratch())
	wantBMem := int(room / cscQ8.KVBytes(4096))
	if wantBMem < 9 || wantBMem > 16 {
		t.Fatalf("fixture drifted: bMem %d outside [9,16]", wantBMem)
	}
	ch := rep.Chosen
	if ch.KVPrecision != "q8" || ch.ContextTokens != 4096 || ch.Slots != 16 || ch.BStar != wantBMem {
		t.Fatalf("chosen %+v, want q8 ctx=4096 slots=16 B*=%d", ch, wantBMem)
	}
	if want := float64(wantBMem) * closedFormRate(wantBMem, kvRead); !relEq(ch.AggregateTokS, want) {
		t.Fatalf("aggregate %v, want %v (hand arithmetic)", ch.AggregateTokS, want)
	}
	if !(ch.TTFTSeconds > 0 && !math.IsInf(ch.TTFTSeconds, 0) && !math.IsNaN(ch.TTFTSeconds)) {
		t.Fatalf("ttft %v, want positive finite", ch.TTFTSeconds)
	}
	if len(rep.Alternatives) > maxAlternatives {
		t.Fatalf("%d alternatives, want <= %d", len(rep.Alternatives), maxAlternatives)
	}
	if len(rep.Alternatives) == 0 || rep.Alternatives[0].KVPrecision != "q8" || rep.Alternatives[0].ContextTokens != 4096 || rep.Alternatives[0].Slots != 8 {
		t.Fatalf("alternatives[0] %+v, want q8 ctx=4096 slots=8", rep.Alternatives)
	}
	// The current boot config is flagged when its axes are pinned: on a pool
	// where it fits one stream, the pinned search must return it flagged.
	in := searchInput(24 * gib)
	in.Workload.Concurrency = 1
	in.Pins = SearchPins{ContextTokens: 32768, KVPrecision: "f32"}
	rep2 := Search(in)
	if rep2.Status != StatusFits || rep2.Chosen == nil || !rep2.Chosen.Current {
		t.Fatalf("pinned current: status %s chosen %+v", rep2.Status, rep2.Chosen)
	}
	if rep2.Chosen.KVPrecision != "f32" || rep2.Chosen.ContextTokens != 32768 || rep2.Chosen.Slots != 1 {
		t.Fatalf("pinned current row %+v, want f32 ctx=32768 slots=1", rep2.Chosen)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFitAutoRefusalsAreTyped(t *testing.T) {
	t.Run("no_rankable_calibration_sol", func(t *testing.T) {
		in := searchInput(16 * gib)
		in.Calibration = sol()
		rep := Search(in)
		if rep.Status != StatusRefused || !errors.Is(error(rep.Refusal), ErrNoRankableCalibration) {
			t.Fatalf("status %s refusal %+v, want no_rankable_calibration", rep.Status, rep.Refusal)
		}
	})
	t.Run("no_rankable_calibration_drift", func(t *testing.T) {
		in := searchInput(16 * gib)
		in.Calibration = Calibration{Tier: TierCalibrated, State: RowDrift, BandwidthEfficiency: 0.55, ComputeEfficiency: 0.5}
		rep := Search(in)
		if rep.Status != StatusRefused || !errors.Is(error(rep.Refusal), ErrNoRankableCalibration) {
			t.Fatalf("status %s refusal %+v, want no_rankable_calibration", rep.Status, rep.Refusal)
		}
	})
	t.Run("needs_ring_moe_weights_overflow", func(t *testing.T) {
		in := SearchInput{
			Model:           moeModel(),
			Weights:         compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 262e9, Scope: compute.MemoryScopeDevice}},
			Hardware:        Hardware{Topology: TopologyDiscrete, PoolUsableBytes: 8 * gib, Peaks: testPeaks()},
			Calibration:     GenericEmpirical(),
			Tiers:           testTiers(),
			ResolvedContext: 4096,
			Workload:        SearchWorkload{PromptTokens: 4096, OutputTokens: 512, Concurrency: 1},
		}
		rep := Search(in)
		if rep.Status != StatusRefused || !errors.Is(error(rep.Refusal), ErrNeedsRing) {
			t.Fatalf("status %s refusal %+v, want needs_ring", rep.Status, rep.Refusal)
		}
		if rep.Refusal.ShortfallBytes != 262e9-8*gib {
			t.Fatalf("shortfall %d, want %d", rep.Refusal.ShortfallBytes, 262e9-8*gib)
		}
	})
	t.Run("fits_only_at_ctx_lt", func(t *testing.T) {
		avail := 4*gib + testScratch() + 1000*int64(testCSC(compute.KVPrecisionF32).KVBytesPerToken())
		in := searchInput(avail)
		in.Workload.PromptTokens = 4096
		rep := Search(in)
		if rep.Status != StatusRefused || !errors.Is(error(rep.Refusal), ErrFitsOnlyAtCtxLT) {
			t.Fatalf("status %s refusal %+v, want fits_only_at_ctx_lt", rep.Status, rep.Refusal)
		}
		// q8's denser KV stretches the same pool to 1920 tokens (f32 reaches 1000).
		if rep.Refusal.MaxContextTokens != 1920 || rep.Refusal.KVPrecision != "q8" || rep.Refusal.RequestedContext != 32768 {
			t.Fatalf("refusal %+v, want max ctx 1920 kv=q8 requested 32768", rep.Refusal)
		}
	})
	t.Run("below_slo_tpot", func(t *testing.T) {
		// 24 GiB keeps two streams at 32768 memory-eligible, so the SLO alone refuses.
		in := searchInput(24 * gib)
		in.SLO = SearchSLO{TPOTMS: 1.0} // needs 1000 tok/s per seq; the roofline tops out near 30
		rep := Search(in)
		if rep.Status != StatusRefused || !errors.Is(error(rep.Refusal), ErrBelowSLO) {
			t.Fatalf("status %s refusal %+v, want below_slo", rep.Status, rep.Refusal)
		}
		if rep.Refusal.SLOBound != "tpot" || rep.Refusal.SLOLimitMS != 1.0 {
			t.Fatalf("refusal %+v, want tpot bound limit 1ms", rep.Refusal)
		}
		wantPred := 1000 / closedFormRate(1, attendedKVBytes(testCSC(compute.KVPrecisionQ8), 4096))
		if !relEq(rep.Refusal.PredictedMS, wantPred) {
			t.Fatalf("predicted %v, want %v", rep.Refusal.PredictedMS, wantPred)
		}
	})
	t.Run("pin_infeasible_absent_kv", func(t *testing.T) {
		in := searchInput(16 * gib)
		in.Pins = SearchPins{KVPrecision: "bf16"}
		rep := Search(in)
		if rep.Status != StatusRefused || !errors.Is(error(rep.Refusal), ErrPinInfeasible) {
			t.Fatalf("status %s refusal %+v, want pin_infeasible", rep.Status, rep.Refusal)
		}
		if rep.Refusal.Pin != "kv_precision=bf16" {
			t.Fatalf("pin %q, want kv_precision=bf16", rep.Refusal.Pin)
		}
	})
	t.Run("pin_infeasible_oversized_ctx", func(t *testing.T) {
		in := searchInput(16 * gib)
		in.Pins = SearchPins{ContextTokens: 131072}
		rep := Search(in)
		if rep.Status != StatusRefused || !errors.Is(error(rep.Refusal), ErrPinInfeasible) {
			t.Fatalf("status %s refusal %+v, want pin_infeasible", rep.Status, rep.Refusal)
		}
		if !strings.Contains(rep.Refusal.Pin, "context_tokens=131072") || rep.Refusal.ShortfallBytes <= 0 {
			t.Fatalf("refusal %+v, want pinned ctx named with a shortfall", rep.Refusal)
		}
	})
}

// fak-test:runtime fast est=5ms lane=default
func TestFitAutoPinnedAxesNeverChange(t *testing.T) {
	honors := func(r SearchRow, p SearchPins) bool {
		if p.ContextTokens > 0 && r.ContextTokens != p.ContextTokens {
			return false
		}
		if s := strings.TrimSpace(p.KVPrecision); s != "" && !strings.EqualFold(r.KVPrecision, s) {
			return false
		}
		if p.Slots > 0 && r.Slots != p.Slots {
			return false
		}
		return true
	}
	avails := []int64{8 * gib, 16 * gib, 32 * gib, 64 * gib}
	ctxs := []int{0, 4096, 32768, 131072}
	kvs := []string{"", "f32", "q8"}
	slots := []int{0, 2, 8}
	for seed := int64(1); seed <= 64; seed++ {
		rng := rand.New(rand.NewSource(seed))
		in := searchInput(avails[rng.Intn(len(avails))])
		in.Pins = SearchPins{
			ContextTokens: ctxs[rng.Intn(len(ctxs))],
			KVPrecision:   kvs[rng.Intn(len(kvs))],
			Slots:         slots[rng.Intn(len(slots))],
		}
		rep := Search(in)
		if rep.Chosen != nil && !honors(*rep.Chosen, in.Pins) {
			t.Fatalf("seed %d: chosen %+v breaks pins %+v", seed, *rep.Chosen, in.Pins)
		}
		for _, a := range rep.Alternatives {
			if !honors(a, in.Pins) {
				t.Fatalf("seed %d: alternative %+v breaks pins %+v", seed, a, in.Pins)
			}
		}
		if anyPinSet(in.Pins) && rep.Status == StatusRefused && rep.Refusal != nil &&
			rep.Refusal.Reason == ReasonPinInfeasible && rep.Refusal.Pin == "" {
			t.Fatalf("seed %d: pin_infeasible without the pin named", seed)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFitAutoMaxSlotContextMatchesBruteForce(t *testing.T) {
	p := newPlanner(Input{
		Model:       denseModel(),
		Weights:     denseWeights(),
		Hardware:    Hardware{Topology: TopologyDiscrete, PoolUsableBytes: 12 * gib, Peaks: testPeaks()},
		Workload:    DefaultWorkload(),
		Calibration: GenericEmpirical(),
	})
	brute := func(c Candidate, n int) int {
		room := p.room(c)
		for i := c.Context.MaxContext; i >= 1; i-- {
			if mulSat(int64(n), perStreamBytes(c, i)) <= room {
				return i
			}
		}
		return 0
	}
	for _, prec := range []compute.KVPrecision{compute.KVPrecisionF32, compute.KVPrecisionQ8} {
		c := cand(prec, 0, false)
		for _, n := range []int{1, 2, 3, 5, 8} {
			if got, want := p.maxSlotContext(c, n), brute(c, n); got != want {
				t.Fatalf("%s n=%d: binary %d, brute %d", prec.String(), n, got, want)
			}
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFitAutoTokenBudgetClamp(t *testing.T) {
	cases := []struct {
		batch, isl, osl, want int
	}{
		{1, 4096, 512, 2048},     // 9 -> floor clamp
		{1, 2048, 512, 2048},     // 5 -> floor clamp
		{300, 4096, 512, 2700},   // in-band
		{4096, 4096, 512, 32768}, // ceiling clamp
		{1000, 8192, 128, 32768}, // ceiling clamp
		{10, 100, 100, 2048},     // 20 -> floor clamp
		{0, 4096, 512, 2048},     // batch 0 reads as 1
	}
	for _, tc := range cases {
		if got := TokenBudget(tc.batch, tc.isl, tc.osl); got != tc.want {
			t.Fatalf("TokenBudget(%d,%d,%d) = %d, want %d", tc.batch, tc.isl, tc.osl, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFitAutoChunkedTTFTMatchesUnchunkedOnShortPrompts(t *testing.T) {
	// With ISL <= 2048 the TRT budget (>= 2048) covers the whole span in one
	// chunk, so the chunked sum must equal Plan's single prefill stage.
	isl := DefaultPromptTokens // 2048
	c := cand(compute.KVPrecisionQ8, 32768, true)
	in := baseInput(16*gib, c)
	in.Workload = Workload{PromptTokens: isl, Concurrency: 1}
	in.Calibration = GenericEmpirical()
	rep := plan(t, in)
	pr := findPred(t, rep, "q8", 32768)
	p := newPlanner(in)
	if got, want := p.chunkedTTFT(c, isl, 1, 512), pr.TTFTSeconds; !relEq(got, want) {
		t.Fatalf("chunked ttft %v, unchunked %v", got, want)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFitAutoSearchIsDeterministic(t *testing.T) {
	a, b := Search(searchInput(16*gib)), Search(searchInput(16*gib))
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("search not deterministic:\n%+v\n%+v", a, b)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFitAutoReportRoundTripsJSON(t *testing.T) {
	rep := Search(searchInput(16 * gib))
	raw, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	var back SearchReport
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, rep) {
		t.Fatalf("round-trip mismatch:\n%+v\n%+v", back, rep)
	}
}
