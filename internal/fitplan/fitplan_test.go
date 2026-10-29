package fitplan

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/compute/strix"
)

const gib = int64(1) << 30

func denseModel() Model {
	return Model{
		PayloadWeightBytes:   4 << 30,
		StreamBytesPerToken:  4 << 30,
		ActiveParamsPerToken: 7e9,
		AttnLayers:           32,
		NumHeads:             32,
		HeadDim:              128,
		MaxContextTokens:     131072,
	}
}

func denseWeights() compute.MemoryPlan {
	return compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 4 << 30, Scope: compute.MemoryScopeDevice}}
}

// testCSC is an 8B-class GQA geometry: f32 KV = 393216 B/token, q8 KV = 204800 B/token.
func testCSC(prec compute.KVPrecision) compute.ContextSizeConfig {
	return compute.ContextSizeConfig{
		KV: compute.KVConfig{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, Precision: prec},
		Scratch: compute.TransformerScratchConfig{
			HiddenSize: 4096, IntermediateSize: 14336, VocabSize: 128256, NumLayers: 32,
			NumHeads: 32, NumKVHeads: 8, HeadDim: 128, IncludeLogits: true,
		},
		MaxContext: 131072,
	}
}

func testPeaks() Peaks { return Peaks{BytesPerSec: 256e9, FLOPsPerSec: 29.7e12, Source: "test"} }

// sol is the speed-of-light calibration: the roofline at full peaks.
func sol() Calibration {
	return Calibration{Tier: TierSOL, BandwidthEfficiency: 1, ComputeEfficiency: 1, Source: "test: roofline at full peaks"}
}

func refusalErr(rep Report) error {
	if rep.Refusal == nil {
		return nil
	}
	return rep.Refusal
}

func testScratch() int64 {
	return compute.EstimateHALTransientMemoryPlan(testCSC(compute.KVPrecisionF32).Scratch).Total()
}

func cand(prec compute.KVPrecision, ctx int, current bool) Candidate {
	return Candidate{KVPrecision: prec.String(), Context: testCSC(prec), ContextTokens: ctx, Current: current}
}

func testTiers() []KVTier {
	return []KVTier{
		{Precision: compute.KVPrecisionF32.String(), Context: testCSC(compute.KVPrecisionF32)},
		{Precision: compute.KVPrecisionQ8.String(), Context: testCSC(compute.KVPrecisionQ8)},
	}
}

func baseInput(avail int64, cands ...Candidate) Input {
	return Input{
		Model:       denseModel(),
		Weights:     denseWeights(),
		Hardware:    Hardware{Topology: TopologyDiscrete, PoolUsableBytes: avail, Peaks: testPeaks()},
		Workload:    DefaultWorkload(),
		Calibration: sol(),
		Candidates:  cands,
	}
}

func plan(t *testing.T, in Input) Report {
	t.Helper()
	rep := Plan(in)
	assertFinite(t, rep)
	return rep
}

func assertFinite(t *testing.T, rep Report) {
	t.Helper()
	preds := append([]Prediction(nil), rep.Candidates...)
	if rep.Chosen != nil {
		preds = append(preds, *rep.Chosen)
	}
	for _, pr := range preds {
		for _, f := range []float64{pr.DecodeTokSPerSeq, pr.DecodeTokSAggregate, pr.DecodeTokSCeiling, pr.PrefillTokS, pr.TTFTSeconds} {
			if math.IsNaN(f) || math.IsInf(f, 0) || f < 0 {
				t.Fatalf("%s: non-finite or negative throughput %v", pr.Name, f)
			}
		}
	}
	for _, f := range []float64{rep.Evidence.ResidentScale, rep.Evidence.Peaks.BytesPerSec, rep.Evidence.Peaks.FLOPsPerSec,
		rep.Evidence.Calibration.BandwidthEfficiency, rep.Evidence.Calibration.ComputeEfficiency} {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("non-finite evidence %v", f)
		}
	}
}

func relEq(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9*math.Max(math.Abs(a), math.Abs(b))
}

func hasOmission(list []Omission, o Omission) bool {
	for _, x := range list {
		if x == o {
			return true
		}
	}
	return false
}

func findPred(t *testing.T, rep Report, prec string, ctx int) Prediction {
	t.Helper()
	for _, pr := range rep.Candidates {
		if pr.KVPrecision == prec && pr.ContextTokens == ctx {
			return pr
		}
	}
	t.Fatalf("no candidate kv=%s ctx=%d", prec, ctx)
	return Prediction{}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnifiedPoolChargesHostScopedWeightsOnce(t *testing.T) {
	weights := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Bytes: 10 * gib, Scope: compute.MemoryScopeDevice},
		{Class: compute.MemoryOffload, Bytes: 20 * gib, Scope: compute.MemoryScopeHost},
	}
	perStream := int64(4096) * 393216
	for _, tc := range []struct {
		topo    Topology
		charged int64
	}{
		{TopologyUnified, 30 * gib},
		{TopologyDiscrete, 10 * gib},
	} {
		in := baseInput(40*gib, cand(compute.KVPrecisionF32, 4096, true))
		in.Weights = weights
		in.Model.PayloadWeightBytes = 30 * gib
		in.Hardware.Topology = tc.topo
		rep := plan(t, in)
		want := int((40*gib - tc.charged - testScratch()) / perStream)
		if got := rep.Candidates[0].MaxConcurrency; got != want || want < 1 {
			t.Fatalf("%s: MaxConcurrency = %d, want %d", tc.topo, got, want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestMaxStreamContextMatchesLargestFittingContext(t *testing.T) {
	avail := 24 * gib
	csc := testCSC(compute.KVPrecisionF32)
	l := compute.LargestFittingContextTokens(csc, denseWeights(), avail)
	if !(l > compute.MinAutoContextTokens && l < 131072) {
		t.Fatalf("precondition: largest fitting context %d not strictly inside the clamp", l)
	}
	rep := plan(t, baseInput(avail, cand(compute.KVPrecisionF32, l, true), cand(compute.KVPrecisionF32, l+1, false)))
	if rep.MaxContextTokens != l {
		t.Fatalf("MaxContextTokens = %d, want %d", rep.MaxContextTokens, l)
	}
	if rep.Candidates[0].MaxConcurrency < 1 || rep.Candidates[1].MaxConcurrency != 0 {
		t.Fatalf("ctx %d conc %d, ctx %d conc %d", l, rep.Candidates[0].MaxConcurrency, l+1, rep.Candidates[1].MaxConcurrency)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestMaxStreamContextBelowFloorIsExact(t *testing.T) {
	avail := 4*gib + testScratch() + 300*393216
	if l := compute.LargestFittingContextTokens(testCSC(compute.KVPrecisionF32), denseWeights(), avail); l != compute.MinAutoContextTokens {
		t.Fatalf("precondition: sizer = %d, want the %d floor clamp", l, compute.MinAutoContextTokens)
	}
	rep := plan(t, baseInput(avail, cand(compute.KVPrecisionF32, 4096, true)))
	if rep.MaxContextTokens != 300 {
		t.Fatalf("MaxContextTokens = %d, want 300", rep.MaxContextTokens)
	}
	if rep.Refusal == nil || rep.Refusal.Reason != ReasonContextExceedsPool || rep.Refusal.MaxContextTokens != 300 {
		t.Fatalf("refusal = %+v, want context_exceeds_pool at 300", rep.Refusal)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRefusalWeightsExceedPool(t *testing.T) {
	tiers := testTiers()[:1]
	rep := plan(t, baseInput(2*gib, BuildCandidates(tiers, 4096)...))
	if rep.Status != StatusRefused {
		t.Fatalf("status = %s, want refused", rep.Status)
	}
	err := refusalErr(rep)
	if !errors.Is(err, ErrNoFit) || !errors.Is(err, ErrWeightsExceedPool) || errors.Is(err, ErrContextExceedsPool) {
		t.Fatalf("err = %v, want ErrNoFit and ErrWeightsExceedPool only", err)
	}
	want := 4*gib + testScratch() + int64(4096)*393216 - 2*gib
	if rep.Refusal.ShortfallBytes != want {
		t.Fatalf("ShortfallBytes = %d, want %d", rep.Refusal.ShortfallBytes, want)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRefusalFitsOnlyBelowContext(t *testing.T) {
	avail := 4*gib + testScratch() + 1000*393216
	rep := plan(t, baseInput(avail, cand(compute.KVPrecisionF32, 4096, true), cand(compute.KVPrecisionF32, 32768, false)))
	if !errors.Is(refusalErr(rep), ErrContextExceedsPool) {
		t.Fatalf("err = %v, want ErrContextExceedsPool", refusalErr(rep))
	}
	if rep.Refusal.MaxContextTokens != 1000 || rep.MaxContextTokens != 1000 {
		t.Fatalf("refusal max ctx %d, report max ctx %d, want 1000", rep.Refusal.MaxContextTokens, rep.MaxContextTokens)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRefusalConcurrencyExceedsPool(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Workload.Concurrency = 8
	rep := plan(t, in)
	if !errors.Is(refusalErr(rep), ErrConcurrencyExceedsPool) {
		t.Fatalf("err = %v, want ErrConcurrencyExceedsPool", refusalErr(rep))
	}
	if rep.Refusal.MaxConcurrency != 1 {
		t.Fatalf("MaxConcurrency = %d, want 1", rep.Refusal.MaxConcurrency)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestChoosesCurrentWhenItFits(t *testing.T) {
	rep := plan(t, baseInput(24*gib, cand(compute.KVPrecisionF32, 4096, false), cand(compute.KVPrecisionF32, 32768, true)))
	if rep.Status != StatusFits || refusalErr(rep) != nil {
		t.Fatalf("status %s err %v, want fits", rep.Status, refusalErr(rep))
	}
	if rep.Chosen == nil || !rep.Chosen.Current || rep.Chosen.ContextTokens != 32768 {
		t.Fatalf("chosen = %+v, want the current 32768 candidate", rep.Chosen)
	}
}

func shrinkInput() Input {
	return baseInput(4*gib+testScratch()+8*gib, BuildCandidates(testTiers(), 32768)...)
}

// fak-test:runtime fast est=1ms lane=default
func TestChoosesLargestFittingContextWhenCurrentRefused(t *testing.T) {
	rep := plan(t, shrinkInput())
	if rep.Status != StatusFits || rep.Chosen == nil {
		t.Fatalf("status %s chosen %+v, want fits", rep.Status, rep.Chosen)
	}
	if rep.Chosen.KVPrecision != "q8" || rep.Chosen.ContextTokens != 32768 || rep.Chosen.Current {
		t.Fatalf("chosen = %+v, want non-current q8@32768", rep.Chosen)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPinnedKVPrecisionIsNeverOverridden(t *testing.T) {
	in := shrinkInput()
	in.Pins.KVPrecision = "f32"
	rep := plan(t, in)
	if rep.Status != StatusFits || rep.Chosen == nil || rep.Chosen.KVPrecision != "f32" || rep.Chosen.ContextTokens != 4096 {
		t.Fatalf("status %s chosen %+v, want f32@4096", rep.Status, rep.Chosen)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPinnedContextRefusesInsteadOfShrinking(t *testing.T) {
	in := shrinkInput()
	in.Pins = Pins{KVPrecision: "f32", ContextTokens: 32768}
	rep := plan(t, in)
	if rep.Status != StatusRefused || !errors.Is(refusalErr(rep), ErrContextExceedsPool) {
		t.Fatalf("status %s err %v, want refused context_exceeds_pool", rep.Status, refusalErr(rep))
	}
	if want := int(8 * gib / 393216); rep.Refusal.MaxContextTokens != want || want != 21845 {
		t.Fatalf("refusal max ctx = %d, want %d", rep.Refusal.MaxContextTokens, want)
	}

	in.Pins = Pins{ContextTokens: 32768}
	rep = plan(t, in)
	if rep.Chosen == nil || rep.Chosen.KVPrecision != "q8" || rep.Chosen.ContextTokens != 32768 {
		t.Fatalf("chosen = %+v, want q8@32768", rep.Chosen)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnknownPoolFailsOpen(t *testing.T) {
	rep := plan(t, baseInput(0, BuildCandidates(testTiers(), 4096)...))
	if rep.Status != StatusUnknown || refusalErr(rep) != nil {
		t.Fatalf("status %s err %v, want unknown and nil", rep.Status, refusalErr(rep))
	}
	for _, pr := range rep.Candidates {
		if pr.Fit != compute.FitUnknown.String() || !hasOmission(pr.Omitted, OmitPoolCapacityUnknown) {
			t.Fatalf("%s: fit %s omitted %v", pr.Name, pr.Fit, pr.Omitted)
		}
	}
	if rep.Chosen == nil || !rep.Chosen.Current || !(rep.Chosen.DecodeTokSPerSeq > 0) || rep.MaxContextTokens != 0 {
		t.Fatalf("chosen %+v max ctx %d", rep.Chosen, rep.MaxContextTokens)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRefusalSentinelsAreClosed(t *testing.T) {
	all := map[RefusalReason]error{
		ReasonWeightsExceedPool:      ErrWeightsExceedPool,
		ReasonContextExceedsPool:     ErrContextExceedsPool,
		ReasonConcurrencyExceedsPool: ErrConcurrencyExceedsPool,
		ReasonPinUnmatched:           ErrPinUnmatched,
		ReasonFitsOnlyAtCtxLT:        ErrFitsOnlyAtCtxLT,
		ReasonNeedsRing:              ErrNeedsRing,
		ReasonBelowSLO:               ErrBelowSLO,
		ReasonPinInfeasible:          ErrPinInfeasible,
		ReasonNoRankableCalibration:  ErrNoRankableCalibration,
	}
	for r, own := range all {
		if r.Sentinel() != own {
			t.Fatalf("%s.Sentinel() mismatch", r)
		}
		err := error(&Refusal{Reason: r})
		if !errors.Is(err, ErrNoFit) || !errors.Is(err, own) {
			t.Fatalf("%s: errors.Is missed ErrNoFit or its sentinel", r)
		}
		for other, sentinel := range all {
			if other != r && errors.Is(err, sentinel) {
				t.Fatalf("%s also matches %s", r, other)
			}
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnavailableReport(t *testing.T) {
	rep := Unavailable(OmitModelFactsUnavailable)
	if rep.Status != StatusUnavailable || refusalErr(rep) != nil {
		t.Fatalf("status %s err %v", rep.Status, refusalErr(rep))
	}
	if rep.Candidates == nil || len(rep.Candidates) != 0 {
		t.Fatalf("candidates = %#v, want a non-nil empty slice", rep.Candidates)
	}
	if !reflect.DeepEqual(rep.Omitted, []Omission{OmitModelFactsUnavailable}) {
		t.Fatalf("omitted = %v", rep.Omitted)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPlanDeterministic(t *testing.T) {
	in := shrinkInput()
	if a, b := plan(t, in), plan(t, in); !reflect.DeepEqual(a, b) {
		t.Fatalf("Plan is not deterministic:\n%+v\n%+v", a, b)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRefusalPoolShortOfScratchIsContextNotWeights(t *testing.T) {
	rep := plan(t, baseInput(4*gib+testScratch()/2, cand(compute.KVPrecisionF32, 4096, true)))
	err := refusalErr(rep)
	if !errors.Is(err, ErrContextExceedsPool) || errors.Is(err, ErrWeightsExceedPool) {
		t.Fatalf("err = %v, want context_exceeds_pool only", err)
	}
	if rep.Refusal.MaxContextTokens != 0 || rep.MaxContextTokens != 0 {
		t.Fatalf("refusal max ctx %d, report max ctx %d, want 0", rep.Refusal.MaxContextTokens, rep.MaxContextTokens)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPinMatchingNoCandidateIsPinUnmatched(t *testing.T) {
	for _, pins := range []Pins{{ContextTokens: 5000}, {KVPrecision: "bf16"}} {
		for _, avail := range []int64{400 * gib, 0} {
			in := baseInput(avail, BuildCandidates(testTiers(), 4096)...)
			in.Pins = pins
			rep := plan(t, in)
			err := refusalErr(rep)
			if rep.Status != StatusRefused || !errors.Is(err, ErrPinUnmatched) || errors.Is(err, ErrContextExceedsPool) || rep.Chosen != nil {
				t.Fatalf("pins %+v avail %d: status %s err %v chosen %+v", pins, avail, rep.Status, err, rep.Chosen)
			}
		}
	}
	in := baseInput(400*gib, BuildCandidates(testTiers(), 4096, 5000)...)
	in.Pins = Pins{ContextTokens: 5000}
	rep := plan(t, in)
	if rep.Status != StatusFits || rep.Chosen == nil || rep.Chosen.ContextTokens != 5000 {
		t.Fatalf("status %s chosen %+v, want the laddered 5000 pin", rep.Status, rep.Chosen)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnknownPoolHonorsPins(t *testing.T) {
	in := baseInput(0, BuildCandidates(testTiers(), 4096)...)
	in.Pins.KVPrecision = "q8"
	rep := plan(t, in)
	if rep.Status != StatusUnknown || rep.Refusal != nil || rep.Chosen == nil || rep.Chosen.KVPrecision != "q8" {
		t.Fatalf("status %s refusal %+v chosen %+v, want unknown with a q8 choice", rep.Status, rep.Refusal, rep.Chosen)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestKVGeometryUnknownOmitsMaxContext(t *testing.T) {
	c := cand(compute.KVPrecisionF32, 4096, true)
	c.Context.KV.NumKVHeads = 0
	rep := plan(t, baseInput(24*gib, c))
	if rep.Status != StatusUnknown || rep.MaxContextTokens != 0 || !hasOmission(rep.Omitted, OmitKVGeometryUnknown) {
		t.Fatalf("status %s max ctx %d omitted %v", rep.Status, rep.MaxContextTokens, rep.Omitted)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestWindowedKVMaxContextCoversEveryFittingCandidate(t *testing.T) {
	csc := compute.ContextSizeConfig{
		KV:         compute.KVConfig{NumLayers: 6, NumKVHeads: 8, HeadDim: 128, WindowPerLayer: []int{1024, 1024, 1024, 1024, 1024, 0}},
		MaxContext: 131072,
	}
	in := baseInput(gib+csc.KVBytes(131072)+(1<<20), BuildCandidates([]KVTier{{Precision: "f32", Context: csc}}, 4096)...)
	in.Weights = compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: gib, Scope: compute.MemoryScopeDevice}}
	in.Model.PayloadWeightBytes = gib
	rep := plan(t, in)
	if rep.MaxContextTokens != 131072 {
		t.Fatalf("MaxContextTokens = %d, want the full 131072 window", rep.MaxContextTokens)
	}
	for _, pr := range rep.Candidates {
		if pr.Fit == compute.FitOK.String() && pr.ContextTokens > rep.MaxContextTokens {
			t.Fatalf("%s fits above MaxContextTokens %d", pr.Name, rep.MaxContextTokens)
		}
	}
}

func discreteOffloadInput(hostAvail int64) Input {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 4096, true))
	in.Weights = compute.MemoryPlan{
		{Class: compute.MemoryWeights, Bytes: 2 * gib, Scope: compute.MemoryScopeDevice},
		{Class: compute.MemoryOffload, Bytes: 64 * gib, Scope: compute.MemoryScopeHost},
	}
	in.Model.PayloadWeightBytes = 66 * gib
	in.Hardware.HostPoolUsableBytes = hostAvail
	return in
}

// fak-test:runtime fast est=1ms lane=default
func TestDiscreteHostScopedWeightsAreChargedToHostPool(t *testing.T) {
	short := plan(t, discreteOffloadInput(20*gib))
	err := refusalErr(short)
	if short.Status != StatusRefused || !errors.Is(err, ErrWeightsExceedPool) || !short.Refusal.HostPool {
		t.Fatalf("status %s refusal %+v, want a host weights_exceed_pool", short.Status, short.Refusal)
	}
	if short.Refusal.ShortfallBytes != 44*gib || short.Evidence.HostPoolUsableBytes != 20*gib || short.MaxContextTokens != 0 {
		t.Fatalf("refusal %+v evidence %+v max ctx %d", short.Refusal, short.Evidence, short.MaxContextTokens)
	}
	if pr := short.Candidates[0]; pr.Fit != compute.FitTooBig.String() || pr.MaxConcurrency != 0 {
		t.Fatalf("candidate %+v, want too_big with no stream", pr)
	}

	unknown := plan(t, discreteOffloadInput(0))
	if unknown.Status != StatusUnknown || unknown.Refusal != nil || unknown.Chosen == nil || !hasOmission(unknown.Omitted, OmitHostPoolCapacityUnknown) {
		t.Fatalf("status %s refusal %+v chosen %+v omitted %v", unknown.Status, unknown.Refusal, unknown.Chosen, unknown.Omitted)
	}

	if roomy := plan(t, discreteOffloadInput(128*gib)); roomy.Status != StatusFits || hasOmission(roomy.Omitted, OmitHostPoolCapacityUnknown) {
		t.Fatalf("status %s omitted %v, want fits", roomy.Status, roomy.Omitted)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnknownConcurrencyDropsItsKey(t *testing.T) {
	geomless := cand(compute.KVPrecisionF32, 4096, true)
	geomless.Context.KV.NumKVHeads = 0
	for _, in := range []Input{baseInput(0, cand(compute.KVPrecisionF32, 4096, true)), baseInput(24*gib, geomless)} {
		pr := plan(t, in).Candidates[0]
		if pr.Fit != compute.FitUnknown.String() {
			t.Fatalf("precondition: fit %s, want unknown", pr.Fit)
		}
		raw, err := json.Marshal(pr)
		if err != nil {
			t.Fatal(err)
		}
		var keys map[string]json.RawMessage
		if err := json.Unmarshal(raw, &keys); err != nil {
			t.Fatal(err)
		}
		if _, ok := keys["max_concurrency"]; ok {
			t.Fatalf("max_concurrency emitted for unknown fit: %s", raw)
		}
	}
}

func moeModel() Model {
	return Model{
		PayloadWeightBytes:        262e9,
		StreamBytesPerToken:       10e9,
		ActiveParamsPerToken:      20e9,
		RoutedExpertBytes:         256e9,
		ActiveRoutedBytesPerToken: 4e9,
		NumExperts:                256,
		ExpertsUsed:               4,
		AttnLayers:                32,
		NumHeads:                  32,
		HeadDim:                   128,
		MaxContextTokens:          131072,
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeMemoryBoundIsBandwidthOverStepBytes(t *testing.T) {
	rep := plan(t, baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true), cand(compute.KVPrecisionQ8, 32768, false)))
	f32KV := testCSC(compute.KVPrecisionF32).KVBytes(2048) * 2 / 3
	pr := findPred(t, rep, "f32", 32768)
	if pr.DecodeBytesPerStep != 4<<30+f32KV {
		t.Fatalf("DecodeBytesPerStep = %d, want %d", pr.DecodeBytesPerStep, 4<<30+f32KV)
	}
	want := 256e9 / float64(pr.DecodeBytesPerStep)
	if !relEq(pr.DecodeTokSPerSeq, want) || !relEq(pr.DecodeTokSCeiling, want) {
		t.Fatalf("per-seq %v ceiling %v, want %v", pr.DecodeTokSPerSeq, pr.DecodeTokSCeiling, want)
	}
	if pr.Bound != BoundMemory {
		t.Fatalf("bound = %s, want memory", pr.Bound)
	}
	q8 := findPred(t, rep, "q8", 32768)
	if kv := q8.DecodeBytesPerStep - 4<<30; !(kv > 0 && kv < f32KV) {
		t.Fatalf("q8 attended KV %d not in (0, %d)", kv, f32KV)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeComputeBoundWhenFLOPsBind(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Hardware.Peaks.FLOPsPerSec = 1e9
	pr := plan(t, in).Candidates[0]
	if pr.Bound != BoundCompute {
		t.Fatalf("bound = %s, want compute", pr.Bound)
	}
	if want := 1e9 / (2*7e9 + 4*32*32*128*2048); !relEq(pr.DecodeTokSPerSeq, want) {
		t.Fatalf("per-seq %v, want %v", pr.DecodeTokSPerSeq, want)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestEmpiricalTierScalesMemoryBoundDecode(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Calibration = GenericEmpirical()
	rep := plan(t, in)
	pr := rep.Candidates[0]
	if !relEq(pr.DecodeTokSPerSeq/pr.DecodeTokSCeiling, 0.55) || rep.Evidence.Calibration.Tier != TierEmpirical {
		t.Fatalf("per-seq/ceiling = %v tier %s", pr.DecodeTokSPerSeq/pr.DecodeTokSCeiling, rep.Evidence.Calibration.Tier)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPrefillTTFTRoofline(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Calibration = GenericEmpirical()
	pr := plan(t, in).Candidates[0]
	flopsPre := 2*7e9*2048 + 2*32*32*128*2048.0*2048
	bytesPre := float64(4<<30 + testCSC(compute.KVPrecisionF32).KVBytes(2048))
	want := math.Max(flopsPre/(0.5*29.7e12), bytesPre/(0.55*256e9))
	if !relEq(pr.TTFTSeconds, want) || !relEq(pr.PrefillTokS*pr.TTFTSeconds, 2048) {
		t.Fatalf("ttft %v prefill %v, want ttft %v", pr.TTFTSeconds, pr.PrefillTokS, want)
	}
	in.Workload.PromptTokens = 4096
	if longer := plan(t, in).Candidates[0]; !(longer.TTFTSeconds > pr.TTFTSeconds) {
		t.Fatalf("ttft at 4096 (%v) not above ttft at 2048 (%v)", longer.TTFTSeconds, pr.TTFTSeconds)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnknownPeakOmitsThroughputWithReason(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Hardware.Peaks = Peaks{}
	pr := plan(t, in).Candidates[0]
	if pr.DecodeTokSPerSeq != 0 || pr.DecodeTokSAggregate != 0 || pr.DecodeTokSCeiling != 0 || pr.PrefillTokS != 0 || pr.TTFTSeconds != 0 {
		t.Fatalf("throughput stated without a bandwidth peak: %+v", pr)
	}
	if !hasOmission(pr.Omitted, OmitPeakBandwidthUnknown) || pr.Fit != compute.FitOK.String() {
		t.Fatalf("omitted %v fit %s", pr.Omitted, pr.Fit)
	}
	raw, err := json.Marshal(pr)
	if err != nil {
		t.Fatal(err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"decode_tok_s_per_seq", "ttft_s"} {
		if _, ok := keys[k]; ok {
			t.Fatalf("key %q emitted for an omitted number", k)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnknownComputePeakStatesDecodeOnly(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Hardware.Peaks.FLOPsPerSec = 0
	pr := plan(t, in).Candidates[0]
	if !(pr.DecodeTokSPerSeq > 0) || pr.Bound != BoundMemory || pr.TTFTSeconds != 0 || !hasOmission(pr.Omitted, OmitPeakComputeUnknown) {
		t.Fatalf("prediction = %+v", pr)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpilledWeightsOnUnifiedCostPoolBandwidth(t *testing.T) {
	run := func(w compute.MemoryPlan) Prediction {
		in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
		in.Hardware.Topology = TopologyUnified
		in.Weights = w
		return plan(t, in).Candidates[0]
	}
	split := run(compute.MemoryPlan{
		{Class: compute.MemoryWeights, Bytes: 2 * gib, Scope: compute.MemoryScopeDevice},
		{Class: compute.MemoryOffload, Bytes: 2 * gib, Scope: compute.MemoryScopeHost},
	})
	resident := run(denseWeights())
	if split.DecodeTokSPerSeq != resident.DecodeTokSPerSeq || split.MaxConcurrency != resident.MaxConcurrency {
		t.Fatalf("split %+v resident %+v", split, resident)
	}
	if split.Bound != BoundMemory || hasOmission(split.Omitted, OmitOffloadBandwidthUnknown) {
		t.Fatalf("split bound %s omitted %v", split.Bound, split.Omitted)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDiscreteOffloadIsLinkBoundUnknown(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Weights = compute.MemoryPlan{
		{Class: compute.MemoryWeights, Bytes: 2 * gib, Scope: compute.MemoryScopeDevice},
		{Class: compute.MemoryOffload, Bytes: 2 * gib, Scope: compute.MemoryScopeHost},
	}
	pr := plan(t, in).Candidates[0]
	if pr.Bound != BoundLink || !hasOmission(pr.Omitted, OmitOffloadBandwidthUnknown) || pr.DecodeTokSPerSeq != 0 {
		t.Fatalf("prediction = %+v", pr)
	}
	if pr.MaxConcurrency <= 0 {
		t.Fatalf("MaxConcurrency = %d, want capacity still computed", pr.MaxConcurrency)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestStreamedWorkingSetIsLinkBound(t *testing.T) {
	for _, topo := range []Topology{TopologyDiscrete, TopologyUnified, TopologyHostOnly} {
		in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
		in.Hardware.Topology = topo
		in.Weights = compute.MemoryPlan{
			{Class: compute.MemoryWeights, Bytes: 2 * gib, Scope: compute.MemoryScopeDevice},
			{Class: compute.MemoryOffload, Bytes: 1 * gib, Scope: compute.MemoryScopeHost, DType: StreamedResidentDType},
		}
		pr := plan(t, in).Candidates[0]
		if pr.Bound != BoundLink || !hasOmission(pr.Omitted, OmitStreamBandwidthUnknown) {
			t.Fatalf("%s: bound %s omitted %v", topo, pr.Bound, pr.Omitted)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestMoEStreamBytesUnion(t *testing.T) {
	m := Model{StreamBytesPerToken: 10e9, ActiveRoutedBytesPerToken: 4e9, RoutedExpertBytes: 256e9, NumExperts: 256, ExpertsUsed: 4}
	if got := m.StreamBytes(1); got != 10e9 {
		t.Fatalf("StreamBytes(1) = %d", got)
	}
	prev := m.StreamBytes(1)
	for b := 2; b <= 64; b++ {
		got := m.StreamBytes(b)
		if got < prev {
			t.Fatalf("StreamBytes(%d) = %d < StreamBytes(%d) = %d", b, got, b-1, prev)
		}
		prev = got
	}
	if !(m.StreamBytes(2) > m.StreamBytes(1)) {
		t.Fatal("a second sequence adds no routed experts")
	}
	if got := m.StreamBytes(1 << 20); got != 6e9+256e9 {
		t.Fatalf("StreamBytes(huge) = %d, want every expert plus the non-routed stream", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSerialDecodeDoesNotScaleWithConcurrency(t *testing.T) {
	in := baseInput(400*gib, cand(compute.KVPrecisionF32, 4096, true))
	one := plan(t, in).Candidates[0]
	in.Workload.Concurrency = 4
	four := plan(t, in).Candidates[0]
	if !relEq(four.DecodeTokSAggregate, one.DecodeTokSPerSeq) || !relEq(four.DecodeTokSPerSeq, four.DecodeTokSAggregate/4) {
		t.Fatalf("one %+v four %+v", one, four)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestBatchedMoEDecodeRaisesAggregate(t *testing.T) {
	in := baseInput(400e9, cand(compute.KVPrecisionF32, 4096, true))
	in.Model = moeModel()
	in.Weights = compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 262e9, Scope: compute.MemoryScopeDevice}}
	one := plan(t, in).Candidates[0]
	in.Workload.Concurrency = 4
	serial := plan(t, in).Candidates[0]
	in.Workload.BatchedDecode = true
	batched := plan(t, in).Candidates[0]
	if !(batched.DecodeTokSAggregate > serial.DecodeTokSAggregate) || !(batched.DecodeTokSPerSeq < one.DecodeTokSPerSeq) {
		t.Fatalf("one %+v serial %+v batched %+v", one, serial, batched)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestInvalidCalibrationOmitsThroughput(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Calibration = Calibration{Tier: TierEmpirical, BandwidthEfficiency: 0, ComputeEfficiency: 0.5, Source: "x"}
	pr := plan(t, in).Candidates[0]
	if !hasOmission(pr.Omitted, OmitCalibrationInvalid) || pr.DecodeTokSPerSeq != 0 || pr.TTFTSeconds != 0 || pr.Bound != "" {
		t.Fatalf("prediction = %+v", pr)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestMoEActiveSetUnknownOmitsThroughput(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Model.ActiveSetUnknown = true
	pr := plan(t, in).Candidates[0]
	if !hasOmission(pr.Omitted, OmitMoEActiveSetUnknown) || pr.DecodeTokSPerSeq != 0 {
		t.Fatalf("prediction = %+v", pr)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDevicePeaksResolvesPublicTargets(t *testing.T) {
	p, ok := DevicePeaks(" GFX1151 ")
	cfg := strix.NewGFX1151Config()
	if !ok || p.BytesPerSec != cfg.PeakBandwidthGBps*1e9 || p.FLOPsPerSec != cfg.PeakFP32TFLOPs*1e12 || p.Source == "" {
		t.Fatalf("gfx1151 peaks = %+v ok=%v", p, ok)
	}
	if _, ok := DevicePeaks("gfx1150"); !ok {
		t.Fatal("gfx1150 unresolved")
	}
	for _, arch := range []string{"sm90", ""} {
		if p, ok := DevicePeaks(arch); ok || p != (Peaks{}) {
			t.Fatalf("%q resolved to %+v", arch, p)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestContextLadder(t *testing.T) {
	for _, tc := range []struct {
		resolved, max int
		extra         []int
		want          []int
	}{
		{10000, 131072, nil, []int{4096, 10000, 32768, 131072}},
		{16, 16, nil, []int{16}},
		{4096, 0, nil, []int{4096}},
		{10000, 131072, []int{50000, 0, 4096, 262144}, []int{4096, 10000, 32768, 50000, 131072}},
	} {
		if got := ContextLadder(tc.resolved, tc.max, tc.extra...); !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("ContextLadder(%d, %d, %v) = %v, want %v", tc.resolved, tc.max, tc.extra, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestBuildCandidatesMarksResolvedCurrentAndDedupesTiers(t *testing.T) {
	tiers := testTiers()
	cs := BuildCandidates(tiers, 10000)
	current := 0
	for _, c := range cs {
		if c.Current {
			current++
			if c.ContextTokens != 10000 || c.KVPrecision != tiers[0].Precision {
				t.Fatalf("current candidate = %+v", c)
			}
		}
	}
	if current != 1 {
		t.Fatalf("%d current candidates, want 1", current)
	}
	one := BuildCandidates(tiers[:1], 10000)
	if dup := BuildCandidates([]KVTier{tiers[0], tiers[0]}, 10000); len(dup) != len(one) {
		t.Fatalf("identical tiers emitted %d rows, want %d", len(dup), len(one))
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRowGatheredEmbeddingIsNotSwept(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	// A 1 GiB-payload untied embedding the arm keeps as a 4 GiB f32 gather table.
	in.Model.PayloadWeightBytes = 5 * gib
	in.Model.GatherPayloadBytes = gib
	in.Model.GatherResidentBytes = 4 * gib
	in.Weights = compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 8 * gib, Scope: compute.MemoryScopeDevice}}
	pr := plan(t, in).Candidates[0]
	f32KV := testCSC(compute.KVPrecisionF32).KVBytes(2048) * 2 / 3
	if pr.DecodeBytesPerStep != 4*gib+f32KV {
		t.Fatalf("DecodeBytesPerStep = %d, want the %d swept weights plus %d KV", pr.DecodeBytesPerStep, 4*gib, f32KV)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPrefillAppliesHeadOnce(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Calibration = GenericEmpirical()
	in.Model.HeadParams = 1e9
	pr := plan(t, in).Candidates[0]
	flopsPre := 2*6e9*2048 + 2*1e9 + 2*32*32*128*2048.0*2048
	bytesPre := float64(4<<30 + testCSC(compute.KVPrecisionF32).KVBytes(2048))
	want := math.Max(flopsPre/(0.5*29.7e12), bytesPre/(0.55*256e9))
	if !relEq(pr.TTFTSeconds, want) {
		t.Fatalf("ttft %v, want %v", pr.TTFTSeconds, want)
	}
	in.Hardware.Peaks.FLOPsPerSec, in.Calibration = 1e9, sol()
	decode := plan(t, in).Candidates[0]
	if want := 1e9 / (2*7e9 + 4*32*32*128*2048); !relEq(decode.DecodeTokSPerSeq, want) {
		t.Fatalf("compute-bound decode %v, want %v with the head charged every step", decode.DecodeTokSPerSeq, want)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestUnknownModelFLOPsStatesBandwidthDecodeOnly(t *testing.T) {
	in := baseInput(24*gib, cand(compute.KVPrecisionF32, 32768, true))
	in.Model.ActiveParamsPerToken, in.Model.AttnLayers, in.Model.NumHeads, in.Model.HeadDim = 0, 0, 0, 0
	pr := plan(t, in).Candidates[0]
	if pr.Bound != BoundMemory || !relEq(pr.DecodeTokSPerSeq, 256e9/float64(pr.DecodeBytesPerStep)) {
		t.Fatalf("bound %s per-seq %v, want the memory bound", pr.Bound, pr.DecodeTokSPerSeq)
	}
	if pr.PrefillTokS != 0 || pr.TTFTSeconds != 0 || !hasOmission(pr.Omitted, OmitModelFLOPsUnknown) || hasOmission(pr.Omitted, OmitPeakComputeUnknown) {
		t.Fatalf("prediction = %+v", pr)
	}
}
