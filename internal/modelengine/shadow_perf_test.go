package modelengine

// shadow_perf_test.go — phase-2 shadow-perf harness witnesses. All results are
// [SW-VERIFIED]: pure in-process CPU simulation, no hardware claims. The
// ticket's acceptance filter is -run ShadowPerf; every test here is named
// TestShadowPerf*.

import (
	"encoding/json"
	"sort"
	"strings"
	"testing"
)

func shadowPerfMedian(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := make([]float64, len(xs))
	copy(s, xs)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2.0
}

// shadowPerfTrimmedMedian drops the fastest and slowest sample before
// mediating: a single scheduling preemption shows up as an extreme sample and
// must not drag the estimate. Requires ≥4 samples; smaller sets fall back to
// the plain median.
func shadowPerfTrimmedMedian(xs []float64) float64 {
	if len(xs) < 4 {
		return shadowPerfMedian(xs)
	}
	s := make([]float64, len(xs))
	copy(s, xs)
	sort.Float64s(s)
	return shadowPerfMedian(s[1 : len(s)-1])
}

// shadowPerfLightOpts keeps the witnesses in the low-second range: the
// tier/ABBA/split-phase contracts are identical at any depth, only the
// simulated layer count and block count shrink. Four blocks give an eight-
// sample set per arm-phase; the ordering witness trims one extreme off each
// side before mediating, so a single preemption cannot drag the estimate.
func shadowPerfLightOpts() ShadowPerfOptions {
	return ShadowPerfOptions{ModelLayers: 16, ABBAReps: 4}
}

// TestShadowPerfFaithfulCandidatePopulatesSplitPhases proves the accept-path
// shape: a bit-exact, faster candidate yields populated, separate prefill and
// decode sample sets on both tiers, with the user path untouched.
func TestShadowPerfFaithfulCandidatePopulatesSplitPhases(t *testing.T) {
	live := shadowPerfCorpusFixture()
	ref := shadowPerfIncumbent()
	cand := shadowPerfFaithfulFastCandidate()

	// Capture the user path BEFORE any candidate work happens. The user path is
	// served by the live reference engine (shadow_traffic.go seam), never by a
	// perf-harness engine.
	liveRef := shadowReferenceEngine()
	var servedBefore []string
	for _, l := range live {
		servedBefore = append(servedBefore, strings.Join(shadowServeLive(l, liveRef), "|"))
	}

	got := ShadowPerfEvaluate(live, ref, cand, shadowPerfLightOpts())
	if !got.VerifyGate.Passed {
		t.Fatalf("faithful candidate failed the verify gate: %+v", got.VerifyGate)
	}
	if !got.PerfCountsValid || !got.IsolationPreserved {
		t.Fatalf("receipt flags wrong: valid=%v isolated=%v", got.PerfCountsValid, got.IsolationPreserved)
	}
	if len(got.Tiers) != 2 {
		t.Fatalf("want two tiers, got %d: %+v", len(got.Tiers), got.Tiers)
	}
	for _, tr := range got.Tiers {
		if len(tr.BaselinePrefillTokS) == 0 || len(tr.CandidatePrefillTokS) == 0 {
			t.Fatalf("%s tier missing prefill_tok_s samples: %+v", tr.Tier, tr)
		}
		if len(tr.BaselineDecodeTokS) == 0 || len(tr.CandidateDecodeTokS) == 0 {
			t.Fatalf("%s tier missing decode_tok_s samples: %+v", tr.Tier, tr)
		}
		for i, v := range tr.BaselinePrefillTokS {
			if v <= 0 {
				t.Fatalf("%s baseline_prefill_tok_s[%d] = %v, want > 0", tr.Tier, i, v)
			}
		}
		for i, v := range tr.CandidateDecodeTokS {
			if v <= 0 {
				t.Fatalf("%s candidate_decode_tok_s[%d] = %v, want > 0", tr.Tier, i, v)
			}
		}
		// Split phases must be distinct measurements, not one blended number
		// copied into two fields.
		if shadowPerfMedian(tr.BaselinePrefillTokS) == shadowPerfMedian(tr.BaselineDecodeTokS) &&
			shadowPerfMedian(tr.CandidatePrefillTokS) == shadowPerfMedian(tr.CandidateDecodeTokS) {
			t.Fatalf("%s prefill and decode medians identical — split labels not independently measured: %+v", tr.Tier, tr)
		}
	}

	// Isolation: what the user receives after the full candidate evaluation is
	// exactly what the incumbent-only serving produced before it.
	for i, l := range live {
		after := strings.Join(shadowServeLive(l, liveRef), "|")
		if after != servedBefore[i] {
			t.Fatalf("user path changed for %s: before=%q after=%q", l.id, servedBefore[i], after)
		}
	}
}

// TestShadowPerfVerifyGateRunsBeforePerfCounts proves the ordering invariant:
// a divergent candidate produces ZERO perf samples — the oracle runs first and
// its refusal means no count counts.
func TestShadowPerfVerifyGateRunsBeforePerfCounts(t *testing.T) {
	live := shadowPerfCorpusFixture()
	ref := shadowPerfIncumbent()
	// A faster-but-divergent candidate: even a big perf win must not count
	// when the oracle refuses it.
	cand := shadowPerfEngine{name: "candidate-sampler-drift", backend: "native-sampler", efficiency: 50, decode: shadowSamplerDriftDefect}

	got := ShadowPerfEvaluate(live, ref, cand, shadowPerfLightOpts())

	if got.VerifyGate.Passed {
		t.Fatalf("divergent candidate passed the verify gate: %+v", got.VerifyGate)
	}
	if got.VerifyGate.DivergenceIndex < 0 {
		t.Fatalf("refusal not localized to a token index: %+v", got.VerifyGate)
	}
	if got.PerfCountsValid {
		t.Fatalf("perf counts marked valid after verify-gate refusal: %+v", got)
	}
	if len(got.Tiers) != 0 {
		t.Fatalf("refused candidate produced tier results: %+v", got.Tiers)
	}
	for _, tr := range got.Tiers {
		if len(tr.BaselinePrefillTokS)+len(tr.BaselineDecodeTokS)+len(tr.CandidatePrefillTokS)+len(tr.CandidateDecodeTokS) != 0 {
			t.Fatalf("refused candidate carries perf samples: %+v", tr)
		}
	}
}

// TestShadowPerfTwoTiersLayerSubsetAndFullDepth proves the tier contract: the
// fast tier measures a layer subset with capped spans, the slow tier measures
// full depth with uncapped spans.
func TestShadowPerfTwoTiersLayerSubsetAndFullDepth(t *testing.T) {
	opts := shadowPerfLightOpts()
	got := ShadowPerfEvaluate(shadowPerfCorpusFixture(), shadowPerfIncumbent(), shadowPerfFaithfulFastCandidate(), opts)
	if len(got.Tiers) != 2 {
		t.Fatalf("want two tiers, got %d", len(got.Tiers))
	}
	fast, slow := got.Tiers[0], got.Tiers[1]
	if fast.Tier != "fast" || slow.Tier != "slow" {
		t.Fatalf("tier order wrong: %q then %q", fast.Tier, slow.Tier)
	}
	if fast.Layers >= slow.Layers {
		t.Fatalf("fast tier must use a layer subset: fast=%d slow=%d", fast.Layers, slow.Layers)
	}
	if fast.SpanCap == 0 || slow.SpanCap != 0 {
		t.Fatalf("span caps wrong: fast=%d (want >0) slow=%d (want 0=uncapped)", fast.SpanCap, slow.SpanCap)
	}
	wantFastCap := opts.ModelLayers / 8
	if wantFastCap < 1 {
		wantFastCap = 1
	}
	if fast.Layers != wantFastCap {
		t.Fatalf("fast tier layers = %d, want configured cap %d", fast.Layers, wantFastCap)
	}
	if slow.Layers != opts.ModelLayers {
		t.Fatalf("slow tier layers = %d, want full depth %d", slow.Layers, opts.ModelLayers)
	}
}

// TestShadowPerfFastTierSandboxAbsorbsCrash proves the crash-absorber contract:
// a candidate that is bit-exact on the probe set but faults on long contexts
// is caught by the sandboxed fast tier; the slow tier never runs, no perf
// counts count, and the user path is unaffected.
func TestShadowPerfFastTierSandboxAbsorbsCrash(t *testing.T) {
	live := shadowPerfCorpusFixture()
	ref := shadowPerfIncumbent()

	got := ShadowPerfEvaluate(live, ref, shadowPerfCrashCandidate(), shadowPerfLightOpts())
	if !got.VerifyGate.Passed {
		t.Fatalf("crash candidate should pass the probe-set verify gate (fault is context-length-dependent): %+v", got.VerifyGate)
	}
	if len(got.Tiers) != 1 {
		t.Fatalf("want exactly the fast tier recorded before the crash, got %d: %+v", len(got.Tiers), got.Tiers)
	}
	fast := got.Tiers[0]
	if !fast.Crashed || fast.CrashReason == "" {
		t.Fatalf("fast tier did not record the absorbed crash: %+v", fast)
	}
	if got.PerfCountsValid {
		t.Fatalf("perf counts marked valid after a candidate crash: %+v", got)
	}
	// Isolation survives the crash: the user still gets exactly the incumbent
	// (reference) response, untouched by the absorbed candidate fault.
	liveRef := shadowReferenceEngine()
	for _, l := range live {
		served := shadowServeLive(l, liveRef)
		want := shadowReferenceDecode(shadowScrub(l))
		if len(served) == 0 || strings.Join(served, "|") != strings.Join(want, "|") {
			t.Fatalf("user path broken after crash absorption for %s", l.id)
		}
	}
}

// TestShadowPerfABBAInterleavingAndSampleCounts proves the ABBA schedule: reps
// blocks of A B B A, each block contributing one outer-pair and one inner-pair
// sample per arm, and the block records mapping back onto the flat arrays.
func TestShadowPerfABBAInterleavingAndSampleCounts(t *testing.T) {
	const reps = 3
	got := ShadowPerfEvaluate(shadowPerfCorpusFixture(), shadowPerfIncumbent(), shadowPerfFaithfulFastCandidate(),
		ShadowPerfOptions{ModelLayers: 16, ABBAReps: reps})
	for _, tr := range got.Tiers {
		if len(tr.Blocks) != reps {
			t.Fatalf("%s tier: want %d ABBA blocks, got %d", tr.Tier, reps, len(tr.Blocks))
		}
		for i, b := range tr.Blocks {
			if b.BaselinePrefill[0] != tr.BaselinePrefillTokS[2*i] || b.BaselinePrefill[1] != tr.BaselinePrefillTokS[2*i+1] {
				t.Fatalf("%s block %d: baseline prefill pair does not map onto flat array: %v vs %v", tr.Tier, i, b.BaselinePrefill, tr.BaselinePrefillTokS)
			}
			if b.CandidateDecode[0] != tr.CandidateDecodeTokS[2*i] || b.CandidateDecode[1] != tr.CandidateDecodeTokS[2*i+1] {
				t.Fatalf("%s block %d: candidate decode pair does not map onto flat array: %v vs %v", tr.Tier, i, b.CandidateDecode, tr.CandidateDecodeTokS)
			}
		}
		if len(tr.BaselinePrefillTokS) != reps*2 || len(tr.CandidateDecodeTokS) != reps*2 {
			t.Fatalf("%s tier: want %d samples per arm-phase, got base=%d cand=%d", tr.Tier, reps*2, len(tr.BaselinePrefillTokS), len(tr.CandidateDecodeTokS))
		}
	}
}

// TestShadowPerfPrivacyNoSecretInReceipt proves the privacy invariant end to
// end: the persisted receipt (marshaled JSON) contains no raw prompt byte and
// no end-user id from either the corpus or the probe set.
func TestShadowPerfPrivacyNoSecretInReceipt(t *testing.T) {
	live := shadowPerfCorpusFixture()
	data, err := json.Marshal(ShadowPerfEvaluate(live, shadowPerfIncumbent(), shadowPerfFaithfulFastCandidate(), shadowPerfLightOpts()))
	if err != nil {
		t.Fatalf("receipt does not marshal: %v", err)
	}
	doc := string(data)
	for _, l := range append(append([]liveRequest{}, live...), shadowPerfProbeSet()...) {
		for _, secret := range shadowSecretsOf(l) {
			if secret != "" && strings.Contains(doc, secret) {
				t.Fatalf("receipt leaked secret %q", secret)
			}
		}
	}
	if !strings.Contains(doc, "promptDigest") && !strings.Contains(doc, "sha256:") {
		t.Fatalf("receipt carries no scrubbed digest evidence: %s", doc[:min(len(doc), 200)])
	}
}

// TestShadowPerfResultPinningTypedCoordinates proves §6.1 pinning: the traffic
// window is always computed and deterministic; generation coordinates are
// typed-unpinned with reasons when the seam is absent, and pinned verbatim
// when a landed seam supplies them. No fabricated values.
func TestShadowPerfResultPinningTypedCoordinates(t *testing.T) {
	live := shadowPerfCorpusFixture()
	one := ShadowPerfEvaluate(live, shadowPerfIncumbent(), shadowPerfFaithfulFastCandidate(), shadowPerfLightOpts())
	two := ShadowPerfEvaluate(live, shadowPerfIncumbent(), shadowPerfFaithfulFastCandidate(), shadowPerfLightOpts())

	if one.Pin.TrafficWindow == "" || !strings.HasPrefix(one.Pin.TrafficWindow, "sha256:") {
		t.Fatalf("traffic window not computed: %+v", one.Pin)
	}
	if one.Pin.TrafficWindow != two.Pin.TrafficWindow {
		t.Fatalf("traffic window not deterministic for the same corpus: %s vs %s", one.Pin.TrafficWindow, two.Pin.TrafficWindow)
	}
	for name, c := range map[string]struct {
		pinned bool
		src    string
	}{
		"weights_epoch": {one.Pin.WeightsEpoch.Pinned, one.Pin.WeightsEpoch.Source},
		"forward_gen":   {one.Pin.ForwardGen.Pinned, one.Pin.ForwardGen.Source},
		"placement_gen": {one.Pin.PlacementGen.Pinned, one.Pin.PlacementGen.Source},
	} {
		if c.pinned || c.src == "" {
			t.Fatalf("coordinate %s must be typed-unpinned with a reason in this base: %+v", name, c)
		}
	}

	// Extension point: a landed seam supplies an observed forward-gen.
	fg := uint64(7)
	pinned := ShadowPerfEvaluate(live, shadowPerfIncumbent(), shadowPerfFaithfulFastCandidate(), ShadowPerfOptions{
		Pins: ShadowPerfPins{ForwardGen: shadowPerfObserved(fg, "modelengine.ForwardSlot.Snapshot()"), HasForwardGen: true},
	})
	if !pinned.Pin.ForwardGen.Pinned || pinned.Pin.ForwardGen.Value != fg || pinned.Pin.ForwardGen.Source == "" {
		t.Fatalf("supplied forward-gen not pinned verbatim: %+v", pinned.Pin.ForwardGen)
	}
	if pinned.Pin.WeightsEpoch.Pinned {
		t.Fatalf("unsupplied weights-epoch must stay unpinned: %+v", pinned.Pin.WeightsEpoch)
	}
}

// TestShadowPerfMeasuredThroughputOrdersArms proves the harness actually
// measures the engineered difference (anti-dead-code sanity): the faster
// candidate's median decode_tok_s exceeds the incumbent's, the slower
// candidate's falls below it — on the authoritative slow tier, both phases.
func TestShadowPerfMeasuredThroughputOrdersArms(t *testing.T) {
	live := shadowPerfCorpusFixture()
	ref := shadowPerfIncumbent()

	fast := ShadowPerfEvaluate(live, ref, shadowPerfFaithfulFastCandidate(), shadowPerfLightOpts())
	slow := ShadowPerfEvaluate(live, ref, shadowPerfRegressingCandidate(), shadowPerfLightOpts())
	ft, st := fast.Tiers[1], slow.Tiers[1] // slow tier is index 1

	if ft.CandidateDecodeTokS[0] <= 0 || ft.BaselineDecodeTokS[0] <= 0 {
		t.Fatalf("slow-tier samples degenerate: fast=%+v slow=%+v", ft, st)
	}
	fastCand, fastBase := shadowPerfTrimmedMedian(ft.CandidateDecodeTokS), shadowPerfTrimmedMedian(ft.BaselineDecodeTokS)
	slowCand, slowBase := shadowPerfTrimmedMedian(st.CandidateDecodeTokS), shadowPerfTrimmedMedian(st.BaselineDecodeTokS)
	fastRatio, slowRatio := fastCand/fastBase, slowCand/slowBase
	// Engineered gaps: 40% of the work (≈2.5× tok/s) and 2.5× the work
	// (≈0.4×). Thresholds sit far inside the true effects so host scheduling
	// noise — even on a degraded, contended CI host — cannot flip the
	// ordering verdict.
	if fastRatio < 1.50 {
		t.Fatalf("faster candidate not measured faster: ratio=%.3f (decode_tok_s trimmed medians cand=%.1f base=%.1f)", fastRatio, fastCand, fastBase)
	}
	if slowRatio > 0.65 {
		t.Fatalf("slower candidate not measured slower: ratio=%.3f (decode_tok_s trimmed medians cand=%.1f base=%.1f)", slowRatio, slowCand, slowBase)
	}
	// Prefill phase carries the same ordering (split labels, same signal).
	if shadowPerfTrimmedMedian(ft.CandidatePrefillTokS) <= shadowPerfTrimmedMedian(ft.BaselinePrefillTokS) {
		t.Fatalf("prefill phase lost the ordering: cand=%.1f base=%.1f", shadowPerfTrimmedMedian(ft.CandidatePrefillTokS), shadowPerfTrimmedMedian(ft.BaselinePrefillTokS))
	}
}
