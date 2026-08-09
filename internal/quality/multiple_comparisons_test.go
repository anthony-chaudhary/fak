package quality

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

// Tests for the multiplicity controller (#4568). The load-bearing one is the NULL
// SIMULATION: it runs thousands of families of purely-null comparisons through the
// shipped ControlMultiplicity path and measures how often each policy manufactures
// a discovery. Under the global null every rejection is false by construction, so
// the measured "at least one rejection" rate IS the family-wise error rate — and
// because all nulls are true, the false discovery proportion is 1 whenever there
// is any rejection, so the same number is the FDR. One simulation therefore checks
// both documented bounds, and the uncorrected arm measures exactly what the
// correction is buying.

const (
	mcAlpha  = 0.05
	mcTrials = 5000
	mcSeed   = int64(4568_2026)
	// mcSlack is Monte-Carlo headroom on the bound. At 5000 trials the standard
	// error of a rate near 0.05 is sqrt(.05*.95/5000) ~= 0.0031, so 0.01 is a bit
	// over three standard errors — loose enough that the assertion is about the
	// procedure rather than about the draw, tight enough that a genuinely
	// uncontrolled policy (0.64, below) cannot hide inside it. The PRNG is
	// seed-pinned, so the measured value is identical on every run regardless.
	mcSlack = 0.01
)

func mcModels() []string { return []string{"fixture-7b", "fixture-70b"} }
func mcSlices() []string { return []string{"short-prompt", "long-context"} }
func mcMetrics() []string {
	return []string{"exact_match", "grounding", "citation_precision", "refusal_rate", "latency_p95"}
}

// mcPrimary declares two of the five metrics primary, so a full grid is 2 models x
// 2 slices x 5 metrics = 20 comparisons: 8 primary and 12 secondary.
func mcPrimary() []string { return []string{"exact_match", "grounding"} }

func mcPolicy(correction MultiplicityCorrection, primary []string) MultiplicityPolicy {
	return MultiplicityPolicy{Alpha: mcAlpha, Correction: correction, Primary: primary}
}

// mcProvenance is complete evidence provenance: model, tokenizer, engine/backend,
// a deterministic oracle, the code revision, and the tolerance/baseline it was
// judged against — the six fields #4568 requires every case to record.
func mcProvenance(model string) EvidenceProvenance {
	return EvidenceProvenance{
		Model:     model,
		Tokenizer: "fixture-tokenizer@sha256:tokenizer-v1",
		Engine:    "fak/cpu",
		Oracle:    "greedy-token-diff",
		Revision:  "git:38185e22199bf791df73195b93ba29bbebeed8e3",
		Baseline:  "sha256:baseline-multiplicity-v1",
	}
}

// mcComparison builds one admissible cell carrying a scrubbed replay artifact, so
// nothing blocks for administrative reasons and every block in these tests is a
// statistical decision.
func mcComparison(model, slice, metric string, p float64) MetricComparison {
	id := fmt.Sprintf("%s-%s-%s", model, slice, metric)
	return MetricComparison{
		Model: model, Slice: slice, Metric: metric, P: p,
		Tier: TierNightly, CostSeconds: 42,
		Evidence: Evidence{
			CaseID:     id,
			State:      StatePass,
			Provenance: mcProvenance(model),
			Replay:     &FailureBundle{CaseID: id, Scrubbed: true, FailingOracle: "distribution-tv", FailingKind: "distribution"},
		},
	}
}

// mcNullFamily draws one full grid of p-values from the GLOBAL NULL: under a true
// null hypothesis a p-value is uniform on [0, 1] by construction, so independent
// uniforms are exactly "nothing regressed anywhere". Any rejection the controller
// makes on this family is therefore a false discovery, with no judgement call.
func mcNullFamily(rng *splitMix64) []MetricComparison {
	var fam []MetricComparison
	for _, m := range mcModels() {
		for _, s := range mcSlices() {
			for _, metric := range mcMetrics() {
				fam = append(fam, mcComparison(m, s, metric, rng.float64()))
			}
		}
	}
	return fam
}

// mcSimulateNull returns the fraction of null families in which the policy
// declared at least one discovery.
func mcSimulateNull(t *testing.T, policy MultiplicityPolicy) float64 {
	t.Helper()
	rng := &splitMix64{state: uint64(mcSeed)}
	errors := 0
	for i := 0; i < mcTrials; i++ {
		d, err := ControlMultiplicity(policy, mcNullFamily(rng))
		if err != nil {
			t.Fatalf("trial %d: ControlMultiplicity: %v", i, err)
		}
		for _, dec := range d.Decisions {
			if dec.Rejected {
				errors++
				break
			}
		}
	}
	return float64(errors) / float64(mcTrials)
}

// TestNullSimulationHonorsDocumentedBounds is the #4568 acceptance criterion: over
// 5000 simulated null grids, every corrected policy keeps its documented rate at
// or under alpha. Three shapes are measured — Holm and BH over a flat family (all
// twenty metrics primary), and Holm under the hierarchical gate (eight primary,
// twelve secondary) — because gatekeeping is a separate claim from the correction
// and could plausibly leak error budget through the second family.
func TestNullSimulationHonorsDocumentedBounds(t *testing.T) {
	all := mcMetrics() // every metric primary: one flat family, no gatekeeping
	for _, tc := range []struct {
		name   string
		policy MultiplicityPolicy
		bound  string
	}{
		{"holm-flat", mcPolicy(CorrectionHolm, all), "family-wise error rate"},
		{"benjamini-hochberg-flat", mcPolicy(CorrectionBenjaminiHochberg, all), "false discovery rate"},
		{"holm-gatekept", mcPolicy(CorrectionHolm, mcPrimary()), "family-wise error rate (8 primary, 12 gated secondary)"},
		{"benjamini-hochberg-gatekept", mcPolicy(CorrectionBenjaminiHochberg, mcPrimary()), "false discovery rate (8 primary, 12 gated secondary)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rate := mcSimulateNull(t, tc.policy)
			t.Logf("%s: measured %s = %.4f over %d null grids at alpha %.2f", tc.name, tc.bound, rate, mcTrials, mcAlpha)
			if rate > mcAlpha+mcSlack {
				t.Errorf("%s: measured rate %.4f exceeds the documented bound %.4f (+%.4f Monte-Carlo slack)",
					tc.name, rate, mcAlpha, mcSlack)
			}
		})
	}
}

// TestUncorrectedFamilyManufacturesFalseDiscoveries is the planted representative
// defect for this contract: judging the SAME null grids at the raw per-comparison
// alpha — the policy a suite has by default when nobody declares a correction —
// declares a regression on well over half of perfectly clean runs. The theoretical
// value is 1 - (1 - 0.05)^20 = 0.6415; measuring it is what makes the correction's
// bound a result rather than an assertion.
func TestUncorrectedFamilyManufacturesFalseDiscoveries(t *testing.T) {
	all := mcMetrics()
	uncorrected := mcSimulateNull(t, mcPolicy(CorrectionNone, all))
	corrected := mcSimulateNull(t, mcPolicy(CorrectionHolm, all))
	t.Logf("uncorrected false-discovery rate %.4f vs holm %.4f over %d null grids (theory: 1-(1-a)^20 = %.4f)",
		uncorrected, corrected, mcTrials, 1-math.Pow(1-mcAlpha, 20))

	if uncorrected < 0.5 {
		t.Errorf("uncorrected rate %.4f < 0.50: the planted defect must visibly manufacture discoveries, or the correction is not load-bearing", uncorrected)
	}
	if corrected <= 0 || uncorrected < 10*corrected {
		t.Errorf("uncorrected %.4f is not an order of magnitude above holm %.4f: the correction is not doing measurable work", uncorrected, corrected)
	}
}

// TestUncorrectedFamilyIsRefusedOutright proves the controller does not merely
// report the uncontrolled rate — it refuses to call such a family a pass, even on
// a grid where no cell was rejected. An unbounded verdict is inconclusive evidence,
// and inconclusive evidence is never a pass.
func TestUncorrectedFamilyIsRefusedOutright(t *testing.T) {
	fam := []MetricComparison{
		mcComparison("fixture-7b", "short-prompt", "exact_match", 0.71),
		mcComparison("fixture-7b", "short-prompt", "grounding", 0.63),
	}
	d, err := ControlMultiplicity(mcPolicy(CorrectionNone, mcPrimary()), fam)
	if err != nil {
		t.Fatalf("ControlMultiplicity: %v", err)
	}
	if len(d.Blocks) != 0 {
		t.Fatalf("no cell should have been rejected; got blocks %+v", d.Blocks)
	}
	if d.Pass {
		t.Error("an uncorrected family must never pass, even with nothing rejected")
	}
	if len(d.Refusals) != 1 || !strings.Contains(d.Refusals[0], "honors no documented family-wise or FDR bound") {
		t.Errorf("refusal must name the missing bound; got %#v", d.Refusals)
	}
	if !strings.Contains(d.Bound, "NO documented bound") {
		t.Errorf("bound statement must say so plainly; got %q", d.Bound)
	}
}

// TestHolmAndBHAdjustedPValues pins the arithmetic itself against hand-computed
// values, so a future edit to the step procedures fails here rather than silently
// shifting every gate. The same five p-values are corrected two ways: Holm rejects
// two, BH rejects all five — the power difference that makes the choice a real
// declaration rather than a formality.
func TestHolmAndBHAdjustedPValues(t *testing.T) {
	raw := []float64{0.001, 0.008, 0.039, 0.041, 0.042}
	metrics := mcMetrics() // five metrics, all declared primary below
	fam := make([]MetricComparison, len(raw))
	for i, p := range raw {
		fam[i] = mcComparison("fixture-7b", "short-prompt", metrics[i], p)
	}

	for _, tc := range []struct {
		correction MultiplicityCorrection
		want       []float64
		wantReject int
	}{
		// Holm: (5*.001, 4*.008, 3*.039, 2*.041, 1*.042) under a running max.
		{CorrectionHolm, []float64{0.005, 0.032, 0.117, 0.117, 0.117}, 2},
		// BH: (5/1*.001, 5/2*.008, 5/3*.039, 5/4*.041, 5/5*.042) under a running min.
		{CorrectionBenjaminiHochberg, []float64{0.005, 0.020, 0.042, 0.042, 0.042}, 5},
	} {
		t.Run(string(tc.correction), func(t *testing.T) {
			d, err := ControlMultiplicity(mcPolicy(tc.correction, metrics), fam)
			if err != nil {
				t.Fatalf("ControlMultiplicity: %v", err)
			}
			rejected := 0
			for i, dec := range d.Decisions {
				if math.Abs(dec.Adjusted-tc.want[i]) > 1e-12 {
					t.Errorf("cell %d (%s) adjusted p = %.6f, want %.6f", i, dec.Cell, dec.Adjusted, tc.want[i])
				}
				if dec.Rejected {
					rejected++
				}
			}
			if rejected != tc.wantReject {
				t.Errorf("%s rejected %d cell(s), want %d", tc.correction, rejected, tc.wantReject)
			}
		})
	}
}

// TestNominalAlarmIsRefusedByTheCorrection is the concrete false discovery the
// contract exists to stop: one cell lands at p = 0.03, "significant" at a naive
// alpha of 0.05, among nineteen clean siblings. Uncorrected it becomes a declared
// regression; Holm multiplies it by the family size to 0.60 and the grid passes.
func TestNominalAlarmIsRefusedByTheCorrection(t *testing.T) {
	fam := mcNullFamily(&splitMix64{state: 99})
	for i := range fam {
		fam[i].P = 0.4 + 0.02*float64(i) // uniformly unremarkable
	}
	fam[3].P = 0.03 // the lone nominal alarm, on a primary metric
	if !mcPolicy(CorrectionHolm, mcMetrics()).isPrimary(fam[3].Metric) {
		t.Fatalf("setup: cell 3 metric %q must be part of the flat primary set", fam[3].Metric)
	}

	naive, err := ControlMultiplicity(mcPolicy(CorrectionNone, mcMetrics()), fam)
	if err != nil {
		t.Fatalf("uncorrected ControlMultiplicity: %v", err)
	}
	if !naive.Decisions[3].Rejected {
		t.Fatal("setup: the uncorrected policy was expected to (wrongly) declare the p=0.03 cell a discovery")
	}

	corrected, err := ControlMultiplicity(mcPolicy(CorrectionHolm, mcMetrics()), fam)
	if err != nil {
		t.Fatalf("corrected ControlMultiplicity: %v", err)
	}
	if corrected.Decisions[3].Rejected {
		t.Errorf("holm must refuse the nominal alarm; adjusted p = %.4f", corrected.Decisions[3].Adjusted)
	}
	if want := math.Min(1, 20*0.03); math.Abs(corrected.Decisions[3].Adjusted-want) > 1e-12 {
		t.Errorf("adjusted p = %.6f, want %.6f", corrected.Decisions[3].Adjusted, want)
	}
	if !corrected.Pass {
		t.Errorf("a grid whose only alarm is noise must pass; got %s", ExplainMultiplicity(corrected))
	}
}

// TestPlantedRegressionBlocksAndTheFixClears is the #4568 witness, driven end to
// end through the spine: a real RunCase failure (the demo decode defect, token 1
// "increased" -> "decreased") is folded into Evidence by EvidenceFromResult, given
// the p-value such a regression produces, and adjudicated. The defect grid blocks
// with the divergence and a scrubbed replay artifact attached; the fixed grid —
// identical but for the engine — clears.
func TestPlantedRegressionBlocksAndTheFixClears(t *testing.T) {
	grid := func(defect string, p float64) []MetricComparison {
		c := DemoCase()
		oracles, err := Lookup(c.Oracles)
		if err != nil {
			t.Fatalf("Lookup(%v): %v", c.Oracles, err)
		}
		res, err := RunCase(c, ReferenceRunner{}, DemoEngine(defect), oracles)
		if err != nil {
			t.Fatalf("RunCase(%q): %v", defect, err)
		}
		if res.Pass == (defect != "") {
			t.Fatalf("setup: RunCase(%q).Pass = %t", defect, res.Pass)
		}
		fam := mcNullFamily(&splitMix64{state: 7})
		for i := range fam {
			fam[i].P = 0.3 + 0.03*float64(i)
		}
		fam[0].Model, fam[0].Slice, fam[0].Metric = "fixture-7b", "short-prompt", "exact_match"
		fam[0].P = p
		fam[0].Evidence = EvidenceFromResult(mcProvenance("fixture-7b"), res)
		return fam
	}

	policy := mcPolicy(CorrectionHolm, mcPrimary())

	defected, err := ControlMultiplicity(policy, grid("decode", 2e-7))
	if err != nil {
		t.Fatalf("defect ControlMultiplicity: %v", err)
	}
	if defected.Pass {
		t.Fatalf("a planted decode regression must block the grid; got %s", ExplainMultiplicity(defected))
	}
	if len(defected.Blocks) != 1 {
		t.Fatalf("want exactly the planted cell blocking, got %d: %s", len(defected.Blocks), ExplainMultiplicity(defected))
	}
	blk := defected.Blocks[0]
	if blk.Cell != "fixture-7b/short-prompt/exact_match" || blk.State != StateFail || !blk.Rejected {
		t.Errorf("block identity = %s/%s/rejected=%t, want the planted primary cell failing", blk.Cell, blk.State, blk.Rejected)
	}
	// Holm over 8 primary comparisons: 8 * 2e-7 = 1.6e-6, still decisively under alpha.
	if want := 8 * 2e-7; math.Abs(blk.Adjusted-want) > 1e-15 {
		t.Errorf("adjusted p = %g, want %g", blk.Adjusted, want)
	}
	if blk.FirstDivergence == nil || blk.FirstDivergence.Index != 1 ||
		blk.FirstDivergence.Reference != "increased" || blk.FirstDivergence.Engine != "decreased" {
		t.Errorf("block must carry the spine's first actionable divergence; got %+v", blk.FirstDivergence)
	}
	if blk.Replay == nil || !blk.Replay.Scrubbed {
		t.Errorf("block must carry a scrubbed replay artifact; got %+v", blk.Replay)
	}
	if !strings.Contains(blk.Reason, "regression declared") || !strings.Contains(blk.Reason, "first divergence at token 1") {
		t.Errorf("reason must name the regression and localize it; got %q", blk.Reason)
	}
	if out := ExplainMultiplicity(defected); !strings.Contains(out, "BLOCKED") || !strings.Contains(out, "-> ") {
		t.Errorf("readout must lead with the first actionable block; got:\n%s", out)
	}

	fixed, err := ControlMultiplicity(policy, grid("", 0.44))
	if err != nil {
		t.Fatalf("fixed ControlMultiplicity: %v", err)
	}
	if !fixed.Pass {
		t.Fatalf("the same grid must clear once the decode defect is fixed; got %s", ExplainMultiplicity(fixed))
	}
}

// TestDiscoveryWithoutAScrubbedReplayArtifactIsNotActionable proves the fail-closed
// half of the replay requirement: a rejection whose evidence carries no scrubbed
// bundle still blocks, but is labelled inconclusive rather than presented as an
// actionable finding.
func TestDiscoveryWithoutAScrubbedReplayArtifactIsNotActionable(t *testing.T) {
	for name, mutate := range map[string]func(*MetricComparison){
		"no-bundle":    func(c *MetricComparison) { c.Evidence.Replay = nil },
		"not-scrubbed": func(c *MetricComparison) { c.Evidence.Replay = &FailureBundle{CaseID: "x", Scrubbed: false} },
	} {
		t.Run(name, func(t *testing.T) {
			fam := []MetricComparison{mcComparison("fixture-7b", "short-prompt", "exact_match", 1e-9)}
			mutate(&fam[0])
			d, err := ControlMultiplicity(mcPolicy(CorrectionHolm, mcPrimary()), fam)
			if err != nil {
				t.Fatalf("ControlMultiplicity: %v", err)
			}
			if d.Pass || len(d.Blocks) != 1 {
				t.Fatalf("an unreplayable discovery must block; got %s", ExplainMultiplicity(d))
			}
			if got := d.Blocks[0].State; got != StateInconclusive {
				t.Errorf("state = %q, want %q", got, StateInconclusive)
			}
			if !strings.Contains(d.Blocks[0].Reason, "cannot be independently reproduced") {
				t.Errorf("reason must name the missing artifact; got %q", d.Blocks[0].Reason)
			}
		})
	}
}

// TestSecondaryMetricsAreGatedBehindThePrimaryFamily proves the hierarchical half
// of the policy in both directions: with every primary metric clean the secondary
// family is not tested at all — even a p = 1e-9 secondary cell yields no discovery
// — and once a primary metric rejects, the gate opens and the secondary family is
// corrected as its own family.
func TestSecondaryMetricsAreGatedBehindThePrimaryFamily(t *testing.T) {
	policy := mcPolicy(CorrectionHolm, mcPrimary())
	build := func(primaryP float64) []MetricComparison {
		return []MetricComparison{
			mcComparison("fixture-7b", "short-prompt", "exact_match", primaryP),   // primary
			mcComparison("fixture-7b", "short-prompt", "grounding", 0.80),         // primary
			mcComparison("fixture-7b", "short-prompt", "latency_p95", 1e-9),       // secondary
			mcComparison("fixture-7b", "long-context", "citation_precision", 0.5), // secondary
		}
	}

	closed, err := ControlMultiplicity(policy, build(0.62))
	if err != nil {
		t.Fatalf("closed-gate ControlMultiplicity: %v", err)
	}
	if !closed.Pass {
		t.Fatalf("a clean primary family must pass with the gate closed; got %s", ExplainMultiplicity(closed))
	}
	if closed.Tested != 2 || closed.Gated != 2 {
		t.Errorf("tested/gated = %d/%d, want 2/2", closed.Tested, closed.Gated)
	}
	if g := closed.Decisions[2]; !g.Gated || g.Rejected {
		t.Errorf("the p=1e-9 secondary cell must be gated and unrejected; got %+v", g)
	}
	if !strings.Contains(closed.Decisions[2].Reason, "hierarchical gate stayed closed") {
		t.Errorf("gated cell must say why it was not tested; got %q", closed.Decisions[2].Reason)
	}
	if !strings.Contains(closed.Bound, "gated untested") {
		t.Errorf("bound must disclose the untested secondary family; got %q", closed.Bound)
	}

	open, err := ControlMultiplicity(policy, build(1e-6))
	if err != nil {
		t.Fatalf("open-gate ControlMultiplicity: %v", err)
	}
	if open.Pass {
		t.Fatalf("a rejecting primary family must block; got %s", ExplainMultiplicity(open))
	}
	if open.Tested != 4 || open.Gated != 0 {
		t.Errorf("tested/gated = %d/%d, want 4/0 once the gate opens", open.Tested, open.Gated)
	}
	if !open.Decisions[2].Rejected {
		t.Errorf("the secondary cell must be adjudicated once the gate opens; got %+v", open.Decisions[2])
	}
	// Blocks lead with the strongest evidence: the secondary cell's 2 * 1e-9 beats
	// the primary cell's 2 * 1e-6.
	if open.Blocks[0].Metric != "latency_p95" {
		t.Errorf("blocks must lead with the strongest adjusted p; got %q first", open.Blocks[0].Cell)
	}
	if !strings.Contains(open.Bound, "the hierarchical gate opened") {
		t.Errorf("bound must disclose that the secondary family was tested; got %q", open.Bound)
	}
}

// TestControlMultiplicityFailsClosed walks every way a family can fail to be
// evidence. Each case must refuse or block — none may pass.
func TestControlMultiplicityFailsClosed(t *testing.T) {
	good := func() MetricComparison { return mcComparison("fixture-7b", "short-prompt", "exact_match", 0.5) }

	t.Run("policy-and-family-errors", func(t *testing.T) {
		for name, tc := range map[string]struct {
			policy MultiplicityPolicy
			family []MetricComparison
			want   string
		}{
			"alpha-zero":        {MultiplicityPolicy{Alpha: 0, Correction: CorrectionHolm, Primary: mcPrimary()}, []MetricComparison{good()}, "alpha"},
			"alpha-one":         {MultiplicityPolicy{Alpha: 1, Correction: CorrectionHolm, Primary: mcPrimary()}, []MetricComparison{good()}, "alpha"},
			"alpha-nan":         {MultiplicityPolicy{Alpha: math.NaN(), Correction: CorrectionHolm, Primary: mcPrimary()}, []MetricComparison{good()}, "alpha"},
			"unknown-corrector": {MultiplicityPolicy{Alpha: mcAlpha, Correction: "bonferroni-ish", Primary: mcPrimary()}, []MetricComparison{good()}, "correction"},
			"no-primary":        {MultiplicityPolicy{Alpha: mcAlpha, Correction: CorrectionHolm}, []MetricComparison{good()}, "no primary metric declared"},
			"empty-primary":     {MultiplicityPolicy{Alpha: mcAlpha, Correction: CorrectionHolm, Primary: []string{" "}}, []MetricComparison{good()}, "non-empty"},
			"empty-family":      {mcPolicy(CorrectionHolm, mcPrimary()), nil, "family is empty"},
		} {
			_, err := ControlMultiplicity(tc.policy, tc.family)
			if err == nil {
				t.Errorf("%s: want an error, got none", name)
				continue
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("%s: error %q does not mention %q", name, err, tc.want)
			}
		}
	})

	t.Run("inadmissible-comparisons", func(t *testing.T) {
		for name, tc := range map[string]struct {
			mutate func(*MetricComparison)
			want   string
		}{
			"no-model":              {func(c *MetricComparison) { c.Model = "" }, "declares no model"},
			"no-slice":              {func(c *MetricComparison) { c.Slice = " " }, "declares no slice"},
			"no-metric":             {func(c *MetricComparison) { c.Metric = "" }, "declares no metric"},
			"no-tier":               {func(c *MetricComparison) { c.Tier = "" }, "not one of pr, nightly, release"},
			"unknown-tier":          {func(c *MetricComparison) { c.Tier = "weekly" }, "not one of pr, nightly, release"},
			"undocumented-cost":     {func(c *MetricComparison) { c.CostSeconds = 0 }, "runtime cost of the evidence must be documented"},
			"infinite-cost":         {func(c *MetricComparison) { c.CostSeconds = math.Inf(1) }, "runtime cost of the evidence must be documented"},
			"no-model-provenance":   {func(c *MetricComparison) { c.Evidence.Provenance.Model = "" }, "missing model"},
			"no-tokenizer":          {func(c *MetricComparison) { c.Evidence.Provenance.Tokenizer = "" }, "missing tokenizer"},
			"no-engine":             {func(c *MetricComparison) { c.Evidence.Provenance.Engine = "" }, "missing engine/backend"},
			"no-seed-or-oracle":     {func(c *MetricComparison) { c.Evidence.Provenance.Oracle = "" }, "missing seed or deterministic oracle"},
			"no-code-revision":      {func(c *MetricComparison) { c.Evidence.Provenance.Revision = "" }, "missing code/module revision"},
			"no-baseline":           {func(c *MetricComparison) { c.Evidence.Provenance.Baseline = "" }, "missing tolerance/baseline provenance"},
			"missing-evidence":      {func(c *MetricComparison) { c.Evidence.State = StateMissing }, "only produced pass/fail evidence"},
			"inconclusive-evidence": {func(c *MetricComparison) { c.Evidence.State = StateInconclusive }, "only produced pass/fail evidence"},
			"p-nan":                 {func(c *MetricComparison) { c.P = math.NaN() }, "not in [0, 1]"},
			"p-negative":            {func(c *MetricComparison) { c.P = -0.01 }, "not in [0, 1]"},
			"p-above-one":           {func(c *MetricComparison) { c.P = 1.5 }, "not in [0, 1]"},
		} {
			// A second, admissible primary cell keeps the family adjudicable so the
			// refusal under test is the one being measured.
			fam := []MetricComparison{good(), mcComparison("fixture-7b", "long-context", "grounding", 0.66)}
			tc.mutate(&fam[0])
			d, err := ControlMultiplicity(mcPolicy(CorrectionHolm, mcPrimary()), fam)
			if err != nil {
				t.Errorf("%s: unexpected error %v", name, err)
				continue
			}
			if d.Pass {
				t.Errorf("%s: an inadmissible comparison must block the grid; got %s", name, ExplainMultiplicity(d))
				continue
			}
			if got := d.Decisions[0].State; got != StateInconclusive {
				t.Errorf("%s: state = %q, want %q", name, got, StateInconclusive)
			}
			if !strings.Contains(d.Decisions[0].Reason, tc.want) {
				t.Errorf("%s: reason %q does not mention %q", name, d.Decisions[0].Reason, tc.want)
			}
		}
	})

	t.Run("no-admissible-primary", func(t *testing.T) {
		fam := []MetricComparison{mcComparison("fixture-7b", "short-prompt", "latency_p95", 1e-9)}
		d, err := ControlMultiplicity(mcPolicy(CorrectionHolm, mcPrimary()), fam)
		if err != nil {
			t.Fatalf("ControlMultiplicity: %v", err)
		}
		if d.Pass {
			t.Fatalf("a grid with no primary comparison must not pass; got %s", ExplainMultiplicity(d))
		}
		if len(d.Refusals) != 1 || !strings.Contains(d.Refusals[0], "no admissible comparison carries a declared primary metric") {
			t.Errorf("refusal must name the missing primary family; got %#v", d.Refusals)
		}
		if d.Decisions[0].Rejected {
			t.Error("a secondary cell must not be adjudicated when the primary family is absent")
		}
	})
}

// TestPerTierCostIsDocumented proves the cost rollup an operator reads to see what
// the grid costs per cadence, including the cells that turned out inconclusive —
// their evidence was paid for either way.
func TestPerTierCostIsDocumented(t *testing.T) {
	fam := []MetricComparison{
		mcComparison("fixture-7b", "short-prompt", "exact_match", 0.4),
		mcComparison("fixture-7b", "long-context", "grounding", 0.5),
		mcComparison("fixture-70b", "short-prompt", "latency_p95", 0.6),
	}
	fam[0].Tier, fam[0].CostSeconds = TierPR, 12
	fam[2].Tier, fam[2].CostSeconds = TierRelease, 900

	d, err := ControlMultiplicity(mcPolicy(CorrectionHolm, mcPrimary()), fam)
	if err != nil {
		t.Fatalf("ControlMultiplicity: %v", err)
	}
	want := []TierCost{
		{Tier: TierPR, Comparisons: 1, CostSeconds: 12},
		{Tier: TierNightly, Comparisons: 1, CostSeconds: 42},
		{Tier: TierRelease, Comparisons: 1, CostSeconds: 900},
	}
	if !reflect.DeepEqual(d.Cost, want) {
		t.Errorf("cost rollup = %+v, want %+v", d.Cost, want)
	}
	if out := ExplainMultiplicity(d); !strings.Contains(out, "cost pr") || !strings.Contains(out, "900.0s") {
		t.Errorf("readout must document per-tier cost; got:\n%s", out)
	}
}

// TestControlMultiplicityIsDeterministic proves the replay contract: the same
// policy and family produce an identical decision every time, including the order
// of the blocking list.
func TestControlMultiplicityIsDeterministic(t *testing.T) {
	fam := mcNullFamily(&splitMix64{state: 31337})
	fam[5].P, fam[11].P = 1e-8, 1e-8 // a tie the block ordering must break stably
	policy := mcPolicy(CorrectionHolm, mcMetrics())

	first, err := ControlMultiplicity(policy, fam)
	if err != nil {
		t.Fatalf("first ControlMultiplicity: %v", err)
	}
	second, err := ControlMultiplicity(policy, fam)
	if err != nil {
		t.Fatalf("second ControlMultiplicity: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("decisions differ across identical runs:\nfirst:  %s\nsecond: %s",
			ExplainMultiplicity(first), ExplainMultiplicity(second))
	}
	if first.Blocks[0].Index > first.Blocks[1].Index {
		t.Errorf("tied adjusted p-values must order by declared index; got %d before %d",
			first.Blocks[0].Index, first.Blocks[1].Index)
	}
}
