package quality

import (
	"encoding/json"
	"strings"
	"testing"
)

// failing wraps a synthetic bundle as the failing Result an operator would hold,
// so every case below exercises the exported Classify path rather than the
// unexported ladder underneath it.
func failing(fb FailureBundle) Result {
	return Result{Schema: ResultSchema, CaseID: fb.CaseID, Pass: false, FailureBundle: &fb}
}

func greedyCase(flags map[string]string) QualityCase {
	c := DemoCase()
	c.Params = SamplingParams{Temperature: 0, MaxTokens: 8}
	c.Metadata.Engine = EngineSpec{Name: "fak", Backend: "cpu", Flags: flags}
	return c
}

// TestClassifySyntheticBundlesLocalizeEachStage is the #4520 acceptance check:
// a bundle assembled by hand, carrying only the evidence a real run would carry,
// localizes to the stage that evidence actually implies. The bundles are
// synthetic on purpose — the classifier reads evidence, never oracle names, so
// nothing here depends on which oracle happened to produce the failure.
func TestClassifySyntheticBundlesLocalizeEachStage(t *testing.T) {
	cases := []struct {
		name   string
		bundle FailureBundle
		want   string
	}{{
		name: "rubric scorer failed, streams never contradicted",
		bundle: FailureBundle{
			CaseID: "rubric", FailingOracle: "grounding-rubric", FailingKind: "rubric",
			Reference: Trace{Tokens: []string{"a"}, Text: "a"},
			Engine:    Trace{Tokens: []string{"a"}, Text: "a"},
		},
		want: "rubric",
	}, {
		name: "engine trace arrived with no tokens at all",
		bundle: FailureBundle{
			CaseID: "transport", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Reference:       Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			Engine:          Trace{},
			FirstDivergence: &Divergence{Index: 0, Reference: "a", Engine: "<end>"},
		},
		want: "transport",
	}, {
		name: "tokens arrived but the assembled body did not",
		bundle: FailureBundle{
			CaseID: "transport-body", FailingOracle: "exact-match", FailingKind: "exact",
			Reference: Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			Engine:    Trace{Tokens: []string{"a", "b"}, Text: ""},
		},
		want: "transport",
	}, {
		name: "identical text segmented two different ways",
		bundle: FailureBundle{
			CaseID: "segmentation", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Reference:       Trace{Tokens: []string{"ab"}, Text: "ab"},
			Engine:          Trace{Tokens: []string{"a", "b"}, Text: "ab"},
			FirstDivergence: &Divergence{Index: 0, Reference: "ab", Engine: "a"},
		},
		want: "tokenization",
	}, {
		name: "identical tokens assembled into different text",
		bundle: FailureBundle{
			CaseID: "detokenize", FailingOracle: "exact-match", FailingKind: "exact",
			Reference: Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			Engine:    Trace{Tokens: []string{"a", "b"}, Text: "ab"},
		},
		want: "tokenization",
	}, {
		name: "identical tokens rendered with different case",
		bundle: FailureBundle{
			CaseID: "case-fold", FailingOracle: "exact-match", FailingKind: "exact",
			Reference: Trace{Tokens: []string{"Throughput", "rose"}, Text: "Throughput rose."},
			Engine:    Trace{Tokens: []string{"Throughput", "rose"}, Text: "throughput  rose."},
		},
		want: "normalization",
	}, {
		name: "divergent tokens differ only in case",
		bundle: FailureBundle{
			CaseID: "token-case", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case:            greedyCase(nil),
			Reference:       Trace{Tokens: []string{"a", "Rose"}, Text: "a Rose"},
			Engine:          Trace{Tokens: []string{"a", "rose"}, Text: "a rose"},
			FirstDivergence: &Divergence{Index: 1, Reference: "Rose", Engine: "rose"},
		},
		want: "normalization",
	}, {
		name: "every logit displaced by one constant: the log-softmax never ran",
		bundle: FailureBundle{
			CaseID: "raw-logits", FailingOracle: "logprob-parity", FailingKind: "differential",
			Case:            greedyCase(nil),
			Reference:       Trace{Tokens: []string{"a", "b"}, Text: "a b", Logits: [][]float64{{-1, -2, -3}, {-1, -4, -5}}},
			Engine:          Trace{Tokens: []string{"a", "b"}, Text: "a b", Logits: [][]float64{{0, -1, -2}, {0, -3, -4}}},
			FirstDivergence: &Divergence{Index: 0, Reference: "a", Engine: "a"},
		},
		want: "normalization",
	}, {
		name: "the distribution itself diverged before selection",
		bundle: FailureBundle{
			CaseID: "logits", FailingOracle: "logit-tolerance", FailingKind: "differential",
			Case:            greedyCase(nil),
			Reference:       Trace{Tokens: []string{"a", "b"}, Text: "a b", Logits: [][]float64{{-1, -2}, {-3, -4}}},
			Engine:          Trace{Tokens: []string{"a", "c"}, Text: "a c", Logits: [][]float64{{-1, -2}, {-3, -9}}},
			FirstDivergence: &Divergence{Index: 1, Reference: "b", Engine: "c"},
		},
		want: "logits",
	}, {
		name: "shared tokens all agree and the engine kept going",
		bundle: FailureBundle{
			CaseID: "overrun", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case:            greedyCase(nil),
			Reference:       Trace{Tokens: []string{"a", "b", "."}, Text: "a b."},
			Engine:          Trace{Tokens: []string{"a", "b", ".", "and", "more"}, Text: "a b. and more"},
			FirstDivergence: &Divergence{Index: 3, Reference: "<end>", Engine: "and"},
		},
		want: "stops",
	}, {
		name: "shared tokens all agree and the engine quit early",
		bundle: FailureBundle{
			CaseID: "truncated", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case:            greedyCase(nil),
			Reference:       Trace{Tokens: []string{"a", "b", "."}, Text: "a b."},
			Engine:          Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			FirstDivergence: &Divergence{Index: 2, Reference: ".", Engine: "<end>"},
		},
		want: "stops",
	}, {
		name: "a stochastic case may legally draw a different token",
		bundle: FailureBundle{
			CaseID: "sampling", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case: func() QualityCase {
				c := greedyCase(nil)
				c.Params = SamplingParams{Temperature: 0.7, TopP: 0.95, MaxTokens: 8, Seed: 11}
				return c
			}(),
			Reference:       Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			Engine:          Trace{Tokens: []string{"a", "z"}, Text: "a z"},
			FirstDivergence: &Divergence{Index: 1, Reference: "b", Engine: "z"},
		},
		want: "sampling",
	}, {
		name: "a greedy case parted mid-stream with reuse enabled",
		bundle: FailureBundle{
			CaseID: "reuse", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case:            greedyCase(map[string]string{"prefix_cache": "on", "chunked_prefill": "off"}),
			Reference:       Trace{Tokens: []string{"a", "b", "c"}, Text: "a b c"},
			Engine:          Trace{Tokens: []string{"a", "b", "z"}, Text: "a b z"},
			FirstDivergence: &Divergence{Index: 2, Reference: "c", Engine: "z"},
		},
		want: "cache",
	}}

	seen := map[string]bool{}
	for _, tc := range cases {
		got := Classify(failing(tc.bundle))
		if got.Stage != tc.want {
			t.Errorf("%s: stage = %q, want %q (reason: %s)", tc.name, got.Stage, tc.want, got.Reason)
			continue
		}
		if got.Abstained() {
			t.Errorf("%s: a named stage must not report as abstained", tc.name)
		}
		if strings.TrimSpace(got.Reason) == "" {
			t.Errorf("%s: stage %q shipped with no reason: an unexplained attribution is not evidence", tc.name, got.Stage)
		}
		seen[got.Stage] = true
	}

	// Completeness: the vocabulary #4520 fixes is closed, so every member of it
	// must be reachable from evidence. A stage no bundle can produce is a stage
	// the classifier only claims to support.
	for _, stage := range Stages {
		if !seen[stage] {
			t.Errorf("stage %q is in the vocabulary but no synthetic bundle reaches it", stage)
		}
	}
	for stage := range seen {
		if !containsStage(stage) {
			t.Errorf("classifier emitted %q, which is outside the closed vocabulary %v", stage, Stages)
		}
	}
}

func containsStage(s string) bool {
	for _, v := range Stages {
		if v == s {
			return true
		}
	}
	return false
}

// TestClassifyAbstainsOnUnknownEvidence is the other half of the acceptance
// criterion: evidence that supports no signature must produce an explicit
// abstention, never a plausible-sounding guess.
func TestClassifyAbstainsOnUnknownEvidence(t *testing.T) {
	cases := []struct {
		name   string
		result Result
	}{{
		name: "a deterministic substitution with no logits and no declared flags",
		result: failing(FailureBundle{
			CaseID: "bare", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case:            greedyCase(nil),
			Reference:       Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			Engine:          Trace{Tokens: []string{"a", "z"}, Text: "a z"},
			FirstDivergence: &Divergence{Index: 1, Reference: "b", Engine: "z"},
		}),
	}, {
		name: "a reuse flag that is declared but switched off implicates nothing",
		result: failing(FailureBundle{
			CaseID: "reuse-off", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case:            greedyCase(map[string]string{"prefix_cache": "off"}),
			Reference:       Trace{Tokens: []string{"a", "b", "c"}, Text: "a b c"},
			Engine:          Trace{Tokens: []string{"a", "b", "z"}, Text: "a b z"},
			FirstDivergence: &Divergence{Index: 2, Reference: "c", Engine: "z"},
		}),
	}, {
		name: "reuse cannot explain a divergence at the very first step",
		result: failing(FailureBundle{
			CaseID: "reuse-step-zero", FailingOracle: "greedy-token-diff", FailingKind: "differential",
			Case:            greedyCase(map[string]string{"prefix_cache": "on"}),
			Reference:       Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			Engine:          Trace{Tokens: []string{"z", "b"}, Text: "z b"},
			FirstDivergence: &Divergence{Index: 0, Reference: "a", Engine: "z"},
		}),
	}, {
		name: "a rate-based verdict pins no step to attribute",
		result: failing(FailureBundle{
			CaseID: "rate", FailingOracle: "statistical-agreement", FailingKind: "statistical",
			Case:      greedyCase(nil),
			Reference: Trace{Tokens: []string{"a", "b"}, Text: "a b"},
			Engine:    Trace{Tokens: []string{"a", "z"}, Text: "a z"},
		}),
	}, {
		name:   "a failing result with no bundle carries no evidence at all",
		result: Result{Schema: ResultSchema, CaseID: "no-bundle", Pass: false},
	}, {
		name:   "a passing result has no first divergence to localize",
		result: Result{Schema: ResultSchema, CaseID: "clean", Pass: true},
	}}

	for _, tc := range cases {
		got := Classify(tc.result)
		if !got.Abstained() {
			t.Errorf("%s: expected an abstention, got stage %q (%s)", tc.name, got.Stage, got.Reason)
		}
		if got.Stage != StageAbstain {
			t.Errorf("%s: abstention must be spelled %q, got %q", tc.name, StageAbstain, got.Stage)
		}
		if strings.TrimSpace(got.Reason) == "" {
			t.Errorf("%s: an abstention with no reason tells an operator nothing", tc.name)
		}
	}
}

// TestAbstainNeverSoftensTheVerdict pins the rule that makes an abstention safe
// to ship: inconclusive evidence localizes nothing, and changes nothing about
// whether the run failed. Missing evidence is never a pass.
func TestAbstainNeverSoftensTheVerdict(t *testing.T) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("decode"), oracles)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("the decode defect must fail the run; got %s", Explain(res))
	}
	got := Classify(res)
	if !got.Abstained() {
		t.Fatalf("a bare greedy substitution carries no layer evidence, so it must abstain; got stage %q", got.Stage)
	}
	out := Explain(res)
	for _, want := range []string{"FAIL", "unclassified", "(abstained)"} {
		if !strings.Contains(out, want) {
			t.Errorf("explain output must contain %q so an abstention reads as loud, not as clean:\n%s", want, out)
		}
	}
}

// TestRunCaseStampsTheStageOnTheBundle checks the machine-readable half: the
// emitted artifact carries the localization, it survives a JSON round trip, and
// it agrees with recomputing it from the bundle. The planted defect is the stop
// mutant, which localizes to a named stage rather than abstaining.
func TestRunCaseStampsTheStageOnTheBundle(t *testing.T) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("stop"), oracles)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("the planted stop defect must fail the run; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil || fb.Classification == nil {
		t.Fatalf("a failing run must emit a bundle carrying its localization; got %+v", fb)
	}
	if fb.Classification.Stage != "stops" {
		t.Fatalf("a stream that decoded past the reference localizes to stops; got %q (%s)",
			fb.Classification.Stage, fb.Classification.Reason)
	}
	if again := Classify(res); again != *fb.Classification {
		t.Fatalf("stamped localization %+v disagrees with recomputing it: %+v", *fb.Classification, again)
	}

	blob, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	var round Result
	if err := json.Unmarshal(blob, &round); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if round.FailureBundle == nil || round.FailureBundle.Classification == nil {
		t.Fatalf("the localization must survive the artifact round trip: %s", blob)
	}
	if *round.FailureBundle.Classification != *fb.Classification {
		t.Fatalf("round-tripped localization %+v != %+v", *round.FailureBundle.Classification, *fb.Classification)
	}
	if !strings.Contains(Explain(round), "stage: stops") {
		t.Fatalf("explain of a reloaded result must name the stage:\n%s", Explain(round))
	}
}

// TestClassifyIsStableAcrossRuns guards the one nondeterminism this classifier
// could plausibly acquire: engine flags live in a map, and Go randomizes map
// order, so a reason string naming whichever flag came first would not replay.
func TestClassifyIsStableAcrossRuns(t *testing.T) {
	fb := FailureBundle{
		CaseID: "stable", FailingOracle: "greedy-token-diff", FailingKind: "differential",
		Case: greedyCase(map[string]string{
			"prefix_cache": "on", "kv_cache_dtype": "fp8", "zz_cache": "on", "aa_cache": "on",
		}),
		Reference:       Trace{Tokens: []string{"a", "b", "c"}, Text: "a b c"},
		Engine:          Trace{Tokens: []string{"a", "b", "z"}, Text: "a b z"},
		FirstDivergence: &Divergence{Index: 2, Reference: "c", Engine: "z"},
	}
	first := Classify(failing(fb))
	for i := 0; i < 64; i++ {
		if got := Classify(failing(fb)); got != first {
			t.Fatalf("classification is not replayable: run %d gave %+v, want %+v", i, got, first)
		}
	}
	if !strings.Contains(first.Reason, "aa_cache") {
		t.Fatalf("the reason must name the lowest-sorted matching flag so it replays; got %s", first.Reason)
	}
}

// TestCleanRunLocalizesNothing pins that the localizer stays silent on a green
// run: there is no stage line to read, and nothing that could be mistaken for a
// finding.
func TestCleanRunLocalizesNothing(t *testing.T) {
	c := DemoCase()
	oracles, err := Lookup(c.Oracles)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(""), oracles)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("the clean engine must pass; got %s", Explain(res))
	}
	if out := Explain(res); strings.Contains(out, "stage:") {
		t.Fatalf("a passing run must not print a stage line:\n%s", out)
	}
	if got := Classify(res); !got.Abstained() {
		t.Fatalf("a passing run localizes nothing; got stage %q", got.Stage)
	}
}
