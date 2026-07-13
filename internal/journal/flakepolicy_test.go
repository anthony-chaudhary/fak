package journal

import "testing"

// naiveLastGreenWins is the PLANTED REPRESENTATIVE DEFECT (#4569 witness): the obvious
// "re-run a flaky case until it goes green, last attempt wins" retry loop. It is exactly
// the bug lm-evaluation-harness / HELM flaky-eval policy exists to prevent — it launders a
// real quality failure into a pass the moment a later sample flickers green. It lives only
// in the test, as the thing the shipped policy must diverge from.
func naiveLastGreenWins(attempts []string) string {
	last := QualityErrorNoVerdict
	for _, a := range attempts {
		switch a {
		case FlakePass:
			last = QualityPass
		case FlakeQualityFailure, FlakeInconclusive:
			last = QualityFail
		}
		// infra errors are skipped by both policies
	}
	return last
}

// TestRealFailureCannotBeRetriedIntoGreen is the captured proof for #4569 acceptance
// criterion 1. The flaky sequence [QUALITY_FAILURE, PASS] is a real failure that flickered
// green on a re-roll. The planted defect PASSES it; the shipped policy FAILS it — both the
// fold (sticky fail) and the admission kernel (refuse the retry at its source).
func TestRealFailureCannotBeRetriedIntoGreen(t *testing.T) {
	flaky := []string{FlakeQualityFailure, FlakePass}

	if got := naiveLastGreenWins(flaky); got != QualityPass {
		t.Fatalf("planted defect precondition: want the naive loop to PASS the flaky case, got %q", got)
	}
	if got := FoldAttempts(flaky); got != QualityFail {
		t.Fatalf("FoldAttempts laundered a real failure into %q; want FAIL (sticky)", got)
	}
	// Order must not matter: a real failure anywhere is sticky.
	if got := FoldAttempts([]string{FlakePass, FlakeQualityFailure}); got != QualityFail {
		t.Fatalf("FoldAttempts([pass, quality_failure]) = %q; want FAIL", got)
	}
	// The admission kernel refuses the retry at its source, so the green is never even sampled.
	if admit, reason := RetryAdmit(FlakeQualityFailure, 0, DefaultMaxInfraRetries); admit || reason != RetryRefuseQualityFailure {
		t.Fatalf("RetryAdmit(QUALITY_FAILURE) = (%v,%q); want (false,%q)", admit, reason, RetryRefuseQualityFailure)
	}
}

// TestInfraErrorIsRetriableWithinBudget covers the other half of the separation: an infra
// error carries no quality signal, so it IS retriable within budget, and a clean green after
// an infra flap legitimately passes (the real eval only ran once, to a pass).
func TestInfraErrorIsRetriableWithinBudget(t *testing.T) {
	if admit, reason := RetryAdmit(FlakeInfraError, 0, 3); !admit || reason != RetryAdmitInfraBudget {
		t.Fatalf("RetryAdmit(INFRA_ERROR, 0/3) = (%v,%q); want (true,%q)", admit, reason, RetryAdmitInfraBudget)
	}
	if admit, reason := RetryAdmit(FlakeInfraError, 3, 3); admit || reason != RetryRefuseBudgetExhausted {
		t.Fatalf("RetryAdmit(INFRA_ERROR, 3/3) = (%v,%q); want (false,%q)", admit, reason, RetryRefuseBudgetExhausted)
	}
	if got := FoldAttempts([]string{FlakeInfraError, FlakePass}); got != QualityPass {
		t.Fatalf("FoldAttempts([infra, pass]) = %q; want PASS (infra is transparent)", got)
	}
	// maxRetries<=0 falls back to the default budget rather than refusing immediately.
	if admit, _ := RetryAdmit(FlakeInfraError, 0, 0); !admit {
		t.Fatalf("RetryAdmit(INFRA_ERROR, 0, 0) should admit under the default budget")
	}
}

// TestMissingOrInconclusiveEvidenceIsNeverPass covers #4569 acceptance criterion 3's
// fail-closed clause: inconclusive → FAIL, and an all-infra / empty log → ERROR_NO_VERDICT.
// Neither is ever PASS.
func TestMissingOrInconclusiveEvidenceIsNeverPass(t *testing.T) {
	cases := []struct {
		name     string
		attempts []string
		want     string
	}{
		{"inconclusive alone", []string{FlakeInconclusive}, QualityFail},
		{"inconclusive dominates a pass", []string{FlakePass, FlakeInconclusive}, QualityFail},
		{"unknown class is fail-closed", []string{"MYSTERY"}, QualityFail},
		{"all infra, no verdict", []string{FlakeInfraError, FlakeInfraError}, QualityErrorNoVerdict},
		{"empty log", nil, QualityErrorNoVerdict},
	}
	for _, tc := range cases {
		if got := FoldAttempts(tc.attempts); got != tc.want {
			t.Errorf("%s: FoldAttempts(%v) = %q; want %q", tc.name, tc.attempts, got, tc.want)
		}
		if got := FoldAttempts(tc.attempts); got == QualityPass {
			t.Errorf("%s: inconclusive/missing evidence was reported PASS", tc.name)
		}
	}
	// Unknown class also fails the admission kernel closed.
	if admit, reason := RetryAdmit("MYSTERY", 0, 3); admit || reason != RetryRefuseUnclassified {
		t.Fatalf("RetryAdmit(unknown) = (%v,%q); want (false,%q)", admit, reason, RetryRefuseUnclassified)
	}
}

// TestProvenanceCompletenessRequiredForPass covers #4569 acceptance criterion 2: a green with
// incomplete provenance is unreproducible, so Decide demotes it to ERROR_NO_VERDICT and names
// the missing axis. A complete case with a real pass stands as PASS.
func TestProvenanceCompletenessRequiredForPass(t *testing.T) {
	full := QualityCaseProvenance{
		Model:     "acme-7b@abc123",
		Tokenizer: "acme-bpe@v3",
		Engine:    "cpu-q8",
		Seed:      "42",
		Revision:  "deadbeef",
		Tolerance: "baseline@nightly-2026-07-01 ±0.5%",
	}
	if ok, missing := full.Complete(); !ok {
		t.Fatalf("full provenance reported incomplete, missing %q", missing)
	}
	// A deterministic oracle satisfies the seed-or-oracle axis in place of Seed.
	oracleProv := full
	oracleProv.Seed = ""
	oracleProv.Oracle = "exact-match-oracle@v1"
	if ok, missing := oracleProv.Complete(); !ok {
		t.Fatalf("oracle in place of seed reported incomplete, missing %q", missing)
	}

	// Each required axis, when blank, is named as the first missing field.
	for _, tc := range []struct {
		field   string
		mutate  func(*QualityCaseProvenance)
		missing string
	}{
		{"model", func(p *QualityCaseProvenance) { p.Model = "" }, "model"},
		{"tokenizer", func(p *QualityCaseProvenance) { p.Tokenizer = "" }, "tokenizer"},
		{"engine", func(p *QualityCaseProvenance) { p.Engine = "" }, "engine"},
		{"seed_or_oracle", func(p *QualityCaseProvenance) { p.Seed = ""; p.Oracle = "" }, "seed_or_oracle"},
		{"revision", func(p *QualityCaseProvenance) { p.Revision = "" }, "revision"},
		{"tolerance", func(p *QualityCaseProvenance) { p.Tolerance = "" }, "tolerance"},
	} {
		p := full
		tc.mutate(&p)
		ok, missing := p.Complete()
		if ok || missing != tc.missing {
			t.Errorf("missing %s: Complete() = (%v,%q); want (false,%q)", tc.field, ok, missing, tc.missing)
		}
	}

	// A folded PASS with incomplete provenance is demoted, not passed.
	incomplete := &QualityQuarantineRow{Provenance: full, Attempts: []string{FlakePass}}
	incomplete.Provenance.Revision = ""
	incomplete.Decide()
	if incomplete.Verdict != QualityErrorNoVerdict {
		t.Fatalf("pass with incomplete provenance = %q; want ERROR_NO_VERDICT", incomplete.Verdict)
	}
	if incomplete.FirstDivergence != "incomplete_provenance:revision" {
		t.Fatalf("first divergence = %q; want incomplete_provenance:revision", incomplete.FirstDivergence)
	}
	// A complete case with a real pass stands.
	good := &QualityQuarantineRow{Provenance: full, Attempts: []string{FlakePass}}
	good.Decide()
	if good.Verdict != QualityPass {
		t.Fatalf("complete pass = %q; want PASS", good.Verdict)
	}
}

// TestTierAndCostRecorded covers #4569 acceptance criterion 4: the tier is a closed set and
// the durable row carries tier + runtime/resource cost.
func TestTierAndCostRecorded(t *testing.T) {
	for _, tier := range []string{QualityTierPR, QualityTierNightly, QualityTierRelease} {
		if !ValidTier(tier) {
			t.Errorf("ValidTier(%q) = false; want true", tier)
		}
	}
	for _, bad := range []string{"", "prod", "hourly", "PR"} {
		if ValidTier(bad) {
			t.Errorf("ValidTier(%q) = true; want false", bad)
		}
	}
}

// TestReplayArtifactScrubbed covers #4569 acceptance criterion 3's scrub clause: a
// secret-bearing replay handle is dropped rather than persisted; a clean handle is kept.
func TestReplayArtifactScrubbed(t *testing.T) {
	if got := scrubReplay("replay://s3/case-42/bearer-secret-token"); got != "" {
		t.Fatalf("secret-bearing replay handle not scrubbed: got %q", got)
	}
	if got := scrubReplay("  replay://blobstore/case-42.jsonl  "); got != "replay://blobstore/case-42.jsonl" {
		t.Fatalf("clean replay handle mangled: got %q", got)
	}
}

// TestQualityQuarantineRowChainsAndVerifies is the durable round-trip: a quarantine decision
// appended through the chain folds to the retry-safe verdict, scrubs the replay handle, carries
// the full correlated record, and leaves the hash chain intact (VerifyRows passes).
func TestQualityQuarantineRowChainsAndVerifies(t *testing.T) {
	j := OpenMemory()
	prov := QualityCaseProvenance{
		Model:     "acme-7b@abc123",
		Tokenizer: "acme-bpe@v3",
		Engine:    "cuda-fp16",
		Oracle:    "exact-match@v1",
		Revision:  "deadbeef",
		Tolerance: "baseline@nightly ±0.5%",
	}
	row := j.AppendQualityQuarantine(QualityQuarantineRow{
		Case:            "acme.reasoning.gsm8k-subset",
		Owner:           "quality-oncall",
		Tier:            QualityTierNightly,
		CostMS:          42000,
		Provenance:      prov,
		Attempts:        []string{FlakeInfraError, FlakeQualityFailure, FlakePass}, // a real failure hid behind a retry-green
		FirstDivergence: "sample#3 expected 42, got 41.5 (>tolerance)",
		ReplayArtifact:  "replay://blobstore/gsm8k-subset.jsonl",
	})

	if row.Kind != KindQualityQuarantine {
		t.Fatalf("row.Kind = %q; want %q", row.Kind, KindQualityQuarantine)
	}
	if row.Verdict != QualityFail {
		t.Fatalf("row.Verdict = %q; want FAIL (the retry-green must not launder the real failure)", row.Verdict)
	}
	if row.Quality == nil {
		t.Fatal("row.Quality carrier is nil")
	}
	if row.Quality.Schema != QualityQuarantineSchema {
		t.Fatalf("carrier schema = %q; want %q", row.Quality.Schema, QualityQuarantineSchema)
	}
	if row.Quality.Tier != QualityTierNightly || row.Quality.CostMS != 42000 {
		t.Fatalf("carrier tier/cost not preserved: tier=%q cost=%d", row.Quality.Tier, row.Quality.CostMS)
	}
	if row.By != "quality-policy" {
		t.Fatalf("row.By = %q; want quality-policy", row.By)
	}

	if _, err := VerifyRows(j.Recent(16)); err != nil {
		t.Fatalf("chain broke after AppendQualityQuarantine: %v", err)
	}

	// A nil receiver is a no-op returning the zero Row, so a guarded-off caller may call it.
	var nilJ *Journal
	if got := nilJ.AppendQualityQuarantine(QualityQuarantineRow{Case: "x"}); got != (Row{}) {
		t.Fatalf("nil-journal append returned %+v; want zero Row", got)
	}
}
