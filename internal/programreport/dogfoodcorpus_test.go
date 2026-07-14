package programreport

import (
	"strings"
	"testing"
)

// reseal recomputes a corpus digest after a test mutation so the mutation under
// test (not a stale digest) is what the gate sees.
func reseal(c DogfoodCorpus) DogfoodCorpus {
	c.Digest = DogfoodCorpusDigest(c)
	return c
}

// TestSeedDogfoodCorpusDefectCasesFailBeforeFix pins the #4550 acceptance
// criteria on the shipped seed corpus: it validates (versioned, calibrated,
// replay-complete), every suspected defect is a failing-before-fix case that
// fails on its declared axis and emits a scrubbed replay artifact, every
// expected-pass case passes, and the case set spans all three run tiers with a
// documented cost on each case.
func TestSeedDogfoodCorpusDefectCasesFailBeforeFix(t *testing.T) {
	c := SeedDogfoodCorpus()
	if err := c.Validate(); err != nil {
		t.Fatalf("seed corpus must validate: %v", err)
	}

	res := GradeDogfoodCorpus(c)
	if !res.Pass {
		t.Fatalf("seed corpus grade must pass (every expectation holding): %+v", res)
	}
	if code, _ := CheckDogfoodGate(res); code != 0 {
		t.Fatalf("gate on seed corpus = %d, want 0", code)
	}

	tiers := map[ReviewTier]bool{}
	fails, passes := 0, 0
	for i, dc := range c.Cases {
		tiers[dc.Review.Tier] = true
		if strings.TrimSpace(dc.Review.CostNote) == "" {
			t.Fatalf("case %q has no documented cost", dc.Review.ID)
		}
		row := res.Cases[i]
		switch dc.Expect {
		case ExpectFail:
			fails++
			if !row.Holding || row.Diverged {
				t.Fatalf("suspected defect %q must be holding (failing before fix): %+v", dc.Review.ID, row)
			}
			if row.Verdict.Pass {
				t.Fatalf("suspected defect %q must not pass before fix", dc.Review.ID)
			}
			if row.Verdict.FirstDivergence != dc.ExpectDivergence {
				t.Fatalf("case %q first divergence %q, want suspected axis %q", dc.Review.ID, row.Verdict.FirstDivergence, dc.ExpectDivergence)
			}
			if row.Verdict.Replay == nil || row.Verdict.Replay.Schema != ReplaySchema {
				t.Fatalf("failing case %q must emit a scrubbed replay artifact: %+v", dc.Review.ID, row.Verdict.Replay)
			}
		case ExpectPass:
			passes++
			if !row.Verdict.Pass || row.Diverged {
				t.Fatalf("expected-pass case %q must pass: %+v", dc.Review.ID, row)
			}
		}
	}
	if fails == 0 || passes == 0 {
		t.Fatalf("seed corpus must carry both defect and adequate cases, got %d fail / %d pass", fails, passes)
	}
	for _, tier := range []ReviewTier{TierPR, TierNightly, TierRelease} {
		if !tiers[tier] {
			t.Fatalf("seed corpus assigns no case to tier %q", tier)
		}
	}
}

// TestDogfoodWitnessPlantedDefectFailsThenFixPasses is the #4550 witness: a
// corpus carrying a planted representative defect (an unsupported speedup
// claim) FAILS against it — first actionable divergence on grounding, scrubbed
// replay emitted — and after the fix (the summary re-captured citing its
// measured signal, re-rated blind, corpus revision bumped) the same grading
// path PASSES. Both grades run on a corpus round-tripped through its serialized
// envelope, so the verdicts are reproduced in a clean, independently replayed
// environment (nothing outside the artifact is consulted).
func TestDogfoodWitnessPlantedDefectFailsThenFixPasses(t *testing.T) {
	replayGrade := func(c DogfoodCorpus) DogfoodCorpusResult {
		blob, err := MarshalDogfoodCorpus(c)
		if err != nil {
			t.Fatalf("marshal corpus: %v", err)
		}
		rt, err := UnmarshalDogfoodCorpus(blob)
		if err != nil {
			t.Fatalf("independent replay must accept the serialized corpus: %v", err)
		}
		return GradeDogfoodCorpus(rt)
	}

	before := SeedDogfoodCorpus()
	res := replayGrade(before)
	if !res.Pass {
		t.Fatalf("before-fix corpus must grade clean (defects holding): %+v", res)
	}
	var planted DogfoodCaseResult
	for _, row := range res.Cases {
		if row.CaseID == "DOG-grounding-unsupported-speedup" {
			planted = row
		}
	}
	if !planted.Holding || planted.Verdict.Pass {
		t.Fatalf("planted defect must fail before fix: %+v", planted)
	}
	if planted.Verdict.FirstDivergence != Grounding || planted.Verdict.Replay == nil {
		t.Fatalf("planted defect must diverge first on grounding with a replay artifact: %+v", planted.Verdict)
	}

	// The fix: the defective rollup is regenerated so the claim traces to a
	// measured signal, re-rated blind, and the case's expectation is flipped at
	// a new corpus revision — the recorded lifecycle of a defect case.
	after := before
	after.Revision = "2026-07-13.2"
	after.Cases = append([]DogfoodCase(nil), before.Cases...)
	for i, dc := range after.Cases {
		if dc.Review.ID != "DOG-grounding-unsupported-speedup" {
			continue
		}
		dc.Review.Subject = "Kernel window active (3 ships/7d, activity proxy — no tok/s claim); cache reuse 0.62 against the trend gate. No operator action needed. Next: hold ratchets."
		dc.Review.Provenance.Revision = "f7f1ec71d+fix"
		dc.Review.Raters = []RaterScores{
			{Rater: "rater-a", Scores: map[ReviewDimension]int{Grounding: 4, Completeness: 4, Salience: 4, Actionability: 4, Clarity: 4, Calibration: 4}},
			{Rater: "rater-b", Scores: map[ReviewDimension]int{Grounding: 5, Completeness: 4, Salience: 4, Actionability: 4, Clarity: 4, Calibration: 4}},
		}
		dc.Expect = ExpectPass
		dc.ExpectDivergence = ""
		dc.DefectRef = ""
		after.Cases[i] = dc
	}
	after = reseal(after)

	fixed := replayGrade(after)
	if !fixed.Pass {
		t.Fatalf("after-fix corpus must pass in the independent replay: %+v", fixed)
	}
	for _, row := range fixed.Cases {
		if row.CaseID == "DOG-grounding-unsupported-speedup" && (!row.Verdict.Pass || row.Verdict.Replay != nil) {
			t.Fatalf("fixed case must pass with no replay artifact: %+v", row)
		}
	}
}

// TestDogfoodDefectCasePassingBeforeFixIsNeverSilentlyGreen proves the
// failing-before-fix ratchet: if a suspected defect's recorded ratings start
// passing while the corpus still expects a failure, the corpus itself FAILS
// with an expectation replay — a before-fix case cannot rot into silent green.
func TestDogfoodDefectCasePassingBeforeFixIsNeverSilentlyGreen(t *testing.T) {
	c := SeedDogfoodCorpus()
	c.Cases = append([]DogfoodCase(nil), c.Cases...)
	for i, dc := range c.Cases {
		if dc.Review.ID != "DOG-grounding-unsupported-speedup" {
			continue
		}
		dc.Review.Raters = []RaterScores{ // ratings drift to adequate; expectation still "fail"
			{Rater: "rater-a", Scores: map[ReviewDimension]int{Grounding: 4, Completeness: 4, Salience: 4, Actionability: 4, Clarity: 4, Calibration: 4}},
			{Rater: "rater-b", Scores: map[ReviewDimension]int{Grounding: 4, Completeness: 4, Salience: 4, Actionability: 4, Clarity: 4, Calibration: 4}},
		}
		c.Cases[i] = dc
	}
	c = reseal(c)

	res := GradeDogfoodCorpus(c)
	if res.Pass {
		t.Fatalf("a passing before-fix defect case must fail the corpus: %+v", res)
	}
	if res.FirstDivergence != "DOG-grounding-unsupported-speedup" {
		t.Fatalf("first divergence = %q, want the drifted defect case", res.FirstDivergence)
	}
	if res.Replay == nil || res.Replay.Kind != "expectation" || res.Replay.Dimension != Grounding {
		t.Fatalf("expectation divergence must carry a replay naming the suspected axis: %+v", res.Replay)
	}
	if code, _ := CheckDogfoodGate(res); code != 1 {
		t.Fatalf("gate = %d, want 1", code)
	}

	// The dual drift: the defect still fails but on a DIFFERENT first axis —
	// the case no longer tracks its suspected defect, so the corpus fails too.
	c2 := SeedDogfoodCorpus()
	c2.Cases = append([]DogfoodCase(nil), c2.Cases...)
	for i, dc := range c2.Cases {
		if dc.Review.ID != "DOG-grounding-unsupported-speedup" {
			continue
		}
		dc.Review.Raters = []RaterScores{ // grounding now adequate, completeness at the floor
			{Rater: "rater-a", Scores: map[ReviewDimension]int{Grounding: 4, Completeness: 1, Salience: 4, Actionability: 4, Clarity: 4, Calibration: 4}},
			{Rater: "rater-b", Scores: map[ReviewDimension]int{Grounding: 4, Completeness: 1, Salience: 4, Actionability: 4, Clarity: 4, Calibration: 4}},
		}
		c2.Cases[i] = dc
	}
	c2 = reseal(c2)
	res2 := GradeDogfoodCorpus(c2)
	if res2.Pass || res2.FirstDivergence != "DOG-grounding-unsupported-speedup" {
		t.Fatalf("attribution drift must fail the corpus at the drifted case: %+v", res2)
	}
	if !strings.Contains(res2.Reason, "moved from") {
		t.Fatalf("attribution drift reason should name the move, got %q", res2.Reason)
	}
}

// TestDogfoodCorpusFailsClosed covers the admission boundaries: an ill-formed
// corpus is refused — never graded, never a pass.
func TestDogfoodCorpusFailsClosed(t *testing.T) {
	mut := func(name string, f func(*DogfoodCorpus), wantErr string) {
		c := SeedDogfoodCorpus()
		c.Cases = append([]DogfoodCase(nil), c.Cases...)
		f(&c)
		err := c.Validate()
		if err == nil {
			t.Fatalf("%s: expected admission refusal", name)
		}
		if wantErr != "" && !strings.Contains(err.Error(), wantErr) {
			t.Fatalf("%s: error %q should mention %q", name, err, wantErr)
		}
		if res := GradeDogfoodCorpus(c); res.Pass {
			t.Fatalf("%s: refused corpus must never grade pass", name)
		} else if code, _ := CheckDogfoodGate(res); code != 1 {
			t.Fatalf("%s: gate on refused corpus must be 1", name)
		}
	}

	mut("wrong schema", func(c *DogfoodCorpus) { c.Schema = "bogus/9"; *c = reseal(*c) }, "schema")
	mut("empty revision", func(c *DogfoodCorpus) { c.Revision = " "; *c = reseal(*c) }, "revision")
	mut("no cases", func(c *DogfoodCorpus) { c.Cases = nil; *c = reseal(*c) }, "empty")
	mut("tampered content", func(c *DogfoodCorpus) { c.Cases[0].Review.Subject += " tampered" }, "digest")
	mut("duplicate id", func(c *DogfoodCorpus) { c.Cases[1] = c.Cases[0]; *c = reseal(*c) }, "duplicate")
	mut("expected-fail without axis", func(c *DogfoodCorpus) { c.Cases[0].ExpectDivergence = ""; *c = reseal(*c) }, "suspected axis")
	mut("expected-fail without defect ref", func(c *DogfoodCorpus) { c.Cases[0].DefectRef = ""; *c = reseal(*c) }, "suspected defect")
	mut("unknown expectation", func(c *DogfoodCorpus) { c.Cases[0].Expect = "maybe"; *c = reseal(*c) }, "expect")
	mut("expected-pass with axis", func(c *DogfoodCorpus) { c.Cases[3].ExpectDivergence = Grounding; *c = reseal(*c) }, "must not declare")
	mut("state without evidence", func(c *DogfoodCorpus) { c.Cases[0].State.Evidence = nil; *c = reseal(*c) }, "evidence")
	mut("state without decision", func(c *DogfoodCorpus) { c.Cases[0].State.Decisions = nil; *c = reseal(*c) }, "decision")
	mut("invalid review case (no tier)", func(c *DogfoodCorpus) { c.Cases[0].Review.Tier = ""; *c = reseal(*c) }, "tier")
	mut("invalid review case (no cost)", func(c *DogfoodCorpus) { c.Cases[0].Review.CostNote = ""; *c = reseal(*c) }, "cost")
	mut("uncalibrated raters", func(c *DogfoodCorpus) {
		// Push every case into beyond-tolerance disagreement so pooled
		// agreement falls under the declared floor.
		for i := range c.Cases {
			c.Cases[i].Review.Raters = []RaterScores{
				{Rater: "a", Scores: map[ReviewDimension]int{Grounding: 1, Completeness: 1, Salience: 1, Actionability: 1, Clarity: 1, Calibration: 1}},
				{Rater: "b", Scores: map[ReviewDimension]int{Grounding: 5, Completeness: 5, Salience: 5, Actionability: 5, Clarity: 5, Calibration: 5}},
			}
		}
		*c = reseal(*c)
	}, "not calibrated")
}

// TestDogfoodCorpusDoesNotFreezeProse pins the #4550 scope clause "without
// freezing prose": rewording a passing case's summary (same recorded ratings,
// digest resealed at a new revision) still grades identically — no exact-string
// oracle anywhere in the corpus contract.
func TestDogfoodCorpusDoesNotFreezeProse(t *testing.T) {
	c := SeedDogfoodCorpus()
	c.Cases = append([]DogfoodCase(nil), c.Cases...)
	for i, dc := range c.Cases {
		if dc.Review.ID != "DOG-pass-grounded-rollup" {
			continue
		}
		dc.Review.Subject = "Lead item: cache reuse is down to 0.41 against its 0.55 floor per the cache-value ledger. Kernel lane shipped 3 times this window (activity only). Operator load fine. Owner cache lane: run `fak cachevalue report`, keep the ratchet held."
		c.Cases[i] = dc
	}
	c.Revision = "2026-07-13.1+reworded"
	c = reseal(c)

	if err := c.Validate(); err != nil {
		t.Fatalf("reworded corpus must still validate: %v", err)
	}
	res := GradeDogfoodCorpus(c)
	if !res.Pass {
		t.Fatalf("reworded prose with unchanged ratings must grade identically: %+v", res)
	}
}

// TestDogfoodCorpusRoundTripRefusesUnknownSchema proves the versioned envelope:
// a mis-versioned serialized corpus is rejected rather than graded.
func TestDogfoodCorpusRoundTripRefusesUnknownSchema(t *testing.T) {
	blob, err := MarshalDogfoodCorpus(SeedDogfoodCorpus())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, err := UnmarshalDogfoodCorpus(blob); err != nil {
		t.Fatalf("round-trip should succeed: %v", err)
	}
	bad := strings.Replace(string(blob), DogfoodCorpusSchema, "fak-program-report-dogfood-corpus/99", 1)
	if _, err := UnmarshalDogfoodCorpus([]byte(bad)); err == nil {
		t.Fatalf("unmarshal must reject an unknown schema")
	}
}

// TestDogfoodReplayArtifactIsScrubbed proves a corpus-path replay artifact
// never carries a credential pasted into a captured summary.
func TestDogfoodReplayArtifactIsScrubbed(t *testing.T) {
	c := SeedDogfoodCorpus()
	c.Cases = append([]DogfoodCase(nil), c.Cases...)
	c.Cases[0].Review.Subject = "Kernel is 40% faster, see token=deadbeefdeadbeefdeadbeefdeadbeef for the raw run."
	c = reseal(c)

	res := GradeDogfoodCorpus(c)
	if !res.Pass {
		t.Fatalf("defect case still holding, corpus should pass: %+v", res)
	}
	var replay string
	for _, row := range res.Cases {
		if row.CaseID == "DOG-grounding-unsupported-speedup" {
			if row.Verdict.Replay == nil {
				t.Fatalf("holding defect case must carry its replay artifact")
			}
			replay = row.Verdict.Replay.SubjectExcerpt
		}
	}
	if strings.Contains(replay, "deadbeefdeadbeef") {
		t.Fatalf("replay artifact leaked a credential-shaped span: %s", replay)
	}
	if !strings.Contains(replay, "[REDACTED]") {
		t.Fatalf("replay artifact should show a [REDACTED] marker: %s", replay)
	}
}
