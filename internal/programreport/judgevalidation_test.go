package programreport

import (
	"encoding/json"
	"strings"
	"testing"
)

// heldOut builds one held-out rating row from positional fields.
func heldOut(id string, judge, expert int, confidence string, escalated bool) HeldOutRating {
	return HeldOutRating{ItemID: id, JudgeScore: judge, ExpertScore: expert, Confidence: confidence, Escalated: escalated}
}

// validJudgeCase is a fully-passing judge-validation case: position-stable,
// repeatable, no verbosity premium, unbiased, correlated with the experts, and
// escalation-honoring. Each fail-closed test below mutates exactly one axis so
// the defect is isolated. Provenance reuses validProvenance() from
// reviewrubric_test.go — the shared #4509 provenance contract.
func validJudgeCase() JudgeCase {
	return JudgeCase{
		Schema:     JudgeSchema,
		ID:         "JUDGE-base",
		Judge:      "claude-opus-4-8/report-judge-v1",
		Provenance: validProvenance(),
		Tier:       TierNightly,
		CostNote:   "held-out set of 4, ~3s/case on gateway; no accelerator time",
		Positions: []PairProbe{
			{ItemA: "sumA", ItemB: "sumB", WinnerForward: "sumA", WinnerSwapped: "sumA"},
		},
		Paraphrases: []ParaphraseProbe{
			{ItemID: "p1", ScoreOriginal: 4, ScoreParaphrase: 4},
		},
		Verbosities: []VerbosityProbe{
			{ItemID: "v1", JudgeScoreConcise: 4, JudgeScoreVerbose: 4},
		},
		HeldOut: []HeldOutRating{
			heldOut("h1", 4, 4, "high", false),
			heldOut("h2", 5, 4, "high", false),
			heldOut("h3", 3, 4, "low", true),
			heldOut("h4", 4, 4, "high", false),
		},
	}
}

// TestJudgeValidationPlantedDefectFailsThenFixPasses is the #4563 witness: a
// representative planted defect — a POSITION-BIASED judge that picks whichever
// summary is shown first, so an order swap flips its winner — FAILS closed and
// names position as the first actionable divergence with a scrubbed replay; the
// fixed judge (order-stable) PASSES in the same clean, independently re-graded
// path. Same provenance, tier and cost — only the judge's recorded behaviour moves.
func TestJudgeValidationPlantedDefectFailsThenFixPasses(t *testing.T) {
	defect := validJudgeCase()
	defect.ID = "JUDGE-position-defect"
	// Position-biased judge: forward (A,B) it picks the first shown (A); swapped
	// (B,A) it again picks the first shown (B). The winner flips with order.
	defect.Positions = []PairProbe{
		{ItemA: "sumA", ItemB: "sumB", WinnerForward: "sumA", WinnerSwapped: "sumB"},
	}

	dv := ReviewJudge(defect)
	if dv.Pass {
		t.Fatalf("position-biased judge must not pass, got: %+v", dv)
	}
	if dv.FirstDivergence != Position {
		t.Fatalf("first divergence = %q, want %q", dv.FirstDivergence, Position)
	}
	if dv.DivergenceKind != "position_flip" {
		t.Fatalf("divergence kind = %q, want position_flip", dv.DivergenceKind)
	}
	if dv.Replay == nil || dv.Replay.Schema != JudgeReplaySchema || dv.Replay.Axis != Position {
		t.Fatalf("failing case must emit a replay artifact naming position, got %+v", dv.Replay)
	}
	if code, _ := CheckJudgeGate(dv); code != 1 {
		t.Fatalf("gate on defect = %d, want 1", code)
	}

	// Fix: an order-stable judge names the same winner regardless of presentation
	// order. Everything else is identical to the defect case.
	fixed := defect
	fixed.ID = "JUDGE-position-fixed"
	fixed.Positions = []PairProbe{
		{ItemA: "sumA", ItemB: "sumB", WinnerForward: "sumA", WinnerSwapped: "sumA"},
	}

	fv := ReviewJudge(fixed)
	if !fv.Pass {
		t.Fatalf("order-stable judge must pass, got fail: %+v", fv)
	}
	if fv.Replay != nil {
		t.Fatalf("passing case must not emit a replay artifact, got %+v", fv.Replay)
	}
	if code, _ := CheckJudgeGate(fv); code != 0 {
		t.Fatalf("gate on fix = %d, want 0", code)
	}
}

// TestJudgeFailsClosedStructural covers "missing or inconclusive evidence is
// never pass" across every structural admission boundary.
func TestJudgeFailsClosedStructural(t *testing.T) {
	if v := ReviewJudge(validJudgeCase()); !v.Pass {
		t.Fatalf("base judge case should pass: %+v", v)
	}

	mut := func(name string, f func(*JudgeCase)) {
		t.Helper()
		c := validJudgeCase()
		f(&c)
		v := ReviewJudge(c)
		if v.Pass {
			t.Fatalf("%s: expected fail-closed, got pass: %+v", name, v)
		}
		if v.DivergenceKind == "" {
			t.Fatalf("%s: non-pass verdict must carry a divergence kind: %+v", name, v)
		}
	}

	mut("wrong schema", func(c *JudgeCase) { c.Schema = "other/1" })
	mut("empty id", func(c *JudgeCase) { c.ID = "" })
	mut("no judge", func(c *JudgeCase) { c.Judge = "  " })
	mut("missing provenance model", func(c *JudgeCase) { c.Provenance.Model = "" })
	mut("no seed and no oracle", func(c *JudgeCase) { c.Provenance.Seed = "" })
	mut("no tier", func(c *JudgeCase) { c.Tier = "" })
	mut("undocumented cost", func(c *JudgeCase) { c.CostNote = "" })
	mut("out-of-range paraphrase score", func(c *JudgeCase) {
		c.Paraphrases = []ParaphraseProbe{{ItemID: "p1", ScoreOriginal: 7, ScoreParaphrase: 4}}
	})
	mut("bad pair winner", func(c *JudgeCase) {
		c.Positions = []PairProbe{{ItemA: "sumA", ItemB: "sumB", WinnerForward: "sumZ", WinnerSwapped: "sumA"}}
	})
	mut("bad confidence label", func(c *JudgeCase) {
		c.HeldOut = append([]HeldOutRating(nil), c.HeldOut...)
		c.HeldOut[0].Confidence = "medium"
	})

	// A seedless case with a deterministic oracle satisfies the seed-or-oracle rule.
	c := validJudgeCase()
	c.Provenance.Seed = ""
	c.Provenance.Oracle = "programreport-judge-oracle/v1"
	if v := ReviewJudge(c); !v.Pass {
		t.Fatalf("seedless case with a deterministic oracle should pass: %+v", v)
	}
}

// TestJudgeInconclusiveOnMissingAxisEvidence proves an axis with NO probe evidence
// is inconclusive (never a pass) and is named as the first divergence in canonical
// order.
func TestJudgeInconclusiveOnMissingAxisEvidence(t *testing.T) {
	cases := []struct {
		name string
		axis JudgeAxis
		f    func(*JudgeCase)
	}{
		{"no positions", Position, func(c *JudgeCase) { c.Positions = nil }},
		{"no paraphrases", Repeatability, func(c *JudgeCase) { c.Paraphrases = nil }},
		{"no verbosities", Verbosity, func(c *JudgeCase) { c.Verbosities = nil }},
		{"no held-out", Bias, func(c *JudgeCase) { c.HeldOut = nil }},
	}
	for _, tc := range cases {
		c := validJudgeCase()
		tc.f(&c)
		v := ReviewJudge(c)
		if v.Pass {
			t.Fatalf("%s: missing-evidence case must not pass: %+v", tc.name, v)
		}
		if v.DivergenceKind != "inconclusive" {
			t.Fatalf("%s: kind = %q, want inconclusive", tc.name, v.DivergenceKind)
		}
		if v.FirstDivergence != tc.axis {
			t.Fatalf("%s: first divergence = %q, want %q", tc.name, v.FirstDivergence, tc.axis)
		}
		if v.Replay == nil || v.Replay.Axis != tc.axis {
			t.Fatalf("%s: inconclusive case must emit a replay naming the axis, got %+v", tc.name, v.Replay)
		}
	}
}

// TestJudgeAxisDivergences plants one representative defect per remaining axis and
// asserts each is caught, named, and attributed in canonical first-divergence
// order.
func TestJudgeAxisDivergences(t *testing.T) {
	cases := []struct {
		name string
		axis JudgeAxis
		kind string
		f    func(*JudgeCase)
	}{
		{
			"unrepeatable paraphrase", Repeatability, "paraphrase_unstable",
			func(c *JudgeCase) {
				c.Paraphrases = []ParaphraseProbe{{ItemID: "p1", ScoreOriginal: 4, ScoreParaphrase: 2}}
			},
		},
		{
			"verbosity premium", Verbosity, "verbosity_preference",
			func(c *JudgeCase) {
				c.Verbosities = []VerbosityProbe{{ItemID: "v1", JudgeScoreConcise: 3, JudgeScoreVerbose: 5}}
			},
		},
		{
			"systematic bias", Bias, "biased",
			func(c *JudgeCase) {
				// Every judge score sits one full point above the expert: agreement
				// is still within ±1, but the mean signed error is +1.0 > 0.75.
				c.HeldOut = []HeldOutRating{
					heldOut("h1", 5, 4, "high", false),
					heldOut("h2", 5, 4, "high", false),
					heldOut("h3", 4, 3, "high", false),
					heldOut("h4", 3, 2, "high", false),
				}
			},
		},
		{
			"below correlation floor", Correlation, "below_correlation_floor",
			func(c *JudgeCase) {
				// Symmetric ±2 disagreement: mean signed error ~0 (bias passes) but
				// no item lands within the ±1 agreement band, so agreement is 0.0.
				c.HeldOut = []HeldOutRating{
					heldOut("h1", 5, 3, "low", true),
					heldOut("h2", 1, 3, "low", true),
					heldOut("h3", 5, 3, "low", true),
					heldOut("h4", 1, 3, "low", true),
				}
			},
		},
		{
			"unescalated low confidence", Escalation, "unescalated_low_confidence",
			func(c *JudgeCase) {
				c.HeldOut = []HeldOutRating{
					heldOut("h1", 4, 4, "high", false),
					heldOut("h2", 3, 4, "low", false), // low but did not escalate
					heldOut("h3", 4, 4, "high", false),
					heldOut("h4", 4, 4, "high", false),
				}
			},
		},
		{
			"overconfident divergence", Escalation, "overconfident_divergence",
			func(c *JudgeCase) {
				// One high-confidence item diverges by 2. Overall agreement stays at
				// 0.75... so raise to 5 items to keep agreement >= floor and bias low.
				c.HeldOut = []HeldOutRating{
					heldOut("h1", 4, 4, "high", false),
					heldOut("h2", 4, 4, "high", false),
					heldOut("h3", 4, 4, "high", false),
					heldOut("h4", 4, 4, "high", false),
					heldOut("h5", 5, 3, "high", false), // confident yet 2 off expert
				}
			},
		},
	}
	for _, tc := range cases {
		c := validJudgeCase()
		tc.f(&c)
		v := ReviewJudge(c)
		if v.Pass {
			t.Fatalf("%s: must not pass: %+v", tc.name, v)
		}
		if v.FirstDivergence != tc.axis {
			t.Fatalf("%s: first divergence = %q, want %q (reason: %s)", tc.name, v.FirstDivergence, tc.axis, v.Reason)
		}
		if v.DivergenceKind != tc.kind {
			t.Fatalf("%s: kind = %q, want %q", tc.name, v.DivergenceKind, tc.kind)
		}
		if v.Replay == nil || v.Replay.Axis != tc.axis || v.Replay.Kind != tc.kind {
			t.Fatalf("%s: replay must name the axis+kind, got %+v", tc.name, v.Replay)
		}
	}
}

// TestJudgeReplayScrubsSecrets proves the replay artifact never carries a
// credential pasted into a judge id or a provenance value.
func TestJudgeReplayScrubsSecrets(t *testing.T) {
	c := validJudgeCase()
	c.ID = "JUDGE-secret"
	c.Judge = "judge sk-abcdef0123456789abcdef0123456789"
	c.Provenance.Baseline = "baseline token=deadbeefdeadbeefdeadbeefdeadbeef"
	// Force a failure so a replay is emitted.
	c.Positions = []PairProbe{{ItemA: "a", ItemB: "b", WinnerForward: "a", WinnerSwapped: "b"}}

	v := ReviewJudge(c)
	if v.Pass || v.Replay == nil {
		t.Fatalf("expected a failing case with a replay artifact: %+v", v)
	}
	blob, err := json.Marshal(v.Replay)
	if err != nil {
		t.Fatalf("marshal replay: %v", err)
	}
	s := string(blob)
	for _, leak := range []string{"sk-abcdef0123456789", "deadbeefdeadbeef"} {
		if strings.Contains(s, leak) {
			t.Fatalf("replay artifact leaked a credential-shaped span %q: %s", leak, s)
		}
	}
	if !strings.Contains(s, "[REDACTED]") {
		t.Fatalf("replay artifact should show a [REDACTED] marker: %s", s)
	}
}

// TestJudgeCaseSchemaRoundTrip proves the versioned envelope rejects an unknown
// schema instead of silently grading it.
func TestJudgeCaseSchemaRoundTrip(t *testing.T) {
	c := validJudgeCase()
	blob, err := MarshalJudgeCase(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, err := UnmarshalJudgeCase(blob); err != nil {
		t.Fatalf("round-trip should succeed: %v", err)
	}
	bad := strings.Replace(string(blob), JudgeSchema, "bogus/9", 1)
	if _, err := UnmarshalJudgeCase([]byte(bad)); err == nil {
		t.Fatalf("unmarshal must reject an unknown schema")
	}
}

// TestJudgeCorpusCalibrated proves the corpus gate: a set of validated judges is
// admissible; a set with one non-pass judge, or an empty set, is not.
func TestJudgeCorpusCalibrated(t *testing.T) {
	good := validJudgeCase()
	good2 := validJudgeCase()
	good2.ID = "JUDGE-base-2"
	if ok, reason := JudgeCorpusCalibrated([]JudgeCase{good, good2}); !ok {
		t.Fatalf("corpus of passing judges should be calibrated: %s", reason)
	}

	bad := validJudgeCase()
	bad.ID = "JUDGE-flip"
	bad.Positions = []PairProbe{{ItemA: "a", ItemB: "b", WinnerForward: "a", WinnerSwapped: "b"}}
	if ok, _ := JudgeCorpusCalibrated([]JudgeCase{good, bad}); ok {
		t.Fatalf("corpus with a position-biased judge must not be calibrated")
	}

	if ok, _ := JudgeCorpusCalibrated(nil); ok {
		t.Fatalf("empty corpus must not be calibrated")
	}
}
