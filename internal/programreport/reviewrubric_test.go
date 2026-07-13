package programreport

import (
	"encoding/json"
	"strings"
	"testing"
)

// validProvenance is a replay-complete provenance stamp shared by the tests so
// each case's identity is fully recorded (model/tokenizer/engine/seed/revision/
// baseline) — the shared #4509 provenance contract.
func validProvenance() ReviewProvenance {
	return ReviewProvenance{
		Model:     "claude-opus-4-8",
		Tokenizer: "claude-bpe-v2",
		Engine:    "fak-gateway",
		Seed:      "20260713",
		Revision:  "4a399adc58af",
		Baseline:  "programreport-review-baseline/2026-07",
	}
}

// scores builds one rater's full six-axis score set from positional ratings in
// canonical ReviewDimensions order.
func scores(rater string, g, c, sa, a, cl, ca int) RaterScores {
	return RaterScores{Rater: rater, Scores: map[ReviewDimension]int{
		Grounding: g, Completeness: c, Salience: sa, Actionability: a, Clarity: cl, Calibration: ca,
	}}
}

// TestReviewPlantedDefectFailsThenFixPasses is the #4562 witness: one
// representative planted defect (an executive summary that asserts an unsupported
// claim, so both blind raters score grounding at the floor) FAILS closed and
// names grounding as the first actionable divergence; the fixed summary (the
// claim now traces to a measured signal) PASSES in the same clean, independently
// re-graded path. Same provenance, tier and cost — only the graded content moves.
func TestReviewPlantedDefectFailsThenFixPasses(t *testing.T) {
	prov := validProvenance()

	// Planted defect: "cache program shipped 40% faster" with no measured signal
	// behind the 40%. Both raters independently mark grounding = 1 (fabricated).
	defect := ReviewCase{
		Schema:     ReviewSchema,
		ID:         "REV-cache-rollup-defect",
		Subject:    "The cache program shipped and is now 40% faster across the board; nothing else needs attention.",
		Provenance: prov,
		Tier:       TierNightly,
		CostNote:   "two human raters, ~6 min/case; no accelerator time",
		Raters: []RaterScores{
			scores("rater-a", 1, 3, 2, 2, 4, 2),
			scores("rater-b", 1, 3, 3, 2, 4, 2),
		},
	}
	dv := Review(defect)
	if dv.Pass {
		t.Fatalf("planted defect must not pass, got pass verdict: %+v", dv)
	}
	if dv.FirstDivergence != Grounding {
		t.Fatalf("first divergence = %q, want %q", dv.FirstDivergence, Grounding)
	}
	if dv.DivergenceKind != "below_floor" {
		t.Fatalf("divergence kind = %q, want below_floor", dv.DivergenceKind)
	}
	if dv.Replay == nil || dv.Replay.Schema != ReplaySchema || dv.Replay.Dimension != Grounding {
		t.Fatalf("failing case must emit a replay artifact naming grounding, got %+v", dv.Replay)
	}
	if code, _ := CheckReviewGate(dv); code != 1 {
		t.Fatalf("gate on defect = %d, want 1", code)
	}

	// Fix: the claim now cites the measured signal and names the residual gap;
	// both raters lift grounding to 4 (grounded) and the summary clears the floor.
	fixed := defect
	fixed.ID = "REV-cache-rollup-fixed"
	fixed.Subject = "Cache program: decode-throughput ledger shows +38% (baseline 2026-07); one program frontier regressed — see `fak cachevalue report`. Next: hold the ratchet."
	fixed.Raters = []RaterScores{
		scores("rater-a", 4, 4, 4, 4, 4, 4),
		scores("rater-b", 4, 4, 4, 4, 5, 4),
	}
	fv := Review(fixed)
	if !fv.Pass {
		t.Fatalf("fixed summary must pass, got fail: %+v", fv)
	}
	if fv.Replay != nil {
		t.Fatalf("passing case must not emit a replay artifact, got %+v", fv.Replay)
	}
	if code, _ := CheckReviewGate(fv); code != 0 {
		t.Fatalf("gate on fix = %d, want 0", code)
	}
}

// TestReviewFailsClosedOnMissingEvidence covers the "missing or inconclusive
// evidence is never pass" clause across every admission boundary.
func TestReviewFailsClosedOnMissingEvidence(t *testing.T) {
	base := ReviewCase{
		Schema:     ReviewSchema,
		ID:         "REV-base",
		Subject:    "a grounded summary",
		Provenance: validProvenance(),
		Tier:       TierPR,
		CostNote:   "2 raters, 5 min",
		Raters: []RaterScores{
			scores("rater-a", 4, 4, 4, 4, 4, 4),
			scores("rater-b", 4, 4, 4, 4, 4, 4),
		},
	}
	// Sanity: the base case passes, so each mutation below isolates one defect.
	if v := Review(base); !v.Pass {
		t.Fatalf("base case should pass: %+v", v)
	}

	mut := func(name string, f func(*ReviewCase)) {
		c := base
		// deep-copy the rater slice so mutations don't leak across sub-tests
		c.Raters = append([]RaterScores(nil), base.Raters...)
		f(&c)
		if v := Review(c); v.Pass {
			t.Fatalf("%s: expected fail-closed, got pass: %+v", name, v)
		} else if v.DivergenceKind == "" {
			t.Fatalf("%s: non-pass verdict must carry a divergence kind: %+v", name, v)
		}
	}

	mut("wrong schema", func(c *ReviewCase) { c.Schema = "other/1" })
	mut("empty id", func(c *ReviewCase) { c.ID = "" })
	mut("empty subject", func(c *ReviewCase) { c.Subject = "  " })
	mut("no tier", func(c *ReviewCase) { c.Tier = "" })
	mut("undocumented cost", func(c *ReviewCase) { c.CostNote = "" })
	mut("one rater", func(c *ReviewCase) { c.Raters = c.Raters[:1] })
	mut("missing provenance model", func(c *ReviewCase) { c.Provenance.Model = "" })
	mut("no seed and no oracle", func(c *ReviewCase) { c.Provenance.Seed = "" })
	mut("out-of-range score", func(c *ReviewCase) {
		c.Raters[0] = scores("rater-a", 7, 4, 4, 4, 4, 4)
	})
	mut("missing axis", func(c *ReviewCase) {
		c.Raters[0] = RaterScores{Rater: "rater-a", Scores: map[ReviewDimension]int{Grounding: 4}}
	})

	// An oracle satisfies the seed-or-oracle rule (seedless deterministic case).
	c := base
	c.Provenance.Seed = ""
	c.Provenance.Oracle = "programreport-review-oracle/v1"
	if v := Review(c); !v.Pass {
		t.Fatalf("seedless case with a deterministic oracle should pass: %+v", v)
	}
}

// TestReviewInconclusiveNeverPasses proves the disagreement process: when raters
// disagree beyond tolerance on an axis, that axis is inconclusive and the case
// cannot pass even though every rating is otherwise high.
func TestReviewInconclusiveNeverPasses(t *testing.T) {
	c := ReviewCase{
		Schema:     ReviewSchema,
		ID:         "REV-disagree",
		Subject:    "a summary two raters read very differently on salience",
		Provenance: validProvenance(),
		Tier:       TierRelease,
		CostNote:   "2 raters, 5 min",
		Raters: []RaterScores{
			scores("rater-a", 5, 5, 2, 5, 5, 5), // salience 2
			scores("rater-b", 5, 5, 5, 5, 5, 5), // salience 5 -> gap 3 > tolerance
		},
	}
	v := Review(c)
	if v.Pass {
		t.Fatalf("case with an inconclusive axis must not pass: %+v", v)
	}
	if v.FirstDivergence != Salience || v.DivergenceKind != "inconclusive" {
		t.Fatalf("want inconclusive @ salience, got %s @ %s", v.DivergenceKind, v.FirstDivergence)
	}
	if v.Consensus[Salience] != 0 {
		t.Fatalf("inconclusive axis consensus must be 0 sentinel, got %d", v.Consensus[Salience])
	}
}

// TestSeedSetInterRaterAgreement proves the corpus-level acceptance gate: a
// blinded seed set of well-calibrated cases reaches the declared agreement floor,
// while a set carrying a genuine disagreement falls below it.
func TestSeedSetInterRaterAgreement(t *testing.T) {
	mk := func(id string, ra, rb RaterScores) ReviewCase {
		return ReviewCase{
			Schema: ReviewSchema, ID: id, Subject: "summary " + id,
			Provenance: validProvenance(), Tier: TierNightly,
			CostNote: "2 raters, 5 min", Raters: []RaterScores{ra, rb},
		}
	}
	calibrated := []ReviewCase{
		mk("s1", scores("a", 4, 4, 4, 4, 4, 4), scores("b", 4, 4, 5, 4, 4, 4)),
		mk("s2", scores("a", 3, 4, 4, 3, 5, 4), scores("b", 4, 4, 4, 4, 5, 4)),
		mk("s3", scores("a", 5, 5, 4, 4, 4, 5), scores("b", 4, 5, 4, 4, 5, 5)),
	}
	ok, frac, reason := SeedSetCalibrated(calibrated)
	if !ok {
		t.Fatalf("calibrated seed set should reach the floor: %s", reason)
	}
	if frac < DeclaredAgreementFloor {
		t.Fatalf("agreement %.3f < floor %.2f", frac, DeclaredAgreementFloor)
	}

	// A set dominated by beyond-tolerance disagreement must NOT be calibrated.
	noisy := []ReviewCase{
		mk("n1", scores("a", 1, 1, 1, 1, 1, 1), scores("b", 5, 5, 5, 5, 5, 5)),
		mk("n2", scores("a", 2, 1, 2, 1, 2, 1), scores("b", 5, 4, 5, 4, 5, 4)),
	}
	if ok, _, _ := SeedSetCalibrated(noisy); ok {
		t.Fatalf("noisy seed set must not be calibrated")
	}

	// Fail-closed: an empty set is never calibrated (no vacuous 1.0).
	if ok, _, _ := SeedSetCalibrated(nil); ok {
		t.Fatalf("empty seed set must not be calibrated")
	}
}

// TestReplayArtifactScrubsSecrets proves the replay artifact never carries a
// credential pasted into a summary or provenance value.
func TestReplayArtifactScrubsSecrets(t *testing.T) {
	c := ReviewCase{
		Schema:     ReviewSchema,
		ID:         "REV-secret",
		Subject:    "summary leaks a token sk-abcdef0123456789abcdef0123456789 into the prose and asserts an unsupported claim",
		Provenance: validProvenance(),
		Tier:       TierPR,
		CostNote:   "2 raters",
		Raters: []RaterScores{
			scores("a", 1, 4, 4, 4, 4, 4),
			scores("b", 1, 4, 4, 4, 4, 4),
		},
	}
	c.Provenance.Baseline = "baseline token=deadbeefdeadbeefdeadbeefdeadbeef"
	v := Review(c)
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

// TestReviewCaseSchemaRoundTrip proves the versioned envelope rejects an unknown
// schema instead of silently grading it.
func TestReviewCaseSchemaRoundTrip(t *testing.T) {
	c := ReviewCase{
		Schema: ReviewSchema, ID: "REV-rt", Subject: "s", Provenance: validProvenance(),
		Tier: TierPR, CostNote: "x", Raters: []RaterScores{scores("a", 4, 4, 4, 4, 4, 4), scores("b", 4, 4, 4, 4, 4, 4)},
	}
	blob, err := MarshalReviewCase(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, err := UnmarshalReviewCase(blob); err != nil {
		t.Fatalf("round-trip should succeed: %v", err)
	}
	bad := strings.Replace(string(blob), ReviewSchema, "bogus/9", 1)
	if _, err := UnmarshalReviewCase([]byte(bad)); err == nil {
		t.Fatalf("unmarshal must reject an unknown schema")
	}
}

// TestSummaryOfGradesReportNarrative ties the rubric to this package's own
// output: the subject under review is a folded Report's Reason + NextAction prose.
func TestSummaryOfGradesReportNarrative(t *testing.T) {
	r := Report{Reason: "programs recorded; 3 ongoing", NextAction: "hold the line"}
	got := SummaryOf(r)
	if !strings.Contains(got, "programs recorded") || !strings.Contains(got, "hold the line") {
		t.Fatalf("SummaryOf should fold Reason and NextAction, got %q", got)
	}
}
