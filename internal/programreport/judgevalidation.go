package programreport

// judgevalidation.go — issue #4563, the judge-validation rung of the
// quality-middle epic (#4509). Its sibling reviewrubric.go (#4562) grades the
// executive-summary NARRATIVE with an anchored expert rubric; this leaf grades
// the LLM JUDGE that would stand in for those experts. A judge that is
// position-biased, verbosity-preferring, unrepeatable, systematically skewed,
// weakly correlated with the experts, or silently confident when wrong is not
// admissible as a stand-in — no matter how plausible its individual scores look.
//
// The contract validates a judge from its RECORDED behaviour on controlled
// probes plus the expert reference ratings, so it is a pure, hermetic leaf:
// stdlib only, no sibling imports (it cannot red architest), no live model call
// (the judge already ran; we adjudicate what it produced). It reuses the shared
// #4509 provenance/tier/cost contract (ReviewProvenance, ReviewTier) and the
// same secret scrubber reviewrubric.go uses, so the two rungs never disagree on
// what a replay-complete, leak-free case is.
//
// Six axes, in canonical first-divergence order (JudgeAxes), map one-to-one onto
// the issue's Scope ("bias, position effects, verbosity preference,
// repeatability, and held-out correlation") plus its "low confidence escalates":
//
//   - POSITION      — an order-swap pair must pick the same winner in both
//     presentation orders; a judge that just prefers whatever is shown first is
//     position-biased.
//   - REPEATABILITY — a paraphrase of the same summary must score within
//     tolerance; a judge whose score swings on surface form is not repeatable.
//   - VERBOSITY     — a longer summary of equal expert quality must not earn a
//     length premium beyond tolerance.
//   - BIAS          — the mean signed judge−expert error on the held-out set must
//     be within a bias tolerance; a judge that runs systematically high or low is
//     biased even when it "correlates".
//   - CORRELATION   — the within-tolerance agreement fraction against the expert
//     ratings on the held-out set must reach a declared floor.
//   - ESCALATION    — every low-confidence held-out item must have escalated, and
//     no high-confidence item may diverge from the expert beyond tolerance (a
//     confident wrong verdict is the worst failure; low confidence must escalate,
//     not silently pass).
//
// Four properties are enforced, not merely documented, mirroring #4562:
//   - FAIL-CLOSED: malformed provenance / missing tier / undocumented cost / an
//     out-of-range probe score / an axis with NO evidence is REFUSED. Missing or
//     inconclusive evidence is never a pass.
//   - FIRST DIVERGENCE: a failing case names the earliest axis (canonical order)
//     that broke, so attribution is deterministic.
//   - SCRUBBED REPLAY: a failing case emits a self-contained replay artifact with
//     credential-shaped spans redacted, so it can be attached to an issue.
//   - TIERED + COSTED: each case carries an explicit PR/nightly/release tier and a
//     documented runtime/resource cost.
//
// NOTE (invalidating assumption, #4563): "held-out correlation" is operationalized
// here as a tolerance-band AGREEMENT fraction plus a signed-bias check, not a
// Pearson/Spearman r. On a 1-5 integer scale with few held-out items a rank
// coefficient is dominated by ties and unstable; the tolerance band is the robust,
// honestly-labeled proxy. Promoting CORRELATION to a rank coefficient once the
// held-out corpus is large enough is the named follow-on.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// JudgeSchema is the versioned envelope tag of a judge-validation case; a reader
// handed a different schema refuses it rather than grading an unversioned corpus.
const JudgeSchema = "fak-program-report-judge/1"

// JudgeReplaySchema tags the scrubbed replay artifact a failing judge case emits.
const JudgeReplaySchema = "fak-program-report-judge-replay/1"

// DefaultCorrelationFloor is the minimum within-tolerance agreement a judge must
// reach against the held-out expert ratings to be admissible. It mirrors
// reviewrubric.go's DeclaredAgreementFloor: a judge that agrees with the experts
// on fewer than 80% of held-out items is not a calibrated stand-in.
const DefaultCorrelationFloor = 0.80

// DefaultBiasTolerance is the largest absolute mean signed (judge − expert) error
// a judge may carry on the held-out set. It is below 1.0 on purpose: a full-point
// systematic skew is a bias to fix, not to average away, even if per-item
// agreement is otherwise within the ±1 band.
const DefaultBiasTolerance = 0.75

// JudgeAxis is one validation axis of the judge contract.
type JudgeAxis string

const (
	Position      JudgeAxis = "position"      // order-swap stability
	Repeatability JudgeAxis = "repeatability" // paraphrase stability
	Verbosity     JudgeAxis = "verbosity"     // no length premium
	Bias          JudgeAxis = "bias"          // signed error vs expert
	Correlation   JudgeAxis = "correlation"   // held-out agreement vs expert
	Escalation    JudgeAxis = "escalation"    // low confidence escalates
)

// JudgeAxes is the fixed, ordered axis set. Canonical order is the
// first-divergence order: a failing case is attributed to the earliest axis here
// that diverges, so attribution is deterministic.
var JudgeAxes = []JudgeAxis{Position, Repeatability, Verbosity, Bias, Correlation, Escalation}

// PairProbe records a judge's pairwise preference under an order swap. The judge
// is shown the same two summaries twice, forward (A, B) and swapped (B, A); a
// position-free judge names the same winner both times. Each winner must be "" (a
// declared tie), ItemA, or ItemB.
type PairProbe struct {
	ItemA         string `json:"item_a"`
	ItemB         string `json:"item_b"`
	WinnerForward string `json:"winner_forward"` // winner when shown (A, B)
	WinnerSwapped string `json:"winner_swapped"` // winner when shown (B, A)
}

func (p PairProbe) stable() bool { return p.WinnerForward == p.WinnerSwapped }

func (p PairProbe) validWinner(w string) bool {
	return w == "" || w == p.ItemA || w == p.ItemB
}

// ParaphraseProbe records a judge's score on a summary and on a meaning-preserving
// paraphrase of it. A repeatable judge scores the two within tolerance.
type ParaphraseProbe struct {
	ItemID          string `json:"item_id"`
	ScoreOriginal   int    `json:"score_original"`
	ScoreParaphrase int    `json:"score_paraphrase"`
}

// VerbosityProbe records a judge's score on a concise summary and on a longer
// summary of EQUAL expert quality. A judge free of verbosity preference gives the
// verbose form no premium beyond tolerance.
type VerbosityProbe struct {
	ItemID            string `json:"item_id"`
	JudgeScoreConcise int    `json:"judge_score_concise"`
	JudgeScoreVerbose int    `json:"judge_score_verbose"`
}

// HeldOutRating pairs a judge's score with the expert consensus on a held-out
// summary the judge did not calibrate on, plus the judge's self-reported
// confidence and whether the case escalated to human review.
type HeldOutRating struct {
	ItemID      string `json:"item_id"`
	JudgeScore  int    `json:"judge_score"`
	ExpertScore int    `json:"expert_score"`
	Confidence  string `json:"confidence"` // "high" | "low"
	Escalated   bool   `json:"escalated"`
}

// JudgeCase is one judge-validation case: the judge under validation, its
// replay-complete provenance, tier and documented cost, and the recorded probe
// evidence on each axis.
type JudgeCase struct {
	Schema           string            `json:"schema"`
	ID               string            `json:"id"`
	Judge            string            `json:"judge"` // identifier of the LLM judge under validation
	Provenance       ReviewProvenance  `json:"provenance"`
	Tier             ReviewTier        `json:"tier"`
	CostNote         string            `json:"cost_note"`
	Positions        []PairProbe       `json:"positions"`
	Paraphrases      []ParaphraseProbe `json:"paraphrases"`
	Verbosities      []VerbosityProbe  `json:"verbosities"`
	HeldOut          []HeldOutRating   `json:"held_out"`
	Tolerance        int               `json:"tolerance,omitempty"`         // 0 => AgreementTolerance
	CorrelationFloor float64           `json:"correlation_floor,omitempty"` // 0 => DefaultCorrelationFloor
	BiasTolerance    float64           `json:"bias_tolerance,omitempty"`    // 0 => DefaultBiasTolerance
}

func (c JudgeCase) tolerance() int {
	if c.Tolerance <= 0 {
		return AgreementTolerance
	}
	return c.Tolerance
}

func (c JudgeCase) correlationFloor() float64 {
	if c.CorrelationFloor <= 0 {
		return DefaultCorrelationFloor
	}
	return c.CorrelationFloor
}

func (c JudgeCase) biasTolerance() float64 {
	if c.BiasTolerance <= 0 {
		return DefaultBiasTolerance
	}
	return c.BiasTolerance
}

// JudgeMetrics is the measured fact set for a case — the report-contract posture:
// an operator reads the numbers behind the verdict without re-deriving them.
type JudgeMetrics struct {
	PositionPairs       int     `json:"position_pairs"`
	PositionUnstable    int     `json:"position_unstable"`
	ParaphraseMaxGap    int     `json:"paraphrase_max_gap"`
	VerbosityMaxPremium int     `json:"verbosity_max_premium"`
	HeldOutPairs        int     `json:"held_out_pairs"`
	MeanSignedError     float64 `json:"mean_signed_error"`
	HeldOutAgreement    float64 `json:"held_out_agreement"`
	LowConfidence       int     `json:"low_confidence"`
	Escalated           int     `json:"escalated"`
}

// JudgeVerdict is the adjudicated result of validating one judge case.
type JudgeVerdict struct {
	CaseID          string       `json:"case_id"`
	JudgeID         string       `json:"judge_id"`
	Pass            bool         `json:"pass"`
	FirstDivergence JudgeAxis    `json:"first_divergence,omitempty"`
	DivergenceKind  string       `json:"divergence_kind,omitempty"`
	Reason          string       `json:"reason"`
	Metrics         JudgeMetrics `json:"metrics"`
	Replay          *JudgeReplay `json:"replay,omitempty"` // present iff Pass is false
}

// JudgeReplay is the scrubbed, self-contained record a failing case emits so the
// first divergence can be independently re-checked. Provenance values and the
// free-text reason pass through the same secret scrubber reviewrubric.go uses.
type JudgeReplay struct {
	Schema     string           `json:"schema"`
	CaseID     string           `json:"case_id"`
	JudgeID    string           `json:"judge_id"`
	Tier       ReviewTier       `json:"tier"`
	Provenance ReviewProvenance `json:"provenance"` // scrubbed
	Axis       JudgeAxis        `json:"axis"`
	Kind       string           `json:"kind"`
	Metrics    JudgeMetrics     `json:"metrics"`
	Reason     string           `json:"reason"`
}

// ValidateJudge is the structural admission gate. It fails closed at every
// boundary: wrong schema, empty id/judge, incomplete provenance, an unassigned
// tier, an undocumented cost, an out-of-range probe score, a malformed pair
// winner, or a held-out row with an unknown confidence label. Axis EMPTINESS is
// NOT rejected here — it is adjudicated as an inconclusive divergence by
// ReviewJudge, so a failing case still names the missing axis.
func (c JudgeCase) ValidateJudge() error {
	if c.Schema != JudgeSchema {
		return fmt.Errorf("programreport: judge case schema %q, want %q", c.Schema, JudgeSchema)
	}
	if strings.TrimSpace(c.ID) == "" {
		return fmt.Errorf("programreport: judge case id is required")
	}
	if strings.TrimSpace(c.Judge) == "" {
		return fmt.Errorf("programreport: judge case %q names no judge under validation", c.ID)
	}
	if err := c.Provenance.validate(); err != nil {
		return fmt.Errorf("programreport: judge case %q provenance: %w", c.ID, err)
	}
	if !c.Tier.valid() {
		return fmt.Errorf("programreport: judge case %q tier %q is not pr/nightly/release", c.ID, c.Tier)
	}
	if strings.TrimSpace(c.CostNote) == "" {
		return fmt.Errorf("programreport: judge case %q must document runtime/resource cost", c.ID)
	}
	for i, p := range c.Positions {
		if strings.TrimSpace(p.ItemA) == "" || strings.TrimSpace(p.ItemB) == "" {
			return fmt.Errorf("programreport: judge case %q position probe %d has an unnamed item", c.ID, i)
		}
		if !p.validWinner(p.WinnerForward) || !p.validWinner(p.WinnerSwapped) {
			return fmt.Errorf("programreport: judge case %q position probe %d winner is not item_a/item_b/tie", c.ID, i)
		}
	}
	for i, p := range c.Paraphrases {
		if strings.TrimSpace(p.ItemID) == "" {
			return fmt.Errorf("programreport: judge case %q paraphrase probe %d has no item id", c.ID, i)
		}
		if !inScore(p.ScoreOriginal) || !inScore(p.ScoreParaphrase) {
			return fmt.Errorf("programreport: judge case %q paraphrase probe %d score out of 1..5", c.ID, i)
		}
	}
	for i, p := range c.Verbosities {
		if strings.TrimSpace(p.ItemID) == "" {
			return fmt.Errorf("programreport: judge case %q verbosity probe %d has no item id", c.ID, i)
		}
		if !inScore(p.JudgeScoreConcise) || !inScore(p.JudgeScoreVerbose) {
			return fmt.Errorf("programreport: judge case %q verbosity probe %d score out of 1..5", c.ID, i)
		}
	}
	for i, h := range c.HeldOut {
		if strings.TrimSpace(h.ItemID) == "" {
			return fmt.Errorf("programreport: judge case %q held-out row %d has no item id", c.ID, i)
		}
		if !inScore(h.JudgeScore) || !inScore(h.ExpertScore) {
			return fmt.Errorf("programreport: judge case %q held-out row %d score out of 1..5", c.ID, i)
		}
		if h.Confidence != "high" && h.Confidence != "low" {
			return fmt.Errorf("programreport: judge case %q held-out row %d confidence %q is not high/low", c.ID, i, h.Confidence)
		}
	}
	return nil
}

func inScore(s int) bool { return s >= 1 && s <= 5 }

// judgeMetrics folds the measured facts once, so the adjudicator and the replay
// artifact report the same numbers.
func (c JudgeCase) metrics() JudgeMetrics {
	m := JudgeMetrics{PositionPairs: len(c.Positions), HeldOutPairs: len(c.HeldOut)}
	for _, p := range c.Positions {
		if !p.stable() {
			m.PositionUnstable++
		}
	}
	for _, p := range c.Paraphrases {
		if g := absInt(p.ScoreOriginal - p.ScoreParaphrase); g > m.ParaphraseMaxGap {
			m.ParaphraseMaxGap = g
		}
	}
	for _, p := range c.Verbosities {
		if prem := p.JudgeScoreVerbose - p.JudgeScoreConcise; prem > m.VerbosityMaxPremium {
			m.VerbosityMaxPremium = prem
		}
	}
	if len(c.HeldOut) > 0 {
		tol, sum, agree := c.tolerance(), 0, 0
		for _, h := range c.HeldOut {
			sum += h.JudgeScore - h.ExpertScore
			if absInt(h.JudgeScore-h.ExpertScore) <= tol {
				agree++
			}
			if h.Confidence == "low" {
				m.LowConfidence++
			}
			if h.Escalated {
				m.Escalated++
			}
		}
		m.MeanSignedError = float64(sum) / float64(len(c.HeldOut))
		m.HeldOutAgreement = float64(agree) / float64(len(c.HeldOut))
	}
	return m
}

// ReviewJudge validates one judge case and fails closed. Adjudication order:
// (1) structural validation — a malformed case is kind "invalid" and never
// passes; (2) each axis in canonical order — an axis with no evidence is
// "inconclusive", otherwise the axis-specific divergence. The first axis that
// diverges is reported. A case with no divergence passes.
func ReviewJudge(c JudgeCase) JudgeVerdict {
	if err := c.ValidateJudge(); err != nil {
		return judgeFail(c, JudgeMetrics{}, "", "invalid", err.Error())
	}
	m := c.metrics()
	tol := c.tolerance()

	// POSITION — order-swap stability.
	if len(c.Positions) == 0 {
		return judgeFail(c, m, Position, "inconclusive", "no order-swap probes: position stability is unmeasured")
	}
	for _, p := range c.Positions {
		if !p.stable() {
			return judgeFail(c, m, Position, "position_flip",
				fmt.Sprintf("order swap flips winner for (%s,%s): forward=%q swapped=%q", p.ItemA, p.ItemB, p.WinnerForward, p.WinnerSwapped))
		}
	}

	// REPEATABILITY — paraphrase stability.
	if len(c.Paraphrases) == 0 {
		return judgeFail(c, m, Repeatability, "inconclusive", "no paraphrase probes: repeatability is unmeasured")
	}
	for _, p := range c.Paraphrases {
		if g := absInt(p.ScoreOriginal - p.ScoreParaphrase); g > tol {
			return judgeFail(c, m, Repeatability, "paraphrase_unstable",
				fmt.Sprintf("paraphrase of %s swings score by %d > tolerance %d (%d vs %d)", p.ItemID, g, tol, p.ScoreOriginal, p.ScoreParaphrase))
		}
	}

	// VERBOSITY — no length premium.
	if len(c.Verbosities) == 0 {
		return judgeFail(c, m, Verbosity, "inconclusive", "no verbosity probes: length preference is unmeasured")
	}
	for _, p := range c.Verbosities {
		if prem := p.JudgeScoreVerbose - p.JudgeScoreConcise; prem > tol {
			return judgeFail(c, m, Verbosity, "verbosity_preference",
				fmt.Sprintf("verbose form of %s earns a +%d premium > tolerance %d (concise %d, verbose %d)", p.ItemID, prem, tol, p.JudgeScoreConcise, p.JudgeScoreVerbose))
		}
	}

	// Held-out set backs BIAS, CORRELATION and ESCALATION — an empty set makes
	// the earliest of those (BIAS) inconclusive.
	if len(c.HeldOut) == 0 {
		return judgeFail(c, m, Bias, "inconclusive", "no held-out ratings: bias, correlation and escalation are unmeasured")
	}

	// BIAS — mean signed error vs expert.
	if bt := c.biasTolerance(); absFloat(m.MeanSignedError) > bt {
		return judgeFail(c, m, Bias, "biased",
			fmt.Sprintf("mean signed judge−expert error %.3f exceeds bias tolerance %.2f over %d held-out items", m.MeanSignedError, bt, len(c.HeldOut)))
	}

	// CORRELATION — within-tolerance agreement vs expert.
	if floor := c.correlationFloor(); m.HeldOutAgreement < floor {
		return judgeFail(c, m, Correlation, "below_correlation_floor",
			fmt.Sprintf("held-out agreement %.3f < declared floor %.2f over %d items (±%d band)", m.HeldOutAgreement, floor, len(c.HeldOut), tol))
	}

	// ESCALATION — low confidence must escalate; high confidence must not diverge.
	for _, h := range c.HeldOut {
		if h.Confidence == "low" && !h.Escalated {
			return judgeFail(c, m, Escalation, "unescalated_low_confidence",
				fmt.Sprintf("held-out item %s is low-confidence but did not escalate", h.ItemID))
		}
		if h.Confidence == "high" && absInt(h.JudgeScore-h.ExpertScore) > tol {
			return judgeFail(c, m, Escalation, "overconfident_divergence",
				fmt.Sprintf("held-out item %s is high-confidence yet diverges from expert by %d > tolerance %d (judge %d, expert %d)", h.ItemID, absInt(h.JudgeScore-h.ExpertScore), tol, h.JudgeScore, h.ExpertScore))
		}
	}

	return JudgeVerdict{
		CaseID:  c.ID,
		JudgeID: c.Judge,
		Pass:    true,
		Metrics: m,
		Reason: fmt.Sprintf("judge %s clears all %d axes: position-stable, repeatable, no verbosity premium, |bias| %.3f, agreement %.3f, escalation honored",
			c.Judge, len(JudgeAxes), m.MeanSignedError, m.HeldOutAgreement),
	}
}

// judgeFail builds a non-pass verdict and its scrubbed replay artifact in one
// place, so every failure path carries an axis, a kind, and a replay.
func judgeFail(c JudgeCase, m JudgeMetrics, axis JudgeAxis, kind, reason string) JudgeVerdict {
	return JudgeVerdict{
		CaseID:          c.ID,
		JudgeID:         c.Judge,
		Pass:            false,
		FirstDivergence: axis,
		DivergenceKind:  kind,
		Reason:          reason,
		Metrics:         m,
		Replay: &JudgeReplay{
			Schema:     JudgeReplaySchema,
			CaseID:     c.ID,
			JudgeID:    scrubText(c.Judge),
			Tier:       c.Tier,
			Provenance: scrubProvenance(c.Provenance),
			Axis:       axis,
			Kind:       kind,
			Metrics:    m,
			Reason:     scrubText(reason),
		},
	}
}

// CheckJudgeGate maps a verdict to a process exit and a one-line summary — the
// report family's CheckGate shape: 0 on pass, 1 on any non-pass. A non-pass
// always carries a first divergence (or the "invalid" kind).
func CheckJudgeGate(v JudgeVerdict) (code int, summary string) {
	if v.Pass {
		return 0, fmt.Sprintf("judge %s PASS: %s", v.CaseID, v.Reason)
	}
	return 1, fmt.Sprintf("judge %s FAIL [%s @ %s]: %s", v.CaseID, v.DivergenceKind, v.FirstDivergence, v.Reason)
}

// JudgeCorpusCalibrated reports whether a set of judge cases is admissible as a
// validated judge corpus: every case validates AND passes. It fails closed — an
// empty set, an invalid case, or any non-pass case is not calibrated.
func JudgeCorpusCalibrated(cases []JudgeCase) (ok bool, reason string) {
	if len(cases) == 0 {
		return false, "empty judge corpus is not calibrated"
	}
	for _, c := range cases {
		v := ReviewJudge(c)
		if !v.Pass {
			_, s := CheckJudgeGate(v)
			return false, s
		}
	}
	return true, fmt.Sprintf("all %d judge cases pass validation", len(cases))
}

func absFloat(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

// MarshalJudgeCase / UnmarshalJudgeCase round-trip a case through its versioned
// envelope; the unmarshaler refuses an unknown schema so a mis-versioned corpus
// is rejected rather than graded.
func MarshalJudgeCase(c JudgeCase) ([]byte, error) { return json.Marshal(c) }

func UnmarshalJudgeCase(b []byte) (JudgeCase, error) {
	var c JudgeCase
	if err := json.Unmarshal(b, &c); err != nil {
		return JudgeCase{}, err
	}
	if c.Schema != JudgeSchema {
		return JudgeCase{}, fmt.Errorf("programreport: judge case schema %q, want %q", c.Schema, JudgeSchema)
	}
	return c, nil
}
