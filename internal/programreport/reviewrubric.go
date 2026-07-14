package programreport

// reviewrubric.go — issue #4562, the report-quality rung of the quality-middle
// epic (#4509): an anchored expert-review rubric + disagreement process for the
// executive-summary NARRATIVE this report family emits (a folded Report's
// Reason + NextAction prose, and its internal/milestonereport sibling). Exact
// string correctness misses the failure modes a decision-support summary
// actually has — fabricated claims, omitted programs, buried priorities — so
// this contract grades six anchored 1-5 axes instead, and gates on whether a
// blinded, independently-rated seed set reaches a declared inter-rater
// agreement.
//
// It is a self-contained leaf: stdlib only, no sibling imports, so it cannot
// red architest. Four properties are enforced, not merely documented:
//
//   - ANCHORED, NOT FREE-FORM: every one of the six axes carries a fixed 1-5
//     anchor descriptor (reviewAnchors). A rating is an index into a published
//     ladder, so two raters mean the same thing by "4".
//   - FAIL-CLOSED: a case with malformed provenance, an out-of-range or missing
//     score, an unassigned tier, or an undocumented cost is REFUSED — it never
//     scores a pass. Inter-rater disagreement beyond tolerance on any axis makes
//     that axis inconclusive, and inconclusive is never a pass (the issue's
//     "missing or inconclusive evidence is never pass").
//   - FIRST DIVERGENCE + SCRUBBED REPLAY: a failing case names the FIRST
//     actionable divergence (in canonical axis order) and emits a replay
//     artifact with credential-shaped spans scrubbed, so a reviewer can re-rate
//     the exact axis that broke without leaking a secret pasted into a summary.
//   - TIERED + COSTED: each case is assigned an explicit PR / nightly / release
//     tier and must document its runtime/resource cost.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ReviewSchema is the versioned envelope tag of a review case. A reader that is
// handed a different schema refuses it rather than guessing — an unversioned or
// wrong-versioned corpus is not silently graded.
const ReviewSchema = "fak-program-report-review/1"

// ReplaySchema tags the scrubbed replay artifact a failing case emits.
const ReplaySchema = "fak-program-report-replay/1"

// AgreementTolerance is the largest absolute gap two raters may show on a 1-5
// axis and still count as agreeing. A five-point anchored scale tolerates ±1
// (adjacent anchors); a gap of 2+ is a genuine disagreement to escalate, not a
// difference to average away.
const AgreementTolerance = 1

// DeclaredAgreementFloor is the minimum pooled within-tolerance inter-rater
// agreement a blinded seed set must reach to be admissible as a calibrated
// review corpus (the issue's "blinded seed set reaches declared inter-rater
// agreement"). It is a property of the CORPUS, checked once at admission.
const DeclaredAgreementFloor = 0.80

// DefaultReviewPassFloor is the consensus rating an axis must reach for a case
// to pass when the case does not set its own PassFloor. 4 is the "adequate"
// anchor across every axis (grounded / covers every program / leads with the
// decision / runnable / clear / verdict matches); 3 and below are partial.
const DefaultReviewPassFloor = 4

// ReviewDimension is one anchored axis of the expert-review rubric.
type ReviewDimension string

const (
	Grounding     ReviewDimension = "grounding"
	Completeness  ReviewDimension = "completeness"
	Salience      ReviewDimension = "salience"
	Actionability ReviewDimension = "actionability"
	Clarity       ReviewDimension = "clarity"
	Calibration   ReviewDimension = "calibration"
)

// ReviewDimensions is the fixed, ordered axis set (#4562 scope). Canonical order
// is also the first-divergence order: a failing case is attributed to the
// earliest axis in this slice that diverges, so attribution is deterministic.
var ReviewDimensions = []ReviewDimension{
	Grounding, Completeness, Salience, Actionability, Clarity, Calibration,
}

// reviewAnchors is the published 1-5 ladder per axis. Index 0 is the anchor for
// rating 1, index 4 for rating 5. These descriptors are what makes the scale
// "anchored" — a rater picks the ladder rung the summary sits on, not a bare
// number.
var reviewAnchors = map[ReviewDimension][5]string{
	Grounding: {
		"fabricated — a claim traces to no measured signal",
		"largely ungrounded — key numbers are unsupported",
		"partially grounded — some claims cite a measured signal",
		"grounded — every material claim traces to a measured signal",
		"grounded and fenced — claims cite signals and name their honesty fences",
	},
	Completeness: {
		"omits most tracked programs",
		"omits a material program or its regression",
		"covers the programs but drops a caveat",
		"covers every tracked program and its direction",
		"covers every program, direction, and unmeasured gap",
	},
	Salience: {
		"buries the actionable item under trivia",
		"mis-orders — trivia lead, the regression trails",
		"names the salient item without leading on it",
		"leads with the most decision-relevant item",
		"leads with it and ranks the rest by impact",
	},
	Actionability: {
		"no next action at all",
		"a vague next action ('investigate')",
		"a next action naming no command or owner",
		"a runnable next action (a named command or ratchet)",
		"a runnable next action with owner and expected signal",
	},
	Clarity: {
		"contradictory or unparseable",
		"ambiguous — admits multiple readings",
		"readable but verbose or over-hedged",
		"clear and unambiguous",
		"clear, concise, and skimmable",
	},
	Calibration: {
		"verdict contradicts the measured signals",
		"overconfident — asserts done on partial evidence",
		"verdict roughly matches but mis-weights confidence",
		"verdict matches the measured state",
		"verdict matches and states confidence plus residual risk",
	},
}

// ReviewTier is the explicit run tier a case is assigned to. A case with no tier
// is refused — the issue requires an explicit PR / nightly / release assignment.
type ReviewTier string

const (
	TierPR      ReviewTier = "pr"
	TierNightly ReviewTier = "nightly"
	TierRelease ReviewTier = "release"
)

func (t ReviewTier) valid() bool {
	switch t {
	case TierPR, TierNightly, TierRelease:
		return true
	}
	return false
}

// ReviewProvenance records the replay-complete origin of a review case — the
// shared #4509 provenance contract: what produced the summary under review and
// against what baseline the ratings are anchored. Every field except the
// seed/oracle pair is required; a case missing any of them is refused (an
// unprovenanced case is never a pass).
type ReviewProvenance struct {
	Model     string `json:"model"`     // model that generated the summary under review
	Tokenizer string `json:"tokenizer"` // tokenizer id
	Engine    string `json:"engine"`    // engine/backend (e.g. "fak-gateway", "reference")
	Seed      string `json:"seed"`      // RNG seed — required unless Oracle is set
	Oracle    string `json:"oracle"`    // deterministic oracle id — required unless Seed is set
	Revision  string `json:"revision"`  // code/module revision the summary was produced at
	Baseline  string `json:"baseline"`  // tolerance/baseline provenance the ratings anchor to
}

// RaterScores is one rater's independent 1-5 score on each axis. Raters rate
// blind (they do not see each other's scores); the disagreement process below
// reconciles them. A score outside 1..5, or a missing axis, is invalid.
type RaterScores struct {
	Rater  string                  `json:"rater"`
	Scores map[ReviewDimension]int `json:"scores"`
	Note   string                  `json:"note,omitempty"`
}

// ReviewCase is one executive-summary review: the summary text under review, its
// provenance, tier and documented cost, and the blind ratings of two or more
// raters.
type ReviewCase struct {
	Schema     string           `json:"schema"`
	ID         string           `json:"id"`
	Subject    string           `json:"subject"` // the executive-summary narrative under review
	Provenance ReviewProvenance `json:"provenance"`
	Tier       ReviewTier       `json:"tier"`
	CostNote   string           `json:"cost_note"`            // documented runtime/resource cost
	Raters     []RaterScores    `json:"raters"`               // >= 2, for inter-rater agreement
	PassFloor  int              `json:"pass_floor,omitempty"` // per-axis consensus floor; 0 => DefaultReviewPassFloor
}

func (c ReviewCase) passFloor() int {
	if c.PassFloor <= 0 {
		return DefaultReviewPassFloor
	}
	return c.PassFloor
}

// Validate is the admission gate. It fails closed at every boundary so an
// ill-formed case can never reach a pass verdict: wrong schema, empty id or
// subject, incomplete provenance, an unassigned tier, an undocumented cost,
// fewer than two raters, or any out-of-range / missing axis score.
func (c ReviewCase) Validate() error {
	if c.Schema != ReviewSchema {
		return fmt.Errorf("programreport: review case schema %q, want %q", c.Schema, ReviewSchema)
	}
	if strings.TrimSpace(c.ID) == "" {
		return fmt.Errorf("programreport: review case id is required")
	}
	if strings.TrimSpace(c.Subject) == "" {
		return fmt.Errorf("programreport: review case %q has no subject summary", c.ID)
	}
	if err := c.Provenance.validate(); err != nil {
		return fmt.Errorf("programreport: review case %q provenance: %w", c.ID, err)
	}
	if !c.Tier.valid() {
		return fmt.Errorf("programreport: review case %q tier %q is not pr/nightly/release", c.ID, c.Tier)
	}
	if strings.TrimSpace(c.CostNote) == "" {
		return fmt.Errorf("programreport: review case %q must document runtime/resource cost", c.ID)
	}
	if len(c.Raters) < 2 {
		return fmt.Errorf("programreport: review case %q needs >= 2 raters for inter-rater agreement, has %d", c.ID, len(c.Raters))
	}
	for _, r := range c.Raters {
		if strings.TrimSpace(r.Rater) == "" {
			return fmt.Errorf("programreport: review case %q has an unnamed rater", c.ID)
		}
		for _, d := range ReviewDimensions {
			s, ok := r.Scores[d]
			if !ok {
				return fmt.Errorf("programreport: review case %q rater %q did not rate %q", c.ID, r.Rater, d)
			}
			if s < 1 || s > 5 {
				return fmt.Errorf("programreport: review case %q rater %q scored %q = %d, want 1..5", c.ID, r.Rater, d, s)
			}
		}
	}
	return nil
}

func (p ReviewProvenance) validate() error {
	for name, v := range map[string]string{
		"model":     p.Model,
		"tokenizer": p.Tokenizer,
		"engine":    p.Engine,
		"revision":  p.Revision,
		"baseline":  p.Baseline,
	} {
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%s is required", name)
		}
	}
	if strings.TrimSpace(p.Seed) == "" && strings.TrimSpace(p.Oracle) == "" {
		return fmt.Errorf("a seed or a deterministic oracle is required")
	}
	return nil
}

// ReviewVerdict is the adjudicated result of grading one case.
type ReviewVerdict struct {
	CaseID          string                  `json:"case_id"`
	Pass            bool                    `json:"pass"`
	Consensus       map[ReviewDimension]int `json:"consensus"` // 0 => inconclusive (raters disagreed)
	FirstDivergence ReviewDimension         `json:"first_divergence,omitempty"`
	DivergenceKind  string                  `json:"divergence_kind,omitempty"` // invalid | inconclusive | below_floor
	Reason          string                  `json:"reason"`
	Replay          *ReplayArtifact         `json:"replay,omitempty"` // present iff Pass is false
}

// ReplayArtifact is the scrubbed, self-contained record a failing case emits so
// the first divergence can be independently re-rated. Free-text spans (subject
// excerpt, rater notes) and provenance values are passed through the secret
// scrubber, so a credential pasted into a summary is redacted before replay.
type ReplayArtifact struct {
	Schema         string           `json:"schema"`
	CaseID         string           `json:"case_id"`
	Tier           ReviewTier       `json:"tier"`
	Provenance     ReviewProvenance `json:"provenance"` // scrubbed
	Dimension      ReviewDimension  `json:"dimension"`  // the first actionable divergence
	Kind           string           `json:"kind"`
	Anchor         string           `json:"anchor,omitempty"` // published anchor at the observed level
	RaterScores    map[string]int   `json:"rater_scores"`     // per-rater score on the divergent axis
	SubjectExcerpt string           `json:"subject_excerpt"`  // scrubbed
	Reason         string           `json:"reason"`
}

// Review grades one case and fails closed. Order of adjudication: (1) validation
// — a malformed case is DivergenceKind "invalid" and never passes; (2) the first
// axis (canonical order) on which raters disagree beyond tolerance is
// "inconclusive"; (3) otherwise the first axis whose consensus is below the pass
// floor is "below_floor". A case with no divergence passes.
func Review(c ReviewCase) ReviewVerdict {
	if err := c.Validate(); err != nil {
		return ReviewVerdict{
			CaseID:         c.ID,
			Pass:           false,
			DivergenceKind: "invalid",
			Reason:         err.Error(),
			Replay:         scrubbedReplay(c, "", "invalid", nil, err.Error()),
		}
	}

	consensus := make(map[ReviewDimension]int, len(ReviewDimensions))
	floor := c.passFloor()
	for _, d := range ReviewDimensions {
		scores := axisScores(c, d)
		if maxInt(scores)-minInt(scores) > AgreementTolerance {
			consensus[d] = 0 // inconclusive — raters disagree beyond tolerance
			continue
		}
		consensus[d] = medianInt(scores)
	}

	// First divergence, canonical order: inconclusive axes and below-floor axes
	// are both actionable; report whichever the earliest axis is.
	for _, d := range ReviewDimensions {
		if consensus[d] == 0 {
			reason := fmt.Sprintf("axis %q inconclusive: raters disagree beyond ±%d (%s)", d, AgreementTolerance, fmtRaterScores(c, d))
			return ReviewVerdict{
				CaseID:          c.ID,
				Pass:            false,
				Consensus:       consensus,
				FirstDivergence: d,
				DivergenceKind:  "inconclusive",
				Reason:          reason,
				Replay:          scrubbedReplay(c, d, "inconclusive", raterMap(c, d), reason),
			}
		}
		if consensus[d] < floor {
			reason := fmt.Sprintf("axis %q consensus %d < pass floor %d — %s", d, consensus[d], floor, anchorFor(d, consensus[d]))
			return ReviewVerdict{
				CaseID:          c.ID,
				Pass:            false,
				Consensus:       consensus,
				FirstDivergence: d,
				DivergenceKind:  "below_floor",
				Reason:          reason,
				Replay:          scrubbedReplay(c, d, "below_floor", raterMap(c, d), reason),
			}
		}
	}

	return ReviewVerdict{
		CaseID:    c.ID,
		Pass:      true,
		Consensus: consensus,
		Reason:    fmt.Sprintf("all %d axes at or above pass floor %d with rater agreement within ±%d", len(ReviewDimensions), floor, AgreementTolerance),
	}
}

// InterRaterAgreement is the corpus-level calibration statistic: the pooled
// fraction of rater pairs, across every axis of every case, that agree within
// tolerance. ok is false when the set carries no comparable pairs (fewer than
// two raters everywhere) — an empty set cannot "reach" an agreement floor, so it
// fails closed rather than reporting a vacuous 1.0.
func InterRaterAgreement(cases []ReviewCase) (fraction float64, pairs int, ok bool) {
	agree, total := 0, 0
	for _, c := range cases {
		for _, d := range ReviewDimensions {
			s := axisScores(c, d)
			for i := 0; i < len(s); i++ {
				for j := i + 1; j < len(s); j++ {
					total++
					if absInt(s[i]-s[j]) <= AgreementTolerance {
						agree++
					}
				}
			}
		}
	}
	if total == 0 {
		return 0, 0, false
	}
	return float64(agree) / float64(total), total, true
}

// SeedSetCalibrated reports whether a blinded seed set is admissible as a
// calibrated review corpus: every case validates AND the pooled inter-rater
// agreement reaches DeclaredAgreementFloor. It fails closed — an empty set, an
// invalid case, or too few pairs is not calibrated.
func SeedSetCalibrated(cases []ReviewCase) (ok bool, fraction float64, reason string) {
	if len(cases) == 0 {
		return false, 0, "empty seed set is not calibrated"
	}
	for _, c := range cases {
		if err := c.Validate(); err != nil {
			return false, 0, "seed set has an invalid case: " + err.Error()
		}
	}
	frac, pairs, hasPairs := InterRaterAgreement(cases)
	if !hasPairs {
		return false, 0, "seed set carries no comparable rater pairs"
	}
	if frac < DeclaredAgreementFloor {
		return false, frac, fmt.Sprintf("inter-rater agreement %.3f over %d pairs < declared floor %.2f", frac, pairs, DeclaredAgreementFloor)
	}
	return true, frac, fmt.Sprintf("inter-rater agreement %.3f over %d pairs >= declared floor %.2f", frac, pairs, DeclaredAgreementFloor)
}

// CheckReviewGate maps a verdict to a process exit and a one-line summary, the
// shape the report family's CheckGate uses: 0 on pass, 1 on any non-pass. A
// non-pass ALWAYS carries a first divergence (or the "invalid" kind), so a CI
// caller can route on the reason without re-deriving it.
func CheckReviewGate(v ReviewVerdict) (code int, summary string) {
	if v.Pass {
		return 0, fmt.Sprintf("review %s PASS: %s", v.CaseID, v.Reason)
	}
	return 1, fmt.Sprintf("review %s FAIL [%s @ %s]: %s", v.CaseID, v.DivergenceKind, v.FirstDivergence, v.Reason)
}

// SummaryOf renders the decision-support narrative a folded program report puts
// in front of an operator — the Reason + NextAction prose this rubric grades. It
// is the in-package subject the #4562 review contract is calibrated on.
func SummaryOf(r Report) string {
	return strings.TrimSpace(strings.TrimSpace(r.Reason) + " -> " + strings.TrimSpace(r.NextAction))
}

// anchorFor returns the published anchor descriptor for an axis at a level, or
// "" for the inconclusive sentinel (level 0).
func anchorFor(d ReviewDimension, level int) string {
	if level < 1 || level > 5 {
		return ""
	}
	return reviewAnchors[d][level-1]
}

// --- helpers (deterministic; no map iteration in output paths) ---

func axisScores(c ReviewCase, d ReviewDimension) []int {
	out := make([]int, 0, len(c.Raters))
	for _, r := range c.Raters {
		out = append(out, r.Scores[d])
	}
	return out
}

func raterMap(c ReviewCase, d ReviewDimension) map[string]int {
	out := make(map[string]int, len(c.Raters))
	for _, r := range c.Raters {
		out[r.Rater] = r.Scores[d]
	}
	return out
}

func fmtRaterScores(c ReviewCase, d ReviewDimension) string {
	parts := make([]string, 0, len(c.Raters))
	for _, r := range c.Raters {
		parts = append(parts, fmt.Sprintf("%s=%d", r.Rater, r.Scores[d]))
	}
	return strings.Join(parts, " ")
}

func scrubbedReplay(c ReviewCase, d ReviewDimension, kind string, raters map[string]int, reason string) *ReplayArtifact {
	level := 0
	if d != "" {
		level = medianInt(axisScores(c, d))
	}
	return &ReplayArtifact{
		Schema:         ReplaySchema,
		CaseID:         c.ID,
		Tier:           c.Tier,
		Provenance:     scrubProvenance(c.Provenance),
		Dimension:      d,
		Kind:           kind,
		Anchor:         anchorFor(d, level),
		RaterScores:    raters,
		SubjectExcerpt: scrubText(excerpt(c.Subject, 240)),
		Reason:         scrubText(reason),
	}
}

func excerpt(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// scrubProvenance redacts credential-shaped values so a replay artifact can be
// attached to an issue without leaking a secret; ids (model, tokenizer, engine,
// revision) pass through the same scrubber harmlessly.
func scrubProvenance(p ReviewProvenance) ReviewProvenance {
	return ReviewProvenance{
		Model:     scrubText(p.Model),
		Tokenizer: scrubText(p.Tokenizer),
		Engine:    scrubText(p.Engine),
		Seed:      scrubText(p.Seed),
		Oracle:    scrubText(p.Oracle),
		Revision:  scrubText(p.Revision),
		Baseline:  scrubText(p.Baseline),
	}
}

// scrubText redacts secret-shaped whitespace-delimited tokens. It is deliberately
// conservative (redacts on any of a small set of shapes) — a scrubbed artifact
// that over-redacts is safe; one that leaks a key is not.
func scrubText(s string) string {
	fields := strings.Fields(s)
	for i, f := range fields {
		if secretShaped(f) {
			fields[i] = "[REDACTED]"
		}
	}
	return strings.Join(fields, " ")
}

func secretShaped(tok string) bool {
	low := strings.ToLower(tok)
	for _, k := range []string{"secret", "token", "password", "apikey", "api_key", "bearer", "key=", "sk-"} {
		if strings.Contains(low, k) {
			return true
		}
	}
	// A long unbroken hex/base64-ish run is credential-shaped.
	if len(tok) >= 24 && isOpaque(tok) {
		return true
	}
	return false
}

func isOpaque(tok string) bool {
	for _, r := range tok {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '+', r == '/', r == '-', r == '_':
		default:
			return false
		}
	}
	return true
}

func medianInt(xs []int) int {
	if len(xs) == 0 {
		return 0
	}
	cp := append([]int(nil), xs...)
	sort.Ints(cp)
	return cp[(len(cp)-1)/2] // lower-middle for even counts — conservative
}

func minInt(xs []int) int {
	if len(xs) == 0 {
		return 0
	}
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func maxInt(xs []int) int {
	if len(xs) == 0 {
		return 0
	}
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// MarshalReviewCase / UnmarshalReviewCase round-trip a case through its versioned
// envelope; the unmarshaler refuses an unknown schema so a mis-versioned corpus
// is rejected rather than graded.
func MarshalReviewCase(c ReviewCase) ([]byte, error) { return json.Marshal(c) }

func UnmarshalReviewCase(b []byte) (ReviewCase, error) {
	var c ReviewCase
	if err := json.Unmarshal(b, &c); err != nil {
		return ReviewCase{}, err
	}
	if c.Schema != ReviewSchema {
		return ReviewCase{}, fmt.Errorf("programreport: review case schema %q, want %q", c.Schema, ReviewSchema)
	}
	return c, nil
}
