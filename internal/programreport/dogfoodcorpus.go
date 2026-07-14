package programreport

// dogfoodcorpus.go — issue #4550, the versioned executive-report dogfood corpus
// of the quality-middle epic (#4509). Its siblings in this package supply the
// grading machinery: reviewrubric.go (#4562) grades one executive summary on
// six anchored axes, and judgevalidation.go (#4563) validates an LLM judge as a
// stand-in for the expert raters. This leaf supplies the thing they grade: a
// VERSIONED corpus of representative dogfood cases — the repo's own
// executive-report surfaces (a folded program report's Reason + NextAction, its
// internal/milestonereport sibling, docs/EXECUTIVE-ROLLUP.md) captured with
// their project state, evidence, decisions, ambiguity, and known defects.
//
// The corpus does NOT freeze prose. A case's expectation is a graded PROPERTY
// (the anchored-rubric verdict and, for a suspected defect, the axis of the
// first actionable divergence) — never an exact string of the summary. Two
// differently-worded summaries with the same recorded ratings grade
// identically; the test suite proves it by rewording a passing case.
//
// The #4550 properties, enforced rather than documented:
//
//   - FAILING-BEFORE-FIX: every currently-suspected report defect is captured
//     as an Expect == "fail" case naming its suspected axis and defect. Grading
//     verifies the defect still reproduces; a defect case that quietly starts
//     passing is a corpus-level FAILURE (either the defect was fixed — re-capture
//     and flip the expectation at a new corpus revision — or the rubric lost
//     sensitivity). A before-fix case can never rot into a silent green.
//   - REPLAY-COMPLETE: every case reuses the shared #4509 provenance contract
//     (ReviewProvenance: model, tokenizer, engine/backend, seed or deterministic
//     oracle, code/module revision, tolerance/baseline provenance), enforced by
//     the same ReviewCase.Validate the #4562 rung uses, so the two rungs never
//     disagree on what a replay-complete case is.
//   - FIRST DIVERGENCE + SCRUBBED REPLAY: a failing or diverging case surfaces
//     the first actionable divergence and emits the same scrubbed replay
//     artifact reviewrubric.go emits; missing or inconclusive evidence is never
//     a pass (inherited from Review's fail-closed adjudication and re-enforced
//     at every corpus admission boundary).
//   - TIERED + COSTED: every case carries an explicit PR / nightly / release
//     tier and a documented runtime/resource cost. Grading the corpus itself is
//     pure and in-memory (stdlib only, no process, sub-millisecond) — the cost
//     that matters is the recorded human-rating cost each case documents.
//   - VERSIONED: the corpus carries a schema tag, a revision, and a content
//     digest. A serialized corpus whose content was mutated after capture, or
//     whose schema is unknown, is REFUSED rather than graded.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// DogfoodCorpusSchema is the versioned envelope tag of the dogfood corpus. A
// reader handed a different schema refuses it rather than guessing.
const DogfoodCorpusSchema = "fak-program-report-dogfood-corpus/1"

// Expectation values a dogfood case may declare. ExpectFail marks a currently
// suspected report defect (a failing-before-fix case); ExpectPass marks a
// summary the raters judged adequate.
const (
	ExpectPass = "pass"
	ExpectFail = "fail"
)

// StateCapture is the representative project state a case's summary was written
// against — the #4550 scope's "project states, evidence, decisions, ambiguity,
// and known defects". It is the context a re-rater needs to judge grounding,
// completeness, and calibration without re-deriving the repo state; it never
// constrains the summary's wording.
type StateCapture struct {
	Programs     []string `json:"programs"`                // tracked programs in scope of the summary
	Evidence     []string `json:"evidence"`                // measured signals available when the summary was written (>= 1 required)
	Decisions    []string `json:"decisions"`               // the operator decision(s) the summary must support (>= 1 required)
	Ambiguity    []string `json:"ambiguity,omitempty"`     // genuinely unresolved facts an honest summary hedges rather than resolves
	KnownDefects []string `json:"known_defects,omitempty"` // known state defects a complete summary must surface
}

func (s StateCapture) validate() error {
	if len(s.Evidence) == 0 {
		return fmt.Errorf("state capture records no evidence")
	}
	if len(s.Decisions) == 0 {
		return fmt.Errorf("state capture records no decision to support")
	}
	return nil
}

// DogfoodCase is one corpus entry: a review case (summary + provenance + tier +
// cost + blind ratings, the #4562 contract) plus the captured project state it
// summarizes and the property the corpus expects of it.
type DogfoodCase struct {
	Review ReviewCase   `json:"review"`
	State  StateCapture `json:"state"`
	// Expect is ExpectPass or ExpectFail. An ExpectFail case is a currently
	// suspected defect: it MUST fail with ExpectDivergence as its first
	// actionable divergence until the underlying report defect is fixed and the
	// case is re-captured at a new corpus revision.
	Expect string `json:"expect"`
	// ExpectDivergence names the suspected defect's axis (required iff
	// Expect == ExpectFail).
	ExpectDivergence ReviewDimension `json:"expect_divergence,omitempty"`
	// DefectRef names the suspected defect an ExpectFail case tracks — the
	// failure mode observed on the report surface, so the case stays actionable
	// after the person who captured it moves on.
	DefectRef string `json:"defect_ref,omitempty"`
}

func (c DogfoodCase) validate() error {
	if err := c.Review.Validate(); err != nil {
		return err
	}
	if err := c.State.validate(); err != nil {
		return fmt.Errorf("case %q: %v", c.Review.ID, err)
	}
	switch c.Expect {
	case ExpectPass:
		if c.ExpectDivergence != "" {
			return fmt.Errorf("case %q: an expected-pass case must not declare a divergence axis", c.Review.ID)
		}
	case ExpectFail:
		if !validDimension(c.ExpectDivergence) {
			return fmt.Errorf("case %q: expected-fail case must name a suspected axis, got %q", c.Review.ID, c.ExpectDivergence)
		}
		if strings.TrimSpace(c.DefectRef) == "" {
			return fmt.Errorf("case %q: expected-fail case must name the suspected defect it tracks", c.Review.ID)
		}
	default:
		return fmt.Errorf("case %q: expect %q is not %q/%q", c.Review.ID, c.Expect, ExpectPass, ExpectFail)
	}
	return nil
}

func validDimension(d ReviewDimension) bool {
	for _, k := range ReviewDimensions {
		if d == k {
			return true
		}
	}
	return false
}

// DogfoodCorpus is the versioned corpus envelope: schema, revision, the cases,
// and a content digest so a serialized corpus mutated after capture is refused
// rather than graded.
type DogfoodCorpus struct {
	Schema   string        `json:"schema"`
	Revision string        `json:"revision"` // corpus content revision, bumped on any case change
	Cases    []DogfoodCase `json:"cases"`
	Digest   string        `json:"digest"`
}

// DogfoodCorpusDigest computes the canonical content digest of a corpus (the
// Digest field itself excluded). Case order is significant — it is the
// first-divergence order — so it is hashed as-is, not sorted.
func DogfoodCorpusDigest(c DogfoodCorpus) string {
	c.Digest = ""
	b, _ := json.Marshal(c)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// Validate is the corpus admission gate; it fails closed at every boundary so
// an ill-formed corpus can never be graded: wrong schema, missing revision,
// stale or tampered digest, no cases, duplicate case ids, an invalid case
// (schema/provenance/tier/cost/raters — the #4562 boundaries), an ill-formed
// expectation, or a blinded rater set below the declared inter-rater agreement
// floor (an uncalibrated corpus is not evidence).
func (c DogfoodCorpus) Validate() error {
	if c.Schema != DogfoodCorpusSchema {
		return fmt.Errorf("programreport: dogfood corpus schema %q, want %q", c.Schema, DogfoodCorpusSchema)
	}
	if strings.TrimSpace(c.Revision) == "" {
		return fmt.Errorf("programreport: dogfood corpus revision is required")
	}
	if len(c.Cases) == 0 {
		return fmt.Errorf("programreport: dogfood corpus is empty")
	}
	if c.Digest != DogfoodCorpusDigest(c) {
		return fmt.Errorf("programreport: dogfood corpus digest mismatch — content mutation or stale digest")
	}
	seen := map[string]bool{}
	reviews := make([]ReviewCase, 0, len(c.Cases))
	for _, dc := range c.Cases {
		if err := dc.validate(); err != nil {
			return fmt.Errorf("programreport: dogfood corpus: %v", err)
		}
		if seen[dc.Review.ID] {
			return fmt.Errorf("programreport: dogfood corpus: duplicate case id %q", dc.Review.ID)
		}
		seen[dc.Review.ID] = true
		reviews = append(reviews, dc.Review)
	}
	if ok, _, reason := SeedSetCalibrated(reviews); !ok {
		return fmt.Errorf("programreport: dogfood corpus is not calibrated: %s", reason)
	}
	return nil
}

// DogfoodCaseResult is one graded corpus entry: the rubric verdict plus whether
// the case matched its declared expectation. Holding means an expected-fail
// case still fails on its suspected axis — the defect reproduces, which is the
// healthy before-fix state.
type DogfoodCaseResult struct {
	CaseID   string        `json:"case_id"`
	Expect   string        `json:"expect"`
	Verdict  ReviewVerdict `json:"verdict"`
	Holding  bool          `json:"holding"`
	Diverged bool          `json:"diverged"`
	Reason   string        `json:"reason"`
}

// DogfoodCorpusResult is the corpus-level grade. Pass means every case matched
// its expectation: passing cases pass, and every suspected defect still
// reproduces on its declared axis. The first case (corpus order) that diverges
// from its expectation sets FirstDivergence and carries a scrubbed replay.
type DogfoodCorpusResult struct {
	Schema          string              `json:"schema"`
	Revision        string              `json:"revision"`
	Pass            bool                `json:"pass"`
	Cases           []DogfoodCaseResult `json:"cases"`
	FirstDivergence string              `json:"first_divergence,omitempty"` // case id
	Reason          string              `json:"reason"`
	Replay          *ReplayArtifact     `json:"replay,omitempty"` // present iff Pass is false
}

// GradeDogfoodCorpus grades every case with the #4562 Review contract and
// adjudicates each against its expectation. It fails closed: an invalid corpus
// never passes, an expected-fail case that PASSES is a corpus failure (a
// silently-green before-fix case), and an expected-fail case whose first
// actionable divergence moved off the suspected axis is a corpus failure
// (attribution drifted — the case no longer tracks its defect).
func GradeDogfoodCorpus(c DogfoodCorpus) DogfoodCorpusResult {
	if err := c.Validate(); err != nil {
		return DogfoodCorpusResult{
			Schema:   DogfoodCorpusSchema,
			Revision: c.Revision,
			Pass:     false,
			Reason:   err.Error(),
		}
	}

	res := DogfoodCorpusResult{Schema: DogfoodCorpusSchema, Revision: c.Revision, Pass: true}
	for _, dc := range c.Cases {
		v := Review(dc.Review)
		row := DogfoodCaseResult{CaseID: dc.Review.ID, Expect: dc.Expect, Verdict: v}
		switch {
		case dc.Expect == ExpectPass && v.Pass:
			row.Reason = "passes as expected"
		case dc.Expect == ExpectFail && !v.Pass && v.FirstDivergence == dc.ExpectDivergence:
			row.Holding = true
			row.Reason = fmt.Sprintf("suspected defect (%s) still reproduces at %q — failing before fix", dc.DefectRef, dc.ExpectDivergence)
		case dc.Expect == ExpectFail && v.Pass:
			row.Diverged = true
			row.Reason = fmt.Sprintf("suspected defect (%s) no longer reproduces: case passed before a recorded fix — re-capture at a new corpus revision or the rubric lost sensitivity", dc.DefectRef)
			if res.Pass {
				res.Pass = false
				res.FirstDivergence = dc.Review.ID
				res.Reason = row.Reason
				res.Replay = scrubbedReplay(dc.Review, dc.ExpectDivergence, "expectation", raterMap(dc.Review, dc.ExpectDivergence), row.Reason)
			}
		case dc.Expect == ExpectFail && !v.Pass:
			row.Diverged = true
			row.Reason = fmt.Sprintf("first actionable divergence moved from suspected %q to %q (%s)", dc.ExpectDivergence, v.FirstDivergence, v.Reason)
			if res.Pass {
				res.Pass = false
				res.FirstDivergence = dc.Review.ID
				res.Reason = row.Reason
				res.Replay = v.Replay
			}
		default: // ExpectPass && !v.Pass
			row.Diverged = true
			row.Reason = fmt.Sprintf("expected-pass case failed: %s", v.Reason)
			if res.Pass {
				res.Pass = false
				res.FirstDivergence = dc.Review.ID
				res.Reason = row.Reason
				res.Replay = v.Replay
			}
		}
		res.Cases = append(res.Cases, row)
	}
	if res.Pass {
		res.Reason = fmt.Sprintf("all %d cases match their expectations (every suspected defect still failing before fix)", len(res.Cases))
	}
	return res
}

// CheckDogfoodGate maps a corpus result to a process exit and a one-line
// summary, the shape the report family's CheckGate uses: 0 on pass, 1 on any
// non-pass. A non-pass always names the first diverging case (or the admission
// failure), so a CI caller can route on the reason without re-deriving it.
func CheckDogfoodGate(r DogfoodCorpusResult) (code int, summary string) {
	if r.Pass {
		return 0, fmt.Sprintf("dogfood corpus %s PASS: %s", r.Revision, r.Reason)
	}
	at := r.FirstDivergence
	if at == "" {
		at = "admission"
	}
	return 1, fmt.Sprintf("dogfood corpus %s FAIL [%s]: %s", r.Revision, at, r.Reason)
}

// MarshalDogfoodCorpus / UnmarshalDogfoodCorpus round-trip the corpus through
// its versioned envelope; the unmarshaler refuses an unknown schema so a
// mis-versioned corpus is rejected rather than graded.
func MarshalDogfoodCorpus(c DogfoodCorpus) ([]byte, error) { return json.Marshal(c) }

func UnmarshalDogfoodCorpus(b []byte) (DogfoodCorpus, error) {
	var c DogfoodCorpus
	if err := json.Unmarshal(b, &c); err != nil {
		return DogfoodCorpus{}, err
	}
	if c.Schema != DogfoodCorpusSchema {
		return DogfoodCorpus{}, fmt.Errorf("programreport: dogfood corpus schema %q, want %q", c.Schema, DogfoodCorpusSchema)
	}
	return c, nil
}

// SeedDogfoodCorpus is corpus revision 2026-07-13.1: five representative cases
// captured from this repo's executive-report surfaces (the programreport /
// milestonereport rollup narratives graded by #4562). The three ExpectFail
// cases are the currently-suspected defect classes observed on those surfaces —
// unsupported claims, omitted regressions, buried priorities (the exact failure
// modes #4550 names as what exact-string correctness misses) — captured as
// failing-before-fix cases. The two ExpectPass cases pin the adequate shape,
// including one whose state carries genuine ambiguity the summary must hedge.
// Tiers span pr / nightly / release; every case documents its rating cost.
func SeedDogfoodCorpus() DogfoodCorpus {
	prov := ReviewProvenance{
		Model:     "claude-opus-4-8",
		Tokenizer: "claude-bpe-v2",
		Engine:    "fak-gateway",
		Oracle:    "recorded-blind-ratings/2026-07-13", // ratings are recorded, so replay is deterministic
		Revision:  "f7f1ec71d",
		Baseline:  "programreport-review-baseline/2026-07",
	}
	rate := func(rater string, g, co, sa, ac, cl, ca int) RaterScores {
		return RaterScores{Rater: rater, Scores: map[ReviewDimension]int{
			Grounding: g, Completeness: co, Salience: sa, Actionability: ac, Clarity: cl, Calibration: ca,
		}}
	}

	c := DogfoodCorpus{
		Schema:   DogfoodCorpusSchema,
		Revision: "2026-07-13.1",
		Cases: []DogfoodCase{
			{
				Review: ReviewCase{
					Schema:     ReviewSchema,
					ID:         "DOG-grounding-unsupported-speedup",
					Subject:    "Kernel program is 40% faster this week and cache is fully optimized; no operator action needed.",
					Provenance: prov,
					Tier:       TierPR,
					CostNote:   "2 blind raters, ~6 min/case; grading pure in-memory, <1ms",
					Raters: []RaterScores{
						rate("rater-a", 1, 3, 3, 2, 4, 2),
						rate("rater-b", 1, 3, 2, 2, 4, 1),
					},
				},
				State: StateCapture{
					Programs:  []string{"kernel-optimization", "cache-optimization", "human-operator-effectiveness"},
					Evidence:  []string{"perf-lane ship window: 3 ships / 7d (activity proxy only — no tok/s claim)", "cache-value ledger reuse ratio 0.62 (trend gate green)"},
					Decisions: []string{"does any program need operator attention this week?"},
				},
				Expect:           ExpectFail,
				ExpectDivergence: Grounding,
				DefectRef:        "rollup asserts a speedup number no ledger row supports (docs/EXECUTIVE-ROLLUP.md unsupported-claim class)",
			},
			{
				Review: ReviewCase{
					Schema:     ReviewSchema,
					ID:         "DOG-completeness-dropped-regression",
					Subject:    "Kernel program active (3 ships in window); cache reuse holding at 0.62 against the trend gate. Next: hold ratchets.",
					Provenance: prov,
					Tier:       TierNightly,
					CostNote:   "2 blind raters, ~6 min/case; grading pure in-memory, <1ms",
					Raters: []RaterScores{
						rate("rater-a", 4, 2, 3, 4, 4, 3),
						rate("rater-b", 4, 2, 3, 4, 5, 3),
					},
				},
				State: StateCapture{
					Programs:     []string{"kernel-optimization", "cache-optimization", "human-operator-effectiveness"},
					Evidence:     []string{"perf-lane ship window: 3 ships / 7d", "cache-value ledger reuse ratio 0.62", "operator-heaviness pressure 74 with hard heaviness_debt present"},
					Decisions:    []string{"does any program need operator attention this week?"},
					KnownDefects: []string{"operator-effectiveness frontier forced to zero by hard heaviness_debt — regression in state"},
				},
				Expect:           ExpectFail,
				ExpectDivergence: Completeness,
				DefectRef:        "rollup silently drops the regressed operator-effectiveness program (omission class)",
			},
			{
				Review: ReviewCase{
					Schema:     ReviewSchema,
					ID:         "DOG-salience-buried-regression",
					Subject:    "Docs churn is down, three lanes renamed cleanly, ledger format stable. Also cache reuse fell below its trend gate. Next: tidy remaining renames.",
					Provenance: prov,
					Tier:       TierNightly,
					CostNote:   "2 blind raters, ~6 min/case; grading pure in-memory, <1ms",
					Raters: []RaterScores{
						rate("rater-a", 4, 4, 1, 2, 4, 3),
						rate("rater-b", 4, 4, 2, 3, 4, 3),
					},
				},
				State: StateCapture{
					Programs:     []string{"kernel-optimization", "cache-optimization", "human-operator-effectiveness"},
					Evidence:     []string{"cache-value ledger reuse ratio 0.41, below trend-gate floor 0.55", "doc-churn scorecard improved"},
					Decisions:    []string{"what is the single most decision-relevant item this week?"},
					KnownDefects: []string{"cache-reuse regression is the actionable item; doc churn is trivia"},
				},
				Expect:           ExpectFail,
				ExpectDivergence: Salience,
				DefectRef:        "rollup leads with trivia and buries the cache regression (prioritization class)",
			},
			{
				Review: ReviewCase{
					Schema:     ReviewSchema,
					ID:         "DOG-pass-grounded-rollup",
					Subject:    "Cache reuse regressed to 0.41 vs floor 0.55 (cache-value ledger) — lead item. Kernel window active (3 ships/7d, activity proxy). Operator heaviness holding. Next: run `fak cachevalue report` and hold the trend ratchet; owner: cache lane.",
					Provenance: prov,
					Tier:       TierPR,
					CostNote:   "2 blind raters, ~6 min/case; grading pure in-memory, <1ms",
					Raters: []RaterScores{
						rate("rater-a", 5, 4, 5, 5, 4, 4),
						rate("rater-b", 4, 4, 5, 4, 4, 4),
					},
				},
				State: StateCapture{
					Programs:  []string{"kernel-optimization", "cache-optimization", "human-operator-effectiveness"},
					Evidence:  []string{"cache-value ledger reuse ratio 0.41, floor 0.55", "perf-lane ship window: 3 ships / 7d", "operator-heaviness pressure 12, no debt"},
					Decisions: []string{"what is the single most decision-relevant item this week?"},
				},
				Expect: ExpectPass,
			},
			{
				Review: ReviewCase{
					Schema:     ReviewSchema,
					ID:         "DOG-pass-ambiguity-hedged",
					Subject:    "Kernel window is quiet (0 ships/7d): HOLDING or stalled is not yet decidable from the activity proxy — do not read it as regression. Next: check the perf-parity RSI loop's last run before acting; cache and operator programs unchanged.",
					Provenance: prov,
					Tier:       TierRelease,
					CostNote:   "2 blind raters, ~7 min/case (ambiguity requires reading the state capture); grading pure in-memory, <1ms",
					Raters: []RaterScores{
						rate("rater-a", 4, 4, 4, 4, 4, 5),
						rate("rater-b", 4, 5, 4, 4, 4, 5),
					},
				},
				State: StateCapture{
					Programs:  []string{"kernel-optimization", "cache-optimization", "human-operator-effectiveness"},
					Evidence:  []string{"perf-lane ship window: 0 ships / 7d", "cache-value ledger reuse ratio 0.62", "operator-heaviness pressure 12, no debt"},
					Decisions: []string{"is the quiet kernel window a regression?"},
					Ambiguity: []string{"a quiet ship window is HOLDING, not regressed — the activity proxy cannot distinguish idle-by-design from stalled"},
				},
				Expect: ExpectPass,
			},
		},
	}
	c.Digest = DogfoodCorpusDigest(c)
	return c
}
