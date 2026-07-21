package livecodebench

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
)

// GradedABSchema identifies the graded raw-vs-fak pass-rate comparison artifact.
const GradedABSchema = "fak.livecodebench-graded-ab.v1"

// Graded A/B verdict tokens. Every token that reports a computed delta is
// prefixed LOCAL_ (and every gated abstain GATED_) so a verdict lifted out of
// the artifact can never be misread as the OFFICIAL lcb_runner result: this
// seam only ever produces a local-ungraded signal.
const (
	// GradedVerdictUnwitnessed: the fairness fence failed — the arms did not
	// provably run the same problems, so no delta is attributable to fak.
	GradedVerdictUnwitnessed = "GATED_UNWITNESSED"
	// GradedVerdictUngraded: no grader/sandbox was available, so the arms were
	// not graded and no pass-rate delta was computed (an honest abstain, never
	// a fabricated zero).
	GradedVerdictUngraded = "GATED_UNGRADED"
	// GradedVerdictLocalFakHigher: fak's local pass@1 exceeds raw's.
	GradedVerdictLocalFakHigher = "LOCAL_FAK_HIGHER"
	// GradedVerdictLocalRawHigher: raw's local pass@1 exceeds fak's.
	GradedVerdictLocalRawHigher = "LOCAL_RAW_HIGHER"
	// GradedVerdictLocalTie: the two arms' local pass@1 are equal (within eps).
	GradedVerdictLocalTie = "LOCAL_TIE"
)

// gradedTieEpsilon is the pass@1 band within which the two arms are called a
// tie, so floating-point dust never flips a verdict.
const gradedTieEpsilon = 1e-9

// gradedClaimBoundary is the fixed honesty boundary every graded A/B artifact
// carries. It states plainly that the delta is a LOCAL sandbox signal, not the
// official number, and names the only path that promotes it.
const gradedClaimBoundary = "LOCAL sandbox signal only: this raw-vs-fak pass-rate delta is graded by a local execution sandbox — a local-ungraded signal — NOT the official lcb_runner number, and backs no published claim. Only the official evaluator grading both arms' saved generations promotes it (export via `livecodebench export --format custom-evaluator`, grade with `python -m lcb_runner.runner.custom_evaluator`, then MarkOfficiallyGraded). Mirrors #2105/#2112."

// GradedArmComparison folds two graded arms into a fairness-fenced, sandbox-
// availability-gated raw-vs-fak pass-rate delta at the local-ungraded evidence
// tier. It is the graded companion to the token-only ArmComparison (fakarm.go):
// where that surface deliberately pins its pass-rate delta to a sentinel, this
// one grades both arms' saved generations and states the delta — but strictly
// as a LOCAL signal that can never carry a result claim (see Validate).
type GradedArmComparison struct {
	Schema   string   `json:"schema"`
	Model    string   `json:"model,omitempty"`
	Release  string   `json:"release,omitempty"`
	Scenario Scenario `json:"scenario"`

	// Raw and Fak are each arm's graded pass@1 / pass@5 summary. They are the
	// zero summary until a delta is actually computed (Delta true).
	Raw CodegenSummary `json:"raw"`
	Fak CodegenSummary `json:"fak"`

	// Pass1Delta and Pass5Delta are fak minus raw. They are meaningful only when
	// Delta is true; an abstained comparison keeps them zero (Validate enforces
	// that no number is smuggled through an abstain).
	Pass1Delta float64 `json:"pass_1_delta"`
	Pass5Delta float64 `json:"pass_5_delta"`

	// Delta reports whether a real pass-rate delta was computed: it is true only
	// when the fairness fence was witnessed AND a grader was available AND both
	// arms produced graded generations.
	Delta bool `json:"delta"`

	// FairnessWitnessed is the WitnessSameTasks verdict: the arms provably ran
	// the same problems in the same order. FairnessReason carries the mismatch
	// detail when it is false, "" when true.
	FairnessWitnessed bool   `json:"fairness_witnessed"`
	FairnessReason    string `json:"fairness_reason,omitempty"`

	// GraderAvailable reports whether a code-execution grader was supplied. When
	// false the comparison abstains rather than fabricate a delta (mirrors
	// GradeCode's no-sandbox abstain).
	GraderAvailable bool `json:"grader_available"`

	EvidenceClass      string `json:"evidence_class"`       // always EvidenceLocalUngraded
	ResultClaimAllowed bool   `json:"result_claim_allowed"` // always false
	Verdict            string `json:"verdict"`
	ClaimBoundary      string `json:"claim_boundary"`
}

// GradedArmDelta grades both arms' saved generations against the suite's tests
// and folds the two graded pass rates into a raw-vs-fak delta, fenced twice:
//
//   - Fairness first. It witnesses that both arms ran the SAME problems in the
//     same order (benchcatalog.WitnessSameTasks over each arm's recorded
//     question ids). A mismatch is a first-class non-claimable OUTCOME, not an
//     error: it returns a GATED_UNWITNESSED comparison with the reason, no delta.
//   - Grader availability next. A nil grader (no code-execution sandbox) yields
//     a GATED_UNGRADED abstain — never a fabricated zero delta — mirroring
//     GradeCode's no-sandbox behavior.
//
// Only a genuinely malformed input is an error: an arm question id the suite
// does not define (grading needs the suite's test cases), or a grading failure
// from ScoreCodegen. The result always carries EvidenceLocalUngraded and
// ResultClaimAllowed=false; this seam can never promote to the official class.
func GradedArmDelta(suite Suite, raw RawArmReport, fak FakArmReport, grade CodegenGrader) (GradedArmComparison, error) {
	c := GradedArmComparison{
		Schema:             GradedABSchema,
		Model:              firstNonEmpty(suite.Model, raw.Model, fak.Model),
		Release:            firstNonEmpty(suite.ReleaseVersion, raw.Release, fak.Release),
		Scenario:           ScenarioCodeGeneration,
		EvidenceClass:      EvidenceLocalUngraded,
		ResultClaimAllowed: false,
		ClaimBoundary:      gradedClaimBoundary,
	}

	// Fence 1: fairness. Compare each arm's RECORDED question ids positionally,
	// so a dropped, extra, or reordered problem is caught, not silently graded.
	rawIDs := questionIDs(raw.Problems)
	fakIDs := questionIDs(fak.Problems)
	same, reason := benchcatalog.WitnessSameTasks(rawIDs, fakIDs)
	c.FairnessWitnessed = same
	c.FairnessReason = reason
	if !same {
		c.Verdict = GradedVerdictUnwitnessed
		return c, nil
	}

	// Fence 2: grader availability. No sandbox ⇒ abstain, never a fabricated delta.
	if grade == nil {
		c.GraderAvailable = false
		c.Verdict = GradedVerdictUngraded
		return c, nil
	}

	// Align each arm's completions to the suite's gradeable problems by
	// question id. The fence guarantees raw.Problems[i] and fak.Problems[i]
	// share a question id, so one index walks both arms.
	suiteByID := make(map[string]Problem, len(suite.Problems))
	for _, p := range suite.Problems {
		suiteByID[p.QuestionID] = p
	}
	problems := make([]Problem, 0, len(raw.Problems))
	rawCompletions := make([][]string, 0, len(raw.Problems))
	fakCompletions := make([][]string, 0, len(fak.Problems))
	for i := range raw.Problems {
		qid := raw.Problems[i].QuestionID
		sp, ok := suiteByID[qid]
		if !ok {
			return GradedArmComparison{}, fmt.Errorf("livecodebench graded ab: arm question_id %q is not defined in the suite (cannot grade a problem with no test cases)", qid)
		}
		problems = append(problems, sp)
		rawCompletions = append(rawCompletions, raw.Problems[i].Completions)
		fakCompletions = append(fakCompletions, fak.Problems[i].Completions)
	}

	// Grade each arm with its own recorded sampling config; grading logic keys
	// off the completion counts, so N/Temperature here only stamp the report.
	rawRep, err := ScoreCodegen(CodegenConfig{Scenario: ScenarioCodeGeneration, N: raw.N, Temperature: raw.Temperature}, problems, rawCompletions, grade)
	if err != nil {
		return GradedArmComparison{}, fmt.Errorf("livecodebench graded ab: grading raw arm: %w", err)
	}
	fakRep, err := ScoreCodegen(CodegenConfig{Scenario: ScenarioCodeGeneration, N: fak.N, Temperature: fak.Temperature}, problems, fakCompletions, grade)
	if err != nil {
		return GradedArmComparison{}, fmt.Errorf("livecodebench graded ab: grading fak arm: %w", err)
	}

	c.GraderAvailable = true
	c.Delta = true
	c.Raw = rawRep.Summary
	c.Fak = fakRep.Summary
	c.Pass1Delta = fakRep.Summary.Pass1 - rawRep.Summary.Pass1
	c.Pass5Delta = fakRep.Summary.Pass5 - rawRep.Summary.Pass5
	switch {
	case c.Pass1Delta > gradedTieEpsilon:
		c.Verdict = GradedVerdictLocalFakHigher
	case c.Pass1Delta < -gradedTieEpsilon:
		c.Verdict = GradedVerdictLocalRawHigher
	default:
		c.Verdict = GradedVerdictLocalTie
	}
	return c, nil
}

// Validate encodes the honesty invariant structurally: this artifact type can
// never carry a claimable pass-rate delta. It is always local-ungraded and
// unclaimable; a computed delta requires both fences to have passed and both
// arms to have graded generations; an abstain must carry no delta.
func (c GradedArmComparison) Validate() error {
	if c.Schema != GradedABSchema {
		return fmt.Errorf("livecodebench graded ab: schema %q, want %q", c.Schema, GradedABSchema)
	}
	if c.EvidenceClass != EvidenceLocalUngraded {
		return fmt.Errorf("livecodebench graded ab: evidence_class %q, want %q (a graded local delta is never the official class)", c.EvidenceClass, EvidenceLocalUngraded)
	}
	if c.ResultClaimAllowed {
		return fmt.Errorf("livecodebench graded ab: result_claim_allowed must be false (a local sandbox delta backs no published claim)")
	}
	if c.Delta {
		if !c.FairnessWitnessed {
			return fmt.Errorf("livecodebench graded ab: a delta requires the fairness fence to be witnessed")
		}
		if !c.GraderAvailable {
			return fmt.Errorf("livecodebench graded ab: a delta requires an available grader")
		}
		if c.Raw.Graded == 0 || c.Fak.Graded == 0 {
			return fmt.Errorf("livecodebench graded ab: a delta requires both arms to have graded generations (raw graded=%d fak graded=%d)", c.Raw.Graded, c.Fak.Graded)
		}
		return nil
	}
	if c.Pass1Delta != 0 || c.Pass5Delta != 0 {
		return fmt.Errorf("livecodebench graded ab: an abstained comparison must not carry a pass-rate delta (pass1=%v pass5=%v)", c.Pass1Delta, c.Pass5Delta)
	}
	return nil
}

// RenderGradedABMarkdown renders the comparison as a compact human report. It
// leads with the claim boundary so a reader can never lift the delta out of
// context, and prints the fence/grader status before any number.
func RenderGradedABMarkdown(c GradedArmComparison) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# LiveCodeBench graded A/B (raw vs fak)\n\n")
	fmt.Fprintf(&b, "- verdict: **%s**\n", c.Verdict)
	fmt.Fprintf(&b, "- evidence_class: `%s`\n", c.EvidenceClass)
	fmt.Fprintf(&b, "- result_claim_allowed: %t\n", c.ResultClaimAllowed)
	if c.Model != "" {
		fmt.Fprintf(&b, "- model: `%s`\n", c.Model)
	}
	if c.Release != "" {
		fmt.Fprintf(&b, "- release: `%s`\n", c.Release)
	}
	fmt.Fprintf(&b, "- fairness_witnessed: %t\n", c.FairnessWitnessed)
	if c.FairnessReason != "" {
		fmt.Fprintf(&b, "- fairness_reason: %s\n", c.FairnessReason)
	}
	fmt.Fprintf(&b, "- grader_available: %t\n\n", c.GraderAvailable)
	fmt.Fprintf(&b, "> %s\n\n", c.ClaimBoundary)
	if c.Delta {
		fmt.Fprintf(&b, "| arm | problems | graded | pass@1 | pass@5 |\n")
		fmt.Fprintf(&b, "| --- | --- | --- | --- | --- |\n")
		fmt.Fprintf(&b, "| raw | %d | %d | %.4f | %.4f |\n", c.Raw.Problems, c.Raw.Graded, c.Raw.Pass1, c.Raw.Pass5)
		fmt.Fprintf(&b, "| fak | %d | %d | %.4f | %.4f |\n", c.Fak.Problems, c.Fak.Graded, c.Fak.Pass1, c.Fak.Pass5)
		fmt.Fprintf(&b, "| **delta (fak−raw)** | | | **%+.4f** | **%+.4f** |\n", c.Pass1Delta, c.Pass5Delta)
	} else {
		fmt.Fprintf(&b, "_No pass-rate delta was computed (%s)._\n", c.Verdict)
	}
	return b.String()
}

// questionIDs collects the recorded question ids of an arm's per-problem rows,
// in order, for the fairness fence.
func questionIDs(problems []RawArmProblem) []string {
	ids := make([]string, len(problems))
	for i, p := range problems {
		ids[i] = p.QuestionID
	}
	return ids
}

// firstNonEmpty returns the first argument with non-whitespace content, or "".
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
