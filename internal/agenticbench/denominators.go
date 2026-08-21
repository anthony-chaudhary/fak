package agenticbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
)

const (
	FrozenCohortSchema       = "fak.agentic-benchmark-planned-cohort.v1"
	DenominatorReceiptSchema = "fak.agentic-benchmark-denominator-receipt.v1"
	DenominatorReportSchema  = "fak.agentic-benchmark-denominator-report.v1"
)

type CohortUnit struct {
	ID      string `json:"id"`
	Stratum string `json:"stratum,omitempty"`
}

type FrozenCohort struct {
	Schema string       `json:"schema"`
	Digest string       `json:"digest"`
	Units  []CohortUnit `json:"units"`
}

// FreezeCohort sorts the declared population before hashing so callers
// cannot create different cohort identities merely by enumerating the same plan
// in a different order.
func FreezeCohort(units []CohortUnit) (FrozenCohort, error) {
	canonical, err := canonicalCohortUnits(units)
	if err != nil {
		return FrozenCohort{}, err
	}
	digest, err := cohortDigest(canonical)
	if err != nil {
		return FrozenCohort{}, err
	}
	return FrozenCohort{Schema: FrozenCohortSchema, Digest: digest, Units: canonical}, nil
}

func (c FrozenCohort) Validate() error {
	if c.Schema != FrozenCohortSchema {
		return fmt.Errorf("planned cohort schema %q, want %q", c.Schema, FrozenCohortSchema)
	}
	canonical, err := canonicalCohortUnits(c.Units)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(c.Units, canonical) {
		return fmt.Errorf("planned cohort units are not in canonical ID order")
	}
	digest, err := cohortDigest(canonical)
	if err != nil {
		return err
	}
	if c.Digest != digest {
		return fmt.Errorf("planned cohort digest mismatch: got %q want %q", c.Digest, digest)
	}
	return nil
}

func canonicalCohortUnits(units []CohortUnit) ([]CohortUnit, error) {
	if len(units) == 0 {
		return nil, fmt.Errorf("planned cohort requires at least one unit")
	}
	canonical := append([]CohortUnit(nil), units...)
	for _, unit := range canonical {
		if unit.ID == "" || strings.TrimSpace(unit.ID) != unit.ID {
			return nil, fmt.Errorf("planned unit ID %q must be non-empty with no surrounding whitespace", unit.ID)
		}
		if strings.TrimSpace(unit.Stratum) != unit.Stratum {
			return nil, fmt.Errorf("planned unit %q stratum has surrounding whitespace", unit.ID)
		}
	}
	sort.Slice(canonical, func(i, j int) bool { return canonical[i].ID < canonical[j].ID })
	for i := 1; i < len(canonical); i++ {
		if canonical[i-1].ID == canonical[i].ID {
			return nil, fmt.Errorf("duplicate planned unit ID %q", canonical[i].ID)
		}
	}
	return canonical, nil
}

func cohortDigest(units []CohortUnit) (string, error) {
	payload := struct {
		Schema string       `json:"schema"`
		Units  []CohortUnit `json:"units"`
	}{Schema: FrozenCohortSchema, Units: units}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal planned cohort: %w", err)
	}
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

type FunnelStage string

const (
	StageDeclared          FunnelStage = "planned"
	StageAdmitted          FunnelStage = "admitted"
	StageAttempted         FunnelStage = "attempted"
	StageAgentTerminal     FunnelStage = "agent_terminal"
	StageEvaluatorTerminal FunnelStage = "evaluator_terminal"
	StageScored            FunnelStage = "scored"
)

var funnelStageRank = map[FunnelStage]int{
	StageDeclared:          0,
	StageAdmitted:          1,
	StageAttempted:         2,
	StageAgentTerminal:     3,
	StageEvaluatorTerminal: 4,
	StageScored:            5,
}

type MissingReason string

const (
	MissingAdmissionRefused   MissingReason = "admission_refused"
	MissingLaunchFailed       MissingReason = "launch_failed"
	MissingAgentTimeout       MissingReason = "agent_timeout"
	MissingEvaluatorFailed    MissingReason = "evaluator_failed"
	MissingArtifactIncomplete MissingReason = "artifact_incomplete"
)

var knownMissingReasons = map[MissingReason]bool{
	MissingAdmissionRefused:   true,
	MissingLaunchFailed:       true,
	MissingAgentTimeout:       true,
	MissingEvaluatorFailed:    true,
	MissingArtifactIncomplete: true,
}

type UnitResult struct {
	UnitID        string        `json:"unit_id"`
	FurthestStage FunnelStage   `json:"furthest_stage"`
	MissingReason MissingReason `json:"missing_reason,omitempty"`
	Score         *float64      `json:"score,omitempty"`
}

type ArmRun struct {
	Arm     string       `json:"arm"`
	Results []UnitResult `json:"results"`
}

type FunnelCounts struct {
	Planned           int `json:"planned"`
	Admitted          int `json:"admitted"`
	Attempted         int `json:"attempted"`
	AgentTerminal     int `json:"agent_terminal"`
	EvaluatorTerminal int `json:"evaluator_terminal"`
	Scored            int `json:"scored"`
	Missing           int `json:"missing"`
}

type ScoreSummary struct {
	Metric           string   `json:"metric"`
	Numerator        string   `json:"numerator"`
	NumeratorValue   float64  `json:"numerator_value"`
	Denominator      string   `json:"denominator"`
	DenominatorCount int      `json:"denominator_count"`
	Value            *float64 `json:"value,omitempty"`
}

type DenominatorReceipt struct {
	Schema          string                `json:"schema"`
	CohortDigest    string                `json:"cohort_digest"`
	Arm             string                `json:"arm"`
	Counts          FunnelCounts          `json:"counts"`
	MissingByReason map[MissingReason]int `json:"missing_by_reason"`
	Score           ScoreSummary          `json:"score"`
	Results         []UnitResult          `json:"results"`
}

func ReconcileArm(cohort FrozenCohort, run ArmRun) (DenominatorReceipt, error) {
	if err := cohort.Validate(); err != nil {
		return DenominatorReceipt{}, err
	}
	if run.Arm == "" || strings.TrimSpace(run.Arm) != run.Arm {
		return DenominatorReceipt{}, fmt.Errorf("arm name must be non-empty with no surrounding whitespace")
	}
	byID := make(map[string]UnitResult, len(run.Results))
	planned := make(map[string]bool, len(cohort.Units))
	for _, unit := range cohort.Units {
		planned[unit.ID] = true
	}
	for _, result := range run.Results {
		if !planned[result.UnitID] {
			return DenominatorReceipt{}, fmt.Errorf("arm %q reports unplanned unit %q", run.Arm, result.UnitID)
		}
		if _, exists := byID[result.UnitID]; exists {
			return DenominatorReceipt{}, fmt.Errorf("arm %q repeats unit %q", run.Arm, result.UnitID)
		}
		byID[result.UnitID] = result
	}

	receipt := DenominatorReceipt{
		Schema:          DenominatorReceiptSchema,
		CohortDigest:    cohort.Digest,
		Arm:             run.Arm,
		MissingByReason: make(map[MissingReason]int),
		Score: ScoreSummary{
			Metric:      "official_score_mean",
			Numerator:   "sum_official_scores",
			Denominator: "scored_rows",
		},
	}
	for _, unit := range cohort.Units {
		result, ok := byID[unit.ID]
		if !ok {
			return DenominatorReceipt{}, fmt.Errorf("arm %q missing planned result for %q", run.Arm, unit.ID)
		}
		if err := validateUnitResult(result); err != nil {
			return DenominatorReceipt{}, fmt.Errorf("arm %q unit %q: %w", run.Arm, unit.ID, err)
		}
		copyResult := result
		if result.Score != nil {
			score := *result.Score
			copyResult.Score = &score
		}
		receipt.Results = append(receipt.Results, copyResult)
		foldFunnelResult(&receipt, result)
	}
	receipt.Counts.Planned = len(cohort.Units)
	receipt.Score.DenominatorCount = receipt.Counts.Scored
	if receipt.Score.DenominatorCount > 0 {
		value := receipt.Score.NumeratorValue / float64(receipt.Score.DenominatorCount)
		receipt.Score.Value = &value
	}
	if err := receipt.Validate(); err != nil {
		return DenominatorReceipt{}, err
	}
	return receipt, nil
}

func validateUnitResult(result UnitResult) error {
	if result.UnitID == "" {
		return fmt.Errorf("unit ID is empty")
	}
	if _, ok := funnelStageRank[result.FurthestStage]; !ok {
		return fmt.Errorf("unknown furthest stage %q", result.FurthestStage)
	}
	if result.FurthestStage == StageScored {
		if result.Score == nil {
			return fmt.Errorf("scored row requires official score")
		}
		if math.IsNaN(*result.Score) || math.IsInf(*result.Score, 0) {
			return fmt.Errorf("official score must be finite")
		}
		if result.MissingReason != "" {
			return fmt.Errorf("scored row cannot carry missing reason %q", result.MissingReason)
		}
		return nil
	}
	if result.Score != nil {
		return fmt.Errorf("unscored row cannot carry official score")
	}
	if !knownMissingReasons[result.MissingReason] {
		return fmt.Errorf("unscored row requires one typed missing reason, got %q", result.MissingReason)
	}
	return nil
}

func foldFunnelResult(receipt *DenominatorReceipt, result UnitResult) {
	rank := funnelStageRank[result.FurthestStage]
	if rank >= funnelStageRank[StageAdmitted] {
		receipt.Counts.Admitted++
	}
	if rank >= funnelStageRank[StageAttempted] {
		receipt.Counts.Attempted++
	}
	if rank >= funnelStageRank[StageAgentTerminal] {
		receipt.Counts.AgentTerminal++
	}
	if rank >= funnelStageRank[StageEvaluatorTerminal] {
		receipt.Counts.EvaluatorTerminal++
	}
	if rank >= funnelStageRank[StageScored] {
		receipt.Counts.Scored++
		receipt.Score.NumeratorValue += *result.Score
		return
	}
	receipt.Counts.Missing++
	receipt.MissingByReason[result.MissingReason]++
}

func (r DenominatorReceipt) Validate() error {
	if r.Schema != DenominatorReceiptSchema {
		return fmt.Errorf("denominator receipt schema %q, want %q", r.Schema, DenominatorReceiptSchema)
	}
	if r.CohortDigest == "" || r.Arm == "" {
		return fmt.Errorf("denominator receipt requires cohort digest and arm")
	}
	c := r.Counts
	for name, count := range map[string]int{
		"planned": c.Planned, "admitted": c.Admitted, "attempted": c.Attempted,
		"agent_terminal": c.AgentTerminal, "evaluator_terminal": c.EvaluatorTerminal,
		"scored": c.Scored, "missing": c.Missing,
	} {
		if count < 0 {
			return fmt.Errorf("%s count must be non-negative", name)
		}
	}
	if c.Planned < c.Admitted || c.Admitted < c.Attempted || c.Attempted < c.AgentTerminal || c.AgentTerminal < c.EvaluatorTerminal || c.EvaluatorTerminal < c.Scored {
		return fmt.Errorf("non-monotone funnel counts: %+v", c)
	}
	if c.Missing != c.Planned-c.Scored {
		return fmt.Errorf("missing count %d does not reconcile planned %d minus scored %d", c.Missing, c.Planned, c.Scored)
	}
	missingTotal := 0
	for reason, count := range r.MissingByReason {
		if !knownMissingReasons[reason] || count < 0 {
			return fmt.Errorf("invalid missingness count %q=%d", reason, count)
		}
		missingTotal += count
	}
	if missingTotal != c.Missing {
		return fmt.Errorf("typed missingness total %d does not reconcile missing count %d", missingTotal, c.Missing)
	}
	if r.Score.Numerator != "sum_official_scores" || r.Score.Denominator != "scored_rows" {
		return fmt.Errorf("score aggregate must name sum_official_scores/scored_rows")
	}
	if r.Score.DenominatorCount != c.Scored {
		return fmt.Errorf("score denominator %d does not reconcile scored count %d", r.Score.DenominatorCount, c.Scored)
	}
	if len(r.Results) != c.Planned {
		return fmt.Errorf("result rows %d do not reconcile planned count %d", len(r.Results), c.Planned)
	}
	recomputed := DenominatorReceipt{
		MissingByReason: make(map[MissingReason]int),
		Score:           ScoreSummary{Numerator: "sum_official_scores", Denominator: "scored_rows"},
	}
	seen := make(map[string]bool, len(r.Results))
	for _, result := range r.Results {
		if seen[result.UnitID] {
			return fmt.Errorf("receipt repeats unit %q", result.UnitID)
		}
		seen[result.UnitID] = true
		if err := validateUnitResult(result); err != nil {
			return fmt.Errorf("receipt unit %q: %w", result.UnitID, err)
		}
		foldFunnelResult(&recomputed, result)
	}
	recomputed.Counts.Planned = len(r.Results)
	recomputed.Score.DenominatorCount = recomputed.Counts.Scored
	if recomputed.Counts != c {
		return fmt.Errorf("reported counts %+v do not match result rows %+v", c, recomputed.Counts)
	}
	for reason := range knownMissingReasons {
		if recomputed.MissingByReason[reason] != r.MissingByReason[reason] {
			return fmt.Errorf("reported missingness %q=%d does not match result rows %d", reason, r.MissingByReason[reason], recomputed.MissingByReason[reason])
		}
	}
	if math.Abs(recomputed.Score.NumeratorValue-r.Score.NumeratorValue) > 1e-12 {
		return fmt.Errorf("reported score numerator %g does not match result rows %g", r.Score.NumeratorValue, recomputed.Score.NumeratorValue)
	}
	if math.IsNaN(r.Score.NumeratorValue) || math.IsInf(r.Score.NumeratorValue, 0) {
		return fmt.Errorf("score numerator must be finite")
	}
	if c.Scored == 0 {
		if r.Score.Value != nil {
			return fmt.Errorf("score value must be absent when scored denominator is zero")
		}
		return nil
	}
	if r.Score.Value == nil || math.IsNaN(*r.Score.Value) || math.IsInf(*r.Score.Value, 0) {
		return fmt.Errorf("score value must be finite when scored denominator is non-zero")
	}
	want := r.Score.NumeratorValue / float64(r.Score.DenominatorCount)
	if math.Abs(*r.Score.Value-want) > 1e-12 {
		return fmt.Errorf("score value %g does not reconcile numerator %g / denominator %d", *r.Score.Value, r.Score.NumeratorValue, r.Score.DenominatorCount)
	}
	return nil
}

type ComparisonScope string

const (
	ScopeFullCohort   ComparisonScope = "full_cohort"
	ScopeScoredSubset ComparisonScope = "scored_subset"
)

type ClaimState string

const (
	ClaimAllowed               ClaimState = "ALLOWED"
	ClaimLimitedToScoredSubset ClaimState = "LIMITED_TO_SCORED_SUBSET"
	ClaimRefused               ClaimState = "REFUSED"
)

type AnalysisRule struct {
	Scope               ComparisonScope `json:"scope"`
	MaxMissingRateDelta float64         `json:"max_missing_rate_delta"`
}

type ArmComparison struct {
	LeftArm    string          `json:"left_arm"`
	RightArm   string          `json:"right_arm"`
	State      ClaimState      `json:"state"`
	Scope      ComparisonScope `json:"scope"`
	LeftScore  ScoreSummary    `json:"left_score"`
	RightScore ScoreSummary    `json:"right_score"`
	Detail     string          `json:"detail"`
}

func CompareArms(left, right DenominatorReceipt, rule AnalysisRule) ArmComparison {
	comparison := ArmComparison{
		LeftArm:    left.Arm,
		RightArm:   right.Arm,
		State:      ClaimRefused,
		Scope:      rule.Scope,
		LeftScore:  left.Score,
		RightScore: right.Score,
	}
	if err := left.Validate(); err != nil {
		comparison.Detail = fmt.Sprintf("left receipt invalid: %v", err)
		return comparison
	}
	if err := right.Validate(); err != nil {
		comparison.Detail = fmt.Sprintf("right receipt invalid: %v", err)
		return comparison
	}
	if left.CohortDigest != right.CohortDigest || left.Counts.Planned != right.Counts.Planned {
		comparison.Detail = fmt.Sprintf("planned cohorts differ: %s/%d vs %s/%d", left.CohortDigest, left.Counts.Planned, right.CohortDigest, right.Counts.Planned)
		return comparison
	}
	if left.Score.Value == nil || right.Score.Value == nil {
		comparison.Detail = "comparison requires at least one scored row in each arm"
		return comparison
	}
	missingDetail := fmt.Sprintf("%s=%d/%d missing; %s=%d/%d missing", left.Arm, left.Counts.Missing, left.Counts.Planned, right.Arm, right.Counts.Missing, right.Counts.Planned)
	switch rule.Scope {
	case ScopeFullCohort:
		if left.Counts.Missing > 0 || right.Counts.Missing > 0 {
			comparison.Detail = "planned-cohort claim refused: " + missingDetail + "; completed scores do not cover the planned population"
			return comparison
		}
		comparison.State = ClaimAllowed
		comparison.Detail = "planned cohorts and complete score denominators reconcile"
		return comparison
	case ScopeScoredSubset:
		if rule.MaxMissingRateDelta < 0 || rule.MaxMissingRateDelta > 1 || math.IsNaN(rule.MaxMissingRateDelta) {
			comparison.Detail = fmt.Sprintf("invalid max missing-rate delta %g", rule.MaxMissingRateDelta)
			return comparison
		}
		leftRate := float64(left.Counts.Missing) / float64(left.Counts.Planned)
		rightRate := float64(right.Counts.Missing) / float64(right.Counts.Planned)
		delta := math.Abs(leftRate - rightRate)
		if delta > rule.MaxMissingRateDelta+1e-12 {
			comparison.Detail = fmt.Sprintf("scored-subset claim refused: missing-rate delta %.6f exceeds declared %.6f; %s", delta, rule.MaxMissingRateDelta, missingDetail)
			return comparison
		}
		if left.Counts.Missing > 0 || right.Counts.Missing > 0 {
			comparison.State = ClaimLimitedToScoredSubset
			comparison.Detail = "claim limited to explicitly named scored subsets: " + missingDetail
			return comparison
		}
		comparison.State = ClaimAllowed
		comparison.Detail = "scored subsets equal the complete planned cohorts"
		return comparison
	default:
		comparison.Detail = fmt.Sprintf("unknown analysis scope %q", rule.Scope)
		return comparison
	}
}

type DenominatorReport struct {
	Schema      string               `json:"schema"`
	Cohort      FrozenCohort         `json:"cohort"`
	Arms        []DenominatorReceipt `json:"arms"`
	Comparisons []ArmComparison      `json:"comparisons"`
}

type ComparisonRequest struct {
	LeftArm  string       `json:"left_arm"`
	RightArm string       `json:"right_arm"`
	Rule     AnalysisRule `json:"rule"`
}

func BuildDenominatorReport(cohort FrozenCohort, runs []ArmRun, requests []ComparisonRequest) (DenominatorReport, error) {
	if err := cohort.Validate(); err != nil {
		return DenominatorReport{}, err
	}
	if len(runs) == 0 {
		return DenominatorReport{}, fmt.Errorf("denominator report requires at least one arm")
	}
	report := DenominatorReport{Schema: DenominatorReportSchema, Cohort: cohort}
	byArm := make(map[string]DenominatorReceipt, len(runs))
	for _, run := range runs {
		if _, exists := byArm[run.Arm]; exists {
			return DenominatorReport{}, fmt.Errorf("duplicate denominator arm %q", run.Arm)
		}
		receipt, err := ReconcileArm(cohort, run)
		if err != nil {
			return DenominatorReport{}, err
		}
		byArm[run.Arm] = receipt
		report.Arms = append(report.Arms, receipt)
	}
	for _, request := range requests {
		left, leftOK := byArm[request.LeftArm]
		right, rightOK := byArm[request.RightArm]
		if !leftOK || !rightOK {
			return DenominatorReport{}, fmt.Errorf("comparison arms %q/%q must name reconciled arms", request.LeftArm, request.RightArm)
		}
		report.Comparisons = append(report.Comparisons, CompareArms(left, right, request.Rule))
	}
	return report, nil
}

func FormatDenominatorMarkdown(report DenominatorReport) string {
	var b strings.Builder
	fmt.Fprintln(&b, "# Agentic Benchmark Denominator Receipt")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- Schema: `%s`\n", report.Schema)
	fmt.Fprintf(&b, "- Planned cohort: `%s`\n", report.Cohort.Digest)
	fmt.Fprintf(&b, "- Planned units: `%d`\n\n", len(report.Cohort.Units))

	fmt.Fprintln(&b, "## Execution Funnel")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| Arm | Planned | Admitted | Attempted | Agent terminal | Evaluator terminal | Scored | Missing |")
	fmt.Fprintln(&b, "|---|---:|---:|---:|---:|---:|---:|---:|")
	for _, arm := range report.Arms {
		c := arm.Counts
		fmt.Fprintf(&b, "| `%s` | %d | %d | %d | %d | %d | %d | %d |\n", arm.Arm, c.Planned, c.Admitted, c.Attempted, c.AgentTerminal, c.EvaluatorTerminal, c.Scored, c.Missing)
	}

	fmt.Fprintln(&b, "\n## Score Aggregates")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| Arm | Metric | Numerator | Numerator value | Denominator | Denominator count | Value |")
	fmt.Fprintln(&b, "|---|---|---|---:|---|---:|---:|")
	for _, arm := range report.Arms {
		value := "n/a"
		if arm.Score.Value != nil {
			value = fmt.Sprintf("%g", *arm.Score.Value)
		}
		fmt.Fprintf(&b, "| `%s` | `%s` | `%s` | %g | `%s` | %d | %s |\n", arm.Arm, arm.Score.Metric, arm.Score.Numerator, arm.Score.NumeratorValue, arm.Score.Denominator, arm.Score.DenominatorCount, value)
	}

	fmt.Fprintln(&b, "\n## Missingness")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| Arm | Reason | Count |")
	fmt.Fprintln(&b, "|---|---|---:|")
	for _, arm := range report.Arms {
		reasons := make([]string, 0, len(arm.MissingByReason))
		for reason := range arm.MissingByReason {
			reasons = append(reasons, string(reason))
		}
		sort.Strings(reasons)
		for _, reason := range reasons {
			fmt.Fprintf(&b, "| `%s` | `%s` | %d |\n", arm.Arm, reason, arm.MissingByReason[MissingReason(reason)])
		}
	}

	fmt.Fprintln(&b, "\n## Claim Gate")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| Arms | Scope | State | Detail |")
	fmt.Fprintln(&b, "|---|---|---|---|")
	for _, comparison := range report.Comparisons {
		fmt.Fprintf(&b, "| `%s` vs `%s` | `%s` | `%s` | %s |\n", comparison.LeftArm, comparison.RightArm, comparison.Scope, comparison.State, mdCell(comparison.Detail))
	}
	return b.String()
}
