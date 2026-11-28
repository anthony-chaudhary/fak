package rsiloop

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

const (
	// LoopVariantArchiveSchema tags kept-variant archive rows. The archive is JSONL
	// so each KEEP becomes one append-only stepping-stone record.
	LoopVariantArchiveSchema = "fak-loopvariant-archive/1"
	// LoopVariantMetricName is the shipgate metric for the metaloop objective. It is
	// a fold over the task-set spec/oracle, not a raw efficiency metric.
	LoopVariantMetricName = "spec_oracle_points"
)

// VariantKind names the loop surface a metaloop candidate changes.
type VariantKind string

const (
	VariantPrompt          VariantKind = "prompt"
	VariantToolPolicy      VariantKind = "tool_policy"
	VariantIterationPolicy VariantKind = "iteration_policy"
)

// LoopVariant is one proposed loop/prompt/tool-policy diff over a baseline loop
// config. Diff is deliberately opaque to this package; the driver owns how it is
// applied in an isolated worktree, while this package owns the keep/revert fold.
type LoopVariant struct {
	ID          string      `json:"id"`
	Kind        VariantKind `json:"kind"`
	Description string      `json:"description,omitempty"`
	Diff        string      `json:"diff,omitempty"`
}

// LoopConfig is the baseline loop surface a metaloop proposer may change. The
// fields are deliberately textual: applying a prompt/tool/iteration-policy diff is
// driver-owned, while this package owns the evidence gate after the driver measures
// the variant.
type LoopConfig struct {
	Prompt          string `json:"prompt,omitempty"`
	ToolPolicy      string `json:"tool_policy,omitempty"`
	IterationPolicy string `json:"iteration_policy,omitempty"`
}

// LoopVariantProposer emits candidate diffs over a baseline loop config. The
// archive argument lets a DGM/SICA-style proposer reuse kept stepping stones without
// trusting them: every new candidate still has to pass EvaluateLoopVariant.
type LoopVariantProposer interface {
	ProposeLoopVariants(baseline LoopConfig, archive []LoopVariantRecord) ([]LoopVariant, error)
}

// StaticLoopVariantProposer is the deterministic proposer used by fixtures and
// simple drivers that already generated candidate diffs elsewhere.
type StaticLoopVariantProposer []LoopVariant

// ProposeLoopVariants returns a defensive copy of the static candidates.
func (p StaticLoopVariantProposer) ProposeLoopVariants(LoopConfig, []LoopVariantRecord) ([]LoopVariant, error) {
	return append([]LoopVariant(nil), p...), nil
}

// DOSEvidence is the non-author read-back attached to a task result. In the live
// driver this is the `dos verify` / `dos improve --observe` receipt for the task.
type DOSEvidence struct {
	ID        string `json:"id"`
	Subject   string `json:"subject,omitempty"`
	Confirmed bool   `json:"confirmed"`
	Summary   string `json:"summary,omitempty"`
}

// TaskWitness is the re-measured outcome of one fixed task under one loop config.
// SpecPassed is the oracle verdict. SpecPoints is credited only when SpecPassed is
// true, so raw metric wins (Turns, LatencyMS, etc.) cannot by themselves move KEEP.
type TaskWitness struct {
	TaskID     string      `json:"task_id"`
	SpecPassed bool        `json:"spec_passed"`
	SpecPoints float64     `json:"spec_points,omitempty"`
	Turns      int         `json:"turns,omitempty"`
	LatencyMS  int         `json:"latency_ms,omitempty"`
	Evidence   DOSEvidence `json:"dos_evidence"`
	Note       string      `json:"note,omitempty"`
}

// TaskSetWitness is the fixed task-set read-back for one baseline or candidate run.
type TaskSetWitness struct {
	Ref   string        `json:"ref,omitempty"`
	Tasks []TaskWitness `json:"tasks"`
}

// SpecFold is the objective fold used for the keep-bit. Raw metrics stay outside
// this structure by design; a shorter run that fails the oracle has zero points.
type SpecFold struct {
	Tasks       int      `json:"tasks"`
	Passed      int      `json:"passed"`
	Points      float64  `json:"points"`
	SpecGreen   bool     `json:"spec_green"`
	TruthClean  bool     `json:"truth_clean"`
	EvidenceIDs []string `json:"evidence_ids,omitempty"`
}

// LoopVariantRecord is the durable archive row for a kept stepping-stone variant,
// and also the in-memory verdict returned for a reverted candidate.
type LoopVariantRecord struct {
	Schema       string           `json:"schema"`
	Variant      LoopVariant      `json:"variant"`
	Decision     string           `json:"decision"`
	Kept         bool             `json:"kept"`
	MetricName   string           `json:"metric_name"`
	Before       SpecFold         `json:"before"`
	After        SpecFold         `json:"after"`
	Score        *Scorecard       `json:"score,omitempty"`
	Witness      shipgate.Witness `json:"witness"`
	DOSEvidence  []DOSEvidence    `json:"dos_evidence"`
	Tasks        []TaskWitness    `json:"tasks"`
	BaselineRef  string           `json:"baseline_ref,omitempty"`
	CandidateRef string           `json:"candidate_ref,omitempty"`
}

// FoldTaskSet folds one task-set witness into the spec/oracle objective. Passing
// tasks with no explicit SpecPoints count as one point; failed tasks count as zero.
func FoldTaskSet(ts TaskSetWitness) SpecFold {
	f := SpecFold{Tasks: len(ts.Tasks), SpecGreen: len(ts.Tasks) > 0, TruthClean: len(ts.Tasks) > 0}
	for _, t := range ts.Tasks {
		if !t.SpecPassed {
			f.SpecGreen = false
		} else {
			f.Passed++
			if t.SpecPoints > 0 {
				f.Points += t.SpecPoints
			} else {
				f.Points++
			}
		}
		if !t.Evidence.Confirmed || t.Evidence.ID == "" {
			f.TruthClean = false
			continue
		}
		f.EvidenceIDs = append(f.EvidenceIDs, t.Evidence.ID)
	}
	return f
}

// EvaluateLoopVariant folds a variant through the existing shipgate keep-bit. The
// baseline and candidate must be the SAME ordered task set; a mismatch is a driver
// bug, because #1177's delta is over a fixed task set.
func EvaluateLoopVariant(v LoopVariant, baseline, candidate TaskSetWitness) (LoopVariantRecord, error) {
	if err := sameTaskSet(baseline, candidate); err != nil {
		return LoopVariantRecord{}, err
	}
	before := FoldTaskSet(baseline)
	after := FoldTaskSet(candidate)
	w := shipgate.Witness{
		Metric:      LoopVariantMetricName,
		Before:      before.Points,
		After:       after.Points,
		LowerBetter: false,
		SuiteGreen:  after.SpecGreen,
		TruthClean:  before.TruthClean && after.TruthClean,
	}
	decision, witness := shipgate.Evaluate(w)
	rec := LoopVariantRecord{
		Schema:       LoopVariantArchiveSchema,
		Variant:      v,
		Decision:     decision.String(),
		Kept:         witness.Kept(),
		MetricName:   LoopVariantMetricName,
		Before:       before,
		After:        after,
		Score:        loopVariantScorecard(before, after),
		Witness:      witness,
		DOSEvidence:  taskSetEvidence(baseline, candidate),
		Tasks:        append([]TaskWitness(nil), candidate.Tasks...),
		BaselineRef:  baseline.Ref,
		CandidateRef: candidate.Ref,
	}
	return rec, nil
}

func loopVariantScorecard(before, after SpecFold) *Scorecard {
	specGreen := 0.0
	if after.SpecGreen {
		specGreen = 1
	}
	truthClean := 0.0
	if before.TruthClean && after.TruthClean {
		truthClean = 1
	}
	return &Scorecard{
		Name:  LoopVariantMetricName,
		Value: after.Points,
		Grade: loopVariantScoreGrade(before, after),
		Components: []ScoreComponent{
			{Name: "before_points", Value: before.Points, Unit: "points"},
			{Name: "after_points", Value: after.Points, Unit: "points"},
			{Name: "point_delta", Value: after.Points - before.Points, Unit: "points"},
			{Name: "before_passed", Value: float64(before.Passed), Unit: "tasks"},
			{Name: "after_passed", Value: float64(after.Passed), Unit: "tasks"},
			{Name: "tasks", Value: float64(after.Tasks), Unit: "tasks"},
			{Name: "spec_green", Value: specGreen, Unit: "bool"},
			{Name: "truth_clean", Value: truthClean, Unit: "bool"},
			{Name: "evidence_ids", Value: float64(len(after.EvidenceIDs)), Unit: "ids"},
		},
	}
}

func loopVariantScoreGrade(before, after SpecFold) string {
	switch {
	case !before.TruthClean || !after.TruthClean:
		return "truth-dirty"
	case !after.SpecGreen:
		return "spec-red"
	case after.Points <= before.Points:
		return "no-gain"
	case after.Passed == after.Tasks:
		return "complete"
	default:
		return "improved"
	}
}

// ArchiveLoopVariant appends a record only when shipgate kept the variant. A REVERT
// is returned to the caller but never becomes a stepping stone.
func ArchiveLoopVariant(v LoopVariant, baseline, candidate TaskSetWitness, a *LoopVariantArchive) (LoopVariantRecord, error) {
	rec, err := EvaluateLoopVariant(v, baseline, candidate)
	if err != nil {
		return LoopVariantRecord{}, err
	}
	if rec.Kept && a != nil {
		if err := a.Append(rec); err != nil {
			return rec, err
		}
	}
	return rec, nil
}

// LoopVariantArchive is an append-only JSONL sink for kept stepping-stone variants.
type LoopVariantArchive struct {
	w      io.WriteCloser
	closer bool
}

// NewLoopVariantArchive opens a kept-variant archive at path. A path of "-" or ""
// writes JSONL to stdout and is not closed.
func NewLoopVariantArchive(path string) (*LoopVariantArchive, error) {
	w, closer, err := openJSONLSink(path)
	if err != nil {
		return nil, err
	}
	return &LoopVariantArchive{w: w, closer: closer}, nil
}

// Append writes one kept variant as a single JSON line. The archive is a stepping-
// stone ledger, so a REVERT record is never admitted even if a caller bypasses
// ArchiveLoopVariant.
func (a *LoopVariantArchive) Append(r LoopVariantRecord) error {
	if !r.Kept || r.Decision != shipgate.KEEP.String() || !r.Witness.Kept() {
		return fmt.Errorf("loopvariant archive admits only kept variants, got decision=%s kept=%v", r.Decision, r.Kept)
	}
	if err := validateArchiveEvidence(r); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = a.w.Write(b)
	return err
}

// Close releases the archive file. Stdout archives are left open.
func (a *LoopVariantArchive) Close() error {
	if a.closer {
		return a.w.Close()
	}
	return nil
}

// ReadLoopVariantArchive loads archive records, skipping torn/non-JSON lines so an
// interrupted append does not hide valid stepping stones already written.
func ReadLoopVariantArchive(path string) ([]LoopVariantRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var rows []LoopVariantRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r LoopVariantRecord
		if err := json.Unmarshal(line, &r); err != nil {
			continue
		}
		rows = append(rows, r)
	}
	return rows, nil
}

func sameTaskSet(a, b TaskSetWitness) error {
	if len(a.Tasks) != len(b.Tasks) {
		return fmt.Errorf("task-set mismatch: baseline has %d task(s), candidate has %d", len(a.Tasks), len(b.Tasks))
	}
	for i := range a.Tasks {
		if a.Tasks[i].TaskID == "" {
			return fmt.Errorf("baseline task %d has empty task_id", i)
		}
		if b.Tasks[i].TaskID == "" {
			return fmt.Errorf("candidate task %d has empty task_id", i)
		}
		if a.Tasks[i].TaskID != b.Tasks[i].TaskID {
			return fmt.Errorf("task-set mismatch at %d: baseline %q candidate %q", i, a.Tasks[i].TaskID, b.Tasks[i].TaskID)
		}
	}
	return nil
}

func taskSetEvidence(taskSets ...TaskSetWitness) []DOSEvidence {
	var n int
	for _, ts := range taskSets {
		n += len(ts.Tasks)
	}
	out := make([]DOSEvidence, 0, n)
	for _, ts := range taskSets {
		for _, t := range ts.Tasks {
			out = append(out, t.Evidence)
		}
	}
	return out
}

func validateArchiveEvidence(r LoopVariantRecord) error {
	if len(r.DOSEvidence) == 0 {
		return fmt.Errorf("loopvariant archive requires dos evidence for kept variant %q", r.Variant.ID)
	}
	for i, ev := range r.DOSEvidence {
		if ev.ID == "" || !ev.Confirmed {
			return fmt.Errorf("loopvariant archive evidence %d for kept variant %q is not confirmed", i, r.Variant.ID)
		}
	}
	return nil
}

// RegimeMetrics records the witnessed baseline or candidate outcome on one
// evaluation regime (for example short, medium, or long horizon).
type RegimeMetrics struct {
	Regime         string      `json:"regime"`
	Score          float64     `json:"score"`
	Completed      int         `json:"completed"`
	RepetitionRate float64     `json:"repetition_rate"`
	Evidence       DOSEvidence `json:"evidence,omitempty"`
}

// MultiRegimeSnapshot is an aligned set of per-regime outcomes for one build
// or loop-policy candidate.
type MultiRegimeSnapshot struct {
	Ref        string          `json:"ref,omitempty"`
	TruthClean bool            `json:"truth_clean"`
	Regimes    []RegimeMetrics `json:"regimes"`
}

// AggregateScore returns the sum of regime scores.
func (s MultiRegimeSnapshot) AggregateScore() float64 {
	var sum float64
	for _, r := range s.Regimes {
		sum += r.Score
	}
	return sum
}

// TotalCompleted returns the sum of completed items across all regimes.
func (s MultiRegimeSnapshot) TotalCompleted() int {
	var total int
	for _, r := range s.Regimes {
		total += r.Completed
	}
	return total
}

// MaxRepetitionRate returns the worst (maximum) repetition rate across regimes.
func (s MultiRegimeSnapshot) MaxRepetitionRate() float64 {
	var worst float64
	for _, r := range s.Regimes {
		if r.RepetitionRate > worst {
			worst = r.RepetitionRate
		}
	}
	return worst
}

// MultiRegimeGatePolicy bounds multi-regime acceptance: no single regime may
// regress past MaxPerRegimeDrop, worst repetition rate may not exceed
// max(RepetitionCeiling, baselineMaxRepetition+MaxRepetitionIncrease), and an
// aggregate score tie keeps only if completions increase by at least
// MinTieBreakCompletions.
type MultiRegimeGatePolicy struct {
	MaxPerRegimeDrop         float64 `json:"max_per_regime_drop"`
	RepetitionCeiling        float64 `json:"repetition_ceiling"`
	MaxRepetitionIncrease    float64 `json:"max_repetition_increase"`
	MinTieBreakCompletions   int     `json:"min_tie_break_completions"`
	TieEpsilon               float64 `json:"tie_epsilon"`
	RequireTruthClean        bool    `json:"require_truth_clean"`
	RequireConfirmedEvidence bool    `json:"require_confirmed_evidence"`
}

// DefaultMultiRegimeGatePolicy returns the standard multi-regime acceptance
// bounds inspired by multi-horizon RSI evaluation: <=2.0 per-regime drop,
// repetition cap max(3.12, baseMaxRep+1.6), and +4 completions on exact score ties.
func DefaultMultiRegimeGatePolicy() MultiRegimeGatePolicy {
	return MultiRegimeGatePolicy{
		MaxPerRegimeDrop:       2.0,
		RepetitionCeiling:      3.12,
		MaxRepetitionIncrease:  1.6,
		MinTieBreakCompletions: 4,
		TieEpsilon:             1e-9,
		RequireTruthClean:      true,
	}
}

// Validate checks that policy bounds are finite and non-negative.
func (p MultiRegimeGatePolicy) Validate() error {
	if math.IsNaN(p.MaxPerRegimeDrop) || math.IsInf(p.MaxPerRegimeDrop, 0) || p.MaxPerRegimeDrop < 0 {
		return fmt.Errorf("multiregime gate: invalid max_per_regime_drop %v", p.MaxPerRegimeDrop)
	}
	if math.IsNaN(p.RepetitionCeiling) || math.IsInf(p.RepetitionCeiling, 0) || p.RepetitionCeiling < 0 {
		return fmt.Errorf("multiregime gate: invalid repetition_ceiling %v", p.RepetitionCeiling)
	}
	if math.IsNaN(p.MaxRepetitionIncrease) || math.IsInf(p.MaxRepetitionIncrease, 0) || p.MaxRepetitionIncrease < 0 {
		return fmt.Errorf("multiregime gate: invalid max_repetition_increase %v", p.MaxRepetitionIncrease)
	}
	if math.IsNaN(p.TieEpsilon) || math.IsInf(p.TieEpsilon, 0) || p.TieEpsilon < 0 {
		return fmt.Errorf("multiregime gate: invalid tie_epsilon %v", p.TieEpsilon)
	}
	if p.MinTieBreakCompletions < 0 {
		return fmt.Errorf("multiregime gate: invalid min_tie_break_completions %d", p.MinTieBreakCompletions)
	}
	return nil
}

// MultiRegimeGateDecision summarizes a multi-regime acceptance adjudication.
type MultiRegimeGateDecision struct {
	Decision            shipgate.Decision `json:"decision"`
	Accepted            bool              `json:"accepted"`
	TieBroken           bool              `json:"tie_broken"`
	Reason              string            `json:"reason"`
	BaseAggregateScore  float64           `json:"base_aggregate_score"`
	CandAggregateScore  float64           `json:"cand_aggregate_score"`
	AggregateScoreDelta float64           `json:"aggregate_score_delta"`
	BaseTotalCompleted  int               `json:"base_total_completed"`
	CandTotalCompleted  int               `json:"cand_total_completed"`
	CompletionDelta     int               `json:"completion_delta"`
	BaseMaxRepetition   float64           `json:"base_max_repetition"`
	CandMaxRepetition   float64           `json:"cand_max_repetition"`
	RepetitionLimit     float64           `json:"repetition_limit"`
	WorstDropRegime     string            `json:"worst_drop_regime,omitempty"`
	WorstDrop           float64           `json:"worst_drop"`
}

func validateMultiRegimePair(baseline, candidate MultiRegimeSnapshot) error {
	if len(baseline.Regimes) == 0 {
		return fmt.Errorf("multiregime gate: baseline has no regimes")
	}
	if len(baseline.Regimes) != len(candidate.Regimes) {
		return fmt.Errorf("multiregime gate: regime count mismatch: baseline %d candidate %d",
			len(baseline.Regimes), len(candidate.Regimes))
	}
	seen := make(map[string]struct{}, len(baseline.Regimes))
	for i := range baseline.Regimes {
		br, cr := baseline.Regimes[i], candidate.Regimes[i]
		if br.Regime == "" {
			return fmt.Errorf("multiregime gate: baseline regime %d has empty name", i)
		}
		if _, dup := seen[br.Regime]; dup {
			return fmt.Errorf("multiregime gate: duplicate regime %q", br.Regime)
		}
		seen[br.Regime] = struct{}{}
		if cr.Regime != br.Regime {
			return fmt.Errorf("multiregime gate: regime mismatch at %d: baseline %q candidate %q",
				i, br.Regime, cr.Regime)
		}
		if err := validateRegimeMetrics("baseline", br); err != nil {
			return err
		}
		if err := validateRegimeMetrics("candidate", cr); err != nil {
			return err
		}
	}
	return nil
}

func validateRegimeMetrics(side string, r RegimeMetrics) error {
	if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
		return fmt.Errorf("multiregime gate: %s regime %q has non-finite score %v", side, r.Regime, r.Score)
	}
	if math.IsNaN(r.RepetitionRate) || math.IsInf(r.RepetitionRate, 0) || r.RepetitionRate < 0 {
		return fmt.Errorf("multiregime gate: %s regime %q has invalid repetition_rate %v", side, r.Regime, r.RepetitionRate)
	}
	if r.Completed < 0 {
		return fmt.Errorf("multiregime gate: %s regime %q has negative completed %d", side, r.Regime, r.Completed)
	}
	return nil
}

// EvaluateMultiRegimeGate evaluates a candidate snapshot across all declared
// regimes against baseline non-regression bounds.
func EvaluateMultiRegimeGate(baseline, candidate MultiRegimeSnapshot, policy MultiRegimeGatePolicy) (MultiRegimeGateDecision, error) {
	if err := policy.Validate(); err != nil {
		return MultiRegimeGateDecision{}, err
	}
	if err := validateMultiRegimePair(baseline, candidate); err != nil {
		return MultiRegimeGateDecision{}, err
	}
	baseAgg := baseline.AggregateScore()
	candAgg := candidate.AggregateScore()
	baseCompleted := baseline.TotalCompleted()
	candCompleted := candidate.TotalCompleted()
	baseMaxRep := baseline.MaxRepetitionRate()
	candMaxRep := candidate.MaxRepetitionRate()
	repLimit := math.Max(policy.RepetitionCeiling, baseMaxRep+policy.MaxRepetitionIncrease)

	out := MultiRegimeGateDecision{
		Decision:            shipgate.REVERT,
		BaseAggregateScore:  baseAgg,
		CandAggregateScore:  candAgg,
		AggregateScoreDelta: candAgg - baseAgg,
		BaseTotalCompleted:  baseCompleted,
		CandTotalCompleted:  candCompleted,
		CompletionDelta:     candCompleted - baseCompleted,
		BaseMaxRepetition:   baseMaxRep,
		CandMaxRepetition:   candMaxRep,
		RepetitionLimit:     repLimit,
	}

	var worstDrop float64
	var worstRegime string
	for i := range baseline.Regimes {
		drop := baseline.Regimes[i].Score - candidate.Regimes[i].Score
		if drop > worstDrop {
			worstDrop = drop
			worstRegime = baseline.Regimes[i].Regime
		}
	}
	out.WorstDrop = worstDrop
	out.WorstDropRegime = worstRegime

	if policy.RequireTruthClean && (!baseline.TruthClean || !candidate.TruthClean) {
		out.Reason = "truth_dirty"
		return out, nil
	}
	if policy.RequireConfirmedEvidence {
		for _, r := range candidate.Regimes {
			if r.Evidence.ID == "" || !r.Evidence.Confirmed {
				out.Reason = fmt.Sprintf("unconfirmed_evidence:%s", r.Regime)
				return out, nil
			}
		}
	}
	for i := range baseline.Regimes {
		drop := baseline.Regimes[i].Score - candidate.Regimes[i].Score
		if drop > policy.MaxPerRegimeDrop+policy.TieEpsilon {
			out.Reason = fmt.Sprintf("regime_drop_floor:%s", baseline.Regimes[i].Regime)
			return out, nil
		}
	}
	if candMaxRep > repLimit+policy.TieEpsilon {
		out.Reason = "repetition_ceiling"
		return out, nil
	}
	switch {
	case out.AggregateScoreDelta > policy.TieEpsilon:
		out.Decision = shipgate.KEEP
		out.Accepted = true
		out.Reason = "aggregate_gain"
	case out.AggregateScoreDelta >= -policy.TieEpsilon:
		if out.CompletionDelta >= policy.MinTieBreakCompletions {
			out.Decision = shipgate.KEEP
			out.Accepted = true
			out.TieBroken = true
			out.Reason = "tie_break_completions"
		} else {
			out.Reason = "tie_break_completion_floor"
		}
	default:
		out.Reason = "aggregate_regression"
	}
	return out, nil
}

// EvaluateMultiRegimeLoopVariant evaluates a multi-regime candidate and folds the
// decision through shipgate.Evaluate into a LoopVariantRecord compatible with
// LoopVariantArchive.
func EvaluateMultiRegimeLoopVariant(v LoopVariant, baseline, candidate MultiRegimeSnapshot, policy MultiRegimeGatePolicy) (LoopVariantRecord, MultiRegimeGateDecision, error) {
	if v.ID == "" {
		return LoopVariantRecord{}, MultiRegimeGateDecision{}, fmt.Errorf("multiregime gate: loop variant has empty id")
	}
	gate, err := EvaluateMultiRegimeGate(baseline, candidate, policy)
	if err != nil {
		return LoopVariantRecord{}, MultiRegimeGateDecision{}, err
	}

	metric := "multi_regime_score"
	beforeVal := gate.BaseAggregateScore
	afterVal := gate.CandAggregateScore
	if gate.TieBroken {
		if gate.CandTotalCompleted > gate.BaseTotalCompleted {
			metric = "multi_regime_completed_tiebreak"
			beforeVal = float64(gate.BaseTotalCompleted)
			afterVal = float64(gate.CandTotalCompleted)
		} else {
			metric = "multi_regime_gate_tiebreak"
			beforeVal = 0
			afterVal = 1
		}
	}

	w := shipgate.Witness{
		Metric:      metric,
		Before:      beforeVal,
		After:       afterVal,
		LowerBetter: false,
		SuiteGreen:  gate.Accepted,
		TruthClean:  !policy.RequireTruthClean || (baseline.TruthClean && candidate.TruthClean),
	}
	decision, witness := shipgate.Evaluate(w)

	evidence := make([]DOSEvidence, 0, len(baseline.Regimes)+len(candidate.Regimes))
	for _, r := range baseline.Regimes {
		if r.Evidence.ID != "" {
			evidence = append(evidence, r.Evidence)
		}
	}
	for _, r := range candidate.Regimes {
		if r.Evidence.ID != "" {
			evidence = append(evidence, r.Evidence)
		}
	}

	beforeFold := SpecFold{
		Points:      gate.BaseAggregateScore,
		Passed:      gate.BaseTotalCompleted,
		Tasks:       len(baseline.Regimes),
		SpecGreen:   true,
		TruthClean:  baseline.TruthClean,
		EvidenceIDs: collectEvidenceIDs(baseline.Regimes),
	}
	afterFold := SpecFold{
		Points:      gate.CandAggregateScore,
		Passed:      gate.CandTotalCompleted,
		Tasks:       len(candidate.Regimes),
		SpecGreen:   gate.Accepted,
		TruthClean:  candidate.TruthClean,
		EvidenceIDs: collectEvidenceIDs(candidate.Regimes),
	}
	rec := LoopVariantRecord{
		Schema:       LoopVariantArchiveSchema,
		Variant:      v,
		Decision:     decision.String(),
		Kept:         witness.Kept(),
		MetricName:   metric,
		Before:       beforeFold,
		After:        afterFold,
		Score:        loopVariantScorecard(beforeFold, afterFold),
		Witness:      witness,
		DOSEvidence:  evidence,
		BaselineRef:  baseline.Ref,
		CandidateRef: candidate.Ref,
	}
	return rec, gate, nil
}

func collectEvidenceIDs(regimes []RegimeMetrics) []string {
	ids := make([]string, 0, len(regimes))
	for _, r := range regimes {
		if r.Evidence.ID != "" {
			ids = append(ids, r.Evidence.ID)
		}
	}
	return ids
}
