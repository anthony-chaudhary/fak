package perfrsiscore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	EvidenceSchema    = "fak-performance-rsi-evidence/1"
	CompositionSchema = "fak-performance-rsi-composition/1"
	CycleSchema       = "fak-performance-rsi-cycle/1"
	ImprovementSchema = "fak-performance-rsi-improvement/1"
	ProvenanceSchema  = "fak-performance-rsi-provenance/1"
	LearningSchema    = "fak-performance-rsi-learning/1"
	HardwareSchema    = "fak-performance-rsi-hardware/1"
	ReportSchema      = "fak-performance-rsi-scorecard/1"
	LoopTurnSchema    = "fak-performance-rsi-loop-turn/1"
	LoopTurnInputEnv  = "FAK_PERFORMANCE_RSI_INPUT"
	TargetMultiple    = 100.0
)

const (
	LoopTurnScored      = "scored"
	LoopTurnUnavailable = "unavailable"
)

type Direction string

const (
	Higher Direction = "higher"
	Lower  Direction = "lower"
)

type Evidence struct {
	Schema           string       `json:"schema"`
	Snapshot         string       `json:"snapshot"`
	TargetMultiplier float64      `json:"target_multiplier"`
	Dimensions       []Dimension  `json:"dimensions"`
	Cycle            *Cycle       `json:"cycle,omitempty"`
	Improvement      *Improvement `json:"improvement,omitempty"`
	Provenance       *Provenance  `json:"provenance,omitempty"`
	Learning         *Learning    `json:"learning,omitempty"`
	Hardware         *Hardware    `json:"hardware,omitempty"`
}

// ComposeInput names one independently validated evidence receipt. Source is
// diagnostic provenance (normally its path), not evidence content.
type ComposeInput struct {
	Source   string
	Evidence Evidence
}

// Hardware is a strict ledger of directly measured accelerator utilization.
// Queue timing is retained as metadata only and never substitutes for a
// utilization measurement.
type Hardware struct {
	Schema string        `json:"schema"`
	Runs   []HardwareRun `json:"runs"`
}

type HardwareRun struct {
	EnqueuedAt           string                    `json:"enqueued_at"`
	StartedAt            string                    `json:"started_at"`
	EndedAt              string                    `json:"ended_at"`
	RequestedDeviceClass string                    `json:"requested_device_class"`
	ActiveUtilization    float64                   `json:"active_utilization"`
	UtilizationUnit      string                    `json:"utilization_unit"`
	WorkloadID           string                    `json:"workload_id"`
	Engine               string                    `json:"engine"`
	TerminalEvidence     *HardwareTerminalEvidence `json:"terminal_evidence,omitempty"`
}

// HardwareTerminalEvidence is an explicitly typed terminal blocker. Hardware
// measurement receipts decode it strictly so validation can reject the blocker
// without inferring terminal state from unrelated measurement fields.
type HardwareTerminalEvidence struct {
	Type string `json:"type"`
}

func (e *HardwareTerminalEvidence) UnmarshalJSON(b []byte) error {
	type terminalEvidenceJSON HardwareTerminalEvidence
	var decoded terminalEvidenceJSON
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("hardware terminal_evidence: trailing JSON value; fix: remove trailing JSON data after terminal_evidence object")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(b, &fields); err != nil {
		return err
	}
	if _, ok := fields["type"]; !ok {
		return errors.New("hardware terminal_evidence type is required; fix: provide non-empty type for terminal_evidence")
	}
	*e = HardwareTerminalEvidence(decoded)
	return nil
}

func (r *HardwareRun) UnmarshalJSON(b []byte) error {
	type hardwareRunJSON HardwareRun
	var decoded hardwareRunJSON
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("hardware run: trailing JSON value; fix: remove trailing JSON data after hardware run object")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(b, &fields); err != nil {
		return err
	}
	for _, name := range []string{
		"enqueued_at", "started_at", "ended_at", "requested_device_class",
		"active_utilization", "utilization_unit", "workload_id", "engine",
	} {
		if _, ok := fields[name]; !ok {
			return fmt.Errorf("hardware run %s is required; fix: populate required field %q in hardware run", name, name)
		}
	}
	*r = HardwareRun(decoded)
	return nil
}

// Learning is an ordered history of predictions, outcomes, and explicit
// learning reuse. Array position, rather than caller-supplied timestamps,
// defines chronology.
type Learning struct {
	Schema string        `json:"schema"`
	Rows   []LearningRow `json:"rows"`
}

type LearningRow struct {
	CycleID                     string  `json:"cycle_id"`
	HypothesisID                string  `json:"hypothesis_id"`
	RecurrenceKey               string  `json:"recurrence_key"`
	PredictedImprovementPercent float64 `json:"predicted_improvement_percent"`
	ConfidencePercent           float64 `json:"confidence_percent"`
	ObservedImprovementPercent  float64 `json:"observed_improvement_percent"`
	LearningID                  string  `json:"learning_id"`
	LearningRecorded            bool    `json:"learning_recorded"`
	LearningReused              bool    `json:"learning_reused"`
	PriorLearningID             string  `json:"prior_learning_id"`
	RepeatedFailure             bool    `json:"repeated_failure"`
	CycleTimeHours              float64 `json:"cycle_time_hours"`
	Engine                      string  `json:"engine"`
	Artifact                    string  `json:"artifact"`
}

func (l *Learning) UnmarshalJSON(b []byte) error {
	type learningJSON Learning
	var decoded learningJSON
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("learning: trailing JSON value; fix: remove trailing JSON data after learning object")
	}
	*l = Learning(decoded)
	return nil
}

// Cycle is one independently versioned, end-to-end performance improvement
// cycle. The engine is explicit so cycle evidence cannot silently cross the
// fak-native boundary.
type Cycle struct {
	Schema                string  `json:"schema"`
	Engine                string  `json:"engine"`
	IdeaAt                string  `json:"idea_at"`
	QueueAt               string  `json:"queue_at"`
	ExecutionAt           string  `json:"execution_at"`
	EvaluationAt          string  `json:"evaluation_at"`
	LandingAt             string  `json:"landing_at"`
	LearningAt            string  `json:"learning_at"`
	OperatorActiveSeconds float64 `json:"operator_active_seconds"`
}

func (c *Cycle) UnmarshalJSON(b []byte) error {
	type cycleJSON Cycle
	var decoded cycleJSON
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("cycle: trailing JSON value; fix: remove trailing JSON data after cycle object")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(b, &fields); err != nil {
		return err
	}
	if _, ok := fields["operator_active_seconds"]; !ok {
		return errors.New("cycle operator_active_seconds is required; fix: specify operator_active_seconds in cycle receipt")
	}
	*c = Cycle(decoded)
	return nil
}

// Improvement is one independently versioned strict receipt for a measured,
// quality-preserving fak-native performance change.
type Improvement struct {
	Schema            string             `json:"schema"`
	Engine            string             `json:"engine"`
	Hypothesis        string             `json:"hypothesis"`
	ChangedModule     string             `json:"changed_module"`
	Baseline          ImprovementMeasure `json:"baseline"`
	Candidate         ImprovementMeasure `json:"candidate"`
	Quality           ImprovementQuality `json:"quality"`
	NetTrueGain       ImprovementGain    `json:"net_true_gain"`
	BaselineEnvelope  OperatingEnvelope  `json:"baseline_envelope"`
	CandidateEnvelope OperatingEnvelope  `json:"candidate_envelope"`
	Causal            ImprovementCausal  `json:"causal"`
}

type ImprovementMeasure struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit"`
}

type ImprovementQuality struct {
	Gate            string `json:"gate"`
	BaselinePassed  *bool  `json:"baseline_passed"`
	CandidatePassed *bool  `json:"candidate_passed"`
	Parity          *bool  `json:"parity"`
}

type ImprovementGain struct {
	Value            float64 `json:"value"`
	Unit             string  `json:"unit"`
	OverheadValue    float64 `json:"overhead_value"`
	OverheadUnit     string  `json:"overhead_unit"`
	IncludesOverhead *bool   `json:"includes_overhead"`
}

type OperatingEnvelope struct {
	Model         string `json:"model"`
	Quantization  string `json:"quantization"`
	Hardware      string `json:"hardware"`
	Workload      string `json:"workload"`
	ContextTokens int    `json:"context_tokens"`
	BatchSize     int    `json:"batch_size"`
}

type ImprovementCausal struct {
	Ablation       string  `json:"ablation"`
	ControlValue   float64 `json:"control_value"`
	TreatmentValue float64 `json:"treatment_value"`
	Unit           string  `json:"unit"`
	IsolatesChange *bool   `json:"isolates_change"`
}

type Dimension struct {
	ID           string    `json:"id"`
	Source       string    `json:"source"`
	Direction    Direction `json:"direction"`
	Current      *float64  `json:"current,omitempty"`
	Target       *float64  `json:"target"`
	Unit         string    `json:"unit"`
	NextAction   string    `json:"next_action"`
	EvidenceKind string    `json:"evidence_kind,omitempty"`
	Engine       string    `json:"engine,omitempty"`
}

type Result struct {
	ID              string   `json:"id"`
	Source          string   `json:"source"`
	EvidenceKind    string   `json:"evidence_kind,omitempty"`
	Engine          string   `json:"engine,omitempty"`
	Status          string   `json:"status"`
	Current         *float64 `json:"current"`
	Target          *float64 `json:"target"`
	Unit            string   `json:"unit"`
	NormalizedRatio *float64 `json:"normalized_ratio"`
	NextAction      string   `json:"next_action"`
}

type Delta struct {
	ID            string   `json:"id"`
	PriorStatus   string   `json:"prior_status"`
	CurrentStatus string   `json:"current_status"`
	PriorRatio    *float64 `json:"prior_ratio,omitempty"`
	CurrentRatio  *float64 `json:"current_ratio,omitempty"`
}

type Comparison struct {
	PriorSnapshot string  `json:"prior_snapshot"`
	Deltas        []Delta `json:"deltas"`
}

// HealthSummary is the bounded presentation signal for loop health. It is
// deliberately scoped away from target achievement: a healthy measurement loop
// can still be far from the explicit performance multiplier, and a high ratio in
// one dimension cannot compensate for missing or behind evidence elsewhere.
type HealthSummary struct {
	Score          float64 `json:"score"`
	Grade          string  `json:"grade"`
	Clean          bool    `json:"clean"`
	Interpretation string  `json:"interpretation"`
}

// DebtEvidence names one dimension that keeps the loop-health report non-clean.
// Source and receipt metadata make the debt re-checkable without treating a
// prose summary as evidence.
type DebtEvidence struct {
	Dimension       string   `json:"dimension"`
	Status          string   `json:"status"`
	NormalizedRatio *float64 `json:"normalized_ratio"`
	Source          string   `json:"source"`
	EvidenceKind    string   `json:"evidence_kind,omitempty"`
	Engine          string   `json:"engine,omitempty"`
	NextAction      string   `json:"next_action"`
}

// DebtSummary is the exhaustive work list for the current metric version.
// performance_rsi_debt counts every BEHIND or UNKNOWN canonical dimension.
type DebtSummary struct {
	PerformanceRSIDebt int            `json:"performance_rsi_debt"`
	Total              int            `json:"total"`
	DimensionsMeasured int            `json:"dimensions_measured"`
	DimensionsTotal    int            `json:"dimensions_total"`
	Behind             int            `json:"behind"`
	Unknown            int            `json:"unknown"`
	Evidence           []DebtEvidence `json:"evidence"`
}

// InvocationOutcome is the observable disposition of one score invocation.
// A refusal means input was unset or supplied evidence could not satisfy the
// strict scorecard contract; an error means the evidence path could not be read.
type InvocationOutcome string

const (
	OutcomeSuccess InvocationOutcome = "success"
	OutcomeRefusal InvocationOutcome = "refusal"
	OutcomeError   InvocationOutcome = "error"
)

// OutcomeCounts makes one invocation's disposition machine-queryable on the
// existing report and automatic loop-turn receipt surfaces.
type OutcomeCounts struct {
	Success int `json:"success"`
	Refusal int `json:"refusal"`
	Error   int `json:"error"`
}

func (c *OutcomeCounts) observe(outcome InvocationOutcome) {
	switch outcome {
	case OutcomeSuccess:
		c.Success++
	case OutcomeRefusal:
		c.Refusal++
	case OutcomeError:
		c.Error++
	}
}

// Total returns the number of score invocations represented by the counts.
func (c OutcomeCounts) Total() int { return c.Success + c.Refusal + c.Error }

func oneOutcome(outcome InvocationOutcome) OutcomeCounts {
	var counts OutcomeCounts
	counts.observe(outcome)
	return counts
}

type Report struct {
	Schema             string         `json:"schema"`
	Snapshot           string         `json:"snapshot"`
	TargetMultiplier   float64        `json:"target_multiplier"`
	Dimensions         []Result       `json:"dimensions"`
	DominantBottleneck string         `json:"dominant_bottleneck"`
	UnknownDebt        int            `json:"unknown_debt"`
	LoopHealth         *HealthSummary `json:"loop_health,omitempty"`
	DebtSummary        *DebtSummary   `json:"debt_summary,omitempty"`
	InvocationOutcomes OutcomeCounts  `json:"invocation_outcomes"`
	Comparison         *Comparison    `json:"comparison,omitempty"`
}

// LoopTurnReceipt is the bounded evidence emitted when a dispatch loop turn
// automatically attempts the performance-RSI score. Unavailable or invalid
// input remains observable without turning score telemetry into a dispatch
// failure.
type LoopTurnReceipt struct {
	Schema                string         `json:"schema"`
	Status                string         `json:"status"`
	Reason                string         `json:"reason"`
	Input                 string         `json:"input,omitempty"`
	Snapshot              string         `json:"snapshot,omitempty"`
	LoopHealth            *HealthSummary `json:"loop_health,omitempty"`
	PerformanceRSIDebt    *int           `json:"performance_rsi_debt,omitempty"`
	DominantBottleneck    string         `json:"dominant_bottleneck,omitempty"`
	UnavailableDiagnostic string         `json:"unavailable_diagnostic,omitempty"`
	InvocationOutcomes    OutcomeCounts  `json:"invocation_outcomes"`
}

var dimensionIDs = []string{
	"cycle_time", "improvement_yield", "evaluation_latency", "receipt_coverage",
	"quality_gate_coverage", "experiment_throughput", "hypothesis_calibration", "discovery_freshness",
	"adaptation_speed", "reuse_ratio", "learning_retention", "production_transfer",
	"hardware_utilization", "attribution_quality", "automation_coverage", "compounding_rate",
}

func DimensionIDs() []string { return append([]string(nil), dimensionIDs...) }

// ScoreLoopTurnFromEnvironment is the single automatic loop-turn entry point.
// It intentionally returns a receipt instead of an error: performance scoring
// is observability at this seam and must not replace the completed dispatch's
// exit status when its independently produced input is absent or unreadable.
func ScoreLoopTurnFromEnvironment() LoopTurnReceipt {
	return ScoreLoopTurn(os.Getenv(LoopTurnInputEnv))
}

// ScoreLoopTurn loads and scores one evidence document using the same strict
// Load and Score path as the performance-rsi-scorecard command.
func ScoreLoopTurn(input string) LoopTurnReceipt {
	input = strings.TrimSpace(input)
	receipt := LoopTurnReceipt{
		Schema:             LoopTurnSchema,
		Status:             LoopTurnUnavailable,
		Reason:             "SCORE_INPUT_UNAVAILABLE",
		Input:              input,
		InvocationOutcomes: oneOutcome(OutcomeRefusal),
	}
	if input == "" {
		receipt.UnavailableDiagnostic = LoopTurnInputEnv + " is not set; fix: set " + LoopTurnInputEnv + " to a valid evidence JSON file path"
		return receipt
	}

	b, err := os.ReadFile(input)
	if err != nil {
		receipt.InvocationOutcomes = oneOutcome(OutcomeError)
		receipt.UnavailableDiagnostic = err.Error()
		return receipt
	}
	evidence, err := Decode(bytes.NewReader(b))
	if err != nil {
		receipt.UnavailableDiagnostic = err.Error()
		return receipt
	}
	report := Score(evidence)
	health, debt := reportHealth(report)
	receipt.Status = LoopTurnScored
	receipt.Reason = "SCORE_COMPLETE"
	receipt.Snapshot = report.Snapshot
	receipt.LoopHealth = &health
	receipt.PerformanceRSIDebt = intPointer(debt.PerformanceRSIDebt)
	receipt.DominantBottleneck = report.DominantBottleneck
	receipt.InvocationOutcomes = report.InvocationOutcomes
	return receipt
}

// FormatLoopTurnReceipt makes the automatic invocation one stable transcript
// line. LoopTurnReceipt contains only JSON-safe bounded scalar data.
func FormatLoopTurnReceipt(receipt LoopTurnReceipt) string {
	b, err := json.Marshal(receipt)
	if err != nil {
		return fmt.Sprintf(`{"schema":%q,"status":"unavailable","reason":"RECEIPT_ENCODE_FAILED","unavailable_diagnostic":%q}`,
			LoopTurnSchema, err.Error())
	}
	return string(b)
}

func intPointer(v int) *int { return &v }

func Load(path string) (Evidence, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Evidence{}, err
	}
	return Decode(bytes.NewReader(b))
}

// LoadAndComposeV1 loads strict receipts and assembles one EvidenceSchema
// document using the versioned CompositionSchema contract.
func LoadAndComposeV1(snapshot string, paths []string) (Evidence, error) {
	inputs := make([]ComposeInput, 0, len(paths))
	for _, path := range paths {
		e, err := Load(path)
		if err != nil {
			return Evidence{}, fmt.Errorf("%s: %w", path, err)
		}
		inputs = append(inputs, ComposeInput{Source: path, Evidence: e})
	}
	return ComposeV1(snapshot, inputs)
}

// ComposeV1 binds every derived dimension to the receipt section that owns it.
// Non-owning copies of that dimension are deliberately ignored. A dimension
// without a supplied owner is retained only when every receipt agrees on its
// scoring contract; its measurement is cleared so the scorecard reports honest
// UNKNOWN debt rather than inheriting an unrelated receipt's baseline.
func ComposeV1(snapshot string, inputs []ComposeInput) (Evidence, error) {
	snapshot = strings.TrimSpace(snapshot)
	if snapshot == "" {
		return Evidence{}, fmt.Errorf("%s: snapshot is required; fix: provide a non-empty snapshot identifier", CompositionSchema)
	}
	if len(inputs) == 0 {
		return Evidence{}, fmt.Errorf("%s: at least one receipt is required; fix: pass one or more receipt paths to compose", CompositionSchema)
	}

	sorted := append([]ComposeInput(nil), inputs...)
	for i := range sorted {
		sorted[i].Source = strings.TrimSpace(sorted[i].Source)
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Source < sorted[j].Source })
	seenSources := make(map[string]bool, len(sorted))
	for i := range sorted {
		if sorted[i].Source == "" {
			return Evidence{}, fmt.Errorf("%s: receipt %d source is required; fix: specify a non-empty source for receipt %d", CompositionSchema, i, i)
		}
		if seenSources[sorted[i].Source] {
			return Evidence{}, fmt.Errorf("%s: receipt source %q appears more than once; fix: ensure each receipt has a unique source path", CompositionSchema, sorted[i].Source)
		}
		seenSources[sorted[i].Source] = true
		validated := sorted[i].Evidence
		if len(validated.Dimensions) > 0 {
			validated.Dimensions = append([]Dimension(nil), validated.Dimensions...)
		}
		if err := validate(&validated); err != nil {
			return Evidence{}, fmt.Errorf("%s: receipt %q is invalid: %w", CompositionSchema, sorted[i].Source, err)
		}
		sorted[i].Evidence = validated
	}

	type sectionOwner struct {
		source   string
		evidence Evidence
	}
	owners := make(map[string]sectionOwner)
	for _, input := range sorted {
		sections := evidenceSections(input.Evidence)
		if len(sections) == 0 {
			return Evidence{}, fmt.Errorf("%s: receipt %q has no composable section; fix: provide a receipt with at least one composable section (cycle, improvement, provenance, learning, or hardware)", CompositionSchema, input.Source)
		}
		for _, section := range sections {
			if prior, exists := owners[section]; exists {
				return Evidence{}, fmt.Errorf(
					"%s: section %q has multiple owners %q and %q; fix: provide exactly one receipt for that section",
					CompositionSchema, section, prior.source, input.Source,
				)
			}
			owners[section] = sectionOwner{source: input.Source, evidence: input.Evidence}
		}
	}

	dimensionsByInput := make([]map[string]Dimension, len(sorted))
	for i, input := range sorted {
		dimensionsByInput[i] = make(map[string]Dimension, len(input.Evidence.Dimensions))
		for _, d := range input.Evidence.Dimensions {
			dimensionsByInput[i][d.ID] = d
		}
	}

	composed := Evidence{
		Schema:           EvidenceSchema,
		Snapshot:         snapshot,
		TargetMultiplier: TargetMultiple,
		Dimensions:       make([]Dimension, 0, len(dimensionIDs)),
	}
	for _, id := range dimensionIDs {
		section := owningSection(id)
		var selected Dimension
		if owner, ok := owners[section]; ok {
			selected = dimensionByID(owner.evidence.Dimensions, id)
		} else {
			selected = dimensionsByInput[0][id]
			for i := 1; i < len(sorted); i++ {
				candidate := dimensionsByInput[i][id]
				if !sameDimensionContract(selected, candidate) {
					return Evidence{}, fmt.Errorf(
						"%s: dimension %q has incompatible contracts without owning section %q: %q (%s) versus %q (%s); fix: add one %q receipt or align the contracts",
						CompositionSchema, id, section,
						sorted[0].Source, renderDimensionContract(selected),
						sorted[i].Source, renderDimensionContract(candidate),
						section,
					)
				}
			}
		}
		selected.Current = nil
		selected.EvidenceKind = ""
		selected.Engine = ""
		composed.Dimensions = append(composed.Dimensions, selected)
	}

	if owner, ok := owners["cycle"]; ok {
		composed.Cycle = owner.evidence.Cycle
	}
	if owner, ok := owners["improvement"]; ok {
		composed.Improvement = owner.evidence.Improvement
	}
	if owner, ok := owners["provenance"]; ok {
		composed.Provenance = owner.evidence.Provenance
	}
	if owner, ok := owners["learning"]; ok {
		composed.Learning = owner.evidence.Learning
	}
	if owner, ok := owners["hardware"]; ok {
		composed.Hardware = owner.evidence.Hardware
	}
	if err := validate(&composed); err != nil {
		return Evidence{}, fmt.Errorf("%s: assembled evidence is invalid: %w", CompositionSchema, err)
	}
	return composed, nil
}

func evidenceSections(e Evidence) []string {
	var sections []string
	if e.Cycle != nil {
		sections = append(sections, "cycle")
	}
	if e.Improvement != nil {
		sections = append(sections, "improvement")
	}
	if e.Provenance != nil {
		sections = append(sections, "provenance")
	}
	if e.Learning != nil {
		sections = append(sections, "learning")
	}
	if e.Hardware != nil {
		sections = append(sections, "hardware")
	}
	return sections
}

func owningSection(id string) string {
	switch id {
	case "cycle_time", "evaluation_latency", "experiment_throughput", "automation_coverage":
		return "cycle"
	case "improvement_yield", "receipt_coverage", "quality_gate_coverage", "attribution_quality":
		return "improvement"
	case "discovery_freshness", "adaptation_speed", "reuse_ratio", "production_transfer":
		return "provenance"
	case "hypothesis_calibration", "learning_retention", "compounding_rate":
		return "learning"
	case "hardware_utilization":
		return "hardware"
	default:
		return ""
	}
}

func dimensionByID(dimensions []Dimension, id string) Dimension {
	for _, d := range dimensions {
		if d.ID == id {
			return d
		}
	}
	return Dimension{}
}

func sameDimensionContract(a, b Dimension) bool {
	if a.ID != b.ID || a.Direction != b.Direction || a.Unit != b.Unit {
		return false
	}
	if a.Target == nil || b.Target == nil {
		return a.Target == nil && b.Target == nil
	}
	return *a.Target == *b.Target
}

func renderDimensionContract(d Dimension) string {
	return fmt.Sprintf("direction=%q unit=%q target=%s", d.Direction, d.Unit, number(d.Target))
}

func Decode(r io.Reader) (Evidence, error) {
	var e Evidence
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&e); err != nil {
		return e, fmt.Errorf("decode evidence: %w; fix: provide valid JSON conforming to %s", err, EvidenceSchema)
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return e, errors.New("decode evidence: trailing JSON value; fix: remove trailing JSON data after evidence object")
	}
	if err := validate(&e); err != nil {
		return e, err
	}
	return e, nil
}

func validate(e *Evidence) error {
	if e.Schema != EvidenceSchema {
		return fmt.Errorf("schema %q, want %q; fix: set schema to %s", e.Schema, EvidenceSchema, EvidenceSchema)
	}
	if strings.TrimSpace(e.Snapshot) == "" {
		return errors.New("snapshot is required; fix: provide a non-empty snapshot identifier")
	}
	if !finite(e.TargetMultiplier) || e.TargetMultiplier != TargetMultiple {
		return fmt.Errorf("target_multiplier must preserve explicit unsaturated 100x target; fix: set target_multiplier to %g", TargetMultiple)
	}
	if len(e.Dimensions) != len(dimensionIDs) {
		return fmt.Errorf("dimensions: got %d, want exactly %d; fix: provide all %d canonical dimensions", len(e.Dimensions), len(dimensionIDs), len(dimensionIDs))
	}
	seen := make(map[string]bool, len(dimensionIDs))
	for _, d := range e.Dimensions {
		if seen[d.ID] {
			return fmt.Errorf("dimension %q appears more than once; fix: include each canonical dimension exactly once", d.ID)
		}
		seen[d.ID] = true
		if strings.TrimSpace(d.Source) == "" || strings.TrimSpace(d.NextAction) == "" || strings.TrimSpace(d.Unit) == "" {
			return fmt.Errorf("dimension %q requires source, unit, and next_action; fix: populate non-empty source, unit, and next_action fields", d.ID)
		}
		if d.Direction != Higher && d.Direction != Lower {
			return fmt.Errorf("dimension %q has invalid direction %q; fix: set direction to %q or %q", d.ID, d.Direction, Higher, Lower)
		}
		if d.Current != nil && (!finite(*d.Current) || *d.Current < 0 || (d.Direction == Lower && *d.Current == 0)) {
			return fmt.Errorf("dimension %q has invalid current; fix: set current to a finite, non-negative value (positive when direction is lower)", d.ID)
		}
		if d.Target != nil && (!finite(*d.Target) || *d.Target <= 0) {
			return fmt.Errorf("dimension %q has invalid target; fix: set target to a finite, positive threshold", d.ID)
		}
		engine := strings.ToLower(strings.TrimSpace(d.Engine))
		if strings.Contains(engine, "llama") {
			return fmt.Errorf("dimension %q: llama.cpp fallback is not native evidence; fix: measure with a native fak-native engine", d.ID)
		}
		if d.EvidenceKind == "native_benchmark" {
			if !strings.HasPrefix(engine, "fak-native") {
				return fmt.Errorf("dimension %q: native benchmark evidence must name a fak-native engine; fix: set engine to a fak-native engine", d.ID)
			}
		}
	}
	for _, id := range dimensionIDs {
		if !seen[id] {
			return fmt.Errorf("missing dimension %q; fix: include canonical dimension %q in dimensions", id, id)
		}
	}
	if e.Cycle != nil {
		if err := applyCycle(e); err != nil {
			return err
		}
	}
	if e.Improvement != nil {
		if err := applyImprovement(e); err != nil {
			return err
		}
	}
	if e.Provenance != nil {
		if err := applyProvenance(e); err != nil {
			return err
		}
	}
	if e.Learning != nil {
		if err := applyLearning(e); err != nil {
			return err
		}
	}
	if e.Hardware != nil {
		if err := applyHardware(e); err != nil {
			return err
		}
	}
	return nil
}

func applyHardware(e *Evidence) error {
	h := e.Hardware
	if h.Schema != HardwareSchema {
		return fmt.Errorf("hardware schema %q, want %q; fix: set hardware schema to %s", h.Schema, HardwareSchema, HardwareSchema)
	}
	if len(h.Runs) == 0 {
		return errors.New("hardware requires at least one measured run; fix: provide at least one measured run in hardware.runs")
	}

	type validatedRun struct {
		durationSeconds float64
		queueSeconds    float64
		utilization     float64
	}
	validated := make([]validatedRun, len(h.Runs))
	totalDuration, weightedUtilization, totalQueue := 0.0, 0.0, 0.0
	for i, run := range h.Runs {
		if strings.TrimSpace(run.RequestedDeviceClass) == "" {
			return fmt.Errorf("hardware run %d requested_device_class is required; fix: specify requested_device_class for run %d", i, i)
		}
		if run.TerminalEvidence != nil {
			switch run.TerminalEvidence.Type {
			case "local-no-gpu":
				return fmt.Errorf("hardware run %d terminal_evidence type %q: local-no-GPU is a terminal blocker, not a hardware utilization measurement", i, run.TerminalEvidence.Type)
			default:
				return fmt.Errorf("hardware run %d terminal_evidence type %q is unsupported; measurement receipts require measured runs, not terminal blockers", i, run.TerminalEvidence.Type)
			}
		}
		if strings.TrimSpace(run.WorkloadID) == "" {
			return fmt.Errorf("hardware run %d workload_id is required; fix: specify workload_id for run %d", i, i)
		}
		if run.Engine != "fak-native" {
			return fmt.Errorf("hardware run %d engine must be exactly fak-native; fix: set engine to fak-native", i)
		}
		if run.UtilizationUnit != "percent" {
			return fmt.Errorf("hardware run %d utilization_unit must be exactly percent; fix: set utilization_unit to percent", i)
		}
		if !finite(run.ActiveUtilization) || run.ActiveUtilization < 0 || run.ActiveUtilization > 100 {
			return fmt.Errorf("hardware run %d active_utilization must be directly measured, finite, and in [0,100]; fix: provide directly measured active_utilization in [0, 100]", i)
		}

		names := []string{"enqueued_at", "started_at", "ended_at"}
		values := []string{run.EnqueuedAt, run.StartedAt, run.EndedAt}
		times := make([]time.Time, len(values))
		for j, value := range values {
			if strings.TrimSpace(value) == "" {
				return fmt.Errorf("hardware run %d %s is required; fix: provide RFC3339 timestamp for %s", i, names[j], names[j])
			}
			parsed, err := time.Parse(time.RFC3339Nano, value)
			if err != nil {
				return fmt.Errorf("hardware run %d %s: malformed RFC3339 timestamp: %w; fix: format timestamp as valid RFC3339 UTC", i, names[j], err)
			}
			_, offset := parsed.Zone()
			if offset != 0 {
				return fmt.Errorf("hardware run %d %s must be UTC; fix: specify UTC timezone offset (Z) for %s", i, names[j], names[j])
			}
			times[j] = parsed
		}
		if times[0].After(times[1]) || !times[2].After(times[1]) {
			return fmt.Errorf("hardware run %d timeline must satisfy enqueued_at <= started_at < ended_at; fix: ensure enqueued_at <= started_at < ended_at", i)
		}
		validated[i] = validatedRun{
			durationSeconds: times[2].Sub(times[1]).Seconds(),
			queueSeconds:    times[1].Sub(times[0]).Seconds(),
			utilization:     run.ActiveUtilization,
		}
	}

	var hardwareIndex = -1
	for i := range e.Dimensions {
		if e.Dimensions[i].ID != "hardware_utilization" {
			continue
		}
		if e.Dimensions[i].Direction != Higher || e.Dimensions[i].Unit != "percent" {
			return errors.New(`dimension "hardware_utilization" must use canonical direction higher and unit percent; fix: set direction to "higher" and unit to "percent"`)
		}
		hardwareIndex = i
		break
	}
	if hardwareIndex < 0 {
		return errors.New(`missing dimension "hardware_utilization"; fix: include dimension "hardware_utilization" in dimensions`)
	}

	for _, run := range validated {
		totalDuration += run.durationSeconds
		weightedUtilization += run.utilization * run.durationSeconds
		totalQueue += run.queueSeconds
	}
	current := weightedUtilization / totalDuration
	d := &e.Dimensions[hardwareIndex]
	d.Current = &current
	d.Source = fmt.Sprintf(
		"hardware:%s;queue_delay_seconds_total=%s;queue_delay_seconds_mean=%s",
		h.Schema,
		formatFact(totalQueue),
		formatFact(totalQueue/float64(len(validated))),
	)
	d.EvidenceKind = "hardware_utilization_receipt"
	d.Engine = "fak-native"
	return nil
}

func formatFact(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func applyLearning(e *Evidence) error {
	l := e.Learning
	if l.Schema != LearningSchema {
		return fmt.Errorf("learning schema %q, want %q; fix: set learning schema to %s", l.Schema, LearningSchema, LearningSchema)
	}
	if len(l.Rows) < 2 {
		return errors.New("learning requires at least two rows of history; fix: provide at least two ordered learning rows")
	}
	cycleIDs := make(map[string]bool, len(l.Rows))
	hypothesisIDs := make(map[string]bool, len(l.Rows))
	type recorded struct {
		key string
	}
	learnings := make(map[string]recorded)
	seenKeys := make(map[string]bool)
	failedKeys := make(map[string]bool)
	originalTime := make(map[string]float64)
	originalOrder := make([]string, 0)
	eligible, validReuses := 0, 0
	confidenceTotal, weightedError := 0.0, 0.0
	latestReuseTime := make(map[string]float64)
	for i, row := range l.Rows {
		if strings.TrimSpace(row.CycleID) == "" || cycleIDs[row.CycleID] {
			return fmt.Errorf("learning row %d has empty or duplicate cycle_id; fix: provide a unique non-empty cycle_id for row %d", i, i)
		}
		if strings.TrimSpace(row.HypothesisID) == "" || hypothesisIDs[row.HypothesisID] {
			return fmt.Errorf("learning row %d has empty or duplicate hypothesis_id; fix: provide a unique non-empty hypothesis_id for row %d", i, i)
		}
		cycleIDs[row.CycleID], hypothesisIDs[row.HypothesisID] = true, true
		if strings.TrimSpace(row.RecurrenceKey) == "" {
			return fmt.Errorf("learning row %d recurrence_key is required; fix: specify non-empty recurrence_key for row %d", i, i)
		}
		for name, value := range map[string]float64{"prediction": row.PredictedImprovementPercent, "confidence": row.ConfidencePercent, "outcome": row.ObservedImprovementPercent} {
			if !finite(value) || value < 0 || value > 100 {
				return fmt.Errorf("learning row %d %s must be finite in [0,100]; fix: set %s to a finite percentage in [0, 100]", i, name, name)
			}
		}
		if !finite(row.CycleTimeHours) || row.CycleTimeHours <= 0 {
			return fmt.Errorf("learning row %d cycle_time_hours must be positive and finite; fix: set cycle_time_hours to a positive finite value", i)
		}
		if row.Engine != "fak-native" || strings.TrimSpace(row.Artifact) == "" {
			return fmt.Errorf("learning row %d requires engine fak-native and artifact; fix: set engine to fak-native and specify non-empty artifact", i)
		}
		if row.LearningRecorded != (strings.TrimSpace(row.LearningID) != "") {
			return fmt.Errorf("learning row %d learning_id must be present iff learning_recorded; fix: provide learning_id when learning_recorded is true and omit otherwise", i)
		}
		if row.LearningID != "" {
			if _, exists := learnings[row.LearningID]; exists {
				return fmt.Errorf("learning row %d has duplicate learning_id; fix: ensure each recorded learning_id is globally unique", i)
			}
		}
		if row.LearningReused != (strings.TrimSpace(row.PriorLearningID) != "") {
			return fmt.Errorf("learning row %d prior_learning_id must be present iff learning_reused; fix: specify prior_learning_id when learning_reused is true and omit otherwise", i)
		}
		wasSeen := seenKeys[row.RecurrenceKey]
		if wasSeen {
			eligible++
		}
		if row.LearningReused {
			if row.PriorLearningID == row.LearningID {
				return fmt.Errorf("learning row %d cannot reference its own learning_id; fix: reference an earlier learning_id instead of row %d's own ID", i, i)
			}
			prior, exists := learnings[row.PriorLearningID]
			if !exists || prior.key != row.RecurrenceKey {
				return fmt.Errorf("learning row %d prior_learning_id must reference earlier learning with same recurrence_key; fix: reference an earlier learning row sharing recurrence_key %q", i, row.RecurrenceKey)
			}
			if !wasSeen {
				return fmt.Errorf("learning row %d cannot reuse learning without an earlier recurrence; fix: ensure an earlier row establishes recurrence_key %q before reuse", i, row.RecurrenceKey)
			}
			validReuses++
			latestReuseTime[row.RecurrenceKey] = row.CycleTimeHours
		}
		if row.LearningID != "" {
			learnings[row.LearningID] = recorded{key: row.RecurrenceKey}
		}
		failed := row.ObservedImprovementPercent <= 0
		actualRepeated := failed && failedKeys[row.RecurrenceKey]
		if row.RepeatedFailure != actualRepeated {
			return fmt.Errorf("learning row %d repeated_failure does not match ordered failure history; fix: set repeated_failure to true only when a prior row with same recurrence_key failed", i)
		}
		if failed {
			failedKeys[row.RecurrenceKey] = true
		}
		if !wasSeen {
			originalTime[row.RecurrenceKey] = row.CycleTimeHours
			originalOrder = append(originalOrder, row.RecurrenceKey)
		}
		seenKeys[row.RecurrenceKey] = true
		confidenceTotal += row.ConfidencePercent
		weightedError += row.ConfidencePercent * math.Abs(row.PredictedImprovementPercent-row.ObservedImprovementPercent)
	}
	if confidenceTotal <= 0 {
		return errors.New("learning requires positive total confidence; fix: ensure total confidence_percent across learning rows is greater than zero")
	}
	if eligible == 0 {
		return errors.New("learning requires a recurrence; fix: include at least two rows sharing a recurrence_key")
	}
	if validReuses == 0 {
		return errors.New("learning requires a valid explicit prior-learning reuse; fix: include a row with learning_reused true referencing an earlier learning_id")
	}
	earliestOriginal, latestReused := 0.0, 0.0
	for _, key := range originalOrder {
		reusedTime, ok := latestReuseTime[key]
		if !ok {
			continue
		}
		earliestOriginal = originalTime[key]
		latestReused = reusedTime
		break
	}
	compounding := 100 * (earliestOriginal - latestReused) / earliestOriginal
	if !finite(compounding) || compounding <= 0 {
		return errors.New("learning compounding must be positive; false or negative compounding is rejected; fix: ensure reused cycle time improves over original cycle time")
	}
	values := map[string]float64{
		"hypothesis_calibration": math.Max(0, 100-weightedError/confidenceTotal),
		"learning_retention":     100 * float64(validReuses) / float64(eligible),
		"compounding_rate":       compounding,
	}
	derivedIndexes := make([]int, 0, len(values))
	for i := range e.Dimensions {
		d := &e.Dimensions[i]
		_, derived := values[d.ID]
		if !derived {
			continue
		}
		if d.Direction != Higher || d.Unit != "percent" {
			return fmt.Errorf("dimension %q must use canonical direction higher and unit percent; fix: set direction to \"higher\" and unit to \"percent\"", d.ID)
		}
		derivedIndexes = append(derivedIndexes, i)
	}
	for _, i := range derivedIndexes {
		d := &e.Dimensions[i]
		value := values[d.ID]
		v := value
		d.Current = &v
		d.Source = "learning:" + l.Schema
		d.EvidenceKind = "performance_rsi_learning_receipt"
		d.Engine = "fak-native"
	}
	return nil
}

func applyCycle(e *Evidence) error {
	c := e.Cycle
	if c.Schema != CycleSchema {
		return fmt.Errorf("cycle schema %q, want %q; fix: set cycle schema to %s", c.Schema, CycleSchema, CycleSchema)
	}
	engine := strings.ToLower(strings.TrimSpace(c.Engine))
	if strings.Contains(engine, "llama") || !strings.HasPrefix(engine, "fak-native") {
		return errors.New("cycle engine must name explicit fak-native provenance without llama.cpp fallback; fix: set cycle engine to a fak-native engine")
	}
	if !finite(c.OperatorActiveSeconds) || c.OperatorActiveSeconds < 0 {
		return errors.New("cycle operator_active_seconds must be nonnegative and finite; fix: set operator_active_seconds to a finite value >= 0")
	}

	names := []string{"idea_at", "queue_at", "execution_at", "evaluation_at", "landing_at", "learning_at"}
	values := []string{c.IdeaAt, c.QueueAt, c.ExecutionAt, c.EvaluationAt, c.LandingAt, c.LearningAt}
	times := make([]time.Time, len(values))
	for i, value := range values {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("cycle %s is required; fix: provide an RFC3339 timestamp for %s", names[i], names[i])
		}
		parsed, err := time.Parse(time.RFC3339Nano, value)
		if err != nil {
			return fmt.Errorf("cycle %s: malformed RFC3339 timestamp: %w; fix: format timestamp as valid RFC3339", names[i], err)
		}
		times[i] = parsed
		if i > 0 && !times[i].After(times[i-1]) {
			return fmt.Errorf("cycle stages must be strictly ordered: %s must follow %s; fix: ensure timestamp for %s is strictly after %s", names[i], names[i-1], names[i], names[i-1])
		}
	}

	cycleSeconds := times[5].Sub(times[0]).Seconds()
	evaluationSeconds := times[3].Sub(times[2]).Seconds()
	if c.OperatorActiveSeconds > cycleSeconds {
		return errors.New("cycle operator_active_seconds exceeds end-to-end cycle time; fix: set operator_active_seconds <= total cycle duration")
	}

	for i := range e.Dimensions {
		d := &e.Dimensions[i]
		var current float64
		var err error
		switch d.ID {
		case "cycle_time":
			current, err = durationInUnit(cycleSeconds, d.Unit)
		case "evaluation_latency":
			current, err = durationInUnit(evaluationSeconds, d.Unit)
		case "experiment_throughput":
			current, err = throughputInUnit(cycleSeconds, d.Unit)
		case "automation_coverage":
			current, err = coverageInUnit(1-c.OperatorActiveSeconds/cycleSeconds, d.Unit)
		default:
			continue
		}
		if err != nil {
			return fmt.Errorf("cycle derivation for %s: %w", d.ID, err)
		}
		d.Current = &current
		d.Source = "cycle:" + c.Schema
		d.EvidenceKind = "cycle_ledger"
		d.Engine = c.Engine
	}
	return nil
}

func applyImprovement(e *Evidence) error {
	r := e.Improvement
	if r.Schema != ImprovementSchema {
		return fmt.Errorf("improvement schema %q, want %q; fix: set improvement schema to %s", r.Schema, ImprovementSchema, ImprovementSchema)
	}
	engine := strings.ToLower(strings.TrimSpace(r.Engine))
	if strings.Contains(engine, "llama") || !strings.HasPrefix(engine, "fak-native") {
		return errors.New("improvement engine must name explicit fak-native provenance without llama.cpp fallback; fix: set improvement engine to a fak-native engine")
	}
	if strings.TrimSpace(r.Hypothesis) == "" {
		return errors.New("improvement hypothesis is required; fix: provide non-empty hypothesis description")
	}
	module := strings.TrimSpace(r.ChangedModule)
	if !strings.Contains(module, "@r") || !strings.Contains(module, "+g") {
		return errors.New("improvement changed_module must name module@rev; fix: specify changed_module in format module@rN+g<sha>")
	}
	if r.Baseline.Unit != "milliseconds" || r.Candidate.Unit != "milliseconds" ||
		!finite(r.Baseline.Value) || !finite(r.Candidate.Value) || r.Baseline.Value <= 0 || r.Candidate.Value <= 0 {
		return errors.New("improvement baseline and candidate require positive finite milliseconds; fix: specify positive finite values with unit \"milliseconds\" for baseline and candidate")
	}
	if strings.TrimSpace(r.Quality.Gate) == "" || r.Quality.BaselinePassed == nil || r.Quality.CandidatePassed == nil || r.Quality.Parity == nil ||
		!*r.Quality.BaselinePassed || !*r.Quality.CandidatePassed || !*r.Quality.Parity {
		return errors.New("improvement quality gate requires passing baseline/candidate parity; fix: verify and set baseline_passed, candidate_passed, and parity to true")
	}
	if r.BaselineEnvelope != r.CandidateEnvelope || !validEnvelope(r.BaselineEnvelope) {
		return errors.New("improvement baseline and candidate operating envelopes must be complete and matched; fix: populate identical non-empty envelope fields across baseline and candidate")
	}
	if r.NetTrueGain.Unit != "percent" || r.NetTrueGain.OverheadUnit != "milliseconds" ||
		r.NetTrueGain.IncludesOverhead == nil || !*r.NetTrueGain.IncludesOverhead ||
		!finite(r.NetTrueGain.OverheadValue) || r.NetTrueGain.OverheadValue < 0 || !finite(r.NetTrueGain.Value) {
		return errors.New("improvement net_true_gain must be percent and explicitly include nonnegative millisecond overhead; fix: set net_true_gain unit to \"percent\", overhead_unit to \"milliseconds\", and includes_overhead to true")
	}
	wantGain := (r.Baseline.Value - (r.Candidate.Value + r.NetTrueGain.OverheadValue)) / r.Baseline.Value * 100
	if wantGain <= 0 || math.Abs(wantGain-r.NetTrueGain.Value) > 1e-9 {
		return errors.New("improvement net_true_gain does not equal baseline minus candidate and overhead; fix: calculate net_true_gain as ((baseline - (candidate + overhead)) / baseline) * 100")
	}
	if strings.TrimSpace(r.Causal.Ablation) == "" || r.Causal.Unit != "milliseconds" || r.Causal.IsolatesChange == nil || !*r.Causal.IsolatesChange ||
		!finite(r.Causal.ControlValue) || !finite(r.Causal.TreatmentValue) || r.Causal.ControlValue <= r.Causal.TreatmentValue {
		return errors.New("improvement causal binding requires an isolating, positive ablation in milliseconds; fix: specify non-empty ablation, isolates_change true, unit \"milliseconds\", and control > treatment")
	}

	values := map[string]float64{
		"improvement_yield":     r.NetTrueGain.Value,
		"receipt_coverage":      100,
		"quality_gate_coverage": 100,
		"attribution_quality":   100,
	}
	for i := range e.Dimensions {
		d := &e.Dimensions[i]
		value, ok := values[d.ID]
		if !ok {
			continue
		}
		if strings.ToLower(strings.TrimSpace(d.Unit)) != "percent" {
			return fmt.Errorf("improvement derivation for %s: unsupported unit %q; fix: set dimension unit to \"percent\"", d.ID, d.Unit)
		}
		d.Current = &value
		d.Source = "improvement:" + r.Schema
		d.EvidenceKind = "improvement_receipt"
		d.Engine = r.Engine
	}
	return nil
}

var (
	lowerHex40 = regexp.MustCompile(`^[0-9a-f]{40}$`)
	moduleRev  = regexp.MustCompile(`^.+@r[1-9][0-9]*\+g([0-9a-f]{7,40})$`)
)

func applyProvenance(e *Evidence) error {
	r := e.Provenance
	if r.Schema != ProvenanceSchema {
		return fmt.Errorf("provenance schema %q, want %q; fix: set provenance schema to %s", r.Schema, ProvenanceSchema, ProvenanceSchema)
	}
	if strings.TrimSpace(r.Source.Repository) == "" {
		return errors.New("provenance source repository is required; fix: specify non-empty source repository")
	}
	if !lowerHex40.MatchString(r.Source.Revision) {
		return errors.New("provenance source revision must be an immutable 40-character lowercase hex revision; fix: set source revision to a 40-character git commit SHA")
	}
	if r.Unit != "hours" {
		return errors.New("provenance unit must be hours; fix: set provenance unit to \"hours\"")
	}
	if !r.AdaptationStartExplicit {
		return errors.New("provenance adaptation start must be explicit; fix: set adaptation_start_explicit to true")
	}
	if !r.Experiment.Linked || strings.TrimSpace(r.Experiment.Artifact) == "" {
		return errors.New("provenance experiment must be linked and name a nonempty artifact; fix: set experiment.linked to true and provide an artifact path")
	}
	if r.Reuse.Classification != "adapted_known_art" {
		return errors.New("provenance reuse classification must be adapted_known_art; fix: set reuse classification to \"adapted_known_art\"")
	}
	if r.Reuse.ReusedMechanisms < 0 || r.Reuse.ReinventedMechanisms < 0 {
		return errors.New("provenance mechanism counts must be nonnegative; fix: specify non-negative counts for reused_mechanisms and reinvented_mechanisms")
	}
	totalMechanisms := r.Reuse.ReusedMechanisms + r.Reuse.ReinventedMechanisms
	if totalMechanisms <= 0 {
		return errors.New("provenance mechanism count total must be positive; fix: ensure total mechanisms (reused + reinvented) is greater than zero")
	}
	if !lowerHex40.MatchString(r.Production.CommitSHA) {
		return errors.New("provenance production commit_sha must be 40-character lowercase hex; fix: set production commit_sha to 40-character git SHA")
	}
	match := moduleRev.FindStringSubmatch(r.Production.ModuleAtRev)
	if match == nil || !strings.HasPrefix(r.Production.CommitSHA, match[1]) {
		return errors.New("provenance production module_at_rev must be module@rN+g<sha-prefix> matching commit_sha; fix: format module_at_rev as module@rN+g<commit_sha_prefix>")
	}
	if r.Production.Engine != "fak-native" {
		return errors.New("provenance production engine must be fak-native; fix: set production engine to fak-native")
	}
	if strings.TrimSpace(r.Production.Artifact) == "" {
		return errors.New("provenance production artifact is required; fix: provide a non-empty production artifact path")
	}

	names := []string{"source.published_at", "discovery_at", "adaptation_started_at", "production.landed_at"}
	values := []string{r.Source.PublishedAt, r.DiscoveryAt, r.AdaptationStartedAt, r.Production.LandedAt}
	times := make([]time.Time, len(values))
	for i, value := range values {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("provenance %s is required; fix: provide RFC3339 timestamp for %s", names[i], names[i])
		}
		parsed, err := time.Parse(time.RFC3339Nano, value)
		if err != nil {
			return fmt.Errorf("provenance %s: malformed RFC3339 timestamp: %w; fix: format timestamp as valid RFC3339", names[i], err)
		}
		times[i] = parsed
		if i > 0 && times[i].Before(times[i-1]) {
			return fmt.Errorf("provenance timeline must satisfy source <= discovery <= adaptation <= landing; fix: ensure timestamps are chronologically ordered (source <= discovery <= adaptation <= landing)")
		}
	}

	valuesByID := map[string]float64{
		"discovery_freshness": times[1].Sub(times[0]).Hours(),
		"adaptation_speed":    times[3].Sub(times[2]).Hours(),
		"reuse_ratio":         float64(r.Reuse.ReusedMechanisms) / float64(totalMechanisms) * 100,
		"production_transfer": times[3].Sub(times[1]).Hours(),
	}
	for _, d := range e.Dimensions {
		_, owned := valuesByID[d.ID]
		if !owned {
			continue
		}
		wantUnit := "hours"
		if d.ID == "reuse_ratio" {
			wantUnit = "percent"
		}
		if strings.ToLower(strings.TrimSpace(d.Unit)) != wantUnit {
			return fmt.Errorf("provenance derivation for %s requires unit %q; fix: set dimension unit to %q", d.ID, wantUnit, wantUnit)
		}
	}

	for i := range e.Dimensions {
		d := &e.Dimensions[i]
		value, owned := valuesByID[d.ID]
		if !owned {
			continue
		}
		d.Current = &value
		d.Source = "provenance:" + r.Schema
		d.EvidenceKind = "research_transfer_receipt"
		d.Engine = r.Production.Engine
	}
	return nil
}

func validEnvelope(e OperatingEnvelope) bool {
	return strings.TrimSpace(e.Model) != "" && strings.TrimSpace(e.Quantization) != "" &&
		strings.TrimSpace(e.Hardware) != "" && strings.TrimSpace(e.Workload) != "" &&
		e.ContextTokens > 0 && e.BatchSize > 0
}

func durationInUnit(seconds float64, unit string) (float64, error) {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "second", "seconds", "s":
		return seconds, nil
	case "minute", "minutes", "min":
		return seconds / 60, nil
	case "hour", "hours", "h":
		return seconds / 3600, nil
	case "day", "days", "d":
		return seconds / 86400, nil
	case "week", "weeks", "w":
		return seconds / (7 * 86400), nil
	default:
		return 0, fmt.Errorf("unsupported duration unit %q; fix: use one of second, minute, hour, day, week", unit)
	}
}

func throughputInUnit(seconds float64, unit string) (float64, error) {
	normalized := strings.ToLower(strings.TrimSpace(unit))
	normalized = strings.ReplaceAll(normalized, "_per_", "/")
	normalized = strings.ReplaceAll(normalized, " per ", "/")
	for _, prefix := range []string{"experiments/", "experiment/", "cycles/", "cycle/"} {
		if strings.HasPrefix(normalized, prefix) {
			period, err := durationInUnit(1, strings.TrimPrefix(normalized, prefix))
			if err == nil {
				return 1 / (seconds * period), nil
			}
		}
	}
	return 0, fmt.Errorf("unsupported throughput unit %q; fix: format throughput unit as experiments/<duration_unit>", unit)
}

func coverageInUnit(fraction float64, unit string) (float64, error) {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "fraction", "ratio":
		return fraction, nil
	case "percent", "percentage", "%":
		return fraction * 100, nil
	default:
		return 0, fmt.Errorf("unsupported coverage unit %q; fix: use fraction or percent for coverage unit", unit)
	}
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func Score(e Evidence) Report {
	r := Report{
		Schema:             ReportSchema,
		Snapshot:           e.Snapshot,
		TargetMultiplier:   e.TargetMultiplier,
		InvocationOutcomes: oneOutcome(OutcomeSuccess),
	}
	worst := math.Inf(1)
	for _, d := range e.Dimensions {
		x := Result{ID: d.ID, Source: d.Source, EvidenceKind: d.EvidenceKind, Engine: d.Engine, Current: d.Current, Target: d.Target, Unit: d.Unit, NextAction: d.NextAction, Status: "UNKNOWN"}
		debt := math.Inf(1)
		if d.Current != nil && d.Target != nil {
			ratio := 0.0
			if d.Direction == Higher {
				ratio = *d.Current / *d.Target
			} else if *d.Current == 0 {
				ratio = math.Inf(1)
			} else {
				ratio = *d.Target / *d.Current
			}
			x.NormalizedRatio = &ratio
			x.Status = "BEHIND"
			if ratio >= 1 {
				x.Status = "MET"
			}
			debt = ratio
		} else {
			r.UnknownDebt++
		}
		r.Dimensions = append(r.Dimensions, x)
		if debt < worst || (math.IsInf(debt, 1) && math.IsInf(worst, 1) && r.DominantBottleneck == "") {
			worst, r.DominantBottleneck = debt, d.ID
		}
	}
	health, debt := summarizeHealth(r.Dimensions)
	r.LoopHealth = &health
	r.DebtSummary = &debt
	return r
}

// Provenance is one independently versioned strict receipt connecting an
// immutable research source to a fak-native production landing.
type Provenance struct {
	Schema                  string               `json:"schema"`
	Source                  ProvenanceSource     `json:"source"`
	DiscoveryAt             string               `json:"discovery_at"`
	AdaptationStartedAt     string               `json:"adaptation_started_at"`
	AdaptationStartExplicit bool                 `json:"adaptation_start_explicit"`
	Experiment              ProvenanceExperiment `json:"experiment"`
	Reuse                   ProvenanceReuse      `json:"reuse"`
	Production              ProvenanceProduction `json:"production"`
	Unit                    string               `json:"unit"`
}

type ProvenanceSource struct {
	Repository  string `json:"repository"`
	Revision    string `json:"revision"`
	PublishedAt string `json:"published_at"`
}

type ProvenanceExperiment struct {
	Artifact string `json:"artifact"`
	Linked   bool   `json:"linked"`
}

type ProvenanceReuse struct {
	ReusedMechanisms     int    `json:"reused_mechanisms"`
	ReinventedMechanisms int    `json:"reinvented_mechanisms"`
	Classification       string `json:"classification"`
}

type ProvenanceProduction struct {
	ModuleAtRev string `json:"module_at_rev"`
	CommitSHA   string `json:"commit_sha"`
	LandedAt    string `json:"landed_at"`
	Engine      string `json:"engine"`
	Artifact    string `json:"artifact"`
}
