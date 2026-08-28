package perfrsiscore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	EvidenceSchema    = "fak-performance-rsi-evidence/1"
	CycleSchema       = "fak-performance-rsi-cycle/1"
	ImprovementSchema = "fak-performance-rsi-improvement/1"
	ReportSchema      = "fak-performance-rsi-scorecard/1"
	TargetMultiple    = 100.0
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
		return errors.New("cycle: trailing JSON value")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(b, &fields); err != nil {
		return err
	}
	if _, ok := fields["operator_active_seconds"]; !ok {
		return errors.New("cycle operator_active_seconds is required")
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

type Report struct {
	Schema             string      `json:"schema"`
	Snapshot           string      `json:"snapshot"`
	TargetMultiplier   float64     `json:"target_multiplier"`
	Dimensions         []Result    `json:"dimensions"`
	DominantBottleneck string      `json:"dominant_bottleneck"`
	UnknownDebt        int         `json:"unknown_debt"`
	Comparison         *Comparison `json:"comparison,omitempty"`
}

var dimensionIDs = []string{
	"cycle_time", "improvement_yield", "evaluation_latency", "receipt_coverage",
	"quality_gate_coverage", "experiment_throughput", "hypothesis_calibration", "discovery_freshness",
	"adaptation_speed", "reuse_ratio", "learning_retention", "production_transfer",
	"hardware_utilization", "attribution_quality", "automation_coverage", "compounding_rate",
}

func DimensionIDs() []string { return append([]string(nil), dimensionIDs...) }

func Load(path string) (Evidence, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Evidence{}, err
	}
	return Decode(bytes.NewReader(b))
}

func Decode(r io.Reader) (Evidence, error) {
	var e Evidence
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&e); err != nil {
		return e, fmt.Errorf("decode evidence: %w", err)
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return e, errors.New("decode evidence: trailing JSON value")
	}
	if err := validate(&e); err != nil {
		return e, err
	}
	return e, nil
}

func validate(e *Evidence) error {
	if e.Schema != EvidenceSchema {
		return fmt.Errorf("schema %q, want %q", e.Schema, EvidenceSchema)
	}
	if strings.TrimSpace(e.Snapshot) == "" {
		return errors.New("snapshot is required")
	}
	if !finite(e.TargetMultiplier) || e.TargetMultiplier != TargetMultiple {
		return fmt.Errorf("target_multiplier must preserve explicit unsaturated 100x target")
	}
	if len(e.Dimensions) != len(dimensionIDs) {
		return fmt.Errorf("dimensions: got %d, want exactly %d", len(e.Dimensions), len(dimensionIDs))
	}
	seen := make(map[string]bool, len(dimensionIDs))
	for _, d := range e.Dimensions {
		if seen[d.ID] {
			return fmt.Errorf("dimension %q appears more than once", d.ID)
		}
		seen[d.ID] = true
		if strings.TrimSpace(d.Source) == "" || strings.TrimSpace(d.NextAction) == "" || strings.TrimSpace(d.Unit) == "" {
			return fmt.Errorf("dimension %q requires source, unit, and next_action", d.ID)
		}
		if d.Direction != Higher && d.Direction != Lower {
			return fmt.Errorf("dimension %q has invalid direction %q", d.ID, d.Direction)
		}
		if d.Current != nil && (!finite(*d.Current) || *d.Current < 0 || (d.Direction == Lower && *d.Current == 0)) {
			return fmt.Errorf("dimension %q has invalid current", d.ID)
		}
		if d.Target != nil && (!finite(*d.Target) || *d.Target <= 0) {
			return fmt.Errorf("dimension %q has invalid target", d.ID)
		}
		engine := strings.ToLower(strings.TrimSpace(d.Engine))
		if strings.Contains(engine, "llama") {
			return fmt.Errorf("dimension %q: llama.cpp fallback is not native evidence", d.ID)
		}
		if d.EvidenceKind == "native_benchmark" {
			if !strings.HasPrefix(engine, "fak-native") {
				return fmt.Errorf("dimension %q: native benchmark evidence must name a fak-native engine", d.ID)
			}
		}
	}
	for _, id := range dimensionIDs {
		if !seen[id] {
			return fmt.Errorf("missing dimension %q", id)
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
	return nil
}

func applyCycle(e *Evidence) error {
	c := e.Cycle
	if c.Schema != CycleSchema {
		return fmt.Errorf("cycle schema %q, want %q", c.Schema, CycleSchema)
	}
	engine := strings.ToLower(strings.TrimSpace(c.Engine))
	if strings.Contains(engine, "llama") || !strings.HasPrefix(engine, "fak-native") {
		return errors.New("cycle engine must name explicit fak-native provenance without llama.cpp fallback")
	}
	if !finite(c.OperatorActiveSeconds) || c.OperatorActiveSeconds < 0 {
		return errors.New("cycle operator_active_seconds must be nonnegative and finite")
	}

	names := []string{"idea_at", "queue_at", "execution_at", "evaluation_at", "landing_at", "learning_at"}
	values := []string{c.IdeaAt, c.QueueAt, c.ExecutionAt, c.EvaluationAt, c.LandingAt, c.LearningAt}
	times := make([]time.Time, len(values))
	for i, value := range values {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("cycle %s is required", names[i])
		}
		parsed, err := time.Parse(time.RFC3339Nano, value)
		if err != nil {
			return fmt.Errorf("cycle %s: malformed RFC3339 timestamp: %w", names[i], err)
		}
		times[i] = parsed
		if i > 0 && !times[i].After(times[i-1]) {
			return fmt.Errorf("cycle stages must be strictly ordered: %s must follow %s", names[i], names[i-1])
		}
	}

	cycleSeconds := times[5].Sub(times[0]).Seconds()
	evaluationSeconds := times[3].Sub(times[2]).Seconds()
	if c.OperatorActiveSeconds > cycleSeconds {
		return errors.New("cycle operator_active_seconds exceeds end-to-end cycle time")
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
		return fmt.Errorf("improvement schema %q, want %q", r.Schema, ImprovementSchema)
	}
	engine := strings.ToLower(strings.TrimSpace(r.Engine))
	if strings.Contains(engine, "llama") || !strings.HasPrefix(engine, "fak-native") {
		return errors.New("improvement engine must name explicit fak-native provenance without llama.cpp fallback")
	}
	if strings.TrimSpace(r.Hypothesis) == "" {
		return errors.New("improvement hypothesis is required")
	}
	module := strings.TrimSpace(r.ChangedModule)
	if !strings.Contains(module, "@r") || !strings.Contains(module, "+g") {
		return errors.New("improvement changed_module must name module@rev")
	}
	if r.Baseline.Unit != "milliseconds" || r.Candidate.Unit != "milliseconds" ||
		!finite(r.Baseline.Value) || !finite(r.Candidate.Value) || r.Baseline.Value <= 0 || r.Candidate.Value <= 0 {
		return errors.New("improvement baseline and candidate require positive finite milliseconds")
	}
	if strings.TrimSpace(r.Quality.Gate) == "" || r.Quality.BaselinePassed == nil || r.Quality.CandidatePassed == nil || r.Quality.Parity == nil ||
		!*r.Quality.BaselinePassed || !*r.Quality.CandidatePassed || !*r.Quality.Parity {
		return errors.New("improvement quality gate requires passing baseline/candidate parity")
	}
	if r.BaselineEnvelope != r.CandidateEnvelope || !validEnvelope(r.BaselineEnvelope) {
		return errors.New("improvement baseline and candidate operating envelopes must be complete and matched")
	}
	if r.NetTrueGain.Unit != "percent" || r.NetTrueGain.OverheadUnit != "milliseconds" ||
		r.NetTrueGain.IncludesOverhead == nil || !*r.NetTrueGain.IncludesOverhead ||
		!finite(r.NetTrueGain.OverheadValue) || r.NetTrueGain.OverheadValue < 0 || !finite(r.NetTrueGain.Value) {
		return errors.New("improvement net_true_gain must be percent and explicitly include nonnegative millisecond overhead")
	}
	wantGain := (r.Baseline.Value - (r.Candidate.Value + r.NetTrueGain.OverheadValue)) / r.Baseline.Value * 100
	if wantGain <= 0 || math.Abs(wantGain-r.NetTrueGain.Value) > 1e-9 {
		return errors.New("improvement net_true_gain does not equal baseline minus candidate and overhead")
	}
	if strings.TrimSpace(r.Causal.Ablation) == "" || r.Causal.Unit != "milliseconds" || r.Causal.IsolatesChange == nil || !*r.Causal.IsolatesChange ||
		!finite(r.Causal.ControlValue) || !finite(r.Causal.TreatmentValue) || r.Causal.ControlValue <= r.Causal.TreatmentValue {
		return errors.New("improvement causal binding requires an isolating, positive ablation in milliseconds")
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
			return fmt.Errorf("improvement derivation for %s: unsupported unit %q", d.ID, d.Unit)
		}
		d.Current = &value
		d.Source = "improvement:" + r.Schema
		d.EvidenceKind = "improvement_receipt"
		d.Engine = r.Engine
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
		return 0, fmt.Errorf("unsupported duration unit %q", unit)
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
	return 0, fmt.Errorf("unsupported throughput unit %q", unit)
}

func coverageInUnit(fraction float64, unit string) (float64, error) {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "fraction", "ratio":
		return fraction, nil
	case "percent", "percentage", "%":
		return fraction * 100, nil
	default:
		return 0, fmt.Errorf("unsupported coverage unit %q", unit)
	}
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func Score(e Evidence) Report {
	r := Report{Schema: ReportSchema, Snapshot: e.Snapshot, TargetMultiplier: e.TargetMultiplier}
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
	return r
}

func Compare(current *Report, prior Report) error {
	if prior.Schema != ReportSchema {
		return fmt.Errorf("prior schema %q, want %q", prior.Schema, ReportSchema)
	}
	pm := map[string]Result{}
	for _, d := range prior.Dimensions {
		pm[d.ID] = d
	}
	c := &Comparison{PriorSnapshot: prior.Snapshot}
	for _, d := range current.Dimensions {
		p, ok := pm[d.ID]
		if !ok {
			return fmt.Errorf("prior snapshot missing dimension %q", d.ID)
		}
		c.Deltas = append(c.Deltas, Delta{ID: d.ID, PriorStatus: p.Status, CurrentStatus: d.Status, PriorRatio: p.NormalizedRatio, CurrentRatio: d.NormalizedRatio})
	}
	current.Comparison = c
	return nil
}

func DecodeReport(r io.Reader) (Report, error) {
	var p Report
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return p, err
	}
	return p, nil
}

func RenderHuman(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "performance RSI: %s | target %.0fx | UNKNOWN debt %d\n", r.Snapshot, r.TargetMultiplier, r.UnknownDebt)
	fmt.Fprintf(&b, "dominant bottleneck: %s\n", r.DominantBottleneck)
	for _, d := range r.Dimensions {
		fmt.Fprintf(&b, "%-25s %-7s current=%s target=%s ratio=%s source=%s next=%s\n", d.ID, d.Status, number(d.Current), number(d.Target), number(d.NormalizedRatio), d.Source, d.NextAction)
	}
	if r.Comparison != nil {
		fmt.Fprintf(&b, "compared with: %s\n", r.Comparison.PriorSnapshot)
	}
	return strings.TrimRight(b.String(), "\n")
}

func RenderMarkdown(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Performance RSI — %s\n\n- Explicit target: **%.0fx** (unsaturated)\n- Dominant bottleneck: `%s`\n- UNKNOWN debt: **%d**\n\n", r.Snapshot, r.TargetMultiplier, r.DominantBottleneck, r.UnknownDebt)
	b.WriteString("| Dimension | Status | Current | Target | Normalized ratio | Source | Next action |\n|---|---:|---:|---:|---:|---|---|\n")
	for _, d := range r.Dimensions {
		fmt.Fprintf(&b, "| %s | %s | %s %s | %s %s | %s | %s | %s |\n", d.ID, d.Status, number(d.Current), d.Unit, number(d.Target), d.Unit, number(d.NormalizedRatio), d.Source, d.NextAction)
	}
	if r.Comparison != nil {
		fmt.Fprintf(&b, "\nCompared with `%s`.\n", r.Comparison.PriorSnapshot)
	}
	return strings.TrimRight(b.String(), "\n")
}

func number(v *float64) string {
	if v == nil {
		return "UNKNOWN"
	}
	if math.IsInf(*v, 1) {
		return "+Inf"
	}
	return fmt.Sprintf("%.6g", *v)
}

func MarshalJSON(r Report) ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

func SortResultsForTest(rs []Result) {
	sort.Slice(rs, func(i, j int) bool { return rs[i].ID < rs[j].ID })
}
