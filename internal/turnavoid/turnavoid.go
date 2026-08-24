// Package turnavoid replays immutable, independently labelled turn decisions.
//
// It deliberately accounts for complete committed model turns separately from
// retained-turn token or latency reductions. The package is offline and pure:
// callers supply rows (or an io.Reader), and no clock, filesystem, model, or
// tool execution participates in the fold.
package turnavoid

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

const (
	TraceSchemaVersion  = "fak.turnavoid.trace/v1"
	ReportSchemaVersion = "fak.turnavoid.report/v1"

	Honest               = "HONEST"
	HonestWithheldCredit = "HONEST_WITHHELD_CREDIT"
)

type Arm string

const (
	ArmControl    Arm = "control"
	ArmExactReuse Arm = "exact-reuse"
	ArmFusedBatch Arm = "fused-batch"
)

var allArms = []Arm{ArmControl, ArmExactReuse, ArmFusedBatch}

type Mechanism string

const (
	MechanismControl       Mechanism = "control"
	MechanismExactReuse    Mechanism = "exact-reuse"
	MechanismFusedBatch    Mechanism = "fused-batch"
	MechanismProviderCache Mechanism = "provider-cache"
	MechanismUnknown       Mechanism = "unknown"
)

type Lifecycle string

const (
	LifecycleOpportunity        Lifecycle = "opportunity"
	LifecycleAttempted          Lifecycle = "attempted"
	LifecycleRealized           Lifecycle = "realized"
	LifecycleInvalidated        Lifecycle = "invalidated"
	LifecycleCounterfactualOnly Lifecycle = "counterfactual_only"
	LifecycleUnknown            Lifecycle = "unknown"
)

type Reason string

const (
	ReasonBaseline                 Reason = "baseline"
	ReasonExactMatch               Reason = "exact-match"
	ReasonSerialRoundTripCollapsed Reason = "serial-round-trip-collapsed"
	ReasonRetainedTurnCheaper      Reason = "retained-turn-cheaper"
	ReasonRequiredEffectSuppressed Reason = "required-effect-suppressed"
	ReasonCounterfactualOnly       Reason = "counterfactual-only"
	ReasonValidationOverhead       Reason = "validation-overhead"
	ReasonNotApplicable            Reason = "not-applicable"
	ReasonUnknown                  Reason = "unknown"
)

// GrossWork is committed model/tool work before turn-avoidance overhead. A
// retained-turn reduction must already be reflected in these candidate values;
// RetainedTurnReduction is attribution and is never subtracted a second time.
type GrossWork struct {
	ModelLatencyMS float64 `json:"model_latency_ms"`
	ToolLatencyMS  float64 `json:"tool_latency_ms"`
	ModelCostUSD   float64 `json:"model_cost_usd"`
	ToolCostUSD    float64 `json:"tool_cost_usd"`
}

func (w GrossWork) latencyMS() float64 { return w.ModelLatencyMS + w.ToolLatencyMS }
func (w GrossWork) costUSD() float64   { return w.ModelCostUSD + w.ToolCostUSD }

// Overhead keeps every candidate-side cost outside gross model/tool work
// visible. Retry is explicit rather than being hidden in recovery.
type Overhead struct {
	ValidationLatencyMS  float64 `json:"validation_latency_ms"`
	SpeculationLatencyMS float64 `json:"speculation_latency_ms"`
	RetryLatencyMS       float64 `json:"retry_latency_ms"`
	RecoveryLatencyMS    float64 `json:"recovery_latency_ms"`
	ValidationCostUSD    float64 `json:"validation_cost_usd"`
	SpeculationCostUSD   float64 `json:"speculation_cost_usd"`
	RetryCostUSD         float64 `json:"retry_cost_usd"`
	RecoveryCostUSD      float64 `json:"recovery_cost_usd"`
}

func (o Overhead) latencyMS() float64 {
	return o.ValidationLatencyMS + o.SpeculationLatencyMS + o.RetryLatencyMS + o.RecoveryLatencyMS
}

func (o Overhead) costUSD() float64 {
	return o.ValidationCostUSD + o.SpeculationCostUSD + o.RetryCostUSD + o.RecoveryCostUSD
}

type EffectObservation struct {
	IndependentObserver string   `json:"independent_observer"`
	ControlRequired     []string `json:"control_required"`
	CandidateRequired   []string `json:"candidate_required"`
}

type RetainedTurnReduction struct {
	Tokens    int64   `json:"tokens"`
	LatencyMS float64 `json:"latency_ms"`
}

// Row is one immutable input evaluated under one arm. Every
// trace_id/unit_id/turn_index input must occur exactly once under every arm with
// the same digest, pre-turn decision basis, and control observation.
type Row struct {
	SchemaVersion            string                 `json:"schema_version"`
	TraceID                  string                 `json:"trace_id"`
	UnitID                   string                 `json:"unit_id"`
	TurnIndex                int                    `json:"turn_index"`
	DecisionBasisThroughTurn int                    `json:"decision_basis_through_turn"`
	InputDigest              string                 `json:"input_digest"`
	Arm                      Arm                    `json:"arm"`
	Mechanism                Mechanism              `json:"mechanism"`
	Lifecycle                Lifecycle              `json:"lifecycle"`
	Reason                   Reason                 `json:"reason"`
	ControlCommittedTurns    int                    `json:"control_committed_model_turns"`
	CandidateCommittedTurns  int                    `json:"candidate_committed_model_turns"`
	Effects                  EffectObservation      `json:"required_effect_observation"`
	ControlGross             GrossWork              `json:"control_gross"`
	CandidateGross           GrossWork              `json:"candidate_gross"`
	Overhead                 Overhead               `json:"overhead"`
	RetainedTurnReduction    *RetainedTurnReduction `json:"retained_turn_reduction,omitempty"`
}

type LifecycleCounts struct {
	Opportunity        int `json:"opportunity"`
	Attempted          int `json:"attempted"`
	Realized           int `json:"realized"`
	Invalidated        int `json:"invalidated"`
	CounterfactualOnly int `json:"counterfactual_only"`
	Unknown            int `json:"unknown"`
}

type Accounting struct {
	GrossControlLatencyMS   float64 `json:"gross_control_latency_ms"`
	GrossCandidateLatencyMS float64 `json:"gross_candidate_latency_ms"`
	GrossLatencySavedMS     float64 `json:"gross_latency_saved_ms"`
	OverheadLatencyMS       float64 `json:"overhead_latency_ms"`
	NetCandidateLatencyMS   float64 `json:"net_candidate_latency_ms"`
	NetLatencySavedMS       float64 `json:"net_latency_saved_ms"`
	GrossControlCostUSD     float64 `json:"gross_control_cost_usd"`
	GrossCandidateCostUSD   float64 `json:"gross_candidate_cost_usd"`
	GrossCostSavedUSD       float64 `json:"gross_cost_saved_usd"`
	OverheadCostUSD         float64 `json:"overhead_cost_usd"`
	NetCandidateCostUSD     float64 `json:"net_candidate_cost_usd"`
	NetCostSavedUSD         float64 `json:"net_cost_saved_usd"`
}

type Attribution struct {
	Mechanism            Mechanism  `json:"mechanism"`
	Lifecycle            Lifecycle  `json:"lifecycle"`
	Reason               Reason     `json:"reason"`
	Decisions            int        `json:"decisions"`
	RealizedTurnsAvoided int        `json:"realized_turns_avoided"`
	WithheldTurns        int        `json:"withheld_turns"`
	RequiredPreserved    int        `json:"required_effects_preserved"`
	RequiredSuppressed   int        `json:"needed_effects_suppressed"`
	Accounting           Accounting `json:"accounting"`
}

type ArmReport struct {
	Arm                          Arm             `json:"arm"`
	Decisions                    int             `json:"decisions"`
	ControlCommittedModelTurns   int             `json:"control_committed_model_turns"`
	CandidateCommittedModelTurns int             `json:"candidate_committed_model_turns"`
	RealizedTurnsAvoided         int             `json:"realized_turns_avoided"`
	WithheldTurns                int             `json:"withheld_turns"`
	Lifecycle                    LifecycleCounts `json:"lifecycle"`
	RetainedTurnsMadeCheaper     int             `json:"retained_turns_made_cheaper"`
	RetainedTurnTokensReduced    int64           `json:"retained_turn_tokens_reduced"`
	RetainedTurnLatencyReducedMS float64         `json:"retained_turn_latency_reduced_ms"`
	RequiredEffectsPreserved     int             `json:"required_effects_preserved"`
	NeededEffectsSuppressed      int             `json:"needed_effects_suppressed"`
	Accounting                   Accounting      `json:"accounting"`
	Attribution                  []Attribution   `json:"attribution"`
	HonestyVerdict               string          `json:"honesty_verdict"`
}

type Report struct {
	SchemaVersion      string      `json:"schema_version"`
	TraceSchemaVersion string      `json:"trace_schema_version"`
	Rows               int         `json:"rows"`
	ImmutableInputs    int         `json:"immutable_inputs"`
	Arms               []ArmReport `json:"arms"`
	HonestyVerdict     string      `json:"honesty_verdict"`
}

// Replay strictly decodes a JSONL trace and folds it. Unknown fields, blank
// lines, multiple JSON values on one line, and lines over 4 MiB are rejected.
func Replay(r io.Reader) (Report, error) {
	rows, err := DecodeJSONL(r)
	if err != nil {
		return Report{}, err
	}
	return Fold(rows)
}

func DecodeJSONL(r io.Reader) ([]Row, error) {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var rows []Row
	for line := 1; s.Scan(); line++ {
		raw := bytes.TrimSpace(s.Bytes())
		if len(raw) == 0 {
			return nil, fmt.Errorf("line %d: blank JSONL row", line)
		}
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		var row Row
		if err := dec.Decode(&row); err != nil {
			return nil, fmt.Errorf("line %d: decode: %w", line, err)
		}
		var trailing any
		if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
			if err == nil {
				return nil, fmt.Errorf("line %d: multiple JSON values", line)
			}
			return nil, fmt.Errorf("line %d: trailing JSON: %w", line, err)
		}
		rows = append(rows, row)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("read JSONL: %w", err)
	}
	if len(rows) == 0 {
		return nil, errors.New("trace has no rows")
	}
	return rows, nil
}

type inputKey struct {
	trace string
	unit  string
	turn  int
}

type sequenceKey struct {
	trace string
	unit  string
	arm   Arm
}

// Fold validates the immutable cross-arm envelope before giving any realized
// credit. Input order is significant: turn indexes must increase within each
// trace/unit/arm sequence.
func Fold(rows []Row) (Report, error) {
	if len(rows) == 0 {
		return Report{}, errors.New("trace has no rows")
	}
	groups := make(map[inputKey]map[Arm]Row)
	lastTurn := make(map[sequenceKey]int)
	for i, row := range rows {
		if err := validateRow(row); err != nil {
			return Report{}, fmt.Errorf("row %d: %w", i+1, err)
		}
		sk := sequenceKey{trace: row.TraceID, unit: row.UnitID, arm: row.Arm}
		if last, ok := lastTurn[sk]; ok && row.TurnIndex <= last {
			return Report{}, fmt.Errorf("row %d: turn_index %d is not greater than prior %d for %s/%s/%s", i+1, row.TurnIndex, last, row.TraceID, row.UnitID, row.Arm)
		}
		lastTurn[sk] = row.TurnIndex

		key := inputKey{trace: row.TraceID, unit: row.UnitID, turn: row.TurnIndex}
		if groups[key] == nil {
			groups[key] = make(map[Arm]Row)
		}
		if _, exists := groups[key][row.Arm]; exists {
			return Report{}, fmt.Errorf("row %d: duplicate arm %q for %s/%s turn %d", i+1, row.Arm, row.TraceID, row.UnitID, row.TurnIndex)
		}
		groups[key][row.Arm] = row
	}

	orderedKeys := make([]inputKey, 0, len(groups))
	for key := range groups {
		orderedKeys = append(orderedKeys, key)
	}
	sort.Slice(orderedKeys, func(i, j int) bool {
		if orderedKeys[i].trace != orderedKeys[j].trace {
			return orderedKeys[i].trace < orderedKeys[j].trace
		}
		if orderedKeys[i].unit != orderedKeys[j].unit {
			return orderedKeys[i].unit < orderedKeys[j].unit
		}
		return orderedKeys[i].turn < orderedKeys[j].turn
	})

	for _, key := range orderedKeys {
		group := groups[key]
		for _, arm := range allArms {
			if _, ok := group[arm]; !ok {
				return Report{}, fmt.Errorf("%s/%s turn %d: missing arm %q", key.trace, key.unit, key.turn, arm)
			}
		}
		control := group[ArmControl]
		for _, arm := range allArms[1:] {
			if err := sameImmutableControl(control, group[arm]); err != nil {
				return Report{}, fmt.Errorf("%s/%s turn %d arm %s: %w", key.trace, key.unit, key.turn, arm, err)
			}
		}
	}

	report := Report{
		SchemaVersion:      ReportSchemaVersion,
		TraceSchemaVersion: TraceSchemaVersion,
		Rows:               len(rows),
		ImmutableInputs:    len(groups),
		HonestyVerdict:     Honest,
	}
	for _, arm := range allArms {
		ar := foldArm(arm, orderedKeys, groups)
		if ar.HonestyVerdict == HonestWithheldCredit {
			report.HonestyVerdict = HonestWithheldCredit
		}
		report.Arms = append(report.Arms, ar)
	}
	return report, nil
}

func foldArm(arm Arm, keys []inputKey, groups map[inputKey]map[Arm]Row) ArmReport {
	ar := ArmReport{Arm: arm, HonestyVerdict: Honest}
	attrs := make(map[string]*Attribution)
	for _, key := range keys {
		row := groups[key][arm]
		ar.Decisions++
		ar.ControlCommittedModelTurns += row.ControlCommittedTurns
		ar.CandidateCommittedModelTurns += row.CandidateCommittedTurns
		addLifecycle(&ar.Lifecycle, row.Lifecycle)

		preserved, suppressed := compareEffects(row.Effects.ControlRequired, row.Effects.CandidateRequired)
		ar.RequiredEffectsPreserved += preserved
		ar.NeededEffectsSuppressed += suppressed
		potential := max(row.ControlCommittedTurns-row.CandidateCommittedTurns, 0)
		credited := 0
		if row.Lifecycle == LifecycleRealized && suppressed == 0 && len(row.Effects.ControlRequired) == len(row.Effects.CandidateRequired) {
			credited = potential
		}
		withheld := potential - credited
		ar.RealizedTurnsAvoided += credited
		ar.WithheldTurns += withheld

		if reduction := row.RetainedTurnReduction; reduction != nil {
			ar.RetainedTurnsMadeCheaper++
			ar.RetainedTurnTokensReduced += reduction.Tokens
			ar.RetainedTurnLatencyReducedMS += reduction.LatencyMS
		}
		accountRow(&ar.Accounting, row)

		attrKey := string(row.Mechanism) + "\x00" + string(row.Lifecycle) + "\x00" + string(row.Reason)
		attr := attrs[attrKey]
		if attr == nil {
			attr = &Attribution{Mechanism: row.Mechanism, Lifecycle: row.Lifecycle, Reason: row.Reason}
			attrs[attrKey] = attr
		}
		attr.Decisions++
		attr.RealizedTurnsAvoided += credited
		attr.WithheldTurns += withheld
		attr.RequiredPreserved += preserved
		attr.RequiredSuppressed += suppressed
		accountRow(&attr.Accounting, row)
	}
	if ar.WithheldTurns > 0 || ar.NeededEffectsSuppressed > 0 {
		ar.HonestyVerdict = HonestWithheldCredit
	}
	for _, attr := range attrs {
		ar.Attribution = append(ar.Attribution, *attr)
	}
	sort.Slice(ar.Attribution, func(i, j int) bool {
		a, b := ar.Attribution[i], ar.Attribution[j]
		if a.Mechanism != b.Mechanism {
			return a.Mechanism < b.Mechanism
		}
		if a.Lifecycle != b.Lifecycle {
			return a.Lifecycle < b.Lifecycle
		}
		return a.Reason < b.Reason
	})
	return ar
}

func accountRow(a *Accounting, row Row) {
	controlLatency := row.ControlGross.latencyMS()
	candidateLatency := row.CandidateGross.latencyMS()
	overheadLatency := row.Overhead.latencyMS()
	controlCost := row.ControlGross.costUSD()
	candidateCost := row.CandidateGross.costUSD()
	overheadCost := row.Overhead.costUSD()
	a.GrossControlLatencyMS += controlLatency
	a.GrossCandidateLatencyMS += candidateLatency
	a.GrossLatencySavedMS += cleanZero(controlLatency - candidateLatency)
	a.OverheadLatencyMS += overheadLatency
	a.NetCandidateLatencyMS += candidateLatency + overheadLatency
	a.NetLatencySavedMS += cleanZero(controlLatency - candidateLatency - overheadLatency)
	a.GrossControlCostUSD += controlCost
	a.GrossCandidateCostUSD += candidateCost
	a.GrossCostSavedUSD += cleanZero(controlCost - candidateCost)
	a.OverheadCostUSD += overheadCost
	a.NetCandidateCostUSD += candidateCost + overheadCost
	a.NetCostSavedUSD += cleanZero(controlCost - candidateCost - overheadCost)
}

func cleanZero(v float64) float64 {
	if v == 0 {
		return 0
	}
	return v
}

func addLifecycle(c *LifecycleCounts, state Lifecycle) {
	switch state {
	case LifecycleOpportunity:
		c.Opportunity++
	case LifecycleAttempted:
		c.Attempted++
	case LifecycleRealized:
		c.Realized++
	case LifecycleInvalidated:
		c.Invalidated++
	case LifecycleCounterfactualOnly:
		c.CounterfactualOnly++
	case LifecycleUnknown:
		c.Unknown++
	}
}

func compareEffects(control, candidate []string) (preserved, suppressed int) {
	ci := make(map[string]struct{}, len(candidate))
	for _, effect := range candidate {
		ci[effect] = struct{}{}
	}
	for _, effect := range control {
		if _, ok := ci[effect]; ok {
			preserved++
		} else {
			suppressed++
		}
	}
	return preserved, suppressed
}

func validateRow(row Row) error {
	if row.SchemaVersion != TraceSchemaVersion {
		return fmt.Errorf("schema_version %q, want %q", row.SchemaVersion, TraceSchemaVersion)
	}
	if err := stableID("trace_id", row.TraceID); err != nil {
		return err
	}
	if err := stableID("unit_id", row.UnitID); err != nil {
		return err
	}
	if row.TurnIndex < 0 {
		return errors.New("turn_index must be non-negative")
	}
	if row.DecisionBasisThroughTurn < -1 || row.DecisionBasisThroughTurn >= row.TurnIndex {
		return fmt.Errorf("decision_basis_through_turn %d must be >= -1 and less than turn_index %d (same-turn result leakage is forbidden)", row.DecisionBasisThroughTurn, row.TurnIndex)
	}
	if err := digest(row.InputDigest); err != nil {
		return err
	}
	if !validArm(row.Arm) {
		return fmt.Errorf("unknown arm %q", row.Arm)
	}
	if !validMechanism(row.Mechanism) {
		return fmt.Errorf("unknown mechanism %q (use %q when intentionally unclassified)", row.Mechanism, MechanismUnknown)
	}
	if !validLifecycle(row.Lifecycle) {
		return fmt.Errorf("unknown lifecycle %q (use %q when intentionally unclassified)", row.Lifecycle, LifecycleUnknown)
	}
	if !validReason(row.Reason) {
		return fmt.Errorf("unknown reason %q (use %q when intentionally unclassified)", row.Reason, ReasonUnknown)
	}
	if row.ControlCommittedTurns < 0 || row.CandidateCommittedTurns < 0 {
		return errors.New("committed model-turn counts must be non-negative")
	}
	if err := stableID("required_effect_observation.independent_observer", row.Effects.IndependentObserver); err != nil {
		return err
	}
	if err := labels("control_required", row.Effects.ControlRequired); err != nil {
		return err
	}
	if err := labels("candidate_required", row.Effects.CandidateRequired); err != nil {
		return err
	}
	if err := nonNegativeFinite("control_gross", grossValues(row.ControlGross)...); err != nil {
		return err
	}
	if err := nonNegativeFinite("candidate_gross", grossValues(row.CandidateGross)...); err != nil {
		return err
	}
	if err := nonNegativeFinite("overhead", overheadValues(row.Overhead)...); err != nil {
		return err
	}
	if reduction := row.RetainedTurnReduction; reduction != nil {
		if reduction.Tokens < 0 {
			return errors.New("retained_turn_reduction.tokens must be non-negative")
		}
		if err := nonNegativeFinite("retained_turn_reduction.latency_ms", reduction.LatencyMS); err != nil {
			return err
		}
		if reduction.Tokens == 0 && reduction.LatencyMS == 0 {
			return errors.New("retained_turn_reduction must reduce tokens or latency")
		}
		if row.CandidateCommittedTurns == 0 {
			return errors.New("retained_turn_reduction requires at least one retained candidate model turn")
		}
	}
	if row.Arm == ArmControl {
		if row.Mechanism != MechanismControl || row.Reason != ReasonBaseline || row.Lifecycle != LifecycleRealized {
			return errors.New("control arm must use control/realized/baseline attribution")
		}
		if row.ControlCommittedTurns != row.CandidateCommittedTurns || row.ControlGross != row.CandidateGross || row.Overhead != (Overhead{}) || row.RetainedTurnReduction != nil || !equalStrings(row.Effects.ControlRequired, row.Effects.CandidateRequired) {
			return errors.New("control arm candidate must equal its control observation with zero overhead and no retained-turn attribution")
		}
	} else if row.Mechanism == MechanismControl {
		return errors.New("candidate arm cannot use the control mechanism")
	}
	return nil
}

func sameImmutableControl(control, candidate Row) error {
	if control.InputDigest != candidate.InputDigest {
		return errors.New("input_digest differs across arms")
	}
	if control.DecisionBasisThroughTurn != candidate.DecisionBasisThroughTurn {
		return errors.New("decision_basis_through_turn differs across arms")
	}
	if control.ControlCommittedTurns != candidate.ControlCommittedTurns {
		return errors.New("control_committed_model_turns differs across arms")
	}
	if control.ControlGross != candidate.ControlGross {
		return errors.New("control_gross differs across arms")
	}
	if control.Effects.IndependentObserver != candidate.Effects.IndependentObserver {
		return errors.New("independent observer differs across arms")
	}
	if !equalStrings(control.Effects.ControlRequired, candidate.Effects.ControlRequired) {
		return errors.New("control required effects differ across arms")
	}
	return nil
}

func stableID(name, value string) error {
	if value == "" || strings.TrimSpace(value) != value {
		return fmt.Errorf("%s must be a non-empty, whitespace-stable value", name)
	}
	return nil
}

func digest(value string) error {
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return errors.New("input_digest must be sha256:<64 lowercase hex characters>")
	}
	hexPart := strings.TrimPrefix(value, "sha256:")
	if hexPart != strings.ToLower(hexPart) {
		return errors.New("input_digest must use lowercase hex")
	}
	if _, err := hex.DecodeString(hexPart); err != nil {
		return fmt.Errorf("input_digest: %w", err)
	}
	return nil
}

func labels(name string, values []string) error {
	for i, value := range values {
		if err := stableID("required_effect_observation."+name, value); err != nil {
			return err
		}
		if i > 0 && values[i-1] >= value {
			return fmt.Errorf("required_effect_observation.%s must be sorted and unique", name)
		}
	}
	return nil
}

func nonNegativeFinite(name string, values ...float64) error {
	for _, value := range values {
		if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("%s values must be non-negative and finite", name)
		}
	}
	return nil
}

func grossValues(w GrossWork) []float64 {
	return []float64{w.ModelLatencyMS, w.ToolLatencyMS, w.ModelCostUSD, w.ToolCostUSD}
}

func overheadValues(o Overhead) []float64 {
	return []float64{o.ValidationLatencyMS, o.SpeculationLatencyMS, o.RetryLatencyMS, o.RecoveryLatencyMS, o.ValidationCostUSD, o.SpeculationCostUSD, o.RetryCostUSD, o.RecoveryCostUSD}
}

func validArm(v Arm) bool {
	return v == ArmControl || v == ArmExactReuse || v == ArmFusedBatch
}

func validMechanism(v Mechanism) bool {
	switch v {
	case MechanismControl, MechanismExactReuse, MechanismFusedBatch, MechanismProviderCache, MechanismUnknown:
		return true
	default:
		return false
	}
}

func validLifecycle(v Lifecycle) bool {
	switch v {
	case LifecycleOpportunity, LifecycleAttempted, LifecycleRealized, LifecycleInvalidated, LifecycleCounterfactualOnly, LifecycleUnknown:
		return true
	default:
		return false
	}
}

func validReason(v Reason) bool {
	switch v {
	case ReasonBaseline, ReasonExactMatch, ReasonSerialRoundTripCollapsed, ReasonRetainedTurnCheaper, ReasonRequiredEffectSuppressed, ReasonCounterfactualOnly, ReasonValidationOverhead, ReasonNotApplicable, ReasonUnknown:
		return true
	default:
		return false
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// RenderText returns a deterministic, concise operator rendering. The report's
// JSON form remains the complete artifact.
func RenderText(report Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "turnavoid %s: %s (%d immutable inputs, %d rows)\n", report.SchemaVersion, report.HonestyVerdict, report.ImmutableInputs, report.Rows)
	for _, arm := range report.Arms {
		fmt.Fprintf(&b, "%s: committed=%d control=%d realized-avoided=%d withheld=%d invalidated=%d counterfactual=%d retained-cheaper=%d\n",
			arm.Arm, arm.CandidateCommittedModelTurns, arm.ControlCommittedModelTurns, arm.RealizedTurnsAvoided, arm.WithheldTurns,
			arm.Lifecycle.Invalidated, arm.Lifecycle.CounterfactualOnly, arm.RetainedTurnsMadeCheaper)
		fmt.Fprintf(&b, "  effects preserved=%d suppressed=%d; latency-saved-ms gross=%.3f net=%.3f (overhead=%.3f); cost-saved-usd gross=%.6f net=%.6f (overhead=%.6f)\n",
			arm.RequiredEffectsPreserved, arm.NeededEffectsSuppressed,
			arm.Accounting.GrossLatencySavedMS, arm.Accounting.NetLatencySavedMS, arm.Accounting.OverheadLatencyMS,
			arm.Accounting.GrossCostSavedUSD, arm.Accounting.NetCostSavedUSD, arm.Accounting.OverheadCostUSD)
	}
	return b.String()
}
