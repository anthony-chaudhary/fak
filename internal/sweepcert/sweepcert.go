package sweepcert

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Axis declares the complete, ordered one-dimensional sampling domain. A
// closed endpoint means the domain, rather than merely this sample, ends there.
type Axis struct {
	Name        string    `json:"name"`
	Unit        string    `json:"unit"`
	Coordinates []float64 `json:"coordinates"`
	LowerClosed bool      `json:"lower_closed"`
	UpperClosed bool      `json:"upper_closed"`
}

// Binding is one comparability identity. Callers bind every fact whose change
// makes two points incomparable.
type Binding struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// Envelope is the canonical identity shared by every comparable point.
type Envelope struct {
	Axis     Axis      `json:"axis"`
	Bindings []Binding `json:"bindings"`
}

// Qwen38NativeEngine is the only engine identity accepted by the Qwen3.8
// prefill certification helpers. It deliberately excludes vague "native"
// labels and external fallback engines.
const Qwen38NativeEngine = "fak-native"

// Qwen38NativeProvenance binds every identity required to compare Qwen3.8
// prefill samples. Values describe a receipt; they are not inferred or filled
// from ambient state.
type Qwen38NativeProvenance struct {
	Artifact       string
	ArtifactDigest string
	Engine         string
	EngineCommit   string
	Backend        string
	Node           string
	Hardware       string
	Tokenizer      string
	Output         string
	Reset          string
	Order          string
	Capacity       string
}

// RangeClosure records proof that the declared prompt-length range reaches the
// real operating-envelope boundary. Without proof, a terminal maximum remains
// censored even when it is the largest measured value.
type RangeClosure struct {
	Proven   bool
	Evidence string
}

// NewQwen38NativeEnvelope constructs the comparable identity for a Qwen3.8
// native prefill sweep. It requires at least three prompt lengths and exact,
// non-empty receipt identities; it never manufactures a live receipt.
func NewQwen38NativeEnvelope(promptLengths []float64, provenance Qwen38NativeProvenance, closure RangeClosure) (Envelope, error) {
	if len(promptLengths) < 3 {
		return Envelope{}, fmt.Errorf("Qwen3.8 native prefill certification requires at least three prompt lengths")
	}
	fields := []struct {
		name  string
		value string
	}{
		{"artifact", provenance.Artifact}, {"artifact_digest", provenance.ArtifactDigest},
		{"engine", provenance.Engine}, {"engine_commit", provenance.EngineCommit},
		{"backend", provenance.Backend}, {"node", provenance.Node},
		{"hardware", provenance.Hardware}, {"tokenizer", provenance.Tokenizer},
		{"output", provenance.Output}, {"reset", provenance.Reset},
		{"order", provenance.Order}, {"capacity", provenance.Capacity},
	}
	for _, field := range fields {
		if strings.TrimSpace(field.value) == "" {
			return Envelope{}, fmt.Errorf("Qwen3.8 native prefill provenance %s must be non-empty", field.name)
		}
	}
	if provenance.Engine != Qwen38NativeEngine {
		return Envelope{}, fmt.Errorf("Qwen3.8 prefill certification requires engine %q, got %q", Qwen38NativeEngine, provenance.Engine)
	}
	if closure.Proven && strings.TrimSpace(closure.Evidence) == "" {
		return Envelope{}, fmt.Errorf("range closure proof requires evidence")
	}
	bindings := make([]Binding, 0, len(fields)+1)
	for _, field := range fields {
		bindings = append(bindings, Binding{Name: field.name, Value: strings.TrimSpace(field.value)})
	}
	bindings = append(bindings, Binding{Name: "model_family", Value: "qwen3.8"})
	envelope := Envelope{
		Axis: Axis{
			Name:        "prompt_length",
			Unit:        "tokens",
			Coordinates: append([]float64(nil), promptLengths...),
			UpperClosed: closure.Proven,
		},
		Bindings: bindings,
	}
	if _, err := CanonicalEnvelopeDigest(envelope); err != nil {
		return Envelope{}, err
	}
	return envelope, nil
}

// ValidateQwen38NativeEvidence validates receipt identity and explicit
// not-measured semantics without converting missing measurements to zero.
func ValidateQwen38NativeEvidence(e Evidence) error {
	if e.Envelope.Axis.Name != "prompt_length" || e.Envelope.Axis.Unit != "tokens" || len(e.Envelope.Axis.Coordinates) < 3 {
		return fmt.Errorf("evidence is not a Qwen3.8 native prefill prompt-length sweep")
	}
	required := map[string]bool{
		"artifact": false, "artifact_digest": false, "engine": false, "engine_commit": false,
		"backend": false, "node": false, "hardware": false, "tokenizer": false,
		"output": false, "reset": false, "order": false, "capacity": false, "model_family": false,
	}
	for _, binding := range e.Envelope.Bindings {
		if _, ok := required[binding.Name]; ok && strings.TrimSpace(binding.Value) != "" {
			required[binding.Name] = true
		}
		if binding.Name == "engine" && binding.Value != Qwen38NativeEngine {
			return fmt.Errorf("evidence engine must be %q", Qwen38NativeEngine)
		}
		if binding.Name == "model_family" && binding.Value != "qwen3.8" {
			return fmt.Errorf("evidence model family must be qwen3.8")
		}
	}
	for name, present := range required {
		if !present {
			return fmt.Errorf("evidence is missing required provenance binding %q", name)
		}
	}
	digest, err := CanonicalEnvelopeDigest(e.Envelope)
	if err != nil {
		return err
	}
	if e.EnvelopeDigest != digest {
		return fmt.Errorf("evidence envelope digest does not match canonical envelope")
	}
	if len(e.Points) != len(e.Envelope.Axis.Coordinates) {
		return fmt.Errorf("evidence must preserve one point per declared prompt length")
	}
	for i, point := range e.Points {
		if point.Coordinate != e.Envelope.Axis.Coordinates[i] || point.EnvelopeDigest != digest {
			return fmt.Errorf("point %d does not match declared prompt-length envelope", i)
		}
		observation, ok := point.Observations["prefill_throughput"]
		if !ok {
			return fmt.Errorf("point %q is missing prefill_throughput observation", point.ID)
		}
		switch point.Status {
		case PointMeasured:
			if observation.Status != ObservationMeasured || observation.Value == nil || !finite(*observation.Value) {
				return fmt.Errorf("measured point %q requires finite measured prefill throughput", point.ID)
			}
			if observation.Provenance.EnvelopeDigest != digest {
				return fmt.Errorf("measured point %q has mismatched observation provenance", point.ID)
			}
		case PointNotMeasured:
			if observation.Status != ObservationNotMeasured || observation.Value != nil || strings.TrimSpace(observation.Reason) == "" {
				return fmt.Errorf("not-measured point %q requires nil value and explicit reason", point.ID)
			}
		case PointInvalid:
			if strings.TrimSpace(point.InvalidReason) == "" {
				return fmt.Errorf("invalid point %q requires a reason", point.ID)
			}
		default:
			return fmt.Errorf("point %q has unknown status %q", point.ID, point.Status)
		}
	}
	return nil
}

// CanonicalEnvelopeDigest returns an order-independent digest of bindings and
// an order-sensitive digest of the declared axis coordinates.
func CanonicalEnvelopeDigest(envelope Envelope) (string, error) {
	if err := validateAxis(envelope.Axis); err != nil {
		return "", err
	}
	bindings := append([]Binding(nil), envelope.Bindings...)
	for i := range bindings {
		bindings[i].Name = strings.TrimSpace(bindings[i].Name)
		bindings[i].Value = strings.TrimSpace(bindings[i].Value)
		if bindings[i].Name == "" || bindings[i].Value == "" {
			return "", fmt.Errorf("sweep envelope binding name and value must be non-empty")
		}
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].Name < bindings[j].Name })
	for i := 1; i < len(bindings); i++ {
		if bindings[i-1].Name == bindings[i].Name {
			return "", fmt.Errorf("duplicate sweep envelope binding %q", bindings[i].Name)
		}
	}
	canonical := struct {
		Schema   string    `json:"schema"`
		Axis     Axis      `json:"axis"`
		Bindings []Binding `json:"bindings"`
	}{Schema: "fak.sweep-envelope.v1", Axis: envelope.Axis, Bindings: bindings}
	raw, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("marshal sweep envelope: %w", err)
	}
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func validateAxis(axis Axis) error {
	if strings.TrimSpace(axis.Name) == "" || strings.TrimSpace(axis.Unit) == "" {
		return fmt.Errorf("sweep axis name and unit must be non-empty")
	}
	if len(axis.Coordinates) < 2 {
		return fmt.Errorf("sweep axis requires at least two declared coordinates")
	}
	for i, coordinate := range axis.Coordinates {
		if !finite(coordinate) {
			return fmt.Errorf("sweep axis coordinate %d is non-finite", i)
		}
		if i > 0 && coordinate <= axis.Coordinates[i-1] {
			return fmt.Errorf("sweep axis coordinates must be strictly increasing")
		}
	}
	return nil
}

type PointStatus string

const (
	PointMeasured    PointStatus = "measured"
	PointNotMeasured PointStatus = "not_measured"
	PointInvalid     PointStatus = "invalid"
)

type ObservationStatus string

const (
	ObservationMeasured    ObservationStatus = "measured"
	ObservationNotMeasured ObservationStatus = "not_measured"
	ObservationInvalid     ObservationStatus = "invalid"
)

// Provenance describes how one raw observation was acquired. Source may vary
// per run; Method, Unit, and EnvelopeDigest must agree for a measured finding.
type Provenance struct {
	Source         string `json:"source"`
	Method         string `json:"method"`
	Unit           string `json:"unit"`
	EnvelopeDigest string `json:"envelope_digest"`
}

// Observation retains its raw value even when its point is invalidated.
// Not-measured is represented by a nil Value, never numeric zero.
type Observation struct {
	Status     ObservationStatus `json:"status"`
	Value      *float64          `json:"value,omitempty"`
	Provenance Provenance        `json:"provenance,omitempty"`
	Reason     string            `json:"reason,omitempty"`
}

// Point is one declared coordinate and all observations made there.
type Point struct {
	ID             string                 `json:"id"`
	Coordinate     float64                `json:"coordinate"`
	Status         PointStatus            `json:"status"`
	EnvelopeDigest string                 `json:"envelope_digest"`
	InvalidReason  string                 `json:"invalid_reason,omitempty"`
	Observations   map[string]Observation `json:"observations,omitempty"`
}

// Evidence contains exactly one point per declared coordinate. Failed runs
// remain explicit not-measured points, so gaps cannot silently disappear.
type Evidence struct {
	Envelope               Envelope `json:"envelope"`
	EnvelopeDigest         string   `json:"envelope_digest"`
	DeclaredInvalidReasons []string `json:"declared_invalid_reasons,omitempty"`
	Points                 []Point  `json:"points"`
}

type FindingStatus string

const (
	FindingMeasured        FindingStatus = "measured"
	FindingLeftCensored    FindingStatus = "left_censored"
	FindingRightCensored   FindingStatus = "right_censored"
	FindingNotIdentifiable FindingStatus = "not_identifiable"
	FindingInvalid         FindingStatus = "invalid"
)

type FindingKind string

const (
	FindingObservedExtremum    FindingKind = "observed_extremum"
	FindingConstrainedExtremum FindingKind = "constrained_extremum"
	FindingFirstThreshold      FindingKind = "first_threshold"
	FindingStableSuffix        FindingKind = "stable_suffix_threshold"
)

type Interval struct {
	LowerPointID string `json:"lower_point_id,omitempty"`
	UpperPointID string `json:"upper_point_id,omitempty"`
}

// Finding references the original points instead of copying or smoothing raw
// values. Support includes every declared point that could affect the result.
type Finding struct {
	Kind             FindingKind   `json:"kind"`
	Status           FindingStatus `json:"status"`
	Metric           string        `json:"metric"`
	PointID          string        `json:"point_id,omitempty"`
	SupportingPoints []string      `json:"supporting_points"`
	Interval         *Interval     `json:"interval,omitempty"`
	Reason           string        `json:"reason,omitempty"`
}

type ExtremumDirection string

const (
	Maximum ExtremumDirection = "maximum"
	Minimum ExtremumDirection = "minimum"
)

type ThresholdOperator string

const (
	AtOrAbove            ThresholdOperator = "at_or_above"
	AtOrBelow            ThresholdOperator = "at_or_below"
	ThresholdDirectAbove ThresholdOperator = AtOrAbove
)

type Constraint struct {
	Metric    string            `json:"metric"`
	Operator  ThresholdOperator `json:"operator"`
	Threshold float64           `json:"threshold"`
}

// ObservedExtremum selects an observed maximum or minimum without pretending
// that a sampled endpoint is an interior optimum.
func ObservedExtremum(evidence Evidence, metric string, direction ExtremumDirection) Finding {
	return extremum(evidence, metric, direction, nil, FindingObservedExtremum)
}

// ConstrainedExtremum selects an extremum among points satisfying every
// caller-declared constraint. Constraint mathematics remains explicit.
func ConstrainedExtremum(evidence Evidence, metric string, direction ExtremumDirection, constraints []Constraint) Finding {
	return extremum(evidence, metric, direction, constraints, FindingConstrainedExtremum)
}

func extremum(e Evidence, metric string, direction ExtremumDirection, constraints []Constraint, kind FindingKind) Finding {
	finding, points, ok := prepare(e, kind, metric, constraintMetrics(constraints))
	if !ok {
		return finding
	}
	if direction != Maximum && direction != Minimum {
		return invalidFinding(kind, metric, points, "undeclared extremum direction")
	}
	for _, constraint := range constraints {
		if strings.TrimSpace(constraint.Metric) == "" || !finite(constraint.Threshold) || constraint.Operator != AtOrAbove && constraint.Operator != AtOrBelow {
			return invalidFinding(kind, metric, points, "invalid constraint declaration")
		}
	}
	eligible := make([]Point, 0, len(points))
	for _, point := range points {
		passes := true
		for _, constraint := range constraints {
			if !thresholdPass(*point.Observations[constraint.Metric].Value, constraint.Operator, constraint.Threshold) {
				passes = false
				break
			}
		}
		if passes {
			eligible = append(eligible, point)
		}
	}
	if len(eligible) == 0 {
		finding.Status = FindingNotIdentifiable
		finding.Reason = "no measured point satisfies every declared constraint"
		return finding
	}
	best := eligible[0]
	for _, point := range eligible[1:] {
		value, bestValue := *point.Observations[metric].Value, *best.Observations[metric].Value
		better := direction == Maximum && value > bestValue || direction == Minimum && value < bestValue
		if better || value == bestValue && point.Coordinate < best.Coordinate {
			best = point
		}
	}
	finding.PointID = best.ID
	finding.Status = endpointStatus(e.Envelope.Axis, best.Coordinate)
	return finding
}

// FirstThreshold finds the first sampled point meeting a threshold. It never
// interpolates a coordinate, and refuses curves with more than one regime.
func FirstThreshold(e Evidence, metric string, operator ThresholdOperator, threshold float64) Finding {
	return thresholdFinding(e, FindingFirstThreshold, metric, operator, threshold)
}

// StableSuffixThreshold finds the earliest point at which the threshold holds
// for the entire remaining suffix. A later collapse makes the regime
// non-identifiable instead of moving the finding past the collapse.
func StableSuffixThreshold(e Evidence, metric string, operator ThresholdOperator, threshold float64) Finding {
	return thresholdFinding(e, FindingStableSuffix, metric, operator, threshold)
}

func thresholdFinding(e Evidence, kind FindingKind, metric string, operator ThresholdOperator, threshold float64) Finding {
	finding, points, ok := prepare(e, kind, metric, nil)
	if !ok {
		return finding
	}
	if !finite(threshold) || operator != AtOrAbove && operator != AtOrBelow {
		return invalidFinding(kind, metric, points, "invalid threshold declaration")
	}
	passes := make([]bool, len(points))
	transitions := 0
	for i, point := range points {
		passes[i] = thresholdPass(*point.Observations[metric].Value, operator, threshold)
		if i > 0 && passes[i] != passes[i-1] {
			transitions++
		}
	}
	if transitions > 1 || passes[0] && transitions > 0 {
		finding.Status = FindingNotIdentifiable
		finding.Reason = "multiple threshold regimes; no single crossing is identifiable"
		return finding
	}
	if passes[0] {
		finding.Status = FindingLeftCensored
		finding.PointID = points[0].ID
		finding.Interval = &Interval{UpperPointID: points[0].ID}
		return finding
	}
	selected := -1
	for i := range passes {
		if passes[i] {
			selected = i
			break
		}
	}
	if selected < 0 {
		finding.Status = FindingRightCensored
		finding.PointID = points[len(points)-1].ID
		finding.Interval = &Interval{LowerPointID: points[len(points)-1].ID}
		return finding
	}
	if kind == FindingStableSuffix {
		for i := selected; i < len(passes); i++ {
			if !passes[i] {
				finding.Status = FindingNotIdentifiable
				finding.Reason = "apparent threshold regime has a later collapse"
				return finding
			}
		}
	}
	finding.Status = FindingMeasured
	finding.PointID = points[selected].ID
	finding.Interval = &Interval{LowerPointID: points[selected-1].ID, UpperPointID: points[selected].ID}
	return finding
}

func prepare(e Evidence, kind FindingKind, metric string, extraMetrics []string) (Finding, []Point, bool) {
	support := make([]string, len(e.Points))
	for i := range e.Points {
		support[i] = e.Points[i].ID
	}
	finding := Finding{Kind: kind, Metric: metric, SupportingPoints: support}
	digest, err := CanonicalEnvelopeDigest(e.Envelope)
	if err != nil || e.EnvelopeDigest != digest {
		if err == nil {
			err = fmt.Errorf("envelope digest does not match canonical envelope")
		}
		finding.Status, finding.Reason = FindingInvalid, err.Error()
		return finding, e.Points, false
	}
	if strings.TrimSpace(metric) == "" {
		finding.Status, finding.Reason = FindingInvalid, "finding metric is empty"
		return finding, e.Points, false
	}
	allowed := make(map[string]bool, len(e.DeclaredInvalidReasons))
	for _, reason := range e.DeclaredInvalidReasons {
		if reason == "" || allowed[reason] {
			finding.Status, finding.Reason = FindingInvalid, "invalid reason vocabulary is empty or duplicated"
			return finding, e.Points, false
		}
		allowed[reason] = true
	}
	if len(e.Points) != len(e.Envelope.Axis.Coordinates) {
		finding.Status, finding.Reason = FindingNotIdentifiable, "missing declared sweep point"
		return finding, e.Points, false
	}
	points := append([]Point(nil), e.Points...)
	sort.Slice(points, func(i, j int) bool { return points[i].Coordinate < points[j].Coordinate })
	methodUnit := make(map[string]string)
	seenID := make(map[string]bool, len(points))
	metrics := append([]string{metric}, extraMetrics...)
	for i, point := range points {
		if strings.TrimSpace(point.ID) == "" || seenID[point.ID] || !finite(point.Coordinate) || point.Coordinate != e.Envelope.Axis.Coordinates[i] {
			return invalidFinding(kind, metric, points, "duplicate, undeclared, or non-finite point coordinate/identity"), points, false
		}
		seenID[point.ID] = true
		switch point.Status {
		case PointInvalid:
			if !allowed[point.InvalidReason] {
				return invalidFinding(kind, metric, points, "point uses an undeclared invalid reason"), points, false
			}
			return invalidFinding(kind, metric, points, "invalid point: "+point.InvalidReason), points, false
		case PointNotMeasured:
			finding.Status, finding.Reason = FindingNotIdentifiable, "a declared point was not measured; transition remains an interval/unknown"
			finding.Interval = &Interval{}
			if i > 0 {
				finding.Interval.LowerPointID = points[i-1].ID
			}
			if i+1 < len(points) {
				finding.Interval.UpperPointID = points[i+1].ID
			}
			return finding, points, false
		case PointMeasured:
			if point.EnvelopeDigest != digest {
				return invalidFinding(kind, metric, points, "mixed point identity"), points, false
			}
		default:
			return invalidFinding(kind, metric, points, "undeclared point status"), points, false
		}
		for _, requiredMetric := range metrics {
			observation, exists := point.Observations[requiredMetric]
			if !exists || observation.Status != ObservationMeasured || observation.Value == nil {
				finding.Status, finding.Reason = FindingNotIdentifiable, "required observation was not measured"
				return finding, points, false
			}
			if !finite(*observation.Value) {
				return invalidFinding(kind, metric, points, "non-finite observation"), points, false
			}
			p := observation.Provenance
			if strings.TrimSpace(p.Source) == "" || strings.TrimSpace(p.Method) == "" || strings.TrimSpace(p.Unit) == "" || p.EnvelopeDigest != digest {
				return invalidFinding(kind, metric, points, "missing or incompatible observation provenance"), points, false
			}
			key := p.Method + "\x00" + p.Unit
			if previous, exists := methodUnit[requiredMetric]; exists && previous != key {
				return invalidFinding(kind, metric, points, "mixed observation provenance"), points, false
			}
			methodUnit[requiredMetric] = key
		}
	}
	finding.SupportingPoints = pointIDs(points)
	return finding, points, true
}

// ValidateFinding checks finding-to-point lineage independently of the fold.
func ValidateFinding(e Evidence, finding Finding) error {
	if finding.Kind != FindingObservedExtremum && finding.Kind != FindingConstrainedExtremum && finding.Kind != FindingFirstThreshold && finding.Kind != FindingStableSuffix {
		return fmt.Errorf("undeclared finding kind %q", finding.Kind)
	}
	if finding.Status != FindingMeasured && finding.Status != FindingLeftCensored && finding.Status != FindingRightCensored && finding.Status != FindingNotIdentifiable && finding.Status != FindingInvalid {
		return fmt.Errorf("undeclared finding status %q", finding.Status)
	}
	if len(finding.SupportingPoints) != len(e.Points) {
		return fmt.Errorf("finding references %d points; want every one of %d declared points", len(finding.SupportingPoints), len(e.Points))
	}
	known := make(map[string]bool, len(e.Points))
	for _, point := range e.Points {
		known[point.ID] = true
	}
	seen := make(map[string]bool, len(finding.SupportingPoints))
	for _, id := range finding.SupportingPoints {
		if !known[id] || seen[id] {
			return fmt.Errorf("finding has unknown or duplicate supporting point %q", id)
		}
		seen[id] = true
	}
	if finding.Interval != nil {
		if finding.Interval.LowerPointID != "" && !known[finding.Interval.LowerPointID] {
			return fmt.Errorf("finding interval has unknown lower point %q", finding.Interval.LowerPointID)
		}
		if finding.Interval.UpperPointID != "" && !known[finding.Interval.UpperPointID] {
			return fmt.Errorf("finding interval has unknown upper point %q", finding.Interval.UpperPointID)
		}
	}
	if finding.Status == FindingMeasured || finding.Status == FindingLeftCensored || finding.Status == FindingRightCensored {
		if len(e.Points) < 2 || finding.PointID == "" || !known[finding.PointID] {
			return fmt.Errorf("comparative finding lacks two points or a selected point")
		}
		_, _, ok := prepare(e, finding.Kind, finding.Metric, nil)
		if !ok {
			return fmt.Errorf("measured/censored finding is not supported by comparable measured observations")
		}
	}
	return nil
}

func invalidFinding(kind FindingKind, metric string, points []Point, reason string) Finding {
	return Finding{Kind: kind, Status: FindingInvalid, Metric: metric, SupportingPoints: pointIDs(points), Reason: reason}
}

func endpointStatus(axis Axis, coordinate float64) FindingStatus {
	if coordinate == axis.Coordinates[0] && !axis.LowerClosed {
		return FindingLeftCensored
	}
	if coordinate == axis.Coordinates[len(axis.Coordinates)-1] && !axis.UpperClosed {
		return FindingRightCensored
	}
	return FindingMeasured
}

func thresholdPass(value float64, operator ThresholdOperator, threshold float64) bool {
	if operator == AtOrAbove {
		return value >= threshold
	}
	if operator == AtOrBelow {
		return value <= threshold
	}
	return false
}

func constraintMetrics(constraints []Constraint) []string {
	metrics := make([]string, 0, len(constraints))
	seen := make(map[string]bool)
	for _, constraint := range constraints {
		if !seen[constraint.Metric] {
			seen[constraint.Metric] = true
			metrics = append(metrics, constraint.Metric)
		}
	}
	return metrics
}

func pointIDs(points []Point) []string {
	ids := make([]string, len(points))
	for i := range points {
		ids[i] = points[i].ID
	}
	return ids
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// MonotonicityType specifies expected directional monotonicity of a metric surface.
type MonotonicityType string

const (
	MonotonicNonDecreasing MonotonicityType = "non-decreasing"
)

// PrefillSample represents one observation point in a prompt prefill response surface.
type PrefillSample struct {
	PromptLength int     `json:"prompt_length"`
	Throughput   float64 `json:"throughput"`
	LatencyMs    float64 `json:"latency_ms"`
}

// CertificationCriteria specifies quality and stability boundaries for certification.
type CertificationCriteria struct {
	MaxVariance       float64          `json:"max_variance,omitempty"`
	MaxCV             float64          `json:"max_cv,omitempty"`
	Monotonicity      MonotonicityType `json:"monotonicity,omitempty"`
	MaxTurnDurationMs float64          `json:"max_turn_duration_ms,omitempty"`
}

// SweepCertResult captures the evaluation verdict and cryptographic receipt digest.
type SweepCertResult struct {
	Passed     bool     `json:"passed"`
	Monotonic  bool     `json:"monotonic"`
	Digest     string   `json:"digest"`
	Violations []string `json:"violations,omitempty"`
}

// TokenLatencyDistribution captures inter-token latency measurements and jitter tolerances.
type TokenLatencyDistribution struct {
	LatenciesMs []float64 `json:"latencies_ms"`
	MaxJitterMs float64   `json:"max_jitter_ms"`
}

// AgentTurnSample captures resource usage and latency of a single conversational turn.
type AgentTurnSample struct {
	TurnIndex        int     `json:"turn_index"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	CumulativeTokens int     `json:"cumulative_tokens"`
	TurnDurationMs   float64 `json:"turn_duration_ms"`
}

// SweepCert evaluates empirical benchmark sweeps against strict performance bounds.
type SweepCert struct{}

// NewSweepCert constructs a new SweepCert validator.
func NewSweepCert() *SweepCert {
	return &SweepCert{}
}

// CertifyPrefillSurface verifies throughput monotonicity and statistical variance.
func (s *SweepCert) CertifyPrefillSurface(samples []PrefillSample, criteria *CertificationCriteria) (*SweepCertResult, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("empty prefill points")
	}
	for _, sample := range samples {
		if !finite(sample.Throughput) || sample.Throughput < 0 {
			return nil, fmt.Errorf("invalid throughput: %v", sample.Throughput)
		}
	}

	monotonic := true
	var violations []string

	for i := 1; i < len(samples); i++ {
		if samples[i].Throughput < samples[i-1].Throughput {
			monotonic = false
			violations = append(violations, fmt.Sprintf("monotonicity violation: throughput regressed from %f to %f at prompt length %d",
				samples[i-1].Throughput, samples[i].Throughput, samples[i].PromptLength))
		}
	}

	throughputs := make([]float64, len(samples))
	for i, sample := range samples {
		throughputs[i] = sample.Throughput
	}
	variance, cv := computeVarianceAndCV(throughputs)

	if criteria != nil {
		if criteria.Monotonicity == MonotonicNonDecreasing && !monotonic {
			// already recorded violation
		}
		if criteria.MaxVariance > 0 && variance > criteria.MaxVariance {
			violations = append(violations, fmt.Sprintf("variance %f exceeds ceiling %f", variance, criteria.MaxVariance))
		}
		if criteria.MaxCV > 0 && cv > criteria.MaxCV {
			violations = append(violations, fmt.Sprintf("CV %f exceeds ceiling %f", cv, criteria.MaxCV))
		}
	}

	h := sha256.New()
	b, _ := json.Marshal(samples)
	h.Write(b)
	digest := "sha256:" + hex.EncodeToString(h.Sum(nil))

	passed := len(violations) == 0 && monotonic
	return &SweepCertResult{
		Passed:     passed,
		Monotonic:  monotonic,
		Digest:     digest,
		Violations: violations,
	}, nil
}

// CertifyTokenLatencyDistribution verifies inter-token latency jitter and distribution stability.
func (s *SweepCert) CertifyTokenLatencyDistribution(dist TokenLatencyDistribution, criteria *CertificationCriteria) (*SweepCertResult, error) {
	if len(dist.LatenciesMs) == 0 {
		return nil, fmt.Errorf("empty token latencies")
	}
	for _, lat := range dist.LatenciesMs {
		if !finite(lat) || lat < 0 {
			return nil, fmt.Errorf("invalid token latency: %v", lat)
		}
	}

	var violations []string
	if dist.MaxJitterMs > 0 {
		for i := 1; i < len(dist.LatenciesMs); i++ {
			jitter := math.Abs(dist.LatenciesMs[i] - dist.LatenciesMs[i-1])
			if jitter > dist.MaxJitterMs {
				violations = append(violations, fmt.Sprintf("jitter %f ms exceeds max %f ms between token %d and %d",
					jitter, dist.MaxJitterMs, i-1, i))
				break
			}
		}
	}

	variance, cv := computeVarianceAndCV(dist.LatenciesMs)
	if criteria != nil {
		if criteria.MaxVariance > 0 && variance > criteria.MaxVariance {
			violations = append(violations, fmt.Sprintf("latency variance %f exceeds ceiling %f", variance, criteria.MaxVariance))
		}
		if criteria.MaxCV > 0 && cv > criteria.MaxCV {
			violations = append(violations, fmt.Sprintf("latency CV %f exceeds ceiling %f", cv, criteria.MaxCV))
		}
	}

	h := sha256.New()
	b, _ := json.Marshal(dist)
	h.Write(b)
	digest := "sha256:" + hex.EncodeToString(h.Sum(nil))

	return &SweepCertResult{
		Passed:     len(violations) == 0,
		Monotonic:  true,
		Digest:     digest,
		Violations: violations,
	}, nil
}

// CertifyAgentTurns verifies monotonic cumulative tokens and bounded turn latency.
func (s *SweepCert) CertifyAgentTurns(turns []AgentTurnSample, criteria *CertificationCriteria) (*SweepCertResult, error) {
	if len(turns) == 0 {
		return nil, fmt.Errorf("empty agent turns")
	}
	for _, turn := range turns {
		if !finite(turn.TurnDurationMs) || turn.TurnDurationMs < 0 {
			return nil, fmt.Errorf("invalid turn duration: %v", turn.TurnDurationMs)
		}
	}

	var violations []string
	for i := 1; i < len(turns); i++ {
		if turns[i].CumulativeTokens < turns[i-1].CumulativeTokens {
			violations = append(violations, fmt.Sprintf("cumulative token rollback at turn %d: %d < %d",
				turns[i].TurnIndex, turns[i].CumulativeTokens, turns[i-1].CumulativeTokens))
			break
		}
	}

	durations := make([]float64, len(turns))
	for i, turn := range turns {
		durations[i] = turn.TurnDurationMs
		if criteria != nil && criteria.MaxTurnDurationMs > 0 && turn.TurnDurationMs > criteria.MaxTurnDurationMs {
			violations = append(violations, fmt.Sprintf("turn %d duration %f ms exceeds ceiling %f ms",
				turn.TurnIndex, turn.TurnDurationMs, criteria.MaxTurnDurationMs))
		}
	}

	variance, cv := computeVarianceAndCV(durations)
	if criteria != nil {
		if criteria.MaxVariance > 0 && variance > criteria.MaxVariance {
			violations = append(violations, fmt.Sprintf("turn duration variance %f exceeds ceiling %f", variance, criteria.MaxVariance))
		}
		if criteria.MaxCV > 0 && cv > criteria.MaxCV {
			violations = append(violations, fmt.Sprintf("turn duration CV %f exceeds ceiling %f", cv, criteria.MaxCV))
		}
	}

	h := sha256.New()
	b, _ := json.Marshal(turns)
	h.Write(b)
	digest := "sha256:" + hex.EncodeToString(h.Sum(nil))

	return &SweepCertResult{
		Passed:     len(violations) == 0,
		Monotonic:  true,
		Digest:     digest,
		Violations: violations,
	}, nil
}

func computeVarianceAndCV(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	var sqDiffSum float64
	for _, v := range values {
		diff := v - mean
		sqDiffSum += diff * diff
	}
	variance := sqDiffSum / float64(len(values))
	stdDev := math.Sqrt(variance)
	cv := 0.0
	if mean > 0 {
		cv = stdDev / mean
	}
	return variance, cv
}
