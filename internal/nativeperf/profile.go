package nativeperf

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

const (
	ProfileSchema        = "fak-native-performance-profile/v1"
	ClassificationSchema = "fak-native-performance-bottleneck/v1"
)

var requiredProfilePhases = [...]string{"load-setup", "prefill", "first-token", "steady-decode", "verification", "teardown"}

var (
	ErrProfileWindowInvalid       = errors.New("native-performance profile counter window invalid")
	ErrProfileEvidenceUnavailable = errors.New("native-performance profile evidence unavailable")
)

type AttributionUnavailableReason string

const (
	AttributionUnavailableBackend AttributionUnavailableReason = "backend-does-not-expose-dispatch-attribution"
	AttributionUnavailableCapture AttributionUnavailableReason = "capture-tool-does-not-export-dispatch-attribution"
)

type ProfileBundle struct {
	Schema                 string                  `json:"schema"`
	EnvelopeID             string                  `json:"envelope_id"`
	Execution              ExecutionIdentity       `json:"execution"`
	Phases                 []ProfilePhase          `json:"phases"`
	CounterWindow          *ProfileCounterWindow   `json:"counter_window,omitempty"`
	Metal                  *MetalCounters          `json:"metal,omitempty"`
	CUDA                   *CUDACounters           `json:"cuda,omitempty"`
	Attributions           []DispatchAttribution   `json:"attributions,omitempty"`
	AttributionUnavailable *AttributionUnavailable `json:"attribution_unavailable,omitempty"`
	CounterComparisons     []CounterComparison     `json:"counter_comparisons,omitempty"`
	Override               *SelectionOverride      `json:"selection_override,omitempty"`
}

type ProfilePhase struct {
	Name                 string  `json:"name"`
	StartMilliseconds    float64 `json:"start_milliseconds"`
	DurationMilliseconds float64 `json:"duration_milliseconds"`
}

type ProfileCounterWindow struct {
	Backend                string   `json:"backend"`
	Phase                  string   `json:"phase"`
	StartMilliseconds      float64  `json:"start_milliseconds"`
	DurationMilliseconds   float64  `json:"duration_milliseconds"`
	DecodeSteps            int      `json:"decode_steps"`
	HostLaunchMilliseconds *float64 `json:"host_launch_milliseconds,omitempty"`
	HostSetupMilliseconds  *float64 `json:"host_setup_milliseconds,omitempty"`
	HostTimingSource       string   `json:"host_timing_source,omitempty"`
}

type MetalCounters struct {
	CommandBuffers       int     `json:"command_buffers"`
	Encoders             int     `json:"encoders"`
	DispatchMilliseconds float64 `json:"dispatch_milliseconds"`
	WaitMilliseconds     float64 `json:"wait_milliseconds"`
	ResidentBytes        uint64  `json:"resident_bytes"`
	WorkingSetBytes      uint64  `json:"working_set_bytes"`
}

type CUDACounters struct {
	Launches                    int     `json:"launches"`
	OccupancyPercent            float64 `json:"occupancy_percent"`
	AchievedBandwidthGBS        float64 `json:"achieved_bandwidth_gbs"`
	PeakBandwidthGBS            float64 `json:"peak_bandwidth_gbs"`
	AchievedComputeTFLOPS       float64 `json:"achieved_compute_tflops"`
	PeakComputeTFLOPS           float64 `json:"peak_compute_tflops"`
	SynchronizationMilliseconds float64 `json:"synchronization_milliseconds"`
}

type DispatchAttribution struct {
	Name    string `json:"name"`
	Layer   int    `json:"layer"`
	LeverID string `json:"lever_id"`
	Count   int    `json:"count"`
}

type AttributionUnavailable struct {
	Reason AttributionUnavailableReason `json:"reason"`
	Detail string                       `json:"detail"`
}

// CounterComparison is a reserved fail-closed input surface. Profile v1
// classifies one backend capture and never normalizes unlike backend counters.
type CounterComparison struct {
	LeftBackend  string `json:"left_backend"`
	LeftCounter  string `json:"left_counter"`
	RightBackend string `json:"right_backend"`
	RightCounter string `json:"right_counter"`
}

type SelectionOverride struct {
	IssueNumber int    `json:"issue_number"`
	Reason      string `json:"reason"`
}

type BottleneckClassification struct {
	Schema             string   `json:"schema"`
	EnvelopeID         string   `json:"envelope_id"`
	Class              string   `json:"class"`
	Confidence         string   `json:"confidence"`
	Evidence           []string `json:"evidence"`
	RecommendedLeverID string   `json:"recommended_lever_id"`
}

func DecodeProfile(data []byte) (ProfileBundle, error) {
	var p ProfileBundle
	rawDecoder := json.NewDecoder(bytes.NewReader(data))
	var raw json.RawMessage
	if err := rawDecoder.Decode(&raw); err != nil {
		return p, fmt.Errorf("decode profile: %w", err)
	}
	var trailing any
	if err := rawDecoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return p, fmt.Errorf("decode profile: multiple JSON values")
		}
		return p, fmt.Errorf("decode profile: trailing data: %w", err)
	}
	if err := validateProfileFieldPresence(raw); err != nil {
		return p, fmt.Errorf("decode profile: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return p, fmt.Errorf("decode profile: %w", err)
	}
	return p, nil
}

func validateProfileFieldPresence(data []byte) error {
	if err := validateCounterWindowJSONKeys(data); err != nil {
		return err
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		return err
	}
	if err := requireJSONFields("profile", root, "schema", "envelope_id", "execution", "phases"); err != nil {
		return err
	}

	var execution map[string]json.RawMessage
	if err := json.Unmarshal(root["execution"], &execution); err != nil {
		return fmt.Errorf("execution must be an object: %w", err)
	}
	if err := requireJSONFields("execution", execution, "engine", "forward_path", "fallback_count"); err != nil {
		return err
	}

	var phases []map[string]json.RawMessage
	if err := json.Unmarshal(root["phases"], &phases); err != nil {
		return fmt.Errorf("phases must be an array of objects: %w", err)
	}
	for i, phase := range phases {
		if err := requireJSONFields(fmt.Sprintf("phases[%d]", i), phase, "name", "start_milliseconds", "duration_milliseconds"); err != nil {
			return err
		}
	}
	if raw, ok := root["counter_window"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		var window map[string]json.RawMessage
		if err := json.Unmarshal(raw, &window); err != nil {
			return fmt.Errorf("%w: counter_window must be an object: %v", ErrProfileWindowInvalid, err)
		}
		if err := requireJSONFields("counter_window", window, "backend", "phase", "start_milliseconds", "duration_milliseconds", "decode_steps"); err != nil {
			return fmt.Errorf("%w: %v", ErrProfileWindowInvalid, err)
		}
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		var typedWindow ProfileCounterWindow
		if err := dec.Decode(&typedWindow); err != nil {
			return fmt.Errorf("%w: %v", ErrProfileWindowInvalid, err)
		}
	}

	if raw, ok := root["metal"]; ok {
		var counters map[string]json.RawMessage
		if err := json.Unmarshal(raw, &counters); err != nil {
			return fmt.Errorf("metal must be an object: %w", err)
		}
		if err := requireJSONFields("metal", counters, "command_buffers", "encoders", "dispatch_milliseconds", "wait_milliseconds", "resident_bytes", "working_set_bytes"); err != nil {
			return err
		}
	}
	if raw, ok := root["cuda"]; ok {
		var counters map[string]json.RawMessage
		if err := json.Unmarshal(raw, &counters); err != nil {
			return fmt.Errorf("cuda must be an object: %w", err)
		}
		if err := requireJSONFields("cuda", counters, "launches", "occupancy_percent", "achieved_bandwidth_gbs", "peak_bandwidth_gbs", "achieved_compute_tflops", "peak_compute_tflops", "synchronization_milliseconds"); err != nil {
			return err
		}
	}
	if raw, ok := root["attributions"]; ok {
		var attributions []map[string]json.RawMessage
		if err := json.Unmarshal(raw, &attributions); err != nil {
			return fmt.Errorf("attributions must be an array of objects: %w", err)
		}
		for i, attribution := range attributions {
			if err := requireJSONFields(fmt.Sprintf("attributions[%d]", i), attribution, "name", "layer", "lever_id", "count"); err != nil {
				return err
			}
		}
	}
	if raw, ok := root["attribution_unavailable"]; ok {
		var unavailable map[string]json.RawMessage
		if err := json.Unmarshal(raw, &unavailable); err != nil {
			return fmt.Errorf("attribution_unavailable must be an object: %w", err)
		}
		if err := requireJSONFields("attribution_unavailable", unavailable, "reason", "detail"); err != nil {
			return err
		}
	}
	if raw, ok := root["selection_override"]; ok {
		var override map[string]json.RawMessage
		if err := json.Unmarshal(raw, &override); err != nil {
			return fmt.Errorf("selection_override must be an object: %w", err)
		}
		if err := requireJSONFields("selection_override", override, "issue_number", "reason"); err != nil {
			return err
		}
	}
	return nil
}

func validateCounterWindowJSONKeys(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	opening, err := dec.Token()
	if err != nil {
		return err
	}
	if opening != json.Delim('{') {
		return nil
	}
	seen := false
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return err
		}
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return err
		}
		name, _ := key.(string)
		if !strings.EqualFold(name, "counter_window") {
			continue
		}
		if name != "counter_window" || seen {
			return fmt.Errorf("%w: counter_window must occur once with its canonical spelling", ErrProfileWindowInvalid)
		}
		seen = true
		if err := validateCounterWindowJSONMembers(raw); err != nil {
			return err
		}
	}
	return nil
}

func validateCounterWindowJSONMembers(data []byte) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	opening, err := dec.Token()
	if err != nil || opening != json.Delim('{') {
		return fmt.Errorf("%w: counter_window must be an object", ErrProfileWindowInvalid)
	}
	seen := make(map[string]bool, 8)
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return fmt.Errorf("%w: %v", ErrProfileWindowInvalid, err)
		}
		name, _ := key.(string)
		canonical := ""
		for _, field := range [...]string{"backend", "phase", "start_milliseconds", "duration_milliseconds", "decode_steps", "host_launch_milliseconds", "host_setup_milliseconds", "host_timing_source"} {
			if strings.EqualFold(name, field) {
				canonical = field
				break
			}
		}
		if canonical == "" || name != canonical || seen[canonical] {
			return fmt.Errorf("%w: counter_window member %q must be known, unique and canonically spelled", ErrProfileWindowInvalid, name)
		}
		seen[canonical] = true
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return fmt.Errorf("%w: %v", ErrProfileWindowInvalid, err)
		}
	}
	return nil
}

func requireJSONFields(label string, object map[string]json.RawMessage, fields ...string) error {
	for _, field := range fields {
		raw, ok := object[field]
		if !ok {
			return fmt.Errorf("%s is missing required field %q", label, field)
		}
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return fmt.Errorf("%s required field %q must not be null", label, field)
		}
	}
	return nil
}

func ValidateProfile(graph Graph, p ProfileBundle) error {
	if err := Validate(graph); err != nil {
		return fmt.Errorf("invalid native-performance profile graph: %w", err)
	}

	var findings []string
	if p.Schema != ProfileSchema {
		findings = append(findings, fmt.Sprintf("schema must be %q", ProfileSchema))
	}

	env := profileEnvelope(graph, p.EnvelopeID)
	if env == nil {
		findings = append(findings, "profile references unknown envelope")
	}
	if p.Execution.Engine != "fak-native" || strings.TrimSpace(p.Execution.ForwardPath) == "" || p.Execution.FallbackCount != 0 {
		findings = append(findings, "profile requires fak-native execution identity with zero fallback")
	}
	if env != nil && p.Execution.ForwardPath != env.ForwardPath {
		findings = append(findings, "execution.forward_path must exactly match the selected envelope")
	}

	findings = append(findings, validateProfilePhases(p.Phases)...)
	windowFindings := validateProfileCounterWindow(env, p)
	findings = append(findings, windowFindings...)
	if env != nil {
		findings = append(findings, validateBackendCounters(*env, p)...)
	}
	findings = append(findings, validateAttributions(graph, p)...)

	if len(p.CounterComparisons) != 0 {
		findings = append(findings, "profile counter comparisons are unsupported; preserve backend-specific counters without cross-backend normalization")
	}
	if p.Override != nil && (p.Override.IssueNumber <= 0 || strings.TrimSpace(p.Override.Reason) == "" || private(p.Override.Reason)) {
		findings = append(findings, "selection override requires a positive issue number and scrubbed issue-backed reason")
	}

	if len(findings) > 0 {
		sort.Strings(findings)
		if len(windowFindings) > 0 {
			return fmt.Errorf("%w: invalid native-performance profile: %s", ErrProfileWindowInvalid, strings.Join(findings, "; "))
		}
		return fmt.Errorf("invalid native-performance profile: %s", strings.Join(findings, "; "))
	}
	return nil
}

func validateProfileCounterWindow(env *Envelope, p ProfileBundle) []string {
	w := p.CounterWindow
	if w == nil {
		return nil
	}
	var findings []string
	if env == nil || w.Backend != env.Backend {
		findings = append(findings, "counter_window backend must match the selected envelope")
	}
	if env == nil || w.DecodeSteps <= 0 || w.DecodeSteps != env.DecodeTokens-1 {
		findings = append(findings, "counter_window decode_steps must match the selected envelope's steady decode steps")
	}
	var phase *ProfilePhase
	for i := range p.Phases {
		if p.Phases[i].Name == "steady-decode" {
			phase = &p.Phases[i]
			break
		}
	}
	if w.Phase != "steady-decode" || phase == nil || !nonnegativeFinite(w.StartMilliseconds) || !positiveFinite(w.DurationMilliseconds) ||
		w.StartMilliseconds != phase.StartMilliseconds || w.DurationMilliseconds != phase.DurationMilliseconds {
		findings = append(findings, "counter_window must exactly match the steady-decode phase boundaries")
	}
	observedHost := w.HostLaunchMilliseconds != nil || w.HostSetupMilliseconds != nil
	if observedHost && w.HostTimingSource != "host-monotonic-exclusive" || !observedHost && w.HostTimingSource != "" {
		findings = append(findings, "counter_window observed host timing requires host-monotonic-exclusive and no source without observations")
	}
	hostMilliseconds := 0.0
	for _, measured := range []*float64{w.HostLaunchMilliseconds, w.HostSetupMilliseconds} {
		if measured == nil {
			continue
		}
		if !nonnegativeFinite(*measured) {
			findings = append(findings, "counter_window host timing must be finite and non-negative")
		}
		hostMilliseconds += *measured
	}
	if !nonnegativeFinite(hostMilliseconds) || hostMilliseconds > w.DurationMilliseconds {
		findings = append(findings, "counter_window exclusive launch and setup timing must not exceed steady decode")
	}
	if p.Metal != nil && (!nonnegativeFinite(p.Metal.WaitMilliseconds) || p.Metal.WaitMilliseconds > w.DurationMilliseconds) {
		findings = append(findings, "counter_window Metal wait time must fit within steady decode")
	}
	if p.CUDA != nil && (!nonnegativeFinite(p.CUDA.SynchronizationMilliseconds) || p.CUDA.SynchronizationMilliseconds > w.DurationMilliseconds) {
		findings = append(findings, "counter_window CUDA synchronization time must fit within steady decode")
	}
	return findings
}

func validateProfilePhases(phases []ProfilePhase) []string {
	if len(phases) != len(requiredProfilePhases) {
		return []string{"profile must contain every ordered phase"}
	}

	var findings []string
	var previousEnd float64
	for i, phase := range phases {
		if phase.Name != requiredProfilePhases[i] {
			findings = append(findings, "profile phases are missing or out of order")
		}
		if !nonnegativeFinite(phase.StartMilliseconds) {
			findings = append(findings, fmt.Sprintf("phase %q start must be finite and non-negative", phase.Name))
		}
		if !positiveFinite(phase.DurationMilliseconds) {
			findings = append(findings, fmt.Sprintf("phase %q duration must be finite and positive", phase.Name))
		}
		if !finite(phase.StartMilliseconds) || !finite(phase.DurationMilliseconds) {
			continue
		}
		end := phase.StartMilliseconds + phase.DurationMilliseconds
		if !finite(end) {
			findings = append(findings, fmt.Sprintf("phase %q end must be finite", phase.Name))
			continue
		}
		if i > 0 && phase.StartMilliseconds < previousEnd {
			findings = append(findings, fmt.Sprintf("phase %q overlaps the previous phase", phase.Name))
		}
		previousEnd = end
	}
	return findings
}

func validateBackendCounters(env Envelope, p ProfileBundle) []string {
	switch env.Backend {
	case "metal":
		if p.Metal == nil || p.CUDA != nil {
			return []string{"Metal envelope requires only Metal counters"}
		}
		return validateMetalCounters(*p.Metal)
	case "cuda":
		if p.CUDA == nil || p.Metal != nil {
			return []string{"CUDA envelope requires only CUDA counters"}
		}
		return validateCUDACounters(*p.CUDA)
	default:
		return []string{fmt.Sprintf("profile backend %q is unsupported", env.Backend)}
	}
}

func validateMetalCounters(c MetalCounters) []string {
	var findings []string
	if c.CommandBuffers <= 0 {
		findings = append(findings, "Metal command_buffers must be positive")
	}
	if c.Encoders <= 0 {
		findings = append(findings, "Metal encoders must be positive")
	}
	if !positiveFinite(c.DispatchMilliseconds) {
		findings = append(findings, "Metal dispatch_milliseconds must be finite and positive")
	}
	if !nonnegativeFinite(c.WaitMilliseconds) {
		findings = append(findings, "Metal wait_milliseconds must be finite and non-negative")
	}
	if c.ResidentBytes == 0 {
		findings = append(findings, "Metal resident_bytes must be positive")
	}
	if c.WorkingSetBytes == 0 {
		findings = append(findings, "Metal working_set_bytes must be positive")
	}
	return findings
}

func validateCUDACounters(c CUDACounters) []string {
	var findings []string
	if c.Launches <= 0 {
		findings = append(findings, "CUDA launches must be positive")
	}
	if !nonnegativeFinite(c.OccupancyPercent) || c.OccupancyPercent > 100 {
		findings = append(findings, "CUDA occupancy_percent must be finite and between 0 and 100")
	}
	if !nonnegativeFinite(c.AchievedBandwidthGBS) {
		findings = append(findings, "CUDA achieved_bandwidth_gbs must be finite and non-negative")
	}
	if !positiveFinite(c.PeakBandwidthGBS) {
		findings = append(findings, "CUDA peak_bandwidth_gbs must be finite and positive")
	}
	if !nonnegativeFinite(c.AchievedComputeTFLOPS) {
		findings = append(findings, "CUDA achieved_compute_tflops must be finite and non-negative")
	}
	if !positiveFinite(c.PeakComputeTFLOPS) {
		findings = append(findings, "CUDA peak_compute_tflops must be finite and positive")
	}
	if !nonnegativeFinite(c.SynchronizationMilliseconds) {
		findings = append(findings, "CUDA synchronization_milliseconds must be finite and non-negative")
	}
	if finite(c.AchievedBandwidthGBS) && finite(c.PeakBandwidthGBS) && c.PeakBandwidthGBS > 0 && c.AchievedBandwidthGBS > c.PeakBandwidthGBS {
		findings = append(findings, "CUDA achieved_bandwidth_gbs must not exceed peak_bandwidth_gbs")
	}
	if finite(c.AchievedComputeTFLOPS) && finite(c.PeakComputeTFLOPS) && c.PeakComputeTFLOPS > 0 && c.AchievedComputeTFLOPS > c.PeakComputeTFLOPS {
		findings = append(findings, "CUDA achieved_compute_tflops must not exceed peak_compute_tflops")
	}
	return findings
}

func validateAttributions(graph Graph, p ProfileBundle) []string {
	if len(p.Attributions) == 0 {
		if p.AttributionUnavailable == nil {
			return []string{"dispatch attribution is required or must carry a typed unavailable reason"}
		}
		var findings []string
		if !validAttributionUnavailableReason(p.AttributionUnavailable.Reason) {
			findings = append(findings, "dispatch attribution unavailable reason is unknown")
		}
		if strings.TrimSpace(p.AttributionUnavailable.Detail) == "" || private(p.AttributionUnavailable.Detail) {
			findings = append(findings, "dispatch attribution unavailable detail must be present and scrubbed")
		}
		return findings
	}
	if p.AttributionUnavailable != nil {
		return []string{"dispatch attribution cannot be both available and unavailable"}
	}

	levers := make(map[string]Lever, len(graph.Levers))
	for _, lever := range graph.Levers {
		levers[lever.ID] = lever
	}
	var findings []string
	seen := make(map[string]struct{}, len(p.Attributions))
	leverID := p.Attributions[0].LeverID
	for i, attribution := range p.Attributions {
		if strings.TrimSpace(attribution.Name) == "" || attribution.Layer < 0 || attribution.Count <= 0 {
			findings = append(findings, fmt.Sprintf("dispatch attribution %d is incomplete", i))
		}
		lever, ok := levers[attribution.LeverID]
		if !ok {
			findings = append(findings, fmt.Sprintf("dispatch attribution %d references unknown lever %q", i, attribution.LeverID))
		} else if lever.Applicability.EnvelopeID != p.EnvelopeID {
			findings = append(findings, fmt.Sprintf("dispatch attribution %d mixes envelope %q lever %q into %q", i, lever.Applicability.EnvelopeID, attribution.LeverID, p.EnvelopeID))
		}
		if attribution.LeverID != leverID {
			findings = append(findings, fmt.Sprintf("dispatch attribution %d mixes lever %q with %q", i, attribution.LeverID, leverID))
		}
		key := fmt.Sprintf("%s\x00%d\x00%s", attribution.Name, attribution.Layer, attribution.LeverID)
		if _, ok := seen[key]; ok {
			findings = append(findings, fmt.Sprintf("dispatch attribution %d duplicates name/layer/lever identity", i))
		}
		seen[key] = struct{}{}
	}
	return findings
}

func validAttributionUnavailableReason(reason AttributionUnavailableReason) bool {
	return reason == AttributionUnavailableBackend || reason == AttributionUnavailableCapture
}

func ClassifyProfile(graph Graph, p ProfileBundle) (BottleneckClassification, error) {
	if err := ValidateProfile(graph, p); err != nil {
		return BottleneckClassification{}, err
	}

	classification := BottleneckClassification{
		Schema:     ClassificationSchema,
		EnvelopeID: p.EnvelopeID,
		Class:      "insufficient-evidence",
		Confidence: "low",
		Evidence:   []string{"backend counters lack matching steady-decode-window evidence"},
	}
	w := p.CounterWindow
	if w == nil {
		return classification, nil
	}
	decode := w.DurationMilliseconds
	classification.Evidence = []string{"matching steady-decode counters do not establish a bottleneck; exclusive host launch or setup timing remains below threshold or unobserved"}
	orchestrationLever := "cuda.default-decode-routing"
	if p.Metal != nil {
		orchestrationLever = "metal.command-buffer-amortization"
	}
	switch {
	case w.HostLaunchMilliseconds != nil && *w.HostLaunchMilliseconds/decode >= 0.30 &&
		(w.HostSetupMilliseconds == nil || *w.HostLaunchMilliseconds >= *w.HostSetupMilliseconds):
		classification.Class = "launch-bound"
		classification.Confidence = "medium"
		classification.Evidence = []string{"measured exclusive host launch time is at least 30% of matching steady decode"}
		classification.RecommendedLeverID = orchestrationLever
	case w.HostSetupMilliseconds != nil && *w.HostSetupMilliseconds/decode >= 0.30:
		classification.Class = "cpu-orchestration-bound"
		classification.Confidence = "medium"
		classification.Evidence = []string{"measured exclusive host setup time is at least 30% of matching steady decode"}
		classification.RecommendedLeverID = orchestrationLever
	}
	if p.Metal != nil && classification.RecommendedLeverID == "" {
		c := p.Metal
		switch {
		case c.WaitMilliseconds/decode >= 0.30:
			classification.Class = "synchronization-bound"
			classification.Confidence = "medium"
			classification.Evidence = []string{"Metal wait time is at least 30% of matching steady decode"}
			classification.RecommendedLeverID = "metal.command-buffer-amortization"
		case c.WorkingSetBytes > c.ResidentBytes:
			classification.Class = "bandwidth-bound"
			classification.Confidence = "medium"
			classification.Evidence = []string{"Metal working set exceeds resident bytes"}
			classification.RecommendedLeverID = "metal.paged-kv"
		case c.DispatchMilliseconds/decode >= 0.70:
			classification.Class = "compute-bound"
			classification.Confidence = "low"
			classification.Evidence = []string{"Metal dispatch time dominates matching steady decode without measured host or residency pressure"}
			classification.RecommendedLeverID = "metal.fused-hybrid-graph-coverage"
		}
	}
	if p.CUDA != nil && classification.RecommendedLeverID == "" {
		c := p.CUDA
		bandwidthRatio := c.AchievedBandwidthGBS / c.PeakBandwidthGBS
		computeRatio := c.AchievedComputeTFLOPS / c.PeakComputeTFLOPS
		switch {
		case c.SynchronizationMilliseconds/decode >= 0.30:
			classification.Class = "synchronization-bound"
			classification.Confidence = "medium"
			classification.Evidence = []string{"CUDA synchronization is at least 30% of matching steady decode"}
			classification.RecommendedLeverID = "cuda.default-decode-routing"
		case bandwidthRatio >= 0.70 && computeRatio < 0.60:
			classification.Class = "bandwidth-bound"
			classification.Confidence = "high"
			classification.Evidence = []string{"CUDA achieved bandwidth is at least 70% while compute is below 60% of backend peak"}
			classification.RecommendedLeverID = "cuda.q8_1-activation-quant"
		case computeRatio >= 0.70:
			classification.Class = "compute-bound"
			classification.Confidence = "high"
			classification.Evidence = []string{"CUDA achieved compute is at least 70% of backend peak"}
			classification.RecommendedLeverID = "cuda.dp4a-q4k-mmvq"
		}
	}
	if classification.RecommendedLeverID == "" {
		return classification, nil
	}

	lever := profileLever(graph, classification.RecommendedLeverID)
	if lever == nil || lever.Applicability.EnvelopeID != p.EnvelopeID {
		return BottleneckClassification{}, fmt.Errorf("classification recommends lever %q outside profile envelope %q", classification.RecommendedLeverID, p.EnvelopeID)
	}
	return classification, nil
}

func NextLeverFromProfile(graph Graph, p ProfileBundle) (*Lever, BottleneckClassification, error) {
	classification, err := ClassifyProfile(graph, p)
	if err != nil {
		return nil, classification, err
	}
	if classification.Class == "insufficient-evidence" {
		return nil, classification, ErrProfileEvidenceUnavailable
	}
	chosen := profileLever(graph, classification.RecommendedLeverID)
	if chosen == nil {
		return nil, classification, fmt.Errorf("classification recommends unknown lever %q", classification.RecommendedLeverID)
	}
	if chosen.Enabled || chosen.Witnessed != nil || !profileLeverDependenciesReady(graph, *chosen) {
		return nil, classification, fmt.Errorf("profile recommendation %q is not an unwitnessed dependency-ready lever", chosen.ID)
	}
	static, err := nextProfileLeverForEnvelope(graph, p.EnvelopeID)
	if err != nil {
		return nil, classification, err
	}
	if static != nil && static.ID != chosen.ID && p.Override == nil {
		return nil, classification, fmt.Errorf("profile contradicts graph-order lever %q with measured recommendation %q; record a positive issue number and issue-backed reason", static.ID, chosen.ID)
	}
	return chosen, classification, nil
}

func nextProfileLeverForEnvelope(graph Graph, envelopeID string) (*Lever, error) {
	if err := Validate(graph); err != nil {
		return nil, err
	}
	for i := range graph.Levers {
		candidate := graph.Levers[i]
		if candidate.Applicability.EnvelopeID != envelopeID || candidate.Enabled || candidate.Witnessed != nil {
			continue
		}
		if profileLeverDependenciesReady(graph, candidate) {
			return &candidate, nil
		}
	}
	return nil, nil
}

func profileLeverDependenciesReady(graph Graph, lever Lever) bool {
	for _, dependencyID := range lever.DependencyIDs {
		dependency := profileLever(graph, dependencyID)
		if dependency == nil || !dependency.Enabled {
			return false
		}
	}
	return true
}

func profileEnvelope(graph Graph, id string) *Envelope {
	for i := range graph.Envelopes {
		if graph.Envelopes[i].ID == id {
			return &graph.Envelopes[i]
		}
	}
	return nil
}

func profileLever(graph Graph, id string) *Lever {
	for i := range graph.Levers {
		if graph.Levers[i].ID == id {
			return &graph.Levers[i]
		}
	}
	return nil
}

func profilePhaseDuration(phases []ProfilePhase, name string) float64 {
	for _, phase := range phases {
		if phase.Name == name {
			return phase.DurationMilliseconds
		}
	}
	return 0
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func nonnegativeFinite(value float64) bool {
	return finite(value) && value >= 0
}

func positiveFinite(value float64) bool {
	return finite(value) && value > 0
}
