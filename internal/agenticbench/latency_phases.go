package agenticbench

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

var requiredLatencyPhases = []string{"queue_wait", "agent_execution", "evaluation"}

type LatencyMeasurement struct {
	Duration      *float64 `json:"duration,omitempty"`
	Unit          string   `json:"unit,omitempty"`
	Source        string   `json:"source,omitempty"`
	Start         *float64 `json:"start,omitempty"`
	End           *float64 `json:"end,omitempty"`
	UnknownReason string   `json:"unknown_reason,omitempty"`
}

type GatewayLatencyObservation struct {
	Name string `json:"name,omitempty"`
	LatencyMeasurement
	Additive bool `json:"additive"`
}

type ArmLatencyStatus struct {
	Role            string                      `json:"role,omitempty"`
	Name            string                      `json:"name"`
	Total           LatencyMeasurement          `json:"total"`
	QueueWait       LatencyMeasurement          `json:"queue_wait"`
	AgentExecution  LatencyMeasurement          `json:"agent_execution"`
	Evaluation      LatencyMeasurement          `json:"evaluation"`
	GatewayRequests []GatewayLatencyObservation `json:"gateway_requests,omitempty"`
}

func checkLatencyArms(doc map[string]any) ([]ArmLatencyStatus, []string) {
	var statuses []ArmLatencyStatus
	var refusals []string
	for i, raw := range anySlice(doc["arms"]) {
		arm := mapValue(raw)
		name := strings.TrimSpace(str(arm, "name"))
		role := strings.TrimSpace(str(arm, "role"))
		if name == "" {
			name = role
		}
		if name == "" {
			name = fmt.Sprintf("arm-%d", i+1)
		}
		latency, ok := arm["latency"].(map[string]any)
		if !ok {
			refusals = append(refusals, fmt.Sprintf("LATENCY_PHASE_MISSING arm %s: latency receipt is required", name))
			continue
		}
		status, armRefusals := parseArmLatency(name, role, latency)
		statuses = append(statuses, status)
		refusals = append(refusals, armRefusals...)
	}
	return statuses, refusals
}

func parseArmLatency(name, role string, latency map[string]any) (ArmLatencyStatus, []string) {
	status := ArmLatencyStatus{
		Name:           name,
		Role:           role,
		Total:          parseLatencyMeasurement(latency["total"]),
		QueueWait:      parseLatencyMeasurement(latency["queue_wait"]),
		AgentExecution: parseLatencyMeasurement(latency["agent_execution"]),
		Evaluation:     parseLatencyMeasurement(latency["evaluation"]),
	}
	var refusals []string
	for i, raw := range anySlice(latency["gateway_requests"]) {
		row := mapValue(raw)
		measurement := parseLatencyMeasurement(row)
		additive, additiveDeclared := row["additive"].(bool)
		observation := GatewayLatencyObservation{
			Name:               strings.TrimSpace(str(row, "name")),
			LatencyMeasurement: measurement,
			Additive:           additive,
		}
		if observation.Name == "" {
			observation.Name = fmt.Sprintf("gateway_request_%d", i+1)
		}
		status.GatewayRequests = append(status.GatewayRequests, observation)
		refusals = append(refusals, validateLatencyMeasurement(name, "gateway_request "+observation.Name, measurement, false)...)
		if !additiveDeclared || additive {
			refusals = append(refusals, fmt.Sprintf("GATEWAY_INTERVAL_ADDITIVE arm %s gateway_request %s: gateway timing must declare additive=false", name, observation.Name))
		}
		if intervalKnown(measurement) && intervalKnown(status.AgentExecution) && !intervalContains(status.AgentExecution, measurement) {
			refusals = append(refusals, fmt.Sprintf("GATEWAY_INTERVAL_OUTSIDE_EXECUTION arm %s gateway_request %s: gateway timing must be nested inside agent_execution", name, observation.Name))
		}
	}
	refusals = append(refusals, validateArmLatency(status)...)
	return status, refusals
}

func parseLatencyMeasurement(raw any) LatencyMeasurement {
	m := mapValue(raw)
	return LatencyMeasurement{
		Duration:      numericPointer(m["duration"]),
		Unit:          strings.TrimSpace(str(m, "unit")),
		Source:        strings.TrimSpace(str(m, "source")),
		Start:         numericPointer(m["start"]),
		End:           numericPointer(m["end"]),
		UnknownReason: strings.TrimSpace(str(m, "unknown_reason")),
	}
}

func numericPointer(raw any) *float64 {
	var value float64
	switch v := raw.(type) {
	case json.Number:
		parsed, err := v.Float64()
		if err != nil {
			return nil
		}
		value = parsed
	case float64:
		value = v
	case float32:
		value = float64(v)
	case int:
		value = float64(v)
	case int64:
		value = float64(v)
	default:
		return nil
	}
	return &value
}

func validateArmLatency(status ArmLatencyStatus) []string {
	var refusals []string
	refusals = append(refusals, validateLatencyMeasurement(status.Name, "total", status.Total, false)...)
	phases := []struct {
		name        string
		measurement LatencyMeasurement
	}{
		{"queue_wait", status.QueueWait},
		{"agent_execution", status.AgentExecution},
		{"evaluation", status.Evaluation},
	}
	allKnown := status.Total.Duration != nil
	anyInterval := intervalKnown(status.Total)
	allIntervals := intervalKnown(status.Total)
	for _, phase := range phases {
		refusals = append(refusals, validateLatencyMeasurement(status.Name, phase.name, phase.measurement, true)...)
		allKnown = allKnown && phase.measurement.Duration != nil && phase.measurement.UnknownReason == ""
		anyInterval = anyInterval || intervalKnown(phase.measurement)
		allIntervals = allIntervals && intervalKnown(phase.measurement)
	}
	if anyInterval && !allIntervals {
		refusals = append(refusals, fmt.Sprintf("LATENCY_INTERVAL_PARTIAL arm %s: total and all three phases must either carry start/end intervals or all use source-attributed durations", status.Name))
	}
	if allKnown {
		phaseTotal := normalizedDuration(status.QueueWait) + normalizedDuration(status.AgentExecution) + normalizedDuration(status.Evaluation)
		if !nearlyEqual(phaseTotal, normalizedDuration(status.Total)) {
			refusals = append(refusals, fmt.Sprintf("LATENCY_TOTAL_INCONSISTENT arm %s: total must equal queue_wait + agent_execution + evaluation; gateway observations are nested and excluded", status.Name))
		}
	}
	if allIntervals {
		queueStart, queueEnd := normalizedInterval(status.QueueWait)
		executionStart, executionEnd := normalizedInterval(status.AgentExecution)
		evaluationStart, evaluationEnd := normalizedInterval(status.Evaluation)
		totalStart, totalEnd := normalizedInterval(status.Total)
		if queueEnd > executionStart+latencyEpsilon || executionEnd > evaluationStart+latencyEpsilon {
			refusals = append(refusals, fmt.Sprintf("LATENCY_INTERVAL_OVERLAP arm %s: queue_wait, agent_execution, and evaluation must not overlap", status.Name))
		}
		if !nearlyEqual(queueStart, totalStart) || !nearlyEqual(queueEnd, executionStart) || !nearlyEqual(executionEnd, evaluationStart) || !nearlyEqual(evaluationEnd, totalEnd) {
			refusals = append(refusals, fmt.Sprintf("LATENCY_TOTAL_INCONSISTENT arm %s: total interval must be exactly decomposed by queue_wait, agent_execution, and evaluation", status.Name))
		}
	}
	return refusals
}

func validateLatencyMeasurement(arm, phase string, measurement LatencyMeasurement, allowUnknown bool) []string {
	var refusals []string
	if measurement.Source == "" {
		refusals = append(refusals, fmt.Sprintf("LATENCY_SOURCE_MISSING arm %s phase %s: source is required", arm, phase))
	}
	if measurement.Duration == nil {
		if allowUnknown && measurement.UnknownReason != "" {
			refusals = append(refusals, fmt.Sprintf("LATENCY_PHASE_UNKNOWN arm %s phase %s: %s", arm, phase, measurement.UnknownReason))
			return refusals
		}
		refusals = append(refusals, fmt.Sprintf("LATENCY_PHASE_MISSING arm %s phase %s: duration or unknown_reason is required", arm, phase))
		return refusals
	}
	if measurement.UnknownReason != "" {
		refusals = append(refusals, fmt.Sprintf("LATENCY_PHASE_CONFLATED arm %s phase %s: duration and unknown_reason are mutually exclusive", arm, phase))
	}
	if measurement.Unit == "" {
		refusals = append(refusals, fmt.Sprintf("LATENCY_PHASE_UNITLESS arm %s phase %s: unit is required", arm, phase))
	} else if _, ok := latencyUnitScale(measurement.Unit); !ok {
		refusals = append(refusals, fmt.Sprintf("LATENCY_UNIT_UNSUPPORTED arm %s phase %s: unit %q is not one of ns, us, ms, or s", arm, phase, measurement.Unit))
	}
	if math.IsNaN(*measurement.Duration) || math.IsInf(*measurement.Duration, 0) || *measurement.Duration < 0 {
		refusals = append(refusals, fmt.Sprintf("LATENCY_PHASE_NEGATIVE arm %s phase %s: duration must be finite and non-negative", arm, phase))
	}
	if (measurement.Start == nil) != (measurement.End == nil) {
		refusals = append(refusals, fmt.Sprintf("LATENCY_INTERVAL_INCOMPLETE arm %s phase %s: start and end must be provided together", arm, phase))
	}
	if intervalKnown(measurement) {
		if *measurement.End < *measurement.Start {
			refusals = append(refusals, fmt.Sprintf("LATENCY_INTERVAL_NEGATIVE arm %s phase %s: end must not precede start", arm, phase))
		} else if scale, ok := latencyUnitScale(measurement.Unit); ok && !nearlyEqual((*measurement.End-*measurement.Start)*scale, *measurement.Duration*scale) {
			refusals = append(refusals, fmt.Sprintf("LATENCY_INTERVAL_MISMATCH arm %s phase %s: duration must equal end-start", arm, phase))
		}
	}
	return refusals
}

const latencyEpsilon = 1e-9

func latencyUnitScale(unit string) (float64, bool) {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "ns":
		return 1e-9, true
	case "us", "µs", "μs":
		return 1e-6, true
	case "ms":
		return 1e-3, true
	case "s":
		return 1, true
	default:
		return 0, false
	}
}

func normalizedDuration(measurement LatencyMeasurement) float64 {
	if measurement.Duration == nil {
		return 0
	}
	scale, _ := latencyUnitScale(measurement.Unit)
	return *measurement.Duration * scale
}

func normalizedInterval(measurement LatencyMeasurement) (float64, float64) {
	scale, _ := latencyUnitScale(measurement.Unit)
	return *measurement.Start * scale, *measurement.End * scale
}

func intervalKnown(measurement LatencyMeasurement) bool {
	return measurement.Start != nil && measurement.End != nil
}

func intervalContains(outer, inner LatencyMeasurement) bool {
	outerStart, outerEnd := normalizedInterval(outer)
	innerStart, innerEnd := normalizedInterval(inner)
	return innerStart+latencyEpsilon >= outerStart && innerEnd <= outerEnd+latencyEpsilon
}

func nearlyEqual(left, right float64) bool {
	scale := math.Max(1, math.Max(math.Abs(left), math.Abs(right)))
	return math.Abs(left-right) <= latencyEpsilon*scale
}
