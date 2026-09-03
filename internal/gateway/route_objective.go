package gateway

import (
	"net/http"
	"sort"
	"strings"
)

type SelectionObjective string

const (
	ObjectiveDefault     SelectionObjective = "default"
	ObjectiveLatency     SelectionObjective = "latency"
	ObjectiveThroughput  SelectionObjective = "throughput"
	ObjectivePortability SelectionObjective = "portability"
	ObjectiveDeterminism SelectionObjective = "determinism"
	ObjectiveDebug       SelectionObjective = "debug"
)

const HeaderSelectionObjective = "x-fak-selection-objective"

func ParseSelectionObjective(raw string) SelectionObjective {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case string(ObjectiveLatency):
		return ObjectiveLatency
	case string(ObjectiveThroughput):
		return ObjectiveThroughput
	case string(ObjectivePortability):
		return ObjectivePortability
	case string(ObjectiveDeterminism):
		return ObjectiveDeterminism
	case string(ObjectiveDebug):
		return ObjectiveDebug
	case string(ObjectiveDefault):
		return ObjectiveDefault
	default:
		return ObjectiveDefault
	}
}

func ExtractSelectionObjective(req *http.Request) SelectionObjective {
	if req == nil || req.Header == nil {
		return ObjectiveDefault
	}
	if val := strings.TrimSpace(req.Header.Get(HeaderSelectionObjective)); val != "" {
		return ParseSelectionObjective(val)
	}
	if val := strings.TrimSpace(req.Header.Get("x-selection-objective")); val != "" {
		return ParseSelectionObjective(val)
	}
	return ObjectiveDefault
}

type RouteEndpoint struct {
	Name             string  `json:"name"`
	LatencyMs        float64 `json:"latency_ms"`
	ThroughputTokS   float64 `json:"throughput_tok_s"`
	PortabilityScore float64 `json:"portability_score"`
	DeterminismScore float64 `json:"determinism_score"`
	BasePriority     int     `json:"base_priority"`
}

func scoreRouteEndpoint(item RouteEndpoint, obj SelectionObjective) float64 {
	lat := item.LatencyMs
	if lat < 0 {
		lat = 0
	}
	latScore := 1000.0 / (lat + 1.0)
	switch obj {
	case ObjectiveLatency:
		return latScore
	case ObjectiveThroughput:
		return item.ThroughputTokS
	case ObjectivePortability:
		return item.PortabilityScore
	case ObjectiveDeterminism:
		return item.DeterminismScore
	case ObjectiveDebug:
		return float64(item.BasePriority)
	case ObjectiveDefault:
		fallthrough
	default:
		return latScore + item.ThroughputTokS + (item.PortabilityScore * 100.0) + (item.DeterminismScore * 100.0) + float64(item.BasePriority)
	}
}

func RankRouteEndpoints(candidates []RouteEndpoint, obj SelectionObjective) []RouteEndpoint {
	if candidates == nil {
		return nil
	}
	if len(candidates) == 0 {
		return []RouteEndpoint{}
	}
	ranked := make([]RouteEndpoint, len(candidates))
	copy(ranked, candidates)
	if len(ranked) == 1 {
		return ranked
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		scoreI := scoreRouteEndpoint(ranked[i], obj)
		scoreJ := scoreRouteEndpoint(ranked[j], obj)
		if scoreI != scoreJ {
			return scoreI > scoreJ
		}
		return ranked[i].Name < ranked[j].Name
	})
	return ranked
}
