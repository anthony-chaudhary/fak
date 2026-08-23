package modelaccept

import (
	"fmt"
	"math"
	"strings"
)

type SpeculativeDisposition string

const (
	SpeculativeBaseline SpeculativeDisposition = "BASELINE"
	SpeculativePromote  SpeculativeDisposition = "PROMOTE"
	SpeculativeReject   SpeculativeDisposition = "REJECT"
)

type SpeculativeArm struct {
	Name                  string  `json:"name"`
	Mode                  string  `json:"mode"`
	RuntimeRevision       string  `json:"runtime_revision,omitempty"`
	Checkpoint            string  `json:"checkpoint,omitempty"`
	HardwareTopology      string  `json:"hardware_topology,omitempty"`
	Loadable              bool    `json:"loadable"`
	FailureBounded        bool    `json:"failure_bounded"`
	StartupSeconds        float64 `json:"startup_seconds"`
	RunSeconds            float64 `json:"run_seconds"`
	Requests              int     `json:"requests"`
	PeakMemoryBytes       uint64  `json:"peak_memory_bytes"`
	EnergyJoules          float64 `json:"energy_joules"`
	TaskQuality           float64 `json:"task_quality"`
	AcceptedTokensPerStep float64 `json:"accepted_tokens_per_step,omitempty"`
}

type SpeculativeArmResult struct {
	SpeculativeArm
	NetSpeedup       float64                `json:"net_speedup"`
	PeakMemoryDelta  int64                  `json:"peak_memory_delta_bytes"`
	EnergyPerRequest float64                `json:"energy_per_request_joules"`
	Disposition      SpeculativeDisposition `json:"disposition"`
	Reasons          []string               `json:"reasons,omitempty"`
}

type SpeculativeFrontier struct {
	RuntimeRevision  string           `json:"runtime_revision"`
	Checkpoint       string           `json:"checkpoint"`
	HardwareTopology string           `json:"hardware_topology"`
	Arms             []SpeculativeArm `json:"arms"`
}

type SpeculativeFrontierResult struct {
	Pass        bool                   `json:"pass"`
	PromotedArm string                 `json:"promoted_arm,omitempty"`
	BaselineArm string                 `json:"baseline_arm"`
	Arms        []SpeculativeArmResult `json:"arms"`
}

// EvaluateSpeculativeFrontier promotes only a pinned, quality-neutral arm whose
// end-to-end elapsed time includes draft startup. Loadability alone never passes.
func EvaluateSpeculativeFrontier(in SpeculativeFrontier) (SpeculativeFrontierResult, error) {
	if strings.TrimSpace(in.RuntimeRevision) == "" || strings.TrimSpace(in.Checkpoint) == "" || strings.TrimSpace(in.HardwareTopology) == "" {
		return SpeculativeFrontierResult{}, fmt.Errorf("runtime revision, checkpoint, and hardware topology must be pinned")
	}
	if len(in.Arms) < 3 {
		return SpeculativeFrontierResult{}, fmt.Errorf("speculative frontier requires at least three arms including baseline")
	}
	baselineIndex := -1
	for i, arm := range in.Arms {
		if arm.RuntimeRevision != "" && arm.RuntimeRevision != in.RuntimeRevision {
			return SpeculativeFrontierResult{}, fmt.Errorf("arm %q runtime revision differs from frontier pin", arm.Name)
		}
		if arm.Checkpoint != "" && arm.Checkpoint != in.Checkpoint {
			return SpeculativeFrontierResult{}, fmt.Errorf("arm %q checkpoint differs from frontier pin", arm.Name)
		}
		if arm.HardwareTopology != "" && arm.HardwareTopology != in.HardwareTopology {
			return SpeculativeFrontierResult{}, fmt.Errorf("arm %q hardware topology differs from frontier pin", arm.Name)
		}
		if arm.Name == "" || arm.Mode == "" || arm.Requests <= 0 || arm.RunSeconds <= 0 || arm.PeakMemoryBytes == 0 || arm.EnergyJoules <= 0 || arm.TaskQuality < 0 || arm.TaskQuality > 1 {
			return SpeculativeFrontierResult{}, fmt.Errorf("arm %d has incomplete or invalid measurements", i)
		}
		if arm.Mode == "none" {
			if baselineIndex >= 0 {
				return SpeculativeFrontierResult{}, fmt.Errorf("frontier has multiple no-speculation baselines")
			}
			baselineIndex = i
		}
	}
	if baselineIndex < 0 {
		return SpeculativeFrontierResult{}, fmt.Errorf("frontier requires one no-speculation baseline")
	}
	baseline := in.Arms[baselineIndex]
	if !baseline.Loadable || !baseline.FailureBounded {
		return SpeculativeFrontierResult{}, fmt.Errorf("baseline must be loadable with bounded failures")
	}
	baselineSecondsPerRequest := (baseline.StartupSeconds + baseline.RunSeconds) / float64(baseline.Requests)
	out := SpeculativeFrontierResult{BaselineArm: baseline.Name, Arms: make([]SpeculativeArmResult, len(in.Arms))}
	bestSpeedup := 1.0
	for i, arm := range in.Arms {
		result := SpeculativeArmResult{SpeculativeArm: arm, Disposition: SpeculativeReject}
		secondsPerRequest := (arm.StartupSeconds + arm.RunSeconds) / float64(arm.Requests)
		result.NetSpeedup = math.Round((baselineSecondsPerRequest/secondsPerRequest)*1000) / 1000
		result.PeakMemoryDelta = int64(arm.PeakMemoryBytes) - int64(baseline.PeakMemoryBytes)
		result.EnergyPerRequest = math.Round((arm.EnergyJoules/float64(arm.Requests))*1000) / 1000
		switch {
		case i == baselineIndex:
			result.Disposition = SpeculativeBaseline
		case !arm.Loadable:
			result.Reasons = append(result.Reasons, "not_loadable")
		case !arm.FailureBounded:
			result.Reasons = append(result.Reasons, "unbounded_failure")
		case arm.TaskQuality < baseline.TaskQuality:
			result.Reasons = append(result.Reasons, "lower_task_quality")
		case result.NetSpeedup <= 1:
			result.Reasons = append(result.Reasons, "no_net_speedup")
		default:
			result.Disposition = SpeculativePromote
			if result.NetSpeedup > bestSpeedup {
				bestSpeedup, out.PromotedArm = result.NetSpeedup, arm.Name
			}
		}
		out.Arms[i] = result
	}
	out.Pass = out.PromotedArm != ""
	return out, nil
}
