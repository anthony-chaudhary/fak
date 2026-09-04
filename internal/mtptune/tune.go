package mtptune

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// TaskCategory identifies a workload category with distinct token predictability.
type TaskCategory string

const (
	TaskCode TaskCategory = "Code"
	TaskMath TaskCategory = "Math"
	TaskJSON TaskCategory = "JSON"
)

// AllTasks returns the canonical standard task categories.
func AllTasks() []TaskCategory {
	return []TaskCategory{TaskCode, TaskMath, TaskJSON}
}

// TaskProfile describes baseline predictability for a task category.
type TaskProfile struct {
	Category    TaskCategory
	BaseAccept  float64 // Base single-token acceptance probability (rho)
	DecayFactor float64 // Step-wise decay across depth
}

// DefaultTaskProfiles returns the empirical task profiles modeled on Qwen 3.8 MTP on AMD Strix Halo.
func DefaultTaskProfiles() map[TaskCategory]TaskProfile {
	return map[TaskCategory]TaskProfile{
		TaskCode: {
			Category:    TaskCode,
			BaseAccept:  0.82,
			DecayFactor: 0.97,
		},
		TaskMath: {
			Category:    TaskMath,
			BaseAccept:  0.52,
			DecayFactor: 0.94,
		},
		TaskJSON: {
			Category:    TaskJSON,
			BaseAccept:  0.90,
			DecayFactor: 0.98,
		},
	}
}

// SweepConfig specifies parameters for the MTP speculative tuning sweep.
type SweepConfig struct {
	KMin               int            `json:"k_min"`
	KMax               int            `json:"k_max"`
	PMin               float64        `json:"p_min"`
	PMax               float64        `json:"p_max"`
	PStep              float64        `json:"p_step"`
	Tasks              []TaskCategory `json:"tasks"`
	BusBandwidthGBs    float64        `json:"bus_bandwidth_gbs"`     // e.g. 200.0 GB/s on 256-bit LPDDR5X
	ModelWeightGB      float64        `json:"model_weight_gb"`       // e.g. 13.55 GB for Qwen 3.8 27B ROCmFP4
	MTPHeadWeightGB    float64        `json:"mtp_head_weight_gb"`    // e.g. 0.45 GB
	KVTrafficPerTokGB  float64        `json:"kv_traffic_per_tok_gb"` // e.g. 0.08 GB
	BaseComputeMs      float64        `json:"base_compute_ms"`       // e.g. 4.0 ms
	DraftStepComputeMs float64        `json:"draft_step_compute_ms"` // e.g. 1.5 ms
}

// DefaultSweepConfig returns standard configuration for sweeping K=1..8 and P=0.0..1.0.
func DefaultSweepConfig() SweepConfig {
	return SweepConfig{
		KMin:               1,
		KMax:               8,
		PMin:               0.0,
		PMax:               1.0,
		PStep:              0.2,
		Tasks:              AllTasks(),
		BusBandwidthGBs:    200.0,
		ModelWeightGB:      13.55,
		MTPHeadWeightGB:    0.45,
		KVTrafficPerTokGB:  0.08,
		BaseComputeMs:      4.0,
		DraftStepComputeMs: 1.5,
	}
}

// MeasurementPoint captures metrics for a specific (Task, K, P) combination.
type MeasurementPoint struct {
	Task           TaskCategory `json:"task"`
	K              int          `json:"k"`
	P              float64      `json:"p"`
	AcceptanceRate float64      `json:"acceptance_rate"`
	AcceptedTokens float64      `json:"accepted_tokens"`
	ProposedTokens int          `json:"proposed_tokens"`
	SimulatedTPS   float64      `json:"simulated_tps"`
	BusSaturation  float64      `json:"bus_saturation"`
	StepLatencyMs  float64      `json:"step_latency_ms"`
}

// ParetoPoint aggregates metrics across tasks for a given (K, P) profile.
type ParetoPoint struct {
	K             int                               `json:"k"`
	P             float64                           `json:"p"`
	AvgTPS        float64                           `json:"avg_tps"`
	AvgAcceptRate float64                           `json:"avg_acceptance_rate"`
	AvgBusSat     float64                           `json:"avg_bus_saturation"`
	IsOptimal     bool                              `json:"is_optimal"`
	TaskPoints    map[TaskCategory]MeasurementPoint `json:"task_points"`
}

// SweepReport contains all results from a tuning sweep.
type SweepReport struct {
	Config         SweepConfig                       `json:"config"`
	Points         []MeasurementPoint                `json:"points"`
	ParetoFront    []ParetoPoint                     `json:"pareto_front"`
	OptimalProfile ParetoPoint                       `json:"optimal_profile"`
	OptimalPerTask map[TaskCategory]MeasurementPoint `json:"optimal_per_task"`
}

// Validate checks the sweep configuration.
func (cfg *SweepConfig) Validate() error {
	if cfg.KMin < 1 || cfg.KMax < cfg.KMin || cfg.KMax > 16 {
		return fmt.Errorf("invalid K range: [%d, %d]", cfg.KMin, cfg.KMax)
	}
	if cfg.PMin < 0 || cfg.PMax > 1.0 || cfg.PStep <= 0 || cfg.PMin > cfg.PMax {
		return fmt.Errorf("invalid P range: [%f, %f] step %f", cfg.PMin, cfg.PMax, cfg.PStep)
	}
	if len(cfg.Tasks) == 0 {
		return errors.New("at least one task category must be specified")
	}
	if cfg.BusBandwidthGBs <= 0 || cfg.ModelWeightGB <= 0 {
		return errors.New("bus bandwidth and model weight must be positive")
	}
	return nil
}

// SimulateMTPStep calculates metrics for one (Task, K, P) point using hardware-calibrated simulation.
func SimulateMTPStep(task TaskCategory, k int, p float64, cfg SweepConfig) MeasurementPoint {
	profiles := DefaultTaskProfiles()
	profile, ok := profiles[task]
	if !ok {
		profile = TaskProfile{Category: task, BaseAccept: 0.70, DecayFactor: 0.95}
	}

	// Calculate per-step acceptance rate
	stepAccept := make([]float64, k)
	sumAccept := 0.0
	cumProd := 1.0
	expectedAcceptedDrafts := 0.0

	for i := 0; i < k; i++ {
		decay := math.Pow(profile.DecayFactor, float64(i))
		baseProb := profile.BaseAccept * decay

		// Threshold effect: confidence threshold P prunes low-probability speculative paths.
		// Higher threshold improves draft acceptance quality up to P=0.6; beyond that, over-filtering hurts.
		var effectiveProb float64
		if p <= 0.6 {
			effectiveProb = baseProb + 0.15*p*(1.0-baseProb)
		} else {
			// Strict thresholding prunes valid tokens
			effectiveProb = baseProb + 0.15*0.6*(1.0-baseProb) - 0.25*(p-0.6)
		}
		effectiveProb = math.Max(0.10, math.Min(0.98, effectiveProb))

		stepAccept[i] = effectiveProb
		sumAccept += effectiveProb
		cumProd *= effectiveProb
		expectedAcceptedDrafts += cumProd
	}

	avgAcceptanceRate := sumAccept / float64(k)
	// 1 target token + expected accepted draft tokens
	expectedTotalAccepted := 1.0 + expectedAcceptedDrafts

	// Memory traffic modeling:
	// Verification streams model weights once.
	// Drafting streams MTP heads K times.
	// KV cache traffic scales with (K + 1).
	// Rollback penalty occurs when draft tokens are rejected: (1.0 - avgAcceptanceRate) * rollback overhead.
	rollbackPenaltyGB := 0.30 * (1.0 - avgAcceptanceRate) * float64(k)
	totalMemoryGB := cfg.ModelWeightGB + float64(k)*cfg.MTPHeadWeightGB + float64(k+1)*cfg.KVTrafficPerTokGB + rollbackPenaltyGB

	// Memory bus streaming time (seconds)
	tBusSec := totalMemoryGB / cfg.BusBandwidthGBs

	// Compute time (seconds)
	tComputeSec := (cfg.BaseComputeMs + float64(k)*cfg.DraftStepComputeMs) / 1000.0

	// Step latency (seconds and ms)
	tStepSec := tBusSec + tComputeSec
	stepLatencyMs := tStepSec * 1000.0

	// Simulated Tokens Per Second (TPS)
	simulatedTPS := expectedTotalAccepted / tStepSec

	// Bus saturation: fraction of step time spent actively consuming bus bandwidth,
	// plus bus bandwidth headroom consumption.
	// As K grows, memory traffic approaches bus throughput ceiling.
	busSaturation := (tBusSec / tStepSec) * math.Min(1.0, totalMemoryGB/(cfg.ModelWeightGB+4.0))
	if k >= 6 {
		// Bus contention penalty on shared memory bus
		busSaturation = math.Min(0.99, busSaturation+0.05*float64(k-5))
	}
	busSaturation = math.Min(1.0, math.Max(0.20, busSaturation))

	return MeasurementPoint{
		Task:           task,
		K:              k,
		P:              p,
		AcceptanceRate: avgAcceptanceRate,
		AcceptedTokens: expectedTotalAccepted,
		ProposedTokens: k,
		SimulatedTPS:   simulatedTPS,
		BusSaturation:  busSaturation,
		StepLatencyMs:  stepLatencyMs,
	}
}

// RunSweep executes the full grid sweep over K and P across all tasks.
func RunSweep(cfg SweepConfig) (*SweepReport, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	var points []MeasurementPoint
	profileMap := make(map[string][]MeasurementPoint)

	// Step grid
	for k := cfg.KMin; k <= cfg.KMax; k++ {
		for p := cfg.PMin; p <= cfg.PMax+1e-9; p += cfg.PStep {
			// Round p to 2 decimal places to avoid floating point drift
			pVal := math.Round(p*100) / 100.0
			profileKey := fmt.Sprintf("%d_%.2f", k, pVal)

			for _, task := range cfg.Tasks {
				pt := SimulateMTPStep(task, k, pVal, cfg)
				points = append(points, pt)
				profileMap[profileKey] = append(profileMap[profileKey], pt)
			}
		}
	}

	// Aggregate per (K, P) across tasks
	var aggregatePoints []ParetoPoint
	for _, pts := range profileMap {
		if len(pts) == 0 {
			continue
		}
		k := pts[0].K
		p := pts[0].P
		totTPS := 0.0
		totAccept := 0.0
		totBusSat := 0.0
		taskPoints := make(map[TaskCategory]MeasurementPoint)

		for _, pt := range pts {
			totTPS += pt.SimulatedTPS
			totAccept += pt.AcceptanceRate
			totBusSat += pt.BusSaturation
			taskPoints[pt.Task] = pt
		}

		n := float64(len(pts))
		aggregatePoints = append(aggregatePoints, ParetoPoint{
			K:             k,
			P:             p,
			AvgTPS:        totTPS / n,
			AvgAcceptRate: totAccept / n,
			AvgBusSat:     totBusSat / n,
			TaskPoints:    taskPoints,
		})
	}

	// Identify Pareto-optimal frontier
	paretoFront := identifyParetoPoints(aggregatePoints)

	// Identify overall optimal profile:
	// Maximizes AvgTPS subject to AvgBusSat <= 0.85 and AvgAcceptRate >= 0.50
	var optimalProfile ParetoPoint
	bestTPS := -1.0

	for _, pt := range paretoFront {
		if pt.AvgBusSat <= 0.85 && pt.AvgAcceptRate >= 0.50 {
			if pt.AvgTPS > bestTPS {
				bestTPS = pt.AvgTPS
				optimalProfile = pt
			}
		}
	}

	// If no point satisfies strict constraints, choose highest TPS in Pareto front
	if bestTPS < 0 && len(paretoFront) > 0 {
		for _, pt := range paretoFront {
			if pt.AvgTPS > bestTPS {
				bestTPS = pt.AvgTPS
				optimalProfile = pt
			}
		}
	}

	// Mark optimal
	for i := range paretoFront {
		if paretoFront[i].K == optimalProfile.K && math.Abs(paretoFront[i].P-optimalProfile.P) < 1e-4 {
			paretoFront[i].IsOptimal = true
			optimalProfile.IsOptimal = true
			break
		}
	}

	// Identify optimal per task
	optimalPerTask := make(map[TaskCategory]MeasurementPoint)
	for _, task := range cfg.Tasks {
		var bestTaskPt MeasurementPoint
		bestTaskTPS := -1.0
		for _, pt := range points {
			if pt.Task == task && pt.BusSaturation <= 0.85 {
				if pt.SimulatedTPS > bestTaskTPS {
					bestTaskTPS = pt.SimulatedTPS
					bestTaskPt = pt
				}
			}
		}
		// If all points for this task exceed 0.85 saturation, select highest TPS point
		if bestTaskTPS < 0 {
			for _, pt := range points {
				if pt.Task == task {
					if pt.SimulatedTPS > bestTaskTPS {
						bestTaskTPS = pt.SimulatedTPS
						bestTaskPt = pt
					}
				}
			}
		}
		optimalPerTask[task] = bestTaskPt
	}

	return &SweepReport{
		Config:         cfg,
		Points:         points,
		ParetoFront:    paretoFront,
		OptimalProfile: optimalProfile,
		OptimalPerTask: optimalPerTask,
	}, nil
}

func identifyParetoPoints(pts []ParetoPoint) []ParetoPoint {
	var pareto []ParetoPoint

	for i, a := range pts {
		dominated := false
		for j, b := range pts {
			if i == j {
				continue
			}
			// b dominates a if b is >= in TPS and AcceptanceRate, <= in BusSaturation,
			// with at least one strictly better.
			if b.AvgTPS >= a.AvgTPS && b.AvgAcceptRate >= a.AvgAcceptRate && b.AvgBusSat <= a.AvgBusSat {
				if b.AvgTPS > a.AvgTPS || b.AvgAcceptRate > a.AvgAcceptRate || b.AvgBusSat < a.AvgBusSat {
					dominated = true
					break
				}
			}
		}
		if !dominated {
			pareto = append(pareto, a)
		}
	}

	// Sort Pareto front by K ascending, then P ascending
	sort.Slice(pareto, func(i, j int) bool {
		if pareto[i].K != pareto[j].K {
			return pareto[i].K < pareto[j].K
		}
		return pareto[i].P < pareto[j].P
	})

	return pareto
}

// FormatReportTable formats the sweep report as an aligned text table.
func FormatReportTable(r *SweepReport) string {
	var sb strings.Builder
	sb.WriteString("========================================================================================\n")
	sb.WriteString("                       AUTOMATED MTP SPECULATIVE TUNING SWEEP\n")
	sb.WriteString("========================================================================================\n")
	sb.WriteString(fmt.Sprintf("Grid: K=%d..%d, P=%.1f..%.1f (step %.1f) | Bus: %.1f GB/s | Model: %.2f GB\n",
		r.Config.KMin, r.Config.KMax, r.Config.PMin, r.Config.PMax, r.Config.PStep,
		r.Config.BusBandwidthGBs, r.Config.ModelWeightGB))
	sb.WriteString("----------------------------------------------------------------------------------------\n")
	sb.WriteString("PARETO OPTIMAL FRONTIER:\n")
	sb.WriteString(fmt.Sprintf("%-4s  %-6s  %-12s  %-16s  %-16s  %-8s\n",
		"K", "P", "Avg TPS", "Avg Accept Rate", "Avg Bus Saturation", "Optimal"))
	sb.WriteString("----------------------------------------------------------------------------------------\n")

	for _, pt := range r.ParetoFront {
		optMarker := ""
		if pt.IsOptimal {
			optMarker = "★ OPTIMAL"
		}
		acceptStr := fmt.Sprintf("%.1f%%", pt.AvgAcceptRate*100)
		busStr := fmt.Sprintf("%.1f%%", pt.AvgBusSat*100)
		sb.WriteString(fmt.Sprintf("%-4d  %-6.2f  %-12.2f  %-16s  %-18s  %-8s\n",
			pt.K, pt.P, pt.AvgTPS, acceptStr, busStr, optMarker))
	}

	sb.WriteString("----------------------------------------------------------------------------------------\n")
	sb.WriteString(fmt.Sprintf("OPTIMAL PROFILE OVERALL: K=%d, P=%.2f -> %.2f tok/s (Accept: %.1f%%, Bus Sat: %.1f%%)\n",
		r.OptimalProfile.K, r.OptimalProfile.P, r.OptimalProfile.AvgTPS,
		r.OptimalProfile.AvgAcceptRate*100, r.OptimalProfile.AvgBusSat*100))
	sb.WriteString("----------------------------------------------------------------------------------------\n")
	sb.WriteString("OPTIMAL BY TASK CATEGORY:\n")
	for _, task := range r.Config.Tasks {
		opt := r.OptimalPerTask[task]
		sb.WriteString(fmt.Sprintf("  - %-6s: K=%d, P=%.2f -> %.2f tok/s (Accept: %.1f%%, Bus Sat: %.1f%%)\n",
			task, opt.K, opt.P, opt.SimulatedTPS, opt.AcceptanceRate*100, opt.BusSaturation*100))
	}
	sb.WriteString("========================================================================================\n")
	return sb.String()
}

// FormatReportJSON serializes the sweep report into formatted JSON.
func FormatReportJSON(r *SweepReport) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
