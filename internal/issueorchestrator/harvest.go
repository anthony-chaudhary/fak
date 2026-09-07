package issueorchestrator

// HarvestReceiptSchema is the canonical schema identifier for wave harvest receipts.
const HarvestReceiptSchema = "fak.issue-orchestrator-harvest.v1"

// LeafState describes the witnessed resolution status of an issue leaf.
type LeafState string

const (
	// LeafStateVerifiedCleared indicates witnessed commit on trunk with verified tests.
	LeafStateVerifiedCleared LeafState = "VERIFIED_CLEARED"

	// LeafStateResidualReview indicates a subject-only or uncorroborated claim requiring human review.
	LeafStateResidualReview LeafState = "RESIDUAL_REVIEW"

	// LeafStateQuietIncomplete indicates claimed done but oracle says not shipped or commit missing.
	LeafStateQuietIncomplete LeafState = "QUIET_INCOMPLETE"

	// LeafStateSpinningStalled indicates spinning or stalled worker with no net progress.
	LeafStateSpinningStalled LeafState = "SPINNING_STALLED"
)

// LeafHarvest records the witnessed state and evidence for an individual issue leaf.
type LeafHarvest struct {
	IssueNumber    int       `json:"issue_number"`
	Lane           string    `json:"lane"`
	State          LeafState `json:"state"`
	Cleared        bool      `json:"cleared"`
	Reason         string    `json:"reason,omitempty"`
	ArtifactPaths  []string  `json:"artifact_paths,omitempty"`
	WitnessCommand string    `json:"witness_command,omitempty"`
	WitnessOutput  string    `json:"witness_output,omitempty"`
}

// HarvestPlan aggregates leaf harvest outcomes across a wave.
type HarvestPlan struct {
	WaveID        string        `json:"wave_id"`
	TotalLeaves   int           `json:"total_leaves"`
	ClearedCount  int           `json:"cleared_count"`
	ResidualCount int           `json:"residual_count"`
	QuietCount    int           `json:"quiet_count"`
	StalledCount  int           `json:"stalled_count"`
	ClearRate     float64       `json:"clear_rate"`
	Leaves        []LeafHarvest `json:"leaves"`
}

// ReconcileResult records the final admission and routing verdict for harvested leaves.
type ReconcileResult struct {
	Schema            string        `json:"schema"`
	WaveID            string        `json:"wave_id"`
	OverallStatus     string        `json:"overall_status"` // "PASSED", "PARTIAL", "FAILED"
	MinClearRate      float64       `json:"min_clear_rate"`
	ActualClearRate   float64       `json:"actual_clear_rate"`
	AdmittedToTrunk   []LeafHarvest `json:"admitted_to_trunk"`
	QueuedForReview   []LeafHarvest `json:"queued_for_review"`
	RequeuedToBacklog []LeafHarvest `json:"requeued_to_backlog"`
	StalledReaped     []LeafHarvest `json:"stalled_reaped"`
}

// ComputeHarvestPlan calculates counts and clear rate across the given leaves.
func ComputeHarvestPlan(waveID string, leaves []LeafHarvest) HarvestPlan {
	plan := HarvestPlan{
		WaveID:      waveID,
		TotalLeaves: len(leaves),
		Leaves:      make([]LeafHarvest, len(leaves)),
	}

	for i, leaf := range leaves {
		if leaf.State == LeafStateVerifiedCleared {
			leaf.Cleared = true
		}
		switch leaf.State {
		case LeafStateVerifiedCleared:
			plan.ClearedCount++
		case LeafStateResidualReview:
			plan.ResidualCount++
		case LeafStateQuietIncomplete:
			plan.QuietCount++
		case LeafStateSpinningStalled:
			plan.StalledCount++
		default:
			if leaf.Cleared {
				plan.ClearedCount++
			}
		}
		plan.Leaves[i] = leaf
	}

	if plan.TotalLeaves > 0 {
		plan.ClearRate = float64(plan.ClearedCount) / float64(plan.TotalLeaves)
	}

	return plan
}

// ReconcileHarvest evaluates the harvest plan against the minimum clear rate and partitions leaves.
func ReconcileHarvest(plan HarvestPlan, minClearRate float64) ReconcileResult {
	result := ReconcileResult{
		Schema:            HarvestReceiptSchema,
		WaveID:            plan.WaveID,
		MinClearRate:      minClearRate,
		ActualClearRate:   plan.ClearRate,
		AdmittedToTrunk:   make([]LeafHarvest, 0),
		QueuedForReview:   make([]LeafHarvest, 0),
		RequeuedToBacklog: make([]LeafHarvest, 0),
		StalledReaped:     make([]LeafHarvest, 0),
	}

	for _, leaf := range plan.Leaves {
		switch leaf.State {
		case LeafStateVerifiedCleared:
			if plan.ClearRate >= minClearRate {
				result.AdmittedToTrunk = append(result.AdmittedToTrunk, leaf)
			}
		case LeafStateResidualReview:
			result.QueuedForReview = append(result.QueuedForReview, leaf)
		case LeafStateQuietIncomplete:
			result.RequeuedToBacklog = append(result.RequeuedToBacklog, leaf)
		case LeafStateSpinningStalled:
			result.StalledReaped = append(result.StalledReaped, leaf)
		default:
			if leaf.Cleared && plan.ClearRate >= minClearRate {
				result.AdmittedToTrunk = append(result.AdmittedToTrunk, leaf)
			}
		}
	}

	if plan.TotalLeaves == 0 || plan.ClearRate < minClearRate {
		result.OverallStatus = "FAILED"
		return result
	}

	if len(result.AdmittedToTrunk) == plan.TotalLeaves {
		result.OverallStatus = "PASSED"
	} else {
		result.OverallStatus = "PARTIAL"
	}

	return result
}
