package issueorchestrator

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
)

// HarvestReceiptSchema is the canonical schema identifier for wave harvest receipts.
const HarvestReceiptSchema = "fak.issue-orchestrator-harvest.v1"

// LeafHarvestState describes the witnessed resolution status of an issue leaf.
type LeafHarvestState string

const (
	StateVerifiedCleared LeafHarvestState = "VERIFIED_CLEARED" // Git commit witnessed & test green -> auto-landable
	StateResidualReview  LeafHarvestState = "RESIDUAL_REVIEW"  // Diff exists but uncorroborated / tests missing -> flagged for review
	StateQuietIncomplete LeafHarvestState = "QUIET_INCOMPLETE" // No commits, no changes -> soft re-queue
	StateSpinningStalled LeafHarvestState = "SPINNING_STALLED" // Stalled or runaway process -> signaled or stopped
)

// Classification sentinel errors for leaf witness evaluation.
var (
	ErrResidualReview  = errors.New("residual review: diff uncorroborated or tests missing")
	ErrQuietIncomplete = errors.New("quiet incomplete: no commits or changes found")
	ErrSpinningStalled = errors.New("spinning stalled: process stalled or runaway")
)

// Backward-compatibility aliases for existing codebase references.
type LeafState = LeafHarvestState

const (
	LeafStateVerifiedCleared = StateVerifiedCleared
	LeafStateResidualReview  = StateResidualReview
	LeafStateQuietIncomplete = StateQuietIncomplete
	LeafStateSpinningStalled = StateSpinningStalled
)

// LeafReceipt describes the outcome and metadata for an individual issue leaf during wave harvest.
type LeafReceipt struct {
	IssueNumber int              `json:"issue_number"`
	Title       string           `json:"title"`
	Lane        string           `json:"lane"`
	CommitSHA   string           `json:"commit_sha,omitempty"`
	State       LeafHarvestState `json:"state"`
	Notes       string           `json:"notes,omitempty"`
	Duration    time.Duration    `json:"duration,omitempty"`
}

// HarvestReceipt represents the structured result of reconciling a wave of issue leaves.
type HarvestReceipt struct {
	Schema        string        `json:"schema,omitempty"`
	WaveID        string        `json:"wave_id"`
	TotalLeaves   int           `json:"total_leaves"`
	ClearedCount  int           `json:"cleared_count"`
	ResidualCount int           `json:"residual_count"`
	QuietCount    int           `json:"quiet_count"`
	StalledCount  int           `json:"stalled_count"`
	ClearRate     float64       `json:"clear_rate"`
	Leaves        []LeafReceipt `json:"leaves"`
	LandedSHAs    []string      `json:"landed_shas,omitempty"`
	ReviewQueue   []LeafReceipt `json:"review_queue,omitempty"`
	AdvisoryLogs  []string      `json:"advisory_logs,omitempty"`
}

// HarvestOptions configures the wave reconciliation process.
type HarvestOptions struct {
	WaveReceiptPath string                                                        `json:"wave_receipt_path,omitempty"`
	WaveID          string                                                        `json:"wave_id,omitempty"`
	Workspace       string                                                        `json:"workspace,omitempty"`
	MinClearRate    float64                                                       `json:"min_clear_rate,omitempty"` // default: 0.0 (progressive acceptance: lands any verified leaf)
	AutoLand        bool                                                          `json:"auto_land,omitempty"`      // whether to land verified cleared leaves
	WitnessChecker  func(leaf LeafReceipt) (verified bool, sha string, err error) `json:"-"`
	GitLander       func(sha string) error                                        `json:"-"`
	Leaves          []LeafReceipt                                                 `json:"leaves,omitempty"`
}

// ReconcileWave reconciles spawned wave receipts against witness checks, categorizes each
// leaf into the 4-state taxonomy, auto-lands verified leaves to trunk if requested,
// and routes residual leaves to the review queue without halting execution.
func ReconcileWave(opts HarvestOptions) (HarvestReceipt, error) {
	waveID := opts.WaveID
	var leaves []LeafReceipt

	if len(opts.Leaves) > 0 {
		leaves = make([]LeafReceipt, len(opts.Leaves))
		copy(leaves, opts.Leaves)
	} else if opts.WaveReceiptPath != "" {
		parsedWaveID, loadedLeaves, err := loadLeavesFromReceipt(opts.WaveReceiptPath)
		if err != nil {
			return HarvestReceipt{}, err
		}
		if waveID == "" {
			waveID = parsedWaveID
		}
		leaves = loadedLeaves
	}

	receipt := HarvestReceipt{
		Schema:       HarvestReceiptSchema,
		WaveID:       waveID,
		TotalLeaves:  len(leaves),
		Leaves:       make([]LeafReceipt, 0, len(leaves)),
		ReviewQueue:  make([]LeafReceipt, 0),
		AdvisoryLogs: make([]string, 0),
		LandedSHAs:   nil,
	}

	if len(leaves) == 0 {
		return receipt, nil
	}

	for _, leaf := range leaves {
		if opts.WitnessChecker != nil {
			start := time.Now()
			verified, sha, err := opts.WitnessChecker(leaf)
			if leaf.Duration == 0 {
				leaf.Duration = time.Since(start)
			}
			if verified {
				leaf.State = StateVerifiedCleared
				if sha != "" {
					leaf.CommitSHA = sha
				}
			} else if err != nil {
				if leaf.Notes == "" {
					leaf.Notes = err.Error()
				}
				errStr := strings.ToLower(err.Error())
				switch {
				case errors.Is(err, ErrSpinningStalled) || strings.Contains(errStr, "stall") || strings.Contains(errStr, "spin") || strings.Contains(errStr, "timeout"):
					leaf.State = StateSpinningStalled
				case errors.Is(err, ErrResidualReview) || strings.Contains(errStr, "residual") || strings.Contains(errStr, "review") || strings.Contains(errStr, "uncorroborated"):
					leaf.State = StateResidualReview
					if sha != "" {
						leaf.CommitSHA = sha
					}
				case errors.Is(err, ErrQuietIncomplete) || strings.Contains(errStr, "quiet") || strings.Contains(errStr, "incomplete") || strings.Contains(errStr, "no commit") || strings.Contains(errStr, "no change"):
					leaf.State = StateQuietIncomplete
				default:
					if sha != "" {
						leaf.State = StateResidualReview
						leaf.CommitSHA = sha
					} else if leaf.State != "" {
						// Preserve leaf state if already specified
					} else {
						leaf.State = StateQuietIncomplete
					}
				}
			} else { // !verified && err == nil
				if sha != "" {
					leaf.State = StateResidualReview
					leaf.CommitSHA = sha
				} else if leaf.State == "" {
					leaf.State = StateQuietIncomplete
				}
			}
		} else {
			if leaf.State == "" {
				if leaf.CommitSHA != "" {
					leaf.State = StateResidualReview
				} else {
					leaf.State = StateQuietIncomplete
				}
			}
		}

		switch leaf.State {
		case StateVerifiedCleared:
			receipt.ClearedCount++
		case StateResidualReview:
			receipt.ResidualCount++
			advisory := fmt.Sprintf("[issueorchestrator:harvest] ADVISORY: leaf #%d in residual review; continuing reconciliation", leaf.IssueNumber)
			receipt.AdvisoryLogs = append(receipt.AdvisoryLogs, advisory)
			receipt.ReviewQueue = append(receipt.ReviewQueue, leaf)
		case StateQuietIncomplete:
			receipt.QuietCount++
		case StateSpinningStalled:
			receipt.StalledCount++
			advisory := fmt.Sprintf("[issueorchestrator:harvest] ADVISORY: leaf #%d spinning stalled; process signaled or stopped", leaf.IssueNumber)
			receipt.AdvisoryLogs = append(receipt.AdvisoryLogs, advisory)
		default:
			receipt.QuietCount++
		}

		receipt.Leaves = append(receipt.Leaves, leaf)
	}

	if receipt.TotalLeaves > 0 {
		receipt.ClearRate = float64(receipt.ClearedCount) / float64(receipt.TotalLeaves)
	}

	if opts.AutoLand {
		if opts.MinClearRate > 0.0 && receipt.ClearRate < opts.MinClearRate {
			receipt.AdvisoryLogs = append(receipt.AdvisoryLogs,
				fmt.Sprintf("[issueorchestrator:harvest] ADVISORY: wave clear rate %.2f below min clear rate %.2f; auto-land skipped", receipt.ClearRate, opts.MinClearRate))
		} else {
			receipt.LandedSHAs = make([]string, 0)
			for _, leaf := range receipt.Leaves {
				if leaf.State == StateVerifiedCleared {
					if opts.GitLander != nil && leaf.CommitSHA != "" {
						if err := opts.GitLander(leaf.CommitSHA); err != nil {
							return receipt, fmt.Errorf("git land leaf #%d (%s): %w", leaf.IssueNumber, leaf.CommitSHA, err)
						}
					}
					if leaf.CommitSHA != "" {
						receipt.LandedSHAs = append(receipt.LandedSHAs, leaf.CommitSHA)
					}
				}
			}
		}
	}

	return receipt, nil
}

func loadLeavesFromReceipt(path string) (string, []LeafReceipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", nil, fmt.Errorf("read wave receipt: %w", err)
	}

	// 1. Try HarvestReceipt / envelope with "leaves"
	var harvestEnvelope struct {
		WaveID string        `json:"wave_id"`
		Leaves []LeafReceipt `json:"leaves"`
	}
	if err := json.Unmarshal(data, &harvestEnvelope); err == nil && len(harvestEnvelope.Leaves) > 0 {
		return harvestEnvelope.WaveID, harvestEnvelope.Leaves, nil
	}

	// 2. Try OpencodeSpawnReceipt format with "chats"
	var opencodeEnvelope struct {
		WaveID string `json:"wave_id"`
		Chats  []struct {
			IssueNumber int    `json:"issue_number"`
			Title       string `json:"title"`
			Lane        string `json:"lane"`
			Status      string `json:"status"`
			Error       string `json:"error,omitempty"`
		} `json:"chats"`
	}
	if err := json.Unmarshal(data, &opencodeEnvelope); err == nil && len(opencodeEnvelope.Chats) > 0 {
		leaves := make([]LeafReceipt, len(opencodeEnvelope.Chats))
		for i, c := range opencodeEnvelope.Chats {
			leaves[i] = LeafReceipt{
				IssueNumber: c.IssueNumber,
				Title:       c.Title,
				Lane:        c.Lane,
				Notes:       c.Error,
			}
		}
		return opencodeEnvelope.WaveID, leaves, nil
	}

	// 3. Try direct []LeafReceipt
	var directLeaves []LeafReceipt
	if err := json.Unmarshal(data, &directLeaves); err == nil && len(directLeaves) > 0 {
		return "", directLeaves, nil
	}

	return "", nil, fmt.Errorf("unable to decode wave receipt from %s", path)
}

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
