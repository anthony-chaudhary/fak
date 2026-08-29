package main

import (
	"path/filepath"
)

const (
	dispatchFinishFirstStateUnobserved        = "UNOBSERVED"
	dispatchFinishFirstStateNormal            = "NORMAL"
	dispatchFinishFirstStateDiverging         = "DIVERGING"
	dispatchFinishFirstStateStaleOldest       = "STALE_OLDEST"
	dispatchFinishFirstStateGitHubUnavailable = "GITHUB_UNAVAILABLE"
	dispatchFinishFirstStateRecovering        = "RECOVERING"
	dispatchFinishFirstStateConverged         = "CONVERGED"
	dispatchFinishFirstStateOverride          = "OVERRIDDEN"

	dispatchFinishFirstStaleMinutes    = int64(24 * 60)
	dispatchFinishFirstRecoveryWindows = 3
)

// dispatchFinishFirstAdmissionInput is the complete, immutable evidence supplied to the
// finish-first admission evaluator. Collection stays outside the evaluator: it performs no
// GitHub calls, process launches, filesystem writes, lease operations, or scheduling.
type dispatchFinishFirstAdmissionInput struct {
	EvidenceAvailable            bool    `json:"evidence_available"`
	WIPFilesDelta                int     `json:"wip_files_delta"`
	WIPLinesDelta                int64   `json:"wip_lines_delta"`
	OldestWIPMinutes             int64   `json:"oldest_wip_minutes"`
	CloseRate                    float64 `json:"close_rate_per_window"`
	GitHubAvailable              bool    `json:"github_available"`
	ConsecutiveDivergingWindows  int     `json:"consecutive_diverging_windows"`
	ConsecutiveConvergingWindows int     `json:"consecutive_converging_windows"`
	RecoveringFromDivergence     bool    `json:"recovering_from_divergence"`
	RequestedFreshStarts         int     `json:"requested_fresh_starts"`
	Finishers                    int     `json:"finishers"`
	Override                     bool    `json:"override"`
}

type dispatchFinishFirstRecovery struct {
	RequiredConvergingWindows int  `json:"required_converging_windows"`
	ObservedConvergingWindows int  `json:"observed_converging_windows"`
	FreshStartCap             int  `json:"fresh_start_cap"`
	Complete                  bool `json:"complete"`
}

// dispatchFinishFirstAdmission is an auditable cap on never-attempted starts. AllowedFinishers
// is deliberately independent of the fresh-start verdict: this result has no representation for
// evicting a worker, releasing a lease, or abandoning an already-recorded intent.
type dispatchFinishFirstAdmission struct {
	State              string                            `json:"state"`
	Inputs             dispatchFinishFirstAdmissionInput `json:"inputs"`
	AllowedFreshStarts int                               `json:"allowed_fresh_starts"`
	DeniedFreshStarts  int                               `json:"denied_fresh_starts"`
	AllowedFinishers   int                               `json:"allowed_finishers"`
	Override           bool                              `json:"override"`
	Reason             string                            `json:"reason"`
	Recovery           dispatchFinishFirstRecovery       `json:"recovery"`
}

// evaluateDispatchFinishFirstAdmission is the pure admission policy. Divergence and missing
// GitHub evidence close only the never-attempted-start valve. Recovery opens that valve by one
// start after two converging windows and fully after three, preventing one noisy sample from
// oscillating a wave between hold and full fan-out.
func evaluateDispatchFinishFirstAdmission(in dispatchFinishFirstAdmissionInput) dispatchFinishFirstAdmission {
	in.RequestedFreshStarts = max(0, in.RequestedFreshStarts)
	in.Finishers = max(0, in.Finishers)
	in.ConsecutiveDivergingWindows = max(0, in.ConsecutiveDivergingWindows)
	in.ConsecutiveConvergingWindows = max(0, in.ConsecutiveConvergingWindows)
	out := dispatchFinishFirstAdmission{
		State: dispatchFinishFirstStateNormal, Inputs: in,
		AllowedFreshStarts: in.RequestedFreshStarts, AllowedFinishers: in.Finishers,
		Override: in.Override,
		Recovery: dispatchFinishFirstRecovery{
			RequiredConvergingWindows: dispatchFinishFirstRecoveryWindows,
			ObservedConvergingWindows: in.ConsecutiveConvergingWindows,
			FreshStartCap:             in.RequestedFreshStarts,
			Complete:                  true,
		},
	}
	setCap := func(cap int) {
		out.AllowedFreshStarts = min(in.RequestedFreshStarts, max(0, cap))
		out.DeniedFreshStarts = in.RequestedFreshStarts - out.AllowedFreshStarts
		out.Recovery.FreshStartCap = out.AllowedFreshStarts
	}

	switch {
	case in.Override:
		out.State = dispatchFinishFirstStateOverride
		out.Reason = "explicit operator override admits never-attempted starts; finishers remain admitted"
	case !in.EvidenceAvailable:
		out.State = dispatchFinishFirstStateUnobserved
		out.Reason = "no paired progress snapshots; preserve legacy fresh-start cap while collecting evidence"
	case !in.GitHubAvailable:
		out.State = dispatchFinishFirstStateGitHubUnavailable
		out.Reason = "GitHub was unavailable at a progress boundary; refuse new fronts without inferring convergence"
		out.Recovery.Complete = false
		setCap(0)
	case in.WIPFilesDelta > 0 || in.WIPLinesDelta > 0:
		out.State = dispatchFinishFirstStateDiverging
		out.Reason = "unfinished file or line inventory grew; admit only finish/reconcile work"
		out.Recovery.Complete = false
		setCap(0)
	case in.OldestWIPMinutes >= dispatchFinishFirstStaleMinutes && in.CloseRate <= 0:
		out.State = dispatchFinishFirstStateStaleOldest
		out.Reason = "oldest unfinished work exceeded 24 hours with no witnessed close in the sampled windows"
		out.Recovery.Complete = false
		setCap(0)
	case in.RecoveringFromDivergence && in.ConsecutiveConvergingWindows < dispatchFinishFirstRecoveryWindows:
		out.State = dispatchFinishFirstStateRecovering
		out.Reason = "inventory is shrinking, but hysteresis requires three consecutive converging windows before full fan-out"
		out.Recovery.Complete = false
		if in.ConsecutiveConvergingWindows >= 2 {
			setCap(1)
		} else {
			setCap(0)
		}
	case in.RecoveringFromDivergence:
		out.State = dispatchFinishFirstStateConverged
		out.Reason = "three consecutive converging windows restored the configured fresh-start cap"
	default:
		out.Reason = "progress evidence does not require a finish-first hold"
	}
	return out
}

// loadDispatchFinishFirstAdmission joins the already-persisted progress inventory and closure
// ledger. It is read-only and fail-open when no paired inventory exists, preserving legacy wave
// behavior until the existing progress collector has produced enough evidence.
func loadDispatchFinishFirstAdmission(root string, requestedFreshStarts, finishers int, override bool) dispatchFinishFirstAdmission {
	in := dispatchFinishFirstAdmissionInput{RequestedFreshStarts: requestedFreshStarts, Finishers: finishers, Override: override}
	path, err := progressInventoryPath(root)
	if err != nil {
		return evaluateDispatchFinishFirstAdmission(in)
	}
	history, err := readProgressInventoryHistory(path)
	if err != nil || len(history.Snapshots) < 2 {
		return evaluateDispatchFinishFirstAdmission(in)
	}
	in = dispatchFinishFirstInputsFromSnapshots(history.Snapshots, dispatchProgressReadRows(filepath.Join(root, dispatchProgressRunsDir)), requestedFreshStarts, finishers, override)
	return evaluateDispatchFinishFirstAdmission(in)
}

func dispatchFinishFirstInputsFromSnapshots(snapshots []progressInventorySnapshot, progressRows []map[string]any, requestedFreshStarts, finishers int, override bool) dispatchFinishFirstAdmissionInput {
	ordered := append([]progressInventorySnapshot(nil), snapshots...)
	sortProgressInventorySnapshots(ordered)
	last, previous := ordered[len(ordered)-1], ordered[len(ordered)-2]
	in := dispatchFinishFirstAdmissionInput{
		EvidenceAvailable: true, WIPFilesDelta: last.WIPFiles - previous.WIPFiles,
		WIPLinesDelta: last.WIPLines - previous.WIPLines, OldestWIPMinutes: last.OldestWIPMinutes,
		GitHubAvailable:      previous.GitHubAvailable && last.GitHubAvailable,
		RequestedFreshStarts: requestedFreshStarts, Finishers: finishers, Override: override,
	}
	for i := len(ordered) - 1; i > 0; i-- {
		files := ordered[i].WIPFiles - ordered[i-1].WIPFiles
		lines := ordered[i].WIPLines - ordered[i-1].WIPLines
		if files > 0 || lines > 0 {
			in.ConsecutiveDivergingWindows++
			if in.ConsecutiveConvergingWindows > 0 {
				in.RecoveringFromDivergence = true
			}
			break
		}
		if files < 0 || lines < 0 {
			in.ConsecutiveConvergingWindows++
			continue
		}
		break
	}
	windowCount := min(3, len(progressRows))
	if windowCount > 0 {
		closed := 0
		for _, row := range progressRows[len(progressRows)-windowCount:] {
			closed += max(0, dispatchMapInt(row, "closed_now"))
		}
		in.CloseRate = float64(closed) / float64(windowCount)
	}
	return in
}

func sortProgressInventorySnapshots(rows []progressInventorySnapshot) {
	for i := 1; i < len(rows); i++ {
		for j := i; j > 0 && rows[j].ObservedAt.Before(rows[j-1].ObservedAt); j-- {
			rows[j], rows[j-1] = rows[j-1], rows[j]
		}
	}
}
