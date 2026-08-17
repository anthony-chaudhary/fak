package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

type superloopResidual struct {
	Checked        bool     `json:"checked"`
	UntrackedCount int      `json:"untracked_count"`
	Untracked      []string `json:"untracked,omitempty"`
	OpenIssues     int      `json:"open_issues"`
	IssueSample    []int    `json:"issue_sample,omitempty"`
	IssueMeasured  bool     `json:"issue_measured"`
	MeasureError   string   `json:"measure_error,omitempty"`
}

var superloopResidualCommand = func(root, name string, args ...string) ([]byte, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = root
	return cmd.Output()
}

// keepSuperloopAlive prevents a completed member roster from being mistaken for a
// drained repository. Untracked files are reconciled before fresh issue dispatch;
// otherwise any open issue keeps the loop in generation so the next cycle can select it.
func keepSuperloopAlive(root string, decision superloop.DriveDecision) (superloop.DriveDecision, superloopResidual) {
	if !decision.Satisfied || decision.Enter {
		return decision, superloopResidual{}
	}
	r := measureSuperloopResidual(root)
	if r.UntrackedCount > 0 {
		return residualDriveDecision(decision, "local-untracked-work", "go run ./cmd/fak sweep --json",
			fmt.Sprintf("repository still has %d untracked path(s); reconcile local work before declaring drain", r.UntrackedCount)), r
	}
	if !r.IssueMeasured {
		decision.Satisfied = false
		decision.Reason = "open-issue liveness is unknown; refusing to declare drain until it is measured"
		return decision, r
	}
	if r.OpenIssues > 0 {
		return residualDriveDecision(decision, "open-issue-backlog", "go run ./cmd/fak dispatch sweep",
			fmt.Sprintf("repository still has %d open issue(s); dispatch the next actionable unit", r.OpenIssues)), r
	}
	return decision, r
}

func residualDriveDecision(base superloop.DriveDecision, ref, action, reason string) superloop.DriveDecision {
	base.Enter = true
	base.Satisfied = false
	base.Member = superloop.Member{Kind: superloop.KindSurface, Ref: ref, Enter: action}
	base.Action = action
	base.Reason = reason
	return base
}

func measureSuperloopResidual(root string) superloopResidual {
	r := superloopResidual{Checked: true}
	if out, err := superloopResidualCommand(root, "git", "ls-files", "--others", "--exclude-standard", "-z"); err != nil {
		r.MeasureError = "untracked: " + err.Error()
	} else {
		for _, path := range strings.Split(string(out), "\x00") {
			if path == "" {
				continue
			}
			r.Untracked = append(r.Untracked, path)
		}
		r.UntrackedCount = len(r.Untracked)
	}

	out, err := superloopResidualCommand(root, "gh", "issue", "list", "--state", "open", "--limit", "100000", "--json", "number")
	if err != nil {
		if r.MeasureError != "" {
			r.MeasureError += "; "
		}
		r.MeasureError += "open issues: " + err.Error()
		return r
	}
	var rows []struct {
		Number json.Number `json:"number"`
	}
	if err := json.Unmarshal(out, &rows); err != nil {
		if r.MeasureError != "" {
			r.MeasureError += "; "
		}
		r.MeasureError += "open issues: " + err.Error()
		return r
	}
	r.IssueMeasured = true
	r.OpenIssues = len(rows)
	for _, row := range rows {
		if len(r.IssueSample) == 5 {
			break
		}
		if n, err := strconv.Atoi(row.Number.String()); err == nil {
			r.IssueSample = append(r.IssueSample, n)
		}
	}
	return r
}
