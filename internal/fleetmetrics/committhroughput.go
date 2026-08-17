package fleetmetrics

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const CommitWindow = 10 * time.Minute

// CommitThroughput is repository progress measured from committed trunk history.
// Current and Previous are adjacent half-open windows, so a boundary commit is
// counted exactly once. Working-tree changes and agent self-reports cannot inflate it.
type CommitThroughput struct {
	Measured       bool
	Window         time.Duration
	Current        int
	Previous       int
	LatestCommitAt time.Time
	Error          string
}

// MeasureCommitThroughput folds commits reachable on HEAD's first-parent history.
// First-parent is the shared trunk's landed-work spine; side-branch commits do not
// count until they land, while a merge landing counts once.
func MeasureCommitThroughput(root string, now time.Time) CommitThroughput {
	out := CommitThroughput{Window: CommitWindow}
	start := now.Add(-2 * CommitWindow)
	cmd := exec.Command("git", "log", "--first-parent", "--format=%ct", "--since=@"+strconv.FormatInt(start.Unix(), 10), "HEAD", "--", ".")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	b, err := cmd.Output()
	if err != nil {
		out.Error = fmt.Sprintf("git log: %v", err)
		return out
	}

	currentStart := now.Add(-CommitWindow)
	for _, raw := range strings.Fields(string(b)) {
		unix, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			out.Error = fmt.Sprintf("git log returned invalid commit timestamp %q", raw)
			return out
		}
		at := time.Unix(unix, 0)
		if at.After(now) || at.Before(start) {
			continue
		}
		if out.LatestCommitAt.IsZero() || at.After(out.LatestCommitAt) {
			out.LatestCommitAt = at
		}
		if !at.Before(currentStart) {
			out.Current++
		} else {
			out.Previous++
		}
	}
	out.Measured = true
	return out
}

// Healthy reports whether an active fleet is producing landed work. An idle
// fleet is neutral; unreadable history and zero commits with live loops are red.
type CommitHealth struct {
	State      string
	Healthy    bool
	Reason     string
	NextAction string
}

// Health turns the rate into a top-level fleet verdict. The previous window
// distinguishes a fresh stall from a sustained one so recovery can escalate
// without redefining the positive-commit target.
func (m CommitThroughput) Health(activeWorkers int) CommitHealth {
	switch {
	case activeWorkers <= 0:
		return CommitHealth{State: "idle", Healthy: true, Reason: "no active loops; commit throughput is neutral"}
	case !m.Measured:
		return CommitHealth{State: "unknown", Reason: "active fleet commit history is unreadable", NextAction: "restore repository history measurement before trusting fleet health"}
	case m.Current > 0:
		return CommitHealth{State: "healthy", Healthy: true, Reason: fmt.Sprintf("%d landed commit(s) in the last 10 minutes", m.Current)}
	case m.Previous > 0:
		return CommitHealth{State: "stalled", Reason: "active fleet landed zero commits in the last 10 minutes", NextAction: "inspect admission, leases, tests, and push failures blocking the fleet"}
	default:
		return CommitHealth{State: "blocked", Reason: "active fleet landed zero commits across two consecutive 10-minute windows", NextAction: "stop fan-out and repair the shared blocker before launching more work"}
	}
}

func (m CommitThroughput) Healthy(activeWorkers int) bool {
	return m.Health(activeWorkers).Healthy
}
