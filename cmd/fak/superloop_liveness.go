package main

import (
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

type superloopResidual struct {
	Checked          bool                             `json:"checked"`
	UntrackedCount   int                              `json:"untracked_count"`
	Untracked        []string                         `json:"untracked,omitempty"`
	UntrackedClass   string                           `json:"untracked_class,omitempty"`
	UntrackedOwner   string                           `json:"untracked_owner,omitempty"`
	OpenIssues       int                              `json:"open_issues"`
	ActionableIssues int                              `json:"actionable_issues"`
	HeldIssues       int                              `json:"held_issues"`
	IssueMeasured    bool                             `json:"issue_measured"`
	CoverageComplete bool                             `json:"coverage_complete"`
	NextQueue        string                           `json:"next_queue,omitempty"`
	RepairQueues     []dispatchtick.RouterRepairQueue `json:"repair_queues,omitempty"`
	MeasureError     string                           `json:"measure_error,omitempty"`
	ActiveWorkers    int                              `json:"active_workers"`
	CommitThroughput fleetmetrics.CommitThroughput    `json:"commit_throughput"`
	CommitHealth     fleetmetrics.CommitHealth        `json:"commit_health"`
}

var superloopResidualCommand = func(root, name string, args ...string) ([]byte, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = root
	return cmd.Output()
}

var superloopResidualRouter = func(root string) (dispatchtick.RouterPayload, error) {
	return dispatchRouteIssuesComplete(root, io.Discard)
}

var superloopCommitThroughput = fleetmetrics.MeasureCommitThroughput
var superloopActiveWorkers = func(root string, now time.Time) int {
	source := fleetMetricsSources{registryPath: defaultSessionRegistryPath(), staleWindow: defaultSessionStaleWindow, maxSessions: defaultFleetMetricsMaxSessions, stderr: io.Discard}
	inv, _, readable := source.liveInventory(now)
	if !readable {
		return 0
	}
	return inv.Count
}

var superloopResidualAttributor = attributeSuperloopResidual

// keepSuperloopAlive prevents a completed member roster from being mistaken for a
// drained repository. Untracked files are reconciled before fresh issue dispatch;
// otherwise any open issue keeps the loop in generation so the next cycle can select it.
func keepSuperloopAlive(root string, decision superloop.DriveDecision) (superloop.DriveDecision, superloopResidual) {
	if !decision.Satisfied || decision.Enter {
		return decision, superloopResidual{}
	}
	r := measureSuperloopResidual(root)
	decision = superloop.GateCommitThroughput(decision, r.CommitThroughput, r.ActiveWorkers)
	if decision.Enter || !decision.Satisfied {
		return decision, r
	}
	if r.UntrackedCount > 0 {
		r.UntrackedClass, r.UntrackedOwner = superloopResidualAttributor(root, r.Untracked)
		switch r.UntrackedClass {
		case "OWNED_RECONCILE":
			return residualDriveDecision(decision, "owned-untracked-work", "go run ./cmd/fak sweep --json",
				fmt.Sprintf("this session still owns %d untracked path(s); reconcile before fresh dispatch", r.UntrackedCount)), r
		case "ABANDONED_RECOVER":
			return residualDriveDecision(decision, "abandoned-untracked-work", "go run ./cmd/fak tree-doctor --json",
				fmt.Sprintf("%d untracked path(s) have no live owner; inspect and recover or park them", r.UntrackedCount)), r
		case "SCRATCH_REAP":
			return residualDriveDecision(decision, "untracked-scratch", "go run ./cmd/fak tree-doctor --sweep-scratch --dry-run --json",
				fmt.Sprintf("%d residual path(s) are declared scratch; preview their safe reap", r.UntrackedCount)), r
		default:
			decision.Satisfied = false
			decision.Reason = fmt.Sprintf("%d untracked path(s) belong to a live or unknown peer%s; wait rather than steal them", r.UntrackedCount, ownerSuffix(r.UntrackedOwner))
			return decision, r
		}
	}

	if !r.IssueMeasured || !r.CoverageComplete {
		decision.Satisfied = false
		decision.Reason = "actionable-backlog liveness is unknown; refusing to declare drain until routing has complete coverage"
		return decision, r
	}
	if r.OpenIssues == 0 {
		return decision, r
	}
	queue := firstRepairQueue(r.RepairQueues)
	switch queue.Kind {
	case "dispatch":
		return residualDriveDecision(decision, "actionable-issue-backlog", "go run ./cmd/fak dispatch sweep",
			fmt.Sprintf("repository still has %d actionable issue(s); dispatch the next routed leaf", queue.Count)), r
	case "human":
		decision.Satisfied = false
		decision.Reason = fmt.Sprintf("repository has %d open issue(s), all currently held by witnessed human blockers", r.OpenIssues)
		return decision, r
	default:
		return residualDriveDecision(decision, "issue-backlog-"+queue.Kind, issueRepairCommand(queue),
			fmt.Sprintf("repository has no dispatchable leaf, but %d %s issue(s) are repairable: %s", queue.Count, queue.Kind, queue.NextAction)), r
	}
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
	now := time.Now()
	r := superloopResidual{Checked: true}
	r.ActiveWorkers = superloopActiveWorkers(root, now)
	r.CommitThroughput = superloopCommitThroughput(root, now)
	r.CommitHealth = r.CommitThroughput.Health(r.ActiveWorkers)
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

	router, err := superloopResidualRouter(root)
	if err != nil {
		appendResidualError(&r, "issue router: "+err.Error())
		return r
	}
	r.IssueMeasured = true
	r.CoverageComplete = router.Coverage.Complete
	r.OpenIssues = router.Coverage.IssuesFetched
	if r.OpenIssues == 0 && router.Coverage.Complete {
		// Pure tests and injected callers may omit the coverage count; the router's
		// partition still preserves the complete open total.
		r.OpenIssues = router.Counts.Open + router.Counts.SkippedHumanBlocked
	}
	r.ActionableIssues = repairQueueCount(router.RepairQueues, "dispatch")
	r.HeldIssues = r.OpenIssues - r.ActionableIssues
	if r.HeldIssues < 0 {
		r.HeldIssues = 0
	}
	r.RepairQueues = append([]dispatchtick.RouterRepairQueue(nil), router.RepairQueues...)
	if !router.Coverage.Complete {
		appendResidualError(&r, "issue router coverage incomplete: "+router.Reason)
	}
	if queue := firstRepairQueue(r.RepairQueues); queue.Kind != "" {
		r.NextQueue = queue.Kind
	}

	return r
}

func firstRepairQueue(queues []dispatchtick.RouterRepairQueue) dispatchtick.RouterRepairQueue {
	for _, queue := range queues {
		if queue.Count > 0 {
			return queue
		}
	}
	return dispatchtick.RouterRepairQueue{}
}

func issueRepairCommand(queue dispatchtick.RouterRepairQueue) string {
	switch queue.Kind {
	case "split":
		return "go run ./cmd/fak-dev issue decompose --json"
	case "scope", "noise", "private", "other":
		return "go run ./cmd/fak-dev issue repair --json"
	case "route", "decide", "duplicate":
		return "go run ./cmd/fak dispatch route --json"
	default:
		return "go run ./cmd/fak dispatch route --json"
	}
}

func appendResidualError(r *superloopResidual, msg string) {
	if r.MeasureError != "" {
		r.MeasureError += "; "
	}
	r.MeasureError += msg
}

func repairQueueCount(queues []dispatchtick.RouterRepairQueue, kind string) int {
	for _, queue := range queues {
		if queue.Kind == kind {
			return queue.Count
		}
	}
	return 0
}
