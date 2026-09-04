package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const (
	worktreeWorkerLifecycleSchema      = "fak-worker-worktree-lifecycle/1"
	worktreeOwnerStampSchema           = "fak-worker-worktree-owner/1"
	worktreeWorkerLifecycleConcurrency = 16
)

type worktreeWorkerEvidenceState string

const (
	worktreeEvidenceAssociated worktreeWorkerEvidenceState = "ASSOCIATED"
	worktreeEvidenceLive       worktreeWorkerEvidenceState = "LIVE"
	worktreeEvidenceDead       worktreeWorkerEvidenceState = "DEAD"
	worktreeEvidenceReleased   worktreeWorkerEvidenceState = "RELEASED"
	worktreeEvidenceClean      worktreeWorkerEvidenceState = "CLEAN"
	worktreeEvidenceDirty      worktreeWorkerEvidenceState = "DIRTY"
	worktreeEvidenceUnknown    worktreeWorkerEvidenceState = "UNKNOWN"
)

type worktreeWorkerLifecycleState string

const (
	worktreeLifecycleReady    worktreeWorkerLifecycleState = "READY"
	worktreeLifecycleRetained worktreeWorkerLifecycleState = "RETAINED"
	worktreeLifecycleCold     worktreeWorkerLifecycleState = "COLD"
	worktreeLifecycleDirty    worktreeWorkerLifecycleState = "DIRTY"
	worktreeLifecycleUnknown  worktreeWorkerLifecycleState = "UNKNOWN"
)

type worktreeWorkerReapVerdict string

const (
	worktreeReapable worktreeWorkerReapVerdict = "REAPABLE"
	worktreeKeep     worktreeWorkerReapVerdict = "KEEP"
)

type worktreeWorkerAssociation struct {
	State    worktreeWorkerEvidenceState `json:"state"`
	Lane     string                      `json:"lane"`
	OwnerPID int                         `json:"owner_pid"`
	LeaseID  string                      `json:"lease_id"`
}

type worktreeWorkerLiveness struct {
	Owner worktreeWorkerEvidenceState `json:"owner"`
	Lease worktreeWorkerEvidenceState `json:"lease"`
}

type worktreeWorkerCleanliness struct {
	State      worktreeWorkerEvidenceState `json:"state"`
	DirtyPaths []string                    `json:"dirty_paths"`
}

type worktreeWorkerReapReadiness struct {
	Reapable bool                      `json:"reapable"`
	Verdict  worktreeWorkerReapVerdict `json:"verdict"`
	Reason   string                    `json:"reason"`
}

// worktreeWorkerLifecycleRow is the stable machine row for one exact registered
// sanctioned worktree. Evidence fields are always present: an unreadable owner
// stamp, liveness store, revision, or status is represented as UNKNOWN rather
// than omitted, and UNKNOWN can never derive a reapable verdict.
type worktreeWorkerIntentStatus string

const (
	worktreeWorkerIntentPresent worktreeWorkerIntentStatus = "PRESENT"
	worktreeWorkerIntentMissing worktreeWorkerIntentStatus = "MISSING"
	worktreeWorkerIntentInvalid worktreeWorkerIntentStatus = "INVALID"
)

type worktreeWorkerIntent struct {
	Status      worktreeWorkerIntentStatus `json:"status"`
	Diagnostic  string                     `json:"diagnostic,omitempty"`
	IssueNumber int                        `json:"issue_number,omitempty"`
	Message     string                     `json:"message,omitempty"`
	Paths       []string                   `json:"paths,omitempty"`
}

type worktreeWorkerLifecycleRow struct {
	Path          string                       `json:"path"`
	Intent        worktreeWorkerIntent         `json:"intent"`
	HeadSHA       string                       `json:"head_sha"`
	BaseSHA       string                       `json:"base_sha"`
	Association   worktreeWorkerAssociation    `json:"association"`
	Liveness      worktreeWorkerLiveness       `json:"liveness"`
	Cleanliness   worktreeWorkerCleanliness    `json:"cleanliness"`
	Lifecycle     worktreeWorkerLifecycleState `json:"lifecycle"`
	ReapReadiness worktreeWorkerReapReadiness  `json:"reap_readiness"`
}

type worktreeWorkerLifecycleOut struct {
	Schema    string                          `json:"schema"`
	Count     int                             `json:"count"`
	Paths     []string                        `json:"paths"`
	Inventory []worktreeWorkerLifecycleRow    `json:"inventory"`
	Capacity  workerworktree.CapacityAdvisory `json:"capacity"`
	Partial   bool                            `json:"partial,omitempty"`
	Timeout   bool                            `json:"timeout,omitempty"`
}

type worktreeWorkerRevisionEvidence struct {
	HeadSHA     string
	BaseSHA     string
	Cleanliness worktreeWorkerEvidenceState
	DirtyPaths  []string
}

type worktreeWorkerLifecycleProbes struct {
	ReadOwner      func(path string) (workerworktree.OwnerStamp, error)
	ProcessAlive   func(pid int) (bool, error)
	LeaseLive      func(leaseID string) (bool, error)
	Inspect        func(repoRoot, path string) (worktreeWorkerRevisionEvidence, error)
	InspectContext func(ctx context.Context, repoRoot, path string) (worktreeWorkerRevisionEvidence, error)
}

func readWorktreeWorkerOwner(path string) (workerworktree.OwnerStamp, error) {
	raw, err := os.ReadFile(workerworktree.OwnerStampPath(path))
	if err != nil {
		return workerworktree.OwnerStamp{}, err
	}
	var stamp workerworktree.OwnerStamp
	if err := json.Unmarshal(raw, &stamp); err != nil {
		return workerworktree.OwnerStamp{}, err
	}
	stamp.LeaseID = strings.TrimSpace(stamp.LeaseID)
	if stamp.Schema != worktreeOwnerStampSchema || stamp.PID <= 0 || stamp.LeaseID == "" || stamp.CreatedAt.IsZero() {
		return workerworktree.OwnerStamp{}, fmt.Errorf("invalid owner stamp")
	}
	stamp.CreatedAt = stamp.CreatedAt.UTC()
	return stamp, nil
}

func worktreeWorkerGitOutput(dir string, args ...string) (string, error) {
	return worktreeWorkerGitOutputContext(context.Background(), dir, args...)
}

func worktreeWorkerGitOutputContext(ctx context.Context, dir string, args ...string) (string, error) {
	cmdArgs := append([]string{"-C", dir}, args...)
	cmd := windowgate.CommandContext(ctx, "git", cmdArgs...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		detail := strings.TrimSpace(string(out))
		if detail == "" {
			detail = err.Error()
		}
		return "", fmt.Errorf("%s", detail)
	}
	return strings.TrimSpace(string(out)), nil
}

func worktreeWorkerDirtyPaths(status string) []string {
	paths := make([]string, 0)
	for _, line := range strings.Split(strings.TrimRight(status, "\r\n"), "\n") {
		line = strings.TrimSuffix(line, "\r")
		if len(line) < 4 {
			continue
		}
		name := strings.TrimSpace(line[3:])
		if i := strings.LastIndex(name, " -> "); i >= 0 {
			name = name[i+4:]
		}
		name = strings.Trim(name, `"`)
		if name != "" {
			paths = append(paths, strings.ReplaceAll(name, "\\", "/"))
		}
	}
	sort.Strings(paths)
	return paths
}

func inspectWorktreeWorker(repoRoot, path string) (worktreeWorkerRevisionEvidence, error) {
	return inspectWorktreeWorkerContext(context.Background(), repoRoot, path)
}

func inspectWorktreeWorkerContext(ctx context.Context, repoRoot, path string) (worktreeWorkerRevisionEvidence, error) {
	evidence := worktreeWorkerRevisionEvidence{
		Cleanliness: worktreeEvidenceUnknown,
		DirtyPaths:  []string{},
	}
	head, err := worktreeWorkerGitOutputContext(ctx, path, "rev-parse", "--verify", "HEAD")
	if err != nil {
		return evidence, fmt.Errorf("read worktree HEAD: %w", err)
	}
	if head == "" {
		return evidence, fmt.Errorf("read worktree HEAD: empty result")
	}
	evidence.HeadSHA = head
	trunkHead, err := worktreeWorkerGitOutputContext(ctx, repoRoot, "rev-parse", "--verify", "HEAD")
	if err != nil {
		return evidence, fmt.Errorf("read trunk HEAD: %w", err)
	}
	if trunkHead == "" {
		return evidence, fmt.Errorf("read trunk HEAD: empty result")
	}
	base, err := worktreeWorkerGitOutputContext(ctx, repoRoot, "merge-base", trunkHead, head)
	if err != nil {
		return evidence, fmt.Errorf("derive worktree base: %w", err)
	}
	if base == "" {
		return evidence, fmt.Errorf("derive worktree base: empty result")
	}
	evidence.BaseSHA = base
	status, err := worktreeWorkerGitOutputContext(ctx, path, "status", "--porcelain=v1", "--untracked-files=all")
	if err != nil {
		return evidence, fmt.Errorf("inspect worktree status: %w", err)
	}
	evidence.DirtyPaths = worktreeWorkerDirtyPaths(status)
	if len(evidence.DirtyPaths) == 0 {
		evidence.Cleanliness = worktreeEvidenceClean
	} else {
		evidence.Cleanliness = worktreeEvidenceDirty
	}
	return evidence, nil
}

func worktreeWorkerLeaseProbe(root string, now time.Time) func(string) (bool, error) {
	live, _, err := leaseref.NewInDir(root).Live(context.Background(), now)
	if err != nil {
		return func(string) (bool, error) { return false, err }
	}
	ids := map[string]bool{}
	lanes := map[string]bool{}
	for _, rec := range live {
		id := strings.ToLower(strings.TrimSpace(rec.ID))
		ids[id] = true
		for _, lane := range dispatchLeaseLanes(id) {
			lanes[strings.ToLower(lane)] = true
		}
	}
	return func(leaseID string) (bool, error) {
		leaseID = strings.ToLower(strings.TrimSpace(leaseID))
		if leaseID == "" {
			return false, fmt.Errorf("lease id is empty")
		}
		if ids[leaseID] {
			return true, nil
		}
		for _, lane := range dispatchLeaseLanes(leaseID) {
			if lanes[strings.ToLower(lane)] {
				return true, nil
			}
		}
		return false, nil
	}
}

func worktreeWorkerLifecycleInventory(repoRoot string, paths []string, probes worktreeWorkerLifecycleProbes) []worktreeWorkerLifecycleRow {
	rows, _ := worktreeWorkerLifecycleInventoryContext(context.Background(), repoRoot, paths, probes)
	return rows
}

func worktreeWorkerLifecycleInventoryContext(ctx context.Context, repoRoot string, paths []string, probes worktreeWorkerLifecycleProbes) ([]worktreeWorkerLifecycleRow, bool) {
	if probes.ReadOwner == nil {
		probes.ReadOwner = readWorktreeWorkerOwner
	}
	if probes.ProcessAlive == nil {
		probes.ProcessAlive = func(pid int) (bool, error) { return dispatchPIDAlive(pid), nil }
	}
	if probes.LeaseLive == nil {
		probes.LeaseLive = worktreeWorkerLeaseProbe(repoRoot, time.Now())
	}
	if probes.Inspect == nil && probes.InspectContext == nil {
		probes.InspectContext = inspectWorktreeWorkerContext
	}

	sortedPaths := append([]string(nil), paths...)
	sort.Slice(sortedPaths, func(i, j int) bool {
		left := strings.ToLower(filepath.ToSlash(sortedPaths[i]))
		right := strings.ToLower(filepath.ToSlash(sortedPaths[j]))
		if left == right {
			return sortedPaths[i] < sortedPaths[j]
		}
		return left < right
	})
	if len(sortedPaths) == 0 {
		return []worktreeWorkerLifecycleRow{}, false
	}

	if ctx != nil && ctx.Err() != nil {
		return []worktreeWorkerLifecycleRow{}, true
	}

	type probeResult struct {
		index int
		row   worktreeWorkerLifecycleRow
		err   error
	}

	workers := min(worktreeWorkerLifecycleConcurrency, len(sortedPaths))
	jobs := make(chan int)
	results := make(chan probeResult, len(sortedPaths))
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			for i := range jobs {
				if ctx != nil && ctx.Err() != nil {
					results <- probeResult{index: i, err: ctx.Err()}
					continue
				}
				row, err := worktreeWorkerLifecycleRowForPathContext(ctx, repoRoot, sortedPaths[i], probes)
				results <- probeResult{index: i, row: row, err: err}
			}
		}()
	}

	go func() {
		for i := range sortedPaths {
			if ctx != nil && ctx.Err() != nil {
				break
			}
			select {
			case jobs <- i:
			case <-ctxDone(ctx):
				close(jobs)
				return
			}
		}
		close(jobs)
	}()

	completedRows := make([]worktreeWorkerLifecycleRow, len(sortedPaths))
	completed := make([]bool, len(sortedPaths))
	var timedOut bool
	totalJobs := len(sortedPaths)
	receivedCount := 0

collectLoop:
	for receivedCount < totalJobs {
		select {
		case res := <-results:
			receivedCount++
			if res.err != nil {
				if errors.Is(res.err, context.DeadlineExceeded) || errors.Is(res.err, context.Canceled) || (ctx != nil && ctx.Err() != nil) || strings.Contains(strings.ToLower(res.err.Error()), "timeout") {
					timedOut = true
				} else {
					completed[res.index] = true
					completedRows[res.index] = res.row
				}
			} else {
				completed[res.index] = true
				completedRows[res.index] = res.row
			}
		case <-ctxDone(ctx):
			timedOut = true
			break collectLoop
		}
	}

	if ctx != nil && ctx.Err() != nil {
		timedOut = true
	}

	if !timedOut {
		return completedRows, false
	}

	out := make([]worktreeWorkerLifecycleRow, 0, len(sortedPaths))
	for i, ok := range completed {
		if ok {
			out = append(out, completedRows[i])
		}
	}
	return out, true
}

func ctxDone(ctx context.Context) <-chan struct{} {
	if ctx == nil {
		return nil
	}
	return ctx.Done()
}

func worktreeWorkerLifecycleRowForPath(repoRoot, path string, probes worktreeWorkerLifecycleProbes) worktreeWorkerLifecycleRow {
	row, _ := worktreeWorkerLifecycleRowForPathContext(context.Background(), repoRoot, path, probes)
	return row
}

func worktreeWorkerLifecycleRowForPathContext(ctx context.Context, repoRoot, path string, probes worktreeWorkerLifecycleProbes) (worktreeWorkerLifecycleRow, error) {
	row := worktreeWorkerLifecycleRow{
		Path: path,
		Association: worktreeWorkerAssociation{
			State: worktreeEvidenceUnknown,
			Lane:  workerworktree.LaneOf(path),
		},
		Liveness: worktreeWorkerLiveness{
			Owner: worktreeEvidenceUnknown,
			Lease: worktreeEvidenceUnknown,
		},
		Cleanliness: worktreeWorkerCleanliness{
			State:      worktreeEvidenceUnknown,
			DirtyPaths: []string{},
		},
	}

	intent, intentErr := workerworktree.LoadIntent(path)
	switch {
	case intentErr == nil:
		row.Intent = worktreeWorkerIntent{
			Status:      worktreeWorkerIntentPresent,
			IssueNumber: intent.IssueNumber,
			Message:     intent.Message,
			Paths:       append([]string(nil), intent.Paths...),
		}
		if row.Intent.Paths == nil {
			row.Intent.Paths = []string{}
		}
	case os.IsNotExist(intentErr):
		row.Intent = worktreeWorkerIntent{Status: worktreeWorkerIntentMissing, Diagnostic: "worker intent not found"}
	default:
		row.Intent = worktreeWorkerIntent{Status: worktreeWorkerIntentInvalid, Diagnostic: intentErr.Error()}
	}

	stamp, ownerErr := probes.ReadOwner(path)
	if ownerErr == nil {
		row.Association.State = worktreeEvidenceAssociated
		row.Association.OwnerPID = stamp.PID
		row.Association.LeaseID = strings.TrimSpace(stamp.LeaseID)
		if live, err := probes.ProcessAlive(stamp.PID); err == nil {
			if live {
				row.Liveness.Owner = worktreeEvidenceLive
			} else {
				row.Liveness.Owner = worktreeEvidenceDead
			}
		}
		if live, err := probes.LeaseLive(stamp.LeaseID); err == nil {
			if live {
				row.Liveness.Lease = worktreeEvidenceLive
			} else {
				row.Liveness.Lease = worktreeEvidenceReleased
			}
		}
	}

	var evidence worktreeWorkerRevisionEvidence
	var inspectErr error
	if probes.InspectContext != nil {
		evidence, inspectErr = probes.InspectContext(ctx, repoRoot, path)
	} else if probes.Inspect != nil {
		evidence, inspectErr = probes.Inspect(repoRoot, path)
	} else {
		evidence, inspectErr = inspectWorktreeWorkerContext(ctx, repoRoot, path)
	}

	row.HeadSHA = strings.TrimSpace(evidence.HeadSHA)
	row.BaseSHA = strings.TrimSpace(evidence.BaseSHA)
	switch evidence.Cleanliness {
	case worktreeEvidenceClean, worktreeEvidenceDirty:
		row.Cleanliness.State = evidence.Cleanliness
		row.Cleanliness.DirtyPaths = append([]string(nil), evidence.DirtyPaths...)
		if row.Cleanliness.DirtyPaths == nil {
			row.Cleanliness.DirtyPaths = []string{}
		}
		sort.Strings(row.Cleanliness.DirtyPaths)
	}
	row.Lifecycle, row.ReapReadiness = lifecycleVerdict(row)
	return row, inspectErr
}

func lifecycleVerdict(row worktreeWorkerLifecycleRow) (worktreeWorkerLifecycleState, worktreeWorkerReapReadiness) {
	keep := func(state worktreeWorkerLifecycleState, reason string) (worktreeWorkerLifecycleState, worktreeWorkerReapReadiness) {
		return state, worktreeWorkerReapReadiness{Reapable: false, Verdict: worktreeKeep, Reason: reason}
	}
	switch {
	case row.Association.State != worktreeEvidenceAssociated || row.Association.Lane == "" ||
		row.Association.OwnerPID <= 0 || strings.TrimSpace(row.Association.LeaseID) == "":
		return keep(worktreeLifecycleUnknown, "ASSOCIATION_UNKNOWN")
	case row.HeadSHA == "" || row.BaseSHA == "":
		return keep(worktreeLifecycleUnknown, "REVISION_UNKNOWN")
	case row.Liveness.Owner == worktreeEvidenceUnknown:
		return keep(worktreeLifecycleUnknown, "OWNER_LIVENESS_UNKNOWN")
	case row.Liveness.Lease == worktreeEvidenceUnknown:
		return keep(worktreeLifecycleUnknown, "LEASE_LIVENESS_UNKNOWN")
	case row.Cleanliness.State == worktreeEvidenceUnknown:
		return keep(worktreeLifecycleUnknown, "CLEANLINESS_UNKNOWN")
	case row.Liveness.Owner == worktreeEvidenceLive:
		return keep(worktreeLifecycleReady, "OWNER_LIVE")
	case row.Liveness.Lease == worktreeEvidenceLive:
		return keep(worktreeLifecycleRetained, "LEASE_LIVE")
	case row.Cleanliness.State == worktreeEvidenceDirty:
		return keep(worktreeLifecycleDirty, "WORKTREE_DIRTY")
	case row.Liveness.Owner == worktreeEvidenceDead &&
		row.Liveness.Lease == worktreeEvidenceReleased &&
		row.Cleanliness.State == worktreeEvidenceClean:
		return worktreeLifecycleCold, worktreeWorkerReapReadiness{
			Reapable: true,
			Verdict:  worktreeReapable,
			Reason:   "COLD_CLEAN",
		}
	default:
		return keep(worktreeLifecycleUnknown, "EVIDENCE_CONTRADICTORY")
	}
}

func worktreeWorkerRetainedTrees(rows []worktreeWorkerLifecycleRow) []workerworktree.RetainedTree {
	trees := make([]workerworktree.RetainedTree, 0, len(rows))
	for _, row := range rows {
		trees = append(trees, workerworktree.RetainedTree{
			Path:         row.Path,
			ColdReapable: row.ReapReadiness.Reapable && row.Cleanliness.State == worktreeEvidenceClean,
			OwnerDead:    row.Liveness.Owner == worktreeEvidenceDead,
			Clean:        row.Cleanliness.State == worktreeEvidenceClean,
		})
	}
	return trees
}

func worktreeWorkerCapacityAdvisory(repoRoot string, census workerworktree.CapacityCensus, prospectiveCount int, reason string, rows []worktreeWorkerLifecycleRow) workerworktree.CapacityAdvisory {
	if census.Known && prospectiveCount > workerworktree.AdvisoryCapacitySetpoint && rows == nil {
		rows = worktreeWorkerLifecycleInventory(repoRoot, census.Paths, worktreeWorkerLifecycleProbes{})
	}
	return workerworktree.AssessCapacity(
		len(census.Paths), prospectiveCount, census.Known, reason, worktreeWorkerRetainedTrees(rows),
	)
}

func worktreeWorkerWriteCapacityHuman(w io.Writer, advisory workerworktree.CapacityAdvisory) {
	if advisory.Status == workerworktree.CapacityWithinSetpoint {
		return
	}
	fmt.Fprintf(w, "capacity advisory: %s\n", advisory.Message)
	for _, recommendation := range advisory.ContractionRecommendations {
		fmt.Fprintf(w, "capacity contraction: candidate=%s basis=%s; inspect with: %s\n",
			recommendation.Path, recommendation.Basis, recommendation.Action)
	}
}
