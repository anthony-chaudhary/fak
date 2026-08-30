package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const wipQueueSchema = "fak-wip-action-queue/1"

type wipQueueRisk string

const (
	wipQueueRiskProtect wipQueueRisk = "PROTECT"
	wipQueueRiskReview  wipQueueRisk = "REVIEW"
	wipQueueRiskRecover wipQueueRisk = "RECOVER"
	wipQueueRiskReap    wipQueueRisk = "REAP_READY"
	wipQueueRiskHealthy wipQueueRisk = "HEALTHY"
)

type wipQueueAge struct {
	Known   bool   `json:"known"`
	Seconds int64  `json:"seconds"`
	Basis   string `json:"basis"`
}

type wipQueueOwner struct {
	State      string `json:"state"`
	PID        int    `json:"pid"`
	LeaseState string `json:"lease_state"`
	LeaseID    string `json:"lease_id"`
	Lane       string `json:"lane"`
}

type wipQueueProvenance struct {
	Source string `json:"source"`
	ID     string `json:"id"`
	SHA    string `json:"sha"`
}

type wipQueueRow struct {
	Priority    int                  `json:"priority"`
	Kind        string               `json:"kind"`
	ID          string               `json:"id"`
	Reason      string               `json:"reason"`
	Risk        wipQueueRisk         `json:"risk"`
	Owner       wipQueueOwner        `json:"owner"`
	Age         wipQueueAge          `json:"age"`
	State       string               `json:"state"`
	NextCommand string               `json:"next_command"`
	Provenance  []wipQueueProvenance `json:"provenance"`
}

type wipQueueOut struct {
	Schema     string        `json:"schema"`
	ObservedAt time.Time     `json:"observed_at"`
	Repository string        `json:"repository"`
	Count      int           `json:"count"`
	Rows       []wipQueueRow `json:"rows"`
	Errors     []string      `json:"errors,omitempty"`
}

func runWIPQueue(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("wip queue", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit schema-versioned JSON")
	root := fs.String("root", "", "repository root (default: discover from cwd)")
	rootC := fs.String("C", "", "repository root (alias for --root)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak wip queue: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = strings.TrimSpace(*rootC)
	}
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}
	abs, err := filepath.Abs(repoRoot)
	if err != nil || repoRoot == "" {
		if err == nil {
			err = fmt.Errorf("could not resolve a git repository root")
		}
		fmt.Fprintf(stderr, "fak wip queue: %v\n", err)
		return 1
	}
	repoRoot = abs
	now := time.Now().UTC()
	_, paths := workerworktree.Count(repoRoot, nil)
	rows := worktreeWorkerLifecycleInventory(repoRoot, paths, worktreeWorkerLifecycleProbes{})
	// Collect is the package's read-only inspection entry point. Passing the same
	// worker root keeps checkpoint and lifecycle evidence in one fleet envelope.
	inventory := wipinventory.Collect(repoRoot, now, wipinventory.GitRunner{}, wipinventory.Options{WorkerRoot: workerworktree.DefaultRoot()})
	out := buildWIPQueue(repoRoot, now, rows, inventory.Checkpoints, inventory.Errors, os.Stat)
	if *jsonOut {
		b, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak wip queue: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
	} else {
		renderWIPQueue(stdout, out)
	}
	if len(out.Errors) > 0 {
		return 1
	}
	return 0
}

type wipQueueStat func(string) (os.FileInfo, error)

func buildWIPQueue(repoRoot string, now time.Time, lifecycle []worktreeWorkerLifecycleRow, checkpoints []wipinventory.Checkpoint, inventoryErrors []string, stat wipQueueStat) wipQueueOut {
	out := wipQueueOut{Schema: wipQueueSchema, ObservedAt: now.UTC(), Repository: filepath.ToSlash(repoRoot), Errors: append([]string(nil), inventoryErrors...)}
	byHead := make(map[string]int)
	headCount := make(map[string]int)
	for _, lifecycleRow := range lifecycle {
		row := queueRowForLifecycle(now, lifecycleRow, stat)
		out.Rows = append(out.Rows, row)
		if lifecycleRow.HeadSHA != "" {
			headCount[lifecycleRow.HeadSHA]++
			byHead[lifecycleRow.HeadSHA] = len(out.Rows) - 1
		}
	}
	for _, checkpoint := range checkpoints {
		if checkpoint.SHA != "" {
			if i, ok := byHead[checkpoint.SHA]; ok && headCount[checkpoint.SHA] == 1 {
				out.Rows[i].Provenance = append(out.Rows[i].Provenance, checkpointProvenance(checkpoint))
				continue
			}
		}
		out.Rows = append(out.Rows, queueRowForCheckpoint(repoRoot, now, checkpoint))
	}
	sort.SliceStable(out.Rows, func(i, j int) bool {
		if out.Rows[i].Priority != out.Rows[j].Priority {
			return out.Rows[i].Priority < out.Rows[j].Priority
		}
		left, right := strings.ToLower(filepath.ToSlash(out.Rows[i].ID)), strings.ToLower(filepath.ToSlash(out.Rows[j].ID))
		if left != right {
			return left < right
		}
		return out.Rows[i].ID < out.Rows[j].ID
	})
	for i := range out.Rows {
		sort.Slice(out.Rows[i].Provenance, func(a, b int) bool {
			left := out.Rows[i].Provenance[a].Source + "\x00" + out.Rows[i].Provenance[a].ID
			right := out.Rows[i].Provenance[b].Source + "\x00" + out.Rows[i].Provenance[b].ID
			return left < right
		})
	}
	sort.Strings(out.Errors)
	out.Count = len(out.Rows)
	return out
}

func queueRowForLifecycle(now time.Time, row worktreeWorkerLifecycleRow, stat wipQueueStat) wipQueueRow {
	q := wipQueueRow{
		Kind:       "WORKTREE",
		ID:         filepath.ToSlash(row.Path),
		Owner:      wipQueueOwner{State: string(row.Liveness.Owner), PID: row.Association.OwnerPID, LeaseState: string(row.Liveness.Lease), LeaseID: row.Association.LeaseID, Lane: row.Association.Lane},
		Age:        wipQueueAge{Basis: "worktree_path_mtime"},
		State:      string(row.Lifecycle),
		Provenance: []wipQueueProvenance{{Source: "WORKTREE_LIFECYCLE", ID: filepath.ToSlash(row.Path), SHA: row.HeadSHA}},
	}
	if info, err := stat(row.Path); err == nil {
		q.Age.Known = true
		q.Age.Seconds = max(0, int64(now.Sub(info.ModTime()).Seconds()))
	}
	quotedPath := shellQuote(row.Path)
	unknown := row.Association.State == worktreeEvidenceUnknown || row.Liveness.Owner == worktreeEvidenceUnknown || row.Liveness.Lease == worktreeEvidenceUnknown || row.Cleanliness.State == worktreeEvidenceUnknown
	switch {
	case (row.Cleanliness.State == worktreeEvidenceDirty && row.Liveness.Owner != worktreeEvidenceLive) || unknown:
		q.Priority, q.Risk = 1, wipQueueRiskProtect
		q.Reason = "DIRTY_DEAD_OR_UNKNOWN_PROTECTION"
		if unknown {
			q.Reason = "UNKNOWN_EVIDENCE_REQUIRES_PROTECTION"
		}
		q.NextCommand = "git -C " + quotedPath + " status --short"
	case row.Cleanliness.State == worktreeEvidenceClean && row.HeadSHA != "" && row.BaseSHA != "" && row.HeadSHA != row.BaseSHA:
		q.Priority, q.Risk = 2, wipQueueRiskReview
		q.Reason = "CLEAN_UNLANDED_COMMITS"
		q.NextCommand = "git -C " + quotedPath + " log --oneline " + shellQuote(row.BaseSHA+".."+row.HeadSHA)
	case row.Lifecycle == worktreeLifecycleCold && row.ReapReadiness.Reapable:
		q.Priority, q.Risk = 4, wipQueueRiskReap
		q.Reason = "LANDED_COLD_REAP_CANDIDATE"
		q.NextCommand = "fak worktree worker reap --worktree " + quotedPath
	default:
		q.Priority, q.Risk = 5, wipQueueRiskHealthy
		q.Reason = "HEALTHY_OWNED_WIP"
		q.NextCommand = "git -C " + quotedPath + " status --short"
	}
	return q
}

func queueRowForCheckpoint(repoRoot string, now time.Time, checkpoint wipinventory.Checkpoint) wipQueueRow {
	age := wipQueueAge{Basis: "checkpoint_commit_time"}
	if checkpoint.Known && checkpoint.Unix > 0 {
		age.Known = true
		age.Seconds = max(0, now.Unix()-checkpoint.Unix)
	}
	session := strings.TrimPrefix(checkpoint.Ref, "refs/fak/wip/")
	return wipQueueRow{
		Priority:    3,
		Kind:        "CHECKPOINT",
		ID:          checkpoint.Ref,
		Reason:      "LOCAL_ONLY_CHECKPOINT",
		Risk:        wipQueueRiskRecover,
		Owner:       wipQueueOwner{State: "UNKNOWN", LeaseState: "UNKNOWN"},
		Age:         age,
		State:       map[bool]string{true: "KNOWN", false: "UNKNOWN"}[checkpoint.Known],
		NextCommand: "fak wip restore -C " + shellQuote(repoRoot) + " " + shellQuote(session),
		Provenance:  []wipQueueProvenance{checkpointProvenance(checkpoint)},
	}
}

func checkpointProvenance(checkpoint wipinventory.Checkpoint) wipQueueProvenance {
	return wipQueueProvenance{Source: "WIP_CHECKPOINT", ID: checkpoint.Ref, SHA: checkpoint.SHA}
}

func renderWIPQueue(w io.Writer, out wipQueueOut) {
	fmt.Fprintf(w, "WIP ACTION QUEUE — %d read-only recommendations\n", out.Count)
	for _, row := range out.Rows {
		age := "unknown"
		if row.Age.Known {
			age = (time.Duration(row.Age.Seconds) * time.Second).Round(time.Second).String()
		}
		fmt.Fprintf(w, "P%d %-10s risk=%-10s state=%-8s age=%s\n", row.Priority, row.Kind, row.Risk, row.State, age)
		fmt.Fprintf(w, "  %s\n  reason: %s\n", row.ID, row.Reason)
		fmt.Fprintf(w, "  owner: state=%s pid=%d lease-state=%s lease-id=%s lane=%s\n", row.Owner.State, row.Owner.PID, row.Owner.LeaseState, row.Owner.LeaseID, row.Owner.Lane)
		for _, provenance := range row.Provenance {
			fmt.Fprintf(w, "  provenance: source=%s id=%s sha=%s\n", provenance.Source, provenance.ID, provenance.SHA)
		}
		fmt.Fprintf(w, "  next: %s\n", row.NextCommand)
	}
	if len(out.Errors) > 0 {
		fmt.Fprintf(w, "unknown/errors: %d (inspect --json; unknown evidence is never treated as safe to reap)\n", len(out.Errors))
	}
}
