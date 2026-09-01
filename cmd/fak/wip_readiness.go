package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
	"github.com/anthony-chaudhary/fak/internal/wipreadiness"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const defaultWIPReadinessMaxAge = 5 * time.Minute

type wipReadinessScanner struct {
	root   string
	remote string
	now    func() time.Time
}

func runWIPReadiness(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("wip readiness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit schema-versioned JSON")
	root := fs.String("root", "", "repository root (default: discover from cwd)")
	rootC := fs.String("C", "", "repository root (alias for --root)")
	maxAge := fs.Duration("max-age", defaultWIPReadinessMaxAge, "maximum age for reusable evidence")
	remote := fs.String("remote", "origin", "remote lifecycle snapshot namespace (empty disables remote coverage)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *maxAge < 0 {
		fmt.Fprintln(stderr, "usage: fak wip readiness --json [-C DIR|--root DIR] [--max-age DURATION] [--remote NAME]")
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
		fmt.Fprintf(stderr, "fak wip readiness: %v\n", err)
		return 1
	}
	now := time.Now().UTC()
	cache := wipreadiness.NewCacheWithClock(wipReadinessScanner{root: abs, remote: strings.TrimSpace(*remote), now: func() time.Time { return now }}, *maxAge, func() time.Time { return now })
	receipt := cache.Receipt(context.Background())
	if *jsonOut {
		return encodeJSONOrFail(stdout, stderr, receipt, "fak wip readiness")
	}
	fmt.Fprintf(stdout, "WIP READINESS — %s (observed %s; reusable until %s)\n", strings.ToUpper(string(receipt.Verdict)), receipt.ObservedAt.Format(time.RFC3339), receipt.ExpiresAt.Format(time.RFC3339))
	if len(receipt.Reasons) > 0 {
		fmt.Fprintf(stdout, "reasons: %s\n", joinWIPReadinessReasons(receipt.Reasons))
	}
	fmt.Fprintf(stdout, "queue=%d inventory=%d lifecycle=%d capacity=%d hosts=%d/%d evidence_only=%t\n", receipt.Queue.Summary.Total, receipt.Inventory.Summary.Total, receipt.Lifecycle.Summary.Total, receipt.Capacity.Summary.Total, len(receipt.Hosts.Observed), len(receipt.Hosts.Expected), receipt.EvidenceOnly)
	return 0
}

func (s wipReadinessScanner) Scan(context.Context) (wipreadiness.Observation, error) {
	now := s.now().UTC()
	inventory := wipinventory.Collect(s.root, now, wipinventory.GitRunner{}, wipinventory.Options{WorkerRoot: workerworktree.DefaultRoot()})
	census := workerworktree.CapacityCensusFor(s.root, nil)
	rows := []worktreeWorkerLifecycleRow{}
	if census.Known {
		rows = worktreeWorkerLifecycleInventory(s.root, census.Paths, worktreeWorkerLifecycleProbes{})
	}
	queue := buildWIPQueue(s.root, now, rows, inventory.Checkpoints, inventory.Errors, os.Stat)
	lifecycle, lifecycleErr := wiplifecycle.ListWithDiagnostics(s.root)
	capacity := worktreeWorkerCapacityAdvisory(s.root, census, len(rows), "", rows)

	observation := wipreadiness.Observation{
		ObservedAt: now,
		Queue:      wipreadiness.Source{Name: "queue", Schema: queue.Schema, ExpectedSchema: wipQueueSchema, ObservedAt: queue.ObservedAt, Available: true, Complete: len(queue.Errors) == 0, Summary: wipreadiness.Summary{Total: queue.Count}},
		Inventory:  wipreadiness.Source{Name: "inventory", Schema: inventory.Schema, ExpectedSchema: wipinventory.Schema, ObservedAt: inventory.ObservedAt, Available: true, Complete: len(inventory.Errors) == 0, Summary: wipreadiness.Summary{Total: inventory.Main.Tracked.Count + inventory.Main.Untracked.Count + len(inventory.Worktrees)}},
		Lifecycle:  wipreadiness.Source{Name: "lifecycle", Schema: wiplifecycle.Schema, ExpectedSchema: wiplifecycle.Schema, ObservedAt: now, Available: lifecycleErr == nil, Complete: lifecycleErr == nil && len(lifecycle.Diagnostics) == 0, Summary: wipreadiness.Summary{Total: len(lifecycle.Receipts)}},
		Capacity:   wipreadiness.Source{Name: "capacity", Schema: capacity.Schema, ExpectedSchema: workerworktree.CapacityAdvisorySchema, ObservedAt: now, Available: census.Known, Complete: census.Known, Summary: wipreadiness.Summary{Total: capacity.CurrentCount}},
		Hosts:      wipreadiness.HostCoverage{},
	}
	observation.Queue.Diagnostics = stringDiagnostics("queue", queue.Errors)
	observation.Inventory.Diagnostics = stringDiagnostics("inventory", inventory.Errors)
	if lifecycleErr != nil {
		observation.Lifecycle.Diagnostics = []wipreadiness.Diagnostic{{Source: "lifecycle", Code: "UNAVAILABLE", Message: lifecycleErr.Error()}}
	} else {
		for _, diagnostic := range lifecycle.Diagnostics {
			observation.Lifecycle.Diagnostics = append(observation.Lifecycle.Diagnostics, wipreadiness.Diagnostic{Source: "lifecycle", Code: diagnostic.Code, Message: diagnostic.Error})
		}
	}
	if !census.Known {
		observation.Capacity.Diagnostics = []wipreadiness.Diagnostic{{Source: "capacity", Code: "UNAVAILABLE", Message: "managed worker worktree census unavailable"}}
	}

	localHost := "host-unknown"
	if hostname, err := os.Hostname(); err == nil && strings.TrimSpace(hostname) != "" {
		localHost = workerworktree.SnapshotHostID(hostname)
	}
	observation.Hosts.Expected = []string{localHost}
	observation.Hosts.Observed = []string{localHost}
	for _, row := range rows {
		observation.Work = append(observation.Work, wipreadiness.Work{ID: filepath.ToSlash(row.Path), Dirty: row.Cleanliness.State == worktreeEvidenceDirty, Ownership: wipreadiness.OwnershipLocal, Host: localHost})
	}
	if s.remote != "" {
		groups, err := workerworktree.ListRemoteSnapshots(s.root, s.remote, now, nil)
		if err != nil {
			observation.Diagnostics = append(observation.Diagnostics, wipreadiness.Diagnostic{Source: "hosts", Code: "REMOTE_SNAPSHOTS_UNAVAILABLE", Message: err.Error()})
		} else {
			for _, group := range groups {
				observation.Hosts.Expected = append(observation.Hosts.Expected, group.Host)
				if group.Authoritative && group.Freshness == workerworktree.SnapshotFresh {
					observation.Hosts.Observed = append(observation.Hosts.Observed, group.Host)
				}
				for i, row := range group.Rows {
					id := row.HeadSHA
					if id == "" {
						id = fmt.Sprintf("%s/%d", group.Host, i)
					}
					observation.Work = append(observation.Work, wipreadiness.Work{ID: id, Dirty: row.Cleanliness.State == string(worktreeEvidenceDirty), Ownership: wipreadiness.OwnershipRemote, Host: group.Host})
				}
				if group.Reason != "" {
					observation.Diagnostics = append(observation.Diagnostics, wipreadiness.Diagnostic{Source: "hosts", Code: "REMOTE_SNAPSHOT_" + string(group.Freshness), Message: group.Reason})
				}
			}
		}
	}
	sort.Strings(observation.Hosts.Expected)
	observation.Hosts.Expected = compactStrings(observation.Hosts.Expected)
	sort.Strings(observation.Hosts.Observed)
	observation.Hosts.Observed = compactStrings(observation.Hosts.Observed)
	for _, work := range observation.Work {
		if work.Dirty {
			observation.Inventory.Summary.Dirty++
		}
		if work.Ownership == wipreadiness.OwnershipRemote {
			observation.Inventory.Summary.RemoteOwned++
		}
	}
	return observation, nil
}

func stringDiagnostics(source string, errors []string) []wipreadiness.Diagnostic {
	out := make([]wipreadiness.Diagnostic, 0, len(errors))
	for _, message := range errors {
		out = append(out, wipreadiness.Diagnostic{Source: source, Code: "COLLECTION_ERROR", Message: message})
	}
	return out
}

func compactStrings(values []string) []string {
	out := values[:0]
	for _, value := range values {
		if len(out) == 0 || out[len(out)-1] != value {
			out = append(out, value)
		}
	}
	return out
}

func joinWIPReadinessReasons(reasons []wipreadiness.ReasonCode) string {
	values := make([]string, len(reasons))
	for i, reason := range reasons {
		values[i] = string(reason)
	}
	return strings.Join(values, ",")
}
