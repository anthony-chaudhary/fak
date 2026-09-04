package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

var worktreeWorkerListProbes worktreeWorkerLifecycleProbes

func worktreeMatchesWorker(path, target string) bool {
	target = strings.TrimSpace(target)
	if target == "" {
		return true
	}
	base := filepath.Base(path)
	if base == target || strings.EqualFold(base, target) {
		return true
	}
	cleanPath := filepath.Clean(path)
	cleanTarget := filepath.Clean(target)
	if cleanPath == cleanTarget || strings.EqualFold(cleanPath, cleanTarget) {
		return true
	}
	slashPath := filepath.ToSlash(cleanPath)
	slashTarget := filepath.ToSlash(cleanTarget)
	if slashPath == slashTarget || strings.EqualFold(slashPath, slashTarget) {
		return true
	}
	if absTarget, err := filepath.Abs(target); err == nil {
		cleanAbs := filepath.Clean(absTarget)
		if cleanPath == cleanAbs || strings.EqualFold(cleanPath, cleanAbs) {
			return true
		}
		if slashPath == filepath.ToSlash(cleanAbs) || strings.EqualFold(slashPath, filepath.ToSlash(cleanAbs)) {
			return true
		}
	}
	return false
}

func worktreeMatchesSession(path, target string) bool {
	target = strings.TrimSpace(target)
	if target == "" {
		return true
	}

	// Check owner stamp
	stampPath := workerworktree.OwnerStampPath(path)
	if raw, err := os.ReadFile(stampPath); err == nil {
		var ownerData struct {
			SessionID string `json:"session_id"`
			Session   string `json:"session"`
			LeaseID   string `json:"lease_id"`
		}
		if err := json.Unmarshal(raw, &ownerData); err == nil {
			if strings.EqualFold(strings.TrimSpace(ownerData.SessionID), target) ||
				strings.EqualFold(strings.TrimSpace(ownerData.Session), target) ||
				strings.EqualFold(strings.TrimSpace(ownerData.LeaseID), target) {
				return true
			}
		}
	}

	// Check intent
	intentPath := filepath.Join(filepath.Dir(path), ".fak-worker-intents", filepath.Base(path)+".json")
	if raw, err := os.ReadFile(intentPath); err == nil {
		var intentData struct {
			SessionID string `json:"session_id"`
			Session   string `json:"session"`
			LeaseID   string `json:"lease_id"`
		}
		if err := json.Unmarshal(raw, &intentData); err == nil {
			if strings.EqualFold(strings.TrimSpace(intentData.SessionID), target) ||
				strings.EqualFold(strings.TrimSpace(intentData.Session), target) ||
				strings.EqualFold(strings.TrimSpace(intentData.LeaseID), target) {
				return true
			}
		}
	}

	// Check lease.json
	if lease, err := workerworktree.ReadWorkerLease(path); err == nil {
		if strings.EqualFold(strings.TrimSpace(lease.SessionID), target) {
			return true
		}
	}

	return false
}

func worktreeWorkerList(argv []string) {
	fs := flag.NewFlagSet("worktree worker list", flag.ExitOnError)
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit stable typed association, liveness, cleanliness, lifecycle, and reap-readiness rows")
	capacityReason := fs.String("capacity-reason", "", "record why capacity above the advisory setpoint is retained (advisory evidence)")
	remote := fs.String("remote", "", "explicit remote whose scrubbed host snapshots are included")
	fetch := fs.Bool("fetch", false, "refresh the remote snapshot mirror before listing")
	worker := fs.String("worker", "", "filter listing to this worker worktree name or path")
	session := fs.String("session", "", "filter listing to this session ID")
	timeout := fs.Duration("timeout", 15*time.Second, "maximum duration before returning partial results (default 15s; 0 disables timeout)")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	census := workerworktree.CapacityCensusFor(repoRoot, nil)
	paths := census.Paths

	workerFilter := strings.TrimSpace(*worker)
	sessionFilter := strings.TrimSpace(*session)
	if workerFilter != "" || sessionFilter != "" {
		filtered := make([]string, 0, len(paths))
		for _, p := range paths {
			if workerFilter != "" && !worktreeMatchesWorker(p, workerFilter) {
				continue
			}
			if sessionFilter != "" && !worktreeMatchesSession(p, sessionFilter) {
				continue
			}
			filtered = append(filtered, p)
		}
		paths = filtered
	}

	var ctx context.Context
	var cancel context.CancelFunc
	if *timeout > 0 {
		ctx, cancel = context.WithTimeout(context.Background(), *timeout)
		defer cancel()
	} else {
		ctx = context.Background()
	}

	if *asJSON {
		rows, timedOut := worktreeWorkerLifecycleInventoryContext(ctx, repoRoot, paths, worktreeWorkerListProbes)
		sortedPaths := make([]string, len(rows))
		for i := range rows {
			sortedPaths[i] = rows[i].Path
		}
		capacity := worktreeWorkerCapacityAdvisory(repoRoot, census, len(rows), *capacityReason, rows)
		worktreeWorkerWriteCapacityHuman(os.Stderr, capacity)
		local := worktreeWorkerLifecycleOut{
			Schema:    worktreeWorkerLifecycleSchema,
			Count:     len(rows),
			Paths:     sortedPaths,
			Inventory: rows,
			Capacity:  capacity,
			Partial:   timedOut,
			Timeout:   timedOut,
		}
		if strings.TrimSpace(*remote) == "" {
			worktreeWorkerEmit(local)
			return
		}
		hostname, hostErr := os.Hostname()
		localHost := workerworktree.SnapshotHostID(hostname)
		if hostErr != nil || strings.TrimSpace(hostname) == "" {
			localHost = "host-unknown"
		}
		out := worktreeWorkerRemoteListOut{
			Schema: "fak-worker-cross-host-lifecycle/1",
			Local:  worktreeWorkerLocalHostGroup{Host: localHost, Provenance: "LOCAL_LIVE", Freshness: workerworktree.SnapshotFresh, ObservedAt: time.Now().UTC(), Authoritative: true, Lifecycle: local},
			Remote: *remote, Hosts: []workerworktree.RemoteSnapshotGroup{},
		}
		if *fetch {
			if err := workerworktree.FetchRemoteSnapshots(repoRoot, *remote, nil); err != nil {
				out.Warning = err.Error()
				worktreeWorkerEmit(out)
				return
			}
			out.Fetched = true
		}
		hosts, err := workerworktree.ListRemoteSnapshots(repoRoot, *remote, time.Now(), nil)
		if err != nil {
			out.Warning = err.Error()
		} else {
			out.Hosts = hosts
		}
		worktreeWorkerEmit(out)
		return
	}

	rows := []workerworktree.InventoryRow{}
	var timedOut bool
	if census.Known {
		var err error
		rows, timedOut, err = workerworktree.InventoryForPathsContext(ctx, repoRoot, paths, nil)
		if err != nil && !timedOut {
			census.Known = false
			rows = []workerworktree.InventoryRow{}
		}
	}
	if rows == nil {
		rows = []workerworktree.InventoryRow{}
	}
	outPaths := make([]string, len(rows))
	for i := range rows {
		outPaths[i] = rows[i].Path
	}
	capacity := worktreeWorkerCapacityAdvisory(repoRoot, census, len(rows), *capacityReason, nil)
	worktreeWorkerWriteCapacityHuman(os.Stderr, capacity)
	worktreeWorkerEmit(worktreeWorkerListOut{
		Count:     len(rows),
		Paths:     outPaths,
		Inventory: rows,
		Capacity:  capacity,
		Partial:   timedOut,
		Timeout:   timedOut,
	})
}
