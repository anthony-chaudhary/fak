package main

import (
	"flag"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func worktreeWorkerList(argv []string) {
	fs := flag.NewFlagSet("worktree worker list", flag.ExitOnError)
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit stable typed association, liveness, cleanliness, lifecycle, and reap-readiness rows")
	capacityReason := fs.String("capacity-reason", "", "record why capacity above the advisory setpoint is retained (advisory evidence)")
	remote := fs.String("remote", "", "explicit remote whose scrubbed host snapshots are included")
	fetch := fs.Bool("fetch", false, "refresh the remote snapshot mirror before listing")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	census := workerworktree.CapacityCensusFor(repoRoot, nil)
	paths := census.Paths
	if *asJSON {
		rows := worktreeWorkerLifecycleInventory(repoRoot, paths, worktreeWorkerLifecycleProbes{})
		sortedPaths := make([]string, len(rows))
		for i := range rows {
			sortedPaths[i] = rows[i].Path
		}
		capacity := worktreeWorkerCapacityAdvisory(repoRoot, census, len(rows), *capacityReason, rows)
		worktreeWorkerWriteCapacityHuman(os.Stderr, capacity)
		local := worktreeWorkerLifecycleOut{Schema: worktreeWorkerLifecycleSchema, Count: len(rows), Paths: sortedPaths, Inventory: rows, Capacity: capacity}
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
	if census.Known {
		var err error
		rows, err = workerworktree.Inventory(repoRoot, nil)
		if err != nil {
			census.Known = false
			rows = []workerworktree.InventoryRow{}
		}
	}
	if rows == nil {
		rows = []workerworktree.InventoryRow{}
	}
	capacity := worktreeWorkerCapacityAdvisory(repoRoot, census, len(paths), *capacityReason, nil)
	worktreeWorkerWriteCapacityHuman(os.Stderr, capacity)
	worktreeWorkerEmit(worktreeWorkerListOut{Count: len(paths), Paths: paths, Inventory: rows, Capacity: capacity})
}
