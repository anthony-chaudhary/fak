package main

import (
	"flag"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type worktreeWorkerLocalHostGroup struct {
	Host          string                           `json:"host"`
	Provenance    string                           `json:"provenance"`
	Freshness     workerworktree.SnapshotFreshness `json:"freshness"`
	ObservedAt    time.Time                        `json:"observed_at"`
	Authoritative bool                             `json:"authoritative"`
	Lifecycle     worktreeWorkerLifecycleOut       `json:"lifecycle"`
}

type worktreeWorkerRemoteListOut struct {
	Schema  string                               `json:"schema"`
	Local   worktreeWorkerLocalHostGroup         `json:"local"`
	Remote  string                               `json:"remote"`
	Fetched bool                                 `json:"fetched"`
	Hosts   []workerworktree.RemoteSnapshotGroup `json:"hosts"`
	Warning string                               `json:"warning,omitempty"`
}

func worktreeWorkerSnapshotRows(rows []worktreeWorkerLifecycleRow) []workerworktree.SnapshotRow {
	out := make([]workerworktree.SnapshotRow, 0, len(rows))
	for _, row := range rows {
		out = append(out, workerworktree.SnapshotRow{
			HeadSHA: row.HeadSHA, BaseSHA: row.BaseSHA,
			Association: workerworktree.SnapshotAssociation{State: string(row.Association.State), Lane: row.Association.Lane, LeaseID: row.Association.LeaseID},
			Liveness:    workerworktree.SnapshotLiveness{Owner: string(row.Liveness.Owner), Lease: string(row.Liveness.Lease)},
			Cleanliness: workerworktree.SnapshotCleanliness{State: string(row.Cleanliness.State)},
			Lifecycle:   string(row.Lifecycle),
		})
	}
	return out
}

func worktreeWorkerPublish(argv []string) {
	fs := flag.NewFlagSet("worktree worker publish", flag.ExitOnError)
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	remote := fs.String("remote", "", "explicit remote receiving this host snapshot")
	dryRun := fs.Bool("dry-run", false, "render and check the publication without writing the remote ref")
	apply := fs.Bool("apply", false, "compare-and-swap publish and read back the host ref")
	fs.Parse(argv)
	if *remote == "" || *dryRun == *apply {
		worktreeWorkerEmit(workerworktree.SnapshotPublishResult{Remote: *remote, Reason: "require --remote and exactly one of --dry-run or --apply"})
		return
	}
	repoRoot := worktreeWorkerRoot(*root)
	census := workerworktree.CapacityCensusFor(repoRoot, nil)
	if !census.Known {
		worktreeWorkerEmit(workerworktree.SnapshotPublishResult{Remote: *remote, Reason: "local lifecycle inventory unavailable; ordinary worker operations remain unaffected"})
		return
	}
	rows := worktreeWorkerLifecycleInventory(repoRoot, census.Paths, worktreeWorkerLifecycleProbes{})
	host, err := os.Hostname()
	if err != nil || strings.TrimSpace(host) == "" {
		worktreeWorkerEmit(workerworktree.SnapshotPublishResult{Remote: *remote, Reason: "hostname unavailable"})
		return
	}
	snapshot, err := workerworktree.NewRemoteSnapshot(host, time.Now(), worktreeWorkerSnapshotRows(rows))
	if err != nil {
		worktreeWorkerEmit(workerworktree.SnapshotPublishResult{Remote: *remote, Reason: err.Error()})
		return
	}
	worktreeWorkerEmit(workerworktree.PublishRemoteSnapshot(repoRoot, *remote, snapshot, *apply, nil))
}

func worktreeWorkerGoBuildVerify(wtPath string) (bool, string) {
	if _, err := exec.LookPath("go"); err != nil {
		return true, "go toolchain not found — skipping build verify (fail open)"
	}
	env, err := workerworktree.EnsureBuildDirs(wtPath)
	if err != nil {
		return false, "prepare isolated Go build directories: " + err.Error()
	}
	cmd := windowgate.Command("go", "build", "./...")
	cmd.Dir = wtPath
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Env = append(os.Environ(), "GOCACHE="+env["GOCACHE"], "GOTMPDIR="+env["GOTMPDIR"])
	out, err := cmd.CombinedOutput()
	if err == nil {
		return true, ""
	}
	detail := strings.TrimSpace(string(out))
	if len(detail) > 500 {
		detail = detail[len(detail)-500:]
	}
	return false, "go build ./... failed: " + detail
}
