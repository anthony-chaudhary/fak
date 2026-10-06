package main

import (
	"flag"
	"os"
	"os/exec"
	"path/filepath"
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
	cmd := windowgate.Command("go", worktreeWorkerGoBuildVerifyArgs()...)
	cmd.Dir = wtPath
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Env = worktreeWorkerGoBuildVerifyEnv(os.Environ(), env, os.UserCacheDir, worktreeWorkerSharedGoCache, worktreeWorkerPathInCheckout)
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

// worktreeWorkerGoBuildVerifyArgs is the verification-only compile check. Land
// candidates live at disposable, randomly named roots, and cmd/go folds a
// main-module package's directory into its action ID unless -trimpath is set, so
// without it no verify could reuse another verify's compiles. -buildvcs=false
// skips VCS stamping of a detached, possibly patched checkout. Same contract as
// buildCheckArgs (internal/devcmd) and validateGoCheckArgs (internal/validate).
func worktreeWorkerGoBuildVerifyArgs() []string {
	return []string{"build", "-trimpath", "-buildvcs=false", "./..."}
}

// worktreeWorkerGoBuildVerifyEnv keeps GOTMPDIR inside the worktree but leaves
// GOCACHE on the caller's shared cache. Go's cache is content-addressed and safe
// for concurrent processes, and a failed compile records no output, so one
// candidate's broken build cannot red another's. A private per-candidate cache
// only made every land verify a cold rebuild of the standard library and the
// whole module graph, which was most of the host's compile load. The isolated
// cache remains the fallback when there is no usable shared cache: GOCACHE set
// to off or a relative path, or unset with no user cache directory.
//
// An absolute GOCACHE inside a checkout is a worker or lane worktree's private
// cache (WorktreeEnv's <wt>/.gocache) inherited by the land process; it starts
// cold for every lane, so the verify uses the shared cache instead and keeps the
// inherited one only when no shared cache resolves.
func worktreeWorkerGoBuildVerifyEnv(base []string, isolated map[string]string, userCacheDir func() (string, error), sharedCache func(base []string) (string, bool), inCheckout func(path string) bool) []string {
	env := append(append([]string(nil), base...), "GOTMPDIR="+isolated["GOTMPDIR"])
	cache, set := goBuildVerifyEnvValue(base, "GOCACHE")
	shared := set && cache != "off" && filepath.IsAbs(cache)
	if !set || cache == "" {
		_, err := userCacheDir()
		shared = err == nil
	}
	if shared && set && cache != "" && inCheckout(cache) {
		if dir, ok := sharedCache(base); ok {
			return append(env, "GOCACHE="+dir)
		}
	}
	if shared {
		return env
	}
	return append(env, "GOCACHE="+isolated["GOCACHE"])
}

// worktreeWorkerSharedGoCache resolves the cache every land verify should
// share: an absolute FAK_SHARED_GOCACHE, else the user's default `go env
// GOCACHE` computed without the inherited per-lane GOCACHE.
func worktreeWorkerSharedGoCache(base []string) (string, bool) {
	if dir, ok := goBuildVerifyEnvValue(base, workerworktree.SharedGoCacheEnv); ok {
		if dir = strings.TrimSpace(dir); filepath.IsAbs(dir) {
			return dir, true
		}
	}
	env := make([]string, 0, len(base))
	for _, kv := range base {
		if name, _, ok := strings.Cut(kv, "="); ok && strings.EqualFold(name, "GOCACHE") {
			continue
		}
		env = append(env, kv)
	}
	cmd := windowgate.Command("go", "env", "GOCACHE")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Env = env
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	dir := strings.TrimSpace(string(out))
	if dir == "" || dir == "off" || !filepath.IsAbs(dir) {
		return "", false
	}
	return dir, true
}

// worktreeWorkerPathInCheckout reports whether path lies inside a git working
// tree: some ancestor directory holds a .git file (a linked worktree) or
// directory.
func worktreeWorkerPathInCheckout(path string) bool {
	dir := filepath.Clean(path)
	for {
		parent := filepath.Dir(dir)
		if parent == dir {
			return false
		}
		dir = parent
		if _, err := os.Lstat(filepath.Join(dir, ".git")); err == nil {
			return true
		}
	}
}

// goBuildVerifyEnvValue returns the last value of key in env, matching the key
// case-insensitively the way os/exec deduplicates a Windows environment.
func goBuildVerifyEnvValue(env []string, key string) (string, bool) {
	value, found := "", false
	for _, kv := range env {
		name, v, ok := strings.Cut(kv, "=")
		if ok && strings.EqualFold(name, key) {
			value, found = v, true
		}
	}
	return value, found
}
