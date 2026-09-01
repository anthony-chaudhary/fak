package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/storagepressure"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type storagePressureDeps struct {
	now         func() time.Time
	repoRoot    func() string
	diskInfo    func(string) (int64, int64, bool)
	leaseOracle func(string, time.Time) workerworktree.LeaseLiveFn
	worktrees   func(string, workerworktree.GitRunner, time.Time, time.Duration, workerworktree.LeaseLiveFn, workerworktree.ColdReapOptions) []workerworktree.ColdWorktree
	goTmp       func(treedoctor.GoTmpOptions, bool) treedoctor.GoTmpReport
	goCache     func(treedoctor.GoCacheOptions, bool) treedoctor.GoCacheReport
	lookupEnv   func(string) string
	userCache   func() (string, error)
}

func defaultStoragePressureDeps() storagePressureDeps {
	return storagePressureDeps{
		now:         time.Now,
		repoRoot:    discoverRepoRoot,
		diskInfo:    compute.DiskInfo,
		leaseOracle: worktreeLiveLeaseOracle,
		worktrees:   workerworktree.ColdReapListWithOptions,
		goTmp:       treedoctor.SweepGoTmp,
		goCache:     treedoctor.SweepGoCache,
		lookupEnv:   os.Getenv,
		userCache:   os.UserCacheDir,
	}
}

func cmdStoragePressure(argv []string) {
	os.Exit(runStoragePressure(os.Stdout, os.Stderr, argv))
}

func runStoragePressure(stdout, stderr io.Writer, argv []string) int {
	return runStoragePressureWith(stdout, stderr, argv, defaultStoragePressureDeps())
}

func runStoragePressureWith(stdout, stderr io.Writer, argv []string, deps storagePressureDeps) int {
	fs := flag.NewFlagSet("storage-pressure", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "storage-pressure")
	asJSON := fs.Bool("json", false, "emit the read-only storage pressure report as JSON")
	root := fs.String("root", "", "repo root and filesystem path to inspect (default: discover from cwd)")
	warningFree := fs.Int64("warning-free-bytes", storagepressure.DefaultWarningFreeBytes, "set warning=true when filesystem free bytes are at or below this threshold (0 disables)")
	refuseFree := fs.Int64("refuse-free-bytes", storagepressure.DefaultRefuseFreeBytes, "set refuse=true when filesystem free bytes are at or below this threshold (0 disables)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "storage-pressure: positional arguments are not supported")
		return 2
	}
	if !*asJSON {
		fmt.Fprintln(stderr, "storage-pressure: --json is required")
		return 2
	}
	if *warningFree < 0 || *refuseFree < 0 {
		fmt.Fprintln(stderr, "storage-pressure: filesystem thresholds must be non-negative byte counts")
		return 2
	}

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = strings.TrimSpace(deps.repoRoot())
	}
	if repoRoot == "" {
		fmt.Fprintln(stderr, "storage-pressure: not in a git repository; pass --root")
		return 1
	}
	if abs, err := filepath.Abs(repoRoot); err == nil {
		repoRoot = abs
	}

	now := deps.now()
	total, free, known := deps.diskInfo(repoRoot)
	filesystem := storagepressure.Filesystem{
		Path: repoRoot, TotalBytes: total, FreeBytes: free, Known: known,
		WarningFreeBytes: *warningFree, RefuseFreeBytes: *refuseFree,
	}

	worktrees := deps.worktrees(
		repoRoot,
		nil,
		now,
		workerworktree.DefaultColdAgeFloor,
		deps.leaseOracle(repoRoot, now),
		workerworktree.ColdReapOptions{Concurrency: worktreeColdStatusConcurrency},
	)

	goTmpRoot := treedoctor.GoTmpRootFromEnv(deps.lookupEnv)
	if goTmpRoot == "" {
		goTmpRoot = filepath.Join(repoRoot, "_scratch", "go-tmp")
	}
	goTmp := deps.goTmp(treedoctor.GoTmpOptions{
		Root: goTmpRoot, RepoRoot: repoRoot, Now: now,
	}, false)

	goCacheRoot := treedoctor.GoCacheRootFromEnv(deps.lookupEnv, deps.userCache)
	goCache := deps.goCache(treedoctor.GoCacheOptions{
		Root: goCacheRoot, Now: now,
	}, false)

	report := storagepressure.New(
		now,
		filesystem,
		storagepressure.Worktrees(worktrees),
		storagepressure.GoTmp(goTmp),
		storagepressure.GoCache(goCache),
	)
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(stderr, "storage-pressure: encode JSON: %v\n", err)
		return 1
	}
	return 0
}
