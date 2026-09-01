package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/storagepressure"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestRunStoragePressureAggregatesDryRunOwnerReceipts(t *testing.T) {
	root := t.TempDir()
	now := time.Unix(123, 0)
	var worktreeRoot, diskPath string
	var leaseRoot string
	var goTmpApply, goCacheApply bool
	deps := storagePressureDeps{
		now:      func() time.Time { return now },
		repoRoot: func() string { return root },
		diskInfo: func(path string) (int64, int64, bool) {
			diskPath = path
			return 1000, 80, true
		},
		leaseOracle: func(gotRoot string, gotNow time.Time) workerworktree.LeaseLiveFn {
			leaseRoot = gotRoot
			if !gotNow.Equal(now) {
				t.Fatalf("lease oracle time = %v, want %v", gotNow, now)
			}
			return func(string) bool { return false }
		},
		worktrees: func(gotRoot string, _ workerworktree.GitRunner, gotNow time.Time, ageFloor time.Duration, oracle workerworktree.LeaseLiveFn, _ workerworktree.ColdReapOptions) []workerworktree.ColdWorktree {
			worktreeRoot = gotRoot
			if !gotNow.Equal(now) {
				t.Fatalf("worktree census time = %v, want %v", gotNow, now)
			}
			if ageFloor != workerworktree.DefaultColdAgeFloor {
				t.Fatalf("worktree age floor = %v, want %v", ageFloor, workerworktree.DefaultColdAgeFloor)
			}
			if oracle == nil || oracle("/candidate") {
				t.Fatal("worktree census did not receive the command-layer lease oracle")
			}
			return []workerworktree.ColdWorktree{{Path: "/cold", Eligible: true, ReclaimBytes: 100, ReclaimBytesKnown: true}}
		},
		goTmp: func(opts treedoctor.GoTmpOptions, apply bool) treedoctor.GoTmpReport {
			goTmpApply = apply
			if opts.Root != filepath.Join(root, "_scratch", "go-tmp") || opts.RepoRoot != root || !opts.Now.Equal(now) {
				t.Fatalf("GOTMP options = %+v", opts)
			}
			return treedoctor.GoTmpReport{TotalBytes: 70, Entries: []treedoctor.GoTmpEntry{{Path: "/tmp/old", Bytes: 40, Verdict: treedoctor.GoTmpReap}}}
		},
		goCache: func(opts treedoctor.GoCacheOptions, apply bool) treedoctor.GoCacheReport {
			goCacheApply = apply
			if opts.Root != "/cache/go-build" || !opts.Now.Equal(now) {
				t.Fatalf("GOCACHE options = root %q now %v", opts.Root, opts.Now)
			}
			return treedoctor.GoCacheReport{BytesBefore: 300, CandidateBytes: 60, ScanComplete: true}
		},
		lookupEnv: func(string) string { return "" },
		userCache: func() (string, error) { return "/cache", nil },
	}

	var stdout, stderr bytes.Buffer
	code := runStoragePressureWith(&stdout, &stderr, []string{"--json", "--warning-free-bytes=100", "--refuse-free-bytes=50"}, deps)
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, stderr.String())
	}
	if goTmpApply || goCacheApply {
		t.Fatalf("owner scans applied mutation: GOTMP=%v GOCACHE=%v", goTmpApply, goCacheApply)
	}
	if diskPath != root || worktreeRoot != root || leaseRoot != root {
		t.Fatalf("roots = disk %q worktrees %q lease %q, want %q", diskPath, worktreeRoot, leaseRoot, root)
	}
	var report storagepressure.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if got, want := report.ObservedBytes, int64(470); got != want {
		t.Fatalf("observed bytes = %d, want %d", got, want)
	}
	if got, want := report.ReclaimableBytes, int64(200); got != want {
		t.Fatalf("reclaimable bytes = %d, want %d", got, want)
	}
	if !report.ReclaimableBytesComplete {
		t.Fatal("reclaimable total should be complete for complete owner receipts")
	}
	if !report.Filesystem.Warning || report.Filesystem.Refuse {
		t.Fatalf("threshold verdict = warning %v refuse %v, want true/false", report.Filesystem.Warning, report.Filesystem.Refuse)
	}
}

func TestRunStoragePressureRequiresJSONAndExposesNoApplyFlag(t *testing.T) {
	deps := storagePressureDeps{}
	for _, args := range [][]string{{}, {"--apply", "--json"}} {
		var stdout, stderr bytes.Buffer
		if code := runStoragePressureWith(&stdout, &stderr, args, deps); code != 2 {
			t.Fatalf("args %v exit = %d, want 2; stderr = %q", args, code, stderr.String())
		}
	}
}
