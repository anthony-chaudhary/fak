package main

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/selfupdate"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

const selfUpdateProbeHelperEnv = "GO_WANT_SELFUPDATE_PROBE_HELPER"
const selfUpdateProbeHelperRev = "1234567890abcdef1234567890abcdef12345678"

func TestSelfUpdateAttemptPinsSelectionAcrossMovingOrigin(t *testing.T) {
	const (
		selected = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		advanced = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	)
	repoRoot := t.TempDir()
	buildDir := filepath.Join(t.TempDir(), "build")
	liveRef, addedRef, fetches := advanced, "", 0
	runner := func(_ context.Context, _, name string, args ...string) (string, bool) {
		if name != "git" {
			return "unexpected command", false
		}
		switch {
		case len(args) >= 1 && args[0] == "fetch":
			fetches++
			return "", true
		case len(args) >= 2 && args[0] == "worktree" && args[1] == "add":
			if want := []string{"--detach", buildDir, selected}; !slices.Equal(args[2:], want) {
				return "unexpected worktree add arguments", false
			}
			addedRef = args[4]
			if err := os.MkdirAll(args[3], 0o755); err != nil {
				return err.Error(), false
			}
			return "", true
		case len(args) >= 2 && args[0] == "worktree" && args[1] == "remove":
			_ = os.RemoveAll(args[len(args)-1])
			return "", true
		case len(args) >= 2 && args[0] == "worktree" && args[1] == "prune":
			return "", true
		default:
			return "unexpected git command", false
		}
	}

	dir, cleanup, err := prepareSelfUpdateAttempt(context.Background(), runner, repoRoot, selected, buildDir)
	if err != nil {
		t.Fatalf("prepare selected attempt: %v", err)
	}
	defer cleanup()
	if dir != buildDir || liveRef != advanced || addedRef != selected || fetches != 0 {
		t.Fatalf("dir=%q live origin=%s worktree ref=%s fetches=%d, want dir=%q live origin B=%s immutable A=%s with no second fetch", dir, liveRef, addedRef, fetches, buildDir, advanced, selected)
	}
	opts := selfUpdateAttemptOptions(buildDir, filepath.Join(repoRoot, "fak.exe"), selected)
	if opts.ExpectedCommit != selected {
		t.Fatalf("install expected commit=%q, want selected attempt %q", opts.ExpectedCommit, selected)
	}
	// Keep the compile-time contract explicit: the production options are the gated
	// selfinstall options whose source/provenance mismatch refuses before build or swap.
	var _ selfinstall.Options = opts
}

func TestSelfUpdateAttemptOptionsUsesCloneSharedCandidateCache(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is required for clone-shared candidate-cache acceptance")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	seed := filepath.Join(root, "seed")
	cloneA := filepath.Join(root, "clone-a")
	cloneB := filepath.Join(root, "clone-b")
	mustSelfUpdateGit(t, ctx, root, "init", "--bare", remote)
	mustSelfUpdateGit(t, ctx, root, "init", seed)
	mustSelfUpdateGit(t, ctx, seed, "config", "user.email", "fak-test@example.invalid")
	mustSelfUpdateGit(t, ctx, seed, "config", "user.name", "FAK Test")
	mustSelfUpdateGit(t, ctx, seed, "checkout", "-b", "main")
	if err := os.WriteFile(filepath.Join(seed, "tracked.txt"), []byte("candidate cache\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustSelfUpdateGit(t, ctx, seed, "add", "tracked.txt")
	mustSelfUpdateGit(t, ctx, seed, "commit", "-m", "seed")
	mustSelfUpdateGit(t, ctx, seed, "remote", "add", "origin", remote)
	mustSelfUpdateGit(t, ctx, seed, "push", "-u", "origin", "main")
	mustSelfUpdateGit(t, ctx, root, "clone", "--branch", "main", remote, cloneA)
	mustSelfUpdateGit(t, ctx, root, "clone", "--branch", "main", remote, cloneB)

	buildA := filepath.Join(root, "build-a")
	buildASibling := filepath.Join(root, "build-a-sibling")
	buildB := filepath.Join(root, "build-b")
	mustSelfUpdateGit(t, ctx, cloneA, "worktree", "add", "--detach", buildA, "HEAD")
	mustSelfUpdateGit(t, ctx, cloneA, "worktree", "add", "--detach", buildASibling, "HEAD")
	mustSelfUpdateGit(t, ctx, cloneB, "worktree", "add", "--detach", buildB, "HEAD")

	const selected = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	cacheA := selfUpdateAttemptOptions(buildA, filepath.Join(root, "fak-a"), selected).CacheDir
	cacheASibling := selfUpdateAttemptOptions(buildASibling, filepath.Join(root, "fak-a-sibling"), selected).CacheDir
	cacheB := selfUpdateAttemptOptions(buildB, filepath.Join(root, "fak-b"), selected).CacheDir
	wantA := selfinstall.CandidateCacheDir(discoverGitCommonDir(buildA))
	if cacheA == "" || cacheA != wantA {
		t.Fatalf("production candidate cache = %q, want clone-shared %q", cacheA, wantA)
	}
	if cacheASibling != cacheA {
		t.Fatalf("same-clone linked worktrees resolved different candidate caches: %q / %q", cacheA, cacheASibling)
	}
	if cacheB == "" || cacheB == cacheA {
		t.Fatalf("independent clones must not share candidate cache: clone A=%q clone B=%q", cacheA, cacheB)
	}
	if got := selfUpdateAttemptOptions(filepath.Join(root, "not-a-worktree"), filepath.Join(root, "fak-miss"), selected).CacheDir; got != "" {
		t.Fatalf("unresolved Git common directory enabled candidate cache at %q", got)
	}
}

func TestSelfUpdateAttemptRejectsUnpinnedSelection(t *testing.T) {
	called := false
	runner := func(context.Context, string, string, ...string) (string, bool) {
		called = true
		return "", true
	}
	if _, _, err := prepareSelfUpdateAttempt(context.Background(), runner, t.TempDir(), "", filepath.Join(t.TempDir(), "build")); err == nil {
		t.Fatal("empty selection fell back to mutable origin/main")
	}
	if called {
		t.Fatal("invalid selection reached Git instead of failing closed")
	}
}

func TestSelfUpdateGateFailureReceiptCarriesActionableCacheDetail(t *testing.T) {
	detail := "vet: Go build cache /tmp/fak-cache became unavailable after the one bounded recovery; stop concurrent cache cleanup and rerun fak self-update"
	receipt := newSelfUpdateReceipt(outcomeGateFailed, filepath.Join("bin", "fak"), detail)
	if receipt.Status != "gate_failed" || receipt.Detail != detail || receipt.NextCommand != "fak self-update" {
		t.Fatalf("gate receipt=%+v, want typed failure with preserved cache remediation", receipt)
	}
}

func TestSelfUpdateAttemptRealGitPinsSelectionAcrossMovingOrigin(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is required for moving-ref acceptance")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	root := t.TempDir()
	remote, seed, clone := filepath.Join(root, "remote.git"), filepath.Join(root, "seed"), filepath.Join(root, "clone")
	mustSelfUpdateGit(t, ctx, root, "init", "--bare", remote)
	mustSelfUpdateGit(t, ctx, root, "init", seed)
	mustSelfUpdateGit(t, ctx, seed, "config", "user.email", "fak-test@example.invalid")
	mustSelfUpdateGit(t, ctx, seed, "config", "user.name", "FAK Test")
	mustSelfUpdateGit(t, ctx, seed, "checkout", "-b", "main")
	tracked := filepath.Join(seed, "tracked.txt")
	if err := os.WriteFile(tracked, []byte("A\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustSelfUpdateGit(t, ctx, seed, "add", "tracked.txt")
	mustSelfUpdateGit(t, ctx, seed, "commit", "-m", "A")
	mustSelfUpdateGit(t, ctx, seed, "remote", "add", "origin", remote)
	mustSelfUpdateGit(t, ctx, seed, "push", "-u", "origin", "main")
	mustSelfUpdateGit(t, ctx, root, "clone", "--branch", "main", remote, clone)
	selected := mustSelfUpdateGit(t, ctx, clone, "rev-parse", "origin/main")

	if err := os.WriteFile(tracked, []byte("B\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustSelfUpdateGit(t, ctx, seed, "add", "tracked.txt")
	mustSelfUpdateGit(t, ctx, seed, "commit", "-m", "B")
	mustSelfUpdateGit(t, ctx, seed, "push", "origin", "main")
	advanced := mustSelfUpdateGit(t, ctx, seed, "rev-parse", "HEAD")
	if selected == advanced {
		t.Fatal("fixture did not advance remote from A to B")
	}
	// Advance the clone's observation before the transaction. The pinned attempt must not
	// fetch again and must still materialize the already-selected A.
	mustSelfUpdateGit(t, ctx, clone, "fetch", "origin", "--quiet")

	buildDir := filepath.Join(root, "selected-worktree")
	dir, cleanup, err := prepareSelfUpdateAttempt(ctx, selfinstall.RealRunner, clone, selected, buildDir)
	if err != nil {
		t.Fatalf("prepare immutable A after remote advanced to B: %v", err)
	}
	defer cleanup()
	if got := mustSelfUpdateGit(t, ctx, clone, "rev-parse", "origin/main"); got != advanced {
		t.Fatalf("fixture fetch did not expose moving origin/main B before pinned prepare: got %s want %s", got, advanced)
	}
	if got := mustSelfUpdateGit(t, ctx, dir, "rev-parse", "HEAD"); got != selected {
		t.Fatalf("prepared worktree followed mutable ref: got %s want immutable A %s", got, selected)
	}
	if got := mustSelfUpdateGit(t, ctx, dir, "status", "--porcelain"); got != "" {
		t.Fatalf("prepared selected worktree is dirty: %q", got)
	}
}

func mustSelfUpdateGit(t *testing.T, ctx context.Context, dir string, args ...string) string {
	t.Helper()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s in %s: %v\n%s", strings.Join(args, " "), dir, err, out)
	}
	return strings.TrimSpace(string(out))
}

func init() {
	if os.Getenv(selfUpdateProbeHelperEnv) != "1" {
		return
	}
	_, _ = os.Stdout.WriteString("fak test helper\nbuild: " + selfUpdateProbeHelperRev + "\n")
	os.Exit(0)
}

func resetSelfUpdateProgressForTest() {
	selfUpdateProgressState.Lock()
	selfUpdateProgressState.percent = 0
	selfUpdateProgressState.operation = ""
	selfUpdateProgressState.Unlock()
}

func resetSelfUpdateTimingForTest() {
	selfUpdateTimingState.Lock()
	selfUpdateTimingState.initialized = false
	selfUpdateTimingState.finished = false
	selfUpdateTimingState.started = time.Time{}
	selfUpdateTimingState.phaseStarted = time.Time{}
	selfUpdateTimingState.active = ""
	selfUpdateTimingState.elapsed = nil
	selfUpdateTimingState.snapshot = selfUpdateTimingSnapshot{}
	selfUpdateTimingState.Unlock()
}

func TestSelfUpdateProgressIsMonotonicAndCompletesOnlyAtOutcome(t *testing.T) {
	oldProgress := selfUpdateProgress
	var stderr strings.Builder
	selfUpdateProgress = &stderr
	resetSelfUpdateProgressForTest()
	t.Cleanup(func() {
		selfUpdateProgress = oldProgress
		resetSelfUpdateProgressForTest()
	})

	reportSelfUpdateProgress(10, "checking provenance")
	reportSelfUpdateProgress(45, "building candidate")
	reportSelfUpdateProgress(30, "late stale update")  // regressions are ignored
	reportSelfUpdateProgress(100, "verifying install") // non-terminal work is capped
	finishSelfUpdateProgress(outcomeInstalled)

	want := "" +
		"self-update: progress=10% operation=\"checking provenance\"\n" +
		"self-update: progress=45% operation=\"building candidate\"\n" +
		"self-update: progress=99% operation=\"verifying install\"\n" +
		"self-update: progress=100% operation=\"terminal outcome: installed\"\n"
	if got := stderr.String(); got != want {
		t.Fatalf("captured progress mismatch\ngot:\n%s\nwant:\n%s", got, want)
	}
}

func TestSelfUpdateHeartbeatUsesDeterministicBoundedSeam(t *testing.T) {
	oldProgress, oldWait := selfUpdateProgress, selfUpdateHeartbeatWait
	var stderr strings.Builder
	selfUpdateProgress = &stderr
	resetSelfUpdateProgressForTest()
	ready := make(chan struct{})
	calls := 0
	selfUpdateHeartbeatWait = func(stop <-chan struct{}, interval time.Duration) bool {
		if interval != selfUpdateHeartbeatInterval {
			t.Errorf("heartbeat interval = %v, want %v", interval, selfUpdateHeartbeatInterval)
		}
		calls++
		if calls <= 2 {
			return true
		}
		close(ready) // two emitted ticks, then block until the operation ends
		<-stop
		return false
	}
	t.Cleanup(func() {
		selfUpdateProgress = oldProgress
		selfUpdateHeartbeatWait = oldWait
		resetSelfUpdateProgressForTest()
	})

	stop := startSelfUpdateHeartbeat(55, "building fak candidate")
	<-ready
	stop()

	want := "" +
		"self-update: progress=55% operation=\"building fak candidate\"\n" +
		"self-update: progress=55% operation=\"building fak candidate\" heartbeat=true\n" +
		"self-update: progress=55% operation=\"building fak candidate\" heartbeat=true\n"
	if got := stderr.String(); got != want {
		t.Fatalf("captured heartbeat mismatch\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// TestSelfUpdateShouldBuild pins the proceed decision, and in particular the case binstamp
// alone gets WRONG: a clean local binary that is AHEAD of origin/main. Under the old
// `verdict == binstamp.Stale` rule that case (rev differs => Stale) rebuilt origin/main OVER
// the newer binary; keying SELF mode off versionskew.Skewed makes Ahead a no-op. This is the
// "previously-collapsed case now drives a distinct decision" the wiring exists to produce.
func TestSelfUpdateShouldBuild(t *testing.T) {
	cases := []struct {
		name  string
		force bool
		fleet bool
		bin   binstamp.Freshness
		skew  versionskew.Verdict
		want  bool
	}{
		// SELF mode: ONLY a provably-behind skew rebuilds.
		{"self behind rebuilds", false, false, binstamp.Stale, versionskew.Skewed, true},
		{"self ahead does NOT rebuild (the fix)", false, false, binstamp.Stale, versionskew.Ahead, false},
		{"self diverged does NOT rebuild", false, false, binstamp.Stale, versionskew.Diverged, false},
		{"self fresh no-op", false, false, binstamp.Fresh, versionskew.Fresh, false},
		{"self dirty no-op", false, false, binstamp.Unknown, versionskew.Dirty, false},
		{"self unstamped no-op", false, false, binstamp.Unknown, versionskew.Unstamped, false},
		{"self unknown no-op", false, false, binstamp.Unknown, versionskew.Unknown, false},
		{"self force overrides a fresh binary", true, false, binstamp.Fresh, versionskew.Fresh, true},
		// FLEET mode: rebuild unless binstamp proves Fresh — regardless of the skew token.
		{"fleet not-fresh rebuilds", false, true, binstamp.Unknown, versionskew.Unknown, true},
		{"fleet behind rebuilds", false, true, binstamp.Stale, versionskew.Skewed, true},
		{"fleet fresh no-op", false, true, binstamp.Fresh, versionskew.Fresh, false},
		{"fleet fresh + force rebuilds", true, true, binstamp.Fresh, versionskew.Fresh, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := selfUpdateShouldBuild(c.force, c.fleet, c.bin, c.skew); got != c.want {
				t.Fatalf("selfUpdateShouldBuild(force=%v fleet=%v bin=%v skew=%v) = %v, want %v",
					c.force, c.fleet, c.bin, c.skew, got, c.want)
			}
		})
	}
}

// TestSelfUpdateSiblingsIncludesInTreeFleetBinary pins the fix for the stale-fleet-binary lag:
// `self-update --target X` converged X and nothing else, while every dispatcher-launched worker
// runs `<root>/tools/.bin/fak[.exe] guard -- claude …` — the path
// tools/dispatch_worker.py resolve_fak_bin prefers AHEAD of PATH. Because that in-tree file
// existed, PATH was never consulted, so the fleet ran a binary no updater targeted and the tick
// still exited 0. The sibling set must therefore contain the in-tree fleet binary.
func TestSelfUpdateSiblingsIncludesInTreeFleetBinary(t *testing.T) {
	root := t.TempDir()
	binDir := filepath.Join(root, "tools", ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fleetBin := filepath.Join(binDir, "fak"+exeSuffix())
	if err := os.WriteFile(fleetBin, []byte("stale"), 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "installed"+exeSuffix())
	if err := os.WriteFile(target, []byte("fresh"), 0o755); err != nil {
		t.Fatal(err)
	}

	got := selfUpdateSiblings(root, target)
	found := false
	for _, p := range got {
		if strings.EqualFold(p, fleetBin) {
			found = true
		}
		if strings.EqualFold(p, target) {
			t.Errorf("sibling set must not repeat the primary --target %q: %v", target, got)
		}
	}
	if !found {
		t.Errorf("selfUpdateSiblings(%q, %q) = %v; want it to include the in-tree fleet binary %q",
			root, target, got, fleetBin)
	}
}

// TestSelfUpdateSiblingsSkipsMissingPaths — we converge binaries that already exist; a path that
// is absent is not an install location we should create. With no tools/.bin on disk the only
// sibling is the running test binary itself, never a phantom <root>/tools/.bin entry.
func TestSelfUpdateSiblingsSkipsMissingPaths(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "installed"+exeSuffix())
	if err := os.WriteFile(target, []byte("fresh"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range selfUpdateSiblings(root, target) {
		if strings.Contains(strings.ToLower(p), filepath.Join("tools", ".bin")) {
			t.Errorf("selfUpdateSiblings returned a non-existent path %q", p)
		}
		if st, err := os.Stat(p); err != nil || st.IsDir() {
			t.Errorf("selfUpdateSiblings returned %q which is not an existing file", p)
		}
	}
}

func TestSelfUpdateFakDevNeedsConvergeTriggersWhenPrimaryIsCurrent(t *testing.T) {
	head := strings.Repeat("a", 40)
	probe := func(path string) (string, bool, bool) {
		if strings.Contains(path, "stale") {
			return strings.Repeat("b", 40), false, true
		}
		return head, false, true
	}
	if !selfUpdateFakDevNeedsConverge([]string{"fak-dev-stale"}, head, probe) {
		t.Fatal("stale fak-dev must force an update even when fak itself is current")
	}
	if selfUpdateFakDevNeedsConverge([]string{"fak-dev-current"}, head, probe) {
		t.Fatal("current fak-dev should not force a rebuild")
	}
}

// TestSelfUpdateFakDevTargetsFindsOnlyInstalledCompanions proves product-only hosts stay
// product-only while a side-by-side developer install joins the same convergence cycle.
func TestSelfUpdateFakDevTargetsFindsOnlyInstalledCompanions(t *testing.T) {
	root := t.TempDir()
	binDir := filepath.Join(root, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fakPath := filepath.Join(binDir, "fak"+exeSuffix())
	if err := os.WriteFile(fakPath, []byte("fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	devPath := filepath.Join(binDir, "fak-dev"+exeSuffix())
	for _, got := range selfUpdateFakDevTargets(root, fakPath) {
		if strings.EqualFold(got, devPath) {
			t.Fatalf("missing companion should not be created: %v", got)
		}
	}
	if err := os.WriteFile(devPath, []byte("stale"), 0o755); err != nil {
		t.Fatal(err)
	}
	got := selfUpdateFakDevTargets(root, fakPath)
	found := false
	for _, candidate := range got {
		if strings.EqualFold(candidate, devPath) {
			found = true
		}
	}
	if !found {
		t.Fatalf("targets=%v; want it to include %s", got, devPath)
	}
}

// TestSelfUpdateProbeReadsOwnPathAfterSwap reproduces the Windows post-swap audit bug. The
// running process still has its old embedded stamp, but invoking its path starts the new bytes.
// The census must read the deployed path or it reports a successful update as divergent.
func TestSelfUpdateProbeReadsOwnPathAfterSwap(t *testing.T) {
	t.Setenv(selfUpdateProbeHelperEnv, "1")
	revision, dirty, attested := selfUpdateProbe(os.Args[0])
	if !attested || dirty || revision != selfUpdateProbeHelperRev {
		t.Fatalf("selfUpdateProbe(own path) = (%q, dirty=%v, attested=%v), want deployed helper stamp %q",
			revision, dirty, attested, selfUpdateProbeHelperRev)
	}
}

// convergeSiblings — the old "is the INVOKER provably fresh?" guard on the sibling swap — is
// gone (#6508). It both over- and under-shot: it re-swapped siblings that were already current
// and never looked at the PATH / Go-bin copies at all. The decision is now per-copy, from the
// role census, and is pinned by TestNeedsConvergeDemandsProofOfFreshness in
// internal/selfinstall — a package whose tests actually build and run.

// TestSelfUpdateSkipOutcome pins the closed outcome vocabulary against the message switch it
// mirrors. The scheduler sees only an exit code, and rc=0 is identical for "installed",
// "already current", "busy" and "--check"; the named outcome is what re-couples the success
// code to whether an update actually happened.
func TestSelfUpdateSkipOutcome(t *testing.T) {
	cases := []struct {
		fleet bool
		skew  versionskew.Verdict
		want  selfUpdateOutcome
	}{
		{true, versionskew.Fresh, outcomeTargetCurrent},
		{true, versionskew.Unknown, outcomeTargetCurrent},
		{false, versionskew.Fresh, outcomeSelfFresh},
		{false, versionskew.Ahead, outcomeSelfAhead},
		{false, versionskew.Dirty, outcomeSelfLocal},
		{false, versionskew.Unstamped, outcomeSelfLocal},
		{false, versionskew.Diverged, outcomeSelfLocal},
		{false, versionskew.Unknown, outcomeSelfUnknown},
	}
	for _, c := range cases {
		if got := selfUpdateSkipOutcome(c.fleet, c.skew); got != c.want {
			t.Errorf("selfUpdateSkipOutcome(fleet=%v, %v) = %q; want %q", c.fleet, c.skew, got, c.want)
		}
	}
}

func TestSelfUpdateReceiptPostures(t *testing.T) {
	oldCorrelation := selfUpdateCorrelationID
	selfUpdateCorrelationID = func() string { return "corr-123" }
	defer func() { selfUpdateCorrelationID = oldCorrelation }()
	selfUpdateReceiptOldRevision = "oldrev"
	selfUpdateReceiptNewRevision = "newrev"
	selfUpdateReceiptTargets = []selfUpdateReceiptTarget{{Role: "primary", Path: filepath.Clean("bin/fak")}}

	cases := []struct {
		name    string
		outcome selfUpdateOutcome
		status  string
		restart bool
		roll    string
	}{
		{"current", outcomeSelfFresh, "current", false, "not_attempted"},
		{"updated", outcomeInstalled, "updated", false, "not_attempted"},
		{"rolled_back", outcomeRolledBack, "rolled_back", false, "succeeded"},
		{"rollback_failed", outcomeRollbackFailed, "rollback_failed", false, "failed"},
		{"busy", outcomeBusy, "busy", false, "not_attempted"},
		{"restart_required", selfUpdateOutcome("restart_required"), "restart_required", true, "not_attempted"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			receipt := newSelfUpdateReceipt(tc.outcome, "bin/fak", "rollback detail")
			encoded, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			var parsed map[string]any
			if err := json.Unmarshal(encoded, &parsed); err != nil {
				t.Fatalf("receipt is not JSON: %v\n%s", err, encoded)
			}
			if receipt.Schema != selfUpdateReceiptSchema || receipt.SchemaVersion != 1 || receipt.CorrelationID != "corr-123" {
				t.Fatalf("unstable envelope: %+v", receipt)
			}
			if receipt.Status != tc.status || receipt.RestartRequired != tc.restart || receipt.RollbackStatus != tc.roll {
				t.Fatalf("posture = %+v", receipt)
			}
			if receipt.OldRevision == nil || *receipt.OldRevision != "oldrev" || receipt.NewRevision == nil || *receipt.NewRevision != "newrev" {
				t.Fatalf("revision fields = %+v", receipt)
			}
			if receipt.NextCommand == "" || len(receipt.Targets) != 1 {
				t.Fatalf("action/targets missing: %+v", receipt)
			}
		})
	}
}

func TestSelfUpdateReceiptCarriesReusedBuildProvenance(t *testing.T) {
	selfUpdateReceiptBuildProvenance = &selfUpdateBuildProvenance{
		SourceCommit:         "89abcdef0123456789abcdef0123456789abcdef",
		ArtifactSourceCommit: "0123456789abcdef0123456789abcdef01234567",
		BuildInputDigest:     "sha256:inputs", BuildEnvelope: map[string]string{"GOVERSION": "go1.26.7"},
		ArtifactDigest: "artifact", ArtifactSize: 42, AppVersion: "1.2.3", Reused: true,
	}
	t.Cleanup(func() { selfUpdateReceiptBuildProvenance = nil })
	receipt := newSelfUpdateReceipt(outcomeInstalled, "bin/fak", "")
	if receipt.BuildProvenance == nil || !receipt.BuildProvenance.Reused ||
		receipt.BuildProvenance.SourceCommit == receipt.BuildProvenance.ArtifactSourceCommit ||
		receipt.BuildProvenance.BuildInputDigest == "" || receipt.BuildProvenance.ArtifactDigest == "" ||
		receipt.BuildProvenance.BuildEnvelope["GOVERSION"] == "" ||
		receipt.BuildProvenance.AppVersion != "1.2.3" {
		t.Fatalf("reused build provenance = %+v", receipt.BuildProvenance)
	}
}

func TestSelfUpdateReceiptCarriesArtifactTransfer(t *testing.T) {
	selfUpdateReceiptTransfer = &selfUpdateTransferReceipt{
		ChosenPath: "full", DeltaBytes: 17, FullBytes: 100, TotalMS: 45,
		Verification: "signed_full_size_sha256_verified", FallbackReason: "zstd_patch_failed",
		FallbackBytes: 100, FallbackMS: 20,
	}
	t.Cleanup(func() { selfUpdateReceiptTransfer = nil })
	receipt := newSelfUpdateReceipt(outcomeInstalled, "bin/fak", "")
	if receipt.Transfer == nil || receipt.Transfer.ChosenPath != "full" ||
		receipt.Transfer.DeltaBytes != 17 || receipt.Transfer.FullBytes != 100 ||
		receipt.Transfer.FallbackReason != "zstd_patch_failed" ||
		receipt.Transfer.FallbackBytes != 100 || receipt.Transfer.FallbackMS != 20 ||
		receipt.Transfer.TotalMS != 45 || receipt.Transfer.Verification != "signed_full_size_sha256_verified" {
		t.Fatalf("artifact transfer receipt = %+v", receipt.Transfer)
	}
}

func TestSelfUpdateMetadataOnlyReceiptDoesNotSignalBinaryUpdate(t *testing.T) {
	selfUpdateReceiptChanged = 0
	receipt := newSelfUpdateReceipt(outcomeMetadataOnly, "bin/fak", "selected-source metadata advanced")
	if receipt.Status != "current" || receipt.RestartRequired {
		t.Fatalf("metadata-only receipt = %+v", receipt)
	}
}

func TestUsableSelfUpdateArtifactFallsBackWhenCatalogHasNoCompleteTarget(t *testing.T) {
	target := &selfUpdateArtifactTarget{URL: "https://updates.example/fak"}
	selection := selfUpdateManifestSelection{Disposition: "update", Artifact: target}
	if got := usableSelfUpdateArtifact(selection, nil); got != target {
		t.Fatalf("usable signed target = %p, want %p", got, target)
	}
	if got := usableSelfUpdateArtifact(selfUpdateManifestSelection{Disposition: "update"}, nil); got != nil {
		t.Fatal("missing catalog target did not fall back to source build")
	}
	if got := usableSelfUpdateArtifact(selection, []string{"fak-dev"}); got != nil {
		t.Fatal("incomplete component catalog bypassed source-build fallback")
	}
}

func TestSelfUpdateCheckOnlyReceiptReportsStaleRevision(t *testing.T) {
	selfUpdateReceiptOldRevision = "oldrev"
	selfUpdateReceiptNewRevision = "newrev"
	t.Cleanup(func() {
		selfUpdateReceiptOldRevision = ""
		selfUpdateReceiptNewRevision = ""
	})

	receipt := newSelfUpdateReceipt(outcomeCheckOnly, "bin/fak", "")
	if receipt.Status != "stale" {
		t.Fatalf("check-only stale status = %q, want stale", receipt.Status)
	}
	if receipt.NextCommand != "fak self-update" {
		t.Fatalf("check-only stale next_command = %q, want fak self-update", receipt.NextCommand)
	}
	if receipt.Attempted != 0 || receipt.Changed != 0 {
		t.Fatalf("check-only receipt mutated targets: attempted=%d changed=%d", receipt.Attempted, receipt.Changed)
	}
}

func TestSelfUpdateCheckOnlyReceiptReportsCurrentRevision(t *testing.T) {
	selfUpdateReceiptOldRevision = "samerev"
	selfUpdateReceiptNewRevision = "samerev"
	t.Cleanup(func() {
		selfUpdateReceiptOldRevision = ""
		selfUpdateReceiptNewRevision = ""
	})

	receipt := newSelfUpdateReceipt(outcomeCheckOnly, "bin/fak", "")
	if receipt.Status != "current" || receipt.NextCommand != "fak version" {
		t.Fatalf("check-only current receipt = status %q next %q", receipt.Status, receipt.NextCommand)
	}
}

func TestSelfUpdateRepairableHotCopyReceiptIsActionable(t *testing.T) {
	receipt := newSelfUpdateReceipt(outcomeHotCopyDivergent, "bin/fak", "")
	if receipt.Status != "divergent" || receipt.NextCommand != "fak self-update" {
		t.Fatalf("divergent receipt = status %q next %q", receipt.Status, receipt.NextCommand)
	}
}

func TestEmitSelfUpdateCheckJSONPostures(t *testing.T) {
	oldCorrelation := selfUpdateCorrelationID
	oldProgress, oldJSON := selfUpdateProgress, selfUpdateJSON
	selfUpdateCorrelationID = func() string { return "corr-check" }
	t.Cleanup(func() {
		selfUpdateCorrelationID = oldCorrelation
		selfUpdateProgress, selfUpdateJSON = oldProgress, oldJSON
	})

	cases := []struct {
		name      string
		freshness binstamp.Freshness
		audit     selfinstall.AuditPartition
		status    selfupdate.CheckStatus
		next      string
	}{
		{"fresh and converged", binstamp.Fresh, selfinstall.AuditPartition{}, selfupdate.StatusCurrent, "fak version"},
		{"revision differs", binstamp.Stale, selfinstall.AuditPartition{}, selfupdate.StatusStale, "fak self-update"},
		{"revision differs and repairable copies drift", binstamp.Stale, selfinstall.Audit{Divergent: []selfinstall.Role{selfinstall.RoleWorker}}.Partition(), selfupdate.StatusStale, "fak self-update"},
		{"repairable hot copies drift", binstamp.Fresh, selfinstall.Audit{Divergent: []selfinstall.Role{selfinstall.RoleWorker}}.Partition(), selfupdate.StatusDivergent, "fak self-update"},
		{"audit-only gate needs attention", binstamp.Fresh, selfinstall.Audit{Dirty: []selfinstall.Role{selfinstall.RoleGate}}.Partition(), selfupdate.StatusAttention, "fak self-update --check"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout strings.Builder
			selfUpdateJSON = &stdout
			selfUpdateProgress = io.Discard
			selfUpdateReceiptOldRevision = "oldrev"
			selfUpdateReceiptNewRevision = "newrev"

			emitSelfUpdateCheckOutcome("bin/fak", "check-only", tc.freshness, tc.audit)

			var receipt selfUpdateReceipt
			if err := json.Unmarshal([]byte(stdout.String()), &receipt); err != nil {
				t.Fatalf("receipt is not JSON: %v: %q", err, stdout.String())
			}
			if receipt.Status != string(tc.status) || receipt.NextCommand != tc.next {
				t.Fatalf("status/next_command = %q/%q, want %q/%q", receipt.Status, receipt.NextCommand, tc.status, tc.next)
			}
			if receipt.Changed != 0 || receipt.Attempted != 0 {
				t.Fatalf("check receipt reports mutation: %+v", receipt)
			}
		})
	}
}

func TestSelfUpdateCheckJSONAuditOnlyHelperProcess(t *testing.T) {
	const helperEnv = "GO_WANT_SELFUPDATE_CHECK_JSON_HELPER"
	if os.Getenv(helperEnv) == "1" {
		selfUpdateProgress = io.Discard
		selfUpdateJSON = os.Stdout
		selfUpdateReceiptOldRevision = "abcdef012345"
		selfUpdateReceiptNewRevision = "abcdef012345"
		audit := selfinstall.Audit{Dirty: []selfinstall.Role{selfinstall.RoleGate}}.Partition()
		emitSelfUpdateCheckOutcome("bin/fak", "audit-only gate drift", binstamp.Fresh, audit)
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestSelfUpdateCheckJSONAuditOnlyHelperProcess$")
	cmd.Env = append(os.Environ(), helperEnv+"=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("self-update check helper failed: %v", err)
	}
	var receipt selfUpdateReceipt
	if err := json.Unmarshal(out, &receipt); err != nil {
		t.Fatalf("helper receipt is not JSON: %v: %q", err, out)
	}
	if receipt.Status != string(selfupdate.StatusAttention) {
		t.Fatalf("audit-only helper status = %q, want attention", receipt.Status)
	}
	if receipt.NextCommand != "fak self-update --check" || strings.Contains(receipt.NextCommand, "--force") {
		t.Fatalf("audit-only helper advertised impossible repair %q", receipt.NextCommand)
	}
}

func TestSelfUpdateCheckDoesNotFetchOrigin(t *testing.T) {
	var calls []string
	runner := func(_ context.Context, dir, name string, args ...string) (string, bool) {
		calls = append(calls, strings.Join(append([]string{dir, name}, args...), " "))
		return "", true
	}

	selfUpdateFetchOrigin(context.Background(), runner, "repo", true)
	if len(calls) != 0 {
		t.Fatalf("--check invoked mutating fetch: %v", calls)
	}

	selfUpdateFetchOrigin(context.Background(), runner, "repo", false)
	if len(calls) != 1 || calls[0] != "repo git fetch origin --quiet" {
		t.Fatalf("update fetch calls = %v", calls)
	}
}

func TestEmitSelfUpdateJSONIsOneObjectWithoutProse(t *testing.T) {
	oldCorrelation := selfUpdateCorrelationID
	oldProgress, oldJSON := selfUpdateProgress, selfUpdateJSON
	oldNow := selfUpdateTimingNow
	selfUpdateCorrelationID = func() string { return "corr-123" }
	base := time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC)
	times := []time.Time{
		base,
		base.Add(10 * time.Millisecond),
		base.Add(50 * time.Millisecond),
		base.Add(70 * time.Millisecond),
	}
	timeCall := 0
	selfUpdateTimingNow = func() time.Time {
		if timeCall >= len(times) {
			t.Fatalf("timing seam called %d times; want at most %d", timeCall+1, len(times))
		}
		now := times[timeCall]
		timeCall++
		return now
	}
	t.Cleanup(func() {
		selfUpdateCorrelationID = oldCorrelation
		selfUpdateProgress, selfUpdateJSON = oldProgress, oldJSON
		selfUpdateTimingNow = oldNow
		resetSelfUpdateProgressForTest()
		resetSelfUpdateTimingForTest()
	})

	var stdout, stderr strings.Builder
	selfUpdateJSON = &stdout
	selfUpdateProgress = &stderr
	resetSelfUpdateProgressForTest()
	beginSelfUpdateTiming()
	startSelfUpdatePhase(selfUpdatePhaseBuild)
	startSelfUpdatePhase(selfUpdatePhaseVet)
	reportSelfUpdateProgress(82, "installing verified binaries")
	emitSelfUpdateOutcome(outcomeBusy, "bin/fak", "single-flight lock held")
	if strings.Count(stdout.String(), "\n") != 1 {
		t.Fatalf("stdout must contain exactly one JSON line: %q", stdout.String())
	}
	decoder := json.NewDecoder(strings.NewReader(stdout.String()))
	var receipt selfUpdateReceipt
	if err := decoder.Decode(&receipt); err != nil {
		t.Fatalf("stdout contains prose or invalid JSON: %v: %q", err, stdout.String())
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		t.Fatalf("stdout contains more than one object: %v: %q", err, stdout.String())
	}
	if receipt.Status != "busy" {
		t.Fatalf("status = %q", receipt.Status)
	}
	if receipt.TotalMS != 70 || receipt.PhaseMS.Check != 10 || receipt.PhaseMS.Build != 40 || receipt.PhaseMS.Vet != 20 {
		t.Fatalf("deterministic timing receipt = %+v", receipt)
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal([]byte(stdout.String()), &envelope); err != nil {
		t.Fatal(err)
	}
	var phases map[string]int64
	if err := json.Unmarshal(envelope["phase_ms"], &phases); err != nil {
		t.Fatalf("phase_ms is not an object: %v", err)
	}
	wantPhases := []string{"check", "lock", "cleanup", "prepare", "companion", "build", "vet", "smoke", "install", "verify", "handoff"}
	if len(phases) != len(wantPhases) {
		t.Fatalf("phase_ms keys = %v; want stable %v", phases, wantPhases)
	}
	for _, phase := range wantPhases {
		if _, ok := phases[phase]; !ok {
			t.Errorf("phase_ms missing stable key %q: %v", phase, phases)
		}
	}
	if timeCall != len(times) {
		t.Fatalf("timing seam calls = %d, want %d", timeCall, len(times))
	}
	if got := stderr.String(); !strings.Contains(got, "progress=82%") ||
		!strings.Contains(got, "progress=100%") ||
		!strings.Contains(got, "timing total_ms=70 dominant_phase=build dominant_ms=40") ||
		strings.Contains(stdout.String(), "installing verified binaries") ||
		strings.Contains(stdout.String(), "dominant_phase") {
		t.Fatalf("progress routing: stdout=%q stderr=%q", stdout.String(), got)
	}
}
