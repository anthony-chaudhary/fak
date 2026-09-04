package dojocal

// worktree_test.go covers the PURE decision core of the dojo worktree arm — the
// part that decides KEEP/REVERT from the measured shard folds + suite/truth
// witnesses. The git/exec integration is exercised by the standalone cmd/dojorsi
// binary against a real corpus; here we prove the keep logic, including the
// acceptance gate "the two-shard gate refuses a single-shard KEEP", without
// needing a worktree or a corpus.

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// fakeEpisodes builds measured OVER_CLAIM episodes for one lever/metric cell with
// the given claimed/realized pair, so a fold over them has a known calib_err.
func fakeEpisodes(lever, metric string, claimed, realized float64, n int) []dojo.Episode {
	out := make([]dojo.Episode, n)
	for i := range out {
		out[i] = dojo.Episode{
			Lever:    lever,
			Metric:   metric,
			Claimed:  claimed,
			Realized: realized,
			Verdict:  dojo.VerdictOverClaim,
		}
	}
	return out
}

// wrapReport wraps episodes in a minimal dojo.Report the fold reads (named to
// avoid colliding with the `report` helper in dojocal_test.go).
func wrapReport(eps []dojo.Episode) dojo.Report {
	return dojo.Report{Schema: dojo.Schema, Episodes: eps}
}

// TestFoldTwoShardsDerivesFullOverCombinedEpisodes proves Full is the fold over
// BOTH shards' episodes combined, not the mean of the two shard values — the
// populations may differ in size, so the combined mean must be re-derived.
func TestFoldTwoShardsDerivesFullOverCombinedEpisodes(t *testing.T) {
	// Shard A: 1 episode, claim 0.9 realized 0.6 (calib_err 0.3). Shard B: 3
	// identical such episodes. The full corpus is 4 episodes mean calib_err 0.3,
	// NOT (0.3 + 0.3)/2 = 0.3 here (equal), so use a starker split to prove the
	// combine happens over episodes, not over shard means.
	repA := wrapReport(fakeEpisodes("resume-posture", "cold_write_share", 0.9, 0.6, 1)) // 1 ep, err 0.3
	repB := wrapReport(fakeEpisodes("resume-posture", "cold_write_share", 0.9, 0.3, 3)) // 3 eps, err 0.6
	got := foldTwoShards(repA, repB)

	fullOverCombined := dojo.FoldCalibrable(append(append([]dojo.Episode(nil), repA.Episodes...), repB.Episodes...)).Value
	if got.Full != fullOverCombined {
		t.Fatalf("Full = %v, want combined-episodes fold %v", got.Full, fullOverCombined)
	}
	if got.ShardA != dojo.FoldCalibrable(repA.Episodes).Value {
		t.Fatalf("ShardA = %v, want %v", got.ShardA, dojo.FoldCalibrable(repA.Episodes).Value)
	}
	if got.ShardB != dojo.FoldCalibrable(repB.Episodes).Value {
		t.Fatalf("ShardB = %v, want %v", got.ShardB, dojo.FoldCalibrable(repB.Episodes).Value)
	}
	if got.Measured != 4 {
		t.Fatalf("Measured = %v, want 4", got.Measured)
	}
}

// TestTwoShardGateRequiresStrictDropOnBothShards is the acceptance gate: a
// candidate that drops only ONE shard must fail the gate.
func TestTwoShardGateRequiresStrictDropOnBothShards(t *testing.T) {
	base := shardFolds{Full: 0.5, ShardA: 0.4, ShardB: 0.6}
	cases := []struct {
		name string
		cand shardFolds
		want bool
	}{
		{"both drop strictly", shardFolds{ShardA: 0.3, ShardB: 0.5}, true},
		{"only A drops", shardFolds{ShardA: 0.3, ShardB: 0.6}, false},
		{"only B drops", shardFolds{ShardA: 0.4, ShardB: 0.5}, false},
		{"neither drops", shardFolds{ShardA: 0.4, ShardB: 0.6}, false},
		{"A drops, B rises (overfit)", shardFolds{ShardA: 0.3, ShardB: 0.7}, false},
		{"equal on A (not strict)", shardFolds{ShardA: 0.4, ShardB: 0.5}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := twoShardGate(base, c.cand); got != c.want {
				t.Fatalf("twoShardGate = %v, want %v", got, c.want)
			}
		})
	}
}

// TestMeasureCandidateKeepsOnCleanTwoShardGain proves the happy path: full drops,
// both shards drop, suite green, truth clean => Measurement that shipgate keeps.
func TestMeasureCandidateKeepsOnCleanTwoShardGain(t *testing.T) {
	base := shardFolds{Full: 0.50, ShardA: 0.40, ShardB: 0.60, Measured: 10}
	cand := shardFolds{Full: 0.35, ShardA: 0.25, ShardB: 0.45, Measured: 10}
	wc := WorktreeCandidate{Lever: "resume-posture", Metric: "cold_write_share", NewClaimed: 0.7}

	m := measureCandidate(base, cand, true, true, "", wc)

	if m.Metric != cand.Full {
		t.Fatalf("Metric = %v, want candidate Full %v", m.Metric, cand.Full)
	}
	if !m.SuiteGreen {
		t.Fatalf("SuiteGreen = false, want true (go-suite green AND two-shard gate passed)")
	}
	if !m.TruthClean {
		t.Fatalf("TruthClean = false, want true")
	}
	if m.Note == "" || m.Note[:4] != "KEPT" {
		t.Fatalf("Note = %q, want a KEPT reason", m.Note)
	}
}

// TestMeasureCandidateRevertsOnSingleShardGain is the core acceptance gate from
// #1024: a candidate whose FULL fold drops but only ONE shard drops must NOT keep
// — the two-shard gate refuses a single-shard KEEP.
func TestMeasureCandidateRevertsOnSingleShardGain(t *testing.T) {
	base := shardFolds{Full: 0.50, ShardA: 0.40, ShardB: 0.60, Measured: 10}
	// Full drops (0.50 -> 0.45) and shard A drops, but shard B does NOT — an
	// overfit to shard A. Suite green + truth clean are irrelevant: the gate must
	// still flip SuiteGreen false so the keep-bit reverts.
	cand := shardFolds{Full: 0.45, ShardA: 0.30, ShardB: 0.60, Measured: 10}
	wc := WorktreeCandidate{Lever: "resume-posture", Metric: "cold_write_share", NewClaimed: 0.7}

	m := measureCandidate(base, cand, true, true, "", wc)

	if m.SuiteGreen {
		t.Fatalf("SuiteGreen = true, want false: a single-shard gain must fail the two-shard gate")
	}
	if m.Note == "" || m.Note[:6] != "REVERT" {
		t.Fatalf("Note = %q, want a REVERT reason naming the two-shard gate", m.Note)
	}
}

// TestMeasureCandidateRevertsOnSuiteRed proves a red go-suite reverts even when
// both shards dropped and the tree is clean.
func TestMeasureCandidateRevertsOnSuiteRed(t *testing.T) {
	base := shardFolds{Full: 0.50, ShardA: 0.40, ShardB: 0.60}
	cand := shardFolds{Full: 0.30, ShardA: 0.20, ShardB: 0.40}
	wc := WorktreeCandidate{Lever: "resume-posture", Metric: "cold_write_share", NewClaimed: 0.7}

	m := measureCandidate(base, cand, false, true, "go test ./... exited non-zero: FAIL", wc)

	if m.SuiteGreen {
		t.Fatalf("SuiteGreen = true, want false on a red suite")
	}
	if m.Note == "" || m.Note[:6] != "REVERT" {
		t.Fatalf("Note = %q, want a REVERT reason naming the red suite", m.Note)
	}
}

// TestMeasureCandidateRevertsOnTruthUnclean proves a worktree that changed more
// than claims.go reverts even when everything else passed.
func TestMeasureCandidateRevertsOnTruthUnclean(t *testing.T) {
	base := shardFolds{Full: 0.50, ShardA: 0.40, ShardB: 0.60}
	cand := shardFolds{Full: 0.30, ShardA: 0.20, ShardB: 0.40}
	wc := WorktreeCandidate{Lever: "resume-posture", Metric: "cold_write_share", NewClaimed: 0.7}

	m := measureCandidate(base, cand, true, false, "", wc)

	if m.TruthClean {
		t.Fatalf("TruthClean = true, want false")
	}
	if m.Note == "" || m.Note[:6] != "REVERT" {
		t.Fatalf("Note = %q, want a REVERT reason naming the truth-unclean tree", m.Note)
	}
}

// TestMeasureCandidateRevertsOnNoFullGain proves a candidate whose full fold did
// not drop reverts even with suite green + truth clean + both shards... well,
// both shards can't drop while full doesn't unless measured counts shift, but the
// guard is explicit: fullDelta <= 0 => revert.
func TestMeasureCandidateRevertsOnNoFullGain(t *testing.T) {
	base := shardFolds{Full: 0.30, ShardA: 0.20, ShardB: 0.40}
	cand := shardFolds{Full: 0.30, ShardA: 0.20, ShardB: 0.40} // identical => no delta
	wc := WorktreeCandidate{Lever: "resume-posture", Metric: "cold_write_share", NewClaimed: 0.7}

	m := measureCandidate(base, cand, true, true, "", wc)

	// Both shards "drop" is false (equal, not strict), so the two-shard gate fires
	// first; either way the candidate must not keep. Assert the hard contract:
	if m.SuiteGreen && m.TruthClean && m.Metric < base.Full {
		t.Fatalf("candidate with no strict gain must not be keepable, got SuiteGreen=%v TruthClean=%v Metric=%v base=%v",
			m.SuiteGreen, m.TruthClean, m.Metric, base.Full)
	}
}

// TestMeasureCandidateScorecardCarriesShardDeltas proves the telemetry scorecard
// records the per-shard before/after so a reader can diagnose a revert.
func TestMeasureCandidateScorecardCarriesShardDeltas(t *testing.T) {
	base := shardFolds{Full: 0.50, ShardA: 0.40, ShardB: 0.60, Measured: 10}
	cand := shardFolds{Full: 0.35, ShardA: 0.25, ShardB: 0.45, Measured: 10}
	wc := WorktreeCandidate{Lever: "resume-posture", Metric: "cold_write_share", NewClaimed: 0.7}

	m := measureCandidate(base, cand, true, true, "", wc)
	if m.Score == nil {
		t.Fatal("Score = nil, want a scorecard")
	}
	want := map[string]float64{
		"baseline_full":     0.50,
		"candidate_full":    0.35,
		"baseline_shard_a":  0.40,
		"candidate_shard_a": 0.25,
		"baseline_shard_b":  0.60,
		"candidate_shard_b": 0.45,
	}
	got := map[string]float64{}
	for _, c := range m.Score.Components {
		got[c.Name] = c.Value
	}
	for name, v := range want {
		if got[name] != v {
			t.Errorf("scorecard %s = %v, want %v", name, got[name], v)
		}
	}
}

// TestSplitCorpusRootIsCallerReapable witnesses the #3295 fix: splitCorpus now returns
// the shard ROOT (previously unreachable — only shardA/shardB were returned), both
// shards live under it, and a single os.RemoveAll(shardRoot) reaps everything. This is
// what lets measureShardFolds defer the cleanup instead of orphaning a corpus-sized
// dojocal-shards-* dir every (>=2x/run) call. An isolated scratch parent keeps the
// count assertion immune to any concurrent dojo run on the shared machine.
func TestSplitCorpusRootIsCallerReapable(t *testing.T) {
	scratch := t.TempDir()
	corpus := filepath.Join(t.TempDir(), "corpus")
	if err := os.MkdirAll(corpus, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"a.jsonl", "b.jsonl", "c.jsonl", "d.jsonl"} {
		if err := os.WriteFile(filepath.Join(corpus, n), []byte(`{"x":1}`+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	shardRoot, shardA, shardB, err := splitCorpus(corpus, scratch)
	if err != nil {
		t.Fatalf("splitCorpus: %v", err)
	}
	if shardRoot == "" {
		t.Fatal("splitCorpus returned an empty shard root — caller cannot reap it")
	}
	if filepath.Dir(shardA) != shardRoot || filepath.Dir(shardB) != shardRoot {
		t.Fatalf("shards not under root: root=%s a=%s b=%s", shardRoot, shardA, shardB)
	}
	if got := shardRootCount(t, scratch); got != 1 {
		t.Fatalf("want 1 dojocal-shards-* dir under scratch, got %d", got)
	}

	os.RemoveAll(shardRoot)
	if got := shardRootCount(t, scratch); got != 0 {
		t.Fatalf("shard root survived caller cleanup: %d dir(s) remain", got)
	}
}

func shardRootCount(t *testing.T, parent string) int {
	t.Helper()
	m, err := filepath.Glob(filepath.Join(parent, "dojocal-shards-*"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	return len(m)
}

func createTestRepoWithFiles(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()

	gitRun := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}

	gitRun("init", "-q")
	gitRun("config", "user.name", "Tester")
	gitRun("config", "user.email", "tester@test.com")

	if err := os.MkdirAll(filepath.Join(repo, "pkg", "a"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "pkg", "b"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(repo, "pkg", "a", "a.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "pkg", "b", "b.go"), []byte("package b\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "docs", "readme.md"), []byte("# docs\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	gitRun("add", ".")
	gitRun("commit", "-q", "-m", "initial commit")

	return repo
}

func createTestWorktree(t *testing.T, repo string) string {
	t.Helper()
	wtDir := filepath.Join(t.TempDir(), "wt")
	cmd := exec.Command("git", "-C", repo, "worktree", "add", "--detach", wtDir, "HEAD")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git worktree add: %v: %s", err, out)
	}
	t.Cleanup(func() {
		_ = exec.Command("git", "-C", repo, "worktree", "remove", "--force", wtDir).Run()
	})
	return wtDir
}

func TestApplyWorktreeIncludePatterns(t *testing.T) {
	repo := createTestRepoWithFiles(t)
	wtDir := createTestWorktree(t, repo)

	if err := ApplyWorktreeInclude(wtDir, []string{"pkg/a/"}); err != nil {
		t.Fatalf("ApplyWorktreeInclude: %v", err)
	}

	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "a", "a.go")); err != nil {
		t.Errorf("expected pkg/a/a.go to exist, got: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "b", "b.go")); !os.IsNotExist(err) {
		t.Errorf("expected pkg/b/b.go to be excluded, got err: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "docs", "readme.md")); !os.IsNotExist(err) {
		t.Errorf("expected docs/readme.md to be excluded, got err: %v", err)
	}
}

func TestApplyWorktreeIncludeFromFile(t *testing.T) {
	repo := createTestRepoWithFiles(t)
	wtDir := createTestWorktree(t, repo)

	includeContent := "# Comments should be ignored\n\n  pkg/a/  \n# trailing comment\n"
	if err := os.WriteFile(filepath.Join(wtDir, ".worktreeinclude"), []byte(includeContent), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := ApplyWorktreeInclude(wtDir, nil); err != nil {
		t.Fatalf("ApplyWorktreeInclude: %v", err)
	}

	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "a", "a.go")); err != nil {
		t.Errorf("expected pkg/a/a.go to exist, got: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "b", "b.go")); !os.IsNotExist(err) {
		t.Errorf("expected pkg/b/b.go to be excluded, got err: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "docs", "readme.md")); !os.IsNotExist(err) {
		t.Errorf("expected docs/readme.md to be excluded, got err: %v", err)
	}
}

func TestApplyWorktreeIncludeFromParentRepo(t *testing.T) {
	repo := createTestRepoWithFiles(t)
	if err := os.WriteFile(filepath.Join(repo, ".worktreeinclude"), []byte("pkg/a/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	wtDir := createTestWorktree(t, repo)

	if err := ApplyWorktreeInclude(wtDir, nil); err != nil {
		t.Fatalf("ApplyWorktreeInclude: %v", err)
	}

	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "a", "a.go")); err != nil {
		t.Errorf("expected pkg/a/a.go to exist, got: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "b", "b.go")); !os.IsNotExist(err) {
		t.Errorf("expected pkg/b/b.go to be excluded, got err: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "docs", "readme.md")); !os.IsNotExist(err) {
		t.Errorf("expected docs/readme.md to be excluded, got err: %v", err)
	}
}

func TestApplyWorktreeIncludeEmptyNoop(t *testing.T) {
	repo := createTestRepoWithFiles(t)
	wtDir := createTestWorktree(t, repo)

	if err := ApplyWorktreeInclude(wtDir, nil); err != nil {
		t.Fatalf("ApplyWorktreeInclude: %v", err)
	}

	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "a", "a.go")); err != nil {
		t.Errorf("expected pkg/a/a.go to exist: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "pkg", "b", "b.go")); err != nil {
		t.Errorf("expected pkg/b/b.go to exist: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wtDir, "docs", "readme.md")); err != nil {
		t.Errorf("expected docs/readme.md to exist: %v", err)
	}
}
