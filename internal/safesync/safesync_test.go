package safesync

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseNameStatusZ(t *testing.T) {
	got := parseNameStatusZ([]byte("M\x00a.txt\x00A\x00new.txt\x00R100\x00old.txt\x00renamed.txt\x00"))
	want := []Entry{
		{Status: "M", Path: "a.txt"},
		{Status: "A", Path: "new.txt"},
		{Status: "R100", Path: "old.txt"},
		{Status: "R100", Path: "renamed.txt"},
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%+v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("entry %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestBehindDirtyIdenticalAssessmentDefersToGitAndPreservesBytes(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "a.txt"), "v2\n")     // M, identical to target
	writeFile(t, filepath.Join(clone, "new.txt"), "n1\n")   // A, identical to target
	writeFile(t, filepath.Join(clone, "mine.txt"), "local") // unrelated dirty work
	headBefore := revString(t, clone, "HEAD")

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateBehind || info.OK || len(info.Divergent) != 2 {
		t.Fatalf("assessment = %+v, want target-identical dirty paths refused without pre-cleaning", info)
	}
	if info.WriteCount != 2 {
		t.Fatalf("write count = %d, want 2", info.WriteCount)
	}

	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if applied.Applied || applied.OK {
		t.Fatalf("apply should defer to git's dirty-worktree refusal: %+v", applied)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved on refusal: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("a.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "new.txt")); got != "n1\n" {
		t.Fatalf("new.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "mine.txt")); got != "local" {
		t.Fatalf("unrelated work was not preserved: %q", got)
	}
}

func TestApplyPreservesPeerCreatedTargetPathRacedAfterAssessment(t *testing.T) {
	clone := behindClone(t)
	headBefore := revString(t, clone, "HEAD")
	created := false
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		res := RealRunner(ctx, repo, args...)
		// Apply revalidates the branch only after classification. Create the peer
		// file immediately after that read: the old Apply path called os.Remove
		// directly in this assess->apply window and silently deleted these bytes.
		if !created && len(args) >= 2 && args[0] == "rev-parse" && args[1] == "--abbrev-ref" {
			writeFile(t, filepath.Join(clone, "new.txt"), "PEER CREATED AFTER ASSESSMENT\n")
			created = true
		}
		return res
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatal(err)
	}
	if !created || info.Applied || info.OK {
		t.Fatalf("apply = %+v, created=%v; want non-mutating refusal", info, created)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved on target-path race: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "new.txt")); got != "PEER CREATED AFTER ASSESSMENT\n" {
		t.Fatalf("peer-created target path was deleted or changed: %q", got)
	}
}

func TestApplyPreservesPeerCreatedIgnoredTargetPath(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, ".git", "info", "exclude"), "new.txt\n")
	headBefore := revString(t, clone, "HEAD")
	created := false
	sawNoOverwriteIgnore := false
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		for _, arg := range args {
			if arg == "--no-overwrite-ignore" {
				sawNoOverwriteIgnore = true
			}
		}
		res := RealRunner(ctx, repo, args...)
		if !created && len(args) >= 2 && args[0] == "rev-parse" && args[1] == "--abbrev-ref" {
			writeFile(t, filepath.Join(clone, "new.txt"), "IGNORED PEER FILE AFTER ASSESSMENT\n")
			created = true
		}
		return res
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatal(err)
	}
	if !created || !sawNoOverwriteIgnore || info.Applied || info.OK {
		t.Fatalf("apply = %+v, created=%v no-overwrite-ignore=%v; want refusal", info, created, sawNoOverwriteIgnore)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved over ignored peer file: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "new.txt")); got != "IGNORED PEER FILE AFTER ASSESSMENT\n" {
		t.Fatalf("ignored peer file was overwritten: %q", got)
	}
}

func TestApplyDisablesConfiguredMergeAutostash(t *testing.T) {
	clone := behindClone(t)
	git(t, clone, "config", "merge.autoStash", "true")
	headBefore := revString(t, clone, "HEAD")
	mutated := false
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		res := RealRunner(ctx, repo, args...)
		if !mutated && isCleanHashFor(args, "a.txt") {
			writeFile(t, filepath.Join(clone, "a.txt"), "PEER EDIT WITH AUTOSTASH CONFIGURED\n")
			mutated = true
		}
		return res
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatal(err)
	}
	if !mutated || info.Applied || info.OK {
		t.Fatalf("apply = %+v, mutated=%v; want refusal despite merge.autoStash=true", info, mutated)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved through configured autostash: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "PEER EDIT WITH AUTOSTASH CONFIGURED\n" {
		t.Fatalf("peer edit was autostashed/reapplied or conflicted: %q", got)
	}
	if status := gitOutput(t, clone, "status", "--porcelain", "--", "a.txt"); strings.Contains(status, "UU") {
		t.Fatalf("autostash left a conflict instead of a refusal: %q", status)
	}
}

func TestApplyUnknownMergeFailureRemainsError(t *testing.T) {
	clone := behindClone(t)
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if len(args) > 0 && args[0] == "merge" {
			return RunResult{Code: 128, Stderr: []byte("fatal: Unable to create '.git/index.lock': Permission denied\n")}
		}
		return RealRunner(ctx, repo, args...)
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err == nil || !strings.Contains(err.Error(), "index.lock") {
		t.Fatalf("Apply error = %v, info=%+v; want infrastructure GitError", err, info)
	}
}

func TestApplyRefusesWhenHeadMovesAfterAssessment(t *testing.T) {
	clone := behindClone(t)
	assessedHead := revString(t, clone, "HEAD")
	peerHead := ""
	sawMerge := false
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if len(args) > 0 && args[0] == "merge" {
			sawMerge = true
		}
		res := RealRunner(ctx, repo, args...)
		if peerHead == "" && len(args) >= 2 && args[0] == "rev-parse" && args[1] == "--abbrev-ref" {
			git(t, clone, "commit", "--allow-empty", "-m", "peer moved HEAD")
			peerHead = revString(t, clone, "HEAD")
		}
		return res
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatal(err)
	}
	if peerHead == "" || peerHead == assessedHead {
		t.Fatalf("test did not move HEAD: before=%s after=%s", assessedHead, peerHead)
	}
	if info.Applied || info.OK || !strings.Contains(info.Reason, "HEAD changed after assessment") {
		t.Fatalf("apply = %+v, want typed HEAD-race refusal", info)
	}
	if sawMerge {
		t.Fatal("apply invoked merge after source HEAD failed revalidation")
	}
	if got := revString(t, clone, "HEAD"); got != peerHead {
		t.Fatalf("apply changed peer HEAD: got %s want %s", got, peerHead)
	}
}

func TestApplyPreservesPeerEditRacedAfterAssessment(t *testing.T) {
	clone := behindClone(t)
	headBefore := revString(t, clone, "HEAD")
	mutated := false
	sawPreclean := false
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if len(args) > 0 && (args[0] == "checkout" || args[0] == "rm") {
			sawPreclean = true
		}
		res := RealRunner(ctx, repo, args...)
		// classify has already confirmed the filtered worktree representation.
		// Mutate immediately after that final read to reproduce the old assess ->
		// checkout/remove data-loss window.
		if !mutated && isCleanHashFor(args, "a.txt") {
			writeFile(t, filepath.Join(clone, "a.txt"), "PEER EDIT AFTER ASSESSMENT\n")
			mutated = true
		}
		return res
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatal(err)
	}
	if !mutated {
		t.Fatal("test runner did not inject the post-assessment edit")
	}
	if info.Applied || info.OK || !strings.Contains(info.Reason, "fast-forward refused after assessment") {
		t.Fatalf("apply = %+v, want typed non-mutating refusal", info)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved on raced refusal: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "PEER EDIT AFTER ASSESSMENT\n" {
		t.Fatalf("peer edit was not preserved: %q", got)
	}
	if sawPreclean {
		t.Fatal("apply ran checkout/rm pre-cleaning in the raced window")
	}
}

func TestApplyPinsAssessedTargetWhenTrackingRefAdvances(t *testing.T) {
	clone := behindClone(t)
	assessedTarget := revString(t, clone, "origin/work")
	origin := strings.TrimSpace(gitOutput(t, clone, "remote", "get-url", "origin"))
	advanced := false
	mergeTarget := ""
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if !advanced && len(args) >= 3 && args[0] == "merge" && args[1] == "--ff-only" {
			mergeTarget = args[len(args)-1]
			writeFile(t, filepath.Join(origin, "late.txt"), "late remote commit\n")
			git(t, origin, "add", "late.txt")
			git(t, origin, "commit", "-m", "c3")
			git(t, clone, "fetch", "origin")
			advanced = true
		}
		return RealRunner(ctx, repo, args...)
	}

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatal(err)
	}
	if !advanced || !info.Applied {
		t.Fatalf("apply = %+v, advanced=%v; want captured target applied", info, advanced)
	}
	if mergeTarget != assessedTarget {
		t.Fatalf("merge target = %q, want assessed immutable SHA %q", mergeTarget, assessedTarget)
	}
	if got := revString(t, clone, "HEAD"); got != assessedTarget {
		t.Fatalf("HEAD = %s, want assessed target %s", got, assessedTarget)
	}
	if got := revString(t, clone, "origin/work"); got == assessedTarget {
		t.Fatalf("test did not advance mutable origin/work beyond %s", assessedTarget)
	}
	if _, err := os.Stat(filepath.Join(clone, "late.txt")); !os.IsNotExist(err) {
		t.Fatalf("late unassessed commit leaked into worktree: err=%v", err)
	}
}

func TestBehindAutocrlfCleanFilterEquivalentWritePathsApply(t *testing.T) {
	clone, target := autocrlfBehindClone(t)

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateBehind || !info.OK || len(info.Divergent) != 0 {
		t.Fatalf("assessment = %+v, want clean-filter-equivalent write paths safe", info)
	}
	if !hasEntry(info.Identical, "M", "modified.txt") || !hasEntry(info.Identical, "D", "deleted.txt") {
		t.Fatalf("safe entries = %+v, want modified.txt M and deleted.txt D", info.Identical)
	}

	var mergeArgs []string
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if len(args) > 0 && args[0] == "merge" {
			mergeArgs = append([]string(nil), args...)
		}
		return RealRunner(ctx, repo, args...)
	}
	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work", Runner: runner})
	if err != nil {
		t.Fatal(err)
	}
	if !applied.Applied || !applied.OK || applied.NewHead != target {
		t.Fatalf("apply = %+v, want exact target %s applied", applied, target)
	}
	wantMerge := []string{"merge", "--ff-only", "--no-autostash", "--no-overwrite-ignore", target}
	if strings.Join(mergeArgs, "\x00") != strings.Join(wantMerge, "\x00") {
		t.Fatalf("merge args = %q, want final immutable ff-only guard %q", mergeArgs, wantMerge)
	}
	if got := readFile(t, filepath.Join(clone, "modified.txt")); got != "v2\r\n" {
		t.Fatalf("modified checkout bytes = %q, want configured CRLF representation", got)
	}
	if _, err := os.Stat(filepath.Join(clone, "deleted.txt")); !os.IsNotExist(err) {
		t.Fatalf("deleted.txt still exists after exact-target apply: %v", err)
	}
	if got := gitOutput(t, clone, "status", "--porcelain", "--", "modified.txt", "deleted.txt", "genuine.txt"); got != "" {
		t.Fatalf("applied target is dirty: %q", got)
	}
}

func TestBehindAutocrlfGenuineModifiedPathRefuses(t *testing.T) {
	clone, _ := autocrlfBehindClone(t)
	headBefore := revString(t, clone, "HEAD")
	writeFile(t, filepath.Join(clone, "genuine.txt"), "LOCAL\r\nEDIT\r\n")

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateBehind || info.OK || len(info.Divergent) != 1 || info.Divergent[0].Path != "genuine.txt" {
		t.Fatalf("assessment = %+v, want only genuine.txt divergent", info)
	}
	if !hasEntry(info.Identical, "M", "modified.txt") || !hasEntry(info.Identical, "D", "deleted.txt") {
		t.Fatalf("safe entries = %+v, want clean-filter-equivalent M/D paths preserved", info.Identical)
	}

	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if applied.Applied || applied.OK {
		t.Fatalf("apply should refuse genuine unstaged divergence: %+v", applied)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved on refusal: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "genuine.txt")); got != "LOCAL\r\nEDIT\r\n" {
		t.Fatalf("genuine local edit changed on refusal: %q", got)
	}
}

func TestBehindCleanTrackedUpdateIsSafeAndApplies(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "mine.txt"), "local") // unrelated dirty work

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateBehind || !info.OK {
		t.Fatalf("assessment = %+v, want safe behind with clean tracked update", info)
	}
	if len(info.Identical) != 2 {
		t.Fatalf("identical/safe entries = %+v, want a.txt M and new.txt A", info.Identical)
	}

	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if !applied.Applied {
		t.Fatalf("apply did not apply: %+v", applied)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("a.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "new.txt")); got != "n1\n" {
		t.Fatalf("new.txt = %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "mine.txt")); got != "local" {
		t.Fatalf("unrelated work was not preserved: %q", got)
	}
}

func TestBehindDivergentTrackedRefusesAndDoesNotMoveHead(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "a.txt"), "LOCAL EDIT\n")
	headBefore := revString(t, clone, "HEAD")

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.OK || len(info.Divergent) != 1 || info.Divergent[0].Path != "a.txt" {
		t.Fatalf("assessment = %+v, want a.txt divergent", info)
	}

	applied, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if applied.Applied {
		t.Fatalf("apply should have refused: %+v", applied)
	}
	if got := revString(t, clone, "HEAD"); got != headBefore {
		t.Fatalf("HEAD moved on refusal: got %s want %s", got, headBefore)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "LOCAL EDIT\n" {
		t.Fatalf("worktree changed on refusal: %q", got)
	}
}

func TestInSyncIsOKNoop(t *testing.T) {
	clone := behindClone(t)
	git(t, clone, "merge", "--ff-only", "origin/work")

	info, err := Apply(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateInSync || !info.OK || info.Applied {
		t.Fatalf("apply in sync = %+v, want ok noop", info)
	}
}

func TestDivergedRefuses(t *testing.T) {
	clone := behindClone(t)
	writeFile(t, filepath.Join(clone, "local.txt"), "local\n")
	git(t, clone, "add", "local.txt")
	git(t, clone, "commit", "-m", "local")

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateDiverged || info.OK {
		t.Fatalf("assessment = %+v, want diverged refusal", info)
	}
}

func TestAheadNamesSafePushPath(t *testing.T) {
	clone := behindClone(t)
	git(t, clone, "merge", "--ff-only", "origin/work")
	writeFile(t, filepath.Join(clone, "local.txt"), "local\n")
	git(t, clone, "add", "local.txt")
	git(t, clone, "commit", "-m", "local")

	info, err := Assess(context.Background(), Options{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil {
		t.Fatal(err)
	}
	if info.State != StateAhead || info.OK {
		t.Fatalf("assessment = %+v, want ahead refusal", info)
	}
	if !strings.Contains(info.Reason, "fak sync push --remote origin --branch work") {
		t.Fatalf("ahead reason should name safe push path, got %q", info.Reason)
	}
}

func TestRenameWriteSetRefuses(t *testing.T) {
	clone := behindClone(t)
	head := revString(t, clone, "HEAD")
	target := revString(t, clone, "origin/work")
	entries := []Entry{{Status: "R100", Path: "old.txt"}, {Status: "R100", Path: "new.txt"}}
	_, divergent := classify(clone, RealRunner, context.Background(), head, target, entries)
	if len(divergent) != 2 {
		t.Fatalf("rename entries should be divergent/refused, got %+v", divergent)
	}
}

func behindClone(t testing.TB) string {
	t.Helper()
	tmp := t.TempDir()
	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "work")
	git(t, origin, "config", "core.autocrlf", "false")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "a.txt"), "v1\n")
	writeFile(t, filepath.Join(origin, "keep.txt"), "keep\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "c1")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "-c", "core.autocrlf=false", "clone", origin, clone)
	git(t, clone, "config", "core.autocrlf", "false")
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")

	writeFile(t, filepath.Join(origin, "a.txt"), "v2\n")
	writeFile(t, filepath.Join(origin, "new.txt"), "n1\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "c2")
	git(t, clone, "fetch", "origin")
	return clone
}

func autocrlfBehindClone(t *testing.T) (string, string) {
	t.Helper()
	tmp := t.TempDir()
	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "work")
	git(t, origin, "config", "core.autocrlf", "false")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "modified.txt"), "v1\n")
	writeFile(t, filepath.Join(origin, "deleted.txt"), "delete me\n")
	writeFile(t, filepath.Join(origin, "genuine.txt"), "v1\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "c1")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "-c", "core.autocrlf=true", "clone", origin, clone)
	git(t, clone, "config", "core.autocrlf", "true")
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")
	if got := readFile(t, filepath.Join(clone, "modified.txt")); got != "v1\r\n" {
		t.Fatalf("fixture checkout bytes = %q, want core.autocrlf CRLF", got)
	}
	if got := readFile(t, filepath.Join(clone, "deleted.txt")); got != "delete me\r\n" {
		t.Fatalf("deleted-path fixture bytes = %q, want core.autocrlf CRLF", got)
	}
	if got := gitOutput(t, clone, "show", "HEAD:modified.txt"); got != "v1\n" {
		t.Fatalf("fixture HEAD blob = %q, want canonical LF bytes", got)
	}

	writeFile(t, filepath.Join(origin, "modified.txt"), "v2\n")
	writeFile(t, filepath.Join(origin, "genuine.txt"), "v2\n")
	git(t, origin, "rm", "deleted.txt")
	git(t, origin, "add", "modified.txt", "genuine.txt")
	git(t, origin, "commit", "-m", "c2")
	git(t, clone, "fetch", "origin")
	return clone, revString(t, clone, "origin/work")
}

func hasEntry(entries []Entry, status, path string) bool {
	for _, entry := range entries {
		if entry.Status == status && entry.Path == path {
			return true
		}
	}
	return false
}

func isCleanHashFor(args []string, path string) bool {
	return len(args) == 4 && args[0] == "hash-object" && args[1] == "--path="+path &&
		args[2] == "--" && args[3] == path
}

func git(t testing.TB, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func gitOutput(t testing.TB, cwd string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

func revString(t testing.TB, cwd, ref string) string {
	t.Helper()
	cmd := exec.Command("git", "rev-parse", "--verify", ref)
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("rev-parse %s: %v", ref, err)
	}
	return strings.TrimSpace(string(out))
}

func mkdir(t testing.TB, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

func writeFile(t testing.TB, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readFile(t testing.TB, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
