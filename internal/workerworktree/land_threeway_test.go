package workerworktree

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// threeWayRel is outside internal/cmd/docs/tools so the land skips the
// whole-tree disambiguation phase; only the isolated apply is under test.
const threeWayRel = "pkg/ctx.txt"

type threeWayRepo struct {
	root, wt, base, msg string
}

func threeWayLines(edit map[int]string) string {
	var b strings.Builder
	for i := 1; i <= 10; i++ {
		line := fmt.Sprintf("line %d", i)
		if v, ok := edit[i]; ok {
			line = v
		}
		b.WriteString(line + "\n")
	}
	return b.String()
}

// newThreeWayRepo builds a ten-line base file, prepares a worker at that base
// with the worker's edit (uncommitted, as a live worker leaves it), then moves
// trunk past the base with a peer edit, so the worker patch is applied to a
// trunk that is no longer its preimage.
func newThreeWayRepo(t *testing.T, worker, trunk map[int]string) threeWayRepo {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "threeway@test.invalid")
	mustGit(t, root, "config", "user.name", "ThreeWay Test")
	mustGit(t, root, "config", "commit.gpgsign", "false")
	mustGit(t, root, "config", "core.autocrlf", "false")
	writeRepoFile(t, root, threeWayRel, threeWayLines(nil))
	mustGit(t, root, "add", threeWayRel)
	mustGit(t, root, "commit", "-q", "-m", "base")
	base := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	prep := Prepare(root, "threeway", "ctx", base, t.TempDir(), nil)
	if !prep.OK {
		t.Fatalf("prepare: %+v", prep)
	}
	writeRepoFile(t, prep.Path, threeWayRel, threeWayLines(worker))

	writeRepoFile(t, root, threeWayRel, threeWayLines(trunk))
	mustGit(t, root, "commit", "-q", "-am", "peer trunk edit")

	msg := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(msg, []byte("feat(pkg): worker edit (fak workerworktree)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return threeWayRepo{root: root, wt: prep.Path, base: base, msg: msg}
}

// A peer trunk edit two lines above the worker's hunk sits inside the patch's
// three-line context window, so a plain `apply --cached` refuses it even though
// the two edits never overlap. The land must accept the clean 3-way merge and
// carry BOTH edits.
func TestLandAcceptsCleanThreeWayWhenTrunkEditsContextLines(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	r := newThreeWayRepo(t, map[int]string{5: "line 5 WORKER"}, map[int]string{3: "line 3 TRUNK"})
	trunkTip := strings.TrimSpace(mustGit(t, r.root, "rev-parse", "refs/heads/main"))

	res := Land(r.root, r.wt, r.base, r.msg, []string{threeWayRel}, nil, nil)
	w := landWireOf(t, res)
	if !w.OK || !w.Committed || w.Code != LandResultSuccess {
		t.Fatalf("clean 3-way land refused: %+v reason=%q detail=%q", w, res.Reason, res.Detail)
	}
	head := strings.TrimSpace(mustGit(t, r.root, "rev-parse", "refs/heads/main"))
	if head == trunkTip || head != w.CommitSHA {
		t.Fatalf("trunk head=%s commit=%s peer tip=%s", head, w.CommitSHA, trunkTip)
	}
	if parent := strings.TrimSpace(mustGit(t, r.root, "rev-parse", head+"^")); parent != trunkTip {
		t.Fatalf("landed commit parent=%s, want peer tip %s", parent, trunkTip)
	}
	want := threeWayLines(map[int]string{3: "line 3 TRUNK", 5: "line 5 WORKER"})
	if got := mustGit(t, r.root, "show", "refs/heads/main:"+threeWayRel); got != want {
		t.Fatalf("landed content lost an edit:\n got %q\nwant %q", got, want)
	}
}

// Worker and trunk rewrite the SAME line differently: a true conflict. The land
// must still refuse with reconciliation-required, leave trunk where the peer
// put it, and keep conflict stages/markers out of the shared root.
func TestLandStillRefusesOverlappingConflict(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	r := newThreeWayRepo(t, map[int]string{5: "line 5 WORKER"}, map[int]string{5: "line 5 TRUNK"})
	trunkTip := strings.TrimSpace(mustGit(t, r.root, "rev-parse", "refs/heads/main"))

	res := Land(r.root, r.wt, r.base, r.msg, []string{threeWayRel}, nil, nil)
	w := landWireOf(t, res)
	if w.OK || w.Committed || w.Code != LandResultReconciliationRequired {
		t.Fatalf("overlapping conflict was not refused for reconciliation: %+v reason=%q", w, res.Reason)
	}
	if !res.Preserved {
		t.Fatalf("conflict refusal must preserve the worker candidate: %+v", res)
	}
	if head := strings.TrimSpace(mustGit(t, r.root, "rev-parse", "refs/heads/main")); head != trunkTip {
		t.Fatalf("conflict refusal moved trunk: %s -> %s", trunkTip, head)
	}
	if status := strings.TrimSpace(mustGit(t, r.root, "status", "--porcelain")); status != "" {
		t.Fatalf("conflict refusal touched the shared root index/worktree:\n%s", status)
	}
	rootBytes, err := os.ReadFile(filepath.Join(r.root, filepath.FromSlash(threeWayRel)))
	if err != nil || strings.Contains(string(rootBytes), "<<<<<<<") {
		t.Fatalf("shared root carries conflict markers or is unreadable: err=%v\n%s", err, rootBytes)
	}
	wtBytes, err := os.ReadFile(filepath.Join(r.wt, filepath.FromSlash(threeWayRel)))
	if err != nil || !strings.Contains(string(wtBytes), "line 5 WORKER") {
		t.Fatalf("worker edit not preserved in its worktree: err=%v\n%s", err, wtBytes)
	}
}
