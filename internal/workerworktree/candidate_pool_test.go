package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// fak-test:runtime medium est=85s lane=default
func TestCandidatePoolAdoptsClosedCheckoutAcrossSessions(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	commit := func(content string) string {
		if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		mustGit(t, root, "add", "a.txt")
		mustGit(t, root, "commit", "-q", "-m", content)
		return strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
	}
	first, second := commit("one"), commit("two")

	type call struct {
		dir  string
		args []string
	}
	var calls []call
	git := func(dir string, args []string) (int, string) {
		calls = append(calls, call{dir: dir, args: append([]string(nil), args...)})
		return defaultGit(dir, args)
	}
	countAdds := func() int {
		n := 0
		for _, c := range calls {
			if slices.Contains(c.args, "worktree") && slices.Contains(c.args, "add") {
				n++
			}
		}
		return n
	}

	var seen []string
	var leftoverSurvived bool
	hook := func(dir string) (bool, string) {
		if _, err := os.Stat(filepath.Join(dir, "session-leftover.txt")); err == nil {
			leftoverSurvived = true
		}
		seen = append(seen, strings.TrimSpace(mustReadFile(t, filepath.Join(dir, "a.txt"))))
		if err := os.WriteFile(filepath.Join(dir, "session-leftover.txt"), []byte("x"), 0o644); err != nil {
			return false, err.Error()
		}
		return true, ""
	}

	firstSession := newTopologyCandidateSession(root, git)
	defer firstSession.close()
	if ok, detail := firstSession.verify(first, "", hook); !ok {
		t.Fatalf("first session verify: %s", detail)
	}
	kept := firstSession.dir
	if !isTopologyCandidatePoolName(filepath.Base(kept)) {
		t.Fatalf("first session dir %q is not a pooled candidate name", kept)
	}
	firstSession.close()
	if _, err := os.Stat(kept); err != nil {
		t.Fatalf("pooled checkout %q should be kept after close: %v", kept, err)
	}

	secondSession := newTopologyCandidateSession(root, git)
	defer secondSession.close()
	if ok, detail := secondSession.verify(second, "", hook); !ok {
		t.Fatalf("second session verify: %s", detail)
	}
	if secondSession.dir != kept {
		t.Fatalf("second session materialized %q, want adoption of %q", secondSession.dir, kept)
	}
	if got, want := seen, []string{"one", "two"}; !slices.Equal(got, want) {
		t.Fatalf("verified contents = %q, want %q", got, want)
	}
	if leftoverSurvived {
		t.Fatal("first session's untracked file survived the adopted reset")
	}
	if adds := countAdds(); adds != 1 {
		t.Fatalf("worktree add ran %d times across two sessions, want 1 (the adoption must not re-materialize)", adds)
	}
	secondSession.close()
	if got := registeredPoolCandidates(t, root); len(got) != 1 || !samePath(got[0], kept) {
		t.Fatalf("pool registrations after two sessions = %v, want exactly %q", got, kept)
	}
}

// fak-test:runtime medium est=95s lane=default
func TestCandidatePoolMoveToResetsExactTree(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("a.txt", "one")
	write("b.txt", "gone-in-second")
	mustGit(t, root, "add", "a.txt", "b.txt")
	mustGit(t, root, "commit", "-q", "-m", "first")
	first := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
	if err := os.Remove(filepath.Join(root, "b.txt")); err != nil {
		t.Fatal(err)
	}
	write("a.txt", "two")
	write("c.txt", "added-in-second")
	mustGit(t, root, "add", "-A")
	mustGit(t, root, "commit", "-q", "-m", "second")
	second := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	hook := func(dir string) (bool, string) { return true, "" }
	session := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := session.verify(first, "", hook); !ok {
		t.Fatalf("first verify: %s", detail)
	}
	session.close()

	adopter := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := adopter.verify(second, "", hook); !ok {
		t.Fatalf("adopting verify: %s", detail)
	}
	dir := adopter.dir
	if strings.TrimSpace(mustReadFile(t, filepath.Join(dir, "a.txt"))) != "two" {
		t.Fatal("adopted checkout did not reset a.txt to the adopted ref")
	}
	if _, err := os.Stat(filepath.Join(dir, "b.txt")); !os.IsNotExist(err) {
		t.Fatalf("file deleted in the adopted ref survived: %v", err)
	}
	if strings.TrimSpace(mustReadFile(t, filepath.Join(dir, "c.txt"))) != "added-in-second" {
		t.Fatal("file added in the adopted ref is missing")
	}
	if out := mustGit(t, dir, "status", "--porcelain"); strings.TrimSpace(out) != "" {
		t.Fatalf("adopted checkout is not clean against its ref: %q", out)
	}
	adopter.close()
}

// fak-test:runtime medium est=95s lane=default
func TestCandidatePoolRespectsHeldLock(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }

	holder := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := holder.verify("HEAD", "", hook); !ok {
		t.Fatalf("holder verify: %s", detail)
	}
	held := holder.dir

	session := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := session.verify("HEAD", "", hook); !ok {
		t.Fatalf("session verify: %s", detail)
	}
	if samePath(session.dir, held) {
		t.Fatal("session adopted a checkout whose pool lock is held by another session")
	}
	heldSecond := session.dir
	session.close()

	// With the holder's session still open, a third session must also not
	// adopt; after the holder closes, a released checkout becomes adoptable
	// and the next session MUST adopt one of them rather than materialize.
	holder.close()
	third := newTopologyCandidateSession(root, defaultGit)
	defer third.close()
	if ok, detail := third.verify("HEAD", "", hook); !ok {
		t.Fatalf("third verify: %s", detail)
	}
	if !samePath(third.dir, held) && !samePath(third.dir, heldSecond) {
		t.Fatalf("third session materialized %q, want adoption of a released checkout (%q or %q)", third.dir, held, heldSecond)
	}
	third.close()
}

// fak-test:runtime medium est=85s lane=default
func TestCandidatePoolCapBoundsKeptCheckouts(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }
	parent := filepath.Dir(mustAbs(t, root))

	// Two concurrently OPEN sessions each hold their own pooled checkout: the
	// second cannot adopt the first (lock held) and the cap still has room.
	first := newTopologyCandidateSession(root, defaultGit)
	defer first.close()
	if ok, detail := first.verify("HEAD", "", hook); !ok {
		t.Fatalf("first verify: %s", detail)
	}
	second := newTopologyCandidateSession(root, defaultGit)
	defer second.close()
	if ok, detail := second.verify("HEAD", "", hook); !ok {
		t.Fatalf("second verify: %s", detail)
	}
	if samePath(first.dir, second.dir) {
		t.Fatal("second session adopted a checkout the first still holds")
	}
	if got := livePoolCount(root, parent, defaultGit); got != TopologyCandidatePoolCap {
		t.Fatalf("live pool count = %d, want %d", got, TopologyCandidatePoolCap)
	}

	// A third session finds the pool full and the locks held, so it falls back
	// to a transient checkout, which close removes.
	overflow := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := overflow.verify("HEAD", "", hook); !ok {
		t.Fatalf("overflow session verify: %s", detail)
	}
	if isTopologyCandidatePoolName(filepath.Base(overflow.dir)) {
		t.Fatal("overflow session created another pooled checkout instead of a transient one")
	}
	kept := overflow.dir
	overflow.close()
	if _, err := os.Stat(kept); !os.IsNotExist(err) {
		t.Fatalf("overflow transient %q should be removed on close: %v", kept, err)
	}
	if got := livePoolCount(root, parent, defaultGit); got != TopologyCandidatePoolCap {
		t.Fatalf("live pool count after overflow = %d, want %d", got, TopologyCandidatePoolCap)
	}

	// After both holders close, a new session adopts instead of materializing.
	firstKept, secondKept := first.dir, second.dir
	first.close()
	second.close()
	adopter := newTopologyCandidateSession(root, defaultGit)
	defer adopter.close()
	if ok, detail := adopter.verify("HEAD", "", hook); !ok {
		t.Fatalf("adopter verify: %s", detail)
	}
	if !samePath(adopter.dir, firstKept) && !samePath(adopter.dir, secondKept) {
		t.Fatalf("adopter materialized %q, want adoption of a released pool checkout (%q or %q)", adopter.dir, firstKept, secondKept)
	}
	if got := livePoolCount(root, parent, defaultGit); got != TopologyCandidatePoolCap {
		t.Fatalf("live pool count after adoption = %d, want %d", got, TopologyCandidatePoolCap)
	}
}

// fak-test:runtime medium est=35s lane=default
func TestCandidatePoolDisabledByEnv(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	t.Setenv(CandidatePoolEnv, "off")
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }
	s := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	if isTopologyCandidatePoolName(filepath.Base(s.dir)) {
		t.Fatal("pool-disabled session created a pooled candidate name")
	}
	dir := s.dir
	s.close()
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("transient candidate %q should be removed on close: %v", dir, err)
	}
	if got := registeredPoolCandidates(t, root); len(got) != 0 {
		t.Fatalf("pool registrations remain: %v", got)
	}
}

// fak-test:runtime medium est=95s lane=default
func TestCandidatePoolSkipsOtherRepositorysCandidate(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, parent := newSiblingTopologyRepo(t)
	other := filepath.Join(parent, "other-repo")
	if err := os.MkdirAll(other, 0o755); err != nil {
		t.Fatal(err)
	}
	mustGit(t, other, "init", "-q", "-b", "main")
	mustGit(t, other, "config", "user.email", "trunk@test")
	mustGit(t, other, "config", "user.name", "trunk")
	mustGit(t, other, "commit", "-q", "--allow-empty", "-m", "init")

	foreign := filepath.Join(parent, topologyCandidatePoolPattern()+"foreign")
	foreign = strings.ReplaceAll(foreign, "*", "x")
	mustGit(t, other, "worktree", "add", "--detach", foreign, "HEAD")

	hook := func(dir string) (bool, string) { return true, "" }
	s := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	if samePath(s.dir, foreign) {
		t.Fatal("adopted another repository's pooled candidate")
	}
	s.close()
	if _, err := os.Stat(foreign); err != nil {
		t.Fatalf("foreign repository's candidate was removed: %v", err)
	}
	if got := mustGit(t, other, "worktree", "list", "--porcelain"); !strings.Contains(got, filepath.ToSlash(foreign)) && !strings.Contains(got, foreign) {
		t.Fatalf("foreign registration was pruned: %q", got)
	}
}

// fak-test:runtime medium est=70s lane=default
func TestCandidatePoolRemovesGuttedAdoptable(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	// An owner-GONE name: a gutted checkout whose creator is dead is the shape
	// the adopt path collects; a live creator's marker is respected (covered by
	// TestCandidatePoolRespectsLiveCreatorMarker).
	gutted := filepath.Join(filepath.Dir(mustAbs(t, root)), ".fak-cand-pool-2147483647-0-gutted")
	mustGit(t, root, "worktree", "add", "--detach", gutted, "HEAD")
	if err := os.RemoveAll(gutted); err != nil {
		t.Fatal(err)
	}

	hook := func(dir string) (bool, string) { return true, "" }
	s := newTopologyCandidateSession(root, defaultGit)
	defer s.close()
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	if samePath(s.dir, gutted) {
		t.Fatal("adopted a gutted candidate")
	}
	s.close()
	if got := registeredPoolCandidates(t, root); len(got) != 1 {
		t.Fatalf("gutted registration was not collected; pool registrations = %v", got)
	}
}

// fak-test:runtime medium est=95s lane=default
func TestCandidatePoolMoveToFailureRemovesAdoptable(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }

	keeper := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := keeper.verify("HEAD", "", hook); !ok {
		t.Fatalf("keeper verify: %s", detail)
	}
	kept := keeper.dir
	keeper.close()

	failing := func(dir string, args []string) (int, string) {
		if samePath(dir, kept) && slices.Contains(args, "checkout") {
			return 128, "fatal: injected checkout failure"
		}
		return defaultGit(dir, args)
	}
	s := newTopologyCandidateSession(root, failing)
	defer s.close()
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify after failed adoption: %s", detail)
	}
	if samePath(s.dir, kept) {
		t.Fatal("session kept a candidate whose reset failed")
	}
	s.close()
	if _, err := os.Stat(kept); !os.IsNotExist(err) {
		t.Fatalf("candidate whose moveTo failed should have been removed: %v", err)
	}
	for _, registered := range registeredPoolCandidates(t, root) {
		if samePath(registered, kept) {
			t.Fatalf("stale registration for the removed candidate remains: %v", registeredPoolCandidates(t, root))
		}
	}
}

func registeredPoolCandidates(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	for _, line := range strings.Split(mustGit(t, root, "worktree", "list", "--porcelain"), "\n") {
		if p, ok := strings.CutPrefix(strings.TrimSpace(line), "worktree "); ok && isTopologyCandidatePoolName(filepath.Base(p)) {
			out = append(out, filepath.Clean(p))
		}
	}
	return out
}

func mustReadFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func mustAbs(t *testing.T, path string) string {
	t.Helper()
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

// fak-test:runtime medium est=70s lane=default
func TestCandidatePoolSkipsMidAddOfAliveCreator(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }

	// The mid-add shape, verified empirically on this host: the checkout is
	// registered and carries .git plus git's `locked` marker while the add is
	// still running — and the creator holds the sidecar lock.
	dir := makeMidAddPoolDir(t, root, strings.TrimSuffix(topologyCandidatePoolPattern(), "*")+"midadd")
	sidecar, ok := lockCandidateSidecar(dir)
	if !ok {
		t.Fatal("could not hold the creation sidecar")
	}

	s := newTopologyCandidateSession(root, defaultGit)
	defer s.close()
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	if samePath(s.dir, dir) {
		t.Fatal("adopted a checkout whose creation sidecar is held")
	}
	if !candidateWorktreeLockPresent(dir) {
		t.Fatal("the live creator's mid-add marker was cleared")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("the mid-add checkout was removed: %v", err)
	}
	s.close()

	releaseCandidateSidecar(sidecar)
	removeCandidateWithRegistration(root, dir, defaultGit)
}

// fak-test:runtime medium est=70s lane=default
func TestCandidatePoolAdoptsKilledMidAdd(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }

	// Same shape, but the creator is provably gone: the marker is a leftover
	// git can never clear itself, and the checkout is fully adoptable.
	dir := makeMidAddPoolDir(t, root, ".fak-cand-pool-2147483647-0-deadmidadd")

	adds := 0
	git := func(gitDir string, args []string) (int, string) {
		if slices.Contains(args, "add") && slices.Contains(args, "worktree") {
			adds++
		}
		return defaultGit(gitDir, args)
	}
	s := newTopologyCandidateSession(root, git)
	defer s.close()
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	if !samePath(s.dir, dir) {
		t.Fatalf("did not adopt the killed creator's checkout: got %q", s.dir)
	}
	if candidateWorktreeLockPresent(dir) {
		t.Fatal("the dead creator's initializing marker was not cleared on adoption")
	}
	if adds != 0 {
		t.Fatalf("adoption ran %d worktree adds, want 0", adds)
	}
	s.close()
}

// fak-test:runtime medium est=70s lane=default
func TestCandidatePoolRespectsLiveCreatorMarker(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }

	// Marker present, creator alive, sidecar free: an explicit marker on a
	// pool dir is respected, never cleared by adoption.
	dir := makeMidAddPoolDir(t, root, strings.TrimSuffix(topologyCandidatePoolPattern(), "*")+"marked")

	s := newTopologyCandidateSession(root, defaultGit)
	defer s.close()
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	if samePath(s.dir, dir) {
		t.Fatal("adopted a checkout whose live creator marked it locked")
	}
	if !candidateWorktreeLockPresent(dir) {
		t.Fatal("the live creator's marker was cleared")
	}
	s.close()
	removeCandidateWithRegistration(root, dir, defaultGit)
}

// fak-test:runtime fast est=5s lane=default
func TestCandidatePoolTrimsExcess(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, parent := newSiblingTopologyRepo(t)
	var created []string
	for i := 0; i < TopologyCandidatePoolCap+2; i++ {
		dir := filepath.Join(parent, strings.TrimSuffix(topologyCandidatePoolPattern(), "*")+"trim"+strconv.Itoa(i))
		mustGit(t, root, "worktree", "add", "--detach", dir, "HEAD")
		created = append(created, dir)
		time.Sleep(50 * time.Millisecond) // distinct mtimes: trim removes the oldest
	}
	newest := created[len(created)-1]

	trimPoolExcess(root, parent, defaultGit)

	if got := livePoolCount(root, parent, defaultGit); got != TopologyCandidatePoolCap {
		t.Fatalf("live pool count after trim = %d, want %d", got, TopologyCandidatePoolCap)
	}
	found := false
	for _, reg := range registeredPoolCandidates(t, root) {
		if samePath(reg, newest) {
			found = true
		}
	}
	if !found {
		t.Fatalf("the newest checkout %q was trimmed instead of the oldest", newest)
	}
	for _, dir := range created {
		removeCandidateWithRegistration(root, dir, defaultGit)
	}
}

// fak-test:runtime medium est=70s lane=default
func TestCandidatePoolLockFailureFallsBackToTransient(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }

	orig := lockCandidatePoolDir
	lockCandidatePoolDir = func(string) (*os.File, bool) { return nil, false }
	defer func() { lockCandidatePoolDir = orig }()

	s := newTopologyCandidateSession(root, defaultGit)
	defer s.close()
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	if isTopologyCandidatePoolName(filepath.Base(s.dir)) {
		t.Fatal("session used a pooled checkout it could not lock")
	}
	dir := s.dir
	s.close()
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("transient fallback %q should be removed on close: %v", dir, err)
	}
	// The pooled checkout the session created before losing its lock remains
	// registered for a later session to adopt.
	if got := registeredPoolCandidates(t, root); len(got) != 1 {
		t.Fatalf("abandoned pooled checkout registrations = %v, want exactly one", got)
	}
	for _, reg := range registeredPoolCandidates(t, root) {
		removeCandidateWithRegistration(root, reg, defaultGit)
	}
}

// fak-test:runtime medium est=70s lane=default
func TestCandidatePoolPruneGoneRespectsHeldLock(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, _ := newSiblingTopologyRepo(t)
	hook := func(dir string) (bool, string) { return true, "" }

	dead := filepath.Join(filepath.Dir(mustAbs(t, root)), ".fak-cand-pool-2147483647-0-gonedir")
	mustGit(t, root, "worktree", "add", "--detach", dead, "HEAD")
	admin := mustAdminDir(t, dead)
	if err := os.RemoveAll(dead); err != nil {
		t.Fatal(err)
	}
	lockFile, err := os.OpenFile(filepath.Join(admin, candidatePoolLockName), os.O_CREATE|os.O_RDWR, poolLockFilePerm)
	if err != nil {
		t.Fatal(err)
	}
	if err := flock.TryLock(lockFile); err != nil {
		t.Fatalf("could not hold the gone checkout's flock: %v", err)
	}

	s := newTopologyCandidateSession(root, defaultGit)
	if ok, detail := s.verify("HEAD", "", hook); !ok {
		t.Fatalf("verify: %s", detail)
	}
	stillThere := false
	for _, reg := range registeredPoolCandidates(t, root) {
		if samePath(reg, dead) {
			stillThere = true
		}
	}
	if !stillThere {
		t.Fatal("held-lock registration was collected despite the flock")
	}
	s.close()
	if err := flock.Unlock(lockFile); err != nil {
		t.Fatal(err)
	}
	if err := lockFile.Close(); err != nil {
		t.Fatal(err)
	}

	again := newTopologyCandidateSession(root, defaultGit)
	defer again.close()
	if ok, detail := again.verify("HEAD", "", hook); !ok {
		t.Fatalf("second verify: %s", detail)
	}
	for _, reg := range registeredPoolCandidates(t, root) {
		if samePath(reg, dead) {
			t.Fatalf("released registration was not collected: %v", registeredPoolCandidates(t, root))
		}
	}
	again.close()
}

// makeMidAddPoolDir materializes a pool checkout carrying git's `locked`
// marker and a live-or-dead creator name, without holding its sidecar.
func makeMidAddPoolDir(t *testing.T, root, name string) string {
	t.Helper()
	dir := filepath.Join(filepath.Dir(mustAbs(t, root)), name)
	mustGit(t, root, "worktree", "add", "--detach", dir, "HEAD")
	admin := mustAdminDir(t, dir)
	if err := os.WriteFile(filepath.Join(admin, "locked"), []byte("initializing"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func mustAdminDir(t *testing.T, dir string) string {
	t.Helper()
	admin := candidateAdminDir(dir)
	if admin == "" {
		t.Fatalf("no git admin dir under %q", dir)
	}
	return admin
}
