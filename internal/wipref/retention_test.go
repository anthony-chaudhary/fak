package wipref

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// retention_test.go is the #3873 RETENTION witness: the half of the durable-object
// contract the pure folds cannot prove on their own. Reconcile/Reap decide, but the
// promise "the checkpoint object is still there when you come back for it" is a
// property of git's object store, so it is witnessed against the REAL git binary in
// a throwaway repo — the same shape internal/leaseref proves its refs/fak/locks/
// round-trip with (leaseref_test.go TestRealGitRoundTrip), skipped when git is
// absent so the pure package still tests without one.
//
// Three claims, in the order the retention edge is crossed:
//
//	1. anchored     — a checkpoint reachable only from refs/fak/wip/<session>
//	                  survives `git gc --prune=now`.
//	2. superseded   — the checkpoint a later writer displaced survives the SAME gc,
//	                  because the CAS write carries --create-reflog and the reflog is
//	                  a reachability root. This is what makes a restore of the
//	                  just-overwritten snapshot possible instead of a lottery.
//	3. reaped       — once Reap's DELETE verdict removes the ref, the objects become
//	                  gc-eligible and the next gc drops them. The retention edge ENDS
//	                  at ref deletion, which is what bounds the namespace's growth.

// retentionGit runs git in dir under hermetic config and returns trimmed stdout.
// The repo-local gc.* settings pinned by initRetentionRepo are what make the three
// claims deterministic: a host whose global config said gc.pruneExpire=never would
// otherwise let claim 3 pass for the wrong reason.
func retentionGit(t *testing.T, dir, stdin string, args ...string) string {
	t.Helper()
	out, err := retentionGitTry(t, dir, stdin, args...)
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return out
}

// retentionGitTry is retentionGit without the fatal: it returns git's exit error so
// a test can assert that a COMPARE-AND-SWAP was correctly REFUSED.
func retentionGitTry(t *testing.T, dir, stdin string, args ...string) (string, error) {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	if stdin != "" {
		c.Stdin = strings.NewReader(stdin)
	}
	c.Env = append(os.Environ(),
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL="+filepath.Join(dir, "absent-global-config"),
		"GIT_CONFIG_SYSTEM="+filepath.Join(dir, "absent-system-config"),
		"HOME="+dir,
	)
	out, err := c.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

func initRetentionRepo(t *testing.T, dir string) {
	t.Helper()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "wip@test.local"},
		{"config", "user.name", "wip test"},
		// Pin every knob the three claims depend on, so the witness grades git's
		// behaviour and not the host's ~/.gitconfig.
		{"config", "gc.auto", "0"},
		{"config", "gc.pruneExpire", "now"},
		{"config", "gc.reflogExpire", "90.days"},
		{"config", "gc.reflogExpireUnreachable", "30.days"},
	} {
		retentionGit(t, dir, "", args...)
	}
}

// retentionZeroOID is the update-ref old-value sentinel for "must not exist".
func retentionZeroOID(t *testing.T, dir string) string {
	t.Helper()
	if retentionGit(t, dir, "", "rev-parse", "--show-object-format") == "sha256" {
		return strings.Repeat("0", 64)
	}
	return strings.Repeat("0", 40)
}

// retentionCheckpoint mints one checkpoint object the way `fak wip checkpoint` does:
// a blob -> a one-entry tree -> a commit over parent carrying the stamp line. It
// returns the commit and its blob, so a claim can assert the DELTA's bytes survived,
// not merely the commit that names them.
func retentionCheckpoint(t *testing.T, dir, parent string, stamp Stamp, body string) (commit, blob string) {
	t.Helper()
	blob = retentionGit(t, dir, body, "hash-object", "-w", "--stdin")
	tree := retentionGit(t, dir, "100644 blob "+blob+"\tnote.txt\n", "mktree")
	msg, err := EncodeStamp(stamp)
	if err != nil {
		t.Fatalf("EncodeStamp: %v", err)
	}
	args := []string{"commit-tree", tree, "-m", msg}
	if parent != "" {
		args = append(args, "-p", parent)
	}
	return retentionGit(t, dir, "", args...), blob
}

// retentionAlive reports whether the object is still in the store.
func retentionAlive(t *testing.T, dir, oid string) bool {
	t.Helper()
	_, err := retentionGitTry(t, dir, "", "cat-file", "-e", oid+"^{object}")
	return err == nil
}

// TestCheckpointObjectRetentionAcrossGC is the #3873 done-condition witness: the
// checkpoint object survives `git gc` until its ref is deleted — and no longer.
func TestCheckpointObjectRetentionAcrossGC(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	initRetentionRepo(t, dir)

	const session = "sess-3873"
	ref := SessionRef(session)
	zero := retentionZeroOID(t, dir)

	// A base commit for HEAD, so the checkpoints hang off a real history the way a
	// captured working-tree delta does.
	base, _ := retentionCheckpoint(t, dir, "", Stamp{SessionID: "base", CheckpointedAt: 1}, "base line\n")
	retentionGit(t, dir, "", "update-ref", "HEAD", base)

	// ---- claim 1: an anchored checkpoint survives gc ----
	first, firstBlob := retentionCheckpoint(t, dir, base,
		Stamp{SessionID: session, StartSHA: base, CheckpointedAt: 1000}, "delta one\n")
	// The CAS create: git's zero-OID old value means "must not already exist", so
	// even the first anchor fails closed against a racing peer.
	retentionGit(t, dir, "", "update-ref", "--create-reflog", ref, first, zero)

	reflog := filepath.Join(dir, ".git", "logs", "refs", "fak", "wip", session)
	if _, err := os.Stat(reflog); err != nil {
		t.Fatalf("no reflog at %s after --create-reflog: %v — the reflog retention arm is not engaged", reflog, err)
	}

	retentionGit(t, dir, "", "gc", "--prune=now", "--quiet")
	if !retentionAlive(t, dir, first) {
		t.Fatal("claim 1: the anchored checkpoint commit was gc'd while its ref still pointed at it")
	}
	if !retentionAlive(t, dir, firstBlob) {
		t.Fatal("claim 1: the anchored checkpoint's delta blob was gc'd while its ref still pointed at it")
	}

	// ---- claim 2: a superseded checkpoint survives the same gc via its reflog ----
	second, secondBlob := retentionCheckpoint(t, dir, base,
		Stamp{SessionID: session, StartSHA: base, CheckpointedAt: 2000}, "delta two\n")
	cur := RefRecord{Ref: ref, Object: first, Stamp: Stamp{SessionID: session, CheckpointedAt: 1000}}
	cand := RefRecord{Ref: ref, Object: second, Stamp: Stamp{SessionID: session, CheckpointedAt: 2000}}
	if _, changed := Reconcile(cur, cand); !changed {
		t.Fatal("Reconcile refused a strictly newer checkpoint — the witness below would prove nothing")
	}
	retentionGit(t, dir, "", "update-ref", "--create-reflog", ref, second, first)

	// No torn ref: a STALE writer whose old-value is the superseded object must be
	// refused by git, not merged into it. This is the real-git half of the
	// concurrent-writer claim the pure table test models in reap_test.go. The argv is
	// deliberately shape-identical to the successful CAS above and differs ONLY in the
	// old value, so a refusal here can be attributed to the stale old value and not to
	// some other malformation — the trap a vacuous "it errored, therefore CAS works"
	// assertion falls into.
	stale, _ := retentionCheckpoint(t, dir, base,
		Stamp{SessionID: session, StartSHA: base, CheckpointedAt: 1500}, "stale delta\n")
	if _, err := retentionGitTry(t, dir, "", "update-ref", "--create-reflog", ref, stale, first); err == nil {
		t.Fatal("a stale-old-value update-ref succeeded: the ref is NOT compare-and-swap protected")
	}
	if got := retentionGit(t, dir, "", "rev-parse", ref); got != second {
		t.Fatalf("ref converged to %s, want the last writer %s", got, second)
	}

	retentionGit(t, dir, "", "gc", "--prune=now", "--quiet")
	if !retentionAlive(t, dir, second) {
		t.Fatal("claim 2: the current checkpoint was gc'd while its ref pointed at it")
	}
	if !retentionAlive(t, dir, first) || !retentionAlive(t, dir, firstBlob) {
		t.Fatal("claim 2: the superseded checkpoint was gc'd — the reflog retention window did not hold it, so a restore of the just-overwritten snapshot would find nothing")
	}

	// ---- claim 3: the retention edge ends at the reap ----
	recs := []RefRecord{{Ref: ref, Object: second, Stamp: Stamp{SessionID: session, CheckpointedAt: 2000}}}
	verdicts := Reap(recs, map[string]OwnerState{session: OwnerLanded})
	if len(verdicts) != 1 || verdicts[0].Action != ReapDelete {
		t.Fatalf("Reap of a LANDED owner = %+v, want one DELETE verdict", verdicts)
	}
	// The delete is itself an old-value CAS, so a ref a concurrent checkpoint
	// advanced since the listing is left alone rather than reaped out from under it.
	retentionGit(t, dir, "", "update-ref", "-d", verdicts[0].Ref, verdicts[0].Object)
	retentionGit(t, dir, "", "gc", "--prune=now", "--quiet")

	for _, o := range []struct {
		name string
		oid  string
	}{{"current checkpoint", second}, {"current delta blob", secondBlob},
		{"superseded checkpoint", first}, {"superseded delta blob", firstBlob}} {
		if retentionAlive(t, dir, o.oid) {
			t.Errorf("claim 3: the %s survived gc after its ref was reaped — the WIP namespace's objects are unbounded", o.name)
		}
	}
}
