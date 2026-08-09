package leaseref

// sync_test.go drives Store.Sync through the injected Runner seam with a canned
// recorder — the same no-real-git discipline as the rest of the package's tests —
// and asserts the EXACT argv issued, the push-before-fetch order, and the
// fail-fast-on-push contract (never force-fetch over unpublished local leases).

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// syncRec is a minimal Runner recorder for the sync tests: it logs every argv and
// answers with a per-verb exit code (default 0) or a hard exec error. refs is the
// clone's LOCAL refs/fak/locks/ population, which the pre-push emptiness probe
// (for-each-ref) reads — an empty refs models a fresh clone that holds no lease.
type syncRec struct {
	calls [][]string
	refs  []string       // local refs the for-each-ref probe reports
	code  map[string]int // args[0] -> exit code
	err   error          // non-nil = git not executable
}

func (r *syncRec) run(ctx context.Context, dir string, args ...string) (string, int, error) {
	if len(args) > 1 && args[0] == "rev-parse" && args[1] == "--path-format=absolute" {
		return "", 0, nil
	}
	r.calls = append(r.calls, args)
	if r.err != nil {
		return "", -1, r.err
	}
	if args[0] == "for-each-ref" {
		return strings.Join(r.refs, "\n"), r.code[args[0]], nil
	}
	return "", r.code[args[0]], nil
}

// wantRefspec is the literal refspec the sync contract promises: the whole locks
// namespace, forced, confined to refs/fak/locks/* on BOTH ends. Asserted as a
// literal so a refactor of the constant cannot silently widen the blast radius.
const wantRefspec = "+refs/fak/locks/*:refs/fak/locks/*"

// wantProbe is the literal pre-push emptiness probe: does this clone hold ANY ref under
// refs/fak/locks/? Asserted as a literal because the whole #5550 fix rests on it being a
// POSITIVE local enumeration — if it ever degrades into inspecting a push failure
// instead, these argv assertions are what notices.
var wantProbe = []string{"for-each-ref", "--count=1", "--format=%(refname)", "refs/fak/locks/"}

// oneLease is the local population of a clone that DOES hold a lease — the state in
// which the push must actually run.
func oneLease() []string { return []string{refPrefix + "lane"} }

// assertArgv compares a recorded call log against the exact argv sequence expected.
func assertArgv(t *testing.T, got, want [][]string, why string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("calls = %v, want %v (%s)", got, want, why)
	}
	for i := range want {
		if strings.Join(got[i], " ") != strings.Join(want[i], " ") {
			t.Fatalf("call %d = %v, want %v (%s)", i, got[i], want[i], why)
		}
	}
}

func TestSyncPushThenFetchExactArgv(t *testing.T) {
	rec := &syncRec{refs: oneLease()}
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, true)
	if err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if !res.Pushed || !res.Fetched {
		t.Fatalf("want Pushed && Fetched, got %+v", res)
	}
	if res.PushSkippedEmpty {
		t.Fatalf("a clone holding a lease must PUSH it, not report an empty skip: %+v", res)
	}
	if res.Refspec != wantRefspec {
		t.Fatalf("refspec = %q, want %q", res.Refspec, wantRefspec)
	}
	assertArgv(t, rec.calls, [][]string{
		wantProbe,
		{"push", "origin", wantRefspec},
		{"fetch", "origin", wantRefspec},
	}, "the emptiness probe precedes the push, and the push precedes the fetch")
}

func TestSyncSingleDirection(t *testing.T) {
	for _, tc := range []struct {
		name        string
		push, fetch bool
		wantCalls   [][]string
		wantPushed  bool
		wantFetched bool
	}{
		{"push-only", true, false, [][]string{wantProbe, {"push", "origin", wantRefspec}}, true, false},
		{"fetch-only", false, true, [][]string{{"fetch", "origin", wantRefspec}}, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &syncRec{refs: oneLease()}
			s := NewWithRunner(rec.run, "")
			res, err := s.Sync(context.Background(), "origin", tc.push, tc.fetch)
			if err != nil {
				t.Fatalf("Sync: %v", err)
			}
			if res.Pushed != tc.wantPushed || res.Fetched != tc.wantFetched {
				t.Fatalf("got %+v, want pushed=%v fetched=%v", res, tc.wantPushed, tc.wantFetched)
			}
			// A fetch-only sync must not probe: the probe exists to protect the PUSH.
			assertArgv(t, rec.calls, tc.wantCalls, "only the requested direction (plus the push's own probe) may run")
		})
	}
}

// A failed push STOPS the sync: the fetch must never run, because force-fetching
// would overwrite the very local records that just failed to publish.
//
// This is also the ANTI-REGRESSION half of #5550. The empty-namespace skip must not
// become a blanket forgiveness of exit 1: here the namespace is NOT empty, so a push
// exit 1 is a real transport/auth/rejection failure and must still error AND still stop
// the sync. If this test ever goes green while reporting Fetched, the fix has started
// swallowing genuine failures and stranding lease state.
func TestSyncPushFailureStopsBeforeFetch(t *testing.T) {
	rec := &syncRec{refs: oneLease(), code: map[string]int{"push": 1}}
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, true)
	if err == nil {
		t.Fatal("want error on push exit 1, got nil")
	}
	if res.Pushed || res.Fetched || res.PushSkippedEmpty {
		t.Fatalf("nothing should be marked done (or excused) after a failed push, got %+v", res)
	}
	assertArgv(t, rec.calls, [][]string{wantProbe, {"push", "origin", wantRefspec}},
		"a failed push must not be followed by a force-fetch")
}

// The skip is authorized ONLY by a clean enumeration. When for-each-ref itself fails the
// answer is unknown, and unknown must fall through to the push — never to a skip, which
// would silently decline to publish leases this clone may well be holding.
func TestSyncUnreadableNamespaceStillPushes(t *testing.T) {
	rec := &syncRec{code: map[string]int{"for-each-ref": 1, "push": 1}}
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, true)
	if err == nil {
		t.Fatal("want the real push failure to surface when the namespace could not be enumerated")
	}
	if res.PushSkippedEmpty {
		t.Fatalf("an unreadable namespace is UNKNOWN, not empty — must not skip the push: %+v", res)
	}
	assertArgv(t, rec.calls, [][]string{wantProbe, {"push", "origin", wantRefspec}},
		"an unreadable probe falls through to the push")
}

// #5550: a clone holding ZERO lease refs has nothing to send. A zero-match push refspec
// exits 1 in git, and since a failed push stops the sync, such a clone could never reach
// the fetch — so it could never learn a peer's leases, and stayed a zero-lease clone
// forever. The push is skipped, no error is reported, and — the whole point — THE FETCH
// STILL RUNS.
func TestSyncEmptyNamespaceSkipsPushAndStillFetches(t *testing.T) {
	rec := &syncRec{} // fresh clone: no refs under refs/fak/locks/
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, true)
	if err != nil {
		t.Fatalf("Sync on an empty namespace: %v, want nil (nothing to send is not a failure)", err)
	}
	if res.Pushed {
		t.Fatalf("Pushed must stay honest — no push ran: %+v", res)
	}
	if !res.PushSkippedEmpty {
		t.Fatalf("want PushSkippedEmpty to record the no-op, got %+v", res)
	}
	if !res.Fetched {
		t.Fatalf("Fetched=false: the fetch is the whole point — a zero-lease clone must still learn its peers': %+v", res)
	}
	assertArgv(t, rec.calls, [][]string{wantProbe, {"fetch", "origin", wantRefspec}},
		"no push subprocess may run, and the fetch must still happen")
}

// The empty-namespace skip is confined to the PUSH direction: a push-only sync of an
// empty namespace reports the honest no-op and never claims a push.
func TestSyncEmptyNamespacePushOnlyIsACleanNoOp(t *testing.T) {
	rec := &syncRec{}
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, false)
	if err != nil {
		t.Fatalf("push-only Sync on an empty namespace: %v, want nil", err)
	}
	if res.Pushed || res.Fetched || !res.PushSkippedEmpty {
		t.Fatalf("got %+v, want Pushed=false Fetched=false PushSkippedEmpty=true", res)
	}
	assertArgv(t, rec.calls, [][]string{wantProbe}, "the probe alone answers a push-only sync of an empty namespace")
}

func TestSyncFetchFailureAfterCleanPush(t *testing.T) {
	rec := &syncRec{refs: oneLease(), code: map[string]int{"fetch": 128}}
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, true)
	if err == nil {
		t.Fatal("want error on fetch exit 128, got nil")
	}
	if !res.Pushed || res.Fetched {
		t.Fatalf("want Pushed=true Fetched=false after a clean push + failed fetch, got %+v", res)
	}
}

func TestSyncGitNotExecutable(t *testing.T) {
	rec := &syncRec{err: errors.New("exec: git not found")}
	s := NewWithRunner(rec.run, "")
	if _, err := s.Sync(context.Background(), "origin", true, true); err == nil {
		t.Fatal("want a hard error when git cannot be executed")
	}
}

// Argv hygiene: a remote that could misparse as a flag or smuggle extra tokens is
// refused BEFORE any git call; so is the neither-direction no-op.
func TestSyncRefusesUnsafeRemoteAndNoDirection(t *testing.T) {
	for _, remote := range []string{"", "-evil", "--force", "ori gin", "o\trigin"} {
		rec := &syncRec{}
		s := NewWithRunner(rec.run, "")
		if _, err := s.Sync(context.Background(), remote, true, true); err == nil {
			t.Fatalf("remote %q: want refusal, got nil", remote)
		}
		if len(rec.calls) != 0 {
			t.Fatalf("remote %q: git must not be invoked, got calls %v", remote, rec.calls)
		}
	}
	rec := &syncRec{}
	s := NewWithRunner(rec.run, "")
	if _, err := s.Sync(context.Background(), "origin", false, false); err == nil {
		t.Fatal("want refusal when neither push nor fetch is enabled")
	}
	if len(rec.calls) != 0 {
		t.Fatalf("no-direction sync must not invoke git, got %v", rec.calls)
	}
}

func TestSyncQuarantinesMalformedLooseRefAndFetches(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	ctx := context.Background()
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	local := filepath.Join(root, "local")
	runGit := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v in %s: %v\n%s", args, dir, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	runGit(root, "init", "--bare", remote)
	runGit(root, "clone", remote, local)
	runGit(local, "-c", "user.name=fak-test", "-c", "user.email=fak-test@example.invalid", "commit", "--allow-empty", "-m", "seed")
	runGit(local, "push", "origin", "HEAD:main")

	validBlob := filepath.Join(root, "valid.json")
	if err := os.WriteFile(validBlob, []byte(`{"ok":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	validOID := runGit(local, "hash-object", "-w", validBlob)
	runGit(local, "update-ref", refPrefix+"valid", validOID)
	common := runGit(local, "rev-parse", "--path-format=absolute", "--git-common-dir")
	badPath := filepath.Join(common, "refs", "fak", "locks", "session-bad")
	if err := os.MkdirAll(filepath.Dir(badPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(badPath, make([]byte, 41), 0o600); err != nil {
		t.Fatal(err)
	}

	cmd := exec.CommandContext(ctx, "git", "fetch", "origin", "main")
	cmd.Dir = local
	if out, err := cmd.CombinedOutput(); err == nil {
		t.Fatalf("raw fetch unexpectedly accepted malformed ref: %s", out)
	}
	got, err := NewInDir(local).Sync(ctx, "origin", false, true)
	if err != nil {
		t.Fatalf("Sync recovery: %v", err)
	}
	if !got.Fetched {
		t.Fatalf("Fetched=false: %+v", got)
	}
	if _, err := os.Stat(badPath); !os.IsNotExist(err) {
		t.Fatalf("malformed ref still present: %v", err)
	}
	if gotOID := runGit(local, "rev-parse", refPrefix+"valid"); gotOID != validOID {
		t.Fatalf("valid ref changed: got %s want %s", gotOID, validOID)
	}
	matches, err := filepath.Glob(filepath.Join(common, "fak", "quarantine", "malformed-lock-refs", "*", "session-bad"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("quarantine matches=%v err=%v", matches, err)
	}
}

// syncRealGitFresh builds a #5550 scenario inside t.TempDir(): a throwaway BARE remote in
// one of three shapes, and a FRESH repo that holds zero lease refs and syncs against it.
// No real fleet remote is ever touched — the "remote" is a local path, exactly as the
// sibling real-git tests here and in drain_test.go do.
//
//   - peerPublishes: a second repo takes a lease and publishes it, so the remote's lock
//     namespace is non-empty and the fresh repo has something to LEARN.
//   - sharesABranch: the remote also carries an ordinary branch that the fresh repo has,
//     i.e. the everyday clone.
//
// Both false leaves the remote advertising NOTHING — the shape that actually breaks.
func syncRealGitFresh(t *testing.T, peerPublishes, sharesABranch bool) (fresh string) {
	t.Helper()
	ctx := context.Background()
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	peer := filepath.Join(root, "peer")
	fresh = filepath.Join(root, "fresh")
	runGit := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v in %s: %v\n%s", args, dir, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	runGit(root, "init", "--bare", remote)
	runGit(root, "init", fresh)
	runGit(fresh, "remote", "add", "origin", filepath.ToSlash(remote))
	if peerPublishes || sharesABranch {
		runGit(root, "init", peer)
		runGit(peer, "remote", "add", "origin", filepath.ToSlash(remote))
	}
	if sharesABranch {
		runGit(peer, "-c", "user.name=fak-test", "-c", "user.email=fak-test@example.invalid",
			"commit", "--allow-empty", "-m", "seed")
		runGit(peer, "push", "origin", "HEAD:main")
		runGit(fresh, "fetch", "origin", "main:refs/heads/main")
	}
	if peerPublishes {
		peerStore := NewInDir(peer)
		if _, err := peerStore.Acquire(ctx, Record{ID: "peer-lane", TreeGlobs: []string{"x/**"}, Holder: "peer-node"}); err != nil {
			t.Fatalf("peer acquire: %v", err)
		}
		if res, err := peerStore.Sync(ctx, "origin", true, false); err != nil || !res.Pushed {
			t.Fatalf("peer publish: res=%+v err=%v", res, err)
		}
	}
	if got := runGit(fresh, "for-each-ref", "--format=%(refname)", refPrefix); got != "" {
		t.Fatalf("precondition: the fresh repo already holds lock refs: %q", got)
	}
	return fresh
}

// TestSyncEmptyNamespaceRealGit is the end-to-end #5550 witness: a machine holding ZERO
// refs/fak/locks/* syncs cleanly against a real (throwaway) remote, and — the half that
// matters — comes back holding the peer's lease. Before the fix the zero-match push could
// exit 1, and because a failed push STOPS the sync the fetch never ran, so a clone with no
// leases could never learn any. The failure perpetuated itself.
//
// The three shapes also pin WHEN git's zero-match push actually fails, which is NARROWER
// than #5550 reported. Measured here on git 2.51.0.windows.1 (the same version the ticket
// cites), pushing +refs/fak/locks/*:refs/fak/locks/* from a repo holding none of them:
//
//   - remote-advertises-nothing (a brand-new bare remote — the shape the sibling wipref
//     harness used, cmd/fak/wip_sync_test.go): exit 1, "fatal: the remote end hung up
//     unexpectedly / No refs in common and none specified; doing nothing." The reported bug.
//   - remote-holds-only-peer-leases (the remote has refs, just none this repo shares):
//     exit 0, "Everything up-to-date".
//   - shares-a-branch (an everyday clone whose main the remote also has): exit 0.
//
// So "the state of every fresh clone" overstates it: a fresh clone of a repo that already
// has a branch does NOT hit this. What does hit it is a remote with nothing in common —
// and, whatever git's exit, pushing a refspec with nothing to send was always a pointless
// subprocess. No subtest ASSERTS git's exit code or wording: that is UI, not contract, and
// pinning it would hostage this test to git's release notes. The raw probe is logged as
// evidence; every assertion is about leaseref's own behaviour.
func TestSyncEmptyNamespaceRealGit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	for _, tc := range []struct {
		name                         string
		peerPublishes, sharesABranch bool
	}{
		{"remote-advertises-nothing", false, false},
		{"remote-holds-only-peer-leases", true, false},
		{"shares-a-branch", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			fresh := syncRealGitFresh(t, tc.peerPublishes, tc.sharesABranch)

			// Evidence: the raw push this sync USED to run from an empty namespace.
			raw := exec.CommandContext(ctx, "git", "push", "origin", syncRefspec)
			raw.Dir = fresh
			rawOut, rawErr := raw.CombinedOutput()
			t.Logf("raw zero-match push: err=%v output=%q", rawErr, strings.TrimSpace(string(rawOut)))

			res, err := NewInDir(fresh).Sync(ctx, "origin", true, true)
			if err != nil {
				t.Fatalf("Sync from a zero-lease repo: %v, want nil — nothing to send is not a failure", err)
			}
			if res.Pushed || !res.PushSkippedEmpty || !res.Fetched {
				t.Fatalf("got %+v, want Pushed=false PushSkippedEmpty=true Fetched=true", res)
			}
			got, ok, err := NewInDir(fresh).Get(ctx, "peer-lane")
			if err != nil {
				t.Fatalf("read after sync: %v", err)
			}
			if ok != tc.peerPublishes {
				t.Fatalf("learned the peer's lease = %v, want %v — the fetch must still run and must actually converge", ok, tc.peerPublishes)
			}
			if ok && got.Holder != "peer-node" {
				t.Fatalf("learned record = %+v, want the peer's holder", got)
			}
		})
	}
}

func TestQuarantineMalformedLockRefsInDirBoundsAndPreservesValid(t *testing.T) {
	refs := t.TempDir()
	quarantine := filepath.Join(t.TempDir(), "q")
	valid := strings.Repeat("a", 40) + "\n"
	for name, body := range map[string]string{"valid": valid, "allzero": strings.Repeat("0", 40) + "\n", "badnul": string(make([]byte, 41)), "short": "deadbeef\n", "active.lock": "busy"} {
		if err := os.WriteFile(filepath.Join(refs, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	moved, err := QuarantineMalformedLockRefsInDir(refs, quarantine)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(moved, ",") != "allzero,badnul,short" {
		t.Fatalf("moved=%v", moved)
	}
	for _, name := range []string{"valid", "active.lock"} {
		if _, err := os.Stat(filepath.Join(refs, name)); err != nil {
			t.Fatalf("%s not preserved: %v", name, err)
		}
	}
	for _, name := range moved {
		if _, err := os.Stat(filepath.Join(quarantine, name)); err != nil {
			t.Fatalf("%s not quarantined: %v", name, err)
		}
	}
}
