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
// answers with a per-verb exit code (default 0) or a hard exec error.
type syncRec struct {
	calls [][]string
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
		return refPrefix + "seed\n", r.code[args[0]], nil
	}
	return "", r.code[args[0]], nil
}

// wantRefspec is the literal refspec the sync contract promises: the whole locks
// namespace, forced, confined to refs/fak/locks/* on BOTH ends. Asserted as a
// literal so a refactor of the constant cannot silently widen the blast radius.
const wantRefspec = "+refs/fak/locks/*:refs/fak/locks/*"

func TestSyncPushThenFetchExactArgv(t *testing.T) {
	rec := &syncRec{}
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, true)
	if err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if !res.Pushed || !res.Fetched {
		t.Fatalf("want Pushed && Fetched, got %+v", res)
	}
	if res.Refspec != wantRefspec {
		t.Fatalf("refspec = %q, want %q", res.Refspec, wantRefspec)
	}
	want := [][]string{
		{"for-each-ref", "--format=%(refname)", refPrefix},
		{"push", "origin", wantRefspec},
		{"fetch", "origin", wantRefspec},
	}
	if len(rec.calls) != len(want) {
		t.Fatalf("calls = %v, want %v", rec.calls, want)
	}
	for i := range want {
		if strings.Join(rec.calls[i], " ") != strings.Join(want[i], " ") {
			t.Fatalf("call %d = %v, want %v (push must precede fetch)", i, rec.calls[i], want[i])
		}
	}
}

func TestSyncSingleDirection(t *testing.T) {
	for _, tc := range []struct {
		name        string
		push, fetch bool
		wantVerb    string
		wantPushed  bool
		wantFetched bool
	}{
		{"push-only", true, false, "push", true, false},
		{"fetch-only", false, true, "fetch", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &syncRec{}
			s := NewWithRunner(rec.run, "")
			res, err := s.Sync(context.Background(), "origin", tc.push, tc.fetch)
			if err != nil {
				t.Fatalf("Sync: %v", err)
			}
			if res.Pushed != tc.wantPushed || res.Fetched != tc.wantFetched {
				t.Fatalf("got %+v, want pushed=%v fetched=%v", res, tc.wantPushed, tc.wantFetched)
			}
			wantCalls := 1
			if tc.push {
				wantCalls = 2
			}
			if len(rec.calls) != wantCalls || rec.calls[len(rec.calls)-1][0] != tc.wantVerb {
				t.Fatalf("calls = %v, want %q as the final call", rec.calls, tc.wantVerb)
			}
		})
	}
}

// A failed push STOPS the sync: the fetch must never run, because force-fetching
// would overwrite the very local records that just failed to publish.
func TestSyncPushFailureStopsBeforeFetch(t *testing.T) {
	rec := &syncRec{code: map[string]int{"push": 1}}
	s := NewWithRunner(rec.run, "")
	res, err := s.Sync(context.Background(), "origin", true, true)
	if err == nil {
		t.Fatal("want error on push exit 1, got nil")
	}
	if res.Pushed || res.Fetched {
		t.Fatalf("nothing should be marked done after a failed push, got %+v", res)
	}
	if len(rec.calls) != 2 || rec.calls[0][0] != "for-each-ref" || rec.calls[1][0] != "push" {
		t.Fatalf("calls = %v, want the ref probe then push only — a failed push must not be followed by a force-fetch", rec.calls)
	}
}

func TestSyncFetchFailureAfterCleanPush(t *testing.T) {
	rec := &syncRec{code: map[string]int{"fetch": 128}}
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

// An empty lease namespace is normal for a fresh clone. Git treats a wildcard
// push with no matching source refs as an error, so Sync must skip that push
// rather than turn an otherwise clean convergence tick into a transport failure.
func TestSyncPushOnlyEmptyNamespaceIsCleanNoOp(t *testing.T) {
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
	runGit(root, "init", local)
	runGit(local, "config", "user.name", "fak-test")
	runGit(local, "config", "user.email", "fak-test@example.invalid")
	if err := os.WriteFile(filepath.Join(local, "README"), []byte("seed\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	runGit(local, "add", "README")
	runGit(local, "commit", "-m", "seed")
	runGit(local, "remote", "add", "origin", remote)
	runGit(local, "push", "origin", "HEAD:main")

	got, err := NewInDir(local).Sync(ctx, "origin", true, false)
	if err != nil {
		t.Fatalf("empty lease namespace must be a clean no-op: %v", err)
	}
	if got.Pushed {
		t.Fatalf("empty lease namespace should skip the git push, got %+v", got)
	}
	if refs := runGit(local, "for-each-ref", "--format=%(refname)", refPrefix); refs != "" {
		t.Fatalf("empty lease namespace changed locally: %q", refs)
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
