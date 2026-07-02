package leaseref

// sync_test.go drives Store.Sync through the injected Runner seam with a canned
// recorder — the same no-real-git discipline as the rest of the package's tests —
// and asserts the EXACT argv issued, the push-before-fetch order, and the
// fail-fast-on-push contract (never force-fetch over unpublished local leases).

import (
	"context"
	"errors"
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
	r.calls = append(r.calls, args)
	if r.err != nil {
		return "", -1, r.err
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
			if len(rec.calls) != 1 || rec.calls[0][0] != tc.wantVerb {
				t.Fatalf("calls = %v, want exactly one %q", rec.calls, tc.wantVerb)
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
	if len(rec.calls) != 1 || rec.calls[0][0] != "push" {
		t.Fatalf("calls = %v, want the push only — a failed push must not be followed by a force-fetch", rec.calls)
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
