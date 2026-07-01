package relay

import (
	"errors"
	"testing"
)

// D1 (issue #1877) done condition: a cursor over a matching git state returns fresh and
// one over a diverged git state returns stale. These are that witness (run: `go test
// ./internal/relay -run ReloadVerify`).

// fakeResolver is a hermetic Resolver: it verifies exactly the refs in its set and calls
// everything else dangling, with an optional forced error to model an unreachable store.
type fakeResolver struct {
	verified map[string]bool
	err      error
}

func (f fakeResolver) Resolve(a Artifact) Resolution {
	if f.err != nil {
		return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: f.err.Error()}
	}
	if f.verified[a.Ref] {
		return Resolution{Artifact: a, Verdict: ResolveVerified, Detail: "known " + a.Ref}
	}
	return Resolution{Artifact: a, Verdict: ResolveDangling, Detail: "unknown " + a.Ref}
}

// TestReloadVerifyFreshVsStale drives the done condition plus the fail-closed edges with a
// hermetic resolver: a matching anchor is fresh; a diverged anchor, an empty anchor, and an
// unreachable store are all stale.
func TestReloadVerifyFreshVsStale(t *testing.T) {
	const anchor = "0123456789abcdef0123456789abcdef01234567"
	matching := fakeResolver{verified: map[string]bool{anchor: true}}
	diverged := fakeResolver{verified: map[string]bool{}} // anchor no longer resolves

	if got := VerifyReload(ProgressCursor{StartSHA: anchor}, matching); got.Verdict != ReloadFresh {
		t.Errorf("matching git state: verdict = %q, want fresh (reason=%s)", got.Verdict, got.Reason)
	}
	if got := VerifyReload(ProgressCursor{StartSHA: anchor}, diverged); got.Verdict != ReloadStale {
		t.Errorf("diverged git state: verdict = %q, want stale (reason=%s)", got.Verdict, got.Reason)
	}
	if got := VerifyReload(ProgressCursor{StartSHA: ""}, matching); got.Verdict != ReloadStale {
		t.Errorf("empty anchor: verdict = %q, want stale (reason=%s)", got.Verdict, got.Reason)
	}
	unreachable := fakeResolver{err: errors.New("git unreachable")}
	if got := VerifyReload(ProgressCursor{StartSHA: anchor}, unreachable); got.Verdict != ReloadStale {
		t.Errorf("unreachable store must fail closed to stale: verdict = %q (reason=%s)", got.Verdict, got.Reason)
	}
}

// TestReloadVerifyAgainstRepo witnesses the verifier over the real git-backed resolver:
// a cursor anchored at HEAD is fresh, and one anchored at an all-zero SHA is stale.
func TestReloadVerifyAgainstRepo(t *testing.T) {
	r := NewCommitResolver(GitCommitExists("."))

	head := VerifyReload(ProgressCursor{StartSHA: "HEAD"}, r)
	if head.Reason == "" {
		t.Fatal("expected a reason string")
	}
	// If git is unavailable the resolver returns Unknown and VerifyReload is stale; only
	// assert the positive case when git actually resolved HEAD.
	if headResolves := NewCommitResolver(GitCommitExists(".")).Resolve(Artifact{Kind: string(ArtifactCommit), Ref: "HEAD"}); headResolves.Verdict == ResolveVerified {
		if head.Verdict != ReloadFresh {
			t.Errorf("HEAD anchor should be fresh: verdict = %q (reason=%s)", head.Verdict, head.Reason)
		}
	} else {
		t.Skipf("git unavailable in this environment: %s", headResolves.Detail)
	}

	bogus := VerifyReload(ProgressCursor{StartSHA: "0000000000000000000000000000000000000000"}, r)
	if bogus.Verdict != ReloadStale {
		t.Errorf("all-zero anchor should be stale: verdict = %q (reason=%s)", bogus.Verdict, bogus.Reason)
	}
}
