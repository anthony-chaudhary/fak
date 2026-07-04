package relay

import (
	"bytes"
	"encoding/json"
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

// J5 (issue #1907) done condition: two reloads of the same durable store + same baton yield
// a BYTE-IDENTICAL reload plan. A reload is a projection of (baton, store) — the successor
// re-reads the durable pointers and re-derives its plan — never a fresh model sample, so the
// same inputs must always produce the same plan bytes. TestReloadDeterminism is that witness
// (run: `go test ./internal/relay -run ReloadDeterminism`). It is hermetic (the resolver is
// the in-memory fakeResolver), so it pins the projection's determinism with no cross-host
// variance (explicitly out of scope for this rung).

// reloadPlan is the aggregate a successor computes when it reloads a baton against durable
// stores: the canonical baton (C2 codec), the D1 cursor verdict, the D-stale outcome, and
// the C3 per-artifact resolutions in baton order. The struct tree has no Go maps, so
// json.Marshal over it is byte-stable — any nondeterminism this test observes therefore comes
// from the reload projection itself (VerifyReload / CheckBatonStale / Resolver.Resolve), which
// is exactly the property being pinned.
type reloadPlan struct {
	Baton       json.RawMessage `json:"baton"`
	Cursor      ReloadResult    `json:"cursor"`
	Stale       StaleOutcome    `json:"stale"`
	Resolutions []Resolution    `json:"resolutions"`
}

// projectReload runs the full reload projection over (b, r) and encodes it as canonical plan
// bytes. It composes only pure, no-clock, no-I/O production functions — Marshal, VerifyReload,
// CheckBatonStale, and Resolver.Resolve over each artifact pointer in baton order — so its
// output depends solely on its inputs, which is what makes two calls comparable byte-for-byte.
func projectReload(t *testing.T, b Baton, r Resolver) []byte {
	t.Helper()
	b = project(b)
	batonBytes, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal baton: %v", err)
	}
	resolutions := make([]Resolution, 0, len(b.Artifacts))
	for _, a := range b.Artifacts {
		resolutions = append(resolutions, r.Resolve(a))
	}
	plan := reloadPlan{
		Baton:       batonBytes,
		Cursor:      VerifyReload(b.ProgressCursor, r),
		Stale:       CheckBatonStale(b, r),
		Resolutions: resolutions,
	}
	out, err := json.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal reload plan: %v", err)
	}
	return out
}

// TestReloadDeterminism pins the done condition: the same durable store + same baton reload to
// a byte-identical plan. It asserts three things — (1) two reloads of the identical inputs are
// byte-equal; (2) independently rebuilt equal inputs project to the same bytes (no hidden
// state or map-iteration order leaks in); and (3) the plan is a real PROJECTION of its inputs,
// not a constant — a different store (the anchor no longer resolves) must change the plan, and
// a resolving anchor must be recorded as a fresh cursor verdict.
func TestReloadDeterminism(t *testing.T) {
	b := sampleBaton()
	// Hermetic store: the cursor anchor and one artifact ref resolve; everything else dangles.
	newStore := func() Resolver {
		return fakeResolver{verified: map[string]bool{
			b.ProgressCursor.StartSHA: true,
			"#1871":                   true,
		}}
	}

	first := projectReload(t, b, newStore())
	second := projectReload(t, b, newStore())
	if !bytes.Equal(first, second) {
		t.Errorf("reload plan is not deterministic across two reloads of the same inputs:\n first =%s\n second=%s", first, second)
	}

	// Independently rebuilt equal inputs must project to the same bytes.
	indep := projectReload(t, sampleBaton(), newStore())
	if !bytes.Equal(first, indep) {
		t.Errorf("equal inputs projected to different plans:\n a=%s\n b=%s", first, indep)
	}

	// Non-vacuous: reload projects the store, so a diverged store (anchor no longer resolves)
	// must produce a different plan (fresh -> stale). A byte-equal plan here would mean the
	// projection ignores its inputs.
	diverged := projectReload(t, b, fakeResolver{verified: map[string]bool{}})
	if bytes.Equal(first, diverged) {
		t.Errorf("a diverged store produced an identical plan; reload is not projecting the store:\n plan=%s", first)
	}
	if !bytes.Contains(first, []byte(`"verdict":"fresh"`)) {
		t.Errorf("plan over a resolving anchor must record a fresh cursor verdict; got %s", first)
	}
}
