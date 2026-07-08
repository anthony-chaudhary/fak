package relay

import "testing"

// B4 (issue #1868) done condition: a baton with a MIXED valid/dangling pointer set
// produces the expected fidelity score and the expected unresolved list. These tests are
// that witness (run: `go test ./internal/relay -run Fidelity`).

// verdictResolver resolves an Artifact by a verdict looked up per store-native ref, so a
// test can compose an exact valid/dangling/unknown mix without any live store. An unlisted
// ref resolves to ResolveUnknown (fail closed), matching resolve.go's doctrine. (Distinct
// from reload_test.go's fakeResolver, which cannot express a per-ref unknown.)
type verdictResolver map[string]ResolveVerdict

func (f verdictResolver) Resolve(a Artifact) Resolution {
	v, ok := f[a.Ref]
	if !ok {
		v = ResolveUnknown
	}
	return Resolution{Artifact: a, Verdict: v, Detail: "fake: " + a.Ref}
}

// TestFidelityMixedValidAndDangling is the done-condition witness: three resolving pointers
// and one dangling pointer score 0.75, and the unresolved set is exactly the dangling one.
func TestFidelityMixedValidAndDangling(t *testing.T) {
	b := Baton{
		Schema:  Schema,
		RelayID: "r1",
		Artifacts: []Artifact{
			{Kind: string(ArtifactCommit), Ref: "good-sha"},
			{Kind: string(ArtifactIssue), Ref: "#1868"},
			{Kind: string(ArtifactMemory), Ref: "good-slug"},
			{Kind: string(ArtifactFile), Ref: "gone/path.go"},
		},
	}
	r := verdictResolver{
		"good-sha":     ResolveVerified,
		"#1868":        ResolveVerified,
		"good-slug":    ResolveVerified,
		"gone/path.go": ResolveDangling,
	}

	f := ScoreBatonFidelity(b, r)

	if f.Total != 4 || f.Verified != 3 || f.Dangling != 1 || f.Unknown != 0 {
		t.Fatalf("counts = {total %d verified %d dangling %d unknown %d}, want {4 3 1 0}",
			f.Total, f.Verified, f.Dangling, f.Unknown)
	}
	if f.Score != 0.75 {
		t.Errorf("score = %v, want 0.75", f.Score)
	}
	if len(f.Unresolved) != 1 {
		t.Fatalf("unresolved = %d rows, want exactly 1 (the dangling pointer)", len(f.Unresolved))
	}
	if got := f.Unresolved[0].Artifact.Ref; got != "gone/path.go" {
		t.Errorf("unresolved[0] ref = %q, want the dangling %q", got, "gone/path.go")
	}
	if f.Unresolved[0].Verdict != ResolveDangling {
		t.Errorf("unresolved[0] verdict = %q, want dangling", f.Unresolved[0].Verdict)
	}
}

// TestFidelityUnknownExcludedFromScore proves an unresolvable-store pointer (unknown) does
// NOT drag the score down: with 1 verified, 1 dangling, 1 unknown the score is 1/2, not
// 1/3 — unknown is excluded from the denominator but still listed in the unresolved set.
func TestFidelityUnknownExcludedFromScore(t *testing.T) {
	b := Baton{
		Schema:  Schema,
		RelayID: "r2",
		Artifacts: []Artifact{
			{Kind: string(ArtifactCommit), Ref: "good"},
			{Kind: string(ArtifactCommit), Ref: "bad"},
			{Kind: string(ArtifactLedger), Ref: "unreachable"},
		},
	}
	r := verdictResolver{
		"good": ResolveVerified,
		"bad":  ResolveDangling,
		// "unreachable" is unlisted -> ResolveUnknown.
	}

	f := ScoreBatonFidelity(b, r)

	if f.Verified != 1 || f.Dangling != 1 || f.Unknown != 1 {
		t.Fatalf("counts = {verified %d dangling %d unknown %d}, want {1 1 1}", f.Verified, f.Dangling, f.Unknown)
	}
	if f.Score != 0.5 {
		t.Errorf("score = %v, want 0.5 (unknown excluded from denominator)", f.Score)
	}
	if len(f.Unresolved) != 2 {
		t.Errorf("unresolved = %d rows, want 2 (dangling + unknown both surfaced)", len(f.Unresolved))
	}
}

// TestFidelityEmptyBatonVacuous: a baton with no pointers is vacuously faithful — score
// 1.0, zero counts, and an empty (non-nil) unresolved slice.
func TestFidelityEmptyBatonVacuous(t *testing.T) {
	f := ScoreBatonFidelity(Baton{Schema: Schema, RelayID: "r3"}, verdictResolver{})
	if f.Total != 0 || f.Score != 1.0 {
		t.Errorf("empty baton: {total %d score %v}, want {0 1}", f.Total, f.Score)
	}
	if f.Unresolved == nil {
		t.Errorf("unresolved must serialize as [], got nil")
	}
}

// TestFidelityAllDangling: every pointer dangling -> score 0.0 and every pointer surfaced.
func TestFidelityAllDangling(t *testing.T) {
	b := Baton{Schema: Schema, RelayID: "r4", Artifacts: []Artifact{
		{Kind: string(ArtifactCommit), Ref: "x"}, {Kind: string(ArtifactCommit), Ref: "y"},
	}}
	f := ScoreBatonFidelity(b, verdictResolver{"x": ResolveDangling, "y": ResolveDangling})
	if f.Score != 0.0 || f.Dangling != 2 || len(f.Unresolved) != 2 {
		t.Errorf("all-dangling: {score %v dangling %d unresolved %d}, want {0 2 2}", f.Score, f.Dangling, len(f.Unresolved))
	}
}

// TestFidelityMultiResolverDispatchesByKind: a MultiResolver routes each pointer to the
// sub-resolver that owns its kind (first non-unknown verdict wins), so a real commit
// resolver + a fake issue resolver together score a two-kind baton with no false unknowns.
func TestFidelityMultiResolverDispatchesByKind(t *testing.T) {
	commits := NewCommitResolver(func(ref string) (bool, error) { return ref == "live-sha", nil })
	issues := verdictResolver{"#1868": ResolveVerified} // owns issue kind in this test

	// issues (fakeResolver) returns a verdict for any ref including commit refs, so order it
	// AFTER commits and only feed it issue refs; commits returns unknown for non-commit kinds.
	m := NewMultiResolver(commits, issueKindOnly{issues})

	b := Baton{Schema: Schema, RelayID: "r5", Artifacts: []Artifact{
		{Kind: string(ArtifactCommit), Ref: "live-sha"}, // -> commits: verified
		{Kind: string(ArtifactCommit), Ref: "dead-sha"}, // -> commits: dangling
		{Kind: string(ArtifactIssue), Ref: "#1868"},     // -> issues: verified
		{Kind: string(ArtifactMemory), Ref: "no-store"}, // -> unknown (no resolver owns memory)
	}}

	f := ScoreBatonFidelity(b, m)
	if f.Verified != 2 || f.Dangling != 1 || f.Unknown != 1 {
		t.Fatalf("multi counts = {verified %d dangling %d unknown %d}, want {2 1 1}", f.Verified, f.Dangling, f.Unknown)
	}
	if f.Score != float64(2)/3 {
		t.Errorf("multi score = %v, want 2/3", f.Score)
	}
}

// issueKindOnly narrows a resolver to the issue kind, returning unknown for other kinds so
// MultiResolver dispatch is exercised honestly (a store owns exactly its kind).
type issueKindOnly struct{ inner Resolver }

func (i issueKindOnly) Resolve(a Artifact) Resolution {
	if a.Kind != string(ArtifactIssue) {
		return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: "issue resolver does not own kind " + a.Kind}
	}
	return i.inner.Resolve(a)
}
