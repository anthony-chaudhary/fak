package relay

import (
	"errors"
	"testing"
)

// C3 (issue #1872) done condition: a resolver classifies a valid and a dangling commit
// pointer to the correct verdicts. These tests are that witness (run: `go test
// ./internal/relay -run Resolver`).

// TestCommitResolverVerdicts drives the full verdict matrix with an injected git probe so
// it is hermetic and deterministic: no live repo, no clock. It covers the two done-condition
// cases (verified + dangling) plus the fail-closed edges (empty ref, unsupported kind,
// unreachable store).
func TestCommitResolverVerdicts(t *testing.T) {
	const known = "0123456789abcdef0123456789abcdef01234567"
	resolver := NewCommitResolver(func(ref string) (bool, error) {
		return ref == known, nil
	})

	cases := []struct {
		name string
		art  Artifact
		want ResolveVerdict
	}{
		{"valid commit pointer", Artifact{Kind: string(ArtifactCommit), Ref: known}, ResolveVerified},
		{"dangling commit pointer", Artifact{Kind: string(ArtifactCommit), Ref: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"}, ResolveDangling},
		{"empty ref", Artifact{Kind: string(ArtifactCommit), Ref: ""}, ResolveDangling},
		{"non-commit kind", Artifact{Kind: string(ArtifactIssue), Ref: "#1872"}, ResolveUnknown},
	}
	for _, tc := range cases {
		got := resolver.Resolve(tc.art)
		if got.Verdict != tc.want {
			t.Errorf("%s: verdict = %q, want %q (detail=%s)", tc.name, got.Verdict, tc.want, got.Detail)
		}
		if got.Artifact != tc.art {
			t.Errorf("%s: resolution must echo the artifact; got %+v", tc.name, got.Artifact)
		}
		if got.Detail == "" {
			t.Errorf("%s: resolution must carry a detail string", tc.name)
		}
	}
}

// TestCommitResolverUnreachableStore asserts a probe error is ResolveUnknown, never
// ResolveDangling — an unreachable store must not be mistaken for a missing object.
func TestCommitResolverUnreachableStore(t *testing.T) {
	resolver := NewCommitResolver(func(string) (bool, error) {
		return false, errors.New("git exploded")
	})
	got := resolver.Resolve(Artifact{Kind: string(ArtifactCommit), Ref: "anything"})
	if got.Verdict != ResolveUnknown {
		t.Errorf("unreachable store: verdict = %q, want unknown (detail=%s)", got.Verdict, got.Detail)
	}
}

// TestGitCommitResolverAgainstRepo witnesses the production git wiring end to end: HEAD is
// a real commit (-> verified) and an all-zero SHA is not (-> dangling). It resolves the
// repo the test runs in; if git cannot be run at all the probe surfaces ResolveUnknown and
// the test skips rather than failing a git-less environment.
func TestGitCommitResolverAgainstRepo(t *testing.T) {
	resolver := NewCommitResolver(GitCommitExists("."))

	head := resolver.Resolve(Artifact{Kind: string(ArtifactCommit), Ref: "HEAD"})
	if head.Verdict == ResolveUnknown {
		t.Skipf("git unavailable in this environment: %s", head.Detail)
	}
	if head.Verdict != ResolveVerified {
		t.Errorf("HEAD should resolve to a commit: verdict = %q (detail=%s)", head.Verdict, head.Detail)
	}

	bogus := resolver.Resolve(Artifact{Kind: string(ArtifactCommit), Ref: "0000000000000000000000000000000000000000"})
	if bogus.Verdict != ResolveDangling {
		t.Errorf("an all-zero SHA should be dangling: verdict = %q (detail=%s)", bogus.Verdict, bogus.Detail)
	}
}
