package relay

import (
	"errors"
	"strings"
	"testing"
)

// TestCrossHostBatonResolverVerdicts pins the injected-probe fold of SessionImageResolver to
// the closed ResolveVerdict, mirroring the CommitResolver contract: a clean load is verified,
// a reachable-but-gone image is dangling, an unreachable store is unknown (fail closed, never
// mistaken for dangling), an empty handle is dangling without touching the store, and a
// foreign kind is unknown without touching the store.
func TestCrossHostBatonResolverVerdicts(t *testing.T) {
	img := Artifact{Kind: string(ArtifactImage), Ref: "bundle-handle"}
	mustNotRun := func(string) (bool, error) { t.Fatal("probe ran when it must not"); return false, nil }
	cases := []struct {
		name  string
		art   Artifact
		loads func(string) (bool, error)
		want  ResolveVerdict
	}{
		{"verified", img, func(string) (bool, error) { return true, nil }, ResolveVerified},
		{"dangling_image_gone", img, func(string) (bool, error) { return false, nil }, ResolveDangling},
		{"unknown_store_unreachable", img, func(string) (bool, error) { return false, errors.New("offload mount gone") }, ResolveUnknown},
		{"dangling_empty_handle", Artifact{Kind: string(ArtifactImage)}, mustNotRun, ResolveDangling},
		{"unknown_foreign_kind", Artifact{Kind: string(ArtifactCommit), Ref: "abc"}, mustNotRun, ResolveUnknown},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := NewSessionImageResolver(c.loads).Resolve(c.art)
			if got.Verdict != c.want {
				t.Fatalf("verdict=%q want %q (detail=%s)", got.Verdict, c.want, got.Detail)
			}
			if got.Artifact != c.art {
				t.Fatalf("resolution echoed a different artifact: %+v", got.Artifact)
			}
		})
	}
}

// TestCrossHostBatonResumeModeFork pins the warm/cold fork: a baton with no image pointer
// short-circuits to cold WITHOUT consulting the resolver (an old baton resumes exactly as
// before); a verified image is the ONLY path to warm; a dangling or unknown image both fall to
// cold — the fail-closed default that guarantees no false-positive warm resume.
func TestCrossHostBatonResumeModeFork(t *testing.T) {
	warm := Baton{ProgressCursor: ProgressCursor{StartSHA: "abc", SessionImage: "handle"}}
	cold := Baton{ProgressCursor: ProgressCursor{StartSHA: "abc"}}
	mustNotRun := func(string) (bool, error) { t.Fatal("resolver ran for an absent pointer"); return false, nil }
	cases := []struct {
		name  string
		b     Baton
		loads func(string) (bool, error)
		want  ResumeMode
	}{
		{"absent_pointer_cold", cold, mustNotRun, ResumeCold},
		{"verified_warm", warm, func(string) (bool, error) { return true, nil }, ResumeWarm},
		{"dangling_cold", warm, func(string) (bool, error) { return false, nil }, ResumeCold},
		{"unknown_cold", warm, func(string) (bool, error) { return false, errors.New("unreachable") }, ResumeCold},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			mode, _ := ResolveResumeMode(c.b, NewSessionImageResolver(c.loads))
			if mode != c.want {
				t.Fatalf("mode=%q want %q", mode, c.want)
			}
		})
	}
}

// TestCrossHostBatonPointerOnly proves the image field keeps the baton's pointer-only /
// no-`claimed` posture: the wire names the bundle HANDLE and never image bytes, a transcript,
// or an injected instruction; the handle round-trips; and an old baton with no image pointer
// omits the key entirely and resumes cold.
func TestCrossHostBatonPointerOnly(t *testing.T) {
	const handle = "/offload/store/img-abc123"
	b := Baton{Schema: Schema, RelayID: "R", ProgressCursor: ProgressCursor{StartSHA: "deadbeef", SessionImage: handle}}
	wire, err := Marshal(b)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(wire)
	if !strings.Contains(s, `"session_image":"`+handle+`"`) {
		t.Fatalf("baton wire dropped the session_image pointer: %s", s)
	}
	for _, forbidden := range []string{"claimed", "transcript", "###system", "cas.json", "manifest.json"} {
		if strings.Contains(strings.ToLower(s), forbidden) {
			t.Fatalf("baton wire carried non-pointer content %q: %s", forbidden, s)
		}
	}
	got, err := Parse(wire)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.ProgressCursor.SessionImage != handle {
		t.Fatalf("session_image did not round-trip: %q", got.ProgressCursor.SessionImage)
	}

	// Back-compat: an old baton with no image pointer omits the key and resumes cold.
	old := Baton{Schema: Schema, RelayID: "R", ProgressCursor: ProgressCursor{StartSHA: "deadbeef"}}
	ow, err := Marshal(old)
	if err != nil {
		t.Fatalf("Marshal(old): %v", err)
	}
	if strings.Contains(string(ow), "session_image") {
		t.Fatalf("an empty session_image must be omitted from the wire: %s", ow)
	}
	mode, _ := ResolveResumeMode(old, NewSessionImageResolver(func(string) (bool, error) {
		t.Fatal("resolver ran for a baton with no image pointer")
		return false, nil
	}))
	if mode != ResumeCold {
		t.Fatalf("old baton must resume cold, got %q", mode)
	}
}
