package recall

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

func TestReadTimeArtifactReverifyWithholdsStaleSHA(t *testing.T) {
	ctx := context.Background()
	body := []byte("commit deadbee fixed the refund fee path")
	d := Digest(body)
	s := (&Session{
		Manifest: Manifest{Version: ManifestVersion, Pages: []Page{{
			Step: 0, Role: "status", Descriptor: "status: commit deadbee fixed refund fee", Digest: d, Len: int64(len(body)),
		}}},
		cas:     map[string][]byte{d: body},
		cleared: map[string]bool{},
		gate:    ctxmmu.New(),
	}).WithArtifactVerifier(func(_ context.Context, claims []ArtifactClaim) []ArtifactFinding {
		if len(claims) != 1 || claims[0].Kind != ArtifactGitSHA || claims[0].Value != "deadbee" {
			t.Fatalf("claims = %+v, want one git SHA deadbee", claims)
		}
		return []ArtifactFinding{{Claim: claims[0], Status: ArtifactStale, Detail: "commit does not resolve"}}
	})

	_, err := s.Resolve(ctx, 0)
	if !errors.Is(err, ErrStale) {
		t.Fatalf("Resolve stale SHA: want ErrStale, got %v", err)
	}
	if set := s.Recall(ctx, "refund fee commit", 1); len(set) != 0 {
		t.Fatalf("stale page was injected through Recall: %+v", set)
	}
}

func TestExtractArtifactClaimsIsConservative(t *testing.T) {
	text := "HEAD commit deadbee touched internal/recall/recall.go and flag --ctx-view-budget; https://example.com/x and /tmp/refund-receipt.txt are not repo paths"
	got := ExtractArtifactClaims(text)
	want := map[ArtifactKind]string{
		ArtifactGitSHA: "deadbee",
		ArtifactPath:   "internal/recall/recall.go",
		ArtifactFlag:   "--ctx-view-budget",
	}
	for kind, value := range want {
		found := false
		for _, c := range got {
			if c.Kind == kind && c.Value == value {
				found = true
			}
		}
		if !found {
			t.Fatalf("claims %+v missing %s %q", got, kind, value)
		}
	}
	for _, c := range got {
		if strings.Contains(c.Value, "example.com") || strings.HasPrefix(c.Value, "/tmp/") {
			t.Fatalf("non-repo path was extracted as a concrete artifact: %+v", got)
		}
	}
}

func TestDefaultArtifactVerifierClassifiesMissingCommitStale(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	if err := cmd.Run(); err != nil {
		t.Skip("not running inside a git checkout")
	}
	claim := ArtifactClaim{Kind: ArtifactGitSHA, Value: "deadbee"}
	got := DefaultArtifactVerifier(context.Background(), []ArtifactClaim{claim})
	if len(got) != 1 {
		t.Fatalf("findings = %+v, want one", got)
	}
	if got[0].Status != ArtifactStale {
		t.Fatalf("missing commit status = %+v, want stale", got[0])
	}
}

func TestDefaultArtifactVerifierClassifiesRevertedCommitStale(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	repo := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=recall-test",
			"GIT_AUTHOR_EMAIL=recall-test@example.com",
			"GIT_COMMITTER_NAME=recall-test",
			"GIT_COMMITTER_EMAIL=recall-test@example.com",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "README.md")
	git("commit", "-q", "-m", "initial")
	if err := os.WriteFile(filepath.Join(repo, "feature.txt"), []byte("feature\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "feature.txt")
	git("commit", "-q", "-m", "add feature")
	featureSHA := git("rev-parse", "HEAD")
	git("revert", "--no-edit", featureSHA)

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(wd); err != nil {
			t.Errorf("restore cwd: %v", err)
		}
	})

	got := DefaultArtifactVerifier(context.Background(), []ArtifactClaim{{Kind: ArtifactGitSHA, Value: featureSHA[:12]}})
	if len(got) != 1 {
		t.Fatalf("findings = %+v, want one", got)
	}
	if got[0].Status != ArtifactStale || !strings.Contains(got[0].Detail, "later reverted") {
		t.Fatalf("reverted commit status = %+v, want stale/later reverted", got[0])
	}
}
