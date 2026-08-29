package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestStudyInventoryCommittedAuthorityIgnoresDirtyAndUntrackedWIP(t *testing.T) {
	repo, git := seedStudySelfRepo(t)
	writeStudySelfFile(t, repo, "README.md", "committed\n")
	gitStudySelf(t, git, "add", "README.md")
	gitStudySelf(t, git, "commit", "-qm", "seed")

	var out, errOut bytes.Buffer
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo, "--json"}); code != 0 {
		t.Fatalf("refresh=%d stderr=%s", code, errOut.String())
	}
	gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
	gitStudySelf(t, git, "commit", "-qm", "manifest")

	writeStudySelfFile(t, repo, "README.md", "dirty peer edit\n")
	writeStudySelfFile(t, repo, "peer-untracked.go", "package peer\n")
	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo, "--json"}); code != 0 {
		t.Fatalf("verify=%d stderr=%s output=%s", code, errOut.String(), out.String())
	}
	var result studySelfInventoryOutput
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Verification == nil || !result.Verification.OK || len(result.Verification.Drift) != 0 {
		t.Fatalf("dirty/untracked WIP changed committed verdict: %+v", result)
	}
}

func TestStudyInventoryCommittedMutationFailsThenRefreshPasses(t *testing.T) {
	repo, git := seedStudySelfRepo(t)
	writeStudySelfFile(t, repo, "README.md", "one\n")
	gitStudySelf(t, git, "add", "README.md")
	gitStudySelf(t, git, "commit", "-qm", "seed")
	var out, errOut bytes.Buffer
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo}); code != 0 {
		t.Fatalf("refresh: %s", errOut.String())
	}
	gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
	gitStudySelf(t, git, "commit", "-qm", "manifest")

	writeStudySelfFile(t, repo, "README.md", "two\n")
	gitStudySelf(t, git, "add", "README.md")
	gitStudySelf(t, git, "commit", "-qm", "mutation")
	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo, "--json"}); code != 1 {
		t.Fatalf("stale verify=%d output=%s stderr=%s", code, out.String(), errOut.String())
	}
	var stale studySelfInventoryOutput
	if err := json.Unmarshal(out.Bytes(), &stale); err != nil {
		t.Fatal(err)
	}
	if stale.Verification == nil || len(stale.Verification.Drift) != 1 || stale.Verification.Drift[0].Kind != studymonitor.SelfDriftContentChanged || stale.Verification.Drift[0].Path != "README.md" {
		t.Fatalf("stale diagnostics = %+v", stale)
	}

	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo}); code != 0 {
		t.Fatalf("second refresh: %s", errOut.String())
	}
	gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
	gitStudySelf(t, git, "commit", "-qm", "refresh")
	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo}); code != 0 {
		t.Fatalf("verify after refresh=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
}

func TestStudyInventoryCommittedTreeMutationMatrix(t *testing.T) {
	type expectedDrift struct {
		kind studymonitor.SelfInventoryDriftKind
		path string
	}
	tests := []struct {
		name       string
		noManifest bool
		mutate     func(*testing.T, string, func(...string) ([]byte, error))
		want       []expectedDrift
	}{
		{name: "default fresh"},
		{name: "empty default", noManifest: true, want: []expectedDrift{{studymonitor.SelfDriftManifestMissing, studymonitor.DefaultSelfInventoryPath}}},
		{
			name: "add", mutate: func(t *testing.T, repo string, git func(...string) ([]byte, error)) {
				writeStudySelfFile(t, repo, "b.go", "package b\n")
				gitStudySelf(t, git, "add", "b.go")
				gitStudySelf(t, git, "commit", "-qm", "add")
			},
			want: []expectedDrift{{studymonitor.SelfDriftPathAdded, "b.go"}},
		},
		{
			name: "delete", mutate: func(t *testing.T, _ string, git func(...string) ([]byte, error)) {
				gitStudySelf(t, git, "rm", "a.md")
				gitStudySelf(t, git, "commit", "-qm", "delete")
			},
			want: []expectedDrift{{studymonitor.SelfDriftPathRemoved, "a.md"}},
		},
		{
			name: "rename", mutate: func(t *testing.T, _ string, git func(...string) ([]byte, error)) {
				gitStudySelf(t, git, "mv", "a.md", "z.md")
				gitStudySelf(t, git, "commit", "-qm", "rename")
			},
			want: []expectedDrift{{studymonitor.SelfDriftPathRemoved, "a.md"}, {studymonitor.SelfDriftPathAdded, "z.md"}},
		},
		{
			name: "content change", mutate: func(t *testing.T, repo string, git func(...string) ([]byte, error)) {
				writeStudySelfFile(t, repo, "a.md", "changed\n")
				gitStudySelf(t, git, "add", "a.md")
				gitStudySelf(t, git, "commit", "-qm", "content")
			},
			want: []expectedDrift{{studymonitor.SelfDriftContentChanged, "a.md"}},
		},
		{
			name: "reclassification", mutate: func(t *testing.T, repo string, git func(...string) ([]byte, error)) {
				mutateStudySelfManifestClass(t, repo, "runtime_source")
				gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
				gitStudySelf(t, git, "commit", "-qm", "reclassify")
			},
			want: []expectedDrift{{studymonitor.SelfDriftClassChanged, "a.md"}},
		},
		{
			name: "unknown class", mutate: func(t *testing.T, repo string, git func(...string) ([]byte, error)) {
				mutateStudySelfManifestClass(t, repo, "unknown_future_class")
				gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
				gitStudySelf(t, git, "commit", "-qm", "unknown class")
			},
			want: []expectedDrift{{studymonitor.SelfDriftClassChanged, "a.md"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo, git := seedStudySelfRepo(t)
			writeStudySelfFile(t, repo, "a.md", "original\n")
			gitStudySelf(t, git, "add", "a.md")
			gitStudySelf(t, git, "commit", "-qm", "seed")
			if !tt.noManifest {
				var out, errOut bytes.Buffer
				if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo, "--json"}); code != 0 {
					t.Fatalf("refresh=%d stderr=%s", code, errOut.String())
				}
				gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
				gitStudySelf(t, git, "commit", "-qm", "manifest")
			}
			if tt.mutate != nil {
				tt.mutate(t, repo, git)
			}

			var out, errOut bytes.Buffer
			code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo, "--ref", "HEAD", "--json"})
			var result studySelfInventoryOutput
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatalf("decode output: %v\nstdout=%s\nstderr=%s", err, out.String(), errOut.String())
			}
			wantCode := 0
			if len(tt.want) > 0 {
				wantCode = 1
			}
			if code != wantCode || result.Verification == nil || result.Verification.OK != (wantCode == 0) {
				t.Fatalf("code=%d result=%+v stderr=%s", code, result, errOut.String())
			}
			if len(result.Verification.Drift) != len(tt.want) {
				t.Fatalf("drift=%+v want=%+v", result.Verification.Drift, tt.want)
			}
			for i, want := range tt.want {
				if result.Verification.Drift[i].Kind != want.kind || result.Verification.Drift[i].Path != want.path {
					t.Fatalf("drift[%d]=%+v want kind=%s path=%s", i, result.Verification.Drift[i], want.kind, want.path)
				}
			}
		})
	}
}

func mutateStudySelfManifestClass(t *testing.T, repo, class string) {
	t.Helper()
	path := filepath.Join(repo, filepath.FromSlash(studymonitor.DefaultSelfInventoryPath))
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest studymonitor.SelfInventory
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.Entries) != 1 {
		t.Fatalf("entries=%d, want 1", len(manifest.Entries))
	}
	manifest.Entries[0].Classification = class
	var out bytes.Buffer
	if err := studymonitor.WriteSelfInventory(&out, manifest); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, out.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func seedStudySelfRepo(t *testing.T) (string, func(...string) ([]byte, error)) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	repo := t.TempDir()
	git := func(args ...string) ([]byte, error) {
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.test", "GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.test")
		return cmd.CombinedOutput()
	}
	if out, err := git("init", "-q"); err != nil {
		t.Skipf("git init: %s", out)
	}
	return repo, git
}

func gitStudySelf(t *testing.T, git func(...string) ([]byte, error), args ...string) {
	t.Helper()
	if out, err := git(args...); err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
}

func writeStudySelfFile(t *testing.T, root, rel, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
