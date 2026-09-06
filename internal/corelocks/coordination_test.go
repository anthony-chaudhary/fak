package corelocks

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func gitRun(t *testing.T, dir string, args ...string) (string, error) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL="+filepath.Join(dir, "no-such-gitconfig"),
		"GIT_AUTHOR_NAME=tester", "GIT_AUTHOR_EMAIL=tester@example.invalid",
		"GIT_COMMITTER_NAME=tester", "GIT_COMMITTER_EMAIL=tester@example.invalid",
	)
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

func countDirEntries(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir %s: %v", dir, err)
	}
	return len(entries)
}

func TestCoordinationAuthorityResolution(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not on PATH; git worktree tests require git")
	}

	t.Run("LinkedWorktreeAndSubdirectories", func(t *testing.T) {
		mainRepo := t.TempDir()
		if out, err := gitRun(t, mainRepo, "init", "-q"); err != nil {
			t.Fatalf("git init: %v: %s", err, out)
		}

		dosToml := `workspace = "."

[coordination]
version = 1
authority_locator = "https://mesh.local/coord/v1"
workspace_id = "ws-declared-42"
`
		if err := os.WriteFile(filepath.Join(mainRepo, "dos.toml"), []byte(dosToml), 0o644); err != nil {
			t.Fatal(err)
		}
		if out, err := gitRun(t, mainRepo, "add", "dos.toml"); err != nil {
			t.Fatalf("git add: %v: %s", err, out)
		}
		if out, err := gitRun(t, mainRepo, "commit", "-m", "init coordination"); err != nil {
			t.Fatalf("git commit: %v: %s", err, out)
		}

		mainSub := filepath.Join(mainRepo, "sub", "inner")
		if err := os.MkdirAll(mainSub, 0o755); err != nil {
			t.Fatal(err)
		}

		wtDir := filepath.Join(t.TempDir(), "wt")
		if out, err := gitRun(t, mainRepo, "worktree", "add", "--detach", wtDir, "HEAD"); err != nil {
			t.Fatalf("git worktree add: %v: %s", err, out)
		}
		t.Cleanup(func() {
			_ = exec.Command("git", "-C", mainRepo, "worktree", "remove", "--force", wtDir).Run()
		})

		wtSub := filepath.Join(wtDir, "sub", "inner")
		if err := os.MkdirAll(wtSub, 0o755); err != nil {
			t.Fatal(err)
		}

		authMainRoot, err := ResolveCoordinationAuthority(mainRepo)
		if err != nil {
			t.Fatalf("resolve mainRoot: %v", err)
		}
		authMainSub, err := ResolveCoordinationAuthority(mainSub)
		if err != nil {
			t.Fatalf("resolve mainSub: %v", err)
		}
		authWTRoot, err := ResolveCoordinationAuthority(wtDir)
		if err != nil {
			t.Fatalf("resolve wtRoot: %v", err)
		}
		authWTSub, err := ResolveCoordinationAuthority(wtSub)
		if err != nil {
			t.Fatalf("resolve wtSub: %v", err)
		}

		all := []*CoordinationAuthority{authMainRoot, authMainSub, authWTRoot, authWTSub}
		for i, a := range all {
			if a.WorkspaceID != "ws-declared-42" {
				t.Errorf("[%d] WorkspaceID = %q, want %q", i, a.WorkspaceID, "ws-declared-42")
			}
			if a.AuthorityLocator != "https://mesh.local/coord/v1" {
				t.Errorf("[%d] AuthorityLocator = %q, want %q", i, a.AuthorityLocator, "https://mesh.local/coord/v1")
			}
			if a.Status != StatusConfigured || !a.IsConfigured() {
				t.Errorf("[%d] Status = %q, want %q", i, a.Status, StatusConfigured)
			}
			if !a.Configured {
				t.Errorf("[%d] Configured = false, want true", i)
			}
			if a.Version != 1 {
				t.Errorf("[%d] Version = %d, want 1", i, a.Version)
			}
		}

		// Verify provenance classifications
		if authMainRoot.RootProvenance.Kind != ProvenanceGit {
			t.Errorf("mainRoot kind = %q, want %q", authMainRoot.RootProvenance.Kind, ProvenanceGit)
		}
		if authMainSub.RootProvenance.Kind != ProvenanceGit {
			t.Errorf("mainSub kind = %q, want %q", authMainSub.RootProvenance.Kind, ProvenanceGit)
		}
		if authWTRoot.RootProvenance.Kind != ProvenanceGitWorktree {
			t.Errorf("wtRoot kind = %q, want %q", authWTRoot.RootProvenance.Kind, ProvenanceGitWorktree)
		}
		if authWTSub.RootProvenance.Kind != ProvenanceGitWorktree {
			t.Errorf("wtSub kind = %q, want %q", authWTSub.RootProvenance.Kind, ProvenanceGitWorktree)
		}

		// CommonDir must match between main repo and linked worktree
		if !sameDir(authMainRoot.RootProvenance.CommonDir, authWTRoot.RootProvenance.CommonDir) {
			t.Errorf("CommonDir mismatch: main=%q, wt=%q", authMainRoot.RootProvenance.CommonDir, authWTRoot.RootProvenance.CommonDir)
		}
	})

	t.Run("DerivedGitWorkspaceIDWithoutExplicitID", func(t *testing.T) {
		mainRepo := t.TempDir()
		if out, err := gitRun(t, mainRepo, "init", "-q"); err != nil {
			t.Fatalf("git init: %v: %s", err, out)
		}

		dosToml := `[coordination]
version = 1
authority_locator = "https://mesh.local/coord/v1"
`
		if err := os.WriteFile(filepath.Join(mainRepo, "dos.toml"), []byte(dosToml), 0o644); err != nil {
			t.Fatal(err)
		}
		if out, err := gitRun(t, mainRepo, "add", "dos.toml"); err != nil {
			t.Fatalf("git add: %v: %s", err, out)
		}
		if out, err := gitRun(t, mainRepo, "commit", "-m", "init coordination"); err != nil {
			t.Fatalf("git commit: %v: %s", err, out)
		}

		wtDir := filepath.Join(t.TempDir(), "wt-derived")
		if out, err := gitRun(t, mainRepo, "worktree", "add", "--detach", wtDir, "HEAD"); err != nil {
			t.Fatalf("git worktree add: %v: %s", err, out)
		}
		t.Cleanup(func() {
			_ = exec.Command("git", "-C", mainRepo, "worktree", "remove", "--force", wtDir).Run()
		})

		mainSub := filepath.Join(mainRepo, "a", "b")
		wtSub := filepath.Join(wtDir, "a", "b")
		_ = os.MkdirAll(mainSub, 0o755)
		_ = os.MkdirAll(wtSub, 0o755)

		authMain, err := ResolveCoordinationAuthority(mainRepo)
		if err != nil {
			t.Fatalf("resolve main: %v", err)
		}
		authMainSub, err := ResolveCoordinationAuthority(mainSub)
		if err != nil {
			t.Fatalf("resolve mainSub: %v", err)
		}
		authWT, err := ResolveCoordinationAuthority(wtDir)
		if err != nil {
			t.Fatalf("resolve wt: %v", err)
		}
		authWTSub, err := ResolveCoordinationAuthority(wtSub)
		if err != nil {
			t.Fatalf("resolve wtSub: %v", err)
		}

		expectedID := filepath.Base(mainRepo)
		for i, a := range []*CoordinationAuthority{authMain, authMainSub, authWT, authWTSub} {
			if a.WorkspaceID != expectedID {
				t.Errorf("[%d] derived WorkspaceID = %q, want %q", i, a.WorkspaceID, expectedID)
			}
			if a.AuthorityLocator != "https://mesh.local/coord/v1" {
				t.Errorf("[%d] AuthorityLocator = %q", i, a.AuthorityLocator)
			}
		}
	})

	t.Run("ExplicitNonGitIdentity", func(t *testing.T) {
		nonGitDir := t.TempDir()
		dosToml := `[coordination]
version = 1
authority_locator = "mesh://non-git-authority"
workspace_id = "ws-non-git-explicit"
`
		if err := os.WriteFile(filepath.Join(nonGitDir, "dos.toml"), []byte(dosToml), 0o644); err != nil {
			t.Fatal(err)
		}
		subDir := filepath.Join(nonGitDir, "nested", "child")
		if err := os.MkdirAll(subDir, 0o755); err != nil {
			t.Fatal(err)
		}

		authRoot, err := ResolveCoordinationAuthority(nonGitDir)
		if err != nil {
			t.Fatalf("resolve nonGit root: %v", err)
		}
		authSub, err := ResolveCoordinationAuthority(subDir)
		if err != nil {
			t.Fatalf("resolve nonGit sub: %v", err)
		}

		for i, a := range []*CoordinationAuthority{authRoot, authSub} {
			if a.WorkspaceID != "ws-non-git-explicit" {
				t.Errorf("[%d] WorkspaceID = %q, want ws-non-git-explicit", i, a.WorkspaceID)
			}
			if a.AuthorityLocator != "mesh://non-git-authority" {
				t.Errorf("[%d] AuthorityLocator = %q", i, a.AuthorityLocator)
			}
			if a.RootProvenance.Kind != ProvenanceNonGit {
				t.Errorf("[%d] Kind = %q, want non-git", i, a.RootProvenance.Kind)
			}
			if a.Status != StatusConfigured {
				t.Errorf("[%d] Status = %q, want configured", i, a.Status)
			}
		}
	})

	t.Run("NonGitMissingWorkspaceID", func(t *testing.T) {
		nonGitDir := t.TempDir()
		dosToml := `[coordination]
version = 1
authority_locator = "mesh://non-git-authority"
`
		if err := os.WriteFile(filepath.Join(nonGitDir, "dos.toml"), []byte(dosToml), 0o644); err != nil {
			t.Fatal(err)
		}

		_, err := ResolveCoordinationAuthorityWithin(nonGitDir, nonGitDir)
		if err == nil {
			t.Fatal("expected error for non-git workspace lacking explicit workspace_id")
		}
		if !errors.Is(err, ErrMalformedDeclaration) && !errors.Is(err, ErrMissingWorkspaceID) {
			t.Errorf("error = %v, want wrapping ErrMalformedDeclaration or ErrMissingWorkspaceID", err)
		}
	})

	t.Run("MissingTableReportsLegacyUnconfigured", func(t *testing.T) {
		legacyDir := t.TempDir()
		dosToml := `workspace = "."

[lanes]
concurrent = ["adjudicator"]
`
		if err := os.WriteFile(filepath.Join(legacyDir, "dos.toml"), []byte(dosToml), 0o644); err != nil {
			t.Fatal(err)
		}

		res, err := ResolveCoordinationAuthorityWithin(legacyDir, legacyDir)
		if err != nil {
			t.Fatalf("missing table must not return error: %v", err)
		}
		if res.Status != StatusLegacyUnconfigured {
			t.Errorf("Status = %q, want %q", res.Status, StatusLegacyUnconfigured)
		}
		if res.Configured {
			t.Errorf("Configured = true, want false")
		}
		if !res.IsLegacy() {
			t.Errorf("IsLegacy() = false, want true")
		}
		if res.IsConfigured() {
			t.Errorf("IsConfigured() = true, want false")
		}
		if res.AuthorityLocator != "" {
			t.Errorf("AuthorityLocator must be empty (never an invented grant), got %q", res.AuthorityLocator)
		}
		if res.ConfigSource != filepath.Join(legacyDir, "dos.toml") {
			t.Errorf("ConfigSource = %q, want %q", res.ConfigSource, filepath.Join(legacyDir, "dos.toml"))
		}
	})

	t.Run("ConflictingDeclarations", func(t *testing.T) {
		cases := []struct {
			name    string
			content string
		}{
			{
				name: "multiple coordination tables",
				content: `[coordination]
version = 1
authority_locator = "https://one.local"
workspace_id = "ws1"

[coordination]
version = 1
authority_locator = "https://two.local"
workspace_id = "ws2"
`,
			},
			{
				name: "conflicting authority keys",
				content: `[coordination]
version = 1
authority = "https://one.local"
authority_locator = "https://two.local"
workspace_id = "ws1"
`,
			},
			{
				name: "conflicting workspace keys",
				content: `[coordination]
version = 1
authority_locator = "https://one.local"
workspace_id = "ws1"
workspace = "ws2"
`,
			},
			{
				name: "conflicting version keys",
				content: `[coordination]
version = 1
schema_version = 2
authority_locator = "https://one.local"
workspace_id = "ws1"
`,
			},
			{
				name: "duplicate key",
				content: `[coordination]
version = 1
authority_locator = "https://one.local"
authority_locator = "https://one.local"
workspace_id = "ws1"
`,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte(tc.content), 0o644); err != nil {
					t.Fatal(err)
				}
				_, err := ResolveCoordinationAuthorityWithin(dir, dir)
				if err == nil {
					t.Fatalf("%s: expected conflict error but got nil", tc.name)
				}
				if !errors.Is(err, ErrConflictingDeclaration) {
					t.Errorf("%s: error %v does not wrap ErrConflictingDeclaration", tc.name, err)
				}
			})
		}
	})

	t.Run("MalformedDeclarations", func(t *testing.T) {
		cases := []struct {
			name    string
			content string
		}{
			{
				name: "missing version",
				content: `[coordination]
authority_locator = "https://one.local"
workspace_id = "ws1"
`,
			},
			{
				name: "unsupported version",
				content: `[coordination]
version = 2
authority_locator = "https://one.local"
workspace_id = "ws1"
`,
			},
			{
				name: "missing authority locator",
				content: `[coordination]
version = 1
workspace_id = "ws1"
`,
			},
			{
				name: "empty authority locator",
				content: `[coordination]
version = 1
authority_locator = ""
workspace_id = "ws1"
`,
			},
			{
				name: "unknown key in coordination",
				content: `[coordination]
version = 1
authority_locator = "https://one.local"
workspace_id = "ws1"
unknown_setting = "disallowed"
`,
			},
			{
				name: "unterminated string",
				content: `[coordination]
version = 1
authority_locator = "https://one.local
workspace_id = "ws1"
`,
			},
			{
				name: "missing equals sign",
				content: `[coordination]
version = 1
authority_locator
workspace_id = "ws1"
`,
			},
			{
				name: "array of tables rejected",
				content: `[[coordination]]
version = 1
authority_locator = "https://one.local"
workspace_id = "ws1"
`,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte(tc.content), 0o644); err != nil {
					t.Fatal(err)
				}
				_, err := ResolveCoordinationAuthorityWithin(dir, dir)
				if err == nil {
					t.Fatalf("%s: expected malformed declaration error but got nil", tc.name)
				}
				if !errors.Is(err, ErrMalformedDeclaration) {
					t.Errorf("%s: error %v does not wrap ErrMalformedDeclaration", tc.name, err)
				}
			})
		}
	})

	t.Run("NoStateCreated", func(t *testing.T) {
		dir := t.TempDir()
		dosToml := `[coordination]
version = 1
authority_locator = "https://coord.local"
workspace_id = "ws-no-state"
`
		if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte(dosToml), 0o644); err != nil {
			t.Fatal(err)
		}
		beforeCount := countDirEntries(t, dir)

		if _, err := ResolveCoordinationAuthorityWithin(dir, dir); err != nil {
			t.Fatalf("resolve: %v", err)
		}
		afterCount := countDirEntries(t, dir)

		if beforeCount != afterCount {
			t.Errorf("state was created: before=%d entries, after=%d entries", beforeCount, afterCount)
		}
		if _, err := os.Stat(filepath.Join(dir, ".dos")); err == nil {
			t.Error(".dos directory was unexpectedly created")
		}
	})

	t.Run("WorkspaceNotFound", func(t *testing.T) {
		dir := t.TempDir()
		_, err := ResolveCoordinationAuthorityWithin(dir, dir)
		if err == nil {
			t.Fatal("expected ErrWorkspaceNotFound")
		}
		if !errors.Is(err, ErrWorkspaceNotFound) {
			t.Errorf("error %v does not wrap ErrWorkspaceNotFound", err)
		}
	})

	t.Run("LiveRepoReportsLegacyUnconfigured", func(t *testing.T) {
		top := repoTop(t)
		res, err := ResolveCoordinationAuthority(top)
		if err != nil {
			t.Fatalf("live repo resolve: %v", err)
		}
		if res.Status != StatusLegacyUnconfigured {
			t.Errorf("live repo Status = %q, want legacy/unconfigured", res.Status)
		}
		if res.Configured {
			t.Errorf("live repo Configured = true, want false")
		}
		if res.AuthorityLocator != "" {
			t.Errorf("live repo AuthorityLocator = %q, want empty", res.AuthorityLocator)
		}
	})
}
