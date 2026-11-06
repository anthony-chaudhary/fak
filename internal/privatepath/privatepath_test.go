package privatepath

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestResolveRunUsesOpaquePrivatePath(t *testing.T) {
	base := t.TempDir()
	public := filepath.Join(base, "fak")
	private := filepath.Join(base, "fak-private")
	for _, dir := range []string{public, private} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got, err := ResolveRun(Options{RepoRoot: public, Now: time.Date(2026, 7, 14, 22, 0, 0, 0, time.UTC), Create: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got.Path, filepath.Join(private, "fleet-runs", "codex")+string(filepath.Separator)) {
		t.Fatalf("path = %q, want private codex run root", got.Path)
	}
	if strings.Contains(strings.ToLower(got.Path), "dgx") {
		t.Fatalf("path leaks hardware identity: %q", got.Path)
	}
	if !got.Created {
		t.Fatal("created = false")
	}
	if info, err := os.Stat(got.Path); err != nil || !info.IsDir() {
		t.Fatalf("created dir: %v", err)
	}
}

func TestResolveRunRefusesPublicRoot(t *testing.T) {
	public := t.TempDir()
	_, err := ResolveRun(Options{RepoRoot: public, Root: public})
	if err == nil || !strings.Contains(err.Error(), "must not be the public checkout") {
		t.Fatalf("err = %v", err)
	}
}

func TestResolveRunRefusesMissingPrivateRoot(t *testing.T) {
	base := t.TempDir()
	public := filepath.Join(base, "fak")
	if err := os.Mkdir(public, 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := ResolveRun(Options{RepoRoot: public})
	if err == nil || !strings.Contains(err.Error(), "private root unavailable") {
		t.Fatalf("err = %v", err)
	}
}

func TestResolveRoot(t *testing.T) {
	base := t.TempDir()
	public := filepath.Join(base, "fak")
	private := filepath.Join(base, "fak-private")
	for _, dir := range []string{public, private} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got, err := ResolveRoot(Options{RepoRoot: public})
	if err != nil {
		t.Fatalf("ResolveRoot failed: %v", err)
	}
	if filepath.Clean(got) != filepath.Clean(private) {
		t.Fatalf("ResolveRoot = %q, want %q", got, private)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestResolveRunChecksSymlinkContainmentBeforeCreating(t *testing.T) {
	for _, tc := range []struct {
		name              string
		privateTarget     string
		privateAlias      bool
		aliasInsidePublic bool
		publicAlias       bool
		wantErr           bool
	}{
		{name: "private alias to public root", privateTarget: "public", privateAlias: true, wantErr: true},
		{name: "private alias to public child", privateTarget: "nested", privateAlias: true, wantErr: true},
		{name: "public root is an alias", privateTarget: "nested", publicAlias: true, wantErr: true},
		{name: "lexical child still refused", privateTarget: "private", privateAlias: true, aliasInsidePublic: true, wantErr: true},
		{name: "outside alias remains valid", privateTarget: "private", privateAlias: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := t.TempDir()
			public := filepath.Join(base, "public")
			targets := map[string]string{
				"public":  public,
				"nested":  filepath.Join(public, "nested"),
				"private": filepath.Join(base, "private"),
			}
			for _, path := range targets {
				if err := os.MkdirAll(path, 0o755); err != nil {
					t.Fatal(err)
				}
			}
			root := targets[tc.privateTarget]
			if tc.privateAlias {
				link := filepath.Join(base, "private-link")
				if tc.aliasInsidePublic {
					link = filepath.Join(public, "private-link")
				}
				if err := os.Symlink(root, link); err != nil {
					t.Skipf("symlinks unavailable: %v", err)
				}
				root = link
			}
			repo := public
			if tc.publicAlias {
				repo = filepath.Join(base, "public-link")
				if err := os.Symlink(public, repo); err != nil {
					t.Skipf("symlinks unavailable: %v", err)
				}
			}
			got, err := ResolveRun(Options{RepoRoot: repo, Root: root, Create: true})
			if tc.wantErr {
				if err == nil || !strings.Contains(err.Error(), "public checkout") {
					t.Fatalf("ResolveRun = %+v, %v; want public checkout refusal", got, err)
				}
				if got != (Result{}) {
					t.Fatalf("refused run returned %+v", got)
				}
				if _, err := os.Stat(filepath.Join(targets[tc.privateTarget], "fleet-runs")); !os.IsNotExist(err) {
					t.Fatalf("refused run created a directory or could not be inspected: %v", err)
				}
				return
			}
			if err != nil || !got.Created || got.Root != root {
				t.Fatalf("valid outside alias = %+v, %v", got, err)
			}
			resolvedPrivate, err := filepath.EvalSymlinks(targets["private"])
			if err != nil {
				t.Fatal(err)
			}
			resolved, err := filepath.EvalSymlinks(got.Path)
			if err != nil || !strings.HasPrefix(resolved, resolvedPrivate+string(filepath.Separator)) {
				t.Fatalf("created path = %q, %v; want separate private directory", resolved, err)
			}
		})
	}
}
