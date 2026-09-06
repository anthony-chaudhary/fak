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
