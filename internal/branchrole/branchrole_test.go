package branchrole

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFileDefaultsWithoutTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dos.toml")
	if err := os.WriteFile(path, []byte("[lanes]\nconcurrent = [\"cmd\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile returned error: %v", err)
	}
	if got != Defaults() {
		t.Fatalf("roles = %+v, want defaults %+v", got, Defaults())
	}
}

func TestLoadFileReadsBranchRoles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dos.toml")
	body := `[branch_roles]
development_branch = "dev" # hot branch
release_branch = "main"
release_source = "dev"
public_front_door = 'main'
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile returned error: %v", err)
	}
	if got.DevelopmentBranch != "dev" || got.ReleaseBranch != "main" || got.ReleaseSource != "dev" || got.PublicFrontDoor != "main" {
		t.Fatalf("roles = %+v", got)
	}
}

func TestLoadWalksUpFromSubdirectory(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[branch_roles]\ndevelopment_branch = \"dev\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(root, "internal", "safecommit")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := Load(sub)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	if got.DevelopmentBranch != "dev" || got.ReleaseBranch != "main" {
		t.Fatalf("roles = %+v, want dev with defaulted release role", got)
	}
}

func TestLoadFileRejectsEmptyKnownRole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dos.toml")
	if err := os.WriteFile(path, []byte("[branch_roles]\ndevelopment_branch = \"\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadFile(path)
	if err == nil {
		t.Fatalf("LoadFile should reject empty development_branch, got %+v", got)
	}
}

func TestLoadFileRejectsDuplicateRole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dos.toml")
	body := "[branch_roles]\ndevelopment_branch = \"main\"\ndevelopment_branch = \"dev\"\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := LoadFile(path); err == nil {
		t.Fatalf("LoadFile should reject duplicate development_branch, got %+v", got)
	}
}

func TestLoadFileRejectsInvalidBranchName(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dos.toml")
	if err := os.WriteFile(path, []byte("[branch_roles]\ndevelopment_branch = \"bad branch\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := LoadFile(path); err == nil {
		t.Fatalf("LoadFile should reject invalid branch names, got %+v", got)
	}
}
