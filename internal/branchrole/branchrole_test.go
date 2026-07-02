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

func writePilotConfig(t *testing.T, pilotLine string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "dos.toml")
	body := "[branch_roles]\ndevelopment_branch = \"main\"\nrelease_branch = \"main\"\nrelease_source = \"main\"\npublic_front_door = \"main\"\n" + pilotLine
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestPilotDeclarationIsInertWithoutOptIn(t *testing.T) {
	t.Setenv(PilotEnv, "")
	got, err := LoadFile(writePilotConfig(t, "pilot_development_branch = \"dev\"\n"))
	if err != nil {
		t.Fatalf("LoadFile returned error: %v", err)
	}
	if got.DevelopmentBranch != "main" || got.PilotDevelopmentBranch != "dev" || got.PilotActive {
		t.Fatalf("declared-but-not-opted-in roles = %+v, want development main + inert pilot", got)
	}
}

func TestPilotOptInFlipsDevelopmentBranch(t *testing.T) {
	t.Setenv(PilotEnv, "1")
	got, err := LoadFile(writePilotConfig(t, "pilot_development_branch = \"dev\"\n"))
	if err != nil {
		t.Fatalf("LoadFile returned error: %v", err)
	}
	if got.DevelopmentBranch != "dev" || !got.PilotActive {
		t.Fatalf("opted-in roles = %+v, want development dev + PilotActive", got)
	}
	if got.ReleaseBranch != "main" || got.PublicFrontDoor != "main" {
		t.Fatalf("opted-in roles = %+v, release/front-door must stay main", got)
	}
}

func TestPilotOptInWithoutDeclarationErrors(t *testing.T) {
	t.Setenv(PilotEnv, "1")
	got, err := LoadFile(writePilotConfig(t, ""))
	if err == nil {
		t.Fatalf("opt-in without a declared pilot branch should error, got %+v", got)
	}
	if got.DevelopmentBranch != "main" || got.PilotActive {
		t.Fatalf("misconfigured pilot roles = %+v, want development main + inactive pilot", got)
	}
}

func TestPilotNonOptInValueIsIgnored(t *testing.T) {
	t.Setenv(PilotEnv, "0")
	got, err := LoadFile(writePilotConfig(t, "pilot_development_branch = \"dev\"\n"))
	if err != nil {
		t.Fatalf("LoadFile returned error: %v", err)
	}
	if got.DevelopmentBranch != "main" || got.PilotActive {
		t.Fatalf("non-opt-in roles = %+v, want development main + inactive pilot", got)
	}
}

func TestPilotMayNotNameReleaseBranch(t *testing.T) {
	t.Setenv(PilotEnv, "")
	if got, err := LoadFile(writePilotConfig(t, "pilot_development_branch = \"main\"\n")); err == nil {
		t.Fatalf("pilot naming the release branch should be rejected, got %+v", got)
	}
}

func TestPilotInvalidBranchNameRejected(t *testing.T) {
	t.Setenv(PilotEnv, "")
	if got, err := LoadFile(writePilotConfig(t, "pilot_development_branch = \"bad branch\"\n")); err == nil {
		t.Fatalf("invalid pilot branch name should be rejected, got %+v", got)
	}
}
