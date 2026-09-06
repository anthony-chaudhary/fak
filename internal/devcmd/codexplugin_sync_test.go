package devcmd

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func writePluginFixture(t *testing.T, root, version, marker string) {
	t.Helper()
	repo := filepath.Dir(root)
	if filepath.Base(root) == "claude-plugin" {
		if err := os.MkdirAll(repo, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(repo, "pyproject.toml"), []byte("[project]\nversion = \""+version+"\"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(repo, ".codex-marketplace-install.json"), []byte(`{"revision":"fixture-revision"}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	files := map[string]string{
		".claude-plugin/plugin.json":     `{"version":"` + version + `"}`,
		"hooks/hooks.json":               `{"hooks":{"Stop":"` + marker + `"}}`,
		"bin/dos-hook-codex.ps1":         "# " + marker,
		"bin/dos-hook-windows-amd64.exe": marker,
	}
	for rel, body := range files {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCodexPluginBackupPathIsHiddenVersionSibling(t *testing.T) {
	destination := filepath.Join(`C:\codex`, "plugins", "cache", "dos", "dos-kernel", "0.30.0")
	got := codexPluginBackupPath(destination, time.Unix(0, 42))
	want := filepath.Join(filepath.Dir(destination), ".fak-plugin-backup-0.30.0-42")
	if got != want {
		t.Fatalf("backup path=%q, want %q", got, want)
	}
	if strings.HasPrefix(filepath.Base(got), "0.30.0") {
		t.Fatalf("backup %q remains discoverable as a 0.30.0 plugin version", got)
	}
}
func TestPrepareCodexPluginStageReusesBoundedPath(t *testing.T) {
	root := t.TempDir()
	destination := filepath.Join(root, "dos-kernel", "0.30.0")
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		t.Fatal(err)
	}
	legacy := filepath.Join(filepath.Dir(destination), ".fak-plugin-stage-123")
	if err := os.Mkdir(legacy, 0o755); err != nil {
		t.Fatal(err)
	}
	ops := codexPluginSyncOps{remove: os.RemoveAll}
	first, err := prepareCodexPluginStage(destination, ops)
	if err != nil {
		t.Fatal(err)
	}
	if pathExists(legacy) {
		t.Fatal("legacy stage was not reaped")
	}
	if err := os.WriteFile(filepath.Join(first, "stale"), []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	second, err := prepareCodexPluginStage(destination, ops)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || filepath.Base(second) != ".fak-plugin-stage-current" {
		t.Fatalf("stages first=%q second=%q", first, second)
	}
	if pathExists(filepath.Join(second, "stale")) {
		t.Fatal("reused stage retained stale content")
	}
}
func TestSyncCodexPluginLockedDestinationRetainsCoherentStage(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	source := filepath.Join(root, "repo", "claude-plugin")
	destination := filepath.Join(home, "plugins", "cache", "dos", "dos-kernel", "0.30.0")
	writePluginFixture(t, source, "0.30.0", "new")
	writePluginFixture(t, destination, "0.30.0", "old")

	r, err := syncCodexPlugin(home, source, destination, root, codexPluginSyncOps{
		rename: func(old, new string) error { return errors.New("sharing violation") },
		remove: os.RemoveAll,
	})
	if err == nil {
		t.Fatal("expected locked cutover")
	}
	if r.Status != "RESTART_REQUIRED" || r.FailureStage != "cutover_locked" {
		t.Fatalf("receipt = %+v", r)
	}
	if !pathExists(r.StagedDestination) {
		t.Fatalf("coherent stage was not retained: %s", r.StagedDestination)
	}
	staged, stageErr := pluginArtifacts(r.StagedDestination)
	if stageErr != nil {
		t.Fatal(stageErr)
	}
	sourceHashes, _ := pluginArtifacts(source)
	if !sameArtifacts(sourceHashes, staged) {
		t.Fatal("retained stage does not match source")
	}
	old, _ := os.ReadFile(filepath.Join(destination, "hooks", "hooks.json"))
	if !strings.Contains(string(old), "old") {
		t.Fatalf("destination changed before cutover: %s", old)
	}
}

func TestSyncCodexPluginCutoverFailureRollsBackOldInstall(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	source := filepath.Join(root, "repo", "claude-plugin")
	destination := filepath.Join(home, "plugins", "cache", "dos", "dos-kernel", "0.30.0")
	writePluginFixture(t, source, "0.30.0", "new")
	writePluginFixture(t, destination, "0.30.0", "old")
	calls := 0
	r, err := syncCodexPlugin(home, source, destination, root, codexPluginSyncOps{
		rename: func(old, new string) error {
			calls++
			if calls == 2 {
				return errors.New("interrupted before cutover")
			}
			return os.Rename(old, new)
		},
		remove: os.RemoveAll,
	})
	if err == nil || r.FailureStage != "cutover" {
		t.Fatalf("receipt=%+v err=%v", r, err)
	}
	old, readErr := os.ReadFile(filepath.Join(destination, "hooks", "hooks.json"))
	if readErr != nil {
		t.Fatal(readErr)
	}
	if !strings.Contains(string(old), "old") {
		t.Fatalf("rollback did not restore old install: %s", old)
	}
	if calls < 3 {
		t.Fatalf("rename calls=%d, want at least backup/cutover/rollback", calls)
	}
}

func TestPluginArtifactsDetectSelectivelyStaleInstall(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	installed := filepath.Join(root, "installed")
	writePluginFixture(t, source, "0.30.0", "same")
	writePluginFixture(t, installed, "0.30.0", "same")
	want, _ := pluginArtifacts(source)
	got, _ := pluginArtifacts(installed)
	if !sameArtifacts(want, got) {
		t.Fatal("identical install reported stale")
	}
	if err := os.WriteFile(filepath.Join(installed, "bin", "dos-hook-codex.ps1"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, _ = pluginArtifacts(installed)
	if sameArtifacts(want, got) {
		t.Fatal("selectively stale adapter was not detected")
	}
	drift := artifactDrift(want, got)
	if len(drift) != 1 || drift[0] != "bin/dos-hook-codex.ps1" {
		t.Fatalf("drift = %v", drift)
	}
}

func TestAttestInstalledHookInventoryRequiresStopFamilyAtInstalledPath(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "0.30.0")
	manifest := filepath.Join(destination, "hooks", "hooks.json")
	report := hookProfileReport{Verdict: "ACTION", Hooks: []effectiveHook{
		{EventName: "preToolUse", SourcePath: manifest, State: "effective"},
		{EventName: "post_tool_use", SourcePath: manifest, State: "effective"},
		{EventName: "Stop", SourcePath: manifest, State: "effective"},
		{EventName: "subagentStop", SourcePath: manifest, State: "effective"},
	}}
	if err := attestInstalledHookInventory(report, destination); err != nil {
		t.Fatal(err)
	}
	report.Hooks = report.Hooks[:3]
	if err := attestInstalledHookInventory(report, destination); err == nil || !strings.Contains(err.Error(), "subagentstop") {
		t.Fatalf("missing SubagentStop not detected: %v", err)
	}
}

func TestSyncCodexPluginVerifyInstallFailureRollsBackOldInstall(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	source := filepath.Join(root, "repo", "claude-plugin")
	destination := filepath.Join(home, "plugins", "cache", "dos", "dos-kernel", "0.30.0")
	writePluginFixture(t, source, "0.30.0", "new")
	writePluginFixture(t, destination, "0.30.0", "old")
	calls := 0
	r, err := syncCodexPlugin(home, source, destination, root, codexPluginSyncOps{
		rename: func(old, new string) error {
			calls++
			if calls == 2 {
				if err := os.Rename(old, new); err != nil {
					return err
				}
				return os.Remove(filepath.Join(new, "bin", "dos-hook-codex.ps1"))
			}
			return os.Rename(old, new)
		},
		remove: os.RemoveAll,
	})
	if err == nil || r.FailureStage != "verify_install" {
		t.Fatalf("receipt=%+v err=%v", r, err)
	}
	old, readErr := os.ReadFile(filepath.Join(destination, "hooks", "hooks.json"))
	if readErr != nil {
		t.Fatal(readErr)
	}
	if !strings.Contains(string(old), "old") {
		t.Fatalf("verify failure did not restore old install: %s", old)
	}
}

func TestCopyTreeRejectsSymlinkEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("creating symlinks requires privileges on Windows")
	}
	root := t.TempDir()
	source := filepath.Join(root, "source")
	destination := filepath.Join(root, "destination")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside")
	if err := os.WriteFile(outside, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(source, "escape")); err != nil {
		t.Fatal(err)
	}
	if err := copyTree(source, destination); err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("copyTree err=%v", err)
	}
}

func TestPackageVersionReadsOnlyProjectTable(t *testing.T) {
	repo := t.TempDir()
	body := "[tool.example]\nversion = \"9.9.9\"\n[project]\nname = \"dos-kernel\"\nversion = \"0.30.0\"\n"
	if err := os.WriteFile(filepath.Join(repo, "pyproject.toml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := packageVersion(repo); got != "0.30.0" {
		t.Fatalf("version=%q", got)
	}
}

func TestSyncCodexPluginBoundsProfileAttestation(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source", "claude-plugin")
	destination := filepath.Join(root, "cache", "0.30.0")
	writePluginFixture(t, source, "0.30.0", "new")
	writePluginFixture(t, destination, "0.30.0", "old")

	oldInspect := codexPluginProfileInspect
	codexPluginProfileInspect = func(ctx context.Context, _, _, _ string) (hookProfileReport, error) {
		<-ctx.Done()
		return hookProfileReport{}, ctx.Err()
	}
	t.Cleanup(func() { codexPluginProfileInspect = oldInspect })

	started := time.Now()
	r, err := syncCodexPlugin(root, source, destination, root, codexPluginSyncOps{rename: os.Rename, remove: os.RemoveAll})
	if err == nil {
		t.Fatal("expected bounded attestation failure")
	}
	if r.Status != "INSTALLED_UNATTESTED" || r.FailureStage != "attest_profile" {
		t.Fatalf("receipt=%+v err=%v", r, err)
	}
	if !strings.Contains(r.Detail, "attestation timed out") || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("detail=%q err=%v", r.Detail, err)
	}
	if elapsed := time.Since(started); elapsed > codexPluginAttestTimeout+2*time.Second {
		t.Fatalf("attestation exceeded bound: %s", elapsed)
	}
	installed, readErr := os.ReadFile(filepath.Join(destination, "hooks", "hooks.json"))
	if readErr != nil {
		t.Fatal(readErr)
	}
	if !strings.Contains(string(installed), "old") {
		t.Fatalf("timeout did not restore old install: %s", installed)
	}
}
