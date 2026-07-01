package appversion

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFromDirWalksUpToVersionMarker(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "fak", "internal", "bench")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "VERSION"), []byte("1.2.3\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, ok := FromDir(nested)
	if !ok {
		t.Fatal("FromDir did not find VERSION")
	}
	if got != "1.2.3" {
		t.Fatalf("version=%q, want 1.2.3", got)
	}
}

func TestFromDirStopsAtRepositoryBoundary(t *testing.T) {
	parent := t.TempDir()
	if err := os.WriteFile(filepath.Join(parent, "VERSION"), []byte("parent-version\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(parent, "sibling-repo")
	nested := filepath.Join(repo, "cmd", "fak")
	if err := os.MkdirAll(filepath.Join(repo, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}

	if got, ok := FromDir(nested); ok {
		t.Fatalf("FromDir crossed the repo boundary and returned %q", got)
	}
}

func TestCurrentPrefersEnvironment(t *testing.T) {
	t.Setenv("FAK_APP_VERSION", "9.9.9-test")
	if got := Current(); got != "9.9.9-test" {
		t.Fatalf("Current()=%q, want environment override", got)
	}
}

func TestCurrentPrefersBuildVersionOverTreeVersion(t *testing.T) {
	oldBuildVersion := BuildVersion
	BuildVersion = "7.7.7-release"
	t.Cleanup(func() { BuildVersion = oldBuildVersion })
	t.Setenv("FAK_APP_VERSION", "")

	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "VERSION"), []byte("1.2.3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldwd) })

	if got := Current(); got != "7.7.7-release" {
		t.Fatalf("Current()=%q, want BuildVersion override", got)
	}
}

func TestDiagnoseBinaryWarnsOnNewerDifferingSibling(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "fak.exe")
	extless := filepath.Join(dir, "fak")
	writeBinaryFixture(t, exe, "old-binary", time.Unix(100, 0))
	writeBinaryFixture(t, extless, "new-binary", time.Unix(200, 0))

	rep := DiagnoseBinary(exe, []string{exe, extless})
	if rep.Findings == 0 {
		t.Fatalf("expected a stale-binary finding, got %+v", rep)
	}
	if severityOfBinary(rep, "binary-shadow") != SeverityWarn {
		t.Fatalf("binary-shadow severity = %q, want warn (%+v)", severityOfBinary(rep, "binary-shadow"), rep.Recommendations)
	}
	if severityOfBinary(rep, "binary-current") != SeverityWarn {
		t.Fatalf("binary-current severity = %q, want warn (%+v)", severityOfBinary(rep, "binary-current"), rep.Recommendations)
	}
}

func TestDiagnoseBinaryCleanWhenSiblingsMatch(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "fak.exe")
	extless := filepath.Join(dir, "fak")
	writeBinaryFixture(t, exe, "same-binary", time.Unix(100, 0))
	writeBinaryFixture(t, extless, "same-binary", time.Unix(200, 0))

	rep := DiagnoseBinary(exe, []string{exe, extless})
	if rep.Findings != 0 {
		t.Fatalf("matching binaries should be healthy, got %+v", rep)
	}
	if severityOfBinary(rep, "binary-shadow") != SeverityOK {
		t.Fatalf("binary-shadow severity = %q, want ok", severityOfBinary(rep, "binary-shadow"))
	}
}

func TestDiagnoseBinaryWarnsOnLiveDifferingProcess(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "fak.exe")
	extless := filepath.Join(dir, "fak")
	writeBinaryFixture(t, exe, "current-binary", time.Unix(200, 0))
	writeBinaryFixture(t, extless, "stale-live-binary", time.Unix(100, 0))

	rep := DiagnoseBinaryWithProcesses(exe, []string{exe, extless}, []BinaryProcess{
		{PID: 123, Path: extless, Command: extless + " sweep --json"},
	}, "")

	if severityOfBinary(rep, "binary-live-process") != SeverityWarn {
		t.Fatalf("binary-live-process severity = %q, want warn (%+v)", severityOfBinary(rep, "binary-live-process"), rep.Recommendations)
	}
	if len(rep.Processes) != 1 {
		t.Fatalf("processes = %d, want 1 (%+v)", len(rep.Processes), rep.Processes)
	}
	if rep.Processes[0].SameCurrent {
		t.Fatalf("live stale process marked same-current: %+v", rep.Processes[0])
	}
}

func TestDiagnoseBinaryDoesNotWarnOnLiveMatchingProcess(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "fak.exe")
	extless := filepath.Join(dir, "fak")
	writeBinaryFixture(t, exe, "same-binary", time.Unix(100, 0))
	writeBinaryFixture(t, extless, "same-binary", time.Unix(200, 0))

	rep := DiagnoseBinaryWithProcesses(exe, []string{exe, extless}, []BinaryProcess{
		{PID: 123, Path: extless, Command: extless + " sweep --json"},
	}, "")

	if severityOfBinary(rep, "binary-live-process") != "" {
		t.Fatalf("binary-live-process should not warn for matching image: %+v", rep.Recommendations)
	}
	if len(rep.Processes) != 1 || !rep.Processes[0].SameCurrent {
		t.Fatalf("matching live process not annotated same-current: %+v", rep.Processes)
	}
}

func writeBinaryFixture(t *testing.T, path, body string, mod time.Time) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mod, mod); err != nil {
		t.Fatal(err)
	}
}

func severityOfBinary(rep BinaryReport, check string) string {
	for _, r := range rep.Recommendations {
		if r.Check == check {
			return r.Severity
		}
	}
	return ""
}
