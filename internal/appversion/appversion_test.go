package appversion

import (
	"os"
	"path/filepath"
	"strings"
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

func TestBenchmarkConceptVersionPreservesFleetContract(t *testing.T) {
	if BenchmarkConceptVersion != "fak.benchmark-concept.v1" {
		t.Fatalf("BenchmarkConceptVersion = %q, want fleet contract value", BenchmarkConceptVersion)
	}
}

func TestFromDirRejectsConflictMarkers(t *testing.T) {
	tests := []struct {
		name   string
		marker string
	}{
		{name: "ours", marker: strings.Repeat("<", 7) + " HEAD"},
		{name: "separator", marker: strings.Repeat("=", 7)},
		{name: "theirs", marker: strings.Repeat(">", 7) + " branch"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "VERSION"), []byte(tt.marker+"\n1.0.0\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			if got, ok := FromDir(dir); ok || got != "" {
				t.Fatalf("FromDir() = %q, %v; want empty, false", got, ok)
			}
		})
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

func TestDiagnoseBinaryNamesNewerSiblingWhenCurrentExeIsNewer(t *testing.T) {
	// Regression: when the CURRENT exe (fak.exe) is itself the NEWER image and the
	// extensionless sibling `fak` is the stale one, the binary-shadow finding must
	// name fak.exe as newer. Each image's Newer flag is "newer than the current
	// exe", which is always false for whichever sibling IS the current exe — so a
	// flag-based direction check saw !fak.Newer && !fakExe.Newer and fell back to
	// the "fak" default, wrongly reporting "fak is newer" and pointing an operator
	// at the wrong binary to replace.
	dir := t.TempDir()
	exe := filepath.Join(dir, "fak.exe")
	extless := filepath.Join(dir, "fak")
	writeBinaryFixture(t, exe, "new-current-binary", time.Unix(200, 0))
	writeBinaryFixture(t, extless, "old-stale-binary", time.Unix(100, 0))

	rep := DiagnoseBinary(exe, []string{exe, extless})
	f := findingOfBinary(rep, "binary-shadow")
	if !strings.Contains(f, "fak.exe is newer") {
		t.Fatalf("binary-shadow finding = %q, want it to name fak.exe (the newer current exe)", f)
	}
	if strings.Contains(f, "; fak is newer") {
		t.Fatalf("binary-shadow finding = %q wrongly names the stale extensionless fak as newer", f)
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

func findingOfBinary(rep BinaryReport, check string) string {
	for _, r := range rep.Recommendations {
		if r.Check == check {
			return r.Finding
		}
	}
	return ""
}

func TestCurrentDoesNotBorrowVersionFromWorkingTree(t *testing.T) {
	t.Setenv("FAK_APP_VERSION", "")
	old := BuildVersion
	BuildVersion = ""
	t.Cleanup(func() { BuildVersion = old })

	// The test executable lives outside dir. This models an old fak installed on PATH
	// and launched from a newer checkout: cwd's VERSION is not the binary's identity.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "VERSION"), []byte("99.88.77\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWD) })

	if got := Current(); got != fallback {
		t.Fatalf("Current() = %q, want %q; installed binary borrowed cwd VERSION", got, fallback)
	}
}

func TestReleaseFromModuleVersionAcceptsOnlyReleaseTags(t *testing.T) {
	// debug.ReadBuildInfo() inside `go test` reports the TEST binary, whose main module
	// version is "(devel)" with no VCS settings at all, so the go-install path cannot be
	// exercised in-process. This pins the pure decision that path delegates to, over exactly
	// the strings the Go toolchain produces: a resolved release tag (`go install …@vX.Y.Z`),
	// "(devel)" and a VCS pseudo-version (`go build` from source), and "" (no module stamp).
	cases := []struct {
		name   string
		in     string
		want   string
		wantOK bool
	}{
		{name: "release tag", in: "v0.42.0", want: "0.42.0", wantOK: true},
		{name: "multi digit release tag", in: "v10.7.123", want: "10.7.123", wantOK: true},
		{name: "devel", in: "(devel)", wantOK: false},
		{name: "no module stamp", in: "", wantOK: false},
		{name: "pseudo version", in: "v0.41.1-0.20260729114657-6ef585379011", wantOK: false},
		{name: "dirty pseudo version", in: "v0.41.1-0.20260729114657-6ef585379011+dirty", wantOK: false},
		{name: "prerelease tag", in: "v0.42.0-rc1", wantOK: false},
		{name: "incompatible", in: "v2.0.0+incompatible", wantOK: false},
		{name: "missing v prefix", in: "0.42.0", wantOK: false},
		{name: "too few fields", in: "v0.42", wantOK: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := releaseFromModuleVersion(tc.in)
			if ok != tc.wantOK {
				t.Fatalf("releaseFromModuleVersion(%q) ok=%v, want %v", tc.in, ok, tc.wantOK)
			}
			if got != tc.want {
				t.Fatalf("releaseFromModuleVersion(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
