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

func TestCurrentPrefersEnvironment(t *testing.T) {
	t.Setenv("FAK_APP_VERSION", "9.9.9-test")
	if got := Current(); got != "9.9.9-test" {
		t.Fatalf("Current()=%q, want environment override", got)
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
