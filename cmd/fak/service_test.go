package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestServiceDryRunRendersHardenedSystemdUnit(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux")
	}
	var out, errout bytes.Buffer
	rc := runService(&out, &errout, []string{"install", "--dry-run", "--unit-dir", t.TempDir(), "--state-dir", filepath.Join(t.TempDir(), "state")})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errout.String())
	}
	for _, w := range []string{"Restart=always", "KillMode=control-group", "NoNewPrivileges=yes", "MemoryMax=1G", "service run --interval 15s"} {
		if !strings.Contains(out.String(), w) {
			t.Fatalf("missing %q", w)
		}
	}
}

func TestServiceDryRunUsesSystemManagerContracts(t *testing.T) {
	var out, errout bytes.Buffer
	tmp := t.TempDir()
	rc := runService(&out, &errout, []string{"install", "--dry-run", "--unit-dir", tmp, "--state-dir", filepath.Join(tmp, "state"), "--principal", "fak-test"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errout.String())
	}
	switch runtime.GOOS {
	case "linux":
		for _, want := range []string{"DynamicUser=yes", "WantedBy=multi-user.target", "FAK_SERVICE_MANAGER=systemd-system"} {
			if !strings.Contains(out.String(), want) {
				t.Fatalf("missing %q in %s", want, out.String())
			}
		}
		for _, forbidden := range []string{"--user", "WantedBy=default.target"} {
			if strings.Contains(out.String(), forbidden) {
				t.Fatalf("user-manager dependency %q in %s", forbidden, out.String())
			}
		}
	case "darwin":
		for _, want := range []string{"<key>UserName</key><string>fak-test</string>", "launchd-system"} {
			if !strings.Contains(out.String(), want) {
				t.Fatalf("missing %q in %s", want, out.String())
			}
		}
	}
}
func TestServiceRunOnceTicksControlPlane(t *testing.T) {
	old := serviceTick
	calls := 0
	serviceTick = func(io.Writer, io.Writer) int { calls++; return 0 }
	t.Cleanup(func() { serviceTick = old })
	if rc := runService(io.Discard, io.Discard, []string{"run", "--once"}); rc != 0 || calls != 1 {
		t.Fatalf("rc=%d calls=%d", rc, calls)
	}
}

func TestLinuxServiceStateAllowsOnlyTraversalToSharedRegistry(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux mode semantics")
	}
	state := filepath.Join(t.TempDir(), "state")
	if err := os.MkdirAll(state, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(state, 0o711); err != nil {
		t.Fatal(err)
	}
	if err := prepareSharedGuardRegistry(filepath.Join(state, "registry")); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(state)
	if err != nil || st.Mode().Perm() != 0o711 {
		t.Fatalf("state mode=%v err=%v", st.Mode(), err)
	}
}
func TestPrepareSharedGuardRegistryAllowsSessionFactsWithoutStateOwnership(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "registry")
	if err := prepareSharedGuardRegistry(dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "guard_sessions.jsonl")); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" {
		if st, err := os.Stat(dir); err != nil || st.Mode().Perm() != 0o733 || st.Mode()&os.ModeSticky == 0 {
			t.Fatalf("dir mode=%v err=%v", st.Mode(), err)
		}
		if st, err := os.Stat(filepath.Join(dir, "guard_sessions.jsonl")); err != nil || st.Mode().Perm() != 0o666 {
			t.Fatalf("index mode=%v err=%v", st.Mode(), err)
		}
	}
}
func TestInstallServiceExecutableStagesImmutableCopy(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "source-fak")
	destination := filepath.Join(dir, "libexec", "fak-guard-control")
	if err := os.WriteFile(source, []byte("v1"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := installServiceExecutable(source, destination); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("mutated"), 0o700); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(destination)
	if err != nil || string(got) != "v1" {
		t.Fatalf("staged=%q err=%v", got, err)
	}
	if st, err := os.Stat(destination); err != nil {
		t.Fatal(err)
	} else if runtime.GOOS != "windows" && st.Mode().Perm() != 0o755 {
		t.Fatalf("mode=%v", st.Mode())
	}
}
func TestWriteFileAtomic(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "unit")
	if err := writeFileAtomic(p, []byte("ok"), 0o600); err != nil {
		t.Fatal(err)
	}
	b, e := os.ReadFile(p)
	if e != nil || string(b) != "ok" {
		t.Fatalf("b=%q err=%v", b, e)
	}
}
