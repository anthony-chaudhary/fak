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
func TestServiceRunOnceTicksControlPlane(t *testing.T) {
	old := serviceTick
	calls := 0
	serviceTick = func(io.Writer, io.Writer) int { calls++; return 0 }
	t.Cleanup(func() { serviceTick = old })
	if rc := runService(io.Discard, io.Discard, []string{"run", "--once"}); rc != 0 || calls != 1 {
		t.Fatalf("rc=%d calls=%d", rc, calls)
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
