package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// A test binary whose HOME sits outside the temp roots must not write a
// garden unit there: that is how a real com.fleet.stale-work-garden unit
// leaked into an operator's home.
func TestGardenRegisterRefusesHomeOutsideTempRoots(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("systemd user-unit registration is linux-only")
	}
	base := t.TempDir()
	tmpRoot := filepath.Join(base, "tmproot")
	home := filepath.Join(base, "home")
	for _, d := range []string{tmpRoot, home} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("TMPDIR", tmpRoot)
	t.Setenv("GOTMPDIR", "")
	t.Setenv("HOME", home)

	var stdout, stderr bytes.Buffer
	code := registerLinuxSystemdTimer(&stdout, &stderr, "/usr/local/bin/fak", base, time.Hour, false)
	if code == 0 {
		t.Fatalf("registerLinuxSystemdTimer exit = 0, want non-zero; stdout=%q", stdout.String())
	}
	assertDirEmpty(t, home)
}

func assertDirEmpty(t *testing.T, dir string) {
	t.Helper()
	var written []string
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if p != dir {
			written = append(written, p)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(written) != 0 {
		t.Fatalf("registration wrote under %s: %v", dir, written)
	}
}
