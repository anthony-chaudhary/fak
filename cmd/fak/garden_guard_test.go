package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type gardenGuardRoots struct {
	base, tmpRoot, goTmp, outside string
}

func newGardenGuardRoots(t *testing.T) gardenGuardRoots {
	t.Helper()
	base := t.TempDir()
	r := gardenGuardRoots{
		base:    base,
		tmpRoot: filepath.Join(base, "tmproot"),
		goTmp:   filepath.Join(base, "gotmp"),
		outside: filepath.Join(base, "outside"),
	}
	for _, d := range []string{r.tmpRoot, r.goTmp, r.outside} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("TMPDIR", r.tmpRoot)
	t.Setenv("GOTMPDIR", "")
	return r
}

func mkGardenGuardHome(t *testing.T, parent string) string {
	t.Helper()
	home := filepath.Join(parent, "home")
	if err := os.MkdirAll(home, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	return home
}

func TestGardenGuardRefusesWindowsRegardlessOfHome(t *testing.T) {
	r := newGardenGuardRoots(t)
	mkGardenGuardHome(t, r.tmpRoot)
	if err := guardGardenTestRegister("windows"); !errors.Is(err, errGardenTestRegisterLive) {
		t.Fatalf("guard(windows) = %v, want errGardenTestRegisterLive", err)
	}
}

func TestGardenGuardAllowsHomeUnderTMPDIR(t *testing.T) {
	r := newGardenGuardRoots(t)
	mkGardenGuardHome(t, r.tmpRoot)
	for _, goos := range []string{"linux", "darwin"} {
		if err := guardGardenTestRegister(goos); err != nil {
			t.Fatalf("guard(%s) with HOME under TMPDIR = %v, want nil", goos, err)
		}
	}
}

func TestGardenGuardAllowsHomeEqualToTMPDIR(t *testing.T) {
	r := newGardenGuardRoots(t)
	t.Setenv("HOME", r.tmpRoot)
	if err := guardGardenTestRegister("linux"); err != nil {
		t.Fatalf("guard with HOME == TMPDIR = %v, want nil", err)
	}
}

func TestGardenGuardAllowsHomeUnderGOTMPDIR(t *testing.T) {
	r := newGardenGuardRoots(t)
	t.Setenv("GOTMPDIR", r.goTmp)
	mkGardenGuardHome(t, r.goTmp)
	for _, goos := range []string{"linux", "darwin"} {
		if err := guardGardenTestRegister(goos); err != nil {
			t.Fatalf("guard(%s) with HOME under GOTMPDIR = %v, want nil", goos, err)
		}
	}
}

func TestGardenGuardRefusesHomeOutsideTempRoots(t *testing.T) {
	r := newGardenGuardRoots(t)
	t.Setenv("GOTMPDIR", r.goTmp)
	mkGardenGuardHome(t, r.outside)
	for _, goos := range []string{"linux", "darwin"} {
		if err := guardGardenTestRegister(goos); !errors.Is(err, errGardenTestRegisterLive) {
			t.Fatalf("guard(%s) with HOME outside temp roots = %v, want errGardenTestRegisterLive", goos, err)
		}
	}
}

func TestGardenGuardBlankGOTMPDIRIsNotARoot(t *testing.T) {
	r := newGardenGuardRoots(t)
	t.Setenv("GOTMPDIR", "   ")
	mkGardenGuardHome(t, r.outside)
	if err := guardGardenTestRegister("linux"); !errors.Is(err, errGardenTestRegisterLive) {
		t.Fatalf("guard with blank GOTMPDIR = %v, want errGardenTestRegisterLive", err)
	}
}

func TestGardenGuardResolvesSymlinkedHome(t *testing.T) {
	r := newGardenGuardRoots(t)
	real := mkGardenGuardHome(t, r.outside)
	link := filepath.Join(r.tmpRoot, "homelink")
	if err := os.Symlink(real, link); err != nil {
		t.Skipf("symlink unsupported: %v", err)
	}
	t.Setenv("HOME", link)
	if err := guardGardenTestRegister("linux"); !errors.Is(err, errGardenTestRegisterLive) {
		t.Fatalf("guard with HOME symlinked out of TMPDIR = %v, want errGardenTestRegisterLive", err)
	}
}

func TestGardenPathUnderDir(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name, path string
		want       bool
	}{
		{"equal", dir, true},
		{"nested", filepath.Join(dir, "a", "b"), true},
		{"sibling prefix", dir + "-sibling", false},
		{"parent", filepath.Dir(dir), false},
		{"outside", filepath.Join(filepath.Dir(dir), "elsewhere", "x"), false},
		{"dotdot escape", filepath.Join(dir, "..", "x"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := pathUnderDir(tc.path, dir); got != tc.want {
				t.Fatalf("pathUnderDir(%q, %q) = %v, want %v", tc.path, dir, got, tc.want)
			}
		})
	}
}

func TestGardenRegisterUnitWritersRefuseHomeOutsideTempRoots(t *testing.T) {
	writers := map[string]func(stdout, stderr *bytes.Buffer, root string) int{
		"systemd": func(o, e *bytes.Buffer, root string) int {
			return registerLinuxSystemdTimer(o, e, "/usr/local/bin/fak", root, time.Hour, false)
		},
		"launchd": func(o, e *bytes.Buffer, root string) int {
			return registerDarwinLaunchdAgent(o, e, "/usr/local/bin/fak", root, time.Hour, false)
		},
	}
	for name, register := range writers {
		t.Run(name, func(t *testing.T) {
			r := newGardenGuardRoots(t)
			home := mkGardenGuardHome(t, r.outside)
			var stdout, stderr bytes.Buffer
			if code := register(&stdout, &stderr, r.base); code != 1 {
				t.Fatalf("exit = %d, want 1; stdout=%q", code, stdout.String())
			}
			if stderr.Len() == 0 {
				t.Fatal("refusal printed nothing to stderr")
			}
			assertDirEmpty(t, home)
		})
	}
}
