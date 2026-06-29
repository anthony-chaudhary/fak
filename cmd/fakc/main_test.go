package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFakcArgvDelegatesToFakCodex(t *testing.T) {
	got := fakcArgv("C:/bin/fak.exe", []string{"--split", "off", "--", "exec", "do x"})
	want := []string{"C:/bin/fak.exe", "codex", "--split", "off", "--", "exec", "do x"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("fakcArgv = %#v, want %#v", got, want)
	}
}

func TestFakcDryRunOnlyBeforeSeparator(t *testing.T) {
	cases := []struct {
		args []string
		want bool
	}{
		{[]string{"--dry-run"}, true},
		{[]string{"-dry-run"}, true},
		{[]string{"--split", "off", "--dry-run"}, true},
		{[]string{"--", "--dry-run"}, false},
		{[]string{"--", "exec", "--dry-run"}, false},
		{[]string{"--split", "off"}, false},
	}
	for _, tc := range cases {
		if got := fakcDryRun(tc.args); got != tc.want {
			t.Fatalf("fakcDryRun(%v) = %v, want %v", tc.args, got, tc.want)
		}
	}
}

func TestResolveFakBinaryPrefersEnv(t *testing.T) {
	got, err := resolveFakBinary(
		func(k string) string {
			if k == "FAK_BIN" {
				return "C:/custom/fak.exe"
			}
			return ""
		},
		func() (string, error) { return "", errors.New("no exe") },
		func(string) (string, error) { return "", errors.New("no path") },
		func() (string, error) { return "", errors.New("no wd") },
		"windows",
	)
	if err != nil || got != "C:/custom/fak.exe" {
		t.Fatalf("resolve env = %q,%v", got, err)
	}
}

func TestResolveFakBinaryFindsSibling(t *testing.T) {
	dir := t.TempDir()
	fak := filepath.Join(dir, "fak.exe")
	if err := os.WriteFile(fak, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := resolveFakBinary(
		func(string) string { return "" },
		func() (string, error) { return filepath.Join(dir, "fakc.exe"), nil },
		func(string) (string, error) { return "", errors.New("no path") },
		func() (string, error) { return "", errors.New("no wd") },
		"windows",
	)
	if err != nil || got != fak {
		t.Fatalf("resolve sibling = %q,%v; want %q", got, err, fak)
	}
}

func TestResolveFakBinaryFallsBackToPath(t *testing.T) {
	got, err := resolveFakBinary(
		func(string) string { return "" },
		func() (string, error) { return "", errors.New("no exe") },
		func(name string) (string, error) {
			if name == "fak" {
				return "/usr/local/bin/fak", nil
			}
			return "", errors.New("no path")
		},
		func() (string, error) { return "", errors.New("no wd") },
		"linux",
	)
	if err != nil || got != "/usr/local/bin/fak" {
		t.Fatalf("resolve path = %q,%v", got, err)
	}
}

func TestRunFakcDryRunWithoutFakStillPrintsDelegation(t *testing.T) {
	orig := fakcResolve
	fakcResolve = func(func(string) string, func() (string, error), func(string) (string, error), func() (string, error), string) (string, error) {
		return "", errors.New("missing fak")
	}
	t.Cleanup(func() { fakcResolve = orig })

	var out, errb bytes.Buffer
	rc := runFakc(&out, &errb, []string{"--dry-run", "--split", "off", "--", "exec", "do x"})
	if rc != 0 {
		t.Fatalf("dry-run rc=%d stderr=%s", rc, errb.String())
	}
	if got := strings.TrimSpace(out.String()); got != "fak codex --dry-run --split off -- exec do x" {
		t.Fatalf("dry-run stdout = %q", got)
	}
}

func TestRunFakcExecSeam(t *testing.T) {
	dir := t.TempDir()
	fak := filepath.Join(dir, "fak.exe")
	if err := os.WriteFile(fak, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_BIN", fak)

	orig := fakcRun
	var gotArgv []string
	fakcRun = func(_, _ io.Writer, argv, _ []string) int {
		gotArgv = append([]string{}, argv...)
		return 23
	}
	t.Cleanup(func() { fakcRun = orig })

	var out, errb bytes.Buffer
	rc := runFakc(&out, &errb, []string{"--split", "off"})
	if rc != 23 {
		t.Fatalf("runFakc rc=%d, want seam rc 23; stderr=%s", rc, errb.String())
	}
	want := []string{fak, "codex", "--split", "off"}
	if !reflect.DeepEqual(gotArgv, want) {
		t.Fatalf("runFakc argv = %#v, want %#v", gotArgv, want)
	}
}
