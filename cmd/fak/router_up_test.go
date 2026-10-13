package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type routerUpFake struct {
	ran     [][]string
	code    int
	healthy bool
}

func (f *routerUpFake) deps(home string) routerUpDeps {
	return routerUpDeps{
		getenv:   func(string) string { return "" },
		homeDir:  func() (string, error) { return home, nil },
		goos:     "linux",
		lookPath: func(string) (string, error) { return "/usr/bin/fak-server", nil },
		healthy:  func(string) bool { return f.healthy },
		run: func(argv []string, _, _ io.Writer) (int, error) {
			f.ran = append(f.ran, argv)
			return f.code, nil
		},
	}
}

func TestRouterHomeLayout(t *testing.T) {
	f := &routerUpFake{}
	d := f.deps("/h")
	if got, _ := routerHome(d); got != filepath.Join("/h", ".fak", "router") {
		t.Fatalf("unix routerHome = %q", got)
	}
	d.goos = "windows"
	d.getenv = func(k string) string {
		if k == "LOCALAPPDATA" {
			return `C:\Users\a\AppData\Local`
		}
		return ""
	}
	if got, _ := routerHome(d); got != filepath.Join(`C:\Users\a\AppData\Local`, "fak", "router") {
		t.Fatalf("windows routerHome = %q", got)
	}
}

func TestRouterUpDerivesGatewayOnlyLaunch(t *testing.T) {
	home := t.TempDir()
	f := &routerUpFake{}
	var out, errOut bytes.Buffer
	if code := runRouterUp(&out, &errOut, nil, f.deps(home)); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if len(f.ran) != 1 {
		t.Fatalf("ran %d times", len(f.ran))
	}
	r := filepath.Join(home, ".fak", "router")
	want := []string{"/usr/bin/fak-server", "route",
		"--gateway-key-file", filepath.Join(r, "gateway.key"),
		"--usage-receipts", filepath.Join(r, "usage.jsonl"),
		"--addr", "127.0.0.1:18095",
		"--routing-config", filepath.Join(r, "routing.json"),
		"--routing-journal", filepath.Join(r, "journal.jsonl"),
		"--init-routing-config"}
	if strings.Join(f.ran[0], "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("argv = %q\nwant  %q", f.ran[0], want)
	}
	for _, a := range f.ran[0] {
		if a == "--local-engine" || a == "--engine-commit" {
			t.Fatalf("router launch carries %s", a)
		}
	}
	info, err := os.Stat(filepath.Join(r, "gateway.key"))
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Errorf("gateway.key mode = %v, want 0600", info.Mode().Perm())
	}
}

func TestEnsureGatewayKeyNeverRewrites(t *testing.T) {
	p := filepath.Join(t.TempDir(), "gateway.key")
	if err := os.WriteFile(p, []byte("existing"), 0o600); err != nil {
		t.Fatal(err)
	}
	if created, err := ensureGatewayKey(p); err != nil || created {
		t.Fatalf("ensureGatewayKey = %v, %v", created, err)
	}
	if b, _ := os.ReadFile(p); string(b) != "existing" {
		t.Fatalf("key rewritten: %q", b)
	}
}

func TestRouterUpDryRunWritesNothing(t *testing.T) {
	home := t.TempDir()
	f := &routerUpFake{}
	var out bytes.Buffer
	if code := runRouterUp(&out, io.Discard, []string{"--dry-run"}, f.deps(home)); code != 0 {
		t.Fatalf("exit %d", code)
	}
	if len(f.ran) != 0 {
		t.Fatal("dry run executed fak-server")
	}
	if !strings.Contains(out.String(), " route ") || !strings.Contains(out.String(), "18095") {
		t.Fatalf("dry run output: %q", out.String())
	}
	if _, err := os.Stat(filepath.Join(home, ".fak")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("dry run created router state")
	}
}

func TestRouterUpAlreadyHealthyExitsZero(t *testing.T) {
	f := &routerUpFake{healthy: true, code: 1}
	if code := runRouterUp(io.Discard, io.Discard, nil, f.deps(t.TempDir())); code != 0 || len(f.ran) != 0 {
		t.Fatalf("exit %d, ran %d", code, len(f.ran))
	}
}

func TestRouterUpPropagatesConfigExit(t *testing.T) {
	f := &routerUpFake{code: exitRouterConfig}
	if code := runRouterUp(io.Discard, io.Discard, nil, f.deps(t.TempDir())); code != exitRouterConfig {
		t.Fatalf("exit %d, want %d", code, exitRouterConfig)
	}
}

func TestRouterUpLogFile(t *testing.T) {
	home := t.TempDir()
	log := filepath.Join(home, "router.log")
	f := &routerUpFake{}
	d := f.deps(home)
	d.run = func(argv []string, out, _ io.Writer) (int, error) {
		_, _ = io.WriteString(out, "hello\n")
		return 0, nil
	}
	if code := runRouterUp(io.Discard, io.Discard, []string{"--log-file", log}, d); code != 0 {
		t.Fatalf("exit %d", code)
	}
	if b, _ := os.ReadFile(log); string(b) != "hello\n" {
		t.Fatalf("log file = %q", b)
	}
}

func TestRouterUnknownSubcommand(t *testing.T) {
	if code := runRouter(io.Discard, io.Discard, []string{"down"}, (&routerUpFake{}).deps(t.TempDir())); code != 2 {
		t.Fatalf("exit %d", code)
	}
}
