package main

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

// planProfile is pure, so the resolved benchmark+profile command -- the whole point of
// the verb -- is pinned here: package + flags -> go test args, and Windows -> WSL via
// test.ps1.
func TestPlanProfile_GoArgs(t *testing.T) {
	cases := []struct {
		name   string
		opts   profileOpts
		goArgs []string
	}{
		{
			"defaults profile every benchmark",
			profileOpts{pkg: "./internal/ctxmmu/"},
			[]string{"-run=^$", "-bench=.", "-benchmem", "-cpuprofile", "cpu.out", "-memprofile", "mem.out", "./internal/ctxmmu/"},
		},
		{
			"custom bench and profile paths",
			profileOpts{pkg: "./internal/recall/", bench: "BenchmarkDigest", cpuProfile: "c.prof", memProfile: "m.prof"},
			[]string{"-run=^$", "-bench=BenchmarkDigest", "-benchmem", "-cpuprofile", "c.prof", "-memprofile", "m.prof", "./internal/recall/"},
		},
		{
			"benchtime is appended before the package",
			profileOpts{pkg: "internal/ctxmmu", benchtime: "2s"},
			[]string{"-run=^$", "-bench=.", "-benchmem", "-cpuprofile", "cpu.out", "-memprofile", "mem.out", "-benchtime", "2s", "internal/ctxmmu"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p, err := planProfile("linux", c.opts)
			if err != nil {
				t.Fatalf("planProfile(%+v) error: %v", c.opts, err)
			}
			if !reflect.DeepEqual(p.GoArgs, c.goArgs) {
				t.Errorf("goArgs = %v, want %v", p.GoArgs, c.goArgs)
			}
		})
	}
}

func TestPlanProfile_WindowsRoutesToWSL(t *testing.T) {
	p, err := planProfile("windows", profileOpts{pkg: "./internal/ctxmmu/"})
	if err != nil {
		t.Fatalf("planProfile error: %v", err)
	}
	if !p.ViaWSL {
		t.Fatalf("windows host must route via WSL, got ViaWSL=false")
	}
	joined := strings.Join(p.Argv, " ")
	if !strings.Contains(joined, "test.ps1") {
		t.Errorf("windows Argv must invoke test.ps1, got %q", joined)
	}
	// The package target must still be forwarded to the wrapper verbatim (last arg).
	if p.Argv[len(p.Argv)-1] != "./internal/ctxmmu/" {
		t.Errorf("windows Argv must forward the package target, got %q", joined)
	}
}

func TestPlanProfile_NonWindowsRunsGoTestDirectly(t *testing.T) {
	p, err := planProfile("darwin", profileOpts{pkg: "./internal/ctxmmu/"})
	if err != nil {
		t.Fatalf("planProfile error: %v", err)
	}
	if p.ViaWSL {
		t.Fatalf("non-windows host must not route via WSL")
	}
	if len(p.Argv) < 2 || p.Argv[0] != "go" || p.Argv[1] != "test" {
		t.Errorf("non-windows Argv must start with `go test`, got %v", p.Argv)
	}
}

func TestPlanProfile_RejectsMissingOrBadTarget(t *testing.T) {
	if _, err := planProfile("linux", profileOpts{pkg: ""}); err == nil {
		t.Errorf("an empty package target must error")
	}
	if _, err := planProfile("linux", profileOpts{pkg: "notapackage"}); err == nil {
		t.Errorf("a non-package token must error rather than be handed to go test")
	}
}

// The dry-run shell prints the resolved command and runs nothing -- the safe path to
// exercise the verb end-to-end without launching a benchmark.
func TestRunProfile_DryRunPrintsCommand(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runProfile(&out, &errb, []string{"-n", "./internal/ctxmmu/"}); rc != 0 {
		t.Fatalf("dry run rc = %d, stderr=%s", rc, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "-cpuprofile") || !strings.Contains(s, "./internal/ctxmmu/") {
		t.Errorf("dry run output missing resolved profile command: %q", s)
	}
	if !strings.Contains(s, "go tool pprof") {
		t.Errorf("dry run output should point at go tool pprof: %q", s)
	}
}

func TestRunProfile_MissingTargetExitsTwo(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runProfile(&out, &errb, []string{"-n"}); rc != 2 {
		t.Errorf("missing target rc = %d, want 2", rc)
	}
}

func TestRunProfile_ListExitsZero(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runProfile(&out, &errb, []string{"--list"}); rc != 0 {
		t.Fatalf("--list rc = %d", rc)
	}
	if !strings.Contains(out.String(), "cpuprofile") {
		t.Errorf("--list output missing capture description: %q", out.String())
	}
}
