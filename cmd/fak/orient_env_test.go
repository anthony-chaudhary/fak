package main

import (
	"bytes"
	"encoding/json"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/preflight"
	"github.com/anthony-chaudhary/fak/internal/testroute"
)

func TestRunOrientEnvJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrientEnv(&stdout, &stderr, []string{"--json", "--leases=false", "--paths", "internal/preflight"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	var rep preflight.EnvReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("output is not JSON: %v\n%s", err, stdout.String())
	}
	if rep.Schema != preflight.EnvPreflightSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, preflight.EnvPreflightSchema)
	}
	if runtime.GOOS == "windows" {
		if rep.TestRoute.Kind != testroute.KindWSL {
			t.Fatalf("test route = %q, want %q on windows", rep.TestRoute.Kind, testroute.KindWSL)
		}
		if rep.GitShell.Shell != preflight.GitShellPowerShell {
			t.Fatalf("git shell = %q, want %q on windows", rep.GitShell.Shell, preflight.GitShellPowerShell)
		}
	} else {
		if rep.TestRoute.Kind != testroute.KindNative {
			t.Fatalf("test route = %q, want %q off windows", rep.TestRoute.Kind, testroute.KindNative)
		}
	}
	if len(rep.Paths) != 1 || rep.Paths[0] != "internal/preflight" {
		t.Fatalf("paths = %v, want the declared target", rep.Paths)
	}
}

func TestRunOrientEnvText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrientEnv(&stdout, &stderr, []string{"--leases=false"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"verdict", "test_route", "git_shell", "leases      none live"} {
		if !strings.Contains(out, want) {
			t.Fatalf("text output missing %q:\n%s", want, out)
		}
	}
}

func TestRunOrientEnvBadFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runOrientEnv(&stdout, &stderr, []string{"--no-such-flag"}); code != 2 {
		t.Fatalf("exit = %d, want 2 for a bad flag", code)
	}
}
