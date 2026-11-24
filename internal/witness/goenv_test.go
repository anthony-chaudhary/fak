package witness

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func envValue(env []string, key string) (string, bool) {
	val, found := "", false
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			val, found = v, true
		}
	}
	return val, found
}

func TestWitnessGoBuildEnv(t *testing.T) {
	shared := filepath.Join(t.TempDir(), "shared")
	for _, tc := range []struct {
		name      string
		env       []string
		wantFlags string
		wantCache string
	}{
		{name: "appends", env: []string{"GOFLAGS=-mod=mod", "GOCACHE=/lane/.gocache"}, wantFlags: "-mod=mod -trimpath", wantCache: "/lane/.gocache"},
		{name: "already-trimmed", env: []string{"GOFLAGS=-trimpath"}, wantFlags: "-trimpath"},
		{name: "opt-out", env: []string{"GOFLAGS=-race", "FAK_NO_TRIMPATH=1"}, wantFlags: "-race"},
		{name: "shared-cache-wins", env: []string{"GOCACHE=/lane/.gocache", "FAK_SHARED_GOCACHE=" + shared}, wantFlags: "-trimpath", wantCache: shared},
		{name: "relative-shared-ignored", env: []string{"GOCACHE=/lane/.gocache", "FAK_SHARED_GOCACHE=rel"}, wantFlags: "-trimpath", wantCache: "/lane/.gocache"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := witnessGoBuildEnv(tc.env)
			if v, _ := envValue(got, "GOFLAGS"); v != tc.wantFlags {
				t.Errorf("GOFLAGS = %q, want %q", v, tc.wantFlags)
			}
			if v, _ := envValue(got, "GOCACHE"); v != tc.wantCache {
				t.Errorf("GOCACHE = %q, want %q", v, tc.wantCache)
			}
		})
	}
}

// The runner applies the rule to every go child, which is how a witness
// snapshot's compiles reach reusable cache entries.
func TestCommandRunnerGoChildGetsTrimpathAndSharedCache(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	shared := t.TempDir()
	t.Setenv("GOFLAGS", "-mod=mod")
	t.Setenv("FAK_NO_TRIMPATH", "")
	t.Setenv("FAK_SHARED_GOCACHE", shared)
	out, code, err := commandRunner(context.Background(), t.TempDir(), "go", "env", "GOFLAGS", "GOCACHE")
	if err != nil || code != 0 {
		t.Fatalf("go env: code=%d err=%v out=%s", code, err, out)
	}
	lines := strings.Split(strings.TrimSpace(strings.ReplaceAll(out, "\r\n", "\n")), "\n")
	if len(lines) != 2 || lines[0] != "-mod=mod -trimpath" || lines[1] != shared {
		t.Fatalf("go env GOFLAGS GOCACHE = %q, want [%q %q]", lines, "-mod=mod -trimpath", shared)
	}

	t.Setenv("FAK_NO_TRIMPATH", "1")
	out, code, err = commandRunner(context.Background(), t.TempDir(), "go", "env", "GOFLAGS")
	if err != nil || code != 0 || strings.TrimSpace(out) != "-mod=mod" {
		t.Fatalf("opt-out go env GOFLAGS = %q (code=%d err=%v), want -mod=mod", out, code, err)
	}
}
