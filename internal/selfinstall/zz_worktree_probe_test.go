package selfinstall

// THROWAWAY empirical probe — DELETE before committing.

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func probeRun4(t *testing.T, env []string, dir, name string, args ...string) (string, bool) {
	t.Helper()
	cmd := exec.CommandContext(context.Background(), name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	if env != nil {
		cmd.Env = append(os.Environ(), env...)
	}
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

const tinyMain4 = `package main
import ("fmt";"runtime/debug")
func main(){ bi,ok:=debug.ReadBuildInfo(); if !ok {fmt.Println("no buildinfo");return}; got:=false; for _,s:=range bi.Settings { if s.Key=="vcs.revision"||s.Key=="vcs.modified"{fmt.Println(s.Key,s.Value);got=true} }; if !got {fmt.Println("NO-VCS-SETTINGS")} }
`

func TestZZProbeDecisive(t *testing.T) {
	root, _ := probeRun4(t, nil, ".", "git", "rev-parse", "--show-toplevel")
	repoRoot := strings.TrimSpace(root)

	tinyDir := filepath.Join(repoRoot, "cmd", "zzztinydec")
	_ = os.MkdirAll(tinyDir, 0o755)
	_ = os.WriteFile(filepath.Join(tinyDir, "main.go"), []byte(tinyMain4), 0o644)
	defer os.RemoveAll(tinyDir)

	for _, tc := range []struct {
		label string
		env   []string
	}{
		{"MAIN(.git=dir) default(1.26.3)", nil},
		{"MAIN(.git=dir) GOTOOLCHAIN=go1.26.5", []string{"GOTOOLCHAIN=go1.26.5"}},
	} {
		gv, _ := probeRun4(t, tc.env, repoRoot, "go", "version")
		bin := filepath.Join(t.TempDir(), "t.exe")
		out, ok := probeRun4(t, tc.env, repoRoot, "go", "build", "-buildvcs=true", "-o", bin, "./cmd/zzztinydec")
		settings := ""
		if ok {
			settings, _ = probeRun4(t, nil, "", bin)
		}
		t.Logf("[%s] goversion=%q build_ok=%v out=%q settings=%q",
			tc.label, strings.TrimSpace(gv), ok, strings.TrimSpace(out), strings.TrimSpace(settings))
	}
	_ = os.RemoveAll(tinyDir)
}
