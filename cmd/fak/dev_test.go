package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestRunDevHandoffExecutesSeparateArtifact(t *testing.T) {
	dir := t.TempDir()
	name := "fak-dev"
	body := "#!/bin/sh\nprintf 'child:%s\\n' \"$*\"\nprintf 'child-err\\n' >&2\nexit 7\n"
	if runtime.GOOS == "windows" {
		name += ".cmd"
		body = "@echo off\r\necho child:%*\r\necho child-err 1>&2\r\nexit /b 7\r\n"
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	old := findFakDev
	findFakDev = func() (string, error) { return path, nil }
	t.Cleanup(func() { findFakDev = old })

	var stdout, stderr bytes.Buffer
	got := runDevHandoff(strings.NewReader(""), &stdout, &stderr, []string{"index", "ownership", "--json"})
	if got != 7 {
		t.Fatalf("exit = %d, want child exit 7; stderr=%s", got, stderr.String())
	}
	if !strings.Contains(stdout.String(), "child:index ownership --json") {
		t.Fatalf("stdout did not prove argv handoff: %q", stdout.String())
	}
	if !strings.Contains(stderr.String(), "child-err") {
		t.Fatalf("stderr was not connected: %q", stderr.String())
	}
}

func TestRunDevHandoffMissingArtifactIsActionable(t *testing.T) {
	old := findFakDev
	findFakDev = func() (string, error) { return "", errors.New("not found") }
	t.Cleanup(func() { findFakDev = old })

	var stderr bytes.Buffer
	if got := runDevHandoff(strings.NewReader(""), io.Discard, &stderr, []string{"index"}); got != 2 {
		t.Fatalf("exit = %d, want 2", got)
	}
	for _, want := range []string{"separate 'fak-dev' executable", "fak-dev <command>"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("missing %q in actionable error:\n%s", want, stderr.String())
		}
	}
}

func TestExactTopLevelDevCommandsPreserveArgvAndChildExit(t *testing.T) {
	old := executeExactDevHandoff
	defer func() { executeExactDevHandoff = old }()
	var captured [][]string
	executeExactDevHandoff = func(_ io.Reader, _, _ io.Writer, argv []string) int {
		captured = append(captured, append([]string(nil), argv...))
		return 7
	}

	for _, tc := range []struct {
		verb string
		args []string
	}{
		{verb: "build", args: []string{"--profile", "release", "--json"}},
		{verb: "study-inventory", args: []string{"--self", "--verify", "--json"}},
	} {
		code, handled := runExactDevHandoff(nil, nil, nil, tc.verb, tc.args)
		if !handled || code != 7 {
			t.Fatalf("%s: handled=%v code=%d, want true/7", tc.verb, handled, code)
		}
		want := append([]string{tc.verb}, tc.args...)
		if got := captured[len(captured)-1]; !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: child argv = %v, want %v", tc.verb, got, want)
		}
	}
	if code, handled := runExactDevHandoff(nil, nil, nil, "buildcheck", []string{"--json"}); handled || code != 0 {
		t.Fatalf("non-exact command handled=%v code=%d, want false/0", handled, code)
	}
}

func TestRuntimeDependencyClosureDoesNotImportDevcmd(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	cmd := exec.Command("go", "list", "-deps", "./cmd/fak")
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list runtime dependencies: %v\n%s", err, out)
	}
	for _, dependency := range strings.Fields(string(out)) {
		if dependency == "github.com/anthony-chaudhary/fak/internal/devcmd" {
			t.Fatal("runtime fak dependency closure imports internal/devcmd")
		}
	}
}
