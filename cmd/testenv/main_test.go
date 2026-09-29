package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// binPath is the real cmd/testenv binary built once in TestMain so every case
// drives the actual process boundary (main -> testenv.Run -> child), never an
// in-memory mock. A case that passed with the binary absent or replaced by
// `exit 0` would fail these assertions.
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "testenv-bin-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	name := "testenv"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binPath = filepath.Join(dir, name)

	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stdout = os.Stderr
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		panic("building cmd/testenv: " + err.Error())
	}
	os.Exit(m.Run())
}

// runBin drives the built binary as a subprocess and returns exit code, stdout
// and stderr. extraEnv is appended to the inherited environment so a case can
// plant a credential-shaped variable the binary is expected to scrub.
func runBin(t *testing.T, extraEnv []string, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	cmd.Env = append(os.Environ(), extraEnv...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			t.Fatalf("running %v: %v", args, err)
		}
		code = ee.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

// echoArgs returns a child command that prints marker on stdout and exits 0.
// It is the delegation target for the success and scrub cases.
func echoArgs(marker string) []string {
	if runtime.GOOS == "windows" {
		return []string{"cmd", "/c", "echo", marker}
	}
	return []string{"sh", "-c", "echo " + marker}
}

func TestNoArgsPrintsUsageToStderr(t *testing.T) {
	code, stdout, stderr := runBin(t, nil)
	if code == 0 {
		t.Fatalf("no args: exit = 0, want nonzero")
	}
	if stdout != "" {
		t.Errorf("no args: stdout = %q, want empty (usage belongs on stderr)", stdout)
	}
	if !strings.Contains(stderr, "testenv: command is required") {
		t.Errorf("no args: stderr = %q, want it to contain %q", stderr, "testenv: command is required")
	}
}

func TestUnknownCommandFails(t *testing.T) {
	code, stdout, stderr := runBin(t, nil, "testenv-no-such-command-xyz")
	if code == 0 {
		t.Fatalf("unknown command: exit = 0, want nonzero")
	}
	if stdout != "" {
		t.Errorf("unknown command: stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "testenv:") {
		t.Errorf("unknown command: stderr = %q, want the testenv error class", stderr)
	}
}

func TestSuccessfulDelegation(t *testing.T) {
	const marker = "testenv-delegate-ok"
	// The leading "--" exercises main's argument-prefix stripping.
	args := append([]string{"--"}, echoArgs(marker)...)
	code, stdout, stderr := runBin(t, nil, args...)
	if code != 0 {
		t.Fatalf("delegation: exit = %d, want 0; stderr = %q", code, stderr)
	}
	if !strings.Contains(stdout, marker) {
		t.Errorf("delegation: stdout = %q, want it to contain %q", stdout, marker)
	}
}

func TestChildFailurePropagates(t *testing.T) {
	var args []string
	if runtime.GOOS == "windows" {
		args = []string{"cmd", "/c", "exit", "3"}
	} else {
		args = []string{"sh", "-c", "exit 3"}
	}
	code, _, stderr := runBin(t, nil, args...)
	if code == 0 {
		t.Fatalf("child failure: exit = 0, want nonzero")
	}
	if !strings.Contains(stderr, "testenv:") {
		t.Errorf("child failure: stderr = %q, want the wrapped testenv error class", stderr)
	}
}

// TestCredentialScrubbedFromChildEnv witnesses the leaf's actual purpose over
// the real process boundary: a credential-shaped variable set in the parent
// must not be visible to the delegated child.
func TestCredentialScrubbedFromChildEnv(t *testing.T) {
	const secret = "supersecret-delegate-value"
	const name = "FAKE_API_KEY" // matches envconfiglint's credential-token rule

	var child []string
	if runtime.GOOS == "windows" {
		child = []string{"cmd", "/c", "echo", "%" + name + "%"}
	} else {
		child = []string{"sh", "-c", "echo \"$" + name + "\""}
	}
	code, stdout, _ := runBin(t, []string{name + "=" + secret}, child...)
	if code != 0 {
		t.Fatalf("scrub delegation: exit = %d, want 0", code)
	}
	if strings.Contains(stdout, secret) {
		t.Errorf("credential leaked to child: stdout = %q contains the planted secret", stdout)
	}
}
