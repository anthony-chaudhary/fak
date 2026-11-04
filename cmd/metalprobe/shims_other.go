//go:build !windows

// Non-windows host shims for the metal-check gate: real environment and PATH lookups
// plus the cgo linkcheck leaf executor. The probe-line parser lives in metalcheck.go
// so every host compiles and tests it.
package main

import (
	"context"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/sysproc"
)

// cgoEnabledFromToolchain is the host source of Requirements.CgoEnabled: the
// selected Go toolchain's effective CGO setting, queried explicitly (#13605
// leaf K / #13611). It is a var so tests can substitute a deterministic
// toolchain answer; the default is the bounded `go env CGO_ENABLED` query.
var cgoEnabledFromToolchain = querySelectedCgoEnabled

// querySelectedCgoEnabled runs `<go> env CGO_ENABLED` against the go executable
// on PATH under a hard bound and returns the trimmed toolchain answer. The
// output of `go env` is toolchain data, not an ambient configuration read: it
// reports the effective setting the selected toolchain will actually build
// with, which an unset ambient variable does not determine. A failed query
// (no go, timeout, nonzero exit) reports "" — the historical unset value — so
// the gate falls through to the GoTool prerequisite, which names the missing
// toolchain explicitly instead of guessing at cgo state.
func querySelectedCgoEnabled() string {
	goExe, err := exec.LookPath("go")
	if err != nil {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := sysproc.CommandContext(ctx, goExe, "env", "CGO_ENABLED")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// checkGoTool verifies the go toolchain is present in PATH.
func checkGoTool() ToolCheck {
	if path, err := exec.LookPath("go"); err == nil {
		return ToolCheck{OK: true, Extra: path}
	}
	return ToolCheck{OK: false, Extra: "'go' not found in PATH"}
}

// checkXcodeCLT verifies the Xcode Command Line Tools are selected (`xcode-select -p`).
func checkXcodeCLT() ToolCheck {
	if err := exec.Command("xcode-select", "-p").Run(); err == nil {
		return ToolCheck{OK: true}
	}
	return ToolCheck{OK: false, Extra: "'xcode-select -p' failed"}
}

// checkClang verifies a clang binary is present in PATH.
func checkClang() ToolCheck {
	if path, err := exec.LookPath("clang"); err == nil {
		return ToolCheck{OK: true, Extra: path}
	}
	return ToolCheck{OK: false, Extra: "'clang' not found in PATH"}
}

// runLinkcheck executes the cgo probe leaf through the go tool, returning its
// combined output and exit code.
func runLinkcheck() (string, int) {
	cmd := sysproc.Command("go", "run", "./cmd/metalprobe/linkcheck")
	outBytes, err := cmd.CombinedOutput()
	exit := 0
	if err != nil {
		exit = 1
		if ee, ok := err.(*exec.ExitError); ok {
			exit = ee.ExitCode()
		}
	}
	return string(outBytes), exit
}
