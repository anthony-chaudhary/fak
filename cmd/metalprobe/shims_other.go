//go:build !windows

// Non-windows host shims for the metal-check gate: real environment and PATH lookups
// plus the cgo linkcheck leaf executor. The probe-line parser lives in metalcheck.go
// so every host compiles and tests it.
package main

import (
	"os/exec"
)

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
	cmd := exec.Command("go", "run", "./cmd/metalprobe/linkcheck")
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
