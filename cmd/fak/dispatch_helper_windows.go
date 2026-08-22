//go:build windows

package main

import (
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func configureDispatchHelperCommand(cmd *exec.Cmd) {
	// Helpers such as gh can launch ordinary console descendants (notably git).
	// Give the short-lived helper tree one hidden console to inherit; a detached
	// helper leaves those descendants to allocate visible desktop consoles.
	windowgate.ConfigureBackgroundCommand(cmd)
}
