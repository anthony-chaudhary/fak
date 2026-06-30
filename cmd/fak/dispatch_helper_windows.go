//go:build windows

package main

import (
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func configureDispatchHelperCommand(cmd *exec.Cmd) {
	windowgate.ConfigureBackgroundCommand(cmd)
}
