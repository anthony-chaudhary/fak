package main

import (
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func configureProcTree(cmd *exec.Cmd) {
	procguard.ConfigureProcessTreeCancel(cmd)
}
