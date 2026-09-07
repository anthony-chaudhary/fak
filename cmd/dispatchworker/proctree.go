package main

import (
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func configureProcTree(cmd *exec.Cmd) {
	procguard.ConfigureProcessTreeCancel(cmd)
	windowgate.ConfigureWorkerCommand(cmd)
}
