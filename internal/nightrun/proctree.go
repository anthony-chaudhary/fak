package nightrun

import (
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func configureProcGroup(cmd *exec.Cmd) {
	procguard.ConfigureProcessTreeCancel(cmd)
}
