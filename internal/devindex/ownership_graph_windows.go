//go:build windows

package devindex

import (
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func graphCommand(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	windowgate.ConfigureDetachedCommand(cmd)
	return cmd
}
