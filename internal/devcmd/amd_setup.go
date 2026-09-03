package devcmd

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// RunAMDSetup runs the AMD GPU hardware governor and TTM memory ceiling configurator.
func RunAMDSetup(stdout, stderr io.Writer, argv []string) int {
	return amdgpu.RunCLI(stdout, stderr, argv)
}
