package devcmd

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// RunAMDStrixPackage generates the AMD Strix Halo installer package with LAN communications and gotcha settings.
func RunAMDStrixPackage(stdout, stderr io.Writer, argv []string) int {
	return amdgpu.RunStrixInstallerCLI(stdout, stderr, argv)
}
