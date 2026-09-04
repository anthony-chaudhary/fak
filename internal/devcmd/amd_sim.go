package devcmd

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// RunAMDStrixSim runs the AMD Strix Halo (Ryzen AI MAX+ 395) agent simulation and hardware verification.
func RunAMDStrixSim(stdout, stderr io.Writer, argv []string) int {
	return amdgpu.RunSimCLI(stdout, stderr, argv)
}
