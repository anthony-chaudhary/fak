package devcmd

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// RunAMDGotchas runs the AMD Strix Halo top 20 gotchas auditor and remediation advisor.
func RunAMDGotchas(stdout, stderr io.Writer, argv []string) int {
	return amdgpu.RunGotchasCLI(stdout, stderr, argv)
}
