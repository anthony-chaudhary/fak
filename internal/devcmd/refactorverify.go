package devcmd

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/refactorverify"
)

// RunRefactorVerify proves that a code-motion refactor dropped no top-level declaration.
func RunRefactorVerify(stdout, stderr io.Writer, argv []string) int {
	return refactorverify.Run(stdout, stderr, argv)
}
