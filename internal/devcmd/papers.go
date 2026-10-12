package devcmd

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/papers"
)

func RunPapers(stdout, stderr io.Writer, argv []string) int {
	return papers.Run(argv, stdout, stderr)
}
