package main

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/localbench"
)

func runBenchLocal(stdout, stderr io.Writer, args []string) int {
	if err := localbench.RunCLI(args, stdout, stderr); err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	return 0
}
