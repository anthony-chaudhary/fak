package main

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/roofline"
)

// runBenchRoofline executes the empirical micro-roofline probe CLI.
// Supports: fak bench roofline --device=gfx1151 --json
func runBenchRoofline(stdout, stderr io.Writer, args []string) int {
	return roofline.RunCLI(stdout, stderr, args)
}
