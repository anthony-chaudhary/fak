package main

import (
	"context"
	"io"

	"github.com/anthony-chaudhary/fak/internal/agentbench"
)

// runBenchAgent executes the bounded same-repository coding-agent benchmark.
func runBenchAgent(stdout, stderr io.Writer, args []string) int {
	return agentbench.RunCLI(context.Background(), stdout, stderr, args)
}
