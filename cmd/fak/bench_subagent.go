package main

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/qwen38campaign"
)

// runBenchSubagent executes the subagent fan-out multi-agent benchmark harness CLI.
// Supports: fak bench subagent --scenario=shared_prefix_forked --concurrency=4 --runs=5 --json
func runBenchSubagent(stdout, stderr io.Writer, args []string) int {
	return qwen38campaign.RunCLI(stdout, stderr, args)
}
