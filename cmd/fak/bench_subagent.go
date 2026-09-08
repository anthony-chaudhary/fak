package main

import (
	"io"

	"github.com/anthony-chaudhary/fak/internal/qwen38campaign"
)

// runBenchSubagent executes the subagent fan-out multi-agent benchmark harness CLI.
// Supports: fak bench subagent --scenario=shared_prefix_forked --concurrency=4 --runs=5 --json
func runBenchSubagent(stdout, stderr io.Writer, args []string) int {
	return runBenchSubagentWithRunner(stdout, stderr, args, nil)
}

// runBenchSubagentWithRunner is the product attachment boundary. Production
// physical mode stays unavailable until the fak-native product path supplies an
// observed-telemetry runner; simulation remains available without one.
func runBenchSubagentWithRunner(stdout, stderr io.Writer, args []string, runner qwen38campaign.PhysicalRunner) int {
	return qwen38campaign.RunWithRunner(stdout, stderr, args, runner)
}
