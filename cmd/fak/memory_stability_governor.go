package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/memorystability"
)

//fak:ctxplan verb=memory-stability-governor enters="nothing live — an offline fold over a replay-trajectory JSON file and the workspace root, scoring per-reload memory drift" pages="nothing into a window — it emits a memory-stability report (per-reload drift vs tau, cumulative budget, per-cycle floor) as text or JSON" warms="nothing — a governor that grades memory-reload stability; it touches no prompt cache or KV"
func cmdMemoryStabilityGovernor(argv []string) {
	os.Exit(runMemoryStabilityGovernor(os.Stdout, os.Stderr, argv))
}

func runMemoryStabilityGovernor(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("memory-stability-governor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	trajectory := fs.String("trajectory", "", "path to a replay-trajectory JSON file")
	tau := fs.Float64("tau", memorystability.DefaultTau, "per-reload point threshold")
	budget := fs.Float64("budget", memorystability.DefaultBudget, "cumulative drift budget")
	floor := fs.Float64("floor", memorystability.DefaultFloor, "per-cycle stability floor")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	root := *workspace
	if root == "" {
		root = resolveRoot("")
		if root == "" {
			root = "."
		}
	}
	report := memorystability.Collect(root, *trajectory, *tau, *budget, *floor)
	if *asJSON {
		data, err := memorystability.MarshalJSON(report)
		if err != nil {
			fmt.Fprintf(stderr, "memory-stability-governor: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintln(stdout, memorystability.Render(report))
	}
	return memorystability.ExitCode(report)
}
