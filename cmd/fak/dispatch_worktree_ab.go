package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

type worktreeABInput struct {
	Baseline scoreboard.WorktreeABArm `json:"baseline"`
	Isolated scoreboard.WorktreeABArm `json:"isolated"`
}

// runDispatchWorktreeAB folds two back-to-back executions of the same fixed wave.
// The runner records each arm as JSON; this command is the reproducible scorer and
// scoreboard-row producer, deliberately separate from live worker spawning.
func runDispatchWorktreeAB(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch worktree-ab", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "two-arm JSON fixture (default: stdin)")
	asJSON := fs.Bool("json", false, "emit machine-readable report")
	if !parseFlags(fs, argv) {
		return 2
	}
	var raw []byte
	var err error
	if *in == "" || *in == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(*in)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch worktree-ab: %v\n", err)
		return 1
	}
	var input worktreeABInput
	if err := json.Unmarshal(raw, &input); err != nil {
		fmt.Fprintf(stderr, "fak dispatch worktree-ab: parse: %v\n", err)
		return 2
	}
	rep := scoreboard.FoldWorktreeAB(input.Baseline, input.Isolated)
	if !scoreboard.WorktreeABEquivalentWave(rep.Baseline, rep.Isolated) {
		fmt.Fprintln(stderr, "fak dispatch worktree-ab: arms must resolve the same non-zero fixed wave")
		return 3
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch worktree-ab")
	}
	fmt.Fprintln(stdout, scoreboard.WorktreeABUpdate(rep).Text())
	return 0
}
