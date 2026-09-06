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
	Baseline scoreboard.WorktreeABArm  `json:"baseline"`
	Isolated scoreboard.WorktreeABArm  `json:"isolated"`
	Trunk    *scoreboard.WorktreeABArm `json:"trunk,omitempty"`
	Worktree *scoreboard.WorktreeABArm `json:"worktree,omitempty"`
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
	if input.Trunk != nil {
		if input.Baseline.WaveID == "" {
			input.Baseline.WaveID = input.Trunk.WaveID
		}
		if input.Baseline.Resolved == 0 {
			input.Baseline.Resolved = input.Trunk.Resolved
		}
		if input.Baseline.DurationSeconds == 0 {
			input.Baseline.DurationSeconds = input.Trunk.DurationSeconds
		}
		if input.Baseline.PoisonIncidents == 0 {
			input.Baseline.PoisonIncidents = input.Trunk.PoisonIncidents
		}
		if input.Baseline.PeakConcurrency == 0 {
			input.Baseline.PeakConcurrency = input.Trunk.PeakConcurrency
		}
		if input.Baseline.HostID == "" {
			input.Baseline.HostID = input.Trunk.HostID
		}
		if len(input.Baseline.DeliveryRecords) == 0 {
			input.Baseline.DeliveryRecords = input.Trunk.DeliveryRecords
		}
		if len(input.Baseline.LifecycleRecords) == 0 {
			input.Baseline.LifecycleRecords = input.Trunk.LifecycleRecords
		}
		if input.Baseline.Accounting.Status == "" && input.Trunk.Accounting.Status != "" {
			input.Baseline.Accounting = input.Trunk.Accounting
		}
	}
	if input.Worktree != nil {
		if input.Isolated.WaveID == "" {
			input.Isolated.WaveID = input.Worktree.WaveID
		}
		if input.Isolated.Resolved == 0 {
			input.Isolated.Resolved = input.Worktree.Resolved
		}
		if input.Isolated.DurationSeconds == 0 {
			input.Isolated.DurationSeconds = input.Worktree.DurationSeconds
		}
		if input.Isolated.PoisonIncidents == 0 {
			input.Isolated.PoisonIncidents = input.Worktree.PoisonIncidents
		}
		if input.Isolated.PeakConcurrency == 0 {
			input.Isolated.PeakConcurrency = input.Worktree.PeakConcurrency
		}
		if input.Isolated.HostID == "" {
			input.Isolated.HostID = input.Worktree.HostID
		}
		if len(input.Isolated.DeliveryRecords) == 0 {
			input.Isolated.DeliveryRecords = input.Worktree.DeliveryRecords
		}
		if len(input.Isolated.LifecycleRecords) == 0 {
			input.Isolated.LifecycleRecords = input.Worktree.LifecycleRecords
		}
		if input.Isolated.Accounting.Status == "" && input.Worktree.Accounting.Status != "" {
			input.Isolated.Accounting = input.Worktree.Accounting
		}
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
