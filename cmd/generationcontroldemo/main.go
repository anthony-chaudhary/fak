package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/generationctl"
	"github.com/anthony-chaudhary/fak/internal/streamrules"
)

type event struct {
	Event      string                    `json:"event"`
	Epoch      *generationctl.Epoch      `json:"epoch,omitempty"`
	Directive  *generationctl.Directive  `json:"directive,omitempty"`
	Checkpoint *generationctl.Checkpoint `json:"checkpoint,omitempty"`
}

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the deterministic continuous-generation steering witness")
	flag.Parse()
	if !*selfcheck {
		fmt.Fprintln(os.Stderr, "usage: generationcontroldemo -selfcheck")
		os.Exit(2)
	}
	if err := run(os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "generationcontroldemo:", err)
		os.Exit(1)
	}
}

func run(out *os.File) error {
	rules := []streamrules.Rule{{
		Name: "no-shell-delete", Tool: "shell", Scope: streamrules.ScopeNamedTool, Pattern: `(?i)remove-item`, Interrupt: true,
		SubstituteAction: "Inspect the target with the read-only inventory tool.",
	}}
	c, err := generationctl.New("trajectory-demo-1", "planner-micro-agent", generationctl.Compute{Worker: "worker-cpu-1", Model: "fast-model", Device: "cpu"}, rules)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)
	e := c.Epoch()
	if err := enc.Encode(event{Event: "epoch_started", Epoch: &e}); err != nil {
		return err
	}
	if err := c.Accept("I will inspect the workspace. "); err != nil {
		return err
	}

	key := streamrules.StreamKey{ToolCallID: "call-1", ToolName: "shell", Scope: streamrules.ScopeNamedTool}
	if _, err := c.ObserveToolDelta(key, `{"command":"Remove-`); err != nil {
		return err
	}
	tr, err := c.ObserveToolDelta(key, `Item -Recurse C:\\work"}`)
	if err != nil {
		return err
	}
	if tr.Checkpoint == nil || tr.Directive.Kind != generationctl.Redirect {
		return fmt.Errorf("redirect was not captured")
	}
	if err := enc.Encode(event{Event: "steering_point", Directive: &tr.Directive, Checkpoint: tr.Checkpoint}); err != nil {
		return err
	}

	next, err := generationctl.Resume(*tr.Checkpoint, "safety-micro-agent", generationctl.Compute{Worker: "worker-gpu-7", Model: "deep-model", Device: "L4"}, rules)
	if err != nil {
		return err
	}
	e = next.Epoch()
	if err := enc.Encode(event{Event: "epoch_started", Epoch: &e}); err != nil {
		return err
	}
	if err := next.Accept("Inventory complete; no destructive action ran."); err != nil {
		return err
	}
	cp := next.Checkpoint()
	if err := enc.Encode(event{Event: "trajectory_checkpoint", Checkpoint: &cp}); err != nil {
		return err
	}

	if cp.TrajectoryID != "trajectory-demo-1" || cp.AfterEpoch != 2 || cp.Accepted != "I will inspect the workspace. Inventory complete; no destructive action ran." {
		return fmt.Errorf("SELF_CHECK_FAIL: continuity invariant")
	}
	fmt.Fprintln(out, "SELF_CHECK_PASS continuous_generation_redirect_handoff")
	return nil
}
