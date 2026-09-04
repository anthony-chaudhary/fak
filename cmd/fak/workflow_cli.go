package main

import (
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func runWorkflowCLI(workflowID string, step bool, checkpointDir string) error {
	_, err := runWorkflowCore(os.Stdout, os.Stderr, workflowID, step, checkpointDir)
	return err
}

func runWorkflowCore(out io.Writer, errOut io.Writer, workflowID string, step bool, checkpointDir string) (*agent.WorkflowState, error) {
	engine := agent.DefaultWorkflowEngine()
	if checkpointDir != "" {
		engine.CheckpointDir = checkpointDir
	}

	var state *agent.WorkflowState
	if step {
		if latest, err := engine.LoadLatestCheckpoint(workflowID); err == nil && latest != nil {
			state = latest
		}
	}
	if state == nil {
		state = &agent.WorkflowState{
			WorkflowID: workflowID,
			Data:       make(map[string]any),
		}
	}

	var resState *agent.WorkflowState
	var err error
	if step {
		resState, err = engine.Step(ctx(), workflowID, state)
	} else {
		resState, err = engine.Execute(ctx(), workflowID, state)
	}

	if resState != nil {
		printWorkflowReceipts(out, resState)
	}

	if err != nil {
		fmt.Fprintf(errOut, "workflow %s execution error: %v\n", workflowID, err)
		return resState, err
	}

	if resState != nil && (resState.Status == agent.WorkflowRefused || resState.Status == agent.WorkflowHalted) {
		if resState.LastRefusal != nil {
			fmt.Fprintf(errOut, "workflow %s %s: [%s] %s\n",
				workflowID, resState.Status, resState.LastRefusal.Kind, resState.LastRefusal.Reason)
			if resState.LastRefusal.RefusalCode != "" {
				fmt.Fprintf(errOut, "refusal code: %s\n", resState.LastRefusal.RefusalCode)
			}
		} else {
			fmt.Fprintf(errOut, "workflow %s ended with status %s\n", workflowID, resState.Status)
		}
		return resState, fmt.Errorf("workflow %s ended with status %s", workflowID, resState.Status)
	}

	return resState, nil
}

func printWorkflowReceipts(out io.Writer, state *agent.WorkflowState) {
	for _, r := range state.Receipts {
		fmt.Fprintf(out, "[workflow %s] phase %d (%s) -> phase %d (%s): %s (%s)\n",
			r.WorkflowID, r.FromIndex, r.FromPhase, r.ToIndex, r.ToPhase, r.Verdict.Kind, r.Verdict.Reason)
		if r.Witness.Command != "" || r.Witness.Artifact != "" || r.Witness.Description != "" {
			fmt.Fprintf(out, "    witness: cmd=%q artifact=%q desc=%q passed=%v\n",
				r.Witness.Command, r.Witness.Artifact, r.Witness.Description, r.Witness.Passed)
		}
	}
}
