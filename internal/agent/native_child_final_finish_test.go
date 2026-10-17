package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

type nativeChildFinalPlanner struct {
	finish string
	calls  atomic.Int32
}

func (p *nativeChildFinalPlanner) Model() string { return "native-child-final-fixture" }

func (p *nativeChildFinalPlanner) Complete(context.Context, []Message, []ToolDef, ...SampleOpt) (*Completion, error) {
	p.calls.Add(1)
	return &Completion{
		Message:      Message{Role: RoleAssistant, Content: "partial-child-output-private-marker"},
		FinishReason: p.finish,
		Usage:        Usage{PromptTokens: 11, CompletionTokens: 7},
	}, nil
}

// fak-test:runtime fast est=1s
func TestNativeChildFinalFinishTaskReceipt(t *testing.T) {
	const partial = "partial-child-output-private-marker"
	const truncated = "native child task: final answer truncated by the output-token limit"
	for _, tc := range []struct {
		name, finish       string
		wrapped, unrelated bool
	}{
		{name: "length", finish: "length"},
		{name: "wrapped length", finish: "length", wrapped: true},
		{name: "stop", finish: "stop"},
		{name: "omitted"},
		{name: "same text unrelated error", unrelated: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			planner := &nativeChildFinalPlanner{finish: tc.finish}
			runner := NewNativeChildTaskRunner(planner, 3, []RunOption{WithInfraRepromptBudget(2)}, nil,
				adjudicator.Policy{Posture: adjudicator.PostureFailClosed})
			if tc.wrapped {
				inner := runner
				runner = func(ctx context.Context, req ChildTaskRunRequest) (any, error) {
					result, err := inner(ctx, req)
					if err != nil {
						err = fmt.Errorf("child wrapper: %w", err)
					}
					return result, err
				}
			}
			if tc.unrelated {
				runner = func(context.Context, ChildTaskRunRequest) (any, error) {
					return partial, errors.New(truncated)
				}
			}
			state := NewTaskStateWithRunner(runner)
			defer state.Close()
			spawn, err := state.Spawn(TaskSpawnRequest{Prompt: "Report the measured results.", ReadOnly: true})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			receipt, err := state.Wait(ctx, TaskWaitRequest{TaskID: spawn.TaskID, TimeoutMs: 1000})
			if err != nil {
				t.Fatal(err)
			}
			// Inspect the actual task_wait payload, not just a runner-local result.
			body, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			var wire TaskWaitReceipt
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatal(err)
			}
			item := wire.Tasks[spawn.TaskID]
			if item == nil || wire.Status != "completed" || wire.Running != 0 {
				t.Fatalf("task did not reach a terminal receipt: %s", body)
			}
			failed := tc.finish == "length" || tc.unrelated
			if failed {
				if item.State != TaskStateFailed || wire.Failed != 1 || wire.Completed != 0 || !strings.Contains(item.Error, truncated) {
					t.Fatalf("incomplete child counted as successful: %s", body)
				}
			} else if item.State != TaskStateCompleted || wire.Completed != 1 || wire.Failed != 0 || item.Error != "" {
				t.Fatalf("clean child changed: %s", body)
			}
			if tc.unrelated {
				if item.Result != nil || planner.calls.Load() != 0 {
					t.Fatalf("unrelated failure retained a partial result: %s", body)
				}
			} else if item.Result != partial || planner.calls.Load() != 1 {
				t.Fatalf("partial result lost or child retried: calls=%d receipt=%s", planner.calls.Load(), body)
			}
			if strings.Contains(item.Error, partial) {
				t.Fatalf("error duplicated model output: %q", item.Error)
			}
		})
	}
}
