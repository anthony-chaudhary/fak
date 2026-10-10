package taskrun

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

// fak-test:runtime medium est=10s lane=default
func TestTaskChildReceiptFileAuthorityFailure(t *testing.T) {
	requireTaskSandbox(t)
	for _, blocked := range []bool{false, true} {
		name := "writable"
		if blocked {
			name = "directory-at-child-path"
		}
		t.Run(name, func(t *testing.T) {
			server, requests := newTaskPlannerServer(t, func(int) taskPlannerReply {
				return taskPlannerReply{content: "No changes made."}
			})
			defer server.Close()
			opts := taskOptions(t, server.URL, name)
			opts.TaskLimit, opts.Concurrency = 1, 1
			fixture := taskfixture.Cases(false)[0]
			taskDir := filepath.Join(opts.OutDir, safeName(fixture.ID))
			childPath := filepath.Join(taskDir, "child.json")
			if blocked {
				// A directory produces a deterministic write failure even when
				// the test process can bypass ordinary file permission bits.
				if err := os.MkdirAll(childPath, 0700); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			receipt, err := Run(ctx, opts)
			if err != nil || len(receipt.Tasks) != 1 {
				t.Fatalf("Run: tasks=%d err=%v", len(receipt.Tasks), err)
			}
			got := receipt.Tasks[0]
			if requests.Load() != 1 || got.Model.SuccessfulRequests != 1 || got.Model.ModelIdentityStatus != "matched" {
				t.Fatalf("child did not reach observed model execution: requests=%d task=%+v", requests.Load(), got)
			}
			if got.Accepted || got.Passed || got.Error == "" {
				t.Fatalf("non-editing child was accepted: %+v", got)
			}
			if blocked {
				if !strings.HasPrefix(got.Error, "decode child receipt: ") || got.PlannerCalls != 0 {
					t.Fatalf("receipt file failure was not surfaced before consuming child evidence: %+v", got)
				}
				if info, err := os.Stat(childPath); err != nil || !info.IsDir() {
					t.Fatalf("blocked child path changed: info=%v err=%v", info, err)
				}
			} else {
				body, err := os.ReadFile(childPath)
				if err != nil {
					t.Fatal(err)
				}
				var child ChildReceipt
				// The current child writes JSON to ReceiptPath; stderr is ancillary.
				if err := json.Unmarshal(body, &child); err != nil || child.PlannerCalls != 1 || child.Error == "" || got.PlannerCalls != 1 {
					t.Fatalf("failed execution receipt was not retained: child=%+v task=%+v err=%v", child, got, err)
				}
				if !strings.HasPrefix(got.Error, "task child exit ") {
					t.Fatalf("persisted failed child was not rejected by exit status: %s", got.Error)
				}
			}
			body, err := os.ReadFile(filepath.Join(taskDir, "task.json"))
			if err != nil {
				t.Fatal(err)
			}
			var persisted TaskReceipt
			if err := json.Unmarshal(body, &persisted); err != nil || persisted.Error != got.Error || persisted.Accepted || persisted.Passed {
				t.Fatalf("task failure receipt differs: persisted=%+v err=%v", persisted, err)
			}
		})
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime fast est=20ms lane=default
func TestReadChildReceiptRejectsMissingDirectoryAndPartial(t *testing.T) {
	for _, mode := range []string{"missing", "directory", "partial"} {
		t.Run(mode, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "child.json")
			switch mode {
			case "directory":
				if err := os.Mkdir(path, 0700); err != nil {
					t.Fatal(err)
				}
			case "partial":
				if err := os.WriteFile(path, []byte(`{"schema":"fak.agentbench.task-child.v2","planner_calls":1`), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := readChildReceipt(path); err == nil {
				t.Fatalf("%s receipt accepted", mode)
			}
		})
	}
}
