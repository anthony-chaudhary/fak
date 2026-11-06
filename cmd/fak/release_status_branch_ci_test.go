package main

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms
func TestReleaseStatusBranchCIRequiresPushEvent(t *testing.T) {
	const unqualified = `[{"event":"workflow_dispatch","conclusion":"success"},{"conclusion":"success"},{"event":null,"conclusion":"success"},{"event":true,"conclusion":"success"},{"event":"pull_request","conclusion":"success"},{"event":"schedule","conclusion":"success"},{"event":"unknown","conclusion":"success"},null]`
	for _, tc := range []struct {
		name, head, branch, want string
		wantCalls                int
		readError                bool
	}{
		{name: "reject both queries", head: unqualified, branch: unqualified, want: "unknown", wantCalls: 2},
		{name: "empty responses", head: `[]`, branch: `[]`, want: "unknown", wantCalls: 2},
		{name: "push head success", head: `[{"event":"push","status":"completed","conclusion":"success"}]`, want: "success", wantCalls: 1},
		{name: "push head failure", head: `[{"event":"push","status":"completed","conclusion":"failure"}]`, want: "failure", wantCalls: 1},
		{name: "push head pending", head: `[{"event":"push","status":"in_progress","conclusion":""}]`, want: "in_progress", wantCalls: 1},
		{name: "push branch fallback", head: unqualified, branch: `[{"event":"push","status":"completed","conclusion":"success"}]`, want: "success", wantCalls: 2},
		{name: "push branch failure", head: unqualified, branch: `[{"event":"push","status":"completed","conclusion":"failure"}]`, want: "failure", wantCalls: 2},
		{name: "manual cannot hide push failure", head: `[{"event":"workflow_dispatch","conclusion":"success"},{"event":"push","status":"completed","conclusion":"failure"}]`, want: "failure", wantCalls: 1},
		{name: "wrapped response", head: `{"value":[{"event":"workflow_dispatch","conclusion":"success"},{"event":"push","status":"completed","conclusion":"failure"}]}`, want: "failure", wantCalls: 1},
		{name: "read errors", want: "unknown", wantCalls: 2, readError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			oldJSON := releaseStatusRunGHJSON
			t.Cleanup(func() { releaseStatusRunGHJSON = oldJSON })
			root := t.TempDir()
			calls := 0
			releaseStatusRunGHJSON = func(gotRoot string, timeout time.Duration, name string, args ...string) (any, error) {
				wantArgs := []string{"run", "list", "--workflow", "ci.yml", "--commit", "source-sha", "--event", "push", "--limit", "1", "--json", "status,conclusion,headSha,url,event"}
				payload := tc.head
				if calls == 1 {
					wantArgs = []string{"run", "list", "--workflow", "ci.yml", "--branch", "dev", "--status", "completed", "--event", "push", "--limit", "1", "--json", "status,conclusion,headSha,url,event"}
					payload = tc.branch
				}
				calls++
				if calls > tc.wantCalls || gotRoot != root || timeout != 30*time.Second || name != "gh" || !reflect.DeepEqual(args, wantArgs) {
					t.Fatalf("unexpected CI query %d: root=%s timeout=%s %s %v", calls, gotRoot, timeout, name, args)
				}
				if tc.readError {
					return nil, fmt.Errorf("gh unavailable or non-JSON output")
				}
				var result any
				if err := json.Unmarshal([]byte(payload), &result); err != nil {
					t.Fatal(err)
				}
				return result, nil
			}
			if got := releaseStatusBranchCI(root, "dev", "source-sha"); got != tc.want || calls != tc.wantCalls {
				t.Fatalf("branch CI=%q calls=%d, want %q calls=%d", got, calls, tc.want, tc.wantCalls)
			}
		})
	}
}
