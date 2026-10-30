package safesync

import (
	"context"
	"strings"
	"testing"
)

func TestSafePushNoFetchOnFailure(t *testing.T) {
	for _, tc := range []struct {
		name, message, reason string
		code                  int
	}{
		{"success", "", "", 0},
		{"hook", "remote rejected: mandatory concept guard refused", PushReasonError, 1},
		{"auth", "remote: Permission to repo denied", PushReasonError, 1},
		{"nonff", "! [rejected] main -> main (non-fast-forward)", PushReasonFetchDeferred, 1},
		{"network", "fatal: unable to access remote: Could not resolve host", PushReasonUnreachable, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			run := func(_ context.Context, _ string, args ...string) RunResult {
				calls++
				if strings.Join(args, " ") != "push origin abc:refs/heads/main" {
					t.Fatalf("unexpected native action: %v", args)
				}
				return RunResult{Code: tc.code, Stderr: []byte(tc.message)}
			}
			r, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", SourceRef: "abc", NoFetchOnFailure: true, MaxRetries: 3, Runner: run})
			if err != nil || calls != 1 || r.Attempts != 1 || r.Reason != tc.reason || r.Pushed != (tc.code == 0) {
				t.Fatalf("calls=%d result=%+v err=%v", calls, r, err)
			}
			if tc.code != 0 && (r.Diagnostics == nil || !strings.Contains(r.Diagnostics.Stderr, tc.message) || r.Velocity.Qualified) {
				t.Fatalf("lost refusal or false delivery: %+v", r)
			}
		})
	}
}
