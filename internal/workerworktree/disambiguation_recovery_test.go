package workerworktree

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms
func TestLandDisambiguationDeadlineRecoveryPreservesVerdict(t *testing.T) {
	// Decode the wire fields so this same witness compiles on the parent and
	// fails on the missing diagnosis, rather than on a new Go field or method.
	recovery := func(d *DisambiguationDiagnostic) (bool, string) {
		t.Helper()
		var wire struct {
			Recoverable bool   `json:"recoverable"`
			Recovery    string `json:"recovery"`
		}
		data, err := json.Marshal(d)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &wire); err != nil {
			t.Fatal(err)
		}
		return wire.Recoverable, wire.Recovery
	}
	oldRead, oldDeadline, oldHas := readDisambiguation, newDeadline, hasDisambiguationContract
	t.Cleanup(func() {
		readDisambiguation, newDeadline, hasDisambiguationContract = oldRead, oldDeadline, oldHas
	})
	hasDisambiguationContract = func(context.Context, string, string) (bool, error) { return true, nil }

	for _, tc := range []struct {
		name      string
		timeoutMS string
		expire    bool
		red       bool
	}{
		{name: "stalled subphase", expire: true},
		{name: "bounded recovery preserves green", timeoutMS: "300000"},
		{name: "bounded recovery preserves red", timeoutMS: "300000", red: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(DisambiguationTimeoutEnv, tc.timeoutMS)
			if tc.timeoutMS == "" {
				if err := os.Unsetenv(DisambiguationTimeoutEnv); err != nil {
					t.Fatal(err)
				}
			}
			manual := newManualDeadlineContext()
			ctx, cancel := context.WithCancel(manual)
			var requested time.Duration
			newDeadline = func(timeout time.Duration) (context.Context, context.CancelFunc) {
				requested = timeout
				return ctx, func() {}
			}
			started, release, exited := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var calls []string
			readDisambiguation = func(ctx context.Context, repo, tree string, phase func(string)) DisambiguationWitness {
				calls = append(calls, repo+"@"+tree)
				if len(calls) == 1 {
					phase("scorecard-command")
					close(started)
					select {
					case <-ctx.Done():
					case <-release:
					}
					close(exited)
				}
				return DisambiguationWitness{
					Tree: tree, Fresh: true, SemanticValid: !tc.red || tree != "treeSHA111",
					CriticalClean: true, Coverage: 100,
				}
			}
			g := isolatedHappyFake()
			msg := writeMsg(t, "fix(workerland): explain deadline recovery (fak workerworktree)")
			type outcome struct {
				result  Result
				handled bool
			}
			done := make(chan outcome, 1)
			finished := make(chan struct{})
			// Join before another subtest can replace the package seams, even if
			// an assertion fires before either worker reaches its normal return.
			// An unexpected stuck fake remains bounded by go test's timeout.
			t.Cleanup(func() {
				cancel()
				<-finished
				if requested != 0 {
					<-exited
				}
			})
			go func() {
				defer close(finished)
				result, handled := landIsolated("/repo", "/worker",
					"diff --git a/internal/workerworktree/disambiguation.go b/internal/workerworktree/disambiguation.go\n@@\n-old\n+new\n",
					msg, []string{"internal/workerworktree/disambiguation.go"}, g.run, g.runEnv)
				done <- outcome{result, handled}
			}()
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("land did not reach the slow subphase")
			}
			if tc.expire {
				manual.expire()
			} else {
				close(release)
			}
			var got outcome
			select {
			case got = <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("land did not return after releasing the slow subphase")
			}
			select {
			case <-exited:
			case <-time.After(time.Second):
				t.Fatal("slow reader did not observe cancellation or release")
			}
			if !got.handled || got.result.Disambiguation == nil {
				t.Fatalf("missing handled oracle result: %+v", got)
			}
			if tc.expire {
				d := got.result.Disambiguation.Before.Diagnostic
				recoverable, instruction := recovery(d)
				if got.result.Code != DisambiguationTimeoutCode || got.result.OK || !got.result.Preserved ||
					d == nil || !recoverable || d.Subphase != "scorecard-command" || d.TimeoutMS != 120000 {
					t.Fatalf("deadline did not carry a recoverable typed refusal: %+v diagnostic=%+v", got.result, d)
				}
				for _, want := range []string{DisambiguationTimeoutEnv, "--disambiguation-timeout-ms", "1..900000", "all three witnesses"} {
					if !strings.Contains(instruction, want) || !strings.Contains(got.result.Detail, want) {
						t.Fatalf("deadline recovery missing %q: %+v", want, got.result)
					}
				}
				if strings.Contains(got.result.Reason, "invariant failed") || LandRefusalRetryable(got.result.Reason) {
					t.Fatalf("deadline was treated as a verdict or automatic retry: %+v", got.result)
				}
				if len(calls) != 1 || requested != 2*time.Minute || len(g.envCallsWithPrefix("commit-tree")) != 0 || len(g.callsWithPrefix("update-ref")) != 0 {
					t.Fatalf("timeout crossed the pre-CAS boundary: calls=%v requested=%v result=%+v", calls, requested, got.result)
				}
				return
			}
			wantCalls := []string{"/repo@HEAD", "/worker@HEAD", "/repo@treeSHA111"}
			if !reflect.DeepEqual(calls, wantCalls) || requested != 5*time.Minute || got.result.OK == tc.red {
				t.Fatalf("bounded recovery changed witnesses or verdict: calls=%v requested=%v result=%+v", calls, requested, got.result)
			}
			if got.result.Code == DisambiguationTimeoutCode {
				t.Fatalf("completed oracle was mislabeled a recoverable deadline: %+v", got.result)
			}
			if tc.red && (got.result.Committed || got.result.Applied || len(g.callsWithPrefix("update-ref")) != 0) {
				t.Fatalf("bounded recovery accepted a red oracle: %+v", got.result)
			}
		})
	}
	if t.Failed() {
		return
	}
	for _, err := range []error{context.Canceled, errors.New("analyzer failed")} {
		d := timeoutResult("tree", "before", "scorecard-command", time.Minute, err).Diagnostic
		if recoverable, instruction := recovery(d); recoverable || instruction != "" {
			t.Fatalf("non-deadline %v must not suggest deadline recovery: %+v", err, d)
		}
	}
	d := timeoutResult("tree", "before", "scorecard-command", time.Minute, fmt.Errorf("analyzer: %w", context.DeadlineExceeded)).Diagnostic
	if recoverable, _ := recovery(d); !recoverable {
		t.Fatalf("wrapped deadline lost recoverable classification: %+v", d)
	}
}
