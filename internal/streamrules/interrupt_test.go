package streamrules

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

func interruptMatch(action string) Match {
	return Match{Rule: "prefer-search", Key: StreamKey{ToolCallID: "call-a", Scope: ScopeAnyTool}, Interrupt: true, SubstituteAction: action}
}

func TestInterruptRequiresSubstituteAtRegistration(t *testing.T) {
	m, diagnostics := Compile([]Rule{{Name: "observe-only", Pattern: "rm", Scope: ScopeAnyTool, Interrupt: true}})
	if len(diagnostics) != 0 {
		t.Fatalf("diagnostics = %v", diagnostics)
	}
	got := m.CheckDelta(StreamKey{ToolCallID: "call-a", Scope: ScopeAnyTool}, "rm")
	if len(got) != 1 || got[0].Interrupt {
		t.Fatalf("match = %#v, want non-interrupting degradation", got)
	}
}

func TestInterruptLifecycleScopesCallDiscardsAndLeadsWithAction(t *testing.T) {
	r := NewRuntime(true)
	var events []string
	var injected string
	h := InterruptHooks{
		Abort:          func(id, reason string) { events = append(events, "abort:"+id+":"+reason) },
		DiscardPartial: func(id string) { events = append(events, "discard:"+id) },
		Inject:         func(message string) { injected = message; events = append(events, "inject") },
		Resume:         func(id string) error { events = append(events, "resume:"+id); return nil },
		Resolve:        func(id string) { events = append(events, "resolve:"+id) },
	}
	retry, ok := r.Begin(interruptMatch("Search the knowledge base first."), 7, "msg-1", h)
	if !ok {
		t.Fatal("Begin did not interrupt")
	}
	if strings.Join(events, ",") != "abort:call-a:substitute-action:prefer-search,discard:call-a,inject" {
		t.Fatalf("events = %v", events)
	}
	if !strings.HasPrefix(injected, "Search the knowledge base first.") {
		t.Fatalf("injected = %q", injected)
	}
	if findings := negframe.Classify("injected.txt", injected); len(findings) != 0 {
		t.Fatalf("negframe findings = %#v", findings)
	}
	if !r.Attempt(retry, 7, "msg-1", h) {
		t.Fatal("resume failed")
	}
	if strings.Contains(strings.Join(events, ","), "call-b") {
		t.Fatalf("sibling was affected: %v", events)
	}
}

func TestRuntimeDefaultOffIsNoOp(t *testing.T) {
	var called bool
	_, ok := NewRuntime(false).Begin(interruptMatch("Search first."), 1, "msg", InterruptHooks{Abort: func(string, string) { called = true }})
	if ok || called {
		t.Fatal("default-off runtime changed behavior")
	}
}

func TestStaleRetriesClearAndResolve(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Runtime, *Retry, *uint64, *string)
	}{
		{"retry-token", func(_ *Runtime, x *Retry, _ *uint64, _ *string) { x.token++ }},
		{"prompt-generation", func(_ *Runtime, _ *Retry, g *uint64, _ *string) { *g++ }},
		{"abort-pending", func(r *Runtime, x *Retry, _ *uint64, _ *string) { r.Cancel(x.toolCallID) }},
		{"target-message", func(_ *Runtime, _ *Retry, _ *uint64, m *string) { *m = "msg-2" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewRuntime(true)
			retry, _ := r.Begin(interruptMatch("Search first."), 1, "msg-1", InterruptHooks{})
			gen := uint64(1)
			msg := "msg-1"
			tt.mutate(r, &retry, &gen, &msg)
			resolved := 0
			resumed := 0
			if r.Attempt(retry, gen, msg, InterruptHooks{Resume: func(string) error { resumed++; return nil }, Resolve: func(string) { resolved++ }}) {
				t.Fatal("stale retry fired")
			}
			if resumed != 0 || resolved != 1 || len(r.pending) != 0 {
				t.Fatalf("resumed=%d resolved=%d pending=%d", resumed, resolved, len(r.pending))
			}
		})
	}
}

func TestFailedResumeStillResolves(t *testing.T) {
	r := NewRuntime(true)
	retry, _ := r.Begin(interruptMatch("Search first."), 1, "msg", InterruptHooks{})
	resolved := 0
	if r.Attempt(retry, 1, "msg", InterruptHooks{Resume: func(string) error { return errors.New("down") }, Resolve: func(string) { resolved++ }}) {
		t.Fatal("failed resume reported success")
	}
	if resolved != 1 {
		t.Fatalf("resolved=%d", resolved)
	}
}

func TestScheduleDefersRetry(t *testing.T) {
	r := NewRuntime(true)
	retry, _ := r.Begin(interruptMatch("Search first."), 3, "msg", InterruptHooks{})
	resumed := make(chan struct{}, 1)
	start := time.Now()
	r.Schedule(retry, func() uint64 { return 3 }, func() string { return "msg" }, InterruptHooks{
		Resume: func(string) error { resumed <- struct{}{}; return nil },
	})
	select {
	case <-resumed:
		if elapsed := time.Since(start); elapsed < RetryDelay/2 {
			t.Fatalf("retry was not deferred: %v", elapsed)
		}
	case <-time.After(time.Second):
		t.Fatal("scheduled retry did not fire")
	}
}
