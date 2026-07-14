package microagent_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type retryFixture struct {
	mu         sync.Mutex
	failures   int
	steps      int
	feedback   []error
	transcript []string
}

func (a *retryFixture) Step(context.Context, microagent.Gateway) (bool, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steps++
	if a.steps <= a.failures {
		err := errors.New("tool stderr: transient connection reset")
		a.transcript = append(a.transcript, "step error: "+err.Error())
		return false, err
	}
	a.transcript = append(a.transcript, "step observed feedback and completed")
	return true, nil
}

func (a *retryFixture) RetryFeedback(_ context.Context, evidence error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.feedback = append(a.feedback, evidence)
	a.transcript = append(a.transcript, "retry evidence: "+evidence.Error())
	return nil
}

type retryBlindFixture struct{ steps int }

func (a *retryBlindFixture) Step(context.Context, microagent.Gateway) (bool, error) {
	a.steps++
	return false, errors.New("tool failed without a feedback seam")
}

type retryAudit struct {
	mu     sync.Mutex
	events []microagent.Event
}

func (s *retryAudit) Record(ev microagent.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
}

func runRetryFixture(t *testing.T, cfg microagent.Config, m microagent.Microagent) microagent.Result {
	t.Helper()
	h, err := microagent.NewHost(retryPlanner{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	if err := h.Spawn("retry-fixture", m); err != nil {
		t.Fatal(err)
	}
	if err := h.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	got := h.Reap()
	if len(got) != 1 {
		t.Fatalf("Reap len = %d, want 1", len(got))
	}
	return got[0]
}

type retryPlanner struct{}

func (retryPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, nil
}
func (retryPlanner) Model() string { return "retry-fixture" }

func TestGroundedRetryImprovesCompletionAndCarriesEvidence(t *testing.T) {
	without := &retryFixture{failures: 1}
	base := runRetryFixture(t, microagent.Config{Workers: 1}, without)
	if base.Done || base.Err == nil || without.steps != 1 || len(without.feedback) != 0 {
		t.Fatalf("off-by-default result = %+v, steps=%d feedback=%d", base, without.steps, len(without.feedback))
	}

	audit := &retryAudit{}
	with := &retryFixture{failures: 1}
	got := runRetryFixture(t, microagent.Config{Workers: 1, MaxRetries: 1, Audit: audit}, with)
	if !got.Done || got.Err != nil || got.Steps != 2 {
		t.Fatalf("grounded retry result = %+v, want done in 2 steps", got)
	}
	if len(with.feedback) != 1 || with.feedback[0].Error() != "tool stderr: transient connection reset" {
		t.Fatalf("feedback = %#v, want exact failed-tool evidence", with.feedback)
	}
	wantTranscript := []string{
		"step error: tool stderr: transient connection reset",
		"retry evidence: tool stderr: transient connection reset",
		"step observed feedback and completed",
	}
	if len(with.transcript) != len(wantTranscript) {
		t.Fatalf("transcript = %#v", with.transcript)
	}
	for i := range wantTranscript {
		if with.transcript[i] != wantTranscript[i] {
			t.Fatalf("transcript[%d] = %q, want %q", i, with.transcript[i], wantTranscript[i])
		}
	}
	audit.mu.Lock()
	defer audit.mu.Unlock()
	found := false
	for _, ev := range audit.events {
		if ev.Kind == microagent.EventRetry {
			found = ev.Err == "tool stderr: transient connection reset" && ev.Steps == 1
		}
	}
	if !found {
		t.Fatalf("audit events = %#v, want retry with exact evidence", audit.events)
	}
}

func TestGroundedRetryCeilingAndNoBlindFallback(t *testing.T) {
	bounded := &retryFixture{failures: 3}
	got := runRetryFixture(t, microagent.Config{Workers: 1, MaxRetries: 2}, bounded)
	if got.Done || got.Err == nil || got.Steps != 3 || bounded.steps != 3 || len(bounded.feedback) != 2 {
		t.Fatalf("bounded result = %+v, steps=%d feedback=%d; want hard ceiling 2", got, bounded.steps, len(bounded.feedback))
	}

	blind := &retryBlindFixture{}
	got = runRetryFixture(t, microagent.Config{Workers: 1, MaxRetries: 9}, blind)
	if got.Done || got.Err == nil || got.Steps != 1 || blind.steps != 1 {
		t.Fatalf("blind result = %+v, steps=%d; retry must require feedback seam", got, blind.steps)
	}
}
