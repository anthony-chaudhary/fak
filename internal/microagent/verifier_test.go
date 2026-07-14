package microagent_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type verifierAgent struct {
	mu       sync.Mutex
	steps    int
	feedback []error
}

func (a *verifierAgent) Step(context.Context, microagent.Gateway) (bool, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steps++
	return true, nil
}

func (a *verifierAgent) RetryFeedback(_ context.Context, evidence error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.feedback = append(a.feedback, evidence)
	return nil
}

func TestVerifierDisabledPreservesBaseline(t *testing.T) {
	agent := &verifierAgent{}
	result := runVerifierHost(t, microagent.Config{Workers: 1}, agent)
	if !result.Done || result.Err != nil || agent.steps != 1 {
		t.Fatalf("baseline changed: result=%+v steps=%d", result, agent.steps)
	}
}

func TestVerifierFailureFeedsEvidenceIntoRetry(t *testing.T) {
	sentinel := errors.New("commit audit: claimed diff is absent")
	calls := 0
	var inputs []microagent.VerificationInput
	verifier := microagent.VerifierFunc(func(_ context.Context, in microagent.VerificationInput) error {
		inputs = append(inputs, in)
		calls++
		if calls == 1 {
			return sentinel
		}
		return nil
	})
	agent := &verifierAgent{}
	audit := &verifyAudit{}
	result := runVerifierHost(t, microagent.Config{Workers: 1, MaxRetries: 1, Verifier: verifier, Audit: audit}, agent)
	if !result.Done || result.Err != nil || agent.steps != 2 {
		t.Fatalf("retry did not recover: result=%+v steps=%d", result, agent.steps)
	}
	if len(agent.feedback) != 1 || !errors.Is(agent.feedback[0], sentinel) {
		t.Fatalf("retry feedback = %#v, want exact verification evidence", agent.feedback)
	}
	if len(inputs) != 2 || inputs[0].Agent != "verify-agent" || inputs[0].Steps != 1 {
		t.Fatalf("verification inputs = %+v", inputs)
	}
	if got := audit.kinds(); !equalKinds(got, []microagent.EventKind{microagent.EventSpawn, microagent.EventVerify, microagent.EventRetry, microagent.EventVerify, microagent.EventDone}) {
		t.Fatalf("audit transcript = %v", got)
	}
}

func TestVerifierFailureRefusesUnwitnessedCompletion(t *testing.T) {
	sentinel := errors.New("tests failed")
	agent := &verifierAgent{}
	result := runVerifierHost(t, microagent.Config{Workers: 1, Verifier: microagent.VerifierFunc(func(context.Context, microagent.VerificationInput) error { return sentinel })}, agent)
	if result.Done || !errors.Is(result.Err, sentinel) {
		t.Fatalf("unwitnessed completion accepted: %+v", result)
	}
	var verification *microagent.VerificationError
	if !errors.As(result.Err, &verification) {
		t.Fatalf("error %T is not VerificationError", result.Err)
	}
}

func runVerifierHost(t *testing.T, cfg microagent.Config, agent microagent.Microagent) microagent.Result {
	t.Helper()
	host, err := microagent.NewHost(stubVerifierGateway{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(host.Close)
	if err := host.Spawn("verify-agent", agent); err != nil {
		t.Fatal(err)
	}
	if err := host.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	results := host.Reap()
	if len(results) != 1 {
		t.Fatalf("results = %+v", results)
	}
	return results[0]
}

type stubVerifierGateway struct{}

func (stubVerifierGateway) Model() string { return "stub" }
func (stubVerifierGateway) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

type verifyAudit struct {
	mu     sync.Mutex
	events []microagent.Event
}

func (a *verifyAudit) Record(e microagent.Event) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.events = append(a.events, e)
}
func (a *verifyAudit) kinds() []microagent.EventKind {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]microagent.EventKind, len(a.events))
	for i, e := range a.events {
		out[i] = e.Kind
	}
	return out
}
func equalKinds(a, b []microagent.EventKind) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
