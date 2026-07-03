package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestNewProxyPlannerPropagatesForceResponsesStream(t *testing.T) {
	planner, err := newProxyPlanner(Config{
		Provider:             string(agent.ProviderOpenAIResponses),
		APIKey:               "test-key",
		ForceResponsesStream: true,
	}, "gpt-test", []string{"https://example.invalid"})
	if err != nil {
		t.Fatalf("newProxyPlanner: %v", err)
	}
	hp, ok := planner.(*agent.HTTPPlanner)
	if !ok {
		t.Fatalf("planner = %T, want *agent.HTTPPlanner", planner)
	}
	if !hp.ForceResponsesStream {
		t.Fatal("ForceResponsesStream was not propagated to HTTPPlanner")
	}
}
