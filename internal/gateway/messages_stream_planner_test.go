// fak-test:runtime fast est=1s lane=default
package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestMessagesPlannerSampleOptsWireReplayTemperature(t *testing.T) {
	s := &Server{}
	omitted := s.plannerSampleOpts(&agent.AnthropicMessagesRequest{
		Model:     "test-model",
		MaxTokens: 128,
	}, servedSessionTurn{})

	var sp agent.SampleParams
	for _, opt := range omitted {
		if opt != nil {
			opt(&sp)
		}
	}
	if sp.Temperature != nil || !sp.InheritUpstreamTemperature {
		t.Fatalf("omitted temperature: got Temperature=%v InheritUpstreamTemperature=%v, want nil and true", sp.Temperature, sp.InheritUpstreamTemperature)
	}

	explicit := s.plannerSampleOpts(&agent.AnthropicMessagesRequest{
		Model:       "test-model",
		MaxTokens:   128,
		Temperature: 0.7,
	}, servedSessionTurn{})
	sp = agent.SampleParams{}
	for _, opt := range explicit {
		if opt != nil {
			opt(&sp)
		}
	}
	if sp.Temperature == nil || *sp.Temperature != 0.7 || sp.InheritUpstreamTemperature {
		t.Fatalf("explicit temperature: got Temperature=%v InheritUpstreamTemperature=%v, want 0.7 and false", sp.Temperature, sp.InheritUpstreamTemperature)
	}
}
