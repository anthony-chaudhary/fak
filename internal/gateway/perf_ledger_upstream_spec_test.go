package gateway

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=500ms lane=default
func TestProxiedCompletionPerfRowCarriesUpstreamDraftCounts(t *testing.T) {
	srv := newTestServer(t)
	drafted := &agent.Timings{PromptN: 600, PromptMS: 1500, PredictedN: 40, PredictedMS: 2000, DraftN: 48, DraftNAccepted: 60}
	srv.metrics.observeCompletionServed(localitySelfHosted, &agent.Completion{FinishReason: "stop", Usage: agent.Usage{PromptTokens: 600, CompletionTokens: 40}, Timings: drafted}, 4*time.Second)
	srv.metrics.observeCompletionServed(localitySelfHosted, &agent.Completion{FinishReason: "stop", Usage: agent.Usage{PromptTokens: 10, CompletionTokens: 2}}, time.Second)

	rep := srv.PerfReport(10)
	if len(rep.Records) != 2 {
		t.Fatalf("records = %d, want 2", len(rep.Records))
	}
	got, plain := rep.Records[0], rep.Records[1]
	if got.UpstreamSpecDraftTokens != 48 || got.UpstreamSpecAcceptedTokens != 48 || got.Engine != nil {
		t.Fatalf("drafted row = draft %d accepted %d engine %v, want 48/48 (accepted clamped) and no native engine", got.UpstreamSpecDraftTokens, got.UpstreamSpecAcceptedTokens, got.Engine)
	}
	if plain.UpstreamSpecDraftTokens != 0 || plain.UpstreamSpecAcceptedTokens != 0 {
		t.Fatalf("undrafted row carries upstream spec counts: %+v", plain)
	}
	if rep.Summary.SpecDraftTokens != 48 || rep.Summary.SpecAcceptRate != 1 || rep.Summary.Native != 0 {
		t.Fatalf("summary spec = %d rate %v native %d", rep.Summary.SpecDraftTokens, rep.Summary.SpecAcceptRate, rep.Summary.Native)
	}
}
