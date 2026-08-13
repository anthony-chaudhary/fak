package gateway

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestPrepareServedAnthropicRequestMigratesRetiredToolSearchContract(t *testing.T) {
	raw := []byte(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"READY"}],"tools":[{"type":"tool_search_tool_20250917","name":"tool_search_tool"},{"name":"Read","description":"read","input_schema":{"type":"object"}}]}`)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	inbound := httptest.NewRequest("POST", "/v1/messages", nil)
	inbound.Header.Set("anthropic-beta", "claude-code-20250219,tool-search-2025-09-17,fine-grained-tool-streaming-2025-05-14")

	prep := (&Server{}).prepareServedAnthropicRequest(context.Background(), inbound, req, "", servedSessionTurn{})
	if strings.Contains(prep.upstreamBeta, retiredToolSearchBeta) {
		t.Fatalf("retired beta survived: %q", prep.upstreamBeta)
	}
	for _, want := range []string{"claude-code-20250219", "fine-grained-tool-streaming-2025-05-14"} {
		if !strings.Contains(prep.upstreamBeta, want) {
			t.Fatalf("supported beta %q was dropped: %q", want, prep.upstreamBeta)
		}
	}
	body := string(req.Raw)
	if strings.Contains(body, "tool_search_tool_20250917") || strings.Contains(body, `"name":"tool_search_tool"`) {
		t.Fatalf("retired descriptor survived: %s", body)
	}
	if !strings.Contains(body, toolSearchToolType) || !strings.Contains(body, toolSearchToolName) {
		t.Fatalf("current descriptor missing: %s", body)
	}
}

func TestMigrateRetiredToolSearchPreservesCurrentAndCustomTools(t *testing.T) {
	raw := []byte(`{"tools":[{"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex"},{"name":"Read","description":"read","input_schema":{"type":"object"}}]}`)
	req := &agent.AnthropicMessagesRequest{Raw: raw}
	if migrateRetiredToolSearch(req) {
		t.Fatal("current contract must remain byte-identical")
	}
	if string(req.Raw) != string(raw) {
		t.Fatalf("non-retired tools changed: %s", req.Raw)
	}
}
