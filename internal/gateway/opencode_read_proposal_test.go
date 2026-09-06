package gateway

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func TestOpenCodeReadProposalPreservesClientArguments(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	monitor := adjudicator.New(adjudicator.DevAgentPolicy())
	abi.RegisterAdjudicator(100, monitor)
	srv, err := New(Config{EngineID: "mock", Model: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	calls := []agent.ToolCall{{ID: "read-1", Type: "function", ThoughtSignature: "opaque-signed-context", Function: agent.Func{Name: "read", Arguments: `{"filePath":"witness.txt","offset":1,"limit":4}`}}}
	kept, adjs, dropped := srv.adjudicateProposed(context.Background(), calls, "opencode-read")
	if dropped != 0 || len(kept) != 1 || !adjs[0].Admitted {
		t.Fatalf("proposal lost: kept=%v adjs=%v dropped=%d", kept, adjs, dropped)
	}
	if kept[0].ThoughtSignature != "opaque-signed-context" {
		t.Fatal("adjudication lost native provider signature")
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(kept[0].Function.Arguments), &got); err != nil {
		t.Fatal(err)
	}
	if kept[0].Function.Name != "read" || got["filePath"] != "witness.txt" || got["offset"] != float64(1) || got["limit"] != float64(4) || got["file_path"] != nil {
		t.Fatalf("client schema changed: %v", kept)
	}
	// A policy denial remains a denial; restoring client spelling grants nothing.
	monitor.SetPolicy(adjudicator.Policy{Deny: map[string]abi.ReasonCode{"read": abi.ReasonPolicyBlock}})
	kept, _, dropped = srv.adjudicateProposed(context.Background(), calls, "opencode-read-deny")
	if dropped != 1 || len(kept) != 0 {
		t.Fatalf("denied read survived: %v", kept)
	}
}
