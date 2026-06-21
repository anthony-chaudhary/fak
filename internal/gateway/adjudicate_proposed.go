package gateway

import (
	"context"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func (s *Server) adjudicateProposed(ctx context.Context, calls []agent.ToolCall, reqTrace string) ([]agent.ToolCall, []ToolAdjudication, int) {
	kept := make([]agent.ToolCall, 0, len(calls))
	adjs := make([]ToolAdjudication, 0, len(calls))
	dropped := 0
	for _, tc := range calls {
		tool := tc.Function.Name
		wv, repaired, aerr := s.adjudicate(ctx, tool, tc.Function.Arguments, false, "", reqTrace)
		if aerr != nil {
			dropped++
			adjs = append(adjs, ToolAdjudication{ToolCallID: tc.ID, Tool: tool, Admitted: false,
				Verdict: WireVerdict{Kind: "DENY", Reason: "MALFORMED", Disposition: "RETRYABLE"}})
			continue
		}
		adj := ToolAdjudication{ToolCallID: tc.ID, Tool: tool, Verdict: wv}
		switch wv.Kind {
		case "ALLOW":
			adj.Admitted = true
			kept = append(kept, tc)
		case "TRANSFORM":
			adj.Admitted = true
			if repaired != "" {
				tc.Function.Arguments = repaired
				adj.RepairedArguments = json.RawMessage(repaired)
			}
			kept = append(kept, tc)
		default:
			dropped++
		}
		adjs = append(adjs, adj)
	}
	return kept, adjs, dropped
}
