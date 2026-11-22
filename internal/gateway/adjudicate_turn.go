package gateway

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// AdjudicateAssistantTurn runs every tool call a buffered assistant turn proposes
// through the served chat floor and rewrites asst in place: only admitted calls
// survive, and refusal or served-inline text is folded into Content. It returns
// the OpenAI finish_reason and one adjudication per proposed call. tools is the
// client's offered set; nil keeps whatever set ctx already carries.
func (s *Server) AdjudicateAssistantTurn(ctx context.Context, asst *agent.Message, tools []agent.ToolDef, upstreamFinish, reqTrace string) (string, []ToolAdjudication) {
	ctx = withClientToolSchemas(ctx, tools)
	kept, adjs, dropped, servedText, servedHits, bodyRefused := s.adjudicateProposedTurn(ctx, *asst, reqTrace)
	return s.applyAdjudicatedTurn(asst, adjs, kept, dropped, servedHits, servedText, bodyRefused, upstreamFinish), adjs
}

// StreamedToolAdjudication is the adjudicated tail of a streamed turn whose prose
// already reached the client. Only Calls may follow that prose on the wire.
type StreamedToolAdjudication struct {
	Calls         []agent.ToolCall
	Adjudications []ToolAdjudication
	Finish        string
	servedText    string
	dropped       int
}

// AdjudicateStreamedToolCalls is the streamed twin of AdjudicateAssistantTurn. The
// prose is already on the wire, so nothing is blanked; Trailer carries the text a
// streamed turn appends instead.
func (s *Server) AdjudicateStreamedToolCalls(ctx context.Context, calls []agent.ToolCall, tools []agent.ToolDef, upstreamFinish, reqTrace string) StreamedToolAdjudication {
	ctx = withClientToolSchemas(ctx, tools)
	kept, adjs, dropped, servedText, servedHits := s.adjudicateProposedServed(ctx, calls, reqTrace)
	if servedHits > 0 {
		s.metrics.recordServedInline(servedHits)
	}
	finish := upstreamFinish
	switch {
	case len(kept) > 0:
		finish = "tool_calls"
	case finish == "" || finish == "tool_calls":
		// No surviving call: none was proposed, every one was dropped or served
		// inline, or the upstream omitted a finish_reason. All are a normal stop.
		finish = "stop"
	}
	return StreamedToolAdjudication{Calls: kept, Adjudications: adjs, Finish: finish, servedText: servedText, dropped: dropped}
}

// Trailer returns, in wire order, the content deltas that follow the streamed
// prose: served-inline answers, then the livelock or all-refused note. proseEmpty
// reports that no model prose reached the client, which selects the short summary.
func (a StreamedToolAdjudication) Trailer(proseEmpty bool) []string {
	var out []string
	if a.servedText != "" {
		out = append(out, "\n"+a.servedText)
	}
	if anyLivelock(a.Adjudications) {
		if note := adjudicationNote(a.Adjudications); note != "" {
			return append(out, note)
		}
	}
	if len(a.Calls) == 0 && a.dropped > 0 {
		if proseEmpty {
			return append(out, denySummary(a.Adjudications))
		}
		return append(out, adjudicationNote(a.Adjudications))
	}
	return out
}
