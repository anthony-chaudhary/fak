package gateway

import (
	"context"
	"encoding/json"
)

// fakReadBatch is the additive batch form of fak_read. Each path crosses fakReadWithOptions on its
// own, so every item gets an independent buildCall + k.Syscall adjudication and its own
// cache hit/miss decision. Results retain request order; an item-level fault is captured
// on that row instead of aborting the remaining independent reads.
func (s *Server) fakReadBatch(ctx context.Context, paths []string, traceID, witness string) FakReadBatchResponse {
	return s.fakReadBatchWithOptions(ctx, paths, 0, 0, false, traceID, witness)
}

func (s *Server) fakReadBatchWithOptions(ctx context.Context, paths []string, offset, limit int, lineNumbers bool, traceID, witness string) FakReadBatchResponse {
	resp := FakReadBatchResponse{
		TraceID:   traceID,
		ItemCount: len(paths),
		Results:   make([]FakReadBatchItem, 0, len(paths)),
	}
	for _, path := range paths {
		wv, env, err := s.fakReadWithOptions(ctx, path, offset, limit, lineNumbers, traceID, witness)
		item := FakReadBatchItem{FilePath: path, Verdict: wv, Result: env}
		if err != nil {
			item.Error = err.Error()
		} else if env != nil && env.Status == "ERROR" {
			item.Error = resultEnvelopeError(env)
		}
		resp.Results = append(resp.Results, item)
	}
	return resp
}

// fakAdmitBatch gives the other per-item MCP verb the same additive batch axis. Top-level
// trace/witness values are defaults; an item may override either. As with fak_read, one
// admission fault is data on that row and never silently drops the remaining items.
func (s *Server) fakAdmitBatch(ctx context.Context, items []AdmitRequest, traceID, witness string) FakAdmitBatchResponse {
	resp := FakAdmitBatchResponse{
		TraceID:   traceID,
		ItemCount: len(items),
		Results:   make([]FakAdmitBatchItem, 0, len(items)),
	}
	for _, req := range items {
		if req.TraceID == "" {
			req.TraceID = traceID
		} else {
			req.TraceID = s.traceFor(req.TraceID)
		}
		if req.Witness == "" {
			req.Witness = witness
		}
		wv, env, err := s.admit(ctx, req.Tool, rawArgs(req.Result), req.Witness, req.TraceID)
		env = s.enrichPagedResultEnvelope(env, req.TraceID)
		item := FakAdmitBatchItem{Tool: req.Tool, TraceID: req.TraceID, Verdict: wv, Result: env}
		if err != nil {
			item.Error = err.Error()
		} else if env != nil && env.Status == "ERROR" {
			item.Error = resultEnvelopeError(env)
		}
		resp.Results = append(resp.Results, item)
	}
	return resp
}

// resultEnvelopeError lifts the read/admit engine's stable {"error":"..."} payload onto
// the batch row. The full envelope remains present for callers that already understand it.
func resultEnvelopeError(env *ResultEnvelope) string {
	if env == nil {
		return ""
	}
	var payload struct {
		Error string `json:"error"`
	}
	if json.Unmarshal([]byte(env.Content), &payload) == nil && payload.Error != "" {
		return payload.Error
	}
	return env.Content
}

// FakReadBatchResponse is the MCP wire shape for {file_paths:[...]}. ItemCount makes
// fusion visible to width observers: one MCP call doing five reads is not mistaken for
// one call doing one read.
type FakReadBatchResponse struct {
	Results   []FakReadBatchItem `json:"results"`
	TraceID   string             `json:"trace_id,omitempty"`
	ItemCount int                `json:"item_count"`
}

type FakReadBatchItem struct {
	FilePath string          `json:"file_path"`
	Verdict  WireVerdict     `json:"verdict"`
	Result   *ResultEnvelope `json:"result,omitempty"`
	Error    string          `json:"error,omitempty"`
}

// FakAdmitBatchResponse is the MCP wire shape for {items:[{tool,result},...]}; rows are
// deliberately independent because each result must cross the result-side safety floor.
type FakAdmitBatchResponse struct {
	Results   []FakAdmitBatchItem `json:"results"`
	TraceID   string              `json:"trace_id,omitempty"`
	ItemCount int                 `json:"item_count"`
}

type FakAdmitBatchItem struct {
	Tool    string          `json:"tool"`
	TraceID string          `json:"trace_id,omitempty"`
	Verdict WireVerdict     `json:"verdict"`
	Result  *ResultEnvelope `json:"result,omitempty"`
	Error   string          `json:"error,omitempty"`
}
