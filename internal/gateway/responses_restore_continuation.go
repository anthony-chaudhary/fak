package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

// #11831: a restored page is input to the model, never a synthetic final answer.
// These budgets apply to gateway-owned tools only; client-registered MCP tools
// retain their ordinary tool round trip and full restoration schema.
const (
	responsesRestoreCallLimit        = 4
	responsesRestoreByteLimit        = 64 << 10
	responsesRestorePageLimit        = 16 << 10
	responsesRestoreIncompleteReason = "fak_context_restore"
)

type responsesRestoreContextKey struct{}

type responsesRestoreContinuation struct {
	clientTools   map[string]bool
	seen          map[string]bool
	calls         int
	bytes         int
	adjudications []ToolAdjudication
	blocked       bool
}

func newResponsesRestoreContinuation(tools []responsesTool) *responsesRestoreContinuation {
	r := &responsesRestoreContinuation{clientTools: make(map[string]bool), seen: make(map[string]bool)}
	for _, tool := range tools {
		if tool.Type == "function" && isRestoreTool(tool.Name) {
			r.clientTools[tool.Name] = true
		}
	}
	return r
}

func responsesRestoreSchema() json.RawMessage {
	for _, descriptor := range contextIntrospectionToolDescriptors() {
		if descriptor["name"] == "fak_context_restore" {
			return descriptor["inputSchema"].(json.RawMessage)
		}
	}
	return nil
}

// continueRestores consumes only gateway-owned restores. A mixed proposal's other
// calls have not run: defer them explicitly and ask the next sample to reissue
// any still needed. Never manufacture results for client-owned tools.
// completeServed applies the same admission, cancellation, and usage debits as the
// original sample. The returned history is also used by denial recovery.
func (r *responsesRestoreContinuation) continueRestores(ctx context.Context, s *Server, turn servedSessionTurn, comp *agent.Completion, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, []agent.Message) {
	for {
		var calls []agent.ToolCall
		for _, call := range comp.Message.ToolCalls {
			if isRestoreTool(call.Function.Name) && !r.clientTools[call.Function.Name] {
				calls = append(calls, call)
			}
		}
		if len(calls) == 0 {
			return comp, messages
		}
		if r.calls+len(calls) > responsesRestoreCallLimit || r.bytes >= responsesRestoreByteLimit {
			return r.incomplete(comp, "the bounded context retrieval budget was exhausted"), messages
		}
		var results []agent.Message
		for _, call := range calls {
			if strings.TrimSpace(call.ID) == "" {
				return r.incomplete(comp, "a restore request had no call ID"), messages
			}
			var req ContextRestoreRequest
			decodeErr := json.Unmarshal([]byte(call.Function.Arguments), &req)
			if req.TraceID == "" {
				req.TraceID = turn.traceID
			}
			req.ID = strings.TrimPrefix(strings.TrimSpace(req.ID), "sha256:")
			if offset, limit, ok := parseRange(req.Range); ok {
				if req.Offset == 0 {
					req.Offset = offset
				}
				if req.Limit == 0 {
					req.Limit = limit
				}
			}
			req.Range = ""
			if req.Offset < 0 {
				req.Offset = 0
			}
			if req.Limit <= 0 || req.Limit > responsesRestorePageLimit {
				req.Limit = responsesRestorePageLimit
			}
			// Bound before restoreContext's offset+limit arithmetic and before any
			// page is admitted into the continuation history.
			if remaining := responsesRestoreByteLimit - r.bytes; req.Limit > remaining {
				req.Limit = remaining
			}
			if req.Limit <= 0 {
				return r.incomplete(comp, "the context byte budget was exhausted"), messages
			}
			keyBytes, _ := json.Marshal(req)
			key := string(keyBytes)
			if decodeErr != nil {
				key = call.Function.Arguments
			}
			if r.seen[key] {
				return r.incomplete(comp, "the model repeated an unchanged context retrieval"), messages
			}
			r.seen[key] = true
			r.calls++
			seq := s.nextOriginSeq()
			adj := ToolAdjudication{ToolCallID: call.ID, Tool: call.Function.Name, ArgsDigest: guardrsi.ArgsDigest(call.Function.Arguments)}
			var result CtxRestoreResult
			err := decodeErr
			if err == nil {
				var repaired string
				adj.Verdict, repaired, err = s.adjudicateWithSeq(ctx, call.Function.Name, string(keyBytes), true, "", turn.traceID, seq)
				if err == nil && adj.Verdict.Kind != "ALLOW" && adj.Verdict.Kind != "TRANSFORM" {
					err = fmt.Errorf("context restoration refused: %s", adj.Verdict.Reason)
				}
				if err == nil && repaired != "" {
					// Execute only the policy's repaired request, provided it remains
					// within this continuation's resource envelope. Do not rewrite a
					// policy decision back into the model's original arguments.
					req = ContextRestoreRequest{}
					err = json.Unmarshal([]byte(repaired), &req)
					if err == nil && (req.Range != "" || req.Offset < 0 || req.Limit <= 0 || req.Limit > responsesRestorePageLimit || req.Limit > responsesRestoreByteLimit-r.bytes) {
						err = fmt.Errorf("repaired restore request exceeds the continuation bounds; reissue a bounded request")
					}
				}
				if err == nil {
					caller, _ := s.traceOwnerOf(turn.traceID)
					result, err = s.restoreContext(caller, req)
					if len(result.Excerpt) > 512 {
						result.Excerpt = result.Excerpt[:512]
					}
				}
			}
			payload := struct {
				CtxRestoreResult
				Error    string       `json:"error,omitempty"`
				Verdict  *WireVerdict `json:"verdict,omitempty"`
				Deferred string       `json:"deferred_calls,omitempty"`
			}{CtxRestoreResult: result}
			adj.Admitted = err == nil
			if err != nil {
				payload.Error = err.Error()
				if adj.Verdict.Kind == "" || adj.Verdict.Kind == "ALLOW" || adj.Verdict.Kind == "TRANSFORM" {
					adj.Verdict = WireVerdict{Kind: "DENY", Reason: "CONTEXT_RESTORE_FAILED", By: "restoreContext", Disposition: "RETRYABLE"}
				}
				payload.Verdict = &adj.Verdict
			}
			if len(calls) < len(comp.Message.ToolCalls) {
				payload.Deferred = "Other proposed calls have not executed. Reissue any still needed after reading this result."
			}
			content, _ := json.Marshal(payload)
			if len(content) > responsesRestoreByteLimit-r.bytes {
				return r.incomplete(comp, "the serialized context result exceeds the remaining retrieval budget"), messages
			}
			if adj.Admitted {
				// Restored bytes cross the same result admission boundary as MCP
				// output before they can enter the next model sample.
				wv, env, admitErr := s.admitOpWithSeq(ctx, "restore_admit", call.Function.Name, string(content), "", turn.traceID, seq)
				if wv.Kind == "QUARANTINE" {
					admitErr = s.resetEngineCacheAfterQuarantine(ctx, []ResultAdmission{{Verdict: wv}})
					if admitErr != nil {
						adj.Admitted = false
						adj.Verdict = wv
						r.adjudications = append(r.adjudications, adj)
						return r.incomplete(comp, "quarantined context could not be cleared from the upstream cache"), messages
					}
				}
				if admitErr != nil || env == nil || (wv.Kind != "ALLOW" && wv.Kind != "TRANSFORM") {
					adj.Admitted = false
					adj.Verdict = wv
					if admitErr != nil || wv.Kind == "" {
						adj.Verdict = WireVerdict{Kind: "DENY", Reason: "ADMIT_ERROR", By: "restore_admit"}
					}
					content, _ = json.Marshal(map[string]any{"error": "Restored context was withheld by result admission. Continue with an allowed approach.", "verdict": adj.Verdict})
				} else {
					content = []byte(env.Content)
				}
			}
			if len(content) > responsesRestoreByteLimit-r.bytes {
				return r.incomplete(comp, "the admitted context result exceeds the remaining retrieval budget"), messages
			}
			r.bytes += len(content)
			r.adjudications = append(r.adjudications, adj)
			results = append(results, agent.Message{Role: agent.RoleTool, ToolCallID: call.ID, Name: call.Function.Name, Content: string(content)})
		}
		proposal := comp.Message
		proposal.Role = agent.RoleAssistant
		proposal.ToolCalls = calls
		if turnBodyClaimsCompletedEdit(proposal.Content) {
			proposal.Content = ""
		}
		continued := make([]agent.Message, 0, len(messages)+1+len(results))
		continued = append(continued, messages...)
		continued = append(continued, proposal)
		continued = append(continued, results...)
		next, err := s.completeServed(ctx, turn, continued, tools, opts...)
		if err != nil || next == nil {
			return r.incomplete(comp, "the next model response could not be obtained"), continued
		}
		copyNext := *next
		copyNext.Usage = foldRecoveryUsage(comp.Usage, next.Usage)
		if next.ToolCallsDropped && len(next.Message.ToolCalls) == 0 {
			return r.incomplete(&copyNext, "the next model response contained unparseable tool calls"), continued
		}
		comp, messages = &copyNext, continued
	}
}

func (r *responsesRestoreContinuation) incomplete(comp *agent.Completion, reason string) *agent.Completion {
	r.blocked = true
	copyComp := *comp
	copyComp.Message = agent.Message{Role: agent.RoleAssistant, Content: fmt.Sprintf("[fak] CONTEXT_RESTORE_INCOMPLETE: %s. The original task remains open.", reason)}
	copyComp.FinishReason = "stop"
	copyComp.ToolCallsDropped = false
	return &copyComp
}
