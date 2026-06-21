package agent

import (
	"context"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TranscriptQuarantine records a tool-result payload held out immediately before
// an API request is serialized. Closed APIs cannot un-attend a provider-owned KV
// cache after the fact, so the enforceable boundary is pre-send.
type TranscriptQuarantine struct {
	Index      int    `json:"index"`
	Tool       string `json:"tool,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	Reason     string `json:"reason"`
	Len        int    `json:"len"`
}

// QuarantineOutboundMessages returns a copy of messages with unsafe tool-result
// bytes held out before any provider adapter can serialize them. It runs the same
// registered ResultAdmitter chain used by the kernel result path, so a binary
// that links the full defconfig inherits normgate, ctxmmu, and IFC source-stamping.
// This boundary is deliberately tool-result scoped: user and assistant messages
// are already authored or accepted as prompt context, while tool results are the
// untrusted cross-boundary bytes the client can still hold out before serialization.
func QuarantineOutboundMessages(messages []Message) ([]Message, []TranscriptQuarantine) {
	out := append([]Message(nil), messages...)
	var qs []TranscriptQuarantine
	ctx := context.Background()
	for i := range out {
		if out[i].Role != RoleTool {
			continue
		}
		body := []byte(out[i].Content)
		call := &abi.ToolCall{Tool: out[i].Name}
		res := &abi.Result{
			Call: call,
			Payload: abi.Ref{
				Kind:   abi.RefInline,
				Inline: append([]byte(nil), body...),
				Len:    int64(len(body)),
				Taint:  abi.TaintTainted,
				Scope:  abi.ScopeAgent,
			},
			Status: abi.StatusOK,
		}
		v := admitOutbound(ctx, call, res)
		switch v.Kind {
		case abi.VerdictQuarantine:
			q := TranscriptQuarantine{
				Index:      i,
				Tool:       out[i].Name,
				ToolCallID: out[i].ToolCallID,
				Reason:     abi.ReasonName(v.Reason),
				Len:        len(body),
			}
			qs = append(qs, q)
			out[i].Content = quarantineStub(len(qs), q)
		case abi.VerdictTransform:
			if content, ok := transformContent(ctx, v, res); ok {
				out[i].Content = content
			}
		}
	}
	return out, qs
}

func admitOutbound(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	chain := abi.ResultAdmitters()
	if len(chain) == 0 {
		chain = []abi.ResultAdmitter{ctxmmu.Default}
	}
	best := abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	bestRank := abi.FoldRank(abi.VerdictAllow)
	for _, ra := range chain {
		if ra == nil {
			continue
		}
		v := ra.Admit(ctx, c, r)
		if rk := abi.FoldRank(v.Kind); rk > bestRank {
			bestRank, best = rk, v
		}
	}
	return best
}

func transformContent(ctx context.Context, v abi.Verdict, r *abi.Result) (string, bool) {
	if tp, ok := v.Payload.(abi.TransformPayload); ok {
		if b, ok := transcriptRefBytes(ctx, tp.NewArgs); ok {
			return string(b), true
		}
	}
	if r != nil {
		if b, ok := transcriptRefBytes(ctx, r.Payload); ok {
			return string(b), true
		}
	}
	return "", false
}

func transcriptRefBytes(ctx context.Context, ref abi.Ref) ([]byte, bool) {
	if ref.Kind == abi.RefInline {
		return append([]byte(nil), ref.Inline...), true
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, ref); err == nil {
			return b, true
		}
	}
	return nil, false
}

func quarantineStub(n int, q TranscriptQuarantine) string {
	stub := map[string]any{
		"_quarantined": true,
		"id":           "preq" + itoa(n),
		"reason":       q.Reason,
		"len":          q.Len,
		"boundary":     "pre_send",
	}
	if q.Tool != "" {
		stub["tool"] = q.Tool
	}
	if q.ToolCallID != "" {
		stub["tool_call_id"] = q.ToolCallID
	}
	b, err := json.Marshal(stub)
	if err != nil {
		return `{"_quarantined":true,"boundary":"pre_send"}`
	}
	return string(b)
}
