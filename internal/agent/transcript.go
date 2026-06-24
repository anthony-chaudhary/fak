package agent

import (
	"context"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/wirescreen"
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

// TranscriptRedaction records one message whose content was span-redacted immediately
// before an API request is serialized: the message index, the redactor that proposed the
// spans, the CAS handle to the UNREDACTED original (wirescreen.Restore returns it
// byte-exact), the spans, and the redacted length. It is the reversibility-witness peer
// of TranscriptQuarantine for an in-place span rewrite (the local-model-on-the-wire
// spine, rung 5 / issue #572): a quarantine HOLDS OUT a whole message; a redaction
// REWRITES the flagged spans and keeps the surrounding bytes on the wire.
type TranscriptRedaction struct {
	Index      int               `json:"index"`
	Tool       string            `json:"tool,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
	By         string            `json:"by"`
	Original   abi.Ref           `json:"original"` // CAS handle; wirescreen.Restore(ctx, Original) returns the bytes byte-exact
	Spans      []wirescreen.Span `json:"spans,omitempty"`
	Len        int               `json:"len"` // redacted content length
}

// RedactOutboundMessages applies the active PII/secret redactor's span rewrite to the
// outbound messages' content — the rung-5 wire point on the non-passthrough re-marshal
// path (issue #572). When wirescreen.ActiveRedactor() selects a Redactor
// (FAK_WIRE_REDACT), each message's content is passed through wirescreen.Apply and the
// flagged spans are replaced with "[REDACTED:<kind>]" placeholders, while the UNREDACTED
// original is pinned in the shared CAS so an authorized caller can Restore it byte-exact
// (the same pageOut + PinResolved witness ctxmmu's quarantine uses). It is strictly
// one-sided (only the flagged spans change; surrounding bytes are untouched) and fails
// open on a miss (an unflagged PII span passes through — honest scope, not a bug).
//
// Default-inert: with no redactor active (FAK_WIRE_REDACT unset) the slice is returned
// UNCHANGED and untouched at zero cost, so the default outbound path is byte-identical to
// today and the inert contract the spine already holds is preserved.
//
// It does NOT ride the flagship `fak guard -- claude` Anthropic passthrough: that route
// forwards req.Raw verbatim (stream.go, WithRawRequestBody) and never serializes these
// messages, so a span rewrite here changes nothing the model reads on it until the
// cache-prefix-preserving req.Raw transform (#555, ctxplan-owned) lands. It lands the
// redaction only where it can reach the wire today — the non-passthrough re-marshal
// (OpenAI/xAI proxy, mock, local serve).
func RedactOutboundMessages(messages []Message) ([]Message, []TranscriptRedaction) {
	return redactOutbound(wirescreen.ActiveRedactor(), messages)
}

// redactOutbound is the testable core of RedactOutboundMessages: it runs r over every
// message's content. A nil r is the inert path (messages returned unchanged, zero cost).
// Splitting it out lets the witness test drive a concrete redactor without env coupling.
func redactOutbound(r wirescreen.Redactor, messages []Message) ([]Message, []TranscriptRedaction) {
	if r == nil {
		return messages, nil
	}
	ctx := context.Background()
	out := append([]Message(nil), messages...)
	var redactions []TranscriptRedaction
	for i := range out {
		if len(out[i].Content) == 0 {
			continue
		}
		red, ok := wirescreen.Apply(ctx, r, []byte(out[i].Content), out[i].Name)
		if !ok {
			continue
		}
		out[i].Content = string(red.Redacted)
		redactions = append(redactions, TranscriptRedaction{
			Index:      i,
			Tool:       out[i].Name,
			ToolCallID: out[i].ToolCallID,
			By:         red.By,
			Original:   red.Original,
			Spans:      red.Spans,
			Len:        len(red.Redacted),
		})
	}
	return out, redactions
}
