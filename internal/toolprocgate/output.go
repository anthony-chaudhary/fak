package toolprocgate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// OutputChannel names the child-owned surface about to enter parent-visible
// state. It is open in practice, but the constants pin the high-risk channels
// issue #2359 tracks.
type OutputChannel string

const (
	ChannelStdout               OutputChannel = "stdout"
	ChannelStderr               OutputChannel = "stderr"
	ChannelStructuredToolResult OutputChannel = "structured_tool_result"
	ChannelMCPHelperResult      OutputChannel = "mcp_helper_result"
	ChannelSubagentSummary      OutputChannel = "subagent_summary"
	ChannelTranscriptSidecar    OutputChannel = "transcript_sidecar"
	ChannelLog                  OutputChannel = "log"
)

// ChildOutput is a byte slice produced by a child/subagent boundary before a
// parent context, transcript sidecar, or normal log observes it.
type ChildOutput struct {
	AgentRunID string
	CallID     string
	Tool       string
	Channel    OutputChannel
	Bytes      []byte
}

// OutputAdmission is the safe-to-forward result of admitting ChildOutput.
// Bytes are the admitted payload or quarantine/deny stub, never the original
// bytes when a result-admission rung held them out.
type OutputAdmission struct {
	AgentRunID  string
	CallID      string
	Channel     OutputChannel
	InputLen    int
	InputSHA256 string
	Verdict     abi.Verdict
	Result      *abi.Result
	Bytes       []byte
	Meta        map[string]string
}

// AdmitChildOutput routes child-owned bytes through the registered result
// admission chain before any parent-visible surface consumes them. The adapter
// intentionally mirrors kernel.AdmitResult's fold semantics for normal
// Allow/Defer/Transform/Quarantine results, with one extra fail-closed branch:
// a deny/require-witness result is converted to a bounded stub so logs and
// transcript sidecars never receive raw bytes the chain refused to admit.
func AdmitChildOutput(ctx context.Context, in ChildOutput) OutputAdmission {
	channel := in.Channel
	if channel == "" {
		channel = ChannelStructuredToolResult
	}
	tool := in.Tool
	if tool == "" {
		tool = "child_output"
	}
	trace := in.CallID
	if trace == "" {
		trace = in.AgentRunID
	}
	body := append([]byte(nil), in.Bytes...)
	sum := sha256.Sum256(body)
	meta := map[string]string{
		"agent_run_id":   in.AgentRunID,
		"source_channel": string(channel),
		"input_len":      strconv.Itoa(len(body)),
		"input_sha256":   hex.EncodeToString(sum[:]),
	}
	c := &abi.ToolCall{
		Tool:    tool,
		TraceID: trace,
		Meta: map[string]string{
			"agent_run_id":   in.AgentRunID,
			"source_channel": string(channel),
		},
	}
	r := &abi.Result{
		Call:    c,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent},
		Meta:    cloneStringMap(meta),
	}
	v := admitChildResult(ctx, c, r)
	if v.Meta == nil {
		v.Meta = map[string]string{}
	}
	for k, val := range meta {
		if val != "" {
			v.Meta[k] = val
		}
		if r.Meta == nil {
			r.Meta = map[string]string{}
		}
		if val != "" {
			r.Meta[k] = val
		}
	}
	if rn := abi.ReasonName(v.Reason); rn != "" {
		r.Meta["reason"] = rn
		v.Meta["reason"] = rn
	}
	return OutputAdmission{
		AgentRunID:  in.AgentRunID,
		CallID:      in.CallID,
		Channel:     channel,
		InputLen:    len(body),
		InputSHA256: meta["input_sha256"],
		Verdict:     v,
		Result:      r,
		Bytes:       resolveOutputBytes(ctx, r.Payload),
		Meta:        cloneStringMap(r.Meta),
	}
}

func admitChildResult(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	chain := abi.ResultAdmittersFor(c)
	if len(chain) == 0 || r == nil {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	}
	best := abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	bestRank := abi.FoldRank(abi.VerdictAllow)
	for _, ra := range chain {
		v := ra.Admit(ctx, c, r)
		if rk := abi.FoldRank(v.Kind); rk > bestRank {
			bestRank, best = rk, v
		}
	}
	switch best.Kind {
	case abi.VerdictQuarantine:
		r.Outcome = abi.OutcomeCommitted
		ensureMeta(r)["admit"] = "quarantined"
	case abi.VerdictTransform:
		if tp, ok := best.Payload.(abi.TransformPayload); ok {
			r.Payload = tp.NewArgs
		}
		ensureMeta(r)["admit"] = "transformed"
	case abi.VerdictDeny, abi.VerdictRequireWitness:
		reason := abi.ReasonName(best.Reason)
		if reason == "" {
			reason = "RESULT_ADMISSION_REFUSED"
		}
		stub := map[string]any{
			"_quarantined": true,
			"reason":       reason,
			"by":           "toolprocgate",
			"len":          payloadLen(ctx, r.Payload),
			"_note":        "child output refused by result admission; payload held out of parent-visible surfaces",
		}
		if ref, ok := putJSON(ctx, stub); ok {
			ref.Taint = abi.TaintQuarantined
			r.Payload = ref
		} else {
			r.Payload = abi.Ref{Kind: abi.RefInline, Taint: abi.TaintQuarantined}
		}
		r.Outcome = abi.OutcomeCommitted
		ensureMeta(r)["admit"] = "denied"
	default:
		ensureMeta(r)["admit"] = "admitted"
	}
	return best
}

func resolveOutputBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return append([]byte(nil), r.Inline...)
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return append([]byte(nil), b...)
		}
	}
	return nil
}

func ensureMeta(r *abi.Result) map[string]string {
	if r.Meta == nil {
		r.Meta = map[string]string{}
	}
	return r.Meta
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
