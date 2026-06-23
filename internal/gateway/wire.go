package gateway

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// ---------------------------------------------------------------------------
// The verdict, rendered for a non-Go client.
// ---------------------------------------------------------------------------

// WireVerdict is the stable, named projection of an abi.Verdict. The ABI carries
// the kind/reason as bare integers with no String(); the gateway attaches the
// stable names (the closed vocabulary) so a client never has to know an integer.
// Disposition is the actionable deny-loopback class — it is what lets a refusal
// cost a non-Go agent zero extra model turns.
type WireVerdict struct {
	Kind        string            `json:"kind"`                  // ALLOW|DENY|TRANSFORM|QUARANTINE|REQUIRE_WITNESS|DEFER|KIND_<n>
	Reason      string            `json:"reason,omitempty"`      // closed refusal vocabulary, e.g. POLICY_BLOCK
	By          string            `json:"by,omitempty"`          // which adjudicator decided (forensics)
	Disposition string            `json:"disposition,omitempty"` // RETRYABLE|WAIT|ESCALATE|TERMINAL
	Detail      map[string]string `json:"detail,omitempty"`      // bounded disclosure (e.g. the offending self-modify glob)
}

// renderVerdict projects a folded abi.Verdict (and the optional result Meta) onto
// the wire. resultMeta is the admitted Result's Meta, consulted only to surface a
// context-MMU quarantine that happened at admit-time (after an Allow verdict).
func renderVerdict(v abi.Verdict, resultMeta map[string]string) WireVerdict {
	w := WireVerdict{Kind: verdictKindName(v.Kind), By: v.By}
	if v.Reason != abi.ReasonNone {
		w.Reason = abi.ReasonName(v.Reason)
	}

	switch v.Kind {
	case abi.VerdictDeny:
		w.Disposition = kernel.Disposition(v.Reason)
	case abi.VerdictRequireWitness:
		// The gate is unresolved at adjudicate-time: route it for escalation rather
		// than collapsing it to a plain refusal, so the client can hand it to a
		// witness / human-approval queue.
		w.Disposition = "ESCALATE"
	default:
		// Any OTHER kind (an unknown/registered restrictive kind, e.g. a plan-CFI
		// RequireApproval) — fail closed. The kernel's Submit holds ANY non-core kind
		// as deny-as-value STRUCTURALLY, regardless of its declared FallbackClass, so
		// we mirror that unconditionally rather than gating on abi.Fallback (which
		// would drop the disposition for a kind declaring FallbackDefer/FallbackAllow
		// that the kernel nonetheless refused). Allow/Transform/Quarantine/Defer are
		// the only non-deny outcomes and are excluded.
		if v.Kind != abi.VerdictAllow && v.Kind != abi.VerdictTransform &&
			v.Kind != abi.VerdictQuarantine && v.Kind != abi.VerdictDefer {
			w.Disposition = "ESCALATE"
		}
	}

	// Bounded disclosure: a witness payload carries ONLY the offending claim/glob
	// (the deny channel is not a policy oracle).
	if wp, ok := v.Payload.(abi.WitnessPayload); ok && wp.Claim != "" {
		w.Detail = map[string]string{"claim": wp.Claim}
	}

	// A result quarantined by the context-MMU at admit-time overrides the (Allow)
	// submit verdict — the poisoned bytes were already paged out.
	if resultMeta["admit"] == "quarantined" {
		w.Kind = "QUARANTINE"
	}
	return w
}

// verdictKindName maps a VerdictKind to its stable wire name. Unknown registered
// kinds render as KIND_<n> rather than leaking an integer or panicking.
func verdictKindName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "REQUIRE_WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	}
	return "KIND_" + itoa(uint64(k))
}

func statusName(s abi.Status) string {
	switch s {
	case abi.StatusOK:
		return "OK"
	case abi.StatusError:
		return "ERROR"
	case abi.StatusPending:
		return "PENDING"
	}
	return "UNKNOWN"
}

// ---------------------------------------------------------------------------
// The fak-native syscall/adjudicate wire (the simplest non-Go integration).
// ---------------------------------------------------------------------------

// SyscallRequest is the body of POST /v1/fak/syscall and /v1/fak/adjudicate, and
// the `arguments` of the fak_syscall / fak_adjudicate MCP tools. Arguments accepts
// EITHER a JSON object (run with these args) OR a JSON-encoded string (the OpenAI
// `function.arguments` convention) — never an abi.Ref. ReadOnly is an optional
// vDSO hint.
type SyscallRequest struct {
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	ReadOnly  bool            `json:"read_only,omitempty"`
	// Witness is an optional external world-state token (a git commit / blob hash /
	// lease epoch) the caller is reading at. It keys the vDSO entry for cross-agent
	// dedup and binds it for causal revocation: a later fak_revoke(witness) evicts every
	// pooled entry admitted under it. Empty => the consistency eraser alone (v0.1).
	Witness string `json:"witness,omitempty"`
	// TraceID correlates a session's calls so the result-side IFC ledger + plan-CFI
	// key their per-session state on it end-to-end. Optional: the gateway mints a
	// fresh non-empty id when the wire omits it (never the empty shared trace).
	TraceID string `json:"trace_id,omitempty"`
	// Principal is the OPTIONAL isolation principal (a tenant / user / auth subject)
	// this call is made on behalf of. When set (here or via the X-Fak-Principal header,
	// which takes precedence), the vDSO scopes its tier-2 cache entry to it: a DIFFERENT
	// principal can neither be served nor fill the same (tool,args) entry — closing the
	// cross-tenant cache leak and the hit/miss timing oracle. Empty => single-tenant
	// (every caller shares, v0.1 behavior). A tool declared vdso-Shareable ignores it
	// (public, identity-independent reads stay cross-tenant shared).
	Principal string `json:"principal,omitempty"`
}

// AdmitRequest is the body of POST /v1/fak/admit and the `arguments` of the
// fak_admit MCP tool. It carries a tool RESULT the CLIENT executed, so the gateway
// can run it through the kernel's result-side stack (context-MMU quarantine + IFC
// source-stamp) — arming the exfil floor on the path where fak does NOT run the
// tool. Result is the raw result content (a JSON object or a JSON-encoded string),
// never an abi.Ref.
type AdmitRequest struct {
	Tool    string          `json:"tool"`
	Result  json.RawMessage `json:"result,omitempty"`
	Witness string          `json:"witness,omitempty"`
	TraceID string          `json:"trace_id,omitempty"`
}

// ChangesRequest drains the cross-agent "what changed" feed. Since is the client's
// cursor (the Seq of the last event it saw); 0 returns everything retained.
type ChangesRequest struct {
	Since uint64 `json:"since,omitempty"`
}

// ChangesResponse is the drained feed slice plus the client's next cursor.
type ChangesResponse struct {
	Events []CoherenceEvent `json:"events"`
	Cursor uint64           `json:"cursor"`
}

// RevokeRequest triggers a fleet-wide refutation of an external world-state witness.
type RevokeRequest struct {
	Witness string `json:"witness"`
}

// RevokeResponse reports how many pooled entries the refutation stranded locally and the
// post-bump integrity epoch (the monotone refutation clock).
type RevokeResponse struct {
	Witness    string `json:"witness"`
	Evicted    int    `json:"evicted"`
	TrustEpoch uint64 `json:"trust_epoch"`
}

// ContextChangeRequest is the body of POST /v1/fak/context/change and the
// `arguments` of the fak_context_change MCP tool. It is deliberately
// negative-only: today the only accepted mutation is a tombstone that suppresses
// one persisted recall page from future model-visible context. The core image's
// CAS bytes are preserved for audit.
type ContextChangeRequest struct {
	ImageDir    string `json:"image_dir"`
	Action      string `json:"action,omitempty"`
	Step        int    `json:"step"`
	Digest      string `json:"digest,omitempty"`
	Reason      string `json:"reason"`
	RequestedBy string `json:"requested_by,omitempty"`
	Witness     string `json:"witness,omitempty"`
}

// ContextChangeResponse is the applied context-control ledger row plus the image
// directory it was persisted to. Tombstoned is a convenience boolean for clients
// that only need to know whether future recall will skip the page.
type ContextChangeResponse struct {
	ImageDir    string `json:"image_dir"`
	ID          string `json:"id"`
	Action      string `json:"action"`
	Step        int    `json:"step"`
	Digest      string `json:"digest"`
	Reason      string `json:"reason"`
	RequestedBy string `json:"requested_by"`
	Witness     string `json:"witness,omitempty"`
	TrustEpoch  uint64 `json:"trust_epoch,omitempty"`
	Applied     bool   `json:"applied"`
	Tombstoned  bool   `json:"tombstoned"`
}

// SyscallResponse is the result of an adjudicate(-and-execute). Result is present
// only on the execute path (fak_syscall); RepairedArguments is present only when
// the verdict is TRANSFORM (the canonical args the client should run instead).
type SyscallResponse struct {
	Verdict           WireVerdict     `json:"verdict"`
	Result            *ResultEnvelope `json:"result,omitempty"`
	RepairedArguments json.RawMessage `json:"repaired_arguments,omitempty"`
	TraceID           string          `json:"trace_id,omitempty"`
}

// ResultEnvelope is a tool result rendered for the wire (bytes resolved, never a
// Ref handle).
type ResultEnvelope struct {
	Status  string            `json:"status"`
	Content string            `json:"content"`
	Meta    map[string]string `json:"meta,omitempty"`
}

// rawArgs normalizes the `arguments` field to the raw argument bytes the kernel
// adjudicates. A JSON string is unquoted (OpenAI convention); a JSON object is
// used verbatim; absent/empty becomes "{}".
func rawArgs(m json.RawMessage) string {
	b := []byte(m)
	// trim leading whitespace to find the first significant byte
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	if i >= len(b) {
		return ""
	}
	if b[i] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err == nil {
			return s
		}
	}
	return string(b)
}

// ---------------------------------------------------------------------------
// The OpenAI /v1/chat/completions wire (exported mirrors of agent's unexported
// chatRequest/chatResponse; the message/tool/usage vocabulary is reused).
// ---------------------------------------------------------------------------

// ChatRequest is the inbound /v1/chat/completions body. The sampling fields
// (max_tokens, temperature, top_p, stop) are parsed and forwarded to the upstream
// model per request — an omitted field falls through to the planner default, so a
// client that asks for a long completion is no longer hard-capped at the planner's
// 1024-token floor. tool_choice and other unknown OpenAI fields are still accepted
// and ignored (drop-in compatibility); there is, by construction, no Ref field to
// smuggle. stream=true is supported by buffering the upstream turn, adjudicating the
// complete proposed tool-call set, then emitting a synthetic SSE stream. Raw
// upstream deltas are never passed through before adjudication.
type ChatRequest struct {
	Model       string          `json:"model"`
	Messages    []agent.Message `json:"messages"`
	Tools       []agent.ToolDef `json:"tools,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	// Stop is raw because the OpenAI wire allows EITHER a bare string OR an array of
	// strings; decoding straight into []string would reject the common `"stop":"\n"`
	// form. normalizeStop folds both shapes to a slice.
	Stop   json.RawMessage `json:"stop,omitempty"`
	Stream bool            `json:"stream,omitempty"`
}

// normalizeStop folds the OpenAI `stop` field (a bare string, an array of strings,
// or absent/null) into a string slice for the planner seam. Anything malformed
// degrades to nil (treated as "no stop sequences") rather than erroring the request.
func normalizeStop(raw json.RawMessage) []string {
	b := []byte(raw)
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	if i >= len(b) {
		return nil
	}
	switch b[i] {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err == nil && s != "" {
			return []string{s}
		}
	case '[':
		var arr []string
		if err := json.Unmarshal(raw, &arr); err == nil {
			return arr
		}
	}
	return nil
}

// ChatResponse is the outbound completion. The `fak` extension carries the
// per-tool-call adjudications for a fak-aware client; a fak-unaware client simply
// never sees the denied tool_calls (they are stripped from the message).
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   agent.Usage  `json:"usage"`
	Fak     *FakExt      `json:"fak,omitempty"`
}

// ChatStreamResponse is the OpenAI-compatible SSE chunk shape emitted when the
// downstream client requests stream=true. Two paths produce it. The LIVE path
// (streamChatLive, taken for a no-tools request whose planner can stream the wire)
// relays each upstream content fragment as its own chunk for a real
// time-to-first-token, then emits a terminal finish/usage chunk. The BUFFERED path
// (writeChatCompletionStream, taken for a tool-bearing request or a non-streaming
// planner) synthesizes the chunks only after the whole turn is adjudicated. Either
// way a tool-call delta carries only filtered/repaired calls — no un-adjudicated
// call is ever streamed.
type ChatStreamResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []ChatStreamChoice `json:"choices"`
	Usage   *agent.Usage       `json:"usage,omitempty"`
	Fak     *FakExt            `json:"fak,omitempty"`
}

// ChatStreamChoice is one streamed completion choice.
type ChatStreamChoice struct {
	Index        int       `json:"index"`
	Delta        ChatDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason"`
}

// ChatDelta is the incremental assistant payload in an SSE chunk.
type ChatDelta struct {
	Role      string              `json:"role,omitempty"`
	Content   string              `json:"content,omitempty"`
	ToolCalls []ChatDeltaToolCall `json:"tool_calls,omitempty"`
}

// ChatDeltaToolCall mirrors OpenAI's streaming tool-call delta entries.
type ChatDeltaToolCall struct {
	Index    int        `json:"index"`
	ID       string     `json:"id,omitempty"`
	Type     string     `json:"type,omitempty"`
	Function agent.Func `json:"function"`
}

// ChatChoice is one completion choice.
type ChatChoice struct {
	Index        int           `json:"index"`
	Message      agent.Message `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// FakExt is the gateway's non-standard response extension: inbound result
// admissions plus the adjudication of every tool_call the model proposed,
// including the dropped ones.
type FakExt struct {
	Adjudications    []ToolAdjudication `json:"adjudications,omitempty"`
	ResultAdmissions []ResultAdmission  `json:"result_admissions,omitempty"`
}

// ToolAdjudication is one proposed tool_call's verdict.
type ToolAdjudication struct {
	ToolCallID        string          `json:"tool_call_id,omitempty"`
	Tool              string          `json:"tool"`
	Admitted          bool            `json:"admitted"`
	Verdict           WireVerdict     `json:"verdict"`
	RepairedArguments json.RawMessage `json:"repaired_arguments,omitempty"`
}

// ResultAdmission is one inbound tool result admitted before it is forwarded to
// an upstream model on the OpenAI-compatible proxy path.
type ResultAdmission struct {
	ToolCallID string      `json:"tool_call_id,omitempty"`
	Tool       string      `json:"tool"`
	Verdict    WireVerdict `json:"verdict"`
}

// itoa is a tiny dependency-free uint formatter (the repo avoids strconv on its
// hot paths; mirror that posture here).
func itoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
