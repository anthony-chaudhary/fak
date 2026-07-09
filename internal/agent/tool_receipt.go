package agent

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// tool_receipt.go — TYPED tool-result receipts for fak's OWNED loop (#2414).
//
// On the PROXY path fak cannot author a tool_result block: the wire forbids a
// server-authored one, so a refusal is spliced into the assistant's own turn as a
// prose "[fak] ..." adjudicationNote (internal/gateway/refusal_notes.go) that the
// model reads in its own voice — and, being prose, routinely ignores (the guardrsi
// livelock envelopes in adjudicate_proposed.go exist precisely because of that). In
// the OWNED loop fak controls the transcript, so a denied / dropped / not-sent call
// gets a REAL, typed tool_result on the exact originating call ID, carrying
// {reason, disposition, fix} from the closed refusal vocabulary — and a call that
// legitimately did nothing reports status=skipped rather than feigning success. This
// is the wire-legal upgrade the proxy prose note only approximated; the notes
// machinery demotes to a proxy-only compatibility shim.

// ToolResultStatus is the CLOSED status vocabulary of an owned-loop tool receipt.
type ToolResultStatus string

const (
	// ToolResultError is a REFUSED call (deny / drop / revoke): the deny-as-value the
	// next planner turn consumes to adapt without another wasted round-trip.
	ToolResultError ToolResultStatus = "error"
	// ToolResultSkipped is a call that legitimately did NOTHING — not-sent / no-effect
	// (e.g. a write barred behind an unconfirmed speculative read). Reported as such so
	// "nothing happened" is never folded into "it worked".
	ToolResultSkipped ToolResultStatus = "skipped"
)

// ToolReceipt is the typed tool_result the owned loop authors on the originating call
// ID. It is serialized as the tool message Content (an owned-loop tool result IS a real
// user-turn tool_result block, unlike the proxy path), so the next planner turn reads a
// structured verdict rather than prose.
type ToolReceipt struct {
	Status      ToolResultStatus `json:"status"`
	Reason      string           `json:"reason,omitempty"`      // closed refusal token, e.g. POLICY_BLOCK
	Disposition string           `json:"disposition,omitempty"` // RETRYABLE|WAIT|ESCALATE|TERMINAL
	Fix         string           `json:"fix,omitempty"`         // sanctioned remedy from the closed vocabulary
	Detail      string           `json:"detail,omitempty"`      // bounded human note (never args/result bytes)
}

// JSON renders the receipt as the tool message Content. Marshaling a fixed small struct
// never errors in practice; the defensive fallback keeps the loop from ever handing the
// model an empty result.
func (r ToolReceipt) JSON() string {
	b, err := json.Marshal(r)
	if err != nil {
		return `{"status":"error","reason":"RECEIPT_MARSHAL"}`
	}
	return string(b)
}

// denyToolReceipt builds the typed error receipt for a refused owned-loop call. The
// reason + disposition come from the kernel's deny-as-value result (kernel.DenyResult);
// the fix/remedy comes from the deciding verdict — the SAME source the gateway wire's
// renderVerdict surfaces: the arg-predicate rung's Meta["fix"] or the reversibility
// rung's Meta["dry_run_hint"]. A bare policy block carries no remedy, so Fix stays empty.
func denyToolReceipt(result *abi.Result, v abi.Verdict) ToolReceipt {
	rc := ToolReceipt{
		Status:      ToolResultError,
		Reason:      metaVal(result, "reason"),
		Disposition: metaVal(result, "disposition"),
	}
	if v.Meta != nil {
		fix := v.Meta["fix"]
		if fix == "" {
			fix = v.Meta["dry_run_hint"]
		}
		rc.Fix = fix
	}
	return rc
}

func metaVal(r *abi.Result, k string) string {
	if r == nil || r.Meta == nil {
		return ""
	}
	return r.Meta[k]
}
