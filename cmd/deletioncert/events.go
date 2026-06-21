package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// decideEvent builds a benign DECIDE event the journal records as a DECIDE row.
// These sit before the anchor so the QUARANTINE row lands mid-chain (a realistic
// position, not row 1) — exercising the prev-hash linkage Verify re-derives.
func decideEvent(tool, taint string) abi.Event {
	call := &abi.ToolCall{Tool: tool, TraceID: "demo-" + tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"q":"` + tool + `"}`)}}
	res := &abi.Result{Call: call, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("ok"), Taint: taintLabel(taint)}}
	return abi.Event{Kind: abi.EvDecide, Call: call,
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "demo"}, Result: res}
}

// quarantineEvent builds the QUARANTINE event that records our eviction. The
// result payload is a secret-shaped inline Ref; the journal stamps its content
// hash as the row's ResultDigest WITHOUT logging the bytes (refDigest hashes the
// inline payload). That digest is what the certificate's Anchor.ResultDigest pins
// the subject to — so the receipt names WHICH data was evicted.
func quarantineEvent(witness string, span []int) abi.Event {
	// A stand-in "secret" payload, content-addressed but never logged verbatim.
	secret := []byte(fmt.Sprintf("sk-DEMO-%v-%s", span, witness))
	call := &abi.ToolCall{Tool: "fetch_credentials", TraceID: "demo-secret",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"resource":"vault"}`)}}
	res := &abi.Result{Call: call, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: secret, Taint: abi.TaintQuarantined}}
	return abi.Event{Kind: abi.EvQuarantine, Call: call,
		Verdict: &abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonSecretExfil,
			By: "ctxmmu", Payload: abi.QuarantinePayload{PageOut: true}},
		Result: res}
}

func taintLabel(s string) abi.TaintLabel {
	switch s {
	case "trusted":
		return abi.TaintTrusted
	case "quarantined":
		return abi.TaintQuarantined
	default:
		return abi.TaintTainted
	}
}
