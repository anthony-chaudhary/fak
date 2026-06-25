package a2achan

// floor.go — the capability floor on messages. The gate functions (gateSend /
// gateRecv / screenIngress) are the ONE source of truth: the Bus calls them
// directly, and the REGISTERED drivers (a2aGate, a2aIngress) call the same
// functions after decoding a synthetic ToolCall. Registering the drivers is what
// makes the message floor first-class IN the kernel — it lives in the same
// Adjudicator + ResultAdmitter registries the kernel walks for every tool call,
// so an "a2a.send"/"a2a.recv" call routed through a real kernel folds this exact
// gate. The drivers are CallScope-scoped to the a2a tools, so they never perturb
// an unrelated tool call's fold.

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Capability tokens — the negotiated send/recv rights, split for least privilege
// (a receiver-only agent cannot send). Absent the right, the floor denies.
const (
	CapA2ASend abi.Capability = "a2a.send"
	CapA2ARecv abi.Capability = "a2a.recv"
)

// Tool names for the synthetic ToolCall the registered drivers fold over. They
// are the lease/training tokens an a2a message would carry through a kernel.
const (
	ToolSend = "a2a.send"
	ToolRecv = "a2a.recv"
)

// Meta keys carrying the addressing on the synthetic ToolCall (Args holds the
// body Ref; Caps holds the advertised rights). Used only on the kernel-routed
// path; the Bus passes typed args to the gate directly.
const (
	metaFrom     = "a2a.from"
	metaToLocale = "a2a.to.locale"
	metaToID     = "a2a.to.id"
)

// Fold ranks for the registered drivers (low runs first; scoped, so the exact
// rank only matters relative to other a2a-scoped drivers — there are none).
const (
	rankGate    = 40
	rankIngress = 40
)

func hasCap(caps []abi.Capability, want abi.Capability) bool {
	for _, c := range caps {
		if c == want {
			return true
		}
	}
	return false
}

// gateSend is the send-time capability floor. Fail-closed:
//   - no CapA2ASend advertised (or it is not a registered/negotiable cap) → deny
//     with ReasonDefaultDeny (no affirmative send-right);
//   - a TaintQuarantined body → deny with ReasonTrustViolation (poison never sent);
//   - a ScopeAgent body to a channel that is not the sender's OWN (To.ID != From)
//     → deny with ReasonTrustViolation (a private payload cannot cross agents; the
//     sender must widen Scope to share).
//
// Otherwise VerdictAllow.
func gateSend(from string, to ChannelKey, body abi.Ref, caps []abi.Capability) abi.Verdict {
	if !hasCap(caps, CapA2ASend) || !abi.Supported(CapA2ASend) {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "a2achan/gate"}
	}
	if body.Taint == abi.TaintQuarantined {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation, By: "a2achan/gate"}
	}
	selfChannel := from != "" && to.ID == from
	if body.Scope == abi.ScopeAgent && !selfChannel {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation, By: "a2achan/gate"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "a2achan/gate"}
}

// gateRecv is the recv-time capability floor: without CapA2ARecv advertised (and
// registered) it denies with ReasonDefaultDeny (no affirmative receive-right).
func gateRecv(caps []abi.Capability) abi.Verdict {
	if !hasCap(caps, CapA2ARecv) || !abi.Supported(CapA2ARecv) {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "a2achan/gate"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "a2achan/gate"}
}

// screenIngress is the recv-time ingress screen (the dual of the context-MMU's
// result admission): a TaintQuarantined message is HELD (VerdictQuarantine), so a
// poisoned message never enters the receiver's context. An admitted message keeps
// its Taint unchanged (sharing a result shares its taint).
func screenIngress(body abi.Ref) abi.Verdict {
	if body.Taint == abi.TaintQuarantined {
		return abi.Verdict{
			Kind:    abi.VerdictQuarantine,
			Payload: abi.QuarantinePayload{PageOut: true},
			Reason:  abi.ReasonTrustViolation,
			By:      "a2achan/ingress",
		}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "a2achan/ingress"}
}

// --- the registered drivers (the kernel-walked floor) --------------------------

// a2aGate is the registered Adjudicator for the a2a tools. It decodes the
// synthetic ToolCall and folds the SAME gateSend/gateRecv used by the Bus, so a
// message routed through a real kernel is gated identically. CallScope-scoped to
// the a2a tools (Defers on everything else), so it never runs on an unrelated call.
type a2aGate struct{}

func (a2aGate) Tools() []string        { return []string{ToolSend, ToolRecv} }
func (a2aGate) Caps() []abi.Capability { return []abi.Capability{CapA2ASend, CapA2ARecv} }
func (a2aGate) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil {
		return abi.Verdict{Kind: abi.VerdictDefer}
	}
	switch c.Tool {
	case ToolSend:
		from := ""
		toID := ""
		if c.Meta != nil {
			from = c.Meta[metaFrom]
			toID = c.Meta[metaToID]
		}
		return gateSend(from, ChannelKey{ID: toID}, c.Args, c.Caps)
	case ToolRecv:
		return gateRecv(c.Caps)
	}
	return abi.Verdict{Kind: abi.VerdictDefer}
}

// a2aIngress is the registered ResultAdmitter for the recv tool: it folds the
// same screenIngress used by the Bus, so a delivered message routed through a real
// kernel's result-admission chain is screened identically. CallScope-scoped to
// ToolRecv (admits every other result as-is).
type a2aIngress struct{}

func (a2aIngress) Tools() []string        { return []string{ToolRecv} }
func (a2aIngress) Caps() []abi.Capability { return nil }
func (a2aIngress) Admit(_ context.Context, _ *abi.ToolCall, r *abi.Result) abi.Verdict {
	if r == nil {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "a2achan/ingress"}
	}
	return screenIngress(r.Payload)
}

// sendCall builds the synthetic ToolCall the kernel-routed path would carry — the
// encode side of a2aGate's decode. Exposed for the test that proves the registered
// adjudicator decides identically to the Bus's direct gate call.
func sendCall(from string, to ChannelKey, body abi.Ref, caps []abi.Capability) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: ToolSend,
		Args: body,
		Caps: caps,
		Meta: map[string]string{
			metaFrom:     from,
			metaToLocale: to.Locale.String(),
			metaToID:     to.ID,
		},
	}
}

func init() {
	abi.RegisterCapability(CapA2ASend)
	abi.RegisterCapability(CapA2ARecv)
	abi.RegisterAdjudicator(rankGate, a2aGate{})
	abi.RegisterResultAdmitter(rankIngress, a2aIngress{})
}
