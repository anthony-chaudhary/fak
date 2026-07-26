package gateway

// principal.go stamps every inbound event with a kernel-assigned AUTHORITY
// principal and type-checks authority-consuming acts against it (#2412, part of the
// harness-native program #2387 / epic #2390).
//
// The move client harnesses learned incident by incident — no relayed message
// (webhook delivery, peer-agent message, scheduled task) may carry USER authority —
// is made once, structurally, here: the inbound decode path labels who a turn came
// from, and a consent-shaped act (today a `_fak_confirm` injection that consumes a
// reversibility preview token) is honored ONLY when that label is the human. A
// peer / timer / network-tool / unknown confirm is stripped BEFORE adjudication, so
// the underlying irreversible call falls back to its REQUIRE_WITNESS hold rather than
// being waved through by a relayed approval. "A webhook approved the destructive
// call" becomes unrepresentable across every wire, not patched per channel.
//
// This is the AUTHORITY principal (who CONSENTED), deliberately distinct from the
// tenant ISOLATION principal already on the wire (principalFor / X-Fak-Principal, an
// auth subject used to scope the vDSO cache in wire.go). They are orthogonal: a
// single tenant's session still carries human vs peer-agent authority turn by turn.

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Principal is the closed set of inbound-event authority labels the kernel assigns.
type Principal string

const (
	// PrincipalHuman is the only label that may consume user authority.
	PrincipalHuman Principal = "human"
	// PrincipalSelfModel is the agent's own model turn (self-issued, not a human act).
	PrincipalSelfModel Principal = "self-model"
	// PrincipalPeerAgent is an A2A / peer-agent relayed message.
	PrincipalPeerAgent Principal = "peer-agent"
	// PrincipalTimer is a scheduled / cron-driven turn.
	PrincipalTimer Principal = "timer"
	// PrincipalNetworkTool is an inbound network tool / webhook delivery.
	PrincipalNetworkTool Principal = "network-tool"
	// PrincipalUnknown is the fail-closed label for an inbound source that named a
	// principal class the kernel does not recognize. It is NOT human, so it can never
	// consume authority — an ambiguous source defaults here, never to human.
	PrincipalUnknown Principal = "unknown"
)

// inboundPrincipalHeader is the wire carrier a trusted front door sets to label a
// RELAYED inbound turn. Absent header => the direct interactive wire => the human at
// the client. A relay (webhook bridge, A2A peer, scheduler) MUST stamp this so its
// turn cannot present as human.
const inboundPrincipalHeader = "X-Fak-Principal-Class"

// ReasonPrincipalNotHuman is the closed refusal reason a consent-shaped act draws
// when it arrives under a non-human principal.
const ReasonPrincipalNotHuman = "PRINCIPAL_NOT_HUMAN"

// IsHuman reports whether this principal may consume user authority. Only the human
// principal can; every other label (and the zero value) fails closed.
func (p Principal) IsHuman() bool { return p == PrincipalHuman }

// classifyPrincipal maps an inbound header value onto the closed label set. An EMPTY
// value is the direct interactive wire (the human at the client presents no relay
// label). Recognized relay classes — plus the common aliases "peer"/"a2a" and
// "webhook" — map through; an UNRECOGNIZED non-empty value fails closed to
// PrincipalUnknown rather than defaulting to human.
func classifyPrincipal(raw string) Principal {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return PrincipalHuman
	case string(PrincipalHuman):
		return PrincipalHuman
	case string(PrincipalSelfModel), "self", "model":
		return PrincipalSelfModel
	case string(PrincipalPeerAgent), "peer", "a2a":
		return PrincipalPeerAgent
	case string(PrincipalTimer), "cron", "schedule", "scheduled":
		return PrincipalTimer
	case string(PrincipalNetworkTool), "webhook", "network":
		return PrincipalNetworkTool
	default:
		return PrincipalUnknown
	}
}

// principalAuthorityRefusal records one consent-shaped act refused because the
// inbound principal was not human — the journalable witness that a relayed approval
// was neutralized.
type principalAuthorityRefusal struct {
	Tool      string
	Principal Principal
	Reason    string
}

// gateInboundAuthority type-checks the authority-consuming acts in a proposed
// tool-call batch against the inbound principal. For the human principal it is a
// no-op passthrough. For every other principal it STRIPS any confirmation token from
// a call's arguments — so the reversibility rung re-holds the underlying irreversible
// call at REQUIRE_WITNESS instead of consuming a relayed approval — and records a
// closed-reason refusal. The calls are RETURNED (not dropped): the call is still
// adjudicated, it just cannot carry a non-human approval. The input slice is never
// mutated in place.
func gateInboundAuthority(p Principal, calls []agent.ToolCall) ([]agent.ToolCall, []principalAuthorityRefusal) {
	if p.IsHuman() || len(calls) == 0 {
		return calls, nil
	}
	var refusals []principalAuthorityRefusal
	out := make([]agent.ToolCall, len(calls))
	copy(out, calls)
	for i := range out {
		stripped, did := stripConfirmationArgs(out[i].Function.Arguments)
		if !did {
			continue
		}
		out[i].Function.Arguments = stripped
		refusals = append(refusals, principalAuthorityRefusal{
			Tool:      out[i].Function.Name,
			Principal: p,
			Reason:    ReasonPrincipalNotHuman,
		})
	}
	return out, refusals
}

// stripConfirmationArgs removes every confirmation key from a JSON tool-argument
// object, returning the re-encoded object and whether anything was stripped. A
// non-object / unparseable argument string is returned unchanged (nothing to strip);
// the remaining keys are preserved verbatim.
func stripConfirmationArgs(raw string) (string, bool) {
	if strings.TrimSpace(raw) == "" {
		return raw, false
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return raw, false
	}
	stripped := false
	for k := range m {
		if isConfirmationArgKey(k) {
			delete(m, k)
			stripped = true
		}
	}
	if !stripped {
		return raw, false
	}
	b, err := json.Marshal(m)
	if err != nil {
		return raw, false
	}
	return string(b), true
}

// isConfirmationArgKey mirrors the adjudicator's closed confirmation-key set
// (internal/adjudicator/reversibility.go isConfirmationKey / ReversibilityConfirmArg —
// the canonical definition). This package cannot reach its unexported form, so the
// closed set is duplicated here with a pointer to the source of truth. Keep in sync:
// a key the adjudicator would consume as a confirmation but this gate does not strip
// would let a relayed approval through.
func isConfirmationArgKey(k string) bool {
	switch strings.ToLower(k) {
	case "_fak_confirm", "_fak_confirm_token", "confirm_token", "confirmation_token":
		return true
	}
	return false
}

// bindTracePrincipal records the AUTHORITY principal a trace's CURRENT turn is
// attributed to. Called from the served-request boundary each time a trace serves a
// turn. Last-writer-wins because authority is a per-turn fact (unlike traceOwner's
// first-writer-wins session ownership): a session that relays a peer turn mid-stream
// must carry the peer principal for THAT turn. A nil server or empty trace is a safe
// no-op. Bounded by the same generational reset as traceOwner so it cannot grow
// unbounded.
func (s *Server) bindTracePrincipal(trace string, p Principal) {
	if s == nil || strings.TrimSpace(trace) == "" {
		return
	}
	s.tracePrincipalMu.Lock()
	defer s.tracePrincipalMu.Unlock()
	if s.tracePrincipal == nil {
		s.tracePrincipal = make(map[string]Principal)
	}
	if len(s.tracePrincipal) >= maxCtxRestoreSessions {
		if _, ok := s.tracePrincipal[trace]; !ok {
			s.tracePrincipal = make(map[string]Principal) // generational reset, like traceOwner
		}
	}
	s.tracePrincipal[trace] = p
}

// tracePrincipalOf returns the authority principal bound to a trace, defaulting to
// PrincipalHuman for an unbound trace (the direct interactive loopback, where the
// human at the client drives every turn).
func (s *Server) tracePrincipalOf(trace string) Principal {
	if s == nil {
		return PrincipalHuman
	}
	s.tracePrincipalMu.RLock()
	defer s.tracePrincipalMu.RUnlock()
	if p, ok := s.tracePrincipal[trace]; ok {
		return p
	}
	return PrincipalHuman
}
