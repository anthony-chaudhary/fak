package gateway

import (
	"encoding/json"
	"strings"
)

// reversibilityGateNote renders the actionable half of a REQUIRE_WITNESS
// preview-confirm verdict for clients that read only the in-band content
// channel — Claude Code on the Anthropic wire is exactly that client.
//
// The reversibility rung (#2156) refuses an irreversible/outward-facing call
// with a deterministic confirm token the caller must echo as `_fak_confirm`.
// That flow assumed a client that reads the `fak` extension or the MCP
// SyscallResponse, where WireVerdict.Detail["claim"] already carries the full
// envelope. A content-channel client saw only "Tool (/ESCALATE)" — no preview,
// no token, no dry-run hint — so the gate's own two-phase confirm could never
// complete: every outward-facing call (a plain `git push` on a repo whose
// contract is commit-AND-push) was a dead end instead of a pause. This note
// closes that gap by rendering the same envelope the other wires expose.
//
// Exposing the token in-band does not weaken the floor: the token is not a
// secret (the MCP wire returns the same envelope to any client, and the token
// is deterministically derivable from the call), and hard denies win before
// the reversibility rung and cannot be confirmed past
// (TestAdjudicateReversibilityGateDoesNotOverrideHardDeny). The gate is a
// deliberate acknowledged pause over a bounded preview, not a capability wall.
//
// The recipe says "byte-identical" out loud because the token hashes the
// call's canonical args minus confirm keys: a model that re-proposes with a
// reworded description or a tweaked command gets a FRESH refusal with a FRESH
// token, and chases tokens turn after turn instead of converging (witnessed
// wedging a fleet session for 1h+).
func reversibilityGateNote(a ToolAdjudication) string {
	if a.Verdict.Kind != "REQUIRE_WITNESS" || a.Verdict.By != "monitor/reversibility" {
		return ""
	}
	var env struct {
		Class        string `json:"class"`
		Preview      string `json:"preview"`
		ConfirmToken string `json:"confirm_token"`
		DryRunHint   string `json:"dry_run_hint"`
	}
	if err := json.Unmarshal([]byte(a.Verdict.Detail["claim"]), &env); err != nil || env.ConfirmToken == "" {
		return ""
	}
	var b strings.Builder
	b.WriteString("preview-confirm gate: ")
	if env.Preview != "" {
		b.WriteString(env.Preview)
	} else {
		b.WriteString(env.Class)
	}
	b.WriteString(`. If this call is deliberate, re-propose it byte-identical — same tool, every other argument unchanged, since any edit issues a different token — with "_fak_confirm":"`)
	b.WriteString(env.ConfirmToken)
	b.WriteString(`" added to the tool input (the kernel verifies and strips the key before dispatch, so the tool never sees it)`)
	if env.DryRunHint != "" {
		b.WriteString("; to preview instead, ")
		b.WriteString(env.DryRunHint)
	}
	b.WriteString(".")
	return b.String()
}

// reasonOrKind names a refusal for the in-band note: the closed-vocabulary
// Reason when the verdict carries one, else the verdict Kind — so a
// REQUIRE_WITNESS refusal renders "(REQUIRE_WITNESS/ESCALATE)" rather than
// the malformed-looking "(/ESCALATE)".
func reasonOrKind(v WireVerdict) string {
	if v.Reason != "" {
		return v.Reason
	}
	return v.Kind
}

// remedyNote renders the refusing rule's sanctioned alternative
// (arg_rules[].fix, carried as WireVerdict.Detail["fix"]) so the refusal and
// the recommended path arrive in the same breath — "great, it refused the bad
// call, but then WHAT?" answered in-band. Empty when the rule declares no fix.
// The generic "choose an allowed alternative" trailer stays; this names the
// alternative for rules whose author declared one.
func remedyNote(a ToolAdjudication) string {
	fix := a.Verdict.Detail["fix"]
	if fix == "" {
		return ""
	}
	return "sanctioned alternative: " + fix
}
