// Package main is the mobile FFI reference: a pure-Go core that routes a
// proposed on-device tool call through fak's adjudicator seam BEFORE it becomes
// an Android Intent / Apple App Intent, plus a thin C-callable shim (see
// libfakmobile.go, built only with cgo) that exposes that core to an NDK or
// Swift host.
//
// The core here is deliberately platform-free and cgo-free so the deny/allow
// round-trip is witnessed on ANY host with `go test` / `go run .` (this repo's
// CI runs CGO_ENABLED=0). The C boundary is an additive shim: it adds the
// `FakAdjudicate` C symbol under `-buildmode=c-archive`, it changes no decision.
package main

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/abi"
)

// Decision is the FFI wire result: a single tool call's verdict, JSON-encoded
// across the C-string boundary. `Allow` is the one bit the host acts on —
// dispatch the intent iff true; everything else is forensic detail the host may
// surface or log. It is intentionally flat and self-describing so a Kotlin/Java
// or Swift decoder needs no fak types.
type Decision struct {
	Tool    string `json:"tool"`
	Allow   bool   `json:"allow"`
	Verdict string `json:"verdict"` // "ALLOW" | "DENY" | "DEFAULT_DENY"
	Reason  string `json:"reason"`  // closed ReasonCode name (e.g. "POLICY_BLOCK")
	By      string `json:"by"`      // which link decided (forensics, not dispatch)
}

// marshalJSON renders the Decision to its wire form. A marshal error is
// impossible for this flat struct, but if it ever occurred the fail-closed
// fallback is a DENY the host can still act on.
func (d Decision) marshalJSON() string {
	b, err := json.Marshal(d)
	if err != nil {
		return `{"allow":false,"verdict":"DENY","reason":"MALFORMED","by":"mobile/ffi"}`
	}
	return string(b)
}

// mobilePolicy is a minimal reference monitor implementing abi.Adjudicator (the
// same interface internal/adjudicator's reference floor implements — pkg/abi
// aliases the identical type). It mirrors the desktop live pilot's shape:
//   - a named dangerous tool is a PROVABLE refusal -> VerdictDeny(POLICY_BLOCK);
//   - a read-shaped benign tool is affirmatively allowed;
//   - everything else Defers, and the fold below fails it closed (DEFAULT_DENY).
//
// It is not the full internal floor (that package is sealed from an out-of-tree
// module); it is the floor's SEAM, exercised end-to-end, which is what the FFI
// boundary carries.
type mobilePolicy struct {
	deny        map[string]bool // dangerous tool names -> POLICY_BLOCK
	allowPrefix []string        // benign read-shaped families -> Allow
}

func newMobilePolicy() mobilePolicy {
	return mobilePolicy{
		// A dangerous on-device action an agent might propose from one coarse
		// grant: send an SMS / place a call / wipe data. Denied at the floor, so
		// it never reaches the Android Intent / App Intent dispatcher.
		deny: map[string]bool{
			"send_sms":       true,
			"place_call":     true,
			"delete_contact": true,
			"factory_reset":  true,
		},
		// The read-only family is safe to continue: query battery, read the
		// calendar, get the location once. Prefix-matched like the real floor.
		allowPrefix: []string{"get_", "read_", "list_", "query_"},
	}
}

// Adjudicate is one link in the fold. A provable refusal denies; a benign
// read-shaped call allows; anything else Defers (fail-to-abstain), leaving the
// fold to resolve it.
func (m mobilePolicy) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil || c.Tool == "" {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonMalformed, By: "mobile/floor"}
	}
	if m.deny[c.Tool] {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "mobile/floor"}
	}
	for _, p := range m.allowPrefix {
		if strings.HasPrefix(c.Tool, p) {
			return abi.Verdict{Kind: abi.VerdictAllow, By: "mobile/floor"}
		}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "mobile/floor"}
}

func (mobilePolicy) Caps() []abi.Capability { return nil }

// fold resolves one adjudicator's verdict the way the kernel's chain does at the
// tail: an explicit Allow/Deny stands; a Defer (nothing affirmatively allowed
// it) fails CLOSED to DEFAULT_DENY. This is fak's floor invariant — a call the
// policy did not recognize does not reach the intent dispatcher by default.
func fold(v abi.Verdict, tool string) Decision {
	switch v.Kind {
	case abi.VerdictAllow:
		return Decision{Tool: tool, Allow: true, Verdict: "ALLOW", By: v.By}
	case abi.VerdictDeny:
		return Decision{Tool: tool, Allow: false, Verdict: "DENY", Reason: abi.ReasonName(v.Reason), By: v.By}
	default: // Defer / anything unrecognized -> fail closed
		return Decision{Tool: tool, Allow: false, Verdict: "DEFAULT_DENY", Reason: abi.ReasonName(abi.ReasonDefaultDeny), By: "mobile/fold"}
	}
}

// floor is the process-wide reference monitor the FFI boundary consults. A real
// host would load a policy manifest; the reference sample pins one.
var floor = newMobilePolicy()

// Decide is the FFI core: it takes a proposed tool call as JSON
// (`{"tool":"send_sms","args":{...}}`), routes it through the floor + fold, and
// returns the Decision. Pure and deterministic — no engine, network, or
// randomness — so the same input always yields the same verdict. The C shim
// (libfakmobile.go) is a thin JSON-string wrapper over exactly this function.
func Decide(toolCallJSON string) Decision {
	var call struct {
		Tool string `json:"tool"`
	}
	if err := json.Unmarshal([]byte(toolCallJSON), &call); err != nil {
		return Decision{Allow: false, Verdict: "DENY", Reason: abi.ReasonName(abi.ReasonMalformed), By: "mobile/ffi"}
	}
	c := &abi.ToolCall{Tool: call.Tool}
	return fold(floor.Adjudicate(context.Background(), c), call.Tool)
}
