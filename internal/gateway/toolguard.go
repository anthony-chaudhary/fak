package gateway

import (
	"fmt"
	"strings"
)

// The tool-guard posture selects whether the gateway INTERCEPTS the tool plane
// on served turns: proposed tool calls (adjudication, deny rungs, inline-serve
// probes) and inbound client tool results (the kernel result-admission chain —
// context-MMU quarantine, normgate secret/PII masking, IFC taint stamping).
//
// The DEFAULT for a raw inference router is OFF: the router is a pass-through —
// every proposed tool call reaches the client unadjudicated and every inbound
// tool result reaches the model byte-identical, with zero guard processing on
// the hot path (the interception seams are not even entered). The guard is an
// explicit opt-in: `fak serve --guard` / FAK_GUARD=on, or the harness products
// (`fak guard`, and every caller that leaves Config.ToolGuard empty, which
// resolves ON so the library's historical fail-closed floor is unchanged).
//
// This is the deliberate flip of the router's product default (2026-11-10):
// credential/PII masking of tool results (the "[fak] masked N credential-shaped
// spans …" note) and every other content interception ran BY DEFAULT on the
// served path before it; raw mode is now the default and the guard is armed
// by name.

const (
	// ToolGuardOn arms the interception floor (the historical behavior).
	ToolGuardOn = "on"
	// ToolGuardOff is raw pass-through: no adjudication, no result admission,
	// no masking, zero interception processing.
	ToolGuardOff = "off"
)

// ParseToolGuardMode validates one explicit ToolGuard posture token. Empty
// resolves to ON — the library/harness default — so every existing caller
// (fak guard, tests, demos) keeps the armed floor byte-for-byte. Accepted
// spellings mirror the repo's env-bool idiom.
func ParseToolGuardMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return ToolGuardOn, nil
	case ToolGuardOn, "1", "true", "yes", "enabled":
		return ToolGuardOn, nil
	case ToolGuardOff, "0", "false", "no", "disabled", "raw":
		return ToolGuardOff, nil
	}
	return "", fmt.Errorf("unknown tool guard mode %q (want \"on\" or \"off\")", raw)
}
