package codetools

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// refusal.go — the toolset's CLOSED refusal vocabulary.
//
// A refusal is a value, never a panic and never a bare error string: the model that
// proposed the call has to be able to act on it, and the kernel has to be able to map it
// onto the abi reason vocabulary the adjudication chain speaks. So each refusal carries
// BOTH — a codetools-local Code naming precisely what was wrong, and the abi.ReasonCode
// the rung denies with. Two layers because the abi vocabulary is deliberately small
// (POLICY_BLOCK covers "a policy said no" for every tool in the system) while a coding
// tool's failure modes are specific enough that collapsing them would throw away the one
// thing the next turn needs: which fix to apply.

// The closed Code set. Anything a caller can hit from a well-formed proposal is here;
// there is no "other".
const (
	CodePathEscape    = "PATH_ESCAPE"    // the canonical path resolves outside the workspace root
	CodeSymlinkEscape = "SYMLINK_ESCAPE" // an in-tree symlink resolves outside the workspace root
	CodeMalformed     = "MALFORMED"      // missing/unknown/ill-typed argument
	CodeDefaultDeny   = "DEFAULT_DENY"   // no policy affirmatively admitted the tool (fail-closed)
	CodeNotFound      = "NOT_FOUND"      // the target path does not exist
	CodeIsDir         = "IS_DIR"         // the target is a directory where a file was required
	CodeCanceled      = "CANCELED"       // the caller's context was canceled
	CodeCacheScope    = "CACHE_SCOPE"    // the call's cache hints contradict the tool's write shape
	CodeIO            = "IO_ERROR"       // the filesystem refused the operation
)

// Refusal is one denied or failed operation: the closed Code, the abi reason the rung
// denies with, and a bounded human detail. Detail never carries file CONTENT — only the
// operand spelling the caller already knows — so a refusal cannot become an exfiltration
// channel for a file the policy just refused to read.
type Refusal struct {
	Code   string         `json:"code"`
	Reason abi.ReasonCode `json:"-"`
	Detail string         `json:"detail,omitempty"`
	Tool   string         `json:"tool,omitempty"`
}

// Error makes a Refusal usable where an error is expected. The Code leads because it is
// the part a caller branches on.
func (r *Refusal) Error() string {
	if r == nil {
		return ""
	}
	if r.Detail == "" {
		return r.Code
	}
	return r.Code + ": " + r.Detail
}

// JSON renders the refusal as the engine's error payload. The shape is stable
// ({"error":{"code","detail","tool"}}) so a loop can parse the code without regexing
// prose, and mirrors readengine's deny-as-value convention: a refused call returns a
// Status=Error RESULT, never a transport failure.
func (r *Refusal) JSON() []byte {
	b, err := json.Marshal(map[string]any{"error": r})
	if err != nil {
		// Marshal of this shape cannot fail; the fallback exists so the engine path is
		// total by construction rather than by argument.
		return []byte(`{"error":{"code":"IO_ERROR"}}`)
	}
	return b
}

// refuse builds a Refusal with the abi reason that matches its code. Centralized so a
// new code cannot silently land with a defaulted (and wrongly permissive) reason.
func refuse(code, detail string) *Refusal {
	return &Refusal{Code: code, Reason: reasonFor(code), Detail: detail}
}

// reasonFor maps the local vocabulary onto the abi reason the kernel's deny-as-value and
// disposition logic understands. Path/symlink escapes are POLICY_BLOCK (a call-shape
// refusal the model must not retry verbatim); a shape/argument fault is MALFORMED
// (model-fixable). Codes that describe a FAILED but admitted operation (NOT_FOUND, IS_DIR,
// CANCELED, IO_ERROR) map to ReasonNone: they are engine results, not adjudication
// refusals, and must never be reported as a kernel denial.
func reasonFor(code string) abi.ReasonCode {
	switch code {
	case CodePathEscape, CodeSymlinkEscape:
		return abi.ReasonPolicyBlock
	case CodeDefaultDeny:
		return abi.ReasonDefaultDeny
	case CodeMalformed:
		return abi.ReasonMalformed
	case CodeCacheScope:
		return abi.ReasonTrustViolation
	default:
		return abi.ReasonNone
	}
}
