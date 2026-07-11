package policy

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// MountRule is one subtree in the T1 mount view (issue #2577). Path is a
// repo-relative prefix that is IN VIEW (the whole tree rooted there exists to
// the agent); Mode is "ro" (read-only) or "rw" (read-write, the default when
// omitted). A path under a "ro" rule may be read but not written; a path under
// no rule at all is outside the view and does not exist.
//
// This mirrors the overlayfs / Plan 9 per-process-namespace idea named in the
// issue's prior art: a set of mounted subtrees, each with a mode, composed into
// the single view a session sees.
type MountRule struct {
	Path string `json:"path"`
	Mode string `json:"mode,omitempty"`
}

// readOnly reports whether this rule mounts its subtree read-only. Only the
// exact token "ro" (case-insensitive) restricts; every other value — "rw", "",
// or an unknown token — is treated as read-write. Mode validation at load
// (rejecting an unknown token fail-loud) is the named promotion step; until
// then the classifier fails OPEN on mode only, never on visibility.
func (r MountRule) readOnly() bool {
	return strings.EqualFold(strings.TrimSpace(r.Mode), "ro")
}

// MountViewRefusal is the T1 reference monitor over paths — the offline,
// no-model-in-the-loop admission kernel behind issue #2577's witness. Given a
// policy's mount view and a target path for an operation of the given write
// shape, it returns the closed-vocabulary refusal the access earns, or ok=true
// when the access is within the view.
//
// Deny-by-default, the same shape the capability floor has over tool names:
//   - EMPTY view  → ok (feature off; no view configured means every path is
//     visible, backward-compatible).
//   - target matches NO rule → DEFAULT_DENY (outside the view: does not exist;
//     nothing affirmatively permitted it).
//   - target in a read-only subtree + write shape → POLICY_BLOCK (an explicit
//     rule denied the write).
//   - otherwise → ok (in view, and the mode permits the shape).
//
// It is a pure function of (view, target, write): no I/O, no clock, no model —
// so a `fak preflight`-style check can replay it deterministically.
func MountViewRefusal(view []MountRule, target string, write bool) (abi.ReasonCode, bool) {
	if len(view) == 0 {
		return abi.ReasonNone, true // no view configured → unrestricted (feature off)
	}
	t := normMountPath(target)
	for _, r := range view {
		if mountPathInScope(t, normMountPath(r.Path)) {
			if write && r.readOnly() {
				return abi.ReasonPolicyBlock, false // in view but read-only; a write is refused
			}
			return abi.ReasonNone, true // in view, mode permits the shape
		}
	}
	return abi.ReasonDefaultDeny, false // matches no rule → outside the view
}

// normMountPath canonicalizes a repo-relative logical path for prefix matching.
// It works on the logical (forward-slash) path space, NOT the host filesystem,
// so the view is identical on Windows and POSIX: backslashes fold to slashes, a
// leading "./" is stripped, and a trailing "/" is trimmed. It deliberately does
// NOT touch "..": an escape attempt is left visible so it matches no in-view
// prefix and falls to DEFAULT_DENY.
func normMountPath(p string) string {
	p = strings.ReplaceAll(p, "\\", "/")
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimSuffix(p, "/")
	return p
}

// mountPathInScope reports whether target sits at or under prefix. An empty
// prefix scopes nothing (fail-closed — a malformed empty rule path can never
// widen the view); "." is the repo root and scopes the whole tree.
func mountPathInScope(target, prefix string) bool {
	if prefix == "" {
		return false
	}
	if prefix == "." {
		return true
	}
	return target == prefix || strings.HasPrefix(target, prefix+"/")
}
