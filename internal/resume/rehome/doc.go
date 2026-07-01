// Package rehome is the Go port of the interactive resume-resolver
// (tools/resume_resolver.py): it decides WHICH account `claude --resume <sid>`
// should run under, re-homing the transcript onto a healthy account when the
// owning account is rate-limited or blocked.
//
// It is a distinct concern from the parent internal/resume package, which prices
// the cold/warm prompt-cache COST of a resume. This package answers the earlier,
// mechanical question the operator hits first: `claude --resume <sid>` is
// CLAUDE_CONFIG_DIR + cwd scoped, so it only finds a conversation under
// <config>/projects/<sanitized-cwd>/<sid>.jsonl and only under the active
// CLAUDE_CONFIG_DIR. When the owning account is throttled, pinning to it yields a
// dead resume (every model call refused until the limit resets). This package
// closes that gap by locating the owner host-last/newest-mtime, checking its live
// availability, and deciding:
//
//   - owner available -> PIN to the owner (no copy; the safe default).
//   - owner blocked   -> REHOME: copy the transcript (+ its <sid>/ sidecar) onto
//     the least-loaded healthy Claude worker and pin there.
//   - no healthy acct -> PIN_BLOCKED: pin to the owner anyway (best effort).
//
// The decision core takes its live-availability facts, owner status, copy, and
// probe as injected inputs, so it is unit-testable with no real account dirs or
// registry — mirroring the injectable design of resume_resolver.resolve.
package rehome
