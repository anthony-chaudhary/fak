package main

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// commit_msg_advisory.go — the COMMIT_MSG subject advisory on the MUTATING `fak commit` path.
//
// Why here. The commit-msg git hook (tools/githooks/commit-msg) and its Go fast path
// (`fak hooks commit-msg`) already warn when a subject is not witness-gradeable, but the
// mutating `fak commit` — the path this shared tree mandates, and the one ~27 concurrent
// sessions actually ship through — computes the same verdict and discards it:
// deriveCommitMessageStamp lints the message only to lift `SuggestedSubject` off the report,
// dropping `Gradeable`/`GradeWhy` on the floor (gate_commitmsg.go says so in its own header:
// "the mutating `fak commit` ... never runs this gate"). The deterministic defects — a
// near-miss type (`feature:` -> `feat:`), an inflected verb (`added` -> `add`) — are already
// auto-healed by that suggestion. What survives is exactly the residue: a genuinely noun-led
// or unrecognized-verb description, which lands silently, ABSTAINs at the DOS commit-audit
// witness, and is immutable on a trunk nobody may amend. This advisory names that residue at
// the one moment it is still cheap: after the message is assembled and stamped, BEFORE any
// commit machinery runs, so it prints whether or not a later gate refuses.
//
// NON-BLOCKING BY CONSTRUCTION. renderCommitMsgAdvisory returns NOTHING — no error, no bool,
// no exit code. There is no value a caller could branch on, so it cannot become a refusal
// without a signature change, and commit_msg_advisory_test.go pins that signature (a compile
// error is a test failure). It also has no "block" mode: unlike the hook gates,
// FLEET_MSG_GUARD can only SILENCE this advisory (`off`), never escalate it. On a tree with
// this many concurrent committers a refusal here would wedge the entire fleet, and the hook
// layer already owns the escalatable enforcement.

// commitMsgAdvisoryHeadline prefixes every line group the advisory emits, so a session (or a
// log scraper) can recognize it and, critically, tell it apart from a refusal.
const commitMsgAdvisoryHeadline = "fak commit: COMMIT_MSG (advisory, non-blocking):"

// renderCommitMsgAdvisory warns — and only warns — when the FINAL commit message (post
// stamp-derivation) carries a subject the DOS commit-audit witness cannot grade. It writes to
// w (the caller passes stderr, never stdout, so `--json` output stays machine-parseable) and
// returns nothing at all.
//
// It re-lints the final message rather than reusing the pre-derivation report so the warning
// describes what will actually land: when the derivation healed the subject, the re-lint is
// clean and the advisory stays silent. LintCommitMessageWithOptions is used instead of the
// bare CommitMsgVerdict because it also honours the exempt subject classes — a `vX.Y.Z:`
// release anchor, a merge/revert, a bookkeeping snapshot — which are intentionally not
// `type(scope): <verb>` and must not be nagged.
func renderCommitMsgAdvisory(w io.Writer, message string, paths []string, root string) {
	mode, escaped := gateModeDefault("FLEET_MSG_GUARD", "FLEET_ALLOW_MSG", "warn")
	if mode == "off" || escaped {
		return
	}
	rep := hooks.LintCommitMessageWithOptions(message, paths, root, false)
	if rep.Gradeable {
		return
	}
	fmt.Fprintf(w, "%s subject is not witness-gradeable — %s\n", commitMsgAdvisoryHeadline, rep.GradeWhy)
	fmt.Fprintf(w, "  subject: %s\n", rep.Subject)
	fmt.Fprintln(w, "  shape:   type(scope): <verb> <what> (fak <leaf>)   e.g. fix(gateway): correct the same-tick ready check (fak gateway)")
	if rep.SuggestedSubject != "" {
		fmt.Fprintf(w, "  → try:   %s\n", rep.SuggestedSubject)
	}
	fmt.Fprintln(w, "  this commit is NOT blocked and proceeds normally; an ungradeable subject ABSTAINs at the commit-audit witness and cannot be amended once it lands on the shared trunk, so spend the fix on the NEXT one (`fak commit --preview -m ... --path ...` grades a subject for free).")
}
