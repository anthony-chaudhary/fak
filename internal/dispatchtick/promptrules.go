package dispatchtick

import (
	"fmt"
	"strings"
)

// promptrules.go — the worker-guidance blocks as DATA instead of prose (#3220).
//
// The `how to work it:` and `git laws:` blocks used to be long free paragraphs baked
// into the RenderIssuePrompt format string. That made them (a) hard for a worker to
// scan and (b) unobservable: nothing could tell whether a given rule still mapped to a
// live, enforced gate. A rule that named a retired refusal token or a renamed verb read
// exactly like one that still bit.
//
// Each rule now carries the WITNESS that proves it is still enforced, so the guidance is
// self-checking: promptrules_test.go folds every witness against the same authoritative
// registries internal/promptlint lints a rendered prompt with (the dos.toml closed reason
// set, safecommit's + pythongate's reason vocabularies, and the live cmd/fak dispatch verb
// catalog). Retire a token or rename a verb and the rule that cites it reds — the drift is
// caught at the source instead of by a confused worker.

// PromptRule is one worker-guidance rule as structured data: a stable ID, ONE short
// Imperative, and the Witness that enforces it.
//
// Witness is deliberately one of exactly two observable shapes, because those are the
// two dimensions promptlint can vouch for:
//   - an UPPER_SNAKE refusal token a gate stamps when the rule is broken (OFF_TRUNK,
//     PATHSPEC_RACE, ...) — checked against the live reason registries;
//   - a runnable command that grades the rule (`dos commit-audit`, `fak dispatch
//     issue-smallness-lint`) — its `fak <verb>` head checked against the live catalog.
//
// A rule whose witness is neither (an unfalsifiable "the hook refuses it") is exactly the
// prose this replaces: it cannot be kept honest, so it does not belong in the set.
type PromptRule struct {
	ID         string `json:"id"`
	Imperative string `json:"imperative"`
	Witness    string `json:"witness"`
}

// WorkRules is the `how to work it:` rule set for one issue in one lane. Content is the
// pre-#3220 prose, condensed to one imperative per rule. New policy rules remain
// explicit structured rows with their own executable witnesses.
func WorkRules(issue int, lane string) []PromptRule {
	return []PromptRule{{
		ID: "lane-lease",
		Imperative: fmt.Sprintf("Take the lease for the lane whose files you will actually edit "+
			"before touching them (`%s` here) and never --force onto a lane a LIVE holder owns. "+
			"A refusal here is a pause, not your stop condition: acquire the lane that matches "+
			"your files, or - if the holder is provably dead, meaning its pid is gone AND its "+
			"lease is long past heartbeat - reap it by naming that dead holder in a release, "+
			"then re-acquire", lane),
		Witness: fmt.Sprintf("dos lease-lane acquire --lane %s --owner <you>", lane),
	}, {
		// The single highest-value rule in this set: forensics over 111 zero-commit worker
		// sessions found 16% ended the instant a turn hit a kernel refusal, because the old
		// lane-lease rule's "honor a REFUSE (pick nothing and stop)" was the ONLY refusal
		// guidance in the packet - it actively taught quitting. Most refusals are pauses with
		// a named resolution; the discriminator below is the live one (a refusal states its
		// own fix), so it stays true as the reason registry grows.
		ID: "refusal-taxonomy",
		Imperative: "Treat a kernel refusal as a PAUSE, not your stop condition: re-read the " +
			"refusal's own stated fix and take the action it names - either re-propose the " +
			"SAME call byte-identical with only the confirm token added, or switch to the " +
			"compiled sidestep it points at (e.g. `fak sync push` for a gated push, which " +
			"needs no token at all). Only a refusal whose own stated fix names NO available " +
			"action is a wall you may stop at; everything else you work through",
		Witness: "REQUIRE_WITNESS",
	}, {
		ID: "smallest-change",
		Imperative: "Make the SMALLEST change that resolves the issue's actual ask - prefer one " +
			"leaf / one file, and on an epic you cannot land whole, land the smallest honest " +
			"increment and say in your report what remains",
		Witness: "fak dispatch issue-smallness-lint",
	}, {
		// The other prompt-shaped cause of the zero-commit sessions: the old packet gated the
		// FIRST commit behind a full green `make ci` at step 4-5, but the median killed worker
		// dies at 8-10 minutes and 27 of 37 killed workers already held edits - so edits were
		// GUARANTEED to strand. A checkpoint bar that is explicitly lower than the ship bar
		// (compiles + its own targeted test) converts that stranded work into landed work
		// without licensing broken commits.
		ID: "checkpoint-commit",
		Imperative: "Commit each working increment AS YOU REACH IT instead of saving every " +
			"commit for the end - your session can be killed at any moment and uncommitted " +
			"edits are simply lost. A checkpoint is honest as soon as it COMPILES and its own " +
			"targeted test passes; that bar is deliberately lower than the full `make ci` ship " +
			"gate, but it is never a licence for broken work. Say only what actually landed in " +
			"the subject and withhold the issue-closing `Fixes` line until the real fix is " +
			"green. Do NOT use a `wip(...)` subject - it forces the claim to none and lands " +
			"your work UNWITNESSED, which silently defeats the witness ledger",
		Witness: "fak commit --path",
	}, {
		ID: "gate-before-done",
		Imperative: "Run the gate yourself before claiming done: the lane's own test " +
			"(`go test ./... -count=1` for the touched package, or the doc/lint check the issue " +
			"names) - a claim with no gate run is not done",
		Witness: "LOOP_DONE_UNWITNESSED",
	}, {
		ID: "proof-by-default",
		Imperative: fmt.Sprintf("Match the proof to the defect: visual/TUI bugs need a captured "+
			"render or screenshot witness; logic/behavior bugs need a failing-before and "+
			"passing-after repro test; docs/operator changes need a lint, render, or exact-output "+
			"fixture; shipped/done claims need a witnessed commit tied to `#%d` and `(fak %s)`. "+
			"Do not stop on narrative alone", issue, lane),
		Witness: "LOOP_DONE_UNWITNESSED",
	}, {
		ID: "browser-display",
		Imperative: "Keep browser automation off the operator desktop. Use headless browser mode and " +
			"captured render/screenshot artifacts by default; do not launch or reuse visible Chrome " +
			"or Edge windows unless this issue explicitly requires an attended visual witness",
		Witness: "LOOP_DONE_UNWITNESSED",
	}, {
		ID: "no-delete",
		Imperative: "NEVER run a delete command (`rm`, `del`, `Remove-Item`, `rmdir`, " +
			"`git clean`) for ANY reason, including your own scratch files - leave them in " +
			"place and name them in your final report. You commit by explicit pathspec, so " +
			"untracked scratch can never contaminate your commit: deleting it is never " +
			"required, and a stray deletion swept into a commit reverts a peer's live work",
		Witness: "SPURIOUS_STAGED_DELETION",
	}, {
		// The honest escape hatch is LOAD-BEARING - a genuinely blocked worker must be able to
		// stop and say so - but it used to be the second clause of the packet's FIRST sentence,
		// which offered quitting as a co-equal opening move. It keeps its force here, gated on
		// having actually attempted the adaptation the blocker named.
		ID: "honest-bail",
		Imperative: "Stopping short is legitimate ONLY after you have tried the sanctioned " +
			"adaptation the blocker itself named and that also failed - then leave a durable " +
			"handoff: post a substantive `gh issue comment <N> --body ...` that gives the exact " +
			"gate you could not reach, what you already tried, and the smallest next checkable " +
			"step (a final chat report alone is not durable). Never stop on the first refusal, " +
			"and never let an unfinished attempt become a silent exit: promote and commit whatever " +
			"already works first; ignored scratch is not a deliverable",
		Witness: "LOOP_DONE_UNWITNESSED",
	}}
}

// GitLawRules is the `git laws:` rule set — the non-negotiables enforced below the agent.
// Content is the pre-#3220 prose plus the two laws the Python renderer already carried
// (message-before-pathspec, pathspec-race-recovery), which this set unifies so both
// renderers state the same laws.
func GitLawRules(issue int, lane, branch string) []PromptRule {
	return []PromptRule{{
		ID: "main-only",
		Imperative: fmt.Sprintf("Work on the configured development branch `%s` ONLY - never "+
			"branch, never a new worktree", branch),
		Witness: "OFF_TRUNK",
	}, {
		ID: "commit-by-path",
		Imperative: "Commit with `git commit -s -m \"<subject>\" -- <explicit paths>`: sign-off " +
			"(DCO), BY PATH only, staging just the files you wrote",
		Witness: "PATHSPEC_RACE",
	}, {
		ID: "no-blanket-add",
		Imperative: "NEVER `git add -A` on this shared multi-session tree - a blanket add steals " +
			"a sibling's in-flight files",
		Witness: "BARE_COMMIT_SWEEP",
	}, {
		ID: "message-before-pathspec",
		Imperative: "Put `-m`/`-F` BEFORE the `--` pathspec separator - git parses everything " +
			"after `--` as a pathspec, so an `-m` placed after it is silently read as a pathspec, " +
			"and a bare `git commit` with no message at all opens the interactive editor and " +
			"hangs headless",
		Witness: "INTERACTIVE_HANG",
	}, {
		ID: "issue-binding-trailer",
		Imperative: fmt.Sprintf("Reference `#%[1]d` in the subject AND end it with a `(fak %[2]s)` "+
			"trailer, lead with a verb (`fix(%[2]s): ... (#%[1]d) (fak %[2]s)`; use "+
			"add/fix/implement/test, NEVER a noun-led description) - miss either and the closure "+
			"auditor never closes your resolved issue", issue, lane),
		Witness: "dos commit-audit",
	}, {
		ID: "no-history-rewrite",
		Imperative: fmt.Sprintf("No push / tag / force-push / history-rewrite / reset / clean / "+
			"checkout-of-tracked-files - just commit on `%s`", branch),
		Witness: "NEVER_AMEND_SHARED",
	}, {
		ID: "pathspec-race-recovery",
		Imperative: "On a race refusal, preserve your working-tree edits, refresh from the current " +
			"trunk witness, and recommit by the same explicit paths - never recover by sweeping " +
			"the tree, and stop with the specific blocker if refresh exposes a conflict or a live " +
			"merge",
		Witness: "PATHSPEC_RACE",
	}}
}

// RenderPromptRules renders a rule set under its header as uniform bullets:
//
//	<header>
//	- <id>: <imperative> - witness `<witness>`
//
// One shape for every rule, so a worker scans the imperatives and an operator (or a
// freshness loop) reads the witness column straight off the rendered packet.
func RenderPromptRules(header string, rules []PromptRule) string {
	var b strings.Builder
	b.WriteString(header)
	for _, r := range rules {
		fmt.Fprintf(&b, "\n- %s: %s - witness `%s`", r.ID, r.Imperative, r.Witness)
	}
	return b.String()
}
