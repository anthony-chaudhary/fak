// foregroundloop.go — the per-item foreground network-loop rung of the
// repo-guard PreToolUse hook (#4595).
//
// A shell `for x in <list>; do <network-call> $x; done` makes one network
// round trip per item. Run in the foreground, N sequential round trips
// serialize into a long wall that routinely blows the turn's time budget and
// the harness kills the command mid-loop — a wasted turn, exactly the class the
// #4595 audit measured (the `for … gh issue view …` fan-outs, and the
// `for … git fetch …` shape). The batched form (`gh issue list --json …`,
// `gh api --paginate …`) or backgrounding the loop finishes it in one turn.
//
// Curation principle (mirrors interactive.go / sleepwait.go): fire ONLY when it
// is provably the mistake. That is the reason this rung is scoped to `for`
// loops over a fixed list — a per-item fan-out that is genuinely batchable —
// and NOT to `while`/`until`, which are as often a legitimate retry/backoff or
// a streaming `while read` as a fan-out, so refusing them would deny benign
// forms. The network call is flagged only inside the loop BODY: the one
// list-generating call in the header (`for x in $(gh pr list …)`) is the
// batched form we are steering toward, never the mistake. Pure string work: no
// filesystem access, hermetically testable like the rest of the core.
package repoguard

import "strings"

// ReasonForegroundNetworkLoop is the structured advisory token for a foreground
// for-loop that makes one network call per iteration.
const ReasonForegroundNetworkLoop = "FOREGROUND_NETWORK_LOOP"

// networkVerbs are the always-network CLIs whose per-iteration invocation in a
// shell for-loop serializes into N round trips. Deliberately narrow to verbs
// that are network calls in EVERY invocation — no false positive on a local
// subcommand.
var networkVerbs = setOf("gh", "curl", "wget")

// gitNetworkSubs are the git subcommands that hit the network; a `git` in a
// loop body is a per-item round trip only for these (a `git log`/`git show`
// loop is local, not this rung's concern).
var gitNetworkSubs = setOf("fetch", "pull", "push", "clone", "ls-remote")

// ClassifyForegroundNetworkLoop returns FOREGROUND_NETWORK_LOOP advisories for a
// shell command.
func ClassifyForegroundNetworkLoop(command string) []Violation {
	return classifyForegroundNetworkLoop(command)
}

func classifyForegroundNetworkLoop(command string) []Violation {
	// A backgrounded loop (`… done &`) does not hold the turn open — the whole
	// point of the fix — so it is never the mistake this rung names.
	if hasBackgroundAmpersand(command) {
		return nil
	}
	var out []Violation
	inForBody := false        // between a for-loop's `do` and its `done`
	forHeaderPending := false // saw a `for` header, awaiting its `do`
	for _, seg := range splitSegments(command) {
		lead, operands, _ := tokenizeSegment(seg)
		switch lead {
		case "for":
			forHeaderPending = true
			continue
		case "done":
			inForBody = false
			continue
		case "do":
			if forHeaderPending {
				inForBody = true
				forHeaderPending = false
			}
		}
		if !inForBody {
			continue
		}
		// Reach the real verb past any leading flow keyword (`do gh …`).
		verb, verbOperands := lead, operands
		for shellFlowKeywords[verb] {
			verb, verbOperands, _ = stripEnvAndEnvVerb(verbOperands)
		}
		if op, ok := networkCall(verb, verbOperands); ok {
			out = append(out, Violation{
				Reason:   ReasonForegroundNetworkLoop,
				Op:       op,
				Target:   strings.TrimSpace(seg),
				Resolved: "<foreground-loop>",
				Why:      "a foreground for-loop runs this network call once per item — the serialized round trips routinely blow the turn's time budget and the command is killed mid-loop",
				Fix:      networkLoopFix(op),
			})
		}
	}
	return out
}

// networkCall reports whether (verb, operands) is a per-item network call and
// returns the display op for the advisory.
func networkCall(verb string, operands []string) (string, bool) {
	if networkVerbs[verb] {
		return verb, true
	}
	if verb == "git" {
		if sub, _ := gitSubcommand(operands); gitNetworkSubs[sub] {
			return "git " + sub, true
		}
	}
	return "", false
}

func networkLoopFix(op string) string {
	if op == "gh" {
		return "batch it into one request — gh issue list --json … / gh api --paginate … / gh search — or run the loop with Bash run_in_background"
	}
	return "batch the calls into one request where the CLI supports it, or run the loop with Bash run_in_background"
}

// renderNetworkLoopReason formats the advisory block for FOREGROUND_NETWORK_LOOP
// findings — the same shape as the other fix-bearing rungs.
func renderNetworkLoopReason(violations []Violation) string {
	parts := make([]string, len(violations))
	for i, v := range violations {
		parts[i] = v.Op + " in `" + v.Target + "` — fix: " + v.Fix
	}
	return ReasonForegroundNetworkLoop + ": a foreground loop makes one network call per item; the serialized round trips routinely blow the turn's time budget and the command is killed mid-loop. " +
		strings.Join(parts, "; ") +
		". To silence this per reason set FAK_REPO_GUARD_SEVERITY=" + ReasonForegroundNetworkLoop + "=record or =off; " +
		"the master switch FAK_REPO_GUARD=warn|off still overrides."
}
