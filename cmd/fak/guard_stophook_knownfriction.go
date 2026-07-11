package main

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// guard_stophook_knownfriction.go -- the stop-time known-friction advisory.
//
// A recurring way an agentic workflow spins is a FALSE-INCOMPLETE signal: the
// agent's own work is diff-witnessed done, but a red it saw is a PRE-EXISTING
// shared blocker (a peer's in-flight WIP, a chronic-red trunk) that it did not
// cause and must not re-fix. The fleet already records those blockers in the
// knownbad ledger (`fak knownbad record`), and the dispatcher already short-
// circuits on them -- but nothing surfaces that knowledge at the ONE moment an
// agent is deciding whether it is done and might otherwise loop trying to "fix"
// a red that was never its own. This advisory closes that gap: at a CLEAN stop,
// if the trees this session MODIFIED intersect a LIVE known-bad signature, it
// prints a one-line heads-up naming the signature, so this agent (or the next
// one to resume the transcript) recognises the red as pre-existing and keeps
// moving instead of re-fixing it.
//
// Advisory-only, exactly like emitUnusedSubstrateAdvisory: it NEVER blocks the
// stop (no enforce rung) and fails open on every read error -- a missing ledger,
// an unreadable transcript, or a malformed row can only make it stay silent.

// knownFrictionFileTools are the tool names whose primary operand (streamTarget)
// is a real file path, so the tree it names is one this session actually WORKED
// in. Grep/Bash operands (a regex or a shell command line) are excluded: feeding
// those to the tree matcher is noise, not a worked tree. The advisory keys off
// trees the session MODIFIED -- the strongest "this is where I worked" signal --
// so a mere Read is excluded too; a pre-existing red in a tree you only browsed
// is not the loop this advisory is guarding against.
var knownFrictionFileTools = map[string]bool{
	"Edit": true, "Write": true, "MultiEdit": true, "NotebookEdit": true,
}

// knownFrictionTouchedTrees folds a session's tool-event stream into the set of
// repo-relative trees it modified: the file_path of each file-mutating tool
// event, made repo-relative against root and canonicalised through
// knownbad.NormalizeTree (which drops absolutes, escapes, and bare stars). The
// result is deduped and order-preserving. An absolute path outside root, or one
// that does not normalise, is dropped rather than matched loosely -- the same
// conservative posture as the rest of the advisory.
func knownFrictionTouchedTrees(events []trajctl.ToolEvent, root string) []string {
	seen := map[string]bool{}
	var out []string
	for _, ev := range events {
		if !knownFrictionFileTools[ev.Tool] {
			continue
		}
		t := strings.TrimSpace(ev.Target)
		if t == "" {
			continue
		}
		// Transcript file_paths are frequently absolute; NormalizeTree keeps an
		// absolute path verbatim (only a leading "/" is rejected, and a Windows
		// "C:/..." has none), so it would never intersect a repo-relative glob.
		// Make it repo-relative first; a path outside root yields a "../" escape
		// that NormalizeTree then drops.
		if filepath.IsAbs(t) && root != "" {
			if rel, err := filepath.Rel(root, t); err == nil {
				t = rel
			}
		}
		n := knownbad.NormalizeTree(filepath.ToSlash(t))
		if n == "" || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	return out
}

// knownFrictionAdvisoryCap bounds how many signatures the advisory line names
// before summarising the remainder as a trailing count, so the heads-up stays
// one readable line even against a busy ledger.
const knownFrictionAdvisoryCap = 3

// knownFrictionAdvisoryLine builds the one-line advisory for the LIVE known-bad
// signatures that intersect the session's touched trees, or "" when none do (so
// the caller stays silent). Pure: records, touched, and nowUnix are all data, so
// the message is deterministic under test.
func knownFrictionAdvisoryLine(records []knownbad.Record, touched []string, nowUnix int64) string {
	if len(touched) == 0 {
		return ""
	}
	matches := knownbad.Match(records, knownbad.Query{TreeGlobs: touched}, nowUnix)
	if len(matches) == 0 {
		return ""
	}
	named := make([]string, 0, knownFrictionAdvisoryCap)
	for i, m := range matches {
		if i >= knownFrictionAdvisoryCap {
			break
		}
		named = append(named, fmt.Sprintf("%s (reason=%s, %s)",
			m.Signature, m.ReasonClass, strings.Join(m.TreeGlobs, ",")))
	}
	more := ""
	if len(matches) > knownFrictionAdvisoryCap {
		more = fmt.Sprintf(" +%d more", len(matches)-knownFrictionAdvisoryCap)
	}
	return fmt.Sprintf("fak guard: heads-up -- a tree you edited has %d LIVE known-bad signature(s): %s%s. "+
		"A red there is a PRE-EXISTING shared blocker the fleet already recorded, not caused by your work -- "+
		"don't re-fix it; `fak knownbad match --tree <t>` to confirm, then park behind the fixer or pick up "+
		"other work. Advisory only -- the stop is allowed.",
		len(matches), strings.Join(named, "; "), more)
}

// emitKnownFrictionAdvisory reads the session transcript and the fleet knownbad
// ledger and prints knownFrictionAdvisoryLine when the session's modified trees
// intersect a live signature. Impure shell: the transcript and ledger reads and
// the clock live here; every one fails open (silent) so the advisory can never
// turn a clean stop into a block.
func emitKnownFrictionAdvisory(stderr io.Writer, transcriptPath string) {
	if transcriptPath == "" {
		return
	}
	events, err := trajctl.ReadToolStream(transcriptPath)
	if err != nil || len(events) == 0 {
		return
	}
	touched := knownFrictionTouchedTrees(events, repoRoot())
	if len(touched) == 0 {
		return
	}
	records, err := readKnownBadLedger(knownBadLedgerPath(""))
	if err != nil || len(records) == 0 {
		return
	}
	if line := knownFrictionAdvisoryLine(records, touched, time.Now().Unix()); line != "" {
		fmt.Fprintln(stderr, line)
	}
}
