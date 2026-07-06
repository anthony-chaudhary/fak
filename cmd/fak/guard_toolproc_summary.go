package main

// guard_toolproc_summary.go — the `fak guard` exit-summary line for the tool
// process table (issue #2445, part of the harness-native program #2387).
//
// A guarded session already installs the toolproc observation hooks, which
// append spawn/exit/session_end rows to the workspace journal
// (.fak/toolproc/journal.jsonl, guard_toolproc_hooks.go). This is the read side
// at session exit: fold that journal and, when any event-stream MONITOR went
// silent past its cadence, say so on ONE line. A stalled monitor is the
// doctrine's whole point — silence looked identical to progress until it folded
// to TOOL_HEARTBEAT_STALLED with kill advice — so the surface every session
// already prints names the count instead of leaving it in the journal unread.
//
// Best-effort by contract, exactly like guardHookLatencySummaryLine: no
// journal, an unreadable/oversized/malformed journal, or a table with nothing
// notable all return "" — the exit summary never grows an error line for an
// observability nicety.

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// guardToolprocSummaryLine reads the workspace toolproc journal and folds it at
// nowMS for the exit summary. It returns a one-line count of RUNNING procs and,
// prominently, STALLED MONITORS — or "" when the journal is absent/unreadable or
// the table has no running procs and no stalled monitors to report.
func guardToolprocSummaryLine(journalPath string, nowMS int64) string {
	f, err := os.Open(journalPath)
	if err != nil {
		return ""
	}
	defer f.Close()
	events, err := toolproc.ParseEvents(f)
	if err != nil {
		return ""
	}
	tab, err := toolproc.Fold(events, nowMS, toolproc.Config{})
	if err != nil {
		return ""
	}
	c := tab.Counts
	if c.Running == 0 && c.StalledMonitors == 0 {
		return "" // nothing outstanding — stay quiet rather than print a vacuous zero row
	}
	var b strings.Builder
	b.WriteString(guardSection("tool processes"))
	b.WriteString(guardRow("running", fmt.Sprintf("%d", c.Running)))
	b.WriteString(guardRow("stalled monitors", fmt.Sprintf("%d", c.StalledMonitors)))
	b.WriteString(guardNote("a stalled monitor is silence folded to TOOL_HEARTBEAT_STALLED with kill advice; `fak toolproc ps --events " + journalPath + "`"))
	return b.String()
}

// guardToolprocSummary is the exit-summary caller: it locates the workspace
// journal the observation hooks feed and renders the line at wall-clock now.
func guardToolprocSummary(now time.Time) string {
	return guardToolprocSummaryLine(guardToolprocJournalRel, now.UnixMilli())
}
