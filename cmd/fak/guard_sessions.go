package main

// guard_sessions.go — `fak guard sessions`, the OPERATOR query surface over the local
// guard-session INDEX (internal/guardsessions). Every `fak guard` launch appends one row
// to a durable JSONL index; this lists the recorded sessions and resolves a short prefix
// (of the handle OR the trace id) to the one session it names, so an operator can reference
// a specific guard session without scraping Slack or grepping the outbox:
//
//	fak guard sessions                 # list recorded guard sessions (newest first)
//	fak guard sessions <prefix>        # resolve a handle/trace prefix to one session
//	fak guard sessions --json          # machine-readable list (or the resolved row)
//
// It is peeled off in cmdGuard before the wrap-a-command flag parse (like `fak guard
// allow`), so a bare leading `sessions` is the query surface and never a program to wrap.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

func cmdGuardSessions(argv []string) { os.Exit(runGuardSessions(os.Stdout, os.Stderr, argv)) }

// recordGuardSessionIndex appends this guard session to the local index under the fleet
// registry dir and returns its short handle (or "" if the append failed). Called once at
// guard start; best-effort by contract — a failed append must never block the launch, so
// the caller only uses the handle for the startup report. The cwd is the process cwd (the
// dir the guarded agent runs from), part of the provenance an operator wants when picking
// a session.
func recordGuardSessionIndex(traceID, agent, auditPath, nonce string, startedAt time.Time) string {
	regDir := resolveSweepRegDir("")
	cwd, _ := os.Getwd()
	row := guardsessions.NewRow(traceID, agent, os.Getpid(), cwd, auditPath, nonce, startedAt)
	if err := guardsessions.Record(regDir, row); err != nil {
		return ""
	}
	return row.Handle
}

// runGuardSessions is the testable core: exit 0 ok, 1 runtime, 2 usage, 3 an ambiguous or
// unmatched resolve (so a script can gate on a clean single-session resolution).
func runGuardSessions(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("guard sessions", flag.ContinueOnError)
	fs.SetOutput(stderr)
	regDirFlag := fs.String("reg-dir", "", "registry dir holding guard_sessions.jsonl (default: $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
	asJSON := fs.Bool("json", false, "emit the session list (or the resolved row) as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	regDir := resolveSweepRegDir(*regDirFlag)
	rows := guardsessions.Load(regDir)

	// A positional query resolves one session by handle/trace prefix.
	if fs.NArg() > 0 {
		return renderGuardSessionResolve(stdout, stderr, regDir, rows, fs.Arg(0), *asJSON)
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":   "fak.guard-sessions.v1",
			"reg_dir":  regDir,
			"sessions": rows,
		}, "fak guard sessions")
	}
	if len(rows) == 0 {
		fmt.Fprintf(stdout, "no recorded guard sessions in %s — start one with `fak guard -- <agent>`\n",
			guardsessions.IndexPath(regDir))
		return 0
	}
	// A5 (#4116): join each session's guard trace to its transcript UUID (the A1/A2 store),
	// so the operator surface names the id `claude --resume` takes, not just the trace.
	_, uuidByTrace := resume.LoadIdentity(regDir)
	renderGuardSessionTable(stdout, rows, uuidByTrace)
	fmt.Fprintf(stdout, "\nreference one with `fak guard sessions <handle-or-trace-prefix>` (index: %s)\n",
		guardsessions.IndexPath(regDir))
	return 0
}

// renderGuardSessionResolve resolves one query to a session, printing the matched row or
// the ambiguity. Exit 1 no match, 3 ambiguous, 0 a unique resolution.
func renderGuardSessionResolve(stdout, stderr io.Writer, regDir string, rows []guardsessions.Row, query string, asJSON bool) int {
	res := guardsessions.Resolve(rows, query)
	_, uuidByTrace := resume.LoadIdentity(regDir) // trace -> transcript UUID for the text renders
	switch {
	case res.Matched == 0:
		if asJSON {
			_ = encodeJSONOrFail(stdout, stderr, map[string]any{"query": query, "matched": 0}, "fak guard sessions")
		} else {
			fmt.Fprintf(stderr, "fak guard sessions: no guard session matches %q in %s\n", query, guardsessions.IndexPath(regDir))
		}
		return 1
	case res.Matched > 1:
		if asJSON {
			_ = encodeJSONOrFail(stdout, stderr, map[string]any{
				"query": query, "matched": res.Matched, "candidates": res.Candidates,
			}, "fak guard sessions")
		} else {
			fmt.Fprintf(stderr, "fak guard sessions: %q is ambiguous — %d sessions match:\n", query, res.Matched)
			renderGuardSessionTable(stderr, res.Candidates, uuidByTrace)
			fmt.Fprintln(stderr, "  narrow the prefix to name exactly one")
		}
		return 3
	default:
		if asJSON {
			return encodeJSONOrFail(stdout, stderr, res.Row, "fak guard sessions")
		}
		renderGuardSessionRow(stdout, res.Row, uuidByTrace)
		return 0
	}
}

// renderGuardSessionTable prints the sessions as an aligned, scannable table: the short
// handle first (the thing to reference), then the trace, the joined transcript UUID (the id
// `claude --resume` takes; a dash when the identity store has no join yet), agent, pid,
// start, and cwd.
func renderGuardSessionTable(w io.Writer, rows []guardsessions.Row, uuidByTrace map[string]string) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "HANDLE\tTRACE\tUUID\tAGENT\tPID\tSTARTED\tCWD\n")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%s\t%s\n",
			r.Handle, orDash(r.TraceID), orDash(uuidByTrace[r.TraceID]), orDash(r.Agent), r.PID, orDash(r.StartedAt), orDash(r.CWD))
	}
	_ = tw.Flush()
}

// renderGuardSessionRow prints one resolved session's full provenance (the detail an
// operator wants after naming it), including the transcript UUID joined from the session's
// trace (a dash when the identity store has no join yet).
func renderGuardSessionRow(w io.Writer, r guardsessions.Row, uuidByTrace map[string]string) {
	fmt.Fprintf(w, "guard session %s\n", r.Handle)
	fmt.Fprintf(w, "  trace_id: %s\n", orDash(r.TraceID))
	fmt.Fprintf(w, "  uuid:     %s\n", orDash(uuidByTrace[r.TraceID]))
	fmt.Fprintf(w, "  agent:    %s\n", orDash(r.Agent))
	fmt.Fprintf(w, "  pid:      %d\n", r.PID)
	fmt.Fprintf(w, "  started:  %s\n", orDash(r.StartedAt))
	fmt.Fprintf(w, "  cwd:      %s\n", orDash(r.CWD))
	fmt.Fprintf(w, "  audit:    %s\n", orDash(r.AuditPath))
	if r.Nonce != "" {
		fmt.Fprintf(w, "  slack_thread_id: %s\n", r.Nonce)
	}
}
