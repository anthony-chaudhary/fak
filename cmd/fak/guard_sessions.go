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
//
// It also holds the PRODUCER side of that index: recordGuardSessionIndex appends the launch
// row, and publishGuardSessionGateway (#5400) re-records it with the session's live gateway
// URL + read-scoped bearer once the listener is serving — the stamp `fak session status`
// and `fak cachevalue census` read back cross-process.

import (
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

func cmdGuardSessions(argv []string) { os.Exit(runGuardSessions(os.Stdout, os.Stderr, argv)) }

var guardSessionIndexRecorder = recordGuardSessionIndex

func maybeRecordGuardSessionIndex(audit *journal.Journal, traceID string, command []string, startedAt time.Time) string {
	if audit == nil || len(command) == 0 || strings.TrimSpace(traceID) == "" {
		return ""
	}
	return guardSessionIndexRecorder(traceID, command[0], audit.Path(), newGuardLaunchNonce(), startedAt)
}

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
	// #5400: keep the recorded row ADDRESSABLE so the post-bind stamp below can re-record
	// THIS row once the gateway is serving. The recorder closure carries the same regDir the
	// launch append used, so the republish lands in the file the launch row is in.
	trackGuardSessionGatewayPublish(&row, func(r guardsessions.Row) error { return guardsessions.Record(regDir, r) })
	return row.Handle
}

// guardSessionGatewayPublication is one row THIS guard process recorded at launch, paired
// with the recorder that wrote it. The launch record necessarily runs BEFORE the listener
// binds (the handle is part of the startup report and the crash-relaunch row must exist
// before the child starts), so the gateway address cannot be known yet. The index is
// append-only and folds to the LATEST row per handle, so re-recording the SAME row with the
// address stamped supersedes the launch row rather than inventing a second session.
type guardSessionGatewayPublication struct {
	row    *guardsessions.Row
	record func(guardsessions.Row) error
}

// guardSessionGatewayPublications accumulates the rows this process recorded (the plain
// index row and, on an operator terminal, the interactive relaunch row — which mirrors into
// the machine registry through its own recorder).
var guardSessionGatewayPublications []guardSessionGatewayPublication

// trackGuardSessionGatewayPublish registers a recorded row for the post-bind gateway stamp.
// The row is held BY POINTER so the stamp is visible to the caller's later writes too — an
// interactive session's clean-exit tombstone then carries the same gateway fields its live
// row did, instead of silently dropping them.
func trackGuardSessionGatewayPublish(row *guardsessions.Row, record func(guardsessions.Row) error) {
	if row == nil || record == nil {
		return
	}
	guardSessionGatewayPublications = append(guardSessionGatewayPublications, guardSessionGatewayPublication{row: row, record: record})
}

// publishGuardSessionGateway is the PRODUCER half of gateway discovery (#5400): it stamps
// the live gateway URL and its read-scoped bearer onto every row this guard process
// recorded and re-records each one, so a second process can reach this session's status
// endpoint from the index alone with no prior port knowledge. Called once, from the guard
// main flow, at the first instant the address is both known AND actually serving.
//
// An empty url is a NO-OP by contract: a guard that binds no gateway keeps its recorded row
// exactly as it was, with both fields omitted (they are omitempty — their absence is a legal
// row shape, and `fak session status` reports it as "published nothing", not as an error).
// Best-effort like the launch append: it returns the append errors for a diagnostic line and
// never fails the launch.
func publishGuardSessionGateway(url, bearer string) (published int, errs []error) {
	if strings.TrimSpace(url) == "" {
		return 0, nil
	}
	for _, pub := range guardSessionGatewayPublications {
		*pub.row = pub.row.WithGateway(url, bearer)
		if err := pub.record(*pub.row); err != nil {
			errs = append(errs, err)
			continue
		}
		published++
	}
	return published, errs
}

// newGuardReadBearer mints THIS launch's read-scoped observability token — the bearer the
// session publishes next to its gateway URL. The gateway accepts it ONLY on the read-only
// paths (/healthz, /debug/vars, /metrics; internal/gateway/http.go withAuth consults it
// after the full-strength credential has already failed and only on those routes), so the
// token that lands in the world-readable session index can read status and can never
// control the session. 128 bits of crypto/rand, fresh per launch. An entropy failure
// returns "" — the session then publishes its URL with NO bearer, which is strictly better
// than publishing a guessable one.
func newGuardReadBearer() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return ""
	}
	return "fakread-" + hex.EncodeToString(b[:])
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
		_, uuidByTrace := resume.LoadIdentity(regDir)
		projected := projectRows(rows, func(row guardsessions.Row) guardSessionJSONRow {
			return guardSessionJSONRow{Row: withoutPublishedBearer(row), TranscriptUUID: uuidByTrace[row.TraceID]}
		})
		return encodeSessionListingJSON(stdout, stderr, "fak.guard-sessions.v1", regDir, projected, "fak guard sessions")
	}
	if len(rows) == 0 {
		reportNoGuardSessions(stdout, regDir)
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

// encodeSessionListingJSON emits the {schema, reg_dir, sessions} envelope that BOTH session
// listings print under --json. `fak guard sessions` and `fak session ls` read the same
// on-disk index and differ only in the schema string, the projected row type, and the stderr
// label; the envelope shape is one contract and stays one construction site.
func encodeSessionListingJSON(stdout, stderr io.Writer, schema, regDir string, sessions any, label string) int {
	return encodeJSONOrFail(stdout, stderr, map[string]any{
		"schema":   schema,
		"reg_dir":  regDir,
		"sessions": sessions,
	}, label)
}

// reportNoGuardSessions prints the shared empty-listing line for those same two commands.
// They read the same index, so an empty listing must point the operator at the same path and
// the same way to start one.
func reportNoGuardSessions(stdout io.Writer, regDir string) {
	fmt.Fprintf(stdout, "no recorded guard sessions in %s — start one with `fak guard -- <agent>`\n",
		guardsessions.IndexPath(regDir))
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
			return encodeJSONOrFail(stdout, stderr, withoutPublishedBearer(res.Row), "fak guard sessions")
		}
		renderGuardSessionRow(stdout, res.Row, uuidByTrace)
		return 0
	}
}

// renderGuardSessionTable prints the sessions as an aligned, scannable table: the short
// handle first (the thing to reference), then the trace, the joined transcript UUID (the id
// `claude --resume` takes; a dash when the identity store has no join yet), agent, pid,
// start, and cwd.
type guardSessionJSONRow struct {
	guardsessions.Row
	TranscriptUUID string `json:"transcript_uuid,omitempty"`
}

// withoutPublishedBearer strips the row's read-scoped gateway token before it is RENDERED.
// Now that the publish half exists (#5400) the recorded row actually carries a live
// credential, so every surface that prints a row has to decide whether to echo it; `fak
// session ls` already answers no (session_query.go), and this is the same answer for the
// sibling query. The gateway_url stays — that is the discovery half, and it is useless to a
// caller who cannot authenticate. A consumer that needs the token reads the index file
// itself, which is where the credential's actual blast radius is scoped and where it is
// already read-only by construction.
func withoutPublishedBearer(r guardsessions.Row) guardsessions.Row {
	r.Bearer = ""
	return r
}

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
