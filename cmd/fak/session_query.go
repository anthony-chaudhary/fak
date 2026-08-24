package main

// session_query.go — cross-process guard-session DISCOVERY + raw status read (#3461).
//
// A running `fak guard` publishes its in-process gateway address and a READ-SCOPED
// bearer into the guard-session index (internal/guardsessions, guard_sessions.go). This
// file is the second process's side of that contract:
//
//	fak session ls                      # enumerate recorded guard sessions (pid-liveness
//	                                    # checked) with each one's gateway URL — NO prior
//	                                    # port knowledge needed
//	fak session status <handle|trace>   # resolve one session and GET its raw status
//	                                    # (<gateway_url>/debug/vars, Authorization: Bearer)
//
// Both verbs live under the existing `fak session` surface (session_cmd.go dispatches
// here when no explicit gateway address was given via --addr/$FAK_ADDR — an explicit
// address keeps the legacy single-gateway drive-state semantics byte-for-byte). Liveness
// is pid-based (procguard.CollectRelations): a dead-pid row still lists, marked stale,
// and `status` refuses to dial a stale row's gateway.

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// sessionQueryCollectRelations is the pid-liveness source (the same process-relations
// snapshot the fleet/dispatch surfaces join on). Var so tests inject a fixture.
var sessionQueryCollectRelations = procguard.CollectRelations

// sessionQueryHTTPClient dials a resolved session's own loopback gateway. Bounded so a
// wedged gateway cannot hang the CLI.
var sessionQueryHTTPClient = &http.Client{Timeout: 10 * time.Second}

// sessionAddrExplicit reports whether the caller named a gateway address (an explicit
// --addr flag or a non-empty $FAK_ADDR). When true, `fak session ls`/`status` keep their
// legacy single-gateway semantics; when false there is NO prior port knowledge, and the
// guard-session index answers instead (#3461).
func sessionAddrExplicit(fs *flag.FlagSet) bool {
	explicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "addr" {
			explicit = true
		}
	})
	return explicit || strings.TrimSpace(os.Getenv("FAK_ADDR")) != ""
}

// sessionIndexAlivePIDs snapshots the live pid set. ok=false means the process table
// could not be read at all — the caller then treats liveness as UNKNOWN (fail open to
// the fetch, whose success/failure is the ground truth) rather than mis-reporting every
// session stale.
func sessionIndexAlivePIDs() (map[int]bool, bool) {
	procs, errStr := sessionQueryCollectRelations()
	if len(procs) == 0 && errStr != "" {
		return nil, false
	}
	alive := make(map[int]bool, len(procs))
	for _, p := range procs {
		alive[p.PID] = true
	}
	return alive, true
}

// sessionIndexState classifies one row: "live" (pid running, not tombstoned), "ended"
// (clean-exit tombstone recorded), or "stale" (no tombstone but the pid is gone — a
// crash or a kill). With an unreadable process table a non-ended row reads "live"
// (liveness unknown; the status fetch is the real probe).
func sessionIndexState(r guardsessions.Row, alive map[int]bool, aliveOK bool) string {
	if strings.TrimSpace(r.EndedAt) != "" {
		return "ended"
	}
	if !aliveOK || alive[r.PID] {
		return "live"
	}
	return "stale"
}

// sessionIndexJSONRow is the machine shape of one listed session. The read bearer is
// REDACTED from the listing (a consumer that needs it reads the index file itself, or
// just calls `fak session status`, which presents it for you); everything else is the
// recorded row plus the computed liveness state.
type sessionIndexJSONRow struct {
	guardsessions.Row
	State string `json:"state"` // live | stale | ended
}

// runSessionIndexLS lists the recorded guard sessions from the index under regDir,
// newest first, pid-liveness checked. Exit 0 even when empty (an empty box is a valid
// answer, not an error).
func runSessionIndexLS(stdout, stderr io.Writer, regDir string, asJSON bool) int {
	rows := guardsessions.Load(regDir)
	alive, aliveOK := sessionIndexAlivePIDs()
	if asJSON {
		projected := projectRows(rows, func(r guardsessions.Row) sessionIndexJSONRow {
			return sessionIndexJSONRow{Row: withoutPublishedBearer(r), State: sessionIndexState(r, alive, aliveOK)}
		})
		return encodeSessionListingJSON(stdout, stderr, "fak.session-ls.v1", regDir, projected, "fak session ls")
	}
	if len(rows) == 0 {
		reportNoGuardSessions(stdout, regDir)
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "HANDLE\tTRACE\tAGENT\tPID\tSTATE\tAGE\tGATEWAY\tCWD\n")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\n",
			r.Handle, orDash(r.TraceID), orDash(r.Agent), r.PID,
			sessionIndexState(r, alive, aliveOK), sessionIndexAge(r), orDash(r.GatewayURL), orDash(r.CWD))
	}
	_ = tw.Flush()
	fmt.Fprintf(stdout, "\nread one with `fak session status <handle-or-trace-prefix>` (index: %s)\n",
		guardsessions.IndexPath(regDir))
	return 0
}

// sessionIndexAge renders how long ago the session started ("-" when unparseable).
func sessionIndexAge(r guardsessions.Row) string {
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(r.StartedAt))
	if err != nil {
		return "-"
	}
	return time.Since(t).Truncate(time.Second).String()
}

// runSessionIndexStatus reads ONE resolved session's raw status from its own published
// gateway: GET <gateway_url>/debug/vars with the row's read-scoped bearer. A non-live
// row reports its state (stale/ended) WITHOUT dialing anything — a dead session's
// recorded port may have been reused by an unrelated process. --json passes the fetched
// JSON body through verbatim (it is the gateway's own /debug/vars document).
func runSessionIndexStatus(stdout, stderr io.Writer, row guardsessions.Row, asJSON bool) int {
	alive, aliveOK := sessionIndexAlivePIDs()
	state := sessionIndexState(row, alive, aliveOK)
	if state != "live" {
		if asJSON {
			_ = encodeJSONOrFail(stdout, stderr, map[string]any{
				"schema": "fak.session-status.v1",
				"handle": row.Handle,
				"trace":  row.TraceID,
				"pid":    row.PID,
				"state":  state,
			}, "fak session status")
		} else {
			fmt.Fprintf(stderr, "fak session status: guard session %s is %s (pid %d) — not querying its recorded gateway\n",
				row.Handle, state, row.PID)
		}
		return 1
	}
	if strings.TrimSpace(row.GatewayURL) == "" {
		// Two DIFFERENT causes wear the same empty field, and the old text guessed the same
		// one ("recorded by an older fak?") for both — which was wrong on every row, because
		// until #5400 no fak build published the field at all. Split them on the row's own
		// start time against the publish epoch, so the version explanation is printed only
		// when the row genuinely predates the producer.
		if row.PredatesGatewayPublish() {
			fmt.Fprintf(stderr, "fak session status: guard session %s carries no gateway_url — it started %s, before fak published session gateways (%s), so no build could have recorded one. Cannot query it cross-process; relaunch it under `fak guard` to get a reachable row.\n",
				row.Handle, orDash(row.StartedAt), guardsessions.GatewayPublishEpoch.Format(time.RFC3339))
		} else {
			fmt.Fprintf(stderr, "fak session status: guard session %s published no gateway_url — this fak stamps one as soon as the session's gateway is serving, so this session bound no gateway (or its index re-record failed). Cannot query it cross-process; read it from the session's own terminal instead.\n",
				row.Handle)
		}
		return 1
	}
	body, err := fetchSessionIndexStatus(row)
	if err != nil {
		fmt.Fprintf(stderr, "fak session status: %s: %v\n", row.Handle, err)
		return 1
	}
	if !asJSON {
		fmt.Fprintf(stdout, "guard session %s (trace %s, pid %d) — %s/debug/vars\n", row.Handle, orDash(row.TraceID), row.PID, row.GatewayURL)
	}
	_, _ = stdout.Write(body)
	if len(body) > 0 && body[len(body)-1] != '\n' {
		fmt.Fprintln(stdout)
	}
	return 0
}

// fetchSessionIndexStatus performs the one cross-process read: the session's raw
// /debug/vars document, authenticated with the row's read-scoped bearer (the token the
// gateway accepts ONLY on its read endpoints). The body is bounded like every other
// gateway response this CLI reads.
func fetchSessionIndexStatus(row guardsessions.Row) ([]byte, error) {
	url := strings.TrimRight(row.GatewayURL, "/") + "/debug/vars"
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if row.Bearer != "" {
		req.Header.Set("Authorization", "Bearer "+row.Bearer)
	}
	resp, err := sessionQueryHTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxSessionRespBytes))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", url, err)
	}
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("GET %s: %s: %s", url, resp.Status, strings.TrimSpace(string(body)))
	}
	return body, nil
}

// sessionIndexResolveStatus is the `fak session status` front half of the #3461 path:
// resolve the query against the guard-session index. Matched==1 answers from the row's
// own gateway; an ambiguous prefix reports the tie (exit 3, like `fak guard sessions`);
// no match returns handled=false so the caller can fall back to the legacy
// single-gateway drive-state read (preserving `fak session status <drive-id>` against a
// $FAK_ADDR-less default gateway for anyone who relied on it).
func sessionIndexResolveStatus(stdout, stderr io.Writer, regDir, query string, asJSON bool) (rc int, handled bool) {
	rows := guardsessions.Load(regDir)
	res := guardsessions.Resolve(rows, query)
	switch {
	case res.Matched == 1:
		return runSessionIndexStatus(stdout, stderr, res.Row, asJSON), true
	case res.Matched > 1:
		fmt.Fprintf(stderr, "fak session status: %q matches %d guard sessions — narrow the prefix:\n", query, res.Matched)
		for _, c := range res.Candidates {
			fmt.Fprintf(stderr, "  %s  %s\n", c.Handle, orDash(c.TraceID))
		}
		return 3, true
	}
	return 0, false
}
