package main

// ps.go — `fak ps` (watch-mode `fak top`), the read-only PROCESS TABLE for a
// live `fak serve`. It folds the one always-available per-session run surface —
// GET /v1/fak/sessions, the drive-state snapshot the gateway already serves
// (#620) — into one aligned row per live session, the way `ps`/`top` render a
// kernel's process table. It is the READ-ONLY twin of the scheduler (#627): both
// read session.Table.Snapshot(); the scheduler decides who yields, `ps` only
// renders. It issues no control verb and mutates no drive state (no Set*/Decide).
//
//	fak ps                 one aligned row per live session (drive state + activity)
//	fak ps --json          the raw GET /v1/fak/sessions body, machine-readable
//	fak ps --watch         refresh every --interval until interrupted (the `top` mode)
//	fak top                fak ps --watch (watch on by default; --watch=false for one shot)
//
// Connection: --addr (default $FAK_ADDR or http://127.0.0.1:8080) and --key
// (default $FAK_KEY) — a loopback gateway with no --require-key needs neither.
//
// Columns are folded from the session drive surface ONLY: TRACE, STATE, the three
// budget axes (TURNS/TOKENS/CTX — "inf" = uncapped, "off" = no context cap), PRIO
// (scheduling rank, lower yields first), REV (the monotonic drive-revision counter,
// a session's activity/iteration proxy), and REASON (the closed token a
// throttled/stopped session carries). The elapsed-turn, ETA, and cost-per-iteration
// columns named in #755 are deferred on purpose: ETA lives on the internal/taskmgr
// snapshot, which carries no session-trace join today, and the cost column lands
// with #755's P2.2 dependency. `fak ps` renders what the live session surface
// actually carries rather than fabricate a progress estimate it cannot source.

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const (
	// psDefaultInterval is the watch-mode refresh cadence (the `top` tick).
	psDefaultInterval = 2 * time.Second
	// psClearScreen homes the cursor and clears the screen between watch frames, so
	// a refreshing `fak top` overwrites the prior frame in place instead of
	// scrolling. A single-shot `fak ps` never emits it.
	psClearScreen = "\033[H\033[2J"
)

// cmdPS / cmdTop are the os.Exit-wrapping entry points dispatched by main. Both
// route through runPS so the dispatch and the tests share one code path (mirroring
// cmdSession/runSession); cmdTop only changes the --watch default.
func cmdPS(argv []string)  { os.Exit(runPS(os.Stdout, os.Stderr, argv, false)) }
func cmdTop(argv []string) { os.Exit(runPS(os.Stdout, os.Stderr, argv, true)) }

// runPS is the testable core. It returns the process exit code (0 ok, 1 a
// transport/HTTP error, 2 a usage error) and takes its streams explicitly so a
// test can drive it against an httptest gateway and assert the rendered table.
// watchDefault sets the --watch default so `fak top` watches by default while
// `fak ps` is single-shot; either can be overridden (--watch / --watch=false).
func runPS(stdout, stderr io.Writer, argv []string, watchDefault bool) int {
	fs := flag.NewFlagSet("ps", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { psUsage(stderr) }
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (only if the gateway sets --require-key)")
	asJSON := fs.Bool("json", false, "emit the raw GET /v1/fak/sessions JSON instead of the human table")
	watch := fs.Bool("watch", watchDefault, "refresh continuously (the `top` mode)")
	interval := fs.Duration("interval", psDefaultInterval, "watch refresh cadence")
	frames := fs.Int("frames", 0, "watch: stop after N frames (0 = until interrupted)")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	// `fak ps` takes only flags; a stray positional is almost always a mistake (a
	// session id meant for `fak session`), so reject it loudly rather than ignore it.
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak ps: unexpected argument %q (fak ps takes only flags; control one session with `fak session`)\n", fs.Arg(0))
		return 2
	}

	c := &sessionClient{base: strings.TrimRight(*addr, "/"), key: *key, hc: &http.Client{Timeout: 15 * time.Second}}
	if !*watch {
		return psFrame(stdout, stderr, c, *asJSON)
	}
	return psWatch(stdout, stderr, c, *asJSON, *interval, *frames)
}

// psFrame fetches the session snapshot once and renders it (JSON or the aligned
// table), returning the process exit code. A transport/HTTP error maps to 1.
func psFrame(stdout, stderr io.Writer, c *sessionClient, asJSON bool) int {
	list, err := c.list()
	if err != nil {
		fmt.Fprintf(stderr, "fak ps: %v\n", err)
		return 1
	}
	if asJSON {
		return emitSessionJSON(stdout, stderr, list)
	}
	renderPSTable(stdout, list)
	return 0
}

// psWatch refreshes the view on the interval until the frame budget is spent
// (frames<=0 ⇒ until the operator interrupts with Ctrl-C). Each frame homes+clears
// the screen so the table updates in place. A transient fetch error is printed and
// the watch CONTINUES — a gateway blip must not tear down a `top` session — so the
// only non-zero exit from the watch path is the flag/usage error caught earlier.
func psWatch(stdout, stderr io.Writer, c *sessionClient, asJSON bool, interval time.Duration, frames int) int {
	if interval <= 0 {
		interval = psDefaultInterval
	}
	for i := 0; frames <= 0 || i < frames; i++ {
		fmt.Fprint(stdout, psClearScreen)
		fmt.Fprintf(stdout, "fak ps — live session process table @ %s (every %s; Ctrl-C to stop)\n\n", c.base, interval)
		list, err := c.list()
		switch {
		case err != nil:
			fmt.Fprintf(stderr, "fak ps: %v\n", err)
		case asJSON:
			emitSessionJSON(stdout, stderr, list)
		default:
			renderPSTable(stdout, list)
		}
		if frames > 0 && i == frames-1 {
			break // last frame rendered; do not sleep past it
		}
		time.Sleep(interval)
	}
	return 0
}

// renderPSTable prints one aligned row per live session. The rows arrive already
// in the scheduler's order (Priority asc, then Rev desc, then TraceID) from the
// gateway's Snapshot, so the table reads top-to-bottom as "who yields first".
func renderPSTable(w io.Writer, list gateway.SessionListResponse) {
	if list.Count == 0 {
		fmt.Fprintln(w, "no live sessions")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "TRACE\tSTATE\tTURNS\tTOKENS\tCTX\tPRIO\tREV\tREASON")
	for _, st := range list.Sessions {
		reason := st.Reason
		if reason == "" {
			reason = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\n",
			st.TraceID, st.Run,
			budgetAxis(st.Budget.TurnsLeft), budgetAxis(st.Budget.TokensLeft), contextBudgetAxis(st.Budget.ContextTokensLeft),
			st.Priority, st.Rev, reason)
	}
	_ = tw.Flush()
	fmt.Fprintf(w, "%d session(s)\n", list.Count)
}

func psUsage(w io.Writer) {
	fmt.Fprint(w, `fak ps — the read-only process table for a live fak serve (watch mode: fak top)

  fak ps                 one aligned row per live session (drive state + activity)
  fak ps --json          the raw GET /v1/fak/sessions body, machine-readable
  fak ps --watch         refresh every --interval until interrupted (the top mode)
  fak top                fak ps --watch (watch on by default)

flags: --addr (default $FAK_ADDR or http://127.0.0.1:8080)  --key ($FAK_KEY)
       --watch  --interval D (default 2s)  --frames N (watch: stop after N)  --json

read-only: fak ps never issues a control verb; control a session with 'fak session'.
`)
}
