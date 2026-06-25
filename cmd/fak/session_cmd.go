package main

// session_cmd.go — `fak session`, the OPERATOR control surface for a served
// session's live DRIVE state. It is the human/script front end to the
// /v1/fak/session(s) routes (#620): read what a session is doing right now, and —
// the goal it serves — CANCEL or UPDATE a session in flight from outside it.
//
//	fak session ls                          # every live session (the snapshot)
//	fak session status <id>                 # one session's drive state
//	fak session stop   <id> [--reason R]    # request a clean stop (drain at the next boundary)
//	fak session pause  <id>                 # hold at the next boundary
//	fak session resume <id>                 # un-pause (a live state flip, not a cold re-attach)
//	fak session throttle <id> [--reason R]  # slow without pausing
//	fak session run    <id> <state>         # set any run-state (running|throttled|paused|draining|stopped)
//	fak session budget <id> [--turns N] [--tokens N]   # re-set the work allotment live
//	fak session pace   <id> [--max-tokens N] [--gap-ms N]  # re-set the per-turn throttle
//	fak session priority <id> <N>           # re-set the scheduling rank (lower yields first)
//
// All write verbs accept --if-rev N: the optimistic-concurrency guard, so a stale
// operator (or a second controller) cannot clobber a newer change — a lost race
// returns a clear 409 to re-read and retry. A partial budget/pace update reads the
// current state first and preserves the axes you did not name, fencing that
// read-modify-write with the observed rev so it stays atomic.
//
// Connection: --addr (default $FAK_ADDR or http://127.0.0.1:8080) and --key (default
// $FAK_KEY) — a loopback gateway with no --require-key needs neither.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// sessionFlagUnset is the sentinel for an integer flag the operator did not set, so a
// partial budget/pace update can tell "leave this axis alone" from a real 0 or -1
// (both of which are meaningful: 0 = planner default for pace, -1 = unbounded for
// budget). It is deliberately an absurd value no operator would type.
const sessionFlagUnset = -1 << 62

// cmdSession is the `fak session` entry point. It delegates to the testable core and
// maps its exit code to the process exit code, mirroring cmdRoute.
func cmdSession(argv []string) { os.Exit(runSession(os.Stdout, os.Stderr, argv)) }

// runSession is the testable core: it returns the process exit code (0 ok, 1 a
// transport/HTTP error, 2 a usage error) and takes its streams explicitly so a test
// can drive it against an httptest gateway and assert the rendered output.
func runSession(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		sessionUsage(stderr)
		return 2
	}
	verb := argv[0]
	args := argv[1:]

	// Positional arity per verb: the fixed leading args (an id, maybe a value) come
	// before any flags, so `fak session status sess-1 --json` parses cleanly.
	arity := map[string]int{
		"ls": 0, "status": 1,
		"stop": 1, "pause": 1, "resume": 1, "throttle": 1,
		"run": 2, "budget": 1, "pace": 1, "priority": 2,
	}
	want, known := arity[verb]
	if !known {
		fmt.Fprintf(stderr, "fak session: unknown verb %q\n", verb)
		sessionUsage(stderr)
		return 2
	}
	if len(args) < want {
		fmt.Fprintf(stderr, "fak session %s: missing argument(s); want %d\n", verb, want)
		sessionUsage(stderr)
		return 2
	}
	pos := args[:want]
	flagArgs := args[want:]

	fs := flag.NewFlagSet("session "+verb, flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", os.Getenv("FAK_KEY"), "bearer credential (only if the gateway sets --require-key)")
	asJSON := fs.Bool("json", false, "emit the raw JSON instead of the human table")
	ifRev := fs.Uint64("if-rev", 0, "optimistic-concurrency guard: apply only if the session's current rev matches (0 = no guard)")
	reason := fs.String("reason", "", "reason token recorded on throttle/stop")
	turns := fs.Int("turns", sessionFlagUnset, "budget: remaining turns (-1 = unbounded)")
	tokens := fs.Int("tokens", sessionFlagUnset, "budget: remaining output tokens (-1 = unbounded)")
	maxTokens := fs.Int("max-tokens", sessionFlagUnset, "pace: max output tokens this turn (0 = planner default)")
	gapMs := fs.Int("gap-ms", sessionFlagUnset, "pace: minimum inter-turn gap in ms (0 = none)")
	if err := fs.Parse(flagArgs); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	c := &sessionClient{base: *addr, key: *key, hc: &http.Client{Timeout: 15 * time.Second}}

	switch verb {
	case "ls":
		return c.renderList(stdout, stderr, *asJSON)
	case "status":
		return c.renderState(stdout, stderr, *asJSON, func() (gateway.SessionState, error) {
			return c.observe(pos[0])
		})
	case "stop":
		return c.runVerb(stdout, stderr, *asJSON, pos[0], "stopped", *reason, *ifRev)
	case "pause":
		return c.runVerb(stdout, stderr, *asJSON, pos[0], "paused", *reason, *ifRev)
	case "resume":
		return c.runVerb(stdout, stderr, *asJSON, pos[0], "running", *reason, *ifRev)
	case "throttle":
		return c.runVerb(stdout, stderr, *asJSON, pos[0], "throttled", *reason, *ifRev)
	case "run":
		return c.runVerb(stdout, stderr, *asJSON, pos[0], pos[1], *reason, *ifRev)
	case "budget":
		return c.budgetVerb(stdout, stderr, *asJSON, pos[0], *turns, *tokens, *ifRev)
	case "pace":
		return c.paceVerb(stdout, stderr, *asJSON, pos[0], *maxTokens, *gapMs, *ifRev)
	case "priority":
		n, err := strconv.Atoi(pos[1])
		if err != nil {
			fmt.Fprintf(stderr, "fak session priority: %q is not an integer\n", pos[1])
			return 2
		}
		return c.renderState(stdout, stderr, *asJSON, func() (gateway.SessionState, error) {
			return c.control(pos[0], "priority", gateway.SessionControlRequest{Priority: &n, IfRev: *ifRev})
		})
	}
	return 2 // unreachable: arity gate already rejected unknown verbs
}

// runVerb applies a run-state change (the cancel/pause/resume/throttle family). A
// non-empty reason rides throttle/stop; running clears it (the table enforces the
// same bookkeeping, this just passes intent).
func (c *sessionClient) runVerb(stdout, stderr io.Writer, asJSON bool, id, state, reason string, ifRev uint64) int {
	return c.renderState(stdout, stderr, asJSON, func() (gateway.SessionState, error) {
		return c.control(id, "run", gateway.SessionControlRequest{Run: state, Reason: reason, IfRev: ifRev})
	})
}

// budgetVerb re-sets the work allotment. Budget is one value (both axes), so a
// partial update (only --turns, say) reads the current state and preserves the other
// axis, fencing the read-modify-write with the observed rev (unless the operator
// passed an explicit --if-rev) so a concurrent change is caught, not clobbered.
func (c *sessionClient) budgetVerb(stdout, stderr io.Writer, asJSON bool, id string, turns, tokens int, ifRev uint64) int {
	if turns == sessionFlagUnset && tokens == sessionFlagUnset {
		fmt.Fprintln(stderr, "fak session budget: set at least one of --turns / --tokens")
		return 2
	}
	b, rev, code := c.mergeBudget(stderr, id, turns, tokens, ifRev)
	if code != 0 {
		return code
	}
	return c.renderState(stdout, stderr, asJSON, func() (gateway.SessionState, error) {
		return c.control(id, "budget", gateway.SessionControlRequest{Budget: &b, IfRev: rev})
	})
}

// paceVerb re-sets the per-turn throttle, with the same preserve-unset-axis +
// fence-with-observed-rev discipline as budgetVerb.
func (c *sessionClient) paceVerb(stdout, stderr io.Writer, asJSON bool, id string, maxTokens, gapMs int, ifRev uint64) int {
	if maxTokens == sessionFlagUnset && gapMs == sessionFlagUnset {
		fmt.Fprintln(stderr, "fak session pace: set at least one of --max-tokens / --gap-ms")
		return 2
	}
	p, rev, code := c.mergePace(stderr, id, maxTokens, gapMs, ifRev)
	if code != 0 {
		return code
	}
	return c.renderState(stdout, stderr, asJSON, func() (gateway.SessionState, error) {
		return c.control(id, "pace", gateway.SessionControlRequest{Pace: &p, IfRev: rev})
	})
}

// mergeBudget fills the axes the operator did not name from the session's current
// state and returns the rev to fence the write with. code != 0 is an early exit (the
// observe failed); the caller returns it.
func (c *sessionClient) mergeBudget(stderr io.Writer, id string, turns, tokens int, ifRev uint64) (gateway.SessionBudget, uint64, int) {
	b := gateway.SessionBudget{TurnsLeft: turns, TokensLeft: tokens}
	rev := ifRev
	if turns == sessionFlagUnset || tokens == sessionFlagUnset || ifRev == 0 {
		cur, err := c.observe(id)
		if err != nil {
			fmt.Fprintf(stderr, "fak session budget: read current state: %v\n", err)
			return b, 0, 1
		}
		if turns == sessionFlagUnset {
			b.TurnsLeft = cur.Budget.TurnsLeft
		}
		if tokens == sessionFlagUnset {
			b.TokensLeft = cur.Budget.TokensLeft
		}
		if ifRev == 0 {
			rev = cur.Rev // fence the read-modify-write so a concurrent change 409s
		}
	}
	return b, rev, 0
}

// mergePace is mergeBudget for the per-turn throttle axes.
func (c *sessionClient) mergePace(stderr io.Writer, id string, maxTokens, gapMs int, ifRev uint64) (gateway.SessionPace, uint64, int) {
	p := gateway.SessionPace{MaxTokensPerTurn: maxTokens, MinTurnGapMs: gapMs}
	rev := ifRev
	if maxTokens == sessionFlagUnset || gapMs == sessionFlagUnset || ifRev == 0 {
		cur, err := c.observe(id)
		if err != nil {
			fmt.Fprintf(stderr, "fak session pace: read current state: %v\n", err)
			return p, 0, 1
		}
		if maxTokens == sessionFlagUnset {
			p.MaxTokensPerTurn = cur.Pace.MaxTokensPerTurn
		}
		if gapMs == sessionFlagUnset {
			p.MinTurnGapMs = cur.Pace.MinTurnGapMs
		}
		if ifRev == 0 {
			rev = cur.Rev
		}
	}
	return p, rev, 0
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

// renderState runs a single-session call and prints its result (JSON or a one-line
// human form), mapping any error to exit 1.
func (c *sessionClient) renderState(stdout, stderr io.Writer, asJSON bool, call func() (gateway.SessionState, error)) int {
	st, err := call()
	if err != nil {
		fmt.Fprintf(stderr, "fak session: %v\n", err)
		return 1
	}
	if asJSON {
		return emitSessionJSON(stdout, stderr, st)
	}
	fmt.Fprintln(stdout, formatSessionState(st))
	return 0
}

// renderList runs the multi-session snapshot call and prints a table (or JSON).
func (c *sessionClient) renderList(stdout, stderr io.Writer, asJSON bool) int {
	list, err := c.list()
	if err != nil {
		fmt.Fprintf(stderr, "fak session ls: %v\n", err)
		return 1
	}
	if asJSON {
		return emitSessionJSON(stdout, stderr, list)
	}
	if list.Count == 0 {
		fmt.Fprintln(stdout, "no live sessions")
		return 0
	}
	for _, st := range list.Sessions {
		fmt.Fprintln(stdout, formatSessionState(st))
	}
	fmt.Fprintf(stdout, "%d session(s)\n", list.Count)
	return 0
}

func emitSessionJSON(stdout, stderr io.Writer, v any) int {
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(stderr, "fak session: encode json: %v\n", err)
		return 1
	}
	return 0
}

// formatSessionState renders one drive record as a compact, fixed-shape line so a
// column scan reads cleanly. Unbounded (-1) budget axes render as "∞"; a reason, when
// present, is appended.
func formatSessionState(st gateway.SessionState) string {
	line := fmt.Sprintf("%-24s %-9s budget(turns=%s tokens=%s) pace(max=%d gap=%dms) prio=%d rev=%d",
		st.TraceID, st.Run,
		budgetAxis(st.Budget.TurnsLeft), budgetAxis(st.Budget.TokensLeft),
		st.Pace.MaxTokensPerTurn, st.Pace.MinTurnGapMs, st.Priority, st.Rev)
	if st.Reason != "" {
		line += " reason=" + st.Reason
	}
	return line
}

// budgetAxis renders an unbounded (negative) budget axis as a stable token rather
// than a raw -1, so an operator never misreads "no cap" as "minus one turn left".
func budgetAxis(v int) string {
	if v < 0 {
		return "inf"
	}
	return strconv.Itoa(v)
}

// ---------------------------------------------------------------------------
// HTTP client (cmd-local: the only consumer of the routes today is this CLI)
// ---------------------------------------------------------------------------

type sessionClient struct {
	base string
	key  string
	hc   *http.Client
}

// observe reads one session's drive state (GET /v1/fak/session/{id}).
func (c *sessionClient) observe(id string) (gateway.SessionState, error) {
	var st gateway.SessionState
	err := c.req(http.MethodGet, "/v1/fak/session/"+id, nil, &st)
	return st, err
}

// list reads every live session's drive state (GET /v1/fak/sessions).
func (c *sessionClient) list() (gateway.SessionListResponse, error) {
	var lr gateway.SessionListResponse
	err := c.req(http.MethodGet, "/v1/fak/sessions", nil, &lr)
	return lr, err
}

// control applies one verb (POST /v1/fak/session/{id}/{verb}) and returns the new
// drive state.
func (c *sessionClient) control(id, verb string, body gateway.SessionControlRequest) (gateway.SessionState, error) {
	var st gateway.SessionState
	err := c.req(http.MethodPost, "/v1/fak/session/"+id+"/"+verb, body, &st)
	return st, err
}

// req is the one HTTP round-trip: it marshals an optional body, sets the bearer
// credential when configured, and decodes a 2xx JSON body into out. A non-2xx is
// turned into a typed error — the 409 (terminal session / lost CAS race) and 404
// (route not configured) get a clear, operator-actionable message.
func (c *sessionClient) req(method, path string, body any, out any) error {
	var rdr io.Reader
	if body != nil {
		buf, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		rdr = bytes.NewReader(buf)
	}
	httpReq, err := http.NewRequestWithContext(context.Background(), method, c.base+path, rdr)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		httpReq.Header.Set("Content-Type", "application/json")
	}
	if c.key != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.key)
	}
	resp, err := c.hc.Do(httpReq)
	if err != nil {
		return fmt.Errorf("%s %s: %w", method, c.base+path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return httpStatusError(resp)
	}
	if out == nil {
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

// httpStatusError maps a non-2xx response to an operator-actionable error, reading
// the OpenAI-style error envelope the gateway emits for the message text.
func httpStatusError(resp *http.Response) error {
	msg := readErrMessage(resp.Body)
	switch resp.StatusCode {
	case http.StatusConflict:
		return fmt.Errorf("refused (409): the session is stopped (terminal) or changed under you — re-read and retry: %s", msg)
	case http.StatusNotFound:
		return fmt.Errorf("not found (404): the session-control routes are not enabled on this gateway: %s", msg)
	case http.StatusUnauthorized:
		return fmt.Errorf("unauthorized (401): pass --key (or set $FAK_KEY) for a gateway with --require-key: %s", msg)
	default:
		return fmt.Errorf("gateway returned %d: %s", resp.StatusCode, msg)
	}
}

// readErrMessage best-effort extracts {"error":{"message":...}} from a body, falling
// back to the raw (bounded) text so a non-JSON error page is still legible.
func readErrMessage(r io.Reader) string {
	raw, _ := io.ReadAll(io.LimitReader(r, 8<<10))
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &env) == nil && env.Error.Message != "" {
		return env.Error.Message
	}
	return string(bytes.TrimSpace(raw))
}

// defaultSessionAddr is the gateway base URL the CLI talks to: $FAK_ADDR if set, else
// the loopback dogfood default. A trailing slash is trimmed so path joins are clean.
func defaultSessionAddr() string {
	if a := os.Getenv("FAK_ADDR"); a != "" {
		return a
	}
	return "http://127.0.0.1:8080"
}

func sessionUsage(w io.Writer) {
	fmt.Fprint(w, `fak session — read and control a served session's live DRIVE state

  fak session ls                              every live session (the snapshot)
  fak session status   <id>                   one session's drive state
  fak session stop     <id> [--reason R]      request a clean stop (drain at the next boundary)
  fak session pause    <id>                   hold at the next turn boundary
  fak session resume   <id>                   un-pause (a live state flip)
  fak session throttle <id> [--reason R]      slow without pausing
  fak session run      <id> <state>           set running|throttled|paused|draining|stopped
  fak session budget   <id> [--turns N] [--tokens N]        re-set the work allotment live
  fak session pace     <id> [--max-tokens N] [--gap-ms N]   re-set the per-turn throttle
  fak session priority <id> <N>               re-set the scheduling rank (lower yields first)

flags: --addr (default $FAK_ADDR or http://127.0.0.1:8080)  --key ($FAK_KEY)
       --if-rev N (optimistic-concurrency guard)  --json
`)
}
