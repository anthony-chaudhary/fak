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
//	fak session budget <id> [--turns N] [--tokens N] [--context-tokens N]   # re-set the work allotment live
//	fak session pace   <id> [--max-tokens N] [--gap-ms N]  # re-set the per-turn throttle
//	fak session envelope <id> <spec>       # parse/apply one managed-context budget envelope (#1573)
//	fak session priority <id> <N>           # re-set the scheduling rank (lower yields first)
//	fak session reset-diff [--in FILE] [--json] [--md]  # offline before/after reset diff (#1575, see session_reset_diff.go)
//
// All write verbs accept --if-rev N: the optimistic-concurrency guard, so a stale
// operator (or a second controller) cannot clobber a newer change — a lost race
// returns a clear 409 to re-read and retry. A partial budget/pace update reads the
// current state first and preserves the axes you did not name, fencing that
// read-modify-write with the observed rev once the session has prior state (rev>=1);
// a fresh, never-written session (rev 0) takes the plain write — it is at its defaults,
// so there is no newer change to clobber.
//
// Connection: --addr (default $FAK_ADDR or http://127.0.0.1:8080) and --key (default
// $FAK_KEY) — a loopback gateway with no --require-key needs neither.

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// sessionFlagUnset is the sentinel for an integer flag the operator did not set, so a
// partial budget/pace update can tell "leave this axis alone" from a real 0 or -1
// (both of which are meaningful: 0 = planner default for pace, -1 = unbounded for
// budget). It is deliberately an absurd value no operator would type.
const sessionFlagUnset = -1 << 62

// maxSessionRespBytes caps a gateway JSON response the CLI will read into memory —
// generous for a SessionListResponse over a large fleet, but bounded so a misbehaving
// gateway cannot stream an unbounded body into the operator's process.
const maxSessionRespBytes = 4 << 20

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

	// reset-diff (#1575) is the one offline verb in this surface: a pure JSON-in,
	// diff-out render over internal/sessionreset.DiffReset that never dials a live
	// gateway, so it is dispatched here before the gateway-shaped arity/flag table
	// below (which assumes every verb talks to a sessionClient).
	if verb == "reset-diff" {
		return runSessionResetDiff(os.Stdin, stdout, stderr, args)
	}
	if verb == "envelope" && (len(args) == 0 || strings.HasPrefix(args[0], "-")) {
		return runSessionEnvelope(stdout, stderr, args)
	}

	// Positional arity per verb: the fixed leading args (an id, maybe a value) come
	// before any flags, so `fak session status sess-1 --json` parses cleanly.
	arity := map[string]int{
		"ls": 0, "status": 1,
		"stop": 1, "pause": 1, "resume": 1, "throttle": 1,
		"run": 2, "budget": 1, "pace": 1, "envelope": 2, "budget-envelope": 2, "priority": 2,
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
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (only if the gateway sets --require-key)")
	asJSON := fs.Bool("json", false, "emit the raw JSON instead of the human table")
	ifRev := fs.Uint64("if-rev", 0, "optimistic-concurrency guard: apply only if the session's current rev matches (0 = no guard)")
	reason := fs.String("reason", "", "reason token recorded on throttle/stop")
	turns := fs.Int("turns", sessionFlagUnset, "budget: remaining turns (-1 = unbounded)")
	tokens := fs.Int("tokens", sessionFlagUnset, "budget: remaining output tokens (-1 = unbounded)")
	contextTokens := fs.Int("context-tokens", sessionFlagUnset, "budget: remaining prompt/context tokens (0 = off)")
	maxTokens := fs.Int("max-tokens", sessionFlagUnset, "pace: max output tokens this turn (0 = planner default)")
	gapMs := fs.Int("gap-ms", sessionFlagUnset, "pace: minimum inter-turn gap in ms (0 = none)")
	inspectOnly := fs.Bool("inspect-only", false, "envelope: parse and print the deterministic budget envelope without applying it")
	if rc, ok := parseFlagsOrHelp(fs, flagArgs); !ok {
		return rc
	}
	// flag.Parse stops at the first non-flag token, so a stray positional (or a flag
	// placed BEFORE the id) would otherwise be silently dropped or misread as the id.
	// Reject leftovers loudly instead: the id (and any value) come first, then flags.
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session %s: unexpected argument %q (the id/value come first, then flags)\n", verb, fs.Arg(0))
		return 2
	}

	c := &sessionClient{base: strings.TrimRight(*addr, "/"), key: *key, hc: &http.Client{Timeout: 15 * time.Second}}

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
		return c.budgetVerb(stdout, stderr, *asJSON, pos[0], *turns, *tokens, *contextTokens, *ifRev)
	case "pace":
		return c.paceVerb(stdout, stderr, *asJSON, pos[0], *maxTokens, *gapMs, *ifRev)
	case "envelope", "budget-envelope":
		return c.envelopeVerb(stdout, stderr, *asJSON, pos[0], pos[1], *ifRev, *inspectOnly)
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

type sessionEnvelopeReport struct {
	Envelope    session.BudgetEnvelope `json:"envelope"`
	Applied     []string               `json:"applied,omitempty"`
	InspectOnly bool                   `json:"inspect_only,omitempty"`
	State       *gateway.SessionState  `json:"state,omitempty"`
}

// runVerb applies a run-state change (the cancel/pause/resume/throttle family). A
// non-empty reason rides throttle/stop; running clears it (the table enforces the
// same bookkeeping, this just passes intent).
func (c *sessionClient) runVerb(stdout, stderr io.Writer, asJSON bool, id, state, reason string, ifRev uint64) int {
	return c.renderState(stdout, stderr, asJSON, func() (gateway.SessionState, error) {
		return c.control(id, "run", gateway.SessionControlRequest{Run: state, Reason: reason, IfRev: ifRev})
	})
}

// budgetVerb re-sets the work allotment. Budget is one value (all axes), so a
// partial update (only --turns, say) reads the current state and preserves the other
// axes, fencing the read-modify-write with the observed rev (unless the operator
// passed an explicit --if-rev) so a concurrent change is caught, not clobbered. The
// fence is real once the session has prior state (rev>=1); a rev-0 (never-written)
// session takes the plain write, since its defaults have nothing newer to clobber.
func (c *sessionClient) budgetVerb(stdout, stderr io.Writer, asJSON bool, id string, turns, tokens, contextTokens int, ifRev uint64) int {
	if turns == sessionFlagUnset && tokens == sessionFlagUnset && contextTokens == sessionFlagUnset {
		fmt.Fprintln(stderr, "fak session budget: set at least one of --turns / --tokens / --context-tokens")
		return 2
	}
	b, rev, code := c.mergeBudget(stderr, id, turns, tokens, contextTokens, ifRev)
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

// envelopeVerb parses issue #1573's one-string managed-context budget envelope and
// applies the axes supported by the existing gateway control API: budget and pace.
// Wall-clock, spend, and throughput remain visible in the parsed report so the caller
// can inspect the deterministic contract even when this gateway build has no direct
// control route for those axes.
func (c *sessionClient) envelopeVerb(stdout, stderr io.Writer, asJSON bool, id, spec string, ifRev uint64, inspectOnly bool) int {
	env, err := session.ParseBudgetEnvelope(spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak session envelope: parse: %v\n", err)
		return 2
	}
	rep := sessionEnvelopeReport{Envelope: env, InspectOnly: inspectOnly}
	if inspectOnly {
		return emitSessionEnvelopeReport(stdout, stderr, asJSON, rep)
	}

	gb := gateway.SessionBudget{
		TurnsLeft:         env.SessionBudget().TurnsLeft,
		TokensLeft:        env.SessionBudget().TokensLeft,
		ContextTokensLeft: env.SessionBudget().ContextTokensLeft,
	}
	st, err := c.control(id, "budget", gateway.SessionControlRequest{Budget: &gb, IfRev: ifRev})
	if err != nil {
		fmt.Fprintf(stderr, "fak session envelope: apply budget: %v\n", err)
		return 1
	}
	rep.Applied = append(rep.Applied, "budget")
	rep.State = &st

	pace := env.SessionPace()
	if pace.MaxTokensPerTurn > 0 || pace.MinTurnGapMs > 0 {
		gp := gateway.SessionPace{MaxTokensPerTurn: pace.MaxTokensPerTurn, MinTurnGapMs: pace.MinTurnGapMs}
		st, err = c.control(id, "pace", gateway.SessionControlRequest{Pace: &gp, IfRev: st.Rev})
		if err != nil {
			fmt.Fprintf(stderr, "fak session envelope: apply pace: %v\n", err)
			return 1
		}
		rep.Applied = append(rep.Applied, "pace")
		rep.State = &st
	}
	return emitSessionEnvelopeReport(stdout, stderr, asJSON, rep)
}

// mergeBudget fills the axes the operator did not name from the session's current
// state and returns the rev to fence the write with. code != 0 is an early exit (the
// observe failed); the caller returns it.
func (c *sessionClient) mergeBudget(stderr io.Writer, id string, turns, tokens, contextTokens int, ifRev uint64) (gateway.SessionBudget, uint64, int) {
	b := gateway.SessionBudget{TurnsLeft: turns, TokensLeft: tokens, ContextTokensLeft: contextTokens}
	rev := ifRev
	if turns == sessionFlagUnset || tokens == sessionFlagUnset || contextTokens == sessionFlagUnset || ifRev == 0 {
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
		if contextTokens == sessionFlagUnset {
			b.ContextTokensLeft = cur.Budget.ContextTokensLeft
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
	if err := writeIndentedJSON(stdout, v); err != nil {
		fmt.Fprintf(stderr, "fak session: encode json: %v\n", err)
		return 1
	}
	return 0
}

func emitSessionEnvelopeReport(stdout, stderr io.Writer, asJSON bool, rep sessionEnvelopeReport) int {
	if asJSON {
		return emitSessionJSON(stdout, stderr, rep)
	}
	fmt.Fprintf(stdout, "budget-envelope %s\n", formatBudgetEnvelope(rep.Envelope))
	if len(rep.Applied) > 0 {
		fmt.Fprintf(stdout, "applied: %s\n", strings.Join(rep.Applied, ","))
	}
	if rep.State != nil {
		fmt.Fprintln(stdout, formatSessionState(*rep.State))
	}
	return 0
}

func formatBudgetEnvelope(env session.BudgetEnvelope) string {
	parts := []string{
		"turns=" + budgetAxis(env.Budget.TurnsLeft),
		"tokens=" + budgetAxis(env.Budget.TokensLeft),
		"context=" + contextBudgetAxis(env.Budget.ContextTokensLeft),
	}
	if env.WallClockLimit() > 0 {
		parts = append(parts, "wall="+env.WallClockLimit().String())
	}
	if !env.Spend.IsZero() {
		parts = append(parts, fmt.Sprintf("spend=%s %.2f", env.Spend.Currency, float64(env.Spend.MaxCents)/100))
	}
	if !env.Throughput.IsZero() {
		parts = append(parts, fmt.Sprintf("throughput=%.3g/s", env.Throughput.ExpectedTokensPerSec))
		if env.Throughput.MinTokensPerSec > 0 {
			parts = append(parts, fmt.Sprintf("min_throughput=%.3g/s", env.Throughput.MinTokensPerSec))
		}
	}
	if env.Pace.MaxTokensPerTurn > 0 || env.Pace.MinTurnGapMs > 0 {
		parts = append(parts, fmt.Sprintf("pace(max=%d gap=%dms)", env.Pace.MaxTokensPerTurn, env.Pace.MinTurnGapMs))
	}
	if env.Budget.ClarificationQueriesCap > 0 || env.Budget.ClarificationQueriesLeft > 0 {
		parts = append(parts, "queries="+budgetAxis(env.Budget.ClarificationQueriesLeft))
	}
	return strings.Join(parts, " ")
}

// formatSessionState renders one drive record as a compact, fixed-shape line so a
// column scan reads cleanly. Unbounded (-1) budget axes render as "inf"; a reason,
// when present, is appended.
func formatSessionState(st gateway.SessionState) string {
	line := fmt.Sprintf("%-24s %-9s budget(turns=%s tokens=%s context=%s) pace(max=%d gap=%dms) prio=%d rev=%d",
		st.TraceID, st.Run,
		budgetAxis(st.Budget.TurnsLeft), budgetAxis(st.Budget.TokensLeft), contextBudgetAxis(st.Budget.ContextTokensLeft),
		st.Pace.MaxTokensPerTurn, st.Pace.MinTurnGapMs, st.Priority, st.Rev)
	if st.Reason != "" {
		line += " reason=" + st.Reason
	}
	if seg := formatSessionTime(st.Time); seg != "" {
		line += " " + seg
	}
	if st.ContinuationID != "" {
		line += " continuation=" + st.ContinuationID
	}
	if st.ParentTrace != "" {
		line += " parent=" + st.ParentTrace
	}
	if st.Generation > 0 {
		line += fmt.Sprintf(" gen=%d", st.Generation)
	}
	return line
}

// formatSessionTime renders the wall-clock budget segment of a session line: the twin of
// the budget(...) token segment, showing where a `--max-duration` / managed-context wall
// axis stands. It returns "" for a zero (never-configured, never-started) time budget so
// the common no-time-budget session line is byte-identical to before this axis existed.
// A bounded envelope shows elapsed + remaining + limit (plus an EXCEEDED marker once the
// wall clock is spent); an unbounded-but-ticking session shows only elapsed, honoring
// "--max-duration 0 … still tracked for session status".
func formatSessionTime(t gateway.SessionTime) string {
	if t.IsZero() {
		return ""
	}
	dur := func(sec int64) string { return (time.Duration(sec) * time.Second).String() }
	if !t.Bounded {
		return "time(elapsed=" + dur(t.ElapsedSeconds) + ")"
	}
	seg := fmt.Sprintf("time(elapsed=%s remaining=%s limit=%s", dur(t.ElapsedSeconds), dur(t.RemainingSeconds), dur(t.LimitSeconds))
	if t.Exceeded {
		seg += " EXCEEDED"
	}
	return seg + ")"
}

// budgetAxis renders an unbounded (negative) budget axis as a stable token rather
// than a raw -1, so an operator never misreads "no cap" as "minus one turn left".
func budgetAxis(v int) string {
	if v < 0 {
		return "inf"
	}
	return strconv.Itoa(v)
}

func contextBudgetAxis(v int) string {
	if v < 0 {
		return "inf"
	}
	if v == 0 {
		return "off"
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
	err := c.req(http.MethodGet, "/v1/fak/session/"+url.PathEscape(id), nil, &st)
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
	err := c.req(http.MethodPost, "/v1/fak/session/"+url.PathEscape(id)+"/"+verb, body, &st)
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
	// Bound the success body too (the error path is already capped): a misbehaving or
	// compromised gateway must not stream an unbounded 200 into the operator's memory.
	// maxSessionRespBytes sits well above a SessionListResponse for a large fleet.
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxSessionRespBytes)).Decode(out); err != nil {
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
// the loopback dogfood default. Any trailing slash is trimmed where the client is
// built (strings.TrimRight, covering both this default and an explicit --addr), so
// path joins stay clean even behind a strict (non-Go) reverse proxy.
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
  fak session budget   <id> [--turns N] [--tokens N] [--context-tokens N]  re-set the work allotment live
  fak session pace     <id> [--max-tokens N] [--gap-ms N]   re-set the per-turn throttle
  fak session envelope <id> <spec>            apply a managed-context budget envelope
                                               spec: turns=20,tokens=200000,context=64000,wall=2h,spend=$25,throughput=40/s,max-tokens=1024,gap=250ms
  fak session priority <id> <N>               re-set the scheduling rank (lower yields first)
  fak session reset-diff [--in FILE] [--json] [--md]
                                               offline before/after diff for one reset
                                               (survived/summarized/expired/must-requery)

flags: --addr (default $FAK_ADDR or http://127.0.0.1:8080)  --key ($FAK_KEY)
       --if-rev N (optimistic-concurrency guard)  --json
       envelope: --inspect-only
`)
}
