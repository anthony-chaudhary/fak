package main

// session_cmd.go — `fak session`, the OPERATOR control surface for a served
// session's live DRIVE state. It is the human/script front end to the
// /v1/fak/session(s) routes (#620): read what a session is doing right now, and —
// the goal it serves — CANCEL or UPDATE a session in flight from outside it.
//
//	fak session ls                          # every live session (the snapshot)
//	fak session status <id>                 # one session's drive state
//
// Without an explicit --addr/$FAK_ADDR there is no prior gateway-port knowledge, so `ls`
// and `status` answer from the cross-process guard-session INDEX instead (#3461,
// session_query.go): --reg-dir names the registry dir holding guard_sessions.jsonl.
//	fak session stop   <id> [--reason R]    # request a clean stop (drain at the next boundary)
//	fak session terminate <id> [--reason R] # forceful stop (#2758): cancel in-flight work at the next safe point
//	fak session pause  <id>                 # hold at the next boundary
//	fak session resume <id>                 # un-pause (a live state flip, not a cold re-attach)
//	fak session throttle <id> [--reason R]  # slow without pausing
//	fak session run    <id> <state>         # set any run-state (running|throttled|paused|draining|terminating|stopped)
//	fak session budget <id> [--turns N] [--tokens N] [--context-tokens N]   # re-set the work allotment live
//	fak session pace   <id> [--max-tokens N] [--gap-ms N]  # re-set the per-turn throttle
//	fak session envelope <id> <spec>       # parse/apply one managed-context budget envelope (#1573)
//	fak session context <id>                # read the managed-context value report
//	fak session priority <id> <N>           # re-set the scheduling rank (lower yields first)
//	fak session audit [summary|actions|discover|audit|deep] ...  # offline transcript audit alias
//	fak session observe [--days N] [--json]  # zero-config recent Codex context health
//	fak session compact-audit [--since D] [--json]  # expert Codex-rollout compaction health (#4763)
//	fak session gate-fatigue [--json]      # offline per-gate approval-without-inspection rate (#4427)
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
	if code, handled := dispatchOfflineSessionVerb(stdout, stderr, verb, args); handled {
		return code
	}

	// Positional arity per verb: the fixed leading args (an id, maybe a value) come
	// before any flags, so `fak session status sess-1 --json` parses cleanly.
	arity := map[string]int{
		"ls": 0, "status": 1,
		"stop": 1, "terminate": 1, "pause": 1, "resume": 1, "throttle": 1,
		"run": 2, "budget": 1, "pace": 1, "envelope": 2, "budget-envelope": 2, "context": 1, "priority": 2,
	}
	want, known := arity[verb]
	if !known {
		fmt.Fprintf(stderr, "fak session: unknown verb %q\n", verb)
		sessionUsage(stderr)
		return 2
	}
	if verb == "resume" || verb == "pause" {
		for _, a := range args {
			if a == "--all" || a == "-all" || strings.HasPrefix(a, "--all=") || strings.HasPrefix(a, "-all=") {
				want = 0
				break
			}
		}
	}
	if len(args) < want {
		if verb == "resume" || verb == "pause" {
			fmt.Fprintf(stderr, "fak session %s: missing argument(s); want 1 (pass <id> or --all)\n", verb)
		} else {
			fmt.Fprintf(stderr, "fak session %s: missing argument(s); want %d\n", verb, want)
		}
		sessionUsage(stderr)
		return 2
	}
	pos := args[:want]
	flagArgs := args[want:]

	fs := flag.NewFlagSet("session "+verb, flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "session")
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (only if the gateway sets --require-key)")
	asJSON := fs.Bool("json", false, "emit the raw JSON instead of the human table")
	ifRev := fs.Uint64("if-rev", 0, "optimistic-concurrency guard: apply only if the session's current rev matches (0 = no guard)")
	reason := fs.String("reason", "", "reason token recorded on throttle/stop")
	all := fs.Bool("all", false, "pause/resume: apply to all matching sessions")
	turns := fs.Int("turns", sessionFlagUnset, "budget: remaining turns (-1 = unbounded)")
	tokens := fs.Int("tokens", sessionFlagUnset, "budget: remaining output tokens (-1 = unbounded)")
	contextTokens := fs.Int("context-tokens", sessionFlagUnset, "budget: remaining prompt/context tokens (0 = off)")
	maxTokens := fs.Int("max-tokens", sessionFlagUnset, "pace: max output tokens this turn (0 = planner default)")
	gapMs := fs.Int("gap-ms", sessionFlagUnset, "pace: minimum inter-turn gap in ms (0 = none)")
	inspectOnly := fs.Bool("inspect-only", false, "envelope: parse and print the deterministic budget envelope without applying it")
	durable := fs.Bool("durable", false, "ls: read the durable session registry (C1, survives restart/eviction) instead of the live gateway snapshot")
	fleet := fs.Bool("fleet", false, "ls: fold in every node's sessions from refs/fak/locks/session-* (C2) after a git fetch (implies --durable)")
	registryPath := fs.String("registry", "", "ls --durable/--fleet: session registry path (default $FAK_SESSION_REGISTRY or the user config dir)")
	remote := fs.String("remote", "origin", "ls --fleet: git remote to fetch the fleet session refs from")
	staleWindow := fs.Duration("stale", defaultSessionStaleWindow, "ls --durable/--fleet: liveness window — a RUNNING session with no heartbeat within it reads STALLED")
	// #3461: the cross-process guard-session INDEX (internal/guardsessions) — a registry
	// DIRECTORY holding guard_sessions.jsonl, the same spelling and default resolution as
	// `fak guard sessions --reg-dir`. Distinct from --registry above, which names the
	// durable C1 session registry FILE that --durable/--fleet read.
	regDir := fs.String("reg-dir", "", "ls/status without --addr/$FAK_ADDR: registry dir holding guard_sessions.jsonl (default: $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
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
		// #1203: --durable / --fleet read the durable C1 registry (and fold in the C2
		// fleet session refs) offline, deterministically — no gateway required. They win
		// over the #3461 index path below, which is the no-flag default.
		if *durable || *fleet {
			return runSessionInventory(stdout, stderr, sessionInventoryOpts{
				asJSON:       *asJSON,
				fleet:        *fleet,
				registryPath: *registryPath,
				remote:       *remote,
				staleWindow:  *staleWindow,
			})
		}
		// #3461: with NO explicit gateway address there is no prior port knowledge, so the
		// guard-session index answers instead of one assumed default gateway — every
		// recorded session, pid-liveness checked, with its own published gateway URL. An
		// explicit --addr/$FAK_ADDR keeps the legacy single-gateway snapshot byte-for-byte.
		if !sessionAddrExplicit(fs) {
			return runSessionIndexLS(stdout, stderr, resolveSweepRegDir(*regDir), *asJSON)
		}
		return c.renderList(stdout, stderr, *asJSON)
	case "status":
		// #3461: with no explicit address, resolve the query against the guard-session
		// index first and read the matched session's OWN published gateway (read-scoped
		// bearer). An unmatched query is not claimed (handled=false) and falls through to
		// the legacy single-gateway drive-state read below.
		if !sessionAddrExplicit(fs) {
			if code, handled := sessionIndexResolveStatus(stdout, stderr, resolveSweepRegDir(*regDir), pos[0], *asJSON); handled {
				return code
			}
		}
		rc := c.renderState(stdout, stderr, *asJSON, func() (gateway.SessionState, error) {
			return c.observe(pos[0])
		})
		// #3057: append the locally-evidenced restart chain (RESTART_HOP journal
		// rows + carryover seeds on this host) so status answers "did my session
		// restart, and did continuity survive?". Human mode only — --json stays
		// the gateway's own SessionState document — and silent for any session
		// with no restart history.
		if rc == 0 && !*asJSON {
			guardRestartChainStatusAddendum(stdout, pos[0])
		}
		return rc
	case "stop":
		return c.runVerb(stdout, stderr, *asJSON, pos[0], "stopped", *reason, *ifRev)
	case "terminate":
		// The forceful stop (#2758): unlike `stop` (drain — the in-flight turn runs to
		// completion), terminate parks the session at Terminating, which cancels the
		// arm's in-flight work at its next safe point and dispatches no further tool
		// call. Rides the same run-state verb wire, so the gateway stays vocabulary-blind.
		return c.runVerb(stdout, stderr, *asJSON, pos[0], "terminating", *reason, *ifRev)
	case "pause":
		if *all {
			return c.pauseAll(stdout, stderr, *asJSON, *reason)
		}
		return c.runVerb(stdout, stderr, *asJSON, pos[0], "paused", *reason, *ifRev)
	case "resume":
		if *all {
			return c.resumeAll(stdout, stderr, *asJSON)
		}
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
	case "context":
		return c.renderContextValue(stdout, stderr, *asJSON, pos[0])
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

func runSessionAuditAlias(stdout, stderr io.Writer, argv []string) int {
	args := normalizeSessionAuditAliasArgs(argv)
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		args = append([]string{"summary"}, args...)
	}
	switch args[0] {
	case "discover", "audit", "summary", "actions":
		if !sessionAuditArgsHaveScope(args[1:]) {
			args = append([]string{args[0], "--here"}, args[1:]...)
		}
	case "deep", "-h", "--help", "help":
	default:
		fmt.Fprintf(stderr, "fak session audit: unknown subcommand %q\n", args[0])
		fmt.Fprintln(stderr, "usage: fak session audit [summary|actions|discover|audit|deep] [session-audit flags]")
		return 2
	}
	return runSessionAudit(stdout, stderr, args)
}

func normalizeSessionAuditAliasArgs(argv []string) []string {
	out := make([]string, len(argv))
	copy(out, argv)
	for i, arg := range out {
		switch {
		case arg == "--days":
			out[i] = "--since-days"
		case strings.HasPrefix(arg, "--days="):
			out[i] = "--since-days=" + strings.TrimPrefix(arg, "--days=")
		}
	}
	return out
}

func sessionAuditArgsHaveScope(args []string) bool {
	for _, arg := range args {
		if arg == "--here" || arg == "--all" ||
			strings.HasPrefix(arg, "--here=") || strings.HasPrefix(arg, "--all=") ||
			arg == "--ns-prefix" || strings.HasPrefix(arg, "--ns-prefix=") {
			return true
		}
	}
	return false
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

// pauseAll holds every currently running or throttled session via GET /v1/fak/sessions and
// POST /v1/fak/session/{id}/run.
func (c *sessionClient) pauseAll(stdout, stderr io.Writer, asJSON bool, reason string) int {
	if reason == "" {
		reason = "operator batch pause"
	}
	list, err := c.list()
	if err != nil {
		fmt.Fprintf(stderr, "fak session pause --all: %v\n", err)
		return 1
	}
	var active []gateway.SessionState
	for _, st := range list.Sessions {
		if strings.EqualFold(st.Run, "running") || strings.EqualFold(st.Run, "throttled") {
			active = append(active, st)
		}
	}
	if len(active) == 0 {
		if asJSON {
			return emitSessionJSON(stdout, stderr, map[string]any{
				"count":    0,
				"paused":   0,
				"sessions": []gateway.SessionState{},
			})
		}
		fmt.Fprintln(stdout, "no running or throttled sessions to pause")
		return 0
	}
	var paused []gateway.SessionState
	for _, st := range active {
		next, err := c.control(st.TraceID, "run", gateway.SessionControlRequest{Run: "paused", Reason: reason})
		if err != nil {
			fmt.Fprintf(stderr, "pause %s: %v\n", st.TraceID, err)
			continue
		}
		paused = append(paused, next)
	}
	if asJSON {
		return emitSessionJSON(stdout, stderr, map[string]any{
			"count":    len(paused),
			"paused":   len(paused),
			"sessions": paused,
		})
	}
	for _, st := range paused {
		fmt.Fprintln(stdout, formatSessionState(st))
	}
	fmt.Fprintf(stdout, "%d session(s) paused\n", len(paused))
	return 0
}

// resumeAll un-pauses every currently paused session via GET /v1/fak/sessions and
// POST /v1/fak/session/{id}/run.
func (c *sessionClient) resumeAll(stdout, stderr io.Writer, asJSON bool) int {
	list, err := c.list()
	if err != nil {
		fmt.Fprintf(stderr, "fak session resume --all: %v\n", err)
		return 1
	}
	var paused []gateway.SessionState
	for _, st := range list.Sessions {
		if strings.EqualFold(st.Run, "paused") {
			paused = append(paused, st)
		}
	}
	if len(paused) == 0 {
		if asJSON {
			return emitSessionJSON(stdout, stderr, map[string]any{
				"count":    0,
				"resumed":  0,
				"sessions": []gateway.SessionState{},
			})
		}
		fmt.Fprintln(stdout, "no paused sessions")
		return 0
	}
	var resumed []gateway.SessionState
	for _, st := range paused {
		next, err := c.control(st.TraceID, "run", gateway.SessionControlRequest{Run: "running"})
		if err != nil {
			fmt.Fprintf(stderr, "resume %s: %v\n", st.TraceID, err)
			continue
		}
		resumed = append(resumed, next)
	}
	if asJSON {
		return emitSessionJSON(stdout, stderr, map[string]any{
			"count":    len(resumed),
			"resumed":  len(resumed),
			"sessions": resumed,
		})
	}
	for _, st := range resumed {
		fmt.Fprintln(stdout, formatSessionState(st))
	}
	fmt.Fprintf(stdout, "%d session(s) resumed\n", len(resumed))
	return 0
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
// applies EVERY stated axis through the gateway control API (#2762): budget —
// including the spend ceiling, which rides the budget wire — and pace, then the
// wall-clock limit ("wall") and the throughput rates ("throughput") through their
// own control verbs. An unstated axis is never written (no verb is issued for it),
// so applying an envelope that names only token budgets is byte-identical to the
// pre-#2762 two-verb apply.
// sessionEnvelopeFacet is one control mutation an envelope decomposes into: the control
// verb name (which also names it in the report's Applied list and in a failure message)
// and the request that carries it. IfRev is filled in at apply time from the chain.
type sessionEnvelopeFacet struct {
	name string
	req  gateway.SessionControlRequest
}

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

	// An envelope lands as an ORDERED sequence of control mutations. Budget is always
	// applied; the other three only when the spec actually set them. Collect the facets
	// first, then apply them below — so "which facets does this spec carry" stays
	// separate from "how a facet is applied and revision-chained".
	sb := env.SessionBudget()
	gb := gateway.SessionBudget{
		TurnsLeft:           sb.TurnsLeft,
		TokensLeft:          sb.TokensLeft,
		ContextTokensLeft:   sb.ContextTokensLeft,
		SpendMicroCentsLeft: sb.SpendMicroCentsLeft,
		SpendMicroCentsCap:  sb.SpendMicroCentsCap,
	}
	facets := []sessionEnvelopeFacet{
		{"budget", gateway.SessionControlRequest{Budget: &gb}},
	}
	if pace := env.SessionPace(); pace.MaxTokensPerTurn > 0 || pace.MinTurnGapMs > 0 {
		gp := gateway.SessionPace{MaxTokensPerTurn: pace.MaxTokensPerTurn, MinTurnGapMs: pace.MinTurnGapMs}
		facets = append(facets, sessionEnvelopeFacet{"pace", gateway.SessionControlRequest{Pace: &gp}})
	}
	if env.WallClockLimitNanos > 0 {
		gw := gateway.SessionWall{LimitNanos: env.WallClockLimitNanos}
		facets = append(facets, sessionEnvelopeFacet{"wall", gateway.SessionControlRequest{Wall: &gw}})
	}
	if !env.Throughput.IsZero() {
		gt := gateway.SessionThroughput{
			ExpectedTokensPerSec: env.Throughput.ExpectedTokensPerSec,
			MinTokensPerSec:      env.Throughput.MinTokensPerSec,
		}
		facets = append(facets, sessionEnvelopeFacet{"throughput", gateway.SessionControlRequest{Throughput: &gt}})
	}

	// The first mutation carries the caller's --if-rev precondition; each later one
	// chains the revision its predecessor returned, so a concurrent writer cannot slip
	// in between and leave a half-applied envelope. A failure stops the sequence and
	// leaves rep.Applied naming exactly the facets that did land.
	rev := ifRev
	for _, f := range facets {
		f.req.IfRev = rev
		st, err := c.control(id, f.name, f.req)
		if err != nil {
			fmt.Fprintf(stderr, "fak session envelope: apply %s: %v\n", f.name, err)
			return 1
		}
		rev = st.Rev
		rep.Applied = append(rep.Applied, f.name)
		rep.State = &st
	}
	return emitSessionEnvelopeReport(stdout, stderr, asJSON, rep)
}

func (c *sessionClient) renderContextValue(stdout, stderr io.Writer, asJSON bool, id string) int {
	snap, err := c.contextValue(id)
	if err != nil {
		fmt.Fprintf(stderr, "fak session context: %v\n", err)
		return 1
	}
	if asJSON {
		return emitSessionJSON(stdout, stderr, snap)
	}
	if len(snap.Sessions) == 0 {
		fmt.Fprintf(stdout, "%s context unknown: no managed-context value row observed yet\n", id)
		return 0
	}
	fmt.Fprintln(stdout, formatSessionContextValue(id, snap.Sessions[0]))
	return 0
}

// mergeBudget fills the axes the operator did not name from the session's current
// state and returns the rev to fence the write with. code != 0 is an early exit (the
// observe failed); the caller returns it. The observe is unconditional: the budget
// write REPLACES the whole Budget value, so the axes this verb has no flag for —
// the spend ceiling (#2762) — must be carried forward from the live state or a
// `fak session budget --turns N` would silently clear a live dollar cap.
func (c *sessionClient) mergeBudget(stderr io.Writer, id string, turns, tokens, contextTokens int, ifRev uint64) (gateway.SessionBudget, uint64, int) {
	b := gateway.SessionBudget{TurnsLeft: turns, TokensLeft: tokens, ContextTokensLeft: contextTokens}
	rev := ifRev
	cur, err := c.observe(id)
	if err != nil {
		fmt.Fprintf(stderr, "fak session budget: read current state: %v\n", err)
		return b, 0, 1
	}
	b.SpendMicroCentsLeft = cur.Budget.SpendMicroCentsLeft
	b.SpendMicroCentsCap = cur.Budget.SpendMicroCentsCap
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
	return encodeJSONOrFail(stdout, stderr, v, "fak session")
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

func formatSessionContextValue(fallbackID string, rep gateway.CtxValueReport) string {
	trace := rep.TraceID
	if trace == "" {
		trace = fallbackID
	}
	budget := contextBudgetAxis(rep.Tokens.BudgetTokens)
	headroom := "n/a"
	if rep.Tokens.Headroom != nil {
		headroom = fmt.Sprintf("%d(%.1f%%)", rep.Tokens.Headroom.Tokens, rep.Tokens.Headroom.Pct)
	}
	return fmt.Sprintf("%s context step=%s basis=%s phase=%s resident=%d peak=%d budget=%s headroom=%s turns=%d events=%d since_event=%d growth=%.1f/t reason=%q",
		trace,
		rep.StepAdvice.StepClass,
		rep.StepAdvice.Basis,
		rep.Session.Phase,
		rep.Tokens.ResidentTokens,
		rep.Tokens.PeakResidentTokens,
		budget,
		headroom,
		rep.Turns.TurnsObserved,
		rep.Turns.ContextEvents,
		rep.Turns.TurnsSinceContextEvent,
		rep.Tokens.GrowthPerTurn,
		rep.StepAdvice.Reason)
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

// contextValue reads one session's managed-context value report.
func (c *sessionClient) contextValue(id string) (gateway.CtxValueSnapshot, error) {
	var snap gateway.CtxValueSnapshot
	q := url.Values{}
	q.Set("trace", id)
	err := c.req(http.MethodGet, "/v1/fak/ctxvalue?"+q.Encode(), nil, &snap)
	return snap, err
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
// sessionHTTPError is the typed transport error the session client returns for a non-2xx
// response. It carries the raw HTTP Status and the closed reason Code (empty when the body
// names none) alongside the rendered message, so a programmatic caller — e.g. `fak doomloop
// drain --deliver` deciding delivered-vs-refused — can branch on the ADJUDICATION outcome
// via errors.As instead of string-matching a human line. Existing callers that only print
// %v/.Error() are unaffected: Error() returns the same rendered text as before.
type sessionHTTPError struct {
	Status int
	Code   string
	msg    string
}

func (e *sessionHTTPError) Error() string { return e.msg }

func httpStatusError(resp *http.Response) error {
	msg, code := readErrEnvelope(resp.Body)
	return &sessionHTTPError{Status: resp.StatusCode, Code: code, msg: renderStatusMessage(resp.StatusCode, code, msg)}
}

// renderStatusMessage builds the human line for a non-2xx session response. A route that
// names a closed reason code gets a reason-specific, truthful line first: the generic status
// text below can be actively misleading (a 409 STEER_NO_OWNED_LOOP is NOT a stale-rev
// conflict — the session is fine, the steer simply has no owned loop to land in), and the
// whole point of the honest-steer contract (#3528) is to not lie about what happened.
func renderStatusMessage(status int, code, msg string) string {
	switch code {
	case "steer_no_owned_loop":
		return fmt.Sprintf("steer not applied (STEER_NO_OWNED_LOOP): %s", msg)
	}
	switch status {
	case http.StatusConflict:
		return fmt.Sprintf("refused (409): the session is stopped (terminal) or changed under you — re-read and retry: %s", msg)
	case http.StatusNotFound:
		return fmt.Sprintf("not found (404): the session-control routes are not enabled on this gateway: %s", msg)
	case http.StatusUnauthorized:
		return fmt.Sprintf("unauthorized (401): pass --key (or set $FAK_KEY) for a gateway with --require-key: %s", msg)
	default:
		return fmt.Sprintf("gateway returned %d: %s", status, msg)
	}
}

// readErrEnvelope best-effort extracts {"error":{"message":...,"code":...}} from a body,
// falling back to the raw (bounded) text as the message so a non-JSON error page is still
// legible. The code (empty when absent) lets the caller render a reason-specific line
// instead of the coarse HTTP-status text.
func readErrEnvelope(r io.Reader) (msg, code string) {
	raw, _ := io.ReadAll(io.LimitReader(r, 8<<10))
	var env struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &env) == nil && env.Error.Message != "" {
		return env.Error.Message, env.Error.Code
	}
	return string(bytes.TrimSpace(raw)), ""
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

  fak session new [--stdin|--clipboard] [--agent claude|codex] [TEXT]
                                               launch a new guarded agent in a terminal
  fak session ls                              every recorded guard session from the local index
                                               (pid-liveness checked, each with its own gateway URL);
                                               with --addr/$FAK_ADDR, the live gateway snapshot
  fak session ls --durable [--registry PATH]  the durable C1 registry (survives restart/eviction)
  fak session ls --fleet   [--remote R]       every node's sessions (C1 + C2 refs after a git fetch)
  fak session status   <id>                   one session's drive state; without --addr/$FAK_ADDR a
                                               handle/trace prefix resolves against the guard-session
                                               index and reads that session's own /debug/vars
  fak session stop     <id> [--reason R]      request a clean stop (drain at the next boundary)
  fak session terminate <id> [--reason R]     forceful stop: cancel in-flight work at the next
                                               safe point (no new tool calls; no drain cleanup)
  fak session pause    [<id>|--all] [--reason R]  hold one session or all active sessions at the next turn boundary
  fak session resume   [<id>|--all]           un-pause one session or all paused sessions
  fak session throttle <id> [--reason R]      slow without pausing
  fak session run      <id> <state>           set running|throttled|paused|draining|terminating|stopped
  fak session budget   <id> [--turns N] [--tokens N] [--context-tokens N]  re-set the work allotment live
  fak session pace     <id> [--max-tokens N] [--gap-ms N]   re-set the per-turn throttle
  fak session envelope <id> <spec>            apply a managed-context budget envelope
                                               spec: turns=20,tokens=200000,context=64000,wall=2h,spend=$25,throughput=40/s,max-tokens=1024,gap=250ms
  fak session context  <id>                   read the managed-context value report
  fak session subscribe <id> [--since N]      re-attach to a running session's event stream by
                                               handle: drain its drive-state revisions after the
                                               cursor (lossless across a controller disconnect)
  fak session priority <id> <N>               re-set the scheduling rank (lower yields first)
  fak session recover [--thread ID] [--limit N] [--apply]
                                               preview crashed Codex sessions; --apply restores each once in a guarded tab
  fak session audit [summary|actions|discover|audit|deep] [--days N] [--json] [--fail-on high]
                                               offline recent transcript audit; defaults to summary --here
  fak session observe [--days N] [--json] [--all-workspaces]
                                               zero-config recent Codex context health for this workspace:
                                               verdict, daily input, compaction fires, and resident shed
  fak session compact-audit [--root DIR] [--since D] [--cwd S] [--json] [--scrub] [--top N] [--top-by fires|peak-resident|cumulative-input]
                                               offline compaction health from native Codex rollouts:
                                               did compaction fire, hold, and cut resident context?
                                               (append-only bytes / cumulative tokens are NOT the signal)
  fak session gate-fatigue [--ledger PATH] [--trace ID] [--threshold F] [--min-fires N] [--json]
                                               offline read-only confirm-fatigue detector: per-gate
                                               approval-without-inspection rate over the guard-stop
                                               stream, flagging RUBBER_STAMPED gates worth coarsening
                                               into a regime instead of firing per call (it only
                                               names them; it never changes a gate)
  fak session reset-diff [--in FILE] [--json] [--md]
                                               offline before/after diff for one reset
                                               (survived/summarized/expired/must-requery)
  fak session branch   <parent-image-dir> --out <branch-dir> [--id ID] [--reason R]
                       [--to-model M] [--to-host H] [--registry PATH] [--json]
                                               offline fork of a checkpoint into a new durable
                                               id (copy-on-write share of the parent's pages)
  fak session checkpoint <image-dir> --out <snap-dir> [--reason R] [--json]
                                               offline on-demand snapshot of a session
                                               (same id, copy-on-write, source unaffected)
  fak session checkpoint-witness <trace> [--repo DIR] [--ledger-dir DIR]
                       [--untracked no|normal|all] [--verify] [--json]
                                               offline two-axis checkpoint: bind the session
                                               ledger head to a git tree witness (HEAD SHA +
                                               dirty-set digest) in one record, and re-check
                                               it later — a failure names the axis that moved
                                               (tree = the workspace drifted, transcript = the
                                               ledger no longer matches the record)
  fak session fork     <parent-image-dir> --out <fork-dir> --checkpoint <branch-point-dir>
                       [--id ID] [--reason R] [--to-model M] [--to-host H] [--registry PATH] [--json]
                                               snapshot-and-branch a session into a divergent
                                               continuation: pin an immutable branch point,
                                               then fork it under a fresh trace (original untouched)
  fak session fork     <trace> [--to ID] [--ledger-dir D] [--json]
                                               DURABLE-LEDGER fork: mint a new trace pointing at
                                               the shared prefix and print it with that prefix
                                               hash (no --out/--checkpoint = this arm, not the
                                               image fork above; nothing is copied)
  fak session export   <trace> [--out FILE] [--ledger-dir D]
                       [--turns N] [--tokens N] [--context-tokens N] [--taint T] [--generation N]
                                               write the session's portable hash closure — ledger
                                               head, the entries reaching it, and the re-arm state
  fak session import   [--in FILE] [--ledger-dir D] [--json]
                                               re-arm a session on THIS host from a closure: the
                                               chain is RE-DERIVED, so a bundle altered in flight
                                               cannot reproduce its head and is refused

flags: --addr (default $FAK_ADDR or http://127.0.0.1:8080)  --key ($FAK_KEY)
       --if-rev N (optimistic-concurrency guard)  --json
       envelope: --inspect-only
       ls: --durable  --fleet  --registry PATH  --remote R (default origin)
       ls/status (no --addr): --reg-dir PATH (guard-session index dir, $FLEET_REG_DIR)
`)
}
