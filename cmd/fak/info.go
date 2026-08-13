package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
	"github.com/anthony-chaudhary/fak/internal/negframe"
	"github.com/anthony-chaudhary/fak/internal/resumemetrics"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
	"golang.org/x/term"
)

// `fak info` — the live fak-info overlay. It polls a running `fak guard` / `fak serve`
// gateway's /debug/vars and prints ONE compact, payload-free line per tick with the turn
// economy an operator running `fak guard -- claude` actually wants visible next to the
// session:
//
//   - whether re-using already-sent text is saving money (how much was re-used, how much
//     cheaper that made it, and the running total saved) — savings that otherwise only
//     surface in the per-turn --debug-stats line Claude's alt-screen buries;
//   - the SAFETY half: how many tool calls fak BLOCKED, FIXED, or SET ASIDE this session —
//     so a refused `rm -rf` or a suspicious result held back from the model is visible at a
//     glance, not only in the exit summary;
//   - how it's going: replies given, work in flight, and how long fak has been running.
//
// It is the 20% pane `fak guard --split` opens beside the 80% interactive agent pane, but
// it is a first-class command in its own right: run it by hand in a second pane against any
// fak gateway — `fak info --gateway-url http://127.0.0.1:PORT`. It NEVER launches an agent
// and writes nothing; it is a read-only poll. On loopback /debug/vars is auth-exempt, so the
// local guard gateway needs no bearer; pass --gateway-key-env for an off-box gateway behind
// --require-key.

// guardInfoVars is the subset of the gateway's /debug/vars JSON the overlay renders. The
// field/JSON-tag names mirror internal/gateway/debug.go (debugVarsResponse); JSON decode
// tolerates the many extra fields we do not surface. VCache is a pointer because the gateway
// OMITS the block until a turn carries provider cache activity (vcacheVarsFromSnapshot
// returns nil), so "no cache yet" is distinguishable from "cache proved zero saving".
type guardInfoVars struct {
	Gateway struct {
		UptimeSeconds    float64 `json:"uptime_seconds"`
		InflightRequests int64   `json:"inflight_requests"`
		VDSO             bool    `json:"vdso"`
	} `json:"gateway"`
	// Runtime is the gateway process's own live resource usage (the subset the
	// resources sub-pane renders). The gateway always reports the block, so a zero
	// NumGoroutine — impossible on a live Go process — is the "no data yet / older
	// gateway" tell the panel hides on, keeping fixtures and stale gateways quiet.
	Runtime struct {
		NumGoroutine int `json:"num_goroutine"`
		Memory       struct {
			HeapAllocBytes uint64 `json:"heap_alloc_bytes"`
			SysBytes       uint64 `json:"sys_bytes"`
			NumGC          uint32 `json:"num_gc"`
		} `json:"memory"`
	} `json:"runtime"`
	Kernel struct {
		Submits      int64 `json:"submits"`
		Admitted     int64 `json:"admitted"`
		Denies       int64 `json:"denies"`
		Transforms   int64 `json:"transforms"`
		Quarantines  int64 `json:"quarantines"`
		ResultDenies int64 `json:"result_denies"`
	} `json:"kernel"`
	Inference struct {
		Turns                 int64   `json:"turns"`
		CompletionTokens      uint64  `json:"completion_tokens"`
		OutputTokensPerSecond float64 `json:"output_tokens_per_second"`
		MeanTTFTSeconds       float64 `json:"mean_ttft_seconds"`
		InflightMaxAgeSeconds float64 `json:"inflight_max_age_seconds"`
	} `json:"inference"`
	// Sessions mirrors the gateway's /debug/vars sessions block: one row per live
	// session — the main agent and any SUB-AGENTS it spawned (a row with a parent
	// trace) — with the remaining budget axes and live wall-clock the agents
	// sub-pane renders. Empty/absent means no session registry is wired (or nothing
	// is running), and the panel stays hidden rather than fabricating rows.
	Sessions []guardInfoSession `json:"sessions"`
	Upstream struct {
		ErrorsByKind         map[string]uint64 `json:"errors_by_kind"`
		Retries              uint64            `json:"retries"`
		AuthRefreshByOutcome map[string]uint64 `json:"auth_refresh_by_outcome"`
	} `json:"upstream"`
	VCache *struct {
		CacheReadTokens int64   `json:"cache_read_tokens"`
		SavedTokenEquiv float64 `json:"saved_token_equiv"`
		HitRate         float64 `json:"hit_rate"`
		Multiplier      float64 `json:"multiplier"`
		Status          string  `json:"status"`
	} `json:"vcache"`
	CacheAttribution *guardInfoCacheAttribution `json:"cache_attribution"`
	// ManagedCache is #2190's managed-cache 1h TTL-upgrade posture: whether the lever is
	// on for this session and, when on, whether it is paying (upgrades) or inert (every
	// head refused). Rendered from /debug/vars only, mirroring debugManagedCacheVars
	// field-for-field. Pointer, like CacheAttribution, so a passive/cold session omits the
	// block rather than fabricating an all-zero posture.
	ManagedCache *guardInfoManagedCache `json:"managed_cache"`
	// PrefixStability is issue #1602's managed-context prefix-stability score: whether
	// the stable/cacheable prefix (system + tools + any protected span) survived the
	// last turn/reset boundary byte-identical. It is a pointer, like VCache, so a
	// gateway build that has not wired a live cachemeta.PrefixStabilityTracker yet
	// (today's build) omits the block entirely rather than fabricating an all-zero
	// unknown; "no field" and "explicitly unknown" stay distinguishable. See
	// guardInfoPrefixStabilityText for the rendering and `fak info --prefix-transcript`
	// for the offline compute-and-display path over a recorded session.
	PrefixStability *guardInfoPrefixStability `json:"prefix_stability"`
	// ManagedContext is issue #1577's concise context status: resident/budget tokens,
	// cache state, reset count, query-needed count, and stale-assumption count, folded
	// through internal/scorecardpane.RenderContextStatusLine. Pointer, like VCache and
	// PrefixStability, so a gateway build that has not wired a live managed-context
	// tracker yet omits the block rather than fabricating all-zero counts.
	ManagedContext *guardInfoManagedContext `json:"managed_context"`
	// Assumptions is the public, active assumption ledger mirrored from the gateway's
	// live session state. It is rendered from /debug/vars only; `fak info` never reads
	// hidden transcript text to infer what the model might be assuming.
	Assumptions []gateway.SessionAssumption `json:"assumptions,omitempty"`
	// Endpoints is the live accounts+nodes block (the fak guard "status area"): which
	// Claude seats and which serving nodes THIS session is using. Decoded straight into
	// the gateway's own shape (info.go already imports gateway). Pointer, like VCache, so
	// a gateway that set no provider — a fak serve gateway, or the accounts half of a
	// non-subscription guard session — omits it rather than fabricating an empty roster.
	Endpoints *gateway.SessionEndpoints `json:"endpoints"`
	Fleet     *gateway.SessionFleet     `json:"fleet,omitempty"`
	// Adjudication is the verdict roll-up promoted from the guard EXIT summary — the
	// HONEST source for the live safety word, because kernel.Counters (the Kernel block
	// above) is structurally ~0 on the guard Decide proxy. Nil on a cold gateway that has
	// decided nothing and observed no tokens.
	Adjudication *gateway.AdjudicationSummary `json:"adjudication"`
	// Harness is the live harness-resource block (kernel CPU/RSS/IO/net) — the /debug/vars
	// twin of the /metrics-only fak_harness_* family, so the pane can show live what the
	// exit summary prints. Nil until the host samples a session.
	Harness *gateway.SessionHarness `json:"harness"`
	// StartupReport is the full startup report the guard recorded at boot — the banner +
	// hook/auth notes an attended launch keeps compact (`fak guard --banner=auto`). The
	// --startup flag prints it verbatim; empty means the gateway recorded none (a fak
	// serve gateway, or a guard build predating the report wiring).
	StartupReport string `json:"startup_report"`
	// Watchdog is the resume/heal watchdog's process-global expvar snapshot (#3803), mirrored
	// from the gateway's /debug/vars "watchdog" block. It shares resumemetrics.Snapshot with the
	// producer (debugWatchdogVars) so emit and decode cannot drift. Pointer, like VCache: the
	// gateway OMITS the block on a cold process that never ran a watchdog (resumemetrics.Active()
	// is false), so nil means "no watchdog signal here" — distinct from a present all-zero snapshot.
	Watchdog *guardInfoWatchdog `json:"watchdog"`
}

// guardInfoWatchdog is the wire shape of the resume/heal watchdog counters the pane renders. It
// is a type alias to resumemetrics.Snapshot — the SAME type the gateway's debugWatchdogVars emits
// — so the block cannot drift between emit and decode, the strongest form of the "shared shape"
// contract guardInfoSession/guardInfoManagedCache already keep.
type guardInfoWatchdog = resumemetrics.Snapshot

// guardInfoSession is one /debug/vars sessions row. Its shape lives in internal/guardvars,
// shared with the gateway producer (debugSessionVars) so the block cannot drift between emit and
// decode. A non-empty ParentTrace marks a CONTINUATION — the same agent re-continued under a
// fresh trace — and Generation counts those re-continuations; the sub-agent axis is SpawnCount
// (see guardvars, and guardInfoAgentText for what the pane may therefore claim). The budget
// fields are what REMAINS of the seeded allotment (0 usually means "never seeded", so the
// renderer omits, not fabricates). LastTool/SpawnCount/InflightSeconds/IdleSeconds are the
// live-status activity cell (#2627); a gateway that predates them omits them and they decode to
// zero (rendered as no clause).
type guardInfoSession = guardvars.SessionVars

// guardInfoManagedContext is the wire shape of a
// scorecardpane.ContextStatusSignals, field-for-field, so a gateway that starts
// populating debugVarsResponse.ManagedContext needs no change on this side.
// Severity is the closed #1579 ContextHealthSeverity string; the renderer fails safe
// on any value scorecardpane.ContextHealthSeverity itself would (an empty or foreign
// severity renders as "(unset)"/"unknown(...)" rather than a fabricated tier).
type guardInfoManagedContext struct {
	Severity             string `json:"severity"`
	ResidentTokens       int    `json:"resident_tokens"`
	BudgetTokens         int    `json:"budget_tokens"`
	CacheState           string `json:"cache_state"`
	ResetCount           int    `json:"reset_count"`
	QueryNeededCount     int    `json:"query_needed_count"`
	StaleAssumptionCount int    `json:"stale_assumption_count"`
}

// guardInfoPrefixStability is the wire shape of a cachemeta.PrefixStabilityScore, field-
// for-field, so a gateway that starts populating debugVarsResponse.PrefixStability needs
// no change on this side. State is the closed three-state string
// ("prefix-stable"|"prefix-mutated"|"prefix-unknown"); the divergence fields are only
// meaningful when State is "prefix-mutated".
type guardInfoPrefixStability struct {
	State                     string `json:"state"`
	FirstDivergentSegment     int    `json:"first_divergent_segment"`
	FirstDivergentTokenOffset int64  `json:"first_divergent_token_offset"`
	FirstDivergentKind        string `json:"first_divergent_kind"`
	ProtectedSpanBroken       bool   `json:"protected_span_broken"`
	Reason                    string `json:"reason"`
}

// guardInfoManagedCache is the gateway's /debug/vars managed_cache posture block. Its shape
// lives in internal/guardvars, shared with the gateway producer (debugManagedCacheVars) so emit
// and decode cannot drift. Active is the 1h TTL-upgrade lever state; Inert marks the #2190
// ACTIVE-but-inert signal (lever on, zero upgrades). Reasons is the per-refusal outcome
// breakdown (refusal-only; the "upgraded" outcome lives in Upgraded).
type guardInfoManagedCache = guardvars.ManagedCacheVars

// guardInfoCacheAttribution is the /debug/vars owner-split block. Its shape lives in
// internal/guardvars, shared with the gateway producer (debugCacheAttributionVars) so emit and
// decode cannot drift.
type guardInfoCacheAttribution = guardvars.CacheAttributionVars

func writerIsTerminal(w io.Writer) bool {
	f, ok := w.(interface{ Fd() uintptr })
	return ok && term.IsTerminal(int(f.Fd()))
}

func cmdInfo(argv []string) {
	os.Exit(runInfo(os.Stdout, os.Stderr, argv))
}

// fetchGuardInfoVars GETs /debug/vars into a guardInfoVars, printing the house error and
// returning ok=false on failure — the probe the --json and --once paths share.
func fetchGuardInfoVars(c *claudeMacDebugClient, stderr io.Writer) (guardInfoVars, bool) {
	var v guardInfoVars
	if err := c.get("/debug/vars", &v); err != nil {
		fmt.Fprintln(stderr, guardInfoFetchErrorLine(c.base, err))
		return v, false
	}
	return v, true
}

// guardInfoUnreachable reports whether err is the "nothing is listening" class — a refused
// connection, a dial/DNS failure, or a connect timeout — as opposed to an HTTP error from a
// gateway that IS answering (those carry "status NNN"). It matches on the platform dial-failure
// phrasings (Windows says "actively refused"/"no connection could be made"; POSIX "connection
// refused") plus the generic dial/timeout tells. A miss just falls back to the raw error, so a
// false negative is harmless; HTTP-status errors contain none of these fragments, so a gateway
// that answers with an error is never mistaken for an absent one.
func guardInfoUnreachable(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	for _, frag := range []string{
		"connection refused",          // POSIX dial refusal
		"actively refused",            // Windows dial refusal (connectex)
		"no connection could be made", // Windows, full phrasing
		"no such host",                // DNS miss / wrong host
		"dial tcp",                    // generic dial failure
		"timeout",                     // dial / handshake timeout
		"deadline exceeded",           // context timeout
		"connection reset",            // peer went away mid-dial
	} {
		if strings.Contains(s, frag) {
			return true
		}
	}
	return false
}

// guardInfoFetchErrorLine turns a /debug/vars fetch error into the single line `fak info` prints.
// When nothing is listening at the gateway — the common first-run case, where `fak info` is run
// before (or without) a `fak guard` — it replaces the raw Go net error with a plain-words,
// actionable hint that names the URL it tried and how to start a gateway, instead of a dial
// stack phrase a non-technical watcher cannot act on. Any other error (an HTTP status from a
// gateway that IS answering, an auth refusal) is passed through verbatim so a real fault stays
// visible.
func guardInfoFetchErrorLine(base string, err error) string {
	if guardInfoUnreachable(err) {
		return fmt.Sprintf("fak info: no fak gateway answering at %s — is `fak guard` running? "+
			"start one with `fak guard -- claude`, or pass --gateway-url for a gateway elsewhere", base)
	}
	return fmt.Sprintf("fak info: %v", err)
}

// guardInfoIdleExitLine is the closing line for the #2340 --max-idle backstop: the gateway never
// answered within the idle budget, so the (typically auto-spawned) pane exits itself rather than
// polling a dead URL forever. It names the URL and the budget so an operator who finds the line in
// a scrollback knows which pane gave up and why — distinct from the "session ended" close, which
// only fires once a gateway HAD been healthy.
func guardInfoIdleExitLine(base string, maxIdle time.Duration) string {
	return fmt.Sprintf("fak info: no gateway answered at %s within %s — exiting (idle backstop)", base, maxIdle)
}

func runInfo(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "info") // #2232: overview verb -> deep help above the flag dump
	gatewayURL := fs.String("gateway-url", envOrDefault("FAK_GATEWAY_URL", "http://127.0.0.1:8080"), "fak guard/serve gateway to poll (the loopback URL fak guard prints as 'gateway')")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer; loopback /debug/vars is auth-exempt so a local guard gateway needs none")
	interval := fs.Duration("interval", 2*time.Second, "refresh interval")
	once := fs.Bool("once", false, "print one snapshot line and exit (no watch loop)")
	watch := fs.Bool("watch", false, "refresh continuously even when stdout is not a terminal")
	asJSON := fs.Bool("json", false, "emit one /debug/vars snapshot (the rendered subset) as JSON and exit")
	style := fs.String("style", envOrDefault("FAK_INFO_STYLE", "visual"), "watch-loop rendering on a TTY: visual (default — task-manager gauges + trend sparklines in stacked sub-panes) or line (a single compact status line); off a TTY both append one line per tick")
	maxIdle := fs.Duration("max-idle", 0, "issue #2340: in watch mode, self-exit (with a closing line) after the gateway has been unreachable for about this long WITHOUT ever answering — a self-terminating backstop so an auto-spawned pane (e.g. from `fak guard --split`) whose gateway never comes up cannot poll a dead URL forever and leak a terminal pane. 0 (default) polls indefinitely, the manual-run behavior. Ignored with --once/--json. Rounds up to a whole --interval tick.")
	prefixTranscript := fs.String("prefix-transcript", "", "issue #1602: score the managed-context prefix-stability of a recorded Claude Code / GLM transcript (JSONL) turn-by-turn, offline, and exit — no gateway needed")
	fromFixture := fs.String("from-fixture", "", "render the overlay OFFLINE from a recorded /debug/vars JSON snapshot (the shape `fak info --json` emits) instead of polling a live gateway — no gateway needed. The deterministic capture path (the twin of `fak console guard --journal`): pairs with --tab and --frame to draw a single static frame for docs/media. See visuals/info-overlay-capture.md.")
	tab := fs.String("tab", "cache", "with --from-fixture: which tab to render — overview, agents, accounts, cache, or safety")
	frame := fs.Bool("frame", true, "with --from-fixture: render ONE static frame and exit (no watch loop, no cursor control). The only mode --from-fixture supports today; kept as an explicit flag so a future replay mode can turn it off.")
	width := fs.Int("width", 0, "with --from-fixture: render at this fixed pane width in cells (0 = the overlay's roomy default). A fixed width makes the captured frame byte-deterministic across terminals.")
	height := fs.Int("height", 0, "with --from-fixture: render at this fixed pane height in rows (0 = roomy — the body renders in full). A fixed height crops/fits the frame exactly as a live pane of that size would.")
	negationTax := fs.Bool("negation-tax", false, "render the negation-tax debt + top offending steer strings from the source corpus and exit (offline; no gateway needed)")
	negationTaxTop := fs.Int("negation-tax-top", 5, "with --negation-tax: maximum offenders to render")
	startup := fs.Bool("startup", false, "print the guarded session's FULL startup report (the banner + hook/MCP/auth notes) and exit. This is the on-demand door to the detail an attended `fak guard -- claude` launch keeps compact: the guard records the full text on its gateway at boot, and this reads it back any time during the session (startup_report on /debug/vars). Relaunching with `fak guard --banner=full` streams it at boot instead.")
	color := fs.String("color", "auto", "colorize the info overlay on a TTY: auto (TTY && NO_COLOR unset), always (force on unless NO_COLOR), or never")
	if !parseFlags(fs, argv) {
		return 2
	}
	var onceSet, watchSet, intervalSet bool
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "once":
			onceSet = true
		case "watch":
			watchSet = true
		case "interval":
			intervalSet = true
		}
	})
	if *once && *watch {
		fmt.Fprintln(stderr, "fak info: --once and --watch cannot be used together")
		return 2
	}
	if *watch {
		*once = false
	} else if !onceSet && !watchSet && !intervalSet && !writerIsTerminal(stdout) {
		// A non-interactive consumer has nobody watching a refresh loop. Preserve
		// the interactive default, while explicit --watch/--interval opts back in.
		*once = true
	}
	if *negationTax {
		if *negationTaxTop < 0 {
			fmt.Fprintln(stderr, "fak info: --negation-tax-top must be non-negative")
			return 2
		}
		pane := scorecardpane.BuildNegframePane(negframe.AllFindings(".", nil), *negationTaxTop)
		fmt.Fprintln(stdout, scorecardpane.RenderNegframePane(pane))
		return 0
	}
	if *prefixTranscript != "" {
		return runInfoPrefixTranscript(stdout, stderr, *prefixTranscript, *asJSON)
	}
	if *fromFixture != "" {
		return runInfoFixtureFrame(stdout, stderr, *fromFixture, *tab, *frame, *width, *height)
	}
	if *interval <= 0 {
		fmt.Fprintln(stderr, "fak info: --interval must be positive")
		return 2
	}
	switch strings.ToLower(strings.TrimSpace(*style)) {
	case "visual", "line":
	default:
		fmt.Fprintf(stderr, "fak info: --style must be visual or line, got %q\n", *style)
		return 2
	}
	switch strings.ToLower(strings.TrimSpace(*color)) {
	case "auto", "always", "never":
	default:
		fmt.Fprintf(stderr, "fak info: --color must be auto, always, or never, got %q\n", *color)
		return 2
	}
	base, err := normalizeTUIAgentGatewayURL(*gatewayURL)
	if err != nil {
		fmt.Fprintf(stderr, "fak info: %v\n", err)
		return 2
	}
	// Reuse the claude-mac debug client's authenticated one-shot GET machinery (it is a
	// generic base+bearer reader); only the decoded shape and the rendered line differ.
	c := &claudeMacDebugClient{
		base: base,
		key:  strings.TrimSpace(os.Getenv(strings.TrimSpace(*keyEnv))),
		hc:   &http.Client{Timeout: 10 * time.Second},
	}

	// --startup is a one-shot read, not a watch: fetch the recorded report, print it
	// verbatim, done. A gateway that answers but recorded no report (fak serve, an older
	// guard build) gets an actionable note instead of a silent empty page.
	if *startup {
		v, ok := fetchGuardInfoVars(c, stderr)
		if !ok {
			return 1
		}
		if strings.TrimSpace(v.StartupReport) == "" {
			fmt.Fprintf(stderr, "fak info: the gateway at %s recorded no startup report (a fak serve gateway, or a fak guard built before the report was wired) — relaunch the guard with --banner=full to stream it at boot\n", c.base)
			return 1
		}
		fmt.Fprint(stdout, v.StartupReport)
		return 0
	}

	if *asJSON {
		v, ok := fetchGuardInfoVars(c, stderr)
		if !ok {
			return 1
		}
		return encodeJSONOrFail(stdout, stderr, v, "fak info")
	}
	// A TTY stdout (the normal split-pane case) lets the watch loop REDRAW one status line
	// in place instead of scrolling a new line every tick — the difference between a clean
	// dashboard and a spam-filled pane. A redirected/piped stdout keeps append-per-line so a
	// captured log stays intact. term.IsTerminal is the same probe guard uses (guard.go).
	infoTTY := !*once && writerIsTerminal(stdout)
	// The pane WIDTH lets the in-place redraw cap the status line so it can never wrap onto a
	// second row — the scroll corruptor in a narrow split pane (the --split right column). 0
	// means the size is unknown (non-TTY, or GetSize failed): "no cap", which is correct since
	// the off-TTY path appends whole lines anyway.
	// The pane HEIGHT lets the visual sub-pane block size its layout (full/compact/mini/tiny) so
	// it always fits the 20% strip without scrolling. 0 means unknown (non-TTY, or GetSize
	// failed): the visual block then assumes a roomy pane and the in-place redraw still pins it.
	infoWidth := 0
	infoHeight := 0
	if infoTTY {
		if w, h, gerr := term.GetSize(int(os.Stdout.Fd())); gerr == nil {
			if w > 0 {
				infoWidth = w
			}
			if h > 0 {
				infoHeight = h
			}
		}
	}
	return runGuardInfoOverlay(stdout, stderr, c, *interval, *once, infoTTY, infoWidth, infoHeight, *style, *color, *maxIdle)
}

// prefixTranscriptTurnResult is one line of `fak info --prefix-transcript` output: the
// turn number and the live cachemeta.PrefixStabilityScore computed for it.
type prefixTranscriptTurnResult struct {
	Turn  int                            `json:"turn"`
	Score cachemeta.PrefixStabilityScore `json:"score"`
}

// prefixTranscriptReport is the full `fak info --prefix-transcript` artifact: every
// turn's score plus the FINAL turn's score again as Summary (the state a live session
// watching this transcript would report right now).
type prefixTranscriptReport struct {
	Turns   []prefixTranscriptTurnResult    `json:"turns"`
	Summary *cachemeta.PrefixStabilityScore `json:"summary"`
}

// runInfoPrefixTranscript is issue #1602's compute-and-display entry point: it reads a
// recorded Claude Code / GLM transcript (the same JSONL shape cmd/prefixlint reads),
// runs a fresh cachemeta.PrefixStabilityTracker turn-by-turn over the PROTECTED span of
// each turn (system + tool-schema + any sealed span — the front cacheable run, capped at
// the first message/tool-result segment), and prints the three-state verdict
// (prefix-stable / prefix-mutated / prefix-unknown) for every turn plus a final summary.
// It needs no running gateway: the whole computation is local and offline, exactly like
// `fak info --json` needs no agent, only here the input is a transcript file instead of
// a live /debug/vars poll.
func runInfoPrefixTranscript(stdout, stderr io.Writer, path string, asJSON bool) int {
	turns, err := loadPrefixTranscriptTurns(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak info: --prefix-transcript: %v\n", err)
		return 1
	}
	if len(turns) == 0 {
		fmt.Fprintf(stderr, "fak info: --prefix-transcript: no assistant turns found in %s\n", path)
		return 1
	}
	tr := cachemeta.NewPrefixStabilityTracker("", abi.ScopeAgent)
	report := prefixTranscriptReport{Turns: make([]prefixTranscriptTurnResult, 0, len(turns))}
	for i, turn := range turns {
		score := tr.Observe(protectedSpanOf(turn))
		report.Turns = append(report.Turns, prefixTranscriptTurnResult{Turn: i + 1, Score: score})
		report.Summary = &report.Turns[len(report.Turns)-1].Score
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak info")
	}
	fmt.Fprintf(stdout, "prefix-stability (%d turns, %s)\n", len(report.Turns), path)
	for _, row := range report.Turns {
		fmt.Fprintf(stdout, "  turn %-4d %-14s %s\n", row.Turn, row.Score.State, row.Score.Reason)
	}
	if report.Summary != nil {
		fmt.Fprintf(stdout, "summary: %s — %s\n", report.Summary.State, report.Summary.Reason)
	}
	return 0
}

// protectedSpanOf returns the leading run of a turn that is meant to stay
// stable/cacheable — every segment up to (but not including) the first ordinary
// message/tool-result segment, INCLUDING a sealed span so a quarantined span still
// caps the baseline (mirroring frontCacheableRun's contract in prefix_stability.go,
// but keeping a sealed segment IN the compared span rather than stopping before it, so
// PrefixStabilityTracker can observe and report the seal itself rather than silently
// truncating it away).
func protectedSpanOf(turn []cachemeta.PromptSegment) []cachemeta.PromptSegment {
	end := 0
	for _, s := range turn {
		switch s.Kind {
		case cachemeta.SegStable, cachemeta.SegToolSchema, cachemeta.SegVolatile:
			end++
			continue
		case cachemeta.SegSealed:
			end++
		}
		break
	}
	return turn[:end]
}

// loadPrefixTranscriptTurns parses a Claude Code / GLM transcript JSONL into the
// per-assistant-request cumulative turns cachemeta.TurnsFromConversation expects — the
// same coarse role-classified parsing cmd/prefixlint's runJSONL uses, kept local so
// `fak info` has no dependency on the prefixlint binary.
func loadPrefixTranscriptTurns(path string) ([][]cachemeta.PromptSegment, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	type jblock struct {
		Type    string          `json:"type"`
		Text    string          `json:"text"`
		Content json.RawMessage `json:"content"`
	}
	type jrecord struct {
		Type    string `json:"type"`
		Message *struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"message"`
	}

	var parts []cachemeta.ConvPart
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var jr jrecord
		if json.Unmarshal([]byte(line), &jr) != nil || jr.Message == nil {
			continue
		}
		role := jr.Message.Role
		var s string
		if json.Unmarshal(jr.Message.Content, &s) == nil {
			parts = append(parts, cachemeta.ConvPart{Role: role, Content: []byte(s)})
			continue
		}
		var blocks []jblock
		if json.Unmarshal(jr.Message.Content, &blocks) != nil {
			continue
		}
		for _, bl := range blocks {
			switch bl.Type {
			case "text":
				parts = append(parts, cachemeta.ConvPart{Role: role, Content: []byte(bl.Text)})
			case "tool_result":
				parts = append(parts, cachemeta.ConvPart{Role: "tool_result", Content: []byte(bl.Content)})
			case "tool_use":
				parts = append(parts, cachemeta.ConvPart{Role: "tool_schema", Content: []byte(bl.Content)})
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for i := range parts {
		if parts[i].Role == "" {
			parts[i].Role = "user"
		}
	}
	return cachemeta.TurnsFromConversation(parts), nil
}

// runGuardInfoOverlay polls /debug/vars and shows one live status line until Ctrl-C — the
// second-pane companion to an interactive `fak guard` session. On a TTY (tty=true) the line
// REDRAWS in place each tick so the pane is a single-line dashboard, not a scrolling log;
// off a TTY each tick appends so a captured log stays whole. It never launches an agent. A
// transient fetch error prints a one-line note and keeps polling; once the gateway HAS been
// seen healthy, a sustained run of misses means the guarded session ended and its in-process
// gateway was torn down — so the overlay prints a closing line and exits 0, which lets the
// pane close itself rather than spin forever on a dead port. --once (once=true) is a scripted
// one-shot: it prints a single line with no header/legend and exits non-zero on a failed fetch.
func runGuardInfoOverlay(stdout, stderr io.Writer, c *claudeMacDebugClient, interval time.Duration, once, tty bool, width, height int, style string, colorMode string, maxIdleOpt ...time.Duration) int {
	// maxIdle is an OPTIONAL trailing arg (variadic) so the seven existing call sites — and the
	// non-watch one-shots — stay byte-for-byte unchanged; only the real watch launch (runInfo) and
	// the focused #2340 test pass it. idleLimit lowers that duration to a consecutive-unreachable
	// tick budget, the same unit the sawHealthy/misses close path already speaks. 0 (the default)
	// disables the backstop: the overlay polls a never-answering gateway forever, as before.
	var maxIdle time.Duration
	if len(maxIdleOpt) > 0 {
		maxIdle = maxIdleOpt[0]
	}
	idleLimit := 0
	if maxIdle > 0 && interval > 0 {
		idleLimit = int((maxIdle + interval - 1) / interval) // ceil(maxIdle/interval), so a sub-interval budget still bites
		if idleLimit < 1 {
			idleLimit = 1
		}
	}
	// --once is a scripted one-shot probe: print ONE line (or fail), no header, no legend —
	// the standing header is noise when there is no watch loop to head.
	if once {
		return runGuardInfoOnce(stdout, stderr, c)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// width/height are loop-MUTABLE from here on: the overlay re-measures the pane on a resize
	// (SIGWINCH) and on a terminal focus-in, so a tab resized while it was hidden repaints at the
	// new geometry instead of drawing the rest of the session at the stale startup size. On a TTY
	// remeasure refreshes both from term.GetSize (keeping the last good value on error); off a TTY
	// it is a no-op, preserving the width=height=0 "size unknown" contract the append path relies on.
	remeasure := func() {
		if !tty {
			return
		}
		// Read the REAL os.Stdout fd (the same source the startup measure at runInfo used), not
		// the stdout writer param — under test that writer is a bytes.Buffer with tty=true, so an
		// fd assertion on it would panic; the GetSize on a non-tty real fd simply errors and is
		// ignored, leaving the test's passed-in width/height untouched. infoTermSize is the
		// pinnable seam over that read (info_resize.go).
		if w, h, err := infoTermSize(); err == nil {
			if w > 0 {
				width = w
			}
			if h > 0 {
				height = h
			}
		}
	}

	visual, colorOn := guardInfoOverlayIntro(stdout, c, interval, tty, width, style, colorMode)

	tr := newGuardInfoTrend(guardInfoTrendCap)
	sawHealthy := false
	misses := 0
	dirty := false // a status line / visual block is currently parked on the cursor (TTY in-place mode)
	prevRows := 0  // rows the last visual frame drew (for the multi-line cursor-up redraw)

	// Focus/resize layer. It is gated to a visual-mode TTY whose STDIN is also a TTY: reading the
	// focus escape bytes (ESC [ I / ESC [ O) needs raw stdin, and the multi-line in-place redraw
	// (prevRows delta) is the surface that benefits. Off a TTY, under --style line, or with a
	// piped stdin the whole layer stays unbuilt — keyCh/resizeCh remain nil so their select cases
	// block forever (a clean no-op), and not a byte of DECSET 1004 is emitted, so those paths are
	// byte-for-byte unchanged. fs starts focused=true: a pane that opens already focused, and the
	// universal case where the terminal never reports focus at all, must run at full cadence.
	fs := focusState{focused: true}
	var keyCh <-chan infoInput
	var resizeCh <-chan struct{}
	var lastSample guardInfoVars
	haveSample := false
	// viewState is the interactive overlay's tabbed-view + glossary UI state (info_tabs.go). It is
	// consulted only on the interactive path (focusable); the non-interactive visual block ignores
	// it, so piped / non-TTY-stdin panes render byte-for-byte as before.
	var viewState infoViewState
	focusable := visual && term.IsTerminal(int(os.Stdin.Fd()))
	if focusable {
		if oldState, err := term.MakeRaw(int(os.Stdin.Fd())); err == nil {
			// MakeRaw just cleared OPOST/ONLCR on the tty DEVICE both fds share, so from here
			// on a bare "\n" no longer implies a carriage return — every multi-row write
			// (frames, notes, the exit line) would staircase across the pane (#5370). Re-add
			// the mapping at the write site for BOTH streams; they are the same terminal.
			// Reassigning the locals is enough: every later print in this function (including
			// the closures below) writes through these variables.
			stdout = newCRLFWriterTUI(stdout)
			stderr = newCRLFWriterTUI(stderr)
			// Register teardown BEFORE emitting any raw/1004/1000 byte so a panic also restores
			// cleanly. LIFO order on return: disable mouse + focus reporting, restore the cooked
			// stdin, then (the existing) trailing newline if a frame is parked. Registered after
			// `defer stop()` so it runs first.
			defer func() {
				writeMouseDisable(stdout)
				writeFocusDisable(stdout)
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
			}()
			writeFocusEnable(stdout)
			writeMouseEnable(stdout)
			keyCh = startGuardInfoInputReader(os.Stdin, stop)
			// Seed the watcher with the geometry this overlay is ABOUT to paint at, so a startup
			// measure that was already wrong (a pane sampled before its host finished laying the
			// split out) fires a repaint on the first poll instead of being mistaken for the
			// steady state.
			rc, stopResize := newInfoResizeChan(width, height)
			resizeCh = rc
			defer stopResize()
		}
		// term.MakeRaw failure (a stdin that claims TTY but rejects raw mode): skip the focus layer
		// entirely and run exactly as before — no raw mode, no 1004, no reader.
	}

	// writeFrame renders one tick. Visual mode pushes the sample into the trend ring and redraws
	// the multi-line sub-pane block in place (cursor-up + clear-down). Line mode keeps the exact
	// single-row \r\033[K redraw on a TTY, and appends one whole line per tick off a TTY so a
	// captured log stays intact. A pending needsRepaint (set by a focus-in or a resize) forces a
	// re-measure first, but KEEPS prevRows real so writeGuardInfoFrame's cursor-up + clear-down
	// still erases the old block — zeroing prevRows here would skip the clear and leave ghost rows.
	writeFrame := func(v guardInfoVars) {
		lastSample, haveSample = v, true
		// Re-measure on EVERY frame, not only when a focus-in / SIGWINCH latched a repaint.
		// The interactive frame is padded to exactly `height` rows and drawn in place, so a
		// `height` LARGER than the real pane overflows it: the pane scrolls, and what is left
		// on screen is the padded TAIL — blank rows. That is the "the 20% pane starts scrolled
		// down and looks empty" report, and it used to be permanent on Windows, which has no
		// SIGWINCH and whose --split pane is deliberately never focused (guard_split.go hands
		// focus back to the agent), so neither latch could ever fire and the geometry measured
		// once at flag-parse time stood for the whole session. One cheap syscall per tick bounds
		// any staleness to a single frame; the resize watcher below just makes it faster.
		remeasure()
		fs.needsRepaint = false
		if visual {
			tr.push(v)
			// The interactive path (a visual-mode TTY whose stdin is also a TTY) draws the TABBED
			// block — a tab bar over the active view or the glossary overlay. Every other visual
			// path keeps the plain stacked-panels block, byte-for-byte as before.
			block := renderGuardInfoVisualBlock(v, tr, width, height)
			if focusable {
				block = renderGuardInfoInteractiveBlock(viewState, v, tr, width, height)
				// Pad the interactive frame to the full pane height so it ALWAYS bottom-parks. The
				// mouse path (blockRelativeRow) translates an absolute click row assuming the block
				// fills the bottom prevRows rows; a frame shorter than the tallest one drawn anchors
				// higher than that and silently offsets every tab/chip click — the "clicks stop
				// working after opening a shorter view" bug. A constant full-height frame keeps
				// prevRows == height, so the translation is exact for every view. Visually identical
				// to the clear-to-end blanks the redraw already shows below the content.
				block = padBlockToHeight(block, height)
			}
			// Layer color on the finished, width-capped block (a no-op off a TTY / under NO_COLOR).
			block = colorizeGuardInfoBlock(block, colorOn)
			prevRows = writeGuardInfoFrame(stdout, block, prevRows)
			dirty = true
			return
		}
		if tty {
			// Redraw one row in place, capped to the pane width so the line can never wrap
			// onto a second row (a wrapped status line is the scroll corruptor: the next
			// tick's \r returns only to the start of the wrapped row, never clearing it).
			fmt.Fprintf(stdout, "\r\033[K%s", colorizeGuardInfoCompactLine(fitGuardInfoStatus(renderGuardInfoLine(v), width), colorOn))
			dirty = true
			return
		}
		fmt.Fprintf(stdout, "  %s\n", renderGuardInfoLine(v))
	}
	// A note (transient error / closing line) must not be clobbered by, or clobber, the parked
	// in-place frame: on a TTY, break to a fresh row first and reset the redraw watermark so the
	// next frame paints clean below the note.
	writeNote := func(w io.Writer, line string) {
		if tty && dirty {
			fmt.Fprintln(stdout)
			dirty = false
			prevRows = 0
		}
		fmt.Fprintln(w, line)
	}

	// Copy/freeze mode hands text selection back to the terminal (mouse reporting off) and FREEZES
	// the in-place redraw, so a watcher can drag-select and copy the pane without the next tick
	// erasing it. Ticks keep fetching (gateway-closed detection stays live); they just stop painting
	// while frozen (see emit + the keyCh arms). Only reachable on the interactive path, where keyCh
	// exists — so the mouse-toggle side effects only fire when mouse reporting was actually on.
	enterCopyMode := func() {
		if viewState.copyMode {
			return
		}
		writeMouseDisable(stdout) // stop capturing drags → native terminal selection works again
		viewState.copyMode = true
		if haveSample {
			writeFrame(lastSample) // paint the frozen frame once, now with the copy banner
		}
	}
	exitCopyMode := func() {
		if !viewState.copyMode {
			return
		}
		viewState.copyMode = false
		writeMouseEnable(stdout) // resume click-to-select-tab / wheel-scroll
		fs.needsRepaint = true
		if haveSample {
			writeFrame(lastSample) // repaint the live tab bar + body from the freshest sample
		}
	}

	// emit fetches + renders once. ok is true when a frame was rendered; stop is true when the
	// watch loop should END — the gateway was healthy and has now been unreachable for a few
	// ticks, i.e. the guarded session exited and tore its in-process gateway down.
	emit := func() (ok, stop bool) {
		var v guardInfoVars
		if err := c.get("/debug/vars", &v); err != nil {
			misses++
			if sawHealthy && misses >= 3 {
				writeNote(stdout, "fak info: gateway closed — guarded session ended")
				return false, true
			}
			// #2340 idle backstop: exit even if the gateway NEVER answered — a pane auto-spawned
			// beside a guard that failed to start, or pointed at a URL that never comes up, would
			// otherwise poll forever and leak a terminal pane. Checked AFTER the sawHealthy path so a
			// normal session that ends still gets the friendlier "session ended" line whenever the
			// idle budget is the usual several ticks. idleLimit==0 (default) skips this entirely.
			if idleLimit > 0 && misses >= idleLimit {
				writeNote(stdout, guardInfoIdleExitLine(c.base, maxIdle))
				return false, true
			}
			writeNote(stderr, guardInfoFetchErrorLine(c.base, err))
			return false, false
		}
		sawHealthy = true
		misses = 0
		if viewState.copyMode {
			lastSample, haveSample = v, true // keep the sample fresh but stay frozen for copy/select
		} else {
			writeFrame(v)
		}
		return true, false
	}

	if _, stop := emit(); stop {
		return 0
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	bg := backgroundInterval(interval) // the throttled cadence while the pane is focused-out
	for {
		select {
		case <-ctx.Done():
			if tty && dirty {
				fmt.Fprintln(stdout) // leave the cursor on a clean row on Ctrl-C
			}
			return 0
		case ev := <-keyCh:
			// A decoded interactive event: a terminal focus report, a quit byte/key, or a
			// view/glossary/mouse action (info_tabs.go). infoInputQuit means Ctrl-C / 'q' arrived as
			// a byte that raw mode swallowed before signal.NotifyContext could see it, so we cancel
			// the context ourselves and let the ctx.Done() arm run the clean teardown.
			switch ev.Kind {
			case infoInputQuit:
				// Ctrl-C / 'q'. In copy mode Ctrl-C is forgiving: it resumes the live pane instead of
				// tearing it down, so the reflexive copy keystroke can never kill the frame a watcher
				// is selecting. Outside copy mode it quits as before.
				if viewState.copyMode {
					exitCopyMode()
				} else {
					stop()
				}
			case infoInputCopyMode:
				if viewState.copyMode {
					exitCopyMode()
				} else {
					enterCopyMode()
				}
			case infoInputFocusIn:
				// Focus-IN edge: the tab may have been resized while hidden. Latch a repaint and
				// paint it now from the last sample (writeFrame re-measures + clears the old block;
				// no extra /debug/vars fetch). If no sample yet, the latch rides to the next tick.
				// Then resume the foreground cadence.
				prev := fs.focused
				fs.focused = true
				fs.needsRepaint = true
				if !prev {
					if haveSample && !viewState.copyMode {
						writeFrame(lastSample) // frozen for copy: the needsRepaint latch repaints on resume
					}
					ticker.Reset(effectiveInterval(true, interval, bg))
				}
			case infoInputFocusOut:
				// Focus-OUT edge: throttle (never pause) so a hidden tab stops churning.
				if fs.focused {
					fs.focused = false
					ticker.Reset(effectiveInterval(false, interval, bg))
				}
			default:
				if viewState.copyMode {
					break // frozen for copy: ignore view/scroll/mouse until copy mode resumes
				}
				// A view switch / glossary toggle / scroll / mouse click. Fold it into the UI state
				// and repaint from the last sample (no new fetch). A mouse click carries ABSOLUTE
				// screen coords; translate the row to block-relative before hit-testing against the
				// tab/chip layout. A scroll event's raw offset is then clamped to the active view's
				// current content (kills drift, lands End on the last page). Only repaint when the
				// state actually changed, so an inert click or an at-the-end scroll is silent.
				if ev.Kind == infoInputMouseClick {
					ev.Y = blockRelativeRow(ev.Y, height, prevRows)
				}
				next := applyInfoInput(viewState, ev)
				// A body click the tab/glossary layout did not claim may still land on a Cache-tab
				// ablation bar → toggle that mechanism's detail sub-panel, resolved against the rows
				// just rendered (so scroll/width are already baked in).
				if ev.Kind == infoInputMouseClick && next == viewState && haveSample {
					next = applyInfoCacheMechClick(next, lastSample, tr, width, height, ev.Y)
				}
				if haveSample {
					next = clampInfoScrollToSample(next, lastSample, tr, width, height)
				}
				if next != viewState {
					viewState = next
					fs.needsRepaint = true
					if haveSample {
						writeFrame(lastSample)
					}
				}
			}
			continue
		case <-resizeCh:
			// SIGWINCH (POSIX): latch a repaint and paint it now from the last sample (writeFrame
			// re-measures to the new geometry + clears the old block). If no sample yet, the latch
			// rides to the next tick.
			fs.needsRepaint = true
			if haveSample && !viewState.copyMode {
				writeFrame(lastSample) // frozen for copy: the needsRepaint latch repaints on resume
			}
			continue
		case <-ticker.C:
			if _, stop := emit(); stop {
				return 0
			}
		}
	}
}

// runGuardInfoOnce is the --once scripted one-shot: fetch /debug/vars, print ONE compact
// status line, and exit — exit non-zero on a failed fetch. No header, no legend, no loop.
func runGuardInfoOnce(stdout, stderr io.Writer, c *claudeMacDebugClient) int {
	v, ok := fetchGuardInfoVars(c, stderr)
	if !ok {
		return 1
	}
	fmt.Fprintf(stdout, "%s\n", renderGuardInfoLine(v))
	return 0
}

// guardInfoOverlayIntro resolves the two run-time display modes for the live overlay and prints
// the appropriate intro/header. visual is the DEFAULT for the live 20% pane (stacked sub-panes
// redrawn in place); it needs a TTY for cursor control, so off a TTY and under --style line the
// single compact status line is kept. colorOn gates print-time SGR color for the live block: a
// real TTY that has not set NO_COLOR (info_color.go), computed once here.
func guardInfoOverlayIntro(stdout io.Writer, c *claudeMacDebugClient, interval time.Duration, tty bool, width int, style string, colorMode string) (visual, colorOn bool) {
	visual = tty && strings.EqualFold(strings.TrimSpace(style), "visual")
	colorOn = resolveGuardInfoColorMode(colorMode, tty)
	if visual {
		// A compact intro line scrolls into history above the live block; the block carries its
		// own labels, so the verbose status-line legend is not printed in visual mode.
		fmt.Fprint(stdout, guardInfoVisualIntro(c.base, interval, width))
	} else {
		fmt.Fprint(stdout, guardInfoStartupHeader(c.base, guardInfoActiveLaneTag(), interval, width))
	}
	return visual, colorOn
}

// renderGuardInfoLine renders one compact live line in plain words a non-technical watcher
// can read at a glance. It leads with whether re-using text is saving money (the headline for
// the default `fak guard -- claude` passthrough, where the decode/serve counters stay 0
// because Anthropic generates the tokens), then what fak blocked or fixed to keep you safe,
// then a small liveness summary. Every value is the gateway's real running total.
func renderGuardInfoLine(v guardInfoVars) string {
	cache := "cache: nothing yet" // until a turn re-uses any text
	if v.VCache != nil {
		cache = guardCacheWord(v.VCache.Status, v.VCache.Multiplier, v.VCache.SavedTokenEquiv, v.VCache.HitRate)
	}
	if split := guardInfoCacheAttributionText(v); split != "" {
		cache += " · " + split
	}
	if posture := guardInfoManagedCacheText(v); posture != "" {
		cache += " · " + posture
	}
	line := fmt.Sprintf("%s · %s · replies %d · busy with %d · running %s",
		cache,
		guardSafetyWord(v),
		v.Inference.Turns, v.Gateway.InflightRequests, humanUptime(v.Gateway.UptimeSeconds))
	// The adjudication "why" rides right after the safety word so the reason a call was refused
	// stays adjacent to the count, and survives a narrow-pane width-trim (which keeps the front).
	if why := guardInfoAdjudicationDetail(v.Adjudication); why != "" {
		line += " · " + why
	}
	if ep := guardInfoEndpointsSummary(v.Endpoints); ep != "" {
		line += " · " + ep
	}
	// Resume/heal watchdog health rides here, next to the liveness summary: a dead resume layer
	// (down / gave-up) is a session-level concern, and the clause is empty on the common case where
	// no watchdog is running, so a plain passthrough line stays quiet (#3802).
	if wd := guardInfoWatchdogText(v.Watchdog); wd != "" {
		line += " · " + wd
	}
	// "turns saved": engine calls fak avoided for the agent this session (the same
	// WITNESSED FakVDSOAvoidedCalls the visual pane's trends row and the exit summary
	// report). Shown only when a call was actually avoided so a session that saved nothing
	// stays quiet; kept in "calls" so it never reads as a provider-cache token saving.
	if saved := guardInfoTurnsSaved(v); saved > 0 {
		line += fmt.Sprintf(" · saved %s calls", guardInfoShortCount(int(saved)))
	}
	if len(v.Sessions) > 0 {
		line += " · agents " + guardInfoAgentsSummary(v.Sessions)
	}
	if prefix := guardInfoPrefixStabilityText(v.PrefixStability); prefix != "" {
		line += " · " + prefix
	}
	if ctx := guardInfoManagedContextText(v.ManagedContext); ctx != "" {
		line += " · " + ctx
	}
	if assumptions := guardInfoAssumptionsText(v.Assumptions); assumptions != "" {
		line += " · " + assumptions
	}
	return line
}

// guardInfoLegend explains each part of the live line above in plain words, printed once at the
// top of the pane (guardInfoStartupHeader) so someone watching in a second pane knows what they
// are looking at without leaving the terminal.
//
// It lives HERE, immediately under renderGuardInfoLine, because that function is the legend's
// only oracle: a clause added to the line has to get its entry added here in the same diff, and
// a legend a file away from the line it documents is the one that silently goes stale. Every
// entry names three things on purpose — the plain word the line shows, the /debug/vars field the
// number is read from (so the pane, `fak info --json` and a raw curl of /debug/vars can be
// reconciled by name), and the units + range it takes. That last part is what keeps an operator
// from guessing: "replies" is a running total that only grows, "busy with" is an instantaneous
// gauge that returns to 0, and "running" restarts at 0 with the gateway.
func guardInfoLegend() string {
	var b strings.Builder
	fmt.Fprintln(&b, "what this means:")
	fmt.Fprintln(&b, "  cache  = fak re-uses text it already sent so the model costs less. \"saving money\" = the re-use has paid off; \"reused %\" = how much was re-used; \"×N cheaper\" = how much cheaper; tokens = how much you've saved so far (can start below zero).")
	fmt.Fprintln(&b, "  safety = what fak did to keep you safe: blocked an unsafe action, fixed a risky one before it ran, or set a suspicious result aside.")
	fmt.Fprintln(&b, "  floor  = the capability floor those safety counts come from: every tool call the agent proposes is adjudicated against it BEFORE it runs, so \"nothing blocked\" means the floor admitted every call so far — never that it is off. The counts are session running totals from 0: blocked = refused outright, fixed = rewritten to a safe form and then allowed, set aside = the result quarantined instead of being fed back to the model.")
	fmt.Fprintln(&b, "  why    = the reason code(s) behind those blocks — the same breakdown fak prints when the session ends, now live — plus anything held for a witness or deferred.")
	fmt.Fprintln(&b, "  saved  = \"turns saved\": engine calls fak avoided for you (served from its own cache or handled in-kernel) so the agent never had to make them — shown only once at least one was avoided.")
	fmt.Fprintln(&b, "  replies = model answers completed this session, read from inference.turns — \"turns\" is this same count's name in the JSON and in the exit summary. A whole number counting up from 0 that never decreases while the gateway is up.")
	fmt.Fprintln(&b, "  busy with = requests in flight through the gateway at this instant, read from gateway.inflight_requests. It is a gauge, not a total: it rises when a call starts and drops back to 0 when the call returns, so 0 between turns is normal and one agent usually reads 0 or 1. Staying above 0 while \"replies\" stops moving is the stuck-call tell.")
	fmt.Fprintln(&b, "  running = how long THIS gateway process has been up, read from gateway.uptime_seconds and shown as whole seconds (Ns), minutes (NmNs) or hours (NhNm). It counts up from 0 at gateway start, so a value that suddenly drops means the gateway restarted — not that the pane stalled.")
	fmt.Fprintln(&b, "  assumptions = active facts the session is relying on, with source class, confidence, expiry, and origin reference from public session/debug state.")
	fmt.Fprintln(&b, "  agents = live sessions running through this fak — the main agent plus any sub-agents it spawned, with remaining budget and wall-clock.")
	fmt.Fprintln(&b, "  watchdog = health of the layer that resumes stranded agents and restarts a dead monitor: the rollup verdict (healthy / healing / down / gave up), how many times it has ticked (alive proof), how many resumes it has proven, and whether any monitor needs attention. Absent when no watchdog is running here.")
	fmt.Fprintln(&b, "  \"nothing yet\" = no re-use has happened.")
	return b.String()
}

func guardInfoAssumptionsText(rows []gateway.SessionAssumption) string {
	rows = cleanGuardInfoAssumptions(rows)
	if len(rows) == 0 {
		return ""
	}
	limit := len(rows)
	if limit > 3 {
		limit = 3
	}
	parts := make([]string, 0, limit+1)
	for _, row := range rows[:limit] {
		detail := fmt.Sprintf("%s %s (%s, expires %s",
			guardInfoAssumptionSource(row.Source),
			guardInfoAssumptionSubject(row),
			guardInfoAssumptionConfidence(row.Confidence),
			guardInfoAssumptionExpiry(row.Expiry))
		if ref := strings.TrimSpace(row.SourceRef); ref != "" {
			detail += ", from " + ref
		}
		detail += ")"
		parts = append(parts, detail)
	}
	if extra := len(rows) - limit; extra > 0 {
		parts = append(parts, fmt.Sprintf("+%d more", extra))
	}
	return fmt.Sprintf("assumptions: %d active — %s", len(rows), strings.Join(parts, "; "))
}

func cleanGuardInfoAssumptions(rows []gateway.SessionAssumption) []gateway.SessionAssumption {
	out := make([]gateway.SessionAssumption, 0, len(rows))
	for _, row := range rows {
		row.TraceID = strings.TrimSpace(row.TraceID)
		row.Key = strings.TrimSpace(row.Key)
		row.Statement = strings.TrimSpace(row.Statement)
		row.Source = strings.TrimSpace(row.Source)
		row.Expiry = strings.TrimSpace(row.Expiry)
		row.SourceRef = strings.TrimSpace(row.SourceRef)
		if row.Key == "" && row.Statement == "" {
			continue
		}
		out = append(out, row)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].TraceID != out[j].TraceID {
			return out[i].TraceID < out[j].TraceID
		}
		if out[i].Key != out[j].Key {
			return out[i].Key < out[j].Key
		}
		if out[i].Statement != out[j].Statement {
			return out[i].Statement < out[j].Statement
		}
		return out[i].Source < out[j].Source
	})
	return out
}

func guardInfoAssumptionSource(source string) string {
	switch strings.ToLower(strings.TrimSpace(source)) {
	case "user_stated", "user-stated", "user":
		return "user-stated"
	case "inferred", "infer":
		return "inferred"
	case "queried", "query", "asked":
		return "queried"
	case "witnessed", "witness":
		return "witnessed"
	case "stale":
		return "stale"
	case "unknown":
		return "unknown"
	default:
		return "unknown"
	}
}

func guardInfoAssumptionSubject(row gateway.SessionAssumption) string {
	key := strings.TrimSpace(row.Key)
	stmt := strings.TrimSpace(row.Statement)
	switch {
	case key != "" && stmt != "" && key != stmt:
		return key + "=" + stmt
	case key != "":
		return key
	case stmt != "":
		return stmt
	default:
		return "assumption"
	}
}

func guardInfoAssumptionConfidence(v float64) string {
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return fmt.Sprintf("%.0f%%", v*100)
}

func guardInfoAssumptionExpiry(expiry string) string {
	if expiry = strings.TrimSpace(expiry); expiry != "" {
		return expiry
	}
	return "none"
}

// guardInfoManagedContextText renders issue #1577's concise managed-context status
// (resident/budget tokens, cache state, reset count, query-needed count, stale-
// assumption count) via internal/scorecardpane.RenderContextStatusLine, or "" when
// the gateway has not reported the block at all (a nil pointer — the same "no field
// yet" contract guardInfoPrefixStabilityText already keeps for PrefixStability, so a
// build that has not wired a live managed-context tracker renders nothing extra
// rather than a fabricated all-zero line).
func guardInfoManagedContextText(m *guardInfoManagedContext) string {
	if m == nil {
		return ""
	}
	return scorecardpane.RenderContextStatusLine(scorecardpane.ContextStatusSignals{
		Severity:             scorecardpane.ContextHealthSeverity(m.Severity),
		ResidentTokens:       m.ResidentTokens,
		BudgetTokens:         m.BudgetTokens,
		CacheState:           m.CacheState,
		ResetCount:           m.ResetCount,
		QueryNeededCount:     m.QueryNeededCount,
		StaleAssumptionCount: m.StaleAssumptionCount,
	})
}

// guardInfoPrefixStabilityText renders issue #1602's managed-context prefix-stability
// score in plain words: "prefix: stable", "prefix: mutated (diverged at segment N,
// offset M tokens — kind)", or nothing when the gateway has not reported the block at
// all (a nil pointer — distinct from an explicit "unknown" state, which DOES render, so
// a first-turn session visibly says "no baseline yet" instead of going silent).
func guardInfoPrefixStabilityText(p *guardInfoPrefixStability) string {
	if p == nil {
		return ""
	}
	switch cachemeta.PrefixStabilityState(p.State) {
	case cachemeta.PrefixStable:
		return "prefix: stable"
	case cachemeta.PrefixMutated:
		detail := fmt.Sprintf("prefix: mutated (diverged at segment %d, offset %d tokens", p.FirstDivergentSegment, p.FirstDivergentTokenOffset)
		if p.FirstDivergentKind != "" {
			detail += ", " + p.FirstDivergentKind
		}
		detail += ")"
		if p.ProtectedSpanBroken {
			detail += " [sealed]"
		}
		return detail
	case cachemeta.PrefixUnknown:
		return "prefix: unknown (no baseline yet)"
	default:
		return ""
	}
}

// guardInfoPrefixStabilityFromScore lowers a live cachemeta.PrefixStabilityScore into
// the wire shape rendered above — the seam a gateway (or the offline --prefix-transcript
// path below) uses to populate guardInfoVars.PrefixStability.
func guardInfoPrefixStabilityFromScore(s cachemeta.PrefixStabilityScore) *guardInfoPrefixStability {
	return &guardInfoPrefixStability{
		State:                     string(s.State),
		FirstDivergentSegment:     s.FirstDivergentSegment,
		FirstDivergentTokenOffset: s.FirstDivergentTokenOffset,
		FirstDivergentKind:        string(s.FirstDivergentKind),
		ProtectedSpanBroken:       s.ProtectedSpanBroken,
		Reason:                    s.Reason,
	}
}

// guardCacheWord puts the re-use savings in plain words. The cache lets fak send the same text
// to the model once and re-use it, which costs less. "saving money" means the re-use has more
// than paid back the small extra cost of setting it up; "not saving yet" means it has not — the
// saved-tokens number is below zero until then, so it carries its own sign. reused% is how much
// of the text was served from the cache; ×N is how many times cheaper that made those tokens.
func guardCacheWord(status string, multiplier, savedTokens, hitRate float64) string {
	lead := "cache: saving money"
	if !strings.EqualFold(strings.TrimSpace(status), "PROVEN") {
		lead = "cache: not saving yet"
	}
	return fmt.Sprintf("%s — reused %.0f%% of text, ×%.2f cheaper, %s tokens",
		lead, hitRate*100, multiplier, signedTokens(savedTokens))
}

func guardInfoCacheAttributionText(v guardInfoVars) string {
	if v.CacheAttribution == nil {
		return ""
	}
	provider := v.CacheAttribution.ProviderTokenEquiv
	fak := v.CacheAttribution.FakTokenEquiv
	total := v.CacheAttribution.TotalTokenEquiv
	providerPct, fakPct := ownerSplitPct(provider, fak, total)
	return fmt.Sprintf("split default cache %.0f%% (~%s tok) + fak %.0f%% (~%s tok)",
		providerPct, gateway.HumanTokenEquiv(provider),
		fakPct, gateway.HumanTokenEquiv(fak))
}

// guardInfoManagedCacheText renders the managed-cache posture clause for the live line: the
// 1h TTL-upgrade lever state and, when ACTIVE, whether it is paying (upgrades) or inert
// (every head refused — the #2190 misconfiguration signal, named with its dominant reason so
// the operator can see WHY the lever bought nothing). Empty when the gateway omitted the
// block (a passive, cold session), so the common passthrough line stays quiet.
func guardInfoManagedCacheText(v guardInfoVars) string {
	mc := v.ManagedCache
	if mc == nil {
		return ""
	}
	if !mc.Active {
		return "managed cache off"
	}
	// Wire-aware: the OpenAI Responses (codex) wire has no 1h-TTL lever, so a zero-upgrade
	// ACTIVE session is NOT inert — fak's lever there is the pinned prompt_cache_key. Name that
	// lever instead of the Anthropic "ACTIVE but inert (0 upgrades)" false posture, mirroring
	// bannerLine's `provider == "openai-responses"` branch. Empty wire keeps Anthropic behavior.
	if mc.WireHasNo1hTTLLever() {
		return "managed cache ACTIVE (OpenAI Responses wire: no 1h-TTL lever; stable prompt_cache_key pinned)"
	}
	if mc.Inert {
		if reason := topTTLUpgradeReason(mc.Reasons); reason != "" {
			return fmt.Sprintf("managed cache ACTIVE but inert (0 upgrades, mostly %s)", reason)
		}
		return "managed cache ACTIVE but inert (0 upgrades)"
	}
	return fmt.Sprintf("managed cache ACTIVE (%d TTL-upgraded head(s))", mc.Upgraded)
}

// topTTLUpgradeReason returns the most-frequent refusal reason in the ttl-upgrade outcome
// map, ties broken by reason name so the line is deterministic despite Go's random map
// iteration order. Empty string when the map is empty.
func topTTLUpgradeReason(reasons map[string]uint64) string {
	best, bestN := "", uint64(0)
	for r, n := range reasons {
		if n > bestN || (n == bestN && (best == "" || r < best)) {
			best, bestN = r, n
		}
	}
	return best
}

func ownerSplitPct(provider, fak, total float64) (providerPct, fakPct float64) {
	if total > 0 {
		return provider / total * 100, fak / total * 100
	}
	denom := absFloat(provider) + absFloat(fak)
	if denom == 0 {
		return 0, 0
	}
	return absFloat(provider) / denom * 100, absFloat(fak) / denom * 100
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// guardSafetyWord is the safety summary the live line + panels render. It prefers the
// operation-ledger ADJUDICATION tally — the honest source on the flagship `fak guard --
// claude` proxy, where the kernel adjudicates with Decide and so the raw kernel.Counters
// (the Kernel block) stay structurally ~0, which made the old "safety: blocked N" read a
// vacuous 0 while the floor was actually refusing calls. It falls back to the kernel
// counters for a fak serve gateway that has no adjudication block. Same formatting either
// way via guardFloorSafetyWord.
func guardSafetyWord(v guardInfoVars) string {
	if a := v.Adjudication; a != nil {
		return guardFloorSafetyWord(int64(a.Denied), int64(a.Transformed), int64(a.Quarantined), 0)
	}
	return guardFloorSafetyWord(v.Kernel.Denies, v.Kernel.Transforms, v.Kernel.Quarantines, v.Kernel.ResultDenies)
}

// guardFloorSafetyWord summarizes, in plain words, what fak did this session to keep you safe:
// blocked an unsafe tool call (Denies), fixed a risky one before it ran (Transforms), or set a
// suspicious result aside instead of feeding it to the model (Quarantines plus result-admission
// denials). A clean session reads "safety: nothing blocked" so the all-clear is visible, not a
// blank.
func guardFloorSafetyWord(denies, transforms, quarantines, resultDenies int64) string {
	setAside := quarantines + resultDenies
	if denies == 0 && transforms == 0 && setAside == 0 {
		return "safety: nothing blocked"
	}
	var parts []string
	if denies > 0 {
		parts = append(parts, fmt.Sprintf("blocked %d", denies))
	}
	if transforms > 0 {
		parts = append(parts, fmt.Sprintf("fixed %d", transforms))
	}
	if setAside > 0 {
		parts = append(parts, fmt.Sprintf("set aside %d", setAside))
	}
	return "safety: " + strings.Join(parts, ", ")
}

// guardInfoAdjudicationDetail promotes the guard EXIT summary's forensic "why" into the LIVE
// pane: the top deny/quarantine reason codes (the ByReason breakdown formatAuditSummary prints
// as "blocked: reason xN"), plus the held-for-witness (escalated) and deferred tallies. The
// safety word above answers "how many blocked"; this answers "blocked WHY" — so an operator
// watching the pane sees the reason a call was refused as it happens, not only in the closing
// summary. Empty when the gateway reported no adjudication block (a nil pointer — an older
// gateway or a fak serve gateway) or the session has refused nothing with a recorded reason and
// holds nothing pending, so a clean session stays silent rather than printing a vacuous "why".
func guardInfoAdjudicationDetail(a *gateway.AdjudicationSummary) string {
	if a == nil {
		return ""
	}
	var parts []string
	if reasons := guardInfoTopReasons(a.ByReason, 3); reasons != "" {
		parts = append(parts, "why "+reasons)
	}
	if a.Escalated > 0 {
		parts = append(parts, fmt.Sprintf("%d held for witness", a.Escalated))
	}
	if a.Deferred > 0 {
		parts = append(parts, fmt.Sprintf("%d deferred", a.Deferred))
	}
	return strings.Join(parts, " · ")
}

// guardInfoTopReasons renders the top-`limit` reason codes by count (count desc, then code asc
// for a stable order regardless of the map's iteration order), each as "code xN", with a
// "+K more" tail folding the reasons past the cap so a long tail can never scroll the pane.
// Empty for an empty/all-zero map.
func guardInfoTopReasons(byReason map[string]uint64, limit int) string {
	if len(byReason) == 0 {
		return ""
	}
	type reasonCount struct {
		code  string
		count uint64
	}
	rows := make([]reasonCount, 0, len(byReason))
	for code, n := range byReason {
		if n == 0 {
			continue
		}
		rows = append(rows, reasonCount{code, n})
	}
	if len(rows) == 0 {
		return ""
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].count != rows[j].count {
			return rows[i].count > rows[j].count
		}
		return rows[i].code < rows[j].code
	})
	shown := rows
	extra := 0
	if limit > 0 && len(rows) > limit {
		shown = rows[:limit]
		extra = len(rows) - limit
	}
	parts := make([]string, 0, len(shown)+1)
	for _, r := range shown {
		parts = append(parts, fmt.Sprintf("%s x%d", r.code, r.count))
	}
	if extra > 0 {
		parts = append(parts, fmt.Sprintf("+%d more", extra))
	}
	return strings.Join(parts, ", ")
}

// signedTokens renders a net saved-token-equiv with an explicit sign, because the value is
// NEGATIVE until cache reads repay the cache-creation premium — a "-1,234" reads correctly as
// "still in the red", where a bare "1234" would look like a saving.
func signedTokens(v float64) string {
	n := int64(v)
	if n < 0 {
		return "-" + groupThousands(-n)
	}
	return "+" + groupThousands(n)
}

// groupThousands formats a non-negative integer with comma separators (12345 -> "12,345").
func groupThousands(n int64) string {
	s := strconv.FormatInt(n, 10)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
		if len(s) > pre {
			b.WriteByte(',')
		}
	}
	for i := pre; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteByte(',')
		}
	}
	return b.String()
}
