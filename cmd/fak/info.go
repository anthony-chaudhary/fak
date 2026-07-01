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
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/gateway"
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
	Kernel struct {
		Submits      int64 `json:"submits"`
		Admitted     int64 `json:"admitted"`
		Denies       int64 `json:"denies"`
		Transforms   int64 `json:"transforms"`
		Quarantines  int64 `json:"quarantines"`
		ResultDenies int64 `json:"result_denies"`
	} `json:"kernel"`
	Inference struct {
		Turns int64 `json:"turns"`
	} `json:"inference"`
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
	// PrefixStability is issue #1602's managed-context prefix-stability score: whether
	// the stable/cacheable prefix (system + tools + any protected span) survived the
	// last turn/reset boundary byte-identical. It is a pointer, like VCache, so a
	// gateway build that has not wired a live cachemeta.PrefixStabilityTracker yet
	// (today's build) omits the block entirely rather than fabricating an all-zero
	// unknown; "no field" and "explicitly unknown" stay distinguishable. See
	// guardInfoPrefixStabilityText for the rendering and `fak info --prefix-transcript`
	// for the offline compute-and-display path over a recorded session.
	PrefixStability *guardInfoPrefixStability `json:"prefix_stability"`
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

type guardInfoCacheAttribution struct {
	ProviderTokenEquiv                        float64 `json:"provider_token_equiv"`
	FakTokenEquiv                             float64 `json:"fak_token_equiv"`
	TotalTokenEquiv                           float64 `json:"total_token_equiv"`
	ProviderPromptCacheReadTokenEquiv         float64 `json:"provider_prompt_cache_read_token_equiv"`
	ProviderPromptCacheWritePremiumTokenEquiv float64 `json:"provider_prompt_cache_write_premium_token_equiv"`
	FakCompactionShedTokens                   uint64  `json:"fak_compaction_shed_tokens"`
	FakKVPrefixReusedTokens                   uint64  `json:"fak_kv_prefix_reused_tokens"`
	FakVDSOAvoidedCalls                       uint64  `json:"fak_vdso_avoided_calls"`
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

func runInfo(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gatewayURL := fs.String("gateway-url", envOrDefault("FAK_GATEWAY_URL", "http://127.0.0.1:8080"), "fak guard/serve gateway to poll (the loopback URL fak guard prints as 'gateway')")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer; loopback /debug/vars is auth-exempt so a local guard gateway needs none")
	interval := fs.Duration("interval", 2*time.Second, "refresh interval")
	once := fs.Bool("once", false, "print one snapshot line and exit (no watch loop)")
	asJSON := fs.Bool("json", false, "emit one /debug/vars snapshot (the rendered subset) as JSON and exit")
	style := fs.String("style", envOrDefault("FAK_INFO_STYLE", "visual"), "watch-loop rendering on a TTY: visual (default — task-manager gauges + trend sparklines in stacked sub-panes) or line (a single compact status line); off a TTY both append one line per tick")
	prefixTranscript := fs.String("prefix-transcript", "", "issue #1602: score the managed-context prefix-stability of a recorded Claude Code / GLM transcript (JSONL) turn-by-turn, offline, and exit — no gateway needed")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *prefixTranscript != "" {
		return runInfoPrefixTranscript(stdout, stderr, *prefixTranscript, *asJSON)
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
	infoTTY := !*once && term.IsTerminal(int(os.Stdout.Fd()))
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
	return runGuardInfoOverlay(stdout, stderr, c, *interval, *once, infoTTY, infoWidth, infoHeight, *style)
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
func runGuardInfoOverlay(stdout, stderr io.Writer, c *claudeMacDebugClient, interval time.Duration, once, tty bool, width, height int, style string) int {
	// --once is a scripted one-shot probe: print ONE line (or fail), no header, no legend —
	// the standing header is noise when there is no watch loop to head.
	if once {
		v, ok := fetchGuardInfoVars(c, stderr)
		if !ok {
			return 1
		}
		fmt.Fprintf(stdout, "%s\n", renderGuardInfoLine(v))
		return 0
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
		// ignored, leaving the test's passed-in width/height untouched.
		if w, h, err := term.GetSize(int(os.Stdout.Fd())); err == nil {
			if w > 0 {
				width = w
			}
			if h > 0 {
				height = h
			}
		}
	}

	// Visual is the DEFAULT for the live 20% pane: stacked sub-panes (trend sparklines + a
	// task-manager gauge pane) redrawn in place. It needs a TTY for the cursor control, so off a
	// TTY (piped/redirected log) and under --style line we keep the single compact status line.
	visual := tty && strings.EqualFold(strings.TrimSpace(style), "visual")
	if visual {
		// A compact intro line scrolls into history above the live block; the block carries its
		// own labels, so the verbose status-line legend is not printed in visual mode.
		fmt.Fprint(stdout, guardInfoVisualIntro(c.base, interval, width))
	} else {
		fmt.Fprint(stdout, guardInfoStartupHeader(c.base, interval, width))
	}

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
	var keyCh <-chan focusEvent
	var resizeCh <-chan struct{}
	var lastSample guardInfoVars
	haveSample := false
	focusable := visual && term.IsTerminal(int(os.Stdin.Fd()))
	if focusable {
		if oldState, err := term.MakeRaw(int(os.Stdin.Fd())); err == nil {
			// Register teardown BEFORE emitting any raw/1004 byte so a panic also restores cleanly.
			// LIFO order on return: disable focus reporting, restore the cooked stdin, then (the
			// existing) trailing newline if a frame is parked. Registered after `defer stop()` so it
			// runs first.
			defer func() {
				writeFocusDisable(stdout)
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
			}()
			writeFocusEnable(stdout)
			keyCh = startGuardInfoFocusReader(os.Stdin, stop)
			rc, stopResize := newInfoResizeChan()
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
		if fs.needsRepaint {
			remeasure()
			fs.needsRepaint = false
		}
		if visual {
			tr.push(v)
			prevRows = writeGuardInfoFrame(stdout, renderGuardInfoVisualBlock(v, tr, width, height), prevRows)
			dirty = true
			return
		}
		if tty {
			// Redraw one row in place, capped to the pane width so the line can never wrap
			// onto a second row (a wrapped status line is the scroll corruptor: the next
			// tick's \r returns only to the start of the wrapped row, never clearing it).
			fmt.Fprintf(stdout, "\r\033[K%s", fitGuardInfoStatus(renderGuardInfoLine(v), width))
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
			writeNote(stderr, guardInfoFetchErrorLine(c.base, err))
			return false, false
		}
		sawHealthy = true
		misses = 0
		writeFrame(v)
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
			// A terminal focus report (or a raw-mode quit byte). focusQuit means Ctrl-C arrived as
			// a 0x03 byte that raw mode swallowed before signal.NotifyContext could see it, so we
			// cancel the context ourselves and let the ctx.Done() arm run the clean teardown.
			if ev == focusQuit {
				stop()
				continue
			}
			prev := fs.focused
			fs = applyFocus(fs, ev)
			switch {
			case fs.focused && !prev:
				// Focus-IN edge: the tab may have been resized while hidden. Latch a repaint and
				// paint it now from the last sample (writeFrame re-measures + clears the old block;
				// no extra /debug/vars fetch). If no sample yet, the latch rides to the next tick.
				// Then resume the foreground cadence.
				fs.needsRepaint = true
				if haveSample {
					writeFrame(lastSample)
				}
				ticker.Reset(effectiveInterval(true, interval, bg))
			case !fs.focused && prev:
				// Focus-OUT edge: throttle (never pause) so a hidden tab stops churning.
				ticker.Reset(effectiveInterval(false, interval, bg))
			}
			continue
		case <-resizeCh:
			// SIGWINCH (POSIX): latch a repaint and paint it now from the last sample (writeFrame
			// re-measures to the new geometry + clears the old block). If no sample yet, the latch
			// rides to the next tick.
			fs.needsRepaint = true
			if haveSample {
				writeFrame(lastSample)
			}
			continue
		case <-ticker.C:
			if _, stop := emit(); stop {
				return 0
			}
		}
	}
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
	line := fmt.Sprintf("%s · %s · replies %d · busy with %d · running %s",
		cache,
		guardFloorSafetyWord(v.Kernel.Denies, v.Kernel.Transforms, v.Kernel.Quarantines, v.Kernel.ResultDenies),
		v.Inference.Turns, v.Gateway.InflightRequests, humanUptime(v.Gateway.UptimeSeconds))
	if prefix := guardInfoPrefixStabilityText(v.PrefixStability); prefix != "" {
		line += " · " + prefix
	}
	return line
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

// guardInfoNarrowCols is the pane-width threshold below which the verbose multi-line legend
// is replaced by a single compact line: a narrow split pane (e.g. the --split right column)
// cannot show the 4-line legend without wrapping it, which crowds out the live status row.
const guardInfoNarrowCols = 80

// fitGuardInfoStatus formats the live status line for the in-place TTY redraw, capped so it
// can NEVER wrap the pane. It prefixes the two-space indent and trims the line to the
// remaining cell budget on a known pane width (width > 0); width <= 0 (size unknown) leaves
// the line whole — trimTUI returns its input for a non-positive budget. A wrapped status
// line is the classic single-line-redraw corruptor: once the text overflows, the terminal
// wraps it to a second row and the next tick's \r returns only to the start of that wrapped
// row, so the overflow is never cleared and the pane scrolls.
func fitGuardInfoStatus(line string, width int) string {
	return "  " + trimTUI(line, width-2)
}

// guardInfoStartupHeader is the one-time header + legend block, sized to the pane. A wide or
// unknown pane (width <= 0 or width >= guardInfoNarrowCols) keeps the full multi-line legend;
// a NARROW split pane gets a single compact legend line so the legend never wraps and crowds
// out the live status row. The header line is trimmed only when the pane width is known.
func guardInfoStartupHeader(base string, interval time.Duration, width int) string {
	var b strings.Builder
	// Lead with the running fak's identity (version + short build id) so the version is visible
	// in the pane for the whole session — the startup banner has scrolled off by then, and a
	// "+"-marked build id is the staleness tell the version alone cannot give. Putting it first
	// means a narrow-pane width-trim drops the interval hint, not the version.
	header := fmt.Sprintf("fak info · %s · %s  (every %s, Ctrl-C to stop)", guardInfoVersionTag(), base, interval)
	if width > 0 {
		header = trimTUI(header, width)
	}
	b.WriteString(header)
	b.WriteByte('\n')
	if width > 0 && width < guardInfoNarrowCols {
		b.WriteString(trimTUI(guardInfoCompactLegend(), width))
		b.WriteByte('\n')
		return b.String()
	}
	b.WriteString(guardInfoLegend())
	return b.String()
}

// guardInfoCompactLegend is the one-line guide for a narrow pane — the same plain words as the
// full guide, shortened so it fits beside the live status row instead of wrapping over it.
func guardInfoCompactLegend() string {
	return "what this means: cache = is re-using text saving money · safety = what fak blocked/fixed/set aside · replies/busy/running = how it's going"
}

// guardInfoLegend explains each part of the live line in plain words, printed once at the top
// so someone watching in a second pane knows what they are looking at without leaving the
// terminal.
func guardInfoLegend() string {
	var b strings.Builder
	fmt.Fprintln(&b, "what this means:")
	fmt.Fprintln(&b, "  cache  = fak re-uses text it already sent so the model costs less. \"saving money\" = the re-use has paid off; \"reused %\" = how much was re-used; \"×N cheaper\" = how much cheaper; tokens = how much you've saved so far (can start below zero).")
	fmt.Fprintln(&b, "  safety = what fak did to keep you safe: blocked an unsafe action, fixed a risky one before it ran, or set a suspicious result aside.")
	fmt.Fprintln(&b, "  replies = answers the model has given · busy with = work happening right now · running = how long fak has been up · \"nothing yet\" = no re-use has happened.")
	return b.String()
}
