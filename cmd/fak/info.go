package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

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
	VCache *struct {
		CacheReadTokens int64   `json:"cache_read_tokens"`
		SavedTokenEquiv float64 `json:"saved_token_equiv"`
		HitRate         float64 `json:"hit_rate"`
		Multiplier      float64 `json:"multiplier"`
		Status          string  `json:"status"`
	} `json:"vcache"`
}

func cmdInfo(argv []string) {
	os.Exit(runInfo(os.Stdout, os.Stderr, argv))
}

// fetchGuardInfoVars GETs /debug/vars into a guardInfoVars, printing the house error and
// returning ok=false on failure — the probe the --json and --once paths share.
func fetchGuardInfoVars(c *claudeMacDebugClient, stderr io.Writer) (guardInfoVars, bool) {
	var v guardInfoVars
	if err := c.get("/debug/vars", &v); err != nil {
		fmt.Fprintf(stderr, "fak info: %v\n", err)
		return v, false
	}
	return v, true
}

func runInfo(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gatewayURL := fs.String("gateway-url", envOrDefault("FAK_GATEWAY_URL", "http://127.0.0.1:8080"), "fak guard/serve gateway to poll (the loopback URL fak guard prints as 'gateway')")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer; loopback /debug/vars is auth-exempt so a local guard gateway needs none")
	interval := fs.Duration("interval", 2*time.Second, "refresh interval")
	once := fs.Bool("once", false, "print one snapshot line and exit (no watch loop)")
	asJSON := fs.Bool("json", false, "emit one /debug/vars snapshot (the rendered subset) as JSON and exit")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *interval <= 0 {
		fmt.Fprintln(stderr, "fak info: --interval must be positive")
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
	infoWidth := 0
	if infoTTY {
		if w, _, gerr := term.GetSize(int(os.Stdout.Fd())); gerr == nil && w > 0 {
			infoWidth = w
		}
	}
	return runGuardInfoOverlay(stdout, stderr, c, *interval, *once, infoTTY, infoWidth)
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
func runGuardInfoOverlay(stdout, stderr io.Writer, c *claudeMacDebugClient, interval time.Duration, once, tty bool, width int) int {
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
	fmt.Fprint(stdout, guardInfoStartupHeader(c.base, interval, width))

	sawHealthy := false
	misses := 0
	dirty := false // a status line is currently parked on the cursor row (TTY in-place mode)

	// On a TTY the status line REDRAWS in place: \r returns to column 0 and \033[K clears to
	// end of line, so each tick overwrites the previous one instead of scrolling — the pane
	// shows a live single-line dashboard, not an ever-growing log. Off a TTY (piped/redirected)
	// every tick appends its own line so a captured log stays whole.
	writeStatus := func(line string) {
		if tty {
			// Redraw one row in place, capped to the pane width so the line can never wrap
			// onto a second row (a wrapped status line is the scroll corruptor: the next
			// tick's \r returns only to the start of the wrapped row, never clearing it).
			fmt.Fprintf(stdout, "\r\033[K%s", fitGuardInfoStatus(line, width))
			dirty = true
			return
		}
		fmt.Fprintf(stdout, "  %s\n", line)
	}
	// A note (transient error / closing line) must not be clobbered by, or clobber, the parked
	// in-place status line: on a TTY, break to a fresh row first.
	writeNote := func(w io.Writer, line string) {
		if tty && dirty {
			fmt.Fprintln(stdout)
			dirty = false
		}
		fmt.Fprintln(w, line)
	}

	// emit fetches + renders once. ok is true when a line was rendered; stop is true when the
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
			writeNote(stderr, fmt.Sprintf("fak info: %v", err))
			return false, false
		}
		sawHealthy = true
		misses = 0
		writeStatus(renderGuardInfoLine(v))
		return true, false
	}

	if _, stop := emit(); stop {
		return 0
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			if tty && dirty {
				fmt.Fprintln(stdout) // leave the cursor on a clean row on Ctrl-C
			}
			return 0
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
	return fmt.Sprintf("%s · %s · replies %d · busy with %d · running %s",
		cache,
		guardFloorSafetyWord(v.Kernel.Denies, v.Kernel.Transforms, v.Kernel.Quarantines, v.Kernel.ResultDenies),
		v.Inference.Turns, v.Gateway.InflightRequests, humanUptime(v.Gateway.UptimeSeconds))
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
	header := fmt.Sprintf("fak info · %s  (every %s, Ctrl-C to stop)", base, interval)
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
