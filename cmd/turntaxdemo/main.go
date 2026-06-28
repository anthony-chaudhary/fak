// Command turntaxdemo is the live, on-box demo of how fak SAVES MODEL TURNS.
//
// It replays a frozen, class-labeled tool-call trace through the REAL kernel
// (internal/turnbench → k.Syscall), call by call, and streams each call's live
// verdict to the browser so TWO lanes advance visibly:
//
//	lane "tuned SOTA (stronger agent + framework)" — a well-built 2026 framework
//	  ELIDES the optional pure/static calls but is STILL FORCED into the recovery
//	  round-trips (a bad arg, a repeated read), so it pays the FORCED turns only
//	  (+5 on airline).
//	lane "fak (1-shot)"        — the kernel resolves the very same condition INSIDE
//	  the syscall the call arrived on (grammar repair / vDSO local serve), so no
//	  second model round-trip fires. The lane's turn counter STAYS FLAT (0).
//
// So the airline slice's tuned agent pays the 5 forced round-trips; fak deletes all
// of them and stays flat at 0 (turn_kinds in the report).
//
// Unlike cmd/demorace (which needs model weights on disk), this demo is FULLY
// SELF-CONTAINED: the trace is replayed through the kernel, not a model, so it
// reproduces identically on any box with no downloads.
//
// The safety floor (a poisoned result quarantined, a destructive op denied) is shown
// on a DELIBERATELY SEPARATE axis — never folded into the turn count — mirroring the
// two-axes discipline of TURN-TAX-RESULTS.md (§1 the moat vs §3 the efficiency upside).
//
// Serve it (browser), or self-check it (headless — CI / cross-platform dog-food):
//
//	go run ./cmd/turntaxdemo -addr 127.0.0.1:8150 -jobs 8
//	# open http://127.0.0.1:8150  → pick a suite → "Replay through the kernel"
//	#   turntax-airline → tuned +5, fak 0  (every lever fires)
//	#   turntax-happy   → all stay at 0              (the anti-inflation control, watchable)
//
//	go run ./cmd/turntaxdemo -print
//	# the 30-second point with ZERO setup: render the tuned-SOTA-vs-fak turn-tax
//	# side-by-side as a colored two-column diff in the terminal (no browser, no
//	# port). -suite picks the trace; honors NO_COLOR. (Pairs with `guarddemo
//	# -print`, the safety-axis twin.)
//
//	go run ./cmd/turntaxdemo -selfcheck
//	# browserless: replay every suite through the kernel, assert the documented
//	# turn-tax + safety-floor invariants, exit non-zero on any drift.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	// Blank-import the built-in driver list so the full ABI (resolver, vDSO,
	// adjudicator, ctx-MMU, normgate, ifc, witness, engines) is wired before
	// kernel.New / agent.Configure run inside turnbench.RunWithCalls.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

//go:embed page.html
var pageFS embed.FS

const version = "fak-turntaxdemo-v1"

var gomax = runtime.GOMAXPROCS(0)

// knownSuites are the fixtures shipped under testdata/turntax. The airline slice
// fires every lever once (baseline +9); the happy slice is the clean control (0).
var knownSuites = []struct{ ID, Label string }{
	{"turntax-airline", "airline support (every lever fires — baseline +9)"},
	{"turntax-happy", "happy path (clean — the anti-inflation control, 0)"},
}

// turnTaxDir resolves the trace fixtures the same way cmd/fak does: prefer a
// testdata/turntax under the working dir (running from fak/), else next to the
// executable, else the source-tree path when invoked via `go run ./cmd/turntaxdemo`.
func turnTaxDir() string {
	cands := []string{
		filepath.Join("testdata", "turntax"),
		filepath.Join("..", "..", "testdata", "turntax"),
	}
	if exe, err := os.Executable(); err == nil {
		cands = append(cands, filepath.Join(filepath.Dir(exe), "testdata", "turntax"))
	}
	for _, d := range cands {
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	// Fallback: climb from the working dir to the module root, so `go run
	// ./cmd/turntaxdemo` finds the fixtures from ANY subdirectory, not just the
	// repo root or cmd/turntaxdemo/. Pure-additive — the cheap relative candidates
	// above still win first; filepath.Dir stops changing at the volume root, which
	// terminates the loop on every OS (win/amd64 and darwin/arm64 alike).
	if wd, err := os.Getwd(); err == nil {
		for d := wd; ; {
			cand := filepath.Join(d, "testdata", "turntax")
			if _, err := os.Stat(cand); err == nil {
				return cand
			}
			parent := filepath.Dir(d)
			if parent == d {
				break
			}
			d = parent
		}
	}
	return cands[0]
}

func suitePath(suite string) string { return filepath.Join(turnTaxDir(), suite+".json") }

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	b, err := pageFS.ReadFile("page.html")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(b)
}

type suiteRow struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Present bool   `json:"present"`
	Calls   int    `json:"calls"`
}

func suiteRows() []suiteRow {
	out := make([]suiteRow, 0, len(knownSuites))
	for _, ks := range knownSuites {
		row := suiteRow{ID: ks.ID, Label: ks.Label}
		if t, err := turnbench.LoadTrace(suitePath(ks.ID)); err == nil {
			row.Present, row.Calls = true, len(t.Calls)
		}
		out = append(out, row)
	}
	return out
}

// handleSuites lists the available trace fixtures and which are present on disk.
func handleSuites(w http.ResponseWriter, r *http.Request) {
	out := suiteRows()
	writeJSON(w, map[string]any{
		"suites":     out,
		"gomaxprocs": gomax,
		"dir":        turnTaxDir(),
		"hardware":   demoui.Probe(), // cores / workers / accelerator this replay actually runs on
	})
}

// handleRun replays one suite through the real kernel and returns the ordered
// per-call dispositions (the lane stream) plus the rolled-up turn-tax report.
func handleRun(w http.ResponseWriter, r *http.Request) {
	suite := r.URL.Query().Get("suite")
	if suite == "" {
		suite = "turntax-airline"
	}
	t, err := turnbench.LoadTrace(suitePath(suite))
	if err != nil {
		http.Error(w, "load trace: "+err.Error(), 400)
		return
	}

	cm := turnbench.DefaultCostModel()
	// Optional cost knobs so a viewer can reprice the same fixed turns.
	if v := r.URL.Query().Get("latency_ms"); v != "" {
		if f, e := strconv.ParseFloat(v, 64); e == nil {
			cm.ModelTurnLatencyMs = f
		}
	}
	if v := r.URL.Query().Get("prompt_tokens"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			cm.PromptTokensPerTurn = n
		}
	}

	rep, calls, err := turnbench.RunWithCalls(r.Context(), t, cm)
	if err != nil {
		http.Error(w, "replay: "+err.Error(), 500)
		return
	}
	writeJSON(w, map[string]any{
		"suite":  suite,
		"calls":  calls,
		"report": rep,
	})
}

// budgetCores maps a fraction-or-percent budget + machine width to a core count in
// [1,cores] with the same grammar and half-up rounding as the model package's
// budgetToWorkers (kept local so this weights-free demo needn't import the model engine).
func budgetCores(raw float64, cores int) (int, bool) {
	if cores < 1 {
		cores = 1
	}
	frac := raw
	if frac > 1 {
		frac = frac / 100.0
	}
	if frac <= 0 || frac > 1 {
		return 0, false
	}
	n := int(float64(cores)*frac + 0.5) // half-up
	if n < 1 {
		n = 1
	}
	if n > cores {
		n = cores
	}
	return n, true
}

func main() {
	const defaultAddr = "127.0.0.1:8150"
	addr := flag.String("addr", defaultAddr, "listen address")
	basePath := demoui.BasePathFlag(flag.CommandLine, "/turntax")
	jobs := flag.Int("jobs", 0, "cap GOMAXPROCS to an ABSOLUTE core count (0 = all cores). On a shared/active box pass e.g. 8 so the demo doesn't starve other work.")
	budget := flag.Float64("budget", 0, "cap GOMAXPROCS to a FRACTION of the machine: 0.75 = 75% of the logical cores (portable; 75 or 0.75 accepted). Mutually exclusive with -jobs. 0 = unset.")
	selfcheck := flag.Bool("selfcheck", false, "run HEADLESS: replay each present suite through the kernel (the same turnbench.RunWithCalls path the browser drives), assert the documented turn-tax + safety-floor invariants, print a witness table, and exit non-zero on any mismatch. No browser, no network — the CI / cross-platform dog-food of this demo's data path.")
	print := flag.Bool("print", false, "render the SOTA-agent vs fak turn-tax side-by-side as a colored TWO-COLUMN diff in the TERMINAL (no browser, no port) and exit. The 30-second point with zero setup. Honors NO_COLOR.")
	suite := flag.String("suite", "turntax-airline", "suite for -print (turntax-airline | turntax-happy)")
	flag.Parse()
	if *jobs > 0 && *budget > 0 {
		fmt.Fprintln(os.Stderr, "-jobs and -budget are mutually exclusive (one is absolute, the other a fraction)")
		os.Exit(2)
	}
	if *jobs > 0 {
		runtime.GOMAXPROCS(*jobs)
		gomax = *jobs
	} else if *budget > 0 {
		// This demo replays through the turn-tax kernel (no model matmul), so a budget
		// only bounds the Go scheduler width. Resolve the same fraction/percent grammar
		// the bench budget uses: <=1 is a fraction, >1 is a percent.
		n, ok := budgetCores(*budget, runtime.NumCPU())
		if !ok {
			fmt.Fprintf(os.Stderr, "budget %g is not a fraction in (0,1] or a percent in (0,100]\n", *budget)
			os.Exit(2)
		}
		runtime.GOMAXPROCS(n)
		gomax = n
	}

	if *selfcheck {
		os.Exit(runSelfcheck())
	}
	if *print {
		os.Exit(runPrint(*suite))
	}

	app := http.NewServeMux()
	app.HandleFunc("/", handleIndex)
	app.HandleFunc("/api/suites", handleSuites)
	app.HandleFunc("/api/run", handleRun)
	mux := http.NewServeMux()
	base := demoui.MountWithBasePath(mux, *basePath, app)

	present := []string{}
	for _, ks := range knownSuites {
		if _, err := os.Stat(suitePath(ks.ID)); err == nil {
			present = append(present, ks.ID)
		}
	}
	bind := demoui.ListenAddr(*addr, defaultAddr)
	fmt.Fprintf(os.Stderr, "turntaxdemo %s on %s (GOMAXPROCS=%d)\n", version, demoui.LocalURL(bind, base), gomax)
	fmt.Fprintf(os.Stderr, "trace dir: %s\n", turnTaxDir())
	if base != "" {
		fmt.Fprintf(os.Stderr, "base path: %s (set by -base-path or %s)\n", base, demoui.DemoBasePathEnv)
	}
	if len(present) == 0 {
		fmt.Fprintf(os.Stderr, "WARNING: no turntax fixtures found — run from the fak/ directory\n")
	} else {
		fmt.Fprintf(os.Stderr, "suites present: %v\n", present)
	}
	fmt.Fprintf(os.Stderr, "open the URL → pick a suite → 'Replay through the kernel'\n")
	if err := http.ListenAndServe(bind, mux); err != nil {
		fmt.Fprintln(os.Stderr, "listen:", err)
		os.Exit(1)
	}
}

// suiteExpect is the documented turn-tax + safety-floor invariant for a known
// fixture — the same headline numbers cmd/fak turntax and TURN-TAX-RESULTS.md
// publish. -selfcheck pins the browser demo's own data path to them, so a
// regression (or a cross-platform divergence between, say, win/amd64 and
// mac/arm64) fails loudly with no browser in the loop.
type suiteExpect struct {
	turnsSaved, forced, elision, vdsoOff         int
	injBaseline, injFak, destrBaseline, destrFak int
}

var selfcheckExpect = map[string]suiteExpect{
	// airline fires every lever once: forced 5 (grammar + dedup) + elision 4
	// (pure + static) = 9; vDSO OFF drops to grammar-only 2; the safety floor
	// admits no injection and executes no destructive op.
	"turntax-airline": {turnsSaved: 9, forced: 5, elision: 4, vdsoOff: 2,
		injBaseline: 1, injFak: 0, destrBaseline: 1, destrFak: 0},
	// the anti-inflation control: a clean path inflates nothing.
	"turntax-happy": {turnsSaved: 0, forced: 0, elision: 0, vdsoOff: 0,
		injBaseline: 0, injFak: 0, destrBaseline: 0, destrFak: 0},
}

// runSelfcheck replays every present suite through the kernel (the exact
// turnbench.RunWithCalls path the browser drives) and asserts its documented
// invariants. Returns a process exit code: 0 = all suites reproduced their
// invariants, 1 = any mismatch / replay error / no fixtures found.
func runSelfcheck() int {
	ctx := context.Background()
	cm := turnbench.DefaultCostModel()
	fmt.Printf("== turntaxdemo -selfcheck: replay each suite through the kernel (browserless) ==\n")
	fmt.Printf("dir: %s   GOMAXPROCS=%d\n\n", turnTaxDir(), gomax)

	ran, failed := 0, 0
	for _, ks := range knownSuites {
		t, err := turnbench.LoadTrace(suitePath(ks.ID))
		if err != nil {
			fmt.Printf("  %-16s SKIP   (fixture absent under %s)\n", ks.ID, turnTaxDir())
			continue
		}
		ran++
		rep, _, err := turnbench.RunWithCalls(ctx, t, cm)
		if err != nil {
			fmt.Printf("  %-16s FAIL   replay error: %v\n", ks.ID, err)
			failed++
			continue
		}

		var check demoui.SelfcheckChecker
		if rep.ConsistencyCheck != "ok" {
			check.Notef("consistency=%q(want \"ok\")", rep.ConsistencyCheck)
		}
		if exp, known := selfcheckExpect[ks.ID]; known {
			check.Check("turns_saved", rep.Net.TurnsSaved, exp.turnsSaved)
			check.Check("forced", rep.TurnKinds.Forced, exp.forced)
			check.Check("elision", rep.TurnKinds.Elision, exp.elision)
			check.Check("vdso_off", rep.VDSOOffNet.TurnsSaved, exp.vdsoOff)
			check.Check("inj_baseline", rep.Safety.InjectionsAdmittedBaseline, exp.injBaseline)
			check.Check("inj_fak", rep.Safety.InjectionsAdmittedFak, exp.injFak)
			check.Check("destr_baseline", rep.Safety.DestructiveExecutedBaseline, exp.destrBaseline)
			check.Check("destr_fak", rep.Safety.DestructiveExecutedFak, exp.destrFak)
		}

		status := "PASS"
		if check.Failed() {
			status, failed = "FAIL", failed+1
		}
		fmt.Printf("  %-16s %s   turns_saved=%d (forced %d + elision %d)  vdso_off=%d  safety inj %d→%d destr %d→%d  consistency=%s\n",
			ks.ID, status, rep.Net.TurnsSaved, rep.TurnKinds.Forced, rep.TurnKinds.Elision,
			rep.VDSOOffNet.TurnsSaved,
			rep.Safety.InjectionsAdmittedBaseline, rep.Safety.InjectionsAdmittedFak,
			rep.Safety.DestructiveExecutedBaseline, rep.Safety.DestructiveExecutedFak,
			rep.ConsistencyCheck)
		if check.Failed() {
			fmt.Printf("                   mismatch: %v\n", check.Mismatches())
		}
	}

	fmt.Println()
	if ran == 0 {
		fmt.Printf("SELFCHECK FAILED — no fixtures found under %s (run from the repo root or the fak/ dir)\n", turnTaxDir())
		return 1
	}
	if failed > 0 {
		fmt.Printf("SELFCHECK FAILED — %d/%d suite(s) mismatched the documented invariants\n", failed, ran)
		return 1
	}
	fmt.Printf("OK — %d/%d suite(s) reproduced the documented turn-tax + safety-floor invariants (browserless)\n", ran, ran)
	return 0
}

// ---------------------------------------------------------------------------
// -print: the terminal twin of the browser turn-tax race. Same ONE live replay,
// rendered as a colored two-column diff — a tuned 2026 SOTA agent (left) racking up
// the FORCED model round-trips it must fire, beside fak (right) staying flat at 0 —
// so the efficiency point lands in ~30s with zero setup (no browser, no port). The
// HONEST headline is fak vs the TUNED 2026 SOTA agent (the forced turns).
// (Pairs with cmd/guarddemo -print, which does the same for the SAFETY axis.)
// ---------------------------------------------------------------------------

// ttPalette / ttPad / ttPaint are the small terminal-rendering helpers, kept local
// (this weights-free demo shares nothing heavier than internal/demoui with siblings).
type ttPalette struct{ red, green, dim, bold, reset string }

func ttColors() ttPalette {
	tty := false
	if fi, err := os.Stdout.Stat(); err == nil {
		tty = fi.Mode()&os.ModeCharDevice != 0
	}
	if os.Getenv("NO_COLOR") != "" || !tty {
		return ttPalette{}
	}
	return ttPalette{red: "\033[31m", green: "\033[32m", dim: "\033[2m", bold: "\033[1m", reset: "\033[0m"}
}

func (p ttPalette) paint(code, s string) string {
	if code == "" {
		return s
	}
	return code + s + p.reset
}

// ttPad pads OR truncates a plain (un-colored) string to exactly w runes, so a later
// color wrap never disturbs column alignment.
func ttPad(s string, w int) string {
	r := []rune(s)
	if len(r) > w {
		if w <= 1 {
			return string(r[:w])
		}
		return string(r[:w-1]) + "…"
	}
	return s + strings.Repeat(" ", w-len(r))
}

// runPrint replays one suite and renders the tuned-SOTA-vs-fak turn-tax side-by-side
// to stdout. Returns a process exit code (0 unless the replay errored / suite absent).
func runPrint(suite string) int {
	p := ttColors()
	t, err := turnbench.LoadTrace(suitePath(suite))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load trace %q: %v (run from the repo root)\n", suite, err)
		return 1
	}
	rep, calls, err := turnbench.RunWithCalls(context.Background(), t, turnbench.DefaultCostModel())
	if err != nil {
		fmt.Fprintf(os.Stderr, "replay: %v\n", err)
		return 1
	}

	const lw, cw, rw = 36, 22, 30
	fmt.Printf("\n  %s — suite: %s (%d calls)\n", p.paint(p.bold, "fak · the turn tax, side by side"), suite, len(calls))
	fmt.Printf("  %s\n\n", p.paint(p.dim, "same tool calls, two agents — count the wasted model round-trips"))
	fmt.Printf("  %s  %s  %s\n",
		p.paint(p.red, ttPad("tuned SOTA agent (2026)", lw)),
		ttPad("the tool call", cw),
		p.paint(p.green, "fak (1-shot kernel)"))
	fmt.Printf("  %s  %s  %s\n", strings.Repeat("─", lw), strings.Repeat("─", cw), strings.Repeat("─", rw))

	for _, d := range calls {
		var lkind, ltext, rkind, rtext string
		switch {
		case d.Axis == "turn-tax" && d.Forced:
			lkind = p.red
			if d.Class == "vdso_dedup" {
				ltext = "x +1 round-trip — dup read"
				rkind, rtext = p.green, "# 1-shot — served from cache"
			} else { // grammar
				ltext = "x +1 round-trip — bad arg"
				rkind, rtext = p.green, "# 1-shot — repaired in-syscall"
			}
		case d.Axis == "turn-tax" && d.Elision:
			lkind, ltext = p.dim, ". elided (optional call)"
			rkind, rtext = p.green, "# 1-shot — served locally"
		case d.Axis == "safety-floor":
			lkind, ltext = p.dim, "! would run it (safety)"
			rkind, rtext = p.dim, "# blocked (see guarddemo)"
		default: // control
			lkind, ltext = p.dim, ". ran"
			rkind, rtext = p.dim, ". ran"
		}
		fmt.Printf("  %s  %s  %s\n",
			p.paint(lkind, ttPad(ltext, lw)),
			p.paint(p.dim, ttPad(d.Tool, cw)),
			p.paint(rkind, rtext))
	}

	fmt.Printf("  %s  %s  %s\n", strings.Repeat("─", lw), strings.Repeat("─", cw), strings.Repeat("─", rw))
	forced := rep.TurnKinds.Forced // fak vs the TUNED agent — the honest headline
	total := rep.Net.TurnsSaved    // per-turn normalizer only (the report's per-turn rate denominator)
	leftScore := fmt.Sprintf("tuned SOTA agent: %d forced round-trip%s", forced, plural(forced))
	fmt.Printf("  %s  %s\n",
		p.paint(p.bold+p.red, ttPad(leftScore, lw+2+cw)),
		p.paint(p.bold+p.green, "fak: 0 extra round-trips"))
	// per-turn price from the report's net (turns are fixed by the kernel; only the
	// per-turn cost is a knob — the hosted-flash default, 1.5s/turn).
	perLatS, perDollar := 0.0, 0.0
	if total > 0 {
		perLatS = rep.Net.LatencySavedMs / 1000 / float64(total)
		perDollar = rep.Net.DollarsSaved / float64(total)
	}
	if forced > 0 {
		fmt.Printf("  %s\n", p.paint(p.dim, fmt.Sprintf(
			"vs even a TUNED 2026 agent, fak deletes %d forced round-trip%s ≈ %.1fs and $%.4f at hosted-flash rates (1.5s/turn).",
			forced, plural(forced), float64(forced)*perLatS, float64(forced)*perDollar)))
		fmt.Printf("  %s\n", p.paint(p.dim, "the safety floor (poison paged out, destructive op refused) is a SEPARATE axis — see `guarddemo -print`."))
	} else {
		fmt.Printf("  %s\n", p.paint(p.dim,
			"a clean happy path inflates nothing — both agents pay the same engine round-trips (the anti-inflation control). "+
				"the gap only opens on aliased/duplicate/optional calls."))
	}
	fmt.Println()
	return 0
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
