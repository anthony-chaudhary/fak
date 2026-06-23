// Command tokendemo is the self-contained demo of two CLEAR WINS the kernel's
// tool-call understanding delivers — counted call by call, each grounded in a LIVE
// kernel verdict (the kernel decides; this demo only counts). It is the token-side
// companion to cmd/guarddemo (safety), cmd/turntaxdemo (turns), and cmd/ctxdemo
// (prefix reuse): same "replay a frozen, class-labeled trace through the REAL kernel"
// discipline, two honest token meters.
//
// The two wins live on DIFFERENT layers, and the demo keeps them separate on purpose:
//
//  1. PREFILTER on a mutating /bad call — a MODEL-CONTEXT win.
//     An agent proposes a write_file / delete_path / run_shell the floor does not
//     sanction. WITHOUT the kernel the call EXECUTES and its result — a confirmation,
//     or (more often) a permission-denied stack trace — lands in the MODEL's context
//     (R tokens it must then read and react to). WITH the kernel the call is refused
//     BEFORE it runs: the big result is NEVER PRODUCED, and only a bounded
//     deny-as-value verdict (~a few dozen tokens) enters context. Those (R − verdict)
//     tokens are genuinely kept out of the model. This is the headline token win.
//
//  2. READING THE SAME FILE — a TOOL-SIDE win.
//     An agent re-reads config.yaml on turn 3 and again on turn 5. WITHOUT tool-call
//     understanding the tool RE-EXECUTES every read (re-fetch from disk / DB / API).
//     WITH it, the kernel knows the re-read is the same idempotent call and serves it
//     1-shot from the content cache (vDSO tier-2) — the tool runs ONCE, not N times.
//     HONEST BOUND: the cached content is still RETURNED to the model (gateway
//     resolveBytes re-materializes it), so this is NOT a model-context cut — it saves
//     the tool round-trip / re-execution (latency, compute, $). The model-side
//     prefill/KV reuse that would ALSO cut the re-read's tokens is a separate axis
//     (cmd/ctxdemo); the live agent loop's KV-eviction half is mechanism-proven, not
//     yet production-served (see docs/FAQ.md). So this demo counts the tool-side win
//     here and does not double-count it as model context.
//
// HONEST SCOPE. The result-token sizes are an explicit, documented per-call knob
// (`result_tokens` in the trace meta) — the same kind turnbench's CostModel and
// ctxdemo's tool sizes are; the magnitudes are illustrative, not a measured
// production bill. The DENY / DEDUP classification underneath is the kernel's own
// LIVE verdict. The SAFETY value of refusing the mutating call (the destructive op
// never runs) is cmd/guarddemo's separate axis, the moat; this counts only the token
// consequence. A clean trace (no bad calls, no re-reads) saves ZERO on both meters —
// the anti-inflation control proves the demo cannot cry wolf.
//
// The world here is a CODING-AGENT FILE WORLD, not the airline world the turntax /
// guard demos use: read_file / list_dir / search_code are allow-listed and cacheable;
// write_file / delete_path / run_shell / apply_patch fall to the structural
// DEFAULT_DENY floor (the capability the agent was never granted). It is installed via
// turnbench.RunWithWorld, so the replay, the live-verdict classification, and the
// consistency check are the exact same grounded machinery the other demos use.
//
// Headless — no model, no GPU, no browser, no network. Deterministic:
//
//	go run ./cmd/tokendemo -print
//	# the 30-second point: render the WITHOUT-kernel vs WITH-kernel ledger as a
//	# colored two-column diff in the terminal. -suite picks the trace; honors NO_COLOR.
//
//	go run ./cmd/tokendemo -print -suite reread-same-file
//	go run ./cmd/tokendemo -json            # the exact per-call ledger as JSON (all suites)
//	go run ./cmd/tokendemo -selfcheck       # browserless: replay each suite through the
//	#   kernel, assert the documented ledger invariants, exit non-zero on drift.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	// Blank-import the built-in driver list so the full ABI (resolver, vDSO,
	// adjudicator, ctx-MMU, engines) is wired before turnbench.RunWithWorld runs.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

const version = "fak-tokendemo-v1"

// denyVerdictTokens is the bounded size of a deny-as-value verdict that enters the
// model's context in place of an executed bad call's result: the tool name, the
// closed reason code (e.g. DEFAULT_DENY), a one-line human message, and the JSON
// envelope. It is a fixed, small constant BY CONSTRUCTION — the refusal vocabulary is
// closed (see docs/mcp-tool-result.md), so a deny can never balloon the way a real
// tool result can. Conservative on the high side.
const denyVerdictTokens = 32

// defaultResultTokens is the per-call result size assumed when a trace call carries
// no explicit `result_tokens` annotation — a modest read.
const defaultResultTokens = 200

var gomax = runtime.GOMAXPROCS(0)

// knownSuites are the fixtures shipped under testdata/tokendemo. Each isolates ONE
// story so the win is unambiguous; clean-control proves a benign session saves zero.
var knownSuites = []struct{ ID, Label string }{
	{"prefilter-bad-calls", "prefilter: mutating /bad calls refused before they run (win 1 — model-context tokens)"},
	{"reread-same-file", "reread: the same file served from cache, the tool not re-run (win 2 — tool round-trips)"},
	{"clean-control", "clean path (no bad calls, no re-reads — the anti-inflation control, 0)"},
}

// ---------------------------------------------------------------------------
// the coding-agent FILE WORLD — installed via turnbench.RunWithWorld.
// ---------------------------------------------------------------------------

// fileEngine is the dispatch target for the ALLOWED read tools (denied tools never
// reach it — they are refused pre-dispatch at the capability floor). It returns a
// small StatusOK result so the vDSO tier-2 cache fills on the first read of a file
// and the second identical read is a real content-cache hit. The payload bytes are
// not what this demo counts (the ledger uses the trace's `result_tokens` annotation);
// the engine exists only so the dedup path is GROUNDED in a real completion.
type fileEngine struct{}

func (fileEngine) Caps() []abi.Capability { return nil }

func (fileEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	out := []byte(`{"tool":"` + c.Tool + `","ok":true}`)
	var ref abi.Ref
	if res := abi.ActiveResolver(); res != nil {
		if r, err := res.Put(ctx, out); err == nil {
			ref = r
		}
	}
	if ref.Kind == 0 && ref.Len == 0 {
		ref = abi.Ref{Kind: abi.RefInline, Inline: out, Len: int64(len(out))}
	}
	return &abi.Result{Call: c, Payload: ref, Status: abi.StatusOK, Meta: map[string]string{"engine": "localtools"}}, nil
}

// configureFileWorld installs the coding-agent file world: the read family is
// affirmatively allowed (and the read calls in the traces carry the read-only +
// idempotent hints that make them vDSO-cacheable); everything write-shaped is left
// OFF the allow-list, so write_file / delete_path / run_shell / apply_patch fall to
// the structural fail-closed DEFAULT_DENY floor (the strongest refusal — the
// capability was never wired up, not a pattern that could be evaded). It overwrites
// the process-global drivers, replacing whatever world was previously installed; the
// engine name "localtools" matches the dispatch target turnbench's replay builds.
func configureFileWorld() {
	abi.RegisterEngine("localtools", fileEngine{})
	adjudicator.Default.SetPolicy(adjudicator.Policy{
		// The read-only tool family a coding agent inspects a repo with. AllowPrefix
		// covers read_file / list_dir / search_code by name shape; none is write-shaped,
		// so each is vDSO fast-path eligible under its read-only+idempotent hints.
		AllowPrefix: []string{"read_", "list_", "search_", "get_", "find_"},
		// Nothing else is allowed: write_file / delete_path / run_shell / apply_patch
		// are unsanctioned AND write-shaped, so they hit the fail-closed DEFAULT_DENY
		// floor and are counted destructive (the baseline would have executed them).
	})
}

// ---------------------------------------------------------------------------
// the two-meter ledger.
//
//	model-context meter: tokens the MODEL must ingest. The prefilter win lives here
//	  (a denied bad call's result is never produced; only the deny verdict enters).
//	  A dedup'd re-read does NOT save here — the cached content is still returned to
//	  the model — so its model-context columns are equal on both arms (honest).
//	tool-side meter: tool round-trips / re-executions. The re-read win lives here
//	  (the tool runs once, not N times); the bytes are served from cache, not re-fetched.
// ---------------------------------------------------------------------------

// callLedger is one replayed call's contribution, on both meters.
type callLedger struct {
	Index        int    `json:"index"`
	Tool         string `json:"tool"`
	Class        string `json:"class"`         // the kernel's live verdict class
	Axis         string `json:"axis"`          // turn-tax | safety-floor | control
	ResultTokens int    `json:"result_tokens"` // R — the result this call carries
	// model-context meter
	CtxWithout int `json:"ctx_without"` // model context tokens, raw loop
	CtxWith    int `json:"ctx_with"`    // model context tokens, behind fak
	CtxSaved   int `json:"ctx_saved"`
	// tool-side meter
	ToolRanWithout int    `json:"tool_ran_without"` // tool executions, raw loop (0/1)
	ToolRanWith    int    `json:"tool_ran_with"`    // tool executions, behind fak (0 on a cache hit)
	Why            string `json:"why"`
}

// ledger is the rolled-up per-suite accounting on both meters.
type ledger struct {
	Suite string       `json:"suite"`
	Calls []callLedger `json:"calls"`
	// model-context meter (the prefilter win lives here)
	CtxWithout        int `json:"ctx_without_total"`
	CtxWith           int `json:"ctx_with_total"`
	ContextTokensKept int `json:"context_tokens_kept_out"` // headline win 1: sum of CtxSaved (denied bad calls)
	Denies            int `json:"denies"`
	// tool-side meter (the re-read win lives here)
	RoundtripsCollapsed int `json:"roundtrips_collapsed"`   // win 2: re-reads served from cache (tool not re-run)
	ToolTokensFromCache int `json:"tool_tokens_from_cache"` // tool-result tokens served from cache instead of re-fetched
	ToolRunsWithout     int `json:"tool_runs_without"`      // tool executions in the raw loop
	ToolRunsWith        int `json:"tool_runs_with"`         // tool executions behind fak (cache hits + denied calls do not run)
	Dedups              int `json:"dedups"`
	Passes              int `json:"passes"`
	DenyVerdictTokens   int `json:"deny_verdict_tokens"`
}

// resultTokens reads the explicit per-call `result_tokens` annotation (the modeled,
// documented knob), falling back to defaultResultTokens when absent or malformed.
func resultTokens(c turnbench.Call) int {
	if c.Meta != nil {
		if s, ok := c.Meta["result_tokens"]; ok {
			if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && n >= 0 {
				return n
			}
		}
	}
	return defaultResultTokens
}

// buildLedger replays the suite through the REAL kernel under the file world and
// scores both meters from the LIVE dispositions zipped with the trace's result-size
// annotations. The classification (deny / dedup / pass) is the kernel's; only the
// token arithmetic is this demo's.
func buildLedger(ctx context.Context, suite string) (ledger, error) {
	t, err := turnbench.LoadTrace(suitePath(suite))
	if err != nil {
		return ledger{}, err
	}
	_, disp, err := turnbench.RunWithWorld(ctx, t, turnbench.DefaultCostModel(), configureFileWorld)
	if err != nil {
		return ledger{}, err
	}
	l := ledger{Suite: suite, DenyVerdictTokens: denyVerdictTokens}
	for i, d := range disp {
		R := defaultResultTokens
		if i < len(t.Calls) {
			R = resultTokens(t.Calls[i])
		}
		row := callLedger{Index: d.Index, Tool: d.Tool, Class: d.Class, Axis: d.Axis, ResultTokens: R}
		switch d.Class {
		case "deny":
			// Win 1 (model-context): refused before it runs. The raw loop executes it →
			// R result tokens enter the MODEL's context; behind fak the result is never
			// produced and only the bounded deny verdict enters. The tool also never runs.
			row.CtxWithout, row.CtxWith = R, denyVerdictTokens
			row.CtxSaved = R - denyVerdictTokens
			row.ToolRanWithout, row.ToolRanWith = 1, 0
			row.Why = "prefilter — refused pre-execution; only a bounded deny verdict enters the model, not the op's " + itoa(R) + "-tok result"
			if row.CtxSaved > 0 {
				l.ContextTokensKept += row.CtxSaved
			}
			l.Denies++
		case "vdso_dedup":
			// Win 2 (tool-side): the same idempotent read, served 1-shot from the content
			// cache — the tool does NOT re-execute. HONEST: the cached content is still
			// RETURNED to the model, so the model-context columns are EQUAL on both arms
			// (no model-context cut here); the win is the eliminated tool round-trip.
			row.CtxWithout, row.CtxWith = R, R
			row.CtxSaved = 0
			row.ToolRanWithout, row.ToolRanWith = 1, 0
			row.Why = "dedup — same file already read; the tool is served from cache (not re-run). The content is still returned to the model (a tool-side win, not a model-context cut)."
			l.RoundtripsCollapsed++
			l.ToolTokensFromCache += R
			l.Dedups++
		default:
			// Control: a first read / legitimate call BOTH arms pay identically, and the
			// tool runs once on both (fak is not free on real work — that honesty is the point).
			row.CtxWithout, row.CtxWith = R, R
			row.CtxSaved = 0
			row.ToolRanWithout, row.ToolRanWith = 1, 1
			row.Why = "control — legitimate work; both arms ingest it and the tool runs once on both"
			l.Passes++
		}
		l.CtxWithout += row.CtxWithout
		l.CtxWith += row.CtxWith
		l.ToolRunsWithout += row.ToolRanWithout
		l.ToolRunsWith += row.ToolRanWith
		l.Calls = append(l.Calls, row)
	}
	return l, nil
}

// ---------------------------------------------------------------------------
// fixture resolution (mirrors cmd/turntaxdemo's turnTaxDir climb).
// ---------------------------------------------------------------------------

func tokenDir() string {
	cands := []string{
		filepath.Join("testdata", "tokendemo"),
		filepath.Join("..", "..", "testdata", "tokendemo"),
	}
	if exe, err := os.Executable(); err == nil {
		cands = append(cands, filepath.Join(filepath.Dir(exe), "testdata", "tokendemo"))
	}
	for _, d := range cands {
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	if wd, err := os.Getwd(); err == nil {
		for d := wd; ; {
			cand := filepath.Join(d, "testdata", "tokendemo")
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

func suitePath(suite string) string { return filepath.Join(tokenDir(), suite+".json") }

func itoa(n int) string { return strconv.Itoa(n) }

func main() {
	jobs := flag.Int("jobs", 0, "cap GOMAXPROCS to an ABSOLUTE core count (0 = all cores). On a shared/active box pass e.g. 8 so the demo doesn't starve other work.")
	print := flag.Bool("print", false, "render the WITHOUT-kernel vs WITH-kernel ledger as a colored TWO-COLUMN diff in the TERMINAL and exit. The 30-second point with zero setup. -suite picks the trace; honors NO_COLOR.")
	asJSON := flag.Bool("json", false, "emit the exact per-call ledger as JSON (all suites, or just -suite) and exit.")
	selfcheck := flag.Bool("selfcheck", false, "run HEADLESS: replay each suite through the kernel (the same turnbench.RunWithWorld path -print/-json drive), assert the documented ledger invariants, and exit non-zero on any drift. The CI / cross-platform dog-food of this demo's data path.")
	suite := flag.String("suite", "prefilter-bad-calls", "suite for -print / -json (prefilter-bad-calls | reread-same-file | clean-control)")
	flag.Parse()
	if *jobs > 0 {
		runtime.GOMAXPROCS(*jobs)
		gomax = *jobs
	}

	switch {
	case *selfcheck:
		os.Exit(runSelfcheck())
	case *asJSON:
		os.Exit(runJSON(*suite))
	case *print:
		os.Exit(runPrint(*suite))
	default:
		// No mode flag: this demo has no browser surface — point the operator at the
		// three headless modes (the value here is the numbers, not a live server).
		fmt.Fprintf(os.Stderr, "tokendemo %s — the tool-call token ledger (no model, no browser)\n", version)
		fmt.Fprintf(os.Stderr, "trace dir: %s\n", tokenDir())
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -print [-suite %s]\n", knownSuites[0].ID)
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -json\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -selfcheck\n")
		os.Exit(2)
	}
}

// runJSON emits the ledger(s) as JSON. suite "" / "all" emits every present suite.
func runJSON(suite string) int {
	ctx := context.Background()
	var out []ledger
	for _, ks := range knownSuites {
		if suite != "" && suite != "all" && ks.ID != suite {
			continue
		}
		l, err := buildLedger(ctx, ks.ID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", ks.ID, err)
			return 1
		}
		out = append(out, l)
	}
	if len(out) == 0 {
		fmt.Fprintf(os.Stderr, "no such suite %q (try: prefilter-bad-calls, reread-same-file, clean-control, all)\n", suite)
		return 2
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	fmt.Println(string(b))
	return 0
}

// ---------------------------------------------------------------------------
// -print: the terminal two-column diff (WITHOUT kernel vs WITH kernel).
// ---------------------------------------------------------------------------

type palette struct{ red, green, dim, bold, reset string }

func colors() palette {
	tty := false
	if fi, err := os.Stdout.Stat(); err == nil {
		tty = fi.Mode()&os.ModeCharDevice != 0
	}
	if os.Getenv("NO_COLOR") != "" || !tty {
		return palette{}
	}
	return palette{red: "\033[31m", green: "\033[32m", dim: "\033[2m", bold: "\033[1m", reset: "\033[0m"}
}

func (p palette) paint(code, s string) string {
	if code == "" {
		return s
	}
	return code + s + p.reset
}

// padTrim pads OR truncates a plain (un-colored) string to exactly w runes so a later
// color wrap never disturbs column alignment.
func padTrim(s string, w int) string {
	r := []rune(s)
	if len(r) > w {
		if w <= 1 {
			return string(r[:w])
		}
		return string(r[:w-1]) + "…"
	}
	return s + strings.Repeat(" ", w-len(r))
}

// commaInt formats an int with thousands separators.
func commaInt(n int) string {
	s := strconv.Itoa(n)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	if neg {
		return "-" + string(out)
	}
	return string(out)
}

func runPrint(suite string) int {
	p := colors()
	l, err := buildLedger(context.Background(), suite)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build ledger %q: %v (run from the repo root)\n", suite, err)
		return 1
	}

	const lw, cw, rw = 32, 26, 32
	fmt.Printf("\n  %s — suite: %s (%d calls)\n", p.paint(p.bold, "fak · the tool-call token ledger"), suite, len(l.Calls))
	fmt.Printf("  %s\n\n", p.paint(p.dim, "same tools, two loops — what reaches the model, and when the tool runs"))
	fmt.Printf("  %s  %s  %s\n",
		p.paint(p.red, padTrim("WITHOUT fak (raw loop)", lw)),
		padTrim("the tool call", cw),
		p.paint(p.green, "WITH fak (kernel)"))
	fmt.Printf("  %s  %s  %s\n", strings.Repeat("─", lw), strings.Repeat("─", cw), strings.Repeat("─", rw))

	for _, c := range l.Calls {
		var lkind, ltext, rkind, rtext string
		switch c.Class {
		case "deny":
			lkind = p.red
			ltext = "x runs it → +" + commaInt(c.CtxWithout) + " model tok"
			rkind, rtext = p.green, "# refused → +"+commaInt(c.CtxWith)+" model tok (verdict)"
		case "vdso_dedup":
			lkind = p.red
			ltext = "x re-runs the tool → +" + commaInt(c.CtxWithout) + " tok"
			rkind, rtext = p.green, "# cache hit → tool skipped (content re-served)"
		default:
			lkind, ltext = p.dim, ". +"+commaInt(c.CtxWithout)+" tok (tool runs once)"
			rkind, rtext = p.dim, ". +"+commaInt(c.CtxWith)+" tok (tool runs once)"
		}
		fmt.Printf("  %s  %s  %s\n",
			p.paint(lkind, padTrim(ltext, lw)),
			p.paint(p.dim, padTrim(c.Tool, cw)),
			p.paint(rkind, rtext))
	}

	fmt.Printf("  %s  %s  %s\n", strings.Repeat("─", lw), strings.Repeat("─", cw), strings.Repeat("─", rw))

	saidSomething := false
	if l.ContextTokensKept > 0 {
		saidSomething = true
		fmt.Printf("  %s\n", p.paint(p.bold+p.green, fmt.Sprintf(
			"→ WIN 1 (model context): fak keeps %s tokens out of the model — %d /bad call%s refused, the result never produced (only a %d-tok deny verdict enters).",
			commaInt(l.ContextTokensKept), l.Denies, plural(l.Denies), denyVerdictTokens)))
		fmt.Printf("  %s\n", p.paint(p.dim,
			"The SAFETY value of refusing it (the destructive op never runs) is a SEPARATE axis — see `guarddemo -print`."))
	}
	if l.RoundtripsCollapsed > 0 {
		saidSomething = true
		fmt.Printf("  %s\n", p.paint(p.bold+p.green, fmt.Sprintf(
			"→ WIN 2 (tool-side): %d re-read%s served from cache — the tool executed %d times, not %d (%s tool-result tokens not re-fetched).",
			l.RoundtripsCollapsed, plural(l.RoundtripsCollapsed), l.ToolRunsWith, l.ToolRunsWithout, commaInt(l.ToolTokensFromCache))))
		fmt.Printf("  %s\n", p.paint(p.dim,
			"HONEST: the cached content is still RETURNED to the model, so this is a tool-side latency/compute/$ win, not a model-context cut. "+
				"The model-side prefill/KV reuse that would also cut the re-read's tokens is `ctxdemo`'s axis (KV-eviction half: mechanism-proven, see FAQ)."))
	}
	if !saidSomething {
		fmt.Printf("  %s\n", p.paint(p.dim, fmt.Sprintf(
			"a clean path saves nothing on either meter — both arms ingest the same %s model tokens and run each tool once (the anti-inflation control). "+
				"fak only saves on a refused bad call or a re-read; it never cries wolf.", commaInt(l.CtxWithout))))
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

// ---------------------------------------------------------------------------
// -selfcheck: replay each suite through the kernel and assert the documented
// ledger invariants. The CI / cross-platform dog-food of this demo's data path.
// ---------------------------------------------------------------------------

// suiteExpect is the documented invariant for a known fixture — the headline numbers
// -print / -json publish. -selfcheck pins the demo's data path to them, so a
// regression (or a cross-platform divergence) fails loudly with no eyes in the loop.
type suiteExpect struct {
	denies, dedups      int
	contextTokensKept   int // win 1
	roundtripsCollapsed int // win 2
	toolTokensFromCache int // win 2
}

var selfcheckExpect = map[string]suiteExpect{
	// 4 mutating /bad calls refused (write_file, delete_path, run_shell, apply_patch),
	// each carrying a large result; only the bounded deny verdict enters the model.
	"prefilter-bad-calls": {denies: 4, dedups: 0,
		contextTokensKept:   (320 - denyVerdictTokens) + (240 - denyVerdictTokens) + (600 - denyVerdictTokens) + (420 - denyVerdictTokens),
		roundtripsCollapsed: 0, toolTokensFromCache: 0},
	// config.yaml read 3× (2 re-reads) + main.go read 2× (1 re-read) = 3 cache hits;
	// the tool runs once per distinct file, not once per read. NO model-context cut.
	"reread-same-file": {denies: 0, dedups: 3,
		contextTokensKept: 0, roundtripsCollapsed: 3, toolTokensFromCache: 180 + 180 + 540},
	// no bad calls, no re-reads — the anti-inflation control saves zero on both meters.
	"clean-control": {denies: 0, dedups: 0, contextTokensKept: 0, roundtripsCollapsed: 0, toolTokensFromCache: 0},
}

func runSelfcheck() int {
	ctx := context.Background()
	fmt.Printf("== tokendemo -selfcheck: replay each suite through the kernel (browserless) ==\n")
	fmt.Printf("dir: %s   GOMAXPROCS=%d   deny_verdict=%d tok\n\n", tokenDir(), gomax, denyVerdictTokens)

	ran, failed := 0, 0
	for _, ks := range knownSuites {
		l, err := buildLedger(ctx, ks.ID)
		if err != nil {
			fmt.Printf("  %-22s FAIL   %v\n", ks.ID, err)
			failed++
			continue
		}
		ran++
		var miss []string
		check := func(name string, got, want int) {
			if got != want {
				miss = append(miss, fmt.Sprintf("%s=%d(want %d)", name, got, want))
			}
		}
		if exp, known := selfcheckExpect[ks.ID]; known {
			check("denies", l.Denies, exp.denies)
			check("dedups", l.Dedups, exp.dedups)
			check("context_tokens_kept_out", l.ContextTokensKept, exp.contextTokensKept)
			check("roundtrips_collapsed", l.RoundtripsCollapsed, exp.roundtripsCollapsed)
			check("tool_tokens_from_cache", l.ToolTokensFromCache, exp.toolTokensFromCache)
		}
		// Invariants true for EVERY suite: the model-context meter never costs MORE
		// behind fak than raw, and a re-read NEVER cuts model context (it is a tool-side
		// win only — the honest bound the dedup framing rests on).
		if l.CtxWith > l.CtxWithout {
			miss = append(miss, "ctx_with>ctx_without")
		}
		if l.Dedups > 0 && l.CtxWithout != l.CtxWith {
			// With no denies, a dedup-only suite must show ZERO model-context delta.
			if l.Denies == 0 {
				miss = append(miss, fmt.Sprintf("dedup-only suite cut model context (without=%d with=%d) — overclaim", l.CtxWithout, l.CtxWith))
			}
		}
		status := "PASS"
		if len(miss) > 0 {
			status, failed = "FAIL", failed+1
		}
		fmt.Printf("  %-22s %s   win1 ctx-kept=%s tok (%d denies)  win2 roundtrips=%d (%s tool tok from cache)\n",
			ks.ID, status, commaInt(l.ContextTokensKept), l.Denies, l.RoundtripsCollapsed, commaInt(l.ToolTokensFromCache))
		if len(miss) > 0 {
			fmt.Printf("                         mismatch: %v\n", miss)
		}
	}

	fmt.Println()
	if ran == 0 {
		fmt.Printf("SELFCHECK FAILED — no fixtures found under %s (run from the repo root)\n", tokenDir())
		return 1
	}
	if failed > 0 {
		fmt.Printf("SELFCHECK FAILED — %d/%d suite(s) mismatched the documented ledger invariants\n", failed, ran)
		return 1
	}
	fmt.Printf("OK — %d/%d suite(s) reproduced the documented ledger invariants (browserless)\n", ran, ran)
	return 0
}
