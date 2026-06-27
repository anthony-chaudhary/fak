package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/demoui"
)

func main() {
	jobs := flag.Int("jobs", 0, "cap GOMAXPROCS to an ABSOLUTE core count (0 = all cores). On a shared/active box pass e.g. 8 so the demo doesn't starve other work.")
	print := flag.Bool("print", false, "render the WITHOUT-kernel vs WITH-kernel ledger as a colored TWO-COLUMN diff in the TERMINAL and exit. The 30-second point with zero setup. -suite picks the trace; honors NO_COLOR.")
	asJSON := flag.Bool("json", false, "emit the exact per-call ledger as JSON (all suites, or just -suite) and exit.")
	timing := flag.Bool("timing", false, "run a measured raw-vs-fak proof for one suite and print per-tool-call latency/source evidence. If -suite is omitted, defaults to reread-same-file.")
	timingJSON := flag.Bool("timing-json", false, "emit the measured raw-vs-fak proof as JSON. If -suite is omitted, defaults to reread-same-file.")
	served := flag.Bool("served", false, "run a served-boundary same-read proof: raw os.ReadFile N times per served surface, then HTTP /v1/fak/syscall and MCP fak_syscall through one gateway, showing first read hits fakread and repeats return served_by=vdso tier=2.")
	servedJSON := flag.Bool("served-json", false, "emit the served-boundary same-read HTTP/MCP proof as JSON.")
	servedCalls := flag.Int("served-calls", defaultServedCalls, "same-file read count per served surface for -served/-served-json; must be at least 2.")
	parallel := flag.Bool("parallel", false, "run a warmed parallel hot-read proof: many workers repeat a small read set and fak serves the hot phase from vDSO tier-2.")
	parallelJSON := flag.Bool("parallel-json", false, "emit the warmed parallel hot-read proof as JSON.")
	engineDelayMS := flag.Int("engine-delay-ms", 15, "fixed local-tool delay used by timing and parallel proof modes so engine re-execution cost is visible and deterministic enough to inspect.")
	parallelWorkers := flag.Int("parallel-workers", 32, "worker count for -parallel/-parallel-json.")
	parallelCalls := flag.Int("parallel-calls", 512, "parallel hot-phase calls for -parallel/-parallel-json.")
	parallelHotFiles := flag.Int("parallel-hot-files", 8, "distinct hot files to prewarm and repeat for -parallel/-parallel-json.")
	parallelCold := flag.Bool("parallel-cold", false, "run the COLD concurrent same-read fill-race probe: N workers released at ONE barrier against a NEVER-SEEN key, counting engine calls before the vDSO tier-2 fill exists. A measurement first — MEASURED_RACE is the expected verdict until singleflight is built.")
	parallelColdJSON := flag.Bool("parallel-cold-json", false, "emit the cold concurrent same-read fill-race probe as JSON.")
	parallelColdWorkers := flag.Int("parallel-cold-workers", 64, "worker count released at the cold barrier for -parallel-cold/-parallel-cold-json.")
	parallelColdTrials := flag.Int("parallel-cold-trials", 24, "independent cold trials (each a fresh empty vDSO world + never-seen key) for -parallel-cold/-parallel-cold-json.")
	selfcheck := flag.Bool("selfcheck", false, "run HEADLESS: replay each suite through the kernel (the same turnbench.RunWithWorld path -print/-json drive), assert the documented ledger invariants, and exit non-zero on any drift. The CI / cross-platform dog-food of this demo's data path.")
	suite := flag.String("suite", "prefilter-bad-calls", "suite for -print / -json (prefilter-bad-calls | reread-same-file | clean-control)")
	flag.Parse()
	suiteSet := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "suite" {
			suiteSet = true
		}
	})
	selectedSuite := *suite
	if (*timing || *timingJSON) && !suiteSet {
		selectedSuite = "reread-same-file"
	}
	if *jobs > 0 {
		runtime.GOMAXPROCS(*jobs)
		gomax = *jobs
	}
	if *engineDelayMS < 0 {
		fmt.Fprintln(os.Stderr, "-engine-delay-ms must be non-negative")
		os.Exit(2)
	}
	if (*served || *servedJSON) && *servedCalls < 2 {
		fmt.Fprintln(os.Stderr, "-served-calls must be at least 2")
		os.Exit(2)
	}
	if *parallelWorkers <= 0 || *parallelCalls <= 0 || *parallelHotFiles <= 0 {
		fmt.Fprintln(os.Stderr, "-parallel-workers, -parallel-calls, and -parallel-hot-files must be positive")
		os.Exit(2)
	}
	if *parallelColdWorkers <= 0 || *parallelColdTrials <= 0 {
		fmt.Fprintln(os.Stderr, "-parallel-cold-workers and -parallel-cold-trials must be positive")
		os.Exit(2)
	}
	switch {
	case *selfcheck:
		os.Exit(runSelfcheck())
	case *servedJSON:
		os.Exit(runServedReadJSON(*servedCalls))
	case *served:
		os.Exit(runServedReadPrint(*servedCalls))
	case *parallelColdJSON:
		os.Exit(runColdJSON(*parallelColdWorkers, *parallelColdTrials, time.Duration(*engineDelayMS)*time.Millisecond))
	case *parallelCold:
		os.Exit(runColdPrint(*parallelColdWorkers, *parallelColdTrials, time.Duration(*engineDelayMS)*time.Millisecond))
	case *parallelJSON:
		os.Exit(runParallelJSON(*parallelWorkers, *parallelCalls, *parallelHotFiles, time.Duration(*engineDelayMS)*time.Millisecond))
	case *parallel:
		os.Exit(runParallelPrint(*parallelWorkers, *parallelCalls, *parallelHotFiles, time.Duration(*engineDelayMS)*time.Millisecond))
	case *timingJSON:
		os.Exit(runTimingJSON(selectedSuite, time.Duration(*engineDelayMS)*time.Millisecond))
	case *timing:
		os.Exit(runTimingPrint(selectedSuite, time.Duration(*engineDelayMS)*time.Millisecond))
	case *asJSON:
		os.Exit(runJSON(selectedSuiteForJSON(*suite, suiteSet)))
	case *print:
		os.Exit(runPrint(*suite))
	default:
		// No mode flag: this demo has no browser surface — point the operator at the
		// three headless modes (the value here is the numbers, not a live server).
		fmt.Fprintf(os.Stderr, "tokendemo %s — the tool-call token ledger (no model, no browser)\n", version)
		fmt.Fprintf(os.Stderr, "trace dir: %s\n", tokenDir())
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -print [-suite %s]\n", knownSuites[0].ID)
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -timing [-suite reread-same-file]\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -served [-served-calls %d]\n", defaultServedCalls)
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -parallel [-parallel-workers 32 -parallel-calls 512]\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -parallel-cold [-parallel-cold-workers 64 -parallel-cold-trials 24]\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -json\n")
		fmt.Fprintf(os.Stderr, "  go run ./cmd/tokendemo -selfcheck\n")
		os.Exit(2)
	}
}

func selectedSuiteForJSON(suite string, explicitlySet bool) string {
	if !explicitlySet {
		return "all"
	}
	return suite
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

func runTimingJSON(suite string, delay time.Duration) int {
	proof, err := buildTimingProof(context.Background(), suite, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "timing proof %q: %v\n", suite, err)
		return 1
	}
	b, _ := json.MarshalIndent(proof, "", "  ")
	fmt.Println(string(b))
	return 0
}

func runTimingPrint(suite string, delay time.Duration) int {
	p := colors()
	proof, err := buildTimingProof(context.Background(), suite, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "timing proof %q: %v\n", suite, err)
		return 1
	}

	fmt.Printf("\n  %s — suite: %s (%d calls, engine delay %dms)\n",
		p.paint(p.bold, "fak · measured tool-cache proof"), suite, len(proof.Calls), proof.EngineDelayMs)
	fmt.Printf("  %s\n\n", p.paint(p.dim, "raw loop executes every tool; fak loop goes through kernel.Syscall with vDSO on"))
	fmt.Printf("  %-3s  %-10s  %-16s  %-12s  %9s  %9s  %-11s  %-7s  %-6s  %-6s\n",
		"#", "tool", "resource", "args", "raw ms", "fak ms", "fak source", "engine", "vDSO", "tokens")
	fmt.Printf("  %s\n", strings.Repeat("─", 105))
	for _, c := range proof.Calls {
		engine := "skip"
		if c.EngineRanFak {
			engine = "run"
		}
		source := c.FakSource
		if c.FakTier != "" && !strings.Contains(source, c.FakTier) {
			source += "/" + c.FakTier
		}
		lineColor := p.dim
		if c.FakSource == "vdso_tier2" || c.FakSource == "fak_deny" {
			lineColor = p.green
		}
		fmt.Printf("  %s\n", p.paint(lineColor, fmt.Sprintf("%-3d  %-10s  %-16s  %-12s  %9s  %9s  %-11s  %-7s  %-6d  %-6d",
			c.Index,
			padTrim(c.Tool, 10),
			padTrim(c.Resource, 16),
			c.ArgsHash,
			formatMs(c.RawToolTimeNs),
			formatMs(c.FakToolTimeNs),
			padTrim(source, 11),
			engine,
			c.KernelVDSODelta,
			c.ResultTokens,
		)))
	}
	fmt.Printf("  %s\n", strings.Repeat("─", 105))
	fmt.Printf("  raw engine calls: %d   fak engine calls: %d   vDSO hits: %d   round-trips collapsed: %d\n",
		proof.RawEngineCalls, proof.FakEngineCalls, proof.VDSOHits, proof.RoundtripsCollapsed)
	fmt.Printf("  measured tool time: raw %.3fms   fak %.3fms   saved %.3fms\n",
		nsToMs(proof.RawTotalNs), nsToMs(proof.FakTotalNs), nsToMs(proof.TimeSavedNs))
	fmt.Printf("  tool-result tokens served from cache: %s   model-context tokens kept out: %s\n\n",
		commaInt(proof.ToolTokensFromCache), commaInt(proof.ContextTokensKeptOut))
	printCacheClaimBoundary(proof.Path, proof.Prewarmed)
	return 0
}

func runParallelJSON(workers, calls, hotFiles int, delay time.Duration) int {
	proof, err := buildParallelProof(context.Background(), workers, calls, hotFiles, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parallel proof: %v\n", err)
		return 1
	}
	b, _ := json.MarshalIndent(proof, "", "  ")
	fmt.Println(string(b))
	return 0
}

func runParallelPrint(workers, calls, hotFiles int, delay time.Duration) int {
	p := colors()
	proof, err := buildParallelProof(context.Background(), workers, calls, hotFiles, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parallel proof: %v\n", err)
		return 1
	}
	fmt.Printf("\n  %s — %d workers, %d hot calls, %d hot files (engine delay %dms)\n",
		p.paint(p.bold, "fak · parallel hot-cache proof"), proof.Workers, proof.Calls, proof.HotFiles, proof.EngineDelayMs)
	fmt.Printf("  %s\n\n", p.paint(p.dim, "fak prewarms one read per hot file, then the parallel hot phase should be vDSO tier-2 only"))
	fmt.Printf("  raw engine calls: %d   fak warmup engine calls: %d   fak hot engine calls: %d\n",
		proof.RawEngineCalls, proof.FakWarmupEngineCalls, proof.FakHotEngineCalls)
	fmt.Printf("  vDSO hits: %d   engine calls avoided: %d (%.1f%%)   tool tokens from cache: %s\n",
		proof.VDSOHits, proof.EngineCallsAvoided, proof.EngineCallAvoidedRate*100, commaInt(proof.ToolTokensFromCache))
	fmt.Printf("  hot-phase total tool time: raw %.3fms   fak %.3fms   saved %.3fms\n",
		nsToMs(proof.RawTotalNs), nsToMs(proof.FakHotTotalNs), nsToMs(proof.TimeSavedNs))
	fmt.Printf("  hot-phase wall time: raw %.3fms   fak %.3fms   warmup %.3fms\n",
		nsToMs(proof.RawWallNs), nsToMs(proof.FakHotWallNs), nsToMs(proof.FakWarmupWallNs))
	fmt.Printf("  p50/p95 per call: raw %s/%sms   fak %s/%sms\n\n",
		formatMs(proof.RawP50Ns), formatMs(proof.RawP95Ns), formatMs(proof.FakP50Ns), formatMs(proof.FakP95Ns))
	printCacheClaimBoundary(proof.Path, proof.Prewarmed)

	fmt.Printf("  %-38s  %8s  %8s  %8s  %8s  %10s  %10s\n",
		"resource", "raw_eng", "warm_eng", "hot_eng", "vdso", "raw_p95", "fak_p95")
	fmt.Printf("  %s\n", strings.Repeat("─", 100))
	for _, r := range proof.PerResource {
		color := p.dim
		if r.FakHotEngineCalls == 0 && r.VDSOHits > 0 {
			color = p.green
		}
		fmt.Printf("  %s\n", p.paint(color, fmt.Sprintf("%-38s  %8d  %8d  %8d  %8d  %10sms  %10sms",
			padTrim(r.Resource, 38),
			r.RawEngineCalls,
			r.FakWarmupEngineCalls,
			r.FakHotEngineCalls,
			r.VDSOHits,
			formatMs(r.RawP95Ns),
			formatMs(r.FakP95Ns),
		)))
	}
	fmt.Println()
	return 0
}

func runServedJSON(calls int, delay time.Duration) int {
	proof, err := buildServedProof(context.Background(), calls, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "served proof: %v\n", err)
		return 1
	}
	b, _ := json.MarshalIndent(proof, "", "  ")
	fmt.Println(string(b))
	return 0
}

func runServedPrint(calls int, delay time.Duration) int {
	p := colors()
	proof, err := buildServedProof(context.Background(), calls, delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "served proof: %v\n", err)
		return 1
	}

	fmt.Printf("\n  %s — %s %q through HTTP + MCP (%d calls/surface, engine delay %dms)\n",
		p.paint(p.bold, "fak · served same-read cache proof"), proof.Tool, proof.Resource, proof.CallsPerSurface, proof.EngineDelayMs)
	fmt.Printf("  %s\n\n", p.paint(p.dim, "raw loop executes every read; served loop POSTs /v1/fak/syscall and MCP fak_syscall against one gateway"))
	fmt.Printf("  %-3s  %-5s  %9s  %9s  %-9s  %-6s  %-7s  %-12s\n",
		"#", "wire", "raw ms", "served ms", "served_by", "tier", "engine", "args")
	fmt.Printf("  %s\n", strings.Repeat("─", 74))
	for _, c := range proof.Calls {
		engine := "skip"
		if c.EngineRanServed {
			engine = "run"
		}
		color := p.dim
		if c.ServedBy == "vdso" && c.Tier == "2" {
			color = p.green
		}
		fmt.Printf("  %s\n", p.paint(color, fmt.Sprintf("%-3d  %-5s  %9s  %9s  %-9s  %-6s  %-7s  %-12s",
			c.Index,
			c.Surface,
			formatMs(c.RawToolTimeNs),
			formatMs(c.ServedTimeNs),
			padTrim(c.ServedBy, 9),
			padTrim(c.Tier, 6),
			engine,
			c.ArgsHash,
		)))
	}
	fmt.Printf("  %s\n", strings.Repeat("─", 74))
	fmt.Printf("  raw engine calls: %d   fak engine calls: %d   response tier-2 hits: %d\n",
		proof.RawEngineCalls, proof.FakEngineCalls, proof.VDSOHits)
	fmt.Printf("  /metrics delta: syscalls=%d   vdso_lookups=%d   vdso_hits=%d   cache_fills=%d   http=%d   mcp=%d\n",
		proof.GatewayMetrics.Delta.GatewaySyscalls,
		proof.GatewayMetrics.Delta.VDSOLookups,
		proof.GatewayMetrics.Delta.VDSOHits,
		proof.GatewayMetrics.Delta.VDSOFills,
		proof.GatewayMetrics.Delta.HTTPSyscallRequests,
		proof.GatewayMetrics.Delta.MCPRequests)
	fmt.Printf("  measured tool time: raw %.3fms   served %.3fms   saved %.3fms\n\n",
		nsToMs(proof.RawTotalNs), nsToMs(proof.ServedTotalNs), nsToMs(proof.TimeSavedNs))
	return 0
}

func nsToMs(ns int64) float64 {
	return float64(ns) / float64(time.Millisecond)
}

func formatMs(ns int64) string {
	ms := nsToMs(ns)
	if ns > 0 && ms < 0.001 {
		return "<0.001"
	}
	return fmt.Sprintf("%.3f", ms)
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
		var check demoui.SelfcheckChecker
		if exp, known := selfcheckExpect[ks.ID]; known {
			check.Check("denies", l.Denies, exp.denies)
			check.Check("dedups", l.Dedups, exp.dedups)
			check.Check("context_tokens_kept_out", l.ContextTokensKept, exp.contextTokensKept)
			check.Check("roundtrips_collapsed", l.RoundtripsCollapsed, exp.roundtripsCollapsed)
			check.Check("tool_tokens_from_cache", l.ToolTokensFromCache, exp.toolTokensFromCache)
		}
		// Invariants true for EVERY suite: the model-context meter never costs MORE
		// behind fak than raw, and a re-read NEVER cuts model context (it is a tool-side
		// win only — the honest bound the dedup framing rests on).
		if l.CtxWith > l.CtxWithout {
			check.Note("ctx_with>ctx_without")
		}
		if l.Dedups > 0 && l.CtxWithout != l.CtxWith {
			// With no denies, a dedup-only suite must show ZERO model-context delta.
			if l.Denies == 0 {
				check.Notef("dedup-only suite cut model context (without=%d with=%d) — overclaim", l.CtxWithout, l.CtxWith)
			}
		}
		status := "PASS"
		if check.Failed() {
			status, failed = "FAIL", failed+1
		}
		fmt.Printf("  %-22s %s   win1 ctx-kept=%s tok (%d denies)  win2 roundtrips=%d (%s tool tok from cache)\n",
			ks.ID, status, commaInt(l.ContextTokensKept), l.Denies, l.RoundtripsCollapsed, commaInt(l.ToolTokensFromCache))
		if check.Failed() {
			fmt.Printf("                         mismatch: %v\n", check.Mismatches())
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
