// Command modelprof is the in-kernel forward pass inspecting ITSELF: it runs the
// modular bottleneck profiler (internal/model/profile.go) over a real decode and a
// real prefill, plus an uninstrumented Session.Step decode measurement. It prints
// per-op-class roofline attribution — MACs, weight-bytes streamed, measured time share,
// arithmetic intensity, and a memory- vs compute-bound verdict anchored to this
// machine's MEASURED memory bandwidth — without presenting instrumentation overhead as
// clean throughput.
//
// It is the observability answer to "where does the time actually go": the profiler
// twin is pinned bit-for-bit to the proven decode path (TestProfileMatchesProven),
// so this is a faithful measurement of the same code the oracle verifies.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func table(p *model.Profile) string {
	s := fmt.Sprintf("\n== %s  (instrumented per-token %.1f ms,  achieved %.1f GFLOP/s, %.1f GB/s = %.0f%% of %.1f GB/s mem ceiling) ==\n",
		p.Mode, p.PerTokenMS, p.AchievedGFLOP, p.AchievedGBps, p.BWUtilPct, p.MemBWGBps)
	s += fmt.Sprintf("%-10s %7s %12s %12s %10s %8s  %s\n", "op-class", "time%", "MMACs", "MB-streamed", "flop/byte", "GB/s", "verdict")
	for _, o := range p.Stats {
		s += fmt.Sprintf("%-10s %6.1f%% %12.1f %12.1f %10.2f %8.1f  %s\n",
			o.Class, o.TimePct, float64(o.MACs)/1e6, float64(o.Bytes)/1e6, o.IntensFB, o.GBps, o.Verdict)
	}
	s += fmt.Sprintf("bottleneck: %s\n", p.Bottleneck)
	return s
}

func cleanDecodeTable(c *model.CleanDecode) string {
	return fmt.Sprintf("\n== clean decode  (Session.Step, uninstrumented %.1f ms/tok over %d steps after %d-token prompt) ==\n%s\n",
		c.PerTokenMS, c.Steps, c.PromptTokens, c.Summary)
}

func main() {
	dir := flag.String("dir", "internal/model/.cache/smollm2-135m", "model export dir")
	out := flag.String("out", "", "write JSON here (default stdout summary)")
	prompt := flag.Int("prompt", 16, "prompt length")
	steps := flag.Int("steps", 24, "decode steps to profile")
	prefillP := flag.Int("prefill", 64, "prefill length to profile")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*dir = pathutil.ExpandTilde(*dir)

	m, err := model.Load(*dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	clean := m.MeasureCleanDecode(*prompt, *steps)
	dec := m.ProfileDecode(*prompt, *steps)
	pre := m.ProfilePrefill(*prefillP)

	fmt.Fprint(os.Stderr, cleanDecodeTable(clean))
	fmt.Fprint(os.Stderr, table(dec))
	fmt.Fprint(os.Stderr, table(pre))

	report := map[string]any{"app_version": appversion.Current(), "clean_decode": clean, "decode": dec, "prefill": pre}
	b, _ := json.MarshalIndent(report, "", "  ")
	if *out != "" {
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write:", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "wrote", *out)
	} else {
		fmt.Println(string(b))
	}
}
