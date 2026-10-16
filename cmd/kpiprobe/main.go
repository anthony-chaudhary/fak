// Command kpiprobe prints the RSI loop's demo KPI — the deterministic LRU
// hit-rate (internal/rsiloop.HitRate) over a fixed reference trace — as a single
// `KPI=<float>` line the loop's worktree Measurer parses. With no flags it reports
// the hit-rate at the committed DefaultCacheSize, so when the loop forks a worktree
// and rewrites that constant, running this probe there MEASURES the candidate.
//
// The metric is wall-clock-free and reproduces bit-for-bit across platforms, which
// is what makes it a legal RSI witness (docs/proofs/00-METHOD.md): the keep/revert
// decision is the same on every box.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("kpiprobe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	size := fs.Int("size", rsiloop.DefaultCacheSize, "LRU cache size to probe (defaults to the committed DefaultCacheSize)")
	dump := fs.Bool("dump", false, "print the hit-rate curve for sizes 1..16 (for picking real candidate sizes)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	if *dump {
		fmt.Fprintf(stdout, "# lru_hit_rate over %d accesses\n", rsiloop.TraceLen())
		for n := 1; n <= 16; n++ {
			fmt.Fprintf(stdout, "size=%2d KPI=%.6f\n", n, rsiloop.HitRate(n))
		}
		return 0
	}
	// The load-bearing line the worktree Measurer parses.
	fmt.Fprintf(stdout, "KPI=%.6f\n", rsiloop.HitRate(*size))
	return 0
}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }
