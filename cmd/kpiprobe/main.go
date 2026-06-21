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
	"flag"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func main() {
	size := flag.Int("size", rsiloop.DefaultCacheSize, "LRU cache size to probe (defaults to the committed DefaultCacheSize)")
	dump := flag.Bool("dump", false, "print the hit-rate curve for sizes 1..16 (for picking real candidate sizes)")
	flag.Parse()

	if *dump {
		fmt.Printf("# lru_hit_rate over %d accesses\n", rsiloop.TraceLen())
		for n := 1; n <= 16; n++ {
			fmt.Printf("size=%2d KPI=%.6f\n", n, rsiloop.HitRate(n))
		}
		return
	}
	// The load-bearing line the worktree Measurer parses.
	fmt.Printf("KPI=%.6f\n", rsiloop.HitRate(*size))
}
