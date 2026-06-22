// Command longctxbench renders the EXACT, contention-free work floor for the
// ULTRA-LONG-CONTEXT regime (per-agent context > 100k tokens) — the proof that the fused
// agent kernel's reread-elimination win holds (and grows) where it matters most, computed as
// closed-form arithmetic from the session shape and the model geometry. No model, no decode,
// no GPU, no wall-clock: the work-elimination ratios A/C (vs naive re-prefill), B/C (vs a warm
// per-agent KV cache) and A/B (the turn-tax) are arithmetic facts a re-run reproduces exactly.
//
// This is the analytic companion to cmd/sessionbench (which runs the same arms LIVE but, like
// every live bench, cannot reach 100k because the naive arm's O(T^2) re-prefill is intractable
// — exactly why sessionbench already COMPUTES its arm A). The token floor here is byte-identical
// to sessionbench's prefillTokens, so the two cross-validate; this tool adds the O(L^2)-aware
// FLOP floor that captures the prefill-attention quadratic the ultra-long-context win rides on.
//
// HONEST SCOPING (BENCHMARK-AUTHORITY law). A/C is vs the NAIVE re-prefill pattern — a worst-case
// REFERENCE, not a serving baseline anyone ships. B/C is vs a WARM per-agent KV cache — the honest
// serving baseline. Both are printed side by side, never conflated.
//
// Usage:
//
//	longctxbench -ladder                      # the canonical regime ladder (single 100k → agent-city)
//	longctxbench -model qwen25-7b -prefix 100000 -turns 10 -agents 1,5,40 -decode 200 -result 500
//	longctxbench -ladder -out experiments/session/ultra-long-context-floor.json
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

func main() {
	fs := flag.NewFlagSet("longctxbench", flag.ExitOnError)
	modelName := fs.String("model", "qwen25-7b", "model geometry: smollm2-135m | qwen25-1.5b | qwen25-7b")
	ladder := fs.Bool("ladder", false, "emit the canonical regime ladder (overrides the shape sweep flags)")
	prefix := fs.String("prefix", "100000", "shared prefix tokens P, comma-separated to sweep")
	turns := fs.String("turns", "10", "turns per agent T, comma-separated to sweep")
	agents := fs.String("agents", "1,5,40", "concurrent agents C, comma-separated to sweep")
	decode := fs.Int("decode", 200, "assistant tokens decoded per turn D")
	result := fs.Int("result", 500, "tool-result tokens ingested per turn R")
	out := fs.String("out", "", "write JSON artifact here (default: stdout summary only)")
	_ = fs.Parse(os.Args[1:])

	shape, ok := turnbench.NamedShape(*modelName)
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown -model %q (smollm2-135m | qwen25-1.5b | qwen25-7b)\n", *modelName)
		os.Exit(1)
	}

	var shapes []turnbench.SessionShape
	if *ladder {
		shapes = turnbench.CanonicalLadder()
	} else {
		for _, P := range parseInts(*prefix) {
			for _, T := range parseInts(*turns) {
				for _, C := range parseInts(*agents) {
					if P < 1 || T < 1 || C < 1 {
						continue
					}
					shapes = append(shapes, turnbench.SessionShape{
						Prefix: P, Turns: T, Agents: C, Decode: *decode, Result: *result,
					})
				}
			}
		}
	}
	if len(shapes) == 0 {
		fmt.Fprintln(os.Stderr, "no session shapes to project (check -prefix/-turns/-agents)")
		os.Exit(1)
	}

	rep := turnbench.RunLongContextLadder(shape, shapes, turnbench.DefaultCostModel())

	// Human summary to stderr (the artifact is the JSON).
	fmt.Fprintf(os.Stderr, "ultra-long-context work floor — model %s (d=%d, L=%d layers), threshold %d tokens\n",
		shape.Name, shape.HiddenSize, shape.NumLayers, rep.Threshold)
	fmt.Fprintf(os.Stderr, "%-7s %-5s %-6s %-9s | %-8s %-8s | %-9s %-9s %-9s | %s\n",
		"P", "T", "C", "maxctx", "tokA/C", "tokB/C", "flopA/C", "flopB/C", "flopA/B", "regime")
	for _, c := range rep.Cells {
		regime := "—"
		if c.UltraLong {
			if c.Shape.Agents <= 1 {
				regime = "ULTRA single"
			} else {
				regime = "ULTRA multi"
			}
		}
		fmt.Fprintf(os.Stderr, "%-7d %-5d %-6d %-9d | %-8s %-8s | %-9s %-9s %-9s | %s\n",
			c.Shape.Prefix, c.Shape.Turns, c.Shape.Agents, c.MaxContextTokens,
			turnbench.FmtRatio(c.TokenAOverC), turnbench.FmtRatio(c.TokenBOverC),
			turnbench.FmtRatio(c.FlopAOverC), turnbench.FmtRatio(c.FlopBOverC), turnbench.FmtRatio(c.FlopAOverB),
			regime)
	}
	if rep.SingleUltraLongIdx >= 0 {
		c := rep.Cells[rep.SingleUltraLongIdx]
		fmt.Fprintf(os.Stderr, "\nHEADLINE single >100k: %s vs naive (A/C) · %s vs tuned (B/C) — C=%d, maxctx=%d\n",
			turnbench.FmtRatio(c.TokenAOverC), turnbench.FmtRatio(c.TokenBOverC), c.Shape.Agents, c.MaxContextTokens)
	}
	if rep.MultiUltraLongIdx >= 0 {
		c := rep.Cells[rep.MultiUltraLongIdx]
		fmt.Fprintf(os.Stderr, "HEADLINE multi  >100k: %s vs naive (A/C) · %s vs tuned (B/C) — C=%d, maxctx=%d\n",
			turnbench.FmtRatio(c.TokenAOverC), turnbench.FmtRatio(c.TokenBOverC), c.Shape.Agents, c.MaxContextTokens)
	}
	fmt.Fprintln(os.Stderr, "\nA/C is vs NAIVE re-prefill (worst-case REFERENCE, not a serving baseline);"+
		" B/C is vs a WARM per-agent KV cache (the serving baseline). WORK floor — no wall-clock.")

	blob := rep.JSON()
	if *out != "" {
		if err := os.WriteFile(*out, blob, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", *out, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", *out)
	} else {
		fmt.Println(string(blob))
	}
}

// parseInts parses a comma-separated list of non-negative integers, skipping non-digits.
func parseInts(s string) []int {
	var out []int
	cur, has := 0, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' {
			cur = cur*10 + int(c-'0')
			has = true
		} else if has {
			out = append(out, cur)
			cur, has = 0, false
		}
	}
	if has {
		out = append(out, cur)
	}
	return out
}
