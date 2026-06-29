package opttarget

// CacheSizeTarget is the rsiloop DefaultCacheSize demo (worktree.go's hand-wired
// harness) re-expressed as a DECLARATION — the Phase 0 proof that an existing
// hand-coded harness becomes data. `values` is the int sweep to try; everything
// else is fixed to the demo's site/metric/direction:
//
//   - Metric "lru_hit_rate", HigherBetter  — matches NewWorktreeHarness.
//   - Site internal/rsiloop/tunable.go : DefaultCacheSize — the anchored literal
//     the worktree Proposer rewrites.
//   - Measurer "worktree-int"           — the registry key a future loader binds
//     to the real worktree probe (Phase 1); Compile takes the value explicitly.
//
// Compiled with a HarnessMeasurer over the real worktree harness it drives the
// identical live loop; compiled with the same in-process measurement the hand-
// wired harness uses, it produces a byte-identical journal (the golden test).
func CacheSizeTarget(values []int) OptTarget {
	return OptTarget{
		Name:        "lru-cache-size",
		Metric:      "lru_hit_rate",
		Direction:   HigherBetter,
		BaselineRef: "main",
		Site:        Site{Path: "internal/rsiloop/tunable.go", Const: "DefaultCacheSize"},
		Grammar:     Grammar{Kind: GrammarIntSweep, Ints: values},
		Measurer:    "worktree-int",
		Guards:      Guards{ChangedPaths: []string{"internal/rsiloop/tunable.go"}},
	}
}
