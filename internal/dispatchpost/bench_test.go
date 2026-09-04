package dispatchpost

import (
	"testing"
)

// BenchmarkDispatchPost exercises card rendering and witness folding in a loop
// to ensure zero regressions in allocation and formatting overhead.
func BenchmarkDispatchPost(b *testing.B) {
	res := Result{
		LoopID:     "nightly-fix",
		RunID:      "run-bench",
		ExitCode:   0,
		DurationMS: 65000,
		Command:    "fak loop run",
		HeadBefore: "aaa1111",
		HeadAfter:  "bbb2222",
		Commits: []string{
			"bbb2222 fix(gateway): treat same-tick ready as positive (fak gateway)",
		},
		Source: "cron",
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		w := CardWitness(res)
		_ = w
		_ = res.Text()
		_ = res.Blocks()
	}
}
