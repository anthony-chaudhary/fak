package compute

import "testing"

// TestResolveGPUBudgetBytes pins the FAK_GPU_BUDGET_MB resolution, including the auto-derivation
// (#3998). It fails before the auto branch exists — "auto" would fall through the numeric parse and
// return 0 even with a known capacity, tripping the auto-known case — and passes after. Capacity is
// injected, so the auto path is exercised with no real GPU present.
func TestResolveGPUBudgetBytes(t *testing.T) {
	const gib = int64(1) << 30
	const mib = int64(1) << 20
	cases := []struct {
		name  string
		raw   string
		total int64
		known bool
		want  int64
	}{
		// Numeric / unset paths must stay byte-identical to the pre-#3998 behavior.
		{"unset unbounded", "", 24 * gib, true, 0},
		{"invalid unbounded", "notanumber", 24 * gib, true, 0},
		{"zero unbounded", "0", 24 * gib, true, 0},
		{"negative unbounded", "-5", 24 * gib, true, 0},
		{"numeric 4096 MiB", "4096", 24 * gib, true, 4096 * mib},
		{"numeric ignores capacity", "2048", 0, false, 2048 * mib},
		// auto derives a positive budget from injected capacity.
		{"auto known derives pct", "auto", 24 * gib, true, (24 * gib) / 100 * gpuBudgetAutoPercent},
		{"auto case-insensitive", "AUTO", 10 * gib, true, (10 * gib) / 100 * gpuBudgetAutoPercent},
		{"auto whitespace", " auto ", 8 * gib, true, (8 * gib) / 100 * gpuBudgetAutoPercent},
		// auto fails open to unbounded when capacity is unknown or non-positive.
		{"auto unknown fail-open", "auto", 0, false, 0},
		{"auto zero-capacity fail-open", "auto", 0, true, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := resolveGPUBudgetBytes(tc.raw, tc.total, tc.known)
			if got != tc.want {
				t.Fatalf("resolveGPUBudgetBytes(%q, %d, %v) = %d, want %d", tc.raw, tc.total, tc.known, got, tc.want)
			}
			// Acceptance: auto on a capacity-probing backend derives a POSITIVE budget.
			if tc.raw == "auto" && tc.known && tc.total > 0 && got <= 0 {
				t.Fatalf("auto with known capacity must derive a positive budget, got %d", got)
			}
		})
	}
}
