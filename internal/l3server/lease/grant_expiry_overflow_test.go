package lease

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestGrantExpiryDoesNotWrap(t *testing.T) {
	t.Parallel()
	const maxExpiry = int64(^uint64(0) >> 1)
	const start = int64(1_700_000_000_000)
	for _, tc := range []struct {
		name     string
		limit    int64
		duration int64
		now      int64
		existing int64
		want     int64
	}{
		{name: "overflowing grant", limit: maxExpiry, duration: maxExpiry, now: start, want: maxExpiry},
		{name: "exact maximum deadline", limit: maxExpiry, duration: maxExpiry - start, now: start, want: maxExpiry},
		{name: "below maximum deadline", limit: maxExpiry, duration: maxExpiry - start - 1, now: start, want: maxExpiry - 1},
		{name: "ordinary maximum clamps request", limit: 30_000, duration: maxExpiry, now: start, want: start + 30_000},
		{name: "default duration", limit: maxExpiry, now: start, want: start + 5000},
		{name: "default duration near maximum", limit: maxExpiry, now: maxExpiry - 10, want: maxExpiry},
		{name: "negative clock remains representable", limit: maxExpiry, duration: maxExpiry, now: -100, want: maxExpiry - 100},
		{name: "overflowing renewal extends", limit: maxExpiry, duration: maxExpiry, now: start, existing: start + 500, want: maxExpiry},
		{name: "renewal never shortens", limit: maxExpiry, duration: 1, now: start, existing: maxExpiry, want: maxExpiry},
		{name: "small maximum keeps default ordering", limit: 1000, now: start, want: start + 5000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			table := NewTable(tc.limit)
			if tc.existing != 0 {
				table.leases[42] = tc.existing
			}
			table.grantAt(42, tc.duration, tc.now)
			if got, ok := table.leases[42]; !ok || got != tc.want {
				t.Fatalf("deadline = %d (present=%v), want %d", got, ok, tc.want)
			}
			if len(table.leases) != 1 {
				t.Fatalf("lease count = %d, want one grant", len(table.leases))
			}
		})
	}
}
