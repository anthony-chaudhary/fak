package cacheprice

import "testing"

func TestDisaggregationRetentionValue(t *testing.T) {
	cases := []struct {
		name                                         string
		dividendPerFetch, expectedFetches, residency int
		want                                         int
	}{
		{"profitable prefix earns its keep", 700, 10, 500, 6500},
		{"break-even over its life is zero", 100, 5, 500, 0},
		{"positive per-fetch but cannot repay residency is a net loss", 700, 1, 800, -100},
		{"negative per-fetch dividend compounds over reuse", -312, 5, 100, -1660},
		{"no fetches, no cost is zero", -50, 0, 0, 0},
		{"pure residency toll with no reuse is fully negative", 0, 0, 256, -256},
		{"negative expectedFetches clamps to 0 (miscount invents no reuse)", 500, -3, 100, -100},
		{"negative residency clamps to 0 (miscount invents no toll)", 500, 4, -50, 2000},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DisaggregationRetentionValue(c.dividendPerFetch, c.expectedFetches, c.residency); got != c.want {
				t.Fatalf("DisaggregationRetentionValue(%d, %d, %d) = %d, want %d",
					c.dividendPerFetch, c.expectedFetches, c.residency, got, c.want)
			}
		})
	}
}

func TestRemoteResidentRetentionValue(t *testing.T) {
	r := RemoteResident{DividendPerFetch: 700, ExpectedFetches: 10, ResidencyCost: 500}
	if got, want := r.RetentionValue(), 6500; got != want {
		t.Fatalf("RetentionValue() = %d, want %d", got, want)
	}
}

func TestAdmitToRemote(t *testing.T) {
	cases := []struct {
		name string
		r    RemoteResident
		want bool
	}{
		{"positive lifetime value is admitted", RemoteResident{DividendPerFetch: 700, ExpectedFetches: 10, ResidencyCost: 500}, true},
		{"break-even is refused (keep the pool for prefixes that pay)", RemoteResident{DividendPerFetch: 100, ExpectedFetches: 5, ResidencyCost: 500}, false},
		{"positive per-fetch that cannot repay residency is refused", RemoteResident{DividendPerFetch: 700, ExpectedFetches: 1, ResidencyCost: 800}, false},
		{"negative per-fetch dividend is refused", RemoteResident{DividendPerFetch: -312, ExpectedFetches: 5, ResidencyCost: 100}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := AdmitToRemote(c.r); got != c.want {
				t.Fatalf("AdmitToRemote(%+v) = %v, want %v", c.r, got, c.want)
			}
		})
	}
}

func TestEvictionVictim(t *testing.T) {
	cases := []struct {
		name      string
		residents []RemoteResident
		want      int
	}{
		{"empty pool has no victim", nil, -1},
		{"single resident is the victim", []RemoteResident{{CapacityTokens: 10, DividendPerFetch: 100, ExpectedFetches: 1}}, 0},
		{
			// The load-bearing inversion: a HOT prefix whose fetch loses to recompute
			// (negative dividend) is evicted before a profitable one, despite more reuse.
			name: "hot-but-losing prefix is evicted before a profitable one",
			residents: []RemoteResident{
				{Key: "losing-hot", DividendPerFetch: -100, ExpectedFetches: 50, CapacityTokens: 100},                    // value -5000
				{Key: "profitable", DividendPerFetch: 200, ExpectedFetches: 10, ResidencyCost: 100, CapacityTokens: 100}, // value 1900
			},
			want: 0,
		},
		{
			// Density, not absolute value: the higher-value prefix is the victim because it
			// returns less compute PER BYTE of pool it holds (900/100 = 9 < 100/10 = 10).
			name: "lower density is the victim even with higher absolute value",
			residents: []RemoteResident{
				{Key: "small-dense", DividendPerFetch: 100, ExpectedFetches: 1, CapacityTokens: 10}, // density 10
				{Key: "big-sparse", DividendPerFetch: 900, ExpectedFetches: 1, CapacityTokens: 100}, // density 9
			},
			want: 1,
		},
		{
			// Equal density breaks toward the larger footprint (frees more pool per eviction).
			name: "equal density evicts the larger footprint",
			residents: []RemoteResident{
				{Key: "small", DividendPerFetch: 100, ExpectedFetches: 1, CapacityTokens: 10}, // density 10
				{Key: "large", DividendPerFetch: 200, ExpectedFetches: 1, CapacityTokens: 20}, // density 10
			},
			want: 1,
		},
		{
			// Full tie (equal density and footprint) is deterministic: lowest index.
			name: "full tie is deterministic on lowest index",
			residents: []RemoteResident{
				{Key: "a", DividendPerFetch: 100, ExpectedFetches: 1, CapacityTokens: 10},
				{Key: "b", DividendPerFetch: 100, ExpectedFetches: 1, CapacityTokens: 10},
			},
			want: 0,
		},
		{
			// A zero/negative footprint is floored to 1 — no divide-by-zero, still ranked.
			name: "zero footprint is floored to one and still ranked by value",
			residents: []RemoteResident{
				{Key: "zero-cap-losing", DividendPerFetch: -10, ExpectedFetches: 1, CapacityTokens: 0}, // value -10, cap→1
				{Key: "paying", DividendPerFetch: 5, ExpectedFetches: 1, CapacityTokens: 1},            // value 5
			},
			want: 0,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := EvictionVictim(c.residents); got != c.want {
				t.Fatalf("EvictionVictim(%d residents) = %d, want %d", len(c.residents), got, c.want)
			}
		})
	}
}
