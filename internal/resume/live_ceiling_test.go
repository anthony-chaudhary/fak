package resume

import "testing"

func seats(specs ...[3]int) []HeadroomSeat {
	// each spec: {available(0/1), throttled(0/1), active_sessions}
	out := make([]HeadroomSeat, 0, len(specs))
	for _, s := range specs {
		out = append(out, HeadroomSeat{Available: s[0] == 1, Throttled: s[1] == 1, ActiveSessions: s[2]})
	}
	return out
}

func TestResolveMaxLiveResumes_Precedence(t *testing.T) {
	healthy := seats([3]int{1, 0, 0}, [3]int{1, 0, 0}, [3]int{1, 0, 0}) // 3 healthy

	cases := []struct {
		name       string
		in         MaxLiveResumesInput
		wantValue  int
		wantSource MaxLiveResumesSource
	}{
		{
			// env wins over everything (flag-over-file), even a policy value + seats.
			name:       "env beats policy and seats",
			in:         MaxLiveResumesInput{EnvPresent: true, EnvValue: 12, ConfigValue: 8, Seats: healthy, Floor: 4, Ceiling: 16, SeatCap: 2},
			wantValue:  12,
			wantSource: MaxLiveFromEnv,
		},
		{
			// env explicitly 0 disables the ceiling and is still authoritative.
			name:       "env zero disables",
			in:         MaxLiveResumesInput{EnvPresent: true, EnvValue: 0, ConfigValue: 8},
			wantValue:  0,
			wantSource: MaxLiveFromEnv,
		},
		{
			name:       "policy file when no env",
			in:         MaxLiveResumesInput{ConfigValue: 8, Seats: healthy, Floor: 4, Ceiling: 16, SeatCap: 2},
			wantValue:  8,
			wantSource: MaxLiveFromConfigFile,
		},
		{
			// the #5093 fix: a fresh host (no env, no policy) scales past the static 4.
			name:       "derived from healthy seats beats static 4",
			in:         MaxLiveResumesInput{Seats: healthy, Floor: 4, Ceiling: 16, SeatCap: 2},
			wantValue:  6, // 3 healthy × 2/seat
			wantSource: MaxLiveFromDerived,
		},
		{
			name:       "nothing set falls to default",
			in:         MaxLiveResumesInput{},
			wantValue:  DefaultMaxLiveResumes,
			wantSource: MaxLiveFromDefault,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ResolveMaxLiveResumes(tc.in)
			if got.Value != tc.wantValue || got.Source != tc.wantSource {
				t.Fatalf("resolve = %d/%s, want %d/%s (detail: %s)",
					got.Value, got.Source, tc.wantValue, tc.wantSource, got.Detail)
			}
			if got.Detail == "" {
				t.Fatalf("provenance detail is empty for %s", tc.wantSource)
			}
		})
	}
}

func TestDeriveMaxLiveResumes_ClampsAndCountsHealthy(t *testing.T) {
	// 4 available seats, one throttled and one unavailable ⇒ 2 healthy; throttled/
	// unavailable seats do not count toward headroom (same rule as DeriveWatchdogCap).
	mixed := seats(
		[3]int{1, 0, 1}, // healthy
		[3]int{1, 1, 0}, // throttled -> excluded
		[3]int{0, 0, 0}, // unavailable -> excluded
		[3]int{1, 0, 3}, // healthy
	)
	// 2 healthy × 3/seat = 6, under the ceiling and over the floor.
	if v, healthy := DeriveMaxLiveResumes(mixed, 4, 16, 3); v != 6 || healthy != 2 {
		t.Fatalf("derive = %d (healthy %d), want 6 (healthy 2)", v, healthy)
	}
	// Floor holds when headroom is thin (0 healthy ⇒ floor, never below).
	if v, _ := DeriveMaxLiveResumes(seats([3]int{1, 1, 0}), 4, 16, 3); v != 4 {
		t.Fatalf("derive with no healthy seats = %d, want floor 4", v)
	}
	// Ceiling caps a large pool.
	big := make([]HeadroomSeat, 20)
	for i := range big {
		big[i] = HeadroomSeat{Available: true}
	}
	if v, healthy := DeriveMaxLiveResumes(big, 4, 16, 2); v != 16 || healthy != 20 {
		t.Fatalf("derive over ceiling = %d (healthy %d), want 16 (healthy 20)", v, healthy)
	}
	// Ceiling <= 0 means no upper bound beyond the floor.
	if v, _ := DeriveMaxLiveResumes(big, 4, 0, 2); v != 40 {
		t.Fatalf("derive with no ceiling = %d, want 40 (20×2)", v)
	}
}
