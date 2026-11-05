package memgate

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestParseLinuxHonorsReportedAvailability(t *testing.T) {
	t.Parallel()
	const counters = "MemTotal: 16000000 kB\nMemFree: 0 kB\nCached: 8000000 kB\n"
	for _, tc := range []struct {
		name      string
		row       string
		available int64
		admit     bool
	}{
		{name: "reported zero", row: "MemAvailable: 0 kB\n"},
		{name: "reported below margin", row: "MemAvailable: 1 kB\n"},
		{name: "reported positive", row: "MemAvailable: 4000000 kB\n", available: 2_096_000_000, admit: true},
		{name: "absent fallback", available: 6_192_000_000, admit: true},
		{name: "unparseable fallback", row: "MemAvailable: unknown kB\n", available: 6_192_000_000, admit: true},
		{name: "empty fallback", row: "MemAvailable:\n", available: 6_192_000_000, admit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mem := ParseLinux(counters + tc.row)
			want := Memory{
				TotalBytes:     16_384_000_000,
				PurgeableBytes: 8_192_000_000,
				AvailableBytes: tc.available,
			}
			if mem != want {
				t.Fatalf("memory = %+v, want %+v", mem, want)
			}
			got := Evaluate(BuildSnapshot("linux", mem, nil), 1)
			if got.Admit == nil || *got.Admit != tc.admit {
				t.Fatalf("one GB admission = %+v, want admit %v", got, tc.admit)
			}
			if !tc.admit && got.ShortfallGB != 1 {
				t.Fatalf("shortfall = %v, want 1 GB", got.ShortfallGB)
			}
		})
	}
}
