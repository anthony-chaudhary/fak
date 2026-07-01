package mutationbudget

import (
	"strings"
	"testing"
)

// nowUnix is a fixed clock for the deterministic tests. The reset times in the
// fixtures are expressed as offsets from it so the humanized "resets in Nm" window
// is stable.
const nowUnix int64 = 1_700_000_000

func TestGuard(t *testing.T) {
	cases := []struct {
		name      string
		remaining int
		resetAt   int64
		limit     int
		planned   int
		reserve   int
		wantAllow bool
		// wantReason substrings that must all appear in the reason (empty slice = no
		// substring assertion beyond the ALLOW/HOLD prefix checked separately).
		wantReason []string
	}{
		{
			name:       "ample budget allows",
			remaining:  500,
			resetAt:    nowUnix + 600,
			limit:      1000,
			planned:    12,
			reserve:    20,
			wantAllow:  true,
			wantReason: []string{"ALLOW", "12 planned", "leave 488", "reserve 20"},
		},
		{
			name:      "burst that breaches reserve HOLDs with actionable reason",
			remaining: 15,
			resetAt:   nowUnix + 14*60, // 14 minutes out
			limit:     1000,
			planned:   12,
			reserve:   20,
			wantAllow: false,
			// remaining 15 is already below reserve 20, so this reports the
			// already-under-reserve shortfall and the reset window.
			wantReason: []string{"HOLD", "remaining 15", "reserve 20", "14m", "wait"},
		},
		{
			name:      "burst drives ample budget below reserve HOLDs naming shortfall + reset",
			remaining: 25,
			resetAt:   nowUnix + 14*60, // 14 minutes out
			limit:     1000,
			planned:   12,
			reserve:   20,
			wantAllow: false,
			// 25 - 12 = 13 < reserve 20; names planned, the after value, reserve, and
			// the reset window.
			wantReason: []string{"HOLD", "12 planned", "leave 13", "reserve 20", "14m", "wait or reduce batch"},
		},
		{
			name:       "exact boundary: remaining-planned == reserve allows",
			remaining:  32,
			resetAt:    nowUnix + 600,
			limit:      1000,
			planned:    12,
			reserve:    20, // 32 - 12 == 20 == reserve
			wantAllow:  true,
			wantReason: []string{"ALLOW", "leave 20", "reserve 20"},
		},
		{
			name:       "exact boundary: remaining-planned == reserve-1 holds",
			remaining:  31,
			resetAt:    nowUnix + 600,
			limit:      1000,
			planned:    12,
			reserve:    20, // 31 - 12 == 19 == reserve-1
			wantAllow:  false,
			wantReason: []string{"HOLD", "leave 19", "reserve 20"},
		},
		{
			name:       "zero planned always allows even at zero remaining",
			remaining:  0,
			resetAt:    nowUnix + 600,
			limit:      1000,
			planned:    0,
			reserve:    20,
			wantAllow:  true,
			wantReason: []string{"ALLOW", "0 planned"},
		},
		{
			name:       "negative planned treated as nothing to spend, allows",
			remaining:  5,
			resetAt:    nowUnix + 600,
			limit:      1000,
			planned:    -3,
			reserve:    20,
			wantAllow:  true,
			wantReason: []string{"ALLOW"},
		},
		{
			name:       "remaining already below reserve, positive burst holds",
			remaining:  3,
			resetAt:    nowUnix + 14*60,
			limit:      1000,
			planned:    1,
			reserve:    20,
			wantAllow:  false,
			wantReason: []string{"HOLD", "remaining 3", "already below reserve 20", "14m"},
		},
		{
			name:       "negative reserve clamped to zero, burst within remaining allows",
			remaining:  10,
			resetAt:    nowUnix + 600,
			limit:      1000,
			planned:    10,
			reserve:    -5, // clamps to 0; 10 - 10 == 0 >= 0
			wantAllow:  true,
			wantReason: []string{"ALLOW", "reserve 0"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := Budget{Remaining: tc.remaining, ResetAtUnix: tc.resetAt, Limit: tc.limit}
			d := Guard(b, tc.planned, tc.reserve, nowUnix)

			if d.Allow != tc.wantAllow {
				t.Fatalf("Allow = %v, want %v (reason: %q)", d.Allow, tc.wantAllow, d.Reason)
			}

			// The reason must always be populated and lead with the verdict token.
			if d.Reason == "" {
				t.Fatalf("Reason is empty; want a populated actionable reason")
			}
			wantPrefix := "HOLD"
			if tc.wantAllow {
				wantPrefix = "ALLOW"
			}
			if !strings.HasPrefix(d.Reason, wantPrefix) {
				t.Errorf("Reason %q does not lead with %q", d.Reason, wantPrefix)
			}

			for _, sub := range tc.wantReason {
				if !strings.Contains(d.Reason, sub) {
					t.Errorf("Reason %q missing expected substring %q", d.Reason, sub)
				}
			}

			// String() is the reason verbatim.
			if d.String() != d.Reason {
				t.Errorf("String() = %q, want == Reason %q", d.String(), d.Reason)
			}

			// The carried numbers must be internally consistent.
			wantReserve := tc.reserve
			if wantReserve < 0 {
				wantReserve = 0
			}
			if d.Reserve != wantReserve {
				t.Errorf("Reserve = %d, want clamped %d", d.Reserve, wantReserve)
			}
			if d.Remaining != tc.remaining {
				t.Errorf("Remaining = %d, want %d", d.Remaining, tc.remaining)
			}
			if d.Planned != tc.planned {
				t.Errorf("Planned = %d, want %d", d.Planned, tc.planned)
			}
			if d.AfterRemaining != tc.remaining-tc.planned {
				t.Errorf("AfterRemaining = %d, want %d", d.AfterRemaining, tc.remaining-tc.planned)
			}
		})
	}
}

// TestGuardHoldReasonNamesShortfallAndWindow pins the acceptance criterion: a burst
// that would breach reserve HOLDs with a reason that names both the shortfall (the
// planned count and the sub-reserve after value) and the reset window, so the
// operator can act (wait or reduce) without inspecting the struct.
func TestGuardHoldReasonNamesShortfallAndWindow(t *testing.T) {
	b := Budget{Remaining: 25, ResetAtUnix: nowUnix + 14*60, Limit: 1000}
	d := Guard(b, 12, 20, nowUnix)

	if d.Allow {
		t.Fatalf("expected HOLD, got Allow=true: %q", d.Reason)
	}
	for _, sub := range []string{"HOLD", "12", "leave 13", "reserve 20", "resets in 14m", "wait or reduce batch"} {
		if !strings.Contains(d.Reason, sub) {
			t.Errorf("hold reason %q missing %q", d.Reason, sub)
		}
	}
}

func TestResetInSec(t *testing.T) {
	cases := []struct {
		name    string
		resetAt int64
		now     int64
		want    int64
	}{
		{"future window", nowUnix + 90, nowUnix, 90},
		{"exactly now", nowUnix, nowUnix, 0},
		{"already past clamps to zero", nowUnix - 30, nowUnix, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := Budget{ResetAtUnix: tc.resetAt}
			if got := b.ResetInSec(tc.now); got != tc.want {
				t.Errorf("ResetInSec(%d) with resetAt %d = %d, want %d", tc.now, tc.resetAt, got, tc.want)
			}
		})
	}
}

func TestHumanizeSec(t *testing.T) {
	cases := []struct {
		sec  int64
		want string
	}{
		{0, "now"},
		{-5, "now"},
		{45, "45s"},
		{60, "1m"},
		{14 * 60, "14m"},
		{3600, "1h"},
		{3600 + 15*60, "1h15m"},
	}
	for _, tc := range cases {
		if got := humanizeSec(tc.sec); got != tc.want {
			t.Errorf("humanizeSec(%d) = %q, want %q", tc.sec, got, tc.want)
		}
	}
}
