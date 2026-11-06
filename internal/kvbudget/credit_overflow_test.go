package kvbudget

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestBlockCreditsClampWithoutOverflow(t *testing.T) {
	for _, tc := range []struct {
		name           string
		held, retained int
		want           int
	}{
		{"combined credits overflow", math.MaxInt, math.MaxInt, 0},
		{"held covers footprint", math.MaxInt, 3, 0},
		{"retained covers footprint", 3, math.MaxInt, 0},
		{"exact coverage", 1, 1, 0},
		{"partial coverage", 1, 0, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := Stream{PromptTokens: 1, MaxNewTokens: 1, BlockSize: 1, HeldBlocks: tc.held, RetainedBlocks: tc.retained}
			for name, got := range map[string]int{
				"reservation":        s.ReserveBlocks(),
				"capped reservation": s.CappedReserveBlocks(1),
				"actual charge":      s.ChargedBlocks(1),
			} {
				if got != tc.want {
					t.Errorf("%s = %d, want %d", name, got, tc.want)
				}
			}
			for name, admit := range map[string]func(*Reservation) Verdict{
				"uncapped": func(r *Reservation) Verdict { return r.Admit(s) },
				"capped":   func(r *Reservation) Verdict { return r.AdmitCapped(s, 1) },
			} {
				r := Reservation{FreeBlocks: tc.want + 1, ReservedBlocks: 1}
				v := admit(&r)
				if !v.Admitted || v.Reason != ReasonAdmitted || v.ReservedBlocks != tc.want || r.ReservedBlocks != tc.want+1 {
					t.Errorf("%s: verdict=%+v budget=%+v, want admitted with %d additional blocks", name, v, r, tc.want)
				}
			}
		})
	}
}
