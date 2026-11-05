package kvbudget

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestAdmissionRejectsTokenLengthOverflow(t *testing.T) {
	for _, tc := range []struct {
		name  string
		admit func(*Reservation, Stream) Verdict
	}{
		{"uncapped", (*Reservation).Admit},
		{"capped", func(r *Reservation, s Stream) Verdict { return r.AdmitCapped(s, 1) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, s := range []Stream{
				{PromptTokens: math.MaxInt, MaxNewTokens: 1, BlockSize: 1},
				// Even when the cap fits, the declared total and verdict's
				// uncapped footprint must remain representable.
				{PromptTokens: math.MaxInt - 1, MaxNewTokens: 2, BlockSize: 1},
			} {
				r := Reservation{FreeBlocks: math.MaxInt, ReservedBlocks: 2}
				before := r
				v := tc.admit(&r, s)
				if v.Admitted || v.Reason != ReasonInvalidRequest || v.ReservedBlocks != 0 || r != before {
					t.Fatalf("overflowed declared length: stream=%+v verdict=%+v budget=%+v", s, v, r)
				}
			}
			r := Reservation{FreeBlocks: math.MaxInt}
			v := tc.admit(&r, Stream{PromptTokens: math.MaxInt - 1, MaxNewTokens: 1, BlockSize: 1})
			if !v.Admitted || v.ReservedBlocks != math.MaxInt || r.Available() != 0 {
				t.Fatalf("exact representable length: verdict=%+v budget=%+v", v, r)
			}
		})
	}
	// A huge but representable declared length still benefits from the cap.
	r := Reservation{FreeBlocks: 2}
	if v := r.AdmitCapped(Stream{MaxNewTokens: math.MaxInt, BlockSize: 1}, 2); !v.Admitted || v.ReservedBlocks != 2 || v.WorstCaseBlocks != math.MaxInt {
		t.Fatalf("representable capped request: %+v", v)
	}
}
