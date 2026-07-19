package cacheprice

import "testing"

func TestTransferBreakEvenLength(t *testing.T) {
	cases := []struct {
		name                                       string
		fixed, recomputePerToken, transferPerToken int
		wantLen                                    int
		wantEver                                   bool
	}{
		{"floor amortizes at fixed/slope + 1", 100, 10, 0, 11, true},
		{"non-divisible floor rounds up past break-even", 105, 10, 0, 11, true},
		{"per-token wire cost eats into the slope", 100, 10, 6, 26, true}, // slope 4 → 100/4+1 = 26
		{"zero floor makes any non-empty prefix worthwhile", 0, 8, 3, 1, true},
		{"equal per-token costs never win at any length", 50, 7, 7, 0, false},
		{"per-token wire cost above recompute never wins", 50, 4, 9, 0, false},
		{"negative inputs clamp — zero floor, positive slope", -20, 5, -3, 1, true}, // slope 5, floor 0
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotLen, gotEver := TransferBreakEvenLength(c.fixed, c.recomputePerToken, c.transferPerToken)
			if gotLen != c.wantLen || gotEver != c.wantEver {
				t.Fatalf("TransferBreakEvenLength(%d, %d, %d) = (%d, %v), want (%d, %v)",
					c.fixed, c.recomputePerToken, c.transferPerToken, gotLen, gotEver, c.wantLen, c.wantEver)
			}
		})
	}
}

func TestTransferWorthwhileAtLength(t *testing.T) {
	cases := []struct {
		name                               string
		prefix, fixed, recompute, transfer int
		want                               bool
	}{
		{"above break-even is worthwhile", 11, 100, 10, 0, true},
		{"exactly at break-even minus one is not", 10, 100, 10, 0, false},
		{"at break-even is worthwhile", 26, 100, 10, 6, true},
		{"zero-length prefix is never worthwhile", 0, 0, 10, 0, false},
		{"flat slope never worthwhile however long", 100000, 1, 5, 5, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := TransferWorthwhileAtLength(c.prefix, c.fixed, c.recompute, c.transfer); got != c.want {
				t.Fatalf("TransferWorthwhileAtLength(%d, %d, %d, %d) = %v, want %v",
					c.prefix, c.fixed, c.recompute, c.transfer, got, c.want)
			}
		})
	}
}

// The two functions must agree: when a break-even exists, a prefix is worthwhile at exactly the
// lengths at or above it; when none exists, no length is ever worthwhile.
func TestBreakEvenLengthAgreesWithAtLength(t *testing.T) {
	for fixed := 0; fixed <= 200; fixed += 40 {
		for recompute := 0; recompute <= 12; recompute += 3 {
			for transfer := 0; transfer <= 12; transfer += 3 {
				breakEven, ever := TransferBreakEvenLength(fixed, recompute, transfer)
				for prefix := 0; prefix <= 60; prefix++ {
					want := ever && prefix >= breakEven
					if got := TransferWorthwhileAtLength(prefix, fixed, recompute, transfer); got != want {
						t.Fatalf("fixed=%d recompute=%d transfer=%d prefix=%d: AtLength=%v but (ever=%v && prefix>=breakEven=%d)=%v",
							fixed, recompute, transfer, prefix, got, ever, breakEven, want)
					}
				}
			}
		}
	}
}
