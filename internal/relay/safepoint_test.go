package relay

import "testing"

// TestSafePointRequiresAllThreeConditions drives every corner of the 3-axis conjunction
// (issue #1880): IsSafe must be true only when NoInFlightTurn, TreeGreenOrParked, and
// NextActionExpressible all hold — a single false axis must make the whole point unsafe.
func TestSafePointRequiresAllThreeConditions(t *testing.T) {
	cases := []struct {
		name string
		sp   SafePoint
		want bool
	}{
		{"all three hold", SafePoint{NoInFlightTurn: true, TreeGreenOrParked: true, NextActionExpressible: true}, true},
		{"zero value", SafePoint{}, false},
		{"in-flight turn blocks", SafePoint{NoInFlightTurn: false, TreeGreenOrParked: true, NextActionExpressible: true}, false},
		{"dirty tree blocks", SafePoint{NoInFlightTurn: true, TreeGreenOrParked: false, NextActionExpressible: true}, false},
		{"mid-thought blocks", SafePoint{NoInFlightTurn: true, TreeGreenOrParked: true, NextActionExpressible: false}, false},
		{"only tree green", SafePoint{TreeGreenOrParked: true}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.sp.IsSafe(); got != tc.want {
				t.Fatalf("SafePoint%+v.IsSafe() = %v, want %v", tc.sp, got, tc.want)
			}
		})
	}
}
