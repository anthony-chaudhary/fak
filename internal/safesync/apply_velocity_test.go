package safesync

import (
	"errors"
	"testing"
	"time"
)

func TestScoreApplyVelocityEffectQualification(t *testing.T) {
	for _, tc := range []struct {
		name      string
		info      Assessment
		err       error
		qualified bool
	}{
		{"applied", Assessment{Applied: true}, nil, true},
		{"in sync no op", Assessment{State: StateInSync, OK: true}, nil, false},
		{"dirty refusal", Assessment{State: StateBehind, Reason: "dirty"}, nil, false},
		{"error", Assessment{}, errors.New("boom"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ScoreApplyVelocity(tc.info, 10*time.Millisecond, time.Second, tc.err)
			if got.Qualified != tc.qualified || (got.Score != nil) != tc.qualified {
				t.Fatalf("got=%+v", got)
			}
		})
	}
}
