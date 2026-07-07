package dispatchtick

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestIssueTierFromLabels pins the C4 label -> IssueTier bridge across the cases
// the issue-contract grammar distinguishes: a valid consistent pair, the security
// shape (optimal MORE demanding than the floor), and every degrade mode — missing,
// the priority/P1-is-not-a-tier disambiguation, conflict, an out-of-range T3, and a
// contradiction (optimal weaker than the required floor). Every degrade mode must
// yield HasTier=false plus the exact closed-vocab flag(s), so an untagged or
// ambiguous issue can never route below the frontier floor.
func TestIssueTierFromLabels(t *testing.T) {
	cases := []struct {
		name      string
		labels    []string
		want      IssueTier
		wantFlags []string
	}{
		{
			name:   "consistent T1/T1",
			labels: []string{"tier/T1-required", "tier/T1-optimal"},
			want:   IssueTier{Required: modelroute.TierT1, Optimal: modelroute.TierT1, HasTier: true},
		},
		{
			name:   "security shape optimal more demanding",
			labels: []string{"tier/T1-required", "tier/T0-optimal"},
			want:   IssueTier{Required: modelroute.TierT1, Optimal: modelroute.TierT0, HasTier: true},
		},
		{
			name:   "duplicate identical tier is not a conflict",
			labels: []string{"tier/T2-required", "tier/T2-required", "tier/T2-optimal"},
			want:   IssueTier{Required: modelroute.TierT2, Optimal: modelroute.TierT2, HasTier: true},
		},
		{
			name:      "only required present",
			labels:    []string{"tier/T2-required"},
			wantFlags: []string{TagFlagOptimalMissing},
		},
		{
			name:      "neither present, priority is not a tier",
			labels:    []string{"priority/P1", "area/dispatch"},
			wantFlags: []string{TagFlagRequiredMissing, TagFlagOptimalMissing},
		},
		{
			name:      "conflicting required",
			labels:    []string{"tier/T0-required", "tier/T1-required", "tier/T1-optimal"},
			wantFlags: []string{TagFlagRequiredConflict},
		},
		{
			name:      "out of range T3 required",
			labels:    []string{"tier/T3-required", "tier/T1-optimal"},
			wantFlags: []string{TagFlagRequiredInvalid},
		},
		{
			name:      "contradiction optimal weaker than floor",
			labels:    []string{"tier/T0-required", "tier/T1-optimal"},
			wantFlags: []string{TagFlagContradiction},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, flags := IssueTierFromLabels(c.labels)
			if got != c.want {
				t.Errorf("tier = %+v, want %+v", got, c.want)
			}
			if !reflect.DeepEqual(flags, c.wantFlags) {
				t.Errorf("flags = %v, want %v", flags, c.wantFlags)
			}
		})
	}
}
