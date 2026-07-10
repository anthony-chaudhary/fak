package dispatchtick

import "testing"

// TestReconcileSetpoint is the #4036 witness: the operator concurrency setpoint folds
// level-triggered against the live fleet -- grow immediately, contract only on drain,
// never a kill -- and a zero (unset) setpoint is a total no-op.
func TestReconcileSetpoint(t *testing.T) {
	cases := []struct {
		name     string
		live     int
		setpoint int
		want     SetpointPlan
	}{
		{
			name: "unset_is_noop", live: 5, setpoint: 0,
			want: SetpointPlan{Mode: "inactive"}, // Active false, all zero
		},
		{
			name: "negative_is_noop", live: 5, setpoint: -3,
			want: SetpointPlan{Mode: "inactive"},
		},
		{
			name: "higher_grows_now", live: 3, setpoint: 8,
			want: SetpointPlan{Active: true, DesiredCap: 8, Mode: "grow"},
		},
		{
			name: "grow_from_idle_fleet", live: 0, setpoint: 4,
			want: SetpointPlan{Active: true, DesiredCap: 4, Mode: "grow"},
		},
		{
			name: "equal_holds_steady", live: 6, setpoint: 6,
			want: SetpointPlan{Active: true, DesiredCap: 6, Mode: "steady"},
		},
		{
			name: "lower_drains_surplus", live: 10, setpoint: 4,
			// contract to 4: 6 surplus workers marked draining, target feeds #4038's cap.
			want: SetpointPlan{Active: true, DesiredCap: 4, ContractionTarget: 4, Draining: 6, Mode: "drain"},
		},
		{
			name: "drain_to_one", live: 3, setpoint: 1,
			want: SetpointPlan{Active: true, DesiredCap: 1, ContractionTarget: 1, Draining: 2, Mode: "drain"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ReconcileSetpoint(tc.live, tc.setpoint)
			if got != tc.want {
				t.Errorf("ReconcileSetpoint(live=%d, setpoint=%d) = %+v, want %+v",
					tc.live, tc.setpoint, got, tc.want)
			}
		})
	}
}

// TestReconcileSetpoint_ShrinkNeverKills asserts the contraction is level-triggered:
// the surplus is marked draining toward the target, and drain count + target always
// reconstruct the live count -- no worker is dropped below the target in one step.
func TestReconcileSetpoint_ShrinkNeverKills(t *testing.T) {
	for live := 1; live <= 12; live++ {
		for setpoint := 1; setpoint < live; setpoint++ {
			p := ReconcileSetpoint(live, setpoint)
			if p.Mode != "drain" {
				t.Fatalf("live=%d setpoint=%d: Mode=%q, want drain", live, setpoint, p.Mode)
			}
			if p.ContractionTarget != setpoint {
				t.Errorf("live=%d setpoint=%d: ContractionTarget=%d, want %d", live, setpoint, p.ContractionTarget, setpoint)
			}
			if p.ContractionTarget+p.Draining != live {
				t.Errorf("live=%d setpoint=%d: target(%d)+draining(%d) != live(%d) -- a worker was killed, not drained",
					live, setpoint, p.ContractionTarget, p.Draining, live)
			}
		}
	}
}

// TestParseConcurrencySetpoint covers the fail-safe parse: a blank, malformed, or
// negative operator string yields 0 (inactive), so a cleared or corrupt setpoint file
// can never drain the fleet to zero.
func TestParseConcurrencySetpoint(t *testing.T) {
	cases := map[string]int{
		"":        0,
		"   ":     0,
		"  6 ":    6,
		"8":       8,
		"0":       0,
		"-4":      0,
		"abc":     0,
		"3.5":     0,
		"12\n":    12,
		"nan":     0,
		" 100\t ": 100,
	}
	for in, want := range cases {
		if got := ParseConcurrencySetpoint(in); got != want {
			t.Errorf("ParseConcurrencySetpoint(%q) = %d, want %d", in, got, want)
		}
	}
}
