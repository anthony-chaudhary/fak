package loopunblock

import "testing"

// sec turns a whole-second offset into unix nanos for readable tick timelines.
func sec(s int64) int64 { return s * 1_000_000_000 }

func tick(a Action, head string, atSec int64) Tick {
	return Tick{Action: a, Head: head, UnixNano: sec(atSec)}
}

func TestAssess_TableDriven(t *testing.T) {
	// A tight policy keeps the timelines short: stalled after 3 same-head blocks,
	// escalate once a stall is 100s old.
	pol := StallPolicy{StallAfter: 3, EscalateAfterSeconds: 100, FastDelaySeconds: 10, SlowDelaySeconds: 90}

	tests := []struct {
		name         string
		history      []Tick
		now          int64
		wantStalled  bool
		wantEscalate bool
		wantStreak   int
		wantDelay    int64
		wantHead     string
	}{
		{
			name:      "no history yet — fast cadence, not stalled",
			history:   nil,
			now:       sec(10),
			wantDelay: 10,
		},
		{
			name:      "newest tick drained — fast cadence, streak resets",
			history:   []Tick{tick(ActionWait, "a", 1), tick(ActionBypass, "a", 2)},
			now:       sec(3),
			wantDelay: 10,
		},
		{
			name:      "clear-then-enter counts as draining — fast cadence",
			history:   []Tick{tick(ActionClearThenEnter, "a", 5)},
			now:       sec(6),
			wantDelay: 10,
		},
		{
			name:      "stand-down (empty queue) — slow cadence, not stalled",
			history:   []Tick{tick(ActionStandDown, "", 5)},
			now:       sec(6),
			wantDelay: 90,
		},
		{
			name:       "one blocked tick — not yet a stall, slow cadence",
			history:    []Tick{tick(ActionWait, "a", 5)},
			now:        sec(6),
			wantStreak: 1,
			wantDelay:  90,
			wantHead:   "a",
		},
		{
			name: "three same-head blocks — head-of-line STALLED (young, no escalate)",
			history: []Tick{
				tick(ActionWait, "a", 10),
				tick(ActionWait, "a", 20),
				tick(ActionWait, "a", 30),
			},
			now:         sec(40), // age 30s < 100s escalate horizon
			wantStalled: true,
			wantStreak:  3,
			wantDelay:   90,
			wantHead:    "a",
		},
		{
			name: "stall aged past the escalation horizon — ESCALATE",
			history: []Tick{
				tick(ActionWait, "a", 1000),
				tick(ActionWait, "a", 1050),
				tick(ActionWait, "a", 1100),
			},
			now:          sec(1200), // age 200s >= 100s
			wantStalled:  true,
			wantEscalate: true,
			wantStreak:   3,
			wantDelay:    90,
			wantHead:     "a",
		},
		{
			name:         "newest tick already Escalate — escalate immediately regardless of age",
			history:      []Tick{tick(ActionEscalate, "a", 95)},
			now:          sec(96), // age 1s, but the decision already escalated
			wantEscalate: true,
			wantStreak:   1,
			wantDelay:    90,
			wantHead:     "a",
		},
		{
			name: "blocking head keeps CHANGING — draining, streak resets to 1, not stalled",
			history: []Tick{
				tick(ActionWait, "a", 10),
				tick(ActionWait, "b", 20),
				tick(ActionWait, "c", 30),
			},
			now:         sec(40),
			wantStalled: false, // three blocks, but on three DIFFERENT heads
			wantStreak:  1,
			wantDelay:   90,
			wantHead:    "c",
		},
		{
			name: "progress in the middle breaks the streak",
			history: []Tick{
				tick(ActionWait, "a", 10),
				tick(ActionBypass, "a", 20), // drained here
				tick(ActionWait, "a", 30),   // fresh block after progress
			},
			now:         sec(40),
			wantStalled: false,
			wantStreak:  1,
			wantDelay:   90,
			wantHead:    "a",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := Assess(tt.history, pol, tt.now)
			if v.Stalled != tt.wantStalled {
				t.Errorf("Stalled = %v, want %v (reason: %s)", v.Stalled, tt.wantStalled, v.Reason)
			}
			if v.Escalate != tt.wantEscalate {
				t.Errorf("Escalate = %v, want %v (reason: %s)", v.Escalate, tt.wantEscalate, v.Reason)
			}
			if v.BlockedStreak != tt.wantStreak {
				t.Errorf("BlockedStreak = %d, want %d", v.BlockedStreak, tt.wantStreak)
			}
			if v.NextDelaySeconds != tt.wantDelay {
				t.Errorf("NextDelaySeconds = %d, want %d", v.NextDelaySeconds, tt.wantDelay)
			}
			if v.Head != tt.wantHead {
				t.Errorf("Head = %q, want %q", v.Head, tt.wantHead)
			}
			if v.Schema != WatchSchema {
				t.Errorf("Schema = %q, want %q", v.Schema, WatchSchema)
			}
		})
	}
}

// The zero-value policy must be usable: Assess fills every threshold with its default.
func TestAssess_ZeroPolicyDefaults(t *testing.T) {
	// DefaultStallAfter (3) same-head blocks -> stalled; fast/slow defaults applied.
	hist := []Tick{
		tick(ActionWait, "a", 0),
		tick(ActionWait, "a", 1),
		tick(ActionWait, "a", 2),
	}
	v := Assess(hist, StallPolicy{}, sec(3))
	if !v.Stalled {
		t.Fatalf("want stalled at DefaultStallAfter=%d, got streak %d", DefaultStallAfter, v.BlockedStreak)
	}
	if v.NextDelaySeconds != DefaultSlowDelaySeconds {
		t.Errorf("blocked NextDelay = %d, want DefaultSlowDelaySeconds %d", v.NextDelaySeconds, DefaultSlowDelaySeconds)
	}
	// A draining newest tick under the zero policy uses the fast default.
	drain := Assess([]Tick{tick(ActionEnter, "a", 0)}, StallPolicy{}, sec(1))
	if drain.NextDelaySeconds != DefaultFastDelaySeconds {
		t.Errorf("draining NextDelay = %d, want DefaultFastDelaySeconds %d", drain.NextDelaySeconds, DefaultFastDelaySeconds)
	}
}

// Progressed is the exported progress/no-progress split the pacing keys on.
func TestAction_Progressed(t *testing.T) {
	progressed := map[Action]bool{
		ActionEnter: true, ActionClearThenEnter: true, ActionBypass: true,
		ActionWait: false, ActionEscalate: false, ActionStandDown: false,
	}
	for a, want := range progressed {
		if a.Progressed() != want {
			t.Errorf("%q.Progressed() = %v, want %v", a, a.Progressed(), want)
		}
	}
}
