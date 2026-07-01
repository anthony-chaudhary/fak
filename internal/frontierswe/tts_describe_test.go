package frontierswe

import "testing"

// taskWithBudget builds a minimal Task carrying just an [agent] timeout_sec, the
// one fact GeometryForTask projects the long-horizon turn count from.
func taskWithBudget(name string, sec float64, trials int) *Task {
	t := &Task{Name: name}
	t.Agent.TimeoutSec = sec
	t.Job.NConcurrentTrial = trials
	return t
}

// TestProjectedTurnsScalesWithBudget pins the budget->turns projection: the 20h
// budget projects 2000 turns (turnsPerHour=100), an 8h budget 800, and a zero/
// negative budget floors at 1 so the arithmetic stays defined.
func TestProjectedTurnsScalesWithBudget(t *testing.T) {
	cases := []struct {
		sec  float64
		want int
	}{
		{72000, 2000}, // 20h * 100 turns/h
		{28800, 800},  // 8h * 100 turns/h
		{3600, 100},   // 1h
		{0, 1},        // no budget -> single-turn floor
		{-5, 1},       // negative -> floor
	}
	for _, c := range cases {
		if got := ProjectedTurns(c.sec); got != c.want {
			t.Errorf("ProjectedTurns(%v) = %d, want %d", c.sec, got, c.want)
		}
	}
}

// TestGeometryForTaskDeterministic checks the derived geometry is a pure function
// of the task's budget + the fixed regime constants — same task in, byte-identical
// geometry out, every call.
func TestGeometryForTaskDeterministic(t *testing.T) {
	task := taskWithBudget("git-to-zig", 72000, 0)
	g1 := GeometryForTask(task)
	g2 := GeometryForTask(task)
	if g1 != g2 {
		t.Fatalf("GeometryForTask not deterministic: %+v != %+v", g1, g2)
	}
	if g1.Name != "git-to-zig" {
		t.Errorf("geometry name = %q, want git-to-zig", g1.Name)
	}
	if g1.Turns != 2000 {
		t.Errorf("turns = %d, want 2000 (20h budget)", g1.Turns)
	}
	if g1.Prefix != defaultPrefixTokens || g1.Decode != defaultDecodeTokens || g1.Result != defaultResultTokens {
		t.Errorf("geometry shape = {P:%d D:%d R:%d}, want {%d %d %d}",
			g1.Prefix, g1.Decode, g1.Result, defaultPrefixTokens, defaultDecodeTokens, defaultResultTokens)
	}
	// The derived geometry reaches the hundreds-of-thousands-of-tokens regime.
	if g1.MaxContext() != defaultPrefixTokens+2000*(defaultDecodeTokens+defaultResultTokens) {
		t.Errorf("MaxContext = %d, want %d", g1.MaxContext(), defaultPrefixTokens+2000*(defaultDecodeTokens+defaultResultTokens))
	}
}

// TestProjectTTSMonotoneInReuse is the acceptance table assertion: across a sweep
// of reuse rates the projection is deterministic and its A/C work-elimination rises
// monotonically while the TTS ratio falls monotonically — the value curve the floor
// projects. It exercises a range of task budgets so the monotonicity holds shape-
// independently.
func TestProjectTTSMonotoneInReuse(t *testing.T) {
	budgets := []float64{72000, 28800, 3600} // 20h, 8h, 1h
	rs := []float64{0, 0.1, 0.25, 0.5, 0.75, 0.85, 1.0}

	for _, sec := range budgets {
		task := taskWithBudget("t", sec, 0)

		var prevAOverC, prevTTS float64
		for i, r := range rs {
			// Determinism: two projections at the same r must be identical.
			p1 := ProjectTTS(task, r, []int{1})
			p2 := ProjectTTS(task, r, []int{1})
			if p1.Arms != p2.Arms {
				t.Fatalf("budget %v r=%v: ProjectTTS not deterministic", sec, r)
			}
			a := p1.Arms
			// At r=0 there is no reuse: A/C == 1 and the TTS ratio == 1 (no speedup).
			if r == 0 {
				if a.AOverC != 1.0 || a.TTSRatio != 1.0 {
					t.Errorf("budget %v r=0: A/C=%v TTS=%v, want 1.0/1.0", sec, a.AOverC, a.TTSRatio)
				}
			}
			if i > 0 {
				if !(a.AOverC > prevAOverC) {
					t.Errorf("budget %v: A/C not strictly increasing at r=%v: %v <= %v", sec, r, a.AOverC, prevAOverC)
				}
				if !(a.TTSRatio < prevTTS) {
					t.Errorf("budget %v: TTS ratio not strictly decreasing at r=%v: %v >= %v", sec, r, a.TTSRatio, prevTTS)
				}
			}
			// A/C and TTS ratio are reciprocals by construction.
			if a.AOverC != 0 {
				if diff := 1.0/a.AOverC - a.TTSRatio; diff > 1e-9 || diff < -1e-9 {
					t.Errorf("budget %v r=%v: TTS %v != 1/(A/C) %v", sec, r, a.TTSRatio, 1.0/a.AOverC)
				}
			}
			prevAOverC, prevTTS = a.AOverC, a.TTSRatio
		}
		// The turn-tax A/B is > 1 for a long-horizon trajectory and is the ceiling
		// A/C climbs to at full reuse.
		full := ProjectTTS(task, 1.0, []int{1}).Arms
		if full.AOverB <= 1.0 {
			t.Errorf("budget %v: turn-tax A/B = %v, want > 1", sec, full.AOverB)
		}
		if full.AOverC != full.AOverB {
			t.Errorf("budget %v: at r=1 A/C (%v) should equal the turn-tax A/B (%v)", sec, full.AOverC, full.AOverB)
		}
	}
}

// TestProjectTTSReuseClamped checks the recorded Reuse and the arms clamp r to
// [0,1], so a projection cannot claim a speedup past the per-agent-KV floor.
func TestProjectTTSReuseClamped(t *testing.T) {
	task := taskWithBudget("t", 72000, 0)
	lo := ProjectTTS(task, -0.5, []int{1})
	if lo.Reuse != 0 || lo.Arms.TTSRatio != ProjectTTS(task, 0, []int{1}).Arms.TTSRatio {
		t.Errorf("negative r should clamp to 0: Reuse=%v", lo.Reuse)
	}
	hi := ProjectTTS(task, 2.0, []int{1})
	if hi.Reuse != 1 || hi.Arms.TTSRatio != ProjectTTS(task, 1, []int{1}).Arms.TTSRatio {
		t.Errorf("r>1 should clamp to 1: Reuse=%v", hi.Reuse)
	}
}

// TestCrossTrialReuseArm pins the cross-trial shared-prefix arm: at N=1 it collapses
// to the single-trial reused work (no fan-out), and at N>1 the work-elimination A/D
// strictly increases with the trial count (concurrent trials share one prefix), while
// the naive all-trials work A_trials scales exactly with N.
func TestCrossTrialReuseArm(t *testing.T) {
	g := GeometryForTask(taskWithBudget("git-to-zig", 72000, 0))
	r := DefaultReuseRate

	// N=1: the cross-trial arm equals the single-trial reused work C(r).
	one := CrossTrialArms(g, r, 1)
	single := DefaultTTSModel().Derive(g, r)
	if diff := one.DShared - single.C; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("N=1: D_shared %v should equal single-trial C(r) %v", one.DShared, single.C)
	}
	if one.Trials != 1 {
		t.Errorf("N=1: Trials = %d, want 1", one.Trials)
	}

	// The naive all-trials work scales exactly with N; the work-elimination A/D
	// rises monotonically as more trials share the one prefix.
	var prevAD float64
	for i, n := range []int{1, 2, 4, 8, 16} {
		ta := CrossTrialArms(g, r, n)
		if wantA := float64(single.A) * float64(n); ta.ATrials != wantA {
			t.Errorf("N=%d: A_trials = %v, want %v (A*N)", n, ta.ATrials, wantA)
		}
		if i > 0 && !(ta.ATrialsD > prevAD) {
			t.Errorf("N=%d: A/D not strictly increasing: %v <= %v", n, ta.ATrialsD, prevAD)
		}
		// The cross-trial TTS ratio is the reciprocal of A/D and stays in (0,1].
		if ta.ATrialsD != 0 {
			if diff := 1.0/ta.ATrialsD - ta.TTSTrial; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("N=%d: cross-trial TTS %v != 1/(A/D) %v", n, ta.TTSTrial, 1.0/ta.ATrialsD)
			}
		}
		prevAD = ta.ATrialsD
	}
}

// TestProjectTTSDefaultTrialSweep checks the trial sweep defaults to {1} when the
// task declares no concurrent trials and {1, N} when it does, so the sweep always
// shows the single-trial floor plus the task's real fan-out.
func TestProjectTTSDefaultTrialSweep(t *testing.T) {
	noTrials := ProjectTTS(taskWithBudget("t", 72000, 0), DefaultReuseRate, nil)
	if len(noTrials.TrialSweep) != 1 || noTrials.TrialSweep[0].Trials != 1 {
		t.Errorf("no declared trials: sweep = %v, want [1]", trialCounts(noTrials.TrialSweep))
	}
	withTrials := ProjectTTS(taskWithBudget("t", 72000, 5), DefaultReuseRate, nil)
	got := trialCounts(withTrials.TrialSweep)
	if len(got) != 2 || got[0] != 1 || got[1] != 5 {
		t.Errorf("declared 5 trials: sweep = %v, want [1 5]", got)
	}
}

func trialCounts(arms []TrialArms) []int {
	out := make([]int, len(arms))
	for i, a := range arms {
		out[i] = a.Trials
	}
	return out
}
