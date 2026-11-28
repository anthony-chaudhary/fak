package ops

import (
	"testing"
)

func TestLoadVerdict(t *testing.T) {
	measuredHost := &HostLoad{
		State:         HostSampleMeasured,
		TotalRAMBytes: 64 << 30,
		AvailRAMBytes: 32 << 30,
		HaveRAM:       true,
	}
	unknownHost := &HostLoad{State: HostSampleUnknown}
	errorHost := &HostLoad{State: HostSampleError, Error: "sensor offline"}

	t.Run("closed verdict vocabulary refuses out-of-vocabulary values", func(t *testing.T) {
		valid := []LoadVerdict{
			LoadProductive,
			LoadIdle,
			LoadSkipped,
			LoadPressureRelief,
			LoadPressureReliefOnly,
			LoadUnknown,
			LoadError,
		}
		for _, v := range valid {
			if !ValidLoadVerdict(v) {
				t.Errorf("ValidLoadVerdict(%q) = false, want true", v)
			}
		}
		invalid := []LoadVerdict{"", "SLEEPY", "BUSY", "PAUSED"}
		for _, v := range invalid {
			if ValidLoadVerdict(v) {
				t.Errorf("ValidLoadVerdict(%q) = true, want false", v)
			}
			c := LoadCycle{Schema: LoadSchemaV1, Tick: 1, Verdict: v}
			if err := c.Validate(); err == nil {
				t.Errorf("LoadCycle.Validate() accepted out-of-vocabulary verdict %q", v)
			}
		}
	})

	t.Run("table-driven classification", func(t *testing.T) {
		tests := []struct {
			name       string
			cycle      LoadCycle
			pauseState PauseState
			want       LoadVerdict
		}{
			{
				name: "tick that evicted a lock is PRODUCTIVE",
				cycle: LoadCycle{
					Schema:    LoadSchemaV1,
					Tick:      1,
					Mutations: 1,
					Host:      measuredHost,
					Phases: []PhaseLoad{
						{Phase: "locks_worktrees", Ran: true, Mutated: true},
						{Phase: "process_runaways", Ran: true, Mutated: false},
					},
				},
				want: LoadProductive,
			},
			{
				name: "tick that ran lock and process sweeps and found nothing is IDLE",
				cycle: LoadCycle{
					Schema:    LoadSchemaV1,
					Tick:      2,
					Mutations: 0,
					Host:      measuredHost,
					Phases: []PhaseLoad{
						{Phase: "locks_worktrees", Ran: true, Mutated: false},
						{Phase: "process_runaways", Ran: true, Mutated: false},
					},
				},
				want: LoadIdle,
			},
			{
				name: "unreadable host sample classifies as UNKNOWN",
				cycle: LoadCycle{
					Schema:    LoadSchemaV1,
					Tick:      3,
					Mutations: 0,
					Host:      unknownHost,
					Phases: []PhaseLoad{
						{Phase: "locks_worktrees", Ran: true, Mutated: false},
						{Phase: "process_runaways", Ran: true, Mutated: false},
					},
				},
				want: LoadUnknown,
			},
			{
				name: "error host sample classifies as UNKNOWN",
				cycle: LoadCycle{
					Schema:    LoadSchemaV1,
					Tick:      4,
					Mutations: 0,
					Host:      errorHost,
					Phases: []PhaseLoad{
						{Phase: "locks_worktrees", Ran: true, Mutated: false},
					},
				},
				want: LoadUnknown,
			},
			{
				name: "failed phase classifies as ERROR",
				cycle: LoadCycle{
					Schema:  LoadSchemaV1,
					Tick:    5,
					Verdict: LoadError,
					Host:    measuredHost,
					Phases: []PhaseLoad{
						{Phase: "locks_worktrees", Ran: true, Mutated: false},
					},
				},
				want: LoadError,
			},
			{
				name: "deferred tick with reason_class classifies as SKIPPED",
				cycle: LoadCycle{
					Schema:      LoadSchemaV1,
					Tick:        6,
					PauseAction: string(ActionSkip),
					ReasonClass: PauseReasonOperatorRequested,
					Host:        measuredHost,
					Phases: []PhaseLoad{
						{Phase: "locks_worktrees", Ran: false, PhaseAction: string(ActionSkip)},
						{Phase: "asset_sync", Ran: false, Deferrable: true, PhaseAction: string(ActionSkip)},
					},
				},
				pauseState: PauseState{
					Schema:      PauseSchemaV1,
					Paused:      true,
					ReasonClass: PauseReasonOperatorRequested,
				},
				want: LoadSkipped,
			},
			{
				name: "pressure-relief only tick classifies as PRESSURE_RELIEF",
				cycle: LoadCycle{
					Schema:      LoadSchemaV1,
					Tick:        7,
					PauseAction: string(ActionSkip),
					ReasonClass: PauseReasonHostPressure,
					Host:        measuredHost,
					Phases: []PhaseLoad{
						{Phase: "locks_worktrees", Ran: true, PhaseAction: string(ActionRunPressureReliefOnly)},
						{Phase: "process_runaways", Ran: true, PhaseAction: string(ActionRunPressureReliefOnly)},
						{Phase: "asset_sync", Ran: false, Deferrable: true, PhaseAction: string(ActionSkip)},
					},
				},
				pauseState: PauseState{
					Schema:      PauseSchemaV1,
					Paused:      true,
					ReasonClass: PauseReasonHostPressure,
				},
				want: LoadPressureRelief,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := ClassifyLoadCycle(tc.cycle, tc.pauseState)
				if got != tc.want {
					t.Errorf("ClassifyLoadCycle() = %q, want %q", got, tc.want)
				}
			})
		}
	})

	t.Run("stretch policy requires 3 consecutive measured IDLE ticks and resets on any other verdict", func(t *testing.T) {
		idleCycle := LoadCycle{
			Schema: LoadSchemaV1,
			Tick:   1,
			Host:   measuredHost,
			Phases: []PhaseLoad{
				{Phase: "locks_worktrees", Ran: true},
				{Phase: "process_runaways", Ran: true},
			},
		}
		unknownCycle := LoadCycle{
			Schema: LoadSchemaV1,
			Tick:   2,
			Host:   unknownHost,
			Phases: []PhaseLoad{
				{Phase: "locks_worktrees", Ran: true},
			},
		}
		errorCycle := LoadCycle{
			Schema:  LoadSchemaV1,
			Tick:    3,
			Verdict: LoadError,
			Host:    measuredHost,
		}
		skippedCycle := LoadCycle{
			Schema:      LoadSchemaV1,
			Tick:        4,
			PauseAction: string(ActionSkip),
			ReasonClass: PauseReasonOperatorRequested,
			Host:        measuredHost,
			Phases: []PhaseLoad{
				{Phase: "asset_sync", Ran: false, Deferrable: true, PhaseAction: string(ActionSkip)},
			},
		}
		productiveCycle := LoadCycle{
			Schema:    LoadSchemaV1,
			Tick:      5,
			Mutations: 1,
			Host:      measuredHost,
			Phases: []PhaseLoad{
				{Phase: "locks_worktrees", Ran: true, Mutated: true},
			},
		}

		policy := StretchPolicy{} // defaults to 3

		// 1st and 2nd idle ticks must NOT license a stretch.
		d1 := ClassifyAndEvaluateStretch(0, idleCycle, PauseState{}, policy)
		if d1.Verdict != LoadIdle || d1.IdleStreak != 1 || d1.StretchAllowed {
			t.Fatalf("tick 1: got %+v, want IDLE streak=1 stretch=false", d1)
		}
		d2 := ClassifyAndEvaluateStretch(d1.IdleStreak, idleCycle, PauseState{}, policy)
		if d2.Verdict != LoadIdle || d2.IdleStreak != 2 || d2.StretchAllowed {
			t.Fatalf("tick 2: got %+v, want IDLE streak=2 stretch=false", d2)
		}

		// UNKNOWN in the middle of a 3-tick idle run resets the streak and licenses no stretch.
		dUnknown := ClassifyAndEvaluateStretch(d2.IdleStreak, unknownCycle, PauseState{}, policy)
		if dUnknown.Verdict != LoadUnknown || dUnknown.IdleStreak != 0 || dUnknown.StretchAllowed {
			t.Fatalf("unknown tick: got %+v, want UNKNOWN streak=0 stretch=false", dUnknown)
		}

		// Rebuild streak to 3: only on the 3rd consecutive measured IDLE tick is stretch licensed.
		d1 = ClassifyAndEvaluateStretch(dUnknown.IdleStreak, idleCycle, PauseState{}, policy)
		d2 = ClassifyAndEvaluateStretch(d1.IdleStreak, idleCycle, PauseState{}, policy)
		d3 := ClassifyAndEvaluateStretch(d2.IdleStreak, idleCycle, PauseState{}, policy)
		if d3.Verdict != LoadIdle || d3.IdleStreak != 3 || !d3.StretchAllowed {
			t.Fatalf("tick 3: got %+v, want IDLE streak=3 stretch=true", d3)
		}

		// ERROR resets streak and licenses no stretch.
		dErr := ClassifyAndEvaluateStretch(d3.IdleStreak, errorCycle, PauseState{}, policy)
		if dErr.Verdict != LoadError || dErr.IdleStreak != 0 || dErr.StretchAllowed {
			t.Fatalf("error tick: got %+v, want ERROR streak=0 stretch=false", dErr)
		}

		// SKIPPED resets streak and is not counted as evidence about the host.
		dSkip := ClassifyAndEvaluateStretch(2, skippedCycle, PauseState{Paused: true, ReasonClass: PauseReasonOperatorRequested}, policy)
		if dSkip.Verdict != LoadSkipped || dSkip.IdleStreak != 0 || dSkip.StretchAllowed {
			t.Fatalf("skipped tick: got %+v, want SKIPPED streak=0 stretch=false", dSkip)
		}

		// PRODUCTIVE resets streak and licenses no stretch.
		dProd := ClassifyAndEvaluateStretch(3, productiveCycle, PauseState{}, policy)
		if dProd.Verdict != LoadProductive || dProd.IdleStreak != 0 || dProd.StretchAllowed {
			t.Fatalf("productive tick: got %+v, want PRODUCTIVE streak=0 stretch=false", dProd)
		}

		// Configurable threshold (e.g. 5).
		custom := StretchPolicy{RequiredIdleStreak: 5}
		dCustom := EvaluateStretch(2, LoadIdle, measuredHost, custom)
		if dCustom.IdleStreak != 3 || dCustom.StretchAllowed {
			t.Fatalf("custom threshold 5 at streak 3: got %+v, want stretch=false", dCustom)
		}
		dCustom5 := EvaluateStretch(4, LoadIdle, measuredHost, custom)
		if dCustom5.IdleStreak != 5 || !dCustom5.StretchAllowed {
			t.Fatalf("custom threshold 5 at streak 5: got %+v, want stretch=true", dCustom5)
		}
	})
}
