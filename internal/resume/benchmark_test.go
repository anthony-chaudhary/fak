package resume

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

var (
	sinkReport              Report
	sinkDiagnosis           Diagnosis
	sinkWatchdogRowDecision WatchdogRowDecision
	sinkRetryDecision       RetryDecision
	sinkResumeState         ResumeState
	sinkNextVerdict         NextVerdict
	sinkWatchdogDrainStatus WatchdogDrainStatus
	sinkDriveStates         map[string]WatchdogDriveState
	sinkDriveCarries        map[string]DriveCarry
	sinkWatchdogCap         WatchdogCap
	sinkRecoveryCost        RecoveryCost
	sinkContinuityWitness   ContinuityWitness
	sinkSoftStateDump       SoftStateDump
	sinkSoftStateDumpBool   bool
	sinkQuarantineDecision  QuarantineDecision
	sinkSourceDecision      SourceDecision
	sinkAttemptErrorClass   AttemptErrorClass
	sinkEnv                 []string
	sinkEarnedBudget        int
	sinkLimitClass          LimitClassification
	sinkLimitFound          bool
)

// BenchmarkPlan measures deterministic resume pricing and strategy recommendations across
// key production operational configurations.
func BenchmarkPlan(b *testing.B) {
	cases := []struct {
		name string
		in   Input
	}{
		{
			name: "Headline250kCold",
			in: Input{
				ResidentTokens: 250000,
				IdleSeconds:    7200,
				TTL:            TTL5m,
				Pricing:        opusPricing,
				HorizonTurns:   20,
			},
		},
		{
			name: "WarmShortHorizon",
			in: Input{
				ResidentTokens: 250000,
				IdleSeconds:    60,
				TTL:            TTL5m,
				Pricing:        opusPricing,
				HorizonTurns:   3,
			},
		},
		{
			name: "WarmLongHorizon",
			in: Input{
				ResidentTokens: 250000,
				IdleSeconds:    60,
				TTL:            TTL5m,
				Pricing:        opusPricing,
				HorizonTurns:   30,
			},
		},
		{
			name: "SmallSession",
			in: Input{
				ResidentTokens: 20000,
				IdleSeconds:    7200,
				TTL:            TTL5m,
				Pricing:        opusPricing,
				HorizonTurns:   20,
			},
		},
		{
			name: "ExtendedTTL1h",
			in: Input{
				ResidentTokens: 250000,
				IdleSeconds:    1800,
				TTL:            TTL1h,
				Pricing:        opusPricing,
				HorizonTurns:   20,
			},
		},
		{
			name: "CrossSessionWarmHit",
			in: Input{
				ResidentTokens:   250000,
				IdleSeconds:      7200,
				TTL:              TTL5m,
				Pricing:          opusPricing,
				HorizonTurns:     20,
				PriorSessionWarm: true,
			},
		},
		{
			name: "unknown_idle",
			in: Input{
				ResidentTokens: 250000,
				IdleSeconds:    -1,
				TTL:            TTL5m,
				Pricing:        opusPricing,
				HorizonTurns:   20,
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			in := tc.in
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkReport = Plan(in)
			}
		})
	}
}

// BenchmarkDiagnose measures transcript event classification and paired restart plan derivation.
func BenchmarkDiagnose(b *testing.B) {
	rateLimitEvents := []Event{
		{Kind: EventRealAssistant, PromptTokens: 45000},
		{Kind: EventRealAssistant, PromptTokens: 90000},
		{Kind: EventRealAssistant, PromptTokens: 140000},
		{Kind: EventRealAssistant, PromptTokens: 190000},
		{Kind: EventRealAssistant, PromptTokens: 245000},
		{Kind: EventRateLimitError, LimitReason: LimitSession},
	}
	cleanEvents := []Event{
		{Kind: EventRealAssistant, PromptTokens: 30000},
		{Kind: EventRealAssistant, PromptTokens: 60000},
		{Kind: EventRealAssistant, PromptTokens: 90000},
		{Kind: EventRealAssistant, PromptTokens: 120000},
	}
	interruptedEvents := []Event{
		{Kind: EventRealAssistant, PromptTokens: 50000},
		{Kind: EventUserTurn},
	}

	makeEvents := func(n int) []Event {
		evs := make([]Event, n)
		for i := 0; i < n; i++ {
			if i == n-1 {
				evs[i] = Event{Kind: EventRateLimitError, LimitReason: LimitWeekly}
			} else if i%3 == 0 {
				evs[i] = Event{Kind: EventRealAssistant, PromptTokens: 10000 + i*100}
			} else if i%3 == 1 {
				evs[i] = Event{Kind: EventUserTurn}
			} else {
				evs[i] = Event{Kind: EventOther}
			}
		}
		return evs
	}

	events100 := makeEvents(100)
	events1000 := makeEvents(1000)

	baseInput := Input{
		IdleSeconds:  7200,
		TTL:          TTL5m,
		Pricing:      opusPricing,
		HorizonTurns: 20,
	}

	cases := []struct {
		name   string
		events []Event
		in     Input
	}{
		{"UnresumedRateLimit", rateLimitEvents, baseInput},
		{"CleanSession", cleanEvents, baseInput},
		{"InterruptedMidTurn", interruptedEvents, Input{IdleSeconds: 600, TTL: TTL5m, Pricing: opusPricing}},
		{"DeepTranscript_100", events100, baseInput},
		{"DeepTranscript_1000", events1000, baseInput},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			evs := tc.events
			in := tc.in
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkDiagnosis = Diagnose(evs, in)
			}
		})
	}
}

// BenchmarkDecideWatchdogRow measures per-row admission gating in the resume watchdog tick loop.
func BenchmarkDecideWatchdogRow(b *testing.B) {
	rowFresh := WatchdogPlanRow{
		Session:   "sid-fresh",
		Account:   ".claude-worker1",
		ConfigDir: "C:/work/.claude-worker1",
	}
	rowRetry := WatchdogPlanRow{
		Session:   "sid-retry",
		Account:   ".claude-worker1",
		ConfigDir: "C:/work/.claude-worker1",
	}
	rowHeld := WatchdogPlanRow{
		Session:   "sid-held",
		Account:   ".claude-worker1",
		ConfigDir: "C:/work/.claude-worker1",
	}
	rowLive := WatchdogPlanRow{
		Session:   "sid-live",
		Account:   ".claude-worker1",
		ConfigDir: "C:/work/.claude-worker1",
	}
	rowCapped := WatchdogPlanRow{
		Session:   "sid-capped",
		Account:   ".claude-worker1",
		ConfigDir: "C:/work/.claude-worker1",
	}
	rowDangling := WatchdogPlanRow{
		Session:       "sid-dangling",
		Account:       ".claude-worker1",
		ConfigDir:     "C:/work/.claude-worker1",
		PartialBlocks: []EmittedBlock{{Kind: BlockToolCall, ToolCallID: "tc-unanswered"}},
	}

	guards := WatchdogGuards{
		WorkerAccounts: map[string]bool{".claude-worker1": true},
		MaxAttempts:    8,
		LiveSIDs:       map[string]bool{"sid-live": true},
		DriveStates:    map[string]WatchdogDriveState{"sid-held": DrivePaused},
		Fleet:          DeclaredFleetTarget(5, "benchmark", "active"),
	}

	historyRetry := []Attempt{
		{UnixSeconds: 1000, Phase: "launched"},
		{UnixSeconds: 2000, Phase: "launched"},
	}

	historyCapped := make([]Attempt, 8)
	for i := 0; i < 8; i++ {
		historyCapped[i] = Attempt{UnixSeconds: int64(1000 + i*600), Phase: "launched"}
	}

	cases := []struct {
		name    string
		row     WatchdogPlanRow
		guards  WatchdogGuards
		history []Attempt
		outcome Outcome
	}{
		{"LaunchAttempt1", rowFresh, guards, nil, OutcomeProgressed},
		{"RecoverableRetry", rowRetry, guards, historyRetry, OutcomeRecoverable},
		{"OperatorHold", rowHeld, guards, nil, OutcomeRecoverable},
		{"LiveSession", rowLive, guards, nil, OutcomeRecoverable},
		{"AttemptCapSpent", rowCapped, guards, historyCapped, OutcomeRecoverable},
		{"ReplaySafetyDanglingTool", rowDangling, guards, nil, OutcomeRecoverable},
		{"QuarantineTarget0", rowFresh, WatchdogGuards{Fleet: DeclaredFleetTarget(0, "bench", "drain")}, nil, OutcomeRecoverable},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			row := tc.row
			g := tc.guards
			hist := tc.history
			out := tc.outcome
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkWatchdogRowDecision = DecideWatchdogRow(row, g, hist, out)
			}
		})
	}
}

// BenchmarkRetryGate measures the outcome-aware retry and continuity gates.
func BenchmarkRetryGate(b *testing.B) {
	historyEarned := make([]Attempt, 20)
	for i := 0; i < 20; i++ {
		gap := int64(300) // thrash
		if i%2 == 0 {
			gap = 1200 // progress
		}
		historyEarned[i] = Attempt{
			UnixSeconds: int64(1000 + int64(i)*gap),
			Phase:       "launched",
		}
	}

	historyOne := []Attempt{{UnixSeconds: 1000, Phase: "launched"}}

	cases := []struct {
		name        string
		history     []Attempt
		outcome     Outcome
		maxAttempts int
		cont        ContinuityWitness
	}{
		{"FirstResume", nil, OutcomeRecoverable, 8, ContinuityWitness{}},
		{"ProgressedBurnOnce", historyOne, OutcomeProgressed, 8, ContinuityWitness{}},
		{"RecoverableEarnedBudget", historyEarned, OutcomeRecoverable, 0, ContinuityWitness{}},
		{"ContinuityTookNoProgress", historyOne, OutcomeProgressed, 8, ContinuityWitness{Witnessed: true, Advanced: false}},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			hist := tc.history
			out := tc.outcome
			max := tc.maxAttempts
			cont := tc.cont
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkRetryDecision = RetryGateContinuity(hist, out, max, cont)
			}
		})
	}
}

// BenchmarkFoldResumeState measures lifecycle state derivation over typed session facts.
func BenchmarkFoldResumeState(b *testing.B) {
	cases := []struct {
		name  string
		facts ResumeFacts
	}{
		{
			name:  "Pending",
			facts: ResumeFacts{Attempts: 0},
		},
		{
			name:  "Took",
			facts: ResumeFacts{Attempts: 1, NewTurns: 3, Outcome: OutcomeProgressed, Continuity: ContinuityWitness{Witnessed: true, Advanced: true}},
		},
		{
			name:  "TookNoProgress",
			facts: ResumeFacts{Attempts: 1, NewTurns: 3, Outcome: OutcomeProgressed, Continuity: ContinuityWitness{Witnessed: true, Advanced: false}},
		},
		{
			name:  "ReStranded",
			facts: ResumeFacts{Attempts: 1, Outcome: OutcomeRecoverable},
		},
		{
			name:  "GaveUp",
			facts: ResumeFacts{Attempts: 8, MaxAttempts: 8, Outcome: OutcomeRecoverable},
		},
		{
			name:  "Settled",
			facts: ResumeFacts{Attempts: 2, OperatorSettled: true},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			f := tc.facts
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkResumeState = FoldResumeState(f)
			}
		})
	}
}

// BenchmarkFoldNextAction measures actionable runbook decision derivation.
func BenchmarkFoldNextAction(b *testing.B) {
	cases := []struct {
		name  string
		input NextInput
	}{
		{
			name: "ActRun",
			input: NextInput{
				State:    ResumeReStranded,
				Outcome:  OutcomeRecoverable,
				Retry:    RetryDecision{Blocked: false},
				Admitted: true,
			},
		},
		{
			name: "ActWaitReset",
			input: NextInput{
				State:       ResumeReStranded,
				Outcome:     OutcomeRecoverable,
				Retry:       RetryDecision{Blocked: false},
				LimitReason: LimitSession,
				IdleSeconds: 3600,
				Admitted:    true,
			},
		},
		{
			name: "ActHoldAdmission",
			input: NextInput{
				State:       ResumeReStranded,
				Outcome:     OutcomeRecoverable,
				Retry:       RetryDecision{Blocked: false},
				Admitted:    false,
				AdmitReason: "529 burst ceiling",
			},
		},
		{
			name: "ActWaitProgress",
			input: NextInput{
				State:    ResumeTookNoProgress,
				Outcome:  OutcomeProgressed,
				Retry:    RetryDecision{Blocked: false},
				Admitted: true,
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			in := tc.input
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkNextVerdict = FoldNextAction(in)
			}
		})
	}
}

// BenchmarkFoldWatchdogStatus measures fleet drain aggregation and SLO page detection.
func BenchmarkFoldWatchdogStatus(b *testing.B) {
	makeStatusInput := func(n int) WatchdogStatusInput {
		plan := make([]WatchdogPlanRow, n)
		events := make([]WatchdogStatusEvent, 0, n*2+2)
		now := int64(20_000)
		for i := 0; i < n; i++ {
			sid := fmt.Sprintf("sid-%d", i)
			plan[i] = WatchdogPlanRow{Session: sid, Account: ".claude-a"}
			events = append(events,
				WatchdogStatusEvent{UnixSeconds: 1000 + int64(i), Session: sid, Phase: "queued", Mode: "LIVE"},
				WatchdogStatusEvent{UnixSeconds: 1200 + int64(i), Session: sid, Phase: "launched", Mode: "LIVE"},
			)
			if i%2 == 0 {
				events = append(events,
					WatchdogStatusEvent{UnixSeconds: 1500 + int64(i), Session: sid, Phase: "progress", Mode: "LIVE", NewTurns: 3},
				)
			}
		}
		events = append(events,
			WatchdogStatusEvent{UnixSeconds: 1000, Phase: "status", Mode: "LIVE", AutoResumeDepth: n},
			WatchdogStatusEvent{UnixSeconds: 2000, Phase: "status", Mode: "LIVE", AutoResumeDepth: n / 2},
		)
		return WatchdogStatusInput{
			Mode:           "LIVE",
			NowUnix:        now,
			SilentSeconds:  3600,
			MonotonicTicks: 3,
			Plan:           plan,
			Events:         events,
		}
	}

	in10 := makeStatusInput(10)
	in100 := makeStatusInput(100)
	in500 := makeStatusInput(500)

	cases := []struct {
		name  string
		input WatchdogStatusInput
	}{
		{"Queue_10", in10},
		{"Queue_100", in100},
		{"Queue_500", in500},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			in := tc.input
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkWatchdogDrainStatus = FoldWatchdogStatus(in)
			}
		})
	}
}

// BenchmarkDriveStateFolds measures operator hold parsing and monotone budget reconciliation.
func BenchmarkDriveStateFolds(b *testing.B) {
	makeDriveRows := func(n int) []DriveStateRow {
		rows := make([]DriveStateRow, n)
		states := []string{"running", "paused", "throttled", "draining", "stopped"}
		for i := 0; i < n; i++ {
			sid := fmt.Sprintf("session-%d", i%20)
			st := states[i%len(states)]
			turns := int64(100 - (i % 50))
			tokens := int64(50000 - (i%50)*1000)
			reArm := (i == n/2)
			rows[i] = DriveStateRow{
				Session:    sid,
				State:      st,
				TurnsLeft:  &turns,
				TokensLeft: &tokens,
				ReArm:      reArm,
				TS:         "2026-09-05T12:00:00Z",
			}
		}
		return rows
	}

	rows100 := makeDriveRows(100)

	b.Run("FoldDriveStates_100Rows", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkDriveStates = FoldDriveStates(rows100)
		}
	})

	b.Run("ReconcileDriveCarry_100Rows", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkDriveCarries = ReconcileDriveCarry(rows100)
		}
	})
}

// BenchmarkDeriveWatchdogCap measures dynamic launch capacity derivation from seat health.
func BenchmarkDeriveWatchdogCap(b *testing.B) {
	makeSeats := func(n int) []HeadroomSeat {
		seats := make([]HeadroomSeat, n)
		for i := 0; i < n; i++ {
			seats[i] = HeadroomSeat{
				Available:      i%5 != 0,
				Throttled:      i%7 == 0,
				ActiveSessions: i % 4,
			}
		}
		return seats
	}

	seats10 := makeSeats(10)
	seats100 := makeSeats(100)

	b.Run("Fleet_10Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkWatchdogCap = DeriveWatchdogCap(seats10, 1, 32, 4)
		}
	})

	b.Run("Fleet_100Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkWatchdogCap = DeriveWatchdogCap(seats100, 1, 32, 4)
		}
	})
}

// BenchmarkFoldRecoveryCost measures observed post-resume provider spend summation.
func BenchmarkFoldRecoveryCost(b *testing.B) {
	makeTurns := func(n int) []RecoveryTurnCost {
		turns := make([]RecoveryTurnCost, n)
		for i := 0; i < n; i++ {
			turns[i] = RecoveryTurnCost{
				UnixSeconds:  int64(1000 + i*60),
				Tokens:       500 + (i%10)*50,
				CostMicroUSD: int64(1500 + (i%10)*150),
			}
		}
		return turns
	}

	turns10 := makeTurns(10)
	turns100 := makeTurns(100)
	turns1000 := makeTurns(1000)

	cases := []struct {
		name  string
		turns []RecoveryTurnCost
	}{
		{"Turns_10", turns10},
		{"Turns_100", turns100},
		{"Turns_1000", turns1000},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			turns := tc.turns
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkRecoveryCost = FoldRecoveryCost(turns, 1500)
			}
		})
	}
}

// BenchmarkFoldW3Continuity measures trajectory W3 verified-progress witness folding.
func BenchmarkFoldW3Continuity(b *testing.B) {
	makeScoreRows := func(n int) []trajctl.ScoreRow {
		rows := make([]trajctl.ScoreRow, n)
		for i := 0; i < n; i++ {
			rows[i] = trajctl.ScoreRow{
				ObjectiveID: "obj-main",
				SessionID:   fmt.Sprintf("sess-%d", i%5),
				Method:      trajctl.CommitScorerMethod,
				Witness:     trajctl.W3,
				Value:       float64(i) / float64(n),
				UnixMillis:  int64(10000 + i*100),
			}
		}
		return rows
	}

	rows10 := makeScoreRows(10)
	rows100 := makeScoreRows(100)
	rows1000 := makeScoreRows(1000)

	cases := []struct {
		name string
		rows []trajctl.ScoreRow
	}{
		{"Rows_10", rows10},
		{"Rows_100", rows100},
		{"Rows_1000", rows1000},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			rows := tc.rows
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkContinuityWitness = FoldW3Continuity(rows, "obj-main", 25000)
			}
		})
	}
}

// BenchmarkDecideSoftStateDump measures clock-free diagnostic capture for stalled sessions.
func BenchmarkDecideSoftStateDump(b *testing.B) {
	now := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)
	stalledObs := SoftWatchdogObservation{
		SessionID:          "sid-stalled",
		Alive:              true,
		Signal:             trajctl.SignalStall,
		LastProgressMarker: "commit-abc",
		LastProgressAt:     now.Add(-5 * time.Minute),
		PendingAction:      "thinking",
		Now:                now,
	}
	healthyObs := SoftWatchdogObservation{
		SessionID:          "sid-healthy",
		Alive:              true,
		Signal:             trajctl.SignalHealthy,
		LastProgressMarker: "commit-def",
		LastProgressAt:     now.Add(-10 * time.Second),
		PendingAction:      "executing",
		Now:                now,
	}

	b.Run("StalledAlive", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkSoftStateDump, sinkSoftStateDumpBool = DecideSoftStateDump(stalledObs, DefaultSoftStallGrace)
		}
	})

	b.Run("HealthyProgressing", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkSoftStateDump, sinkSoftStateDumpBool = DecideSoftStateDump(healthyObs, DefaultSoftStallGrace)
		}
	})
}

// BenchmarkAdmitQuarantine measures the fail-closed fleet quarantine gate.
func BenchmarkAdmitQuarantine(b *testing.B) {
	postureDeclared := DeclaredFleetTarget(10, "benchmark", "fleet running")
	postureQuarantined := DeclaredFleetTarget(0, "benchmark", "operator hold")

	cases := []struct {
		name    string
		posture FleetPosture
		action  RecoveryAction
	}{
		{"DeclaredTarget", postureDeclared, RecoveryResumeSession},
		{"QuarantinedTarget0", postureQuarantined, RecoveryResumeSession},
		{"ReadOnlyAction", postureQuarantined, RecoveryStatusRead},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			p := tc.posture
			act := tc.action
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkQuarantineDecision = AdmitQuarantine(p, act)
			}
		})
	}
}

// BenchmarkAdmitSource measures host-wide 529 burst and spacing floor admission.
func BenchmarkAdmitSource(b *testing.B) {
	now := time.Unix(300, 0).UTC()
	snapUnderCeiling := SourceSnapshot{
		LiveResumeCount: 2,
		LaunchUnixTimes: []int64{100, 200},
		LastLaunchUnix:  200,
	}
	snapRateLimited := SourceSnapshot{
		LiveResumeCount: 1,
		LaunchUnixTimes: []int64{280, 285, 290, 295},
		LastLaunchUnix:  295,
	}

	polCeiling := SourcePolicy{
		MaxLiveResumes: 4,
	}
	polRate := SourcePolicy{
		WindowSeconds:        60,
		MaxLaunchesPerWindow: 3,
	}

	b.Run("Permissive", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkSourceDecision = AdmitSource(snapUnderCeiling, SourcePolicy{}, now)
		}
	})

	b.Run("UnderCeiling", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkSourceDecision = AdmitSource(snapUnderCeiling, polCeiling, now)
		}
	})

	b.Run("RateLimited", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkSourceDecision = AdmitSource(snapRateLimited, polRate, now)
		}
	})
}

// BenchmarkWatchdogChildEnv measures child environment sanitization and isolation.
func BenchmarkWatchdogChildEnv(b *testing.B) {
	environ := make([]string, 60)
	for i := 0; i < 60; i++ {
		environ[i] = fmt.Sprintf("VAR_%d=value_%d", i, i)
	}
	// Inject the forbidden keys
	for _, key := range WatchdogChildEnvDrop {
		environ = append(environ, key+"=secret_or_inherited")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkEnv = WatchdogChildEnv(environ, "C:/work/.claude-target")
	}
}

// BenchmarkCodexWatchdogChildEnv measures Codex child environment sanitization and isolation.
func BenchmarkCodexWatchdogChildEnv(b *testing.B) {
	environ := make([]string, 60)
	for i := 0; i < 60; i++ {
		environ[i] = fmt.Sprintf("VAR_%d=value_%d", i, i)
	}
	// Inject the forbidden keys
	for _, key := range WatchdogChildEnvDrop {
		environ = append(environ, key+"=secret_or_inherited")
	}
	environ = append(environ, "CLAUDE_CONFIG_DIR=secret_or_inherited")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkEnv = CodexWatchdogChildEnv(environ)
	}
}

// BenchmarkClassifyAttemptError measures child stderr pattern parsing into typed causes.
func BenchmarkClassifyAttemptError(b *testing.B) {
	cases := []struct {
		name string
		text string
	}{
		{"Malformed400", "fatal error: child_crash with status 400 bad request in model stream"},
		{"Auth", "error: authentication failed - invalid api key for seat"},
		{"Usage", "rate limit exceeded: usage limit reached for tier"},
		{"WireCrash", "transport failure: connection reset by peer on wire"},
		{"Unknown", "unexpected internal agent error during execution"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			txt := tc.text
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkAttemptErrorClass = ClassifyAttemptError(txt)
			}
		})
	}
}

func makeAttemptHistory(n int) []Attempt {
	attempts := make([]Attempt, n)
	for i := 0; i < n; i++ {
		attempts[i] = Attempt{
			Phase:       "launch",
			UnixSeconds: int64(1000 + i*600),
		}
	}
	return attempts
}

func BenchmarkEarnedResumeBudget(b *testing.B) {
	for _, count := range []int{2, 10, 50} {
		b.Run(fmt.Sprintf("launches_%d", count), func(b *testing.B) {
			history := makeAttemptHistory(count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkEarnedBudget = EarnedResumeBudget(history)
			}
		})
	}
}

func BenchmarkClassifyLimitResponse(b *testing.B) {
	openaiBody := []byte(`{"error":{"message":"Rate limit reached for requests","type":"tokens","param":null,"code":"rate_limit_exceeded"}}`)
	anthropicBody := []byte(`{"type":"error","error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded your daily limit"}}`)
	plainBody := []byte(`session limit reached, resets in 4 hours`)

	cases := []struct {
		name   string
		status int
		body   []byte
	}{
		{"openai_429", 429, openaiBody},
		{"anthropic_429", 429, anthropicBody},
		{"plain_429", 429, plainBody},
		{"non_429_status", 500, openaiBody},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			status, body := tc.status, tc.body
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkLimitClass, sinkLimitFound = ClassifyLimitResponse(status, body)
			}
		})
	}
}
