package resume

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

var (
	benchReport           Report
	benchWatchdogDecision WatchdogRowDecision
	benchResumeState      ResumeState
	benchDriveStates      map[string]WatchdogDriveState
	benchDriveCarries     map[string]DriveCarry
	benchQuarantineDec    QuarantineDecision
	benchContinuity       ContinuityWitness
	benchEarnedBudget     int
	benchLimitClass       LimitClassification
	benchLimitFound       bool
	benchRetryDec         RetryDecision
)

func BenchmarkPlan(b *testing.B) {
	b.ReportAllocs()
	cases := []struct {
		name  string
		input Input
	}{
		{
			name: "cold_250k_5m",
			input: Input{
				ResidentTokens: 250000,
				IdleSeconds:    7200,
				TTL:            TTL5m,
				Pricing:        Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
				HorizonTurns:   20,
			},
		},
		{
			name: "cold_250k_1h",
			input: Input{
				ResidentTokens: 250000,
				IdleSeconds:    7200,
				TTL:            TTL1h,
				Pricing:        Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
				HorizonTurns:   20,
			},
		},
		{
			name: "warm_intact",
			input: Input{
				ResidentTokens: 250000,
				IdleSeconds:    60,
				TTL:            TTL5m,
				Pricing:        Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
				HorizonTurns:   20,
			},
		},
		{
			name: "small_session",
			input: Input{
				ResidentTokens: 15000,
				IdleSeconds:    7200,
				TTL:            TTL5m,
				Pricing:        Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
				HorizonTurns:   20,
			},
		},
		{
			name: "unknown_idle",
			input: Input{
				ResidentTokens: 250000,
				IdleSeconds:    -1,
				TTL:            TTL5m,
				Pricing:        Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
				HorizonTurns:   20,
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			in := tc.input
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchReport = Plan(in)
			}
		})
	}
}

func BenchmarkDecideWatchdogRow(b *testing.B) {
	guards := WatchdogGuards{
		SelfSID:        "sess-self-watchdog",
		WorkerAccounts: map[string]bool{".claude-worker1": true, ".claude-worker2": true},
		MaxAttempts:    4,
		LiveSIDs:       map[string]bool{"sess-active-driver": true},
		DriveStates: map[string]WatchdogDriveState{
			"sess-held-paused": DrivePaused,
		},
		Fleet: FleetPosture{
			State:          FleetTargetDeclared,
			DesiredWorkers: 8,
			Source:         "dos.loop",
		},
	}

	cases := []struct {
		name    string
		row     WatchdogPlanRow
		g       WatchdogGuards
		history []Attempt
		outcome Outcome
	}{
		{
			name: "launch_recoverable",
			row: WatchdogPlanRow{
				Session:   "sess-crashed-1",
				Account:   ".claude-worker1",
				ConfigDir: "/etc/claude/.claude-worker1",
				Disp:      "STOPPED_APIERR",
			},
			g: guards,
			history: []Attempt{
				{Phase: "launch", UnixSeconds: 1000},
			},
			outcome: OutcomeRecoverable,
		},
		{
			name: "skip_self",
			row: WatchdogPlanRow{
				Session:   "sess-self-watchdog",
				Account:   ".claude-worker1",
				ConfigDir: "/etc/claude/.claude-worker1",
			},
			g:       guards,
			outcome: OutcomeRecoverable,
		},
		{
			name: "skip_live",
			row: WatchdogPlanRow{
				Session:   "sess-active-driver",
				Account:   ".claude-worker1",
				ConfigDir: "/etc/claude/.claude-worker1",
			},
			g:       guards,
			outcome: OutcomeRecoverable,
		},
		{
			name: "skip_operator_hold",
			row: WatchdogPlanRow{
				Session:   "sess-held-paused",
				Account:   ".claude-worker1",
				ConfigDir: "/etc/claude/.claude-worker1",
			},
			g:       guards,
			outcome: OutcomeRecoverable,
		},
		{
			name: "skip_quarantine",
			row: WatchdogPlanRow{
				Session:   "sess-quarantine-blocked",
				Account:   ".claude-worker1",
				ConfigDir: "/etc/claude/.claude-worker1",
			},
			g: WatchdogGuards{
				Fleet: FleetPosture{
					State:          FleetTargetDeclared,
					DesiredWorkers: 0,
					Source:         "dos.loop",
				},
			},
			outcome: OutcomeRecoverable,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			row, g, hist, out := tc.row, tc.g, tc.history, tc.outcome
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchWatchdogDecision = DecideWatchdogRow(row, g, hist, out)
			}
		})
	}
}

func BenchmarkFoldResumeState(b *testing.B) {
	cases := []struct {
		name  string
		facts ResumeFacts
	}{
		{
			name: "progressed_witnessed",
			facts: ResumeFacts{
				Attempts:    2,
				MaxAttempts: 5,
				NewTurns:    4,
				Outcome:     OutcomeProgressed,
				Continuity: ContinuityWitness{
					Witnessed: true,
					Advanced:  true,
					PreValue:  0.25,
					PostValue: 0.50,
					W3Rows:    6,
				},
			},
		},
		{
			name: "took_no_progress",
			facts: ResumeFacts{
				Attempts:    2,
				MaxAttempts: 5,
				NewTurns:    4,
				Outcome:     OutcomeProgressed,
				Continuity: ContinuityWitness{
					Witnessed: true,
					Advanced:  false,
					PreValue:  0.50,
					PostValue: 0.50,
					W3Rows:    4,
				},
			},
		},
		{
			name: "re_stranded",
			facts: ResumeFacts{
				Attempts:    2,
				MaxAttempts: 5,
				NewTurns:    0,
				Outcome:     OutcomeRecoverable,
			},
		},
		{
			name: "gave_up_unrecoverable",
			facts: ResumeFacts{
				Attempts:    1,
				MaxAttempts: 5,
				NewTurns:    0,
				Outcome:     OutcomeUnrecoverable,
			},
		},
		{
			name: "pending_fresh",
			facts: ResumeFacts{
				Attempts:    0,
				MaxAttempts: 5,
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			facts := tc.facts
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchResumeState = FoldResumeState(facts)
			}
		})
	}
}

func makeDriveStateRows(n int) []DriveStateRow {
	states := []string{"running", "paused", "draining", "stopped", "throttled"}
	rows := make([]DriveStateRow, n)
	for i := 0; i < n; i++ {
		rows[i] = DriveStateRow{
			Session: fmt.Sprintf("session-%d", i%20),
			State:   states[i%len(states)],
			Via:     "fak resume hold",
		}
	}
	return rows
}

func BenchmarkFoldDriveStates(b *testing.B) {
	for _, size := range []int{10, 50, 250} {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			rows := makeDriveStateRows(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchDriveStates = FoldDriveStates(rows)
			}
		})
	}
}

func makeDriveCarryRows(n int) []DriveStateRow {
	rows := make([]DriveStateRow, n)
	for i := 0; i < n; i++ {
		val := int64(100 - (i % 50))
		rows[i] = DriveStateRow{
			Session:    fmt.Sprintf("session-%d", i%15),
			State:      "running",
			TurnsLeft:  &val,
			TokensLeft: &val,
		}
	}
	return rows
}

func BenchmarkReconcileDriveCarry(b *testing.B) {
	for _, size := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			rows := makeDriveCarryRows(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchDriveCarries = ReconcileDriveCarry(rows)
			}
		})
	}
}

func BenchmarkAdmitQuarantine(b *testing.B) {
	cases := []struct {
		name    string
		posture FleetPosture
		action  RecoveryAction
	}{
		{
			name: "declared_positive",
			posture: FleetPosture{
				State:          FleetTargetDeclared,
				DesiredWorkers: 5,
				Source:         "supervisor",
			},
			action: RecoveryResumeSession,
		},
		{
			name: "quarantined_zero",
			posture: FleetPosture{
				State:          FleetTargetDeclared,
				DesiredWorkers: 0,
				Source:         "dos.loop",
			},
			action: RecoveryResumeSession,
		},
		{
			name: "undeclared",
			posture: FleetPosture{
				State:  FleetTargetUndeclared,
				Source: "untracked",
			},
			action: RecoveryResumeSession,
		},
		{
			name: "status_read_always_admitted",
			posture: FleetPosture{
				State:          FleetTargetDeclared,
				DesiredWorkers: 0,
				Source:         "dos.loop",
			},
			action: RecoveryStatusRead,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			p, act := tc.posture, tc.action
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchQuarantineDec = AdmitQuarantine(p, act)
			}
		})
	}
}

func makeScoreRows(n int) []trajctl.ScoreRow {
	rows := make([]trajctl.ScoreRow, n)
	for i := 0; i < n; i++ {
		rows[i] = trajctl.ScoreRow{
			Witness:     trajctl.W3,
			ObjectiveID: "OBJ-RESUME-1",
			UnixMillis:  int64(1000 + i*100),
			Value:       float64(i) / float64(n),
		}
	}
	return rows
}

func BenchmarkFoldW3Continuity(b *testing.B) {
	for _, size := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			rows := makeScoreRows(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchContinuity = FoldW3Continuity(rows, "OBJ-RESUME-1", 1000+(int64(size/2)*100))
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
				benchEarnedBudget = EarnedResumeBudget(history)
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
			b.ReportAllocs()
			status, body := tc.status, tc.body
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchLimitClass, benchLimitFound = ClassifyLimitResponse(status, body)
			}
		})
	}
}

func BenchmarkRetryGateContinuity(b *testing.B) {
	history := []Attempt{
		{Phase: "launch", UnixSeconds: 1000},
		{Phase: "launch", UnixSeconds: 1600},
	}
	cont := ContinuityWitness{
		Witnessed: true,
		Advanced:  false,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRetryDec = RetryGateContinuity(history, OutcomeProgressed, 4, cont)
	}
}
