package stopgate

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

var (
	benchDecisionSink Decision
	benchStringSink   string
	benchBoolSink     bool
	benchStageSink    Stage
)

// BenchmarkCanonicalJSONArgs measures JSON canonicalization throughput across varying payload shapes.
func BenchmarkCanonicalJSONArgs(b *testing.B) {
	flatJSON := `{"timeout":30,"command":"git status --porcelain","workdir":"/workspace/fak"}`
	nestedJSON := `{"session":{"id":"sess-12345","trace":true},"tool":{"name":"edit","params":{"path":"/src/main.go","offset":42}}}`
	nonJSON := `plain non-json argument string with spaces`

	b.Run("Flat", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = CanonicalJSONArgs(flatJSON)
		}
	})

	b.Run("Nested", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = CanonicalJSONArgs(nestedJSON)
		}
	})

	b.Run("NonJSON", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = CanonicalJSONArgs(nonJSON)
		}
	})
}

// BenchmarkToolInvocationSignature measures signature hashing latency for tool calls.
func BenchmarkToolInvocationSignature(b *testing.B) {
	tool := "run_command"
	args := `{"command":"go test ./internal/...","timeout":60,"env":{"CGO_ENABLED":"0"}}`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = ToolInvocationSignature(tool, args)
	}
}

// BenchmarkCircuitBreakerRecordInvocation measures failure tracking and circuit tripping latency.
func BenchmarkCircuitBreakerRecordInvocation(b *testing.B) {
	tool := "fetch_logs"
	args := `{"tail":100,"filter":"error"}`

	b.Run("Success", func(b *testing.B) {
		cb := NewCircuitBreaker()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = cb.RecordInvocation(tool, args, false, true)
		}
	})

	b.Run("FailureIdentical", func(b *testing.B) {
		cb := NewCircuitBreakerWithThreshold(1_000_000_000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = cb.RecordFailure(tool, args)
		}
	})

	b.Run("FailureAlternating", func(b *testing.B) {
		cb := NewCircuitBreakerWithThreshold(1_000_000_000)
		toolA := "tool_a"
		toolB := "tool_b"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if i%2 == 0 {
				benchBoolSink = cb.RecordFailure(toolA, args)
			} else {
				benchBoolSink = cb.RecordFailure(toolB, args)
			}
		}
	})

	b.Run("SignedRefusal", func(b *testing.B) {
		receipt := &BoundaryRefusalReceipt{
			Tool:        tool,
			Reason:      "POLICY_BLOCK",
			Disposition: "TERMINAL",
			Signature:   "ed25519:test_bench_signature_12345",
			Verified:    true,
		}
		cb := NewCircuitBreaker()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cb.Reset()
			benchBoolSink = cb.RecordSignedRefusal(receipt)
		}
	})
}

// BenchmarkLadderEvaluateDenyAll measures graduated ladder adjudication latency.
func BenchmarkLadderEvaluateDenyAll(b *testing.B) {
	cfg := DefaultLadderConfig()
	ladder := NewLadder(cfg)

	b.Run("BlindAllow", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateDenyAll(0, 0, false)
		}
	})

	b.Run("BlindNudge", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateDenyAll(1, 0, false)
		}
	})

	b.Run("BlindWarn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateDenyAll(4, 0, false)
		}
	})

	b.Run("BlindGiveUp", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateDenyAll(10, 0, false)
		}
	})

	b.Run("SameIssueWarn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateDenyAll(0, 4, true)
		}
	})

	b.Run("SameIssueGiveUp", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateDenyAll(0, 7, true)
		}
	})
}

// BenchmarkLadderEvaluateToolFeedback measures tool feedback retry adjudication.
func BenchmarkLadderEvaluateToolFeedback(b *testing.B) {
	cfg := DefaultLadderConfig()
	ladder := NewLadder(cfg)

	b.Run("Nudge", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateToolFeedback(5)
		}
	})

	b.Run("GiveUp", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = ladder.EvaluateToolFeedback(30)
		}
	})
}

// BenchmarkEvaluateWitness measures witness gate evaluation across verification states.
func BenchmarkEvaluateWitness(b *testing.B) {
	cfgEnforce := WitnessGateConfig{Mode: ModeEnforce, Max: 3}
	cfgShadow := WitnessGateConfig{Mode: ModeShadow, Max: 3}

	notClaimed := WitnessClaim{Claimed: false}
	witnessed := WitnessClaim{
		Claimed:   true,
		Witnessed: true,
		Commit:    "8f9a2b1c4d",
		Detail:    "witnessed proof artifact",
	}
	unwitnessed := WitnessClaim{
		Claimed:   true,
		Witnessed: false,
		Reason:    "CLAIM_UNWITNESSED",
		Detail:    "missing test execution witness",
	}

	b.Run("NotClaimed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateWitness(cfgEnforce, notClaimed, 0)
		}
	})

	b.Run("Witnessed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateWitness(cfgEnforce, witnessed, 0)
		}
	})

	b.Run("UnwitnessedEnforce", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateWitness(cfgEnforce, unwitnessed, 1)
		}
	})

	b.Run("UnwitnessedShadow", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateWitness(cfgShadow, unwitnessed, 1)
		}
	})
}

// BenchmarkBoundaryClassification measures predicate classification performance.
func BenchmarkBoundaryClassification(b *testing.B) {
	receiptTerminal := &BoundaryRefusalReceipt{
		Reason:      "POLICY_BLOCK",
		Disposition: "TERMINAL",
		Verified:    true,
	}
	receiptTransient := &BoundaryRefusalReceipt{
		Reason:      "LOCK_BUSY",
		Disposition: "RETRYABLE",
		Verified:    true,
	}

	b.Run("IsTransientHurdle", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = IsTransientHurdle("LOCK_BUSY", "RETRYABLE")
		}
	})

	b.Run("IsTerminalBoundary", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = IsTerminalBoundary(receiptTerminal)
		}
	})

	b.Run("IsTerminalBoundaryTransient", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = IsTerminalBoundary(receiptTransient)
		}
	})

	b.Run("IsSurrenderNoteMatch", func(b *testing.B) {
		text := "I cannot proceed further with the requested changes"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = IsSurrenderNote(text)
		}
	})

	b.Run("IsSurrenderNoteClean", func(b *testing.B) {
		text := "all tests green and verified against main branch"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = IsSurrenderNote(text)
		}
	})
}

// BenchmarkEvaluateBoundary measures unified turn-boundary lifecycle adjudication.
func BenchmarkEvaluateBoundary(b *testing.B) {
	ladder := DefaultLadderConfig()
	witness := WitnessGateConfig{Mode: ModeEnforce, Max: 3}

	inClean := BoundaryInput{}

	cbTripped := NewCircuitBreaker()
	cbTripped.Trip("circuit breaker tripped in benchmark")
	inCircuitBreaker := BoundaryInput{
		CircuitBreaker: cbTripped,
	}

	inDenyAll := BoundaryInput{
		ConsecutiveDenyAll: 3,
	}

	inToolFeedback := BoundaryInput{
		ConsecutiveToolFeedback: 4,
	}

	inTerminalWrapup := BoundaryInput{
		NotedNoAllowedPath: true,
		BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
			Reason:      "POLICY_BLOCK",
			Disposition: "TERMINAL",
			Verified:    true,
		},
	}

	inUnwitnessedNoPath := BoundaryInput{
		NotedNoAllowedPath: true,
	}

	inGoalSurrender := BoundaryInput{
		GoalActive:    true,
		GoalObjective: "implement substantive stopgate benchmarks",
		SurrenderNote: "giving up on task due to complexity",
	}

	b.Run("CleanCompletion", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateBoundary(ladder, witness, inClean)
		}
	})

	b.Run("CircuitBreakerTripped", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateBoundary(ladder, witness, inCircuitBreaker)
		}
	})

	b.Run("DenyAllContinue", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateBoundary(ladder, witness, inDenyAll)
		}
	})

	b.Run("ToolFeedbackContinue", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateBoundary(ladder, witness, inToolFeedback)
		}
	})

	b.Run("NoAllowedPathTerminalWrapup", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateBoundary(ladder, witness, inTerminalWrapup)
		}
	})

	b.Run("NoAllowedPathUnwitnessed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateBoundary(ladder, witness, inUnwitnessedNoPath)
		}
	})

	b.Run("GoalSurrenderBlocked", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDecisionSink = EvaluateBoundary(ladder, witness, inGoalSurrender)
		}
	})
}

// BenchmarkEvaluateBoundaryParallel measures concurrent turn-boundary evaluation throughput.
func BenchmarkEvaluateBoundaryParallel(b *testing.B) {
	ladder := DefaultLadderConfig()
	witness := WitnessGateConfig{Mode: ModeEnforce, Max: 3}
	in := BoundaryInput{
		Turn:               5,
		ConsecutiveDenyAll: 2,
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.Run("Parallel", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				dec := EvaluateBoundary(ladder, witness, in)
				if !dec.Blocked {
					b.Fatal("expected decision to be blocked")
				}
			}
		})
	})
}

// BenchmarkThresholdNormalization measures threshold normalization performance.
func BenchmarkThresholdNormalization(b *testing.B) {
	b.Run("DenyAll", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			w, f, m := NormalizeDenyAllThresholds(3, 7, 9)
			benchBoolSink = (w + f + m) > 0
		}
	})

	b.Run("SameStop", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			w, f, s := NormalizeSameStop(6)
			benchBoolSink = (w + f + s) > 0
		}
	})
}

// BenchmarkStageCalculation measures ladder stage mapping logic.
func BenchmarkStageCalculation(b *testing.B) {
	b.Run("DenyAll", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStageSink = StageForDenyAll(i%12, 3, 7, 9)
		}
	})

	b.Run("SameIssue", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStageSink = StageForSameIssue(i%10, 6)
		}
	})
}

// BenchmarkReasonCodes measures reason code lookup and mapping latency.
func BenchmarkReasonCodes(b *testing.B) {
	names := []string{"POLICY_BLOCK", "LOCK_BUSY", "RATE_LIMITED", "SELF_MODIFY", "LEASE_HELD"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		name := names[i%len(names)]
		code, _ := abi.ReasonByName(name)
		benchStringSink = abi.ReasonName(code)
	}
}

// TestBenchmarkExecutionSanity ensures all benchmark paths run without error.
func TestBenchmarkExecutionSanity(t *testing.T) {
	benchmarks := []struct {
		name string
		fn   func(b *testing.B)
	}{
		{"BenchmarkCanonicalJSONArgs", BenchmarkCanonicalJSONArgs},
		{"BenchmarkToolInvocationSignature", BenchmarkToolInvocationSignature},
		{"BenchmarkCircuitBreakerRecordInvocation", BenchmarkCircuitBreakerRecordInvocation},
		{"BenchmarkLadderEvaluateDenyAll", BenchmarkLadderEvaluateDenyAll},
		{"BenchmarkLadderEvaluateToolFeedback", BenchmarkLadderEvaluateToolFeedback},
		{"BenchmarkEvaluateWitness", BenchmarkEvaluateWitness},
		{"BenchmarkBoundaryClassification", BenchmarkBoundaryClassification},
		{"BenchmarkEvaluateBoundary", BenchmarkEvaluateBoundary},
		{"BenchmarkEvaluateBoundaryParallel", BenchmarkEvaluateBoundaryParallel},
		{"BenchmarkThresholdNormalization", BenchmarkThresholdNormalization},
		{"BenchmarkStageCalculation", BenchmarkStageCalculation},
		{"BenchmarkReasonCodes", BenchmarkReasonCodes},
	}

	for _, bm := range benchmarks {
		t.Run(bm.name, func(t *testing.T) {
			res := testing.Benchmark(bm.fn)
			if res.N <= 0 {
				t.Fatalf("expected benchmark %s to run iterations > 0, got %d", bm.name, res.N)
			}
		})
	}
}
