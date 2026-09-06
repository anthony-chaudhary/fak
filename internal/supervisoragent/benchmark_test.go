package supervisoragent

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetmon"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

var (
	benchInputSink     SupervisorInput
	benchVerdictSink   EnvelopeVerdict
	benchEffectSink    ActionEffect
	benchKeepSink      KeepVerdict
	benchWitnessesSink []string
	benchBoolSink      bool
)

type benchVerbs struct{}

func (benchVerbs) Arbitrate(lane string, tree []string) (Lease, error) {
	return Lease{Lane: lane, Kind: "cluster", Tree: tree}, nil
}

func (benchVerbs) Admit(issue, lane, supersedes string) (AdmitReceipt, error) {
	return AdmitReceipt{RunID: "RID-bench", Issue: issue, Lane: lane, Supersedes: supersedes}, nil
}

func (benchVerbs) EmitEscalation(head Escalation) (Escalation, error) {
	head.ID = "PKT-bench"
	return head, nil
}

func makeBenchSources(workerCount, escalationCount, leaseCount int) Sources {
	workers := make([]WorkerCensus, workerCount)
	classes := []fleetmon.Classification{
		fleetmon.ClassHealthy,
		fleetmon.ClassCompletedFinal,
		fleetmon.ClassDead,
		fleetmon.ClassStaleTranscript,
		fleetmon.ClassStaleChild,
		fleetmon.ClassAuthRateBlocked,
		fleetmon.ClassAttention,
	}
	for i := 0; i < workerCount; i++ {
		workers[i] = WorkerCensus{
			RunID: fmt.Sprintf("RID-%03d", i),
			Issue: fmt.Sprintf("ISSUE-%04d", 4000+i),
			Lane:  fmt.Sprintf("lane-%02d", i%8),
			Class: classes[i%len(classes)],
		}
	}

	escalations := make([]Escalation, escalationCount)
	for i := 0; i < escalationCount; i++ {
		escalations[i] = Escalation{
			ID:         fmt.Sprintf("PKT-%03d", i),
			RunID:      fmt.Sprintf("RID-%03d", i),
			Issue:      fmt.Sprintf("ISSUE-%04d", 4000+i),
			Class:      "stall",
			Severity:   "operator",
			ReasonCode: "REQUIRE_WITNESS",
		}
	}

	leases := make([]leaseref.ArbiterLease, leaseCount)
	for i := 0; i < leaseCount; i++ {
		laneName := fmt.Sprintf("lane-%02d", i)
		leases[i] = leaseref.ArbiterLease{
			Lane:     laneName,
			LaneKind: "cluster",
			Tree:     []string{fmt.Sprintf("internal/%s/**", laneName)},
		}
	}

	return Sources{
		Liveness: Seen(Liveness{
			RunID:     "RID-root",
			Class:     "moving",
			Commits:   12,
			SilentFor: 45 * time.Second,
			Region: []LaneRef{
				{Lane: "supervisoragent", Kind: "cluster"},
			},
		}),
		Workers:     Seen(workers),
		Escalations: Seen(escalations),
		Leases:      Seen(leases),
	}
}

func BenchmarkAssemble(b *testing.B) {
	typical := makeBenchSources(10, 3, 5)
	scaled := makeBenchSources(100, 20, 30)
	withAbsent := Sources{
		Liveness:    Seen(Liveness{RunID: "RID-root", Class: "moving", Commits: 5}),
		Workers:     Absent[[]WorkerCensus](),
		Escalations: Seen([]Escalation{{ID: "PKT-1", RunID: "RID-1", Issue: "4478", Class: "stall", Severity: "operator", ReasonCode: "REQUIRE_WITNESS"}}),
		Leases:      Absent[[]leaseref.ArbiterLease](),
	}

	b.Run("TypicalFleet", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchInputSink = Assemble(typical)
		}
	})

	b.Run("ScaledFleet", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchInputSink = Assemble(scaled)
		}
	})

	b.Run("WithAbsent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchInputSink = Assemble(withAbsent)
		}
	})
}

func BenchmarkAbsentWitnesses(b *testing.B) {
	inPresent := Assemble(makeBenchSources(10, 2, 4))
	inAbsent := SupervisorInput{
		Liveness:    Seen(Liveness{RunID: "RID-1"}),
		Workers:     Absent[[]WorkerVerdict](),
		Escalations: Absent[[]Escalation](),
		Leases:      Seen([]Lease{}),
	}

	b.Run("AllPresent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchWitnessesSink = inPresent.AbsentWitnesses()
			benchBoolSink = inPresent.AnyAbsent()
		}
	})

	b.Run("TwoAbsent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchWitnessesSink = inAbsent.AbsentWitnesses()
			benchBoolSink = inAbsent.AnyAbsent()
		}
	})
}

func BenchmarkAuthorize(b *testing.B) {
	in := Assemble(makeBenchSources(8, 2, 3))
	spawnAct := SpawnAction{Issue: "4480", Lane: "supervisoragent"}
	holdAct := HoldAction{}
	escAct := EscalateAction{RunID: "RID-1", Issue: "4480", Class: "stall", Severity: "operator", ReasonCode: "REQUIRE_WITNESS"}

	demandSpawn, _ := DemandFor(ActionSpawn)
	earnedEnv := Envelope{Widths: map[ActionKind]int{ActionSpawn: demandSpawn}}
	subEnv := Envelope{Widths: map[ActionKind]int{ActionSpawn: demandSpawn - 1}}
	refusedEnv := Envelope{Widths: map[ActionKind]int{ActionSpawn: demandSpawn}, WitnessRefused: true}

	absentIn := in
	absentIn.Workers = Absent[[]WorkerVerdict]()

	b.Run("Spawn_Unattended", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = Authorize(in, earnedEnv, ModeEnforce, spawnAct)
		}
	})

	b.Run("Spawn_ConfirmRequired", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = Authorize(in, subEnv, ModeEnforce, spawnAct)
		}
	})

	b.Run("Spawn_WarnSoak", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = Authorize(in, subEnv, ModeWarn, spawnAct)
		}
	})

	b.Run("FailClosed_WitnessRefused", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = Authorize(in, refusedEnv, ModeEnforce, spawnAct)
		}
	})

	b.Run("FailClosed_SurfaceAbsent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = Authorize(absentIn, earnedEnv, ModeEnforce, spawnAct)
		}
	})

	b.Run("Narrowing_Hold", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = Authorize(in, Envelope{}, ModeEnforce, holdAct)
		}
	})

	b.Run("Narrowing_Escalate", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVerdictSink = Authorize(in, Envelope{}, ModeEnforce, escAct)
		}
	})
}

func BenchmarkLower(b *testing.B) {
	verbs := benchVerbs{}
	spawn := SpawnAction{Issue: "4480", Lane: "supervisoragent"}
	replace := ReplaceAction{RunID: "RID-dead", Issue: "4480", Lane: "supervisoragent"}
	redispatch := RedispatchAction{Issue: "4480", Lane: "supervisoragent"}
	widen := WidenAction{Lane: "supervisoragent", Tree: []string{"internal/supervisoragent/**"}}
	escalate := EscalateAction{RunID: "RID-1", Issue: "4480", Class: "stall", Severity: "operator", ReasonCode: "REQUIRE_WITNESS"}
	hold := HoldAction{}

	b.Run("Spawn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eff, err := Lower(spawn, verbs)
			if err != nil {
				b.Fatal(err)
			}
			benchEffectSink = eff
		}
	})

	b.Run("Replace", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eff, err := Lower(replace, verbs)
			if err != nil {
				b.Fatal(err)
			}
			benchEffectSink = eff
		}
	})

	b.Run("Redispatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eff, err := Lower(redispatch, verbs)
			if err != nil {
				b.Fatal(err)
			}
			benchEffectSink = eff
		}
	})

	b.Run("Widen", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eff, err := Lower(widen, verbs)
			if err != nil {
				b.Fatal(err)
			}
			benchEffectSink = eff
		}
	})

	b.Run("Escalate", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eff, err := Lower(escalate, verbs)
			if err != nil {
				b.Fatal(err)
			}
			benchEffectSink = eff
		}
	})

	b.Run("Hold", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			eff, err := Lower(hold, verbs)
			if err != nil {
				b.Fatal(err)
			}
			benchEffectSink = eff
		}
	})
}

func BenchmarkLowerInEnvelope(b *testing.B) {
	in := Assemble(makeBenchSources(6, 1, 2))
	verbs := benchVerbs{}
	spawn := SpawnAction{Issue: "4480", Lane: "supervisoragent"}
	hold := HoldAction{}

	demandSpawn, _ := DemandFor(ActionSpawn)
	earnedEnv := Envelope{Widths: map[ActionKind]int{ActionSpawn: demandSpawn}}
	subEnv := Envelope{Widths: map[ActionKind]int{ActionSpawn: demandSpawn - 1}}

	b.Run("Unattended_Spawn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			verd, eff, err := LowerInEnvelope(spawn, verbs, in, earnedEnv, ModeEnforce)
			if err != nil {
				b.Fatal(err)
			}
			benchVerdictSink = verd
			benchEffectSink = eff
		}
	})

	b.Run("WarnSoak_Spawn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			verd, eff, err := LowerInEnvelope(spawn, verbs, in, subEnv, ModeWarn)
			if err != nil {
				b.Fatal(err)
			}
			benchVerdictSink = verd
			benchEffectSink = eff
		}
	})

	b.Run("ConfirmRefusal_Spawn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			verd, eff, err := LowerInEnvelope(spawn, verbs, in, subEnv, ModeEnforce)
			if err == nil {
				b.Fatal("expected ErrConfirmRequired")
			}
			benchVerdictSink = verd
			benchEffectSink = eff
		}
	})

	b.Run("Narrowing_Hold", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			verd, eff, err := LowerInEnvelope(hold, verbs, in, Envelope{}, ModeEnforce)
			if err != nil {
				b.Fatal(err)
			}
			benchVerdictSink = verd
			benchEffectSink = eff
		}
	})
}

func BenchmarkKeepBitVerdict(b *testing.B) {
	baseline := KeepSample{Measured: true, Value: 3.0}

	makeSoak := func(n int) []KeepSample {
		out := make([]KeepSample, n)
		for i := 0; i < n; i++ {
			out[i] = KeepSample{
				Measured: i%5 != 4,
				Value:    2.8 - float64(i)*0.01,
			}
		}
		return out
	}

	soak10 := makeSoak(10)
	soak100 := makeSoak(100)

	b.Run("Checkpoints_10", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchKeepSink = KeepBitVerdict(baseline, soak10)
		}
	})

	b.Run("Checkpoints_100", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchKeepSink = KeepBitVerdict(baseline, soak100)
		}
	})
}
