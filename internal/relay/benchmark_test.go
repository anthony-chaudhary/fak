package relay

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

var (
	benchBatonSink    Baton
	benchBytesSink    []byte
	benchOutcomeSink  LegOutcome
	benchFidelitySink Fidelity
	benchStaleSink    StaleOutcome
	benchGateSink     ExternalizeGate
	benchFactsSink    []LoadBearingFact
	benchActionSink   NextActionVerdict
	benchAxisSink     Axis
	benchStateSink    RotationState
	benchPointersSink []string
	benchBoolSink     bool
	benchGuardSink    GuardVerdict
	benchRematSink    RematerializeOutcome
)

const (
	benchSHA1 = "1111111111111111111111111111111111111111"
	benchSHA2 = "2222222222222222222222222222222222222222"
)

type benchResolver struct {
	verdicts map[string]ResolveVerdict
}

func (r benchResolver) Resolve(a Artifact) Resolution {
	if r.verdicts != nil {
		if v, ok := r.verdicts[a.Ref]; ok {
			return Resolution{Artifact: a, Verdict: v, Detail: "bench resolved"}
		}
	}
	return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: "bench unknown"}
}

type benchLedger struct {
	steps []ProgressStep
}

func (l benchLedger) ReadProgress(ledgerRef string) ([]ProgressStep, error) {
	return l.steps, nil
}

type benchWipRestorer struct {
	result WipApplyResult
}

func (r benchWipRestorer) Restore(objectID string) WipApplyResult {
	return r.result
}

func makeBenchBatons() (Baton, Baton) {
	minBaton := Baton{
		Schema:      Schema,
		RelayID:     "RLY-BENCH-MIN",
		Leg:         1,
		ParentTrace: "trace-min-0",
		Objective:   ctxplan.NewObjectivePin("pin-min", "Minimal relay objective", 1),
		DoneWhen:    "minimal objective verified",
		ProgressCursor: ProgressCursor{
			StartSHA: benchSHA1,
		},
		NextAction: "run next minimal step",
	}

	prodBaton := Baton{
		Schema:      Schema,
		RelayID:     "RLY-20260905-0042",
		Leg:         12,
		ParentTrace: "trace-leg-11-prod",
		Objective:   ctxplan.NewObjectivePin("pin-relay-prod", "Run perpetual session with robust baton rotation.", 4),
		DoneWhen:    "relay goal achieved with witnessed commits",
		ProgressCursor: ProgressCursor{
			StartSHA:   benchSHA1,
			LedgerRef:  ".dos/runs/relay-bench.jsonl",
			HeldRegion: []string{"internal/relay/**", "internal/session/**"},
			WipTree:    "refs/fak/wip/session-42",
		},
		NextAction: "execute next boundary in rotation loop",
		Artifacts: []Artifact{
			{Kind: string(ArtifactCommit), Ref: benchSHA1},
			{Kind: string(ArtifactIssue), Ref: "#1894"},
			{Kind: string(ArtifactMemory), Ref: "memory:relay-core-architecture"},
			{Kind: string(ArtifactLedger), Ref: "row-42"},
			{Kind: string(ArtifactFile), Ref: "internal/relay/driver.go"},
		},
		OpenQuestions: []string{
			"Does the successor inherit lease region automatically?",
			"Should rotation cap default to 20 per hour?",
		},
		DoNotRederive: []string{
			"memory:dead-end-unbuffered-driver",
			"commit:badsha00000000000000000000000000000000",
		},
		Tombstone: Tombstone{
			Reason: "RELAY_ROTATED",
			AtSHA:  benchSHA1,
			Note:   "normal soft-mark rotation at turn boundary",
		},
	}

	return minBaton, prodBaton
}

func BenchmarkBatonMarshal(b *testing.B) {
	minBaton, prodBaton := makeBenchBatons()

	b.Run("Minimal", func(b *testing.B) {
		raw, err := Marshal(minBaton)
		if err != nil {
			b.Fatalf("Marshal failed: %v", err)
		}
		b.SetBytes(int64(len(raw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := Marshal(minBaton)
			if err != nil {
				b.Fatal(err)
			}
			benchBytesSink = out
		}
	})

	b.Run("Production", func(b *testing.B) {
		raw, err := Marshal(prodBaton)
		if err != nil {
			b.Fatalf("Marshal failed: %v", err)
		}
		b.SetBytes(int64(len(raw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := Marshal(prodBaton)
			if err != nil {
				b.Fatal(err)
			}
			benchBytesSink = out
		}
	})
}

func BenchmarkBatonParse(b *testing.B) {
	minBaton, prodBaton := makeBenchBatons()
	minRaw, err := Marshal(minBaton)
	if err != nil {
		b.Fatalf("setup Marshal min failed: %v", err)
	}
	prodRaw, err := Marshal(prodBaton)
	if err != nil {
		b.Fatalf("setup Marshal prod failed: %v", err)
	}

	b.Run("Minimal", func(b *testing.B) {
		b.SetBytes(int64(len(minRaw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			parsed, err := Parse(minRaw)
			if err != nil {
				b.Fatal(err)
			}
			benchBatonSink = parsed
		}
	})

	b.Run("Production", func(b *testing.B) {
		b.SetBytes(int64(len(prodRaw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			parsed, err := Parse(prodRaw)
			if err != nil {
				b.Fatal(err)
			}
			benchBatonSink = parsed
		}
	})
}

func BenchmarkBatonRoundTrip(b *testing.B) {
	_, prodBaton := makeBenchBatons()
	raw, _ := Marshal(prodBaton)
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wire, err := Marshal(prodBaton)
		if err != nil {
			b.Fatal(err)
		}
		parsed, err := Parse(wire)
		if err != nil {
			b.Fatal(err)
		}
		benchBatonSink = parsed
	}
}

func BenchmarkDriveLeg(b *testing.B) {
	minBaton, _ := makeBenchBatons()
	resolver := benchResolver{
		verdicts: map[string]ResolveVerdict{
			benchSHA1: ResolveVerified,
			benchSHA2: ResolveVerified,
		},
	}
	ledger := benchLedger{
		steps: []ProgressStep{
			{Ref: benchSHA1, Note: "benchmark step 1"},
			{Ref: benchSHA2, Note: "benchmark step 2"},
		},
	}

	b.Run("FirstLeg", func(b *testing.B) {
		cfg := LegConfig{
			RelayID:       "RLY-BENCH-FIRST",
			Objective:     ctxplan.NewObjectivePin("pin-bench-1", "First leg benchmark", 1),
			DoneWhen:      "benchmark complete",
			HeldRegion:    []string{"internal/relay/**"},
			TraceID:       "trace-bench-0",
			Triggers:      ArmTriggers{SoftMark: 0.55},
			MaxBoundaries: 10,
			Work: func(_ Orientation, boundary int) (BoundaryObs, error) {
				if boundary == 0 {
					return BoundaryObs{
						Usage:        BudgetUsage{Context: AxisUsage{Used: 30, Cap: 100}},
						TurnInFlight: true,
						Tree:         TreeStatus{DirtyPaths: []string{"internal/relay/driver.go"}},
						NextSteps:    []string{"work in flight"},
						Facts:        []LoadBearingFact{{Label: "in flight fact"}},
						AtSHA:        benchSHA1,
					}, nil
				}
				return BoundaryObs{
					Usage:        BudgetUsage{Context: AxisUsage{Used: 65, Cap: 100}},
					TurnInFlight: false,
					Tree:         TreeStatus{},
					NextSteps:    []string{"rotate to successor"},
					Facts: []LoadBearingFact{
						{Label: "externalized fact", Backing: Artifact{Kind: string(ArtifactCommit), Ref: benchSHA1}},
					},
					AtSHA: benchSHA1,
				}, nil
			},
			WriteBaton: func(wire []byte) error {
				benchBytesSink = wire
				return nil
			},
			Recontinue: func(b Baton) (string, error) {
				benchBatonSink = b
				return "trace-bench-succ", nil
			},
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outcome, err := DriveLeg(cfg)
			if err != nil {
				b.Fatal(err)
			}
			benchOutcomeSink = outcome
		}
	})

	b.Run("SuccessorLeg", func(b *testing.B) {
		cfg := LegConfig{
			Incoming:      minBaton,
			TraceID:       "trace-bench-succ-leg",
			Triggers:      ArmTriggers{SoftMark: 0.55},
			MaxBoundaries: 10,
			Resolver:      resolver,
			Ledger:        ledger,
			Work: func(_ Orientation, boundary int) (BoundaryObs, error) {
				if boundary == 0 {
					return BoundaryObs{
						Usage:        BudgetUsage{Context: AxisUsage{Used: 40, Cap: 100}},
						TurnInFlight: true,
						Tree:         TreeStatus{DirtyPaths: []string{"internal/relay/driver.go"}},
						NextSteps:    []string{"stepping"},
						Facts:        []LoadBearingFact{{Label: "temp fact"}},
						AtSHA:        benchSHA1,
					}, nil
				}
				return BoundaryObs{
					Usage:        BudgetUsage{Context: AxisUsage{Used: 70, Cap: 100}},
					TurnInFlight: false,
					Tree:         TreeStatus{},
					NextSteps:    []string{"hand off to next leg"},
					Facts: []LoadBearingFact{
						{Label: "safe fact", Backing: Artifact{Kind: string(ArtifactCommit), Ref: benchSHA1}},
					},
					AtSHA: benchSHA1,
				}, nil
			},
			WriteBaton: func(wire []byte) error {
				benchBytesSink = wire
				return nil
			},
			Recontinue: func(b Baton) (string, error) {
				benchBatonSink = b
				return "trace-bench-next", nil
			},
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outcome, err := DriveLeg(cfg)
			if err != nil {
				b.Fatal(err)
			}
			benchOutcomeSink = outcome
		}
	})

	b.Run("DoneCheckShortCircuit", func(b *testing.B) {
		cfg := LegConfig{
			RelayID:       "RLY-BENCH-DONE",
			Objective:     ctxplan.NewObjectivePin("pin-bench-done", "Already done relay", 1),
			DoneWhen:      "already satisfied",
			TraceID:       "trace-bench-done",
			MaxBoundaries: 5,
			DoneCheck: func(doneWhen string) (bool, error) {
				return true, nil
			},
			Work: func(_ Orientation, _ int) (BoundaryObs, error) {
				return BoundaryObs{}, nil
			},
			WriteBaton: func(wire []byte) error {
				return nil
			},
			Recontinue: func(b Baton) (string, error) {
				return "", nil
			},
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			outcome, err := DriveLeg(cfg)
			if err != nil {
				b.Fatal(err)
			}
			benchOutcomeSink = outcome
		}
	})
}

func BenchmarkArmTriggers(b *testing.B) {
	triggers := ArmTriggers{
		SoftMark: 0.55,
	}

	below := BudgetUsage{
		Context: AxisUsage{Used: 30, Cap: 100},
		Turns:   AxisUsage{Used: 5, Cap: 20},
		Wall:    AxisUsage{Used: 20, Cap: 100},
		Spend:   AxisUsage{Used: 100, Cap: 1000},
	}
	contextCrossed := BudgetUsage{
		Context: AxisUsage{Used: 60, Cap: 100},
		Turns:   AxisUsage{Used: 5, Cap: 20},
		Wall:    AxisUsage{Used: 20, Cap: 100},
		Spend:   AxisUsage{Used: 100, Cap: 1000},
	}
	multipleCrossed := BudgetUsage{
		Context: AxisUsage{Used: 75, Cap: 100},
		Turns:   AxisUsage{Used: 18, Cap: 20},
		Wall:    AxisUsage{Used: 95, Cap: 100},
		Spend:   AxisUsage{Used: 900, Cap: 1000},
	}

	b.Run("BelowSoftMark", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = triggers.Crossed(below.Context, below.Turns, below.Wall, below.Spend)
		}
	})

	b.Run("ContextCrossed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			axis, crossed := triggers.Cross(contextCrossed.Context, contextCrossed.Turns, contextCrossed.Wall, contextCrossed.Spend)
			benchAxisSink = axis
			benchBoolSink = crossed
		}
	})

	b.Run("MultipleCrossedPriority", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			axis, crossed := triggers.Cross(multipleCrossed.Context, multipleCrossed.Turns, multipleCrossed.Wall, multipleCrossed.Spend)
			benchAxisSink = axis
			benchBoolSink = crossed
		}
	})
}

func BenchmarkArmFireStep(b *testing.B) {
	safe := SafePoint{
		NoInFlightTurn:        true,
		TreeGreenOrParked:     true,
		NextActionExpressible: true,
	}
	unsafe := SafePoint{
		NoInFlightTurn:        false,
		TreeGreenOrParked:     false,
		NextActionExpressible: false,
	}

	b.Run("DisarmedToArmed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var af ArmFire
			benchStateSink = af.Step(true, unsafe)
		}
	})

	b.Run("ArmedToFired", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var af ArmFire
			af.Step(true, unsafe)                // arms
			benchStateSink = af.Step(true, safe) // fires
		}
	})

	b.Run("FullCycle", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var af ArmFire
			benchStateSink = af.Step(false, unsafe)
			benchStateSink = af.Step(true, unsafe)
			benchStateSink = af.Step(true, safe)
			benchStateSink = af.Step(true, safe) // terminal
		}
	})
}

func BenchmarkSafePointEvaluation(b *testing.B) {
	sp := SafePoint{
		NoInFlightTurn:        true,
		TreeGreenOrParked:     true,
		NextActionExpressible: true,
	}
	treeClean := TreeStatus{}
	treeParked := TreeStatus{
		DirtyPaths:  []string{"a.go", "b.go"},
		ParkedPaths: []string{"a.go", "b.go"},
	}

	b.Run("IsSafe", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = sp.IsSafe()
		}
	})

	b.Run("GuardRotation", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchGuardSink = GuardRotation(sp, false)
		}
	})

	b.Run("TreeGateClean", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchGuardSink = TreeGate(sp, treeClean)
		}
	})

	b.Run("TreeGateParked", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchGuardSink = TreeGate(sp, treeParked)
		}
	})
}

func BenchmarkNextActionExtraction(b *testing.B) {
	cleanState := LegState{NextSteps: []string{"land witnessed commit and rotate"}}
	whitespaceState := LegState{NextSteps: []string{"  land witnessed commit  ", "land witnessed commit"}}
	ambiguousState := LegState{NextSteps: []string{"step one", "step two", "step three"}}

	b.Run("SingleClean", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchActionSink = ExtractNextAction(cleanState)
		}
	})

	b.Run("WhitespaceAndDedup", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchActionSink = ExtractNextAction(whitespaceState)
		}
	})

	b.Run("Ambiguous", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchActionSink = ExtractNextAction(ambiguousState)
		}
	})
}

func BenchmarkExternalizeGate(b *testing.B) {
	cleanFacts := []LoadBearingFact{
		{Label: "fact 1", Backing: Artifact{Kind: string(ArtifactCommit), Ref: benchSHA1}},
		{Label: "fact 2", Backing: Artifact{Kind: string(ArtifactIssue), Ref: "#123"}},
		{Label: "fact 3", Backing: Artifact{Kind: string(ArtifactMemory), Ref: "slug-abc"}},
		{Label: "fact 4", Backing: Artifact{Kind: string(ArtifactLedger), Ref: "row-7"}},
		{Label: "fact 5", Backing: Artifact{Kind: string(ArtifactFile), Ref: "pkg/file.go"}},
	}
	mixedFacts := []LoadBearingFact{
		{Label: "fact 1", Backing: Artifact{Kind: string(ArtifactCommit), Ref: benchSHA1}},
		{Label: "fact 2 (unbacked)", Backing: Artifact{Kind: "", Ref: ""}},
		{Label: "fact 3", Backing: Artifact{Kind: string(ArtifactIssue), Ref: "#456"}},
		{Label: "fact 4 (unbacked)", Backing: Artifact{Kind: "invalid", Ref: "x"}},
	}
	candidates := []Candidate{
		{LoadBearingFact: cleanFacts[0], Durability: ctxplan.DurabilityDurable},
		{LoadBearingFact: cleanFacts[1], Durability: ctxplan.DurabilitySession},
		{LoadBearingFact: cleanFacts[2], Durability: ctxplan.DurabilityTurn},
		{LoadBearingFact: cleanFacts[3], Durability: ""},
		{LoadBearingFact: cleanFacts[4], Durability: ctxplan.DurabilityBounded},
	}

	b.Run("CheckGateClean", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchGateSink = CheckExternalizeGate(cleanFacts)
		}
	})

	b.Run("CheckGateRefuse", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchGateSink = CheckExternalizeGate(mixedFacts)
		}
	})

	b.Run("LoadBearingFilter", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFactsSink = LoadBearing(candidates)
		}
	})
}

func BenchmarkFidelityScoring(b *testing.B) {
	makeFidelityBaton := func(n int) (Baton, benchResolver) {
		artifacts := make([]Artifact, n)
		verdicts := make(map[string]ResolveVerdict, n)
		for i := 0; i < n; i++ {
			ref := fmt.Sprintf("ref-%04d", i)
			artifacts[i] = Artifact{Kind: string(ArtifactCommit), Ref: ref}
			switch i % 3 {
			case 0:
				verdicts[ref] = ResolveVerified
			case 1:
				verdicts[ref] = ResolveDangling
			default:
				verdicts[ref] = ResolveUnknown
			}
		}
		return Baton{Artifacts: artifacts}, benchResolver{verdicts: verdicts}
	}

	for _, n := range []int{10, 50, 100} {
		b.Run(fmt.Sprintf("%d_Artifacts", n), func(b *testing.B) {
			baton, resolver := makeFidelityBaton(n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchFidelitySink = ScoreBatonFidelity(baton, resolver)
			}
		})
	}
}

func BenchmarkDoNotRederiveIndex(b *testing.B) {
	pointers := []string{
		"memory:dead-end-1",
		"commit:badsha1",
		"memory:dead-end-2",
		"issue:#999",
		"commit:badsha2",
		"memory:dead-end-1", // duplicate
		"commit:badsha1",    // duplicate
		"file:stale/path.go",
	}

	b.Run("BuildAndExport", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := NewDoNotRederiveIndex(pointers)
			benchPointersSink = idx.Pointers()
		}
	})

	b.Run("DeduplicateAdd", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var idx DoNotRederiveIndex
			for _, p := range pointers {
				idx.Add(p)
			}
			benchPointersSink = idx.Pointers()
		}
	})
}

func BenchmarkCheckBatonStale(b *testing.B) {
	resolver := benchResolver{
		verdicts: map[string]ResolveVerdict{
			benchSHA1: ResolveVerified,
			benchSHA2: ResolveDangling,
		},
	}
	freshBaton := Baton{
		ProgressCursor: ProgressCursor{StartSHA: benchSHA1},
		Tombstone:      Tombstone{AtSHA: benchSHA1, Reason: "RELAY_ROTATED"},
	}
	staleBaton := Baton{
		ProgressCursor: ProgressCursor{StartSHA: benchSHA2},
		Tombstone:      Tombstone{AtSHA: benchSHA2, Reason: "RELAY_ROTATED"},
	}

	b.Run("Fresh", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStaleSink = CheckBatonStale(freshBaton, resolver)
		}
	})

	b.Run("Stale", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStaleSink = CheckBatonStale(staleBaton, resolver)
		}
	})
}

func BenchmarkRotationCapAdmit(b *testing.B) {
	b.Run("Allowed", func(b *testing.B) {
		cap := RotationCap{MaxPerHour: 100}
		baseTime := int64(1700000000)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cap.accepted = cap.accepted[:0]
			allowed, _ := cap.Admit(baseTime)
			benchBoolSink = allowed
		}
	})

	b.Run("Capped", func(b *testing.B) {
		cap := RotationCap{MaxPerHour: 5}
		baseTime := int64(1700000000)
		// fill cap
		for i := 0; i < 5; i++ {
			cap.Admit(baseTime + int64(i))
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			allowed, _ := cap.Admit(baseTime + 10)
			benchBoolSink = allowed
		}
	})
}

func BenchmarkArmHysteresisMayArm(b *testing.B) {
	var h ArmHysteresis
	h.MinSteps = 3

	prog1 := VerifiedProgress{Verdict: ProgressVerified, Steps: []ProgressStep{{Ref: "1"}, {Ref: "2"}, {Ref: "3"}}}
	progStalled := VerifiedProgress{Verdict: ProgressVerified, Steps: []ProgressStep{{Ref: "1"}, {Ref: "2"}, {Ref: "3"}}}
	progAdvanced := VerifiedProgress{Verdict: ProgressVerified, Steps: []ProgressStep{{Ref: "1"}, {Ref: "2"}, {Ref: "3"}, {Ref: "4"}, {Ref: "5"}, {Ref: "6"}}}

	b.Run("InitialArm", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var freshH ArmHysteresis
			freshH.MinSteps = 3
			benchBoolSink = freshH.MayArm(prog1)
		}
	})

	b.Run("ReArmRefused", func(b *testing.B) {
		hCopy := h
		hCopy.NoteArmed(prog1)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = hCopy.MayArm(progStalled)
		}
	})

	b.Run("ReArmVerified", func(b *testing.B) {
		hCopy := h
		hCopy.NoteArmed(prog1)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = hCopy.MayArm(progAdvanced)
		}
	})
}

func BenchmarkProjectShadowBaton(b *testing.B) {
	input := ShadowBatonInput{
		RelayID:     "RLY-SHADOW-01",
		Leg:         5,
		ParentTrace: "trace-4",
		Objective:   ctxplan.NewObjectivePin("pin-shadow", "Shadow projection benchmark", 2),
		DoneWhen:    "done when verified",
		NextAction:  "prepare shadow baton",
		StartSHA:    benchSHA1,
		LedgerRef:   "ledger-ref-1",
		HeldRegion:  []string{"internal/relay/**"},
		Artifacts: []Artifact{
			{Kind: string(ArtifactCommit), Ref: benchSHA1},
			{Kind: string(ArtifactIssue), Ref: "#123"},
		},
		DoNotRederive: []string{"dead-end-1"},
		OpenQuestions: []string{"question-1"},
		ExitReason:    "RELAY_ARMED",
		AtSHA:         benchSHA1,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBatonSink = ProjectShadowBaton(input)
	}
}

func BenchmarkRematerializeWip(b *testing.B) {
	restorer := benchWipRestorer{
		result: WipApplyResult{Verdict: WipApplied, Detail: "clean applied"},
	}
	cursor := ProgressCursor{
		StartSHA: benchSHA1,
		WipTree:  "refs/fak/wip/session-bench",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRematSink = RematerializeWip(cursor, restorer)
	}
}
