package agentic

import (
	"testing"
)

var (
	benchSinkPlan      Plan
	benchSinkInference Inference
	benchSinkBounds    Bounds
	benchSinkUnits     []WorkUnit
	benchSinkCohorts   []Cohort
	benchSinkDirs      []directionSpec
	benchSinkPaths     []string
	benchSinkBytes     []byte
	benchSinkString    string
)

const (
	benchObjectiveDev        = "Add a CLI and tests for internal/widget/widget.go and cmd/fak/widget.go."
	benchObjectiveNative     = "Optimize Qwen3.8 CUDA decode throughput in internal/engine and benchmark the result"
	benchObjectiveMixed      = "Build 100x better agentic performance processes across development and native model running: smaller issue-sized work, concurrent experiments, expand-contract cycles, outcome learning, OSS research, trigger graphs, and typed effects."
	benchObjectiveNounFilter = "Improve docs for Qwen3.8 performance settings and update GGUF configuration without execution"
)

// TestBenchmarkAgenticSanity verifies all benchmark fixtures and operational paths
// run and produce valid plans before benchmarking.
func TestBenchmarkAgenticSanity(t *testing.T) {
	for _, obj := range []string{benchObjectiveDev, benchObjectiveNative, benchObjectiveMixed, benchObjectiveNounFilter} {
		plan, err := Compile(obj)
		if err != nil {
			t.Fatalf("sanity Compile(%q) failed: %v", obj, err)
		}
		if plan.ObjectiveID == "" || len(plan.WorkUnits) == 0 {
			t.Fatalf("sanity plan incomplete for %q: %+v", obj, plan)
		}
		rendered := Render(plan)
		if len(rendered) == 0 {
			t.Fatalf("sanity Render produced empty string for %q", obj)
		}
		marshaled, err := Marshal(plan)
		if err != nil || len(marshaled) == 0 {
			t.Fatalf("sanity Marshal failed for %q: %v", obj, err)
		}
	}
}

// BenchmarkAgenticPatternEvaluation measures raw intent pattern extraction,
// clause parsing, path isolation, and domain discriminator filtering.
func BenchmarkAgenticPatternEvaluation(b *testing.B) {
	b.Run("DevelopmentPattern", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkInference = infer(benchObjectiveDev)
		}
	})

	b.Run("NativeModelPattern", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkInference = infer(benchObjectiveNative)
		}
	})

	b.Run("MixedPattern", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkInference = infer(benchObjectiveMixed)
		}
	})

	b.Run("TechNounFiltering", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkInference = infer(benchObjectiveNounFilter)
		}
	})

	b.Run("PathExtraction", func(b *testing.B) {
		const complexPathsObj = "Refactor cmd/fak/main.go, internal/agentic/agentic.go, internal/ctxmmu/mmu.go, docs/spec.md, and examples/demo.json"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkPaths = extractPaths(complexPathsObj)
		}
	})
}

// BenchmarkTransitionScoring measures uncertainty, scope, and depth scoring,
// bounds inference, candidate direction scoring, and cohort partitioning.
func BenchmarkTransitionScoring(b *testing.B) {
	devInference := infer(benchObjectiveDev)
	mixedInference := infer(benchObjectiveMixed)

	b.Run("InferenceScoring", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkInference = infer(benchObjectiveMixed)
		}
	})

	b.Run("BoundsCalculation", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBounds = inferBounds(mixedInference)
		}
	})

	b.Run("CandidateDirectionScoringDev", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDirs = candidateDirections(devInference)
		}
	})

	b.Run("CandidateDirectionScoringMixed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDirs = candidateDirections(mixedInference)
		}
	})

	b.Run("CohortGeneration", func(b *testing.B) {
		bounds := inferBounds(mixedInference)
		units := buildWorkUnits(benchObjectiveMixed, mixedInference, bounds)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkCohorts = buildCohorts(mixedInference, units)
		}
	})
}

// BenchmarkActionArbitration measures work-unit graph assembly, dependency
// wiring from candidate exploration to experiment and contract stages,
// full compilation arbitration, and deterministic plan rendering.
func BenchmarkActionArbitration(b *testing.B) {
	mixedInference := infer(benchObjectiveMixed)
	mixedBounds := inferBounds(mixedInference)
	mixedPlan, _ := Compile(benchObjectiveMixed)

	b.Run("WorkUnitArbitration", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkUnits = buildWorkUnits(benchObjectiveMixed, mixedInference, mixedBounds)
		}
	})

	b.Run("CompileFocused", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkPlan, _ = Compile(benchObjectiveDev)
		}
	})

	b.Run("CompileMixedSystem", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkPlan, _ = Compile(benchObjectiveMixed)
		}
	})

	b.Run("RenderPlan", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = Render(mixedPlan)
		}
	})

	b.Run("MarshalPlan", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBytes, _ = Marshal(mixedPlan)
		}
	})
}
