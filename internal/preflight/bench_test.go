package preflight

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/testroute"
)

// BenchmarkPreflightCheck measures throughput and memory allocations when evaluating
// tool call arguments through the preflight adjudication rung ladder.
func BenchmarkPreflightCheck(b *testing.B) {
	b.ReportAllocs()
	l := New()
	l.SetSchema("bench_tool", Schema{
		Required: map[string]FieldType{
			"action":  TypeString,
			"retries": TypeNumber,
			"verbose": TypeBool,
		},
	})
	ctx := context.Background()
	call := inlineCall("bench_tool", `{"action":"fetch","retries":3,"verbose":false}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := l.Adjudicate(ctx, call)
		if v.Kind != abi.VerdictDefer {
			b.Fatalf("expected VerdictDefer, got %v", v.Kind)
		}
	}
}

// BenchmarkWorkspaceInspect measures planning performance when inspecting workspace
// read/write sets, package dependency graphs, and test route resolution.
func BenchmarkWorkspaceInspect(b *testing.B) {
	b.ReportAllocs()
	graph := PackageGraph{
		TotalPackages: 4,
		FileToPackage: map[string]string{
			"internal/preflight/preflight.go": "github.com/anthony-chaudhary/fak/internal/preflight",
			"internal/preflight/workspace.go": "github.com/anthony-chaudhary/fak/internal/preflight",
			"cmd/fak/main.go":                 "github.com/anthony-chaudhary/fak/cmd/fak",
			"internal/engine/engine.go":       "github.com/anthony-chaudhary/fak/internal/engine",
		},
		Edges: map[string][]string{
			"github.com/anthony-chaudhary/fak/cmd/fak":         {"github.com/anthony-chaudhary/fak/internal/preflight"},
			"github.com/anthony-chaudhary/fak/internal/engine": {"github.com/anthony-chaudhary/fak/internal/preflight"},
		},
	}
	input := WorkspacePreflightInput{
		TaskID:       "task-bench-101",
		Actor:        "bench-agent",
		ReadGlobs:    []string{"docs/**", "internal/preflight/*.go"},
		WriteGlobs:   []string{"internal/preflight/**"},
		PackageGraph: graph,
		LiveLeases: []LeaseObservation{
			{ID: "lease-peer-other", Holder: "worker-peer", Tree: []string{"cmd/fak/**"}},
		},
		TestProbe: testroute.Probe{
			GOOS:              "windows",
			NativeTestAllowed: true,
			WSLPresent:        false,
		},
		TestArgs:        []string{"-short"},
		VerifyWarmBuild: true,
		WarmThresholdMS: 2000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan := PlanWorkspacePreflight(input)
		if plan.Verdict != WorkspaceVerdictReady {
			b.Fatalf("expected WorkspaceVerdictReady, got %s", plan.Verdict)
		}
	}
}
