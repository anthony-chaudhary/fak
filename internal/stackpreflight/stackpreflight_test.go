package stackpreflight

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stackresolve"
	"github.com/anthony-chaudhary/fak/internal/supportgraph"
	"github.com/anthony-chaudhary/fak/internal/workloadfit"
)

func TestRequiredRecommendedAndFallback(t *testing.T) {
	at := time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC)
	allow := stackresolve.Receipt{Status: "allow"}
	fit := workloadfit.Assessment{Status: "fit"}
	good := supportgraph.Tuple{Artifact: "a", Hardware: "l4"}
	bad := supportgraph.Tuple{Artifact: "a", Hardware: "t4"}
	graph := supportgraph.Graph{Schema: supportgraph.Schema, Edges: []supportgraph.Edge{
		{Tuple: good, Required: []string{"sm>=80"}, Recommended: []string{"memory>=24GiB"}, Evidence: []supportgraph.Evidence{{ID: "w", State: supportgraph.Supported, Tier: supportgraph.Witnessed, Authority: "lab", Source: "w"}}},
		{Tuple: bad, Required: []string{"sm>=80"}, Evidence: []supportgraph.Evidence{{ID: "r", State: supportgraph.Unsupported, Tier: supportgraph.Witnessed, Authority: "lab", Source: "r", Fallback: "cpu", Penalty: "10x latency"}}},
	}}
	accepted := Run(Input{Stack: allow, Fitness: fit, Graph: graph, Tuple: good, AsOf: at, CapacityTarget: "24GiB"})
	if accepted.Status != "allow" || len(accepted.Required) != 1 || len(accepted.Warnings) != 1 {
		t.Fatalf("allow=%+v", accepted)
	}
	refused := Run(Input{Stack: allow, Fitness: fit, Graph: graph, Tuple: bad, AsOf: at, CapacityTarget: "24GiB"})
	if refused.Status != "refuse" || len(refused.Alternatives) != 1 || refused.Alternatives[0].Impact != "10x latency" {
		t.Fatalf("bad=%+v", refused)
	}
}

func TestMandatoryBlockersCannotBeBypassed(t *testing.T) {
	result := Run(Input{Stack: stackresolve.Receipt{Status: "refuse", Conflict: &stackresolve.Conflict{Code: "UNSAT", Wanted: "gpu"}}, Fitness: workloadfit.Assessment{Status: "fit"}})
	if result.Status != "refuse" || len(result.Blockers) != 1 {
		t.Fatal(result)
	}
}

func TestRecommendationNeverBlocks(t *testing.T) {
	graph := supportgraph.Graph{Schema: supportgraph.Schema, Edges: []supportgraph.Edge{{Tuple: supportgraph.Tuple{Artifact: "a"}, Recommended: []string{"sm>=90"}, Evidence: []supportgraph.Evidence{{ID: "w", State: supportgraph.Supported, Tier: supportgraph.Witnessed, Authority: "lab", Source: "w"}}}}}
	result := Run(Input{Stack: stackresolve.Receipt{Status: "allow"}, Fitness: workloadfit.Assessment{Status: "fit"}, Graph: graph, Tuple: graph.Edges[0].Tuple, AsOf: time.Now()})
	if result.Status != "allow" {
		t.Fatal(result)
	}
}

var benchRunResult Result

func BenchmarkRun(b *testing.B) {
	asOf := time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC)
	stack := stackresolve.Receipt{Status: "allow"}
	fitness := workloadfit.Assessment{Status: "fit"}
	tuple := supportgraph.Tuple{Artifact: "a", Hardware: "l4"}
	graph := supportgraph.Graph{
		Schema: supportgraph.Schema,
		Edges: []supportgraph.Edge{
			{
				Tuple:       tuple,
				Required:    []string{"sm>=80"},
				Recommended: []string{"memory>=24GiB"},
				Evidence: []supportgraph.Evidence{
					{
						ID:        "w",
						State:     supportgraph.Supported,
						Tier:      supportgraph.Witnessed,
						Authority: "lab",
						Source:    "w",
					},
				},
			},
		},
	}
	in := Input{
		Stack:          stack,
		Fitness:        fitness,
		Graph:          graph,
		Tuple:          tuple,
		AsOf:           asOf,
		CapacityTarget: "24GiB",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRunResult = Run(in)
	}
}
