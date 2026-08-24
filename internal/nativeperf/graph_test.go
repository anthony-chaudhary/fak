package nativeperf

import (
	"strings"
	"testing"
)

func TestActiveGraphIsValidAndPinsSeparateEnvelopes(t *testing.T) {
	graph := ActiveGraph()
	if err := Validate(graph); err != nil {
		t.Fatal(err)
	}
	if graph.Envelope.PromptTokens != 32 || graph.Envelope.DecodeTokens != 64 || len(graph.Rungs) != 3 {
		t.Fatalf("unexpected legacy graph view: %+v", graph)
	}
	if len(graph.Envelopes) != 2 || graph.Envelopes[0].Backend != "metal" || graph.Envelopes[1].Backend != "cuda" {
		t.Fatalf("Metal and CUDA must remain separate pinned envelopes: %+v", graph.Envelopes)
	}
	if graph.Rungs[0].Witnessed == nil || graph.Rungs[0].Witnessed.TokensPerSecond != 3.3 || graph.Comparison.TokensPerSecond != 6.966061 {
		t.Fatalf("legacy evidence changed: %+v", graph)
	}
	for _, lever := range graph.Levers {
		if lever.Applicability.EnvelopeID == graph.Envelopes[1].ID && lever.Witnessed != nil {
			t.Fatalf("CUDA planning target was conflated with a witness: %+v", lever)
		}
	}
}

func TestNextLeverIsDeterministicAndDependencyReady(t *testing.T) {
	graph := ActiveGraph()
	first, err := NextLever(graph)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NextLever(graph)
	if err != nil {
		t.Fatal(err)
	}
	if first == nil || second == nil || first.ID != "metal.command-buffer-amortization" || first.ID != second.ID {
		t.Fatalf("unexpected next lever: first=%+v second=%+v", first, second)
	}
	if first.OwningIssue.Number != 8324 || !strings.Contains(first.NextWitness, "OFF/ON") {
		t.Fatalf("next action lacks owner or exact witness: %+v", first)
	}
}

func TestDOTHasClustersDependencyAndConflictEdges(t *testing.T) {
	dot, err := DOT(ActiveGraph())
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"cluster_qwen38_27b_q4km_m3pro_p32_t64",
		"cluster_qwen38_27b_q4k_a100_p1_decode",
		`"metal.resident-q4k-weights" -> "metal.command-buffer-amortization" [label="depends"]`,
		`label="conflicts"`,
	} {
		if !strings.Contains(dot, want) {
			t.Fatalf("DOT missing %q:\n%s", want, dot)
		}
	}
}

func TestValidateRejectsUnsafeLeverGraphs(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Graph)
		want string
	}{
		{"duplicate lever", func(g *Graph) { g.Levers[1].ID = g.Levers[0].ID }, `duplicate lever id "metal.resident-q4k-weights"`},
		{"unknown reference", func(g *Graph) { g.Levers[1].DependencyIDs = []string{"missing"} }, `references unknown lever "missing"`},
		{"cycle", func(g *Graph) { g.Levers[0].DependencyIDs = []string{"metal.command-buffer-amortization"} }, "lever dependency cycle:"},
		{"cross envelope", func(g *Graph) { g.Levers[1].DependencyIDs = []string{"cuda.scalar-f32-activation-baseline"} }, "cross-envelope edge"},
		{"invalid applicability", func(g *Graph) { g.Levers[1].Applicability.Platform = "linux/amd64+nvidia-a100" }, "invalid applicability"},
		{"asymmetric conflict", func(g *Graph) { g.Levers[8].ConflictIDs = nil }, "is not symmetric"},
		{"enabled conflict", func(g *Graph) { g.Levers[9].Enabled = true; g.Levers[9].Status = StatusPartial }, "are both enabled"},
		{"expected witness conflation", func(g *Graph) { g.Levers[1].Expected.Classification = ClassWitnessed }, "witnessed evidence belongs in witnessed"},
		{"witness projection conflation", func(g *Graph) { g.Levers[0].Witnessed.Classification = ClassHypothesis }, "projections belong in expected"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := ActiveGraph()
			tt.edit(&graph)
			err := Validate(graph)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestValidateRetainsLegacyFailures(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Graph)
		want string
	}{
		{"duplicate rung", func(g *Graph) { g.Rungs[1].ID = g.Rungs[0].ID }, `duplicate rung id "resident-q4k-baseline"`},
		{"unknown rung", func(g *Graph) { g.Rungs[1].DependencyIDs = []string{"missing"} }, `depends on unknown rung "missing"`},
		{"rung cycle", func(g *Graph) { g.Rungs[0].DependencyIDs = []string{"matched-native-parity"} }, "dependency cycle:"},
		{"range", func(g *Graph) { g.Rungs[1].Expected.FloorTokensPerSecond = 8 }, "invalid expected range"},
		{"duplicate feature", func(g *Graph) { g.Features[1].ID = g.Features[0].ID }, "duplicate feature id"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := ActiveGraph()
			tt.edit(&graph)
			err := Validate(graph)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestActiveGraphReturnsIndependentValues(t *testing.T) {
	first := ActiveGraph()
	first.Rungs[0].DependencyIDs = append(first.Rungs[0].DependencyIDs, "mutated")
	first.Rungs[0].Witnessed.TokensPerSecond = 99
	first.Levers[1].DependencyIDs[0] = "mutated"
	second := ActiveGraph()
	if len(second.Rungs[0].DependencyIDs) != 0 || second.Rungs[0].Witnessed.TokensPerSecond != 3.3 || second.Levers[1].DependencyIDs[0] != "metal.resident-q4k-weights" {
		t.Fatalf("ActiveGraph leaked caller mutation: %+v", second)
	}
}
