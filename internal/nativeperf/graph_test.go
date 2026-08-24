package nativeperf

import (
	"strings"
	"testing"
)

func TestActiveGraphIsValidAndPinsAuthoritativeValues(t *testing.T) {
	graph := ActiveGraph()
	if err := Validate(graph); err != nil {
		t.Fatal(err)
	}
	if graph.Envelope.PromptTokens != 32 || graph.Envelope.DecodeTokens != 64 || len(graph.Rungs) != 3 {
		t.Fatalf("unexpected active graph envelope: %+v", graph)
	}
	if graph.Rungs[0].Witnessed == nil || graph.Rungs[0].Witnessed.TokensPerSecond != 3.3 {
		t.Fatalf("native baseline must retain the 3.3 tok/s witness: %+v", graph.Rungs[0])
	}
	if graph.Comparison.TokensPerSecond != 6.966061 || graph.Comparison.Classification != ClassComparison {
		t.Fatalf("comparison must retain the issue #8697 value: %+v", graph.Comparison)
	}
}

func TestValidateRejectsDuplicateUnknownAndCyclicDependencies(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Graph)
		want string
	}{
		{
			name: "duplicate",
			edit: func(g *Graph) { g.Rungs[1].ID = g.Rungs[0].ID },
			want: `duplicate rung id "resident-q4k-baseline"`,
		},
		{
			name: "unknown dependency",
			edit: func(g *Graph) { g.Rungs[1].DependencyIDs = []string{"not-committed"} },
			want: `depends on unknown rung "not-committed"`,
		},
		{
			name: "cycle",
			edit: func(g *Graph) { g.Rungs[0].DependencyIDs = []string{"matched-native-parity"} },
			want: "dependency cycle:",
		},
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

func TestValidateRejectsInvalidRangesAndEvidenceConflation(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Graph)
		want string
	}{
		{
			name: "reversed range",
			edit: func(g *Graph) {
				g.Rungs[1].Expected.FloorTokensPerSecond = 7
				g.Rungs[1].Expected.RoofTokensPerSecond = 5
			},
			want: "invalid expected range",
		},
		{
			name: "witness in expected",
			edit: func(g *Graph) { g.Rungs[0].Expected.Classification = ClassWitnessed },
			want: "measurements belong in witnessed",
		},
		{
			name: "projection in witnessed",
			edit: func(g *Graph) { g.Rungs[0].Witnessed.Classification = ClassHypothesis },
			want: "projections belong in expected",
		},
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
	first.Features[0].ID = "mutated"
	second := ActiveGraph()
	if len(second.Rungs[0].DependencyIDs) != 0 || second.Rungs[0].Witnessed.TokensPerSecond != 3.3 || second.Features[0].ID != "resident-q4k-weights" {
		t.Fatalf("ActiveGraph leaked caller mutation: %+v", second.Rungs[0])
	}
}

func TestValidateRejectsInvalidFeatureCatalog(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Graph)
		want string
	}{
		{"duplicate", func(g *Graph) { g.Features[1].ID = g.Features[0].ID }, `duplicate feature id "resident-q4k-weights"`},
		{"unknown owner", func(g *Graph) { g.Features[1].RungID = "missing" }, `owns unknown rung "missing"`},
		{"enabled absent", func(g *Graph) { g.Features[1].Enabled = true; g.Features[1].Status = StatusAbsent }, `is enabled but absent`},
		{"present disabled", func(g *Graph) { g.Features[0].Enabled = false }, `is present but disabled`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := ActiveGraph()
			tt.edit(&g)
			err := Validate(g)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want %q", err, tt.want)
			}
		})
	}
}
