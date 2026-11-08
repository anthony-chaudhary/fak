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
	if len(graph.Envelopes) != 3 || graph.Envelopes[0].Backend != "metal" || graph.Envelopes[1].Backend != "cuda" || graph.Envelopes[2].Backend != "cuda" {
		t.Fatalf("Metal, CUDA A100, and CUDA RTX 5090 must remain separate pinned envelopes: %+v", graph.Envelopes)
	}
	if graph.Envelopes[2].ID != "qwen38-27b-q4km-rtx5090-p1-decode" || graph.Envelopes[2].Hardware != "NVIDIA RTX 5090" || graph.Envelopes[2].MemoryGiB != 32 || graph.Envelopes[2].Quantization != "Q4_K_M" || graph.Envelopes[2].ModelRevision != "issue-8635-pinned-revision" || graph.Envelopes[2].ArtifactSHA256 != "issue-8635-pinned-artifact" {
		t.Fatalf("RTX 5090 envelope drifted from the issue #8635 pinned form: %+v", graph.Envelopes[2])
	}
	if graph.Rungs[0].Witnessed == nil || graph.Rungs[0].Witnessed.TokensPerSecond != 3.3 || graph.Comparison.TokensPerSecond != 6.966061 {
		t.Fatalf("legacy evidence changed: %+v", graph)
	}
	for _, lever := range graph.Levers {
		for _, envelope := range graph.Envelopes {
			if envelope.Backend != "cuda" {
				continue
			}
			if lever.Applicability.EnvelopeID == envelope.ID && lever.Witnessed != nil {
				t.Fatalf("CUDA planning target was conflated with a witness: %+v", lever)
			}
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

// TestRTX5090LeverChainResolvesInOrder climbs the committed RTX 5090 chain:
// each step ReadyLevers names exactly the next lever in order, and climbing
// toggles the lever ON with its conflicting scalar baseline OFF, mirroring the
// committed A100 arm shape.
func TestRTX5090LeverChainResolvesInOrder(t *testing.T) {
	const envID = "qwen38-27b-q4km-rtx5090-p1-decode"
	graph := ActiveGraph()
	byID := make(map[string]int, len(graph.Levers))
	for i := range graph.Levers {
		byID[graph.Levers[i].ID] = i
	}
	want := []string{"rtx5090.q8_1-activation-quant", "rtx5090.dp4a-q4k-mmvq", "rtx5090.default-decode-routing"}
	for _, id := range want {
		ready, err := ReadyLevers(graph)
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for _, lever := range ready {
			if lever.Applicability.EnvelopeID == envID {
				got = append(got, lever.ID)
			}
		}
		if len(got) != 1 || got[0] != id {
			t.Fatalf("ready RTX 5090 levers = %v, want [%s]", got, id)
		}
		i := byID[id]
		graph.Levers[i].Enabled = true
		graph.Levers[i].Status = StatusPresent
		// Retire the conflicting arm as captured: its NextWitness is exactly the
		// repeated same-artifact baseline receipt, so promoting the candidate
		// presupposes that evidence exists and the baseline lever closes out.
		for _, conflict := range graph.Levers[i].ConflictIDs {
			j := byID[conflict]
			graph.Levers[j].Enabled = false
			graph.Levers[j].Status = StatusAbsent
			graph.Levers[j].Witnessed = &WitnessedEffect{Summary: "repeated same-artifact baseline captured", Classification: ClassWitnessed, Provenance: "climb test", Receipt: "climb-test-receipt"}
		}
	}
	ready, err := ReadyLevers(graph)
	if err != nil {
		t.Fatal(err)
	}
	for _, lever := range ready {
		if lever.Applicability.EnvelopeID == envID {
			t.Fatalf("RTX 5090 chain exhausted but %q is still ready", lever.ID)
		}
	}
}

// TestRTX5090DependencyOnA100LeverIsCrossEnvelope pins envelope disjointness:
// an A100 lever can never satisfy an RTX 5090 lever's dependency.
func TestRTX5090DependencyOnA100LeverIsCrossEnvelope(t *testing.T) {
	graph := ActiveGraph()
	for i := range graph.Levers {
		if graph.Levers[i].ID == "rtx5090.q8_1-activation-quant" {
			graph.Levers[i].DependencyIDs = []string{"cuda.scalar-f32-activation-baseline"}
		}
	}
	err := Validate(graph)
	if err == nil || !strings.Contains(err.Error(), `cross-envelope edge "rtx5090.q8_1-activation-quant" -> "cuda.scalar-f32-activation-baseline"`) {
		t.Fatalf("Validate() error = %v, want cross-envelope edge refusal", err)
	}
}

func TestValidPlatformCoversCommittedBackendPairs(t *testing.T) {
	tests := []struct {
		platform string
		backend  string
		want     bool
	}{
		{"darwin/arm64", "metal", true},
		{"linux/amd64+nvidia-a100", "cuda", true},
		{"linux/amd64+nvidia-rtx5090", "cuda", true},
		{"linux/amd64+nvidia-rtx4090", "cuda", false},
		{"linux/amd64+nvidia-rtx5090", "metal", false},
		{"darwin/arm64", "cuda", false},
	}
	for _, tt := range tests {
		if got := validPlatform(tt.platform, tt.backend); got != tt.want {
			t.Errorf("validPlatform(%q, %q) = %v, want %v", tt.platform, tt.backend, got, tt.want)
		}
	}
}
