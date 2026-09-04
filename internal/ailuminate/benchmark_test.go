package ailuminate

import (
	"encoding/json"
	"testing"
)

var (
	benchContractSink Contract
	benchStringSink   string
	benchBytesSink    []byte
)

// BenchmarkBuildNoGoContract measures assembling the scoping contract under the default
// condition where fak has no completion-path content filter (StatusNoGoPrereqUnmet).
func BenchmarkBuildNoGoContract(b *testing.B) {
	in := ContractInput{
		GeneratedAt:                 "2026-09-04T00:00:00Z",
		FakCommit:                   "a1b2c3d4e5f6",
		CompletionPathContentFilter: false,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := Build(in)
		if c.Status != StatusNoGoPrereqUnmet {
			b.Fatalf("unexpected status: %s", c.Status)
		}
		benchContractSink = c
	}
}

// BenchmarkBuildGoContract measures assembling the scoping contract with full lineage
// and an active completion-path content filter (StatusReadyOperatorRun).
func BenchmarkBuildGoContract(b *testing.B) {
	in := ContractInput{
		GeneratedAt:                 "2026-09-04T00:00:00Z",
		FakCommit:                   "a1b2c3d4e5f6",
		CompletionPathContentFilter: true,
		ContentFilterEvidence: []string{
			"internal/gateway/http.go:handleChatCompletions — inspected",
			"internal/gateway/moderations.go — in-path classifier active",
		},
		FrontedModelID: "meta-llama/Llama-3.1-70B-Instruct",
		ModelProvider:  "vllm",
		ModelDate:      "2026-08-01",
		HarnessCommit:  "fedcba654321",
		RunDateTime:    "2026-09-04T12:00:00Z",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := Build(in)
		if c.Status != StatusReadyOperatorRun {
			b.Fatalf("unexpected status: %s", c.Status)
		}
		benchContractSink = c
	}
}

// BenchmarkRenderMarkdown measures formatting the full AILuminate scoping and go/no-go
// contract into committable Markdown tables and evidence blocks.
func BenchmarkRenderMarkdown(b *testing.B) {
	c := Build(ContractInput{
		GeneratedAt:                 "2026-09-04T00:00:00Z",
		FakCommit:                   "a1b2c3d4e5f6",
		CompletionPathContentFilter: false,
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		md := RenderMarkdown(c)
		if len(md) == 0 {
			b.Fatal("unexpected empty markdown render")
		}
		benchStringSink = md
	}
}

// BenchmarkHazardCategoriesMapping measures the taxonomy resolution across all 12 AILuminate
// hazard categories and classification of their movability.
func BenchmarkHazardCategoriesMapping(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cats := hazardCategories()
		if len(cats) != 12 {
			b.Fatalf("expected 12 categories, got %d", len(cats))
		}
	}
}

// BenchmarkContractJSONMarshal measures JSON serialization throughput of the Contract struct.
func BenchmarkContractJSONMarshal(b *testing.B) {
	c := Build(ContractInput{
		GeneratedAt:                 "2026-09-04T00:00:00Z",
		FakCommit:                   "a1b2c3d4e5f6",
		CompletionPathContentFilter: true,
		FrontedModelID:              "Qwen/Qwen2.5-72B-Instruct",
		ModelProvider:               "sglang",
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bytes, err := json.Marshal(c)
		if err != nil {
			b.Fatalf("json.Marshal failed: %v", err)
		}
		benchBytesSink = bytes
	}
}

// BenchmarkContractPipelineEndToEnd measures the complete lifecycle: input synthesis,
// contract evaluation, Markdown evidence generation, and JSON transport serialization.
func BenchmarkContractPipelineEndToEnd(b *testing.B) {
	in := ContractInput{
		GeneratedAt:                 "2026-09-04T00:00:00Z",
		FakCommit:                   "a1b2c3d4e5f6",
		CompletionPathContentFilter: false,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := Build(in)
		md := RenderMarkdown(c)
		data, err := json.Marshal(c)
		if err != nil {
			b.Fatalf("marshal failed: %v", err)
		}
		var decoded Contract
		if err := json.Unmarshal(data, &decoded); err != nil {
			b.Fatalf("unmarshal failed: %v", err)
		}
		benchContractSink = decoded
		benchStringSink = md
	}
}

// TestBenchmarkSanity ensures benchmark routines execute without panic and perform iterations.
func TestBenchmarkSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	res := testing.Benchmark(BenchmarkContractPipelineEndToEnd)
	if res.N <= 0 {
		t.Fatalf("expected positive iterations, got %d", res.N)
	}
}
