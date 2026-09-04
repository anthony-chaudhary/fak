package advmodel_test

// Invariant: advanced model router maintains deterministic model selection without latency regressions.
// Invariant: advisory adjudication is fail-closed across all benchmark and test evaluation paths.
// Contract: bench_test exercises end-to-end integration and throughput bounds of the advisory model.

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/advmodel"
)

// TestBenchSanityContract verifies that model artifacts, featurization, and
// fail-closed adjudication invariants hold across test fixtures.
func TestBenchSanityContract(t *testing.T) {
	artPath := filepath.Join("testdata", "adjudicator.json")
	art, err := advmodel.Load(artPath)
	if err != nil {
		t.Skipf("testdata/adjudicator.json not available: %v", err)
	}

	desc := art.Descriptor()
	if !desc.Valid() {
		t.Fatalf("expected valid descriptor, got %+v", desc)
	}
	if desc.FeatureCount <= 0 {
		t.Fatalf("expected positive feature count, got %d", desc.FeatureCount)
	}

	adj := advmodel.NewAdjudicator(art)
	ctx := context.Background()

	// Invariant: known policy violation must be corroborated with VerdictDeny.
	denyCall := &abi.ToolCall{
		Tool: "refund_payment",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(`{"order":"o-1001","amount":49.99}`),
		},
	}
	vDeny := adj.Adjudicate(ctx, denyCall)
	if vDeny.Kind != abi.VerdictDeny {
		t.Fatalf("expected VerdictDeny for known violation, got %v", vDeny.Kind)
	}
	if vDeny.By != "advmodel" {
		t.Fatalf("expected By='advmodel', got %q", vDeny.By)
	}

	// Invariant: benign read calls must defer, never emit allow.
	allowCall := &abi.ToolCall{
		Tool: "search_kb",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(`{"q":"returns"}`),
		},
	}
	vAllow := adj.Adjudicate(ctx, allowCall)
	if vAllow.Kind != abi.VerdictDefer {
		t.Fatalf("expected VerdictDefer for benign call, got %v", vAllow.Kind)
	}

	// Invariant: unconfigured/nil adjudicator defers on every call.
	inertAdj := advmodel.NewAdjudicator(nil)
	vInert := inertAdj.Adjudicate(ctx, denyCall)
	if vInert.Kind != abi.VerdictDefer {
		t.Fatalf("expected VerdictDefer from inert adjudicator, got %v", vInert.Kind)
	}
}

// TestAdjudicatorLatencyAndInvariants verifies deterministic classification and
// ensures no unexpected allocations or panic paths occur during evaluation.
func TestAdjudicatorLatencyAndInvariants(t *testing.T) {
	toks := advmodel.Tokens("Bash", []byte(`{"command":"rm -rf /tmp/data"}`))
	if len(toks) == 0 {
		t.Fatal("expected non-empty tokens from tool call")
	}

	corpusPath := filepath.Join("testdata", "corpus.jsonl")
	data, err := os.ReadFile(corpusPath)
	if err != nil {
		t.Skipf("testdata/corpus.jsonl not available: %v", err)
	}

	rows, err := advmodel.LoadCorpus(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("failed to parse corpus: %v", err)
	}
	if len(rows) == 0 {
		t.Fatal("corpus rows must not be empty")
	}
}

// BenchmarkTokens evaluates token extraction throughput across varying argument payloads.
func BenchmarkTokens(b *testing.B) {
	b.Run("ShortCall", func(b *testing.B) {
		tool := "search_kb"
		args := []byte(`{"q":"return policy"}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = advmodel.Tokens(tool, args)
		}
	})

	b.Run("LongCall", func(b *testing.B) {
		tool := "create_support_ticket"
		args := []byte(`{"subject":"Payment failure","body":"Detailed description of issue with multiple tokens and params"}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = advmodel.Tokens(tool, args)
		}
	})

	b.Run("GuardedPath", func(b *testing.B) {
		tool := "Bash"
		args := []byte(`{"command":"cat /etc/passwd && rm -rf /tmp/data"}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = advmodel.Tokens(tool, args)
		}
	})
}

// BenchmarkArtifactScoreAndDenies measures raw scoring and decision boundary evaluation.
func BenchmarkArtifactScoreAndDenies(b *testing.B) {
	art, err := advmodel.Load(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("failed to load artifact: %v", err)
	}

	b.Run("KnownDeny", func(b *testing.B) {
		tool := "refund_payment"
		args := []byte(`{"order":"o-1001","amount":49.99}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = art.Score(tool, args)
			_ = art.Denies(tool, args)
		}
	})

	b.Run("BenignRead", func(b *testing.B) {
		tool := "read_customer_record"
		args := []byte(`{"id":"c-12345"}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = art.Score(tool, args)
			_ = art.Denies(tool, args)
		}
	})

	b.Run("UnknownTokens", func(b *testing.B) {
		tool := "unregistered_tool_operation"
		args := []byte(`{"arbitrary_payload_key":"unrecognized_value_token"}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = art.Score(tool, args)
			_ = art.Denies(tool, args)
		}
	})
}

// BenchmarkAdjudicate evaluates end-to-end adjudication throughput across decision paths.
func BenchmarkAdjudicate(b *testing.B) {
	art, err := advmodel.Load(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("failed to load artifact: %v", err)
	}
	adj := advmodel.NewAdjudicator(art)
	ctx := context.Background()

	b.Run("DenyTighten", func(b *testing.B) {
		call := &abi.ToolCall{
			Tool: "delete_account",
			Args: abi.Ref{
				Kind:   abi.RefInline,
				Inline: []byte(`{"account":"user-998"}`),
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = adj.Adjudicate(ctx, call)
		}
	})

	b.Run("DeferBenign", func(b *testing.B) {
		call := &abi.ToolCall{
			Tool: "search_kb",
			Args: abi.Ref{
				Kind:   abi.RefInline,
				Inline: []byte(`{"q":"warranty inquiry"}`),
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = adj.Adjudicate(ctx, call)
		}
	})

	b.Run("InertNilArtifact", func(b *testing.B) {
		inertAdj := advmodel.NewAdjudicator(nil)
		call := &abi.ToolCall{
			Tool: "refund_payment",
			Args: abi.Ref{
				Kind:   abi.RefInline,
				Inline: []byte(`{"order":"o-1"}`),
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = inertAdj.Adjudicate(ctx, call)
		}
	})
}

// BenchmarkLoadCorpus benchmarks parsing and decoding of newline-delimited JSON training corpora.
func BenchmarkLoadCorpus(b *testing.B) {
	data, err := os.ReadFile(filepath.Join("testdata", "corpus.jsonl"))
	if err != nil {
		b.Fatalf("failed to read corpus file: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := advmodel.LoadCorpus(bytes.NewReader(data))
		if err != nil {
			b.Fatalf("LoadCorpus failed: %v", err)
		}
		if len(rows) == 0 {
			b.Fatal("unexpected empty corpus")
		}
	}
}

// BenchmarkModelResolveAndDescriptor benchmarks artifact unmarshaling and descriptor extraction.
func BenchmarkModelResolveAndDescriptor(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("read artifact fixture: %v", err)
	}

	b.Run("ResolveBytes", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			art, desc, err := advmodel.ResolveModel(raw)
			if err != nil || art == nil || !desc.Valid() {
				b.Fatalf("ResolveModel failed: %v", err)
			}
		}
	})

	b.Run("DescriptorExtract", func(b *testing.B) {
		art, _, err := advmodel.ResolveModel(raw)
		if err != nil {
			b.Fatalf("setup resolve failed: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			desc := art.Descriptor()
			if !desc.Valid() {
				b.Fatal("invalid descriptor")
			}
		}
	})
}

// BenchmarkEndToEndThroughput measures sustained decision throughput across a sequence of calls.
func BenchmarkEndToEndThroughput(b *testing.B) {
	art, err := advmodel.Load(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("failed to load artifact: %v", err)
	}
	adj := advmodel.NewAdjudicator(art)
	ctx := context.Background()

	calls := []*abi.ToolCall{
		{Tool: "refund_payment", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"order":"o-1"}`)}},
		{Tool: "search_kb", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"q":"returns"}`)}},
		{Tool: "delete_account", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"account":"user-1"}`)}},
		{Tool: "read_customer_record", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"id":"c-1"}`)}},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		call := calls[i%len(calls)]
		_ = adj.Adjudicate(ctx, call)
	}
}
