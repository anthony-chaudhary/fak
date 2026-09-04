package advmodel

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// BenchmarkModelResolve benchmarks parsing serialized model artifacts and
// resolving their operational descriptors.
func BenchmarkModelResolve(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("read artifact fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		art, desc, err := ResolveModel(raw)
		if err != nil {
			b.Fatalf("resolve model: %v", err)
		}
		if !desc.Valid() {
			b.Fatal("invalid resolved descriptor")
		}
		if desc.FeatureCount != len(art.Features) {
			b.Fatalf("feature count mismatch: got %d want %d", desc.FeatureCount, len(art.Features))
		}
	}
}

// BenchmarkModelDescriptor benchmarks extracting operational descriptors from
// an in-memory trained model artifact.
func BenchmarkModelDescriptor(b *testing.B) {
	art, _, err := ResolvePath(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("resolve fixture: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		desc := art.Descriptor()
		if !desc.Valid() {
			b.Fatal("invalid descriptor")
		}
	}
}

// BenchmarkModelTokens benchmarks featurization of incoming tool calls into
// unique token representations.
func BenchmarkModelTokens(b *testing.B) {
	const tool = "refund_payment"
	args := []byte(`{"order":"o-1001","amount":49.99,"reason":"duplicate_charge"}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toks := Tokens(tool, args)
		if len(toks) == 0 {
			b.Fatal("unexpected empty tokens")
		}
	}
}

// BenchmarkModelScore benchmarks raw logit evaluation against trained feature weights.
func BenchmarkModelScore(b *testing.B) {
	art, _, err := ResolvePath(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("resolve fixture: %v", err)
	}
	const tool = "refund_payment"
	args := []byte(`{"order":"o-1001","amount":49.99}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		score := art.Score(tool, args)
		if score == 0 {
			b.Fatal("unexpected zero score for known deny tool")
		}
	}
}

// BenchmarkModelAdjudicate benchmarks end-to-end fail-closed adjudication throughput
// using the advisory model.
func BenchmarkModelAdjudicate(b *testing.B) {
	art, _, err := ResolvePath(filepath.Join("testdata", "adjudicator.json"))
	if err != nil {
		b.Fatalf("resolve fixture: %v", err)
	}
	adj := NewAdjudicator(art)
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool: "refund_payment",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"order":"o-1001","amount":49.99}`)},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verdict := adj.Adjudicate(ctx, call)
		if verdict.Kind != abi.VerdictDeny {
			b.Fatalf("unexpected verdict: got %v want Deny", verdict.Kind)
		}
	}
}
