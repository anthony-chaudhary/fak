package skillfootprint

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// syntheticCards generates n distinct CapCard definitions with realistic metadata.
func syntheticCards(n int) []capindex.CapCard {
	cards := make([]capindex.CapCard, n)
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("synthetic-skill-%03d", i)
		trigger := fmt.Sprintf("Use when executing autonomous workflow %d: handles tool admission, context inspection, and verification rungs.", i)
		intent := fmt.Sprintf("Execute autonomous workflow %d with verification rungs.", i)
		cards[i] = capindex.CapCard{
			Ref: capindex.CapRef{
				Kind:    capindex.CapKindSkill,
				Name:    name,
				Version: "1.0.0",
			},
			Trigger:   trigger,
			Intent:    intent,
			Digest:    fmt.Sprintf("sha256:%064x", i),
			CardBytes: []byte(fmt.Sprintf(`{"name":%q,"version":"1.0.0","intent":%q}`, name, intent)),
		}
	}
	return cards
}

// BenchmarkFold measures the throughput of folding real production skill cards into a resident floor.
func BenchmarkFold(b *testing.B) {
	root := repoRootForTest(b)
	cards := capindex.NewSkillResolver(SkillsDir(root)).Index()
	if len(cards) == 0 {
		b.Fatal("no skill cards found in real .claude/skills tree")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fp := Fold(cards)
		if fp.SkillCount != len(cards) {
			b.Fatalf("unexpected skill count: got %d, want %d", fp.SkillCount, len(cards))
		}
	}
}

// BenchmarkFoldSyntheticScaling measures fold performance across varied skill catalog sizes.
func BenchmarkFoldSyntheticScaling(b *testing.B) {
	for _, count := range []int{10, 50, 100, 200} {
		b.Run(fmt.Sprintf("skills=%d", count), func(b *testing.B) {
			cards := syntheticCards(count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fp := Fold(cards)
				if fp.SkillCount != count {
					b.Fatalf("unexpected skill count: got %d, want %d", fp.SkillCount, count)
				}
			}
		})
	}
}

// BenchmarkMeasure measures end-to-end pricing of the real repository skills tree.
func BenchmarkMeasure(b *testing.B) {
	root := repoRootForTest(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fp := Measure(root)
		if fp.SkillCount == 0 || fp.DescFloor == 0 {
			b.Fatalf("unexpected zero floor from Measure: %+v", fp)
		}
	}
}

// BenchmarkCheckDescriptions measures gate evaluation against the committed ceiling on the real floor.
func BenchmarkCheckDescriptions(b *testing.B) {
	root := repoRootForTest(b)
	fp := Measure(root)
	if fp.SkillCount == 0 {
		b.Fatal("no skills measured")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := CheckDescriptions(fp); err != nil {
			b.Fatalf("unexpected gate refusal: %v", err)
		}
	}
}

// BenchmarkCheckDescriptionsRefusal measures structured error creation on budget overflow.
func BenchmarkCheckDescriptionsRefusal(b *testing.B) {
	cards := []capindex.CapCard{
		skillCard("overflow-skill", strings.Repeat("x", SkillDescriptionBudgetBytes+100)),
	}
	fp := Fold(cards)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := CheckDescriptions(fp)
		if err == nil {
			b.Fatal("expected refusal for budget overflow")
		}
	}
}

// BenchmarkApproxTokens measures the throughput of token estimation arithmetic.
func BenchmarkApproxTokens(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toks := ApproxTokens(SkillDescriptionBudgetBytes)
		if toks <= 0 {
			b.Fatalf("unexpected non-positive token count: %d", toks)
		}
	}
}

// BenchmarkSkillsDir measures resolution of the .claude/skills path under repo root.
func BenchmarkSkillsDir(b *testing.B) {
	root := repoRootForTest(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dir := SkillsDir(root)
		if len(dir) == 0 {
			b.Fatal("empty skills dir")
		}
	}
}

// BenchmarkSkillFootprintLifecycle measures end-to-end measuring, gating, and token projection.
func BenchmarkSkillFootprintLifecycle(b *testing.B) {
	root := repoRootForTest(b)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fp := Measure(root)
		if err := CheckDescriptions(fp); err != nil {
			b.Fatalf("check descriptions failed: %v", err)
		}
		toks := ApproxTokens(fp.DescFloor)
		if toks <= 0 || fp.SkillCount == 0 {
			b.Fatalf("invalid lifecycle results: toks=%d count=%d", toks, fp.SkillCount)
		}
	}
}

// TestBenchmarkSanity verifies that the skillfootprint benchmark suite executes cleanly.
func TestBenchmarkSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	res := testing.Benchmark(BenchmarkSkillFootprintLifecycle)
	if res.N <= 0 {
		t.Fatalf("expected iterations > 0, got %d", res.N)
	}
}
