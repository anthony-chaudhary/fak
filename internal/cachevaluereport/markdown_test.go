package cachevaluereport

import (
	"strings"
	"testing"
)

func sampleTwoTrack() TwoTrackReport {
	return TwoTrackReport{
		Verdict:         "MEASURED",
		Finding:         "Track 1 realized reuse 0.780 (improved); Track 2 cumulative $0.5000 (broke even)",
		ProjectionFence: projectionFence,
		Track1: Report{
			LatestReuseRatio: 0.78,
			LatestTrend:      TrendImproved,
			Buckets: []Bucket{
				{Period: "2026-W25", RealizedReuseRatio: 0.50},
				{Period: "2026-W26", RealizedReuseRatio: 0.78},
			},
		},
		Track2: []SavingsBucket{
			{Period: "2026-W25", NetUSD: -0.50, CumulativeNetUSD: -0.50},
			{Period: "2026-W26", NetUSD: 1.00, CumulativeNetUSD: 0.50},
		},
		OwnerAttribution: []OwnerAttributionBucket{
			{Period: "2026-W25", ProviderPromptCacheTokenEquiv: 100, FakAuthoredTokenEquiv: 50, FakKVPrefixReusedTokens: 50},
			{Period: "2026-W26", ProviderPromptCacheTokenEquiv: 900, FakAuthoredTokenEquiv: 1100, FakKVPrefixReusedTokens: 800, FakCompactionShedTokens: 300},
		},
		LatestNetUSD:     1.00,
		CumulativeNetUSD: 0.50,
		BrokeEven:        true,
	}
}

// #1305 acceptance: the markdown render carries a mermaid block, sparkline glyphs, and the
// provenance labels that keep WITNESSED reuse and the OBSERVED $ projection distinct.
func TestRenderTwoTrackMarkdown_hasMermaidSparklineAndProvenance(t *testing.T) {
	md := RenderTwoTrackMarkdown(sampleTwoTrack())
	if !strings.Contains(md, "```mermaid") || !strings.Contains(md, "xychart-beta") {
		t.Fatalf("markdown must contain a mermaid xychart block:\n%s", md)
	}
	if !strings.ContainsAny(md, string(sparkGlyphs)) {
		t.Fatalf("markdown must contain sparkline block glyphs:\n%s", md)
	}
	if !strings.Contains(md, "WITNESSED") || !strings.Contains(md, "OBSERVED") {
		t.Fatalf("markdown must label both provenances (WITNESSED + OBSERVED):\n%s", md)
	}
	if !strings.Contains(md, "provider prompt-cache token-equiv") || !strings.Contains(md, "fak-authored token-equiv") {
		t.Fatalf("markdown must carry the owner attribution token split:\n%s", md)
	}
	// A mermaid chart per track (Track 1 reuse + Track 2 net).
	if n := strings.Count(md, "```mermaid"); n != 2 {
		t.Fatalf("expected one mermaid chart per track (2), got %d:\n%s", n, md)
	}
}

// An empty report renders no chart but still produces valid, labelled markdown (no panic,
// no half-built mermaid block).
func TestRenderTwoTrackMarkdown_emptyIsGraceful(t *testing.T) {
	md := RenderTwoTrackMarkdown(TwoTrackReport{Verdict: "INSUFFICIENT", ProjectionFence: projectionFence})
	if strings.Contains(md, "```mermaid") {
		t.Fatalf("an empty report must not emit a mermaid block:\n%s", md)
	}
	if !strings.Contains(md, "INSUFFICIENT") {
		t.Fatalf("the verdict must still be rendered:\n%s", md)
	}
}

func TestMarkdownSparkline(t *testing.T) {
	if s := markdownSparkline(nil); s != "" {
		t.Errorf("empty series must yield empty sparkline, got %q", s)
	}
	s := []rune(markdownSparkline([]float64{0, 0.5, 1}))
	if len(s) != 3 {
		t.Fatalf("expected 3 glyphs, got %d", len(s))
	}
	if s[0] != sparkGlyphs[0] || s[2] != sparkGlyphs[len(sparkGlyphs)-1] {
		t.Errorf("min→max series must span low→high glyph; got %q..%q", string(s[0]), string(s[2]))
	}
}
