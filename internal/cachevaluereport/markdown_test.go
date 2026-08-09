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

// #2039: the markdown P&L surfaces the compaction lever's fire/starve/shed trend, so a
// lever that fired but shed nothing (anchor-starved, #1407) is visible as fired>0,
// anchor_starved>0, shed_tok=0 — not silence — in the --markdown render path too.
func TestRenderTwoTrackMarkdown_showsCompactionLeverHealth(t *testing.T) {
	rep := sampleTwoTrack()
	rep.Track2 = append(rep.Track2, SavingsBucket{
		Period:                  "2026-W26",
		Provider:                "fak",
		Mechanism:               "compaction_shed",
		Sessions:                3,
		CompactionFired:         8,
		CompactionBailed:        1,
		CompactionAnchorStarved: 5,
		CompactionShedTokens:    0,
		CompactionBudget:        48000,
	})
	md := RenderTwoTrackMarkdown(rep)
	if !strings.Contains(md, "Compaction lever health") {
		t.Fatalf("markdown must show the compaction health section:\n%s", md)
	}
	for _, want := range []string{"8", "5", "48000"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown compaction health must include %q:\n%s", want, md)
		}
	}
}

// #2039: a report with no compaction buckets emits no compaction-health section (an idle
// lever must not fabricate a health row), mirroring the terminal render's guard.
func TestRenderTwoTrackMarkdown_omitsCompactionHealthWhenIdle(t *testing.T) {
	md := RenderTwoTrackMarkdown(sampleTwoTrack())
	if strings.Contains(md, "Compaction lever health") {
		t.Fatalf("a report with no compaction buckets must not emit the section:\n%s", md)
	}
}

// #3394: the markdown P&L surfaces the savings_feed freshness/drain-lag row so a stale
// savings dashboard flags its own staleness in the --markdown render path, not only in
// the JSON ComponentHealth. A stale row renders a prominent banner carrying its
// hours-behind reason and next action; a measured row renders the quiet fresh line;
// and a report with no freshness row renders no banner at all (no fabricated freshness).
func TestRenderTwoTrackMarkdown_surfacesFeedFreshness(t *testing.T) {
	base := sampleTwoTrack()

	stale := base
	stale.ComponentHealth = []ComponentHealth{{
		Plane: "savings_feed", Component: "feed_freshness", Status: "stale",
		Reason: "last savings row is 180.0h behind now (threshold 48h)", NextAction: "re-run the savings feed",
	}}
	md := RenderTwoTrackMarkdown(stale)
	if !strings.Contains(md, "savings feed STALE") || !strings.Contains(md, "180.0h behind") || !strings.Contains(md, "re-run the savings feed") {
		t.Fatalf("a stale feed must render a banner with its reason and next action:\n%s", md)
	}

	measured := base
	measured.ComponentHealth = []ComponentHealth{{
		Plane: "savings_feed", Component: "feed_freshness", Status: "measured",
		Reason: "last savings row is 2.0h behind now (threshold 48h)",
	}}
	md = RenderTwoTrackMarkdown(measured)
	if strings.Contains(md, "savings feed STALE") {
		t.Fatalf("a measured feed must not raise the stale banner:\n%s", md)
	}
	if !strings.Contains(md, "savings feed fresh") || !strings.Contains(md, "2.0h behind") {
		t.Fatalf("a measured feed must render the quiet fresh line:\n%s", md)
	}

	if md := RenderTwoTrackMarkdown(base); strings.Contains(md, "savings feed STALE") || strings.Contains(md, "savings feed fresh") {
		t.Fatalf("a report with no freshness row must render no feed banner:\n%s", md)
	}
}

func TestRenderMarkdownDebitsRecallInjectionInsideKPI(t *testing.T) {
	r := sampleTwoTrack()
	r.CumulativeNetUSD = 1.0
	r.BrokeEven = true
	r.RecallInjectionDebit = RecallInjectionDebit{Injections: 1, Records: 2, EstimatedTokens: 400, EstimatedUSD: 0.25}
	md := RenderTwoTrackMarkdown(r)
	if !strings.Contains(md, "| recall injection debit | -$0.2500") {
		t.Fatalf("missing measured recall debit row: %s", md)
	}
	if !strings.Contains(md, "| cumulative net | $0.7500") {
		t.Fatalf("net did not subtract recall debit: %s", md)
	}
}
