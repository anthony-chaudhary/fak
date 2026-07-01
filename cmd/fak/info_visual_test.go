package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

// TestSparklineTUI pins the sparkline contract: empty/zero-width render nothing, the output is
// exactly min(len,width) single-cell runes, an ascending series rises monotonically to the top
// rune, a flat series renders a single repeated mid baseline (not the floor), and a window wider
// than the budget samples the TAIL.
func TestSparklineTUI(t *testing.T) {
	if got := sparklineTUI(nil, 8); got != "" {
		t.Fatalf("empty series must render nothing, got %q", got)
	}
	if got := sparklineTUI([]float64{1, 2, 3}, 0); got != "" {
		t.Fatalf("zero width must render nothing, got %q", got)
	}

	asc := sparklineTUI([]float64{0, 1, 2, 3, 4, 5, 6, 7}, 16)
	if r := []rune(asc); len(r) != 8 {
		t.Fatalf("8 samples within budget must yield 8 cells, got %d (%q)", len(r), asc)
	}
	rs := []rune(asc)
	if rs[0] != guardInfoSparkRunes[0] {
		t.Errorf("ascending series must start at the lowest rune, got %q", string(rs[0]))
	}
	if rs[len(rs)-1] != guardInfoSparkRunes[len(guardInfoSparkRunes)-1] {
		t.Errorf("ascending series must end at the highest rune, got %q", string(rs[len(rs)-1]))
	}
	for i := 1; i < len(rs); i++ {
		if rs[i] < rs[i-1] {
			t.Errorf("ascending series must not dip at %d: %q", i, asc)
		}
	}

	// A flat series is steady, not zero: every cell is the same mid-ramp rune.
	flat := sparklineTUI([]float64{5, 5, 5, 5}, 8)
	mid := guardInfoSparkRunes[(len(guardInfoSparkRunes)-1)/2]
	if flat != strings.Repeat(string(mid), 4) {
		t.Errorf("flat series must be a repeated mid baseline %q, got %q", string(mid), flat)
	}

	// A window wider than the budget keeps the TAIL (the most recent samples).
	tail := sparklineTUI([]float64{0, 0, 0, 0, 9}, 2)
	if r := []rune(tail); len(r) != 2 || r[len(r)-1] != guardInfoSparkRunes[len(guardInfoSparkRunes)-1] {
		t.Errorf("tail sampling must keep the last (peak) sample, got %q", tail)
	}
}

// TestGaugeBarTUI pins the gauge: it is exactly width cells, 0 is all empty, 1 is all filled,
// a mid fraction is half filled, and out-of-range fractions clamp.
func TestGaugeBarTUI(t *testing.T) {
	if got := gaugeBarTUI(0.5, 0); got != "" {
		t.Fatalf("zero width must render nothing, got %q", got)
	}
	for _, tc := range []struct {
		frac     float64
		fill     int
		expempty int
	}{
		{0, 0, 10},
		{1, 10, 0},
		{0.5, 5, 5},
		{-1, 0, 10}, // clamp low
		{2, 10, 0},  // clamp high
	} {
		got := gaugeBarTUI(tc.frac, 10)
		if dispWidthTUI(got) != 10 {
			t.Errorf("frac %v: gauge must be 10 cells, got %d (%q)", tc.frac, dispWidthTUI(got), got)
		}
		if fill := strings.Count(got, "█"); fill != tc.fill {
			t.Errorf("frac %v: want %d filled cells, got %d (%q)", tc.frac, tc.fill, fill, got)
		}
		if empty := strings.Count(got, "░"); empty != tc.expempty {
			t.Errorf("frac %v: want %d empty cells, got %d (%q)", tc.frac, tc.expempty, empty, got)
		}
	}
}

// provenVisualVars is a snapshot with a PROVEN cache and a non-trivial safety surface, used by the
// content/fit tests.
func provenVisualVars() guardInfoVars {
	var v guardInfoVars
	v.Gateway.UptimeSeconds = 3 * 3600
	v.Gateway.InflightRequests = 1
	v.Inference.Turns = 5
	v.Kernel.Denies = 1
	v.Kernel.Transforms = 2
	v.Kernel.Quarantines = 1
	v.Kernel.ResultDenies = 1 // folds into set-aside => 2
	v.VCache = &struct {
		CacheReadTokens int64   `json:"cache_read_tokens"`
		SavedTokenEquiv float64 `json:"saved_token_equiv"`
		HitRate         float64 `json:"hit_rate"`
		Multiplier      float64 `json:"multiplier"`
		Status          string  `json:"status"`
	}{CacheReadTokens: 1000, SavedTokenEquiv: 12345, HitRate: 0.88, Multiplier: 2.1, Status: "PROVEN"}
	return v
}

// TestRenderGuardInfoVisualBlockContent proves the full-height block carries both sub-panes and
// the live numbers in plain words: the trends gutter labels, the tasks gutter labels, the section
// rules, the gauge/sparkline glyphs, the cache verdict, and the safety counters.
func TestRenderGuardInfoVisualBlockContent(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	for i := 0; i < 6; i++ {
		v := provenVisualVars()
		v.VCache.SavedTokenEquiv = float64(i * 1000) // a rising savings trend to shape the sparkline
		v.Inference.Turns = int64(i)
		tr.push(v)
	}
	block := renderGuardInfoVisualBlock(provenVisualVars(), tr, 120, 0 /*roomy*/)

	for _, want := range []string{
		"── trends ", "── tasks ", // both sub-pane rules
		" save  ", " hit   ", " work  ", // trend gutter labels
		" cache  ", " safety ", // task gutter labels
		"saving money", // cache verdict
		"88%", "×2.10", // hit + multiplier
		"+12,345 tok",                         // signed savings
		"blocked 1", "fixed 2", "set aside 2", // safety counters
	} {
		if !strings.Contains(block, want) {
			t.Errorf("visual block missing %q:\n%s", want, block)
		}
	}
	// The block must actually be drawing a gauge and a sparkline (not just text).
	if !strings.ContainsAny(block, "▁▂▃▄▅▆▇█") {
		t.Errorf("visual block has no sparkline/gauge glyphs:\n%s", block)
	}
	if !strings.Contains(block, "░") {
		t.Errorf("cache gauge must show an unfilled remainder for 88%%:\n%s", block)
	}
}

func TestRenderGuardInfoVisualBlockIncidentRegion(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := provenVisualVars()
	v.Upstream.ErrorsByKind = map[string]uint64{
		"auth":         1,
		"rate_limited": 2,
	}
	v.Upstream.AuthRefreshByOutcome = map[string]uint64{
		"exhausted": 1,
	}
	v.Upstream.Retries = 3
	for i := 0; i < 3; i++ {
		tr.push(v)
	}

	block := renderGuardInfoVisualBlock(v, tr, 120, 8)
	for _, want := range []string{
		" incident ",
		"upstream auth/401 x1",
		"rate_limited/429 x2",
		"auth-refresh exhausted x1",
		"retries x3",
	} {
		if !strings.Contains(block, want) {
			t.Fatalf("incident panel missing %q:\n%s", want, block)
		}
	}
	if strings.Contains(renderGuardInfoLine(v), "auth/401") || strings.Contains(renderGuardInfoLine(v), "rate_limited/429") {
		t.Fatalf("upstream incident leaked into compact status line:\n%s", renderGuardInfoLine(v))
	}
}

func TestRenderGuardInfoVisualBlockCacheAttributionSplit(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := provenVisualVars()
	v.CacheAttribution = cacheAttributionFixture(120, 0, 120)
	for i := 0; i < 3; i++ {
		tr.push(v)
	}

	block := renderGuardInfoVisualBlock(v, tr, 180, 0)
	for _, want := range []string{
		"split default cache 100%",
		"fak 0%",
		"~120 tok",
		"~0 tok",
	} {
		if !strings.Contains(block, want) {
			t.Fatalf("visual cache attribution missing %q:\n%s", want, block)
		}
	}
}

// TestRenderGuardInfoVisualBlockFits proves the block never wraps and never exceeds the pane: for
// a range of widths and heights every row is within the width budget and the row count respects
// the height budget (the property that keeps the in-place redraw exact in a small 20% pane).
func TestRenderGuardInfoVisualBlockFits(t *testing.T) {
	tr := newGuardInfoTrend(guardInfoTrendCap)
	for i := 0; i < 12; i++ {
		tr.push(provenVisualVars())
	}
	for _, width := range []int{24, 40, 80, 120, 200} {
		for _, height := range []int{1, 2, 3, 4, 5, 6, 8, 10} {
			block := renderGuardInfoVisualBlock(provenVisualVars(), tr, width, height)
			rows := strings.Split(block, "\n")
			if height > 0 && len(rows) > height {
				t.Errorf("w=%d h=%d: %d rows exceeds height budget", width, height, len(rows))
			}
			if len(rows) == 0 {
				t.Errorf("w=%d h=%d: block must have at least one row", width, height)
			}
			for _, r := range rows {
				if dw := dispWidthTUI(r); dw > width {
					t.Errorf("w=%d h=%d: row %d cells wide, must be <= %d: %q", width, height, dw, width, r)
				}
			}
		}
	}
}

// TestWriteGuardInfoFrame pins the multi-line in-place redraw: the first paint moves no cursor and
// returns the row count, and a redraw of a multi-row block moves the cursor up to the block top and
// clears down before reprinting.
func TestWriteGuardInfoFrame(t *testing.T) {
	var b bytes.Buffer
	rows := writeGuardInfoFrame(&b, "a\nb\nc", 0)
	if rows != 3 {
		t.Fatalf("first frame must report 3 rows, got %d", rows)
	}
	if got := b.String(); got != "a\nb\nc" {
		t.Fatalf("first frame must print the block with no leading cursor move, got %q", got)
	}
	if strings.Contains(b.String(), "\033[") {
		t.Fatalf("first frame must emit NO cursor escapes, got %q", b.String())
	}

	b.Reset()
	rows = writeGuardInfoFrame(&b, "x\ny\nz", 3)
	out := b.String()
	if !strings.HasPrefix(out, "\033[2A\r\033[J") {
		t.Fatalf("redraw of a 3-row block must move up 2 + clear down first, got %q", out)
	}
	if !strings.HasSuffix(out, "x\ny\nz") {
		t.Fatalf("redraw must reprint the new block, got %q", out)
	}
	if rows != 3 {
		t.Fatalf("redraw must report the new row count, got %d", rows)
	}

	// A single-row block needs no cursor-up, just a carriage-return + clear-down.
	b.Reset()
	writeGuardInfoFrame(&b, "solo", 1)
	if got := b.String(); got != "\r\033[Jsolo" {
		t.Fatalf("single-row redraw must be \\r + clear-down + text, got %q", got)
	}
}

// TestRunInfoOverlayVisualTTY proves the visual mode end-to-end: on a TTY it prints the live
// intro, redraws a multi-line sub-pane frame (cursor-up + clear-down, with sparkline/gauge
// glyphs), and still ends on the gateway-closed line when the guarded session goes away.
func TestRunInfoOverlayVisualTTY(t *testing.T) {
	c := healthyThenGoneClient(t, 2)
	var stdout, stderr bytes.Buffer
	code := runGuardInfoOverlay(&stdout, &stderr, c, time.Millisecond, false /*once*/, true /*tty*/, 80 /*width*/, 8 /*height*/, "visual")
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "live sub-panes") {
		t.Fatalf("visual mode must print the live intro:\n%s", out)
	}
	if !strings.Contains(out, "\033[J") {
		t.Fatalf("visual mode must redraw the frame in place (clear-down escape):\n%q", out)
	}
	if !strings.ContainsAny(out, "▁▂▃▄▅▆▇█") {
		t.Fatalf("visual frame must draw sparkline/gauge glyphs:\n%s", out)
	}
	if !strings.Contains(out, "fak info: gateway closed") {
		t.Fatalf("must end on the gateway-closed line:\n%s", out)
	}
}

// TestRunInfoRejectsBadStyle proves an unknown --style is a usage error (exit 2), not a silent
// fallback — a typo should be visible, not quietly ignored.
func TestRunInfoRejectsBadStyle(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runInfo(&stdout, &stderr, []string{"--gateway-url", "http://127.0.0.1:1", "--style", "fancy"}); code != 2 {
		t.Fatalf("bad style exit = %d, want 2", code)
	}
}
