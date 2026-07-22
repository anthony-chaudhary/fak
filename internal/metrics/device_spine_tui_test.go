package metrics

import (
	"strings"
	"testing"
)

// TestSparklineRingBounded proves the sparkline ring is bounded: pushing past
// cap never grows the retained window beyond cap, and the retained window is
// always the most-recent cap samples (oldest dropped first). This is the
// "sparkline ring bounded" half of the renderer-contract gate.
func TestSparklineRingBounded(t *testing.T) {
	s := newSparkline(4)
	for i := 0; i < 100; i++ {
		s.push(float64(i))
	}
	if len(s.vals) != 4 {
		t.Fatalf("ring must stay bounded at cap=4, got len=%d", len(s.vals))
	}
	want := []float64{96, 97, 98, 99}
	for i, v := range want {
		if s.vals[i] != v {
			t.Fatalf("ring should keep the most-recent window %v, got %v", want, s.vals)
		}
	}
	// The rendered spark has one rune per retained sample — bounded too.
	if got := []rune(s.render()); len(got) != 4 {
		t.Fatalf("render must be bounded to cap runes, got %d", len(got))
	}
	// A flat window renders to the lowest block, not a torn scale.
	flat := newSparkline(4)
	for i := 0; i < 4; i++ {
		flat.push(5)
	}
	if flat.render() != strings.Repeat(string(sparkRunes[0]), 4) {
		t.Fatalf("flat window should render lowest block, got %q", flat.render())
	}
	// A rising ramp ends on the top block.
	if r := []rune(s.render()); r[len(r)-1] != sparkRunes[len(sparkRunes)-1] {
		t.Fatalf("rising ramp should peak at the top block, got %q", s.render())
	}
}

// TestFleetViewObserveFeedsRingsFromSnapshot proves Observe consumes ONLY the
// shared snapshot: each present metric pushes onto its device×metric ring, an
// unread metric pushes nothing (a gap stays a gap), and the rings stay bounded
// across many polls.
func TestFleetViewObserveFeedsRingsFromSnapshot(t *testing.T) {
	v := NewFleetView()
	for i := 0; i < sparkCap+10; i++ {
		snap := []DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(float64(i))}}
		v.Observe(snap)
	}
	r := v.rings["nvml\x00gpu0\x00tokens_per_second"]
	if r == nil {
		t.Fatal("Observe should have created the tokens_per_second ring")
	}
	if len(r.vals) != sparkCap {
		t.Fatalf("ring must stay bounded at sparkCap=%d across polls, got %d", sparkCap, len(r.vals))
	}
	// queue_depth was never present in any snapshot: no ring, no zeros invented.
	if _, ok := v.rings["nvml\x00gpu0\x00queue_depth"]; ok {
		t.Fatal("an unread metric must not create a ring (no invented zeros)")
	}
}

// TestRenderOverviewDrivenByMetricTable proves the overview grid body is driven
// by the SAME metricTable as every other renderer: each device card carries a
// title line plus exactly one line per metric in table order, present metrics
// show their value and absent ones show "-" (never 0).
func TestRenderOverviewDrivenByMetricTable(t *testing.T) {
	v := NewFleetView()
	snap := Federate(
		[]DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(42), UtilizationRatio: f(0.8)}},
	)
	v.Observe(snap)
	// One card, very wide terminal: a single band, one column.
	out := string(v.RenderOverview(snap, 400))
	lines := nonBlank(out)
	if len(lines) != 1+len(metricTable) {
		t.Fatalf("card must be title + one line per metricTable row (%d), got %d:\n%s",
			1+len(metricTable), len(lines), out)
	}
	if !strings.HasPrefix(lines[0], "nvml/gpu0 (local)") {
		t.Fatalf("title line wrong: %q", lines[0])
	}
	// Metric lines are in metricTable order and each names its key.
	for i, m := range metricTable {
		if !strings.HasPrefix(strings.TrimSpace(lines[1+i]), m.Key) {
			t.Fatalf("metric line %d not driven by metricTable: got %q want key %q",
				i, lines[1+i], m.Key)
		}
	}
	// Present value renders, absent renders "-".
	tps := lines[1+metricIndex(t, "tokens_per_second")]
	if !strings.Contains(tps, "42") {
		t.Fatalf("present tokens_per_second should show 42: %q", tps)
	}
	qd := lines[1+metricIndex(t, "queue_depth")]
	if !strings.Contains(qd, "-") || strings.Contains(qd, "0.0") {
		t.Fatalf("absent queue_depth should show '-' not a zero: %q", qd)
	}
}

// TestRenderOverviewResponsiveColumnWrap proves the layout is responsive: a
// wide terminal packs several device cards side-by-side into one band while a
// narrow terminal wraps to one card per band. Same snapshot, same view — only
// the width changes.
func TestRenderOverviewResponsiveColumnWrap(t *testing.T) {
	v := NewFleetView()
	snap := []DeviceMetrics{
		{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(1)},
		{Backend: "nvml", DeviceID: "gpu1", TokensPerSecond: f(2)},
		{Backend: "nvml", DeviceID: "gpu2", TokensPerSecond: f(3)},
	}
	v.Observe(snap)

	cw := tuiCardWidth()
	wide := string(v.RenderOverview(snap, cw*3+tuiCardGap*2+4))
	narrow := string(v.RenderOverview(snap, cw)) // fits exactly one column

	// Narrow: three bands (one card each) → three title lines on their own line.
	if got := countLinesWith(narrow, "nvml/gpu0"); got != 1 {
		t.Fatalf("narrow layout should place gpu0 alone, got %d occurrences", got)
	}
	narrowBands := blankSeparatedBands(narrow)
	wideBands := blankSeparatedBands(wide)
	if narrowBands <= wideBands {
		t.Fatalf("narrow layout should use MORE bands than wide (wrap): narrow=%d wide=%d\nwide:\n%s\nnarrow:\n%s",
			narrowBands, wideBands, wide, narrow)
	}
	if wideBands != 1 {
		t.Fatalf("wide layout should pack all three cards into one band, got %d bands:\n%s", wideBands, wide)
	}
	// In the wide band, all three titles share one row (side-by-side).
	firstRow := strings.SplitN(strings.TrimLeft(wide, "\n"), "\n", 2)[0]
	for _, id := range []string{"gpu0", "gpu1", "gpu2"} {
		if !strings.Contains(firstRow, id) {
			t.Fatalf("wide first row should carry all three cards side-by-side, missing %s: %q", id, firstRow)
		}
	}
}

// TestRenderDrilldownFullRingAndTable proves the per-device drill-down is driven
// by the shared table and shows the FULL sparkline ring (the whole retained
// history), and that an unknown device renders a self-describing line rather
// than a torn pane.
func TestRenderDrilldownFullRingAndTable(t *testing.T) {
	v := NewFleetView()
	for i := 0; i < 12; i++ {
		v.Observe([]DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(float64(i))}})
	}
	out := string(v.RenderDrilldown([]DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(11)}}, "nvml", "gpu0"))
	lines := nonBlank(out)
	if len(lines) != 1+len(metricTable) {
		t.Fatalf("drilldown must be header + one line per metricTable row, got %d:\n%s", len(lines), out)
	}
	// The tokens_per_second line carries a full 12-sample spark (>= overview's 8).
	tps := lines[1+metricIndex(t, "tokens_per_second")]
	spark := []rune(strings.TrimSpace(tps))
	// Count trailing block runes.
	blocks := 0
	for i := len(spark) - 1; i >= 0; i-- {
		if isBlock(spark[i]) {
			blocks++
		} else {
			break
		}
	}
	if blocks != 12 {
		t.Fatalf("drilldown should render the full 12-sample ring, got %d block runes: %q", blocks, tps)
	}
	// Unknown device: self-describing, not a panic or empty pane.
	missing := string(v.RenderDrilldown(nil, "nvml", "ghost"))
	if !strings.Contains(missing, "no such device") {
		t.Fatalf("unknown device should render a self-describing line, got %q", missing)
	}
}

func metricIndex(t *testing.T, key string) int {
	t.Helper()
	for i, m := range metricTable {
		if m.Key == key {
			return i
		}
	}
	t.Fatalf("metric %q not in table", key)
	return -1
}

func isBlock(r rune) bool {
	for _, b := range sparkRunes {
		if r == b {
			return true
		}
	}
	return false
}

func nonBlank(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}

func countLinesWith(s, sub string) int {
	n := 0
	for _, l := range strings.Split(s, "\n") {
		if strings.Contains(l, sub) {
			n++
		}
	}
	return n
}

// blankSeparatedBands counts the bands (card groups) a RenderOverview produced,
// each terminated by a blank separator line.
func blankSeparatedBands(s string) int {
	n := 0
	for _, l := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		if strings.TrimSpace(l) == "" {
			n++
		}
	}
	return n + 1 // trailing band has no counted blank after TrimRight
}
