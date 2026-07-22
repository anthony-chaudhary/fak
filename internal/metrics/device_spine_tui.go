package metrics

// device_spine_tui.go — the interactive fleet TUI as a THIN renderer over the
// device-telemetry spine (issue #4366, parent #3237, epic #3236). ZML's `tui/*`
// is a thin reader over the shared snapshot; this is the Go form of that
// mechanism. It adds NO new collection path: every surface here consumes a
// snapshot the caller already got from Buffer.Snapshot()/Federate, exactly the
// slice RenderText/RenderJSON/RenderProm read. The interactive state that the
// one-shot RenderText cannot hold — the in-memory sparkline history — lives in
// FleetView, fed by successive Buffer.Poll snapshots via Observe. The grid and
// drill-down bodies are driven by the SAME metricTable as every other renderer,
// so a metric added to the table appears on the TUI for free.
//
// Generation: gen/next. Promotion evidence (toward "now"): wire FleetView into
// a live `fak status`/`fak metrics --watch` verb polling Buffer.Poll on a
// ticker, and capture an operator dogfood screenshot on a real multi-device
// host. Demotion / retirement evidence: if operators only ever consume the
// one-shot RenderText or the Prom/JSON scrape, the interactive surface is dead
// weight — retire FleetView and keep the one-shot. Invalidating assumption: the
// sparkline history is assumed to fit an in-memory bounded ring per
// device×metric; a fleet wide enough that the ring set outgrows memory would
// need a downsampled or disk-backed store instead of this flat map of rings.

import (
	"sort"
	"strconv"
	"strings"
)

// sparkCap bounds each in-memory sparkline ring: at most this many of the most
// recent samples are retained, so the history is O(1) memory per device×metric
// no matter how long the TUI runs.
const sparkCap = 32

// sparkRunes is the eight-level unicode block ramp the sparkline renders onto,
// lowest amplitude first.
var sparkRunes = []rune("▁▂▃▄▅▆▇█")

// sparkline is a bounded ring buffer of recent metric samples fed by successive
// Buffer.Poll snapshots. Push appends the newest sample and drops the oldest
// once the ring is full, so len(vals) never exceeds cap — the "sparkline ring
// bounded" contract. Render maps the retained window onto the block ramp,
// scaled to the window's own min..max so a flat series reads flat.
type sparkline struct {
	vals []float64 // most-recent-last, len always <= cap
	cap  int
}

func newSparkline(capacity int) *sparkline {
	if capacity < 1 {
		capacity = 1
	}
	return &sparkline{cap: capacity}
}

// push appends v and trims the ring back to cap from the front, dropping the
// oldest sample. The retained slice is always the most-recent cap values.
func (s *sparkline) push(v float64) {
	s.vals = append(s.vals, v)
	if len(s.vals) > s.cap {
		s.vals = s.vals[len(s.vals)-s.cap:]
	}
}

// render draws the whole retained window as block runes. An empty ring renders
// to the empty string; a flat window renders to the lowest block, not a torn
// scale.
func (s *sparkline) render() string { return s.renderLast(len(s.vals)) }

// renderLast draws only the most-recent n samples (n clamped to the retained
// window), so the overview grid can show a short spark while the drill-down
// shows the full ring — both off the same bounded history.
func (s *sparkline) renderLast(n int) string {
	if n > len(s.vals) {
		n = len(s.vals)
	}
	if n <= 0 {
		return ""
	}
	window := s.vals[len(s.vals)-n:]
	min, max := window[0], window[0]
	for _, v := range window {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	span := max - min
	var b strings.Builder
	for _, v := range window {
		idx := 0
		if span > 0 {
			idx = int((v-min)/span*float64(len(sparkRunes)-1) + 0.5)
		}
		b.WriteRune(sparkRunes[idx])
	}
	return b.String()
}

// FleetView is the interactive TUI's in-memory state: one bounded sparkline ring
// per device×metric, plus the first-seen device order so the grid is stable
// across polls. It holds the ONLY state the one-shot renderers lack — the
// sample history — and holds it off the shared snapshot, never a second
// collection path. Construct with NewFleetView, feed it with Observe(snapshot)
// once per Buffer.Poll, then Render* the same snapshot.
type FleetView struct {
	cap   int
	rings map[string]*sparkline // key: deviceKey(d) \x00 metric.Key
	order []string              // device keys in first-seen order
	seen  map[string]struct{}
}

// NewFleetView returns an empty view whose rings fill as Observe is called.
func NewFleetView() *FleetView {
	return &FleetView{cap: sparkCap, rings: map[string]*sparkline{}, seen: map[string]struct{}{}}
}

// Observe folds one snapshot into the sparkline history: every PRESENT metric
// pushes its value onto that device×metric's bounded ring; an unread (nil)
// metric pushes nothing, so a gap in collection is a gap in the spark, never a
// zero. Device order is recorded first-seen so the grid stays stable even as
// devices come and go across polls.
func (v *FleetView) Observe(snapshot []DeviceMetrics) {
	for _, d := range snapshot {
		dk := deviceKey(d)
		if _, ok := v.seen[dk]; !ok {
			v.seen[dk] = struct{}{}
			v.order = append(v.order, dk)
		}
		for _, m := range metricTable {
			val, ok := m.get(d)
			if !ok {
				continue
			}
			rk := dk + "\x00" + m.Key
			r := v.rings[rk]
			if r == nil {
				r = newSparkline(v.cap)
				v.rings[rk] = r
			}
			r.push(val)
		}
	}
}

// spark returns the rendered sparkline for a device×metric, or "" when no
// samples have been observed.
func (v *FleetView) spark(d DeviceMetrics, key string, last int) string {
	r := v.rings[deviceKey(d)+"\x00"+key]
	if r == nil {
		return ""
	}
	return r.renderLast(last)
}

// tuiKeyWidth is the width of the metric-label column, the widest metric key.
func tuiKeyWidth() int {
	w := 0
	for _, m := range metricTable {
		if len(m.Key) > w {
			w = len(m.Key)
		}
	}
	return w
}

const (
	tuiValWidth      = 7 // right-aligned value column
	tuiOverviewSpark = 8 // sparkline width inside an overview card
	tuiCardGap       = 2 // spaces between side-by-side cards
)

// tuiCardWidth is the inner width of one device card: label + value + spark.
func tuiCardWidth() int {
	return tuiKeyWidth() + 1 + tuiValWidth + 1 + tuiOverviewSpark
}

// formatVal renders a metric value compactly for a fixed-width cell, matching
// RenderText's null-on-error contract: a present value formats with minimal
// digits, an absent one is "-", never a zero.
func formatVal(v float64, ok bool) string {
	if !ok {
		return "-"
	}
	return strconv.FormatFloat(v, 'g', 4, 64)
}

// deviceCard renders one device as a fixed-height card: a title line then one
// line per metric in metricTable order (present → value+spark, absent → "-").
// Every card has exactly 1+len(metricTable) lines so cards join cleanly
// side-by-side in the overview grid — the grid body is driven entirely by the
// shared table.
func (v *FleetView) deviceCard(d DeviceMetrics) []string {
	kw := tuiKeyWidth()
	cw := tuiCardWidth()
	lines := make([]string, 0, 1+len(metricTable))
	title := d.Backend + "/" + d.DeviceID + " (" + textOrigin(d) + ")"
	lines = append(lines, padRight(clip(title, cw), cw))
	for _, m := range metricTable {
		val, ok := m.get(d)
		cell := padRight(m.Key, kw) + " " +
			padLeft(formatVal(val, ok), tuiValWidth) + " " +
			v.spark(d, m.Key, tuiOverviewSpark)
		lines = append(lines, padRight(clip(cell, cw), cw))
	}
	return lines
}

// RenderOverview renders the fleet overview grid: one fixed-height card per
// device, arranged into as many columns as `width` fits (responsive
// column-wrap — a narrow terminal stacks one card per band, a wide one packs
// several side-by-side). Device order is the view's stable first-seen order;
// devices absent from the view (never Observed) fall back to snapshot order.
// The snapshot is the SAME slice every other renderer reads — no new collection
// path. A nil snapshot renders nothing.
func (v *FleetView) RenderOverview(snapshot []DeviceMetrics, width int) []byte {
	if len(snapshot) == 0 {
		return nil
	}
	ordered := v.orderSnapshot(snapshot)

	cw := tuiCardWidth()
	cols := (width + tuiCardGap) / (cw + tuiCardGap)
	if cols < 1 {
		cols = 1
	}

	var buf strings.Builder
	for start := 0; start < len(ordered); start += cols {
		end := start + cols
		if end > len(ordered) {
			end = len(ordered)
		}
		band := ordered[start:end]
		cards := make([][]string, len(band))
		height := 0
		for i, d := range band {
			cards[i] = v.deviceCard(d)
			if len(cards[i]) > height {
				height = len(cards[i])
			}
		}
		for row := 0; row < height; row++ {
			for i, card := range cards {
				if i > 0 {
					buf.WriteString(strings.Repeat(" ", tuiCardGap))
				}
				if row < len(card) {
					buf.WriteString(card[row])
				} else {
					buf.WriteString(strings.Repeat(" ", cw))
				}
			}
			buf.WriteByte('\n')
		}
		buf.WriteByte('\n') // blank separator between bands
	}
	return []byte(buf.String())
}

// orderSnapshot returns the snapshot rows in the view's stable first-seen
// order, appending any rows the view never Observed in snapshot order.
func (v *FleetView) orderSnapshot(snapshot []DeviceMetrics) []DeviceMetrics {
	pos := make(map[string]int, len(v.order))
	for i, k := range v.order {
		pos[k] = i
	}
	ordered := append([]DeviceMetrics(nil), snapshot...)
	sort.SliceStable(ordered, func(i, j int) bool {
		pi, oki := pos[deviceKey(ordered[i])]
		pj, okj := pos[deviceKey(ordered[j])]
		if oki && okj {
			return pi < pj
		}
		return oki && !okj // known-order rows before unknown ones
	})
	return ordered
}

// RenderDrilldown renders one device's per-metric detail: every metric in
// metricTable order with its current value and the FULL sparkline ring (the
// whole retained history, not the short overview window). The device is
// selected by backend+device identity; an unknown device renders a
// self-describing "no such device" line so the drill-down never renders a torn
// pane. Body is driven by the shared table, over the same snapshot.
func (v *FleetView) RenderDrilldown(snapshot []DeviceMetrics, backend, device string) []byte {
	var target *DeviceMetrics
	for i := range snapshot {
		if snapshot[i].Backend == backend && snapshot[i].DeviceID == device {
			target = &snapshot[i]
			break
		}
	}
	var buf strings.Builder
	if target == nil {
		buf.WriteString("no such device: " + backend + "/" + device + "\n")
		return []byte(buf.String())
	}
	kw := tuiKeyWidth()
	buf.WriteString(target.Backend + "/" + target.DeviceID + " (" + textOrigin(*target) + ")\n")
	for _, m := range metricTable {
		val, ok := m.get(*target)
		buf.WriteString(padRight(m.Key, kw) + " " +
			padLeft(formatVal(val, ok), tuiValWidth) + " " +
			v.spark(*target, m.Key, sparkCap) + "\n")
	}
	return []byte(buf.String())
}

// clip truncates s to at most w runes so an over-long title/cell never breaks
// card alignment.
func clip(s string, w int) string {
	r := []rune(s)
	if len(r) <= w {
		return s
	}
	if w <= 0 {
		return ""
	}
	return string(r[:w])
}

func padRight(s string, w int) string {
	if n := w - len([]rune(s)); n > 0 {
		return s + strings.Repeat(" ", n)
	}
	return s
}

func padLeft(s string, w int) string {
	if n := w - len([]rune(s)); n > 0 {
		return strings.Repeat(" ", n) + s
	}
	return s
}
