package metrics

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// SOTA parity performance dashboard (issue #196, track G foundation).
//
// The repo already provisions Grafana (tools/grafana) and already keeps a link
// registry for the #grafana channel (docs/grafana/links.json), so "add a Grafana
// stack" is NOT the missing piece. What was missing is the thing those surfaces
// would point at: a maintained, testable model of fak's PARITY performance —
// per-metric history plus the SOTA baseline each point was measured against —
// and an honest answer to "is it published yet?".
//
// This file is that model. It is pure and stdlib-only: a caller folds dated
// samples (from the committed benchmark corpus — experiments/parity, the
// docs/nightrun ledgers, experiments/benchmark/runs) into ParityDashboardInput,
// and gets back
//
//   - one panel per tracked metric carrying the FULL historical series
//     (the "historical tracking" ask),
//   - the latest fak-vs-SOTA comparison with a direction-aware verdict
//     (the "SOTA comparison" ask), and
//   - a publication state that resolves a public dashboard URL ONLY from a
//     caller-supplied absolute base (the "public URL" ask).
//
// The publication rule is the load-bearing honesty here, and it matches the
// standing rule in docs/grafana/README.md — "No URL here is fabricated." A
// dashboard with no configured public Grafana renders as `unpublished` with a
// named reason; it never invents a plausible-looking https URL to tick a box.
// Public HOSTING itself is infrastructure this package cannot conjure, so the
// model reports its absence instead of pretending.
//
// OpenMetrics() feeds the same snapshot to Prometheus through this package's
// existing exposition renderer, so a Grafana panel has a real data path rather
// than a hand-maintained screenshot.
//
// Generation: gen/next. The issue's foundation (Grafana provisioning, the link
// registry, a benchmark corpus) already exists on trunk, and nothing here needs
// an architecture bet — but the public-hosting half needs infrastructure and an
// exposure decision fak does not own yet, which is what keeps it out of gen/now.
//
//   - Promotion evidence: a real parity corpus folds through BuildParityDashboard
//     on an operator surface, and a `category: public-demo` entry with an absolute
//     url lands in docs/grafana/links.json — Publication then resolves to
//     ParityPublished and the last acceptance box is witnessable.
//   - Demotion / retirement evidence: retire this shape if the parity corpus stops
//     being maintained (no dated sample newer than the review cadence), or if a
//     hosted upstream dashboard subsumes it and the repo stops folding its own.
//   - Invalidating assumption: that fak-vs-SOTA is a SCALAR ratio per metric. If
//     parity becomes multi-dimensional per axis (quality x cost x latency traded
//     against each other), ParityComparison.Ratio stops being meaningful and the
//     verdict rule must move before anything else here.

// ParityDirection is the closed set of "which way is better" for a metric. It is
// what makes a verdict direction-aware: 1.2x throughput is ahead, 1.2x latency is
// behind, and a dashboard that ignores the distinction reports regressions as wins.
type ParityDirection string

const (
	// ParityHigherIsBetter marks a metric where a larger value is better
	// (tokens/sec, benchmark pass rate, cache reuse).
	ParityHigherIsBetter ParityDirection = "higher-is-better"
	// ParityLowerIsBetter marks a metric where a smaller value is better
	// (p95 latency, cost per task, time-to-first-token).
	ParityLowerIsBetter ParityDirection = "lower-is-better"
)

// ParityVerdict is the closed set of SOTA-comparison outcomes for one metric.
type ParityVerdict string

const (
	// ParityAhead means fak beats the SOTA baseline by more than the metric's band.
	ParityAhead ParityVerdict = "ahead"
	// ParityAtParity means fak is within the metric's tolerance band of the baseline.
	ParityAtParity ParityVerdict = "at-parity"
	// ParityBehind means fak trails the SOTA baseline by more than the band.
	ParityBehind ParityVerdict = "behind"
	// ParityUnknown means no comparison was possible — no sample for the metric, or
	// a baseline of zero. It is reported, never silently rendered as at-parity.
	ParityUnknown ParityVerdict = "unknown"
)

// ParityMetric declares one tracked performance axis of the dashboard.
type ParityMetric struct {
	// Slug is the stable key samples bind to (kebab-case, e.g. "tokens-per-second").
	Slug string `json:"slug"`
	// Title is the human panel label.
	Title string `json:"title"`
	// Unit is the display unit (e.g. "tok/s", "ms", "%").
	Unit string `json:"unit"`
	// Direction says which way is better. Empty defaults to ParityHigherIsBetter.
	Direction ParityDirection `json:"direction"`
	// Band is the fractional tolerance within which fak counts as at-parity
	// (0.05 = +/-5%). Zero means an exact-match band.
	Band float64 `json:"band"`
}

// ParitySample is one dated measurement: fak's value and the SOTA baseline it was
// measured against, on the same run. Both sides are carried per sample because the
// baseline itself moves — a dashboard that pins one static baseline goes stale
// silently, which is the failure mode this shape exists to avoid.
type ParitySample struct {
	// Metric is the ParityMetric.Slug this sample belongs to.
	Metric string `json:"metric"`
	// At is when the measurement was taken. Rendered in UTC.
	At time.Time `json:"at"`
	// Fak is fak's measured value.
	Fak float64 `json:"fak"`
	// SOTA is the reference stack's value on the same axis and run.
	SOTA float64 `json:"sota"`
	// Baseline names the reference stack the SOTA value came from (e.g. "vllm-0.9").
	Baseline string `json:"baseline"`
	// Source is provenance — the committed run or ledger the numbers were read from.
	Source string `json:"source"`
}

// ParityDashboardInput is the caller-supplied snapshot the dashboard folds. This
// package reads no disk and no clock, so a test can assert exact bytes.
type ParityDashboardInput struct {
	// Title is the dashboard's human title.
	Title string `json:"title"`
	// UID is the Grafana dashboard uid, used to resolve the public URL.
	UID string `json:"uid"`
	// Metrics is the declared axis set. A sample naming an undeclared metric is
	// an error, not a silently-dropped row.
	Metrics []ParityMetric `json:"metrics"`
	// Samples is the dated corpus, in any order.
	Samples []ParitySample `json:"samples"`
	// PublicBaseURL is the absolute base of the PUBLIC Grafana (e.g.
	// "https://grafana.example.org"). Empty, or anything that is not an absolute
	// http(s) URL, leaves the dashboard unpublished — never a fabricated link.
	PublicBaseURL string `json:"public_base_url"`
}

// ParityPoint is one point of a metric's historical series.
type ParityPoint struct {
	At       time.Time `json:"at"`
	Fak      float64   `json:"fak"`
	SOTA     float64   `json:"sota"`
	Baseline string    `json:"baseline"`
	Source   string    `json:"source"`
}

// ParityComparison is the latest fak-vs-SOTA readout for one metric.
type ParityComparison struct {
	// At is the timestamp of the most recent sample.
	At time.Time `json:"at"`
	// Fak and SOTA are that sample's two sides.
	Fak  float64 `json:"fak"`
	SOTA float64 `json:"sota"`
	// Ratio is fak relative to the baseline, NORMALIZED so >1 always means fak is
	// ahead regardless of the metric's direction. Zero when Verdict is ParityUnknown.
	Ratio float64 `json:"ratio"`
	// Verdict is the direction-aware outcome.
	Verdict ParityVerdict `json:"verdict"`
	// Baseline names the reference stack compared against.
	Baseline string `json:"baseline"`
}

// ParityPanel is one metric's panel: the declared axis, its full history, and the
// latest comparison.
type ParityPanel struct {
	Metric ParityMetric     `json:"metric"`
	Points []ParityPoint    `json:"points"`
	Latest ParityComparison `json:"latest"`
}

// ParityPublishState is the closed set of publication states.
type ParityPublishState string

const (
	// ParityPublished means a public URL was resolved from a real absolute base.
	ParityPublished ParityPublishState = "published"
	// ParityUnpublished means no public URL exists yet, with Reason naming why.
	ParityUnpublished ParityPublishState = "unpublished"
)

// ParityPublication is the honest publication state of the dashboard. It exists so
// "Public URL" is a checkable fact rather than a claim: URL is non-empty only when
// it was derived from a caller-supplied absolute base and a uid.
type ParityPublication struct {
	State ParityPublishState `json:"state"`
	// URL is the absolute public dashboard link, empty when unpublished.
	URL string `json:"url"`
	// Reason names why the dashboard is unpublished. Empty when published.
	Reason string `json:"reason"`
}

// ParityDashboard is the folded snapshot: one panel per declared metric (in
// declaration order) plus the publication state.
type ParityDashboard struct {
	// Schema is the versioned shape tag, never edited in place.
	Schema      string            `json:"schema"`
	Title       string            `json:"title"`
	UID         string            `json:"uid"`
	Panels      []ParityPanel     `json:"panels"`
	Publication ParityPublication `json:"publication"`
}

// ParityDashboardSchema is the versioned shape tag carried by every dashboard.
const ParityDashboardSchema = "fak-parity-dashboard/1"

// BuildParityDashboard folds declared metrics and dated samples into the dashboard
// snapshot. It is total over its declared inputs: an undeclared metric slug or a
// duplicate declaration is an error, so a typo cannot silently erase a panel.
func BuildParityDashboard(in ParityDashboardInput) (ParityDashboard, error) {
	if strings.TrimSpace(in.UID) == "" {
		return ParityDashboard{}, fmt.Errorf("metrics: parity dashboard needs a uid")
	}

	bySlug := make(map[string]int, len(in.Metrics))
	panels := make([]ParityPanel, 0, len(in.Metrics))
	for _, m := range in.Metrics {
		slug := strings.TrimSpace(m.Slug)
		if slug == "" {
			return ParityDashboard{}, fmt.Errorf("metrics: parity metric needs a slug")
		}
		if _, dup := bySlug[slug]; dup {
			return ParityDashboard{}, fmt.Errorf("metrics: duplicate parity metric %q", slug)
		}
		if m.Direction == "" {
			m.Direction = ParityHigherIsBetter
		}
		if m.Direction != ParityHigherIsBetter && m.Direction != ParityLowerIsBetter {
			return ParityDashboard{}, fmt.Errorf("metrics: parity metric %q has unknown direction %q", slug, m.Direction)
		}
		if m.Band < 0 {
			return ParityDashboard{}, fmt.Errorf("metrics: parity metric %q has negative band %v", slug, m.Band)
		}
		m.Slug = slug
		bySlug[slug] = len(panels)
		panels = append(panels, ParityPanel{Metric: m, Latest: ParityComparison{Verdict: ParityUnknown}})
	}

	for _, s := range in.Samples {
		idx, ok := bySlug[strings.TrimSpace(s.Metric)]
		if !ok {
			return ParityDashboard{}, fmt.Errorf("metrics: parity sample names undeclared metric %q", s.Metric)
		}
		panels[idx].Points = append(panels[idx].Points, ParityPoint{
			At:       s.At.UTC(),
			Fak:      s.Fak,
			SOTA:     s.SOTA,
			Baseline: s.Baseline,
			Source:   s.Source,
		})
	}

	for i := range panels {
		points := panels[i].Points
		sort.SliceStable(points, func(a, b int) bool {
			if !points[a].At.Equal(points[b].At) {
				return points[a].At.Before(points[b].At)
			}
			return points[a].Source < points[b].Source
		})
		panels[i].Points = points
		if len(points) > 0 {
			panels[i].Latest = compareParity(panels[i].Metric, points[len(points)-1])
		}
	}

	return ParityDashboard{
		Schema:      ParityDashboardSchema,
		Title:       in.Title,
		UID:         strings.TrimSpace(in.UID),
		Panels:      panels,
		Publication: resolveParityPublication(strings.TrimSpace(in.PublicBaseURL), strings.TrimSpace(in.UID)),
	}, nil
}

// compareParity reduces one point to a direction-aware verdict. Ratio is
// normalized so that >1 means fak is ahead on BOTH kinds of metric.
func compareParity(m ParityMetric, p ParityPoint) ParityComparison {
	out := ParityComparison{At: p.At, Fak: p.Fak, SOTA: p.SOTA, Baseline: p.Baseline, Verdict: ParityUnknown}
	if p.SOTA == 0 || p.Fak == 0 || math.IsNaN(p.SOTA) || math.IsNaN(p.Fak) || math.IsInf(p.SOTA, 0) || math.IsInf(p.Fak, 0) {
		return out
	}
	ratio := p.Fak / p.SOTA
	if m.Direction == ParityLowerIsBetter {
		ratio = p.SOTA / p.Fak
	}
	out.Ratio = ratio
	switch {
	case ratio > 1+m.Band:
		out.Verdict = ParityAhead
	case ratio < 1-m.Band:
		out.Verdict = ParityBehind
	default:
		out.Verdict = ParityAtParity
	}
	return out
}

// resolveParityPublication derives the public URL, or names why there is none. It
// refuses anything that is not an absolute http(s) base: a relative or scheme-less
// value is a configuration mistake, and turning it into a link would publish a
// URL that does not resolve.
func resolveParityPublication(base, uid string) ParityPublication {
	if base == "" {
		return ParityPublication{
			State:  ParityUnpublished,
			Reason: "no public Grafana base URL configured — set PublicBaseURL once a public host exists",
		}
	}
	if !strings.HasPrefix(base, "http://") && !strings.HasPrefix(base, "https://") {
		return ParityPublication{
			State:  ParityUnpublished,
			Reason: fmt.Sprintf("public base %q is not an absolute http(s) URL", base),
		}
	}
	return ParityPublication{
		State: ParityPublished,
		URL:   strings.TrimSuffix(base, "/") + "/d/" + uid,
	}
}

// OpenMetrics exposes the latest comparison per metric as Prometheus gauge
// families, so a Grafana panel scrapes real numbers instead of tracking a
// hand-copied figure. Panels with no sample are omitted — an absent series is
// honest, a zero-valued one is not.
func (d ParityDashboard) OpenMetrics() []OpenMetricFamily {
	value := OpenMetricFamily{
		Name: "fak_parity_value",
		Help: "latest measured value per parity metric, by side (fak or the SOTA baseline)",
		Type: OpenMetricGauge,
	}
	ratio := OpenMetricFamily{
		Name: "fak_parity_ratio",
		Help: "latest fak-to-SOTA ratio per parity metric, normalized so >1 means fak is ahead",
		Type: OpenMetricGauge,
	}
	for _, p := range d.Panels {
		if p.Latest.Verdict == ParityUnknown {
			continue
		}
		labels := func(side string) []OpenMetricLabel {
			return []OpenMetricLabel{
				{Name: "baseline", Value: p.Latest.Baseline},
				{Name: "metric", Value: p.Metric.Slug},
				{Name: "side", Value: side},
			}
		}
		value.Samples = append(value.Samples,
			OpenMetricSample{Labels: labels("fak"), Value: p.Latest.Fak},
			OpenMetricSample{Labels: labels("sota"), Value: p.Latest.SOTA},
		)
		ratio.Samples = append(ratio.Samples, OpenMetricSample{
			Labels: []OpenMetricLabel{
				{Name: "baseline", Value: p.Latest.Baseline},
				{Name: "metric", Value: p.Metric.Slug},
				{Name: "verdict", Value: string(p.Latest.Verdict)},
			},
			Value: p.Latest.Ratio,
		})
	}
	return []OpenMetricFamily{ratio, value}
}

// Render produces a deterministic operator card: the publication state, then one
// block per metric with the latest SOTA comparison and the full dated history.
// Pure (no clock, no disk) so a test can assert its exact bytes.
func (d ParityDashboard) Render() string {
	var b strings.Builder
	title := d.Title
	if title == "" {
		title = "Parity performance"
	}
	b.WriteString(title)
	b.WriteString(" (")
	b.WriteString(d.UID)
	b.WriteString(")\n")

	b.WriteString("Public URL: ")
	if d.Publication.State == ParityPublished {
		b.WriteString(d.Publication.URL)
	} else {
		b.WriteString("unpublished — ")
		b.WriteString(d.Publication.Reason)
	}
	b.WriteString("\n")

	if len(d.Panels) == 0 {
		b.WriteString("\nno parity metrics declared\n")
		return b.String()
	}

	for _, p := range d.Panels {
		b.WriteString("\n")
		b.WriteString(parityPanelHeader(p))
		b.WriteString("\n")
		if len(p.Points) == 0 {
			b.WriteString("  no samples\n")
			continue
		}
		for _, pt := range p.Points {
			b.WriteString("  ")
			b.WriteString(pt.At.UTC().Format(time.RFC3339))
			b.WriteString("  fak=")
			b.WriteString(formatParityValue(pt.Fak))
			b.WriteString("  sota=")
			b.WriteString(formatParityValue(pt.SOTA))
			if pt.Source != "" {
				b.WriteString("  src=")
				b.WriteString(pt.Source)
			}
			b.WriteString("\n")
		}
	}
	return b.String()
}

// parityPanelHeader is the one-line summary of a panel: the metric, its unit and
// direction, and the latest verdict against the named baseline.
func parityPanelHeader(p ParityPanel) string {
	title := p.Metric.Title
	if title == "" {
		title = p.Metric.Slug
	}
	head := title
	if p.Metric.Unit != "" {
		head += " [" + p.Metric.Unit + "]"
	}
	head += " (" + string(p.Metric.Direction) + ")"
	if p.Latest.Verdict == ParityUnknown {
		return head + ": unknown"
	}
	head += ": " + string(p.Latest.Verdict) + " " + formatParityValue(p.Latest.Ratio) + "x"
	if p.Latest.Baseline != "" {
		head += " vs " + p.Latest.Baseline
	}
	return head
}

// formatParityValue trims a float to at most 4 significant decimals without an
// exponent, so the card stays stable across runs and readable to an operator.
func formatParityValue(v float64) string {
	s := strconv.FormatFloat(v, 'f', 4, 64)
	s = strings.TrimRight(s, "0")
	return strings.TrimSuffix(s, ".")
}
