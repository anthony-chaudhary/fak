package metrics

import (
	"strings"
	"testing"
	"time"
)

func parityAt(day int) time.Time {
	return time.Date(2026, 8, day, 12, 0, 0, 0, time.UTC)
}

func mustParityDashboard(t *testing.T, in ParityDashboardInput) ParityDashboard {
	t.Helper()
	got, err := BuildParityDashboard(in)
	if err != nil {
		t.Fatalf("BuildParityDashboard: %v", err)
	}
	return got
}

// The direction trap: a metric where fak's value is LARGER than the baseline is a
// win for throughput and a loss for latency. A dashboard that ignores direction
// reports the latency regression as a win, which is the whole point of the axis
// declaration.
func TestParityDashboardVerdictIsDirectionAware(t *testing.T) {
	got := mustParityDashboard(t, ParityDashboardInput{
		Title: "SOTA parity",
		UID:   "fak-sota-parity",
		Metrics: []ParityMetric{
			{Slug: "tokens-per-second", Title: "Decode throughput", Unit: "tok/s", Direction: ParityHigherIsBetter, Band: 0.05},
			{Slug: "p95-latency", Title: "p95 latency", Unit: "ms", Direction: ParityLowerIsBetter, Band: 0.05},
		},
		Samples: []ParitySample{
			{Metric: "tokens-per-second", At: parityAt(2), Fak: 120, SOTA: 100, Baseline: "vllm-0.9"},
			{Metric: "p95-latency", At: parityAt(2), Fak: 120, SOTA: 100, Baseline: "vllm-0.9"},
		},
	})

	if got.Schema != ParityDashboardSchema {
		t.Fatalf("schema = %q, want %q", got.Schema, ParityDashboardSchema)
	}
	if len(got.Panels) != 2 {
		t.Fatalf("panels = %d, want 2", len(got.Panels))
	}
	if v := got.Panels[0].Latest.Verdict; v != ParityAhead {
		t.Errorf("higher-is-better 120 vs 100: verdict = %q, want %q", v, ParityAhead)
	}
	if r := got.Panels[0].Latest.Ratio; r != 1.2 {
		t.Errorf("higher-is-better ratio = %v, want 1.2", r)
	}
	if v := got.Panels[1].Latest.Verdict; v != ParityBehind {
		t.Errorf("lower-is-better 120 vs 100: verdict = %q, want %q (a slower fak is not a win)", v, ParityBehind)
	}
	if r := got.Panels[1].Latest.Ratio; r >= 1 {
		t.Errorf("lower-is-better ratio = %v, want <1 so >1 always means fak is ahead", r)
	}
}

// The tolerance band decides at-parity, and the latest point (not the last one
// supplied) drives the verdict — history arrives out of order from a corpus walk.
func TestParityDashboardOrdersHistoryAndUsesLatest(t *testing.T) {
	got := mustParityDashboard(t, ParityDashboardInput{
		UID: "fak-sota-parity",
		Metrics: []ParityMetric{
			{Slug: "pass-rate", Direction: ParityHigherIsBetter, Band: 0.05},
		},
		Samples: []ParitySample{
			{Metric: "pass-rate", At: parityAt(3), Fak: 0.98, SOTA: 1.0, Source: "run-c"},
			{Metric: "pass-rate", At: parityAt(1), Fak: 0.50, SOTA: 1.0, Source: "run-a"},
			{Metric: "pass-rate", At: parityAt(2), Fak: 0.80, SOTA: 1.0, Source: "run-b"},
		},
	})

	points := got.Panels[0].Points
	if len(points) != 3 {
		t.Fatalf("points = %d, want 3", len(points))
	}
	for i, want := range []string{"run-a", "run-b", "run-c"} {
		if points[i].Source != want {
			t.Errorf("point %d source = %q, want %q (history must be ascending by time)", i, points[i].Source, want)
		}
	}
	if got.Panels[0].Latest.Verdict != ParityAtParity {
		t.Errorf("0.98 vs 1.0 inside a 5%% band: verdict = %q, want %q", got.Panels[0].Latest.Verdict, ParityAtParity)
	}
	if !got.Panels[0].Latest.At.Equal(parityAt(3)) {
		t.Errorf("latest.At = %v, want the newest sample %v", got.Panels[0].Latest.At, parityAt(3))
	}
}

// "Public URL" is only ticked by a real absolute base. Everything else reports
// unpublished with a reason — docs/grafana/README.md: no URL here is fabricated.
func TestParityDashboardPublicationNeverFabricatesAURL(t *testing.T) {
	metrics := []ParityMetric{{Slug: "pass-rate"}}

	unset := mustParityDashboard(t, ParityDashboardInput{UID: "fak-sota-parity", Metrics: metrics})
	if unset.Publication.State != ParityUnpublished {
		t.Errorf("no base: state = %q, want %q", unset.Publication.State, ParityUnpublished)
	}
	if unset.Publication.URL != "" {
		t.Errorf("no base: URL = %q, want empty", unset.Publication.URL)
	}
	if unset.Publication.Reason == "" {
		t.Error("no base: want a named reason for the unpublished state")
	}

	relative := mustParityDashboard(t, ParityDashboardInput{
		UID: "fak-sota-parity", Metrics: metrics, PublicBaseURL: "grafana.example.org",
	})
	if relative.Publication.State != ParityUnpublished || relative.Publication.URL != "" {
		t.Errorf("scheme-less base: got %+v, want unpublished with no URL", relative.Publication)
	}

	published := mustParityDashboard(t, ParityDashboardInput{
		UID: "fak-sota-parity", Metrics: metrics, PublicBaseURL: "https://grafana.example.org/",
	})
	if published.Publication.State != ParityPublished {
		t.Fatalf("absolute base: state = %q, want %q", published.Publication.State, ParityPublished)
	}
	if want := "https://grafana.example.org/d/fak-sota-parity"; published.Publication.URL != want {
		t.Errorf("URL = %q, want %q", published.Publication.URL, want)
	}
	if published.Publication.Reason != "" {
		t.Errorf("published: reason = %q, want empty", published.Publication.Reason)
	}
}

func TestParityDashboardRejectsBadInput(t *testing.T) {
	cases := map[string]ParityDashboardInput{
		"no uid": {Metrics: []ParityMetric{{Slug: "a"}}},
		"empty metric slug": {
			UID: "u", Metrics: []ParityMetric{{Slug: "  "}},
		},
		"duplicate metric": {
			UID: "u", Metrics: []ParityMetric{{Slug: "a"}, {Slug: "a"}},
		},
		"unknown direction": {
			UID: "u", Metrics: []ParityMetric{{Slug: "a", Direction: ParityDirection("sideways")}},
		},
		"negative band": {
			UID: "u", Metrics: []ParityMetric{{Slug: "a", Band: -0.1}},
		},
		"sample for undeclared metric": {
			UID: "u", Metrics: []ParityMetric{{Slug: "a"}},
			Samples: []ParitySample{{Metric: "typo", At: parityAt(1), Fak: 1, SOTA: 1}},
		},
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := BuildParityDashboard(in); err == nil {
				t.Fatalf("BuildParityDashboard(%s) = nil error, want a refusal", name)
			}
		})
	}
}

// A missing or zero baseline is reported as unknown, not folded into a fake
// at-parity, and it must not reach the scrape surface as a zero-valued series.
func TestParityDashboardUnknownVerdictStaysOffTheScrape(t *testing.T) {
	got := mustParityDashboard(t, ParityDashboardInput{
		UID: "fak-sota-parity",
		Metrics: []ParityMetric{
			{Slug: "measured", Direction: ParityHigherIsBetter},
			{Slug: "no-baseline", Direction: ParityHigherIsBetter},
			{Slug: "no-samples", Direction: ParityHigherIsBetter},
		},
		Samples: []ParitySample{
			{Metric: "measured", At: parityAt(1), Fak: 2, SOTA: 1, Baseline: "sglang"},
			{Metric: "no-baseline", At: parityAt(1), Fak: 2, SOTA: 0, Baseline: "sglang"},
		},
	})

	for i, want := range []ParityVerdict{ParityAhead, ParityUnknown, ParityUnknown} {
		if v := got.Panels[i].Latest.Verdict; v != want {
			t.Errorf("panel %q verdict = %q, want %q", got.Panels[i].Metric.Slug, v, want)
		}
	}
	if r := got.Panels[1].Latest.Ratio; r != 0 {
		t.Errorf("zero baseline ratio = %v, want 0", r)
	}

	families := got.OpenMetrics()
	text, err := RenderOpenMetricsText(families)
	if err != nil {
		t.Fatalf("RenderOpenMetricsText: %v", err)
	}
	rendered := string(text)
	if !strings.Contains(rendered, `fak_parity_value{baseline="sglang",metric="measured",side="fak"} 2`) {
		t.Errorf("scrape missing the measured fak series:\n%s", rendered)
	}
	if !strings.Contains(rendered, `fak_parity_ratio{baseline="sglang",metric="measured",verdict="ahead"} 2`) {
		t.Errorf("scrape missing the measured ratio series:\n%s", rendered)
	}
	if strings.Contains(rendered, `metric="no-baseline"`) || strings.Contains(rendered, `metric="no-samples"`) {
		t.Errorf("unknown panels must not reach the scrape as zero-valued series:\n%s", rendered)
	}
}

func TestParityDashboardRenderIsDeterministic(t *testing.T) {
	got := mustParityDashboard(t, ParityDashboardInput{
		Title: "SOTA parity",
		UID:   "fak-sota-parity",
		Metrics: []ParityMetric{
			{Slug: "tokens-per-second", Title: "Decode throughput", Unit: "tok/s", Direction: ParityHigherIsBetter, Band: 0.05},
			{Slug: "no-samples", Title: "Cost per task", Unit: "usd", Direction: ParityLowerIsBetter},
		},
		Samples: []ParitySample{
			{Metric: "tokens-per-second", At: parityAt(1), Fak: 80, SOTA: 100, Baseline: "vllm-0.9", Source: "run-a"},
			{Metric: "tokens-per-second", At: parityAt(2), Fak: 120, SOTA: 100, Baseline: "vllm-0.9", Source: "run-b"},
		},
	})

	want := strings.Join([]string{
		"SOTA parity (fak-sota-parity)",
		"Public URL: unpublished — no public Grafana base URL configured — set PublicBaseURL once a public host exists",
		"",
		"Decode throughput [tok/s] (higher-is-better): ahead 1.2x vs vllm-0.9",
		"  2026-08-01T12:00:00Z  fak=80  sota=100  src=run-a",
		"  2026-08-02T12:00:00Z  fak=120  sota=100  src=run-b",
		"",
		"Cost per task [usd] (lower-is-better): unknown",
		"  no samples",
		"",
	}, "\n")

	if diff := got.Render(); diff != want {
		t.Errorf("Render() mismatch\n got:\n%s\nwant:\n%s", diff, want)
	}
}
