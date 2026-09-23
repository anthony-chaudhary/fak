package grafana

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakOpsStatusesPath is the hand-authored Ops Plane Statuses dashboard
// (uid fak-ops-statuses) rendered by the fak_ops exporter scrape job. No
// generator owns this artifact, so this test is its contract: it pins the
// population/time-scope wording operators rely on when reading the gauges.
const fakOpsStatusesPath = "dashboards/fak-ops-statuses.json"

// fakOpsPanel is the subset of the Grafana panel schema these contracts read.
type fakOpsPanel struct {
	ID          int    `json:"id"`
	Type        string `json:"type"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Targets     []struct {
		Expr string `json:"expr"`
	} `json:"targets"`
}

type fakOpsDashboard struct {
	Title  string         `json:"title"`
	UID    string         `json:"uid"`
	Panels []fakOpsPanel  `json:"panels"`
	Links  []interface{}  `json:"links"`
	Time   map[string]any `json:"time"`
}

// loadFakOpsStatuses parses the dashboard relative to this test file so the
// contract holds regardless of the test's working directory.
func loadFakOpsStatuses(t *testing.T) fakOpsDashboard {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Join(filepath.Dir(file), filepath.FromSlash(fakOpsStatusesPath))
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("dashboard artifact missing at %s: %v", path, err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	// json.Unmarshal is strict about well-formed JSON and rejects the BOM-less
	// mojibake-free document this test requires; a parse error fails here.
	var d fakOpsDashboard
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("%s is not well-formed dashboard JSON: %v", path, err)
	}
	if d.UID != "fak-ops-statuses" {
		t.Fatalf("expected uid fak-ops-statuses, got %q", d.UID)
	}
	return d
}

// panelID indexes panels by their stable Grafana id.
func panelID(t *testing.T, d fakOpsDashboard, id int) fakOpsPanel {
	t.Helper()
	for _, p := range d.Panels {
		if p.ID == id {
			return p
		}
	}
	t.Fatalf("panel id %d not found in %s", id, fakOpsStatusesPath)
	return fakOpsPanel{}
}

// panelText is the searchable population-labeling text of a panel: title plus
// description. Titles are included because operators read them first.
func panelText(p fakOpsPanel) string {
	return strings.ToLower(p.Title + "\n" + p.Description)
}

func mustContain(t *testing.T, what, haystack string, needles ...string) {
	t.Helper()
	for _, n := range needles {
		if !strings.Contains(haystack, strings.ToLower(n)) {
			t.Errorf("%s must mention %q; got text:\n%s", what, n, haystack)
		}
	}
}

func mustNotContain(t *testing.T, what, haystack string, banned ...string) {
	t.Helper()
	for _, b := range banned {
		if strings.Contains(haystack, strings.ToLower(b)) {
			t.Errorf("%s must not claim %q; got text:\n%s", what, b, haystack)
		}
	}
}

// deniesRecompute reports whether text negates a graph-range recomputation of
// the exporter's report fold. Wording varies across panels ("not recompute",
// "not a ratio recomputed", "does NOT recompute"), so match the concept.
func deniesRecompute(text string) bool {
	for _, neg := range []string{"not recompute", "not recomputed", "not a ratio recomputed", "does not recompute", "never recompute"} {
		if strings.Contains(text, neg) {
			return true
		}
	}
	return false
}

// TestFakOpsStatusesPopulationLabels is the scoped witness for #13391: every
// panel that reports a denominator, an inventory, or a stuck census must state
// the population it measures.
func TestFakOpsStatusesPopulationLabels(t *testing.T) {
	d := loadFakOpsStatuses(t)

	t.Run("report panels disclose the exporter-configured window", func(t *testing.T) {
		// Panels 7, 16, 30 report the failure ratio/outcomes fold. Their time
		// scope is the exporter's configured trailing window (default 24h),
		// which Grafana's graph range does NOT recompute. Panels 14/15 are
		// counters whose increase() IS computed over the graph range, so they
		// must say so instead of inheriting the report-window wording.
		for _, id := range []int{7, 16, 30} {
			p := panelID(t, d, id)
			text := panelText(p)
			mustContain(t, "report panel "+p.Title,
				text,
				"exporter-configured report window",
				"default 24h",
			)
			// The wording must stay truthful for a custom exporter window:
			// "default" plus "configured" avoids implying 24h is fixed.
			mustNotContain(t, "report panel "+p.Title,
				text,
				"fixed 24h",
				"always 24h",
			)
			// A graph-range selection must never be described as controlling
			// the report window: the text must negate recomputation.
			if !deniesRecompute(text) {
				t.Errorf("report panel %q must state Grafana's graph range does not recompute the report window; got:\n%s", p.Title, text)
			}
		}
		// The rate()/increase() panels must attribute their window to the
		// graph range, the opposite direction of the report panels.
		for _, id := range []int{14, 15} {
			p := panelID(t, d, id)
			text := panelText(p)
			mustContain(t, "counter panel "+p.Title,
				text,
				"grafana graph range",
				"not over the exporter-configured report window",
			)
		}
		// The failure denominator must exclude unknown outcomes.
		rateText := panelText(panelID(t, d, 7))
		mustContain(t, "panel 7", rateText,
			"unknown outcomes are excluded",
		)
	})

	t.Run("inventory panels describe retained snapshots", func(t *testing.T) {
		// Panels 10-12 (and the routines-enabled stat, 4) are held state
		// values re-read each scrape, not events inside the graph range.
		for _, id := range []int{4, 10, 11, 12} {
			p := panelID(t, d, id)
			text := panelText(p)
			mustContain(t, "inventory panel "+p.Title,
				text,
				"retained snapshot",
			)
		}
		for _, id := range []int{10, 11, 12} {
			mustContain(t, "inventory panel "+panelID(t, d, id).Title,
				panelText(panelID(t, d, id)),
				"re-read each scrape",
			)
		}
	})

	t.Run("retired is decommissioned, not failed or retried", func(t *testing.T) {
		// Both places that name routine lifecycle state must define retired.
		for _, id := range []int{4, 10} {
			p := panelID(t, d, id)
			text := panelText(p)
			mustContain(t, "routine-state panel "+p.Title,
				text,
				"retired",
				"decommissioned by design",
			)
			// The wording may vary, but it must deny both inversions: retired is
			// not a failure, and it is not retried.
			if !(strings.Contains(text, "not a failure") || strings.Contains(text, "not failed")) {
				t.Errorf("routine-state panel %q must state retired is not a failure; got:\n%s", p.Title, text)
			}
			if !strings.Contains(text, "not being retried") && !strings.Contains(text, "not failed or retried") {
				t.Errorf("routine-state panel %q must state retired is not retried; got:\n%s", p.Title, text)
			}
			// Guard the specific inversion this issue exists to prevent.
			mustNotContain(t, "routine-state panel "+p.Title, text,
				"retired means failed",
				"retired is a failure",
				"retired routines are retried",
			)
		}
	})

	t.Run("partial and failed-plane indicators stay visible", func(t *testing.T) {
		// The three partial/failed surfaces must remain panels with targets.
		for _, id := range []int{8, 22, 41} {
			p := panelID(t, d, id)
			if len(p.Targets) == 0 || strings.TrimSpace(p.Targets[0].Expr) == "" {
				t.Errorf("partial/failed indicator panel id %d %q must keep a query", id, p.Title)
			}
			if !strings.Contains(strings.ToLower(p.Description), "partial") {
				t.Errorf("panel id %d %q must describe its partial semantics; got %q", id, p.Title, p.Description)
			}
		}
	})

	t.Run("stuck counts warn about incompleteness and overlapping reasons", func(t *testing.T) {
		// Aggregate stuck panels must warn the count can be incomplete...
		for _, id := range []int{6, 19, 22} {
			mustContain(t, "stuck aggregate panel "+panelID(t, d, id).Title,
				panelText(panelID(t, d, id)),
				"incomplete",
			)
		}
		// ...and reason-partitioned panels must warn reasons can overlap.
		for _, id := range []int{6, 19, 20} {
			mustContain(t, "stuck reason panel "+panelID(t, d, id).Title,
				panelText(panelID(t, d, id)),
				"overlap",
			)
		}
	})

	t.Run("existing queries are unchanged and rate/increase never touch a gauge", func(t *testing.T) {
		// Metric contracts are frozen by this issue: spot-check the exact
		// expressions the repro inspected (panels 7 and 10-16).
		want := map[int]string{
			7:  `fak_ops_reports_failure_rate{job="fak_ops"}`,
			10: `sum by (state) (fak_ops_routines{job="fak_ops"})`,
			11: `sum by (state) (fak_ops_items{job="fak_ops"})`,
			12: `sum by (state) (fak_ops_intervals{job="fak_ops"})`,
			14: `increase(fak_ops_reports_intervals_total{job="fak_ops"}[$__rate_interval])`,
			15: `increase(fak_ops_reports_items_admitted_total{job="fak_ops"}[$__rate_interval])`,
			16: `fak_ops_reports_results{job="fak_ops"}`,
		}
		for id, expr := range want {
			p := panelID(t, d, id)
			if len(p.Targets) == 0 {
				t.Fatalf("panel id %d %q lost its target", id, p.Title)
			}
			if got := p.Targets[0].Expr; got != expr {
				t.Errorf("panel id %d %q query changed: got %q want %q", id, p.Title, got, expr)
			}
		}
		// Gold-plating boundary: rate()/increase() must never be applied to a
		// retained-state gauge. Only the two _total counters may use increase().
		for _, p := range d.Panels {
			for _, tgt := range p.Targets {
				if !strings.Contains(tgt.Expr, "increase(") && !strings.Contains(tgt.Expr, "rate(") {
					continue
				}
				if !strings.Contains(tgt.Expr, "_total{") {
					t.Errorf("panel %d %q applies rate/increase to a non-counter (gauge) series: %q",
						p.ID, p.Title, tgt.Expr)
				}
			}
		}
	})
}

// TestDockerComposeDefaultsLightTheme pins the light-theme default for every
// Grafana surface provisioned by this stack. Read-heavy fleet dashboards are
// meant to be legible on white backgrounds; the theme is an instance default
// (GF_USERS_DEFAULT_THEME, the env alias of [users] default_theme), not a
// dashboard-JSON field, so the compose file is where it must be asserted.
// Anonymous Viewers inherit it too — verified against grafana/grafana:11.5.2.
func TestDockerComposeDefaultsLightTheme(t *testing.T) {
	raw, err := os.ReadFile("docker-compose.yml")
	if err != nil {
		t.Fatalf("read docker-compose.yml: %v", err)
	}
	compose := string(raw)
	if !strings.Contains(compose, "GF_USERS_DEFAULT_THEME=light") {
		t.Error("docker-compose.yml must set GF_USERS_DEFAULT_THEME=light so dashboards default to the light theme")
	}
	if strings.Contains(compose, "GF_USERS_DEFAULT_THEME=dark") {
		t.Error("docker-compose.yml must not force the dark theme")
	}
}
