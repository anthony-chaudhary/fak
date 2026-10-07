package grafana

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// fakEngineBatchingPath is the hand-authored continuous-batching board (uid
// fak-engine-batching). No generator owns it, so this test is its contract: every
// query reads only series the gateway actually emits, and the board shows both
// decode paths and every non-sample phase of the native serving loop.
const fakEngineBatchingPath = "dashboards/fak-engine-batching.json"

type fakEngineBatchingPanel struct {
	ID      int    `json:"id"`
	Type    string `json:"type"`
	Title   string `json:"title"`
	Targets []struct {
		Expr string `json:"expr"`
	} `json:"targets"`
	Panels []fakEngineBatchingPanel `json:"panels"` // collapsed rows nest their panels
}

type fakEngineBatchingDashboard struct {
	Title  string                   `json:"title"`
	UID    string                   `json:"uid"`
	Panels []fakEngineBatchingPanel `json:"panels"`
}

func loadFakEngineBatching(t *testing.T) fakEngineBatchingDashboard {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Join(filepath.Dir(file), filepath.FromSlash(fakEngineBatchingPath))
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var d fakEngineBatchingDashboard
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("%s is not well-formed dashboard JSON: %v", path, err)
	}
	return d
}

func flattenFakEngineBatchingPanels(ps []fakEngineBatchingPanel) []fakEngineBatchingPanel {
	var out []fakEngineBatchingPanel
	for _, p := range ps {
		out = append(out, p)
		out = append(out, flattenFakEngineBatchingPanels(p.Panels)...)
	}
	return out
}

// fakEngineBatchingAllowedMetrics is the closed set of families the board may
// query: the enginestep families plus the gateway scheduler/serving/inference
// series it explains the cycle against.
func fakEngineBatchingAllowedMetrics() map[string]bool {
	allowed := map[string]bool{}
	for _, m := range enginestep.MetricFamilies {
		allowed[m] = true
	}
	for _, m := range stepobs.MetricFamilies {
		allowed[m] = true
	}
	for _, m := range []string{
		"fak_sched_running", "fak_sched_waiting", "fak_sched_tokens_in_use", "fak_sched_queued_tokens",
		"fak_sched_admitted_total", "fak_sched_queued_total", "fak_sched_shed_total",
		"fak_sched_preempt_total", "fak_sched_preempt_swap_total", "fak_sched_preempt_recompute_total",
		"fak_serving_num_requests_running", "fak_serving_num_requests_waiting",
		"fak_serving_kv_cache_usage_perc", "fak_serving_prefix_cache_hit_rate",
		"fak_gateway_inference_requests_total",
		"fak_gateway_inference_completion_tokens_total", "fak_gateway_inference_prompt_tokens_total",
		"fak_gateway_kv_memory_evictions_total", "fak_gateway_kv_memory_resident_bytes",
		// Serving-latency SLO row.
		"fak_gateway_inference_ttft_seconds", "fak_gateway_inference_tpot_seconds", "fak_gateway_inference_e2e_seconds",
		"fak_gateway_inference_prefill_tokens_per_second", "fak_gateway_inference_decode_tokens_per_second",
		// Live-engine KV cache-event stream (engine.DefaultCacheEvents).
		"fak_engine_cache_events_observed", "fak_engine_cache_hits_total", "fak_engine_cache_misses_total",
		"fak_engine_cache_faults_total", "fak_engine_cache_restore_miss_total", "fak_engine_cache_restore_fault_total",
		"fak_engine_cache_bytes_moved_breakdown_total",
		// OTLP span export.
		"fak_otlp_spans_total", "fak_otlp_queue_depth",
	} {
		allowed[m] = true
	}
	return allowed
}

var (
	fakMetricNameRE   = regexp.MustCompile(`\bfak_[a-zA-Z0-9_]+`)
	quotedRE          = regexp.MustCompile(`"[^"]*"`) // label values (job="fak_gateway") are not metric names
	phaseEqRE         = regexp.MustCompile(`phase="([^"]*)"`)
	phaseReRE         = regexp.MustCompile(`phase=~"([^"]*)"`)
	phaseNeRE         = regexp.MustCompile(`phase!="([^"]*)"`)
	pathEqRE          = regexp.MustCompile(`path="([^"]*)"`)
	byLabelsRE        = regexp.MustCompile(`by\s*\(([^)]*)\)`)
	histogramSuffixes = []string{"_bucket", "_sum", "_count"}
)

func metricAllowed(allowed map[string]bool, name string) bool {
	if allowed[name] {
		return true
	}
	for _, sfx := range histogramSuffixes {
		if base, ok := strings.CutSuffix(name, sfx); ok && allowed[base] {
			return true
		}
	}
	return false
}

func groupsBy(expr, label string) bool {
	for _, m := range byLabelsRE.FindAllStringSubmatch(expr, -1) {
		for _, l := range strings.Split(m[1], ",") {
			if strings.TrimSpace(l) == label {
				return true
			}
		}
	}
	return false
}

func TestFakEngineBatchingDashboardContract(t *testing.T) {
	d := loadFakEngineBatching(t)
	if d.UID != "fak-engine-batching" {
		t.Errorf("expected uid fak-engine-batching, got %q", d.UID)
	}
	if d.Title == "" {
		t.Error("dashboard must carry a title")
	}

	allowed := fakEngineBatchingAllowedMetrics()
	seenIDs := map[int]string{}
	phases := map[string]bool{}
	paths := map[string]bool{}
	exprs := 0
	for _, p := range flattenFakEngineBatchingPanels(d.Panels) {
		if prev, dup := seenIDs[p.ID]; dup {
			t.Errorf("duplicate panel id %d: %q and %q", p.ID, prev, p.Title)
		}
		seenIDs[p.ID] = p.Title
		for _, tgt := range p.Targets {
			expr := tgt.Expr
			if strings.TrimSpace(expr) == "" {
				continue
			}
			exprs++
			for _, name := range fakMetricNameRE.FindAllString(quotedRE.ReplaceAllString(expr, `""`), -1) {
				if !metricAllowed(allowed, name) {
					t.Errorf("panel %d %q queries %s, which is outside the emitted allowlist (expr: %s)", p.ID, p.Title, name, expr)
				}
			}
			if strings.Contains(expr, enginestep.MetricPhaseSeconds) {
				collectPhases(expr, phases)
			}
			if strings.Contains(expr, "fak_engine_decode_") {
				if m := pathEqRE.FindStringSubmatch(expr); m != nil {
					paths[m[1]] = true
				} else if groupsBy(expr, "path") {
					for _, path := range enginestep.Paths {
						paths[path] = true
					}
				}
			}
		}
	}
	if exprs == 0 {
		t.Fatal("dashboard carries no query targets")
	}
	for _, path := range enginestep.Paths {
		if !paths[path] {
			t.Errorf("decode path %q is not visible on any panel", path)
		}
	}
	for _, ph := range enginestep.Phases {
		if ph == enginestep.PhaseSample {
			continue
		}
		if !phases[string(ph)] {
			t.Errorf("phase %q is not visible on any panel", ph)
		}
	}
}

// collectPhases marks the phases an fak_engine_phase_seconds query shows: an
// explicit phase="x", a phase=~"a|b" alternation, or a by(phase) grouping over
// every phase minus any phase!="x" exclusion.
func collectPhases(expr string, phases map[string]bool) {
	for _, m := range phaseEqRE.FindAllStringSubmatch(expr, -1) {
		phases[m[1]] = true
	}
	for _, m := range phaseReRE.FindAllStringSubmatch(expr, -1) {
		for _, alt := range strings.Split(m[1], "|") {
			phases[alt] = true
		}
	}
	if phaseEqRE.MatchString(expr) || phaseReRE.MatchString(expr) || !groupsBy(expr, "phase") {
		return
	}
	excluded := map[string]bool{}
	for _, m := range phaseNeRE.FindAllStringSubmatch(expr, -1) {
		excluded[m[1]] = true
	}
	for _, ph := range enginestep.Phases {
		if !excluded[string(ph)] {
			phases[string(ph)] = true
		}
	}
}
