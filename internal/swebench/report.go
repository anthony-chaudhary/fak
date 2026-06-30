package swebench

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// This file assembles the fak<->bench SWE-bench Verified COMPARISON — fak's
// numbers keyed to bench's own metric vocabulary (the Benchmark repo's
// docs/guides/swebench/reference/critical-metrics.md: an AGENT stream and a
// SERVER stream) so they slot directly beside a bench results_<run_id>.json. The
// four headline families the user asked for:
//
//	1. wall-time + prefill/KV reuse  — fak's prefill work-elimination is a
//	   cache-reuse metric directly comparable to bench's SERVER-stream cache-hit /
//	   KV-reuse (computed deterministically here; live wall-clock on the GPU server).
//	2. turns + tokens                — fak geometry Turns maps to bench's
//	   actual_agent_steps = (len(messages)-2)//2 (AGENT stream).
//	3. in-process adjudication cost  — fak-native; bench has no analog (the
//	   spawn-per-hook vs in-process gate, from internal/bench).
//	4. resolve-rate + safety         — resolved_count / pass_rate_pct (AGENT
//	   stream), gated to a Docker box; safety is fak-native.
//
// Each metric is tagged comparable|fak_native and computed|live|gated so the
// report never passes a deterministic floor off as a measured wall-clock, nor a
// fak-vs-fak ablation off as a fair-comparator win (the METRICS-HONESTY rule).

// Kind classifies whether a metric has a bench analog or is a fak differentiator.
const (
	KindComparable = "comparable" // bench measures an analog; a true head-to-head
	KindFakNative  = "fak_native" // bench has no analog; fak's differentiator
)

// Provenance is the honesty tag on a number's origin.
const (
	ProvComputed = "computed_deterministic" // exact arithmetic, no model (contention-free floor)
	ProvLive     = "live_measured"          // real wall-clock / real run
	ProvGated    = "gated_gpu_server_only"  // needs Docker / a capable model — not produced on this box
)

// Metric is one named number with a unit and an honesty note.
type Metric struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
	Unit  string  `json:"unit"`
	Note  string  `json:"note,omitempty"`
}

// MetricFamily is one of the four headline families.
type MetricFamily struct {
	Name        string   `json:"name"`
	BenchAnalog string   `json:"bench_analog"` // the bench metric this compares to, or a "no analog" note
	Kind        string   `json:"kind"`         // comparable | fak_native
	Provenance  string   `json:"provenance"`   // computed | live | gated
	Metrics     []Metric `json:"metrics"`
}

// AdjCost is the in-process vs spawn-per-hook adjudication gate (family 3),
// supplied by the caller from internal/bench so this package stays decoupled from
// the kernel. Zero p50s mean "not measured this run".
type AdjCost struct {
	InProcessP50Ns int64   `json:"in_process_p50_ns"`
	SpawnHookP50Ns int64   `json:"spawn_hook_p50_ns"`
	SpeedupX       float64 `json:"speedup_x"`
}

// BenchSide is the subset of a bench results_<run_id>.json this comparison reads
// to render a true side-by-side. Fields are best-effort across schema v4..v6.
type BenchSide struct {
	Path             string  `json:"path"`
	SchemaVersion    int     `json:"schema_version"`
	ProfileName      string  `json:"profile_name"`
	TokenHitRatioPct float64 `json:"token_hit_ratio_pct"` // SERVER stream cache hit (cache_verdict.per_server)
	CacheStatus      string  `json:"cache_status"`
	TotalRunSeconds  float64 `json:"total_run_seconds"`
	ResolvedCount    int     `json:"resolved_count,omitempty"` // AGENT stream, if the rollup carried it
	GradedTotal      int     `json:"graded_total,omitempty"`
	PassRatePct      float64 `json:"pass_rate_pct,omitempty"`
	Present          bool    `json:"present"`
}

// Comparison is the whole artifact.
type Comparison struct {
	Schema      string         `json:"schema"`
	GeneratedAt string         `json:"generated_at,omitempty"`
	Dataset     DatasetRef     `json:"dataset"`
	Workers     []int          `json:"workers"`
	Families    []MetricFamily `json:"families"`
	Resolve     *EvalResult    `json:"resolve,omitempty"`
	Bench       *BenchSide     `json:"bench_side,omitempty"`
	Methodology string         `json:"methodology"`
	Honesty     []string       `json:"honesty"`
}

// DatasetRef is the lightweight dataset descriptor embedded in the comparison.
type DatasetRef struct {
	Instances      int            `json:"instances"`
	DifficultyDist map[string]int `json:"difficulty_distribution"`
	GeometrySource map[string]int `json:"geometry_sources"`
	TotalTurns     int64          `json:"total_turns"`
}

// CompareInputs is everything BuildComparison needs; optional fields may be nil.
type CompareInputs struct {
	Dataset      *Dataset
	Geometry     GeometryModel
	Workers      []int
	Eval         *EvalResult // resolve family (gated)
	Adjudication *AdjCost    // adjudication family (from internal/bench)
	BenchResult  string      // optional path to a bench results_*.json
	GeneratedAt  string      // caller-supplied timestamp (the pkg takes no clock)
}

// BuildComparison assembles the four-family comparison from whatever inputs are
// present. It always produces the deterministic families (1 and 2); families 3
// and 4 are populated when their inputs are supplied, else marked gated.
func BuildComparison(in CompareInputs) Comparison {
	workers := in.Workers
	if len(workers) == 0 {
		workers = []int{1, 2, 4, 8}
	}
	summary := Describe(in.Dataset, in.Geometry, workers)

	c := Comparison{
		Schema:      "fak-swebench-compare/1",
		GeneratedAt: in.GeneratedAt,
		Workers:     workers,
		Dataset: DatasetRef{
			Instances:      summary.Instances,
			DifficultyDist: summary.DifficultyDist,
			GeometrySource: summary.GeometrySources,
			TotalTurns:     summary.TotalTurns,
		},
		Methodology: "fak metrics keyed to bench's critical-metrics vocabulary (agent stream + server stream). " +
			"Prefill work-elimination is the deterministic, contention-free floor (sessionbench A/B/C formula) over the real " +
			"instance set — comparable to bench's server-side cache-hit/KV-reuse. Turns map to mini-swe-agent actual_agent_steps. " +
			"Resolve-rate uses the official harness (gated to a Docker box). Adjudication cost is the in-process vs spawn-per-hook gate.",
	}

	// Family 1 — prefill / KV-reuse work-elimination (fak-native: these are fak's
	// OWN ablation arms — naive-re-prefill vs per-agent-KV vs fak-fused — NOT a
	// head-to-head vs a tuned SGLang/llama server, which also reuses a shared prefix
	// under seq_cp/kv_unified. Thematically the cache-reuse story bench's server-side
	// token_hit_ratio also tracks, but a different quantity (an unbounded work-ratio,
	// not a bounded 0-100% hit %). Live TTFT/wall-clock is the gated GPU-server number.).
	fam1 := MetricFamily{
		Name:        "prefill / KV-reuse work-elimination (deterministic)",
		BenchAnalog: "related to (NOT the same quantity as) bench's server-stream cache-hit/KV-reuse; live TTFT/wall-clock is the gated GPU-server number, not produced here",
		Kind:        KindFakNative,
		Provenance:  ProvComputed,
	}
	for _, p := range summary.Prefill {
		fam1.Metrics = append(fam1.Metrics,
			Metric{Name: fmt.Sprintf("prefill_work_elimination_A_over_C@workers=%d", p.Workers), Value: round2(p.AOverC), Unit: "x", Note: "fak's naive-re-prefill arm (A) vs fak-fused arm (C) — a fak-vs-fak ablation, not a fair-comparator win"},
			Metric{Name: fmt.Sprintf("cross_worker_prefix_reuse_B_over_C@workers=%d", p.Workers), Value: round2(p.BOverC), Unit: "x", Note: "fak's per-agent-KV arm (B) vs fak-fused shared-prefix arm (C) — a deterministic work-ratio, NOT bench's bounded token_hit_ratio_pct and NOT a head-to-head"},
		)
	}
	c.Families = append(c.Families, fam1)

	// Family 2 — turns + tokens (comparable to agent-stream actual_agent_steps).
	fam2 := MetricFamily{
		Name:        "turns + tokens",
		BenchAnalog: "agent-stream actual_agent_steps = (len(messages)-2)//2; prompt+completion tokens",
		Kind:        KindComparable,
		Provenance:  ProvComputed,
		Metrics: []Metric{
			{Name: "total_agent_steps", Value: float64(summary.TotalTurns), Unit: "steps", Note: "Σ geometry turns ≈ Σ actual_agent_steps over the set"},
			{Name: "median_agent_steps", Value: float64(summary.TurnsMedian), Unit: "steps"},
			{Name: "turn_tax_A_over_B", Value: round2(firstTurnTax(summary)), Unit: "x", Note: "extra prefill work a re-prefill harness pays per turn vs KV persistence (worker-independent)"},
		},
	}
	c.Families = append(c.Families, fam2)

	// Family 3 — in-process adjudication cost (fak-native; bench has no analog).
	fam3 := MetricFamily{
		Name:        "in-process adjudication cost",
		BenchAnalog: "none — bench measures a served endpoint, not the tool-call adjudication boundary",
		Kind:        KindFakNative,
		Provenance:  ProvGated,
	}
	if in.Adjudication != nil && in.Adjudication.SpawnHookP50Ns > 0 {
		fam3.Provenance = ProvLive
		fam3.Metrics = []Metric{
			{Name: "in_process_adjudication_p50", Value: float64(in.Adjudication.InProcessP50Ns), Unit: "ns"},
			{Name: "spawn_per_hook_p50", Value: float64(in.Adjudication.SpawnHookP50Ns), Unit: "ns"},
			{Name: "fusion_speedup", Value: round2(in.Adjudication.SpeedupX), Unit: "x", Note: "in-process vs the deployed process-per-hook status quo"},
		}
	} else {
		fam3.Metrics = []Metric{{Name: "in_process_adjudication", Value: 0, Unit: "ns", Note: "run `fak bench --suite tau2-smoke` to measure; not folded this run"}}
	}
	c.Families = append(c.Families, fam3)

	// Family 4 — resolve-rate + safety (agent-stream resolved_count; gated).
	fam4 := MetricFamily{
		Name:        "resolve-rate + safety",
		BenchAnalog: "agent-stream resolved_count / pass_rate_pct (report.json); safety is fak-native",
		Kind:        KindComparable,
		Provenance:  ProvGated,
	}
	if in.Eval != nil && in.Eval.Available {
		fam4.Provenance = ProvLive
		fam4.Metrics = []Metric{
			{Name: "resolved_count", Value: float64(in.Eval.Resolved), Unit: "instances"},
			{Name: "graded_total", Value: float64(in.Eval.Total), Unit: "instances"},
			{Name: "pass_rate_pct", Value: round2(in.Eval.ResolveRatePct), Unit: "%"},
		}
		c.Resolve = in.Eval
	} else {
		reason := "no predictions graded — resolve-rate needs a capable model + Docker (GPU server); ~0 with the local 135M model"
		if in.Eval != nil && in.Eval.Reason != "" {
			reason = in.Eval.Reason
		}
		fam4.Metrics = []Metric{{Name: "pass_rate_pct", Value: 0, Unit: "%", Note: reason}}
		if in.Eval != nil {
			c.Resolve = in.Eval
		}
	}
	c.Families = append(c.Families, fam4)

	// Optional: parse a bench results_*.json for the true side-by-side.
	if in.BenchResult != "" {
		if bs, err := ParseBenchResult(in.BenchResult); err == nil {
			c.Bench = bs
		} else {
			c.Bench = &BenchSide{Path: in.BenchResult, Present: false}
		}
	}

	c.Honesty = []string{
		"prefill work-elimination is a deterministic floor (exact token arithmetic), NOT a measured wall-clock — live timing is the GPU server headline",
		"cross-worker reuse (B/C) is the value-stack lever; the A/C and turn-tax (A/B) numbers are fak-vs-harness-arms, reported as such, not as a tuned-SGLang head-to-head",
		"resolve-rate is ~0 with the local 135M model; the real resolve number comes from a Qwen3.6-27B-class model on the GPU server via the same harness",
		"to be scraped by bench like SGLang, fak serve needs a Prometheus /metrics route exposing kernel.Counters() — gap noted, not yet shipped",
	}
	return c
}

// ParseBenchResult reads the subset of a bench results_<run_id>.json this report
// renders. Tolerant across schema v4..v6 (the loader accepts 2..6).
func ParseBenchResult(path string) (*BenchSide, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	bs := &BenchSide{Path: path, Present: true}
	intField(raw, "schema_version", &bs.SchemaVersion)
	strField(raw, "profile_name", &bs.ProfileName)

	// total_run_time.seconds
	if tr, ok := raw["total_run_time"]; ok {
		var m map[string]json.RawMessage
		if json.Unmarshal(tr, &m) == nil {
			floatField(m, "seconds", &bs.TotalRunSeconds)
		}
	}
	// cache_verdict.per_server.<srv>.token_hit_ratio_pct + status
	if cv, ok := raw["cache_verdict"]; ok {
		var m map[string]json.RawMessage
		if json.Unmarshal(cv, &m) == nil {
			strField(m, "status", &bs.CacheStatus)
			if ps, ok := m["per_server"]; ok {
				var servers map[string]map[string]json.RawMessage
				if json.Unmarshal(ps, &servers) == nil && len(servers) > 0 {
					// Deterministic across runs: pick the lowest server key, not an
					// arbitrary map-iteration order (Go randomizes it).
					names := make([]string, 0, len(servers))
					for name := range servers {
						names = append(names, name)
					}
					sort.Strings(names)
					if r, ok := servers[names[0]]["token_hit_ratio_pct"]; ok {
						_ = json.Unmarshal(r, &bs.TokenHitRatioPct)
					}
				}
			}
		}
	}
	return bs, nil
}

// RenderMarkdown renders the comparison as a house-style results table.
func RenderMarkdown(c Comparison) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# fak ↔ bench — SWE-bench Verified comparison\n\n")
	if c.GeneratedAt != "" {
		fmt.Fprintf(&b, "_generated %s_\n\n", c.GeneratedAt)
	}
	fmt.Fprintf(&b, "**Instances:** %d  ·  **worker sweep:** %s  ·  **difficulty:** %s\n\n",
		c.Dataset.Instances, intsJoin(c.Workers), countsJoin(c.Dataset.DifficultyDist))
	fmt.Fprintf(&b, "Geometry provenance: %s. Σ agent steps: %d.\n\n", countsJoin(c.Dataset.GeometrySource), c.Dataset.TotalTurns)

	fmt.Fprintf(&b, "## The four metric families (fak, keyed to bench's vocabulary)\n\n")
	fmt.Fprintf(&b, "| family | vs bench | provenance | headline |\n|---|---|---|---|\n")
	for _, f := range c.Families {
		head := ""
		if len(f.Metrics) > 0 {
			m := pickHeadline(f)
			head = fmt.Sprintf("%s = %s%s", m.Name, trimNum(m.Value), m.Unit)
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", f.Name, kindLabel(f.Kind), provLabel(f.Provenance), head)
	}
	b.WriteString("\n")

	for _, f := range c.Families {
		fmt.Fprintf(&b, "### %s\n_bench analog: %s — %s, %s_\n\n", f.Name, f.BenchAnalog, kindLabel(f.Kind), provLabel(f.Provenance))
		for _, m := range f.Metrics {
			note := ""
			if m.Note != "" {
				note = "  — " + m.Note
			}
			fmt.Fprintf(&b, "- `%s` = **%s%s**%s\n", m.Name, trimNum(m.Value), m.Unit, note)
		}
		b.WriteString("\n")
	}

	if c.Bench != nil && c.Bench.Present {
		fmt.Fprintf(&b, "## Side-by-side vs the bench run\n\n")
		fmt.Fprintf(&b, "bench `%s` (schema v%d): cache status **%s**, token-hit-ratio **%.1f%%** (server-side, bounded 0–100%%), run %.0fs.\n",
			c.Bench.ProfileName, c.Bench.SchemaVersion, c.Bench.CacheStatus, c.Bench.TokenHitRatioPct, c.Bench.TotalRunSeconds)
		fmt.Fprintf(&b, "fak's cross-worker prefix-reuse (B/C above) is a *related but distinct* quantity — a deterministic fak-vs-fak work-ratio (≥1), not the server's measured hit fraction. The two track the same theme (don't redo prefill) but are not the same number.\n\n")
	}

	fmt.Fprintf(&b, "## Honesty\n\n")
	for _, h := range c.Honesty {
		fmt.Fprintf(&b, "- %s\n", h)
	}
	return b.String()
}

// ---- helpers ----

func round2(f float64) float64 {
	return float64(int64(f*100+0.5)) / 100
}

func firstTurnTax(s Summary) float64 {
	if len(s.Prefill) > 0 {
		return s.Prefill[0].AOverB
	}
	return 0
}

func pickHeadline(f MetricFamily) Metric {
	// prefer an "x" ratio, else the first metric
	for _, m := range f.Metrics {
		if m.Unit == "x" {
			return m
		}
	}
	return f.Metrics[0]
}

func kindLabel(k string) string {
	if k == KindFakNative {
		return "fak-native"
	}
	return "comparable"
}

func provLabel(p string) string {
	switch p {
	case ProvComputed:
		return "computed (deterministic)"
	case ProvLive:
		return "live (measured)"
	case ProvGated:
		return "gated (GPU-server-only)"
	}
	return p
}

func trimNum(f float64) string {
	s := fmt.Sprintf("%.2f", f)
	s = strings.TrimRight(strings.TrimRight(s, "0"), ".")
	return s
}

func intsJoin(xs []int) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = fmt.Sprintf("%d", x)
	}
	return strings.Join(parts, ",")
}

func countsJoin(m map[string]int) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, " ")
}

func intField(m map[string]json.RawMessage, k string, dst *int) {
	if r, ok := m[k]; ok {
		_ = json.Unmarshal(r, dst)
	}
}
func strField(m map[string]json.RawMessage, k string, dst *string) {
	if r, ok := m[k]; ok {
		_ = json.Unmarshal(r, dst)
	}
}
func floatField(m map[string]json.RawMessage, k string, dst *float64) {
	if r, ok := m[k]; ok {
		_ = json.Unmarshal(r, dst)
	}
}
