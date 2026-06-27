package gateway

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// This file instruments the ONE-TIME boot of a fak gateway: the cost of every
// phase between process start and "ready to serve", plus an optional model-load
// profile when the host eagerly loads weights at boot (fak serve --gguf). A
// time-series scrape model has no native notion of a one-shot event, so the boot
// timeline is held as GAUGES — measured once as the gateway comes up, then served
// unchanged for the life of the process. That lets a Grafana panel show "what did
// this process's startup cost" at any later moment, not only in the first scrape
// window after boot (which a counter/rate would miss entirely on a fast restart).

// StartupPhase is one completed stage of process boot. Phases the host CLI timed
// BEFORE gateway.New (e.g. loading the capability-floor policy) are passed in via
// Config.StartupPhases; the gateway appends the phases it can time itself.
type StartupPhase struct {
	// Name is the phase label exposed as the {phase="..."} metric dimension.
	Name string
	// Dur is the wall-clock the phase took.
	Dur time.Duration
}

// startupProfile records the boot timeline: the process start instant, the per-
// phase costs observed while coming up, and the instant the gateway became able to
// serve. It is read at scrape time by renderMetrics.
type startupProfile struct {
	mu     sync.Mutex
	start  time.Time
	ready  time.Time
	phases []StartupPhase
}

func newStartupProfile(start time.Time) *startupProfile {
	if start.IsZero() {
		start = time.Now()
	}
	return &startupProfile{start: start}
}

// phase records a completed boot phase. A zero or negative duration is still
// recorded (a phase that ran is worth showing even at sub-microsecond cost).
func (p *startupProfile) phase(name string, dur time.Duration) {
	if p == nil || name == "" {
		return
	}
	p.mu.Lock()
	p.phases = append(p.phases, StartupPhase{Name: name, Dur: dur})
	p.mu.Unlock()
}

// markReady stamps the instant the gateway became able to serve. The FIRST call
// wins: a listener that restarts (ServeStdio after a re-bind, a test re-serving the
// same Server) must not move the boot mark, which is a property of process start.
func (p *startupProfile) markReady(now time.Time) {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.ready.IsZero() {
		p.ready = now
	}
	p.mu.Unlock()
}

type startupSnapshot struct {
	start  time.Time
	ready  time.Time
	phases []StartupPhase
}

func (p *startupProfile) snapshot() startupSnapshot {
	if p == nil {
		return startupSnapshot{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return startupSnapshot{
		start:  p.start,
		ready:  p.ready,
		phases: append([]StartupPhase(nil), p.phases...),
	}
}

// timeToReady is the total boot wall-clock (start -> ready), or 0 until ready.
func (s startupSnapshot) timeToReady() float64 {
	if s.ready.IsZero() || s.start.IsZero() || s.ready.Before(s.start) {
		return 0
	}
	if s.ready.Equal(s.start) {
		return time.Nanosecond.Seconds()
	}
	return s.ready.Sub(s.start).Seconds()
}

// ModelLoadPhase is one aggregate phase of weight loading, surfaced from the GGUF
// load profiler (internal/ggufload.LoadPhaseStat) by the host when it eagerly loads
// a model at boot. It is the gateway's import-decoupled mirror of that struct.
type ModelLoadPhase struct {
	Phase   string
	Seconds float64
	Bytes   int64
	Tensors int
}

// ModelLoadPath is one per-quant-type row of the boot-time load-path breakdown: how many
// tensors + on-disk bytes of a GGUF quant type took the raw-RESIDENT fast path vs paid the
// f32 DEQUANT round-trip, split by expert-vs-dense. It surfaces (on /metrics) which
// mixed-quant weights still pay the slow load — the GLM-5.2 load-time diagnosis made legible
// without an external gguf-dump.
type ModelLoadPath struct {
	QuantType       string
	Expert          bool
	ResidentTensors int
	ResidentBytes   int64
	DequantTensors  int
	DequantBytes    int64
}

// ModelLoadMemoryDemand is one classed row from the memory plan used to admit a
// boot-time model load. Detail is safe for /debug/vars, but Prometheus aggregates
// by class+scope to avoid high-cardinality tensor/path labels.
type ModelLoadMemoryDemand struct {
	Class  string
	Scope  string
	Bytes  int64
	Detail string
	DType  string
}

// ModelLoadMemoryCapacity is the backend capacity snapshot used with the memory
// plan. Unknown capacity is represented explicitly so a scrape can distinguish
// "not probed" from "zero bytes".
type ModelLoadMemoryCapacity struct {
	Scope      string
	TotalBytes int64
	FreeBytes  int64
	Known      bool
	FreeKnown  bool
}

// ModelLoadProfile is the boot-time weight-load breakdown the dashboard renders.
// nil (the default for a mock / proxy serve that loads no weights) suppresses every
// fak_model_load_* metric entirely, so an empty series never masquerades as a 0ms
// load.
type ModelLoadProfile struct {
	Source              string
	Mode                string
	TotalSeconds        float64
	Bytes               int64
	Tensors             int
	Bottleneck          string
	Phases              []ModelLoadPhase
	LoadPaths           []ModelLoadPath
	MemoryPlan          []ModelLoadMemoryDemand
	MemoryCapacities    []ModelLoadMemoryCapacity
	MemoryHeadroomRatio float64
}

func (p *ModelLoadProfile) clone() *ModelLoadProfile {
	if p == nil {
		return nil
	}
	out := *p
	out.Phases = append([]ModelLoadPhase(nil), p.Phases...)
	out.LoadPaths = append([]ModelLoadPath(nil), p.LoadPaths...)
	out.MemoryPlan = append([]ModelLoadMemoryDemand(nil), p.MemoryPlan...)
	out.MemoryCapacities = append([]ModelLoadMemoryCapacity(nil), p.MemoryCapacities...)
	return &out
}

// sorted returns the phases ordered by descending cost so the exposition (and the
// bar-gauge that reads it) leads with the bottleneck.
func (p *ModelLoadProfile) sorted() []ModelLoadPhase {
	if p == nil {
		return nil
	}
	out := append([]ModelLoadPhase(nil), p.Phases...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Seconds > out[j].Seconds })
	return out
}

func (p *ModelLoadProfile) memoryPlanByClassScope() []ModelLoadMemoryDemand {
	if p == nil {
		return nil
	}
	type key struct {
		class string
		scope string
	}
	by := map[key]int64{}
	for _, row := range p.MemoryPlan {
		if row.Bytes <= 0 {
			continue
		}
		k := key{class: modelLoadClass(row.Class), scope: modelLoadScope(row.Scope)}
		by[k] += row.Bytes
	}
	out := make([]ModelLoadMemoryDemand, 0, len(by))
	for k, bytes := range by {
		out = append(out, ModelLoadMemoryDemand{Class: k.class, Scope: k.scope, Bytes: bytes})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Scope != out[j].Scope {
			return out[i].Scope < out[j].Scope
		}
		return out[i].Class < out[j].Class
	})
	return out
}

func (p *ModelLoadProfile) memoryPlanByClassScopeDType() []ModelLoadMemoryDemand {
	if p == nil {
		return nil
	}
	type key struct {
		class string
		scope string
		dtype string
	}
	by := map[key]int64{}
	for _, row := range p.MemoryPlan {
		if row.Bytes <= 0 {
			continue
		}
		k := key{class: modelLoadClass(row.Class), scope: modelLoadScope(row.Scope), dtype: modelLoadDType(row.DType)}
		by[k] += row.Bytes
	}
	out := make([]ModelLoadMemoryDemand, 0, len(by))
	for k, bytes := range by {
		out = append(out, ModelLoadMemoryDemand{Class: k.class, Scope: k.scope, DType: k.dtype, Bytes: bytes})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Scope != out[j].Scope {
			return out[i].Scope < out[j].Scope
		}
		if out[i].Class != out[j].Class {
			return out[i].Class < out[j].Class
		}
		return out[i].DType < out[j].DType
	})
	return out
}

func (p *ModelLoadProfile) sortedMemoryCapacities() []ModelLoadMemoryCapacity {
	if p == nil {
		return nil
	}
	out := append([]ModelLoadMemoryCapacity(nil), p.MemoryCapacities...)
	for i := range out {
		out[i].Scope = modelLoadScope(out[i].Scope)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Scope < out[j].Scope })
	return out
}

func modelLoadClass(class string) string {
	class = strings.TrimSpace(class)
	if class == "" {
		return "unknown"
	}
	return class
}

func modelLoadScope(scope string) string {
	scope = strings.TrimSpace(scope)
	if scope == "" {
		return "device"
	}
	return scope
}

func modelLoadDType(dtype string) string {
	dtype = strings.ToLower(strings.TrimSpace(dtype))
	if dtype == "" {
		return "unknown"
	}
	return dtype
}

func modelLoadPathClass(expert bool) string {
	if expert {
		return "expert"
	}
	return "dense"
}

// writeStartupMetrics renders the boot-timeline gauges. They are emitted on every
// scrape with their once-at-boot values so a Grafana panel can show this process's
// startup cost at any later moment, not only in the scrape window right after boot.
func (s *Server) writeStartupMetrics(b *strings.Builder) {
	snap := s.startup.snapshot()

	writeHelpType(b, "fak_gateway_time_to_ready_seconds", "Total wall-clock from process start to the gateway becoming ready to serve (0 until ready).", "gauge")
	fmt.Fprintf(b, "fak_gateway_time_to_ready_seconds %s\n", promFloat(snap.timeToReady()))

	writeHelpType(b, "fak_gateway_ready_time_seconds", "Unix instant the gateway became ready to serve (0 until ready).", "gauge")
	ready := int64(0)
	if !snap.ready.IsZero() {
		ready = snap.ready.Unix()
	}
	fmt.Fprintf(b, "fak_gateway_ready_time_seconds %d\n", ready)

	// Per-phase boot costs. Aggregate by name so a phase recorded more than once
	// (defensive) sums rather than emitting a duplicate series. The running total
	// is also kept so the unaccounted-boot gauge (below) can report how much of
	// time-to-ready the named phases do NOT explain.
	writeHelpType(b, "fak_gateway_startup_phase_duration_seconds", "Wall-clock cost of each fak gateway boot phase (flag-parse, policy-load, planner-init, vdso-config, kernel-init, listener-bind, and model-load with --gguf).", "gauge")
	sums := map[string]float64{}
	order := make([]string, 0, len(snap.phases))
	var phaseTotal float64
	for _, ph := range snap.phases {
		if _, seen := sums[ph.Name]; !seen {
			order = append(order, ph.Name)
		}
		sums[ph.Name] += ph.Dur.Seconds()
	}
	for _, name := range order {
		v := sums[name]
		phaseTotal += v
		fmt.Fprintf(b, "fak_gateway_startup_phase_duration_seconds{phase=\"%s\"} %s\n", promQuote(name), promFloat(v))
	}

	// Unaccounted boot time is the "is startup fully instrumented" signal: boot
	// wall-clock the named phases do NOT explain (time_to_ready - sum of phases),
	// clamped at 0 so scrape-timing skew can't render a negative boot. Near-zero
	// means every bit of startup is attributed to a phase; a large value means a
	// phase is missing or host-side work ran between New and MarkReady that no
	// phase records. 0 (like time_to_ready) until the boot completes.
	writeHelpType(b, "fak_gateway_startup_unaccounted_seconds", "Boot wall-clock not explained by any named startup phase (time_to_ready minus the sum of phase durations). Near-zero means startup is fully instrumented.", "gauge")
	unaccounted := snap.timeToReady() - phaseTotal
	if unaccounted < 0 {
		unaccounted = 0
	}
	fmt.Fprintf(b, "fak_gateway_startup_unaccounted_seconds %s\n", promFloat(unaccounted))
}

// writeModelLoadMetrics renders the boot-time weight-load breakdown when the host
// captured one (fak serve --gguf). A nil profile emits NOTHING — a mock/proxy serve
// that loads no weights must not publish a phantom 0-byte, 0ms model load.
func (s *Server) writeModelLoadMetrics(b *strings.Builder) {
	p := s.modelLoadProfile()
	if p == nil {
		return
	}

	writeHelpType(b, "fak_model_load_info", "Static labels for the model loaded at boot: source path, loader mode, and the slowest (bottleneck) load phase.", "gauge")
	fmt.Fprintf(b, "fak_model_load_info{source=\"%s\",mode=\"%s\",bottleneck=\"%s\"} 1\n",
		promQuote(p.Source), promQuote(p.Mode), promQuote(p.Bottleneck))

	writeHelpType(b, "fak_model_load_duration_seconds", "Total wall-clock to load model weights at boot.", "gauge")
	fmt.Fprintf(b, "fak_model_load_duration_seconds %s\n", promFloat(p.TotalSeconds))
	writeHelpType(b, "fak_model_load_bytes", "Total bytes read/materialized while loading model weights at boot.", "gauge")
	fmt.Fprintf(b, "fak_model_load_bytes %d\n", p.Bytes)
	writeHelpType(b, "fak_model_load_tensors", "Number of tensors materialized while loading model weights at boot.", "gauge")
	fmt.Fprintf(b, "fak_model_load_tensors %d\n", p.Tensors)

	phases := p.sorted()
	writeHelpType(b, "fak_model_load_phase_duration_seconds", "Wall-clock cost of each model-weight load phase at boot.", "gauge")
	for _, ph := range phases {
		fmt.Fprintf(b, "fak_model_load_phase_duration_seconds{phase=\"%s\"} %s\n", promQuote(ph.Phase), promFloat(ph.Seconds))
	}
	writeHelpType(b, "fak_model_load_phase_bytes", "Bytes processed by each model-weight load phase at boot.", "gauge")
	for _, ph := range phases {
		fmt.Fprintf(b, "fak_model_load_phase_bytes{phase=\"%s\"} %d\n", promQuote(ph.Phase), ph.Bytes)
	}
	writeHelpType(b, "fak_model_load_phase_tensors", "Tensors processed by each model-weight load phase at boot.", "gauge")
	for _, ph := range phases {
		fmt.Fprintf(b, "fak_model_load_phase_tensors{phase=\"%s\"} %d\n", promQuote(ph.Phase), ph.Tensors)
	}

	if len(p.LoadPaths) > 0 {
		writeHelpType(b, "fak_model_load_path_tensors", "Tensors materialized at boot by GGUF quant type and expert/dense class, split by load path: resident (raw quant bytes, no dequant) vs dequant (f32 round-trip). A nonzero dequant row for a large expert quant type is the slow-load signal.", "gauge")
		for _, r := range p.LoadPaths {
			class := modelLoadPathClass(r.Expert)
			fmt.Fprintf(b, "fak_model_load_path_tensors{quant_type=\"%s\",class=\"%s\",path=\"resident\"} %d\n", promQuote(r.QuantType), class, r.ResidentTensors)
			fmt.Fprintf(b, "fak_model_load_path_tensors{quant_type=\"%s\",class=\"%s\",path=\"dequant\"} %d\n", promQuote(r.QuantType), class, r.DequantTensors)
		}
		writeHelpType(b, "fak_model_load_path_bytes", "On-disk bytes materialized at boot by GGUF quant type and expert/dense class, split by load path (resident raw bytes vs f32 dequant round-trip).", "gauge")
		for _, r := range p.LoadPaths {
			class := modelLoadPathClass(r.Expert)
			fmt.Fprintf(b, "fak_model_load_path_bytes{quant_type=\"%s\",class=\"%s\",path=\"resident\"} %d\n", promQuote(r.QuantType), class, r.ResidentBytes)
			fmt.Fprintf(b, "fak_model_load_path_bytes{quant_type=\"%s\",class=\"%s\",path=\"dequant\"} %d\n", promQuote(r.QuantType), class, r.DequantBytes)
		}
	}

	if rows := p.memoryPlanByClassScope(); len(rows) > 0 {
		writeHelpType(b, "fak_model_load_memory_plan_bytes", "Classed memory demand used to admit the boot-time model load, aggregated by memory class and scope.", "gauge")
		for _, row := range rows {
			fmt.Fprintf(b, "fak_model_load_memory_plan_bytes{class=\"%s\",scope=\"%s\"} %d\n",
				promQuote(row.Class), promQuote(row.Scope), row.Bytes)
		}
	}
	if rows := p.memoryPlanByClassScopeDType(); len(rows) > 0 {
		writeHelpType(b, "fak_model_load_memory_plan_dtype_bytes", "Classed memory demand used to admit the boot-time model load, aggregated by memory class, scope, and bounded dtype/storage label.", "gauge")
		for _, row := range rows {
			fmt.Fprintf(b, "fak_model_load_memory_plan_dtype_bytes{class=\"%s\",scope=\"%s\",dtype=\"%s\"} %d\n",
				promQuote(row.Class), promQuote(row.Scope), promQuote(row.DType), row.Bytes)
		}
	}
	if p.MemoryHeadroomRatio > 0 {
		writeHelpType(b, "fak_model_load_memory_headroom_ratio", "Fraction of reported capacity reserved by the boot-time model-load fit check for runtime headroom.", "gauge")
		fmt.Fprintf(b, "fak_model_load_memory_headroom_ratio %s\n", promFloat(p.MemoryHeadroomRatio))
	}
	if caps := p.sortedMemoryCapacities(); len(caps) > 0 {
		writeHelpType(b, "fak_model_load_memory_capacity_known", "Whether the backend reported capacity for a memory scope used by the boot-time model-load fit check.", "gauge")
		for _, cap := range caps {
			known := 0
			if cap.Known {
				known = 1
			}
			fmt.Fprintf(b, "fak_model_load_memory_capacity_known{scope=\"%s\"} %d\n", promQuote(cap.Scope), known)
		}
		writeHelpType(b, "fak_model_load_memory_capacity_free_known", "Whether the backend reported current free bytes for a memory scope used by the boot-time model-load fit check.", "gauge")
		for _, cap := range caps {
			known := 0
			if cap.Known && cap.FreeKnown {
				known = 1
			}
			fmt.Fprintf(b, "fak_model_load_memory_capacity_free_known{scope=\"%s\"} %d\n", promQuote(cap.Scope), known)
		}
		writeHelpType(b, "fak_model_load_memory_capacity_bytes", "Reported backend capacity bytes for the boot-time model load. The free row is omitted when current free bytes are unknown.", "gauge")
		for _, cap := range caps {
			if !cap.Known {
				continue
			}
			fmt.Fprintf(b, "fak_model_load_memory_capacity_bytes{scope=\"%s\",kind=\"total\"} %d\n",
				promQuote(cap.Scope), cap.TotalBytes)
			if cap.FreeKnown {
				fmt.Fprintf(b, "fak_model_load_memory_capacity_bytes{scope=\"%s\",kind=\"free\"} %d\n",
					promQuote(cap.Scope), cap.FreeBytes)
			}
		}
	}
	if rows := modelLoadMemoryFitRows(p.MemoryPlan, p.MemoryCapacities, p.MemoryHeadroomRatio); len(rows) > 0 {
		writeHelpType(b, "fak_model_load_memory_fit_bytes", "Headroom-adjusted fit summary for the boot-time model load by scope. kind=want is planned bytes; kind=budget and kind=margin are omitted when capacity is unknown.", "gauge")
		for _, row := range rows {
			fmt.Fprintf(b, "fak_model_load_memory_fit_bytes{scope=\"%s\",kind=\"want\"} %d\n",
				promQuote(row.Scope), row.WantBytes)
			if !row.CapacityKnown {
				continue
			}
			fmt.Fprintf(b, "fak_model_load_memory_fit_bytes{scope=\"%s\",kind=\"budget\"} %d\n",
				promQuote(row.Scope), row.BudgetBytes)
			fmt.Fprintf(b, "fak_model_load_memory_fit_bytes{scope=\"%s\",kind=\"margin\"} %d\n",
				promQuote(row.Scope), row.MarginBytes)
		}
	}
}
