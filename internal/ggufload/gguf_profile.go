package ggufload

import (
	"fmt"
	"io"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// hostMemStatus is compute.ReadHostMemStatus, indirected so progress-line tests can
// inject deterministic memory snapshots.
var hostMemStatus = compute.ReadHostMemStatus

// LoadPhaseStat is one aggregate phase in the GGUF quant-on-load path.
type LoadPhaseStat struct {
	Phase   string  `json:"phase"`
	Calls   int     `json:"calls"`
	Tensors int     `json:"tensors,omitempty"`
	Bytes   int64   `json:"bytes,omitempty"`
	Nanos   int64   `json:"nanos"`
	MS      float64 `json:"ms"`
	TimePct float64 `json:"time_pct"`
}

// LoadTensorStat records per-tensor timing so a 27B load profile can identify the
// specific tensor(s) causing page churn or allocator pressure.
type LoadTensorStat struct {
	Name           string  `json:"name"`
	CanonicalName  string  `json:"canonical_name"`
	Type           string  `json:"type"`
	Shape          []int   `json:"shape,omitempty"`
	PayloadBytes   int64   `json:"payload_bytes,omitempty"`
	Values         int     `json:"values,omitempty"`
	ReadNanos      int64   `json:"read_nanos,omitempty"`
	DequantNanos   int64   `json:"dequant_nanos,omitempty"`
	NormalizeNanos int64   `json:"normalize_nanos,omitempty"`
	AddNanos       int64   `json:"add_nanos,omitempty"`
	TotalNanos     int64   `json:"total_nanos"`
	TotalMS        float64 `json:"total_ms"`
}

// LoadPathStat is one row of the per-quant-type load-path breakdown: for a GGUF quant type
// (and expert-vs-dense class) how many model tensors + on-disk bytes took the raw-RESIDENT
// fast path vs paid the f32 DEQUANT round-trip. It is the visibility that makes the
// mixed-quant load cost legible WITHOUT an external gguf-dump: a row like
// {Q6_K, expert, dequant_tensors>0} names exactly the bulk the resident path does not yet
// cover (the S2 lever in docs/notes/GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md).
type LoadPathStat struct {
	QuantType       string `json:"quant_type"`
	Expert          bool   `json:"expert"`
	ResidentTensors int    `json:"resident_tensors,omitempty"`
	ResidentBytes   int64  `json:"resident_bytes,omitempty"`
	DequantTensors  int    `json:"dequant_tensors,omitempty"`
	DequantBytes    int64  `json:"dequant_bytes,omitempty"`
}

// LoadProfile is a machine-readable load-phase report for modelbench. It is scoped
// to the pure GGUF->resident-model path, not tokenizer or inference.
type LoadProfile struct {
	Mode        string           `json:"mode"`
	Source      string           `json:"source,omitempty"`
	TensorCount int              `json:"tensor_count"`
	TotalNanos  int64            `json:"total_nanos"`
	TotalMS     float64          `json:"total_ms"`
	Phases      []LoadPhaseStat  `json:"phases"`
	TopTensors  []LoadTensorStat `json:"top_tensors,omitempty"`
	Bottleneck  string           `json:"bottleneck"`
	// LoadPaths is the per-quant-type resident-vs-dequant breakdown (deterministically
	// ordered). Empty unless the loader recorded it (the resident-Q4_K GLM path does).
	LoadPaths []LoadPathStat `json:"load_paths,omitempty"`
	// Alerts retains the bounded, safety-relevant notices observed while loading. It is
	// the durable twin of AlertWriter: a serve can show these after readiness without
	// replaying ordinary percentage progress through the user's terminal.
	Alerts []LoadAlert `json:"alerts,omitempty"`
}

// LoadAlert is one safety-relevant condition observed during model startup. Kind and
// Level are bounded labels for structured consumers; Text is the complete operator note.
type LoadAlert struct {
	Kind  string `json:"kind"`
	Level string `json:"level"`
	Text  string `json:"text"`
}

// loadPathKey keys the per-quant-type load-path tally by (GGUF quant type, expert class).
type loadPathKey struct {
	quantType string
	expert    bool
}

// LoadProfiler records opt-in GGUF load timings. Nil keeps the loader on its
// existing behavior with no timing or per-tensor bookkeeping.
type LoadProfiler struct {
	stat    map[string]*LoadPhaseStat
	order   []string
	tensors []LoadTensorStat
	TopN    int
	Trace   io.Writer
	Every   int

	// Progress, when non-nil, receives a human-readable one-line load status
	// (percent, tensors, GB, elapsed, throughput) emitted periodically as tensors
	// are loaded — so a multi-minute large-model load is not a silent black box.
	// Total is the expected tensor count (set by the loader before the loop) so the
	// percent is meaningful; ProgressEvery throttles the lines (default every 5%).
	Progress      io.Writer
	Total         int
	ProgressEvery float64 // percent step between progress lines (default 5)
	// AlertWriter receives only safety-relevant preflight/warning lines. Production
	// startup can therefore stay free of percentage churn while retaining the one line
	// that may explain a pre-ready OOM kill. When nil, alerts fall back to Progress for
	// compatibility with existing explicit progress consumers.
	AlertWriter io.Writer
	loadStart   time.Time
	cumBytes    int64
	ggufSeen    int // GGUF tensors consumed (advances even for split/merged tensors)
	lastPct     float64
	memWarned   bool // the headroom-collapse warning fired (emit once per load)
	alerts      []LoadAlert

	// loadPaths tallies the per-quant-type resident-vs-dequant breakdown. Written only by
	// the serial load collector (one goroutine), so it needs no lock even under the parallel
	// load pipeline.
	loadPaths map[loadPathKey]*LoadPathStat
}

// NewLoadProfiler returns an enabled load profiler that records per-phase timings and
// keeps the top 16 slowest tensors by default.
func NewLoadProfiler() *LoadProfiler {
	return &LoadProfiler{stat: map[string]*LoadPhaseStat{}, loadPaths: map[loadPathKey]*LoadPathStat{}, TopN: 16}
}

// recordLoadPath tallies one GGUF tensor against the per-quant-type load-path breakdown:
// which raw quant type, expert-vs-dense, and whether it took the raw-resident fast path
// (resident=true) or the f32 dequant round-trip (resident=false). tensors is the number of
// model tensors the GGUF tensor produced (E for a batched expert blob); bytes is its on-disk
// payload. Called ONLY from the serial load collector, so it is lock-free. Safe on nil.
func (p *LoadProfiler) recordLoadPath(quantType string, expert, resident bool, bytes int64, tensors int) {
	if p == nil || quantType == "" {
		return
	}
	if p.loadPaths == nil {
		p.loadPaths = map[loadPathKey]*LoadPathStat{}
	}
	k := loadPathKey{quantType: quantType, expert: expert}
	st := p.loadPaths[k]
	if st == nil {
		st = &LoadPathStat{QuantType: quantType, Expert: expert}
		p.loadPaths[k] = st
	}
	if resident {
		st.ResidentTensors += tensors
		st.ResidentBytes += bytes
	} else {
		st.DequantTensors += tensors
		st.DequantBytes += bytes
	}
}

// SetTotal records the expected tensor count and starts the load clock so the
// Progress writer can report a meaningful percentage and elapsed time. Safe on nil.
func (p *LoadProfiler) SetTotal(n int) {
	if p == nil {
		return
	}
	p.Total = n
	p.loadStart = time.Now()
	p.lastPct = -1
}

// Tick advances the GGUF-tensor progress counter by one (adding payloadBytes to the
// running total) and emits a throttled progress line. The loader calls it once per GGUF
// tensor it consumes — including the batched experts and the split KV-b halves, which do
// NOT go through recordTensor — so the percentage tracks GGUF tensors read, not just the
// canonical tensors added. Safe on nil / when Progress is unset.
func (p *LoadProfiler) Tick(payloadBytes int64) {
	if p == nil {
		return
	}
	p.ggufSeen++
	p.cumBytes += payloadBytes
	p.emitProgress()
}

// emitProgress writes a throttled one-line load-progress status to p.Progress.
// no-op when Progress is unset or Total is unknown.
func (p *LoadProfiler) emitProgress() {
	if p == nil || (p.Progress == nil && p.AlertWriter == nil) || p.Total <= 0 {
		return
	}
	n := p.ggufSeen
	pct := 100 * float64(n) / float64(p.Total)
	step := p.ProgressEvery
	if step <= 0 {
		step = 5
	}
	// Emit on the first tensor, every `step` percent, and on the last tensor.
	if n != 1 && n != p.Total && pct-p.lastPct < step {
		return
	}
	p.lastPct = pct
	elapsed := time.Duration(0)
	if !p.loadStart.IsZero() {
		elapsed = time.Since(p.loadStart)
	}
	gb := float64(p.cumBytes) / (1 << 30)
	var rate float64
	if s := elapsed.Seconds(); s > 0 {
		rate = gb / s
	}
	mem := hostMemStatus()
	if n == 1 {
		p.emitMemPreflight(mem)
	}
	if p.Progress != nil {
		fmt.Fprintf(p.Progress, "fak: loading model %.0f%% (%d/%d tensors, %.1f GB, %s elapsed, %.2f GB/s%s)\n",
			pct, n, p.Total, gb, elapsed.Round(time.Second), rate, memSuffix(mem))
	}
	p.emitMemCliffWarning(mem)
}

func (p *LoadProfiler) alertWriter() io.Writer {
	if p.AlertWriter != nil {
		return p.AlertWriter
	}
	return p.Progress
}

func (p *LoadProfiler) recordAlert(kind, level, text string) {
	if p == nil || text == "" {
		return
	}
	p.alerts = append(p.alerts, LoadAlert{Kind: kind, Level: level, Text: text})
}

// emitMemPreflight prints a one-time confinement notice before the first progress line.
// It fires ONLY when allocations are strictly confined to a NUMA-node subset, because
// that is the regime where the box-wide MemAvailable number everyone looks at is a lie:
// the load must fit in the confined nodes' free memory or the kernel SIGKILLs the
// process with no Go-side trace (CONSTRAINT_MEMORY_POLICY in dmesg is the only record).
func (p *LoadProfiler) emitMemPreflight(mem compute.HostMemStatus) {
	if !mem.Constrained {
		return
	}
	free := "unknown"
	if mem.PolicyFree > 0 {
		free = fmt.Sprintf("%.1f GB", float64(mem.PolicyFree)/(1<<30))
	}
	avail := "unknown"
	if mem.HostAvail > 0 {
		avail = fmt.Sprintf("%.1f GB", float64(mem.HostAvail)/(1<<30))
	}
	text := fmt.Sprintf("allocations CONFINED to numa node(s) %s (%s): %s free there now, host MemAvailable %s — if the load outgrows the confined nodes the kernel kills fak silently (dmesg: CONSTRAINT_MEMORY_POLICY)",
		mem.PolicyNodes, mem.PolicyLabel, free, avail)
	p.recordAlert("memory-preflight", "warning", text)
	fmt.Fprintln(p.alertWriter(), "fak: memory preflight: "+text)
}

// emitMemCliffWarning fires once when a confined load's remaining node-local free
// memory drops under max(4 GiB, rss/8) — close enough to the cliff that the very next
// tensors may be the ones that draw the OOM kill. Its job is to be the last legible
// line in the log when the kill happens.
func (p *LoadProfiler) emitMemCliffWarning(mem compute.HostMemStatus) {
	if p.memWarned || !mem.Constrained || mem.PolicyFree <= 0 || mem.RSS <= 0 {
		return
	}
	floor := int64(4 << 30)
	if r := mem.RSS / 8; r > floor {
		floor = r
	}
	if mem.PolicyFree >= floor {
		return
	}
	p.memWarned = true
	text := fmt.Sprintf("numa-confined memory nearly exhausted: rss %.1f GB, only %.1f GB free on allowed node(s) %s — imminent risk of a silent kernel OOM kill; widen the membind (or set it only when the node's free memory clears the load peak)",
		float64(mem.RSS)/(1<<30), float64(mem.PolicyFree)/(1<<30), mem.PolicyNodes)
	p.recordAlert("memory-cliff", "warning", text)
	fmt.Fprintln(p.alertWriter(), "fak: WARNING: "+text)
}

// memSuffix renders the live memory tail of a progress line: resident set always (when
// known), plus the confined nodes' current free memory when a strict policy applies —
// the two numbers whose convergence IS the OOM countdown.
func memSuffix(mem compute.HostMemStatus) string {
	s := ""
	if mem.RSS > 0 {
		s += fmt.Sprintf(", rss %.1f GB", float64(mem.RSS)/(1<<30))
	}
	if mem.Constrained && mem.PolicyFree > 0 {
		s += fmt.Sprintf(", node-free %.1f GB", float64(mem.PolicyFree)/(1<<30))
	}
	return s
}

func loadProfileStart(p *LoadProfiler) time.Time {
	if p == nil {
		return time.Time{}
	}
	return time.Now()
}

func loadProfileEnd(p *LoadProfiler, phase string, start time.Time, bytes int64, tensors int) int64 {
	if p == nil || start.IsZero() {
		return 0
	}
	nanos := time.Since(start).Nanoseconds()
	p.record(phase, nanos, bytes, tensors)
	return nanos
}

func (p *LoadProfiler) record(phase string, nanos, bytes int64, tensors int) {
	if p == nil {
		return
	}
	st := p.stat[phase]
	if st == nil {
		st = &LoadPhaseStat{Phase: phase}
		p.stat[phase] = st
		p.order = append(p.order, phase)
	}
	st.Calls++
	st.Tensors += tensors
	st.Bytes += bytes
	st.Nanos += nanos
}

func (p *LoadProfiler) recordTensor(st LoadTensorStat) {
	if p == nil {
		return
	}
	st.TotalMS = float64(st.TotalNanos) / 1e6
	p.tensors = append(p.tensors, st)
	if p.Trace != nil {
		n := len(p.tensors)
		every := p.Every
		if every <= 0 {
			every = 1
		}
		if n == 1 || n%every == 0 {
			fmt.Fprintf(p.Trace, "[gguf load] tensor %d %s -> %s type=%s payload=%.1fMB total=%.1fms read=%.1fms dequant=%.1fms normalize=%.1fms add=%.1fms\n",
				n, st.Name, st.CanonicalName, st.Type, float64(st.PayloadBytes)/1e6,
				st.TotalMS, float64(st.ReadNanos)/1e6, float64(st.DequantNanos)/1e6,
				float64(st.NormalizeNanos)/1e6, float64(st.AddNanos)/1e6)
		}
	}
}

// Snapshot renders the accumulated timings into a LoadProfile: per-phase stats with
// time percentages, the slowest phase as the bottleneck, and the TopN slowest tensors.
// Returns nil for a nil (disabled) profiler.
func (p *LoadProfiler) Snapshot(mode, source string, totalNanos int64) *LoadProfile {
	if p == nil {
		return nil
	}
	denom := totalNanos
	if denom <= 0 {
		for _, st := range p.stat {
			denom += st.Nanos
		}
	}
	out := &LoadProfile{
		Mode:        mode,
		Source:      source,
		TensorCount: len(p.tensors),
		TotalNanos:  totalNanos,
		TotalMS:     float64(totalNanos) / 1e6,
	}
	for _, key := range p.order {
		src := p.stat[key]
		st := *src
		st.MS = float64(st.Nanos) / 1e6
		if denom > 0 {
			st.TimePct = 100 * float64(st.Nanos) / float64(denom)
		}
		out.Phases = append(out.Phases, st)
	}
	sort.Slice(out.Phases, func(i, j int) bool { return out.Phases[i].Nanos > out.Phases[j].Nanos })
	if len(out.Phases) > 0 {
		out.Bottleneck = out.Phases[0].Phase
	}
	top := append([]LoadTensorStat(nil), p.tensors...)
	sort.Slice(top, func(i, j int) bool { return top[i].TotalNanos > top[j].TotalNanos })
	n := p.TopN
	if n <= 0 || n > len(top) {
		n = len(top)
	}
	out.TopTensors = top[:n]
	out.LoadPaths = p.loadPathRows()
	out.Alerts = append([]LoadAlert(nil), p.alerts...)
	return out
}

// loadPathRows renders the per-quant-type load-path tally in a deterministic order:
// experts before dense, then by quant type, so a fixed load yields identical output.
func (p *LoadProfiler) loadPathRows() []LoadPathStat {
	if p == nil || len(p.loadPaths) == 0 {
		return nil
	}
	rows := make([]LoadPathStat, 0, len(p.loadPaths))
	for _, st := range p.loadPaths {
		rows = append(rows, *st)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Expert != rows[j].Expert {
			return rows[i].Expert // experts first (the bulk that dominates the load)
		}
		return rows[i].QuantType < rows[j].QuantType
	})
	return rows
}

// EmitLoadPathSummary writes the per-quant-type resident-vs-dequant breakdown to w (one line
// per row), so an operator watching a multi-minute load sees, in-band, exactly which quant
// types took the fast raw-resident path and which paid the slow f32 round-trip — the
// mixed-quant diagnosis without an external gguf-dump. No-op on nil / no recorded rows.
func (p *LoadProfiler) EmitLoadPathSummary(w io.Writer) {
	if p == nil || w == nil {
		return
	}
	rows := p.loadPathRows()
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "fak: load-path breakdown (resident = raw bytes, no dequant; dequant = f32 round-trip):\n")
	for _, r := range rows {
		class := "dense"
		if r.Expert {
			class = "expert"
		}
		fmt.Fprintf(w, "fak:   %-5s %-6s  resident=%d (%.1f GB)  dequant=%d (%.1f GB)\n",
			r.QuantType, class,
			r.ResidentTensors, float64(r.ResidentBytes)/(1<<30),
			r.DequantTensors, float64(r.DequantBytes)/(1<<30))
	}
}
