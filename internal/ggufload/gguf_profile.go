package ggufload

import (
	"fmt"
	"io"
	"sort"
	"time"
)

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
	loadStart     time.Time
	cumBytes      int64
	ggufSeen      int // GGUF tensors consumed (advances even for split/merged tensors)
	lastPct       float64
}

// NewLoadProfiler returns an enabled load profiler that records per-phase timings and
// keeps the top 16 slowest tensors by default.
func NewLoadProfiler() *LoadProfiler {
	return &LoadProfiler{stat: map[string]*LoadPhaseStat{}, TopN: 16}
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
	if p == nil || p.Progress == nil || p.Total <= 0 {
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
	fmt.Fprintf(p.Progress, "fak: loading model %.0f%% (%d/%d tensors, %.1f GB, %s elapsed, %.2f GB/s)\n",
		pct, n, p.Total, gb, elapsed.Round(time.Second), rate)
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
	return out
}
