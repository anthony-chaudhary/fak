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
}

func NewLoadProfiler() *LoadProfiler {
	return &LoadProfiler{stat: map[string]*LoadPhaseStat{}, TopN: 16}
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
