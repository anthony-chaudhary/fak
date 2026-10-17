package nativeperf

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// processCPUTime is swappable so tests can drive deterministic CPU deltas.
var processCPUTime = readProcessCPUTime

// decodeSpansInFlight and decodeSpanStarts detect overlapping decode windows.
// Process CPU and device counters are process-wide, so an overlapped window
// cannot be attributed to one request and its host measurement is dropped.
var decodeSpansInFlight, decodeSpanStarts atomic.Int64

// DecodeSpan samples process CPU time and device-work counters around one
// request's decode loop. The zero value is inert.
type DecodeSpan struct {
	started   time.Time
	cpu       time.Duration
	cpuOK     bool
	device    []compute.DeviceWork
	inFlight  int64
	startSeq  int64
	finished  bool
	measuring bool
}

// StartDecodeSpan begins one decode-window sample. Cost is one process CPU
// syscall plus lock-free counter reads; it never blocks on device work.
func StartDecodeSpan() DecodeSpan {
	s := DecodeSpan{measuring: true}
	s.inFlight = decodeSpansInFlight.Add(1)
	s.startSeq = decodeSpanStarts.Add(1)
	s.device = compute.DeviceWorkSnapshot()
	s.cpu, s.cpuOK = processCPUTime()
	s.started = time.Now()
	return s
}

// Abandon releases a span that never reached Finish (an early return or panic)
// without recording it. Deferring it after StartDecodeSpan is always safe.
func (s *DecodeSpan) Abandon() {
	if s == nil || !s.measuring || s.finished {
		return
	}
	s.finished = true
	decodeSpansInFlight.Add(-1)
}

// DecodeHostWork is one decode window's host/device split. CPU fields are nil
// when the platform cannot read process CPU time or another decode overlapped;
// Kernel is nil when no device backend reported submit-to-fence time.
type DecodeHostWork struct {
	Wall       time.Duration
	Steps      int
	CPU        *time.Duration
	Kernel     *time.Duration
	Backend    string
	Dispatches uint64
	Readback   uint64
	Overlapped bool
}

// Finish closes the span. steps counts decode steps inside the window (the
// first token comes from prefill and is excluded, matching DecodeTPS). The
// result is folded into the request accumulator carried by ctx, if any.
func (s *DecodeSpan) Finish(ctx context.Context, steps int) DecodeHostWork {
	if s == nil || !s.measuring || s.finished {
		return DecodeHostWork{}
	}
	s.finished = true
	w := DecodeHostWork{Wall: time.Since(s.started), Steps: max(steps, 0)}
	cpu, cpuOK := processCPUTime()
	w.Overlapped = s.inFlight > 1 || decodeSpanStarts.Load() != s.startSeq
	decodeSpansInFlight.Add(-1)
	// A zero delta is below the clock's resolution (a Windows tick is ~15.6ms),
	// not a decode loop that used no CPU, so it stays absent.
	if cpuOK && s.cpuOK && !w.Overlapped && cpu > s.cpu {
		d := cpu - s.cpu
		w.CPU = &d
	}
	before := make(map[string]compute.DeviceWork, len(s.device))
	for _, d := range s.device {
		before[d.Backend] = d
	}
	for _, after := range compute.DeviceWorkSnapshot() {
		prev, ok := before[after.Backend]
		if !ok || after.Dispatches < prev.Dispatches {
			continue
		}
		dispatches := after.Dispatches - prev.Dispatches
		if dispatches == 0 && w.Backend != "" {
			continue
		}
		w.Backend, w.Dispatches = after.Backend, dispatches
		if after.ReadbackBytes >= prev.ReadbackBytes {
			w.Readback = after.ReadbackBytes - prev.ReadbackBytes
		}
		if !w.Overlapped && after.SubmitWaitNS >= prev.SubmitWaitNS {
			if k := time.Duration(after.SubmitWaitNS - prev.SubmitWaitNS); k <= w.Wall {
				w.Kernel = &k
			}
		}
	}
	if acc := decodeHostWorkFrom(ctx); acc != nil {
		acc.add(w)
	}
	return w
}

type decodeHostWorkKey struct{}

// DecodeHostWorkAccumulator sums every decode window of one served request
// (an agent loop may decode several times).
type DecodeHostWorkAccumulator struct {
	mu        sync.Mutex
	windows   int
	wall      time.Duration
	steps     int
	cpu       time.Duration
	cpuGap    bool
	kernel    time.Duration
	submitWaitGap bool
	backend   string
	claimed   bool
}

// WithDecodeHostWork installs a per-request accumulator on ctx.
func WithDecodeHostWork(ctx context.Context) (context.Context, *DecodeHostWorkAccumulator) {
	if ctx == nil {
		ctx = context.Background()
	}
	acc := &DecodeHostWorkAccumulator{}
	return context.WithValue(ctx, decodeHostWorkKey{}, acc), acc
}

// RequestDecodeHostWork returns the request accumulator, or nil.
func RequestDecodeHostWork(ctx context.Context) *DecodeHostWorkAccumulator {
	return decodeHostWorkFrom(ctx)
}

func decodeHostWorkFrom(ctx context.Context) *DecodeHostWorkAccumulator {
	if ctx == nil {
		return nil
	}
	acc, _ := ctx.Value(decodeHostWorkKey{}).(*DecodeHostWorkAccumulator)
	return acc
}

func (a *DecodeHostWorkAccumulator) add(w DecodeHostWork) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.windows++
	a.wall += w.Wall
	a.steps += w.Steps
	if w.CPU == nil {
		a.cpuGap = true
	} else {
		a.cpu += *w.CPU
	}
	if w.Kernel == nil {
		a.submitWaitGap = true
	} else {
		a.kernel += *w.Kernel
	}
	if w.Backend != "" {
		a.backend = w.Backend
	}
}

// DecodeHostWorkTotals is a request's folded decode windows. A field is set
// only when every window measured it: one gap makes the request total absent.
type DecodeHostWorkTotals struct {
	Windows int
	Wall    time.Duration
	Steps   int
	CPU     *time.Duration
	Kernel  *time.Duration
	Backend string
}

// Totals returns a coherent copy; ok is false when no decode window ran.
func (a *DecodeHostWorkAccumulator) Totals() (DecodeHostWorkTotals, bool) {
	if a == nil {
		return DecodeHostWorkTotals{}, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.windows == 0 {
		return DecodeHostWorkTotals{}, false
	}
	t := DecodeHostWorkTotals{Windows: a.windows, Wall: a.wall, Steps: a.steps, Backend: a.backend}
	if !a.cpuGap {
		cpu := a.cpu
		t.CPU = &cpu
	}
	if !a.submitWaitGap {
		kernel := a.kernel
		t.Kernel = &kernel
	}
	return t, true
}

// ClaimPhaseReceipt returns the request's decode-kernel phase receipt at most
// once, so a request that writes two perf rows books its kernel time once.
func (a *DecodeHostWorkAccumulator) ClaimPhaseReceipt() (PhaseReceipt, bool) {
	if a == nil {
		return PhaseReceipt{}, false
	}
	totals, ok := a.Totals()
	if !ok {
		return PhaseReceipt{}, false
	}
	a.mu.Lock()
	claimed := a.claimed
	a.claimed = true
	a.mu.Unlock()
	if claimed {
		return PhaseReceipt{}, false
	}
	return totals.DecodeWindowReceipt()
}

// CPUMSPerToken is process CPU milliseconds per decode step; ok is false when
// CPU or steps are unmeasured.
func (t DecodeHostWorkTotals) CPUMSPerToken() (float64, bool) {
	if t.CPU == nil || t.Steps <= 0 {
		return 0, false
	}
	return float64(*t.CPU) / float64(time.Millisecond) / float64(t.Steps), true
}

// HostShare is process CPU time over decode wall time, in cores: 1.0 means one
// core busy for the whole window, and it can exceed 1 on multi-threaded hosts.
func (t DecodeHostWorkTotals) HostShare() (float64, bool) {
	if t.CPU == nil || t.Wall <= 0 {
		return 0, false
	}
	return float64(*t.CPU) / float64(t.Wall), true
}

// DecodeWindowReceipt projects the decode windows into the bounded phase
// receipt: Kernel is host-observed submit-to-fence wall, Decode is the rest of
// the window. ok is false when kernel time was not measured for every window.
func (t DecodeHostWorkTotals) DecodeWindowReceipt() (PhaseReceipt, bool) {
	if t.Kernel == nil || t.Wall <= 0 || *t.Kernel > t.Wall {
		return PhaseReceipt{}, false
	}
	kernel, rest := *t.Kernel, t.Wall-*t.Kernel
	r := PhaseReceipt{Engine: "inkernel", Backend: t.Backend, ForwardPath: "decode_window", Wall: t.Wall}
	for _, p := range phaseOrder {
		a := PhaseAccounting{Phase: p}
		switch p {
		case PhaseDecode:
			a.Inclusive = PhaseTiming{Wall: t.Wall, Active: t.Wall}
			a.Exclusive = PhaseTiming{Wall: rest, Active: rest}
		case PhaseKernel:
			a.Parent = PhaseDecode
			a.Inclusive = PhaseTiming{Wall: kernel, Active: kernel}
			a.Exclusive = a.Inclusive
		}
		r.Phases = append(r.Phases, a)
	}
	return r, true
}
