package nativeperf

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// processCPUTime is swappable so tests can drive deterministic CPU deltas.
var processCPUTime = readProcessCPUTime

// graphReplayWitness and h2dCensus are swappable so tests can drive deterministic
// engagement and copy-byte deltas without a capture-once backend or FAK_H2D_CENSUS.
var (
	graphReplayWitness = model.GraphReplayWitness
	h2dCensusSnapshot  = compute.H2DCensusSnapshot
	h2dCensusEnabled   = compute.H2DCensusEnabled
)

// decodeSpansInFlight and decodeSpanStarts detect overlapping decode windows.
// Process CPU and device counters are process-wide, so an overlapped window
// cannot be attributed to one request and its host measurement is dropped.
var decodeSpansInFlight, decodeSpanStarts atomic.Int64

// nonDecodeWorkInFlight and nonDecodeWorkStarts detect device work that is not
// a decode step (prefill, restore, admission) running inside a decode window.
// CPU, transfer, graph and device counters are process-wide, so their
// attribution must be dropped even when kernel timestamp profiling is off.
var nonDecodeWorkInFlight, nonDecodeWorkStarts atomic.Int64

// NonDecodeWork marks one stretch of non-decode device work. The zero value is
// inert. Marks remain active without kernel profiling because other global
// counters still need ownership boundaries.
type NonDecodeWork struct{ active bool }

// BeginNonDecodeWork marks non-decode device work as in flight until End.
func BeginNonDecodeWork() NonDecodeWork {
	nonDecodeWorkInFlight.Add(1)
	nonDecodeWorkStarts.Add(1)
	return NonDecodeWork{active: true}
}

// End closes the mark; calling it again, or deferring it after an explicit
// End, is safe.
func (w *NonDecodeWork) End() {
	if w == nil || !w.active {
		return
	}
	w.active = false
	nonDecodeWorkInFlight.Add(-1)
}

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
	kernels   *compute.KernelProfile
	// nonDecodeSeq and nonDecodeBusy are read before every opening snapshot,
	// so work that starts inside a snapshot still drops the window.
	nonDecodeSeq  int64
	nonDecodeBusy bool
	// witnessEngaged/witnessDegraded/witnessArmed snapshot the graph-replay
	// engagement counters (model.GraphReplayWitness) at span start; h2dCopies/
	// h2dBytes snapshot the FAK_H2D_CENSUS totals the same way.
	witnessEngaged  uint64
	witnessDegraded uint64
	witnessArmed    bool
	h2dCopies       int64
	h2dBytes        int64
}

// StartDecodeSpan begins one decode-window sample. By default the cost is one
// process CPU syscall plus lock-free counter reads and it never blocks on
// device work. With a kernel profile reader registered (the backend's
// timestamp profile is on) it also snapshots that profile, which takes the
// backend's device lock: it must not be called while that lock is held.
func StartDecodeSpan() DecodeSpan {
	s := DecodeSpan{measuring: true}
	s.inFlight = decodeSpansInFlight.Add(1)
	s.startSeq = decodeSpanStarts.Add(1)
	s.nonDecodeSeq = nonDecodeWorkStarts.Load()
	s.nonDecodeBusy = nonDecodeWorkInFlight.Load() > 0
	s.device = compute.DeviceWorkSnapshot()
	if compute.KernelProfileReaderRegistered() {
		s.kernels = compute.KernelProfileSnapshot()
	}
	s.witnessEngaged, s.witnessDegraded, s.witnessArmed = graphReplayWitness()
	s.h2dCopies, s.h2dBytes = h2dCensusSnapshot()
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
// when the platform cannot read process CPU time or another decode/non-decode
// operation overlapped;
// Kernel is nil unless the selected backend explicitly reports complete
// host submit-to-fence timing coverage. This compatibility field is host wall
// elapsed time, never GPU event time or device-active execution time.
// Kernels is the window's per-kernel GPU timestamp delta: nil unless a kernel
// profile was kept, the window had steps, and neither another decode nor
// non-decode device work (NonDecodeOverlap) ran inside it.
type DecodeHostWork struct {
	Wall             time.Duration
	Steps            int
	CPU              *time.Duration
	Kernel           *time.Duration
	Backend          string
	Dispatches       uint64
	Readback         uint64
	Overlapped       bool
	Kernels          *compute.KernelProfile
	NonDecodeOverlap bool
	// DeviceIdle is the window's device-idle host wall and SyncSites its host
	// syncs by site and stage (see compute.DeviceWork). Both stay absent (nil,
	// SyncsObserved false) unless the backend observed every submit and wait at
	// both ends, no other decode/non-decode work overlapped, and no counter went backwards;
	// DeviceIdle is also absent when it exceeds Wall or the window submitted,
	// drained or began with a pipelined batch (compute.DeviceIdleDelta).
	DeviceIdle    *time.Duration
	SyncSites     []compute.DeviceSyncSite
	SyncsObserved bool
	// H2DCopyBytes is the window's FAK_H2D_CENSUS byte delta; nil when the census
	// was off, a counter went backwards, or another decode/non-decode operation
	// overlapped. A measured zero stays present.
	H2DCopyBytes *uint64
	// GraphEngaged and GraphDegraded are the window's graph-replay engagement
	// delta (model.GraphReplayWitness): tokens served by the captured-graph fast
	// path versus tokens that fell to the per-layer host loop on a
	// capture-once-capable backend. GraphArmed reports whether any step ran on
	// such a backend at all, which separates "unsupported" from "unobserved";
	// GraphObserved is false unless armed, the window had steps, no other
	// decode/non-decode work overlapped, and neither counter went backwards.
	GraphEngaged  uint64
	GraphDegraded uint64
	GraphArmed    bool
	GraphObserved bool
}

// Finish closes the span. steps counts decode steps inside the window (the
// first token comes from prefill and is excluded, matching DecodeTPS). The
// result is folded into the request accumulator carried by ctx, if any.
func (s *DecodeSpan) Finish(ctx context.Context, steps int) DecodeHostWork {
	if s == nil || !s.measuring || s.finished {
		return DecodeHostWork{}
	}
	s.finished = true
	// Closing snapshots may block while peer work begins. Keep this span visible
	// through every read, and release it even when a snapshot panics.
	defer decodeSpansInFlight.Add(-1)
	w := DecodeHostWork{Wall: time.Since(s.started), Steps: max(steps, 0)}
	cpu, cpuOK := processCPUTime()
	w.Overlapped = s.inFlight > 1 || decodeSpanStarts.Load() != s.startSeq
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
		// Each candidate owns its values. A later backend with unavailable or
		// reset counters must not inherit the prior backend's measurement.
		w.Readback, w.Kernel = 0, nil
		w.DeviceIdle, w.SyncSites, w.SyncsObserved = nil, nil, false
		if !w.Overlapped && prev.SyncObserved && after.SyncObserved {
			w.DeviceIdle, w.SyncSites, w.SyncsObserved = decodeWindowSyncs(prev, after, w.Wall)
		}
		if after.ReadbackBytes >= prev.ReadbackBytes {
			w.Readback = after.ReadbackBytes - prev.ReadbackBytes
		}
		if !w.Overlapped && prev.SubmitWaitObserved && after.SubmitWaitObserved && after.SubmitWaitNS >= prev.SubmitWaitNS {
			// Bound before converting to signed Duration so overflow cannot mint
			// a negative, apparently valid sample. An observed zero stays valid.
			if delta := after.SubmitWaitNS - prev.SubmitWaitNS; delta <= uint64(w.Wall) {
				k := time.Duration(delta)
				w.Kernel = &k
			}
		}
	}
	if s.kernels != nil {
		after := compute.KernelProfileSnapshot()
		if w.Steps > 0 {
			w.Kernels = compute.KernelProfileDelta(s.kernels, after)
		}
	}
	if engAfter, degAfter, armedAfter := graphReplayWitness(); engAfter >= s.witnessEngaged && degAfter >= s.witnessDegraded {
		w.GraphEngaged = engAfter - s.witnessEngaged
		w.GraphDegraded = degAfter - s.witnessDegraded
		w.GraphArmed = armedAfter
		w.GraphObserved = armedAfter && !w.Overlapped && w.Steps > 0
	}
	// With the census off the counters never move, so a zero delta would be an
	// empty measurement dressed as a measured zero; the field stays absent.
	if h2dCensusEnabled() {
		if copiesAfter, bytesAfter := h2dCensusSnapshot(); copiesAfter >= s.h2dCopies && bytesAfter >= s.h2dBytes {
			b := uint64(bytesAfter - s.h2dBytes)
			w.H2DCopyBytes = &b
		}
	}
	// One final ownership verdict covers all closing snapshots, including work
	// that starts inside their callbacks. Wall/steps belong to this span and
	// backend/graph capability remain useful, but global deltas do not.
	w.Overlapped = s.inFlight > 1 || decodeSpanStarts.Load() != s.startSeq
	w.NonDecodeOverlap = s.nonDecodeBusy || nonDecodeWorkStarts.Load() != s.nonDecodeSeq
	if w.Overlapped || w.NonDecodeOverlap {
		w.CPU, w.Kernel, w.Kernels = nil, nil, nil
		w.DeviceIdle, w.SyncSites, w.SyncsObserved = nil, nil, false
		w.Dispatches, w.Readback = 0, 0
		w.H2DCopyBytes = nil
		w.GraphEngaged, w.GraphDegraded, w.GraphObserved = 0, 0, false
	}
	if acc := decodeHostWorkFrom(ctx); acc != nil {
		acc.add(w)
	}
	return w
}

// decodeWindowSyncs is the window's idle and per-site sync delta between two
// observed snapshots; observed is false on any counter reset.
func decodeWindowSyncs(prev, after compute.DeviceWork, wall time.Duration) (idle *time.Duration, sites []compute.DeviceSyncSite, observed bool) {
	if after.DeviceIdleNS < prev.DeviceIdleNS {
		return nil, nil, false
	}
	sites, observed = compute.DeviceSyncSitesDelta(prev.SyncSites, after.SyncSites)
	if !observed {
		return nil, nil, false
	}
	if delta, ok := compute.DeviceIdleDelta(prev, after); ok && delta <= uint64(wall) {
		d := time.Duration(delta)
		idle = &d
	}
	return idle, sites, true
}

type decodeHostWorkKey struct{}

// DecodeHostWorkAccumulator sums every decode window of one served request
// (an agent loop may decode several times).
type DecodeHostWorkAccumulator struct {
	mu                  sync.Mutex
	windows             int
	wall                time.Duration
	steps               int
	cpu                 time.Duration
	cpuGap              bool
	kernel              time.Duration
	submitWaitGap       bool
	backend             string
	claimed             bool
	hostOverheadClaimed bool
	kernels             *compute.KernelProfile
	kernelWindows       int
	kernelGap           bool
	deviceIdle          time.Duration
	deviceIdleGap       bool
	syncSites           []compute.DeviceSyncSite
	syncGap             bool
	h2dBytes            uint64
	h2dGap              bool
	graphEngaged        uint64
	graphDegraded       uint64
	graphWindows        int
	graphArmed          bool
	graphGap            bool
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
	if w.DeviceIdle == nil {
		a.deviceIdleGap = true
	} else {
		a.deviceIdle += *w.DeviceIdle
	}
	if !w.SyncsObserved {
		a.syncGap, a.syncSites = true, nil
	} else if !a.syncGap {
		a.syncSites = compute.DeviceSyncSitesAdd(a.syncSites, w.SyncSites)
	}
	switch {
	case w.Kernels == nil:
		a.kernelGap = true
	case a.kernelWindows == 0:
		k := *w.Kernels
		k.Rows = append([]compute.KernelRow(nil), w.Kernels.Rows...)
		a.kernels = &k
		a.kernelWindows++
	default:
		if a.kernels = compute.KernelProfileAdd(a.kernels, w.Kernels); a.kernels == nil {
			a.kernelGap = true
		}
		a.kernelWindows++
	}
	if w.H2DCopyBytes == nil {
		a.h2dGap = true
	} else {
		a.h2dBytes += *w.H2DCopyBytes
	}
	a.graphArmed = a.graphArmed || w.GraphArmed
	switch {
	case w.Steps == 0:
		// A zero-step window carries no tokens, so it neither observes nor gaps
		// the engagement witness.
	case w.GraphObserved:
		a.graphWindows++
		a.graphEngaged += w.GraphEngaged
		a.graphDegraded += w.GraphDegraded
	default:
		a.graphGap = true
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
	Kernels *compute.KernelProfile
	// DeviceIdle and SyncSites fold every window's device-idle wall and host
	// syncs; SyncsObserved is false (SyncSites nil) when any window lacked them.
	DeviceIdle    *time.Duration
	SyncSites     []compute.DeviceSyncSite
	SyncsObserved bool
	// H2DCopyBytes sums every window's FAK_H2D_CENSUS byte delta; nil unless
	// every window observed it. H2DCensusDisabled reports that the census was
	// off for the process, which is why a nil H2DCopyBytes reads as disabled
	// rather than unobserved in the receipt's gap reasons.
	H2DCopyBytes      *uint64
	H2DCensusDisabled bool
	// GraphReplayEngaged and GraphReplayDegraded roll up every token-bearing
	// window's graph-replay engagement witness; nil unless at least one such
	// window observed it and none gapped. GraphBackendSupported reports whether
	// any window saw a capture-once-capable backend, which separates "the
	// backend cannot replay" from "it replayed zero times".
	GraphReplayEngaged    *uint64
	GraphReplayDegraded   *uint64
	GraphBackendSupported bool
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
	if !a.kernelGap && a.kernelWindows == a.windows && a.kernels != nil {
		k := *a.kernels
		k.Rows = append([]compute.KernelRow(nil), a.kernels.Rows...)
		t.Kernels = &k
	}
	if !a.deviceIdleGap {
		idle := a.deviceIdle
		t.DeviceIdle = &idle
	}
	if !a.syncGap {
		t.SyncSites = append([]compute.DeviceSyncSite{}, a.syncSites...)
		t.SyncsObserved = true
	}
	if !a.h2dGap {
		b := a.h2dBytes
		t.H2DCopyBytes = &b
	}
	t.H2DCensusDisabled = !h2dCensusEnabled()
	if !a.graphGap && a.graphWindows > 0 {
		e, d := a.graphEngaged, a.graphDegraded
		t.GraphReplayEngaged = &e
		t.GraphReplayDegraded = &d
	}
	t.GraphBackendSupported = a.graphArmed
	return t, true
}

// ClaimHostOverhead projects the request's folded decode windows into rep's v3
// host-overhead decomposition at most once, so a request that writes two
// receipts books its decomposition once. It fills every field the collector
// measured and records a closed-vocabulary gap reason for every field it could
// not observe; ok is false when no decode window ran. This is the production
// producer for the receipt-v3 fields on the decode-window receipt path: the
// spans started at the planner's decode sites fold into this accumulator via
// ctx, and one claim projects the whole request.
func (a *DecodeHostWorkAccumulator) ClaimHostOverhead(rep *Repetition) bool {
	if a == nil || rep == nil {
		return false
	}
	totals, ok := a.Totals()
	if !ok {
		return false
	}
	a.mu.Lock()
	claimed := a.hostOverheadClaimed
	a.hostOverheadClaimed = true
	a.mu.Unlock()
	if claimed {
		return false
	}
	rep.FillHostOverhead(totals)
	return true
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

// GPUIdleMSPerToken is device-idle host wall in milliseconds per decode step:
// host work the device waited on in the serial chain.
func (t DecodeHostWorkTotals) GPUIdleMSPerToken() (float64, bool) {
	if t.DeviceIdle == nil || t.Steps <= 0 {
		return 0, false
	}
	return float64(*t.DeviceIdle) / float64(time.Millisecond) / float64(t.Steps), true
}

// SyncsPerToken is host waits on device completion per decode step.
func (t DecodeHostWorkTotals) SyncsPerToken() (float64, bool) {
	if !t.SyncsObserved || t.Steps <= 0 {
		return 0, false
	}
	var syncs uint64
	for _, s := range t.SyncSites {
		syncs += s.Syncs
	}
	return float64(syncs) / float64(t.Steps), true
}

// DecodeSyncSiteRate is one (site, stage) host-sync cell per decode step.
type DecodeSyncSiteRate struct {
	Site             string
	Stage            string
	SyncsPerToken    float64
	WaitMSPerToken   float64
	D2HBytesPerToken float64
}

// SyncSiteRates is every observed sync cell per decode step, sorted by Site
// then Stage; ok is false when syncs or steps are unmeasured.
func (t DecodeHostWorkTotals) SyncSiteRates() ([]DecodeSyncSiteRate, bool) {
	if !t.SyncsObserved || t.Steps <= 0 {
		return nil, false
	}
	steps := float64(t.Steps)
	out := make([]DecodeSyncSiteRate, 0, len(t.SyncSites))
	for _, s := range t.SyncSites {
		out = append(out, DecodeSyncSiteRate{
			Site: s.Site, Stage: s.Stage,
			SyncsPerToken:    float64(s.Syncs) / steps,
			WaitMSPerToken:   float64(s.WaitNS) / float64(time.Millisecond) / steps,
			D2HBytesPerToken: float64(s.D2HBytes) / steps,
		})
	}
	return out, true
}

// DecodeWindowReceipt projects the decode windows into the bounded phase
// receipt: Kernel is host-observed submit-to-fence wall, Synchronization is
// host wait on device completion beyond that fence wall, and Decode is the rest
// of the window. ok is false when kernel time was not measured for every
// window, or when the observed sync wait exceeds the window wall.
//
// Overlap is booked once. A backend books every synchronous submit-to-fence
// wait into both the kernel counter and a sync site (vulkan_shim submitWait),
// so the sync sites contain the kernel wall rather than add to it: kernel keeps
// the fence wall and synchronization keeps only sync wait above it (pipeline
// drains, batch flushes). Sync is booked only when SyncsObserved; otherwise the
// receipt is the two-phase kernel/decode split. D2H bytes the waits delivered
// go to HostDownloadBytes; no copy duration is derived from bytes.
func (t DecodeHostWorkTotals) DecodeWindowReceipt() (PhaseReceipt, bool) {
	if t.Kernel == nil || t.Wall <= 0 || *t.Kernel > t.Wall {
		return PhaseReceipt{}, false
	}
	kernel := *t.Kernel
	var sync time.Duration
	var d2h uint64
	if t.SyncsObserved {
		var waitNS uint64
		for _, s := range t.SyncSites {
			if waitNS > math.MaxUint64-s.WaitNS || d2h > math.MaxUint64-s.D2HBytes {
				return PhaseReceipt{}, false
			}
			waitNS += s.WaitNS
			d2h += s.D2HBytes
		}
		if waitNS > uint64(t.Wall) {
			return PhaseReceipt{}, false
		}
		if wait := time.Duration(waitNS); wait > kernel {
			sync = wait - kernel
		}
	}
	rest := t.Wall - kernel - sync
	r := PhaseReceipt{Engine: "inkernel", Backend: t.Backend, ForwardPath: "decode_window", Wall: t.Wall, HostDownloadBytes: d2h}
	for _, p := range phaseOrder {
		a := PhaseAccounting{Phase: p}
		switch p {
		case PhaseDecode:
			a.Inclusive = PhaseTiming{Wall: t.Wall, Active: t.Wall - sync, Wait: sync}
			a.Exclusive = PhaseTiming{Wall: rest, Active: rest}
		case PhaseKernel:
			a.Parent = PhaseDecode
			a.Inclusive = PhaseTiming{Wall: kernel, Active: kernel}
			a.Exclusive = a.Inclusive
		case PhaseSynchronization:
			a.Parent = PhaseDecode
			a.Inclusive = PhaseTiming{Wall: sync, Wait: sync}
			a.Exclusive = a.Inclusive
		default:
			// The decode window observes only decode, kernel and sync time; other phases stay zero.
		}
		r.Phases = append(r.Phases, a)
	}
	return r, true
}
