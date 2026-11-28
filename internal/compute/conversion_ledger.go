package compute

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

// conversion_ledger.go implements Go measurement hooks and accounting types for
// the prefill conversion-ledger chain:
//
//   GPU ms/tok -> dispatches+submits/tok -> host ms/tok -> end-to-end tok/s
//
// Every field records observed/derived measurements without fabricating
// throughput or conflating GPU-only kernel time with end-to-end token time.

// ErrConversionLedgerInvalid is returned when a conversion-ledger sample carries
// invalid token counts, negative durations, or zero wall duration.
var ErrConversionLedgerInvalid = errors.New("compute: invalid conversion-ledger sample")

// ConversionRegime identifies the quantized/numeric execution regime recorded
// in a conversion-ledger measurement.
type ConversionRegime string

const (
	ConversionRegimeFP8   ConversionRegime = "fp8"
	ConversionRegimeNVFP4 ConversionRegime = "nvfp4"
	ConversionRegimeQ4K   ConversionRegime = "q4_k"
	ConversionRegimeQ8_0  ConversionRegime = "q8_0"
	ConversionRegimeF32   ConversionRegime = "f32"
)

// ConversionRegimeFromDtype maps a compute.Dtype onto its conversion-ledger regime label.
func ConversionRegimeFromDtype(dt Dtype) ConversionRegime {
	switch dt {
	case FP8:
		return ConversionRegimeFP8
	case NVFP4:
		return ConversionRegimeNVFP4
	case Q4_K:
		return ConversionRegimeQ4K
	case Q8_0:
		return ConversionRegimeQ8_0
	default:
		return ConversionRegime(dt.String())
	}
}

// ConversionNodeMeta identifies the target single-node hardware and kernel path
// associated with a conversion-ledger entry.
type ConversionNodeMeta struct {
	Label            string           `json:"label,omitempty"`
	Regime           ConversionRegime `json:"regime"`
	Arch             string           `json:"arch,omitempty"`
	NumGPUs          int              `json:"num_gpus"`
	VRAMPerGPUBytes  int64            `json:"vram_per_gpu_bytes"`
	AttentionBackend string           `json:"attention_backend,omitempty"`
}

// ConversionSampleInput provides the raw measured quantities for one
// conversion-ledger observation.
type ConversionSampleInput struct {
	Meta         ConversionNodeMeta
	Tokens       int
	Dispatches   uint64
	Submits      uint64
	HostReads    uint64
	HostUploads  uint64
	ReadbackByte uint64
	GPUDuration  time.Duration
	HostDuration time.Duration
	WallDuration time.Duration
}

// ConversionMetrics holds one complete conversion-ledger row, tying GPU time,
// dispatch/submit overhead, and host time directly to end-to-end tok/s.
type ConversionMetrics struct {
	Meta         ConversionNodeMeta `json:"meta"`
	Tokens       int                `json:"tokens"`
	Dispatches   uint64             `json:"dispatches"`
	Submits      uint64             `json:"submits"`
	HostReads    uint64             `json:"host_reads"`
	HostUploads  uint64             `json:"host_uploads"`
	ReadbackByte uint64             `json:"readback_bytes"`
	GPUDuration  time.Duration      `json:"gpu_duration_ns"`
	HostDuration time.Duration      `json:"host_duration_ns"`
	WallDuration time.Duration      `json:"wall_duration_ns"`

	// Explicit conversion-ledger chain metrics:
	GPUMSPerTok                 float64 `json:"gpu_ms_per_tok"`
	DispatchesPlusSubmitsPerTok float64 `json:"dispatches_plus_submits_per_tok"`
	HostMSPerTok                float64 `json:"host_ms_per_tok"`
	EndToEndTokPerSec           float64 `json:"end_to_end_tok_per_s"`

	GPUFractionOfWall    float64 `json:"gpu_fraction_of_wall"`
	HostOverheadFraction float64 `json:"host_overhead_fraction"`
}

// BuildConversionMetrics validates sample measurements and derives the
// conversion-ledger metrics deterministically.
func BuildConversionMetrics(in ConversionSampleInput) (ConversionMetrics, error) {
	if in.Tokens <= 0 {
		return ConversionMetrics{}, fmt.Errorf("%w: tokens=%d must be positive", ErrConversionLedgerInvalid, in.Tokens)
	}
	if in.GPUDuration < 0 || in.HostDuration < 0 {
		return ConversionMetrics{}, fmt.Errorf("%w: negative gpu (%v) or host (%v) duration", ErrConversionLedgerInvalid, in.GPUDuration, in.HostDuration)
	}
	if in.WallDuration <= 0 {
		return ConversionMetrics{}, fmt.Errorf("%w: wall duration %v must be positive", ErrConversionLedgerInvalid, in.WallDuration)
	}

	tokF := float64(in.Tokens)
	gpuMS := float64(in.GPUDuration) / float64(time.Millisecond)
	hostMS := float64(in.HostDuration) / float64(time.Millisecond)
	wallS := float64(in.WallDuration) / float64(time.Second)
	wallMS := float64(in.WallDuration) / float64(time.Millisecond)
	ops := float64(in.Dispatches) + float64(in.Submits)

	out := ConversionMetrics{
		Meta:                        in.Meta,
		Tokens:                      in.Tokens,
		Dispatches:                  in.Dispatches,
		Submits:                     in.Submits,
		HostReads:                   in.HostReads,
		HostUploads:                 in.HostUploads,
		ReadbackByte:                in.ReadbackByte,
		GPUDuration:                 in.GPUDuration,
		HostDuration:                in.HostDuration,
		WallDuration:                in.WallDuration,
		GPUMSPerTok:                 gpuMS / tokF,
		DispatchesPlusSubmitsPerTok: ops / tokF,
		HostMSPerTok:                hostMS / tokF,
		EndToEndTokPerSec:           tokF / wallS,
		GPUFractionOfWall:           gpuMS / wallMS,
		HostOverheadFraction:        hostMS / wallMS,
	}
	if math.IsNaN(out.EndToEndTokPerSec) || math.IsInf(out.EndToEndTokPerSec, 0) {
		return ConversionMetrics{}, fmt.Errorf("%w: non-finite end-to-end tok/s", ErrConversionLedgerInvalid)
	}
	return out, nil
}

// ConversionBaselineComparison compares a candidate run against a same-path
// token-serial or legacy baseline, catching GPU-only wins ("mirages") that do
// not improve end-to-end tok/s.
type ConversionBaselineComparison struct {
	CandidateGPUSpeedup       float64 `json:"candidate_gpu_speedup"`
	EndToEndSpeedup           float64 `json:"end_to_end_speedup"`
	DispatchSubmitDeltaPerTok float64 `json:"dispatch_submit_delta_per_tok"`
	HostMSDeltaPerTok         float64 `json:"host_ms_delta_per_tok"`
	MirageDetected            bool    `json:"mirage_detected"`
	EndToEndWin               bool    `json:"end_to_end_win"`
}

// CompareConversionMetrics compares candidate against baseline across the full
// conversion chain. If GPU ms/tok drops while end-to-end tok/s fails to
// improve, MirageDetected is true.
func CompareConversionMetrics(baseline, candidate ConversionMetrics) (ConversionBaselineComparison, error) {
	if baseline.Tokens <= 0 || candidate.Tokens <= 0 || baseline.WallDuration <= 0 || candidate.WallDuration <= 0 {
		return ConversionBaselineComparison{}, fmt.Errorf("%w: invalid baseline or candidate metrics", ErrConversionLedgerInvalid)
	}
	gpuSpeedup := 0.0
	if candidate.GPUMSPerTok > 0 {
		gpuSpeedup = baseline.GPUMSPerTok / candidate.GPUMSPerTok
	}
	e2eSpeedup := candidate.EndToEndTokPerSec / baseline.EndToEndTokPerSec
	gpuImproved := candidate.GPUMSPerTok < baseline.GPUMSPerTok
	e2eImproved := candidate.EndToEndTokPerSec > baseline.EndToEndTokPerSec
	return ConversionBaselineComparison{
		CandidateGPUSpeedup:       gpuSpeedup,
		EndToEndSpeedup:           e2eSpeedup,
		DispatchSubmitDeltaPerTok: candidate.DispatchesPlusSubmitsPerTok - baseline.DispatchesPlusSubmitsPerTok,
		HostMSDeltaPerTok:         candidate.HostMSPerTok - baseline.HostMSPerTok,
		MirageDetected:            gpuImproved && !e2eImproved,
		EndToEndWin:               e2eImproved,
	}, nil
}

// ConversionHook receives each recorded ConversionMetrics entry synchronously on Record.
type ConversionHook func(ConversionMetrics)

// ConversionLedger records ordered conversion-ledger measurements and notifies
// registered hooks.
type ConversionLedger struct {
	mu      sync.Mutex
	entries []ConversionMetrics
	hooks   []ConversionHook
}

// NewConversionLedger constructs an empty ConversionLedger.
func NewConversionLedger() *ConversionLedger {
	return &ConversionLedger{}
}

// RegisterHook attaches a synchronous callback invoked whenever a measurement
// entry is committed to the ledger.
func (l *ConversionLedger) RegisterHook(fn ConversionHook) {
	if l == nil || fn == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.hooks = append(l.hooks, fn)
}

// Record validates, stores, and broadcasts a conversion-ledger sample.
func (l *ConversionLedger) Record(in ConversionSampleInput) (ConversionMetrics, error) {
	m, err := BuildConversionMetrics(in)
	if err != nil {
		return ConversionMetrics{}, err
	}
	if l == nil {
		return m, nil
	}
	l.mu.Lock()
	l.entries = append(l.entries, m)
	hooks := append([]ConversionHook(nil), l.hooks...)
	l.mu.Unlock()
	for _, fn := range hooks {
		fn(m)
	}
	return m, nil
}

// Entries returns a copy of all recorded conversion-ledger entries in order.
func (l *ConversionLedger) Entries() []ConversionMetrics {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]ConversionMetrics(nil), l.entries...)
}

// ConversionSession brackets a single backend operation with before/after
// counter reads and monotonic wall timing.
type ConversionSession struct {
	ledger       *ConversionLedger
	backend      Backend
	meta         ConversionNodeMeta
	started      time.Time
	before       BackendCounterRead
	counterOK    bool
	beforeKernel *KernelProfile
	kernelOK     bool
}

// Begin snapshots backend counters, optional kernel timestamp profile, and
// start time for a subsequent Finish call.
func (l *ConversionLedger) Begin(be Backend, meta ConversionNodeMeta) *ConversionSession {
	s := &ConversionSession{
		ledger:  l,
		backend: be,
		meta:    meta,
		started: time.Now(),
	}
	if be != nil {
		if read, ok, err := ReadBackendCounters(be); err == nil && ok {
			s.before = read
			s.counterOK = true
		}
	}
	if kp := KernelProfileSnapshot(); kp != nil {
		s.beforeKernel = kp
		s.kernelOK = true
	}
	return s
}

// ConversionHostCounters captures host round-trip counts observed during a
// measured prefill step when using host/resident seams.
type ConversionHostCounters struct {
	DispatchesOverride uint64
	SubmitsOverride    uint64
	HostReads          uint64
	HostUploads        uint64
	GPUDuration        time.Duration
	HostDuration       time.Duration
	WallDuration       time.Duration
}

// Finish closes the measurement bracket, computes counter and duration deltas,
// records them into the ConversionLedger, and returns the resulting metrics.
func (s *ConversionSession) Finish(tokens int, obs ConversionHostCounters) (ConversionMetrics, error) {
	if s == nil {
		return ConversionMetrics{}, fmt.Errorf("%w: nil conversion session", ErrConversionLedgerInvalid)
	}
	wall := obs.WallDuration
	if wall <= 0 {
		wall = time.Since(s.started)
		if wall <= 0 {
			wall = time.Nanosecond
		}
	}

	var dispatches, submits, readback uint64
	gpuDur := obs.GPUDuration
	hostDur := obs.HostDuration

	if s.backend != nil && s.counterOK {
		if after, ok, err := ReadBackendCounters(s.backend); err == nil && ok {
			if after.WorkObserved && s.before.WorkObserved && after.Work.Dispatches >= s.before.Work.Dispatches {
				dispatches = after.Work.Dispatches - s.before.Work.Dispatches
			} else if after.DispatchCountsLive && s.before.DispatchCountsLive &&
				after.Counters.ComputeDispatches >= s.before.Counters.ComputeDispatches {
				dispatches = after.Counters.ComputeDispatches - s.before.Counters.ComputeDispatches
			}
			if after.SubmitsObserved && s.before.SubmitsObserved && after.Submits >= s.before.Submits {
				submits = after.Submits - s.before.Submits
			} else if after.DispatchCountsLive && s.before.DispatchCountsLive &&
				after.Counters.DispatchSubmits >= s.before.Counters.DispatchSubmits {
				submits = after.Counters.DispatchSubmits - s.before.Counters.DispatchSubmits
			}
			if after.WorkObserved && s.before.WorkObserved && after.Work.ReadbackBytes >= s.before.Work.ReadbackBytes {
				readback = after.Work.ReadbackBytes - s.before.Work.ReadbackBytes
			}
			if hostDur == 0 && after.WorkObserved && s.before.WorkObserved &&
				after.Work.SubmitWaitNS >= s.before.Work.SubmitWaitNS {
				hostDur = time.Duration(after.Work.SubmitWaitNS - s.before.Work.SubmitWaitNS)
			}
		}
	}

	if obs.DispatchesOverride > 0 {
		dispatches = obs.DispatchesOverride
	}
	if obs.SubmitsOverride > 0 {
		submits = obs.SubmitsOverride
	}
	if gpuDur == 0 && s.kernelOK {
		if afterKP := KernelProfileSnapshot(); afterKP != nil && afterKP.GPUNS >= s.beforeKernel.GPUNS {
			gpuDur = time.Duration(afterKP.GPUNS - s.beforeKernel.GPUNS)
		}
	}
	if hostDur == 0 && wall > gpuDur {
		hostDur = wall - gpuDur
	}

	return s.ledger.Record(ConversionSampleInput{
		Meta:         s.meta,
		Tokens:       tokens,
		Dispatches:   dispatches,
		Submits:      submits,
		HostReads:    obs.HostReads,
		HostUploads:  obs.HostUploads,
		ReadbackByte: readback,
		GPUDuration:  gpuDur,
		HostDuration: hostDur,
		WallDuration: wall,
	})
}

// PrefillBatchObserved executes PrefillBatch on be under a ConversionLedger
// session and records GPU/host/dispatch/throughput metrics automatically.
func PrefillBatchObserved(be Backend, args PrefillBatchArgs, ledger *ConversionLedger, meta ConversionNodeMeta) (PrefillBatchResult, ConversionMetrics, error) {
	sess := ledger.Begin(be, meta)
	res, err := PrefillBatch(be, args)
	if err != nil {
		return res, ConversionMetrics{}, err
	}
	metrics, merr := sess.Finish(res.Tokens, ConversionHostCounters{})
	return res, metrics, merr
}
