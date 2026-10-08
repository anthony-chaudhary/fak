package agent

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_moe_residency.go — the SERVE half of R6 (#5617, epic #5606): carry activated-expert
// residency out of a request and into something an operator can read.
//
// The gap this closes is structural, not cosmetic. model.Session.MoEResidency reports one session's
// ring, and on the device path generateReusedContextWithBias builds a session per request
// (p.m.NewBackendSession) and `defer s.Close()`s it. So the entire ledger the ladder spent seven
// rungs building — page-ins, hits, evictions, page-in bytes, coverage, placement drift — is
// constructed, filled, and destroyed inside one request, and no serve surface ever sees a byte of
// it. An operator sizing --n-cpu-moe against a live workload had to infer the ring's behavior from
// tokens per second.
//
// noteMoEResidency folds each finished request's report into a planner-scoped ledger just before
// that session's teardown, following the same shape RequestMemoryStats already uses for per-request
// memory: a small mutex, cumulative counters, and the last full report kept whole so a reader can
// see one request in detail rather than only aggregates.
//
// What the aggregate deliberately does NOT do is average rates. Averaging per-request hit rates
// would weight a two-token request equally with a two-thousand-token one; summing the raw counters
// and dividing once at read time is the number an operator actually tunes against. Every derived
// rate here is therefore computed from the sums, and each guards its denominator so a serve that
// never engaged a ring reports 0 rather than NaN in its telemetry JSON.
//
// Reading PageInBytes/Tokens against a per-request session scope. Because each request gets a fresh
// ring, an expert paged in for request N is gone by request N+1, and the bytes-per-token this
// ledger reports is therefore the COLD-start cost repeated every turn — not the steady-state cost a
// long-lived ring would show. That is a real property of the current serve wiring and the number is
// the honest measure of it, which is precisely why it is worth surfacing: it is the evidence for
// giving the planner a model-scoped ring (the SharedExpertRing R7/#5618 already built at
// (model, device) scope) instead of a guess that one would help.

// MoEResidencyLedger is a serve's activated-expert residency. Requests and Tokens frame only
// requests that engaged a routed-expert ring; Checkpoint independently carries the latest
// model-lifetime checkpoint-tier snapshot and phase attribution may be present when Requests is zero.
type MoEResidencyLedger struct {
	// Requests is how many completed requests contributed, and Tokens how many tokens those
	// requests actually forwarded through the model (prompt tokens not served from the prefix cache,
	// plus generated ones). Tokens is the denominator of the byte rates, so it counts FORWARDED
	// tokens rather than prompt length: a token served from the radix cache activated no expert and
	// would make the ring look cheaper than it is.
	Requests int64 `json:"requests"`
	Tokens   int64 `json:"tokens"`
	// The ring ledger summed across requests. Lookups is every staging request the rings received;
	// Hits resident reuses; PageIns cold uploads; Evictions page-outs; Refusals stagings no budget
	// could admit, which fall back to permanent unbounded residency and are the first sign a budget
	// is being abandoned rather than enforced.
	Lookups   int64 `json:"lookups"`
	Hits      int64 `json:"hits"`
	PageIns   int64 `json:"page_ins"`
	Evictions int64 `json:"evictions"`
	Refusals  int64 `json:"refusals"`
	// PageInBytes is the device bytes cold uploads actually moved — the numerator of the
	// bytes-per-token an operator sizes a budget against, and not recoverable from PageIns because
	// routed projections differ in size and quantization.
	PageInBytes int64 `json:"page_in_bytes"`
	// BudgetBytes is the declared ceiling most recently observed and PeakBytes the high-water
	// footprint across every request. PeakBytes <= BudgetBytes is the boundedness claim, held here
	// across requests rather than only within one.
	BudgetBytes int64 `json:"budget_bytes"`
	PeakBytes   int64 `json:"peak_bytes"`
	// ReconciliationFailures counts requests whose own report failed its internal identity checks
	// (hits+page-ins+refusals == lookups, resident within budget, and the cross-agent pairs under a
	// shared ring). It should be 0 forever. A non-zero value means the ring's accounting disagreed
	// with itself, so every number above it is suspect — which is worth an alarm rather than a
	// silently wrong dashboard, and is the reason the per-request report computes those checks from
	// independent increments instead of restating one.
	ReconciliationFailures int64 `json:"reconciliation_failures"`
	// Checkpoint is the latest model-lifetime checkpoint-tier snapshot. Unlike the
	// request-scoped ring counters above, these cumulative source counters are
	// replaced on each observation rather than summed across requests.
	Checkpoint *CheckpointResidencySnapshot `json:"checkpoint,omitempty"`
	// V41Phases is the latest model-lifetime V4.1 phase attribution. These
	// cumulative source counters are replaced, never added per request.
	V41Phases *V41PhaseLifetimeSnapshot `json:"v41_phases,omitempty"`
	// Last is the most recent request's full report, kept whole. The aggregate answers "what is this
	// serve costing"; Last answers "what did one request actually do", including the placement basis
	// and drift, which do not sum across requests in any meaningful way.
	Last model.MoEResidencyReport `json:"last,omitempty"`
}

// CheckpointResidencySnapshot is the bounded, non-sensitive checkpoint-tier
// state exposed by the planner. It deliberately excludes source paths and the
// checkpoint tier's last raw error.
type CheckpointResidencySnapshot struct {
	Scope            string `json:"scope"`
	Reads            int    `json:"reads"`
	Hits             int    `json:"hits"`
	BytesRead        int64  `json:"bytes_read"`
	Evictions        int    `json:"evictions"`
	Failures         int    `json:"failures"`
	BudgetBytes      int64  `json:"budget_bytes"`
	ResidentBytes    int64  `json:"resident_bytes"`
	PeakBytes        int64  `json:"peak_bytes"`
	ResidentCount    int    `json:"resident_count"`
	OverlayRows      int    `json:"overlay_rows"`
	OverlayBytesRead int64  `json:"overlay_bytes_read"`
}

// V41PhaseLifetimeSnapshot is the bounded model-lifetime view of V4.1 work.
type V41PhaseLifetimeSnapshot struct {
	Scope       string                             `json:"scope"`
	Prefill     V41PhaseSnapshot                   `json:"prefill"`
	Decode      V41PhaseSnapshot                   `json:"decode"`
	Projections V41ProjectionPhaseLifetimeSnapshot `json:"projections"`
	Compressor  V41CompressorPhaseLifetimeSnapshot `json:"compressor"`
	Indexer     V41IndexerPhaseLifetimeSnapshot    `json:"indexer"`
}

// V41PhaseSnapshot deliberately contains only fixed-shape numeric counters.
type V41PhaseSnapshot struct {
	Tokens                               int   `json:"tokens"`
	AttentionContractionCalls            int   `json:"attention_contraction_calls"`
	AttentionContractionNanos            int64 `json:"attention_contraction_nanos"`
	Faults                               int   `json:"faults"`
	FaultedBytes                         int64 `json:"faulted_bytes"`
	FaultNanos                           int64 `json:"fault_nanos"`
	DequantBytes                         int64 `json:"dequant_bytes"`
	DequantNanos                         int64 `json:"dequant_nanos"`
	Contractions                         int   `json:"contractions"`
	ContractionNanos                     int64 `json:"contraction_nanos"`
	DenseProjectionDeviceCalls           int   `json:"dense_projection_device_calls"`
	DenseProjectionHostCalls             int   `json:"dense_projection_host_calls"`
	DenseProjectionNanos                 int64 `json:"dense_projection_nanos"`
	DenseProjectionActivationUploadBytes int64 `json:"dense_projection_activation_upload_bytes"`
	DenseProjectionReadbackBytes         int64 `json:"dense_projection_readback_bytes"`
	GroupedOutputDeviceCalls             int   `json:"grouped_output_device_calls"`
	GroupedOutputHostCalls               int   `json:"grouped_output_host_calls"`
	GroupedOutputNanos                   int64 `json:"grouped_output_nanos"`
	GroupedOutputActivationUploadBytes   int64 `json:"grouped_output_activation_upload_bytes"`
	GroupedOutputReadbackBytes           int64 `json:"grouped_output_readback_bytes"`
	ExpertActivationDeviceCalls          int   `json:"expert_activation_device_calls"`
	ExpertActivationHostCalls            int   `json:"expert_activation_host_calls"`
	ExpertActivationNanos                int64 `json:"expert_activation_nanos"`
	ExpertActivationReadbackBytes        int64 `json:"expert_activation_readback_bytes"`
	IncrementalEngramInjections          int   `json:"incremental_engram_injections"`
	IncrementalEngramNanos               int64 `json:"incremental_engram_nanos"`
	MHCProjectionDeviceCalls             int   `json:"mhc_projection_device_calls"`
	MHCProjectionHostCalls               int   `json:"mhc_projection_host_calls"`
	MHCProjectionNanos                   int64 `json:"mhc_projection_nanos"`
	MHCProjectionActivationUploadBytes   int64 `json:"mhc_projection_activation_upload_bytes"`
	MHCProjectionReadbackBytes           int64 `json:"mhc_projection_readback_bytes"`
}

// V41ProjectionPhaseLifetimeSnapshot attributes projection work by forward phase.
type V41ProjectionPhaseLifetimeSnapshot struct {
	Prefill V41ProjectionPhaseSnapshot `json:"prefill"`
	Decode  V41ProjectionPhaseSnapshot `json:"decode"`
}

// V41ProjectionPhaseSnapshot is the fixed numeric view of head and Engram work.
type V41ProjectionPhaseSnapshot struct {
	HeadProjectionDeviceCalls             int   `json:"head_projection_device_calls"`
	HeadProjectionHostCalls               int   `json:"head_projection_host_calls"`
	HeadProjectionDeviceRows              int   `json:"head_projection_device_rows"`
	HeadProjectionHostRows                int   `json:"head_projection_host_rows"`
	HeadProjectionActivationUploadBytes   int64 `json:"head_projection_activation_upload_bytes"`
	HeadProjectionReadbackBytes           int64 `json:"head_projection_readback_bytes"`
	HeadProjectionNanos                   int64 `json:"head_projection_nanos"`
	EngramProjectionDeviceCalls           int   `json:"engram_projection_device_calls"`
	EngramProjectionHostCalls             int   `json:"engram_projection_host_calls"`
	EngramProjectionDeviceRows            int   `json:"engram_projection_device_rows"`
	EngramProjectionHostRows              int   `json:"engram_projection_host_rows"`
	EngramProjectionMatMulCalls           int   `json:"engram_projection_matmul_calls"`
	EngramProjectionActivationUploadBytes int64 `json:"engram_projection_activation_upload_bytes"`
	EngramProjectionReadbackBytes         int64 `json:"engram_projection_readback_bytes"`
	EngramProjectionNanos                 int64 `json:"engram_projection_nanos"`
	EngramProjectionHostWeightF32Bytes    int64 `json:"engram_projection_host_weight_f32_bytes"`
}

type V41CompressorPhaseLifetimeSnapshot struct {
	Prefill V41CompressorPhaseSnapshot `json:"prefill"`
	Decode  V41CompressorPhaseSnapshot `json:"decode"`
}

type V41CompressorPhaseSnapshot struct {
	CompressorProjectionDeviceCalls           int   `json:"compressor_projection_device_calls"`
	CompressorProjectionHostCalls             int   `json:"compressor_projection_host_calls"`
	CompressorProjectionDeviceRows            int   `json:"compressor_projection_device_rows"`
	CompressorProjectionHostRows              int   `json:"compressor_projection_host_rows"`
	CompressorProjectionActivationUploadBytes int64 `json:"compressor_projection_activation_upload_bytes"`
	CompressorProjectionReadbackBytes         int64 `json:"compressor_projection_readback_bytes"`
	CompressorProjectionNanos                 int64 `json:"compressor_projection_nanos"`
}

type V41IndexerPhaseLifetimeSnapshot struct {
	Prefill V41IndexerPhaseSnapshot `json:"prefill"`
	Decode  V41IndexerPhaseSnapshot `json:"decode"`
}

type V41IndexerPhaseSnapshot struct {
	IndexerProjectionDeviceCalls           int   `json:"indexer_projection_device_calls"`
	IndexerProjectionHostCalls             int   `json:"indexer_projection_host_calls"`
	IndexerProjectionDeviceRows            int   `json:"indexer_projection_device_rows"`
	IndexerProjectionHostRows              int   `json:"indexer_projection_host_rows"`
	IndexerProjectionActivationUploadBytes int64 `json:"indexer_projection_activation_upload_bytes"`
	IndexerProjectionReadbackBytes         int64 `json:"indexer_projection_readback_bytes"`
	IndexerProjectionNanos                 int64 `json:"indexer_projection_nanos"`
	IndexerScoringCalls                    int   `json:"indexer_scoring_calls"`
	IndexerScoringNanos                    int64 `json:"indexer_scoring_nanos"`
}

func v41IndexerPhaseSnapshot(in model.V41ExpertFaultAttributionPhase) V41IndexerPhaseSnapshot {
	return V41IndexerPhaseSnapshot{
		IndexerProjectionDeviceCalls:           in.IndexerProjectionDeviceCalls,
		IndexerProjectionHostCalls:             in.IndexerProjectionHostCalls,
		IndexerProjectionDeviceRows:            in.IndexerProjectionDeviceRows,
		IndexerProjectionHostRows:              in.IndexerProjectionHostRows,
		IndexerProjectionActivationUploadBytes: in.IndexerProjectionActivationUploadBytes,
		IndexerProjectionReadbackBytes:         in.IndexerProjectionReadbackBytes,
		IndexerProjectionNanos:                 in.IndexerProjectionNanos,
		IndexerScoringCalls:                    in.IndexerScoringCalls,
		IndexerScoringNanos:                    in.IndexerScoringNanos,
	}
}

// HitRate is Hits/(Hits+PageIns) over the whole serve — the activated-set hit rate, weighted by
// staging volume rather than by request count. It answers 0 when no staging ever happened, which
// reads as "not measured" and not as "everything missed".
func (l MoEResidencyLedger) HitRate() float64 {
	staged := l.Hits + l.PageIns
	if staged <= 0 {
		return 0
	}
	return float64(l.Hits) / float64(staged)
}

// RefusalRate is Refusals/Lookups. Any non-zero value is a sizing bug, not a tuning dial: a refused
// staging did not fail the forward, it silently promoted that weight to permanent residency, so the
// budget the operator declared stopped being the bound.
func (l MoEResidencyLedger) RefusalRate() float64 {
	if l.Lookups <= 0 {
		return 0
	}
	return float64(l.Refusals) / float64(l.Lookups)
}

// ExpertBytesPerToken is the device bytes each forwarded token cost in expert page-ins — the number
// --n-cpu-moe is actually sized against, and the one that falls when residency is working.
func (l MoEResidencyLedger) ExpertBytesPerToken() float64 {
	if l.Tokens <= 0 {
		return 0
	}
	return float64(l.PageInBytes) / float64(l.Tokens)
}

// PeakBudgetUsed is PeakBytes/BudgetBytes. Well under 1 across a real workload means the budget is
// larger than the activated working set needs and the difference could be given back to KV.
func (l MoEResidencyLedger) PeakBudgetUsed() float64 {
	if l.BudgetBytes <= 0 {
		return 0
	}
	return float64(l.PeakBytes) / float64(l.BudgetBytes)
}

// noteMoEResidency folds one finished request's residency into the planner ledger. It must be
// called while the session is still alive — on the device path the caller's `defer s.Close()` takes
// the ring with it. It is a no-op only when the session has neither a ring nor a checkpoint tier;
// checkpoint-only sessions retain the latest cumulative source snapshot.
//
// tokens is the count actually FORWARDED, which the caller computes; a request that served its
// whole prompt from the prefix cache and generated nothing contributes a ring read and no tokens,
// and must not be allowed to divide by zero downstream.
func (p *InKernelPlanner) noteMoEResidency(s *model.Session, tokens int64) {
	if p == nil || s == nil {
		return
	}
	// Serialize the model-lifetime source snapshot with its publication. Without
	// this span, an older overlapping request could capture first, stall, and then
	// overwrite a newer cumulative checkpoint snapshot after it was published.
	p.moeMu.Lock()
	defer p.moeMu.Unlock()
	p.foldMoEResidencyLocked(s.MoEResidency(model.MoEResidencyOptions{Tokens: tokens}), tokens)
	if s.M != nil {
		p.foldV41PhasesLocked(s.M.V41ExpertFaultAttribution())
	}
}

func (p *InKernelPlanner) foldV41PhasesLocked(at model.V41ExpertFaultAttribution) {
	prefill := v41PhaseSnapshot(at.Prefill)
	decode := v41PhaseSnapshot(at.Decode)
	projections := V41ProjectionPhaseLifetimeSnapshot{
		Prefill: v41ProjectionPhaseSnapshot(at.Prefill),
		Decode:  v41ProjectionPhaseSnapshot(at.Decode),
	}
	compressor := V41CompressorPhaseLifetimeSnapshot{
		Prefill: v41CompressorPhaseSnapshot(at.Prefill),
		Decode:  v41CompressorPhaseSnapshot(at.Decode),
	}
	indexer := V41IndexerPhaseLifetimeSnapshot{Prefill: v41IndexerPhaseSnapshot(at.Prefill), Decode: v41IndexerPhaseSnapshot(at.Decode)}
	if prefill == (V41PhaseSnapshot{}) && decode == (V41PhaseSnapshot{}) && projections == (V41ProjectionPhaseLifetimeSnapshot{}) && compressor == (V41CompressorPhaseLifetimeSnapshot{}) && indexer == (V41IndexerPhaseLifetimeSnapshot{}) {
		return
	}
	p.moeResidency.V41Phases = &V41PhaseLifetimeSnapshot{
		Scope: "model_lifetime", Prefill: prefill, Decode: decode, Projections: projections, Compressor: compressor, Indexer: indexer,
	}
}

func v41PhaseSnapshot(in model.V41ExpertFaultAttributionPhase) V41PhaseSnapshot {
	return V41PhaseSnapshot{
		Tokens:                    in.Tokens,
		AttentionContractionCalls: in.AttentionContractionCalls,
		AttentionContractionNanos: in.AttentionContractionNanos,
		Faults:                    in.Faults, FaultedBytes: in.FaultedBytes, FaultNanos: in.FaultDoorNanos,
		DequantBytes: in.DequantBytes, DequantNanos: in.DequantNanos,
		Contractions: in.Contractions, ContractionNanos: in.ContractionNanos,
		DenseProjectionDeviceCalls:           in.DenseProjectionDeviceCalls,
		DenseProjectionHostCalls:             in.DenseProjectionHostCalls,
		DenseProjectionNanos:                 in.DenseProjectionNanos,
		DenseProjectionActivationUploadBytes: in.DenseProjectionActivationUploadBytes,
		DenseProjectionReadbackBytes:         in.DenseProjectionReadbackBytes,
		GroupedOutputDeviceCalls:             in.GroupedOutputDeviceCalls,
		GroupedOutputHostCalls:               in.GroupedOutputHostCalls,
		GroupedOutputNanos:                   in.GroupedOutputNanos,
		GroupedOutputActivationUploadBytes:   in.GroupedOutputActivationUploadBytes,
		GroupedOutputReadbackBytes:           in.GroupedOutputReadbackBytes,
		ExpertActivationDeviceCalls:          in.ExpertActivationDeviceCalls,
		ExpertActivationHostCalls:            in.ExpertActivationHostCalls,
		ExpertActivationNanos:                in.ExpertActivationNanos,
		ExpertActivationReadbackBytes:        in.ExpertActivationReadbackBytes,
		IncrementalEngramInjections:          in.IncrementalEngramInjections,
		IncrementalEngramNanos:               in.IncrementalEngramNanos,
		MHCProjectionDeviceCalls:             in.MHCProjectionDeviceCalls,
		MHCProjectionHostCalls:               in.MHCProjectionHostCalls,
		MHCProjectionNanos:                   in.MHCProjectionNanos,
		MHCProjectionActivationUploadBytes:   in.MHCProjectionActivationUploadBytes,
		MHCProjectionReadbackBytes:           in.MHCProjectionReadbackBytes,
	}
}

func v41CompressorPhaseSnapshot(in model.V41ExpertFaultAttributionPhase) V41CompressorPhaseSnapshot {
	return V41CompressorPhaseSnapshot{
		CompressorProjectionDeviceCalls:           in.CompressorProjectionDeviceCalls,
		CompressorProjectionHostCalls:             in.CompressorProjectionHostCalls,
		CompressorProjectionDeviceRows:            in.CompressorProjectionDeviceRows,
		CompressorProjectionHostRows:              in.CompressorProjectionHostRows,
		CompressorProjectionActivationUploadBytes: in.CompressorProjectionActivationUploadBytes,
		CompressorProjectionReadbackBytes:         in.CompressorProjectionReadbackBytes,
		CompressorProjectionNanos:                 in.CompressorProjectionNanos,
	}
}

func v41ProjectionPhaseSnapshot(in model.V41ExpertFaultAttributionPhase) V41ProjectionPhaseSnapshot {
	return V41ProjectionPhaseSnapshot{
		HeadProjectionDeviceCalls:             in.HeadProjectionDeviceCalls,
		HeadProjectionHostCalls:               in.HeadProjectionHostCalls,
		HeadProjectionDeviceRows:              in.HeadProjectionDeviceRows,
		HeadProjectionHostRows:                in.HeadProjectionHostRows,
		HeadProjectionActivationUploadBytes:   in.HeadProjectionActivationUploadBytes,
		HeadProjectionReadbackBytes:           in.HeadProjectionReadbackBytes,
		HeadProjectionNanos:                   in.HeadProjectionNanos,
		EngramProjectionDeviceCalls:           in.EngramProjectionDeviceCalls,
		EngramProjectionHostCalls:             in.EngramProjectionHostCalls,
		EngramProjectionDeviceRows:            in.EngramProjectionDeviceRows,
		EngramProjectionHostRows:              in.EngramProjectionHostRows,
		EngramProjectionMatMulCalls:           in.EngramProjectionMatMulCalls,
		EngramProjectionActivationUploadBytes: in.EngramProjectionActivationUploadBytes,
		EngramProjectionReadbackBytes:         in.EngramProjectionReadbackBytes,
		EngramProjectionNanos:                 in.EngramProjectionNanos,
		EngramProjectionHostWeightF32Bytes:    in.EngramProjectionHostWeightF32Bytes,
	}
}

// foldMoEResidency is the accumulation itself, split from the session read so the arithmetic is
// witnessable against hand-built reports — a ledger that summed wrong would otherwise only be
// catchable by driving a real ring and reasoning backwards from the total.
func (p *InKernelPlanner) foldMoEResidency(rep model.MoEResidencyReport, tokens int64) {
	p.moeMu.Lock()
	defer p.moeMu.Unlock()
	p.foldMoEResidencyLocked(rep, tokens)
}

func (p *InKernelPlanner) foldMoEResidencyLocked(rep model.MoEResidencyReport, tokens int64) {
	if !rep.Ring.Enabled && !rep.Checkpoint.Enabled {
		return
	}
	l := &p.moeResidency
	if rep.Ring.Enabled {
		if tokens < 0 {
			tokens = 0
		}
		l.Requests++
		l.Tokens += tokens
		l.Lookups += int64(rep.Ring.Lookups)
		l.Hits += int64(rep.Ring.Hits)
		l.PageIns += int64(rep.Ring.PageIns)
		l.Evictions += int64(rep.Ring.Evictions)
		l.Refusals += int64(rep.Ring.Refusals)
		l.PageInBytes += rep.Ring.PageInBytes
		l.BudgetBytes = rep.Ring.BudgetBytes
		if rep.Ring.PeakBytes > l.PeakBytes {
			l.PeakBytes = rep.Ring.PeakBytes
		}
		if !rep.Reconciliation.OK {
			l.ReconciliationFailures++
		}
	}
	if rep.Checkpoint.Enabled {
		ck := rep.Checkpoint
		l.Checkpoint = &CheckpointResidencySnapshot{
			Scope:            "model_lifetime",
			Reads:            ck.Reads,
			Hits:             ck.Hits,
			BytesRead:        ck.BytesRead,
			Evictions:        ck.Evictions,
			Failures:         ck.Failures,
			BudgetBytes:      ck.BudgetBytes,
			ResidentBytes:    ck.ResidentBytes,
			PeakBytes:        ck.PeakBytes,
			ResidentCount:    ck.ResidentCount,
			OverlayRows:      ck.OverlayRows,
			OverlayBytesRead: ck.OverlayBytesRead,
		}
	}
	l.Last = rep
}

// MoEResidencyStats returns the serve's activated-expert residency. It is the accessor a telemetry
// surface or a `fak` verb reads; the zero value (Requests==0) is the honest answer for a serve that
// never engaged a ring, and callers should render that as "not engaged" rather than as all-zero
// metrics that look like a ring doing nothing.
func (p *InKernelPlanner) MoEResidencyStats() MoEResidencyLedger {
	if p == nil {
		return MoEResidencyLedger{}
	}
	p.moeMu.Lock()
	defer p.moeMu.Unlock()
	out := p.moeResidency
	if out.Checkpoint != nil {
		checkpoint := *out.Checkpoint
		out.Checkpoint = &checkpoint
	}
	if out.V41Phases != nil {
		phases := *out.V41Phases
		out.V41Phases = &phases
	}
	return out
}

// moeResidencyState is the planner-side storage, embedded rather than declared inline in the
// InKernelPlanner struct so this rung adds one field to that already-long definition.
type moeResidencyState struct {
	moeMu        sync.Mutex
	moeResidency MoEResidencyLedger
}
