package agent

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// nativePhaseLogCap bounds the per-planner native-phase ledger. It is small because the
// ledger holds only the MOST RECENT observation per trace id, and one serve usually has a
// handful of in-flight requests; the cap exists so an unbounded stream of distinct trace ids
// (or the empty-trace-id bucket) cannot grow the map without limit.
const nativePhaseLogCap = 64

// recordNativePhase stores the most recent observation for traceID. It is a no-op for an
// unknown phase token, so the closed vocabulary cannot be widened by a caller's typo. The
// empty trace id is a valid key: a request that carried no trace id is still attributed to
// its phase rather than dropped (NativePhaseObservation.TraceID stays empty, never fabricated).
func (p *InKernelPlanner) recordNativePhase(traceID string, phase NativePhase, at time.Time, elapsed time.Duration, completed bool) {
	if p == nil || !nativePhaseKnown(phase) {
		return
	}
	obs := NativePhaseObservation{TraceID: traceID, Phase: phase, At: at, Elapsed: elapsed, Completed: completed}
	p.storeNativePhaseObservation(obs, false)
}

func (p *InKernelPlanner) storeNativePhaseObservation(obs NativePhaseObservation, reset bool) {
	traceID := obs.TraceID
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	if p.nativePhaseLog == nil {
		p.nativePhaseLog = make(map[string]NativePhaseObservation, nativePhaseLogCap)
	}
	if _, seen := p.nativePhaseLog[traceID]; !seen {
		for len(p.nativePhaseSeq) >= nativePhaseLogCap {
			oldest := p.nativePhaseSeq[0]
			p.nativePhaseSeq = p.nativePhaseSeq[1:]
			if _, still := p.nativePhaseLog[oldest]; still {
				delete(p.nativePhaseLog, oldest)
				break
			}
		}
	}
	p.touchNativePhaseLocked(traceID)
	if !reset {
		obs.FirstDraw = p.nativePhaseLog[traceID].FirstDraw
	}
	obs.UpdatedAt = time.Now()
	p.nativePhaseLog[traceID] = obs
}

func (p *InKernelPlanner) touchNativePhaseLocked(traceID string) {
	for i, key := range p.nativePhaseSeq {
		if key == traceID {
			p.nativePhaseSeq = append(p.nativePhaseSeq[:i], p.nativePhaseSeq[i+1:]...)
			break
		}
	}
	p.nativePhaseSeq = append(p.nativePhaseSeq, traceID)
}

func cloneNativePhaseObservation(obs NativePhaseObservation) NativePhaseObservation {
	if obs.FirstDraw != nil {
		draw := *obs.FirstDraw
		if draw.TokenID != nil {
			token := *draw.TokenID
			draw.TokenID = &token
		}
		obs.FirstDraw = &draw
	}
	return obs
}

func (p *InKernelPlanner) beginNativeFirstDraw(traceID string, ceiling int) {
	obs := NativePhaseObservation{TraceID: traceID, Phase: NativePhasePrefill, At: time.Now(),
		FirstDraw: &NativeFirstDrawObservation{ResolvedOutputCeiling: ceiling, Classification: "not_drawn"}}
	p.storeNativePhaseObservation(obs, true)
}

func (p *InKernelPlanner) recordNativeFirstDraw(traceID string, token int, stop bool) {
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	obs, ok := p.nativePhaseLog[traceID]
	if !ok || obs.FirstDraw == nil || obs.FirstDraw.Observed {
		return
	}
	draw := *obs.FirstDraw
	draw.Observed, draw.TokenID, draw.Classification = true, &token, "non_stop_token"
	if token < 0 {
		draw.Classification = "negative_token"
	} else if stop {
		draw.Classification = "stop_token"
	}
	obs.FirstDraw = &draw
	obs.UpdatedAt = time.Now()
	p.nativePhaseLog[traceID] = obs
	p.touchNativePhaseLocked(traceID)
}

// LatestNativePhaseObservation reads the latest actual phase from the same
// bounded ledger. Empty trace IDs remain valid exact keys in the original API.
func (p *InKernelPlanner) LatestNativePhaseObservation() (NativePhaseObservation, bool) {
	if p == nil {
		return NativePhaseObservation{}, false
	}
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	for i := len(p.nativePhaseSeq) - 1; i >= 0; i-- {
		if obs, ok := p.nativePhaseLog[p.nativePhaseSeq[i]]; ok {
			return cloneNativePhaseObservation(obs), true
		}
	}
	return NativePhaseObservation{}, false
}

// NativePhaseObservation reports the most recent native-phase observation recorded for traceID.
// It implements NativePhaseReporter, so the gateway can type-assert a planner for this optional
// seam and emit nothing for a proxy planner that does not implement it. A nil planner reports
// not-observed.
func (p *InKernelPlanner) NativePhaseObservation(traceID string) (NativePhaseObservation, bool) {
	if p == nil {
		return NativePhaseObservation{}, false
	}
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	obs, ok := p.nativePhaseLog[traceID]
	return cloneNativePhaseObservation(obs), ok
}

// Complete renders the transcript as ChatML and runs one in-kernel decode turn,
// returning the generated assistant text. Mirrors cmd/fakchat's hybrid path. The
// per-request SampleOpts override this planner's configured decode length,
// temperature, TopP (nucleus cutoff), and TopK (top-k cutoff) for THIS turn, and a
// per-request Stop sequence ends the turn early (string-suffix stop, orthogonal to
// the token-ID <|im_end|>/EOS stops). All five per-request sampling controls the
// HTTP wires forward are now honored on the in-kernel path too.
// InKernelOOMError is the agent-level, recovered form of an in-kernel device allocation
// failure (a *compute.DeviceAllocError that unwound out of a device decode path). It is
// in-kernel BY CONSTRUCTION â€” only the in-kernel planner / compute backend can produce it,
// never a real upstream â€” so the gateway can safely render a specific, actionable client
// message for it (an over-large prompt on a small GPU) without any risk of leaking upstream
// content. Bytes is the device allocation that failed; Class and Site preserve the allocator
// category for operator visibility without exposing model/provider content.
type InKernelOOMError struct {
	Bytes int
	Class compute.MemoryClass
	Site  string
}

func (e *InKernelOOMError) Error() string {
	class := e.Class
	if class == "" {
		class = compute.MemoryUnknown
	}
	if class == compute.MemoryUnknown {
		return fmt.Sprintf("in-kernel GPU out of memory (device allocation of %d bytes failed)", e.Bytes)
	}
	return fmt.Sprintf("in-kernel GPU out of memory (%s allocation of %d bytes failed)", class, e.Bytes)
}

// InKernelCapacityError is the request-time companion to InKernelOOMError: a backend
// with known capacity can refuse the planned in-kernel request memory before the device
// allocator is touched. It is still a local OOM-class resource exhaustion, but it is
// earlier and more actionable than a recovered DeviceAllocError.
type InKernelCapacityError struct {
	Want  int64
	Avail int64
	Class compute.MemoryClass
	Scope compute.MemoryScope
	Site  string
	// Detail carries the typed cause when no byte figure exists (e.g. the runtime-extras
	// estimator's missing bound), so the message never reports a misleading "needs 0 bytes".
	Detail string
	// Cause is the typed upstream reason when the refusal is not a byte-budget verdict
	// (the *compute.RuntimeExtraCapacityUnknownError behind runtime-extras-unknown), so
	// callers can errors.As the missing bound instead of parsing Detail.
	Cause error
	// AvailSigned is the unclamped budget (ceiling - used - reserved) on the host-memory
	// arm; negative means the resident footprint already exceeds the ceiling. Avail stays
	// clamped at zero for existing callers.
	AvailSigned int64
	// Structural marks a host-memory refusal with no turn in flight and the resident
	// footprint at or above the ceiling: waiting cannot free room, so it is not retryable.
	Structural bool
}

// inKernelRuntimeExtrasUnknownSite marks a refusal where the Qwen3.8 runtime-extras
// estimator could not bound the request plan (no Want/Avail figures exist).
const inKernelRuntimeExtrasUnknownSite = InKernelCapacitySiteRuntimeExtrasUnknown

// InKernelCapacitySiteRuntimeExtrasUnknown is the exported Site of a refusal raised because
// a required runtime-extras bound could not be priced. Want/Avail carry no meaning there.
const InKernelCapacitySiteRuntimeExtrasUnknown = "runtime-extras-unknown"

// ErrInKernelRuntimeExtrasUnknown matches (errors.Is) any InKernelCapacityError whose Site is
// InKernelCapacitySiteRuntimeExtrasUnknown: the closed contract for "the engine could not bound
// its runtime extras", distinct from a known-too-small byte budget.
var ErrInKernelRuntimeExtrasUnknown = errors.New("in-kernel runtime-extras capacity unknown")

// Is reports whether target is ErrInKernelRuntimeExtrasUnknown and this refusal carries that site.
func (e *InKernelCapacityError) Is(target error) bool {
	return target == ErrInKernelRuntimeExtrasUnknown && e != nil && e.Site == InKernelCapacitySiteRuntimeExtrasUnknown
}

// Unwrap exposes the typed cause (if any) to errors.As.
func (e *InKernelCapacityError) Unwrap() error { return e.Cause }

func (e *InKernelCapacityError) Error() string {
	class := e.Class
	if class == "" {
		class = compute.MemoryUnknown
	}
	scope := e.Scope
	if scope == "" {
		scope = compute.MemoryScopeDevice
	}
	subject := "GPU"
	if scope == compute.MemoryScopeHost {
		subject = "host-memory"
	}
	if e.Site == inKernelRuntimeExtrasUnknownSite {
		// No byte figure exists here: the runtime-extras estimator could not bound the plan.
		// Printing "plan needs 0 bytes, available budget is 0 bytes" hid the real cause.
		reason := strings.TrimSpace(e.Detail)
		if reason == "" {
			reason = "runtime extras capacity unknown"
		}
		return fmt.Sprintf("in-kernel %s capacity precheck refused request (%s %s plan could not be bounded: %s; site=%s)", subject, scope, class, reason, e.Site)
	}
	if e.AvailSigned < 0 {
		return fmt.Sprintf("in-kernel %s capacity precheck refused request (%s %s plan needs %d bytes, available budget is %d bytes: resident usage exceeds the ceiling by %d bytes)", subject, scope, class, e.Want, e.AvailSigned, -e.AvailSigned)
	}
	return fmt.Sprintf("in-kernel %s capacity precheck refused request (%s %s plan needs %d bytes, available budget is %d bytes)", subject, scope, class, e.Want, e.Avail)
}

// recoverDevicePanic is the body of Complete's deferred recover, factored out so it is
// unit-testable without a GPU (the panic payload is an ordinary Go value). It converts a
// recovered in-kernel device failure (allocation OOM, typed CUDA backend error, or device fault)
// or a declared Flash / V4.1 forward-admission refusal into a typed,
// actionable error and reports handled=true; for ANY other recovered value it
// reports handled=false so the caller re-panics â€” the recover stays surgical and never swallows
// a genuine bug (a nil deref, an unrelated validation panic).
func recoverDevicePanic(r any) (err error, handled bool) {
	if e, ok := r.(error); ok {
		if errors.Is(e, ErrKVPrecisionUnrealized) {
			return e, true
		}
		// Direct session APIs have no error return. Recover only the declared
		// unsupported/malformed Flash schedule refusals, never arbitrary model
		// validation errors or string panics, and preserve the original cause.
		if errors.Is(e, model.ErrV4FlashAttentionUnimplemented) || errors.Is(e, model.ErrV4FlashAttentionRatioInvalid) {
			return e, true
		}
		// V4.1 Prefill/Step panic their typed stage-admission refusal before any
		// math; under the default policy it must fail the request (or warmup),
		// not the serving process.
		if _, ok := model.V41RefusalSiteOf(e); ok {
			return e, true
		}
		var operation *model.BackendForwardOperationError
		if errors.As(e, &operation) {
			return e, true
		}
		var projectionOperation *model.V41ProjectionOperationError
		if errors.As(e, &projectionOperation) {
			return e, true
		}
		var expertOperation *model.V41ExpertOperationError
		if errors.As(e, &expertOperation) {
			return e, true
		}
		var stall metalgemm.MetalCommandBufferStallError
		if errors.As(e, &stall) {
			return e, true
		}
		var stallPtr *metalgemm.MetalCommandBufferStallError
		if errors.As(e, &stallPtr) {
			return e, true
		}
	}
	var dae *compute.DeviceAllocError
	if e, ok := r.(error); ok && errors.As(e, &dae) {
		return &InKernelOOMError{Bytes: dae.Bytes, Class: dae.DemandClass(), Site: dae.Site}, true
	}
	var cudaLaunchErr *compute.CUDALaunchError
	if e, ok := r.(error); ok && errors.As(e, &cudaLaunchErr) {
		return e, true
	}
	var cudaOpErr *compute.CUDAOpError
	if e, ok := r.(error); ok && errors.As(e, &cudaOpErr) {
		return e, true
	}
	var faultErr *compute.DeviceFaultError
	if e, ok := r.(error); ok && errors.As(e, &faultErr) {
		return e, true
	}
	return nil, false
}

// recoverDeviceCall preserves panic presence separately from its payload.
// Recovery is local to the inner call; Goexit never reaches the conversion below.
func recoverDeviceCall[T any](call func() (T, error)) (result T, err error) {
	returned := false
	var payload any
	func() {
		defer func() { payload = recover() }()
		result, err = call()
		returned = true
	}()
	if !returned {
		if converted, handled := recoverDevicePanic(payload); handled {
			var zero T
			return zero, converted
		}
		panic(payload)
	}
	return result, err
}

type inKernelGenerateResult struct {
	gen, promptTok, matched int
	// cacheable is the lookup-side prefix match (#3390): tokens the radix index matched
	// BEFORE servability (nil KV, exact-hit refeed, unsupported truncate) could trim the
	// realized `matched` below it. cacheable >= matched always; 0 when reuse is off.
	cacheable         int
	sourceTier        radixkv.SnapshotTier
	prefillS, decodeS float64
	stopped           bool
	batchReceipt      InKernelBatchReceipt
	vulkanMTP         *VulkanMTPExecution
	// spec is this request's own verified draft rounds (zero off the speculative
	// paths); enginestep.Default only holds the process-wide sum.
	spec     SpeculativeDecodeTally
	specCost *SpeculativeDecodeCost
}

// VulkanMTPExecution is the request-local execution receipt for the resident
// Vulkan MTP route. It is attached to Completion so the gateway reports what
// the completed request actually ran instead of reading mutable planner state.
type VulkanMTPExecution struct {
	Engine         string
	Backend        string
	RequestedDepth int
	EffectiveDepth int
	ProposalRounds int
	ProposedTokens int
	AcceptedTokens int
	RejectedTokens int
	RollbackTokens int
	// The target counters below cover observed resident-device verification
	// receipts. They do not include ordinary target work after a downgrade.
	TargetOperations      int
	TargetDecodeSteps     int
	FullTargetReplaySteps int
	RecurrentRepairTokens int
	Used                  bool
	DowngradeReason       string
	Cancelled             bool
	Elapsed               time.Duration
	// RoundDiagnostics preserve post-commit target accounting, not additive GPU spans.
	RoundDiagnostics        []*VulkanMTPRoundDiagnostic
	RoundDiagnosticsDropped int
	ClosedRoundTotals       *VulkanMTPClosedRoundTotals
	AcceptanceFallback      *VulkanMTPFallbackDiagnostic
}

const (
	vulkanMTPExecutionEngine             = "mtp-vulkan"
	cudaMTPExecutionEngine               = "mtp-cuda"
	vulkanMTPDowngradeConstructorRefused = "constructor-refused"
	vulkanMTPDowngradeEmptyProposal      = "empty-proposal"
	vulkanMTPDowngradeProposalError      = "proposal-error"
	vulkanMTPDowngradeTargetVerifier     = "target-verifier-downgrade"
	vulkanMTPDowngradeAcceptanceFloor    = "acceptance-floor-fallback"
	vulkanMTPDowngradeCancelled          = "cancelled"
)

// residentMTPExecutionEngine names the backend that admitted resident MTP so a
// CUDA run is never reported as Vulkan. Every non-CUDA backend keeps the legacy
// "mtp-vulkan" value byte-identical for existing receipts.
func residentMTPExecutionEngine(backendName string) string {
	if strings.EqualFold(strings.TrimSpace(backendName), "cuda") {
		return cudaMTPExecutionEngine
	}
	return vulkanMTPExecutionEngine
}

func (p *InKernelPlanner) generateReusedRecovering(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool, measurementOpt ...*nativeInferenceMeasurement) (res inKernelGenerateResult, err error) {
	return recoverDeviceCall(func() (res inKernelGenerateResult, err error) {
		// Speculative acceptance and rollback must share request-local constraints and seeds.
		_, seeded := nativeTargetSampleSeedFromContext(ctx)
		visionPrefill, _ := ctx.Value(visionPrefillKey{}).(*nativeVisionPrefill)
		structured := nativeStructuredConstraintFromContext(ctx)
		structuredSerial := structured != nil && (!p.structuredSpeculationSupported() || structured.reasoningArmed() || !p.greedySpeculativeRequestEligible(temp, logitBias, freqPenalty, presPenalty))
		if structured != nil && len(measurementOpt) > 0 && measurementOpt[0] != nil && measurementOpt[0].topLogprobsK > 0 {
			// Speculative diagnostic rows are raw model logits, not the masked
			// distribution required by a structured top-logprob response.
			structuredSerial = true
		}
		targetOnly := seeded || nativeThinkingConstraintFromContext(ctx) != nil || structuredSerial || visionPrefill != nil
		if coord := p.MetalMTPCoordinator(); !targetOnly && coord != nil && p.qwen38MTPCanaryAllowsExecution() {
			if err := p.checkExecutionDeadline(ctx, len(ids), 0, maxNew); err != nil {
				return inKernelGenerateResult{}, err
			}
			return p.generateReusedMetalMTP(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
		} else if coord != nil {
			targetOnly = true
		}
		if !targetOnly && (p.greedySpeculativeRequestEligible(temp, logitBias, freqPenalty, presPenalty) || p.sampledSpeculativeRequestEligible(temp, logitBias, freqPenalty, presPenalty)) {
			if err := p.checkExecutionDeadline(ctx, len(ids), 0, maxNew); err != nil {
				return inKernelGenerateResult{}, err
			}
			return p.generateReusedSpeculative(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
		}
		gen, promptTok, cacheable, matched, sourceTier, prefillS, decodeS, stopped, err := p.generateReusedContextWithBias(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
		if err != nil {
			return inKernelGenerateResult{}, err
		}
		return inKernelGenerateResult{
			gen:        gen,
			promptTok:  promptTok,
			cacheable:  cacheable,
			matched:    matched,
			sourceTier: sourceTier,
			prefillS:   prefillS,
			decodeS:    decodeS,
			stopped:    stopped,
		}, nil
	})
}

func (p *InKernelPlanner) generateReusedWithOOMRetry(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool, onRetry func(), measurementOpt ...*nativeInferenceMeasurement) (inKernelGenerateResult, error) {
	if err := p.validateV41EngramLookAhead(); err != nil {
		return inKernelGenerateResult{}, err
	}
	if p.v41EngramLookAhead {
		// Fresh state per generation, shared only with this invocation's retry.
		ctx = context.WithValue(ctx, inKernelEngramAttemptContextKey{}, &inKernelEngramAttempt{})
	}
	res, err := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	if err == nil {
		return res, nil
	}
	if res.vulkanMTP != nil && p.sampledSpeculativeRequestEligible(temp, logitBias, freqPenalty, presPenalty) {
		// A sampled transaction failure does not certify an unobserved,
		// replayable proposal. Only its explicit no-proposal verdict permits
		// serial fallback; do not restart it through the outer OOM retry seam.
		return res, err
	}
	if retry, retryGate := p.admitDeviceOOMRetry(ctx, err, len(ids), maxNew); !retry {
		p.releaseFailedOOMRequest(ctx, err)
		return res, retryGate
	}
	if onRetry != nil {
		onRetry()
	}
	retryRes, retryErr := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	p.recordInKernelOOMRetry(err, retryErr == nil)
	if retryErr != nil {
		p.releaseFailedOOMRequest(ctx, retryErr)
	}
	return retryRes, retryErr
}

type inKernelObserverDeliveryContextKey struct{}

// admitDeviceOOMRetry decides whether a failed first attempt may run once more, and
// returns the error to surface when it may not. It is bounded (#13267): a cancelled
// request, a coalesced lane that already crossed the shared decode pass (a replay would
// fail closed with the fresh lane's nil error and book an empty completion as a
// successful retry), an invoked decode observer, and a non-OOM failure never retry.
// A retry is admitted only after idle pools were trimmed AND the request re-prices
// under the recovered capacity,
// otherwise it refuses typed instead of re-growing into the same wall.
func (p *InKernelPlanner) admitDeviceOOMRetry(ctx context.Context, err error, promptTokens, maxNew int) (bool, error) {
	if attempt, _ := ctx.Value(inKernelEngramAttemptContextKey{}).(*inKernelEngramAttempt); attempt != nil && attempt.selected {
		// Selection may have started speculative I/O even if the forward failed.
		// Preserve the primary failure, including when cancellation also arrived.
		return false, err
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return false, ctxErr
	}
	if req, ok := ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest); ok && req.decodePass.Load() > 0 {
		return false, err
	}
	if delivered, _ := ctx.Value(inKernelObserverDeliveryContextKey{}).(*atomic.Bool); delivered != nil && delivered.Load() {
		// Observer state and external output cannot be rewound by onRetry. Keep
		// the original failure and do not prepare or book a retry after delivery;
		// releaseFailedOOMRequest still drops the failed request's idle pools.
		return false, err
	}
	if !p.prepareDeviceOOMRetry(err) {
		return false, err
	}
	revokePromptPanelHeadroom(ctx)
	if capErr := p.refuseOversizeRequest(promptTokens, maxNew); capErr != nil {
		// No retry ran, so none is booked: the retry counter counts attempted retries.
		return false, capErr
	}
	return true, nil
}

func (p *InKernelPlanner) prepareDeviceOOMRetry(err error) bool {
	if p == nil || p.backend == nil {
		return false
	}
	var oom *InKernelOOMError
	if !errors.As(err, &oom) {
		return false
	}
	released := p.trimBackendIdlePools()
	if released {
		log.Printf("inkernel_chat oom-retry model=%s backend=%s class=%s site=%s bytes=%d action=trim-idle-pools",
			p.modelID, p.backend.Name(), oom.Class, oom.Site, oom.Bytes)
	}
	return released
}

func (p *InKernelPlanner) trimBackendIdlePools() bool {
	if p == nil || p.backend == nil {
		return false
	}
	// This maintenance boundary is outside acquiring model entrypoints. A
	// planner-local gate cannot exclude another client of the same device domain.
	// Optional cleanup declines a busy or invalid owner without touching pools.
	if provider, ok := p.backend.(compute.ForwardExecutionOwnerBackend); ok {
		owner := provider.ForwardExecutionOwner()
		if owner == nil || !owner.TryLock() {
			return false
		}
		defer owner.Unlock()
	}
	released := false
	if r, ok := p.backend.(interface{ Recycle() }); ok {
		r.Recycle()
		released = true
	}
	if t, ok := p.backend.(interface{ Trim() }); ok {
		t.Trim()
		released = true
	}
	if t, ok := p.backend.(interface{ TrimLarge(int) }); ok {
		t.TrimLarge(0)
		released = true
	}
	return released
}
