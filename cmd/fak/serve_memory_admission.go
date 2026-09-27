package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// streamedQ4KFreeCPUReservationPeakBytes is the candidate reservation bound
// for the exact Qwen3.8-27B Q4_K_M streamed FreeCPU profile. The two #8964 runs
// peaked at 17,085,792 and 17,895,520 KiB RSS; 20 GiB rounds above the larger
// readiness observation with margin. The first bounded run must still monitor
// swap and measured RSS. This is distinct from the 36 GiB minimum host-size gate
// and does not qualify another model, load recipe, or context envelope.
const streamedQ4KFreeCPUReservationPeakBytes int64 = 20 << 30

const qwen38Q4KMArtifactBytes int64 = 17106775008

// Admission state-envelope constants (issue #9587).
//
// The legacy Metal reservation sizes only the model weights (plus a load-staging
// margin). That undercounts the enforced serve envelope: a running gateway also
// reserves aggregate KV/session state and decode-cohort scratch. These constants
// build a conservative, deterministic state envelope from the gateway admission
// policy's token budget when the model's exact attention geometry is not reachable
// from a raw gguf path. Everything here is gated behind FAK_ADMISSION_STATE_ENVELOPE=1;
// with the gate off the reservation is byte-identical to the legacy weights-only plan.
const (
	// stateEnvelopeKVBytesPerToken is a deliberately pessimistic per-token KV-cache
	// cost used when exact geometry is unavailable. Real decoders with GQA and a
	// quantized KV tier charge less; this scalar keeps the reservation above the true
	// resident envelope so a load that cannot hold its cache is refused rather than
	// silently oversubscribed.
	stateEnvelopeKVBytesPerToken int64 = 64 << 10 // 64 KiB per token
	// stateEnvelopeScratchBytes is the per-session transient/activation scratch.
	stateEnvelopeScratchBytes int64 = 64 << 20 // 64 MiB per session
	// stateEnvelopeMaxMetalSessions caps the enforced session count on the Metal path.
	// Serialized Metal runs a single session, and coalesced decode caps a cohort at 8
	// (internal/agent/inkernel_batch_coordinator.go inKernelDecodeCohortMax). The
	// gateway default MaxNumSeqs (256) is far above what this backend can hold, so the
	// envelope uses the smaller of the two.
	stateEnvelopeMaxMetalSessions = 8
)

// stateEnvelopeEnabled reports whether the additive session/cache state envelope is
// enforced for this process. Default off: legacy weights-only behavior is preserved.
func stateEnvelopeEnabled() bool {
	return os.Getenv("FAK_ADMISSION_STATE_ENVELOPE") == "1"
}

// metalStateEnvelopeMaxSessions is the enforced session bound for the Metal path: the
// smaller of the gateway admission policy's MaxNumSeqs and the in-kernel decode cohort
// cap. A non-positive gateway bound falls back to the cohort cap.
func metalStateEnvelopeMaxSessions() int {
	max := gateway.DefaultAdmissionPolicy().MaxNumSeqs
	if max <= 0 || max > stateEnvelopeMaxMetalSessions {
		max = stateEnvelopeMaxMetalSessions
	}
	return max
}

// estimateModelStateEnvelope builds the complete enforced session/cache state envelope
// for a serve: the weights residency plus maxSessions copies of a per-session KV-cache
// and scratch plan plus the coalesced-decode cohort scratch. It is pure and deterministic
// (no host sampling) so it is unit-testable and reproducible.
//
// The per-session KV demand is a conservative scalar (stateEnvelopeKVBytesPerToken) over
// the gateway token budget; exact per-model attention geometry would tighten it but is
// not reachable from a raw gguf path here. maxSessions is caller-capped (see
// metalStateEnvelopeMaxSessions) rather than taken blindly from the gateway policy.
func estimateModelStateEnvelope(weightsTotal, tokenBudget, maxSessions int) localadmission.SessionResidencyPlan {
	if weightsTotal < 0 {
		weightsTotal = 0
	}
	if tokenBudget < 0 {
		tokenBudget = 0
	}
	if maxSessions < 0 {
		maxSessions = 0
	}
	weights := localadmission.EnvelopePlan{{
		Class:  localadmission.MemClassWeights,
		Bytes:  int64(weightsTotal),
		Detail: "model-weights",
	}}
	perContext := localadmission.EnvelopePlan{
		{
			Class: localadmission.MemClassKVCache,
			// saturatingEnvelopeMul refuses to wrap: an overflowing product must
			// stay huge (and keep the envelope conservative), never collapse the
			// KV term toward zero and under-reserve.
			Bytes:  saturatingEnvelopeMul(int64(tokenBudget), stateEnvelopeKVBytesPerToken),
			Detail: "session-kv-cache-conservative-scalar",
		},
		{
			Class:  localadmission.MemClassActivation,
			Bytes:  stateEnvelopeScratchBytes,
			Detail: "session-scratch",
		},
	}
	// cohort is a COUNT: a value >1 charges one extra copy of the per-context state
	// (the conservative in-flight buffer bound for a grouped decode step). On Metal the
	// decode cohort is capped at stateEnvelopeMaxMetalSessions, so pass that as the
	// conservative bound rather than the raw MaxNumSeqs.
	cohort := maxSessions
	if cohort < 1 {
		cohort = 1
	}
	return localadmission.NewSessionResidencyPlan(weights, perContext, maxSessions, cohort)
}

// memoryClassesToStrings renders a memory-class breakdown as the string-keyed
// map persisted on a local admission reservation.
func memoryClassesToStrings(classes map[localadmission.MemClass]int64) map[string]int64 {
	if len(classes) == 0 {
		return nil
	}
	out := make(map[string]int64, len(classes))
	for class, bytes := range classes {
		out[string(class)] = bytes
	}
	return out
}

// saturatingEnvelopeMul multiplies two non-negative envelope quantities and
// saturates at math.MaxInt64 instead of wrapping. A wrapped product could shrink
// the KV term and under-reserve a load, so overflow must fail toward "too big",
// matching the reservation envelope's saturating arithmetic.
func saturatingEnvelopeMul(a, b int64) int64 {
	if a <= 0 || b <= 0 {
		return 0
	}
	if a > math.MaxInt64/b {
		return math.MaxInt64
	}
	return a * b
}

func isWitnessedQwen38Q4KM(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Size() == qwen38Q4KMArtifactBytes &&
		strings.EqualFold(filepath.Base(path), "Qwen3.8-27B-Q4_K_M.gguf")
}

// qwen38UDQ2KXLArtifactBytes is the exact byte size of the canonical upstream
// Qwen3.8-27B-UD-Q2_K_XL.gguf artifact (9,828,981,664 bytes) pinned in #11961.
// The gate pairs the size with the exact filename so a renamed or truncated file
// is never admitted by coincidence.
const qwen38UDQ2KXLArtifactBytes int64 = 9828981664

func isWitnessedQwen38UDQ2KXL(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Size() == qwen38UDQ2KXLArtifactBytes &&
		strings.EqualFold(filepath.Base(path), "Qwen3.8-27B-UD-Q2_K_XL.gguf")
}

var serveReadMemory = memgate.ReadMemory

// reservationAllocatable applies the turnkey stable-budget floor to a live
// allocatable reading. live is the byte-precise sample; fitFloor (when non-nil)
// is the reserve-based ceiling (total x (1-headroom)) the turnkey plan was
// admitted against at the loader layer.
//
// The rule is deliberately asymmetric so a momentary dip cannot defeat an
// SLO plan while genuine exhaustion still refuses:
//
//   - unknown or critical pressure -> return live unchanged. The kernel has a
//     real signal the host is exhausted (or cannot be read); the reservation
//     must fail closed rather than paper over it.
//   - normal/warning -> max(live, budget), capped at the host total. The floor
//     only ever RAISES a depressed reading up to the reserve-based envelope;
//     a live reading already above the floor is preserved, and the cap keeps
//     the value physically representable.
func reservationAllocatable(live int64, fitFloor *serveFitBudget, pressure memgate.Pressure, total int64) int64 {
	if fitFloor == nil {
		return live
	}
	switch pressure {
	case memgate.PressureCritical, memgate.PressureUnknown:
		return live
	}
	floor := fitFloor.avail()
	if floor <= 0 {
		return live
	}
	if total > 0 && floor > total {
		floor = total
	}
	if floor > live {
		return floor
	}
	return live
}

func defaultLocalReservationDir() string {
	if dir := os.Getenv("FAK_RESERVATION_DIR"); dir != "" {
		return dir
	}
	root, err := os.UserCacheDir()
	if err != nil {
		root = os.TempDir()
	}
	return filepath.Join(root, "fak", "reservations")
}

// estimateMetalModelMemoryBounds estimates the startup peak and steady resident bytes
// for a model path on Apple Silicon unified memory, and reports a typed error when the
// gguf cannot be opened or its weight plan cannot be derived. Test environments may
// inject explicit bounds via FAK_TEST_STARTUP_PEAK_BYTES / FAK_TEST_STEADY_BYTES.
//
// When FAK_ADMISSION_STATE_ENVELOPE=1 the weights-only plan is widened to the complete
// enforced session/cache state envelope (issue #9587); otherwise the returned plan is
// byte-identical to the weights-only estimate.
func estimateMetalModelMemoryBounds(ggufPath string) (localadmission.MemoryPlan, error) {
	plan, _, err := estimateMetalModelMemoryBoundsWithEnvelope(ggufPath)
	return plan, err
}

// estimateMetalModelMemoryBoundsWithEnvelope returns the reservation plan, the
// SessionResidencyPlan whose class breakdown the caller threads onto the reservation
// request when the state envelope is enabled, and a typed error when the gguf cannot be
// opened or its weight plan cannot be derived. A nil envelope means the feature gate is
// off and the plan is the weights-only bound.
func estimateMetalModelMemoryBoundsWithEnvelope(ggufPath string) (localadmission.MemoryPlan, *localadmission.SessionResidencyPlan, error) {
	weights, err := estimateMetalWeightsMemoryBounds(ggufPath)
	if err != nil {
		return localadmission.MemoryPlan{}, nil, err
	}
	if !stateEnvelopeEnabled() {
		return weights, nil, nil
	}
	env := estimateModelStateEnvelope(
		int(weights.SteadyBytes),
		gateway.DefaultAdmissionPolicy().TokenBudget,
		metalStateEnvelopeMaxSessions(),
	)
	plan := localadmission.MemoryPlan{
		StartupPeakBytes: env.StartupPeakBytes(),
		SteadyBytes:      env.SteadyBytes(),
	}
	if plan.StartupPeakBytes < plan.SteadyBytes {
		plan.StartupPeakBytes = plan.SteadyBytes
	}
	return plan, &env, nil
}

// estimateMetalWeightsMemoryBounds is the route-aware weights-only estimator: it derives
// the transformed native weight residency (not the raw payload proxy) and the launcher
// startup peak for the resolved serve load arm. An unopenable gguf or a route whose
// qualified plan cannot be derived returns a typed error so admission refuses before any
// model bytes are loaded.
func estimateMetalWeightsMemoryBounds(ggufPath string) (localadmission.MemoryPlan, error) {
	if peakStr := os.Getenv("FAK_TEST_STARTUP_PEAK_BYTES"); peakStr != "" {
		if peak, err := strconv.ParseInt(peakStr, 10, 64); err == nil && peak > 0 {
			steady := peak * 2 / 3
			if steadyStr := os.Getenv("FAK_TEST_STEADY_BYTES"); steadyStr != "" {
				if s, err := strconv.ParseInt(steadyStr, 10, 64); err == nil && s > 0 {
					steady = s
				}
			}
			return localadmission.MemoryPlan{
				StartupPeakBytes: peak,
				SteadyBytes:      steady,
			}, nil
		}
	}

	ws, err := ggufload.OpenWeights(strings.TrimSpace(ggufPath))
	if err != nil {
		return localadmission.MemoryPlan{}, err
	}
	defer ws.Close()
	arm := resolveMetalServeLoadArm(ws)
	if os.Getenv("FAK_STREAM_Q4K") == "1" || os.Getenv("FAK_METAL_STREAM_Q4K") == "1" {
		// Preserve the separately qualified streaming reservation policy. Its raw
		// payload proxy is not the nonstreamed transformed resident-weight estimate.
		var plan compute.MemoryPlan
		if arm == serveLoadArmQuantProfileQ8 {
			plan, err = ws.EstimateQ8LoadMemoryPlan()
		} else {
			plan, err = ws.EstimateLoadMemoryPlan()
		}
		if err != nil {
			return localadmission.MemoryPlan{}, err
		}
		steady := plan.Total()
		if steady <= 0 {
			return localadmission.MemoryPlan{}, fmt.Errorf("empty streamed Metal memory plan")
		}
		total, _, known := compute.HostSystemMemoryInfo()
		processPeak, _, _ := streamedQ4KMetalCapacity(total, known, os.Getenv("FAK_Q4K_FREE_CPU") == "1")
		if processPeak > 0 && os.Getenv("FAK_Q4K_FREE_CPU") == "1" && isWitnessedQwen38Q4KM(ggufPath) {
			processPeak = streamedQ4KFreeCPUReservationPeakBytes
		}
		return localadmission.MemoryPlan{StartupPeakBytes: max(processPeak, steady), SteadyBytes: steady}, nil
	}
	plan, rawBasis, err := serveMetalGGUFAdmissionWeights(ggufPath, ws)
	if err != nil {
		return localadmission.MemoryPlan{}, err
	}
	steady := plan.Total()
	// Route both the refusal path and this launcher plan through the single
	// shared peak helper. The resident arm passes the historical raw-payload
	// floor as the steady basis; every legacy arm passes its raw plan total.
	peakBasis := steady
	if arm == serveLoadArmResidentQ4K {
		peakBasis = max(steady, rawBasis)
	}
	peak, _ := metalServeStartupPeakBytes(arm, peakBasis, 0, false)
	return localadmission.MemoryPlan{StartupPeakBytes: peak, SteadyBytes: steady}, nil
}

// metalLeaseRefusalAdvice builds the actionable tail of the `fak up` Metal
// residency refusal (#13131). The lease carries only the holder's pid, so the
// historical advice was an unconditional "stop the holder process" — which, on
// 2026-09-15, would have killed a healthy 17-minute modelbench prefill sweep.
// The holder-progress verdict lets the refusal name a progressing holder as
// legitimate and reserve the release advice for a dead or stalled one.
func metalLeaseRefusalAdvice(path string) string {
	return metalLeaseRefusalAdviceFor(gpulease.ProbeHolderProgress(gpulease.HolderProgressOptions{Path: path}), path)
}

// metalLeaseRefusalAdviceFor renders the refusal tail for an already-computed
// holder probe. Split from metalLeaseRefusalAdvice so the wording for each
// verdict is testable without needing a live holder in each state.
func metalLeaseRefusalAdviceFor(probe gpulease.HolderProbe, path string) string {
	const doctor = "inspect the holder with 'fak doctor serve'"
	switch probe.Verdict {
	case gpulease.HolderProgressLiveProgressing:
		return fmt.Sprintf("the lease is held by %s and that holder is making progress (%s), so wait for it to finish rather than killing it; run a CPU/non-Metal serve, or %s, or release %s once it exits",
			gpulease.FormatHolderPID(probe.PID), probe.Detail, doctor, path)
	case gpulease.HolderProgressDead:
		return fmt.Sprintf("the recorded holder (%s) is no longer running, so %s and retry, or release %s",
			gpulease.FormatHolderPID(probe.PID), doctor, path)
	case gpulease.HolderProgressStalled:
		return fmt.Sprintf("the lease is held by %s and that holder looks stalled (%s), so %s and release it once confirmed, or release %s",
			gpulease.FormatHolderPID(probe.PID), probe.Detail, doctor, path)
	default:
		return fmt.Sprintf("stop the holder process and retry, or run a CPU/non-Metal serve; %s (%s), or release %s",
			doctor, probe.Detail, path)
	}
}

// gpuWaitBoundFromEnv is the caller's willingness to queue for the machine-wide
// GPU lease, read from FAK_GPU_WAIT_BOUND (a Go duration such as "120s" or "2m").
// Zero means the caller cannot bound the wait, which makes a progressing-but-
// unattachable holder a TRUTHFUL_REFUSE rather than a WAIT_WITH_BOUND: the
// default must not invent a bound the operator never offered.
func gpuWaitBoundFromEnv() time.Duration {
	raw := strings.TrimSpace(os.Getenv("FAK_GPU_WAIT_BOUND"))
	if raw == "" {
		return 0
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d < 0 {
		return 0
	}
	return d
}

// metalLeaseRefusalError builds the bounded, typed Metal residency refusal. It
// replaces the historical bare "stop the holder process and retry" tail with the
// gpulease scale-out decision, so the error names one of ATTACH_OWNER /
// WAIT_WITH_BOUND / TRUTHFUL_REFUSE and carries the retry bound when one applies.
// Split from the acquisition branch so the wording is testable directly.
func metalLeaseRefusalError(err error, path string, waitBound time.Duration) error {
	rec := gpulease.DecideScaleOut(err, gpulease.ScaleOutOptions{Path: path, WaitBound: waitBound})
	return fmt.Errorf("fak local launcher: Metal residency admission refused before model load: %w; %s; %s",
		err, rec.String(), metalLeaseRefusalAdvice(path))
}

// loadLocalLauncherModelWithMetalLease is the historical admission entry point
// whose loader cannot report failure: a normal return counts as a resident
// model. New callers that can observe a failed load, or that own model
// teardown, use admitLocalMetalModel so a nil model never holds a steady
// reservation and capacity is returned only after the weights are freed.
func loadLocalLauncherModelWithMetalLease(useMetal bool, ggufPath string, opts gpulease.Options, load func(), fitFloors ...*serveFitBudget) (release func(), err error) {
	spec := metalAdmissionSpec{Load: func() bool { load(); return true }}
	if len(fitFloors) > 0 {
		spec.FitFloor = fitFloors[0]
	}
	return admitLocalMetalModel(useMetal, ggufPath, opts, spec)
}

// metalAdmissionSpec is one native Metal load whose reservation lifecycle the
// admission seam owns end to end (#9587).
type metalAdmissionSpec struct {
	// Load performs the model load and reports whether a model is now resident.
	// false is a load failure: the reservation is released without ever being
	// marked steady and the GPU lease is dropped, exactly once, before
	// admitLocalMetalModel returns (with a no-op release and a nil error, so the
	// caller's own "model failed to load" handling still runs).
	Load func() bool
	// Teardown frees the resident model. The returned release runs it BEFORE it
	// hands the reservation and the GPU lease back, so capacity is returned only
	// after the model's bytes are freed. nil leaves teardown to the caller.
	Teardown func() error
	// FitFloor is the STABLE reserve-based budget the turnkey path pins (total
	// unified memory minus the 20% OS reserve); see reservationAllocatable. nil
	// keeps the pure live-probe admission.
	FitFloor *serveFitBudget
	// ReceiptOut receives the lifecycle receipt lines; nil writes them to
	// serveAdmissionReceiptOut (stderr). A caller that owns the terminal (guard's
	// quiet/auto banner) passes io.Discard.
	ReceiptOut io.Writer
}

// Native admission modes (FAK_NATIVE_ADMISSION).
const (
	// nativeAdmissionDefault reserves aggregate bytes AND retains the GPU lease
	// for the whole residency.
	nativeAdmissionDefault = "default"
	// nativeAdmissionAggregate reserves aggregate bytes and drops the GPU lease
	// after load when the model is small enough to coexist.
	nativeAdmissionAggregate = "aggregate"
	// nativeAdmissionExclusive is the conservative rollback: the GPU lease alone
	// serializes residency and no aggregate reservation is written.
	nativeAdmissionExclusive = "exclusive"
)

func resolveNativeAdmissionMode() string {
	switch strings.TrimSpace(os.Getenv("FAK_NATIVE_ADMISSION")) {
	case nativeAdmissionAggregate:
		return nativeAdmissionAggregate
	case nativeAdmissionExclusive:
		return nativeAdmissionExclusive
	}
	return nativeAdmissionDefault
}

// admitLocalMetalModel coordinates local native Metal memory admission. Before
// any model byte is allocated it combines the model's startup/steady memory
// plan, the current Darwin allocatable memory and pressure, and the active FAK
// reservations into one reservation transaction under the machine-wide GPU
// lease, and it emits a decision receipt at every lifecycle stage.
//
//   - An unreadable host sample, unknown or critical pressure, or an aggregate
//     that does not fit refuses before the loader (typed reason + remedy).
//   - The reservation holds the startup peak during load and drops to the
//     steady residency only after spec.Load reports success; a failed load
//     releases it without ever marking it steady.
//   - The returned release runs spec.Teardown first, then returns the
//     reservation and the lease, exactly once.
//
// FAK_NATIVE_ADMISSION=exclusive is the conservative rollback: the lease alone
// serializes residency for the whole serve and no reservation is written.
// FAK_NATIVE_ADMISSION=aggregate drops the lease after a small load so
// proven-small models can coexist; the default keeps the lease as well as the
// reservation.
func admitLocalMetalModel(useMetal bool, ggufPath string, opts gpulease.Options, spec metalAdmissionSpec) (release func(), err error) {
	if !useMetal || strings.TrimSpace(ggufPath) == "" {
		spec.Load()
		return func() {}, nil
	}

	mode := resolveNativeAdmissionMode()

	opts.NoWait = true
	opts.Timeout = 0
	lease, err := gpulease.Acquire(opts)
	if err != nil {
		path := opts.Path
		if path == "" {
			path = gpulease.DefaultPath()
		}
		if errors.Is(err, gpulease.ErrBusy) {
			return func() {}, metalLeaseRefusalError(err, path, gpuWaitBoundFromEnv())
		}
		return func() {}, fmt.Errorf("fak local launcher: acquire Metal residency lease %s before model load: %w", path, err)
	}

	store := localadmission.NewReservationStore(defaultLocalReservationDir())
	plan, stateEnvelope, err := estimateMetalModelMemoryBoundsWithEnvelope(ggufPath)
	if err != nil {
		lease.Release()
		return func() {}, fmt.Errorf("fak local launcher: Metal memory estimate refused before model load: %w", err)
	}

	rc := newMetalAdmissionReceipt(ggufPath, mode, plan, stateEnvelope)
	rc.out = spec.ReceiptOut
	var resID string
	retainLease := mode != nativeAdmissionAggregate

	host, hostKnown := serveAdmissionHostSample()
	rc.observeHost(host)
	if hostKnown && host.Pressure == localadmission.PressureWarning {
		compPct := float64(host.CompressedBytes) / float64(host.TotalBytes) * 100.0
		allocGiB := float64(host.AllocatableBytes) / (1 << 30)
		fmt.Fprintf(os.Stderr, "fak local launcher: advisory: ambient memory pressure is warning (compressed %.1f%%, %.2f GiB allocatable); close background apps if paging occurs\n", compPct, allocGiB)
	}

	if mode == nativeAdmissionExclusive {
		rc.LeaseRetained = true
		rc.stage("admit", "ADMIT", "exclusive_lease")
		rc.emit()
	} else {
		// An unreadable host sample is submitted as-is (unknown pressure, zero
		// capacity) so the store refuses it with a typed reason before the loader.
		req := localadmission.ReservationRequest{
			OwnerPID: os.Getpid(),
			Plan:     plan,
			Host:     host,
			Policy:   os.Getenv("FAK_ADMISSION_POLICY"),
		}
		if hostKnown {
			req.Host.AllocatableBytes = reservationAllocatable(host.AllocatableBytes, spec.FitFloor, memgate.Pressure(host.Pressure), host.TotalBytes)
		}
		if stateEnvelope != nil {
			req.Classes = memoryClassesToStrings(stateEnvelope.Classes())
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		dec, rerr := store.Reserve(ctx, req)
		cancel()
		if rerr != nil {
			lease.Release()
			return func() {}, fmt.Errorf("fak local launcher: local memory reservation error: %w", rerr)
		}
		rc.observeDecision(dec)
		if !dec.Admit {
			lease.Release()
			if dec.Reason != "aggregate_capacity" {
				// Early refusals never read the ledger, so the decision carries no
				// peer total; report what peers actually hold (best effort).
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				if reserved, err := store.TotalReservedBytes(ctx); err == nil {
					rc.ReservedBytes = reserved
					rc.AvailableBytes = max(rc.AllocatableBytes-reserved, 0)
				}
				cancel()
			}
			hint := reservationRefusalHint(dec, plan)
			rc.RemedyHint = hint
			rc.stage("refuse", "REJECT", dec.Reason)
			rc.emit()
			return func() {}, fmt.Errorf("fak local launcher: local memory reservation refused: %s (%s)", dec.Reason, hint)
		}
		if dec.Reservation != nil {
			resID = dec.Reservation.ID
		}
		if opts.Path != "" || plan.StartupPeakBytes > host.AllocatableBytes/2 || plan.SteadyBytes > host.AllocatableBytes/2 {
			retainLease = true
		}
		rc.LeaseRetained = retainLease
		rc.stage("admit", "ADMIT", dec.Reason)
		rc.emit()
	}

	releaseReservation := func(stage string) bool {
		if resID == "" {
			return true
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		err := store.Release(ctx, resID)
		cancel()
		if err != nil {
			// The row stays until its owner exits and is reaped: say so rather
			// than silently leaking capacity for the process lifetime.
			fmt.Fprintf(os.Stderr, "fak local launcher: %s: release memory reservation %s: %v (the dead-owner reap reclaims it after this process exits)\n", stage, resID, err)
			return false
		}
		return true
	}

	loaded := false
	defer func() {
		if !loaded {
			// A panicking loader never reaches steady: return the startup
			// reservation and the lease before the panic propagates.
			releaseReservation("aborted load")
			lease.Release()
		}
	}()
	ok := spec.Load()
	loaded = true
	if !ok {
		released := releaseReservation("failed load")
		lease.Release()
		failedStore := store
		if mode == nativeAdmissionExclusive {
			failedStore = nil
		}
		rc.observeRelease(failedStore, released)
		rc.stage("load_failed", "RELEASED", "load_failed")
		rc.emit()
		return func() {}, nil
	}
	if resID != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		r, err := store.MarkSteady(ctx, resID)
		cancel()
		if err != nil {
			// Failing to downshift keeps the startup peak held (fail-closed) but
			// starves peers, so it must be visible.
			fmt.Fprintf(os.Stderr, "fak local launcher: mark memory reservation %s steady: %v (the startup peak stays reserved)\n", resID, err)
		} else {
			rc.Phase = r.Phase
			// The held bytes dropped from the startup peak to the steady residency.
			rc.ReservedBytes -= r.StartupPeakBytes - r.HeldBytes
			rc.AvailableBytes = max(rc.AllocatableBytes-rc.ReservedBytes, 0)
			rc.stage("steady", "ADMIT", "steady")
			rc.emit()
		}
	}
	if !retainLease {
		lease.Release()
	}

	// Exclusive mode writes no reservation, so its receipts must not read (or
	// reap) the shared ledger either.
	ledgerStore := store
	if mode == nativeAdmissionExclusive {
		ledgerStore = nil
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			if spec.Teardown != nil {
				if err := spec.Teardown(); err != nil {
					rc.TeardownError = err.Error()
					var active *fakmodel.WeightSessionsActiveError
					if errors.As(err, &active) {
						// The weights stay resident until the last session closes, so
						// handing the capacity back now would let a peer admit against
						// bytes that are still allocated. Keep the reservation (and a
						// retained lease) until this process exits: the kernel drops the
						// lease flock and the dead-owner reap reclaims the ledger row.
						rc.Cleanup = "deferred_to_exit"
						rc.stage("release", "DEFERRED", "weight_sessions_active")
						rc.emit()
						return
					}
				}
			}
			released := releaseReservation("teardown")
			if retainLease {
				lease.Release()
			}
			rc.observeRelease(ledgerStore, released)
			rc.stage("release", "RELEASED", "teardown")
			rc.emit()
		})
	}, nil
}

// closeAdmittedModelWeights is the teardown an admitted Metal load runs before
// its reservation is released. CloseWeights is idempotent, so a later close on
// the same model (e.g. serve's closeEPGroup) is a no-op. When sessions are still
// attached, the free completes when the last one closes; the typed error is
// recorded on the release receipt.
func closeAdmittedModelWeights(m *fakmodel.Model) error {
	if m == nil {
		return nil
	}
	return m.CloseWeights()
}

// reservationRefusalHint returns the store's remedy hint, or a reason-accurate
// fallback. The capacity arithmetic is only quoted for an aggregate-capacity
// refusal; other reasons must not claim the plan "exceeds" anything.
func reservationRefusalHint(dec localadmission.ReservationDecision, plan localadmission.MemoryPlan) string {
	if dec.RemedyHint != "" {
		return dec.RemedyHint
	}
	if dec.Reason != "aggregate_capacity" {
		return fmt.Sprintf("reservation request not admissible (startup peak %d bytes, steady %d bytes)", plan.StartupPeakBytes, plan.SteadyBytes)
	}
	avail := dec.CapacityBytes - dec.ReservedBytes
	if avail < 0 {
		avail = 0
	}
	return fmt.Sprintf("requested startup peak %.2f GiB (steady %.2f GiB) exceeds available allocatable capacity %.2f GiB (active reservations %.2f GiB)",
		float64(dec.RequestedPeakBytes)/(1<<30),
		float64(plan.SteadyBytes)/(1<<30),
		float64(avail)/(1<<30),
		float64(dec.ReservedBytes)/(1<<30))
}

// loadLocalLauncherModelWithVulkanLease coordinates local native Vulkan memory admission,
// acquiring the canonical GPU lease before model allocation and retaining it across the
// entire serving lifetime. In-kernel Vulkan serving requires sole ownership of the
// machine-wide GPU lease so concurrent GPU-heavy workloads queue instead of stacking.
func loadLocalLauncherModelWithVulkanLease(useVulkan bool, ggufPath string, opts gpulease.Options, load func()) (release func(), err error) {
	if !useVulkan || ggufPath == "" {
		load()
		return func() {}, nil
	}

	opts.NoWait = true
	opts.Timeout = 0
	opts.Mode = gpulease.ModeExclusive
	opts.Shared = false
	lease, err := gpulease.Acquire(opts)
	if err != nil {
		path := opts.Path
		if path == "" {
			path = gpulease.DefaultPath()
		}
		if errors.Is(err, gpulease.ErrBusy) {
			rec := gpulease.DecideScaleOut(err, gpulease.ScaleOutOptions{Path: path, WaitBound: gpuWaitBoundFromEnv()})
			return func() {}, fmt.Errorf("fak local launcher: Vulkan residency admission refused before model load: %w; %s", err, rec.String())
		}
		return func() {}, fmt.Errorf("fak local launcher: acquire Vulkan residency lease %s before model load: %w", path, err)
	}

	loaded := false
	defer func() {
		if !loaded {
			lease.Release()
		}
	}()
	load()
	loaded = true

	var once sync.Once
	return func() {
		once.Do(func() {
			lease.Release()
		})
	}, nil
}

// loadServeModelWithVulkanLease is the cmdServe admission seam. Any backend
// selector that resolves to usable Vulkan, paired with an exact nonempty --gguf,
// acquires residency; --model is advertised/delegated identity and never
// substitutes for local bytes.
func loadServeModelWithVulkanLease(sf *serveFlags, opts gpulease.Options, load func()) (release func(), err error) {
	ggufPath := ""
	if sf != nil && sf.ggufPath != nil {
		ggufPath = *sf.ggufPath
	}
	return loadLocalLauncherModelWithVulkanLease(isServeVulkan(sf), ggufPath, opts, load)
}
