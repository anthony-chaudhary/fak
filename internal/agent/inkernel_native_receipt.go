package agent

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

func validateNativeInferenceReceiptForwardCoverage(cfg model.Config, receipt *model.NativeInferenceReceipt) error {
	if receipt == nil {
		return nil
	}
	if len(receipt.ForwardCoverage) == 0 && !cfg.IsDeepSeekV41() {
		return nil
	}
	if err := model.ValidateNativeForwardCoverage(receipt.ForwardCoverage, len(receipt.TokenIDs)); err != nil {
		return err
	}
	for i := range receipt.ForwardCoverage {
		coverage := receipt.ForwardCoverage[i]
		if coverage.ModelType != cfg.ModelType {
			return fmt.Errorf("%w: output %d model_type %q != live model_type %q", model.ErrNativeForwardCoverageInvalid, i, coverage.ModelType, cfg.ModelType)
		}
		if coverage.ConfiguredLayers != cfg.NumLayers {
			return fmt.Errorf("%w: output %d configured_layers %d != live configured_layers %d", model.ErrNativeForwardCoverageInvalid, i, coverage.ConfiguredLayers, cfg.NumLayers)
		}
		if coverage.HeadDim != cfg.HeadDim {
			return fmt.Errorf("%w: output %d head_dim %d != live head_dim %d", model.ErrNativeForwardCoverageInvalid, i, coverage.HeadDim, cfg.HeadDim)
		}
	}
	return nil
}

func (p *InKernelPlanner) buildNativeInferenceReceipt(measurement *nativeInferenceMeasurement, prefillS, decodeS float64, accounting ...*model.NativeCacheAccounting) *model.NativeInferenceReceipt {
	backend, forwardPath := p.executionIdentity()
	nativeSelection, nativeSelectionDigest, _ := p.nativeSelectionIdentity()
	var qwen35MetalForwardSequence *model.Qwen35MetalForwardSequenceReceipt
	if measurement.qwen35MetalForwardSequence.EvidenceState != "" || measurement.qwen35MetalForwardSequence.Available {
		snapshot := measurement.qwen35MetalForwardSequence
		qwen35MetalForwardSequence = &snapshot
	}
	var qwen35MetalStateIdentity *model.Qwen35MetalStateIdentityReceipt
	if measurement.qwen35MetalStateIdentity != nil && measurement.qwen35MetalStateIdentity.Available {
		qwen35MetalStateIdentity = cloneQwen35MetalStateIdentityReceipt(*measurement.qwen35MetalStateIdentity)
	}
	var cudaImmutableWeightUploads *model.NativeCUDAImmutableWeightUploadDelta
	if after, ok := cudaImmutableWeightUploadSnapshot(p.backend); ok && measurement.cudaImmutableWeightUploadsAvailable {
		before := measurement.cudaImmutableWeightUploadsBefore
		if after.Calls >= before.Calls && after.TransferBytes >= before.TransferBytes && after.ResidentBytes >= before.ResidentBytes {
			cudaImmutableWeightUploads = &model.NativeCUDAImmutableWeightUploadDelta{
				Before: before,
				After:  after,
				Delta: model.NativeCUDAImmutableWeightUploadCounters{
					Calls:         after.Calls - before.Calls,
					TransferBytes: after.TransferBytes - before.TransferBytes,
					ResidentBytes: after.ResidentBytes - before.ResidentBytes,
				},
			}
		}
	}
	return &model.NativeInferenceReceipt{
		TokenIDs:                   append([]int(nil), measurement.tokenIDs...),
		TokenLogprobs:              append([]float64(nil), measurement.logprobs...),
		PrefillSeconds:             prefillS,
		TTFTSeconds:                measurement.ttftS,
		DecodeSeconds:              decodeS,
		Model:                      p.modelID,
		Engine:                     "inkernel",
		Planner:                    "inkernel",
		Owner:                      "fak",
		Backend:                    backend,
		ForwardPath:                forwardPath,
		Q4K:                        p.q4k,
		FallbackActive:             measurement.hostFallbackObserved || len(measurement.hostStages) > 0,
		HostPathBypass:             measurement.hostPathBypass,
		HostStages:                 append([]string(nil), measurement.hostStages...),
		PrefillChunkTokens:         p.nativeInferencePrefillChunkTokens(),
		NativeSelection:            nativeSelection,
		NativeSelectionDigest:      nativeSelectionDigest,
		ForwardCoverage:            append([]model.NativeForwardCoverage(nil), measurement.forwardCoverage...),
		Qwen35MetalForwardSequence: qwen35MetalForwardSequence,
		Qwen35MetalStateIdentity:   qwen35MetalStateIdentity,
		Qwen35SequencePrefillRoute: cloneNativeSequencePrefillRouteReceipt(measurement.qwen35SequencePrefillRoute),
		CUDAImmutableWeightUploads: cudaImmutableWeightUploads,
		CUDAUploadHostRetention:    cudaUploadHostRetentionSnapshot(p.backend),
		CUDASequenceLockProfile:    cudaSequenceLockProfileSnapshot(p.backend),
		NativeCacheAccounting:      firstNativeCacheAccounting(accounting),
	}
}

// firstNativeCacheAccounting returns the first non-nil accounting block from the
// optional variadic argument, or nil. It lets existing 3-argument call sites
// (tests and internal builders that predate #13339) compile unchanged while the
// served path attaches the accounting block.
func firstNativeCacheAccounting(accounting []*model.NativeCacheAccounting) *model.NativeCacheAccounting {
	for _, a := range accounting {
		if a != nil {
			return a
		}
	}
	return nil
}

// nativeCacheAccountingFacts carries the served-request facts the cache
// accounting block is derived from. They are the SAME values the served path
// already computes for the witness line and the provider-cache projection
// (cacheable from #3390's lookup-side match, matched from the realized serve),
// so the accounting block is a re-projection of observed numbers, never a
// second, independently-guessed source of truth.
type nativeCacheAccountingFacts struct {
	promptTokens    int
	cacheableTokens int
	matchedTokens   int
	sourceTier      radixkv.SnapshotTier
	prefillSecs     float64
	generated       int
}

// nativeCacheAccountingFor re-projects the served request's already-observed
// facts into the optional model.NativeCacheAccounting block (#13339). It
// separates the request's own restore from the startup preparation that made
// that restore possible: the block reports restored vs computed tokens and the
// restore duration of THIS request, and startup priming (a populate that
// generates nothing and never enters demand) is accounted by the warming path
// that performs it, not folded into a served turn.
//
// Conservation is structural: computed is prompt - matched and restored is the
// matched serve, so restored + computed == prompt by construction and the
// block passes model.NativeCacheAccounting.Validate() for every branch. A cold
// turn (no reuse) is the always-usable fallback and needs no cache state.
//
// The block is attached only when a native receipt was requested, so an
// ordinary receipt keeps its existing shape. It is NOT attached when nothing
// was prompt-presented or generated: an absent block reads as "no cache state
// to account", which is more honest than a fabricated zero-token record.
func (p *InKernelPlanner) nativeCacheAccountingFor(f nativeCacheAccountingFacts) *model.NativeCacheAccounting {
	prompt := f.promptTokens
	if prompt < 0 {
		prompt = 0
	}
	matched := f.matchedTokens
	if matched < 0 {
		matched = 0
	}
	if matched > prompt {
		matched = prompt
	}
	cacheable := f.cacheableTokens
	if cacheable < 0 {
		cacheable = 0
	}
	if cacheable > prompt {
		cacheable = prompt
	}
	if cacheable < matched {
		cacheable = matched
	}
	if prompt == 0 && f.generated == 0 {
		// Nothing was prompt-presented or generated: leave the block absent
		// rather than emit a zero-token record that implies a decision.
		return nil
	}
	computed := prompt - matched
	src := nativeCacheRestoreSourceFor(f.sourceTier, matched)

	acct := &model.NativeCacheAccounting{
		TotalPromptTokens: prompt,
		StableTokens:      cacheable,
		CachedTokens:      cacheable,
		RestoredTokens:    matched,
		ComputedTokens:    computed,
		RestoreSource:     src,
		GeneratedTokens:   f.generated,
	}
	if matched > 0 {
		acct.RestoreDurationSecs = f.prefillSecs
	}
	switch {
	case matched == 0:
		acct.Operation = model.NativeCacheCold
		acct.RestoreSource = model.NativeCacheRestoreNone
		acct.RestoreDurationSecs = 0
	case matched == prompt:
		acct.Operation = model.NativeCacheExactHit
	default:
		acct.Operation = model.NativeCachePartialHit
	}
	return acct
}

// nativeCacheRestoreSourceFor maps a radix snapshot tier onto the closed
// restore-source vocabulary. A miss (or any non-positive match) is "none";
// device-resident state is "device"; every host-side and remote transfer plane
// is "host" (it crossed the host boundary, not the device-resident L1 cache).
func nativeCacheRestoreSourceFor(tier radixkv.SnapshotTier, matched int) model.NativeCacheRestoreSource {
	if matched <= 0 {
		return model.NativeCacheRestoreNone
	}
	switch tier {
	case radixkv.SnapshotTierDeviceL1:
		return model.NativeCacheRestoreDevice
	case radixkv.SnapshotTierHostL2, radixkv.SnapshotTierRemoteL3, radixkv.SnapshotTierLocalL3:
		return model.NativeCacheRestoreHost
	default:
		// A live match under an unknown tier cannot be claimed as a device hit;
		// leave it source-less so the block reads as an un-attributed restore
		// rather than an optimistic device claim.
		return model.NativeCacheRestoreNone
	}
}

func (p *InKernelPlanner) nativeSelectionIdentity() (model.NativeSelectionIdentity, string, error) {
	backend, forwardPath := p.executionIdentity()
	quantization := p.nativeSelectionQuantization()
	identity := model.NativeSelectionIdentity{
		Schema:              model.NativeSelectionIdentitySchemaV1,
		ModelRef:            p.modelID,
		Backend:             backend,
		ForwardPath:         forwardPath,
		Quantization:        quantization,
		PrefillChunkTokens:  p.nativeInferencePrefillChunkTokens(),
		CPUOffloadExperts:   p.nativeSelectionCPUOffloadExperts(),
		Q4KGateUpOutputSlab: p.q4kGateUpOutputSlab && quantization == model.NativeSelectionQuantizationQ4K,
	}
	digest, err := identity.Digest()
	if err != nil {
		return model.NativeSelectionIdentity{}, "", err
	}
	return identity, digest, nil
}

func (p *InKernelPlanner) nativeSelectionQuantization() string {
	if p != nil && p.q4k {
		// Resident Q2_0 and Prism PQ2_0 reuse Session.Q4K's mixed-quant
		// execution path. The attached Prism contract distinguishes the two
		// on-disk formats after both reach the same packed Q2 weight store.
		if p.m != nil && p.m.Q2Count() > 0 {
			if p.m.HasPrismHadamard() {
				return model.NativeSelectionQuantizationPQ2_0
			}
			return model.NativeSelectionQuantizationQ2_0
		}
		return model.NativeSelectionQuantizationQ4K
	}
	if p != nil && p.quant {
		return model.NativeSelectionQuantizationQ8_0
	}
	return model.NativeSelectionQuantizationF32
}

func (p *InKernelPlanner) nativeSelectionCPUOffloadExperts() int {
	if p == nil {
		return 0
	}
	if p.expertSpill != nil {
		return p.expertSpill.Fit.SpillLayers
	}
	if p.cpuOffloadExperts && p.m != nil {
		return len(p.m.MoEExpertLayers())
	}
	return 0
}

func cloneQwen35MetalStateIdentityReceipt(src model.Qwen35MetalStateIdentityReceipt) *model.Qwen35MetalStateIdentityReceipt {
	src.States = append([]model.Qwen35MetalStateDigest(nil), src.States...)
	return &src
}

type cudaImmutableWeightUploadSnapshotter interface {
	CUDAImmutableWeightUploadSnapshot() (calls, transferBytes, residentBytes uint64)
}

func cudaImmutableWeightUploadSnapshot(be compute.Backend) (model.NativeCUDAImmutableWeightUploadCounters, bool) {
	provider, ok := be.(cudaImmutableWeightUploadSnapshotter)
	if !ok {
		return model.NativeCUDAImmutableWeightUploadCounters{}, false
	}
	calls, transferBytes, residentBytes := provider.CUDAImmutableWeightUploadSnapshot()
	return model.NativeCUDAImmutableWeightUploadCounters{Calls: calls, TransferBytes: transferBytes, ResidentBytes: residentBytes}, true
}

type cudaUploadHostRetentionSnapshotter interface {
	CUDAUploadHostRetentionSnapshot() (entries, payloadViewBytes, scaleViewBytes uint64)
}

// Only buildNativeInferenceReceipt calls this observer. Ordinary completions
// neither walk the CUDA upload cache nor acquire its mutex for this diagnostic.
func cudaUploadHostRetentionSnapshot(be compute.Backend) *model.NativeCUDAUploadHostRetention {
	provider, ok := be.(cudaUploadHostRetentionSnapshotter)
	if !ok {
		return nil
	}
	entries, payload, scales := provider.CUDAUploadHostRetentionSnapshot()
	return &model.NativeCUDAUploadHostRetention{
		Scope:                "process_cuda_upload_cache",
		Accounting:           "logical_views_per_cache_entry",
		Backing:              "unknown",
		EntriesWithHostOwner: entries,
		PayloadViewBytes:     payload,
		ScaleViewBytes:       scales,
	}
}

type cudaSequenceLockProfileSnapshotter interface {
	CUDAQwen35SequenceLockProfileSnapshot() (completed, waitNS, holdNS uint64, available bool)
}

// Only the requested native receipt builder reads this optional observer.
// The backend's existing timestamp profiler must already be enabled; this
// consumer never enables profiling or changes CUDA execution policy.
func cudaSequenceLockProfileSnapshot(be compute.Backend) *model.NativeCUDASequenceLockProfile {
	provider, ok := be.(cudaSequenceLockProfileSnapshotter)
	if !ok {
		return nil
	}
	completed, waitNS, holdNS, available := provider.CUDAQwen35SequenceLockProfileSnapshot()
	if !available {
		return nil
	}
	return &model.NativeCUDASequenceLockProfile{
		Scope:                 "process_cuda_qwen35_sequence_prefill",
		Accounting:            "cumulative_completed_acquisitions",
		Gate:                  "cuda_timestamp_profile",
		CompletedAcquisitions: completed,
		WaitNS:                waitNS,
		HoldNS:                holdNS,
	}
}
