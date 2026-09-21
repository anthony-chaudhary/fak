package model

import "errors"

// NativeCacheOperation names what a request did with the native KV cache. It is
// a closed vocabulary: an emitter either declares one of these or omits the
// accounting block entirely, and an unknown value is refused rather than mapped
// onto a nearby known one.
type NativeCacheOperation string

const (
	// NativeCacheCold is a request that restored nothing and computed the whole
	// prompt. It is the always-usable fallback and needs no cache state.
	NativeCacheCold NativeCacheOperation = "cold"
	// NativeCacheExactHit restored the entire prompt from a prepared prefix.
	NativeCacheExactHit NativeCacheOperation = "exact_hit"
	// NativeCachePartialHit restored a prefix and computed the remaining tail.
	NativeCachePartialHit NativeCacheOperation = "partial_hit"
	// NativeCacheStartupPrime is cache preparation that runs before any demand:
	// it may compute prompt tokens but must generate none, so it can never be
	// read as a throughput sample.
	NativeCacheStartupPrime NativeCacheOperation = "startup_prime"
)

// NativeCacheRestoreSource names where restored prefix state came from. It is
// deliberately a small closed set so a receipt cannot imply a reuse it did not
// perform.
type NativeCacheRestoreSource string

const (
	// NativeCacheRestoreNone is the cold path: nothing was restored.
	NativeCacheRestoreNone NativeCacheRestoreSource = "none"
	// NativeCacheRestoreHost is a host-side prepared prefix.
	NativeCacheRestoreHost NativeCacheRestoreSource = "host"
	// NativeCacheRestoreDevice is a device-resident prepared prefix.
	NativeCacheRestoreDevice NativeCacheRestoreSource = "device"
)

// ErrNativeCacheAccountingInvalid is returned when a cache accounting record
// does not conserve its token counts, carries a negative counter, or violates
// the zero-work rule for startup priming. It is a typed refusal so callers can
// distinguish a malformed receipt from an unsupported one.
var ErrNativeCacheAccountingInvalid = errors.New("model: native cache accounting invalid")

// NativeCacheAccounting is the optional, additive accounting block for a native
// inference receipt. It records what one request did with the prepared cache
// without changing any runtime emitter: the fields are populated by callers
// (anthony-chaudhary/fak#13339), and an absent block preserves byte-for-byte
// receipt compatibility for every existing producer.
//
// Conservation: for every operation, RestoredTokens + ComputedTokens must equal
// TotalPromptTokens, and CachedTokens must be <= TotalPromptTokens across the
// whole block. The state counts are per-request, not cumulative.
type NativeCacheAccounting struct {
	Operation NativeCacheOperation `json:"operation"`
	// TotalPromptTokens is the full prompt length for this request, whether the
	// tokens were restored or computed.
	TotalPromptTokens int `json:"total_prompt_tokens"`
	// StableTokens is the length of the stable (cacheable) prefix the plan
	// targets. CachedTokens is how much of the prompt was actually held as
	// reusable cache state. RestoredTokens is how much this request restored and
	// ComputedTokens how much it computed; they must sum to TotalPromptTokens.
	StableTokens   int `json:"stable_tokens"`
	CachedTokens   int `json:"cached_tokens"`
	RestoredTokens int `json:"restored_tokens"`
	ComputedTokens int `json:"computed_tokens"`
	// RestoreSource is "none" on the cold path and names the plane otherwise.
	RestoreSource       NativeCacheRestoreSource `json:"restore_source"`
	RestoreDurationSecs float64                  `json:"restore_duration_seconds"`
	// PrimeDurationSecs is reported separately from restore so startup
	// preparation cost is never hidden inside a request-restore figure.
	PrimeDurationSecs float64 `json:"prime_duration_seconds"`
	// GeneratedTokens is carried here only to enforce the startup-prime
	// zero-work rule; it is the same count the parent receipt reports and is
	// never a second source of truth.
	GeneratedTokens int `json:"generated_tokens"`
}

// Validate refuses an accounting block that does not conserve its tokens, holds
// a negative counter, names an unknown operation/source, or claims startup
// priming while generating tokens. A nil receiver is valid: absent accounting
// is compatible with every existing receipt.
func (a *NativeCacheAccounting) Validate() error {
	if a == nil {
		return nil
	}
	switch a.Operation {
	case NativeCacheCold, NativeCacheExactHit, NativeCachePartialHit, NativeCacheStartupPrime:
	default:
		return errNativeCache(a, "unknown operation")
	}
	switch a.RestoreSource {
	case NativeCacheRestoreNone, NativeCacheRestoreHost, NativeCacheRestoreDevice:
	default:
		return errNativeCache(a, "unknown restore source")
	}
	for _, c := range []struct {
		name  string
		value int
	}{
		{"total_prompt_tokens", a.TotalPromptTokens},
		{"stable_tokens", a.StableTokens},
		{"cached_tokens", a.CachedTokens},
		{"restored_tokens", a.RestoredTokens},
		{"computed_tokens", a.ComputedTokens},
		{"generated_tokens", a.GeneratedTokens},
	} {
		if c.value < 0 {
			return errNativeCache(a, "negative "+c.name)
		}
	}
	if a.RestoreDurationSecs < 0 || a.PrimeDurationSecs < 0 {
		return errNativeCache(a, "negative duration")
	}
	if a.RestoredTokens+a.ComputedTokens != a.TotalPromptTokens {
		return errNativeCache(a, "restored+computed != total prompt")
	}
	if a.CachedTokens > a.TotalPromptTokens {
		return errNativeCache(a, "cached tokens exceed total prompt")
	}
	if a.RestoredTokens > a.CachedTokens {
		return errNativeCache(a, "restored tokens exceed cached tokens")
	}
	if a.StableTokens > a.TotalPromptTokens {
		return errNativeCache(a, "stable tokens exceed total prompt")
	}
	switch a.Operation {
	case NativeCacheCold:
		if a.RestoredTokens != 0 || a.RestoreSource != NativeCacheRestoreNone {
			return errNativeCache(a, "cold operation restored state")
		}
	case NativeCacheExactHit:
		if a.ComputedTokens != 0 || a.RestoredTokens != a.TotalPromptTokens {
			return errNativeCache(a, "exact hit did not restore the whole prompt")
		}
		if a.RestoreSource == NativeCacheRestoreNone {
			return errNativeCache(a, "exact hit without a restore source")
		}
	case NativeCachePartialHit:
		if a.RestoredTokens == 0 || a.ComputedTokens == 0 {
			return errNativeCache(a, "partial hit must restore and compute")
		}
		if a.RestoreSource == NativeCacheRestoreNone {
			return errNativeCache(a, "partial hit without a restore source")
		}
	case NativeCacheStartupPrime:
		if a.GeneratedTokens != 0 {
			return errNativeCache(a, "startup prime generated tokens")
		}
	}
	return nil
}

// SatisfiesDemandThroughput reports whether the block may be read as a demand
// throughput sample. Startup priming is preparation, not demand: even though it
// computes prompt tokens, it must never be counted as a served request.
func (a *NativeCacheAccounting) SatisfiesDemandThroughput() bool {
	if a == nil {
		return false
	}
	return a.Operation != NativeCacheStartupPrime
}

func errNativeCache(a *NativeCacheAccounting, why string) error {
	return &NativeCacheAccountingError{Operation: string(a.Operation), Reason: why}
}

// NativeCacheAccountingError is the typed carrier for ErrNativeCacheAccountingInvalid.
// It keeps the refused operation and the specific conservation rule that failed
// so a caller reports a named refusal instead of an opaque error.
type NativeCacheAccountingError struct {
	Operation string
	Reason    string
}

func (e *NativeCacheAccountingError) Error() string {
	return "native cache accounting " + e.Operation + ": " + e.Reason
}

func (e *NativeCacheAccountingError) Unwrap() error { return ErrNativeCacheAccountingInvalid }

// NativeInferenceReceipt binds generated tokens and their normalized chosen-token
// log probabilities to the native model execution which produced them. Logprobs
// are log_softmax over the unmodified model logits, so receipt capture is supported
// only for greedy sampling without bias or repetition penalties.
type NativeInferenceReceipt struct {
	TokenIDs              []int                   `json:"token_ids"`
	TokenLogprobs         []float64               `json:"token_logprobs"`
	PromptTokenIDs        []int                   `json:"prompt_token_ids,omitempty"`
	TokenizerID           string                  `json:"tokenizer_id,omitempty"`
	RendererID            string                  `json:"renderer_id,omitempty"`
	RenderedSHA256        string                  `json:"rendered_sha256,omitempty"`
	PrefillSeconds        float64                 `json:"prefill_seconds"`
	TTFTSeconds           float64                 `json:"ttft_seconds"`
	DecodeSeconds         float64                 `json:"decode_seconds"`
	Model                 string                  `json:"model"`
	Engine                string                  `json:"engine"`
	Planner               string                  `json:"planner"`
	Owner                 string                  `json:"owner"`
	Backend               string                  `json:"backend"`
	ForwardPath           string                  `json:"forward_path"`
	Q4K                   bool                    `json:"q4k"`
	FallbackActive        bool                    `json:"fallback_active"`
	PrefillChunkTokens    int                     `json:"prefill_chunk_tokens"`
	NativeSelection       NativeSelectionIdentity `json:"kernel_selection"`
	NativeSelectionDigest string                  `json:"kernel_selection_digest"`
	// Qwen35MetalForwardSequence is present only when the model session produced
	// terminal evidence for the native whole-sequence Metal graph.
	Qwen35MetalForwardSequence *Qwen35MetalForwardSequenceReceipt `json:"qwen35_metal_forward_sequence,omitempty"`
	// Qwen35MetalStateIdentity is present only for an explicitly requested native
	// receipt whose fresh exact-P32 Metal session sealed model-owned state identity.
	Qwen35MetalStateIdentity   *Qwen35MetalStateIdentityReceipt      `json:"qwen35_metal_state_identity,omitempty"`
	Qwen35SequencePrefillRoute *NativeSequencePrefillRouteReceipt    `json:"qwen35_sequence_prefill_route,omitempty"`
	CUDAImmutableWeightUploads *NativeCUDAImmutableWeightUploadDelta `json:"cuda_immutable_weight_uploads,omitempty"`
	// NativeCacheAccounting is present only when the producer recorded what the
	// request did with the prepared native cache. Absent (nil) preserves the
	// existing receipt shape for every producer that predates the field.
	NativeCacheAccounting *NativeCacheAccounting `json:"native_cache_accounting,omitempty"`
}

// NativeSequencePrefillRouteReceipt aggregates the route decisions observed for
// every prefill call executed by one request. Status is the representative
// observed decision: the first fallback or other non-qualifying decision is
// sticky. Its packed-row fields describe only that represented call, not all
// calls in a mixed request. NativePerformanceQualifying summarizes only those
// model route decisions; it does not claim numerical parity or hardware
// qualification.
type NativeSequencePrefillRouteReceipt struct {
	PrefillCalls                int                               `json:"prefill_calls"`
	ObservedCalls               int                               `json:"observed_calls"`
	Complete                    bool                              `json:"complete"`
	NativePerformanceQualifying bool                              `json:"native_performance_qualifying"`
	Status                      *Qwen35SequencePrefillRouteStatus `json:"status,omitempty"`
}

// NativeCUDAImmutableWeightUploadCounters is one cumulative CUDA-backend
// snapshot. TransferBytes counts actual host payload crossing H2D; ResidentBytes
// counts the resulting device layout, including quant scale sidecars.
type NativeCUDAImmutableWeightUploadCounters struct {
	Calls         uint64 `json:"calls"`
	TransferBytes uint64 `json:"transfer_bytes"`
	ResidentBytes uint64 `json:"resident_bytes"`
}

// NativeCUDAImmutableWeightUploadDelta is the request's cumulative-backend
// observation window. Delta is exact for the parent's deliberately serialized
// campaign; concurrent requests may contribute to the same backend window.
type NativeCUDAImmutableWeightUploadDelta struct {
	Before NativeCUDAImmutableWeightUploadCounters `json:"before"`
	After  NativeCUDAImmutableWeightUploadCounters `json:"after"`
	Delta  NativeCUDAImmutableWeightUploadCounters `json:"delta"`
}

// NativeInferenceReceiptUnsupportedError is returned before model execution when
// receipt semantics would be ambiguous. The receipt contract is intentionally
// narrower than ordinary sampling rather than guessing what modified logits mean.
type NativeInferenceReceiptUnsupportedError struct {
	Reason string
}

func (e *NativeInferenceReceiptUnsupportedError) Error() string {
	return "native inference receipt unsupported: " + e.Reason
}

// InKernelQwenQ4KPrefillChunkConfigError is retained by a planner when the
// bounded Qwen resident-Q4_K prefill control is outside the supported range.
// Targeted requests return it before tokenization or model
// execution; unrelated model paths do not acquire a new failure mode.
type InKernelQwenQ4KPrefillChunkConfigError struct {
	Value string
}

func (e *InKernelQwenQ4KPrefillChunkConfigError) Error() string {
	return "native Qwen Q4_K prefill chunk tokens " + e.Value + ": want 128..8192 (set with --native-qwen-q4k-prefill-chunk-tokens)"
}
