package model

// NativeInferenceReceipt binds generated tokens and their normalized chosen-token
// log probabilities to the native model execution which produced them. Logprobs
// are log_softmax over the unmodified model logits, so receipt capture is supported
// only for greedy sampling without bias or repetition penalties.
type NativeInferenceReceipt struct {
	TokenIDs              []int                   `json:"token_ids"`
	TokenLogprobs         []float64               `json:"token_logprobs"`
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
	CUDAImmutableWeightUploads *NativeCUDAImmutableWeightUploadDelta `json:"cuda_immutable_weight_uploads,omitempty"`
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
