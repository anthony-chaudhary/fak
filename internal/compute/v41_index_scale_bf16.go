package compute

// V41IndexScaleBF16Backend optionally performs a separately rounded binary32
// multiply followed by the ordinary finite-only BF16 publication at fixed site 25.
// Capability refusal precedes effects. Selected failures are fatal, never replayed.
// Output and its single 16-byte status record are fresh caller-owned allocations,
// including partial results returned with errors. Input remains borrowed until
// completion. Status uses V41DeferredBF16Publish, Count N and fixed
// V41BF16PublishIndexerScaledWeight; the invalid index is local to the input row.
// Exact F32 RNE, subnormal and signed-zero preservation are required. NaN/Inf
// products and finite BF16 overflow produce a fault and no published output.
type V41IndexScaleBF16Backend interface {
	SupportsV41IndexScaleBF16(input Tensor, scale float32) bool
	V41IndexScaleBF16Deferred(input Tensor, scale float32) (Tensor, V41DeferredStatus, error)
}
