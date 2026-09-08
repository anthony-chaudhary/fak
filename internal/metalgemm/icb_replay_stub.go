//go:build !(darwin && arm64 && cgo)

package metalgemm

// ICBDecodeMode defines the orchestration mechanism for the autoregressive decode pass.
type ICBDecodeMode int

const (
	// DecodeModeDirectMultiCB executes multiple command buffers per step with per-op/block host synchronization.
	DecodeModeDirectMultiCB ICBDecodeMode = 0
	// DecodeModeDirectOneCB executes a single command buffer per step with direct CPU-encoded dispatches.
	DecodeModeDirectOneCB ICBDecodeMode = 1
	// DecodeModeICBReplay executes transformer decode layers via pre-allocated Indirect Command Buffers (ICB).
	DecodeModeICBReplay ICBDecodeMode = 2
)

// ICBDecodeReceipt records execution facts and hardware timing witnessed during a decode step.
type ICBDecodeReceipt struct {
	CommandBuffers   int           `json:"command_buffers"`
	Encoders         int           `json:"encoders"`
	ICBDispatches    int           `json:"icb_dispatches"`
	ContiguousBlocks int           `json:"contiguous_blocks"`
	HostEncodeMs     float64       `json:"host_encode_ms"`
	HostWaitMs       float64       `json:"host_wait_ms"`
	GPUMs            float64       `json:"gpu_ms"`
	TotalMs          float64       `json:"total_ms"`
	ICBUsed          bool          `json:"icb_used"`
	Mode             ICBDecodeMode `json:"mode"`
}

// SetDecodeDispatchMode sets the active decode dispatch mode in stub builds.
func SetDecodeDispatchMode(mode ICBDecodeMode) {}

// GetDecodeDispatchMode returns the default decode dispatch mode in stub builds.
func GetDecodeDispatchMode() ICBDecodeMode { return DecodeModeDirectOneCB }

// SetICBDecodeEnabled toggles ICB decode replay in stub builds.
func SetICBDecodeEnabled(enabled bool) {}

// ICBDecodeSupported reports false in non-Metal or non-darwin builds.
func ICBDecodeSupported() bool { return false }

// DecodeStepWithReceipt is unavailable in stub builds.
func DecodeStepWithReceipt(xEmbed, Kctx, Vctx []float32, L, nLayers, w, H, vocab int, seed bool) (lastPre, newKpost, newV, logits []float32, receipt ICBDecodeReceipt, ok bool) {
	return nil, nil, nil, nil, ICBDecodeReceipt{}, false
}

// DecodeStepWithMode is unavailable in stub builds.
func DecodeStepWithMode(mode ICBDecodeMode, xEmbed, Kctx, Vctx []float32, L, nLayers, w, H, vocab int, seed bool) (lastPre, newKpost, newV, logits []float32, receipt ICBDecodeReceipt, ok bool) {
	return nil, nil, nil, nil, ICBDecodeReceipt{}, false
}
