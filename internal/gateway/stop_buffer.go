package gateway

import "github.com/anthony-chaudhary/fak/pkg/anthropicbridge"

// StopHoldbackBuffer is the incremental stop-string holdback buffer for streaming
// generation (#10719, borrowed from the vLLM detokenizer): it delays emitting up
// to max(len(stop)) - 1 characters until the stop condition resolves (and the
// stop sequence is trimmed) or generation finishes (and the held buffer is
// flushed), preventing partial stop markers from prematurely leaking across SSE
// chunk boundaries to streaming clients. The implementation lives in the public
// translator (pkg/anthropicbridge.StopHoldbackBuffer) — the exact code the routed
// /v1/messages arm runs — so both wires share one holdback semantic.
type StopHoldbackBuffer = anthropicbridge.StopHoldbackBuffer

// NewStopHoldbackBuffer creates an incremental holdback buffer for the specified stop sequences.
func NewStopHoldbackBuffer(stops []string) *StopHoldbackBuffer {
	return anthropicbridge.NewStopHoldbackBuffer(stops)
}
