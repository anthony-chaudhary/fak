package model

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41CompressorPoolFunc owns one completed compressor group's gated pool (Refs
// #13668, #13812): kv and scores are the group's [ratio, width] projected rows
// in token order, and the result is the raw pooled row before the BF16 cast and
// learned RMSNorm tail. It has no decline outcome: once selected, any failure
// closes the session instead of replaying the softmax on the host.
type v41CompressorPoolFunc func(layer int, kv, scores []float32, ratio, width int) ([]float32, error)

var errV41CompressorPoolResult = errors.New("model: invalid selected V4.1 device compressor pool result")

func v41CompressorPoolOperationErr(layer int, cause error) error {
	if cause == nil {
		cause = errV41CompressorPoolResult
	}
	return &V41ProjectionOperationError{
		Layer: layer, Leaf: "attn.compressor.wgate.weight", Stage: string(v41StageCompress),
		Cause: v41StageErr(v41StageCompress, layer, cause),
	}
}

// v41CompressorPoolFunc binds the gated pool when the backend exposes the
// dedicated operation. Each call uploads the group (2*ratio*width*4 bytes),
// dispatches once and reads back one width-wide row.
func (s *Session) v41CompressorPoolFunc() v41CompressorPoolFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	op, ok := s.Backend.(compute.V41CompressorPoolBackend)
	if !ok || !op.SupportsV41CompressorPool() {
		return nil
	}
	return func(layer int, kv, scores []float32, ratio, width int) ([]float32, error) {
		pooled, err := v41SelectedDeviceCall(s, layer, "v41-compressor-pool", func(stage *string) ([]float32, error) {
			n, ok := checkedMulInt(ratio, width)
			if !ok || ratio < 2 || width <= 0 || len(kv) != n || len(scores) != n {
				return nil, errV41CompressorPoolResult
			}
			*stage = "upload"
			kt := s.uploadHostF32([]int{ratio, width}, kv, compute.MemoryActivation, "V4.1 compressor pool kv")
			defer s.Backend.Free(kt)
			st := s.uploadHostF32([]int{ratio, width}, scores, compute.MemoryActivation, "V4.1 compressor pool scores")
			defer s.Backend.Free(st)
			*stage = "pool"
			out, err := op.V41CompressorPool(kt, st, ratio, width)
			if out.Buf() != nil && out.Buf() != kt.Buf() && out.Buf() != st.Buf() {
				defer s.Backend.Free(out)
			}
			if err != nil {
				return nil, err
			}
			if !v41MHCFreshOutput(s, out, []int{width}, kt, st) {
				return nil, errV41CompressorPoolResult
			}
			*stage = "readback"
			return s.v41DeviceSumReadback(out, width, errV41CompressorPoolResult)
		})
		if err != nil {
			return nil, v41CompressorPoolOperationErr(layer, err)
		}
		return pooled, nil
	}
}

// bindGatedPool routes c's completed ratio >= 2 groups through pool. A ratio-one
// pool has no gate, so it stays unbound.
func (c *V41CompressorPool) bindGatedPool(layer int, pool v41CompressorPoolFunc) {
	if c == nil || pool == nil || c.ratio < 2 {
		return
	}
	ratio, width := c.ratio, c.width
	c.gatedPool = func(kv, scores []float32) ([]float32, error) {
		return pool(layer, kv, scores, ratio, width)
	}
}
