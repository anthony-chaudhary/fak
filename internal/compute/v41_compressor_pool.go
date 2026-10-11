package compute

import (
	"fmt"
	"math"
)

// V41CompressorPoolBackend is the optional DeepSeek V4.1 compressor gated pool
// (Ref #13668, #13812). It transcribes the host V41CompressorPool.Push group
// emission for ratio >= 2: per latent dimension d, a softmax over the ratio token
// rows of scores[:,d] gates kv[:,d],
//
//	m = max_t scores[t,d]
//	w_t = exp(scores[t,d] - m), denom = serial sum w_t
//	pooled[d] = serial sum kv[t,d] * (w_t / denom)
//
// in token order with separately rounded binary32 products, sums and correctly
// rounded quotients. exp is a Cody-Waite binary32 evaluation within about one
// ulp of the host's F64-narrowed exp, so pooled values track the host within a
// few ulps; the BF16 cast and learned RMSNorm tail stay with the compressor norm.
//
// kv and scores are finite unquantized row-major F32 [ratio,width] tensors on
// this backend. The result is a fresh caller-owned F32 [width] tensor. Absence
// permits host selection; after dispatch is selected, an error never permits
// host replay.
type V41CompressorPoolBackend interface {
	SupportsV41CompressorPool() bool
	V41CompressorPool(kv, scores Tensor, ratio, width int) (Tensor, error)
}

// V41CompressorPoolMaxRatio bounds the serial per-dimension token loop.
const V41CompressorPoolMaxRatio = 1 << 12

func validateV41CompressorPool(kv, scores Tensor, ratio, width int) error {
	if ratio < 2 || ratio > V41CompressorPoolMaxRatio || width <= 0 {
		return fmt.Errorf("compute: V4.1 compressor pool ratio=%d width=%d outside the gated domain", ratio, width)
	}
	if uint64(ratio)*uint64(width) > math.MaxInt32 {
		return fmt.Errorf("compute: V4.1 compressor pool ratio=%d width=%d exceeds the shader index range", ratio, width)
	}
	for _, operand := range []struct {
		name string
		t    Tensor
	}{{"kv", kv}, {"scores", scores}} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || operand.t.Numel() != ratio*width {
			return fmt.Errorf("compute: V4.1 compressor pool %s must be an unquantized row-major F32 [%d,%d] tensor", operand.name, ratio, width)
		}
	}
	return nil
}

func vulkanV41CompressorPoolStatusError(status int, reason string) *BackendError {
	err := vulkanV41IndexerScoreStatusError(status, reason)
	err.Site = "V41CompressorPool"
	err.Message = fmt.Sprintf("dedicated compressor pool failed closed (code %d): %s", status, reason)
	return err
}
