package compute

import (
	"fmt"
	"math"
	"math/bits"
	"slices"
)

// V41ExactRMSNormMaxWidth bounds the serial per-row reduction of the exact
// RMSNorm kernel. V4.1 binds it at the compressor (512) and index-key (128) widths.
const V41ExactRMSNormMaxWidth = 1 << 16

// V41ExactRMSNormBackend is the optional bit-exact RMSNorm the V4.1 compressor
// and index-key norms bind under device-only execution (Ref #13668). Each row
// equals norm.SerialRMSNorm bit for bit: an in-order binary32 sum of separately
// rounded squares, the mean as that sum times the exact reciprocal of a
// power-of-two width, plus eps, then float32(1/sqrt(float64(mean+eps))), and
// (x*inv)*w per lane. Products and sums are binary32 round-to-nearest-even with
// denormals preserved; there is no FMA, reassociation or tree reduction. The
// generic Backend.RMSNorm keeps its faster reduction and makes no such promise.
//
// x is an unquantized row-major F32 [rows, width] (or [width]) tensor and weight
// an F32 [width] tensor on this backend. The output is a fresh caller-owned
// tensor shaped like x. A selected failure propagates without CPU replay.
type V41ExactRMSNormBackend interface {
	// SupportsV41ExactRMSNorm reports whether width/eps is inside the exact
	// domain on this backend: width a power of two up to the bound, eps a finite
	// positive normal binary32, and the exact module and float controls loaded.
	SupportsV41ExactRMSNorm(width int, eps float32) bool
	V41ExactRMSNorm(x, weight Tensor, eps float32) (Tensor, error)
}

// V41ExactRMSNormDomain is the backend-independent half of the qualification:
// the mean is an exact power-of-two scaling only for these widths, and a normal
// eps keeps mean+eps normal so the scalar inverse never sees zero.
func V41ExactRMSNormDomain(width int, eps float32) bool {
	return width > 0 && width <= V41ExactRMSNormMaxWidth && bits.OnesCount(uint(width)) == 1 &&
		!math.IsNaN(float64(eps)) && !math.IsInf(float64(eps), 0) && eps >= math.SmallestNonzeroFloat32*(1<<23)
}

// validateV41ExactRMSNorm checks geometry and dtypes before any native call.
// It neither reads device data nor allocates.
func validateV41ExactRMSNorm(x, weight Tensor, eps float32) (rows, width int, err error) {
	if weight.Dtype != F32 || weight.Layout != RowMajor || weight.Quant != nil || len(weight.Shape) != 1 {
		return 0, 0, fmt.Errorf("compute: V4.1 exact RMSNorm weight must be an unquantized row-major F32 vector")
	}
	width = weight.Shape[0]
	if !V41ExactRMSNormDomain(width, eps) {
		return 0, 0, fmt.Errorf("compute: V4.1 exact RMSNorm width=%d eps=%g is outside the exact domain", width, eps)
	}
	if x.Dtype != F32 || x.Layout != RowMajor || x.Quant != nil || len(x.Shape) == 0 || len(x.Shape) > 2 || x.Shape[len(x.Shape)-1] != width {
		return 0, 0, fmt.Errorf("compute: V4.1 exact RMSNorm input must be unquantized row-major F32 [rows,%d]", width)
	}
	rows = 1
	if len(x.Shape) == 2 {
		rows = x.Shape[0]
	}
	if rows <= 0 || uint64(rows)*uint64(width) > math.MaxInt32 {
		return 0, 0, fmt.Errorf("compute: V4.1 exact RMSNorm rows=%d exceeds the shader index range", rows)
	}
	if slices.Contains(x.Shape, 0) {
		return 0, 0, fmt.Errorf("compute: V4.1 exact RMSNorm input is empty")
	}
	return rows, width, nil
}

func vulkanV41ExactRMSNormStatusError(status int, reason string) *BackendError {
	err := vulkanV41IndexerScoreStatusError(status, reason)
	err.Site = "V41ExactRMSNorm"
	err.Message = fmt.Sprintf("dedicated exact RMSNorm failed closed (code %d): %s", status, reason)
	return err
}
