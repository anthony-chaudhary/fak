package compute

import "errors"

// V41FP8WKVWeights owns the resident immutable E4M3 WKV pair. A failed Release
// retains its recovery handle; callers must not reuse uncertain device bytes.
type V41FP8WKVWeights interface{ Release() error }

// V41FP8WKVBackend exposes an explicit native WKV operation, not ordinary model
// admission. Supported means an actual image/device can be selected; Qualified
// separately requires returned physical numerical evidence. Input is contiguous
// device F32 [M,5120], M a positive multiple of128; output is fresh BF16 [M,512].
type V41FP8WKVBackend interface {
	Backend
	V41FP8WKVSupported() bool
	V41FP8WKVQualified() bool
	UploadV41FP8WKV(codes, tileScales []byte) (V41FP8WKVWeights, error)
	V41FP8WKV(V41FP8WKVWeights, Tensor, int) (Tensor, error)
}

var ErrV41FP8WKVUnavailable = errors.New("V4.1 FP8 WKV lacks qualified native admission")

// packV41WKVScales repeats each32-row tile and packs four UE8M0 bytes per
// column-major word. This is DeepGEMM's MN-major layout, not cuBLASLt scales.
func packV41WKVScales(raw []byte) ([]byte, error) {
	if len(raw) != 16*160 {
		return nil, errors.New("WKV scale grid must be16x160")
	}
	for _, v := range raw {
		if v == 255 {
			return nil, errors.New("WKV scale NaN")
		}
	}
	out := make([]byte, 512*160)
	for row := 0; row < 512; row++ {
		for k := 0; k < 160; k++ {
			out[((k/4)*512+row)*4+k%4] = raw[(row/32)*160+k]
		}
	}
	return out, nil
}
