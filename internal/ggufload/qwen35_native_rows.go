package ggufload

import (
	"fmt"
	"os"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// qwen35_native_rows.go — native K-quant residency for the Qwen3.5-family projections whose
// canonical normalization only permutes OUTPUT rows (fak#13567).
//
// normalizeCanonicalTensorData reorders linear_attn.in_proj_qkv/z/a/b and unpermutes the
// self_attn q/k rotary layout. Every one of those transforms moves whole rows (row width =
// hidden size) and leaves each row's values untouched, so applying it to raw K-quant rows (each
// an independent run of 256-weight super-blocks) yields exactly the bytes that dequantize to the
// normalized tensor — no requantization. The row map is DERIVED from the float normalizer itself
// (qwen35CanonicalRowSource) so there is one copy of the permutation math.

// metalQ5KNativeRows reports whether this host's resident Metal graph can encode raw Q5_K rows
// (q5k_mul_mv / q5k_mul_mm). It is a var so tests can pin either host.
var metalQ5KNativeRows = func() bool {
	return runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" && metalgemm.Q5KAvailable()
}

// qwen35NativeRowResident is the ONE predicate, shared by computeQ4KTensorWork and
// EstimateQ4KLoadMemoryPlan, deciding whether a tensor takes the native-row route instead of
// dequant -> normalize -> Q8. canon may be the loader's pre-resolution canonical name or the
// estimator's resolved one (model.Qwen35RowNormalizedProjection resolves idempotently).
//
// Admitted: Q4_K always (resident q4kw), Q6_K only when the options retain dense Q6_K (kqw; a
// backend without a Q6_K kernel keeps the proven Q8 path exactly as for any dense Q6_K weight),
// and Q5_K only where the Metal Q5_K graph kernels run (Apple Silicon) and dense k-quants are
// retained. Every other type stays on Q8. The resident Metal decode graph encodes each admitted
// type natively, and the Q8 band shrinks to what stays Q8. The exact 64-layer Qwen3.8 runtime
// requires these weights as Q8 and is excluded; WithNativeLinearAttnRows(false) forces it off.
func qwen35NativeRowResident(cfg model.Config, canon string, t TensorType, shape []int, o q4kLoadOptions) bool {
	if o.nativeRowsOff {
		return false
	}
	if qwen35DeviceQKNativeRow(cfg, canon, t, shape, o) {
		return true
	}
	switch t {
	case TensorQ4_K:
	case TensorQ6_K:
		if !denseKQuantRetained(o, t) {
			return false
		}
	case TensorQ5_K:
		if !denseKQuantRetained(o, t) || !metalQ5KNativeRows() {
			return false
		}
	default:
		return false
	}
	// Every admitted transform reorders rows of width cfg.HiddenSize; the row map below is
	// derived at that width, so any other reduction dim is outside the proven contract.
	if len(shape) != 2 || shape[0] <= 0 || shape[1] != cfg.HiddenSize || shape[1]%qkK != 0 {
		return false
	}
	if model.Qwen38ExactMetalQ8Runtime(cfg) {
		return false
	}
	return model.Qwen35RowNormalizedProjection(cfg, canon)
}

// qwen35CanonicalRowSource returns src such that canonical row dst is GGUF row src[dst]. It runs
// the production float normalizer on a one-column probe whose value is the row index (the cfg
// copy sets HiddenSize — the row width every admitted transform uses — to 1), then proves the
// result is a permutation, so a transform that ever stops being row-only fails closed here
// instead of silently corrupting raw rows.
func qwen35CanonicalRowSource(canon string, rows int, cfg model.Config) ([]int, error) {
	if rows <= 0 || rows > 1<<24 { // float32 represents every row index exactly below 2^24
		return nil, fmt.Errorf("gguf: tensor %s has %d rows, outside the exact row-probe range", canon, rows)
	}
	probeCfg := cfg
	probeCfg.HiddenSize = 1
	probe := make([]float32, rows)
	for i := range probe {
		probe[i] = float32(i)
	}
	got, err := normalizeCanonicalTensorData(canon, probe, probeCfg)
	if err != nil {
		return nil, err
	}
	if len(got) != rows {
		return nil, fmt.Errorf("gguf: tensor %s row probe returned %d rows, want %d", canon, len(got), rows)
	}
	src := make([]int, rows)
	seen := make([]bool, rows)
	for dst, v := range got {
		r := int(v)
		if float32(r) != v || r < 0 || r >= rows || seen[r] {
			return nil, fmt.Errorf("gguf: tensor %s normalization is not a pure row permutation (row %d -> %v)", canon, dst, v)
		}
		seen[r] = true
		src[dst] = r
	}
	return src, nil
}

// permuteRawRows moves whole raw rows into canonical order (dst row i = src row src[i]). An
// identity map (e.g. nV == nK, or a k_proj stored in HF layout) returns raw unchanged.
func permuteRawRows(canon string, raw []byte, src []int) ([]byte, error) {
	rows := len(src)
	if rows == 0 || len(raw)%rows != 0 {
		return nil, fmt.Errorf("gguf: tensor %s has %d bytes, not divisible into %d rows", canon, len(raw), rows)
	}
	identity := true
	for dst, s := range src {
		if dst != s {
			identity = false
			break
		}
	}
	if identity {
		return raw, nil
	}
	rowBytes := len(raw) / rows
	dst := make([]byte, len(raw))
	for row, s := range src {
		copy(dst[row*rowBytes:(row+1)*rowBytes], raw[s*rowBytes:(s+1)*rowBytes])
	}
	return dst, nil
}

// normalizeQwen35NativeRows reorders a raw K-quant payload into canonical row order.
func normalizeQwen35NativeRows(canon string, shape []int, raw []byte, cfg model.Config) ([]byte, error) {
	src, err := qwen35CanonicalRowSource(canon, shape[0], cfg)
	if err != nil {
		return nil, err
	}
	return permuteRawRows(canon, raw, src)
}

// qwen35DeviceQKNativeRow admits the full-attention q_proj/k_proj rows and the GDN
// in_proj_qkv/in_proj_z rows of a raw-i-quant device backend (Vulkan iq*_matvec, ticket 05f) in
// their native format, including the exact Qwen3.8 topology: its Q8 requirement belongs to the
// Metal runtime, which never retains raw i-quants. The type must be one the backend already keeps
// packed; a raw in-proj makes the fused Q8 GDN in-proj decline to its composed MatMul path.
// FAK_QWEN35_NATIVE_QK_ROWS=0 / FAK_QWEN35_NATIVE_GDN_INPROJ_ROWS=0 restore Q8 per class.
func qwen35DeviceQKNativeRow(cfg model.Config, canon string, t TensorType, shape []int, o q4kLoadOptions) bool {
	if len(o.residentDenseIQ) == 0 {
		return false
	}
	if t != TensorQ4_K && !denseKQuantRetained(o, t) {
		return false
	}
	if bw, _, ok := residentExpertBlockGeometry(t); !ok || bw != qkK {
		return false
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] != cfg.HiddenSize || shape[1]%qkK != 0 {
		return false
	}
	switch {
	case model.Qwen35FullAttentionQKProjection(cfg, canon):
		return os.Getenv("FAK_QWEN35_NATIVE_QK_ROWS") != "0"
	case model.Qwen35GDNInProjQKVZProjection(cfg, canon):
		return os.Getenv("FAK_QWEN35_NATIVE_GDN_INPROJ_ROWS") != "0"
	}
	return false
}
