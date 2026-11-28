package compute

import "errors"

// ADAPT of OpenNMT/CTranslate2 (MIT), src/cpu/primitives.cc:1 @09e0165f68c8:
// int8 CPU GEMM blocking and B-matrix panel packing for quantized CPU decode.
// Component and host oracle only: no hardware or speed claim.

// Int8GEMMBlockConfig defines cache-blocking tile sizes for int8 CPU GEMM.
type Int8GEMMBlockConfig struct {
	MC int
	NC int
	KC int
}

// DefaultInt8GEMMBlockConfig returns standard L1/L2-friendly blocking sizes.
func DefaultInt8GEMMBlockConfig() Int8GEMMBlockConfig {
	return Int8GEMMBlockConfig{MC: 64, NC: 64, KC: 256}
}

// PackInt8B packs row-major B[k][n] into contiguous (KC x NC) panels for blocked GEMM.
func PackInt8B(b []int8, k, n int, cfg Int8GEMMBlockConfig) ([]int8, error) {
	if k <= 0 || n <= 0 || len(b) != k*n || cfg.KC <= 0 || cfg.NC <= 0 {
		return nil, errors.New("compute: invalid int8 B matrix or block config")
	}
	packed := make([]int8, 0, k*n)
	for jc := 0; jc < n; jc += cfg.NC {
		nc := min(cfg.NC, n-jc)
		for pc := 0; pc < k; pc += cfg.KC {
			kc := min(cfg.KC, k-pc)
			for p := 0; p < kc; p++ {
				row := (pc + p) * n
				packed = append(packed, b[row+jc:row+jc+nc]...)
			}
		}
	}
	return packed, nil
}

// BlockedInt8GEMM computes C = (A * B_packed) * rowScaleA * colScaleB using
// cache-blocked (MC, NC, KC) tiles.
func BlockedInt8GEMM(a []int8, packedB []int8, m, n, k int, rowScaleA, colScaleB []float32, cfg Int8GEMMBlockConfig) ([]float32, error) {
	if m <= 0 || n <= 0 || k <= 0 || len(a) != m*k || len(packedB) != k*n ||
		len(rowScaleA) != m || len(colScaleB) != n || cfg.MC <= 0 || cfg.NC <= 0 || cfg.KC <= 0 {
		return nil, errors.New("compute: invalid blocked int8 GEMM dimensions")
	}
	acc := make([]int32, m*n)
	bOffset := 0
	for jc := 0; jc < n; jc += cfg.NC {
		nc := min(cfg.NC, n-jc)
		for pc := 0; pc < k; pc += cfg.KC {
			kc := min(cfg.KC, k-pc)
			panel := packedB[bOffset : bOffset+kc*nc]
			bOffset += kc * nc
			for ic := 0; ic < m; ic += cfg.MC {
				mc := min(cfg.MC, m-ic)
				for i := 0; i < mc; i++ {
					aRow := a[(ic+i)*k+pc : (ic+i)*k+pc+kc]
					cRow := acc[(ic+i)*n+jc : (ic+i)*n+jc+nc]
					for p := 0; p < kc; p++ {
						av := int32(aRow[p])
						bRow := panel[p*nc : (p+1)*nc]
						for j := 0; j < nc; j++ {
							cRow[j] += av * int32(bRow[j])
						}
					}
				}
			}
		}
	}
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[i*n+j] = float32(acc[i*n+j]) * rowScaleA[i] * colScaleB[j]
		}
	}
	return out, nil
}
