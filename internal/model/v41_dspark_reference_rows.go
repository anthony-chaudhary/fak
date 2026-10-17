package model

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// v41DSparkRows joins the already validated original-format draft inventories.
// It borrows immutable shards; its caller keeps them open until the reference
// closes. No whole tensor or checkpoint expansion is retained.
type v41DSparkRows struct {
	inv    *V41DSparkInventory
	shards map[string]*V41DSparkPackedShard
}

func newV41DSparkRows(cfg Config, shards []*V41DSparkPackedShard) (*v41DSparkRows, error) {
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		return nil, err
	}
	w := &v41DSparkRows{inv: inv, shards: map[string]*V41DSparkPackedShard{}}
	for _, shard := range shards {
		if shard == nil {
			return nil, fmt.Errorf("%w: nil draft shard", errV41DSparkReference)
		}
		err := func() error {
			shard.mu.RLock()
			defer shard.mu.RUnlock()
			if shard.closed {
				return fmt.Errorf("%w: closed draft shard", errV41DSparkReference)
			}
			for name := range shard.sources {
				spec, ok := inv.Lookup(name)
				ref, err := shard.referenceLocked(name)
				if !ok || err != nil || !sameShape(spec.Shape, ref.Ref.Shape) || !sameShape(spec.LogicalShape, ref.Ref.LogicalShape) {
					return fmt.Errorf("%w: incompatible draft tensor %s", errV41DSparkReference, name)
				}
				if _, exists := w.shards[name]; exists {
					return fmt.Errorf("%w: duplicate draft tensor %s", errV41DSparkReference, name)
				}
				w.shards[name] = shard
			}
			return nil
		}()
		if err != nil {
			return nil, err
		}
	}
	for _, name := range inv.Names() {
		if w.shards[name] == nil {
			return nil, fmt.Errorf("%w: missing draft tensor %s", errV41DSparkReference, name)
		}
	}
	return w, nil
}

const v41DSparkReferenceRowBytes = 256 << 10

// row reads one logical matrix row (a vector is row zero). The lock spans both
// payload and scale reads. A short/error read publishes no decoded row.
func (w *v41DSparkRows) row(ctx context.Context, name string, row int) ([]float32, error) {
	if ctx == nil || w == nil || w.inv == nil {
		return nil, fmt.Errorf("%w: missing row reader/context", errV41DSparkReference)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	spec, ok := w.inv.Lookup(name)
	shard := w.shards[name]
	if !ok || shard == nil {
		return nil, fmt.Errorf("%w: unknown draft row %s", errV41DSparkReference, name)
	}
	rows, width := 1, spec.LogicalShape[0]
	if len(spec.LogicalShape) == 2 {
		rows, width = spec.LogicalShape[0], spec.LogicalShape[1]
	}
	if row < 0 || row >= rows || width <= 0 || width > v41DSparkReferenceRowBytes/4 {
		return nil, fmt.Errorf("%w: invalid bounded row %s[%d]", errV41DSparkReference, name, row)
	}
	shard.mu.RLock()
	defer shard.mu.RUnlock()
	ref, err := shard.referenceLocked(name)
	if err != nil {
		return nil, err
	}
	bytesPerRow := width
	switch ref.Ref.Dtype {
	case "F32":
		bytesPerRow *= 4
	case "BF16":
		bytesPerRow *= 2
	case "I8":
		bytesPerRow /= 2
	case "F8_E4M3":
	default:
		return nil, fmt.Errorf("%w: unsupported draft dtype %s", errV41DSparkReference, ref.Ref.Dtype)
	}
	read := func(start, count, end int64) ([]byte, error) {
		if count <= 0 || count > v41DSparkReferenceRowBytes || start < 0 || start > end-count {
			return nil, fmt.Errorf("%w: draft row span out of bounds", errV41DSparkReference)
		}
		b := make([]byte, int(count))
		n, err := shard.packed.file.r.ReadAt(b, start)
		if err == nil && n != len(b) {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			return nil, fmt.Errorf("DSpark row %s: %w", name, err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return b, nil
	}
	raw, err := read(ref.Start+int64(row)*int64(bytesPerRow), int64(bytesPerRow), ref.End)
	if err != nil {
		return nil, err
	}
	var scales []byte
	if ref.HasScale() {
		scaleWidth := spec.ScaleShape[1]
		scaleRow := row
		if ref.Ref.Dtype == "F8_E4M3" {
			scaleRow /= 32
		}
		scales, err = read(ref.ScaleStart+int64(scaleRow)*int64(scaleWidth), int64(scaleWidth), ref.ScaleEnd)
		if err != nil {
			return nil, err
		}
		for _, scale := range scales {
			if scale == 255 {
				return nil, fmt.Errorf("%w: NaN E8M0 scale %s", errV41DSparkReference, name)
			}
		}
	}
	out := make([]float32, width)
	for k := range out {
		switch ref.Ref.Dtype {
		case "F32":
			out[k] = math.Float32frombits(binary.LittleEndian.Uint32(raw[k*4:]))
		case "BF16":
			out[k] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[k*2:])) << 16)
		case "F8_E4M3":
			out[k] = fp8E4M3ToF32(raw[k]) * float32(math.Ldexp(1, int(scales[k/32])-127))
		case "I8":
			nibble := raw[k/2] & 15
			if k%2 != 0 {
				nibble = raw[k/2] >> 4
			}
			out[k] = float32(math.Ldexp(float64(v4ExpertE2M1Values[nibble]), int(scales[k/32])-127))
		}
		if !finite32(out[k]) {
			return nil, fmt.Errorf("%w: non-finite draft row %s[%d,%d]", errV41DSparkReference, name, row, k)
		}
	}
	return out, nil
}

// v41DSparkActFP8 is the native scalar act_quant reference: E4M3 with an
// E8M0 power-of-two scale per 32 values, amax floor 1e-4, ties-to-even.
// This preserves the original draft activation format independently of target
// NVFP4. It does not claim GPU GEMM accumulation or transcendental bit parity.
func v41DSparkActFP8(x []float32) ([]float32, error) {
	if len(x) == 0 || len(x)%32 != 0 {
		return nil, fmt.Errorf("%w: FP8 activation width", errV41DSparkReference)
	}
	out := make([]float32, len(x))
	for start := 0; start < len(x); start += 32 {
		amax := float32(1e-4)
		for _, value := range x[start : start+32] {
			if !finite32(value) {
				return nil, fmt.Errorf("%w: non-finite FP8 input", errV41DSparkReference)
			}
			amax = max(amax, float32(math.Abs(float64(value))))
		}
		ratio := float32(amax * float32(1.0/448.0))
		bits := math.Float32bits(ratio)
		exponent := int((bits>>23)&255) - 127
		if bits&((1<<23)-1) != 0 {
			exponent++
		}
		scale := float32(math.Ldexp(1, exponent))
		if !finite32(scale) || scale <= 0 {
			return nil, fmt.Errorf("%w: invalid FP8 activation scale", errV41DSparkReference)
		}
		for j, value := range x[start : start+32] {
			v := float32(value / scale)
			sign := float32(1)
			if math.Signbit(float64(v)) {
				sign = -1
				v = -v
			}
			v = min(v, float32(448))
			lo, hi := 0, 126
			for lo < hi {
				mid := (lo + hi) / 2
				if fp8E4M3ToF32(byte(mid)) < v {
					lo = mid + 1
				} else {
					hi = mid
				}
			}
			code := lo
			if code > 0 {
				upper, lower := fp8E4M3ToF32(byte(code)), fp8E4M3ToF32(byte(code-1))
				du, dl := float32(upper-v), float32(v-lower)
				if dl < du || dl == du && (code-1)%2 == 0 {
					code--
				}
			}
			out[start+j] = v41RoundBF16(sign * fp8E4M3ToF32(byte(code)) * scale)
			if !finite32(out[start+j]) {
				return nil, fmt.Errorf("%w: non-finite FP8 output", errV41DSparkReference)
			}
		}
	}
	return out, nil
}

func (f *v41DSparkReference) linear(ctx context.Context, name string, x []float32, rounded bool) ([]float32, error) {
	spec, ok := f.w.inv.Lookup(name)
	if !ok || len(spec.LogicalShape) != 2 || len(x) != spec.LogicalShape[1] {
		return nil, fmt.Errorf("%w: linear shape %s", errV41DSparkReference, name)
	}
	for _, v := range x {
		if !finite32(v) {
			return nil, fmt.Errorf("%w: linear input %s", errV41DSparkReference, name)
		}
	}
	var err error
	if spec.PackedWeight || spec.AllowedDtypes[0] == "F8_E4M3" {
		x, err = v41DSparkActFP8(x)
		if err != nil {
			return nil, err
		}
	}
	out := make([]float32, spec.LogicalShape[0])
	for row := range out {
		weight, err := f.w.row(ctx, name, row)
		if err != nil {
			return nil, err
		}
		var sum float32
		for k, v := range weight {
			sum += float32(v * x[k])
		}
		if rounded {
			sum = v41RoundBF16(sum)
		}
		if !finite32(sum) {
			return nil, fmt.Errorf("%w: linear output %s", errV41DSparkReference, name)
		}
		out[row] = sum
	}
	return out, nil
}

func (f *v41DSparkReference) norm(ctx context.Context, name string, x []float32) ([]float32, error) {
	weight, err := f.w.row(ctx, name, 0)
	if err != nil {
		return nil, err
	}
	if len(weight) != len(x) || len(x) == 0 {
		return nil, fmt.Errorf("%w: norm shape %s", errV41DSparkReference, name)
	}
	for _, v := range x {
		if !finite32(v) {
			return nil, fmt.Errorf("%w: norm input %s", errV41DSparkReference, name)
		}
	}
	return v41MHCBF16(-1, rmsnorm(x, weight, float32(f.cfg.RMSNormEps)))
}
