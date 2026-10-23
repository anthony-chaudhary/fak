package compute

import (
	"bytes"
	"errors"
	"fmt"
	"math"
)

// ErrRoutedExpertResidentReleased is the typed refusal of a GEMV or Release on a
// RoutedExpertResidentWeight that was already released: its device bytes are gone.
var ErrRoutedExpertResidentReleased = errors.New("compute: routed-expert resident weight was released")

// ErrRoutedExpertResidentStale is the typed refusal of a GEMV on a resident weight uploaded
// before a device fault the session has since observed: a reconstructed context does not vouch
// for bytes that predate the fault, so the caller must Release and upload again.
var ErrRoutedExpertResidentStale = errors.New("compute: routed-expert resident weight predates a device fault")

// ErrExpertDeviceKernelAbsent is the fail-closed verdict a routed-expert device
// matmul returns when the target Backend carries no FP8/MXFP4 expert kernel. It is
// deliberately TYPED and returned (never a scalar fallback): the expert tier binds
// decode throughput, so a roofline caller must be able to distinguish "ran on the
// device" from "no device kernel exists" and refuse rather than silently degrade to
// the memory-bound scalar loop. errors.Is(err, ErrExpertDeviceKernelAbsent) pins it.
var ErrExpertDeviceKernelAbsent = errors.New("compute: routed-expert FP8/MXFP4 device kernel is absent")

// RoutedExpertDeviceKernel is the OPTIONAL backend capability a device (CUDA/Metal/
// ROCm) implements to serve routed-expert FP8/MXFP4 GEMVs natively. A backend that
// does not implement it is refused by GemvFP8Block32ExpertDevice /
// GemvMXFP4ExpertDevice with ErrExpertDeviceKernelAbsent — never scalar-fallback.
//
// The interface is intentionally byte-in/byte-out: weight bytes stay PACKED (E4M3 for
// FP8, I8-packed E2M1 nibbles for MXFP4) and the kernel decodes inline against the
// per-32-block E8M0 scale, so installing a resident expert never materializes a full
// dequantized f32 matrix.
type RoutedExpertDeviceKernel interface {
	// GemvExpertFP8Block32E8M0 computes y[o] = sum_i decode(weight[o,i], scale[o/32,i/32]) * x[i]
	// for one routed-expert projection resident on the device. weight holds O*I packed
	// E4M3 bytes, scales holds ceil(O/32)*ceil(I/32) raw E8M0 bytes, x has I entries,
	// and the returned slice has O entries. err is non-nil (fail closed) on any shape,
	// dtype or device fault; a nil error with a short y is never returned.
	GemvExpertFP8Block32E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error)
	// GemvExpertMXFP4E8M0 computes y[o] = sum_i w[o,i] * x[i] for one
	// routed-expert projection with unpacked shape [O,I]. O must be positive,
	// and I must be positive and even.
	// weight holds O rows of I/2 packed I8 bytes: byte j holds E2M1 codes
	// for w[o,2*j] in its low nibble and w[o,2*j+1] in its high nibble.
	// scales holds O*ceil(I/32) raw E8M0 bytes in row-major [O,ceil(I/32)]
	// order; scales[o*ceil(I/32)+i/32] scales w[o,i]. This is one scale per
	// row and 32 unpacked K values, not the FP8 method's 32x32 tile layout.
	// x has I entries, and the returned slice has O entries. Fail-closed as above.
	GemvExpertMXFP4E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error)
}

// RoutedExpertResidentKernel is the OPTIONAL resident-weight companion of
// RoutedExpertDeviceKernel: a routed expert's packed weight and E8M0 scale bytes are
// uploaded ONCE and every later GEMV moves only x and y, instead of paying the packed
// weight H2D on each call. The layouts are exactly those of the RoutedExpertDeviceKernel
// method of the same format, and upload applies that method's acceptance predicate minus
// x, so an upload plus a GEMV accepts and refuses exactly what the scalar oracle does and
// returns bit-identical outputs.
type RoutedExpertResidentKernel interface {
	// UploadExpertFP8Block32E8M0 makes one FP8 block-32 E8M0 projection resident
	// (layout of GemvExpertFP8Block32E8M0). The caller owns the handle and must Release it.
	UploadExpertFP8Block32E8M0(O, I int, weight, scales []byte) (RoutedExpertResidentWeight, error)
	// UploadExpertMXFP4E8M0 makes one MXFP4 E8M0 projection resident (layout of
	// GemvExpertMXFP4E8M0). The caller owns the handle and must Release it.
	UploadExpertMXFP4E8M0(O, I int, weight, scales []byte) (RoutedExpertResidentWeight, error)
}

// RoutedExpertResidentWeight is a typed handle to one device-resident routed-expert
// projection. Its lifetime is explicit: nothing frees the device bytes but Release.
type RoutedExpertResidentWeight interface {
	// Gemv returns y[o] = sum_i w[o,i] * x[i] over the resident bytes. x must hold I
	// entries. It fails closed, never with a short y: after Release with
	// ErrRoutedExpertResidentReleased, and once the session has recovered from a device
	// fault observed after the upload with ErrRoutedExpertResidentStale.
	Gemv(x []float32) ([]float32, error)
	// Release frees the device bytes. A second Release returns
	// ErrRoutedExpertResidentReleased.
	Release() error
}

// fp8_block32.go - the DeepSeek-V4.1 dense FP8 block-scale reference for the compute HAL.
// V4.1 replaces V4-Flash's 128x128 weight_scale_inv tile with a 32x32 edge, so a single
// 2-D [O,I] float8_e4m3fn weight carries ceil(O/32)*ceil(I/32) companion scales. This is
// the compute-side sibling of internal/model's decodeFP8BlockScale (the 128x128 V4-Flash
// path): model imports compute, not the reverse, so the HAL must own its own block decode.
// Two scale dtypes ship in the wild - a widened f32 `weight_scale_inv` and the raw
// F8_E8M0 exponent byte (one octet per tile, value 2^(b-127)) - so both decodes land here
// as separate, fail-closed functions. No global dequantization: GemvFP8Block32E8M0 decodes
// inline per element so a streaming engine never materializes the full f32 matrix.

// FP8Block32Dim is the DeepSeek-V4.1 dense FP8 block edge: one scale accompanies each
// FP8Block32Dim x FP8Block32Dim tile of a 2-D weight.
const FP8Block32Dim = 32

func fp8Block32ScaleShape(O, I int) (sO, sI int) {
	return (O + FP8Block32Dim - 1) / FP8Block32Dim, (I + FP8Block32Dim - 1) / FP8Block32Dim
}

// fp8Block32Product returns a*b with an overflow guard, so a hostile shape cannot wrap
// into a small positive element count and slip past the length checks below.
func fp8Block32Product(a, b int) (int, bool) {
	if a <= 0 || b <= 0 {
		return 0, false
	}
	p := a * b
	if p/b != a || p <= 0 {
		return 0, false
	}
	return p, true
}

// decodeE8M0Scale decodes one raw F8_E8M0 exponent byte as 2^(b-127). Byte 0xff is the
// E8M0 NaN encoding and is rejected; every other byte yields an exactly representable
// power of two, so the only failure mode is the NaN slot.
func decodeE8M0Scale(b byte) (float32, error) {
	if b == 0xff {
		return 0, fmt.Errorf("e8m0 scale byte 0xff is NaN")
	}
	v := float32(math.Ldexp(1, int(b)-127))
	if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
		return 0, fmt.Errorf("e8m0 scale byte %#02x decodes to a non-finite value", b)
	}
	return v, nil
}

// DecodeFP8Block32 dequantizes a row-major [O,I] float8_e4m3fn weight stored with one f32
// scale per FP8Block32Dim x FP8Block32Dim output/input tile. scaleInv holds one f32 per
// tile in row-major [ceil(O/FP8Block32Dim), ceil(I/FP8Block32Dim)] order - the widened
// `weight_scale_inv` companion tensor a DeepSeek-V4.1 FP8 checkpoint ships. Ragged final
// tiles are cropped by integer division (o/32, i/32), so the same layout covers an
// O or I that is not a multiple of the block edge.
//
// Fail-closed: a non-positive dimension, a weight length that disagrees with [O,I], or a
// scale length that disagrees with the block grid is a named error, never a silent
// mis-load. name is used only for error context.
func DecodeFP8Block32(name string, O, I int, weight []byte, scaleInv []float32) ([]float32, error) {
	if O <= 0 || I <= 0 {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] has a non-positive dimension", name, O, I)
	}
	elems, ok := fp8Block32Product(O, I)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] overflows element count", name, O, I)
	}
	if len(weight) != elems {
		return nil, fmt.Errorf("fp8 block32 %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	scaleElems, ok := fp8Block32Product(sO, sI)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: scale shape [%d %d] overflows element count", name, sO, sI)
	}
	if len(scaleInv) != scaleElems {
		return nil, fmt.Errorf("fp8 block32 %s: scaleInv has %d entries, shape [%d %d] (blocks of %d) implies %d", name, len(scaleInv), O, I, FP8Block32Dim, scaleElems)
	}
	out := make([]float32, elems)
	for o := 0; o < O; o++ {
		scaleRow := (o / FP8Block32Dim) * sI
		for i := 0; i < I; i++ {
			out[o*I+i] = DecodeE4M3(weight[o*I+i]) * scaleInv[scaleRow+i/FP8Block32Dim]
		}
	}
	return out, nil
}

// DecodeFP8Block32E8M0 dequantizes the same row-major [O,I] float8_e4m3fn / 32x32-tile
// layout as DecodeFP8Block32, but the scales are raw F8_E8M0 bytes - one exponent octet
// per tile, decoded as 2^(b-127). A byte of 0xff (E8M0 NaN), a weight byte decoding to
// NaN/Inf (the E4M3 NaN slots 0x7f/0xff), or any non-finite product is refused. All
// metadata and lengths are validated BEFORE the output is allocated, so a malformed pair
// fails before any large allocation.
func DecodeFP8Block32E8M0(name string, O, I int, weight, scales []byte) ([]float32, error) {
	if O <= 0 || I <= 0 {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] has a non-positive dimension", name, O, I)
	}
	elems, ok := fp8Block32Product(O, I)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: shape [%d %d] overflows element count", name, O, I)
	}
	if len(weight) != elems {
		return nil, fmt.Errorf("fp8 block32 %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	scaleElems, ok := fp8Block32Product(sO, sI)
	if !ok {
		return nil, fmt.Errorf("fp8 block32 %s: scale shape [%d %d] overflows element count", name, sO, sI)
	}
	if len(scales) != scaleElems {
		return nil, fmt.Errorf("fp8 block32 %s: scales has %d bytes, shape [%d %d] (blocks of %d) implies %d", name, len(scales), O, I, FP8Block32Dim, scaleElems)
	}
	scaleF := make([]float32, scaleElems)
	for k, b := range scales {
		v, err := decodeE8M0Scale(b)
		if err != nil {
			return nil, fmt.Errorf("fp8 block32 %s: %w", name, err)
		}
		scaleF[k] = v
	}
	for k, b := range weight {
		d := DecodeE4M3(b)
		if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
			return nil, fmt.Errorf("fp8 block32 %s: weight byte %d (%#02x) decodes to a non-finite value", name, k, b)
		}
	}
	out := make([]float32, elems)
	for o := 0; o < O; o++ {
		scaleRow := (o / FP8Block32Dim) * sI
		for i := 0; i < I; i++ {
			out[o*I+i] = DecodeE4M3(weight[o*I+i]) * scaleF[scaleRow+i/FP8Block32Dim]
		}
	}
	return out, nil
}

// e4m3NaNByte reports whether b is a float8_e4m3fn NaN encoding (0x7f or 0xff). E4M3
// has no Inf, so this is exactly the byte set on which DecodeE4M3 is non-finite; an
// exhaustive 256-byte witness holds the two to the same set.
func e4m3NaNByte(b byte) bool { return b&0x7f == 0x7f }

// firstE4M3NaNByte returns the index of the first E4M3 NaN byte in w, or -1. It is the
// vectorized form of scanning DecodeE4M3 over every byte: a 16 MiB expert is checked in
// two IndexByte passes instead of 16M float64 decodes.
func firstE4M3NaNByte(w []byte) int {
	k := bytes.IndexByte(w, 0x7f)
	limit := w
	if k >= 0 {
		limit = w[:k]
	}
	if j := bytes.IndexByte(limit, 0xff); j >= 0 {
		return j
	}
	return k
}

// firstE8M0ScaleError returns the decodeE8M0Scale refusal for the first scale byte it
// refuses, or nil. decodeE8M0Scale refuses exactly byte 0xff (an exhaustive witness pins
// that), so the scan is one IndexByte and the error is still decodeE8M0Scale's own.
func firstE8M0ScaleError(scales []byte) error {
	k := bytes.IndexByte(scales, 0xff)
	if k < 0 {
		return nil
	}
	_, err := decodeE8M0Scale(scales[k])
	return err
}

// checkGemvFP8Block32E8M0 is the single acceptance predicate of a routed-expert FP8
// block-32 E8M0 GEMV: the scalar oracle GemvFP8Block32E8M0 and every device kernel call
// it, so a device accepts and refuses exactly the inputs the oracle does. It allocates
// nothing and returns the scale-grid column count sI.
func checkGemvFP8Block32E8M0(name string, O, I int, weight, scales []byte, x []float32) (int, error) {
	return checkFP8Block32E8M0(name, O, I, weight, scales, x, true)
}

// checkFP8Block32E8M0Weight is checkGemvFP8Block32E8M0 without the x clause: the predicate a
// resident upload applies once, leaving routedExpertXError for each later GEMV.
func checkFP8Block32E8M0Weight(name string, O, I int, weight, scales []byte) (int, error) {
	return checkFP8Block32E8M0(name, O, I, weight, scales, nil, false)
}

// routedExpertXError is the x-length clause both acceptance predicates share.
func routedExpertXError(format, name string, O, I int, x []float32) error {
	if len(x) != I {
		return fmt.Errorf("%s %s: x has %d entries, shape [%d %d] implies %d", format, name, len(x), O, I, I)
	}
	return nil
}

func checkFP8Block32E8M0(name string, O, I int, weight, scales []byte, x []float32, withX bool) (int, error) {
	if O <= 0 || I <= 0 {
		return 0, fmt.Errorf("fp8 block32 %s: shape [%d %d] has a non-positive dimension", name, O, I)
	}
	elems, ok := fp8Block32Product(O, I)
	if !ok {
		return 0, fmt.Errorf("fp8 block32 %s: shape [%d %d] overflows element count", name, O, I)
	}
	if len(weight) != elems {
		return 0, fmt.Errorf("fp8 block32 %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	if withX {
		if err := routedExpertXError("fp8 block32", name, O, I, x); err != nil {
			return 0, err
		}
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	scaleElems, ok := fp8Block32Product(sO, sI)
	if !ok {
		return 0, fmt.Errorf("fp8 block32 %s: scale shape [%d %d] overflows element count", name, sO, sI)
	}
	if len(scales) != scaleElems {
		return 0, fmt.Errorf("fp8 block32 %s: scales has %d bytes, shape [%d %d] (blocks of %d) implies %d", name, len(scales), O, I, FP8Block32Dim, scaleElems)
	}
	if err := firstE8M0ScaleError(scales); err != nil {
		return 0, fmt.Errorf("fp8 block32 %s: %w", name, err)
	}
	if k := firstE4M3NaNByte(weight); k >= 0 {
		return 0, fmt.Errorf("fp8 block32 %s: weight byte %d (%#02x) decodes to a non-finite value", name, k, weight[k])
	}
	return sI, nil
}

// GemvFP8Block32E8M0 is a portable scalar GEMV over a V4.1 E8M0 block-scaled weight:
// y[o] = sum_i decode32(weight[o,i], scale[o/32,i/32]) * x[i]. It decodes each weight
// element inline against its tile scale, never materializing the full dequantized matrix,
// so a memory-bound engine can stream the FP8 weight and the fused scale in one pass.
// Fail-closed: len(x) must equal I, and all weight/scale lengths are validated before y
// is allocated.
//
// It is also the bit-exact oracle for device kernels: each row accumulates serially in
// i order, and the explicit float32 conversion rounds every product before its add. The
// Go spec forbids fusing across an explicit conversion, so this is the same function on
// every GOARCH (the spec permits fusion; arm64, ppc64le, s390x, riscv64, loong64 and
// amd64 with GOAMD64>=v3 do fuse a plain acc += w*x).
func GemvFP8Block32E8M0(name string, O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	sI, err := checkGemvFP8Block32E8M0(name, O, I, weight, scales, x)
	if err != nil {
		return nil, err
	}
	scaleF := make([]float32, len(scales))
	for k, b := range scales {
		v, err := decodeE8M0Scale(b)
		if err != nil {
			return nil, fmt.Errorf("fp8 block32 %s: %w", name, err)
		}
		scaleF[k] = v
	}
	y := make([]float32, O)
	for o := 0; o < O; o++ {
		scaleRow := (o / FP8Block32Dim) * sI
		var acc float32
		for i := 0; i < I; i++ {
			w := DecodeE4M3(weight[o*I+i]) * scaleF[scaleRow+i/FP8Block32Dim]
			acc += float32(w * x[i])
		}
		y[o] = acc
	}
	return y, nil
}

// ---- Routed-expert device matmul seam ------------------------------------------

// mxE2M1Values is the OCP MX E2M1 finite value table indexed by one unpacked nibble.
// PyTorch float4_e2m1fn_x2 stores val0 in the low nibble and val1 in the high nibble,
// matching internal/model's v4ExpertE2M1Values (this copy keeps compute free of a model
// import; the two are held to the same table by the expert-GEMV witness).
var mxE2M1Values = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6,
}

// routedExpertDeviceKernel type-asserts be for the optional device capability, returning
// nil when the backend carries none. It is the single seam both dispatchers share so the
// fail-closed rule lives in one place.
func routedExpertDeviceKernel(be Backend) RoutedExpertDeviceKernel {
	if be == nil {
		return nil
	}
	k, ok := be.(RoutedExpertDeviceKernel)
	if !ok {
		return nil
	}
	return k
}

// GemvFP8Block32ExpertDevice dispatches a routed-expert FP8 block-32 E8M0 GEMV to the
// backend's device kernel, FAILING CLOSED with ErrExpertDeviceKernelAbsent when the
// backend implements no RoutedExpertDeviceKernel. It never calls the portable scalar
// GemvFP8Block32E8M0: the expert tier binds decode throughput, so an absent kernel must
// be a visible refusal, not a silent roofline-regressing fallback. The weight/scale
// bytes are passed through PACKED, so no full-checkpoint dequantization occurs here.
func GemvFP8Block32ExpertDevice(name string, be Backend, O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	k := routedExpertDeviceKernel(be)
	if k == nil {
		return nil, fmt.Errorf("fp8 block32 expert %s: %w", name, ErrExpertDeviceKernelAbsent)
	}
	y, err := k.GemvExpertFP8Block32E8M0(O, I, weight, scales, x)
	if err != nil {
		return nil, fmt.Errorf("fp8 block32 expert %s: %w", name, err)
	}
	if len(y) != O {
		return nil, fmt.Errorf("fp8 block32 expert %s: device kernel returned %d outputs, want %d", name, len(y), O)
	}
	return y, nil
}

// GemvMXFP4ExpertDevice dispatches a routed-expert MXFP4 (E2M1 x E8M0, 32-K block) GEMV
// to the backend's device kernel, failing closed exactly as GemvFP8Block32ExpertDevice.
// The packed I8 weight (two E2M1 nibbles per byte) and its raw E8M0 scale bytes are
// handed to the device unchanged — no scalar unpack, no f32 materialization.
func GemvMXFP4ExpertDevice(name string, be Backend, O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	k := routedExpertDeviceKernel(be)
	if k == nil {
		return nil, fmt.Errorf("mxfp4 expert %s: %w", name, ErrExpertDeviceKernelAbsent)
	}
	y, err := k.GemvExpertMXFP4E8M0(O, I, weight, scales, x)
	if err != nil {
		return nil, fmt.Errorf("mxfp4 expert %s: %w", name, err)
	}
	if len(y) != O {
		return nil, fmt.Errorf("mxfp4 expert %s: device kernel returned %d outputs, want %d", name, len(y), O)
	}
	return y, nil
}

// GemvMXFP4E8M0 is the PORTABLE SCALAR reference for a routed-expert MXFP4 GEMV: it is
// the non-roofline oracle the device kernel is held to, never a fallback the dispatch
// above reaches for. Weight rows are O packed rows of I/2 bytes (low nibble = val0,
// high nibble = val1); one E8M0 byte scales each group of 32 unpacked K values, so a
// row carries ceil(I/32) scale bytes. Fail-closed: every length, dtype and E8M0 NaN
// byte is validated before y is allocated. Each row accumulates serially in i order with
// every product rounded before its add (explicit float32 conversion, as in
// GemvFP8Block32E8M0), so the oracle is fusion-free on every GOARCH.
func GemvMXFP4E8M0(name string, O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	scaleCols, err := checkGemvMXFP4E8M0(name, O, I, weight, scales, x)
	if err != nil {
		return nil, err
	}
	packedCols := I / 2
	scaleF := make([]float32, len(scales))
	for k, b := range scales {
		v, err := decodeE8M0Scale(b)
		if err != nil {
			return nil, fmt.Errorf("mxfp4 %s: %w", name, err)
		}
		scaleF[k] = v
	}
	y := make([]float32, O)
	for o := 0; o < O; o++ {
		scaleRow := o * scaleCols
		var acc float32
		for i := 0; i < I; i++ {
			b := weight[o*packedCols+i/2]
			nib := b & 0x0f
			if i%2 == 1 {
				nib = b >> 4
			}
			w := mxE2M1Values[nib] * scaleF[scaleRow+i/FP8Block32Dim]
			acc += float32(w * x[i])
		}
		y[o] = acc
	}
	return y, nil
}

// checkGemvMXFP4E8M0 is the single acceptance predicate of a routed-expert MXFP4 E8M0
// GEMV, shared by the scalar oracle GemvMXFP4E8M0 and every device kernel so both accept
// and refuse the same inputs. Every E2M1 nibble is finite, so only shapes, lengths and
// the E8M0 NaN byte can refuse. It allocates nothing and returns ceil(I/32).
func checkGemvMXFP4E8M0(name string, O, I int, weight, scales []byte, x []float32) (int, error) {
	return checkMXFP4E8M0(name, O, I, weight, scales, x, true)
}

// checkMXFP4E8M0Weight is checkGemvMXFP4E8M0 without the x clause (see
// checkFP8Block32E8M0Weight).
func checkMXFP4E8M0Weight(name string, O, I int, weight, scales []byte) (int, error) {
	return checkMXFP4E8M0(name, O, I, weight, scales, nil, false)
}

func checkMXFP4E8M0(name string, O, I int, weight, scales []byte, x []float32, withX bool) (int, error) {
	if O <= 0 || I <= 0 || I%2 != 0 {
		return 0, fmt.Errorf("mxfp4 %s: shape [%d %d] needs positive O and even I", name, O, I)
	}
	packedCols := I / 2
	elems, ok := fp8Block32Product(O, packedCols)
	if !ok {
		return 0, fmt.Errorf("mxfp4 %s: shape [%d %d] overflows packed byte count", name, O, I)
	}
	if len(weight) != elems {
		return 0, fmt.Errorf("mxfp4 %s: weight has %d bytes, shape [%d %d] implies %d", name, len(weight), O, I, elems)
	}
	scaleCols := (I + FP8Block32Dim - 1) / FP8Block32Dim
	scaleElems, ok := fp8Block32Product(O, scaleCols)
	if !ok {
		return 0, fmt.Errorf("mxfp4 %s: scale shape [%d %d] overflows element count", name, O, scaleCols)
	}
	if len(scales) != scaleElems {
		return 0, fmt.Errorf("mxfp4 %s: scales has %d bytes, shape [%d %d] (blocks of %d) implies %d", name, len(scales), O, I, FP8Block32Dim, scaleElems)
	}
	if withX {
		if err := routedExpertXError("mxfp4", name, O, I, x); err != nil {
			return 0, err
		}
	}
	if err := firstE8M0ScaleError(scales); err != nil {
		return 0, fmt.Errorf("mxfp4 %s: %w", name, err)
	}
	return scaleCols, nil
}
