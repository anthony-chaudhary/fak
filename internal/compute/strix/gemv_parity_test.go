package strix

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// gemv_parity_test.go -- first-order self-checks for the gfx1151 RDNA 3.5 Wave32 decode GEMV
// (Q3_K, Q4_K, BF16, FP16). These are compile-shape, config/admission, coalescing-model,
// error-ladder and one small numeric parity per format. The adversarial, roofline-saturating
// acceptance test is owned by a separate context (gemv_test.go / TestRooflineSaturation).

// gemvF32ToF16Bits is a round-to-nearest-even float32 -> IEEE-754 half converter for fixtures.
// It is the local fixture inverse of gemvF16BitsToF32 (distinct from the Q2_K test file's
// f32ToF16Bits so the two test files share no code).
func gemvF32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127 + 15
	man := bits & 0x7fffff
	switch {
	case (bits & 0x7f800000) == 0x7f800000:
		return sign | 0x7c00 | uint16(man>>13)
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint32(14 - exp)
		half := uint32(1) << (shift - 1)
		man = (man + half) >> shift
		return sign | uint16(man)
	default:
		man += 0x1000
		if man&0x800000 != 0 {
			man = 0
			exp++
			if exp >= 0x1f {
				return sign | 0x7c00
			}
		}
		return sign | uint16(exp)<<10 | uint16(man>>13)
	}
}

// encodeQ3KSuperBlock builds one 110-byte Q3_K super-block fixture in the GGUF layout
// Q3KDequantSuperBlock reads: 32 hmask bytes, 64 low-code bytes, 12 packed scale bytes, f16 d.
func encodeQ3KSuperBlock(hmask, q, scales []byte, d float32) []byte {
	blk := make([]byte, Q3KSuperBlockBytes)
	copy(blk[0:32], hmask)
	copy(blk[32:96], q)
	copy(blk[96:108], scales)
	h := gemvF32ToF16Bits(d)
	blk[108] = byte(h)
	blk[109] = byte(h >> 8)
	return blk
}

// encodeQ4KSuperBlock builds one 144-byte Q4_K super-block fixture: f16 d, f16 dmin, 12 scale
// bytes, 128 nibble bytes.
func encodeQ4KSuperBlock(scales, q []byte, d, dmin float32) []byte {
	blk := make([]byte, Q4KSuperBlockBytes)
	dh := gemvF32ToF16Bits(d)
	dm := gemvF32ToF16Bits(dmin)
	blk[0], blk[1] = byte(dh), byte(dh>>8)
	blk[2], blk[3] = byte(dm), byte(dm>>8)
	copy(blk[4:16], scales)
	copy(blk[16:144], q)
	return blk
}

// gemvUnpackQ3KScales is an independent Q3_K scale unpacker for the reference dot (duplicates
// the 12-byte 6-bit expansion without calling unpackQ3KScalesMirror).
func gemvUnpackQ3KScales(raw []byte) [16]int8 {
	const (
		kmask1 = uint32(0x03030303)
		kmask2 = uint32(0x0f0f0f0f)
	)
	aux0 := uint32(raw[0]) | uint32(raw[1])<<8 | uint32(raw[2])<<16 | uint32(raw[3])<<24
	aux1 := uint32(raw[4]) | uint32(raw[5])<<8 | uint32(raw[6])<<16 | uint32(raw[7])<<24
	aux2 := uint32(raw[8]) | uint32(raw[9])<<8 | uint32(raw[10])<<16 | uint32(raw[11])<<24
	tmp := aux2
	words := [4]uint32{
		(aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4),
		(aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
		((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
		((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
	}
	var s [16]int8
	for i, w := range words {
		for j := 0; j < 4; j++ {
			s[i*4+j] = int8(byte(w >> (8 * j)))
		}
	}
	return s
}

// gemvQ3KReferenceDot is an independent float64 scalar reference for one Q3_K super-block dot.
func gemvQ3KReferenceDot(blk []byte, x []float32) float64 {
	hmask, q := blk[:32], blk[32:96]
	scales := gemvUnpackQ3KScales(blk[96:108])
	d := float64(gemvF16BitsToF32(uint16(blk[108]) | uint16(blk[109])<<8))
	var sum float64
	qi, is := 0, 0
	mask := byte(1)
	for n := 0; n < 256; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			dl := d * float64(int(scales[is])-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+l] >> shift) & 3)
				if hmask[l]&mask == 0 {
					code -= 4
				}
				sum += dl * float64(code) * float64(x[n+j*32+l])
			}
			dl = d * float64(int(scales[is])-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+16+l] >> shift) & 3)
				if hmask[16+l]&mask == 0 {
					code -= 4
				}
				sum += dl * float64(code) * float64(x[n+j*32+16+l])
			}
			shift += 2
			mask <<= 1
		}
		qi += 32
	}
	return sum
}

// gemvQ4KReferenceDot is an independent float64 scalar reference for one Q4_K super-block dot.
func gemvQ4KReferenceDot(blk []byte, x []float32) float64 {
	d := float64(gemvF16BitsToF32(uint16(blk[0]) | uint16(blk[1])<<8))
	dmin := float64(gemvF16BitsToF32(uint16(blk[2]) | uint16(blk[3])<<8))
	scales, q := blk[4:16], blk[16:144]
	scaleMin := func(j int) (uint8, uint8) {
		if j < 4 {
			return scales[j] & 63, scales[j+4] & 63
		}
		return (scales[j+4] & 0x0f) | ((scales[j-4] >> 6) << 4), (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	}
	var sum float64
	qi, is := 0, 0
	for j := 0; j < 256; j += 64 {
		sc, m := scaleMin(is)
		d1, m1 := d*float64(sc), dmin*float64(m)
		sc, m = scaleMin(is + 1)
		d2, m2 := d*float64(sc), dmin*float64(m)
		for l := 0; l < 32; l++ {
			sum += (d1*float64(q[qi+l]&0x0f) - m1) * float64(x[j+l])
		}
		for l := 0; l < 32; l++ {
			sum += (d2*float64(q[qi+l]>>4) - m2) * float64(x[j+32+l])
		}
		qi += 32
		is += 2
	}
	return sum
}

// --- config / admission -------------------------------------------------------------------

func TestGemvWave32ConfigsValidate(t *testing.T) {
	if err := DefaultQ3KDecodeConfig().Validate(); err != nil {
		t.Fatalf("Q3_K default config invalid: %v", err)
	}
	if err := DefaultQ4KDecodeConfig().Validate(); err != nil {
		t.Fatalf("Q4_K default config invalid: %v", err)
	}
	if err := DefaultBF16DecodeConfig().Validate(); err != nil {
		t.Fatalf("BF16 default config invalid: %v", err)
	}
	if err := DefaultFP16DecodeConfig().Validate(); err != nil {
		t.Fatalf("FP16 default config invalid: %v", err)
	}

	bad := DefaultQ3KDecodeConfig()
	bad.SuperBlockBytes = 84
	if err := bad.Validate(); err == nil {
		t.Fatal("Q3_K config with misstated super-block bytes must not validate")
	}
	bad4 := DefaultQ4KDecodeConfig()
	bad4.WaveSize = 64
	if err := bad4.Validate(); err == nil {
		t.Fatal("Q4_K config with Wave64 must not validate")
	}
	badB := DefaultBF16DecodeConfig()
	badB.ElementBytes = 4
	if err := badB.Validate(); err == nil {
		t.Fatal("BF16 config with 4-byte elements must not validate")
	}
}

func TestGemvWave32AdmissionFailClosed(t *testing.T) {
	// No proven launch path => everything must be unavailable and dispatch must refuse.
	if DefaultWave32Q3KGEMVDecodeKernel().Available() {
		t.Fatal("Q3_K kernel must be unavailable without a proven launch path")
	}
	if DefaultWave32Q4KGEMVDecodeKernel().Available() {
		t.Fatal("Q4_K kernel must be unavailable without a proven launch path")
	}
	if DefaultWave32BF16GEMVDecodeKernel().Available() {
		t.Fatal("BF16 kernel must be unavailable without a proven launch path")
	}
	if DefaultWave32FP16GEMVDecodeKernel().Available() {
		t.Fatal("FP16 kernel must be unavailable without a proven launch path")
	}

	if _, err := DefaultWave32Q3KGEMVDecodeKernel().DispatchQ3KGEMV(nil, nil, 1, 256); !errors.Is(err, ErrWave32GEMVUnavailable) {
		t.Fatalf("Q3_K dispatch must refuse with ErrWave32GEMVUnavailable, got %v", err)
	}
	if _, err := DefaultWave32Q4KGEMVDecodeKernel().DispatchQ4KGEMV(nil, nil, 1, 256); !errors.Is(err, ErrWave32GEMVUnavailable) {
		t.Fatalf("Q4_K dispatch must refuse with ErrWave32GEMVUnavailable, got %v", err)
	}
	if _, err := DefaultWave32BF16GEMVDecodeKernel().DispatchBF16GEMV(nil, nil, 1, 256); !errors.Is(err, ErrWave32GEMVUnavailable) {
		t.Fatalf("BF16 dispatch must refuse with ErrWave32GEMVUnavailable, got %v", err)
	}
	if _, err := DefaultWave32FP16GEMVDecodeKernel().DispatchFP16GEMV(nil, nil, 1, 256); !errors.Is(err, ErrWave32GEMVUnavailable) {
		t.Fatalf("FP16 dispatch must refuse with ErrWave32GEMVUnavailable, got %v", err)
	}

	// A proven launch path admits only the pinned geometry; any drift refuses.
	adm := NewWave32Q3KGEMVDecodeKernel(DefaultQ3KDecodeConfig(), true).Admission()
	if !adm.Admitted || adm.Reason != "admitted" || !adm.KernelCompiled || !adm.DeviceVisible {
		t.Fatalf("admitted Q3_K admission unexpected: %+v", adm)
	}
	drift := DefaultQ3KDecodeConfig()
	drift.Arch = ArchGFX1150
	if NewWave32Q3KGEMVDecodeKernel(drift, true).Available() {
		t.Fatal("non-gfx1151 arch must refuse even with a launch path")
	}
	drift4 := DefaultQ4KDecodeConfig()
	drift4.CoalescedBytes = 256
	if NewWave32Q4KGEMVDecodeKernel(drift4, true).Available() {
		t.Fatal("non-512-byte coalescing must refuse even with a launch path")
	}
}

// --- lane / coalescing model --------------------------------------------------------------

func TestGemvWave32LaneLoadModel(t *testing.T) {
	m := DefaultWave32LaneLoadModel()
	if err := m.Validate(); err != nil {
		t.Fatalf("canonical lane model invalid: %v", err)
	}
	if m.Lanes != 32 || m.PayloadBytes != 512 || m.Bursts != 4 {
		t.Fatalf("lane model shape = %d lanes / %d bytes / %d bursts, want 32/512/4", m.Lanes, m.PayloadBytes, m.Bursts)
	}
	if !CoalescedBurstAligned(m.BaseOffset) {
		t.Fatalf("canonical wavefront base offset %d must be 128-byte burst-aligned", m.BaseOffset)
	}
	if len(m.Loads) != 32 {
		t.Fatalf("lane model has %d loads, want 32", len(m.Loads))
	}
	for i, l := range m.Loads {
		if l.ByteOffset != i*16 {
			t.Fatalf("lane %d offset %d, want %d", i, l.ByteOffset, i*16)
		}
	}

	if !CoalescedBurstAligned(0) || !CoalescedBurstAligned(128) || !CoalescedBurstAligned(512) {
		t.Fatal("128-byte multiples must be burst-aligned")
	}
	if CoalescedBurstAligned(16) || CoalescedBurstAligned(127) || CoalescedBurstAligned(-128) {
		t.Fatal("non-multiples (and negatives) must not be burst-aligned")
	}

	// A 512-byte wavefront reads with exactly one dwordx4; any spill costs a whole extra load.
	cases := []struct {
		bytes int64
		want  int64
	}{{0, 0}, {-1, 0}, {1, 1}, {512, 1}, {513, 2}, {110, 1}, {1024, 2}, {1025, 3}}
	for _, c := range cases {
		if got := GlobalLoadDwordX4Count(c.bytes); got != c.want {
			t.Fatalf("GlobalLoadDwordX4Count(%d) = %d, want %d", c.bytes, got, c.want)
		}
	}

	// A wavefront base that is not a whole 128-byte burst must fail Validate.
	bad := DefaultWave32LaneLoadModel()
	bad.BaseOffset = 64
	if err := bad.Validate(); err == nil {
		t.Fatal("lane model with a burst-unaligned base offset must not validate")
	}
	// A lane read that escapes the 512-byte payload must fail Validate.
	bad = DefaultWave32LaneLoadModel()
	bad.Loads[31].ByteOffset = 512
	if err := bad.Validate(); err == nil {
		t.Fatal("lane model whose lane read escapes the payload must not validate")
	}
	// A non-16-byte-aligned base is not a valid vector-load address.
	bad = DefaultWave32LaneLoadModel()
	bad.BaseOffset = 8
	if err := bad.Validate(); err == nil {
		t.Fatal("lane model with a non-vector-aligned base must not validate")
	}
}

// --- error ladder -------------------------------------------------------------------------

func TestGemvWave32ErrorLadder(t *testing.T) {
	out, in := 3, 512

	if _, err := Wave32Q3KGEMVStream(nil, nil, 0, in); !errors.Is(err, ErrInvalidDimensions) {
		t.Fatalf("out<=0 must be ErrInvalidDimensions, got %v", err)
	}
	if _, err := Wave32Q3KGEMVStream(nil, nil, out, 0); !errors.Is(err, ErrInvalidDimensions) {
		t.Fatalf("in<=0 must be ErrInvalidDimensions, got %v", err)
	}
	if _, err := Wave32Q3KGEMVStream(nil, nil, out, 255); !errors.Is(err, ErrInvalidDimensions) {
		t.Fatalf("in %% 256 != 0 must wrap ErrInvalidDimensions, got %v", err)
	}
	if _, err := Wave32Q3KGEMVStream(make([]byte, 110), make([]float32, in), out, in); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("short payload must be ErrDimensionMismatch, got %v", err)
	}
	raw := make([]byte, out*(in/Q3KSuperBlockElems)*Q3KSuperBlockBytes)
	if _, err := Wave32Q3KGEMVStream(raw, make([]float32, in-1), out, in); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("short activation must be ErrDimensionMismatch, got %v", err)
	}

	if _, err := Wave32Q4KGEMVStream(nil, nil, out, 300); !errors.Is(err, ErrInvalidDimensions) {
		t.Fatalf("Q4_K in %% 256 != 0 must be ErrInvalidDimensions, got %v", err)
	}
	if _, err := Wave32Q4KGEMVStream(make([]byte, 144), make([]float32, in), out, in); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("Q4_K short payload must be ErrDimensionMismatch, got %v", err)
	}

	if _, err := Wave32BF16GEMVStream(nil, nil, 0, in); !errors.Is(err, ErrInvalidDimensions) {
		t.Fatalf("BF16 out<=0 must be ErrInvalidDimensions, got %v", err)
	}
	if _, err := Wave32BF16GEMVStream(make([]uint16, in), make([]float32, in), out, in); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("BF16 short payload must be ErrDimensionMismatch, got %v", err)
	}
	if _, err := Wave32FP16GEMVStream(make([]uint16, out*in), make([]float32, in-1), out, in); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("FP16 short activation must be ErrDimensionMismatch, got %v", err)
	}
}

// --- numeric parity (one small case per format) -------------------------------------------

func TestGemvWave32Q3KParity(t *testing.T) {
	const out, in = 11, 512 // 2 super-blocks per row
	rowBlocks := in / Q3KSuperBlockElems
	rng := rand.New(rand.NewSource(0x33C0FFEE))

	raw := make([]byte, out*rowBlocks*Q3KSuperBlockBytes)
	for r := 0; r < out*rowBlocks; r++ {
		hmask := make([]byte, 32)
		q := make([]byte, 64)
		scales := make([]byte, 12)
		for i := range hmask {
			hmask[i] = byte(rng.Intn(256))
		}
		for i := range q {
			q[i] = byte(rng.Intn(256))
		}
		for i := range scales {
			scales[i] = byte(rng.Intn(256))
		}
		d := 0.001 + rng.Float32()*0.05
		copy(raw[r*Q3KSuperBlockBytes:], encodeQ3KSuperBlock(hmask, q, scales, d))
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	got, err := Wave32Q3KGEMVStream(raw, x, out, in)
	if err != nil {
		t.Fatalf("Wave32Q3KGEMVStream: %v", err)
	}
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		for b := 0; b < rowBlocks; b++ {
			blk := raw[(o*rowBlocks+b)*Q3KSuperBlockBytes : (o*rowBlocks+b+1)*Q3KSuperBlockBytes]
			want[o] += gemvQ3KReferenceDot(blk, x[b*Q3KSuperBlockElems:(b+1)*Q3KSuperBlockElems])
		}
	}
	assertGEMVParity(t, got, want, "Q3_K")
}

func TestGemvWave32Q4KParity(t *testing.T) {
	const out, in = 13, 512
	rowBlocks := in / Q4KSuperBlockElems
	rng := rand.New(rand.NewSource(0x44C0FFEE))

	raw := make([]byte, out*rowBlocks*Q4KSuperBlockBytes)
	for r := 0; r < out*rowBlocks; r++ {
		scales := make([]byte, 12)
		q := make([]byte, 128)
		for i := range scales {
			scales[i] = byte(rng.Intn(256))
		}
		for i := range q {
			q[i] = byte(rng.Intn(256))
		}
		d := 0.001 + rng.Float32()*0.05
		dmin := 0.0005 + rng.Float32()*0.02
		copy(raw[r*Q4KSuperBlockBytes:], encodeQ4KSuperBlock(scales, q, d, dmin))
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	got, err := Wave32Q4KGEMVStream(raw, x, out, in)
	if err != nil {
		t.Fatalf("Wave32Q4KGEMVStream: %v", err)
	}
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		for b := 0; b < rowBlocks; b++ {
			blk := raw[(o*rowBlocks+b)*Q4KSuperBlockBytes : (o*rowBlocks+b+1)*Q4KSuperBlockBytes]
			want[o] += gemvQ4KReferenceDot(blk, x[b*Q4KSuperBlockElems:(b+1)*Q4KSuperBlockElems])
		}
	}
	assertGEMVParity(t, got, want, "Q4_K")
}

func TestGemvWave32BF16Parity(t *testing.T) {
	const out, in = 9, 130
	rng := rand.New(rand.NewSource(0xB1F16))

	vals := make([]float32, out*in)
	raw := make([]uint16, out*in)
	for i := range vals {
		vals[i] = rng.Float32()*4 - 2
		raw[i] = FP32ToBF16(vals[i])
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	got, err := Wave32BF16GEMVStream(raw, x, out, in)
	if err != nil {
		t.Fatalf("Wave32BF16GEMVStream: %v", err)
	}
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		for j := 0; j < in; j++ {
			want[o] += float64(BF16ToFP32(raw[o*in+j])) * float64(x[j])
		}
	}
	assertGEMVParity(t, got, want, "BF16")
}

func TestGemvWave32FP16Parity(t *testing.T) {
	const out, in = 9, 130
	rng := rand.New(rand.NewSource(0xF91B))

	raw := make([]uint16, out*in)
	for i := range raw {
		raw[i] = gemvF32ToF16Bits(rng.Float32()*4 - 2)
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	got, err := Wave32FP16GEMVStream(raw, x, out, in)
	if err != nil {
		t.Fatalf("Wave32FP16GEMVStream: %v", err)
	}
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		for j := 0; j < in; j++ {
			want[o] += float64(gemvF16BitsToF32(raw[o*in+j])) * float64(x[j])
		}
	}
	assertGEMVParity(t, got, want, "FP16")
}

// assertGEMVParity checks a float32 kernel result against a float64 reference: a non-degenerate
// output, exact argmax, relative L2 <= 1e-4, and cosine >= 0.99999.
func assertGEMVParity(t *testing.T, got []float32, want []float64, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: output len %d != reference len %d", label, len(got), len(want))
	}
	f64 := make([]float32, len(want))
	var refMax float64
	for i, w := range want {
		f64[i] = float32(w)
		if math.Abs(w) > refMax {
			refMax = math.Abs(w)
		}
	}
	if refMax < 1e-6 {
		t.Fatalf("%s: degenerate all-zero reference", label)
	}

	gotArg, wantArg := 0, 0
	for i := 1; i < len(got); i++ {
		if got[i] > got[gotArg] {
			gotArg = i
		}
		if f64[i] > f64[wantArg] {
			wantArg = i
		}
	}
	if gotArg != wantArg {
		t.Fatalf("%s: argmax mismatch got %d want %d", label, gotArg, wantArg)
	}

	var num, den, refDen float64
	for i := range got {
		d := float64(got[i]) - want[i]
		num += d * d
		den += want[i] * want[i]
		refDen += float64(got[i]) * float64(got[i])
	}
	relL2 := math.Sqrt(num / den)
	if relL2 > 1e-4 {
		t.Fatalf("%s: rel-L2 %.3e > 1e-4", label, relL2)
	}
	if refDen == 0 {
		t.Fatalf("%s: zero-norm kernel output", label)
	}

	cos, err := CosineSimilarity(got, f64)
	if err != nil {
		t.Fatalf("%s: CosineSimilarity: %v", label, err)
	}
	if cos < 0.99999 {
		t.Fatalf("%s: cosine %.9f < 0.99999", label, cos)
	}
	t.Logf("%s parity: argmax=%d rel-L2=%.3e cosine=%.9f", label, gotArg, relL2, cos)
}

// TestGemvWave32PayloadBytes pins the per-format payload accounting against the streamed
// byte lengths the GEMV models actually consume.
func TestGemvWave32PayloadBytes(t *testing.T) {
	const out, in = 7, 512
	if got := Q3KPayloadBytes(out, in); got != int64(out*2*Q3KSuperBlockBytes) {
		t.Fatalf("Q3KPayloadBytes = %d, want %d", got, out*2*Q3KSuperBlockBytes)
	}
	if got := Q4KPayloadBytes(out, in); got != int64(out*2*Q4KSuperBlockBytes) {
		t.Fatalf("Q4KPayloadBytes = %d, want %d", got, out*2*Q4KSuperBlockBytes)
	}
	if got := BF16PayloadBytes(out, in); got != int64(out*in*2) {
		t.Fatalf("BF16PayloadBytes = %d, want %d", got, out*in*2)
	}
	if got := FP16PayloadBytes(out, in); got != int64(out*in*2) {
		t.Fatalf("FP16PayloadBytes = %d, want %d", got, out*in*2)
	}
	if Q3KPayloadBytes(0, in) != 0 || Q3KPayloadBytes(out, 255) != 0 {
		t.Fatal("Q3KPayloadBytes must be 0 for invalid geometry")
	}
	if Q4KPayloadBytes(out, 0) != 0 || BF16PayloadBytes(0, 0) != 0 || FP16PayloadBytes(-1, in) != 0 {
		t.Fatal("payload helpers must be 0 for non-positive dims")
	}
}
