// Package strix -- gfx1151 RDNA 3.5 Wave32 decode GEMV for the Q3_K, Q4_K, BF16 and FP16
// weight layouts.
//
// The Q2_K Wave32 decode GEMV in wave32_wmma.go established the shape this file mirrors: a
// pinned launch geometry (arch/wave/coalescing/super-block), a fail-closed admission gate, a
// device kernel body expressed as a numerically-exact Go model, and a byte-for-byte dequant
// mirror of the internal/compute quant owner. This file extends that same contract to the
// formats the published DeepSeek V4 Flash artifact stores its routed-expert slabs and its
// attention/embedding weights in -- Q3_K and Q4_K k-quants, and the unquantized BF16/FP16 rows.
//
// strix cannot import internal/compute (the quant owners import compute, and the boundary
// forbids private-to-core peaks in this direction), so the super-block arithmetic is restated
// here as the physical device contract and pinned against the reference by the parity test.
package strix

import (
	"fmt"
	"math"
	"sync"
)

// --- Shared Wave32 lane/coalescing model --------------------------------------------------

// BytesPerLaneDwordX4 is the per-lane payload of one global_load_dwordx4 (16 bytes / 128 bits).
const BytesPerLaneDwordX4 = BytesPerVectorLoadDwordX4

// CoalescedBurstBytes is the LPDDR5X physical burst granularity the coalesced loads must honor
// (128 bytes). Every wavefront transaction is an integer number of these bursts.
const CoalescedBurstBytes = LPDDR5XBurstSizeBytes

// GlobalLoadDwordX4Count returns the number of 128-bit global_load_dwordx4 instructions a
// Wave32 wavefront must issue to stream payloadBytes, rounded up to a whole wavefront. It is
// the instruction-count form of the coalescing invariant: 32 lanes * 16 bytes = 512 bytes of
// payload per instruction, so a payload is read with exactly ceil(payloadBytes/512) dwordx4
// loads (each load being 4 x 128-byte bursts). A non-positive payload reads nothing (0).
func GlobalLoadDwordX4Count(payloadBytes int64) int64 {
	if payloadBytes <= 0 {
		return 0
	}
	return (payloadBytes + int64(WavefrontCoalescedBytes) - 1) / int64(WavefrontCoalescedBytes)
}

// CoalescedBurstAligned reports whether byteOffset sits on a 128-byte LPDDR5X burst boundary.
// A coalesced Wave32 dwordx4 load must begin at a 128-byte-aligned address so the 512-byte
// wavefront payload maps onto whole physical bursts; an unaligned start splits a burst and
// costs a second transaction per wavefront.
func CoalescedBurstAligned(byteOffset int) bool {
	return byteOffset >= 0 && byteOffset%CoalescedBurstBytes == 0
}

// Wave32LaneLoad is the resolved byte offset a single lane reads within one dwordx4 load.
type Wave32LaneLoad struct {
	// Lane is the Wave32 lane index (0..31).
	Lane int `json:"lane"`
	// ByteOffset is the lane-relative byte offset (lane * 16) within the wavefront payload.
	ByteOffset int `json:"byte_offset"`
	// DwordIndex is the 32-bit dword index within the 512-byte wavefront payload.
	DwordIndex int `json:"dword_index"`
}

// Wave32LaneLoadModel resolves the canonical 32-lane x global_load_dwordx4 access pattern for
// one coalesced wavefront read. It is the address model the .hip kernel body must reproduce:
// lane L reads 16 contiguous bytes at L*16, so the 32 lanes together cover exactly
// WavefrontCoalescedBytes (512) bytes = 4 x 128-byte bursts.
type Wave32LaneLoadModel struct {
	// Lanes is the number of lanes (always 32 for RDNA 3.5 native Wave32).
	Lanes int `json:"lanes"`
	// PayloadBytes is the per-wavefront payload (512 bytes).
	PayloadBytes int `json:"payload_bytes"`
	// Bursts is the number of 128-byte bursts per wavefront transaction (4).
	Bursts int `json:"bursts"`
	// BaseOffset is the wavefront transaction's start byte offset. It must be a whole
	// LPDDR5X burst multiple (128): the coalesced 512-byte wavefront payload then maps onto
	// exactly Bursts whole 128-byte physical bursts.
	BaseOffset int `json:"base_offset"`
	// Loads is one resolved entry per lane.
	Loads []Wave32LaneLoad `json:"loads"`
}

// DefaultWave32LaneLoadModel returns the canonical 32-lane x 16-byte dwordx4 lane map.
func DefaultWave32LaneLoadModel() Wave32LaneLoadModel {
	loads := make([]Wave32LaneLoad, RDNA35NativeWaveSize)
	for lane := 0; lane < RDNA35NativeWaveSize; lane++ {
		loads[lane] = Wave32LaneLoad{
			Lane:       lane,
			ByteOffset: lane * BytesPerLaneDwordX4,
			DwordIndex: lane * (BytesPerLaneDwordX4 / 4),
		}
	}
	return Wave32LaneLoadModel{
		Lanes:        RDNA35NativeWaveSize,
		PayloadBytes: WavefrontCoalescedBytes,
		Bursts:       BurstsPerWavefrontTransaction,
		Loads:        loads,
	}
}

// Validate checks the lane model against the physical coalescing invariants: 32 lanes, 16 bytes
// per lane, a 512-byte payload, 4 whole 128-byte bursts, a 128-byte-aligned wavefront base, and
// every lane read contained within the payload.
func (m Wave32LaneLoadModel) Validate() error {
	if m.Lanes != RDNA35NativeWaveSize {
		return fmt.Errorf("strix/wave32: lane model has %d lanes, want %d", m.Lanes, RDNA35NativeWaveSize)
	}
	if m.PayloadBytes != WavefrontCoalescedBytes {
		return fmt.Errorf("strix/wave32: lane model payload %d != %d", m.PayloadBytes, WavefrontCoalescedBytes)
	}
	if m.Bursts != BurstsPerWavefrontTransaction {
		return fmt.Errorf("strix/wave32: lane model bursts %d != %d", m.Bursts, BurstsPerWavefrontTransaction)
	}
	if len(m.Loads) != RDNA35NativeWaveSize {
		return fmt.Errorf("strix/wave32: lane model has %d load entries, want %d", len(m.Loads), RDNA35NativeWaveSize)
	}
	if !CoalescedBurstAligned(m.BaseOffset) {
		return fmt.Errorf("strix/wave32: wavefront base offset %d is not 128-byte burst-aligned", m.BaseOffset)
	}
	for i, l := range m.Loads {
		if l.Lane != i {
			return fmt.Errorf("strix/wave32: load entry %d carries lane %d", i, l.Lane)
		}
		if l.ByteOffset != i*BytesPerLaneDwordX4 {
			return fmt.Errorf("strix/wave32: lane %d offset %d != %d", i, l.ByteOffset, i*BytesPerLaneDwordX4)
		}
		if l.ByteOffset+BytesPerLaneDwordX4 > m.PayloadBytes {
			return fmt.Errorf("strix/wave32: lane %d read [%d,%d) escapes the %d-byte payload", i, l.ByteOffset, l.ByteOffset+BytesPerLaneDwordX4, m.PayloadBytes)
		}
	}
	if m.BaseOffset%BytesPerLaneDwordX4 != 0 {
		return fmt.Errorf("strix/wave32: wavefront base offset %d is not 16-byte vector-aligned", m.BaseOffset)
	}
	return nil
}

// --- Shared fail-closed decode GEMV kernel shape ------------------------------------------

// wave32GEMVAdmission is the device-visible admission record for a format-specific decode
// GEMV. It is the hinge of the fail-closed contract: dispatch commits only when Admitted is
// true, and a device that cannot prove the gfx1151 Wave32 launch path is admitted=false by
// construction.
type wave32GEMVAdmission struct {
	// Admitted is the single fail-closed bit. It starts false and is set true only when every
	// physical precondition (arch, wave size, coalescing, super-block geometry) is satisfied.
	Admitted bool `json:"admitted"`
	// Reason names the first unmet precondition, or "admitted" when Admitted is true.
	Reason string `json:"reason"`
	// DeviceVisible reports the toggle is reachable from the decode MatMul path.
	DeviceVisible bool `json:"device_visible"`
	// KernelCompiled reports the gfx1151 kernel object was built/validated for this target.
	KernelCompiled bool `json:"kernel_compiled"`
}

// resolveWave32GEMVAdmission evaluates the shared physical preconditions in a fixed order. Any
// unmet precondition (or a missing launch path) leaves Admitted false with a named reason.
// geometryOK is the caller-supplied super-block geometry predicate for its format.
func resolveWave32GEMVAdmission(arch RDNAArch, waveSize, coalescedBytes int, geometryOK bool, geometryReason string, launchAvailable bool) wave32GEMVAdmission {
	adm := wave32GEMVAdmission{DeviceVisible: true, KernelCompiled: false}
	switch {
	case arch != ArchGFX1151:
		adm.Reason = fmt.Sprintf("arch %q is not gfx1151", arch)
		return adm
	case waveSize != RDNA35NativeWaveSize:
		adm.Reason = fmt.Sprintf("wave size %d is not native Wave32 (32)", waveSize)
		return adm
	case coalescedBytes != WavefrontCoalescedBytes:
		adm.Reason = fmt.Sprintf("coalesced bytes %d != %d (Wave32 128-bit loads)", coalescedBytes, WavefrontCoalescedBytes)
		return adm
	case !geometryOK:
		adm.Reason = geometryReason
		return adm
	}
	// Preconditions hold: the kernel is compiled for the target, but it is only admitted when a
	// real launch path is proven. Without that proof the toggle fails closed.
	adm.KernelCompiled = true
	if !launchAvailable {
		adm.Reason = "no validated gfx1151 hsaco/AQL launch path"
		return adm
	}
	adm.Admitted = true
	adm.Reason = "admitted"
	return adm
}

// --- Q3_K decode GEMV ---------------------------------------------------------------------

// Q3_K super-block geometry mirrored in the strix package (the public quant owner lives in
// internal/compute/quant_q3k.go; strix cannot import it, so the constants are restated as the
// physical device contract and asserted against the reference in the parity test).
const (
	// Q3KSuperBlockBytes is the Q3_K super-block byte length (32 hmask + 64 low codes + 12
	// packed 6-bit scales + 2 f16 d).
	Q3KSuperBlockBytes = 110
	// Q3KSuperBlockElems is the Q3_K super-block element count (256).
	Q3KSuperBlockElems = 256
)

// Q3KDecodeConfig pins the physical launch shape of the gfx1151 Q3_K decode GEMV.
type Q3KDecodeConfig struct {
	// Arch is the RDNA 3.5 target ("gfx1151"). A non-Strix Halo arch is not admissible.
	Arch RDNAArch `json:"arch"`
	// WaveSize is the native Wave32 execution width (32). Any other value is inadmissible.
	WaveSize int `json:"wave_size"`
	// CoalescedBytes is the per-wavefront payload: 32 lanes * 16B global_load_dwordx4 = 512B.
	CoalescedBytes int `json:"coalesced_bytes"`
	// SuperBlockBytes is the Q3_K on-disk super-block size (110 bytes / 256 weights).
	SuperBlockBytes int `json:"super_block_bytes"`
	// SuperBlockElems is the Q3_K super-block element count (256).
	SuperBlockElems int `json:"super_block_elems"`
}

// DefaultQ3KDecodeConfig returns the canonical gfx1151 Q3_K decode GEMV configuration.
func DefaultQ3KDecodeConfig() Q3KDecodeConfig {
	return Q3KDecodeConfig{
		Arch:            ArchGFX1151,
		WaveSize:        RDNA35NativeWaveSize,
		CoalescedBytes:  WavefrontCoalescedBytes,
		SuperBlockBytes: Q3KSuperBlockBytes,
		SuperBlockElems: Q3KSuperBlockElems,
	}
}

// Validate checks configuration invariants.
func (c Q3KDecodeConfig) Validate() error {
	if c.WaveSize != RDNA35NativeWaveSize {
		return fmt.Errorf("strix/wave32: Q3_K WaveSize (%d) must be 32", c.WaveSize)
	}
	if c.CoalescedBytes != WavefrontCoalescedBytes {
		return fmt.Errorf("strix/wave32: Q3_K CoalescedBytes (%d) must be %d", c.CoalescedBytes, WavefrontCoalescedBytes)
	}
	if c.SuperBlockBytes != Q3KSuperBlockBytes || c.SuperBlockElems != Q3KSuperBlockElems {
		return fmt.Errorf("strix/wave32: Q3_K super-block geometry mismatch (%d,%d)", c.SuperBlockBytes, c.SuperBlockElems)
	}
	return nil
}

// Wave32Q3KGEMVDecodeKernel is the gfx1151 Wave32 Q3_K decode GEMV. It owns the physical launch
// shape and the fail-closed admission gate; Available() is false absent a proven hsaco/AQL path.
type Wave32Q3KGEMVDecodeKernel struct {
	cfg       Q3KDecodeConfig
	admission wave32GEMVAdmission
	mu        sync.RWMutex
}

// NewWave32Q3KGEMVDecodeKernel builds the Q3_K decode GEMV and resolves its fail-closed
// admission. launchAvailable is the caller's proof that a real gfx1151 launch path exists.
func NewWave32Q3KGEMVDecodeKernel(cfg Q3KDecodeConfig, launchAvailable bool) *Wave32Q3KGEMVDecodeKernel {
	k := &Wave32Q3KGEMVDecodeKernel{cfg: cfg}
	k.admission = resolveWave32GEMVAdmission(cfg.Arch, cfg.WaveSize, cfg.CoalescedBytes,
		cfg.SuperBlockBytes == Q3KSuperBlockBytes && cfg.SuperBlockElems == Q3KSuperBlockElems,
		"Q3_K super-block geometry mismatch", launchAvailable)
	return k
}

// DefaultWave32Q3KGEMVDecodeKernel builds the canonical kernel with the launch path unresolved.
func DefaultWave32Q3KGEMVDecodeKernel() *Wave32Q3KGEMVDecodeKernel {
	return NewWave32Q3KGEMVDecodeKernel(DefaultQ3KDecodeConfig(), false)
}

// Admission returns the current device-visible admission record.
func (k *Wave32Q3KGEMVDecodeKernel) Admission() wave32GEMVAdmission {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.admission
}

// Available reports whether the Q3_K decode GEMV may be dispatched. False is the fail-closed state.
func (k *Wave32Q3KGEMVDecodeKernel) Available() bool { return k.Admission().Admitted }

// Config returns the pinned launch configuration.
func (k *Wave32Q3KGEMVDecodeKernel) Config() Q3KDecodeConfig {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.cfg
}

// DispatchQ3KGEMV dispatches one decode GEMV token, computing y[o] = dot(dequant(weight row o), x).
// It refuses (ErrWave32GEMVUnavailable) whenever the device-visible toggle is not admitted -- it
// never falls back to the scalar CPU path.
func (k *Wave32Q3KGEMVDecodeKernel) DispatchQ3KGEMV(raw []byte, x []float32, out, in int) ([]float32, error) {
	if !k.Available() {
		return nil, fmt.Errorf("%w: %s", ErrWave32GEMVUnavailable, k.Admission().Reason)
	}
	return Wave32Q3KGEMVStream(raw, x, out, in)
}

// Q3KDequantSuperBlock writes the 256 weights of one 110-byte Q3_K super-block into dst. It is a
// byte-for-byte mirror of internal/compute/quant_q3k.go's q3kDequantSuperBlock; the parity test
// pins the two implementations together.
func Q3KDequantSuperBlock(dst []float32, blk []byte) {
	if len(blk) < Q3KSuperBlockBytes {
		panic("strix/wave32: short Q3_K super-block")
	}
	if len(dst) < Q3KSuperBlockElems {
		panic("strix/wave32: short destination for Q3_K dequant")
	}
	hmask := blk[:Q3KSuperBlockElems/8]
	q := blk[Q3KSuperBlockElems/8 : Q3KSuperBlockElems/8+Q3KSuperBlockElems/4]
	scales := unpackQ3KScalesMirror(blk[Q3KSuperBlockElems/8+Q3KSuperBlockElems/4 : Q3KSuperBlockElems/8+Q3KSuperBlockElems/4+12])
	d := gemvF16BitsToF32(uint16(blk[Q3KSuperBlockBytes-2]) | uint16(blk[Q3KSuperBlockBytes-1])<<8)
	qi := 0
	is := 0
	mask := byte(1)
	for n := 0; n < Q3KSuperBlockElems; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			dl := d * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+l] >> shift) & 3)
				if hmask[l]&mask == 0 {
					code -= 4
				}
				dst[n+j*32+l] = dl * float32(code)
			}

			dl = d * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+16+l] >> shift) & 3)
				if hmask[16+l]&mask == 0 {
					code -= 4
				}
				dst[n+j*32+16+l] = dl * float32(code)
			}
			shift += 2
			mask <<= 1
		}
		qi += 32
	}
}

// unpackQ3KScalesMirror expands the 12 packed 6-bit sub-scale bytes into 16 signed 6-bit scales,
// byte-for-byte internal/compute/quant_q3k.go's unpackQ3KScales (llama.cpp get_scale_min_k4-style
// Q3_K unpacking).
func unpackQ3KScalesMirror(raw []byte) [16]int8 {
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
	var scales [16]int8
	for i, word := range words {
		for j := 0; j < 4; j++ {
			scales[i*4+j] = int8(byte(word >> (8 * j)))
		}
	}
	return scales
}

// Wave32Q3KGEMVStream is the device kernel body: the Wave32 lane-coalesced Q3_K decode GEMV
// model. It is the numerical contract the real gfx1151 launch must reproduce exactly; the parity
// test binds it to the Q3_K row-dot.
func Wave32Q3KGEMVStream(raw []byte, x []float32, out, in int) ([]float32, error) {
	if out <= 0 || in <= 0 {
		return nil, ErrInvalidDimensions
	}
	if in%Q3KSuperBlockElems != 0 {
		return nil, fmt.Errorf("%w: reduction dim %d is not a multiple of %d", ErrInvalidDimensions, in, Q3KSuperBlockElems)
	}
	rowBytes := (in / Q3KSuperBlockElems) * Q3KSuperBlockBytes
	if len(raw) != out*rowBytes {
		return nil, fmt.Errorf("%w: Q3_K payload len %d != out*rowBytes %d", ErrDimensionMismatch, len(raw), out*rowBytes)
	}
	if len(x) != in {
		return nil, fmt.Errorf("%w: activation len %d != in %d", ErrDimensionMismatch, len(x), in)
	}

	y := make([]float32, out)
	scratch := make([]float32, Q3KSuperBlockElems)
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		var sum float32
		for off, xi := 0, 0; off < len(row); off, xi = off+Q3KSuperBlockBytes, xi+Q3KSuperBlockElems {
			Q3KDequantSuperBlock(scratch, row[off:off+Q3KSuperBlockBytes])
			for j := 0; j < Q3KSuperBlockElems; j++ {
				sum += scratch[j] * x[xi+j]
			}
		}
		y[o] = sum
	}
	return y, nil
}

// --- Q4_K decode GEMV ---------------------------------------------------------------------

// Q4_K super-block geometry mirrored in the strix package (the public quant owner lives in
// internal/compute/quant_q4k.go; strix cannot import it).
const (
	// Q4KSuperBlockBytes is the Q4_K super-block byte length (2 f16 d + 2 f16 dmin + 12 scales
	// + 128 nibbles).
	Q4KSuperBlockBytes = 144
	// Q4KSuperBlockElems is the Q4_K super-block element count (256).
	Q4KSuperBlockElems = 256
)

// Q4KDecodeConfig pins the physical launch shape of the gfx1151 Q4_K decode GEMV.
type Q4KDecodeConfig struct {
	// Arch is the RDNA 3.5 target ("gfx1151"). A non-Strix Halo arch is not admissible.
	Arch RDNAArch `json:"arch"`
	// WaveSize is the native Wave32 execution width (32). Any other value is inadmissible.
	WaveSize int `json:"wave_size"`
	// CoalescedBytes is the per-wavefront payload: 32 lanes * 16B global_load_dwordx4 = 512B.
	CoalescedBytes int `json:"coalesced_bytes"`
	// SuperBlockBytes is the Q4_K on-disk super-block size (144 bytes / 256 weights).
	SuperBlockBytes int `json:"super_block_bytes"`
	// SuperBlockElems is the Q4_K super-block element count (256).
	SuperBlockElems int `json:"super_block_elems"`
}

// DefaultQ4KDecodeConfig returns the canonical gfx1151 Q4_K decode GEMV configuration.
func DefaultQ4KDecodeConfig() Q4KDecodeConfig {
	return Q4KDecodeConfig{
		Arch:            ArchGFX1151,
		WaveSize:        RDNA35NativeWaveSize,
		CoalescedBytes:  WavefrontCoalescedBytes,
		SuperBlockBytes: Q4KSuperBlockBytes,
		SuperBlockElems: Q4KSuperBlockElems,
	}
}

// Validate checks configuration invariants.
func (c Q4KDecodeConfig) Validate() error {
	if c.WaveSize != RDNA35NativeWaveSize {
		return fmt.Errorf("strix/wave32: Q4_K WaveSize (%d) must be 32", c.WaveSize)
	}
	if c.CoalescedBytes != WavefrontCoalescedBytes {
		return fmt.Errorf("strix/wave32: Q4_K CoalescedBytes (%d) must be %d", c.CoalescedBytes, WavefrontCoalescedBytes)
	}
	if c.SuperBlockBytes != Q4KSuperBlockBytes || c.SuperBlockElems != Q4KSuperBlockElems {
		return fmt.Errorf("strix/wave32: Q4_K super-block geometry mismatch (%d,%d)", c.SuperBlockBytes, c.SuperBlockElems)
	}
	return nil
}

// Wave32Q4KGEMVDecodeKernel is the gfx1151 Wave32 Q4_K decode GEMV.
type Wave32Q4KGEMVDecodeKernel struct {
	cfg       Q4KDecodeConfig
	admission wave32GEMVAdmission
	mu        sync.RWMutex
}

// NewWave32Q4KGEMVDecodeKernel builds the Q4_K decode GEMV and resolves its fail-closed admission.
func NewWave32Q4KGEMVDecodeKernel(cfg Q4KDecodeConfig, launchAvailable bool) *Wave32Q4KGEMVDecodeKernel {
	k := &Wave32Q4KGEMVDecodeKernel{cfg: cfg}
	k.admission = resolveWave32GEMVAdmission(cfg.Arch, cfg.WaveSize, cfg.CoalescedBytes,
		cfg.SuperBlockBytes == Q4KSuperBlockBytes && cfg.SuperBlockElems == Q4KSuperBlockElems,
		"Q4_K super-block geometry mismatch", launchAvailable)
	return k
}

// DefaultWave32Q4KGEMVDecodeKernel builds the canonical kernel with the launch path unresolved.
func DefaultWave32Q4KGEMVDecodeKernel() *Wave32Q4KGEMVDecodeKernel {
	return NewWave32Q4KGEMVDecodeKernel(DefaultQ4KDecodeConfig(), false)
}

// Admission returns the current device-visible admission record.
func (k *Wave32Q4KGEMVDecodeKernel) Admission() wave32GEMVAdmission {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.admission
}

// Available reports whether the Q4_K decode GEMV may be dispatched. False is the fail-closed state.
func (k *Wave32Q4KGEMVDecodeKernel) Available() bool { return k.Admission().Admitted }

// Config returns the pinned launch configuration.
func (k *Wave32Q4KGEMVDecodeKernel) Config() Q4KDecodeConfig {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.cfg
}

// DispatchQ4KGEMV dispatches one decode GEMV token. It refuses (ErrWave32GEMVUnavailable) whenever
// the device-visible toggle is not admitted -- it never falls back to the scalar CPU path.
func (k *Wave32Q4KGEMVDecodeKernel) DispatchQ4KGEMV(raw []byte, x []float32, out, in int) ([]float32, error) {
	if !k.Available() {
		return nil, fmt.Errorf("%w: %s", ErrWave32GEMVUnavailable, k.Admission().Reason)
	}
	return Wave32Q4KGEMVStream(raw, x, out, in)
}

// Q4KDequantSuperBlock writes the 256 weights of one 144-byte Q4_K super-block into dst. It is a
// byte-for-byte mirror of internal/compute/quant_q4k.go's q4kDequantBlock; the parity test pins
// the two implementations together.
func Q4KDequantSuperBlock(dst []float32, blk []byte) {
	if len(blk) < Q4KSuperBlockBytes {
		panic("strix/wave32: short Q4_K super-block")
	}
	if len(dst) < Q4KSuperBlockElems {
		panic("strix/wave32: short destination for Q4_K dequant")
	}
	d := gemvF16BitsToF32(uint16(blk[0]) | uint16(blk[1])<<8)
	dmin := gemvF16BitsToF32(uint16(blk[2]) | uint16(blk[3])<<8)
	scales := blk[4 : 4+12]
	q := blk[4+12 : Q4KSuperBlockBytes]
	qi, is := 0, 0
	for j := 0; j < Q4KSuperBlockElems; j += 64 {
		sc, m := scaleMinK4Mirror(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = scaleMinK4Mirror(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			dst[j+l] = d1*float32(q[qi+l]&0x0f) - m1
		}
		for l := 0; l < 32; l++ {
			dst[j+32+l] = d2*float32(q[qi+l]>>4) - m2
		}
		qi += 32
		is += 2
	}
}

// scaleMinK4Mirror unpacks the j-th 6-bit scale and minimum pair from a 12-byte scales field. It
// is a byte-for-byte mirror of internal/kquantbits.ScaleMinK4 (strix cannot import kquantbits).
func scaleMinK4Mirror(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// Wave32Q4KGEMVStream is the device kernel body: the Wave32 lane-coalesced Q4_K decode GEMV
// model. It is the numerical contract the real gfx1151 launch must reproduce exactly; the parity
// test binds it to the Q4_K row-dot.
func Wave32Q4KGEMVStream(raw []byte, x []float32, out, in int) ([]float32, error) {
	if out <= 0 || in <= 0 {
		return nil, ErrInvalidDimensions
	}
	if in%Q4KSuperBlockElems != 0 {
		return nil, fmt.Errorf("%w: reduction dim %d is not a multiple of %d", ErrInvalidDimensions, in, Q4KSuperBlockElems)
	}
	rowBytes := (in / Q4KSuperBlockElems) * Q4KSuperBlockBytes
	if len(raw) != out*rowBytes {
		return nil, fmt.Errorf("%w: Q4_K payload len %d != out*rowBytes %d", ErrDimensionMismatch, len(raw), out*rowBytes)
	}
	if len(x) != in {
		return nil, fmt.Errorf("%w: activation len %d != in %d", ErrDimensionMismatch, len(x), in)
	}

	y := make([]float32, out)
	scratch := make([]float32, Q4KSuperBlockElems)
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		var sum float32
		for off, xi := 0, 0; off < len(row); off, xi = off+Q4KSuperBlockBytes, xi+Q4KSuperBlockElems {
			Q4KDequantSuperBlock(scratch, row[off:off+Q4KSuperBlockBytes])
			for j := 0; j < Q4KSuperBlockElems; j++ {
				sum += scratch[j] * x[xi+j]
			}
		}
		y[o] = sum
	}
	return y, nil
}

// --- BF16 decode GEMV -----------------------------------------------------------------------

// BF16DecodeConfig pins the physical launch shape of the gfx1151 BF16 decode GEMV. There is no
// super-block: BF16 rows are a flat [out, in] stream of 2-byte weights, so the geometry fields
// pin the element width instead.
type BF16DecodeConfig struct {
	// Arch is the RDNA 3.5 target ("gfx1151"). A non-Strix Halo arch is not admissible.
	Arch RDNAArch `json:"arch"`
	// WaveSize is the native Wave32 execution width (32). Any other value is inadmissible.
	WaveSize int `json:"wave_size"`
	// CoalescedBytes is the per-wavefront payload: 32 lanes * 16B global_load_dwordx4 = 512B.
	CoalescedBytes int `json:"coalesced_bytes"`
	// ElementBytes is the BF16 on-disk element width (2 bytes).
	ElementBytes int `json:"element_bytes"`
}

// DefaultBF16DecodeConfig returns the canonical gfx1151 BF16 decode GEMV configuration.
func DefaultBF16DecodeConfig() BF16DecodeConfig {
	return BF16DecodeConfig{
		Arch:           ArchGFX1151,
		WaveSize:       RDNA35NativeWaveSize,
		CoalescedBytes: WavefrontCoalescedBytes,
		ElementBytes:   2,
	}
}

// Validate checks configuration invariants.
func (c BF16DecodeConfig) Validate() error {
	if c.WaveSize != RDNA35NativeWaveSize {
		return fmt.Errorf("strix/wave32: BF16 WaveSize (%d) must be 32", c.WaveSize)
	}
	if c.CoalescedBytes != WavefrontCoalescedBytes {
		return fmt.Errorf("strix/wave32: BF16 CoalescedBytes (%d) must be %d", c.CoalescedBytes, WavefrontCoalescedBytes)
	}
	if c.ElementBytes != 2 {
		return fmt.Errorf("strix/wave32: BF16 ElementBytes (%d) must be 2", c.ElementBytes)
	}
	return nil
}

// Wave32BF16GEMVDecodeKernel is the gfx1151 Wave32 BF16 decode GEMV.
type Wave32BF16GEMVDecodeKernel struct {
	cfg       BF16DecodeConfig
	admission wave32GEMVAdmission
	mu        sync.RWMutex
}

// NewWave32BF16GEMVDecodeKernel builds the BF16 decode GEMV and resolves its fail-closed admission.
func NewWave32BF16GEMVDecodeKernel(cfg BF16DecodeConfig, launchAvailable bool) *Wave32BF16GEMVDecodeKernel {
	k := &Wave32BF16GEMVDecodeKernel{cfg: cfg}
	k.admission = resolveWave32GEMVAdmission(cfg.Arch, cfg.WaveSize, cfg.CoalescedBytes,
		cfg.ElementBytes == 2, "BF16 element width is not 2 bytes", launchAvailable)
	return k
}

// DefaultWave32BF16GEMVDecodeKernel builds the canonical kernel with the launch path unresolved.
func DefaultWave32BF16GEMVDecodeKernel() *Wave32BF16GEMVDecodeKernel {
	return NewWave32BF16GEMVDecodeKernel(DefaultBF16DecodeConfig(), false)
}

// Admission returns the current device-visible admission record.
func (k *Wave32BF16GEMVDecodeKernel) Admission() wave32GEMVAdmission {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.admission
}

// Available reports whether the BF16 decode GEMV may be dispatched.
func (k *Wave32BF16GEMVDecodeKernel) Available() bool { return k.Admission().Admitted }

// Config returns the pinned launch configuration.
func (k *Wave32BF16GEMVDecodeKernel) Config() BF16DecodeConfig {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.cfg
}

// DispatchBF16GEMV dispatches one decode GEMV token. It refuses (ErrWave32GEMVUnavailable)
// whenever the device-visible toggle is not admitted -- it never falls back to the CPU path.
func (k *Wave32BF16GEMVDecodeKernel) DispatchBF16GEMV(raw []uint16, x []float32, out, in int) ([]float32, error) {
	if !k.Available() {
		return nil, fmt.Errorf("%w: %s", ErrWave32GEMVUnavailable, k.Admission().Reason)
	}
	return Wave32BF16GEMVStream(raw, x, out, in)
}

// Wave32BF16GEMVStream is the device kernel body: the Wave32 lane-coalesced BF16 decode GEMV
// model. Each BF16 weight is widened to f32 via the package BF16ToFP32 helper (an exact
// left-shift of the bit pattern, no rounding), accumulated in row order.
func Wave32BF16GEMVStream(raw []uint16, x []float32, out, in int) ([]float32, error) {
	if out <= 0 || in <= 0 {
		return nil, ErrInvalidDimensions
	}
	if len(raw) != out*in {
		return nil, fmt.Errorf("%w: BF16 payload len %d != out*in %d", ErrDimensionMismatch, len(raw), out*in)
	}
	if len(x) != in {
		return nil, fmt.Errorf("%w: activation len %d != in %d", ErrDimensionMismatch, len(x), in)
	}

	y := make([]float32, out)
	for o := 0; o < out; o++ {
		row := raw[o*in : (o+1)*in]
		var sum float32
		for j := 0; j < in; j++ {
			sum += BF16ToFP32(row[j]) * x[j]
		}
		y[o] = sum
	}
	return y, nil
}

// --- FP16 decode GEMV -----------------------------------------------------------------------

// FP16DecodeConfig pins the physical launch shape of the gfx1151 FP16 decode GEMV.
type FP16DecodeConfig struct {
	// Arch is the RDNA 3.5 target ("gfx1151"). A non-Strix Halo arch is not admissible.
	Arch RDNAArch `json:"arch"`
	// WaveSize is the native Wave32 execution width (32). Any other value is inadmissible.
	WaveSize int `json:"wave_size"`
	// CoalescedBytes is the per-wavefront payload: 32 lanes * 16B global_load_dwordx4 = 512B.
	CoalescedBytes int `json:"coalesced_bytes"`
	// ElementBytes is the FP16 on-disk element width (2 bytes).
	ElementBytes int `json:"element_bytes"`
}

// DefaultFP16DecodeConfig returns the canonical gfx1151 FP16 decode GEMV configuration.
func DefaultFP16DecodeConfig() FP16DecodeConfig {
	return FP16DecodeConfig{
		Arch:           ArchGFX1151,
		WaveSize:       RDNA35NativeWaveSize,
		CoalescedBytes: WavefrontCoalescedBytes,
		ElementBytes:   2,
	}
}

// Validate checks configuration invariants.
func (c FP16DecodeConfig) Validate() error {
	if c.WaveSize != RDNA35NativeWaveSize {
		return fmt.Errorf("strix/wave32: FP16 WaveSize (%d) must be 32", c.WaveSize)
	}
	if c.CoalescedBytes != WavefrontCoalescedBytes {
		return fmt.Errorf("strix/wave32: FP16 CoalescedBytes (%d) must be %d", c.CoalescedBytes, WavefrontCoalescedBytes)
	}
	if c.ElementBytes != 2 {
		return fmt.Errorf("strix/wave32: FP16 ElementBytes (%d) must be 2", c.ElementBytes)
	}
	return nil
}

// Wave32FP16GEMVDecodeKernel is the gfx1151 Wave32 FP16 decode GEMV.
type Wave32FP16GEMVDecodeKernel struct {
	cfg       FP16DecodeConfig
	admission wave32GEMVAdmission
	mu        sync.RWMutex
}

// NewWave32FP16GEMVDecodeKernel builds the FP16 decode GEMV and resolves its fail-closed admission.
func NewWave32FP16GEMVDecodeKernel(cfg FP16DecodeConfig, launchAvailable bool) *Wave32FP16GEMVDecodeKernel {
	k := &Wave32FP16GEMVDecodeKernel{cfg: cfg}
	k.admission = resolveWave32GEMVAdmission(cfg.Arch, cfg.WaveSize, cfg.CoalescedBytes,
		cfg.ElementBytes == 2, "FP16 element width is not 2 bytes", launchAvailable)
	return k
}

// DefaultWave32FP16GEMVDecodeKernel builds the canonical kernel with the launch path unresolved.
func DefaultWave32FP16GEMVDecodeKernel() *Wave32FP16GEMVDecodeKernel {
	return NewWave32FP16GEMVDecodeKernel(DefaultFP16DecodeConfig(), false)
}

// Admission returns the current device-visible admission record.
func (k *Wave32FP16GEMVDecodeKernel) Admission() wave32GEMVAdmission {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.admission
}

// Available reports whether the FP16 decode GEMV may be dispatched.
func (k *Wave32FP16GEMVDecodeKernel) Available() bool { return k.Admission().Admitted }

// Config returns the pinned launch configuration.
func (k *Wave32FP16GEMVDecodeKernel) Config() FP16DecodeConfig {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.cfg
}

// DispatchFP16GEMV dispatches one decode GEMV token. It refuses (ErrWave32GEMVUnavailable)
// whenever the device-visible toggle is not admitted -- it never falls back to the CPU path.
func (k *Wave32FP16GEMVDecodeKernel) DispatchFP16GEMV(raw []uint16, x []float32, out, in int) ([]float32, error) {
	if !k.Available() {
		return nil, fmt.Errorf("%w: %s", ErrWave32GEMVUnavailable, k.Admission().Reason)
	}
	return Wave32FP16GEMVStream(raw, x, out, in)
}

// Wave32FP16GEMVStream is the device kernel body: the Wave32 lane-coalesced FP16 decode GEMV
// model. Each FP16 weight is widened to f32 via gemvF16BitsToF32, accumulated in row order.
func Wave32FP16GEMVStream(raw []uint16, x []float32, out, in int) ([]float32, error) {
	if out <= 0 || in <= 0 {
		return nil, ErrInvalidDimensions
	}
	if len(raw) != out*in {
		return nil, fmt.Errorf("%w: FP16 payload len %d != out*in %d", ErrDimensionMismatch, len(raw), out*in)
	}
	if len(x) != in {
		return nil, fmt.Errorf("%w: activation len %d != in %d", ErrDimensionMismatch, len(x), in)
	}

	y := make([]float32, out)
	for o := 0; o < out; o++ {
		row := raw[o*in : (o+1)*in]
		var sum float32
		for j := 0; j < in; j++ {
			sum += gemvF16BitsToF32(row[j]) * x[j]
		}
		y[o] = sum
	}
	return y, nil
}

// gemvF16BitsToF32 converts an IEEE 754 half-precision bit pattern to float32. It is the
// strix-local equivalent of the unexported f16BitsToF32 in wave32_wmma.go (that file is owned by
// the Q2_K lane and cannot be edited here) and mirrors internal/kquantbits.F16BitsToF32Bits.
func gemvF16BitsToF32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	var bits uint32
	switch {
	case exp == 0:
		if man == 0 {
			bits = sign << 31
		} else {
			e := uint32(127 - 15 + 1)
			for man&0x400 == 0 {
				man <<= 1
				e--
			}
			man &= 0x3ff
			bits = sign<<31 | e<<23 | man<<13
		}
	case exp == 0x1f:
		bits = sign<<31 | 0xff<<23 | man<<13
	default:
		bits = sign<<31 | (exp+127-15)<<23 | man<<13
	}
	return math.Float32frombits(bits)
}

// --- Shared decode-GEMV payload accounting --------------------------------------------------

// Q3KPayloadBytes returns the weight byte length of an [out, in] Q3_K matrix row-set, i.e. the
// exact streamed payload a decode GEMV must read: out * (in/256) * 110. It is the input to
// EstimateDecodeBandwidthGBps for the roofline floor; it returns 0 for non-positive dims or a
// reduction dim that is not a Q3_K super-block multiple.
func Q3KPayloadBytes(out, in int) int64 {
	if out <= 0 || in <= 0 || in%Q3KSuperBlockElems != 0 {
		return 0
	}
	return int64(out) * int64(in/Q3KSuperBlockElems) * int64(Q3KSuperBlockBytes)
}

// Q4KPayloadBytes returns the weight byte length of an [out, in] Q4_K matrix row-set:
// out * (in/256) * 144. It returns 0 for non-positive dims or a non-multiple reduction dim.
func Q4KPayloadBytes(out, in int) int64 {
	if out <= 0 || in <= 0 || in%Q4KSuperBlockElems != 0 {
		return 0
	}
	return int64(out) * int64(in/Q4KSuperBlockElems) * int64(Q4KSuperBlockBytes)
}

// BF16PayloadBytes returns the weight byte length of an [out, in] BF16 matrix row-set:
// out * in * 2. It returns 0 for non-positive dims.
func BF16PayloadBytes(out, in int) int64 {
	if out <= 0 || in <= 0 {
		return 0
	}
	return int64(out) * int64(in) * 2
}

// FP16PayloadBytes returns the weight byte length of an [out, in] FP16 matrix row-set:
// out * in * 2. It returns 0 for non-positive dims.
func FP16PayloadBytes(out, in int) int64 {
	if out <= 0 || in <= 0 {
		return 0
	}
	return int64(out) * int64(in) * 2
}
