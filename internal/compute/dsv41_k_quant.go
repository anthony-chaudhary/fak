// dsv41_k_quant.go — DSv4.1 mxfp8 K-plane quantize-and-insert + dequantize-and-gather.
//
// ADAPTED from vllm-project/vllm,
// vllm/models/deepseek_v41/common/ops/cache_utils.py:184,528
// (_quantize_and_insert_k_mxfp8_kernel, _dequantize_and_gather_k_mxfp8_kernel)
// at ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0; attribution in
// THIRD_PARTY_NOTICES.md): the V4.1 all-dims MXFP8 K record (512 e4m3 payload
// bytes + 16 ue8m0 scale bytes of 32 dims each, interleaved per token inside
// the paged page), quantized at paged-cache insert and dequantized during
// decode-attention gather. Re-expressed here as the CPU oracle that pins that
// contract; the CUDA kernels live in dsv41_k_quant.cu with the shared record
// helpers in cuda/dsv41_k_quant_vllm.cuh. No upstream code copied.
// Ref oss-port-vllmtop-dsv41-k-quant-cache-20261103.
//
// SEMANTICS (pinned by dsv41_k_quant_test.go):
//   - One record is DSV41KBytesPerToken = 528 bytes: dims 0..511 as OCP
//     e4m3fn, then one ue8m0 scale byte per 32-dim tile.
//   - Tile scale: amax = max(|tile|) floored at 1e-4 (an all-zero tile still
//     lands on a finite scale), exponent = ceil(log2(amax/448)) taken exactly
//     from the f32 bits of amax/448, stored biased by 127 and clamped to
//     [0,255] (upstream clamps to 255, not the e8m0 NaN byte).
//   - Payload: value/scale rounded to nearest e4m3fn, ties to even, signed
//     saturation at +/-448 (codes 0x7E/0xFE). The 0x7F NaN code is never
//     emitted.
//   - Paged slot addressing reuses the paged_kv_write_offsets decomposition
//     (pos = firstOffset + token; page = blocks[pos/16]; offset = pos%16)
//     with the mxfp8 page extent: page*8448 + offset*528 bytes. The page holds
//     16 tokens because the fak paged writer (cuda/paged_kv_write_map.h)
//     admits block_size == 16 only; upstream's 64/128-token pages are the
//     documented delta.
//   - Round trip: |dequant - original| <= 16 * tileScale for every dim, the
//     half-spacing of the top e4m3 bucket at that scale.

package compute

import (
	"errors"
	"math"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

const (
	// DSV41KHeadDim is the full V4.1 K width, RoPE dims included.
	DSV41KHeadDim = 512
	// DSV41KQuantBlock is the number of dims sharing one ue8m0 scale.
	DSV41KQuantBlock = 32
	// DSV41KNumScales is the scale count per record (512/32).
	DSV41KNumScales = 16
	// DSV41KBytesPerToken is the mxfp8 record size: 512 payload + 16 scales.
	DSV41KBytesPerToken = DSV41KHeadDim + DSV41KNumScales // 528
	// DSV41KPageSize is the paged page extent in tokens; the fak paged writer
	// admits block_size == 16 only (cuda/paged_kv_write_map.h).
	DSV41KPageSize = 16
	// DSV41KPageBytes is one mxfp8 K page: 16 records of 528 bytes.
	DSV41KPageBytes = DSV41KPageSize * DSV41KBytesPerToken // 8448
	// dsv41kMaxPages mirrors kWriteMaxPages/cudaPagedKVWriteMaxPages: the
	// by-value page map stays under the legacy 4 KiB kernel-argument limit.
	dsv41kMaxPages = 64
)

var (
	// ErrDSV41KEmpty rejects a zero-token call.
	ErrDSV41KEmpty = errors.New("compute: dsv41 k quant: empty input")

	// ErrDSV41KShape rejects a row width, buffer length or gather window that
	// does not match the declared geometry.
	ErrDSV41KShape = errors.New("compute: dsv41 k quant: shape mismatch")

	// ErrDSV41KPageMap rejects a paged write map or block table whose page
	// count, range, order or arena extent fails admission.
	ErrDSV41KPageMap = errors.New("compute: dsv41 k quant: invalid paged map")

	// ErrDSV41KRecord rejects a record byte slice that is not 528 bytes long.
	ErrDSV41KRecord = errors.New("compute: dsv41 k quant: record size mismatch")

	// ErrDSV41KNonFinite rejects a K value that is NaN or infinite before any
	// quantization, so a non-finite can never become a plausible byte code.
	ErrDSV41KNonFinite = errors.New("compute: dsv41 k quant: non-finite value")
)

// DSV41KMaxTileError returns the closed round-trip bound for one tile at the
// given power-of-two scale: half the spacing of the top e4m3 bucket
// ([224s, 448s], spacing 32s), which bounds the absolute error of every dim
// in the tile under RNE quantization.
func DSV41KMaxTileError(scale float64) float64 { return 16 * scale }

// DSV41KQuantizeRow packs one f32 K row (exactly 512 dims) into the V4.1
// mxfp8 record: 512 e4m3 payload bytes followed by 16 ue8m0 scale bytes, one
// per 32-dim tile. It mirrors the per-token math of the upstream
// _quantize_and_insert_k_mxfp8_kernel without the paged addressing.
func DSV41KQuantizeRow(row []float32) ([]byte, error) {
	if len(row) != DSV41KHeadDim {
		return nil, ErrDSV41KShape
	}
	rec := make([]byte, DSV41KBytesPerToken)
	for t := 0; t < DSV41KNumScales; t++ {
		tile := row[t*DSV41KQuantBlock : (t+1)*DSV41KQuantBlock]
		exp := dsv41kTileExponent(tile)
		inv := float32(math.Ldexp(1, -exp)) // 2^-exp, exact
		for i, v := range tile {
			// NaN or infinite values would turn into plausible-looking byte
			// codes; refuse them before any quantization.
			if math.Float32bits(v)&0x7f800000 == 0x7f800000 {
				return nil, ErrDSV41KNonFinite
			}
			rec[t*DSV41KQuantBlock+i] = dsv41kEncodeE4M3(v * inv)
		}
		rec[DSV41KHeadDim+t] = dsv41kEncodeScale(exp)
	}
	return rec, nil
}

// DSV41KDequantizeRow decodes a 528-byte mxfp8 record back to f32: each dim is
// its e4m3 value times its tile's power-of-two scale, matching the upstream
// dequant multiply order.
func DSV41KDequantizeRow(rec []byte) ([]float32, error) {
	if len(rec) != DSV41KBytesPerToken {
		return nil, ErrDSV41KRecord
	}
	out := make([]float32, DSV41KHeadDim)
	for t := 0; t < DSV41KNumScales; t++ {
		scale := float32(math.Ldexp(1, int(rec[DSV41KHeadDim+t])-127))
		base := t * DSV41KQuantBlock
		for i := 0; i < DSV41KQuantBlock; i++ {
			out[base+i] = mathx.DecodeE4M3(rec[base+i]) * scale
		}
	}
	return out, nil
}

// dsv41kTileExponent returns the ue8m0 exponent for one tile: the smallest
// integer e with 2^e >= amax/448, amax floored at 1e-4. raw = amax/448 is a
// positive normal f32, so ceil(log2(raw)) resolves from its bits without a
// transcendental: unbiased exponent field, plus one when the mantissa is
// nonzero. This matches the upstream fmaxf(amax, 1e-4) + log2/ceil pipeline
// for every finite input.
func dsv41kTileExponent(tile []float32) int {
	var amax float32
	for _, v := range tile {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax = a
		}
	}
	const floor = 1e-4
	if amax < floor {
		amax = floor
	}
	raw := amax / 448.0
	bits := math.Float32bits(raw)
	exp := int(bits>>23)&0xff - 127
	if bits&0x7fffff != 0 {
		exp++
	}
	return exp
}

// dsv41kEncodeScale biases the exponent by 127 and clamps to the uint8 range.
// Upstream clamps to 255 (not 254), so the e8m0 NaN byte is a valid stored
// scale here; our exponent range makes it unreachable in practice.
func dsv41kEncodeScale(exp int) byte {
	enc := exp + 127
	if enc < 0 {
		enc = 0
	}
	if enc > 255 {
		enc = 255
	}
	return byte(enc)
}

// dsv41kEncodeE4M3 rounds one already-scaled value to the nearest OCP
// e4m3fn byte: ties to even, signed saturation at +/-448 (0x7E/0xFE). The
// 0x7F NaN code is never emitted. It is the bit-exact host mirror of
// dsv41_kq_e4m3_encode in cuda/dsv41_k_quant_vllm.cuh.
func dsv41kEncodeE4M3(x float32) byte {
	if x >= 448.0 {
		return 0x7E
	}
	if x <= -448.0 {
		return 0xFE
	}
	bits := math.Float32bits(x)
	sign := bits >> 31
	mag := bits & 0x7fffffff
	if mag == 0 {
		return byte(sign << 7)
	}
	exp := int(mag>>23)&0xff - 127
	if exp >= -6 {
		frac := mag & 0x7fffff
		m := uint32(frac >> 20)
		rem := uint32(frac & 0xfffff)
		if rem > 0x80000 || (rem == 0x80000 && m&1 != 0) {
			m++
		}
		e := exp + 7
		if m == 8 {
			m = 0
			e++
		}
		if e > 15 {
			return byte(sign<<7 | 0x7E)
		}
		return byte(sign<<7 | uint32(e)<<3 | m)
	}
	// Subnormal target: q = |x| * 8192 is exact in f32 and lies in [0, 8).
	q := float64(math.Float32frombits(mag)) * 8192
	m := uint32(q)
	r := q - float64(m)
	if r > 0.5 || (r == 0.5 && m&1 != 0) {
		m++
	}
	if m > 7 {
		m = 7
	}
	return byte(sign<<7 | m)
}

// DSV41KQuantInsert quantizes numTokens rows of k and scatters their records
// into the paged mxfp8 K arena through a by-value write map: blocks covers
// exactly the pages touched by logical positions firstOffset ..
// firstOffset+numTokens-1, with the same admission rules as
// paged_kv_write_map_valid (count within dsv41kMaxPages, in-range distinct
// page IDs, contiguous page coverage). The slot decomposition is verbatim
// paged_kv_write_offsets': pos = firstOffset + token; page =
// blocks[pos/DSV41KPageSize]; offset = pos % DSV41KPageSize.
func DSV41KQuantInsert(k []float32, numTokens, firstOffset int, blocks []int32,
	capacityBlocks int, arena []byte) error {
	if numTokens <= 0 {
		return ErrDSV41KEmpty
	}
	if len(k) != numTokens*DSV41KHeadDim {
		return ErrDSV41KShape
	}
	if err := dsv41kPageMapValid(blocks, firstOffset, numTokens, capacityBlocks); err != nil {
		return err
	}
	if int64(capacityBlocks)*int64(DSV41KPageBytes) > int64(len(arena)) {
		return ErrDSV41KShape
	}
	for token := 0; token < numTokens; token++ {
		rec, err := DSV41KQuantizeRow(k[token*DSV41KHeadDim : (token+1)*DSV41KHeadDim])
		if err != nil {
			return err
		}
		pos := firstOffset + token
		base := dsv41kRecordSlot(int(blocks[pos/DSV41KPageSize]), pos)
		copy(arena[base:base+DSV41KBytesPerToken], rec)
	}
	return nil
}

// dsv41kPageMapValid mirrors paged_kv_write_map_valid for the mxfp8 plane:
// the f32 head geometry terms are dropped because the record layout carries
// no head split, everything else is identical.
func dsv41kPageMapValid(blocks []int32, firstOffset, tokens, capacityBlocks int) error {
	if len(blocks) < 1 || len(blocks) > dsv41kMaxPages {
		return ErrDSV41KPageMap
	}
	if capacityBlocks <= 0 {
		return ErrDSV41KPageMap
	}
	if firstOffset < 0 || firstOffset >= DSV41KPageSize {
		return ErrDSV41KPageMap
	}
	if tokens <= 0 {
		return ErrDSV41KPageMap
	}
	end := firstOffset + tokens
	if (end-1)/DSV41KPageSize+1 != len(blocks) {
		return ErrDSV41KPageMap
	}
	for i, b := range blocks {
		if b < 0 || int(b) >= capacityBlocks {
			return ErrDSV41KPageMap
		}
		for j := 0; j < i; j++ {
			if blocks[j] == b {
				return ErrDSV41KPageMap
			}
		}
	}
	return nil
}

// dsv41kRecordSlot is the byte base of one token record inside the paged
// mxfp8 K arena: page*8448 + inPageOffset*528. It shares its (page, offset)
// decomposition with paged_kv_write_offsets so both planes address the same
// physical slot.
func dsv41kRecordSlot(pageIdx, pos int) int64 {
	return int64(pageIdx)*int64(DSV41KPageBytes) + int64(pos%DSV41KPageSize)*int64(DSV41KBytesPerToken)
}

// DSV41KDequantGatherOnePass reads the last gatherLen logical positions of one
// sequence from the paged mxfp8 K arena and writes their dequantized rows to
// out[offset*512 : (offset+gatherLen)*512]. Each record is read in one pass:
// the whole 528-byte record is pulled before any tile is decoded, mirroring
// upstream's one-pass gather kernel.
func DSV41KDequantGatherOnePass(out []float32, arena []byte, blockTable []int32,
	seqLen, gatherLen, offset int) error {
	if err := dsv41kGatherValidate(out, arena, blockTable, seqLen, gatherLen, offset); err != nil {
		return err
	}
	start := seqLen - gatherLen
	for i := 0; i < gatherLen; i++ {
		base, ok := dsv41kGatherRecordBase(arena, blockTable, start+i)
		if !ok {
			return ErrDSV41KPageMap
		}
		var rec [DSV41KBytesPerToken]byte
		copy(rec[:], arena[base:base+DSV41KBytesPerToken])
		row := out[(offset+i)*DSV41KHeadDim : (offset+i+1)*DSV41KHeadDim]
		for t := 0; t < DSV41KNumScales; t++ {
			scale := float32(math.Ldexp(1, int(rec[DSV41KHeadDim+t])-127))
			baseT := t * DSV41KQuantBlock
			for d := 0; d < DSV41KQuantBlock; d++ {
				row[baseT+d] = mathx.DecodeE4M3(rec[baseT+d]) * scale
			}
		}
	}
	return nil
}

// DSV41KDequantGatherPerBlock is the per-tile form of the same gather: for
// each record it walks the 16 tiles in order, loading one tile's 32 payload
// bytes plus its scale byte and storing that tile's 32 outputs before moving
// on, mirroring upstream's per-quant-block loop. It must agree element-wise
// with DSV41KDequantGatherOnePass on every input.
func DSV41KDequantGatherPerBlock(out []float32, arena []byte, blockTable []int32,
	seqLen, gatherLen, offset int) error {
	if err := dsv41kGatherValidate(out, arena, blockTable, seqLen, gatherLen, offset); err != nil {
		return err
	}
	start := seqLen - gatherLen
	for i := 0; i < gatherLen; i++ {
		base, ok := dsv41kGatherRecordBase(arena, blockTable, start+i)
		if !ok {
			return ErrDSV41KPageMap
		}
		rec := arena[base : base+DSV41KBytesPerToken]
		row := out[(offset+i)*DSV41KHeadDim : (offset+i+1)*DSV41KHeadDim]
		for t := 0; t < DSV41KNumScales; t++ {
			scale := float32(math.Ldexp(1, int(rec[DSV41KHeadDim+t])-127))
			tile := rec[t*DSV41KQuantBlock : (t+1)*DSV41KQuantBlock]
			dst := row[t*DSV41KQuantBlock : (t+1)*DSV41KQuantBlock]
			for d := 0; d < DSV41KQuantBlock; d++ {
				dst[d] = mathx.DecodeE4M3(tile[d]) * scale
			}
		}
	}
	return nil
}

// dsv41kGatherValidate closes the gather contract: non-negative window inside
// the output, and every page the gathered positions touch mapped to a
// non-negative ID whose full page extent lies inside the arena. Only the
// pages of the gathered window are consulted, matching upstream's block-table
// reads.
func dsv41kGatherValidate(out []float32, arena []byte, blockTable []int32,
	seqLen, gatherLen, offset int) error {
	if seqLen < 0 || gatherLen < 0 || gatherLen > seqLen || offset < 0 {
		return ErrDSV41KShape
	}
	if offset+gatherLen > len(out)/DSV41KHeadDim {
		return ErrDSV41KShape
	}
	if seqLen == 0 {
		return nil
	}
	if len(blockTable) < (seqLen-1)/DSV41KPageSize+1 {
		return ErrDSV41KPageMap
	}
	start := seqLen - gatherLen
	for p := start / DSV41KPageSize; p <= (seqLen-1)/DSV41KPageSize; p++ {
		b := blockTable[p]
		if b < 0 {
			return ErrDSV41KPageMap
		}
		if int64(b)*int64(DSV41KPageBytes)+int64(DSV41KPageBytes) > int64(len(arena)) {
			return ErrDSV41KPageMap
		}
	}
	return nil
}

// dsv41kGatherRecordBase resolves one logical position to its record byte
// base, re-checking the arena bound so a caller cannot alias a short arena.
func dsv41kGatherRecordBase(arena []byte, blockTable []int32, pos int) (int64, bool) {
	base := dsv41kRecordSlot(int(blockTable[pos/DSV41KPageSize]), pos)
	if base+DSV41KBytesPerToken > int64(len(arena)) {
		return 0, false
	}
	return base, true
}
