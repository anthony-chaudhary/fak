package compute

// Adapted from flashinfer-ai/flashinfer (Apache-2.0), files
// flashinfer/experimental/qsa_nvfp4/csrc/qsa_nvfp4_attention.cuh and
// flashinfer/experimental/qsa_nvfp4/backend.py at commit
// kernels are not ported; this file re-expresses their decode-step contract
// (page math, split count, merge plan and reference, E2M1/E4M3 to BF16
// conversion) as host Go reference code.

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
)

// QSAKVFormat is the KV cache element format of a QSA sparse paged decode.
type QSAKVFormat uint8

const (
	QSAKVNVFP4 QSAKVFormat = iota
	QSAKVFP8
	QSAKVBF16
)

const (
	QSAHeadDim      = 256
	QSAMaxGroupSize = 16
	// QSAMergeDepth is the number of splits one merge warp holds in flight.
	QSAMergeDepth = 8
	// qsaMaxPageSize keeps 1<<shift inside 32 bits for the reciprocal divisor.
	qsaMaxPageSize = 1 << 31
)

var (
	ErrQSADecodeFormat = errors.New("compute: unknown QSA KV format")
	ErrQSADecodeShape  = errors.New("compute: invalid QSA decode shape")
)

func (f QSAKVFormat) valid() bool { return f <= QSAKVBF16 }

func (f QSAKVFormat) String() string {
	switch f {
	case QSAKVNVFP4:
		return "nvfp4"
	case QSAKVFP8:
		return "fp8"
	case QSAKVBF16:
		return "bf16"
	}
	return fmt.Sprintf("QSAKVFormat(%d)", uint8(f))
}

// TileTokens is the selection width one attention tile covers. BF16 halves it
// because two 64-token BF16 stages exceed SM86/SM89 shared memory.
func (f QSAKVFormat) TileTokens() int {
	if f == QSAKVBF16 {
		return 32
	}
	return 64
}

// MergeWarps is the most warps one merge CTA runs.
func (f QSAKVFormat) MergeWarps() int {
	if f == QSAKVNVFP4 {
		return 8
	}
	return 16
}

// MergeOnePassSplits is the most splits a merge handles with every warp
// draining a single group: 64 for NVFP4, 128 for FP8 and BF16.
func (f QSAKVFormat) MergeOnePassSplits() int { return f.MergeWarps() * QSAMergeDepth }

// ReciprocalPageMath reports whether the format resolves selection indices to
// pages with the multiply-shift divisor; the others divide plainly.
func (f QSAKVFormat) ReciprocalPageMath() bool { return f == QSAKVNVFP4 }

// QSAPageDivisor divides a non-negative int32 index by a fixed page size as
// (mulhi(index, Mul) + index) >> Shift, which is exact for index < 2^31 and
// avoids a 64-bit division whose call would wait on every outstanding load.
type QSAPageDivisor struct {
	PageSize uint32
	Shift    uint32
	Mul      uint32
}

// NewQSAPageDivisor derives Mul - 1 = floor(n * 2^32 / pageSize), with
// n = 2^Shift - pageSize, from a float32 reciprocal estimate followed by two
// exact remainder corrections, the same steps a device can run without
// integer division.
func NewQSAPageDivisor(pageSize uint32) (QSAPageDivisor, error) {
	if pageSize == 0 || pageSize > qsaMaxPageSize {
		return QSAPageDivisor{}, fmt.Errorf("%w: page size %d outside [1, 2^31]", ErrQSADecodeShape, pageSize)
	}
	shift := uint32(32 - bits.LeadingZeros32(pageSize-1))
	n := uint32((uint64(1) << shift) - uint64(pageSize))
	scaled := int64(uint64(n) << 32)
	rcp := 1 / float32(pageSize)
	q := qsaFloat32ToUint32RZ(float32(n) * 4294967296 * rcp)
	residual := scaled - int64(uint64(q)*uint64(pageSize))
	q += uint32(qsaFloat32ToInt32RD(float32(residual) * rcp))
	r := scaled - int64(uint64(q)*uint64(pageSize))
	switch {
	case r < 0:
		q--
	case r >= int64(pageSize):
		q++
	}
	return QSAPageDivisor{PageSize: pageSize, Shift: shift, Mul: q + 1}, nil
}

// Split returns (index / PageSize, index % PageSize); negative indices are
// clamped to 0 first, as the decode does for masked selection slots.
func (d QSAPageDivisor) Split(index int32) (page, offset int32) {
	safe := uint32(max(index, 0))
	hi, _ := bits.Mul32(safe, d.Mul)
	page = int32((hi + safe) >> d.Shift)
	return page, int32(safe) - page*int32(d.PageSize)
}

func qsaFloat32ToUint32RZ(v float32) uint32 {
	switch {
	case v != v || v <= 0:
		return 0
	case v >= 4294967296:
		return math.MaxUint32
	}
	return uint32(v)
}

func qsaFloat32ToInt32RD(v float32) int32 {
	f := math.Floor(float64(v))
	switch {
	case v != v:
		return 0
	case f >= math.MaxInt32:
		return math.MaxInt32
	case f <= math.MinInt32:
		return math.MinInt32
	}
	return int32(f)
}

// QSAPageTable is the per-request block table one decode reads: BlockTable[r]
// maps request r's logical pages to physical blocks in [0, NumBlocks).
type QSAPageTable struct {
	PageSize   int32
	NumBlocks  int32
	BlockTable [][]int32
	divisor    QSAPageDivisor
}

// QSAPageSlot is where a selection index lands. Page is -1 when the index,
// request or logical page is out of range; Valid additionally requires the
// physical block to exist. Invalid slots are zero-filled, never read.
type QSAPageSlot struct {
	Page   int32
	Offset int32
	Valid  bool
}

// NewQSAPageTable validates a rectangular block table.
func NewQSAPageTable(pageSize, numBlocks int32, blockTable [][]int32) (*QSAPageTable, error) {
	if pageSize <= 0 || numBlocks <= 0 || len(blockTable) == 0 || len(blockTable[0]) == 0 {
		return nil, fmt.Errorf("%w: page size %d, %d blocks, %d requests", ErrQSADecodeShape, pageSize, numBlocks, len(blockTable))
	}
	width := len(blockTable[0])
	for r, row := range blockTable {
		if len(row) != width {
			return nil, fmt.Errorf("%w: block table row %d has width %d, want %d", ErrQSADecodeShape, r, len(row), width)
		}
	}
	d, err := NewQSAPageDivisor(uint32(pageSize))
	if err != nil {
		return nil, err
	}
	return &QSAPageTable{PageSize: pageSize, NumBlocks: numBlocks, BlockTable: blockTable, divisor: d}, nil
}

// Locate resolves selection index of request through the format's page math.
func (t *QSAPageTable) Locate(f QSAKVFormat, request, index int32) QSAPageSlot {
	var logical, offset int32
	if f.ReciprocalPageMath() {
		logical, offset = t.divisor.Split(index)
	} else {
		safe := max(index, 0)
		logical = safe / t.PageSize
		offset = safe - logical*t.PageSize
	}
	page := int32(-1)
	if request >= 0 && int(request) < len(t.BlockTable) && index >= 0 && int(logical) < len(t.BlockTable[request]) {
		page = t.BlockTable[request][logical]
	}
	return QSAPageSlot{Page: page, Offset: offset, Valid: page >= 0 && page < t.NumBlocks}
}

// QSADecodeStep is one prologue step of a decode CTA, before its first tile.
type QSADecodeStep uint8

const (
	QSAStepLoadIndices QSADecodeStep = iota
	QSAStepLoadQRegisters
	QSAStepStoreQShared
	QSAStepLookupPages
	QSAStepIssueKVCopies
	QSAStepBarrier
)

// QSAPrologueOrder is the issue order of a decode CTA's prologue. NVFP4 holds
// Q in registers until the first tile's copies are issued, so its page lookups
// never wait on Q stores; FP8 and BF16 store Q first because unrolling the
// late store slowed their large single-split shapes.
func QSAPrologueOrder(f QSAKVFormat) []QSADecodeStep {
	if f == QSAKVNVFP4 {
		return []QSADecodeStep{
			QSAStepLoadIndices, QSAStepLoadQRegisters, QSAStepLookupPages,
			QSAStepIssueKVCopies, QSAStepStoreQShared, QSAStepBarrier,
		}
	}
	return []QSADecodeStep{
		QSAStepLoadIndices, QSAStepLoadQRegisters, QSAStepStoreQShared,
		QSAStepLookupPages, QSAStepIssueKVCopies, QSAStepBarrier,
	}
}

// QSANumTiles is the tile count over a selection of selectionWidth indices.
func QSANumTiles(f QSAKVFormat, selectionWidth int) (int, error) {
	if !f.valid() {
		return 0, ErrQSADecodeFormat
	}
	if selectionWidth <= 0 {
		return 0, fmt.Errorf("%w: selection width %d", ErrQSADecodeShape, selectionWidth)
	}
	return qsaCeilDiv(selectionWidth, f.TileTokens()), nil
}

// QSAResidency is the device occupancy the split rule sizes against.
type QSAResidency struct {
	NumSMs    int
	CTAsPerSM int
}

// QSANumSplits chooses how many splits each (row, KV head) is cut into. NVFP4
// fills two CTAs per SM and keeps every split; the others fill one wave of
// resident CTAs, BF16 drops to one CTA per SM until that leaves eight or more
// tiles per split, and the count is then shrunk to the fewest splits with the
// same tiles per split.
func QSANumSplits(f QSAKVFormat, numRows, numKVHeads, numTiles int, r QSAResidency) (int, error) {
	if !f.valid() {
		return 0, ErrQSADecodeFormat
	}
	if numRows <= 0 || numKVHeads <= 0 || numTiles <= 0 || r.NumSMs <= 0 || r.CTAsPerSM <= 0 {
		return 0, fmt.Errorf("%w: rows %d, kv heads %d, tiles %d, residency %+v", ErrQSADecodeShape, numRows, numKVHeads, numTiles, r)
	}
	work := numRows * numKVHeads
	if f == QSAKVNVFP4 {
		return min(numTiles, max(1, 2*r.NumSMs/work)), nil
	}
	splits := min(numTiles, max(1, r.CTAsPerSM*r.NumSMs/work))
	if f == QSAKVBF16 {
		onePerSM := min(numTiles, max(1, r.NumSMs/work))
		if qsaCeilDiv(numTiles, onePerSM) < 8 {
			splits = onePerSM
		}
	}
	return qsaCeilDiv(numTiles, qsaCeilDiv(numTiles, splits)), nil
}

// QSAMergePlan is everything the split merge reads besides its buffers; the
// merge launch deliberately carries this slim set instead of the full decode
// parameters.
type QSAMergePlan struct {
	Format QSAKVFormat
	Splits int
	// Warps is one per QSAMergeDepth splits, capped at Format.MergeWarps().
	Warps int
	// Chunk is the contiguous run of splits each warp merges.
	Chunk int
	// OneWarp selects the NVFP4 single-warp merge: its splits fit one group,
	// so the sum stays in registers and skips the cross-warp rescale.
	OneWarp bool
}

// Needed reports whether a merge runs at all; one split writes the output.
func (p QSAMergePlan) Needed() bool { return p.Splits > 1 }

// PlanQSAMerge sizes the merge launch for numSplits.
func PlanQSAMerge(f QSAKVFormat, numSplits int) (QSAMergePlan, error) {
	if !f.valid() {
		return QSAMergePlan{}, ErrQSADecodeFormat
	}
	if numSplits <= 0 {
		return QSAMergePlan{}, fmt.Errorf("%w: %d splits", ErrQSADecodeShape, numSplits)
	}
	warps := min(f.MergeWarps(), qsaCeilDiv(numSplits, QSAMergeDepth))
	return QSAMergePlan{
		Format:  f,
		Splits:  numSplits,
		Warps:   warps,
		Chunk:   qsaCeilDiv(numSplits, warps),
		OneWarp: f == QSAKVNVFP4 && warps == 1,
	}, nil
}

// MergeQSASplits merges split partial outputs through their log2-domain LSE
// in the plan's warp and group order and returns the normalized output,
// before BF16 rounding and the output gate. An LSE of -Inf marks a split with
// no values; when every split is empty the output is zero.
func MergeQSASplits(p QSAMergePlan, partialOut [][]float32, partialLSE []float32) ([]float32, error) {
	if p.Splits <= 0 || p.Warps <= 0 || p.Chunk*p.Warps < p.Splits || len(partialOut) != p.Splits || len(partialLSE) != p.Splits {
		return nil, fmt.Errorf("%w: plan %+v with %d partials and %d LSEs", ErrQSADecodeShape, p, len(partialOut), len(partialLSE))
	}
	dims := len(partialOut[0])
	for k, part := range partialOut {
		if len(part) != dims {
			return nil, fmt.Errorf("%w: partial %d has %d dims, want %d", ErrQSADecodeShape, k, len(part), dims)
		}
	}
	negInf := float32(math.Inf(-1))
	warpM := make([]float32, p.Warps)
	warpDen := make([]float32, p.Warps)
	warpAcc := make([][]float32, p.Warps)
	for w := range p.Warps {
		first := w * p.Chunk
		last := min(first+p.Chunk, p.Splits)
		m, den, acc := negInf, float32(0), make([]float32, dims)
		for base := first; base < last; base += QSAMergeDepth {
			end := min(base+QSAMergeDepth, last)
			nextM := m
			for k := base; k < end; k++ {
				nextM = max(nextM, partialLSE[k])
			}
			if nextM == negInf {
				continue
			}
			alpha := qsaExp2(m - nextM)
			den *= alpha
			for j := range acc {
				acc[j] *= alpha
			}
			m = nextM
			for k := base; k < end; k++ {
				weight := qsaExp2(partialLSE[k] - m)
				den += weight
				for j, v := range partialOut[k] {
					acc[j] += weight * v
				}
			}
		}
		warpM[w], warpDen[w], warpAcc[w] = m, den, acc
	}
	if p.OneWarp {
		return qsaNormalize(warpAcc[0], warpDen[0]), nil
	}
	maxM := negInf
	for _, m := range warpM {
		maxM = max(maxM, m)
	}
	total, merged := float32(0), make([]float32, dims)
	if maxM > negInf {
		for w := range p.Warps {
			scale := qsaExp2(warpM[w] - maxM)
			total += scale * warpDen[w]
			for j, v := range warpAcc[w] {
				merged[j] += scale * v
			}
		}
	}
	return qsaNormalize(merged, total), nil
}

func qsaExp2(x float32) float32 { return float32(math.Exp2(float64(x))) }

func qsaNormalize(acc []float32, total float32) []float32 {
	inv := float32(0)
	if total > 0 {
		inv = 1 / total
	}
	out := make([]float32, len(acc))
	for j, v := range acc {
		out[j] = v * inv
	}
	return out
}

// qsaE2M1MagnitudeBF16 is the BF16 encoding of each E2M1 magnitude code times
// 2^120. The bias lets the E2M1 exponent land directly in the BF16 exponent
// field, and the E4M3 scale conversion carries the matching 2^-120.
var qsaE2M1MagnitudeBF16 = [8]uint16{0x0000, 0x7B00, 0x7B80, 0x7BC0, 0x7C00, 0x7C40, 0x7C80, 0x7CC0}

// qsaE2M1HiTable and qsaE2M1LoTable split qsaE2M1MagnitudeBF16 into the two 8-byte lookup
// tables (high and low bytes) a byte permute indexes by magnitude code.
var qsaE2M1HiTable, qsaE2M1LoTable = func() (hi, lo [2]uint32) {
	for code, v := range qsaE2M1MagnitudeBF16 {
		hi[code/4] |= uint32(v>>8) << (8 * (code % 4))
		lo[code/4] |= uint32(v&0xFF) << (8 * (code % 4))
	}
	return hi, lo
}()

// qsaPermuteBytes is the PTX prmt.b32 default mode: output byte i is byte
// (sel nibble i) & 7 of the eight bytes {b:a}, replaced by its sign bit
// replicated eight times when the nibble's bit 3 is set.
func qsaPermuteBytes(a, b, sel uint32) uint32 {
	src := uint64(b)<<32 | uint64(a)
	var r uint32
	for i := range 4 {
		s := sel >> (4 * i) & 0xF
		v := uint32(src >> (8 * (s & 7)) & 0xFF)
		if s&8 != 0 {
			v = uint32(int32(int8(v)) >> 7 & 0xFF)
		}
		r |= v << (8 * i)
	}
	return r
}

// qsaE2M1Bytes decodes the eight nibbles of packed to BF16 byte planes: lo
// and hi hold the low and high bytes of nibbles 0..3 (A) and 4..7 (B).
func qsaE2M1Bytes(packed uint32) (loA, hiA, loB, hiB uint32) {
	mag := packed & 0x77777777
	magB := qsaPermuteBytes(mag, mag, 0x3232)
	hiA = qsaPermuteBytes(qsaE2M1HiTable[0], qsaE2M1HiTable[1], mag)
	hiB = qsaPermuteBytes(qsaE2M1HiTable[0], qsaE2M1HiTable[1], magB)
	loA = qsaPermuteBytes(qsaE2M1LoTable[0], qsaE2M1LoTable[1], mag)
	loB = qsaPermuteBytes(qsaE2M1LoTable[0], qsaE2M1LoTable[1], magB)
	// Nibble k's sign is bit 7 of byte k/2 of packed (odd k) or of packed<<4 (even k).
	hiA |= qsaPermuteBytes(packed<<4, packed, 0xD9C8) & 0x80808080
	hiB |= qsaPermuteBytes(packed<<4, packed, 0xFBEA) & 0x80808080
	return loA, hiA, loB, hiB
}

// QSADecodeE2M1x8 decodes eight E2M1 codes (nibble k is element k) to BF16
// times 2^120, two per word: word w holds element 2w low and 2w+1 high.
func QSADecodeE2M1x8(packed uint32) [4]uint32 {
	loA, hiA, loB, hiB := qsaE2M1Bytes(packed)
	return [4]uint32{
		qsaPermuteBytes(loA, hiA, 0x5140),
		qsaPermuteBytes(loA, hiA, 0x7362),
		qsaPermuteBytes(loB, hiB, 0x5140),
		qsaPermuteBytes(loB, hiB, 0x7362),
	}
}

// QSADecodeE2M1x8Pairs is QSADecodeE2M1x8 with nibbles paired 0,2 / 1,3 /
// 4,6 / 5,7: fed two tokens' bytes interleaved, each word holds one dim of
// both tokens.
func QSADecodeE2M1x8Pairs(packed uint32) [4]uint32 {
	loA, hiA, loB, hiB := qsaE2M1Bytes(packed)
	return [4]uint32{
		qsaPermuteBytes(loA, hiA, 0x6240),
		qsaPermuteBytes(loA, hiA, 0x7351),
		qsaPermuteBytes(loB, hiB, 0x6240),
		qsaPermuteBytes(loB, hiB, 0x7351),
	}
}

// QSADecodeE4M3x4 decodes four E4M3 block scales (byte k is scale k) to BF16
// times 2^-120: E4M3 exponent and mantissa bits drop into the BF16 fields
// unchanged. Times a QSADecodeE2M1x8 value the product is the exact decoded
// element, so no bias multiply is needed. NaN codes are not preserved.
func QSADecodeE4M3x4(packed uint32) [4]uint16 {
	q := qsaPermuteBytes(packed, packed, 0x1302)
	x := q&0x80008000 | (q&0x7F007F00)>>4
	y := (q<<8)&0x80008000 | ((q<<8)&0x7F007F00)>>4
	return [4]uint16{uint16(x), uint16(x >> 16), uint16(y), uint16(y >> 16)}
}

func qsaCeilDiv(a, b int) int { return (a + b - 1) / b }
