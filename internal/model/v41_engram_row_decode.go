package model

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Engram table rows are activation input: the stored bytes of the demanded
// rows are gathered on the host (a byte read, like a weight read), and the
// decode to F32 runs where the row is consumed. A session with the device
// decode hands the stored bytes to compute.V41EngramRowDecodeBackend; the host
// arm decodes with decodeV41EngramRow. Verified Q2_K and IQ4_NL sources are
// cached as their bound 84-byte and 144-byte rows, so neither arm dequantizes
// at cache fill (Ref #13668).

const v41EngramIQ4NLEncodedRowBytes = qkK / iq4nlBlockWeights * iq4nlBlockBytes

// v41HostEngramRowDecodes counts host Engram row decodes while the BF16 witness
// is on (enableV41BF16Witness), beside v41HostBF16Rounds.
var v41HostEngramRowDecodes atomic.Int64

func v41NoteHostEngramRowDecode() {
	if v41HostBF16Witness.Load() {
		v41HostEngramRowDecodes.Add(1)
	}
}

// v41EngramBoundEncoded returns reader's encoded-row seam when digest commits
// to a supported exact encoded dialect, never merely a matching row width.
func v41EngramBoundEncoded(reader io.ReaderAt, digest string) V41EngramEncodedRows {
	_, scheme, err := v41EngramParseDigest(digest)
	if err != nil || (scheme != v41EngramDigestTreePrefix+V41EngramEncodingQ2K &&
		scheme != v41EngramDigestTreePrefix+V41EngramEncodingIQ4NL) {
		return nil
	}
	enc, _ := reader.(V41EngramEncodedRows)
	return enc
}

// v41EngramQ2KRowSource retains its historical name and original encoded
// owner/lifetime forwarding. Its immutable encoding/width now also name IQ4_NL.
// Q4_K's equal 144-byte width does not make it an IQ4_NL source.
type v41EngramQ2KRowSource struct {
	enc      V41EngramEncodedRows
	shard    string
	base     int
	rows     int
	encoding string
	rowBytes int
}

// v41EngramEncodedForm returns only an authenticated, supported encoded source.
// Binding offsets retain served-F32 units; source reads use encoded-row units.
func v41EngramEncodedForm(src V41EngramRowSource) (*v41EngramQ2KRowSource, bool) {
	v, ok := src.(*v41VerifiedEngramRowSource)
	if !ok || v == nil || v.encoded == nil || v.rowBytes != V41EngramF32RowBytes || v.binding.Offset%int64(v.rowBytes) != 0 {
		return nil, false
	}
	_, scheme, err := v41EngramParseDigest(v.binding.Digest)
	var encoding string
	switch scheme {
	case v41EngramDigestTreePrefix + V41EngramEncodingQ2K:
		encoding = V41EngramEncodingQ2K
	case v41EngramDigestTreePrefix + V41EngramEncodingIQ4NL:
		encoding = V41EngramEncodingIQ4NL
	default:
		return nil, false
	}
	base := v.binding.Offset / int64(v.rowBytes)
	if err != nil || v.binding.Rows <= 0 || base < 0 || base > int64(math.MaxInt-v.binding.Rows) {
		return nil, false
	}
	return &v41EngramQ2KRowSource{enc: v.encoded, shard: v.binding.Shard, base: int(base), rows: v.binding.Rows,
		encoding: encoding, rowBytes: v41EngramEncodedRowBytes[encoding]}, true
}

func (s *v41EngramQ2KRowSource) RowBytes() int { return s.rowBytes }

func (s *v41EngramQ2KRowSource) ReadRows(start, count int, dst []byte) (int, error) {
	if start < 0 || count <= 0 || start > s.rows || count > s.rows-start {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamOutOfBounds, Row: start, Shard: s.shard,
			Err: fmt.Errorf("row range [%d,%d) outside [0,%d)", start, start+count, s.rows)}
	}
	want, ok := checkedMulInt(count, s.rowBytes)
	if !ok || s.rowBytes <= 0 || len(dst) < want {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: start, Shard: s.shard,
			Err: fmt.Errorf("destination bytes=%d want=%d", len(dst), want)}
	}
	r, offset, size, encoding, err := s.enc.V41EngramEncodedRows(s.base+start, count)
	if err != nil {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamOutOfBounds, Row: start, Shard: s.shard, Err: err}
	}
	if r == nil || encoding != s.encoding || offset < 0 || size != int64(want) || offset > math.MaxInt64-size {
		return 0, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: start, Shard: s.shard,
			Err: fmt.Errorf("encoded rows name %q extent offset=%d size=%d, want %d %s bytes", encoding, offset, size, want, s.encoding)}
	}
	n, err := r.ReadAt(dst[:want], offset)
	if n != want {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return n, &V41EngramStreamError{Kind: V41EngramStreamShortRead, Row: start, Shard: s.shard,
			Err: fmt.Errorf("read bytes=%d want=%d: %w", n, want, err)}
	}
	if err != nil && !errors.Is(err, io.EOF) {
		return n, err
	}
	return n, nil
}

// dequantV41EngramQ2KRow decodes one 84-byte Q2_K row with the host scalar
// super-block dequantization and returns its first dim values.
func dequantV41EngramQ2KRow(row []byte, dim int) ([]float32, error) {
	if len(row) != v41EngramQ2KEncodedRowBytes {
		return nil, fmt.Errorf("%w: Engram q2_k row has %d bytes, want %d", ErrV41NativeUnsupported, len(row), v41EngramQ2KEncodedRowBytes)
	}
	if dim <= 0 || dim > qkK {
		return nil, fmt.Errorf("%w: Engram q2_k row dim %d outside [1,%d]", ErrV41NativeUnsupported, dim, qkK)
	}
	var block [qkK]float32
	q2kDequantSuperBlockScalar(block[:], row)
	for j, value := range block[:dim] {
		if !finite32(value) {
			return nil, fmt.Errorf("%w: Engram q2_k row value %d is non-finite", ErrV41NativeUnsupported, j)
		}
	}
	return append([]float32(nil), block[:dim]...), nil
}

// dequantV41EngramIQ4NLRow reuses the existing split-half scalar block decoder.
// This is the pre-selected host arm, never replay of a selected device failure.
func dequantV41EngramIQ4NLRow(row []byte, dim int) ([]float32, error) {
	if len(row) != v41EngramIQ4NLEncodedRowBytes {
		return nil, fmt.Errorf("%w: Engram iq4_nl row has %d bytes, want %d", ErrV41NativeUnsupported, len(row), v41EngramIQ4NLEncodedRowBytes)
	}
	if dim <= 0 || dim > qkK {
		return nil, fmt.Errorf("%w: Engram iq4_nl row dim %d outside [1,%d]", ErrV41NativeUnsupported, dim, qkK)
	}
	var block [qkK]float32
	for b := 0; b < qkK/iq4nlBlockWeights; b++ {
		iq4nlDequantBlock(block[b*iq4nlBlockWeights:], row[b*iq4nlBlockBytes:])
	}
	for j, value := range block[:dim] {
		if !finite32(value) {
			return nil, fmt.Errorf("%w: Engram iq4_nl row value %d is non-finite", ErrV41NativeUnsupported, j)
		}
	}
	return append([]float32(nil), block[:dim]...), nil
}

// v41EngramGatheredRows is one Engram layer's gathered rows on the resident
// path, position-major [positions*cols], still in the stage's stored dialect.
type v41EngramGatheredRows struct {
	layer    int
	stage    *v41EngramStage
	cacheIdx int
	rows     [][]byte
	dim      int
	// encoded owns the contiguous resident gather, never cache or pooled storage.
	encoded []byte
}

// encoding names the device decode dialect of the stage's rows.
func (g *v41EngramGatheredRows) encoding() (compute.V41EngramRowEncoding, bool) {
	if g == nil || g.stage == nil {
		return 0, false
	}
	var encoding compute.V41EngramRowEncoding
	switch g.stage.rowBytes {
	case V41EngramPackedRowBytes:
		encoding = compute.V41EngramRowPacked
	case v41EngramQ2KEncodedRowBytes:
		encoding = compute.V41EngramRowQ2K
	case v41EngramIQ4NLEncodedRowBytes:
		encoding = compute.V41EngramRowIQ4NL
	case V41EngramF32RowBytes:
		encoding = compute.V41EngramRowF32
	default:
		return 0, false
	}
	return encoding, g.dim > 0 && g.dim <= compute.V41EngramRowDecodeMaxDim
}

// stored borrows the resident gather's owned panel for one synchronous upload.
// Hand-built or noncontiguous rows retain the historical concatenation path.
func (g *v41EngramGatheredRows) stored() ([]byte, bool) {
	if g == nil || g.stage == nil {
		return nil, false
	}
	stride := g.stage.rowBytes
	bytes, ok := checkedMulInt(len(g.rows), stride)
	if !ok || stride <= 0 {
		return nil, false
	}
	contiguous := g.encoded != nil && len(g.encoded) == bytes
	for i, row := range g.rows {
		if len(row) != stride {
			return nil, false
		}
		if contiguous && &row[0] != &g.encoded[i*stride] {
			contiguous = false
		}
	}
	if contiguous {
		return g.encoded, true
	}
	out := make([]byte, 0, bytes)
	for _, row := range g.rows {
		out = append(out, row...)
	}
	return out, true
}

// decodeHost is the host arm: rows decode in position-major order and the first
// refusal invalidates the layer's cache, as the host injection does.
func (g *v41EngramGatheredRows) decodeHost() ([]float32, error) {
	panel := make([]float32, len(g.rows)*g.dim)
	for i, row := range g.rows {
		values, err := decodeV41EngramRow(row, g.dim, g.stage.rowBytes)
		if err != nil {
			g.stage.caches[g.cacheIdx].invalidateRows()
			return nil, v41StageErr(v41StageEngram, g.layer, err)
		}
		copy(panel[i*g.dim:], values)
	}
	return panel, nil
}

// deviceFault returns the host decode's refusal for the first faulting row of a
// device status readback, or nil when every row decoded cleanly.
func (g *v41EngramGatheredRows) deviceFault(words []float32) error {
	for _, word := range words {
		bits := math.Float32bits(word)
		if bits == 0 {
			continue
		}
		j, nanByte, ok := compute.V41EngramRowFault(bits)
		var err error
		switch {
		case !ok || j >= g.dim:
			err = fmt.Errorf("%w: Engram row decode status %#x is malformed", ErrV41NativeUnsupported, bits)
		case nanByte && g.stage.rowBytes == V41EngramPackedRowBytes:
			err = fmt.Errorf("%w: Engram row has a NaN code/scale byte at %d", ErrV41NativeUnsupported, j)
		case nanByte:
			err = fmt.Errorf("%w: Engram row decode status %#x names a NaN byte outside the packed dialect", ErrV41NativeUnsupported, bits)
		case g.stage.rowBytes == V41EngramPackedRowBytes:
			err = fmt.Errorf("%w: Engram row value %d is non-finite", ErrV41NativeUnsupported, j)
		case g.stage.rowBytes == v41EngramIQ4NLEncodedRowBytes:
			err = fmt.Errorf("%w: Engram iq4_nl row value %d is non-finite", ErrV41NativeUnsupported, j)
		case g.stage.rowBytes == V41EngramF32RowBytes:
			err = fmt.Errorf("%w: Engram f32 row value %d is non-finite", ErrV41NativeUnsupported, j)
		default:
			err = fmt.Errorf("%w: Engram q2_k row value %d is non-finite", ErrV41NativeUnsupported, j)
		}
		g.stage.caches[g.cacheIdx].invalidateRows()
		return v41StageErr(v41StageEngram, g.layer, err)
	}
	return nil
}

// v41EngramGatherRows gathers layer l's stored rows for positions tokens
// without decoding them.
func (m *Model) v41EngramGatherRows(l, positions int, layerRows []uint32, eps float32, geometry v41EngramGeometry) (*v41EngramGatheredRows, error) {
	stage, cacheIdx, cols := geometry.stage, geometry.cacheIdx, geometry.cols
	count, ok := checkedMulInt(positions, cols)
	if positions < 0 || cols <= 0 || !ok || len(layerRows) != count {
		return nil, v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram row identifiers %d, want %d positions x %d columns", ErrV41NativeUnsupported, len(layerRows), positions, cols))
	}
	if err := v41EngramGateParamsErr(l, eps, geometry.qNorm, geometry.kNorm); err != nil {
		return nil, err
	}
	gathered, encoded, err := stage.caches[cacheIdx].gatherKnownRowsOwned(layerRows, true)
	if err != nil {
		return nil, v41StageErr(v41StageEngram, l, fmt.Errorf("%w: %w", ErrV41NativeUnsupported, err))
	}
	return &v41EngramGatheredRows{layer: l, stage: stage, cacheIdx: cacheIdx, rows: gathered, dim: geometry.dim, encoded: encoded}, nil
}

// v41EngramRowDecoder returns the backend's device row decode, or nil.
func (s *Session) v41EngramRowDecoder() compute.V41EngramRowDecodeBackend {
	if s == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	op, ok := s.Backend.(compute.V41EngramRowDecodeBackend)
	if !ok || !op.SupportsV41EngramRowDecode() {
		return nil
	}
	return op
}
