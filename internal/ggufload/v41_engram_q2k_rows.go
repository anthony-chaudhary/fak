package ggufload

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41EngramQ2KRowBytes is the DEQUANTIZED f32 width of one DeepSeek-V4.1 Flash
// Engram table row read from the published Q2_K encoding: qkK = 256 elements at
// 4 bytes per float32 = 1024 bytes. This is the seam's RowBytes() value, NOT the
// on-disk width.
//
// The load-bearing invariant is that ONE Engram row is exactly ONE Q2_K
// super-block: the published tensor declares dims [256, rows] and a Q2_K
// super-block covers qkK = 256 elements, so dims[0] == qkK. A row therefore
// occupies blockQ2KBytes = 84 bytes on disk ([16 scale bytes][64 q bytes][f16
// d][f16 min]) and expands to 256 float32 (1024 bytes) after dequantization. The
// byte-oriented model.V41EngramRowCache is geometry-agnostic: it reads
// count*RowBytes bytes from this source, so reporting the DEQUANTIZED width is
// correct and the source handles the 84-byte-per-row on-disk stride internally.
const v41EngramQ2KRowBytes = qkK * 4

// V41EngramQ2KErrorKind classifies fail-closed Q2_K Engram row-source refusals.
type V41EngramQ2KErrorKind string

const (
	// V41EngramQ2KTensorType is an unsupported tensor encoding. It is the typed
	// negative that a caller must never paper over with zero-filled rows.
	V41EngramQ2KTensorType V41EngramQ2KErrorKind = "tensor_type"
	// V41EngramQ2KGeometry is malformed metadata, dims, or a row-count mismatch.
	V41EngramQ2KGeometry V41EngramQ2KErrorKind = "geometry"
	// V41EngramQ2KRange is an invalid or out-of-range row read.
	V41EngramQ2KRange V41EngramQ2KErrorKind = "range"
	// V41EngramQ2KExtent is a payload extent that overruns its owning shard.
	V41EngramQ2KExtent V41EngramQ2KErrorKind = "extent"
	// V41EngramQ2KRead is a bounded-read failure (short read or IO error).
	V41EngramQ2KRead V41EngramQ2KErrorKind = "read"
)

// V41EngramQ2KError is the historical typed refusal shared by the Q2_K and
// Q4_K Engram row sources. Callers (and the negative witness) assert on the type via
// errors.As, so an unsupported encoding is distinguishable from a plain string
// and can never be mistaken for a successful zero-filled read.
type V41EngramQ2KError struct {
	Kind   V41EngramQ2KErrorKind
	Tensor string
	Type   TensorType
	Detail string
	Err    error
}

func (e *V41EngramQ2KError) Error() string {
	if e == nil {
		return "gguf: nil V4.1 Engram K-quant error"
	}
	where := ""
	if e.Tensor != "" {
		where += " tensor=" + e.Tensor
	}
	if e.Type != 0 {
		where += fmt.Sprintf(" type=%s", e.Type)
	}
	if e.Detail != "" {
		where += " " + e.Detail
	}
	if e.Err != nil {
		return fmt.Sprintf("gguf: V4.1 Engram K-quant %s%s: %v", e.Kind, where, e.Err)
	}
	return fmt.Sprintf("gguf: V4.1 Engram K-quant %s%s", e.Kind, where)
}

func (e *V41EngramQ2KError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// V41EngramKSource reads one 256-element K-quant super-block per Engram row
// from its owning shard and decodes only the requested rows to little-endian f32.
// It borrows its WeightSource and must not outlive it. The zero-value dialect is
// Q2_K, preserving existing package-local Q2_K source literals.
type V41EngramKSource struct {
	r          io.ReaderAt
	tableRows  int
	payloadOff int64
	q4K        bool
}

// V41EngramQ2KSource retains the existing Q2_K row-source API.
type V41EngramQ2KSource = V41EngramKSource

func (s *V41EngramKSource) encodedRow() (int, string) {
	if s.q4K {
		return blockQ4KBytes, model.V41EngramEncodingQ4K
	}
	return blockQ2KBytes, model.V41EngramEncodingQ2K
}

func (s *V41EngramKSource) dequantRows(dst []float32, raw []byte) {
	if s.q4K {
		dequantQ4K(dst, raw)
	} else {
		dequantQ2K(dst, raw)
	}
}

// V41EngramQ2KOpen resolves the published Q2_K Engram table for one decoder layer
// from an already-open WeightSource and returns a bounded, dequantizing row
// source over it.
//
// OpenWeights does not populate File.DeepSeek41Engram, so this factory first
// calls File.Config to run applyDeepSeek41Config, then locates decoderLayer in
// DeepSeek41Engram.LayerIDs. Unlike the row264 raw-I8 dialect it REQUIRES no
// engram.encoding marker: the published vcruz dialect omits it, and the tensor
// type (Q2_K, 10) is this source's discriminator.
//
// The row count prefers the declared DeepSeek41Engram.NumEmbeddings at the layer's
// positional index when present. A present entry must be positive AND agree with
// the tensor's dims[1]: a non-positive declared count is malformed metadata and a
// disagreeing one is a named geometry refusal, both rather than a silent
// preference for either. Only when the vcruz dialect omits engram.rows entirely
// (NumEmbeddings is empty) is the count derived from the tensor's dims[1].
//
// The resolved blk.<layer>.engram_embd.weight tensor must be:
//
//   - TensorQ2_K (type 10) with dims [qkK, rows] (dims[0] == 256); a raw-I8
//     (type 24) [264,rows] table is redirected to V41EngramGGUFOpen, and any
//     other type fails closed with a *V41EngramQ2KError of kind tensor_type;
//   - a representable rows*blockQ2KBytes extent inside the owning shard.
//
// Every refusal is named and fires before any payload read. The returned source
// borrows ws and must not outlive it.
func V41EngramQ2KOpen(ws *WeightSource, decoderLayer int) (*V41EngramQ2KSource, error) {
	return openV41EngramK(ws, decoderLayer, TensorQ2_K)
}

// V41EngramQ4KOpen opens a Q4_K Engram table with dims [256, rows]: one
// 144-byte super-block per row, served as 1024 f32 bytes. Binding names the
// encoded Q4_K bytes; neither opening nor binding expands the whole table.
// Geometry, shard ownership, and typed refusals are shared with the Q2_K path.
func V41EngramQ4KOpen(ws *WeightSource, decoderLayer int) (*V41EngramKSource, error) {
	return openV41EngramK(ws, decoderLayer, TensorQ4_K)
}

func openV41EngramK(ws *WeightSource, decoderLayer int, typ TensorType) (*V41EngramKSource, error) {
	if ws == nil {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Detail: "requires a WeightSource"}
	}
	if ws.File == nil {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Detail: "requires a parsed File"}
	}
	if _, err := ws.File.Config(); err != nil {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Detail: "config", Err: err}
	}
	eng := ws.File.DeepSeek41Engram
	if eng == nil {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Detail: "file declares no Engram metadata"}
	}

	idx := -1
	for i, id := range eng.LayerIDs {
		if id == decoderLayer {
			idx = i
			break
		}
	}
	if idx < 0 {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry,
			Detail: fmt.Sprintf("layer %d is not an Engram layer (declared %v)", decoderLayer, eng.LayerIDs)}
	}

	name := fmt.Sprintf("blk.%d.%s", decoderLayer, v41EngramEmbdSuffix)
	info, ok := ws.Tensor(name)
	if !ok {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Tensor: name, Detail: "missing tensor"}
	}
	if info.Type != typ {
		if info.Type == v41EngramRawI8Type {
			return nil, &V41EngramQ2KError{Kind: V41EngramQ2KTensorType, Tensor: name, Type: info.Type,
				Detail: "raw-I8 row264 table: use V41EngramGGUFOpen instead"}
		}
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KTensorType, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("unsupported Engram tensor type %s (want %s %d)", info.Type, typ, typ)}
	}
	if len(info.Dims) != 2 || info.Dims[0] != uint64(qkK) {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("dims %v, want [%d rows]", info.Dims, qkK)}
	}

	// Prefer the declared row count; fall back to the tensor's own dims[1] for the
	// vcruz dialect that omits engram.rows entirely (idx >= len(NumEmbeddings)).
	// A declared entry that is present must be positive AND agree with dims[1]:
	// a non-positive declared count is malformed metadata (a named refusal), and a
	// positive-but-disagreeing count is a named mismatch. Only a fully-absent
	// array falls back to dims[1].
	// Check before narrowing or multiplying: binding also addresses the decoded
	// f32 extent with int64 offsets, even though hashing reads encoded bytes.
	if info.Dims[1] > uint64(math.MaxInt) || info.Dims[1] > uint64(math.MaxInt64/v41EngramQ2KRowBytes) {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Tensor: name, Type: info.Type,
			Detail: "row count or decoded extent is not representable"}
	}
	tensorRows := int(info.Dims[1])
	rows := tensorRows
	if idx < len(eng.NumEmbeddings) {
		rows = eng.NumEmbeddings[idx]
		if rows <= 0 {
			return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Tensor: name, Type: info.Type,
				Detail: fmt.Sprintf("declared rows %d for layer %d is non-positive (want > 0 agreeing with tensor dims[1] %d)", rows, decoderLayer, tensorRows)}
		}
		if rows != tensorRows {
			return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Tensor: name, Type: info.Type,
				Detail: fmt.Sprintf("declared rows %d disagree with tensor dims[1] %d", rows, tensorRows)}
		}
	}
	if rows <= 0 {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("layer %d resolves %d rows", decoderLayer, rows)}
	}

	r, size, err := ws.tensorReader(info)
	if err != nil {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KRead, Tensor: name, Err: err}
	}
	payload, err := tensorPayloadBytes(info)
	if err != nil {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KGeometry, Tensor: name, Type: info.Type, Err: err}
	}
	stride := blockQ2KBytes
	if typ == TensorQ4_K {
		stride = blockQ4KBytes
	}
	// The geometry check bounds both the encoded and decoded int64 extents.
	extent := int64(rows) * int64(stride)
	if payload != uint64(extent) {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KExtent, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("payload %d bytes is inconsistent with %d rows * %d bytes/row = %d", payload, rows, stride, extent)}
	}
	if info.FileOffset < 0 || info.FileOffset > math.MaxInt64-extent || info.FileOffset+extent > size {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KExtent, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("extent offset=%d size=%d overruns its shard (%d bytes)", info.FileOffset, extent, size)}
	}
	return &V41EngramKSource{r: r, tableRows: rows, payloadOff: info.FileOffset, q4K: typ == TensorQ4_K}, nil
}

// RowBytes returns the DEQUANTIZED f32 width of one Engram row (qkK*4 = 1024).
func (s *V41EngramKSource) RowBytes() int { return v41EngramQ2KRowBytes }

// TableRows returns the table's resolved row count.
func (s *V41EngramKSource) TableRows() int { return s.tableRows }

// V41EngramEncodedRows names the K-quant super-blocks rows [start, start+count)
// decode from: exactly the bytes ReadRows reads for that range. A binding then
// addresses the checkpoint bytes instead of their f32 expansion.
func (s *V41EngramKSource) V41EngramEncodedRows(start, count int) (io.ReaderAt, int64, int64, string, error) {
	if s.r == nil {
		return nil, 0, 0, "", &V41EngramQ2KError{Kind: V41EngramQ2KRange, Detail: "source is closed"}
	}
	if start < 0 || count <= 0 || start > s.tableRows || count > s.tableRows-start {
		return nil, 0, 0, "", &V41EngramQ2KError{Kind: V41EngramQ2KRange,
			Detail: fmt.Sprintf("encoded range start=%d count=%d tableRows=%d", start, count, s.tableRows)}
	}
	stride, encoding := s.encodedRow()
	return s.r, s.payloadOff + int64(start)*int64(stride), int64(count) * int64(stride), encoding, nil
}

// ReadRows issues one bounded read for count encoded rows, then writes
// count*1024 f32 bytes to dst. Range and destination checks precede allocation
// and IO; all short reads are refused. A zero count is a no-op.
func (s *V41EngramKSource) ReadRows(start, count int, dst []byte) (int, error) {
	if s.r == nil {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRange, Detail: "source is closed"}
	}
	if start < 0 || count < 0 {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRange,
			Detail: fmt.Sprintf("negative range start=%d count=%d", start, count)}
	}
	if count == 0 {
		return 0, nil
	}
	if start > s.tableRows || count > s.tableRows-start {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRange,
			Detail: fmt.Sprintf("out of range start=%d count=%d tableRows=%d", start, count, s.tableRows)}
	}
	if count > math.MaxInt/v41EngramQ2KRowBytes {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRange, Detail: "decoded row byte count overflows int"}
	}
	want := count * v41EngramQ2KRowBytes
	if len(dst) < want {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRange,
			Detail: fmt.Sprintf("destination %d bytes, need %d", len(dst), want)}
	}

	stride, _ := s.encodedRow()
	rawBytes := count * stride
	raw := make([]byte, rawBytes)
	off := s.payloadOff + int64(start)*int64(stride)
	n, err := s.r.ReadAt(raw, off)
	// A ReadAt that fills the request may still return io.EOF; that is not short.
	// Any other error, or any short count, is a typed read refusal.
	if err != nil && !(err == io.EOF && n == rawBytes) {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRead,
			Detail: fmt.Sprintf("[%d,%d)", start, start+count), Err: err}
	}
	if n != rawBytes {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRead,
			Detail: fmt.Sprintf("[%d,%d): short read %d of %d bytes", start, start+count, n, rawBytes)}
	}

	// Only validated encoded rows reach the decoder. Aligned little-endian
	// destinations avoid a scratch f32 copy; other destinations serialize below.
	if v41EngramHostLittleEndian && uintptr(unsafe.Pointer(&dst[0]))%4 == 0 {
		s.dequantRows(unsafe.Slice((*float32)(unsafe.Pointer(&dst[0])), count*qkK), raw)
		return want, nil
	}
	row := make([]float32, count*qkK)
	s.dequantRows(row, raw)
	for i, v := range row {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v))
	}
	return want, nil
}

// v41EngramHostLittleEndian gates ReadRows' in-place decode: on a little-endian
// host an aligned float32 view of dst holds exactly the bytes the
// PutUint32(Float32bits) loop would write, so the decode skips the f32 scratch
// row and the per-element copy.
var v41EngramHostLittleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// compile-time proof the source satisfies the model-side read and encoded seams.
var (
	_ model.V41EngramRowSource   = (*V41EngramQ2KSource)(nil)
	_ model.V41EngramEncodedRows = (*V41EngramQ2KSource)(nil)
)
