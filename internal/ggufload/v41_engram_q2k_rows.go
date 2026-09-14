package ggufload

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

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

// V41EngramQ2KError is the typed fail-closed refusal returned by the Q2_K Engram
// row source. Callers (and the negative witness) assert on the type via
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
		return "gguf: nil V4.1 Engram Q2_K error"
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
		return fmt.Sprintf("gguf: V4.1 Engram Q2_K %s%s: %v", e.Kind, where, e.Err)
	}
	return fmt.Sprintf("gguf: V4.1 Engram Q2_K %s%s", e.Kind, where)
}

func (e *V41EngramQ2KError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// V41EngramQ2KSource is a bounded, shard-aware model.V41EngramRowSource over the
// published DeepSeek-V4.1-Flash Q2_K engram_embd.weight table. It reads the
// quantized 84-byte super-blocks with one bounded ReaderAt per request and
// dequantizes them to f32 rows; it never returns zero-filled bytes for an
// unsupported tensor type (the factory refuses first). The source borrows its
// WeightSource and is invalid once that WeightSource is closed.
type V41EngramQ2KSource struct {
	r          io.ReaderAt
	tableRows  int
	payloadOff int64
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
	if info.Type != TensorQ2_K {
		if info.Type == v41EngramRawI8Type {
			return nil, &V41EngramQ2KError{Kind: V41EngramQ2KTensorType, Tensor: name, Type: info.Type,
				Detail: "raw-I8 row264 table: use V41EngramGGUFOpen instead"}
		}
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KTensorType, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("unsupported Engram tensor type %s (want Q2_K %d)", info.Type, TensorQ2_K)}
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
	// The payload must be exactly rows*blockQ2KBytes: dims[1] supplies both rows
	// and the derived payload, so this equality is the self-consistency guard
	// (a >/< comparison here would be identically false). The real overrun guard
	// is the shard-extent check below.
	if payload != uint64(rows)*blockQ2KBytes {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KExtent, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("payload %d bytes is inconsistent with %d rows * %d bytes/row = %d", payload, rows, blockQ2KBytes, uint64(rows)*blockQ2KBytes)}
	}
	extent := int64(rows) * int64(blockQ2KBytes)
	if info.FileOffset < 0 || info.FileOffset > math.MaxInt64-extent || info.FileOffset+extent > size {
		return nil, &V41EngramQ2KError{Kind: V41EngramQ2KExtent, Tensor: name, Type: info.Type,
			Detail: fmt.Sprintf("extent [%d,%d) overruns its shard (%d bytes)", info.FileOffset, info.FileOffset+extent, size)}
	}
	return &V41EngramQ2KSource{r: r, tableRows: rows, payloadOff: info.FileOffset}, nil
}

// RowBytes returns the DEQUANTIZED f32 width of one Engram row (qkK*4 = 1024).
func (s *V41EngramQ2KSource) RowBytes() int { return v41EngramQ2KRowBytes }

// TableRows returns the table's resolved row count.
func (s *V41EngramQ2KSource) TableRows() int { return s.tableRows }

// ReadRows reads count contiguous Q2_K rows starting at row index start. It issues
// ONE bounded ReaderAt at payloadOff+start*blockQ2KBytes for count*84 raw bytes,
// dequantizes the whole buffer with dequantQ2K, and writes count*1024 f32 bytes
// into dst. It validates the range and destination length before any IO, and
// rejects every short read including a nil-error short read. count==0 is a no-op.
func (s *V41EngramQ2KSource) ReadRows(start, count int, dst []byte) (int, error) {
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
	want := count * v41EngramQ2KRowBytes
	if len(dst) < want {
		return 0, &V41EngramQ2KError{Kind: V41EngramQ2KRange,
			Detail: fmt.Sprintf("destination %d bytes, need %d", len(dst), want)}
	}

	rawBytes := count * blockQ2KBytes
	raw := make([]byte, rawBytes)
	off := s.payloadOff + int64(start)*int64(blockQ2KBytes)
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

	// dequantQ2K writes qkK float32 per 84-byte super-block; count super-blocks
	// fill row[:want] exactly. The dequantized row is then serialized into the
	// caller's byte-oriented dst as little-endian float32. Never zero-fill: the
	// typed refusals above guarantee only a validated Q2_K range reaches here.
	row := make([]float32, count*qkK)
	dequantQ2K(row, raw)
	for i, v := range row {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v))
	}
	return want, nil
}

// compile-time proof the source satisfies the model-side read seam.
var _ model.V41EngramRowSource = (*V41EngramQ2KSource)(nil)
