package ggufload

import (
	"fmt"
	"io"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41EngramRowBytes is the fixed encoded width of one DeepSeek-V4.1 Flash Engram
// table row: 256 E4M3 embedding values plus eight E8M0 block scales (32 values
// per scale) = 256 + 8 = 264 bytes. It is the per-row byte width the packed
// engram_embd.weight table uses, and the row stride a bounded read advances by.
const v41EngramRowBytes = 264

// v41EngramEmbdSuffix is the per-layer packed Engram table tensor suffix.
const v41EngramEmbdSuffix = "engram_embd.weight"

// v41EngramEncoding is the required metadata encoding marker for a packed V4.1
// Engram table.
const v41EngramEncoding = "e4m3_e8m0_32_row264"

// v41EngramRawI8Type is the private V4.1 raw-int8 GGUF tensor encoding code the
// converter assigns to a packed Engram table. The shared tensor enum deliberately
// does not name it, so this leaf pins the raw code locally and does not widen
// tensorPayloadBytes.
const v41EngramRawI8Type TensorType = 24

// V41EngramGGUFSource is a bounded, shard-aware model.V41EngramRowSource backed
// by a real GGUF engram_embd.weight table. It serves encoded row bytes through a
// single bounded ReaderAt per request, never materializing the whole table, and
// never dequantizing. The source borrows its WeightSource and is invalid once
// that WeightSource is closed.
type V41EngramGGUFSource struct {
	r          io.ReaderAt
	tableRows  int
	payloadOff int64
}

// V41EngramGGUFOpen resolves the packed Engram table for one decoder layer from an
// already-open WeightSource and returns a bounded row source over it.
//
// OpenWeights does not populate File.DeepSeek41Engram, so this factory first
// calls File.Config to run applyDeepSeek41Config. It then locates decoderLayer in
// DeepSeek41Engram.LayerIDs and uses the row count at the same index in
// NumEmbeddings (the row counts are declared positionally, so index alignment is
// load-bearing). It resolves blk.<layer>.engram_embd.weight and requires:
//
//   - metadata encoding e4m3_e8m0_32_row264;
//   - the private V4.1 raw-I8 tensor type (24);
//   - dims [264, rows] whose row count matches the declared count;
//   - a representable extent inside the owning shard.
//
// Every refusal is named and fires before any payload read. The returned source
// borrows ws and must not outlive it.
func V41EngramGGUFOpen(ws *WeightSource, decoderLayer int) (*V41EngramGGUFSource, error) {
	if ws == nil {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source requires a WeightSource")
	}
	if ws.File == nil {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source requires a parsed File")
	}
	if _, err := ws.File.Config(); err != nil {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source config: %w", err)
	}
	eng := ws.File.DeepSeek41Engram
	if eng == nil {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: file declares no Engram metadata")
	}
	if eng.Encoding != v41EngramEncoding {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: encoding %q is not %q", eng.Encoding, v41EngramEncoding)
	}

	idx := -1
	for i, id := range eng.LayerIDs {
		if id == decoderLayer {
			idx = i
			break
		}
	}
	if idx < 0 {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: layer %d is not an Engram layer (declared %v)", decoderLayer, eng.LayerIDs)
	}
	if idx >= len(eng.NumEmbeddings) {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: layer %d has no declared row count (%d rows entries)", decoderLayer, len(eng.NumEmbeddings))
	}
	rows := eng.NumEmbeddings[idx]
	if rows <= 0 {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: layer %d declares %d rows", decoderLayer, rows)
	}

	name := fmt.Sprintf("blk.%d.%s", decoderLayer, v41EngramEmbdSuffix)
	info, ok := ws.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: missing tensor %s", name)
	}
	if info.Type != v41EngramRawI8Type {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: tensor %s type %s is not the V4.1 raw I8 code %d", name, info.Type, v41EngramRawI8Type)
	}
	if len(info.Dims) != 2 || info.Dims[0] != v41EngramRowBytes || info.Dims[1] != uint64(rows) {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: tensor %s dims %v, want [%d %d]", name, info.Dims, v41EngramRowBytes, rows)
	}

	r, size, err := ws.tensorReader(info)
	if err != nil {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: %w", err)
	}
	// The packed table's byte extent is rows*264. tensorPayloadBytes deliberately
	// does not support the private raw-I8 code, so check the extent directly
	// against the owning shard rather than widening the shared tensor enum.
	extent := int64(rows) * int64(v41EngramRowBytes)
	if info.FileOffset < 0 || info.FileOffset > math.MaxInt64-extent || info.FileOffset+extent > size {
		return nil, fmt.Errorf("gguf: V4.1 Engram row source: tensor %s extent [%d,%d) overruns its shard (%d bytes)", name, info.FileOffset, info.FileOffset+extent, size)
	}
	return &V41EngramGGUFSource{r: r, tableRows: rows, payloadOff: info.FileOffset}, nil
}

// RowBytes returns the fixed encoded width of one Engram row.
func (s *V41EngramGGUFSource) RowBytes() int { return v41EngramRowBytes }

// TableRows returns the table's declared row count.
func (s *V41EngramGGUFSource) TableRows() int { return s.tableRows }

// ReadRows reads count contiguous encoded rows starting at row index start into
// dst via one bounded ReaderAt at payloadOff+start*264. It validates arithmetic,
// the row range, and the destination length before IO, and rejects every short
// read including a nil-error short read.
func (s *V41EngramGGUFSource) ReadRows(start, count int, dst []byte) (int, error) {
	if s.r == nil {
		return 0, fmt.Errorf("gguf: V4.1 Engram row source is closed")
	}
	if start < 0 || count < 0 {
		return 0, fmt.Errorf("gguf: V4.1 Engram ReadRows negative range start=%d count=%d", start, count)
	}
	if count == 0 {
		return 0, nil
	}
	if start > s.tableRows || count > s.tableRows-start {
		return 0, fmt.Errorf("gguf: V4.1 Engram ReadRows out of range start=%d count=%d tableRows=%d", start, count, s.tableRows)
	}
	want := count * v41EngramRowBytes
	if len(dst) < want {
		return 0, fmt.Errorf("gguf: V4.1 Engram ReadRows destination %d bytes, need %d", len(dst), want)
	}
	off := s.payloadOff + int64(start*v41EngramRowBytes)
	n, err := s.r.ReadAt(dst[:want], off)
	if err != nil {
		if err == io.EOF && n == want {
			return n, nil
		}
		return n, fmt.Errorf("gguf: V4.1 Engram ReadRows [%d,%d): %w", start, start+count, err)
	}
	if n != want {
		return n, fmt.Errorf("gguf: V4.1 Engram ReadRows [%d,%d): short read %d of %d bytes", start, start+count, n, want)
	}
	return n, nil
}

// compile-time proof the source satisfies the model-side read seam.
var _ model.V41EngramRowSource = (*V41EngramGGUFSource)(nil)
