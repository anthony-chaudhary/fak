package ggufload

// deepseek41_engram_serve.go is the ordinary-Q4K serve-load attachment for the
// DeepSeek-V4.1 Flash Engram stage (public #13662, leaf of parent #12640). Before
// this file the Q4K builder refused any GGUF that DECLARED Engram (the interim
// fence in quant_q4k_loader.go), because the row-source and forward pieces
// existed but nothing wired them. This file wires them: each declared layer's
// table is opened through the Q2_K (dequantizing) or raw-I8 Ge4 row source,
// admitted only after its bytes match a verified artifact binding, assembled into
// a model.V41EngramLayout from the artifact's own metadata, and attached to the
// built model. The fence's job is then done by the attachment itself: a declared
// layer with no openable/verified table still refuses.
//
// Scope: attachment/admission only. No forward arithmetic, no row decoder, no GPU,
// no checkpoint, no throughput or parity claim. The row sources BORROW the
// WeightSource, so the caller must keep it open for the life of the model (the
// same contract V41EngramQ2KOpen / V41EngramGGUFOpen already document).

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41EngramServeBudgetBytes bounds the resident row cache the attached stage
// keeps per layer. A handful of rows is enough for the bounded cache to function;
// this is not a memory or performance claim.
const v41EngramServeBudgetRows = 4

// v41EngramRowReaderAt adapts a model.V41EngramRowSource (row-addressed) to the
// io.ReaderAt the verified binding constructor hashes and reads through. The
// binding hashes the DECLARED extent once; row reads afterwards are the source's
// normal bounded ReadRows, so streaming is preserved. ReadAt slices whole rows so
// an unaligned or partial request is served from the same verified rows.
type v41EngramRowReaderAt struct {
	src      model.V41EngramRowSource
	rows     int
	rowBytes int
}

func (w *v41EngramRowReaderAt) RowBytes() int { return w.rowBytes }

func (w *v41EngramRowReaderAt) ReadRows(start, count int, dst []byte) (int, error) {
	return w.src.ReadRows(start, count, dst)
}

func (w *v41EngramRowReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, fmt.Errorf("gguf: V4.1 Engram ReadAt negative offset %d", off)
	}
	if len(p) == 0 {
		return 0, nil
	}
	rb := int64(w.rowBytes)
	startRow := off / rb
	endRow := (off + int64(len(p)) + rb - 1) / rb
	if startRow < 0 || endRow > int64(w.rows) {
		return 0, fmt.Errorf("gguf: V4.1 Engram ReadAt [%d,%d) outside table (%d rows)", off, off+int64(len(p)), w.rows)
	}
	buf := make([]byte, (endRow-startRow)*rb)
	n, err := w.src.ReadRows(int(startRow), int(endRow-startRow), buf)
	if err != nil {
		return 0, err
	}
	base := off - startRow*rb
	if base < 0 || base+int64(len(p)) > int64(n) {
		return 0, fmt.Errorf("gguf: V4.1 Engram ReadAt short read: have %d, need [%d,%d)", n, base, base+int64(len(p)))
	}
	copy(p, buf[base:base+int64(len(p))])
	return len(p), nil
}

// openV41EngramRowSource resolves the row source for one declared Engram layer.
// A raw-I8 (row264) table routes to V41EngramGGUFOpen; everything else goes
// through the Q2_K dequantizing source, which names its own refusals.
func openV41EngramRowSource(ws *WeightSource, decoderLayer int) (model.V41EngramRowSource, int, int, error) {
	name := fmt.Sprintf("blk.%d.%s", decoderLayer, v41EngramEmbdSuffix)
	if info, ok := ws.Tensor(name); ok && info.Type == v41EngramRawI8Type {
		src, err := V41EngramGGUFOpen(ws, decoderLayer)
		if err != nil {
			return nil, 0, 0, err
		}
		return src, src.TableRows(), src.RowBytes(), nil
	}
	src, err := V41EngramQ2KOpen(ws, decoderLayer)
	if err != nil {
		return nil, 0, 0, err
	}
	return src, src.TableRows(), src.RowBytes(), nil
}

// buildV41EngramLayout assembles the model hash layout from the parsed
// DeepSeek41Engram metadata. It invents no geometry: per-layer Primes are the
// declared flat slice split at (MaxNgramSize-1)*NHeads, per-layer Multipliers at
// MaxNgramSize, and each layer's row count is the sum of its primes (the exact
// identity V41EngramLayout.validate re-derives). CompressedVocab is the smallest
// vocabulary consistent with the declared token map and pad id; the vcruz dialect
// omits it. A shape that does not divide cleanly is refused rather than padded.
func buildV41EngramLayout(eng *DeepSeek41Engram) (model.V41EngramLayout, error) {
	if eng == nil {
		return model.V41EngramLayout{}, fmt.Errorf("gguf: no Engram metadata")
	}
	nLayers := len(eng.LayerIDs)
	maxN := eng.MaxNgramSize
	heads := eng.NHeads
	if nLayers == 0 || maxN < 2 || heads <= 0 {
		return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram geometry layers=%d max_ngram=%d heads=%d", nLayers, maxN, heads)
	}
	wantCols := (maxN - 1) * heads
	if len(eng.Primes) != nLayers*wantCols {
		return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram primes %d, want %d (layers %d * cols %d)", len(eng.Primes), nLayers*wantCols, nLayers, wantCols)
	}
	if len(eng.Multipliers) != nLayers*maxN {
		return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram multipliers %d, want %d (layers %d * max_ngram %d)", len(eng.Multipliers), nLayers*maxN, nLayers, maxN)
	}
	if len(eng.TokenMap) == 0 {
		return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram declares no token map")
	}

	tokenMap := make([]uint32, len(eng.TokenMap))
	maxID := int(eng.PadTokenID)
	for i, t := range eng.TokenMap {
		if t < 0 {
			return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram token map[%d]=%d is negative", i, t)
		}
		if t > maxID {
			maxID = t
		}
		tokenMap[i] = uint32(t)
	}
	compressed := maxID + 1
	if eng.CompressedVocabSize > compressed {
		compressed = eng.CompressedVocabSize
	}
	if eng.PadTokenID < 0 || eng.PadTokenID >= compressed {
		return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram pad id %d outside compressed vocab %d", eng.PadTokenID, compressed)
	}

	rows := make([]uint32, nLayers)
	primes := make([][]uint32, nLayers)
	multipliers := make([][]uint64, nLayers)
	for l := 0; l < nLayers; l++ {
		p := eng.Primes[l*wantCols : (l+1)*wantCols]
		rp := make([]uint32, wantCols)
		var sum uint64
		for c, v := range p {
			if v < 2 {
				return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram layer %d prime[%d]=%d is not >= 2", l, c, v)
			}
			rp[c] = uint32(v)
			sum += uint64(v)
		}
		if sum == 0 || sum > uint64(^uint32(0)) {
			return model.V41EngramLayout{}, fmt.Errorf("gguf: V4.1 Engram layer %d row count %d out of range", l, sum)
		}
		rows[l] = uint32(sum)
		primes[l] = rp
		m := eng.Multipliers[l*maxN : (l+1)*maxN]
		multipliers[l] = append([]uint64(nil), m...)
	}

	return model.V41EngramLayout{
		TokenMap:        tokenMap,
		CompressedVocab: uint32(compressed),
		PadID:           uint32(eng.PadTokenID),
		Rows:            rows,
		Multipliers:     multipliers,
		Primes:          primes,
		MaxNgramSize:    maxN,
		HeadsPerNgram:   heads,
	}, nil
}

// attachV41EngramServing attaches an executable, verified Engram stage to a
// just-built model from the WeightSource's parsed Engram declaration. It is a
// no-op for a model that declares no Engram. Every failure is a typed
// ErrV41NativeUnsupported wrapped in a V41ForwardError at the engram stage, so a
// declared-but-unsupported table still fails closed (the fence's contract,
// preserved).
func attachV41EngramServing(m *model.Model, ws *WeightSource) error {
	if m == nil || ws == nil || ws.File == nil {
		return nil
	}
	eng := ws.File.DeepSeek41Engram
	if eng == nil || len(eng.LayerIDs) == 0 {
		return nil
	}
	refuse := func(err error) error {
		return &model.V41ForwardError{Stage: "engram", Layer: eng.LayerIDs[0],
			Err: fmt.Errorf("%w: %v", model.ErrV41NativeUnsupported, err)}
	}

	layout, err := buildV41EngramLayout(eng)
	if err != nil {
		return refuse(err)
	}

	srcs := make([]model.V41EngramRowSource, len(eng.LayerIDs))
	rowBytes := 0
	for i, id := range eng.LayerIDs {
		inner, tableRows, innerRowBytes, err := openV41EngramRowSource(ws, id)
		if err != nil {
			return refuse(fmt.Errorf("layer %d: %w", id, err))
		}
		if uint32(tableRows) != layout.Rows[i] {
			return refuse(fmt.Errorf("layer %d table rows %d disagree with hash layout %d", id, tableRows, layout.Rows[i]))
		}
		if i == 0 {
			rowBytes = innerRowBytes
		} else if innerRowBytes != rowBytes {
			return refuse(fmt.Errorf("layer %d row width %d disagrees with %d (one dialect per stage)", id, innerRowBytes, rowBytes))
		}

		// Admit only after the declared extent's bytes match a binding over that
		// exact extent: the verified route, never the unverified prepared reader.
		shardName := fmt.Sprintf("blk.%d.%s", id, v41EngramEmbdSuffix)
		size := int64(tableRows) * int64(innerRowBytes)
		reader := &v41EngramRowReaderAt{src: inner, rows: tableRows, rowBytes: innerRowBytes}
		digest, err := model.V41EngramDigestExtent(reader, 0, size)
		if err != nil {
			return refuse(fmt.Errorf("layer %d: hash Engram extent: %w", id, err))
		}
		binding := model.V41EngramArtifactBinding{
			Shard: shardName, Offset: 0, Size: size,
			Rows: tableRows, RowBytes: innerRowBytes, Digest: digest,
		}
		shard := model.V41EngramShard{
			ID: shardName, ExpectedID: shardName,
			Offset: 0, Size: size, Rows: tableRows,
		}
		verified, err := model.NewV41VerifiedEngramRowSource(reader, shard, binding)
		if err != nil {
			return refuse(fmt.Errorf("layer %d: verify Engram binding: %w", id, err))
		}
		srcs[i] = verified
	}

	spec := model.V41EngramAttachSpec{
		LayerIDs:            append([]int(nil), eng.LayerIDs...),
		MaxNgramSize:        eng.MaxNgramSize,
		VocabSize:           eng.VocabSize,
		NHeads:              eng.NHeads,
		HeadDim:             eng.HeadDim,
		PadTokenID:          eng.PadTokenID,
		CompressedVocabSize: int(layout.CompressedVocab),
	}
	if err := m.AttachV41Engram(spec, layout, srcs, int64(rowBytes)*v41EngramServeBudgetRows); err != nil {
		return refuse(err)
	}
	return nil
}

// compile-time proof the adapter satisfies the binding constructor's reader seam.
var _ io.ReaderAt = (*v41EngramRowReaderAt)(nil)
