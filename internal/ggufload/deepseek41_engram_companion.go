package ggufload

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const deepSeek41CompanionHeaderLimit = 64 << 10

func engramCompanionRefusal(reason string) error {
	return fmt.Errorf("gguf: declared DeepSeek41 Engram companion: %s: %w", reason, model.ErrV41ConfigAdmission)
}

// attachDeepSeek41EngramCompanion finishes an unpublished ordinary source. The
// declared source string binds headers, not authenticated whole-file identity.
// It never changes tokenizer metadata or the authenticated-manifest open path.
// Checkpoint files must remain immutable, as for other retained shard readers.
func attachDeepSeek41EngramCompanion(ws *WeightSource, backbone string, mmap bool) error {
	arch, _ := ws.File.String("general.architecture")
	if !archIsDeepSeek41(canonicalGGUFArch(arch)) {
		return nil
	}
	p := arch + "."
	if _, present := ws.File.lookup(p + "engram.file"); !present {
		return nil
	}
	name, ok := ws.File.String(p + "engram.file")
	if !ok || name == "" || name == "." || name == ".." || filepath.Base(name) != name ||
		filepath.IsAbs(name) || strings.ContainsAny(name, `/\:`) {
		return engramCompanionRefusal("file must be a nonempty sibling filename")
	}
	source, ok := ws.File.String(p + "engram.source")
	if !ok || source == "" {
		return engramCompanionRefusal("missing source identity")
	}
	layers, err := engramCompanionArray(ws.File, p, ds41KeyEngramLayerIDs, ds41LegacyEngramLayerIDs)
	if err != nil {
		return err
	}
	rows, err := engramCompanionArray(ws.File, p, ds41KeyEngramNumEmbedding, "engram.num_embeddings", ds41LegacyEngramNumEmbedding)
	if err != nil {
		return err
	}
	width, err := engramCompanionWidth(ws.File, p)
	if err != nil {
		return err
	}
	blocks, validBlocks := ws.File.Uint64(p + "block_count")
	if len(layers) == 0 || len(layers) > deepSeek41CompanionHeaderLimit/32 || len(rows) != len(layers) ||
		!validBlocks || blocks == 0 || width != qkK {
		return engramCompanionRefusal("invalid declared layer/row/width geometry")
	}
	want := make(map[string]uint64, len(layers))
	for i, layer := range layers {
		key := fmt.Sprintf("blk.%d.%s", layer, v41EngramEmbdSuffix)
		if layer < 0 || uint64(layer) >= blocks || rows[i] <= 0 || want[key] != 0 {
			return engramCompanionRefusal("invalid or duplicate declared layer/row")
		}
		if _, present := ws.byName[key]; present {
			return engramCompanionRefusal("table already exists in backbone: " + key)
		}
		want[key] = uint64(rows[i])
	}
	path := filepath.Join(filepath.Dir(backbone), name)
	f, gg, size, err := openBoundedEngramCompanion(path, len(layers))
	if err != nil {
		return err
	}
	owned := true
	defer func() {
		if owned {
			_ = f.Close()
		}
	}()
	companionArch, _ := gg.String("general.architecture")
	companionSource, _ := gg.String(p + "engram.source")
	if companionArch != "deepseek41-engram" || companionSource != source {
		return engramCompanionRefusal("architecture/source identity mismatch")
	}
	seen := make(map[string]bool, len(layers))
	type extent struct{ start, end int64 }
	extents := make([]extent, 0, len(layers))
	for _, info := range gg.Tensors {
		if seen[info.Name] || want[info.Name] == 0 || len(info.Dims) != 2 ||
			info.Dims[0] != uint64(width) || info.Dims[1] != want[info.Name] {
			return engramCompanionRefusal("unexpected, duplicate, or mismatched table " + info.Name)
		}
		switch info.Type {
		case TensorQ2_K, TensorQ4_K, TensorIQ4_NL:
		default:
			return engramCompanionRefusal("unsupported table encoding " + info.Type.String())
		}
		n, err := tensorPayloadBytes(info)
		if err != nil || n > math.MaxInt64 || info.FileOffset < 0 || info.FileOffset > size || n > uint64(size-info.FileOffset) {
			return engramCompanionRefusal("table extent overruns companion: " + info.Name)
		}
		seen[info.Name] = true
		extents = append(extents, extent{info.FileOffset, info.FileOffset + int64(n)})
	}
	sort.Slice(extents, func(i, j int) bool { return extents[i].start < extents[j].start })
	for i := 1; i < len(extents); i++ {
		if extents[i].start < extents[i-1].end {
			return engramCompanionRefusal("overlapping table extents")
		}
	}
	// No Config or payload read is needed. retainShardReader owns f on every
	// return; a mapped companion and an ordinary one share this lifetime.
	r, size, closer, data, err := retainShardReader(path, f, size, mmap)
	owned = false
	if err != nil {
		return err
	}
	view := *ws.File
	view.Tensors = append(slices.Clone(ws.File.Tensors), gg.Tensors...)
	readers := make([]io.ReaderAt, len(view.Tensors))
	sizes := make([]int64, len(view.Tensors))
	mappings := make([][]byte, len(view.Tensors))
	byName := make(map[string]int, len(view.Tensors))
	for i, info := range view.Tensors {
		byName[info.Name] = i
		if i >= len(ws.File.Tensors) {
			readers[i], sizes[i], mappings[i] = r, size, data
			if mappings[i] == nil {
				// Explicit empty view prevents fallback to the backbone's mmap.
				mappings[i] = []byte{}
			}
			continue
		}
		readers[i], sizes[i], mappings[i] = ws.r, ws.size, ws.data
		if i < len(ws.readerFor) && ws.readerFor[i] != nil {
			readers[i], sizes[i] = ws.readerFor[i], ws.sizeFor[i]
		}
		if i < len(ws.dataFor) && ws.dataFor[i] != nil {
			mappings[i] = ws.dataFor[i]
		}
	}
	// Do not copy WeightSource: its readLife and lazy auxiliary callback belong
	// to this exact owner. Only the unpublished directory/routing is extended.
	ws.File, ws.byName = &view, byName
	ws.readerFor, ws.sizeFor, ws.dataFor = readers, sizes, mappings
	ws.closers = append(ws.closers, closer)
	return nil
}

func engramCompanionArray(f *File, p string, keys ...string) ([]int, error) {
	var result []int
	found := false
	for _, key := range keys {
		if _, present := f.lookup(p + key); !present {
			continue
		}
		values, valid := f.IntArray(p + key)
		if !valid || (found && !slices.Equal(result, values)) {
			return nil, engramCompanionRefusal("invalid or conflicting " + p + key)
		}
		result, found = values, true
	}
	if !found {
		return nil, engramCompanionRefusal("missing " + p + keys[0])
	}
	return result, nil
}

func engramCompanionWidth(f *File, p string) (uint64, error) {
	var width uint64
	found := false
	for _, key := range []string{ds41KeyEngramKeyLength, ds41LegacyEngramHeadDim} {
		if _, present := f.lookup(p + key); !present {
			continue
		}
		x, valid := f.Uint64(p + key)
		if !valid || (found && x != width) {
			return 0, engramCompanionRefusal("invalid or conflicting " + p + key)
		}
		width, found = x, true
	}
	return width, nil
}

func openBoundedEngramCompanion(path string, tables int) (*os.File, *File, int64, error) {
	st, err := os.Lstat(path)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("gguf: declared Engram companion: %w", err)
	}
	if !st.Mode().IsRegular() {
		return nil, nil, 0, engramCompanionRefusal("companion must be a regular sibling file, not a symlink")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	fail := func(err error) (*os.File, *File, int64, error) {
		_ = f.Close()
		return nil, nil, 0, err
	}
	opened, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	if !opened.Mode().IsRegular() || !os.SameFile(st, opened) {
		return fail(engramCompanionRefusal("companion changed while opening"))
	}
	var prefix [24]byte
	if _, err := f.ReadAt(prefix[:], 0); err != nil {
		return fail(err)
	}
	if binary.LittleEndian.Uint64(prefix[8:16]) != uint64(tables) || binary.LittleEndian.Uint64(prefix[16:24]) > 16 {
		return fail(engramCompanionRefusal("unexpected table count or oversized metadata header"))
	}
	limit := min(opened.Size(), int64(deepSeek41CompanionHeaderLimit))
	header := make([]byte, int(limit))
	if _, err := f.ReadAt(header, 0); err != nil {
		return fail(err)
	}
	// A limited reader alone does not bound the generic GGUF parser's
	// variable-width array allocation. Companion metadata is scalar-only;
	// reject arrays before value() can allocate their declared element count.
	rr := countingReader{r: bytes.NewReader(header[24:]), n: 24, mapped: limit}
	keys := make(map[string]bool, 16)
	for i := uint64(0); i < binary.LittleEndian.Uint64(prefix[16:24]); i++ {
		key, err := rr.str()
		if err != nil {
			return fail(err)
		}
		typ, err := rr.valueType()
		if err != nil {
			return fail(err)
		}
		if keys[key] || typ == TypeArray {
			return fail(engramCompanionRefusal("duplicate key or array metadata in scalar companion header"))
		}
		keys[key] = true
		if _, err := rr.value(typ); err != nil {
			return fail(err)
		}
	}
	gg, err := ReadSize(bytes.NewReader(header), limit)
	if err != nil {
		return fail(fmt.Errorf("gguf: bounded Engram companion header: %w", err))
	}
	return f, gg, opened.Size(), nil
}
