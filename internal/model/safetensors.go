package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// This file closes an honesty gap the adversarial review caught: weights.go reads a
// torch-DECODED weights.f32, so the bf16->f32 decode is done in Python, not Go —
// "Go diffs torch's decode against itself" is a tautology. Here is a pure-Go
// safetensors reader + bf16/f16->f32 decoder, so the WEIGHTS are Go-authored too.
// Its witness (safetensors_test.go) is bitwise equality to the torch export: bf16->f32
// is a lossless 16-bit widening, and f16->f32 is the exact IEEE-754 half widening
// production fp16 checkpoints need.

type stEntry struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"` // [start, end) into the data buffer
}

type safetensorsFile struct {
	r        io.ReaderAt
	closer   io.Closer
	hdr      map[string]json.RawMessage
	dataBase int64
	size     int64
	// data is the read-only memory-mapped file bytes when the file was opened via mmap, else
	// nil. When set, tensorBytes returns a zero-copy slice into it instead of a per-tensor
	// heap ReadAt, so a single-file checkpoint is never fully resident in the process heap.
	data []byte
}

// openSafetensorsFile opens a single-file safetensors checkpoint, preferring a read-only
// memory map (zero-copy per-tensor slices, the whole file never resident in the process heap)
// and falling back to os.Open + per-tensor ReadAt where mmap is unavailable or fails. Both
// paths produce a byte-identical decode (TestLoadSafetensorsMmapMatchesReadFile).
func openSafetensorsFile(path string) (*safetensorsFile, error) {
	if sf, err := openSafetensorsFileMmap(path); err == nil {
		return sf, nil
	}
	return openSafetensorsFileReadAt(path)
}

// openSafetensorsFileMmap forces the memory-mapped path; it returns errMmapUnsupported (or a
// header-parse error) when the file cannot be mapped/parsed, never silently falling back.
func openSafetensorsFileMmap(path string) (*safetensorsFile, error) {
	data, closer, err := mmapOpen(path)
	if err != nil {
		return nil, err
	}
	return newSafetensorsFileMmap(data, closer)
}

// openSafetensorsFileReadAt forces the portable os.Open + ReadAt path (one transient tensor
// buffer at a time, never the whole file). This is the fallback used on platforms without an
// mmap impl and the historical behaviour the mmap path is proven byte-identical against.
func openSafetensorsFileReadAt(path string) (*safetensorsFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	return newSafetensorsFile(f, st.Size(), f)
}

// newSafetensorsFileMmap parses the header out of the mapped bytes and records them so
// tensorBytes can return zero-copy slices. closer munmaps the region; newSafetensorsFile
// closes it on a header-parse error, so there is no leak on the error path.
func newSafetensorsFileMmap(data []byte, closer io.Closer) (*safetensorsFile, error) {
	sf, err := newSafetensorsFile(bytes.NewReader(data), int64(len(data)), closer)
	if err != nil {
		return nil, err
	}
	sf.data = data
	return sf, nil
}

func newSafetensorsFile(r io.ReaderAt, size int64, closer io.Closer) (*safetensorsFile, error) {
	if size < 8 {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, fmt.Errorf("safetensors: short file")
	}
	var lenBuf [8]byte
	if _, err := r.ReadAt(lenBuf[:], 0); err != nil {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, fmt.Errorf("safetensors: header length: %w", err)
	}
	hlen := binary.LittleEndian.Uint64(lenBuf[:])
	if hlen > uint64(size-8) {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, fmt.Errorf("safetensors: header overruns file")
	}
	maxInt := int64(^uint(0) >> 1)
	if int64(hlen) > maxInt {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, fmt.Errorf("safetensors: header too large")
	}
	header := make([]byte, int(hlen))
	if _, err := r.ReadAt(header, 8); err != nil {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, fmt.Errorf("safetensors: header read: %w", err)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(header, &hdr); err != nil {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, fmt.Errorf("safetensors: header json: %w", err)
	}
	return &safetensorsFile{r: r, closer: closer, hdr: hdr, dataBase: 8 + int64(hlen), size: size}, nil
}

// Close releases the underlying file or memory map (munmap on the mmap path, file
// close otherwise); it is a no-op when the file holds no closer.
func (sf *safetensorsFile) Close() error {
	if sf.closer == nil {
		return nil
	}
	return sf.closer.Close()
}

func (sf *safetensorsFile) tensorBytes(e stEntry) ([]byte, error) {
	start, end, err := safetensorsDataBounds(sf.dataBase, sf.size, e)
	if err != nil {
		return nil, err
	}
	n := end - start
	maxInt := int64(^uint(0) >> 1)
	if n > maxInt {
		return nil, fmt.Errorf("safetensors: tensor too large")
	}
	if sf.data != nil {
		// Zero-copy: slice directly into the read-only mmap. safetensorsDataBounds already
		// proved [start,end) is within the file, so len(sf.data)==sf.size makes this in-range.
		// The three-index cap stops any downstream append from scribbling past this tensor
		// into the next one's bytes. Every consumer only READS this slice (decoding/quantizing
		// into freshly-allocated owned memory) before the next tensor, so the source is freed
		// before the next — the single-file analogue of LoadSafetensorsQuantDir's per-shard free.
		return sf.data[start:end:end], nil
	}
	b := make([]byte, int(n))
	if _, err := sf.r.ReadAt(b, start); err != nil {
		return nil, fmt.Errorf("safetensors: tensor read: %w", err)
	}
	return b, nil
}

func parseSafetensorsHeader(buf []byte) (map[string]json.RawMessage, int, error) {
	if len(buf) < 8 {
		return nil, 0, fmt.Errorf("safetensors: short file")
	}
	hlen := binary.LittleEndian.Uint64(buf[:8])
	if hlen > uint64(len(buf)-8) {
		return nil, 0, fmt.Errorf("safetensors: header overruns file")
	}
	maxInt := int64(^uint(0) >> 1)
	if int64(hlen) > maxInt {
		return nil, 0, fmt.Errorf("safetensors: header too large")
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(buf[8:8+int(hlen)], &hdr); err != nil {
		return nil, 0, fmt.Errorf("safetensors: header json: %w", err)
	}
	return hdr, 8 + int(hlen), nil
}

func safetensorsTensorNames(hdr map[string]json.RawMessage) []string {
	names := make([]string, 0, len(hdr))
	for name := range hdr {
		if name == "__metadata__" {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func safetensorsTiedHeader(hdr map[string]json.RawMessage) bool {
	_, hasHead := hdr["lm_head.weight"]
	return !hasHead
}

func safetensorsDataBounds(dataBase, fileSize int64, e stEntry) (int64, int64, error) {
	if len(e.DataOffsets) != 2 {
		return 0, 0, fmt.Errorf("safetensors: malformed data_offsets")
	}
	relStart, relEnd := int64(e.DataOffsets[0]), int64(e.DataOffsets[1])
	if relStart < 0 || relEnd < relStart {
		return 0, 0, fmt.Errorf("safetensors: malformed data_offsets")
	}
	dataLen := fileSize - dataBase
	if relEnd > dataLen {
		return 0, 0, fmt.Errorf("safetensors: tensor data overruns file")
	}
	return dataBase + relStart, dataBase + relEnd, nil
}

func safetensorsBufferBytes(buf []byte, dataBase int, e stEntry) ([]byte, error) {
	start, end, err := safetensorsDataBounds(int64(dataBase), int64(len(buf)), e)
	if err != nil {
		return nil, err
	}
	return buf[int(start):int(end)], nil
}

func decodeSafetensorF32(name string, e stEntry, src []byte) ([]byte, error) {
	switch e.Dtype {
	case "BF16":
		if len(src)%2 != 0 {
			return nil, fmt.Errorf("safetensors: bf16 byte length is odd for %s", name)
		}
		return decodeBF16(src), nil
	case "F16":
		if len(src)%2 != 0 {
			return nil, fmt.Errorf("safetensors: f16 byte length is odd for %s", name)
		}
		return decodeF16(src), nil
	case "F32":
		if len(src)%4 != 0 {
			return nil, fmt.Errorf("safetensors: f32 byte length is not divisible by 4 for %s", name)
		}
		return src, nil
	default:
		return nil, fmt.Errorf("safetensors: unsupported dtype %q for %s", e.Dtype, name)
	}
}

// LoadSafetensors reads a HuggingFace .safetensors file in Go (header + bf16 decode),
// streaming one source tensor at a time into the resident f32 Model. No torch in the loop.
// Tied embeddings are handled by Model.lmHead (no lm_head tensor in the file).
func LoadSafetensors(path string, cfg Config) (*Model, error) {
	sf, err := openSafetensorsFile(path)
	if err != nil {
		return nil, err
	}
	defer sf.Close()
	return loadSafetensorsFile(sf, cfg)
}

// LoadSafetensorsDir loads a HuggingFace snapshot directory. If the directory has a
// model.safetensors.index.json weight map, each shard is streamed into the same packed f32
// Model representation as LoadSafetensors; otherwise it falls back to model.safetensors.
func LoadSafetensorsDir(dir string, cfg Config) (*Model, error) {
	return loadSafetensorsDir(dir, cfg, openSafetensorsFile)
}

func loadSafetensorsDir(dir string, cfg Config, open safetensorsFileOpener) (*Model, error) {
	idxPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(idxPath); err != nil {
		return loadSafetensorsFilePath(filepath.Join(dir, "model.safetensors"), cfg, open)
	}
	ib, err := os.ReadFile(idxPath)
	if err != nil {
		return nil, err
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(ib, &index); err != nil {
		return nil, fmt.Errorf("safetensors index: %w", err)
	}
	shardSet := map[string]bool{}
	for _, shard := range index.WeightMap {
		shardSet[shard] = true
	}
	shards := make([]string, 0, len(shardSet))
	for shard := range shardSet {
		shards = append(shards, shard)
	}
	sort.Strings(shards)

	man := map[string]tensorMeta{}
	var raw []byte
	off := 0
	for _, shard := range shards {
		sf, err := open(filepath.Join(dir, shard))
		if err != nil {
			return nil, fmt.Errorf("shard %s: %w", shard, err)
		}
		err = func() error {
			defer sf.Close()
			return appendSafetensorsFileInto(sf, man, &raw, &off, cfg)
		}()
		if err != nil {
			return nil, fmt.Errorf("shard %s: %w", shard, err)
		}
	}
	return newModel(cfg, man, raw)
}

func loadSafetensorsFilePath(path string, cfg Config, open safetensorsFileOpener) (*Model, error) {
	sf, err := open(path)
	if err != nil {
		return nil, err
	}
	defer sf.Close()
	return loadSafetensorsFile(sf, cfg)
}

func loadSafetensorsFile(sf *safetensorsFile, cfg Config) (*Model, error) {
	// Decode every tensor into one packed f32 buffer + manifest, mirroring the torch
	// export's layout (offset = running byte position) so Model.tensor works as-is.
	man := map[string]tensorMeta{}
	var raw []byte
	off := 0
	if err := appendSafetensorsFileInto(sf, man, &raw, &off, cfg); err != nil {
		return nil, err
	}
	return newModel(cfg, man, raw)
}

// skipLoadTensor drops tensors the text forward never reads BEFORE they are decoded into
// the f32 buffer. For Qwen3.5/Qwen3-Next that is the vision tower ("model.visual.") and the
// multi-token-prediction head ("mtp."), which together would otherwise expand to several
// GB of f32 and OOM the load on a memory-tight box. The same two tensor families exist in
// GLM-5.2 (glm_moe_dsa): a multimodal vision encoder and an MTP head for speculative
// decoding, neither read by the text causal-LM forward pass, so the skip is generalized to
// any model whose config marks it mtp-bearing. No-op for a plain Llama/Qwen dense checkpoint.
func skipLoadTensor(cfg Config, name string) bool {
	if !cfg.dropsMtpAndVisualAtLoad() {
		return false
	}
	return strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "mtp.")
}

// dropsMtpAndVisualAtLoad reports whether the load path should drop the vision tower and
// MTP-head tensors before decoding. True for Qwen3.5/Qwen3-Next hybrid checkpoints (the
// original OOM-avoidance gate), for GLM-family checkpoints, and for MiniMax-M3, which
// carry a multimodal vision encoder plus an MTP speculative-decoding head that the text
// forward never reads. False for every dense Llama/Qwen/Mistral/etc. checkpoint, so the
// load path there is unchanged.
func (c Config) dropsMtpAndVisualAtLoad() bool {
	return c.IsQwen35Hybrid() || c.isGLM() || c.isMiniMax()
}

func appendSafetensorsFileInto(sf *safetensorsFile, man map[string]tensorMeta, raw *[]byte, off *int, cfg Config) error {
	names := safetensorsTensorNames(sf.hdr)
	// Pre-size the f32 buffer to the exact kept-tensor total so the per-tensor appends
	// never reallocate. Without this, append's grow-and-copy transiently holds two copies
	// of a multi-GB buffer and OOMs the load on a memory-tight box (e.g. a 1.9B model =
	// ~7.5 GB f32 spiking past WSL's 15 GB). Gated on the mtp/vision-bearing models that
	// drop tensors at load and therefore need an exact pre-size to avoid the reallocate.
	if cfg.dropsMtpAndVisualAtLoad() {
		need := len(*raw)
		for _, name := range names {
			if skipLoadTensor(cfg, name) {
				continue
			}
			var e stEntry
			if json.Unmarshal(sf.hdr[name], &e) != nil {
				continue
			}
			elems, ok := checkedShapeProduct(e.Shape...)
			if !ok {
				return fmt.Errorf("safetensors: tensor %s shape %v overflows element count", name, e.Shape)
			}
			if elems > (math.MaxInt-need)/4 {
				return fmt.Errorf("safetensors: tensor %s f32 byte size overflows int", name)
			}
			need += elems * 4
		}
		if cap(*raw) < need {
			grown := make([]byte, len(*raw), need)
			copy(grown, *raw)
			*raw = grown
		}
	}
	consumed := map[string]bool{}
	for _, name := range names {
		if consumed[name] {
			continue
		}
		if skipLoadTensor(cfg, name) {
			continue
		}
		if _, ok := man[name]; ok {
			return fmt.Errorf("safetensors: duplicate tensor %s", name)
		}
		if strings.HasSuffix(name, "_blocks") {
			base := strings.TrimSuffix(name, "_blocks")
			scaleName := base + "_scales"
			scaleRaw, ok := sf.hdr[scaleName]
			if ok {
				var blockEntry, scaleEntry stEntry
				if err := json.Unmarshal(sf.hdr[name], &blockEntry); err != nil {
					return fmt.Errorf("safetensors: entry %s: %w", name, err)
				}
				if err := json.Unmarshal(scaleRaw, &scaleEntry); err != nil {
					return fmt.Errorf("safetensors: entry %s: %w", scaleName, err)
				}
				blocks, err := sf.tensorBytes(blockEntry)
				if err != nil {
					return fmt.Errorf("safetensors: tensor %s: %w", name, err)
				}
				scales, err := sf.tensorBytes(scaleEntry)
				if err != nil {
					return fmt.Errorf("safetensors: tensor %s: %w", scaleName, err)
				}
				f32, shape, err := decodeMXFP4Blocks(base, blockEntry, scaleEntry, blocks, scales)
				if err != nil {
					return err
				}
				if _, ok := man[base]; ok {
					return fmt.Errorf("safetensors: duplicate tensor %s", base)
				}
				man[base] = tensorMeta{Dtype: "f32", Shape: shape, Offset: *off, Nbytes: len(f32)}
				*raw = append(*raw, f32...)
				*off += len(f32)
				consumed[name] = true
				consumed[scaleName] = true
				continue
			}
		}
		var e stEntry
		if err := json.Unmarshal(sf.hdr[name], &e); err != nil {
			return fmt.Errorf("safetensors: entry %s: %w", name, err)
		}
		if skipSafetensorsTensor(name, e) {
			continue
		}
		src, err := sf.tensorBytes(e)
		if err != nil {
			return fmt.Errorf("safetensors: tensor %s: %w", name, err)
		}
		f32, err := decodeSafetensorF32(name, e, src)
		if err != nil {
			return err
		}
		man[name] = tensorMeta{Dtype: "f32", Shape: e.Shape, Offset: *off, Nbytes: len(f32)}
		*raw = append(*raw, f32...)
		*off += len(f32)
	}
	return nil
}

func skipSafetensorsTensor(name string, e stEntry) bool {
	return e.Dtype == "U8" &&
		strings.HasPrefix(name, "gpt_neox.layers.") &&
		strings.HasSuffix(name, ".attention.bias")
}

var mxfp4Values = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6,
}

func decodeMXFP4Blocks(name string, blocksEntry, scalesEntry stEntry, blocks, scales []byte) ([]byte, []int, error) {
	if blocksEntry.Dtype != "U8" {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 blocks %s dtype %q, want U8", name, blocksEntry.Dtype)
	}
	if scalesEntry.Dtype != "U8" {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 scales %s dtype %q, want U8", name, scalesEntry.Dtype)
	}
	if len(blocksEntry.Shape) != 4 {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 blocks %s shape %v, want rank-4 [E,R,G,B]", name, blocksEntry.Shape)
	}
	E, R, G, B := blocksEntry.Shape[0], blocksEntry.Shape[1], blocksEntry.Shape[2], blocksEntry.Shape[3]
	for _, dim := range []int{E, R, G, B} {
		if dim <= 0 {
			return nil, nil, fmt.Errorf("safetensors: MXFP4 blocks %s shape %v has invalid dimension %d", name, blocksEntry.Shape, dim)
		}
	}
	if !sameShape(scalesEntry.Shape, []int{E, R, G}) {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 scales %s shape %v, want %v", name, scalesEntry.Shape, []int{E, R, G})
	}
	blockElems, ok := checkedShapeProduct(E, R, G, B)
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 blocks %s shape %v overflows element count", name, blocksEntry.Shape)
	}
	scaleElems, ok := checkedShapeProduct(E, R, G)
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 scales %s shape %v overflows element count", name, scalesEntry.Shape)
	}
	if len(blocks) != blockElems {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 blocks %s has %d bytes, shape implies %d", name, len(blocks), blockElems)
	}
	if len(scales) != scaleElems {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 scales %s has %d bytes, shape implies %d", name, len(scales), scaleElems)
	}
	C, ok := checkedShapeProduct(G, B, 2)
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 blocks %s decoded column count overflows", name)
	}
	outElems, ok := checkedShapeProduct(E, C, R)
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: MXFP4 blocks %s decoded shape [%d %d %d] overflows element count", name, E, C, R)
	}
	outVals := make([]float32, outElems)
	for e := 0; e < E; e++ {
		for r := 0; r < R; r++ {
			for g := 0; g < G; g++ {
				exp := int(scales[(e*R+r)*G+g]) - 127
				for b := 0; b < B; b++ {
					packed := blocks[((e*R+r)*G+g)*B+b]
					c := g*B*2 + b*2
					outVals[(e*C+c)*R+r] = float32(math.Ldexp(float64(mxfp4Values[packed&0x0f]), exp))
					outVals[(e*C+c+1)*R+r] = float32(math.Ldexp(float64(mxfp4Values[packed>>4]), exp))
				}
			}
		}
	}
	out := make([]byte, len(outVals)*4)
	for i, v := range outVals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out, []int{E, C, R}, nil
}

func checkedShapeProduct(dims ...int) (int, bool) {
	n := 1
	for _, d := range dims {
		if d <= 0 || n > int(^uint(0)>>1)/d {
			return 0, false
		}
		n *= d
	}
	return n, true
}

// decodeBF16 widens little-endian bf16 to little-endian f32. bf16 is the top 16 bits
// of an IEEE-754 f32, so the widening is lossless: f32bits = uint32(bf16) << 16. This
// is exactly what torch's .float() does, hence bitwise-identical (incl. inf/nan/subnormal).
func decodeBF16(b []byte) []byte {
	n := len(b) / 2
	out := make([]byte, n*4)
	for i := 0; i < n; i++ {
		u16 := binary.LittleEndian.Uint16(b[i*2:])
		binary.LittleEndian.PutUint32(out[i*4:], uint32(u16)<<16)
	}
	return out
}

// decodeF16 widens little-endian IEEE-754 binary16 to little-endian f32. Unlike bf16,
// f16 has a different exponent bias and subnormal layout, so the conversion is explicit
// instead of a shift.
func decodeF16(b []byte) []byte {
	n := len(b) / 2
	out := make([]byte, n*4)
	for i := 0; i < n; i++ {
		u16 := binary.LittleEndian.Uint16(b[i*2:])
		binary.LittleEndian.PutUint32(out[i*4:], f16bitsToF32bits(u16))
	}
	return out
}

func f16bitsToF32bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1f)
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return sign
		}
		exp = -14
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x03ff
		return sign | uint32(exp+127)<<23 | frac<<13
	case 0x1f:
		return sign | 0x7f800000 | frac<<13
	default:
		return sign | uint32(exp-15+127)<<23 | frac<<13
	}
}

// f32frombits is exported for the witness test's per-element comparison.
func f32frombits(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }
