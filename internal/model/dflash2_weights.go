package model

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"sync"
)

const (
	qwen38DFlash2WeightRevision = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
	qwen38DFlash2WeightSHA256   = "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
	qwen38DFlash2WeightBytes    = int64(3848817896)
	dflash2MaxHeaderBytes       = 1 << 20
	dflash2MaxReadElements      = 1 << 16
)

// dflash2Weights owns an immutable BF16 sidecar, not a target Model. Opening it
// does not admit execution: the future drafter must separately bind the exact
// target weights, tokenizer, template, feature taps, and borrowed embedding/head.
// The file must remain immutable until Close. Concurrent reads are supported;
// Close waits for readers, closes the file once, and rejects subsequent reads.
// No tensor or complete checkpoint is expanded into a persistent f32 allocation.
type dflash2Weights struct {
	mu       sync.RWMutex
	r        io.ReaderAt
	closer   io.Closer
	tensors  map[string]dflash2Tensor
	closed   bool
	closeErr error
}

type dflash2Tensor struct {
	shape    []int
	offset   int64
	elements int64
}

type dflash2WeightContract struct {
	sha256    string
	fileBytes int64
	shapes    map[string][]int
}

// qwen38DFlash2WeightContract describes the stock five-layer Z Lab artifact at
// the revision above, not a generic Qwen causal LM or a classic MTP head.
// Shapes are source-derived and are checked against the actual file on open.
func qwen38DFlash2WeightContract() dflash2WeightContract {
	shapes := map[string][]int{
		"fc.weight":          {5120, 25600},
		"hidden_norm.weight": {5120},
		"norm.weight":        {5120},
		"candidate_selector.hidden_projection.weight": {256, 5120},
		"candidate_selector.predecessor_codebook":     {248320, 256},
		"candidate_selector.successor_codebook":       {248320, 256},
	}
	layer := map[string][]int{
		"input_layernorm.weight":                  {5120},
		"post_attention_layernorm.weight":         {5120},
		"self_attn.q_proj.weight":                 {4096, 5120},
		"self_attn.k_proj.weight":                 {1024, 5120},
		"self_attn.v_proj.weight":                 {1024, 5120},
		"self_attn.o_proj.weight":                 {5120, 4096},
		"self_attn.q_norm.weight":                 {128},
		"self_attn.k_norm.weight":                 {128},
		"mlp.gate_proj.weight":                    {17408, 5120},
		"mlp.up_proj.weight":                      {17408, 5120},
		"mlp.down_proj.weight":                    {5120, 17408},
		"attention_conv.kernel_projection.weight": {1280, 5120},
		"attention_conv.base_kernel":              {2, 2, 5120},
		"mlp_conv.kernel_projection.weight":       {1280, 5120},
		"mlp_conv.base_kernel":                    {2, 2, 5120},
	}
	for i := 0; i < 5; i++ {
		for name, shape := range layer {
			shapes[fmt.Sprintf("layers.%d.%s", i, name)] = append([]int(nil), shape...)
		}
	}
	return dflash2WeightContract{qwen38DFlash2WeightSHA256, qwen38DFlash2WeightBytes, shapes}
}

func openQwen38DFlash2Weights(ctx context.Context, path string) (*dflash2Weights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	if !st.Mode().IsRegular() {
		return nil, errors.Join(fmt.Errorf("dflash2: weights must be a regular immutable file"), f.Close())
	}
	return newDFlash2Weights(ctx, f, st.Size(), f, qwen38DFlash2WeightContract())
}

// newDFlash2Weights owns closer on both success and failure. The configurable
// private contract permits tiny loader tests; the production opener above always
// supplies the pinned stock contract. Parsing, hashing, and reads remain bounded.
func newDFlash2Weights(ctx context.Context, r io.ReaderAt, size int64, closer io.Closer, contract dflash2WeightContract) (_ *dflash2Weights, err error) {
	keep := false
	defer func() {
		if !keep && closer != nil {
			err = errors.Join(err, closer.Close())
		}
	}()
	if ctx == nil || r == nil {
		return nil, fmt.Errorf("dflash2: context and reader are required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	wantDigest, err := hex.DecodeString(contract.sha256)
	if err != nil || len(wantDigest) != sha256.Size || len(contract.shapes) == 0 {
		return nil, fmt.Errorf("dflash2: invalid weight contract")
	}
	if size < 8 || size != contract.fileBytes {
		return nil, fmt.Errorf("dflash2: weight file size %d, want %d", size, contract.fileBytes)
	}
	var length [8]byte
	if n, err := r.ReadAt(length[:], 0); err != nil || n != len(length) {
		return nil, fmt.Errorf("dflash2: header length: %w", errors.Join(err, io.ErrUnexpectedEOF))
	}
	headerBytes := binary.LittleEndian.Uint64(length[:])
	if headerBytes == 0 || headerBytes > dflash2MaxHeaderBytes || headerBytes > uint64(size-8) {
		return nil, fmt.Errorf("dflash2: header exceeds bounded file contract")
	}
	// Keep the already bounded length: do not reread it through an allocator
	// whose limit is the entire checkpoint size. Reuse native header validation.
	header := make([]byte, 8+int(headerBytes))
	copy(header, length[:])
	if n, err := r.ReadAt(header[8:], 8); err != nil || n != len(header)-8 {
		return nil, fmt.Errorf("dflash2: header read: %w", errors.Join(err, io.ErrUnexpectedEOF))
	}
	hdr, dataBase, err := parseSafetensorsHeader(header)
	if err != nil {
		return nil, fmt.Errorf("dflash2: %w", err)
	}
	sf := &safetensorsFile{hdr: hdr, dataBase: int64(dataBase), size: size}
	tensors, err := dflash2ValidateTensors(sf, contract.shapes)
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	// Authenticate exactly the header we parsed, rather than a second read that
	// could differ. The immutable-file lifetime precondition still applies.
	_, _ = h.Write(header)
	section := io.NewSectionReader(r, sf.dataBase, size-sf.dataBase)
	buf := make([]byte, 64<<10)
	for remaining := size - sf.dataBase; remaining > 0; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := int64(len(buf))
		if remaining < n {
			n = remaining
		}
		if _, err := io.ReadFull(section, buf[:int(n)]); err != nil {
			return nil, fmt.Errorf("dflash2: hash read: %w", err)
		}
		_, _ = h.Write(buf[:int(n)])
		remaining -= n
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !bytes.Equal(h.Sum(nil), wantDigest) {
		return nil, fmt.Errorf("dflash2: weight SHA256 mismatch")
	}
	keep = true
	return &dflash2Weights{r: r, closer: closer, tensors: tensors}, nil
}

func dflash2ValidateTensors(sf *safetensorsFile, shapes map[string][]int) (map[string]dflash2Tensor, error) {
	if metadata, ok := sf.hdr["__metadata__"]; ok {
		var values map[string]string
		if err := json.Unmarshal(metadata, &values); err != nil || values == nil {
			return nil, fmt.Errorf("dflash2: invalid safetensors metadata")
		}
	}
	if len(safetensorsTensorNames(sf.hdr)) != len(shapes) {
		return nil, fmt.Errorf("dflash2: tensor key set mismatch")
	}
	tensors := make(map[string]dflash2Tensor, len(shapes))
	ranges := make([]dflash2Tensor, 0, len(shapes))
	for name, wantShape := range shapes {
		raw, ok := sf.hdr[name]
		if !ok {
			return nil, fmt.Errorf("dflash2: missing tensor %q", name)
		}
		var entry stEntry
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&entry); err != nil || entry.Dtype != "BF16" || len(entry.Shape) != len(wantShape) || len(wantShape) == 0 {
			return nil, fmt.Errorf("dflash2: tensor %q dtype/shape metadata mismatch", name)
		}
		elements := int64(1)
		for i, dim := range wantShape {
			if dim <= 0 || entry.Shape[i] != dim || elements > (math.MaxInt64/2)/int64(dim) {
				return nil, fmt.Errorf("dflash2: tensor %q invalid shape", name)
			}
			elements *= int64(dim)
		}
		start, end, err := safetensorsDataBounds(sf.dataBase, sf.size, entry)
		if err != nil || end-start != elements*2 || (start-sf.dataBase)%2 != 0 {
			return nil, fmt.Errorf("dflash2: tensor %q invalid byte range", name)
		}
		tensor := dflash2Tensor{append([]int(nil), wantShape...), start, elements}
		tensors[name] = tensor
		ranges = append(ranges, tensor)
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].offset < ranges[j].offset })
	end := sf.dataBase
	for _, tensor := range ranges {
		if tensor.offset != end {
			return nil, fmt.Errorf("dflash2: tensor ranges overlap or leave a gap")
		}
		end += tensor.elements * 2
	}
	if end != sf.size {
		return nil, fmt.Errorf("dflash2: unclaimed trailing weight bytes")
	}
	return tensors, nil
}

// ReadBF16 widens one bounded, contiguous row/span without exposing file storage.
// An error leaves dst unchanged. Successful reads may execute concurrently into
// caller-owned, disjoint destinations. The caller owns synchronization of dst.
func (w *dflash2Weights) ReadBF16(ctx context.Context, name string, first int64, dst []float32) error {
	if ctx == nil {
		return fmt.Errorf("dflash2: context is required")
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		return fmt.Errorf("dflash2: weights closed")
	}
	tensor, ok := w.tensors[name]
	if !ok || first < 0 || first > tensor.elements || int64(len(dst)) > tensor.elements-first || len(dst) > dflash2MaxReadElements {
		return fmt.Errorf("dflash2: invalid bounded tensor span %q", name)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	buf := make([]byte, 2*len(dst))
	if len(buf) != 0 {
		if n, err := w.r.ReadAt(buf, tensor.offset+2*first); err != nil || n != len(buf) {
			return fmt.Errorf("dflash2: tensor span read: %w", errors.Join(err, io.ErrUnexpectedEOF))
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	for i := range dst {
		dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[2*i:])) << 16)
	}
	return nil
}

func (w *dflash2Weights) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.closed {
		w.closed = true
		if w.closer != nil {
			w.closeErr = w.closer.Close()
		}
		w.r, w.closer, w.tensors = nil, nil, nil
	}
	return w.closeErr
}
