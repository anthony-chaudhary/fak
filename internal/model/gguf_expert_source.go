package model

import (
	"errors"
	"fmt"
	"io"
	"math"
)

// gguf_expert_source.go — ggufExpertSource, the lazy per-expert range-read source for fused
// GGUF routed-expert slabs (#3174, the M11 expert-weight residency/streaming memory concept;
// SSD-offload epic #2722). A GGUF MoE checkpoint stores each layer's routed experts as ONE
// fused [E, out, in] tensor whose outermost dimension varies slowest, so expert e's weights
// are one contiguous stride-byte segment at offset + e*stride. Today's loader path
// (internal/ggufload's splitGLMMoeDsaExpertsRawQuant, a separate lane) reads the WHOLE
// E-expert blob resident before splitting, so a top-k route pays E*stride bytes of IO for
// the k*stride it uses. ggufExpertSource inverts that: it indexes {offset, [E,out,in],
// quant-block geometry} per fused tensor at construction (no payload IO) and reads exactly
// ONE expert's sub-range [offset + e*stride, offset + (e+1)*stride) on demand through a
// caller-supplied io.ReaderAt.
//
// The ReaderAt parameter is the composition seam: it accepts an *os.File (the plain pread
// path), the mmap/ReadAt seam safetensors.go established, or an
// internal/stripeload.StripedReaderAt that fans the one expert read across byte-identical
// mirrors by bandwidth — the hardware-free core of the 3-tier GPU / pinned-CPU / SSD
// expert-cache ladder (#2726) this leaf feeds. It is the GGUF analogue of
// safetensorsFile.readExpertSlice (#4359): that method slices a safetensors file it owns;
// this source owns no file and derives each stride from GGUF quant-block geometry instead
// of trusting a caller-computed byte count.
//
// GENERATION FRAME (gen/second-next, #3174 — architectural option, never default exposure):
// nothing in the default load/decode path constructs a ggufExpertSource yet, so no default
// behaviour changes. PROMOTION evidence: gguf_expert_source_test.go proves streamed-expert
// bytes are bit-identical to the fully-resident slab slice on both a direct and a
// striped-mirror ReaderAt, and that a read moves exactly stride bytes, never E*stride;
// promote by wiring it under gguf_glm_tensors.go's expert split (ggufload lane) behind a
// measured bytes/token witness, with pagedRing (#2726) as the residency tier above it.
// DEMOTION/retirement: retire if fully-resident loading remains the only shipped path or
// the ggufload seam grows its own range reader — a second unwired slicer is slop.
// INVALIDATING assumption: experts are contiguous, equal-stride, unpadded segments of the
// fused tensor; a GGUF layout that interleaves or pads per-expert segments breaks the
// stride math and must fail at construction, never read garbage.

var (
	// ErrGGUFExpertMetadata reports a malformed fused-expert tensor description.
	// Construction fails before any payload IO.
	ErrGGUFExpertMetadata = errors.New("invalid gguf fused-expert tensor description")
	// ErrGGUFExpertNotFound reports a read of a tensor name the source does not index.
	ErrGGUFExpertNotFound = errors.New("gguf fused-expert tensor not found")
	// ErrGGUFExpertOutOfRange reports an expert index outside [0, Experts).
	ErrGGUFExpertOutOfRange = errors.New("gguf expert index out of range")
)

// ggufExpertTensorDesc describes one fused GGUF routed-expert tensor as Experts contiguous
// per-expert segments starting at Offset. Rows/Cols are one expert's 2-D weight shape
// ([out, in] — Cols is the reduction dimension), and BlockWeights/BlockBytes are the GGUF
// quant-block geometry (weights per block, bytes per block; 1/4 for f32). The per-expert
// stride is derived, never caller-supplied: (Rows*Cols/BlockWeights)*BlockBytes.
type ggufExpertTensorDesc struct {
	Name         string
	Offset       int64
	Experts      int
	Rows         int
	Cols         int
	BlockWeights int
	BlockBytes   int
}

// expertStride returns the derived per-expert byte stride, validating the description's
// geometry. Like splitGLMMoeDsaExpertsRawQuant, it gates on the REDUCTION dim (Cols): a
// raw-quant expert row must be a whole number of quant blocks, since a GEMV dequantizes
// blocks ALONG each row — Rows*Cols alignment alone can pass while Cols alignment fails
// and would then mis-block every row.
func (d ggufExpertTensorDesc) expertStride() (int64, error) {
	if d.Experts <= 0 || d.Rows <= 0 || d.Cols <= 0 || d.BlockWeights <= 0 || d.BlockBytes <= 0 {
		return 0, fmt.Errorf("%w: %s has non-positive geometry [e=%d out=%d in=%d blockWeights=%d blockBytes=%d]",
			ErrGGUFExpertMetadata, d.Name, d.Experts, d.Rows, d.Cols, d.BlockWeights, d.BlockBytes)
	}
	if d.Cols%d.BlockWeights != 0 {
		return 0, fmt.Errorf("%w: %s reduction dim %d is not a whole number of %d-weight quant blocks",
			ErrGGUFExpertMetadata, d.Name, d.Cols, d.BlockWeights)
	}
	if int64(d.Rows) > math.MaxInt64/int64(d.Cols) {
		return 0, fmt.Errorf("%w: %s expert weight count overflows", ErrGGUFExpertMetadata, d.Name)
	}
	blocks := int64(d.Rows) * int64(d.Cols) / int64(d.BlockWeights)
	if blocks > math.MaxInt64/int64(d.BlockBytes) {
		return 0, fmt.Errorf("%w: %s expert byte stride overflows", ErrGGUFExpertMetadata, d.Name)
	}
	return blocks * int64(d.BlockBytes), nil
}

// ggufExpertEntry is one indexed fused tensor: its description plus the stride derived once
// at construction, so every read reuses validated geometry.
type ggufExpertEntry struct {
	desc   ggufExpertTensorDesc
	stride int64
}

// ggufExpertSource lazily range-reads per-expert segments of fused GGUF expert tensors from
// an io.ReaderAt. Construction validates every description against the declared source size
// and performs no payload IO; readExpert faults exactly one expert's byte range. It is safe
// for concurrent use iff the underlying ReaderAt is (os.File, bytes.Reader, and
// stripeload.StripedReaderAt all are).
type ggufExpertSource struct {
	r       io.ReaderAt
	size    int64
	tensors map[string]ggufExpertEntry
}

// newGGUFExpertSource indexes the given fused-expert tensor descriptions over r, whose
// readable extent is size bytes. Every description's derived slab [Offset, Offset +
// Experts*stride) must lie inside [0, size); a description that does not is a construction
// error, never a deferred read fault.
func newGGUFExpertSource(r io.ReaderAt, size int64, descs []ggufExpertTensorDesc) (*ggufExpertSource, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: nil reader", ErrGGUFExpertMetadata)
	}
	if size < 0 {
		return nil, fmt.Errorf("%w: negative source size %d", ErrGGUFExpertMetadata, size)
	}
	s := &ggufExpertSource{r: r, size: size, tensors: make(map[string]ggufExpertEntry, len(descs))}
	for _, d := range descs {
		if d.Name == "" {
			return nil, fmt.Errorf("%w: empty tensor name", ErrGGUFExpertMetadata)
		}
		if _, dup := s.tensors[d.Name]; dup {
			return nil, fmt.Errorf("%w: duplicate tensor %s", ErrGGUFExpertMetadata, d.Name)
		}
		stride, err := d.expertStride()
		if err != nil {
			return nil, err
		}
		if d.Offset < 0 {
			return nil, fmt.Errorf("%w: %s has negative offset %d", ErrGGUFExpertMetadata, d.Name, d.Offset)
		}
		if stride > math.MaxInt64/int64(d.Experts) {
			return nil, fmt.Errorf("%w: %s slab size overflows", ErrGGUFExpertMetadata, d.Name)
		}
		slab := stride * int64(d.Experts)
		if d.Offset > size-slab {
			return nil, fmt.Errorf("%w: %s slab [%d,%d) outside source [0,%d)",
				ErrGGUFExpertMetadata, d.Name, d.Offset, d.Offset+slab, size)
		}
		s.tensors[d.Name] = ggufExpertEntry{desc: d, stride: stride}
	}
	return s, nil
}

// readExpert reads exactly expert e's stride-byte segment of the named fused tensor —
// [Offset + e*stride, Offset + (e+1)*stride) — into a freshly-owned buffer. It never reads
// another expert's bytes: a top-k route over this source moves k*stride bytes, not the
// E*stride whole-slab read the resident split path pays.
func (s *ggufExpertSource) readExpert(name string, e int) ([]byte, error) {
	entry, ok := s.tensors[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrGGUFExpertNotFound, name)
	}
	if e < 0 || e >= entry.desc.Experts {
		return nil, fmt.Errorf("%w: expert %d of %s outside [0,%d)", ErrGGUFExpertOutOfRange, e, name, entry.desc.Experts)
	}
	maxInt := int64(^uint(0) >> 1)
	if entry.stride > maxInt {
		return nil, fmt.Errorf("%w: %s expert stride %d exceeds addressable buffer size", ErrGGUFExpertMetadata, name, entry.stride)
	}
	buf := make([]byte, entry.stride)
	if _, err := s.r.ReadAt(buf, entry.desc.Offset+int64(e)*entry.stride); err != nil {
		return nil, fmt.Errorf("model: gguf expert read %s[%d]: %w", name, e, err)
	}
	return buf, nil
}

// expertStride returns the derived per-expert byte stride of the named fused tensor.
func (s *ggufExpertSource) expertStride(name string) (int64, bool) {
	entry, ok := s.tensors[name]
	if !ok {
		return 0, false
	}
	return entry.stride, true
}

// len reports how many fused expert tensors the source indexes.
func (s *ggufExpertSource) len() int { return len(s.tensors) }
