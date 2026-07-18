package model

import (
	"bytes"
	"errors"
	"io"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stripeload"
)

// gguf_expert_source_test.go — the promotion witness for ggufExpertSource (#3174, epic
// #2722): streamed per-expert bytes are bit-identical to the fully-resident slab slice, on
// both a direct ReaderAt and a stripeload striped-mirror ReaderAt, and a read moves exactly
// one expert's stride — never the whole E-expert slab.

// ggufExpertFixture is a synthetic single-file layout of fused expert slabs with junk gaps
// between them, so an off-by-stride read cannot accidentally still match.
type ggufExpertFixture struct {
	file  []byte
	descs []ggufExpertTensorDesc
}

// fixtureStride recomputes a description's per-expert byte stride with the same arithmetic
// splitGLMMoeDsaExpertsRawQuant uses ((rows*cols/blockWeights)*blockBytes), independently of
// the production expertStride, so the two implementations witness each other.
func fixtureStride(d ggufExpertTensorDesc) int {
	return (d.Rows * d.Cols / d.BlockWeights) * d.BlockBytes
}

// newGGUFExpertFixture lays out three fused expert tensors of distinct quant geometries
// (Q4_K-, f32-, and Q8_0-shaped) over deterministic non-repeating bytes.
func newGGUFExpertFixture(t *testing.T) ggufExpertFixture {
	t.Helper()
	descs := []ggufExpertTensorDesc{
		// Q4_K geometry: 256-weight, 144-byte super-blocks; stride 3*512/256*144 = 864.
		{Name: "blk.0.ffn_gate_exps.weight", Experts: 4, Rows: 3, Cols: 512, BlockWeights: 256, BlockBytes: 144},
		// f32 geometry: stride 2*8*4 = 64.
		{Name: "blk.0.ffn_up_exps.weight", Experts: 3, Rows: 2, Cols: 8, BlockWeights: 1, BlockBytes: 4},
		// Q8_0 geometry: 32-weight, 34-byte blocks; stride 4*64/32*34 = 272.
		{Name: "blk.1.ffn_down_exps.weight", Experts: 5, Rows: 4, Cols: 64, BlockWeights: 32, BlockBytes: 34},
	}
	off := int64(100) // junk prefix before the first slab
	for i := range descs {
		descs[i].Offset = off
		off += int64(descs[i].Experts*fixtureStride(descs[i])) + 32 // junk gap between slabs
	}
	file := make([]byte, off)
	// Deterministic LCG fill: every byte position gets a distinct pseudo-random value, so
	// bit-equality can only pass by reading exactly the right range.
	state := uint32(0x3174)
	for i := range file {
		state = state*1664525 + 1013904223
		file[i] = byte(state >> 24)
	}
	return ggufExpertFixture{file: file, descs: descs}
}

// residentExpert slices expert e out of the fully-resident fixture bytes — the
// all-resident reference the streamed read must be bit-identical to.
func (f ggufExpertFixture) residentExpert(d ggufExpertTensorDesc, e int) []byte {
	stride := fixtureStride(d)
	start := int(d.Offset) + e*stride
	return f.file[start : start+stride]
}

func TestGGUFExpertSourceStreamedBitEqualResident(t *testing.T) {
	fix := newGGUFExpertFixture(t)
	src, err := newGGUFExpertSource(bytes.NewReader(fix.file), int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	if got, want := src.len(), len(fix.descs); got != want {
		t.Fatalf("indexed %d tensors, want %d", got, want)
	}
	for _, d := range fix.descs {
		stride, ok := src.expertStride(d.Name)
		if !ok {
			t.Fatalf("expertStride(%s): not indexed", d.Name)
		}
		if want := int64(fixtureStride(d)); stride != want {
			t.Fatalf("%s stride = %d, want %d", d.Name, stride, want)
		}
		for e := 0; e < d.Experts; e++ {
			streamed, err := src.readExpert(d.Name, e)
			if err != nil {
				t.Fatalf("readExpert(%s, %d): %v", d.Name, e, err)
			}
			if !bytes.Equal(streamed, fix.residentExpert(d, e)) {
				t.Fatalf("readExpert(%s, %d): streamed bytes differ from resident slab slice", d.Name, e)
			}
		}
	}
}

func TestGGUFExpertSourceStripedMirrorsBitEqualResident(t *testing.T) {
	fix := newGGUFExpertFixture(t)
	// Three byte-identical mirrors with unequal bandwidth weights, and a split floor far
	// below every expert stride, so each expert read genuinely fans across the mirror set.
	striped, err := stripeload.New([]stripeload.Source{
		{R: bytes.NewReader(fix.file), BWWeight: 3},
		{R: bytes.NewReader(fix.file), BWWeight: 1},
		{R: bytes.NewReader(fix.file), BWWeight: 2},
	}, stripeload.WithMinChunk(16))
	if err != nil {
		t.Fatalf("stripeload.New: %v", err)
	}
	src, err := newGGUFExpertSource(striped, int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	for _, d := range fix.descs {
		for e := 0; e < d.Experts; e++ {
			streamed, err := src.readExpert(d.Name, e)
			if err != nil {
				t.Fatalf("readExpert(%s, %d) over striped mirrors: %v", d.Name, e, err)
			}
			if !bytes.Equal(streamed, fix.residentExpert(d, e)) {
				t.Fatalf("readExpert(%s, %d) over striped mirrors: bytes differ from resident slab slice", d.Name, e)
			}
		}
	}
}

// expertRecordedRead is one observed ReadAt request: absolute offset and requested length.
type expertRecordedRead struct {
	off int64
	n   int
}

// expertRangeRecorder records every ReadAt request before delegating, so a test can assert
// exactly which byte ranges a read faulted (recordingReaderAt counts reads but not offsets).
type expertRangeRecorder struct {
	r     io.ReaderAt
	calls []expertRecordedRead
}

func (r *expertRangeRecorder) ReadAt(p []byte, off int64) (int, error) {
	r.calls = append(r.calls, expertRecordedRead{off: off, n: len(p)})
	return r.r.ReadAt(p, off)
}

func TestGGUFExpertSourceReadsOnlyTheRequestedExpert(t *testing.T) {
	fix := newGGUFExpertFixture(t)
	rec := &expertRangeRecorder{r: bytes.NewReader(fix.file)}
	src, err := newGGUFExpertSource(rec, int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	if len(rec.calls) != 0 {
		t.Fatalf("construction performed %d payload reads, want 0", len(rec.calls))
	}
	d := fix.descs[0]
	const e = 2
	if _, err := src.readExpert(d.Name, e); err != nil {
		t.Fatalf("readExpert(%s, %d): %v", d.Name, e, err)
	}
	stride := fixtureStride(d)
	if len(rec.calls) != 1 {
		t.Fatalf("readExpert issued %d reads, want exactly 1", len(rec.calls))
	}
	call := rec.calls[0]
	if wantOff := d.Offset + int64(e*stride); call.off != wantOff {
		t.Fatalf("readExpert read at offset %d, want %d", call.off, wantOff)
	}
	if call.n != stride {
		t.Fatalf("readExpert moved %d bytes, want exactly one expert stride %d (never the %d-byte slab)",
			call.n, stride, d.Experts*stride)
	}
}

func TestGGUFExpertSourceRejectsMalformedDescriptions(t *testing.T) {
	fix := newGGUFExpertFixture(t)
	valid := fix.descs[0]
	mutate := func(f func(*ggufExpertTensorDesc)) []ggufExpertTensorDesc {
		d := valid
		f(&d)
		return []ggufExpertTensorDesc{d}
	}
	cases := []struct {
		name  string
		r     io.ReaderAt
		size  int64
		descs []ggufExpertTensorDesc
	}{
		{name: "nil reader", r: nil, size: int64(len(fix.file)), descs: fix.descs},
		{name: "negative size", r: bytes.NewReader(fix.file), size: -1, descs: fix.descs},
		{name: "empty name", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: mutate(func(d *ggufExpertTensorDesc) { d.Name = "" })},
		{name: "duplicate name", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: []ggufExpertTensorDesc{valid, valid}},
		{name: "zero experts", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: mutate(func(d *ggufExpertTensorDesc) { d.Experts = 0 })},
		{name: "negative rows", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: mutate(func(d *ggufExpertTensorDesc) { d.Rows = -3 })},
		{name: "zero block bytes", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: mutate(func(d *ggufExpertTensorDesc) { d.BlockBytes = 0 })},
		{name: "reduction dim not block aligned", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: mutate(func(d *ggufExpertTensorDesc) { d.Cols = 100 })},
		{name: "negative offset", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: mutate(func(d *ggufExpertTensorDesc) { d.Offset = -1 })},
		{name: "slab past end of source", r: bytes.NewReader(fix.file), size: int64(len(fix.file)),
			descs: mutate(func(d *ggufExpertTensorDesc) { d.Offset = int64(len(fix.file)) - 10 })},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := newGGUFExpertSource(tc.r, tc.size, tc.descs); !errors.Is(err, ErrGGUFExpertMetadata) {
				t.Fatalf("newGGUFExpertSource = %v, want ErrGGUFExpertMetadata", err)
			}
		})
	}

	src, err := newGGUFExpertSource(bytes.NewReader(fix.file), int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	if _, err := src.readExpert("blk.9.ffn_gate_exps.weight", 0); !errors.Is(err, ErrGGUFExpertNotFound) {
		t.Fatalf("readExpert(unknown) = %v, want ErrGGUFExpertNotFound", err)
	}
	for _, e := range []int{-1, valid.Experts} {
		if _, err := src.readExpert(valid.Name, e); !errors.Is(err, ErrGGUFExpertOutOfRange) {
			t.Fatalf("readExpert(%s, %d) = %v, want ErrGGUFExpertOutOfRange", valid.Name, e, err)
		}
	}
	if _, ok := src.expertStride("blk.9.ffn_gate_exps.weight"); ok {
		t.Fatal("expertStride(unknown) reported ok")
	}
}
