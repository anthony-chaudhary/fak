//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdlib.h>
#include "packed_gguf.h"
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

//go:embed packed_gguf_codebooks.metal
var packedCodebooks string

//go:embed packed_gguf_dequant.metal
var packedDequant string

//go:embed packed_gguf_gemv.metal
var packedGEMV string

//go:embed packed_gguf_panel.metal
var packedPanel string

//go:embed packed_gguf_moe.metal
var packedMoE string

// PackedGGUFStore owns Metal weight views and reusable activation buffers.
// Its caller owns the mapping and must close this store before unmapping.
type PackedGGUFStore struct {
	mu         sync.Mutex
	mapped     []byte
	ctx        unsafe.Pointer
	maxBuffer  int
	buffers    []unsafe.Pointer
	batch      []C.mg_packed_job
	panels     []C.mg_packed_panel
	panelIndex map[packedProjection]int
}

type packedView struct {
	handle unsafe.Pointer
	offset int
}

type PackedGGUFMatrix struct {
	store                                       *PackedGGUFStore
	typ                                         uint32
	rows, cols, experts, stride, expertsPerView int
	views                                       []packedView
}

// NewPackedGGUFStore takes a whole-file mmap starting at a VM page boundary.
func NewPackedGGUFStore(mapped []byte) (*PackedGGUFStore, error) {
	if !Available() {
		return nil, ErrMetalUnavailable
	}
	if len(mapped) == 0 || uintptr(unsafe.Pointer(&mapped[0]))%uintptr(os.Getpagesize()) != 0 {
		return nil, ErrUnalignedMemory
	}
	src := C.CString(packedCodebooks + packedDequant + packedGEMV + packedPanel + packedMoE)
	defer C.free(unsafe.Pointer(src))
	ctx := C.mg_packed_create(src)
	if ctx == nil {
		return nil, errors.New("metalgemm: packed GGUF pipelines unavailable")
	}
	return &PackedGGUFStore{mapped: mapped, ctx: ctx, maxBuffer: int(C.mg_packed_max_buffer())}, nil
}

func packedBlock(typ uint32) (int, int, error) {
	switch typ {
	case 40: // GGML NVFP4 stores four UE4M3 scales with 64 E2M1 values.
		return 64, 36, nil
	case 0, 1, 8, 11, 12, 13, 14, 17, 18, 20, 21, 22, 30:
		d, err := quantmeta.FromGGUF(typ)
		return d.BlockSize, d.BlockBytes, err
	default:
		return 0, 0, fmt.Errorf("%w: %d", ErrPackedGGUFType, typ)
	}
}

// Matrix wraps the contiguous GGUF payload at the absolute byte offset.
// Expert stacks wider than a Metal buffer are split at expert boundaries.
func (s *PackedGGUFStore) Matrix(offset int, ggmlType uint32, rows, cols, experts int) (*PackedGGUFMatrix, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		return nil, ErrPackedGGUFClosed
	}
	block, bytesPerBlock, err := packedBlock(ggmlType)
	if err != nil {
		return nil, err
	}
	if offset < 0 || offset >= len(s.mapped) || offset%4 != 0 || rows <= 0 ||
		rows > math.MaxInt32 || cols <= 0 || cols > math.MaxInt32 || experts <= 0 || cols%block != 0 {
		return nil, ErrPackedGGUFShape
	}
	rowBytes := cols / block * bytesPerBlock
	remaining := len(s.mapped) - offset
	if rows > remaining/rowBytes || experts > remaining/(rows*rowBytes) {
		return nil, ErrPackedGGUFShape
	}
	stride := rows * rowBytes
	page := os.Getpagesize()
	m := &PackedGGUFMatrix{store: s, typ: ggmlType, rows: rows, cols: cols, experts: experts, stride: stride}
	// One extra page covers either unaligned edge of an adjacent view.
	m.expertsPerView = min(experts, (s.maxBuffer-page)/stride)
	if m.expertsPerView < 1 {
		return nil, fmt.Errorf("%w: expert needs %d bytes", ErrPackedGGUFShape, stride)
	}
	for first := 0; first < experts; first += m.expertsPerView {
		at := offset + first*stride
		n := min(experts-first, m.expertsPerView)
		base := at &^ (page - 1)
		length := (at - base + n*stride + page - 1) &^ (page - 1)
		// mmap reserves through the last VM page, even when EOF is not page aligned.
		handle := C.mg_zerocopy_create_shared_buffer(unsafe.Pointer(&s.mapped[base]), C.size_t(length))
		if handle == nil {
			for _, v := range m.views {
				C.mg_zerocopy_buffer_free(v.handle)
			}
			return nil, fmt.Errorf("metalgemm: rejected %d-byte GGUF view", length)
		}
		m.views = append(m.views, packedView{handle: handle, offset: at - base})
	}
	for _, v := range m.views {
		s.buffers = append(s.buffers, v.handle)
	}
	return m, nil
}

func (m *PackedGGUFMatrix) job(expert, inputOffset, outputOffset int) C.mg_packed_job {
	v := m.views[expert/m.expertsPerView]
	off := v.offset + (expert%m.expertsPerView)*m.stride
	return C.mg_packed_job{weight: C.uintptr_t(uintptr(v.handle)), byte_offset: C.size_t(off),
		_type: C.uint32_t(m.typ), rows: C.uint32_t(m.rows), cols: C.uint32_t(m.cols),
		input_offset: C.size_t(inputOffset), output_offset: C.size_t(outputOffset)}
}

// Run submits all jobs in one command buffer and returns after readback.
func (s *PackedGGUFStore) Run(jobs []PackedGGUFJob) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		return ErrPackedGGUFClosed
	}
	if len(jobs) == 0 {
		return nil
	}
	if cap(s.batch) < len(jobs) {
		s.batch = make([]C.mg_packed_job, len(jobs))
	}
	batch := s.batch[:len(jobs)]
	nx, ny := 0, 0
	for i, j := range jobs {
		m := j.Matrix
		if m == nil || m.store != s || j.Expert < 0 || j.Expert >= m.experts ||
			len(j.X) != m.cols || len(j.Dst) != m.rows {
			return fmt.Errorf("%w: job %d", ErrPackedGGUFShape, i)
		}
		batch[i] = m.job(j.Expert, nx, ny)
		// Every packed activation can be loaded as a float4.
		nx += (m.cols + 3) &^ 3
		ny += (m.rows + 3) &^ 3
		if nx > s.maxBuffer/4 || ny > s.maxBuffer/4 {
			return ErrPackedGGUFShape
		}
	}
	var xp, yp *C.float
	if C.mg_packed_prepare(s.ctx, C.size_t(nx), C.size_t(ny), &xp, &yp) == 0 {
		return errors.New("metalgemm: packed GGUF activation allocation failed")
	}
	x := unsafe.Slice((*float32)(unsafe.Pointer(xp)), nx)
	for i, j := range jobs {
		copy(x[int(batch[i].input_offset):], j.X)
	}
	var grouped *C.mg_packed_panel
	count := len(batch)
	if nx <= math.MaxUint32 && ny <= math.MaxUint32 && s.groupProjections(jobs) {
		grouped, count = &s.panels[0], len(s.panels)
	}
	if C.mg_packed_run(s.ctx, &batch[0], C.size_t(count), grouped) == 0 {
		return errors.New("metalgemm: packed GGUF command failed")
	}
	y := unsafe.Slice((*float32)(unsafe.Pointer(yp)), ny)
	for i, j := range jobs {
		copy(j.Dst, y[int(batch[i].output_offset):])
	}
	return nil
}

func (s *PackedGGUFStore) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, h := range s.buffers {
		C.mg_zerocopy_buffer_free(h)
	}
	s.buffers = nil
	C.mg_packed_close(s.ctx)
	s.ctx = nil
	s.mapped = nil
}

// ExpertMLP keeps the gate/up rows, SITU activation and down projection on
// the GPU until the selected experts have been reduced to a single row.
func (s *PackedGGUFStore) ExpertMLP(job PackedExpertMLP, x, dst []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		return ErrPackedGGUFClosed
	}
	gu, down := job.GateUp, job.Down
	k := len(job.Selected)
	if gu == nil || down == nil || gu.store != s || down.store != s ||
		gu.experts != down.experts || k == 0 || k > gu.experts || len(job.Weights) != k ||
		(len(job.GateUpScale) != 0 && len(job.GateUpScale) != gu.experts) ||
		(len(job.DownScale) != 0 && len(job.DownScale) != down.experts) {
		return ErrPackedGGUFShape
	}
	width, hidden := down.cols, down.rows
	if gu.rows != 2*width || gu.cols != hidden || len(x) != hidden || len(dst) != hidden {
		return ErrPackedGGUFShape
	}
	limit := min(s.maxBuffer/4, math.MaxInt32)
	stride := (width + 3) &^ 3
	if k > (limit-hidden-3)/(stride+3) || k > (limit-hidden)/(gu.rows+hidden) {
		return ErrPackedGGUFShape
	}
	activationBase := (hidden + 3*k + 3) &^ 3
	downBase := k * gu.rows
	sumBase := downBase + k*hidden
	if cap(s.batch) < 2*k {
		s.batch = make([]C.mg_packed_job, 2*k)
	}
	batch := s.batch[:2*k]
	for slot, expert := range job.Selected {
		if expert < 0 || expert >= gu.experts {
			return ErrPackedGGUFShape
		}
		batch[slot] = gu.job(expert, 0, slot*gu.rows)
		batch[k+slot] = down.job(expert, activationBase+slot*stride, downBase+slot*hidden)
	}
	var xp, yp *C.float
	if C.mg_packed_prepare(s.ctx, C.size_t(activationBase+k*stride), C.size_t(sumBase+hidden), &xp, &yp) == 0 {
		return errors.New("metalgemm: packed expert activation allocation failed")
	}
	input := unsafe.Slice((*float32)(unsafe.Pointer(xp)), hidden+3*k)
	copy(input, x)
	params := input[hidden:]
	for slot, expert := range job.Selected {
		upScale, downScale := float32(1), float32(1)
		if len(job.GateUpScale) != 0 {
			upScale = job.GateUpScale[expert]
		}
		if len(job.DownScale) != 0 {
			downScale = job.DownScale[expert]
		}
		copy(params[3*slot:3*slot+3], []float32{upScale, downScale, job.Weights[slot]})
	}
	shape := C.mg_packed_moe_shape{slots: C.uint32_t(k), hidden: C.uint32_t(hidden), width: C.uint32_t(width),
		activation_stride: C.uint32_t(stride), beta: C.float(job.Beta), linear_beta: C.float(job.LinearBeta)}
	if C.mg_packed_moe(s.ctx, &batch[0], shape) == 0 {
		return errors.New("metalgemm: packed expert command failed")
	}
	result := unsafe.Slice((*float32)(unsafe.Pointer(yp)), sumBase+hidden)
	copy(dst, result[sumBase:])
	return nil
}

type packedProjection struct {
	matrix *PackedGGUFMatrix
	expert int
}

// Group equal weight views into register tiles. Decode batches contain a
// handful of distinct weights; skip the map when the batch has no repeats.
func (s *PackedGGUFStore) groupProjections(jobs []PackedGGUFJob) bool {
	if len(jobs) <= 8 {
		repeated := false
		for i, j := range jobs {
			for _, prev := range jobs[:i] {
				repeated = repeated || (j.Matrix == prev.Matrix && j.Expert == prev.Expert)
			}
		}
		if !repeated {
			return false
		}
	}
	if s.panelIndex == nil {
		s.panelIndex = make(map[packedProjection]int)
	}
	clear(s.panelIndex)
	s.panels = s.panels[:0]
	for i, j := range jobs {
		key := packedProjection{j.Matrix, j.Expert}
		g, ok := s.panelIndex[key]
		if !ok || s.panels[g].count == 4 {
			g = len(s.panels)
			s.panelIndex[key] = g
			s.panels = append(s.panels, C.mg_packed_panel{})
		}
		p := &s.panels[g]
		p.job[p.count] = C.uint32_t(i)
		p.count++
	}
	return len(s.panels) < len(jobs)
}
