//go:build darwin && arm64 && cgo

package ggufload

import (
	"fmt"
	"io"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2MetalDeviceSet owns the Metal views of one mapped GGUF. The views
// alias the mapping, so Close drains the device before it unmaps.
type inkling2MetalDeviceSet struct {
	be      compute.Backend
	store   *metalgemm.PackedGGUFStore
	mapping io.Closer
	data    []byte
}

func (s *inkling2MetalDeviceSet) Close() error {
	if s.mapping == nil {
		return nil
	}
	// A forward that failed part way leaves launches encoded against the views;
	// a read is the fence that runs them while the pages are still mapped.
	if dev, ok := s.be.(compute.Inkling2Device); ok {
		if t, err := dev.Ink2Alloc(1); err == nil {
			_, _ = dev.Ink2Read(compute.Ink2At(t, 0), 1)
			s.be.Free(t)
		}
	}
	s.store.Close()
	err := s.mapping.Close()
	s.mapping = nil
	return err
}

// packed views raw, a tensor payload inside the mapping, as a packed matrix.
// ok is false for a payload the packed kernels cannot read in place.
func (s *inkling2MetalDeviceSet) packed(typ TensorType, raw []byte, rows, cols, experts int) (*metalgemm.PackedGGUFMatrix, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	offset := int(uintptr(unsafe.Pointer(unsafe.SliceData(raw))) - uintptr(unsafe.Pointer(unsafe.SliceData(s.data))))
	if offset < 0 || offset+len(raw) > len(s.data) {
		return nil, false
	}
	m, err := s.store.Matrix(offset, uint32(typ), rows, cols, experts)
	return m, err == nil
}

// matrix rebinds a host quant matrix as an in-place device weight. A matrix
// the kernels cannot read stays host-side: the resident forward widens a host
// matrix to F32 and uploads it.
func (s *inkling2MetalDeviceSet) matrix(m model.Inkling2LoopMatrix) (model.Inkling2LoopMatrix, error) {
	q, ok := m.(*inkling2QuantMatrix)
	if !ok {
		return m, nil
	}
	pm, ok := s.packed(q.typ, q.raw, q.rows, q.cols, 1)
	if !ok {
		return m, nil
	}
	w, err := compute.MetalInkling2Weight(pm, 0)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", q.name, err)
	}
	return &inkling2DeviceMatrix{be: s.be, w: w, rows: q.rows, cols: q.cols, padK: q.cols}, nil
}

func (s *inkling2MetalDeviceSet) experts(e model.Inkling2LoopExperts) (model.Inkling2LoopExperts, error) {
	q, ok := e.(*inkling2QuantExperts)
	if !ok {
		return e, nil
	}
	pm, ok := s.packed(q.typ, q.raw, q.rows, q.cols, q.experts)
	if !ok {
		return nil, fmt.Errorf("inkling2_loop: %s expert stack [%d %d %d] has no packed Metal kernel", q.typ, q.experts, q.rows, q.cols)
	}
	stack, err := compute.MetalInkling2Weight(pm, -1)
	if err != nil {
		return nil, err
	}
	out := &inkling2DeviceExperts{be: s.be, stack: stack, w: make([]compute.Tensor, q.experts), rows: q.rows, cols: q.cols, padK: q.cols}
	for x := range out.w {
		if out.w[x], err = compute.MetalInkling2Weight(pm, x); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// OpenInkling2LoopMetalDevice maps a single-file inkling2_loop GGUF and binds
// its matrices and expert stacks as weights of the Metal inkling2 device ops,
// read in place from the mapping. The returned backend runs the resident
// forward through the ordinary loop session and Inkling2LoopGenerate; the model
// has no host-driven fallback. The token embedding, the vectors and the loop
// controllers stay host-side for the resident forward to upload.
func OpenInkling2LoopMetalDevice(path string) (*Inkling2LoopModel, compute.Backend, error) {
	be, ok := compute.MetalInkling2Backend()
	if !ok {
		return nil, nil, metalgemm.ErrMetalUnavailable
	}
	data, mapping, ok, err := model.MmapOpen(path)
	if err != nil {
		return nil, nil, err
	}
	if !ok {
		return nil, nil, fmt.Errorf("inkling2_loop: cannot map %q", path)
	}
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		mapping.Close()
		return nil, nil, err
	}
	s := &inkling2MetalDeviceSet{be: be, store: store, mapping: mapping, data: data}
	m, err := bindInkling2Shards([][]byte{data}, nil)
	if err == nil {
		m.closer = s
		err = s.bind(m)
	}
	if err != nil {
		s.Close()
		return nil, nil, err
	}
	return m, be, nil
}

func (s *inkling2MetalDeviceSet) bind(m *Inkling2LoopModel) (err error) {
	w := m.Weights
	mat := func(dst *model.Inkling2LoopMatrix) {
		if err == nil && *dst != nil {
			*dst, err = s.matrix(*dst)
		}
	}
	mat(&w.Output)
	for i := range w.Layers {
		lw := &w.Layers[i]
		for _, p := range []*model.Inkling2LoopMatrix{&lw.Wq, &lw.Wk, &lw.Wv, &lw.Wo, &lw.Gate, &lw.Rel,
			&lw.SinkProj, &lw.Router, &lw.KConvProj, &lw.VConvProj} {
			mat(p)
		}
		for _, p := range []*model.Inkling2LoopExperts{&lw.GateUp, &lw.Down} {
			if err == nil && *p != nil {
				*p, err = s.experts(*p)
			}
		}
	}
	if err != nil {
		return err
	}
	if !inkling2ResidentEligible(m, s.be) {
		return fmt.Errorf("inkling2_loop: %s is not eligible for the resident Metal forward", m.Arch)
	}
	return nil
}
