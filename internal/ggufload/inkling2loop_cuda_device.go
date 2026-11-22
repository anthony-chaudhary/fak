package ggufload

import (
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2HostDenseLimit keeps small dense (F32/F16/BF16) matrices on the
// host: router and loop-controller GEMVs are cheaper there than a device
// round trip per call.
const inkling2HostDenseLimit = 4 << 20

// inkling2DeviceMatrix is a weight resident on be; every MatVec uploads x,
// multiplies on the device and reads the result back. padCols > cols means the
// weight rows were zero-padded to a kernel block multiple.
type inkling2DeviceMatrix struct {
	be               compute.Backend
	w                compute.Tensor
	rows, cols, padK int
}

func (m *inkling2DeviceMatrix) Rows() int { return m.rows }
func (m *inkling2DeviceMatrix) Cols() int { return m.cols }
func (m *inkling2DeviceMatrix) MatVec(dst, x []float32) {
	inkling2DeviceMatVec(m.be, m.w, m.cols, m.padK, dst, x)
}

// inkling2DeviceExperts multiplies through per-expert row views of one stacked
// device tensor owned by the inkling2DeviceSet.
type inkling2DeviceExperts struct {
	be               compute.Backend
	w                []compute.Tensor
	rows, cols, padK int
}

func (e *inkling2DeviceExperts) Experts() int { return len(e.w) }
func (e *inkling2DeviceExperts) Rows() int    { return e.rows }
func (e *inkling2DeviceExperts) Cols() int    { return e.cols }
func (e *inkling2DeviceExperts) ExpertMatVec(expert int, dst, x []float32) {
	inkling2DeviceMatVec(e.be, e.w[expert], e.cols, e.padK, dst, x)
}

type inkling2ClassedUploader interface {
	UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor
}

func inkling2DeviceMatVec(be compute.Backend, w compute.Tensor, cols, padK int, dst, x []float32) {
	in := x[:cols]
	if padK > cols {
		in = make([]float32, padK)
		copy(in, x[:cols])
	}
	host := compute.NewF32(be, []int{len(in)}, in)
	var xt compute.Tensor
	if cu, ok := be.(inkling2ClassedUploader); ok {
		xt = cu.UploadClass(host, compute.F32, compute.MemoryActivation, "inkling2-activation")
	} else {
		xt = be.Upload(host, compute.F32)
	}
	y := be.MatMul(w, xt)
	copy(dst, be.Read(y))
	be.Free(y)
	be.Free(xt)
}

// inkling2DeviceWeight uploads one [rows, cols] GGUF matrix to be in the
// narrowest form the backend multiplies natively and returns the K width the
// activation must be padded to.
func inkling2DeviceWeight(be compute.Backend, typ TensorType, raw []byte, rows, cols int) (w compute.Tensor, padK int, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("inkling2_loop: device upload of %s [%d %d]: %v", typ, rows, cols, r)
		}
	}()
	shape := []int{rows, cols}
	switch typ {
	case TensorNVFP4:
		packed, scales, err := repackInkling2NVFP4(raw, rows, cols)
		if err != nil {
			return compute.Tensor{}, 0, err
		}
		host, err := compute.NewNVFP4(be, shape, packed, scales, 1)
		if err != nil {
			return compute.Tensor{}, 0, err
		}
		return be.Upload(host, compute.NVFP4), cols, nil
	case TensorQ8_0:
		const blk, blkBytes = 32, 34
		nblk := rows * cols / blk
		if cols%blk != 0 || len(raw) != nblk*blkBytes {
			return compute.Tensor{}, 0, fmt.Errorf("inkling2_loop: Q8_0 [%d %d] payload of %d bytes", rows, cols, len(raw))
		}
		codes := make([]int8, rows*cols)
		scale := make([]float32, nblk)
		for b := 0; b < nblk; b++ {
			src := raw[b*blkBytes : (b+1)*blkBytes]
			scale[b] = f16At(src, 0)
			for i, v := range src[2:] {
				codes[b*blk+i] = int8(v)
			}
		}
		return be.Upload(compute.NewQ8(be, shape, codes, scale, blk), compute.Q8_0), cols, nil
	case TensorQ3_K:
		return be.Upload(compute.NewQ3K(be, shape, raw), compute.Q3_K), cols, nil
	case TensorQ4_K:
		return be.Upload(compute.NewQ4K(be, shape, raw), compute.Q4_K), cols, nil
	case TensorQ5_K:
		return be.Upload(compute.NewQ5K(be, shape, raw), compute.Q5_K), cols, nil
	case TensorQ6_K:
		return be.Upload(compute.NewQ6K(be, shape, raw), compute.Q6_K), cols, nil
	case TensorIQ2_XS, TensorIQ2_S, TensorIQ3_XXS, TensorIQ3_S, TensorIQ4_XS:
		dt := inkling2RawIQDtype[typ]
		return be.Upload(compute.NewRawIQ(be, dt, shape, raw), dt), cols, nil
	case TensorIQ4_NL:
		padded, padK := inkling2PadIQ4NL(raw, rows, cols)
		return be.Upload(compute.NewRawIQ(be, compute.IQ4_NL, []int{rows, padK}, padded), compute.IQ4_NL), padK, nil
	case TensorF32, TensorF16, TensorBF16:
		rb, err := inkling2RowBytes(typ, cols)
		if err != nil {
			return compute.Tensor{}, 0, err
		}
		f := make([]float32, rows*cols)
		for r := 0; r < rows; r++ {
			if err := inkling2DequantRow(typ, raw[r*rb:(r+1)*rb], f[r*cols:(r+1)*cols]); err != nil {
				return compute.Tensor{}, 0, err
			}
		}
		return be.Upload(compute.NewF32(be, shape, f), compute.F32), cols, nil
	}
	return compute.Tensor{}, 0, &UnsupportedInkling2LoopTensorTypeError{Type: typ}
}

var inkling2RawIQDtype = map[TensorType]compute.Dtype{
	TensorIQ2_XS: compute.IQ2_XS, TensorIQ2_S: compute.IQ2_S, TensorIQ3_XXS: compute.IQ3_XXS,
	TensorIQ3_S: compute.IQ3_S, TensorIQ4_XS: compute.IQ4_XS,
}

// inkling2PadIQ4NL zero-pads IQ4_NL rows to whole 256-value super-blocks (the
// raw-IQ device geometry). A zero block has d = 0, so padded K contributes 0.
func inkling2PadIQ4NL(raw []byte, rows, cols int) ([]byte, int) {
	const blk, blkBytes = 32, 18
	padK := (cols + 255) / 256 * 256
	if padK == cols {
		return raw, cols
	}
	src, dst := cols/blk*blkBytes, padK/blk*blkBytes
	out := make([]byte, rows*dst)
	for r := 0; r < rows; r++ {
		copy(out[r*dst:r*dst+src], raw[r*src:(r+1)*src])
	}
	return out, padK
}

// inkling2DeviceSet owns every tensor uploaded for one model.
type inkling2DeviceSet struct {
	be      compute.Backend
	mu      sync.Mutex
	tensors []compute.Tensor
	next    io.Closer
}

func (s *inkling2DeviceSet) keep(t compute.Tensor) {
	s.mu.Lock()
	s.tensors = append(s.tensors, t)
	s.mu.Unlock()
}

func (s *inkling2DeviceSet) Close() error {
	s.mu.Lock()
	for _, t := range s.tensors {
		s.be.Free(t)
	}
	s.tensors = nil
	s.mu.Unlock()
	if s.next != nil {
		return s.next.Close()
	}
	return nil
}

func (s *inkling2DeviceSet) matrix(m model.Inkling2LoopMatrix) (model.Inkling2LoopMatrix, error) {
	q, ok := m.(*inkling2QuantMatrix)
	if !ok {
		return m, nil
	}
	switch q.typ {
	case TensorF32, TensorF16, TensorBF16:
		if q.rows*q.cols < inkling2HostDenseLimit {
			return m, nil
		}
	}
	w, padK, err := inkling2DeviceWeight(s.be, q.typ, q.raw, q.rows, q.cols)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", q.name, err)
	}
	s.keep(w)
	return &inkling2DeviceMatrix{be: s.be, w: w, rows: q.rows, cols: q.cols, padK: padK}, nil
}

// experts uploads the whole [experts*rows, cols] stack as one device tensor and
// hands out per-expert row views: one allocation per GGUF tensor instead of
// one per expert, whose allocator rounding overflowed a 16 GB card.
func (s *inkling2DeviceSet) experts(e model.Inkling2LoopExperts) (model.Inkling2LoopExperts, error) {
	q, ok := e.(*inkling2QuantExperts)
	if !ok {
		return e, nil
	}
	stack, padK, err := inkling2DeviceWeight(s.be, q.typ, q.raw, q.experts*q.rows, q.cols)
	if err != nil {
		return nil, err
	}
	s.keep(stack)
	out := &inkling2DeviceExperts{be: s.be, w: make([]compute.Tensor, q.experts), rows: q.rows, cols: q.cols, padK: padK}
	for x := range out.w {
		if out.w[x], err = compute.RowView(s.be, stack, x*q.rows, q.rows); err != nil {
			return nil, fmt.Errorf("inkling2_loop: expert %d of %s stack: %w", x, q.typ, err)
		}
	}
	return out, nil
}

// bindInkling2Device swaps every packed matrix and expert stack of a host-bound
// model for be-resident equivalents. The token embedding and small dense
// matrices stay host-side; Close frees the device tensors before the mapping.
func bindInkling2Device(m *Inkling2LoopModel, be compute.Backend) (err error) {
	s := &inkling2DeviceSet{be: be}
	defer func() {
		if err != nil {
			s.Close()
		}
	}()
	w := m.Weights
	mat := func(dst *model.Inkling2LoopMatrix) {
		if err == nil && *dst != nil {
			*dst, err = s.matrix(*dst)
		}
	}
	exps := func(dst *model.Inkling2LoopExperts) {
		if err == nil && *dst != nil {
			*dst, err = s.experts(*dst)
		}
	}
	mat(&w.Output)
	for i := range w.Layers {
		lw := &w.Layers[i]
		for _, p := range []*model.Inkling2LoopMatrix{&lw.Wq, &lw.Wk, &lw.Wv, &lw.Wo, &lw.Gate, &lw.Rel,
			&lw.SinkProj, &lw.Router, &lw.KConvProj, &lw.VConvProj} {
			mat(p)
		}
		exps(&lw.GateUp)
		exps(&lw.Down)
	}
	for i := range w.Controllers {
		cw := &w.Controllers[i]
		for c := range cw.Down {
			mat(&cw.Down[c])
			mat(&cw.Gain[c])
			mat(&cw.Forget[c])
		}
	}
	if err != nil {
		return err
	}
	s.next = m.closer
	m.closer = s
	return nil
}

// OpenInkling2LoopDevice maps an inkling2_loop GGUF like OpenInkling2Loop and
// moves its packed weights onto be.
func OpenInkling2LoopDevice(path string, be compute.Backend) (*Inkling2LoopModel, error) {
	m, err := OpenInkling2Loop(path)
	if err != nil {
		return nil, err
	}
	if err := bindInkling2Device(m, be); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

// Inkling2LoopGreedyStats times one greedy run. Decode covers the steps that
// consume generated tokens; the first generated token comes from prefill.
type Inkling2LoopGreedyStats struct {
	Prefill, Decode time.Duration
	DecodeSteps     int
}

func (s Inkling2LoopGreedyStats) DecodeTokensPerSecond() float64 {
	if s.Decode <= 0 {
		return 0
	}
	return float64(s.DecodeSteps) / s.Decode.Seconds()
}

// Inkling2LoopGreedyTimed is Inkling2LoopGreedy with prefill/decode timing. A
// backend with a token-boundary Recycle is recycled after every step; every
// device intermediate is already read back by then.
func Inkling2LoopGreedyTimed(m *Inkling2LoopModel, be compute.Backend, prompt []int, n int) ([]int, Inkling2LoopGreedyStats, error) {
	var st Inkling2LoopGreedyStats
	if len(prompt) == 0 {
		return nil, st, fmt.Errorf("inkling2_loop: empty prompt")
	}
	h, err := model.NewInkling2LoopHost(m.HP, m.Weights)
	if err != nil {
		return nil, st, err
	}
	recycle := func() {}
	if r, ok := be.(interface{ Recycle() }); ok {
		recycle = r.Recycle
	}
	start := time.Now()
	var logits []float32
	for _, t := range prompt {
		logits = h.Step(t)
		recycle()
	}
	st.Prefill = time.Since(start)
	out := make([]int, 0, n)
	start = time.Now()
	for len(out) < n {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		out = append(out, best)
		if len(out) == n {
			break
		}
		logits = h.Step(best)
		recycle()
		st.DecodeSteps++
	}
	st.Decode = time.Since(start)
	return out, st, nil
}
