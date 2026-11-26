package ggufload

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
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
	stack            compute.Tensor
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
	xt := inkling2UploadActivation(be, x, cols, padK)
	y := be.MatMul(w, xt)
	copy(dst, be.Read(y))
	be.Free(y)
	be.Free(xt)
}

// inkling2DeviceBatcher runs a batch as: upload each distinct input once,
// launch every device GEMV back to back, run host-resident jobs while the
// device works, then read the outputs. Each blocking copy used to drain the
// stream once per GEMV; now only the first read waits on device work.
type inkling2DeviceBatcher struct{ be compute.Backend }

func (b *inkling2DeviceBatcher) MatVecBatch(jobs []model.Inkling2LoopMatVecJob) {
	type devJob struct {
		w          compute.Tensor
		cols, padK int
		dst, x     []float32
	}
	devs := make([]devJob, 0, len(jobs))
	host := make([]model.Inkling2LoopMatVecJob, 0, len(jobs))
	for _, j := range jobs {
		switch m := j.Matrix.(type) {
		case *inkling2DeviceMatrix:
			devs = append(devs, devJob{m.w, m.cols, m.padK, j.Dst, j.X})
		case nil:
			if e, ok := j.Experts.(*inkling2DeviceExperts); ok {
				devs = append(devs, devJob{e.w[j.Expert], e.cols, e.padK, j.Dst, j.X})
			} else {
				host = append(host, j)
			}
		default:
			host = append(host, j)
		}
	}
	type inputKey struct {
		p    *float32
		n, k int
	}
	if pk, ok := b.be.(compute.PackedMatVecer); ok && len(devs) > 0 {
		for _, j := range host {
			inkling2HostMatVec(j)
		}
		align := func(n int) int {
			return (n + compute.PackedMatVecAlign - 1) / compute.PackedMatVecAlign * compute.PackedMatVecAlign
		}
		ws := make([]compute.Tensor, len(devs))
		xOff, yOff := make([]int, len(devs)), make([]int, len(devs))
		at := make(map[inputKey]int, 3)
		var xs []float32
		yLen := 0
		for i, d := range devs {
			key := inputKey{&d.x[0], d.cols, d.padK}
			off, seen := at[key]
			if !seen {
				off = len(xs)
				at[key] = off
				xs = append(xs, make([]float32, align(d.padK))...)
				copy(xs[off:off+d.cols], d.x[:d.cols])
			}
			ws[i], xOff[i], yOff[i] = d.w, off, yLen
			yLen += align(len(d.dst))
		}
		y, err := pk.MatVecPacked(ws, xs, xOff, yOff, yLen)
		if err != nil {
			panic(fmt.Errorf("inkling2_loop: packed matvec: %w", err))
		}
		for i, d := range devs {
			copy(d.dst, y[yOff[i]:yOff[i]+len(d.dst)])
		}
		return
	}
	inputs := make(map[inputKey]compute.Tensor, 3)
	xs := make([]compute.Tensor, len(devs))
	for i, d := range devs {
		key := inputKey{&d.x[0], d.cols, d.padK}
		xt, ok := inputs[key]
		if !ok {
			xt = inkling2UploadActivation(b.be, d.x, d.cols, d.padK)
			inputs[key] = xt
		}
		xs[i] = xt
	}
	ys := make([]compute.Tensor, len(devs))
	for i, d := range devs {
		ys[i] = b.be.MatMul(d.w, xs[i])
	}
	for _, j := range host {
		inkling2HostMatVec(j)
	}
	for i, d := range devs {
		copy(d.dst, b.be.Read(ys[i]))
		b.be.Free(ys[i])
	}
	for _, xt := range inputs {
		b.be.Free(xt)
	}
}

// ExpertMLP runs the routed MLP in one device round trip when the backend has
// the fused situ MoE and both expert stacks are device-resident.
func (b *inkling2DeviceBatcher) ExpertMLP(lw *model.Inkling2LoopLayerWeights, hp *model.Inkling2LoopHParams, top []int, weights, x []float32) ([]float32, bool) {
	sm, ok := b.be.(compute.SituMoE)
	gu, okG := lw.GateUp.(*inkling2DeviceExperts)
	dn, okD := lw.Down.(*inkling2DeviceExperts)
	if !ok || !okG || !okD {
		return nil, false
	}
	scale := func(s []float32, e int) float32 {
		if s == nil {
			return 1
		}
		return s[e]
	}
	g, d := make([]compute.Tensor, len(top)), make([]compute.Tensor, len(top))
	gs, ds := make([]float32, len(top)), make([]float32, len(top))
	for i, e := range top {
		g[i], d[i] = gu.w[e], dn.w[e]
		gs[i], ds[i] = scale(lw.GateUpScale, e), scale(lw.DownScale, e)
	}
	out, err := sm.SituMoE(g, d, x[:gu.cols], weights, gs, ds, hp.SituBeta, hp.SituLinearBeta)
	if err != nil {
		panic(fmt.Errorf("inkling2_loop: situ moe: %w", err))
	}
	return out, true
}

func inkling2HostMatVec(j model.Inkling2LoopMatVecJob) {
	if j.Matrix != nil {
		j.Matrix.MatVec(j.Dst, j.X)
	} else {
		j.Experts.ExpertMatVec(j.Expert, j.Dst, j.X)
	}
}

func inkling2UploadActivation(be compute.Backend, x []float32, cols, padK int) compute.Tensor {
	in := x[:cols]
	if padK > cols {
		in = make([]float32, padK)
		copy(in, x[:cols])
	}
	host := compute.NewF32(be, []int{len(in)}, in)
	if cu, ok := be.(inkling2ClassedUploader); ok {
		return cu.UploadClass(host, compute.F32, compute.MemoryActivation, "inkling2-activation")
	}
	return be.Upload(host, compute.F32)
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
	default:
		// Tensor types without a device weight layout take the typed unsupported
		// refusal below.
	}
	return compute.Tensor{}, 0, &UnsupportedInkling2LoopTensorTypeError{Type: typ}
}

//enumlint:exempt only these five i-quant tensor types have a raw device dtype; every other tensor type is uploaded through its own case in inkling2DeviceWeight
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
	// placement is the budget plan the bind applied.
	placement Inkling2DevicePlacement
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
	if inkling2MatrixStaysHost(q) {
		return m, nil
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
	out := &inkling2DeviceExperts{be: s.be, stack: stack, w: make([]compute.Tensor, q.experts), rows: q.rows, cols: q.cols, padK: padK}
	for x := range out.w {
		if out.w[x], err = compute.RowView(s.be, stack, x*q.rows, q.rows); err != nil {
			return nil, fmt.Errorf("inkling2_loop: expert %d of %s stack: %w", x, q.typ, err)
		}
	}
	return out, nil
}

// upload moves every bind item to the device. A non-zero window streams the
// cold load: uploads run in mapping order while a background readahead keeps
// window bytes of the upcoming payloads queued on the disk.
func (s *inkling2DeviceSet) upload(m *Inkling2LoopModel, resident []bool, window int64) error {
	items := inkling2UploadItems(m, resident)
	var ra *inkling2UploadReadahead
	if window > 0 {
		inkling2SortUploadsByMapping(items)
		payloads := make([][]byte, 0, len(items))
		for _, it := range items {
			if r := it.raw(); len(r) > 0 {
				payloads = append(payloads, r)
			}
		}
		ra = startInkling2UploadReadahead(newInkling2ReadaheadChunks(payloads, window, inkling2UploadReadaheadChunk), inkling2UploadAdvise)
		defer ra.close()
	}
	for _, it := range items {
		n := len(it.raw())
		var err error
		if it.mat != nil {
			*it.mat, err = s.matrix(*it.mat)
		} else {
			*it.exps, err = s.experts(*it.exps)
		}
		if err != nil {
			return err
		}
		if ra != nil && n > 0 {
			ra.advance(n)
		}
	}
	return nil
}

// inkling2ExpertResidencyOrder lists layer indices in the order their expert
// stacks claim device memory: layers in the loop core [CoreStart, CoreEnd) run
// Passes times per token, so a byte of their experts saves Passes reads where
// any other layer's saves one. Ties keep layer order. A schedule that does not
// fit inside layers degrades to plain layer order rather than guessing.
// Adapted from Niko1221/Strata engine/src/inkling2/ink_manifest.h:495-509 @
// c6f6427d9de9 (MIT), re-expressed without copied code.
func inkling2ExpertResidencyOrder(hp model.Inkling2LoopHParams, layers int) []int {
	order := make([]int, 0, layers)
	core := hp.Passes > 1 && hp.CoreStart >= 0 && hp.CoreStart < hp.CoreEnd && hp.CoreEnd <= layers
	if core {
		for i := hp.CoreStart; i < hp.CoreEnd; i++ {
			order = append(order, i)
		}
	}
	for i := 0; i < layers; i++ {
		if !core || i < hp.CoreStart || i >= hp.CoreEnd {
			order = append(order, i)
		}
	}
	return order
}

// bindInkling2Device swaps every packed matrix and, while they fit the device
// budget, expert stacks (chosen in residency order) of a host-bound model for
// be-resident equivalents. Experts over budget stay host quant experts that
// MatVecBatch runs on the CPU. The token embedding and small dense matrices
// stay host-side; Close frees the device tensors before the mapping.
func bindInkling2Device(m *Inkling2LoopModel, be compute.Backend) (err error) {
	budget, err := inkling2DeviceBudget(be, os.LookupEnv)
	if err != nil {
		return err
	}
	resident, placement, ramRegions, err := planInkling2Device(m, budget)
	if err != nil {
		return err
	}
	s := &inkling2DeviceSet{be: be, placement: placement}
	defer func() {
		if err != nil {
			s.Close()
		}
	}()
	if err = s.upload(m, resident, inkling2UploadReadaheadWindow(be, os.LookupEnv)); err != nil {
		return err
	}
	// Prefetch only now, so cold-stack reads never race the uploads on NVMe;
	// it wraps the mapping's closer, so Close stops it before the unmap.
	if len(ramRegions) > 0 {
		m.closer = inkling2StartRAMPrefetch(ramRegions, m.closer)
	}
	s.next = m.closer
	m.closer = s
	m.Weights.Batcher = &inkling2DeviceBatcher{be: be}
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
	// A spill drops the resident fast path, so it must never be silent.
	if p, ok := Inkling2DevicePlacementOf(m); ok && !p.FullyResident() {
		fmt.Fprintln(os.Stderr, p.String())
	}
	return m, nil
}

// Inkling2LoopGreedyStats times one greedy run. Decode covers the steps that
// consume generated tokens; the first generated token comes from prefill.
type Inkling2LoopGreedyStats struct {
	Prefill, Decode time.Duration
	DecodeSteps     int
	// Passes is the loop depth the run executed; set by Inkling2LoopGenerate.
	Passes int
	// PrefillChunk is the widest batched prefill forward the run made, in tokens; 0
	// when the prompt went through the per-token step.
	PrefillChunk int
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
	if inkling2ResidentEligible(m, be) {
		if depth := inkling2SpecDepth(); depth > 0 {
			return inkling2SpecGreedy(m, be, prompt, n, depth)
		}
		return inkling2ResidentGreedy(m, be, prompt, n)
	}
	s, err := newInkling2HostSession(m, be)
	if err != nil {
		return nil, st, err
	}
	return inkling2SessionGreedy(s, prompt, n)
}

// inkling2SessionGreedy decodes n greedy tokens from a fresh session; the first
// comes from prefill.
func inkling2SessionGreedy(s Inkling2LoopSession, prompt []int, n int) ([]int, Inkling2LoopGreedyStats, error) {
	var st Inkling2LoopGreedyStats
	start := time.Now()
	next, err := s.Prefill(context.Background(), prompt)
	if err != nil {
		return nil, st, err
	}
	st.Prefill = time.Since(start)
	out := make([]int, 0, n)
	start = time.Now()
	for len(out) < n {
		out = append(out, next)
		if len(out) == n {
			break
		}
		if next, err = s.Step(next); err != nil {
			return nil, st, err
		}
		st.DecodeSteps++
	}
	st.Decode = time.Since(start)
	return out, st, nil
}

// Inkling2LoopRequest is one native inkling2_loop generation. Loops selects the
// core pass count for this request only; 0 runs the checkpoint's loop.count.
type Inkling2LoopRequest struct {
	Prompt    []int
	MaxTokens int
	Loops     int
}

// Inkling2LoopGenerate greedily decodes one request at its own loop depth. The
// loaded weights are shared; only the per-request cache is sized for the
// selected pass count, so a shallower request also holds fewer KV slots.
func Inkling2LoopGenerate(m *Inkling2LoopModel, be compute.Backend, req Inkling2LoopRequest) ([]int, Inkling2LoopGreedyStats, error) {
	hp, err := m.HP.WithPasses(req.Loops)
	if err != nil {
		return nil, Inkling2LoopGreedyStats{}, err
	}
	run := &Inkling2LoopModel{HP: hp, Weights: m.Weights}
	out, st, err := Inkling2LoopGreedyTimed(run, be, req.Prompt, req.MaxTokens)
	st.Passes = hp.Passes
	return out, st, err
}

// inkling2BackendSM is the compute capability of a CUDA backend from its
// "sm_120" tier, 0 for any other backend or an unparseable tier.
func inkling2BackendSM(be compute.Backend) int {
	if be == nil || be.Name() != "cuda" {
		return 0
	}
	sm, err := strconv.Atoi(strings.TrimPrefix(be.Tier(), "sm_"))
	if err != nil || sm <= 0 {
		return 0
	}
	return sm
}

// inkling2DefaultOn resolves an Inkling-2 on/off switch through the per-arch
// default table, so a value the user set wins and each arch default cites its
// run. Only an explicit "0" turns the switch off: an arch with no row, or one
// whose kill switch is set, keeps the built-in on.
func inkling2DefaultOn(be compute.Backend, env string, lookup func(string) (string, bool)) bool {
	return compute.ResolveArchDefault(inkling2BackendSM(be), env, lookup).Value != "0"
}

// inkling2ResidentEligible routes a device-bound model on a backend with the
// inkling2 device ops through the resident forward; FAK_INKLING2_RESIDENT=0 keeps
// the host-driven per-matrix path. A model with any cold (host) expert stack
// is ineligible: the resident forward has no host-expert fallback.
func inkling2ResidentEligible(m *Inkling2LoopModel, be compute.Backend) bool {
	if !inkling2DefaultOn(be, "FAK_INKLING2_RESIDENT", os.LookupEnv) || len(m.Weights.Layers) == 0 {
		return false
	}
	if _, ok := be.(compute.Inkling2Device); !ok {
		return false
	}
	if p, ok := Inkling2DevicePlacementOf(m); ok && !p.FullyResident() {
		return false
	}
	if _, ok := m.Weights.Layers[0].GateUp.(*inkling2DeviceExperts); !ok {
		return false
	}
	return inkling2ExpertsAllResident(m.Weights)
}
