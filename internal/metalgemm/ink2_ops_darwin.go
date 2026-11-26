//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
)

//go:embed ink2_ops.metal
var ink2OpsSource string

// These mirror the limits compiled into ink2_ops.metal.
const (
	ink2OpsThreads      = 256
	ink2OpsRouteThreads = 32
	ink2OpsMaxStates    = 16
	// Weight slots of an RMS launch, queries of a mix, controllers of a control.
	ink2OpsMaxRows    = 4
	ink2OpsMaxExperts = 512
	ink2OpsMaxTopK    = 16
	// The kernels take a mean as sum / float(n), which is exact up to here.
	ink2OpsMaxWidth = 1 << 24
)

type ink2RMSShape struct {
	n                uint32
	eps              float32
	weighted, perRow uint32
}

type ink2MixShape struct {
	n, states, norm, weighted uint32
	eps, invSqrtN             float32
}

type ink2CtrlShape struct{ n, rank uint32 }

type ink2RouteShape struct {
	experts, slots, biased uint32
	weightScale            float32
}

type ink2SituShape struct {
	width, actStride uint32
	beta, linearBeta float32
}

type ink2CombineShape struct{ slots, n uint32 }

type ink2ArgmaxShape struct {
	n       uint32
	softcap float32
}

// opsKernel returns a kernel of ink2_ops.metal. A reducing kernel sums across
// 32-wide SIMD groups of a full ink2OpsThreads threadgroup.
func (d *Ink2Device) opsKernel(name string, reducing bool) (Ink2Pipeline, error) {
	p, err := d.PipelinePrecise(ink2OpsSource, name)
	if err != nil {
		return Ink2Pipeline{}, err
	}
	if reducing && (p.ThreadWidth() != 32 || p.MaxThreads() < ink2OpsThreads) {
		return Ink2Pipeline{}, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of %d threads", name, ink2OpsThreads)
	}
	return p, nil
}

// opsEach encodes one thread per element of a w x h grid.
func (d *Ink2Device) opsEach(p Ink2Pipeline, w, h int, args ...Ink2Arg) error {
	return d.DispatchThreads(p, [3]int{w, h, 1}, [3]int{min(w, ink2OpsThreads, p.MaxThreads()), 1, 1}, args...)
}

func ink2OpsOverlap(a Ink2Ref, an int, b Ink2Ref, bn int) bool {
	return a.Valid() && a.Buf == b.Buf && a.Off < b.Off+bn && b.Off < a.Off+an
}

// ink2OpsSlots pads refs to the kernels' four binding slots.
func ink2OpsSlots(refs []Ink2Ref) (slots [ink2OpsMaxRows]Ink2Ref) {
	copy(slots[:], refs)
	return slots
}

// RMS encodes, for each of rows rows of width n, y = x * s (* w) with
// s = 1/sqrt(mean(x^2) + eps). Every row shares the optional weight w; y may be x.
func (d *Ink2Device) RMS(x, w, y Ink2Ref, rows, n int, eps float32) error {
	if !x.Valid() || !y.Valid() || rows < 1 || n < 1 || n > ink2OpsMaxWidth || rows > math.MaxInt32/n {
		return fmt.Errorf("%w: rms of %d rows of %d", ErrInk2Shape, rows, n)
	}
	return d.rmsRows(x, y, []Ink2Ref{w}, false, rows, n, eps, ink2OpsLegacy.rms)
}

// rmsRows is RMS with, when perRow, row r weighted by w[r] (absent for none).
func (d *Ink2Device) rmsRows(x, y Ink2Ref, w []Ink2Ref, perRow bool, rows, n int, eps float32, legacy bool) error {
	p, _, err := d.opsOrLean("ink2_rms", "ink2_rms_lean", legacy, n)
	if err != nil {
		return err
	}
	shape := ink2RMSShape{n: uint32(n), eps: eps, perRow: uint32(btoi(perRow))}
	args := []Ink2Arg{x.Span(rows * n), y.Span(rows * n)}
	for i, slot := range ink2OpsSlots(w) {
		if slot.Valid() {
			shape.weighted |= 1 << i
		}
		args = append(args, slot.Span(n))
	}
	return d.Dispatch(p, [3]int{rows, 1, 1}, [3]int{ink2OpsThreads, 1, 1}, append(args, Ink2Const(shape))...)
}

// CtrlPair encodes the unweighted RMS (eps 1e-6) of cur and of msg into the
// two halves of pair ([2][n]).
func (d *Ink2Device) CtrlPair(cur, msg, pair Ink2Ref, n int) error {
	if !cur.Valid() || !msg.Valid() || !pair.Valid() || n < 1 || n > ink2OpsMaxWidth {
		return fmt.Errorf("%w: ctrl pair of width %d", ErrInk2Shape, n)
	}
	p, err := d.opsKernel("ink2_ctrl_pair", true)
	if err != nil {
		return err
	}
	return d.Dispatch(p, [3]int{2, 1, 1}, [3]int{ink2OpsThreads, 1, 1}, cur.Span(n), msg.Span(n), pair.Span(2*n), Ink2Const(uint32(n)))
}

// Mix encodes nq depth mixes of states (each [n]) into the rows of out
// ([nq][n]): every state RMS-normalized (eps 1e-6), scored against query q of
// queries ([nq][n]), softmaxed and summed in state order. With norms non-nil
// each mixed row is then RMS-normalized (eps) and scaled by norms[q], an
// absent entry leaving it unscaled. out must not overlap an input.
func (d *Ink2Device) Mix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n int) error {
	ns := len(states)
	if ns < 1 || ns > ink2OpsMaxStates || nq < 1 || nq > ink2OpsMaxRows || len(norms) > ink2OpsMaxRows ||
		n < 1 || n > ink2OpsMaxWidth || !queries.Valid() || !out.Valid() || ink2OpsOverlap(queries, nq*n, out, nq*n) {
		return fmt.Errorf("%w: mix of %d states by %d queries of width %d", ErrInk2Shape, ns, nq, n)
	}
	p, _, err := d.opsOrLean("ink2_mix", "ink2_mix_lean", ink2OpsLegacy.mix, n)
	if err != nil {
		return err
	}
	args := make([]Ink2Arg, 0, ink2OpsMaxStates+ink2OpsMaxRows+3)
	for i := range ink2OpsMaxStates {
		var s Ink2Ref
		if i < ns {
			if s = states[i]; !s.Valid() || ink2OpsOverlap(s, n, out, nq*n) {
				return fmt.Errorf("%w: mix state %d is absent or overlaps the output", ErrInk2Shape, i)
			}
		}
		args = append(args, s.Span(n))
	}
	args = append(args, queries.Span(nq*n))
	shape := ink2MixShape{n: uint32(n), states: uint32(ns), norm: uint32(btoi(norms != nil)), eps: eps,
		invSqrtN: float32(1 / math.Sqrt(float64(n)))}
	for q, w := range ink2OpsSlots(norms) {
		if q >= nq {
			w = Ink2Ref{}
		}
		if w.Valid() {
			shape.weighted |= 1 << q
		}
		args = append(args, w.Span(n))
	}
	args = append(args, out.Span(nq*n), Ink2Const(shape))
	return d.Dispatch(p, [3]int{nq, 1, 1}, [3]int{ink2OpsThreads, 1, 1}, args...)
}

// Control encodes len(ctl.Cur) loop controllers into the rows of out
// ([j][n]): g_j = Down_j [rms(cur_j), rms(msg_j)] into the scratch g
// ([j][rank]), then out_j = cur_j - tanh(F_j g_j)*cur_j + tanh(G_j g_j)*msg_j.
// Down_j is [rank][2n]; Gain_j and Forget_j are [n][rank]. With norms non-nil
// each output row is then RMS-normalized (eps) and scaled by its weight.
func (d *Ink2Device) Control(ctl Ink2Controls, g, out Ink2Ref, n, rank int, norms []Ink2Ref, eps float32) error {
	nj := len(ctl.Cur)
	if nj < 1 || nj > ink2OpsMaxRows || len(ctl.Msg) != nj || len(ctl.Down) != nj || len(ctl.Gain) != nj ||
		len(ctl.Forget) != nj || len(norms) > nj || !g.Valid() || !out.Valid() ||
		n < 1 || n > ink2OpsMaxWidth || rank < 1 || rank > math.MaxInt32/(2*n) {
		return fmt.Errorf("%w: control of %d controllers, width %d, rank %d", ErrInk2Shape, nj, n, rank)
	}
	for j := range nj {
		if !ctl.Cur[j].Valid() || !ctl.Msg[j].Valid() || !ctl.Down[j].Valid() || !ctl.Gain[j].Valid() || !ctl.Forget[j].Valid() {
			return fmt.Errorf("%w: controller %d is missing an operand", ErrInk2Shape, j)
		}
	}
	down, _, err := d.opsOrLean("ink2_ctrl_down", "ink2_ctrl_down_lean", ink2OpsLegacy.ctrl, n)
	if err != nil {
		return err
	}
	apply, err := d.ctrlApplyKernel(n, rank)
	if err != nil {
		return err
	}
	bind := func(args []Ink2Arg, refs []Ink2Ref, span int) []Ink2Arg {
		for _, r := range ink2OpsSlots(refs) {
			args = append(args, r.Span(span))
		}
		return args
	}
	shape := Ink2Const(ink2CtrlShape{n: uint32(n), rank: uint32(rank)})
	inputs := bind(bind(make([]Ink2Arg, 0, 19), ctl.Cur, n), ctl.Msg, n)
	args := append(bind(inputs[:len(inputs):len(inputs)], ctl.Down, rank*2*n), g.Span(nj*rank), shape)
	if err := d.Dispatch(down, [3]int{rank, nj, 1}, [3]int{ink2OpsThreads, 1, 1}, args...); err != nil {
		return err
	}
	args = append(bind(bind(inputs, ctl.Gain, n*rank), ctl.Forget, n*rank), g.Span(nj*rank), out.Span(nj*n), shape)
	if err := d.opsEach(apply, n, nj, args...); err != nil {
		return err
	}
	if norms == nil {
		return nil
	}
	return d.rmsRows(out, out, norms, true, nj, n, eps, ink2OpsLegacy.rms || ink2OpsLegacy.ctrl)
}

// SigmoidMul encodes x *= sigmoid(gate) over n elements.
func (d *Ink2Device) SigmoidMul(x, gate Ink2Ref, n int) error {
	if !x.Valid() || !gate.Valid() || n < 1 || n > math.MaxInt32 {
		return fmt.Errorf("%w: sigmoid mul of %d", ErrInk2Shape, n)
	}
	p, err := d.opsKernel("ink2_sigmoid_mul", false)
	if err != nil {
		return err
	}
	return d.opsEach(p, n, 1, x.Span(n), gate.Span(n))
}

// Route encodes the MoE routing of nexp router logits: the k experts with the
// largest sigmoid(logit) + bias (bias optional, ties to the lower index) as
// int32 ids, and their sigmoid probabilities renormalized over the selection
// and multiplied by a nonzero scale as wts.
func (d *Ink2Device) Route(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32) error {
	return d.RouteRows(logits, bias, ids, wts, nexp, k, scale, 1)
}

// RouteRows is Route for T tokens in one launch: token t's logits at
// logits + t*nexp, its ids and weights at t*k. Each token is Route's bits.
func (d *Ink2Device) RouteRows(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	if !logits.Valid() || !ids.Valid() || !wts.Valid() || nexp < 1 || nexp > ink2OpsMaxExperts || k < 1 || k > ink2OpsMaxTopK || k > nexp ||
		T < 1 || T > math.MaxInt32/nexp {
		return fmt.Errorf("%w: route of %d experts to %d slots for %d tokens", ErrInk2Shape, nexp, k, T)
	}
	p, _, err := d.opsOrLean("ink2_route", "ink2_route_lean", ink2OpsLegacy.route || nexp > ink2LeanMaxExperts, 1)
	if err != nil {
		return err
	}
	return d.Dispatch(p, [3]int{T, 1, 1}, [3]int{ink2OpsRouteThreads, 1, 1},
		logits.Span(T*nexp), bias.Span(nexp), ids.Span(T*k), wts.Span(T*k),
		Ink2Const(ink2RouteShape{experts: uint32(nexp), slots: uint32(k), biased: uint32(btoi(bias.Valid())), weightScale: scale}))
}

// Situ encodes the SITU activation of k routed slots: slot t reads its
// [gate | up] row of 2*width values at gu + t*2*width and writes width values
// at act + t*actStride, leaving the rest of the stride untouched. MSL has no
// float64, so the betas and the activation are float32.
func (d *Ink2Device) Situ(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	if !gu.Valid() || !act.Valid() || k < 1 || width < 1 || actStride < width || actStride > math.MaxInt32/k {
		return fmt.Errorf("%w: situ of %d slots of width %d, stride %d", ErrInk2Shape, k, width, actStride)
	}
	p, err := d.opsKernel("ink2_situ", false)
	if err != nil {
		return err
	}
	return d.opsEach(p, width, k, gu.Span(k*2*width), act.Span((k-1)*actStride+width),
		Ink2Const(ink2SituShape{width: uint32(width), actStride: uint32(actStride), beta: float32(beta), linearBeta: float32(linearBeta)}))
}

// Combine encodes out[j] = sum over the k routed rows of down ([k][n]), in
// routing order, of wts[t] * down[t][j].
func (d *Ink2Device) Combine(down, wts, out Ink2Ref, k, n int) error {
	return d.CombineRows(down, wts, out, k, n, 1)
}

// CombineRows is Combine for T tokens in one launch: token t's [k][n] down rows
// at down + t*k*n, its weights at t*k and its output at t*n.
func (d *Ink2Device) CombineRows(down, wts, out Ink2Ref, k, n, T int) error {
	if !down.Valid() || !wts.Valid() || !out.Valid() || k < 1 || n < 1 || T < 1 || k > math.MaxInt32/n || T > math.MaxInt32/(k*n) {
		return fmt.Errorf("%w: moe combine of %d slots of width %d for %d tokens", ErrInk2Shape, k, n, T)
	}
	p, err := d.opsKernel("ink2_combine", false)
	if err != nil {
		return err
	}
	return d.opsEach(p, n, T, down.Span(T*k*n), wts.Span(T*k), out.Span(T*n), Ink2Const(ink2CombineShape{slots: uint32(k), n: uint32(n)}))
}

// Add encodes out = a + b over n elements.
func (d *Ink2Device) Add(a, b, out Ink2Ref, n int) error {
	if !a.Valid() || !b.Valid() || !out.Valid() || n < 1 || n > math.MaxInt32 {
		return fmt.Errorf("%w: add of %d", ErrInk2Shape, n)
	}
	p, err := d.opsKernel("ink2_add", false)
	if err != nil {
		return err
	}
	return d.opsEach(p, n, 1, a.Span(n), b.Span(n), out.Span(n))
}

// SoftcapArgmax encodes logits = softcap*tanh(logits/softcap) in place over n
// elements (skipped when softcap is 0) and the index of the largest, ties to
// the lower index, as an int32 at outIdx. outIdx is four words: the two after
// the index's are the reduction scratch.
func (d *Ink2Device) SoftcapArgmax(logits Ink2Ref, n int, softcap float32, outIdx Ink2Ref) error {
	if !logits.Valid() || !outIdx.Valid() || n < 1 || n > math.MaxInt32-ink2OpsThreads {
		return fmt.Errorf("%w: softcap argmax of %d", ErrInk2Shape, n)
	}
	begin, err := d.opsKernel("ink2_argmax_begin", false)
	if err != nil {
		return err
	}
	fold, err := d.opsKernel("ink2_softcap_max", true)
	if err != nil {
		return err
	}
	index, err := d.opsKernel("ink2_argmax_index", false)
	if err != nil {
		return err
	}
	slot, shape := outIdx.Span(4), Ink2Const(ink2ArgmaxShape{n: uint32(n), softcap: softcap})
	if err := d.opsEach(begin, 1, 1, slot); err != nil {
		return err
	}
	groups := (n + ink2OpsThreads - 1) / ink2OpsThreads
	if err := d.Dispatch(fold, [3]int{groups, 1, 1}, [3]int{ink2OpsThreads, 1, 1}, logits.Span(n), slot, shape); err != nil {
		return err
	}
	return d.opsEach(index, n, 1, logits.Span(n), slot, shape)
}
