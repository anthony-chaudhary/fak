package ggufload

import (
	"context"
	"fmt"
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2PrefillChunk is the token count of one batched prefill forward. Local
// slots keep a window+chunk ring so a chunk's appended keys never evict a key an
// earlier query of the same chunk still attends to.
const inkling2PrefillChunk = 512

// inkling2BatchedPrefill reports whether the resident path prefills a prompt in
// token-axis batches, which decode the same tokens as the per-token step;
// FAK_INKLING2_BATCHED_PREFILL=0 keeps the per-token step.
func inkling2BatchedPrefill() bool { return os.Getenv("FAK_INKLING2_BATCHED_PREFILL") != "0" }

// inkling2GroupedExperts reports whether NVFP4 expert stacks run the expert-grouped
// GEMM in prefill; FAK_INKLING2_PREFILL_GROUPED=0 runs every (token, slot) pair
// through the decode expert GEMV instead.
func inkling2GroupedExperts() bool { return os.Getenv("FAK_INKLING2_PREFILL_GROUPED") != "0" }

// inkling2PrefillControlBatch reports whether a prefill visit runs its Q/K/V
// controllers as one grid per stage; FAK_INKLING2_PREFILL_CTRL_BATCH=1 opts in.
func inkling2PrefillControlBatch() bool { return os.Getenv("FAK_INKLING2_PREFILL_CTRL_BATCH") == "1" }

// ink2Prefill is one batched prefill of a resident session. Values that outlive a
// layer visit (layer outputs, residual blocks) take keep slots; everything else is
// layer scratch, reset at the start of every visit.
type ink2Prefill struct {
	s          *inkling2Resident
	pd         compute.Inkling2PrefillDevice
	half       compute.Inkling2PrefillHalfKV // set when s stores f16 global caches
	fp8        compute.Inkling2FP8KVDevice   // set when s stores fp8 global caches
	nvfp4      compute.Inkling2NVFP4KVDevice // set when s stores nvfp4 global caches
	T, pos0    int
	keep, scr  ink2Ref
	keepOff    int
	scrOff     int
	keepFloats int
	scrFloats  int
	grouped    bool
	// verify is a speculative verify chunk: convs log their inputs for a later
	// commit and the head runs on every row.
	verify bool
	vd     compute.Inkling2VerifyDevice
	// rows, when set, pins 2..127-row Q8_0 matmuls to the row-tile panel GEMM
	rows compute.Inkling2BatchMemory
	// fewRows runs dense matmuls, mixes and experts on vd's few-row kernels as a
	// verify chunk does, without the verify chunk's conv log, device positions or head.
	fewRows bool
	// q8 maps a scratch activation offset to its Q8_0 codes for this visit
	q8 map[int]ink2Ref
	// in holds the chunk's embedding rows; unset reads the session's prefillIn
	in ink2Ref
	// cohort, when set, splits the chunk into the prompt segments of cohort sequences
	cohort *ink2PrefillRows
}

func (p *ink2Prefill) fail(err error) { p.s.fail(err) }

func (p *ink2Prefill) take(base ink2Ref, off *int, limit, n int, what string) ink2Ref {
	n = (n + 63) &^ 63
	if *off+n > limit {
		p.fail(fmt.Errorf("inkling2 prefill: %s arena exhausted at %d floats", what, *off+n))
		return base
	}
	r := compute.Ink2At(base.T, base.Off+*off)
	*off += n
	return r
}

func (p *ink2Prefill) tmp(n int) ink2Ref  { return p.take(p.scr, &p.scrOff, p.scrFloats, n, "scratch") }
func (p *ink2Prefill) hold(n int) ink2Ref { return p.take(p.keep, &p.keepOff, p.keepFloats, n, "keep") }

// prefillArenas sizes the two prefill arenas for chunks of T tokens.
func (s *inkling2Resident) prefillArenas(T int) (keep, scr int) {
	return inkling2PrefillArenaFloats(s.hp, T, s.maxPadK())
}

// maxPadK is the widest padded expert row across the session's layers.
func (s *inkling2Resident) maxPadK() int {
	padK := 0
	for i := range s.layers {
		padK = max(padK, s.layers[i].down.padK, s.layers[i].gateUp.padK)
	}
	return padK
}

// inkling2PrefillArenaFloats is the prefill arena sizing as a function of the shape,
// so the device-memory fit can size the arenas before any session exists.
func inkling2PrefillArenaFloats(hp model.Inkling2LoopHParams, T, padK int) (keep, scr int) {
	visits := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	keep = (visits + 2) * (T*hp.Hidden + 64)
	kvDim := hp.KVHeads * hp.HeadDim
	k := hp.ExpertsUsed
	perTok := 4*4*hp.Hidden + // msgs
		3*hp.Hidden*3 + // mix, control, rms inputs
		hp.Heads*hp.HeadDim*3 + hp.Heads*hp.RelLength + hp.Heads + 4*kvDim + 2*hp.KVHeads*hp.ConvKernel +
		6*hp.Hidden + 256 + 2*k +
		k*(hp.Hidden+padK+2*hp.ExpertWidth+padK+hp.Hidden) +
		k*(1+max(hp.Hidden, padK)+max(2*hp.ExpertWidth, hp.Hidden)) // sorted-pair staging
	scr = T*perTok + 64*64 + compute.Ink2PGroupScratch(T*k, 256) + 64
	return keep, scr
}

// prefill runs prompt through the batched forward in chunks, leaving every cache
// slot and conv history where the per-token path would, and returns the greedy
// token after the last prompt position. ctx is checked before each chunk, so a
// cancelled prefill stops after the chunk in flight and leaves s.pos at the chunks
// that ran.
func (s *inkling2Resident) prefill(ctx context.Context, prompt []int) (int, error) {
	pd, ok := s.be.(compute.Inkling2PrefillDevice)
	if !ok {
		return 0, fmt.Errorf("inkling2 prefill: backend %s has no batched prefill ops", s.be.Name())
	}
	hp := s.hp
	if s.pos+len(prompt) > s.ctx {
		return 0, fmt.Errorf("inkling2 prefill: %d tokens exceed the %d-token global cache", s.pos+len(prompt), s.ctx)
	}
	T := inkling2PrefillChunkFor(len(prompt), s.chunkMax)
	startPos := s.pos
	p := &ink2Prefill{s: s, pd: pd, grouped: inkling2GroupedExperts()}
	if s.kvHalf {
		// inkling2ResidentKVDType has already checked these ops exist on this backend
		p.half, _ = s.be.(compute.Inkling2PrefillHalfKV)
	}
	if s.kvFP8 {
		p.fp8 = s.fp8Dev
	}
	if s.kvNVFP4 {
		p.nvfp4 = s.nvfp4Dev
	}
	s.ensurePrefillArenas(T)
	p.keepFloats, p.scrFloats = s.prefillArenas(s.prefillCap)
	p.keep, p.scr = s.prefillKeep, s.prefillScr
	if s.err != nil {
		return 0, s.err
	}
	rows := make([]float32, T*hp.Hidden)
	for start := 0; start < len(prompt); start += T {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		n := min(T, len(prompt)-start)
		for i, tok := range prompt[start : start+n] {
			s.w.TokEmbd.Row(tok, rows[i*hp.Hidden:(i+1)*hp.Hidden])
		}
		s.fail(s.dev.Ink2Write(s.prefillIn, rows[:n*hp.Hidden]))
		p.T, p.pos0 = n, s.pos
		p.keepOff, p.scrOff = 0, 0
		if s.draft != nil {
			s.fail(s.dev.Ink2Write(s.verifyPos, []float32{intBits(s.pos)}))
		}
		p.forward(start+n == len(prompt))
		if s.err != nil {
			return 0, s.err
		}
		s.pos += n
		s.prefillWidest = max(s.prefillWidest, n)
	}
	got, err := s.dev.Ink2ReadInts(s.argIdx, 1)
	if err != nil {
		return 0, err
	}
	if overflow, oerr := s.checkKVOverflow(); oerr != nil {
		return 0, oerr
	} else if overflow {
		if startPos == 0 {
			if s.kvNVFP4 {
				if ferr := s.fallbackNVFP4ToHigher(); ferr != nil {
					return 0, ferr
				}
				return s.prefill(ctx, prompt)
			}
			if ferr := s.fallbackFP8ToFP16(); ferr != nil {
				return 0, ferr
			}
			return s.prefill(ctx, prompt)
		}
		if s.kvNVFP4 {
			return 0, ErrInkling2NVFP4ScaleOverflow
		}
		return 0, ErrInkling2FP8ScaleOverflow
	}
	if r, ok := s.be.(interface{ Recycle() }); ok {
		r.Recycle()
	}
	return int(got[0]), nil
}

func (p *ink2Prefill) mm(m ink2Mat, x ink2Ref) ink2Ref {
	y := p.tmp(p.T * m.rows)
	if p.s.err != nil {
		return y
	}
	// a verify chunk reads each weight once for all its rows, each row equal to the
	// decode GEMV
	if (p.verify || p.fewRows) && p.T <= 9 {
		ok, err := inkling2RowsMatMul(p.vd, m, x, y, p.T, p.q8Codes)
		if err == nil && !ok && (m.t.Dtype == compute.F32 || p.verify && inkling2VerifyMMA(p.s.be)) {
			// the prompt path's dense F32 GEMM is slow at a few rows and sums in another
			// order; in MMA mode every verify weight takes the MMA rows kernel
			ok, err = p.vd.Ink2VRows(m.t, x, y, p.T)
		}
		if err != nil || ok {
			p.fail(err)
			return y
		}
	}
	if p.rows != nil && p.T >= 2 && p.T < 128 && m.t.Dtype == compute.Q8_0 {
		p.fail(p.rows.Ink2PMatMulRows(m.t, x, y, p.T))
		return y
	}
	p.fail(p.pd.Ink2PMatMul(m.t, x, y, p.T))
	return y
}

// mmMany is mm for each (ms[i], xs[i]); on a verify chunk the Q8_0 weights of one
// input width share a single launch.
func (p *ink2Prefill) mmMany(ms []ink2Mat, xs []ink2Ref) []ink2Ref {
	ys := make([]ink2Ref, len(ms))
	done := make([]bool, len(ms))
	if p.verify && p.T <= 9 && p.s.err == nil {
		var err error
		done, err = inkling2SegsMatMul(p.vd, ms, xs, ys, p.T, p.tmp)
		p.fail(err)
	}
	for i, m := range ms {
		if !done[i] {
			ys[i] = p.mm(m, xs[i])
		}
	}
	return ys
}

// inkling2SegsMatMul runs the Q8_0 weights of ms that share the first one's input
// width in one launch, writing their ys (allocated with tmp), and reports which
// products it ran.
func inkling2SegsMatMul(vd compute.Inkling2VerifyDevice, ms []ink2Mat, xs, ys []ink2Ref, rows int, tmp func(int) ink2Ref) ([]bool, error) {
	done := make([]bool, len(ms))
	var idx []int
	cols := 0
	for i, m := range ms {
		if m.t.Dtype != compute.Q8_0 || len(idx) == 8 {
			continue
		}
		if len(idx) == 0 {
			cols = m.cols
		}
		if m.cols == cols {
			idx = append(idx, i)
		}
	}
	if len(idx) < 2 {
		return done, nil
	}
	ws, sx, sy := make([]compute.Tensor, len(idx)), make([]ink2Ref, len(idx)), make([]ink2Ref, len(idx))
	distinct := 0
	for j, i := range idx {
		ws[j], sx[j], sy[j] = ms[i].t, xs[i], tmp(rows*ms[i].rows)
		seen := false
		for _, e := range idx[:j] {
			seen = seen || (xs[e].T.Buf() == xs[i].T.Buf() && xs[e].Off == xs[i].Off)
		}
		if !seen {
			distinct++
		}
	}
	ok, err := vd.Ink2VQ8Segs(ws, sx, sy, tmp(distinct*rows*(cols/4+cols/32)), rows)
	if err != nil || !ok {
		return done, err
	}
	for j, i := range idx {
		ys[i], done[i] = sy[j], true
	}
	return done, nil
}

// q8Codes returns the Q8_0 code buffer for activation x of rows*n floats and whether
// it still has to be filled. Scratch activations are never rewritten before the next
// visit resets the scratch, so their codes are shared by every weight that reads them.
func (p *ink2Prefill) q8Codes(x ink2Ref, n int) (ink2Ref, bool) {
	size := p.T * (n/4 + n/32)
	if x.T.Buf() != p.scr.T.Buf() {
		return p.tmp(size), true
	}
	if q, ok := p.q8[x.Off]; ok {
		return q, false
	}
	if p.q8 == nil {
		p.q8 = map[int]ink2Ref{}
	}
	q := p.tmp(size)
	p.q8[x.Off] = q
	return q, true
}

// inkling2RowsMatMul runs y = m.x[t] for rows <= 9 through the few-row kernels; ok is
// false (nothing launched) for a dtype without one. codes returns x's Q8_0 code
// buffer and whether it must be (re)quantized.
func inkling2RowsMatMul(vd compute.Inkling2VerifyDevice, m ink2Mat, x, y ink2Ref, rows int, codes func(ink2Ref, int) (ink2Ref, bool)) (bool, error) {
	switch m.t.Dtype {
	case compute.Q8_0:
		q, fresh := codes(x, m.cols)
		return vd.Ink2VQ8Rows(m.t, x, ink2Ref{}, q, y, rows, fresh)
	case compute.Q5_K, compute.Q6_K:
		return vd.Ink2VRows(m.t, x, y, rows)
	}
	return false, nil
}

// mmGated is mm(m, x * sigmoid(gate)) for x of n floats per row; on a verify chunk's
// Q8_0 weight the gate is applied while x is quantized, so x itself stays unscaled.
func (p *ink2Prefill) mmGated(m ink2Mat, x, gate ink2Ref, n int) ink2Ref {
	s := p.s
	if s.err == nil && p.verify && p.T <= 9 && m.t.Dtype == compute.Q8_0 {
		y := p.tmp(p.T * m.rows)
		ok, err := p.vd.Ink2VQ8Rows(m.t, x, gate, p.tmp(p.T*(m.cols/4+m.cols/32)), y, p.T, true)
		if err != nil || ok {
			p.fail(err)
			return y
		}
	}
	if s.err == nil {
		p.fail(s.dev.Ink2SigmoidMul(x, gate, p.T*n))
	}
	return p.mm(m, x)
}

// rmsRows RMS-normalizes rows contiguous rows of x by w in one launch.
func (p *ink2Prefill) rmsRows(x, w ink2Ref, rows, n int, eps float32) ink2Ref {
	y := p.tmp(rows * n)
	if p.s.err == nil {
		p.fail(p.s.dev.Ink2RMS(x, w, y, rows, n, eps))
	}
	return y
}

func (p *ink2Prefill) rms(x, w ink2Ref, n int, eps float32) ink2Ref {
	y := p.tmp(p.T * n)
	p.rmsInto(x, w, y, n, eps)
	return y
}

// rmsInto RMS-normalizes the chunk's rows of x by w into y. A verify chunk keeps the
// decode kernel: its tokens must be the decode step's, which the prompt kernels'
// fp32 reductions do not promise.
func (p *ink2Prefill) rmsInto(x, w, y ink2Ref, n int, eps float32) {
	switch {
	case p.s.err != nil:
	case p.verify:
		p.fail(p.s.dev.Ink2RMS(x, w, y, p.T, n, eps))
	default:
		p.fail(p.pd.Ink2PRMS(x, w, y, p.T, n, eps))
	}
}

// mixNorm is mix with query q RMS-normalized by norms[q] (norms nil: no norm). A
// verify chunk runs both in one cluster launch per token row.
func (p *ink2Prefill) mixNorm(states []ink2Ref, queries ink2Ref, nq int, norms []ink2Ref) []ink2Ref {
	s := p.s
	h := s.hp.Hidden
	if (p.verify || p.fewRows) && s.err == nil {
		out := p.tmp(nq * p.T * h)
		ok, err := p.vd.Ink2VMix(states, queries, nq, norms, s.hp.NormEps, out, h, p.T)
		if err != nil || ok {
			p.fail(err)
			refs := make([]ink2Ref, nq)
			for q := range refs {
				refs[q] = at(out, q*p.T*h)
			}
			return refs
		}
	}
	refs := p.mixPlain(states, queries, nq)
	for q := range refs {
		if q < len(norms) {
			refs[q] = p.rms(refs[q], norms[q], h, s.hp.NormEps)
		}
	}
	return refs
}

// mix returns the nq query outputs, each a [T][hidden] matrix.
func (p *ink2Prefill) mix(states []ink2Ref, queries ink2Ref, nq int) []ink2Ref {
	return p.mixNorm(states, queries, nq, nil)
}

func (p *ink2Prefill) mixPlain(states []ink2Ref, queries ink2Ref, nq int) []ink2Ref {
	h := p.s.hp.Hidden
	out := p.tmp(nq * p.T * h)
	if p.s.err == nil {
		p.fail(p.pd.Ink2PMix(states, queries, nq, out, h, p.T))
	}
	refs := make([]ink2Ref, nq)
	for q := range refs {
		refs[q] = at(out, q*p.T*h)
	}
	return refs
}

// controls runs controllers 0..len(cur)-1 of c on (cur[j], msg[j]) in one launch per
// stage, the result of each equal to control's.
func (p *ink2Prefill) controls(cur, msg []ink2Ref, c *ink2Ctrl) []ink2Ref {
	return p.controlsAt(cur, msg, c, 0)
}

// controlsAt is controls for controllers j0..j0+len(cur)-1, each with the decode
// controller's float64 tanh.
func (p *ink2Prefill) controlsAt(cur, msg []ink2Ref, c *ink2Ctrl, j0 int) []ink2Ref {
	h := p.s.hp.Hidden
	J := len(cur)
	g, out := make([]ink2Ref, J), make([]ink2Ref, J)
	// the outputs are contiguous, so one rms can normalize them all
	gs, outs := p.tmp(J*p.T*c.rank), p.tmp(J*p.T*h)
	for j := range g {
		g[j], out[j] = at(gs, j*p.T*c.rank), at(outs, j*p.T*h)
	}
	if p.s.err == nil {
		p.fail(p.vd.Ink2PControlMulti(cur, msg, c.down[j0:j0+J], c.gain[j0:j0+J], c.forget[j0:j0+J], g, out, h, c.rank, p.T))
	}
	return out
}

// controlBatch runs controllers 0..len(cur)-1 of c on the prefill controller
// kernels in one grid per stage, each output equal to control's; ok is false when
// the device or the toggle does not offer it.
func (p *ink2Prefill) controlBatch(cur, msg []ink2Ref, c *ink2Ctrl) ([]ink2Ref, bool) {
	bd, ok := p.pd.(compute.Inkling2PrefillControlBatchDevice)
	if !ok || !inkling2PrefillControlBatch() {
		return nil, false
	}
	h := p.s.hp.Hidden
	J := len(cur)
	g, outs := p.tmp(J*p.T*c.rank), p.tmp(J*p.T*h)
	if p.s.err == nil {
		p.fail(bd.Ink2PCtrlBatch(cur, msg, c.down[:J], c.gain[:J], c.forget[:J], g, outs, h, c.rank, p.T))
	}
	out := make([]ink2Ref, J)
	for j := range out {
		out[j] = at(outs, j*p.T*h)
	}
	return out, true
}

func (p *ink2Prefill) control(cur, msg ink2Ref, c *ink2Ctrl, j int) ink2Ref {
	h := p.s.hp.Hidden
	g := p.tmp(p.T * c.rank)
	if p.s.err == nil {
		p.fail(p.pd.Ink2PCtrlDown(cur, msg, c.down[j], g, h, c.rank, p.T))
	}
	out := p.tmp(p.T * h)
	if p.s.err == nil {
		p.fail(p.pd.Ink2PCtrlApply(cur, msg, g, c.gain[j], c.forget[j], out, h, c.rank, p.T))
	}
	return out
}

func (p *ink2Prefill) conv(slot int, x, weight, dyn, resid ink2Ref, n, heads, offset int, keep bool) ink2Ref {
	s := p.s
	kn := s.hp.ConvKernel
	norm := float32(1 / math.Sqrt(float64(kn+1)))
	if dyn.T.Buf() != nil {
		norm = float32(1 / math.Sqrt(float64(2*kn+1)))
	}
	var out ink2Ref
	if keep {
		out = p.hold(p.T * n)
	} else {
		out = p.tmp(p.T * n)
	}
	if s.err == nil && p.cohort != nil {
		p.cohort.conv(p, slot, x, weight, dyn, resid, out, n, max(heads, 1), offset, norm)
	} else if s.err == nil && p.verify {
		L := kn - 1 + p.T
		p.fail(p.vd.Ink2PConvLog(x, weight, dyn, at(s.hist[slot], offset*(kn-1)), at(s.convLog[slot], offset*L), out, resid,
			n, max(heads, 1), kn, norm, p.T))
	} else if s.err == nil {
		p.fail(p.pd.Ink2PConv(x, weight, dyn, at(s.hist[slot], offset*(kn-1)), out, resid, n, max(heads, 1), kn, norm, p.T))
	}
	return out
}

// convKV runs a verify chunk's logged k and v short convs in one launch; each equals
// the conv call it replaces.
func (p *ink2Prefill) convKV(slot int, kx, vx ink2Ref, l *ink2Layer, kDyn, vDyn ink2Ref) (ink2Ref, ink2Ref) {
	s := p.s
	hp := s.hp
	kn := hp.ConvKernel
	kvDim := hp.KVHeads * hp.HeadDim
	L := kn - 1 + p.T
	k, v := p.tmp(p.T*kvDim), p.tmp(p.T*kvDim)
	if s.err == nil {
		p.fail(p.vd.Ink2PConvPairLog(
			[]ink2Ref{kx, vx}, []ink2Ref{l.kconvW, l.vconvW}, []ink2Ref{kDyn, vDyn},
			[]ink2Ref{s.hist[slot], at(s.hist[slot], kvDim*(kn-1))},
			[]ink2Ref{s.convLog[slot], at(s.convLog[slot], kvDim*L)},
			[]ink2Ref{k, v}, []float32{inkling2ConvNorm(kn, kDyn), inkling2ConvNorm(kn, vDyn)},
			kvDim, hp.KVHeads, kn, p.T))
	}
	return k, v
}

func (p *ink2Prefill) layer(slot, ip int, cur ink2Ref, blocks *[]ink2Ref, msgs []ink2Ref, c *ink2Ctrl) ink2Ref {
	s := p.s
	hp := s.hp
	l := &s.layers[ip]
	local := hp.LocalLayer[ip]
	h := hp.Hidden
	kvDim := hp.KVHeads * hp.HeadDim
	inpSA := cur

	in := [3]ink2Ref{cur, cur, cur}
	// without a controller between them the input norms fold into the mix
	normed := false
	if l.resQRows > 0 {
		var norms []ink2Ref
		if msgs == nil {
			normed = true
			norms = []ink2Ref{l.qNorm, l.kNorm, l.vNorm}
			if local {
				norms = []ink2Ref{l.attnNorm}
			}
		}
		if local {
			m := p.mixNorm(with(*blocks, cur), l.resQ, 1, norms)
			in = [3]ink2Ref{m[0], m[0], m[0]}
		} else {
			m := p.mixNorm(with(*blocks, cur), l.resQ, 3, norms)
			in = [3]ink2Ref{m[0], m[1], m[2]}
		}
	}
	if msgs != nil {
		if p.verify {
			o := p.controls(in[:], msgs[:3], c)
			in = [3]ink2Ref{o[0], o[1], o[2]}
		} else if o, ok := p.controlBatch(in[:], msgs[:3], c); ok {
			in = [3]ink2Ref{o[0], o[1], o[2]}
		} else {
			for j := 0; j < 3; j++ {
				in[j] = p.control(in[j], msgs[j], c, j)
			}
		}
	}
	switch {
	case normed:
	case local && msgs != nil && p.verify:
		o := p.rmsRows(in[0], l.attnNorm, 3*p.T, h, hp.NormEps)
		in = [3]ink2Ref{o, at(o, p.T*h), at(o, 2*p.T*h)}
	case local && msgs != nil:
		for j := 0; j < 3; j++ {
			in[j] = p.rms(in[j], l.attnNorm, h, hp.NormEps)
		}
	case local:
		n0 := p.rms(in[0], l.attnNorm, h, hp.NormEps)
		in = [3]ink2Ref{n0, n0, n0}
	default:
		in[0] = p.rms(in[0], l.qNorm, h, hp.NormEps)
		in[1] = p.rms(in[1], l.kNorm, h, hp.NormEps)
		in[2] = p.rms(in[2], l.vNorm, h, hp.NormEps)
	}

	// every product of the visit's attention inputs
	fm := []ink2Mat{l.wq, l.wk, l.wv, l.rel, l.sink, l.gate}
	fx := []ink2Ref{in[0], in[1], in[2], in[0], in[0], in[0]}
	if !local {
		fm = append(fm, l.kconv, l.vconv)
		fx = append(fx, in[1], in[2])
	}
	fy := p.mmMany(fm, fx)
	q, r, sinks, gate := fy[0], fy[3], fy[4], fy[5]
	var kDyn, vDyn ink2Ref
	if !local {
		kDyn, vDyn = fy[6], fy[7]
	}
	var k, v ink2Ref
	if p.verify {
		k, v = p.convKV(slot, fy[1], fy[2], l, kDyn, vDyn)
	} else {
		k = p.conv(slot, fy[1], l.kconvW, kDyn, ink2Ref{}, kvDim, hp.KVHeads, 0, false)
		v = p.conv(slot, fy[2], l.vconvW, vDyn, ink2Ref{}, kvDim, hp.KVHeads, kvDim, false)
	}

	ap := compute.Ink2AttnParams{
		Heads: hp.Heads, KVHeads: hp.KVHeads, HeadDim: hp.HeadDim, RelLen: hp.RelLength,
		QKClip: hp.QKClip, RelClip: hp.RelClip, Eps: hp.NormEps, Global: !local,
		Alpha: hp.LogScaleAlpha, Floor: hp.LogScaleFloor, Scale: float32(1 / math.Sqrt(float64(hp.HeadDim))),
	}
	cap := s.caps[slot]
	extent, window := hp.RelExtent, 0
	if local {
		extent, window = hp.SlidingWindow, hp.SlidingWindow
	}
	if s.err == nil && p.verify {
		p.fail(p.vd.Ink2PQKVPostDev(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], s.verifyPos, ap, cap, p.T))
	} else if s.err == nil && p.cohort == nil && s.kvNVFP4Slot(slot) {
		p.fail(p.nvfp4.Ink2PQKVPostNVFP4(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.kvOverflow, ap, cap, p.pos0, p.T))
	} else if s.err == nil && p.cohort == nil && s.kvFP8Slot(slot) {
		p.fail(p.fp8.Ink2PQKVPostFP8(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.kvOverflow, ap, cap, p.pos0, p.T))
	} else if s.err == nil && p.cohort == nil && s.kvHalfSlot(slot) {
		p.fail(p.half.Ink2PQKVPostHalf(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], ap, cap, p.pos0, p.T))
	} else if s.err == nil && p.cohort == nil {
		p.fail(p.pd.Ink2PQKVPost(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], ap, cap, p.pos0, p.T))
	}
	attn := p.tmp(p.T * hp.Heads * hp.HeadDim)
	if s.err == nil && p.cohort != nil {
		p.cohort.attend(p, slot, l, q, r, k, v, sinks, attn, ap, extent, window)
	} else if s.err == nil && p.verify {
		p.fail(p.vd.Ink2PAttendDev(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], attn, s.verifyPos, ap, extent, window, cap, p.T))
	} else if s.err == nil && s.kvNVFP4Slot(slot) {
		p.fail(p.nvfp4.Ink2PAttendNVFP4(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], attn, ap, extent, window, cap, p.pos0, p.T))
	} else if s.err == nil && s.kvFP8Slot(slot) {
		p.fail(p.fp8.Ink2PAttendFP8(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], attn, ap, extent, window, cap, p.pos0, p.T))
	} else if s.err == nil && s.kvHalfSlot(slot) {
		p.fail(p.half.Ink2PAttendHalf(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], attn, ap, extent, window, cap, p.pos0, p.T))
	} else if s.err == nil {
		p.fail(p.pd.Ink2PAttend(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], attn, ap, extent, window, cap, p.pos0, p.T))
	}
	o := p.mmGated(l.wo, attn, gate, hp.Heads*hp.HeadDim)
	if ip%hp.AttnResBlock == 0 {
		*blocks = append(*blocks, inpSA)
		cur = p.conv(slot, o, l.attnPostConv, ink2Ref{}, ink2Ref{}, h, 0, 2*kvDim, false)
	} else {
		cur = p.conv(slot, o, l.attnPostConv, ink2Ref{}, inpSA, h, 0, 2*kvDim, false)
	}
	ffnInp := cur

	var x ink2Ref
	if msgs != nil {
		x = p.mix(with(*blocks, cur), l.ffnResQ, 1)[0]
		if p.verify {
			x = p.controlsAt([]ink2Ref{x}, msgs[3:4], c, 3)[0]
		} else {
			x = p.control(x, msgs[3], c, 3)
		}
		x = p.rms(x, l.ffnNorm, h, hp.NormEps)
	} else {
		x = p.mixNorm(with(*blocks, cur), l.ffnResQ, 1, []ink2Ref{l.ffnNorm})[0]
	}
	down, wts := p.moeRows(l, x)
	if p.verify && s.err == nil {
		// the combine is folded into the logged post conv
		kn := hp.ConvKernel
		out := p.hold(p.T * h)
		offset := 2*kvDim + h
		p.fail(p.vd.Ink2PCombineConvLog(down, wts, hp.ExpertsUsed, l.ffnPostConv, at(s.hist[slot], offset*(kn-1)),
			at(s.convLog[slot], offset*(kn-1+p.T)), out, ffnInp, h, kn, inkling2ConvNorm(kn, ink2Ref{}), p.T))
		return out
	}
	x = p.tmp(p.T * h)
	if s.err == nil {
		p.fail(p.pd.Ink2PCombine(down, wts, x, hp.ExpertsUsed, h, p.T))
	}
	return p.conv(slot, x, l.ffnPostConv, ink2Ref{}, ffnInp, h, 0, 2*kvDim+h, true)
}

// ink2Bucket is the device bucketing of one layer's routed ids, shared by its
// gate/up and down expert GEMMs.
type ink2Bucket struct {
	nexp    int
	scratch ink2Ref
	filled  bool
}

// experts runs npairs routed pairs through e: the grouped GEMM when the dtype has
// one, else the decode expert GEMV over every pair in expert order.
func (p *ink2Prefill) experts(e *ink2Experts, ids, x ink2Ref, xDiv, xStride int, y ink2Ref, rows int, b *ink2Bucket) {
	s := p.s
	if s.err != nil {
		return
	}
	npairs := p.T * s.hp.ExpertsUsed
	nexp := e.stack.Shape[0] / rows
	if b.nexp != nexp {
		*b = ink2Bucket{nexp: nexp, scratch: p.tmp(compute.Ink2PGroupScratch(npairs, nexp))}
	}
	scratch := b.scratch
	if p.grouped {
		ok, err := p.pd.Ink2PExpertGrouped(e.stack, rows, nexp, ids, x, xDiv, xStride, y, npairs, e.scale, scratch, b.filled)
		if err != nil || ok {
			b.filled = err == nil
			p.fail(err)
			return
		}
	}
	b.filled = false
	ok, err := p.pd.Ink2PExpertSorted(e.stack, rows, nexp, ids, x, xDiv, xStride, y, npairs, e.scale, scratch,
		p.tmp(npairs), p.tmp(npairs*xStride), p.tmp(npairs*rows))
	if err == nil && !ok {
		err = fmt.Errorf("inkling2 prefill: expert dtype %s has no indexed kernel", e.stack.Dtype)
	}
	p.fail(err)
}

// moe is the MoE output for x: the routed pairs' down rows combined by their weights.
func (p *ink2Prefill) moe(l *ink2Layer, x ink2Ref) ink2Ref {
	down, wts := p.moeRows(l, x)
	out := p.tmp(p.T * p.s.hp.Hidden)
	if p.s.err == nil {
		p.fail(p.pd.Ink2PCombine(down, wts, out, p.s.hp.ExpertsUsed, p.s.hp.Hidden, p.T))
	}
	return out
}

// moeRows returns the routed pairs' down rows ([T*k][hidden]) and weights, before the
// combine.
func (p *ink2Prefill) moeRows(l *ink2Layer, x ink2Ref) (ink2Ref, ink2Ref) {
	s := p.s
	hp := s.hp
	k, width := hp.ExpertsUsed, hp.ExpertWidth
	logits := p.mm(l.router, x)
	ids, wts := p.tmp(p.T*k), p.tmp(p.T*k)
	if s.err == nil {
		p.fail(p.pd.Ink2PRoute(logits, l.probsBias, ids, wts, l.router.rows, k, hp.ExpertWeightsScale, p.T))
	}
	npairs := p.T * k
	gu := p.tmp(npairs * 2 * width)
	var bucket ink2Bucket
	// a verify chunk runs the decode expert GEMV straight over its T*k pairs: at a
	// few rows, bucketing pairs by expert costs more than it saves
	pairwise := (p.verify || p.fewRows) && !p.grouped
	switch {
	case pairwise && p.verifyExperts(&l.gateUp, ids, x, k, hp.Hidden, gu, 2*width, npairs):
	case pairwise:
		xr := p.tmp(npairs * hp.Hidden)
		if s.err == nil {
			p.fail(p.vd.Ink2VRepeatRows(x, xr, hp.Hidden, p.T, k))
		}
		p.pairExperts(&l.gateUp, ids, xr, hp.Hidden, gu, 2*width, npairs)
	default:
		p.experts(&l.gateUp, ids, x, k, hp.Hidden, gu, 2*width, &bucket)
	}
	padK := l.down.padK
	act := p.tmp(npairs * padK)
	switch {
	case s.err != nil:
	case p.verify:
		p.fail(s.dev.Ink2Situ(gu, act, npairs, width, padK, float64(hp.SituBeta), float64(hp.SituLinearBeta)))
	default:
		p.fail(p.pd.Ink2PSitu(gu, act, npairs, width, padK, float64(hp.SituBeta), float64(hp.SituLinearBeta)))
	}
	down := p.tmp(npairs * hp.Hidden)
	switch {
	case pairwise && p.verifyExperts(&l.down, ids, act, 1, padK, down, hp.Hidden, npairs):
	case pairwise:
		p.pairExperts(&l.down, ids, act, padK, down, hp.Hidden, npairs)
	default:
		p.experts(&l.down, ids, act, 1, padK, down, hp.Hidden, &bucket)
	}
	return down, wts
}

// verifyExperts runs npairs pairs through e reading each routed expert once for
// all its pairs, on a device whose grouped kernel keeps the decode GEMV's bits;
// false (nothing launched) sends the caller to the pairwise GEMV.
func (p *ink2Prefill) verifyExperts(e *ink2Experts, ids, x ink2Ref, xDiv, xStride int, y ink2Ref, rows, npairs int) bool {
	s := p.s
	ve, ok := s.be.(compute.Inkling2VerifyMoE)
	if !ok || s.err != nil {
		return s.err != nil
	}
	ok, err := ve.Ink2VExperts(e.stack, rows, ids, x, xDiv, xStride, y, npairs, e.scale)
	p.fail(err)
	return ok || err != nil
}

// pairExperts runs npairs (token, slot) pairs through e with the decode expert GEMV,
// pair i reading x row i (stride xStride) and writing y row i.
func (p *ink2Prefill) pairExperts(e *ink2Experts, ids, x ink2Ref, xStride int, y ink2Ref, rows, npairs int) {
	s := p.s
	if s.err != nil {
		return
	}
	ok, err := s.dev.Ink2ExpertMatVec(e.stack, rows, ids, x, xStride, y, npairs, e.scale, nil)
	if err == nil && !ok {
		err = fmt.Errorf("inkling2 verify: expert dtype %s has no indexed kernel", e.stack.Dtype)
	}
	p.fail(err)
}

// forward runs one chunk through every layer visit; on the last chunk it also
// enqueues the last token's head and argmax into s.argIdx.
func (p *ink2Prefill) forward(last bool) {
	s := p.s
	hp := s.hp
	h := hp.Hidden
	cur := p.hold(p.T * h)
	in := p.in
	if in.T.Buf() == nil {
		in = s.prefillIn
	}
	p.rmsInto(in, s.tokNorm, cur, h, hp.NormEps)
	// visit resets the layer scratch and, with a draft attached, keeps the visit's
	// input when it is a draft tap
	var taps []ink2Ref
	nv := 0
	visit := func() {
		p.scrOff = 0
		clear(p.q8)
		if s.draft != nil {
			for _, v := range s.draft.tapVisits() {
				if v == nv {
					taps = append(taps, cur)
				}
			}
		}
		nv++
	}
	var blocks []ink2Ref
	for il := 0; il < hp.CoreStart; il++ {
		visit()
		cur = p.layer(il, il, cur, &blocks, nil, nil)
	}
	prelude := cur
	preludeBlocks := append([]ink2Ref(nil), blocks...)
	core := hp.CoreEnd - hp.CoreStart
	for pass := 0; pass < hp.Passes; pass++ {
		var bank []ink2Ref
		var bankOut ink2Ref
		if pass > 0 {
			bank = append(bank, blocks[len(preludeBlocks)+1:]...)
			bankOut = cur
			cur = prelude
			blocks = append([]ink2Ref(nil), preludeBlocks...)
		}
		for ip := hp.CoreStart; ip < hp.CoreEnd; ip++ {
			visit()
			if pass == 0 {
				cur = p.layer(ip, ip, cur, &blocks, nil, nil)
				continue
			}
			slot := hp.Layers + (pass-1)*core + (ip - hp.CoreStart)
			c := &s.ctrls[ip-hp.CoreStart]
			msgs := p.mix(with(bank, bankOut), c.msgQ, 4)
			cur = p.layer(slot, ip, cur, &blocks, msgs, c)
		}
	}
	for il := hp.CoreEnd; il < hp.Layers; il++ {
		visit()
		cur = p.layer(il, il, cur, &blocks, nil, nil)
	}
	p.scrOff = 0
	if p.verify {
		final := p.headRows(blocks, cur)
		if s.draft != nil {
			s.draft.commitTaps(p, append(taps, final))
		}
		return
	}
	if s.draft != nil {
		x := p.mixNorm(with(blocks, cur), s.outResQ, 1, []ink2Ref{s.outNorm})[0]
		s.draft.commitTaps(p, append(taps, x))
	}
	if p.cohort != nil {
		p.cohort.head(p, blocks, cur)
		return
	}
	if !last {
		return
	}
	p.scrOff = 0
	// the head runs on the last token only: every state viewed at its last row
	lastRow := func(r ink2Ref) ink2Ref { return at(r, (p.T-1)*h) }
	states := make([]ink2Ref, 0, len(blocks)+1)
	for _, b := range blocks {
		states = append(states, lastRow(b))
	}
	states = append(states, lastRow(cur))
	x := p.tmp(h)
	if s.err == nil {
		p.fail(s.dev.Ink2Mix(states, s.outResQ, 1, nil, 0, x, h))
	}
	xn := p.tmp(h)
	if s.err == nil {
		p.fail(s.dev.Ink2RMS(x, s.outNorm, xn, 1, h, hp.NormEps))
	}
	logits := p.tmp(s.output.rows)
	s.logits = logits
	if s.err == nil {
		p.fail(s.dev.Ink2MatVec(s.output.t, xn, logits))
	}
	if s.err == nil {
		p.fail(s.dev.Ink2SoftcapArgmax(logits, hp.VocabUnpadded, hp.FinalLogitSoftcap, s.argIdx))
	}
}

// ensurePrefillArenas sizes the batched-forward arenas for chunks of up to T tokens.
// A longer chunk replaces them: the old arenas are freed rather than left beside the
// new ones, and every capture that addressed them is recaptured.
func (s *inkling2Resident) ensurePrefillArenas(T int) {
	if s.prefillKeep.T.Buf() != nil && T <= s.prefillCap {
		return
	}
	T = max(T, s.prefillCap)
	if s.prefillKeep.T.Buf() != nil {
		// captured verify graphs address the arenas being replaced
		s.dropCaptures()
		s.release(s.prefillKeep, s.prefillScr, s.prefillIn)
	}
	keep, scr := s.prefillArenas(T)
	s.prefillKeep, s.prefillScr = s.alloc(keep), s.alloc(scr)
	s.prefillIn = s.alloc(T * s.hp.Hidden)
	s.prefillCap = T
	if s.err != nil {
		s.prefillKeep, s.prefillScr, s.prefillIn, s.prefillCap = ink2Ref{}, ink2Ref{}, ink2Ref{}, 0
	}
}

// headRows runs the output head on every row of the chunk and leaves row t's greedy
// token at s.verifyIdx[4t].
func (p *ink2Prefill) headRows(blocks []ink2Ref, cur ink2Ref) ink2Ref {
	s := p.s
	hp := s.hp
	xn := p.mixNorm(with(blocks, cur), s.outResQ, 1, []ink2Ref{s.outNorm})[0]
	rows := s.output.rows
	if s.err == nil {
		ok, err := p.vd.Ink2VRows(s.output.t, xn, s.verifyLogits, p.T)
		if err == nil && !ok {
			err = p.pd.Ink2PMatMul(s.output.t, xn, s.verifyLogits, p.T)
		}
		p.fail(err)
	}
	for t := 0; t < p.T && s.err == nil; t++ {
		p.fail(s.dev.Ink2SoftcapArgmax(at(s.verifyLogits, t*rows), hp.VocabUnpadded, hp.FinalLogitSoftcap, at(s.verifyIdx, 4*t)))
	}
	return xn
}
