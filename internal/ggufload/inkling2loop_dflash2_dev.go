package ggufload

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// inkling2DFlash2Ring is the context ring of the device draft: a query sees
// window-1 earlier positions, and a verify writes up to a chunk of rejected
// rows past the committed position before they are overwritten.
const inkling2DFlash2Ring = 2048 + 2*inkling2SpecMaxDraft

// inkling2DFlash2Rows is the most rows one draft launch covers.
const inkling2DFlash2Rows = 8

type ink2DF2Layer struct {
	inNorm, postNorm, qNorm, kNorm                  ink2Ref
	attnBase, mlpBase                               compute.Tensor
	attnDyn, mlpDyn, wq, wk, wv, wo, gate, up, down compute.Tensor
	kc, vc                                          ink2Ref // context rings [inkling2DFlash2Ring][kvDim]
}

// ink2DFlash2Draft is the DFlash2 draft resident beside a target session, the
// device port of Inkling2DFlash2Draft. Every target forward writes its context
// K/V from the four target taps, one ring row per position, so rows of
// rejected drafts sit past the committed position and are overwritten. A
// proposal runs the block over the anchor and Block-1 masks and returns the
// selector's greedy chain.
type ink2DFlash2Draft struct {
	s      *inkling2Resident
	dd     compute.Inkling2DFlash2Device
	cfg    Inkling2DFlash2Config
	visits []int
	layers []ink2DF2Layer
	fc     compute.Tensor
	sel    compute.Tensor
	pred   compute.Tensor
	succ   compute.Tensor
	store  *inkling2DFlash2Store

	hiddenNorm, norm                                    ink2Ref
	x, n, dyn, conv, q, k, v, attn, o, g, u, down, proj ink2Ref
	logits, scratch, path, pos, anchor, ch, chn, ck, cv ink2Ref
	rows, mask                                          []float32
	eps                                                 float32
	// tapSink, when set, receives each target forward's taps ([tap][T*Hidden])
	// so a test can feed the host reference the same context.
	tapSink func(T int, taps [][]float32)
}

// attachDFlash2 binds dm on the session's device; every later target forward
// also writes the draft context. BF16 matrices are read in place from the
// checkpoint mapping where the device can, else uploaded once.
func (s *inkling2Resident) attachDFlash2(dm *Inkling2DFlash2Model) (*ink2DFlash2Draft, error) {
	dd, ok := s.be.(compute.Inkling2DFlash2Device)
	if !ok {
		return nil, fmt.Errorf("inkling2 dflash2: backend %s has no draft ops", s.be.Name())
	}
	c := dm.Config
	hp := s.hp
	visits := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	taps := c.DFlash.Taps
	if c.Hidden != hp.Hidden || len(taps) != 4 || taps[len(taps)-1] != visits || hp.VocabUnpadded <= c.DFlash.Mask ||
		hp.VocabUnpadded > c.Vocab || s.output.rows < hp.VocabUnpadded || c.DFlash.Block > inkling2DFlash2Rows {
		return nil, fmt.Errorf("inkling2 dflash2: draft %+v does not fit a %d-wide target of %d visits", c, hp.Hidden, visits)
	}
	d := &ink2DFlash2Draft{s: s, dd: dd, cfg: c, visits: append([]int(nil), taps[:len(taps)-1]...), eps: float32(c.Eps)}
	if len(dm.mapped) > 0 {
		// a mapping the device cannot alias leaves every matrix to the upload path
		d.store, _ = openInkling2DFlash2Store(dm.mapped)
	}
	bf := func(m *inkling2BF16Matrix) compute.Tensor { return d.bf16(m.raw, m.rows, m.cols) }
	base := func(v []float32) compute.Tensor {
		raw := make([]byte, 2*len(v))
		for i, f := range v {
			binary.LittleEndian.PutUint16(raw[2*i:], uint16(math.Float32bits(f)>>16))
		}
		return s.bf16Upload(raw, 2*inkling2DFlash2Taps, c.Hidden)
	}
	H, kvDim := c.Hidden, c.KVHeads*c.HeadDim
	for l := range dm.layers {
		L := &dm.layers[l]
		d.layers = append(d.layers, ink2DF2Layer{
			inNorm: s.upload(L.inputNorm), postNorm: s.upload(L.postNorm), qNorm: s.upload(L.qNorm), kNorm: s.upload(L.kNorm),
			attnBase: base(L.attnBase), mlpBase: base(L.mlpBase),
			attnDyn: bf(L.attnDyn), mlpDyn: bf(L.mlpDyn), wq: bf(L.wq), wk: bf(L.wk), wv: bf(L.wv), wo: bf(L.wo),
			gate: bf(L.gate), up: bf(L.up), down: bf(L.down),
			kc: s.alloc(inkling2DFlash2Ring * kvDim), vc: s.alloc(inkling2DFlash2Ring * kvDim),
		})
	}
	d.fc, d.sel, d.pred, d.succ = bf(dm.fc), bf(dm.selProj), bf(dm.predCB), bf(dm.succCB)
	d.hiddenNorm, d.norm = s.upload(dm.hiddenNorm), s.upload(dm.norm)
	B := c.DFlash.Block
	qDim := c.Heads * c.HeadDim
	d.x, d.n, d.conv, d.o, d.down = s.alloc(B*H), s.alloc(B*H), s.alloc(B*H), s.alloc(B*H), s.alloc(B*H)
	d.dyn = s.alloc(B * c.dynamicWidth())
	d.q, d.attn = s.alloc(B*qDim), s.alloc(B*qDim)
	d.k, d.v = s.alloc(B*kvDim), s.alloc(B*kvDim)
	d.g, d.u = s.alloc(B*c.Intermediate), s.alloc(B*c.Intermediate)
	d.proj = s.alloc(B * c.DFlash.Rank)
	d.logits = s.alloc((B - 1) * s.output.rows)
	d.scratch = s.alloc(metalgemm.DraftSelectScratch(B-1, c.DFlash.TopK))
	d.path, d.pos, d.anchor = s.alloc(B), s.alloc(4), s.alloc(4)
	d.ch, d.chn = s.alloc(inkling2DFlash2Rows*H), s.alloc(inkling2DFlash2Rows*H)
	d.ck, d.cv = s.alloc(inkling2DFlash2Rows*kvDim), s.alloc(inkling2DFlash2Rows*kvDim)
	d.mask = d.tokenRow(c.DFlash.Mask)
	d.rows = make([]float32, B*H)
	for r := 1; r < B; r++ {
		copy(d.rows[r*H:(r+1)*H], d.mask)
	}
	if s.err != nil {
		d.close()
		return nil, s.err
	}
	s.draft = d
	return d, nil
}

// bf16 is the device tensor of a BF16 payload: a view of the mapping, or an
// upload of its bytes.
func (d *ink2DFlash2Draft) bf16(raw []byte, rows, cols int) compute.Tensor {
	if t, ok := d.store.view(raw, rows, cols); ok {
		return t
	}
	return d.s.bf16Upload(raw, rows, cols)
}

// bf16Upload copies a BF16 payload into a device buffer typed as a BF16
// [rows, cols] tensor.
func (s *inkling2Resident) bf16Upload(raw []byte, rows, cols int) compute.Tensor {
	words := make([]float32, (len(raw)+3)/4)
	for i := range words {
		var w [4]byte
		copy(w[:], raw[4*i:min(len(raw), 4*i+4)])
		words[i] = math.Float32frombits(binary.LittleEndian.Uint32(w[:]))
	}
	r := s.upload(words)
	t := r.T
	t.Dtype, t.Shape = compute.BF16, []int{rows, cols}
	return t
}

// tokenRow is a token's target first-layer input row, as the host reference
// normalizes it.
func (d *ink2DFlash2Draft) tokenRow(tok int) []float32 {
	s := d.s
	e := make([]float32, s.hp.Hidden)
	s.w.TokEmbd.Row(tok, e)
	return inkling2DFlash2RMS(e, s.w.TokNorm, s.hp.NormEps)
}

func (d *ink2DFlash2Draft) tapVisits() []int { return d.visits }

// dropCaptures is a no-op: the DFlash2 draft runs eagerly and captures no graph.
func (d *ink2DFlash2Draft) dropCaptures() {}

func (d *ink2DFlash2Draft) run(err error) {
	if d.s.err == nil {
		d.s.fail(err)
	}
}

// commitTaps writes the context K/V of the T rows of a target forward from
// its four taps at positions *s.verifyPos + t, in launches of up to eight rows.
func (d *ink2DFlash2Draft) commitTaps(p *ink2Prefill, taps []ink2Ref) {
	s, c := d.s, &d.cfg
	if len(taps) != len(c.DFlash.Taps) {
		s.fail(fmt.Errorf("inkling2 dflash2: %d target taps, want %d", len(taps), len(c.DFlash.Taps)))
		return
	}
	H, kvDim := c.Hidden, c.KVHeads*c.HeadDim
	if d.tapSink != nil && s.err == nil {
		host := make([][]float32, len(taps))
		for i, t := range taps {
			v, err := s.dev.Ink2Read(t, p.T*H)
			s.fail(err)
			host[i] = v
		}
		if s.err == nil {
			d.tapSink(p.T, host)
		}
	}
	segs := make([]ink2Ref, len(taps))
	for g0 := 0; g0 < p.T && s.err == nil; g0 += inkling2DFlash2Rows {
		n := min(inkling2DFlash2Rows, p.T-g0)
		for i, t := range taps {
			segs[i] = at(t, g0*H)
		}
		d.run(d.dd.Ink2DFLinear(segs, d.fc, d.ch, n))
		d.run(s.dev.Ink2RMS(d.ch, d.hiddenNorm, d.chn, n, H, d.eps))
		for l := range d.layers {
			L := &d.layers[l]
			d.run(d.dd.Ink2DFLinear([]ink2Ref{d.chn}, L.wk, d.ck, n))
			d.run(d.dd.Ink2DFLinear([]ink2Ref{d.chn}, L.wv, d.cv, n))
			d.run(d.dd.Ink2DFQKRope(d.ck, L.kNorm, L.kc, s.verifyPos, n, c.KVHeads, c.HeadDim, inkling2DFlash2Ring, g0, true,
				d.eps, c.Rope.Theta))
			d.run(d.dd.Ink2DFRingStore(d.cv, L.vc, s.verifyPos, n, kvDim, inkling2DFlash2Ring, g0))
		}
	}
}

// forward enqueues the block forward from d.x at anchor position *d.pos and
// leaves the greedy chain of depth rows in d.path.
func (d *ink2DFlash2Draft) forward(depth int) {
	d.layersForward()
	d.head(depth)
	d.selectPath(depth)
}

// layersForward enqueues the draft layers and final norm from d.x into d.n.
func (d *ink2DFlash2Draft) layersForward() {
	s, c := d.s, &d.cfg
	B, H := c.DFlash.Block, c.Hidden
	G, kn := c.DFlash.Group, inkling2DFlash2Taps
	lin := func(x ink2Ref, w compute.Tensor, y ink2Ref, rows int) {
		d.run(d.dd.Ink2DFLinear([]ink2Ref{x}, w, y, rows))
	}
	for l := range d.layers {
		L := &d.layers[l]
		d.run(s.dev.Ink2RMS(d.x, L.inNorm, d.n, B, H, d.eps))
		lin(d.n, L.attnDyn, d.dyn, B)
		d.run(d.dd.Ink2DFConv(d.n, d.dyn, L.attnBase, ink2Ref{}, d.conv, B, H, G, kn, 0))
		lin(d.conv, L.wq, d.q, B)
		lin(d.conv, L.wk, d.k, B)
		lin(d.conv, L.wv, d.v, B)
		d.run(d.dd.Ink2DFQKRope(d.q, L.qNorm, d.q, d.pos, B, c.Heads, c.HeadDim, 0, 0, false, d.eps, c.Rope.Theta))
		d.run(d.dd.Ink2DFQKRope(d.k, L.kNorm, d.k, d.pos, B, c.KVHeads, c.HeadDim, 0, 0, false, d.eps, c.Rope.Theta))
		d.run(d.dd.Ink2DFAttend(d.q, d.k, d.v, L.kc, L.vc, d.attn, d.pos, B, c.Heads, c.KVHeads, c.HeadDim, c.Window,
			inkling2DFlash2Ring))
		lin(d.attn, L.wo, d.o, B)
		d.run(d.dd.Ink2DFConv(d.o, d.dyn, L.attnBase, d.x, d.x, B, H, G, kn, 1))
		d.run(s.dev.Ink2RMS(d.x, L.postNorm, d.n, B, H, d.eps))
		lin(d.n, L.mlpDyn, d.dyn, B)
		d.run(d.dd.Ink2DFConv(d.n, d.dyn, L.mlpBase, ink2Ref{}, d.conv, B, H, G, kn, 0))
		lin(d.conv, L.gate, d.g, B)
		lin(d.conv, L.up, d.u, B)
		d.run(d.dd.Ink2DFSiluMul(d.g, d.u, d.g, B*c.Intermediate))
		lin(d.g, L.down, d.down, B)
		d.run(d.dd.Ink2DFConv(d.down, d.dyn, L.mlpBase, d.x, d.x, B, H, G, kn, 1))
	}
	d.run(s.dev.Ink2RMS(d.x, d.norm, d.n, B, H, d.eps))
}

// inkling2DFlash2RowsHead sends the draft head through the verify rows
// kernel, the A side of a timing comparison.
var inkling2DFlash2RowsHead bool

// head enqueues the target LM head over the depth proposal rows of d.n.
func (d *ink2DFlash2Draft) head(depth int) {
	s := d.s
	fin := at(d.n, d.cfg.Hidden)
	if !inkling2DFlash2RowsHead {
		if ok, err := d.dd.Ink2DFHead(s.output.t, fin, d.logits, depth); ok || err != nil {
			d.run(err)
			return
		}
	}
	vd := s.be.(compute.Inkling2VerifyDevice)
	if ok, err := vd.Ink2VRows(s.output.t, fin, d.logits, depth); err != nil || !ok {
		d.run(err)
		d.run(s.be.(compute.Inkling2PrefillDevice).Ink2PMatMul(s.output.t, fin, d.logits, depth))
	}
}

// selectPath enqueues the selector projection and greedy chain into d.path.
func (d *ink2DFlash2Draft) selectPath(depth int) {
	s, c := d.s, &d.cfg
	d.run(d.dd.Ink2DFLinear([]ink2Ref{at(d.n, c.Hidden)}, d.sel, d.proj, depth))
	d.run(d.dd.Ink2DFSelect(d.logits, depth, s.output.rows, s.hp.VocabUnpadded, c.DFlash.TopK, d.proj, d.pred, d.succ,
		d.anchor, float32(c.DFlash.Softcap), d.scratch, d.path, ink2Ref{}))
}

// propose drafts up to k tokens after anchor, the token at position pos whose
// context (positions < pos) is already committed.
func (d *ink2DFlash2Draft) propose(anchor, pos, k int) ([]int, error) {
	s := d.s
	depth := min(k, d.cfg.DFlash.Block-1)
	if depth < 1 {
		return nil, nil
	}
	copy(d.rows, d.tokenRow(anchor))
	s.fail(s.dev.Ink2Write(d.x, d.rows))
	s.fail(s.dev.Ink2Write(d.pos, []float32{intBits(pos)}))
	s.fail(s.dev.Ink2Write(d.anchor, []float32{intBits(anchor)}))
	if s.err == nil {
		d.forward(depth)
	}
	if s.err != nil {
		return nil, s.err
	}
	got, err := s.dev.Ink2ReadInts(d.path, depth)
	if err != nil {
		return nil, err
	}
	out := make([]int, depth)
	for i := range out {
		out[i] = int(got[i])
	}
	return out, nil
}

// close releases the in-place weight views once the device has drained; the
// session frees the uploaded buffers.
func (d *ink2DFlash2Draft) close() {
	if d == nil || d.store == nil {
		return
	}
	if t, err := d.s.dev.Ink2Alloc(1); err == nil {
		_, _ = d.s.dev.Ink2Read(compute.Ink2At(t, 0), 1)
		d.s.be.Free(t)
	}
	d.store.Close()
	d.store = nil
}

// inkling2DFlash2Drafter adapts the device draft to the speculative loop.
type inkling2DFlash2Drafter struct{ d *ink2DFlash2Draft }

func (a inkling2DFlash2Drafter) Draft(history []int, k int) []int {
	props, err := a.d.propose(history[len(history)-1], a.d.s.pos, k)
	if err != nil {
		a.d.s.fail(err)
		return nil
	}
	return props
}

func (inkling2DFlash2Drafter) Observe(*inkling2Resident, []int, int) error { return nil }
