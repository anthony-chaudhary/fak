//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
	"sync"
)

//go:embed ink2_prefill.metal
var ink2PrefillSource string

//go:embed ink2_prefill_attend.metal
var ink2PrefillAttendSource string

//go:embed ink2_prefill_packed.metal
var ink2PrefillPacked string

//go:embed ink2_prefill_gemm.metal
var ink2PrefillGemm string

//go:embed ink2_prefill_flash.metal
var ink2PrefillFlashSource string

const (
	// The register tile of the batched weight kernels: weight rows a lane
	// carries, and the activation rows, or routed pairs, that share one decode
	// of them. Measured on an M3 Pro: more of either spills the tile.
	ink2PrefillRowsR     = 2
	ink2PrefillRowsTile  = 8
	ink2PrefillGroupTile = 8
	// The conv kernel keeps a channel's history in a register array of this.
	ink2PrefillMaxKernel = 8
	// The K step and threadgroup (four SIMD groups) of the GEMM kernels.
	// Measured on an M3 Pro: a K step of 64, or two staged slabs, cost more
	// occupancy than their fewer barriers win.
	ink2GemmBK      = 32
	ink2GemmThreads = 128
)

// A GEMM tile: SIMD groups split gm x gn, each sm x sn 8x8 accumulators.
type ink2GemmTile struct{ gm, gn, sm, sn int }

func (t ink2GemmTile) bm() int { return 8 * t.gm * t.sm }
func (t ink2GemmTile) bn() int { return 8 * t.gn * t.sn }

// The tiles of the rows kernel and the expert-grouped kernel: each SIMD group
// owns 8 rows by all 64 columns, so one past an expert run's pairs idles
// whole. Measured on an M3 Pro: 64-row or 128-column tiles, or 2 x 2 splits,
// are slower; expert runs average a tile of pairs.
var (
	ink2GemmRows    = ink2GemmTile{4, 1, 1, 8}
	ink2GemmGrouped = ink2GemmTile{4, 1, 1, 8}
)

// The batched weight kernels append to the packed decoders they instantiate.
var ink2PrefillPackedSource = sync.OnceValue(func() string {
	return packedCodebooks + packedDequant + packedGEMV +
		fmt.Sprintf("#define INK2P_ROWS_R %d\n#define INK2P_ROWS_TILE %d\n#define INK2P_GROUP_TILE %d\n", ink2PrefillRowsR, ink2PrefillRowsTile, ink2PrefillGroupTile) +
		ink2PrefillPacked +
		fmt.Sprintf("#define INK2G_BK %d\n#define INK2G_R_GM %d\n#define INK2G_R_GN %d\n#define INK2G_R_SM %d\n#define INK2G_R_SN %d\n#define INK2G_G_GM %d\n#define INK2G_G_GN %d\n#define INK2G_G_SM %d\n#define INK2G_G_SN %d\n",
			ink2GemmBK, ink2GemmRows.gm, ink2GemmRows.gn, ink2GemmRows.sm, ink2GemmRows.sn, ink2GemmGrouped.gm, ink2GemmGrouped.gn, ink2GemmGrouped.sm, ink2GemmGrouped.sn) +
		ink2PrefillGemm
})

var ink2RowsKernel, ink2GroupedKernel = ink2PackedKernels("rows_"), ink2PackedKernels("grouped_")

var ink2GemmRowsKernel, ink2GemmGroupedKernel = ink2PackedKernels("gemm_rows_"), ink2PackedKernels("gemm_grouped_")

// ink2VerifyRowsKernel names the verify rows kernels over the same packed decoders.
var ink2VerifyRowsKernel = ink2PackedKernels("ink2v_rows_")

type ink2PMixShape struct {
	n, states, tokens uint32
	invSqrtN          float32
}

type ink2PCtrlShape struct{ n, rank, tokens uint32 }

type ink2PConvShape struct {
	n, perHead, kn, tokens uint32
	dynamic, residual      uint32
	norm                   float32
}

type ink2PQKVShape struct {
	heads, kvHeads, headDim, relLen uint32
	cap, lanes                      uint32
	global, pos0                    uint32
	qkClip, relClip, eps            float32
	alpha, floorTok                 float32
}

type ink2PAttendShape struct {
	heads, perKV, headDim, relLen uint32
	kvDim, extent, window, cap    uint32
	threads, width                uint32
	span, tok0, pos0              uint32
	scale                         float32
}

type ink2PRouteShape struct {
	experts, slots, biased uint32
	weightScale            float32
}

type ink2PCombineShape struct{ slots, n, tokens uint32 }

type ink2PBucketShape struct{ npairs, nexp, threads uint32 }

type ink2PackedRows struct{ tokens, xStride, yStride uint32 }

type ink2PackedGroup struct{ npairs, xDiv, xStride, scaled uint32 }

func (d *Ink2Device) prefillKernel(name string, reducing bool) (Ink2Pipeline, error) {
	p, err := d.PipelinePrecise(ink2PrefillSource, name)
	if err != nil {
		return Ink2Pipeline{}, err
	}
	if reducing && (p.ThreadWidth() != 32 || p.MaxThreads() < ink2OpsThreads) {
		return Ink2Pipeline{}, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of %d threads", name, ink2OpsThreads)
	}
	return p, nil
}

// packedBatch is packed for the kernels of ink2_prefill_packed.metal.
func (d *Ink2Device) packedBatch(w Ink2Weight, kernels *[41]string, x Ink2Ref, xStride int) (Ink2Pipeline, int, error) {
	if !w.Buf.Valid() || w.ByteOff < 0 || w.Rows < 1 || w.Rows > math.MaxInt32 || w.Cols < 1 || w.Cols > math.MaxInt32 ||
		w.Experts < 1 || int(w.Type) >= len(kernels) || kernels[w.Type] == "" {
		return Ink2Pipeline{}, 0, fmt.Errorf("%w: weight type %d [%d %d] x%d", ErrInk2Shape, w.Type, w.Rows, w.Cols, w.Experts)
	}
	block, blockBytes, err := packedBlock(w.Type)
	if err != nil {
		return Ink2Pipeline{}, 0, err
	}
	// The block kernels load the activation as float4 vectors.
	if w.Cols%block != 0 || (block > 1 && (x.Off%4 != 0 || xStride%4 != 0)) {
		return Ink2Pipeline{}, 0, fmt.Errorf("%w: type %d weight of %d columns with the activation at offset %d, stride %d",
			ErrInk2Shape, w.Type, w.Cols, x.Off, xStride)
	}
	p, err := d.Pipeline(ink2PrefillPackedSource(), kernels[w.Type])
	if err != nil {
		return Ink2Pipeline{}, 0, err
	}
	if p.width != 32 || p.maxThreads < 128 {
		return Ink2Pipeline{}, 0, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of 128 threads", kernels[w.Type])
	}
	return p, w.Rows * (w.Cols / block * blockBytes), nil
}

// RowsMatMul encodes y[t] = W x[t] for rows activation rows, row t at
// x + t*xStride and its output at y + t*yStride. Each row equals MatVec of it.
func (d *Ink2Device) RowsMatMul(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) error {
	if !x.Valid() || !y.Valid() || rows < 1 || xStride < w.Cols || yStride < w.Rows ||
		rows > math.MaxInt32/max(xStride, yStride, 1) {
		return fmt.Errorf("%w: rows matmul of %d rows, strides %d and %d", ErrInk2Shape, rows, xStride, yStride)
	}
	p, bytes, err := d.packedBatch(w, &ink2RowsKernel, x, xStride)
	if err != nil {
		return err
	}
	args := append(w.args(x.Span((rows-1)*xStride+w.Cols), y.Span((rows-1)*yStride+w.Rows), bytes),
		Ink2Const(ink2PackedRows{uint32(rows), uint32(xStride), uint32(yStride)}))
	return d.Dispatch(p, [3]int{(w.Rows + 4*ink2PrefillRowsR - 1) / (4 * ink2PrefillRowsR), (rows + ink2PrefillRowsTile - 1) / ink2PrefillRowsTile, 1}, [3]int{128, 1, 1}, args...)
}

// gemmBatch is packedBatch for the GEMM kernels, which also load the
// activation as float4 vectors for the scalar types; ok is false for a shape
// they do not tile (columns not a multiple of ink2GemmBK).
func (d *Ink2Device) gemmBatch(w Ink2Weight, kernels *[41]string, x Ink2Ref, xStride int) (Ink2Pipeline, int, bool, error) {
	if w.Cols%ink2GemmBK != 0 || x.Off%4 != 0 || xStride%4 != 0 {
		return Ink2Pipeline{}, 0, false, nil
	}
	p, bytes, err := d.packedBatch(w, kernels, x, xStride)
	if err == nil && p.maxThreads < ink2GemmThreads {
		return p, bytes, false, fmt.Errorf("metalgemm: inkling2 kernel %s needs threadgroups of %d threads", kernels[w.Type], ink2GemmThreads)
	}
	return p, bytes, err == nil, err
}

// RowsGEMM is RowsMatMul on the tiled GEMM kernels: the same products, summed
// in another order than MatVec's. ok is false (nothing encoded) for a shape
// the kernels do not tile.
func (d *Ink2Device) RowsGEMM(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) (bool, error) {
	if !x.Valid() || !y.Valid() || rows < 1 || xStride < w.Cols || yStride < w.Rows ||
		rows > math.MaxInt32/max(xStride, yStride, 1) {
		return false, fmt.Errorf("%w: rows GEMM of %d rows, strides %d and %d", ErrInk2Shape, rows, xStride, yStride)
	}
	p, bytes, ok, err := d.gemmBatch(w, &ink2GemmRowsKernel, x, xStride)
	if err != nil || !ok {
		return false, err
	}
	args := append(w.args(x.Span((rows-1)*xStride+w.Cols), y.Span((rows-1)*yStride+w.Rows), bytes),
		Ink2Const(ink2PackedRows{uint32(rows), uint32(xStride), uint32(yStride)}))
	return true, d.Dispatch(p, [3]int{(w.Rows + ink2GemmRows.bn() - 1) / ink2GemmRows.bn(), (rows + ink2GemmRows.bm() - 1) / ink2GemmRows.bm(), 1}, [3]int{ink2GemmThreads, 1, 1}, args...)
}

// PExpertGEMM is PExpertGrouped on the tiled GEMM kernels, which also read
// the offsets ([nexp+1]) PBucket left beside order; the sums run in another
// order than ExpertMatVec's. ok is false (nothing encoded) for a shape the
// kernels do not tile.
func (d *Ink2Device) PExpertGEMM(w Ink2Weight, ids, order, offsets, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale Ink2Ref) (bool, error) {
	if !ids.Valid() || !order.Valid() || !offsets.Valid() || !x.Valid() || !y.Valid() || npairs < 1 || xDiv < 1 || xStride < w.Cols ||
		npairs > math.MaxInt32/max(xStride, w.Rows, 1) {
		return false, fmt.Errorf("%w: grouped expert GEMM of %d pairs, input stride %d", ErrInk2Shape, npairs, xStride)
	}
	p, bytes, ok, err := d.gemmBatch(w, &ink2GemmGroupedKernel, x, xStride)
	if err != nil || !ok {
		return false, err
	}
	rows := (npairs-1)/xDiv + 1
	args := append(w.args(x.Span((rows-1)*xStride+w.Cols), y.Span(npairs*w.Rows), w.Experts*bytes),
		ids.Span(npairs), order.Span(npairs), scale.Span(w.Experts),
		Ink2Const(ink2PackedGroup{uint32(npairs), uint32(xDiv), uint32(xStride), uint32(btoi(scale.Valid()))}),
		offsets.Span(w.Experts+1), Ink2Const(uint32(w.Experts)))
	// Every expert run takes at most one partial tile.
	tiles := (npairs+ink2GemmGrouped.bm()-1)/ink2GemmGrouped.bm() + w.Experts
	return true, d.Dispatch(p, [3]int{(w.Rows + ink2GemmGrouped.bn() - 1) / ink2GemmGrouped.bn(), tiles, 1}, [3]int{ink2GemmThreads, 1, 1}, args...)
}

// PMix encodes nq depth mixes of every token row: states are [T][n], queries
// [nq][n], and out is query-major [nq][T][n]. Row t of query q equals Mix of
// the states' rows t. out must not overlap a state.
func (d *Ink2Device) PMix(states []Ink2Ref, queries Ink2Ref, nq int, out Ink2Ref, n, T int) error {
	ns := len(states)
	if ns < 1 || ns > ink2OpsMaxStates || nq < 1 || nq > ink2OpsMaxRows || n < 1 || n > ink2OpsMaxWidth ||
		T < 1 || T > math.MaxInt32/(nq*n) || !queries.Valid() || !out.Valid() || ink2OpsOverlap(queries, nq*n, out, nq*T*n) {
		return fmt.Errorf("%w: prefill mix of %d states by %d queries, %d rows of %d", ErrInk2Shape, ns, nq, T, n)
	}
	p, err := d.prefillKernel("ink2p_mix", true)
	if err != nil {
		return err
	}
	args := make([]Ink2Arg, 0, ink2OpsMaxStates+3)
	for i := range ink2OpsMaxStates {
		var s Ink2Ref
		if i < ns {
			if s = states[i]; !s.Valid() || ink2OpsOverlap(s, T*n, out, nq*T*n) {
				return fmt.Errorf("%w: prefill mix state %d is absent or overlaps the output", ErrInk2Shape, i)
			}
		}
		args = append(args, s.Span(T*n))
	}
	args = append(args, queries.Span(nq*n), out.Span(nq*T*n),
		Ink2Const(ink2PMixShape{n: uint32(n), states: uint32(ns), tokens: uint32(T), invSqrtN: float32(1 / math.Sqrt(float64(n)))}))
	return d.Dispatch(p, [3]int{nq, T, 1}, [3]int{ink2OpsThreads, 1, 1}, args...)
}

func ink2PBind(args []Ink2Arg, refs []Ink2Ref, span int) []Ink2Arg {
	for _, r := range ink2OpsSlots(refs) {
		args = append(args, r.Span(span))
	}
	return args
}

func ink2PCtrlCheck(what string, n, rank, T int, sets ...[]Ink2Ref) error {
	nj := len(sets[0])
	bad := nj < 1 || nj > ink2OpsMaxRows || n < 1 || n > ink2OpsMaxWidth || rank < 1 || rank > math.MaxInt32/(2*n) ||
		T < 1 || T > math.MaxInt32/max(n, rank)
	for _, set := range sets {
		bad = bad || len(set) != nj
		for _, r := range set {
			bad = bad || !r.Valid()
		}
	}
	if bad {
		return fmt.Errorf("%w: prefill %s of %d controllers, %d rows of %d, rank %d", ErrInk2Shape, what, nj, T, n, rank)
	}
	return nil
}

// PCtrlDown encodes g[j][t] = Down_j [rms(cur_j[t]), rms(msg_j[t])] for
// len(cur) controllers over T rows; cur[j] and msg[j] are [T][n], down[j] is
// [rank][2n] and g[j] is [T][rank].
func (d *Ink2Device) PCtrlDown(cur, msg, down, g []Ink2Ref, n, rank, T int) error {
	if err := ink2PCtrlCheck("ctrl down", n, rank, T, cur, msg, down, g); err != nil {
		return err
	}
	p, err := d.prefillKernel("ink2p_ctrl_down", true)
	if err != nil {
		return err
	}
	args := ink2PBind(ink2PBind(ink2PBind(ink2PBind(make([]Ink2Arg, 0, 17), cur, T*n), msg, T*n), down, rank*2*n), g, T*rank)
	args = append(args, Ink2Const(ink2PCtrlShape{uint32(n), uint32(rank), uint32(T)}))
	return d.Dispatch(p, [3]int{rank, T, len(cur)}, [3]int{ink2OpsThreads, 1, 1}, args...)
}

// PCtrlApply encodes out[j][t] = cur - tanh(F_j g)*cur + tanh(G_j g)*msg with
// g[j] from PCtrlDown; gain[j] and forget[j] are [n][rank], out[j] is [T][n].
func (d *Ink2Device) PCtrlApply(cur, msg, g, gain, forget, out []Ink2Ref, n, rank, T int) error {
	if err := ink2PCtrlCheck("ctrl apply", n, rank, T, cur, msg, g, gain, forget, out); err != nil {
		return err
	}
	p, err := d.prefillKernel("ink2p_ctrl_apply", false)
	if err != nil {
		return err
	}
	args := ink2PBind(ink2PBind(make([]Ink2Arg, 0, 25), cur, T*n), msg, T*n)
	args = ink2PBind(ink2PBind(args, gain, n*rank), forget, n*rank)
	args = ink2PBind(ink2PBind(args, g, T*rank), out, T*n)
	args = append(args, Ink2Const(ink2PCtrlShape{uint32(n), uint32(rank), uint32(T)}))
	return d.DispatchThreads(p, [3]int{n, T, len(cur)}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1}, args...)
}

type ink2PCtrlRows struct{ n, tokens uint32 }

// CtrlGEMMTiles reports whether PCtrlDown's and PCtrlApply's products of
// width n and rank tile into RowsGEMM.
func CtrlGEMMTiles(n, rank int) bool { return n%ink2GemmBK == 0 && rank%ink2GemmBK == 0 }

// PCtrlDownGEMM is PCtrlDown of one controller as a GEMM over the normalized
// inputs, which take xs ([T][2n]) as scratch.
func (d *Ink2Device) PCtrlDownGEMM(cur, msg, down, g, xs Ink2Ref, n, rank, T int) error {
	if err := ink2PCtrlCheck("ctrl down", n, rank, T, []Ink2Ref{cur}, []Ink2Ref{msg}, []Ink2Ref{down}, []Ink2Ref{g}, []Ink2Ref{xs}); err != nil {
		return err
	}
	if !CtrlGEMMTiles(n, rank) {
		return fmt.Errorf("%w: ctrl GEMM of width %d, rank %d", ErrInk2Shape, n, rank)
	}
	p, err := d.prefillKernel("ink2p_ctrl_norm", true)
	if err != nil {
		return err
	}
	if err := d.Dispatch(p, [3]int{T, 1, 1}, [3]int{ink2OpsThreads, 1, 1}, cur.Span(T*n), msg.Span(T*n), xs.Span(2*T*n),
		Ink2Const(ink2PCtrlRows{uint32(n), uint32(T)})); err != nil {
		return err
	}
	_, err = d.RowsGEMM(Ink2DenseF32(down, rank, 2*n), xs, 2*n, g, rank, T)
	return err
}

// PCtrlApplyGEMM is PCtrlApply of one controller as two GEMMs, whose outputs
// take gs ([2][T][n]) as scratch.
func (d *Ink2Device) PCtrlApplyGEMM(cur, msg, g, gain, forget, out, gs Ink2Ref, n, rank, T int) error {
	if err := ink2PCtrlCheck("ctrl apply", n, rank, T, []Ink2Ref{cur}, []Ink2Ref{msg}, []Ink2Ref{g}, []Ink2Ref{gain},
		[]Ink2Ref{forget}, []Ink2Ref{out}, []Ink2Ref{gs}); err != nil {
		return err
	}
	if !CtrlGEMMTiles(n, rank) {
		return fmt.Errorf("%w: ctrl GEMM of width %d, rank %d", ErrInk2Shape, n, rank)
	}
	fs := gs.At(T * n)
	if _, err := d.RowsGEMM(Ink2DenseF32(gain, n, rank), g, rank, gs, n, T); err != nil {
		return err
	}
	if _, err := d.RowsGEMM(Ink2DenseF32(forget, n, rank), g, rank, fs, n, T); err != nil {
		return err
	}
	p, err := d.prefillKernel("ink2p_ctrl_gate", false)
	if err != nil {
		return err
	}
	return d.DispatchThreads(p, [3]int{n, T, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		cur.Span(T*n), msg.Span(T*n), gs.Span(T*n), fs.Span(T*n), out.Span(T*n), Ink2Const(ink2PCtrlRows{uint32(n), uint32(T)}))
}

// PConv encodes the causal short conv of n channels along T token rows, each
// row as Conv of it in token order, and leaves the final history in hist. dyn
// is [T][heads][kn]. out must not overlap x or resid.
func (d *Ink2Device) PConv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	if !x.Valid() || !w.Valid() || !hist.Valid() || !out.Valid() || n < 1 || kn < 2 || kn > ink2PrefillMaxKernel || heads < 1 ||
		T < 1 || T > math.MaxInt32/n || n > math.MaxInt32/kn || (dyn.Valid() && n%heads != 0) ||
		ink2OpsOverlap(x, T*n, out, T*n) || ink2OpsOverlap(resid, T*n, out, T*n) {
		return fmt.Errorf("%w: prefill conv of %d rows of %d channels, %d heads, kernel %d", ErrInk2Shape, T, n, heads, kn)
	}
	p, err := d.prefillKernel("ink2p_conv", false)
	if err != nil {
		return err
	}
	perHead, taps := n, 0
	if dyn.Valid() {
		perHead, taps = n/heads, T*heads*kn
	}
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		x.Span(T*n), w.Span(n*kn), dyn.Span(taps), hist.Span(n*(kn-1)), out.Span(T*n), resid.Span(T*n),
		Ink2Const(ink2PConvShape{n: uint32(n), perHead: uint32(perHead), kn: uint32(kn), tokens: uint32(T),
			dynamic: uint32(btoi(dyn.Valid())), residual: uint32(btoi(resid.Valid())), norm: norm}))
}

// PQKVPost encodes the attention prep of T token rows at positions pos0+t:
// QKVPost of each row without its folded convs, appending k and v into the
// cache rows of those positions. T must not exceed cap.
func (d *Ink2Device) PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	const limit = math.MaxInt32
	if p.Heads < 1 || p.KVHeads < 1 || p.HeadDim < 1 || p.RelLen < 0 || cap < 1 || cap > limit || T < 1 || T > cap ||
		pos0 < 0 || pos0 > limit-T || p.Heads > limit/p.HeadDim/T || p.Heads > limit/max(p.RelLen, 1)/T ||
		cap > limit/p.KVHeads/p.HeadDim || p.Heads > limit-2*p.KVHeads {
		return fmt.Errorf("%w: prefill qkv post of %d rows at %d, %d heads, %d kv heads, head dim %d, cache of %d",
			ErrInk2Shape, T, pos0, p.Heads, p.KVHeads, p.HeadDim, cap)
	}
	ok := p.RelLen == 0 || (r.Valid() && rclip.Valid())
	for _, ref := range []Ink2Ref{q, k, v, qclip, kcache, vcache} {
		ok = ok && ref.Valid()
	}
	if !ok {
		return fmt.Errorf("%w: prefill qkv post without an operand", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		r, rclip = Ink2Ref{}, Ink2Ref{}
	}
	pipe, err := d.prefillKernel("ink2p_qkv_post", false)
	if err != nil {
		return err
	}
	// Each head reduces across one SIMD group, so that is the threadgroup.
	lanes := pipe.ThreadWidth()
	if lanes < 1 || lanes > pipe.MaxThreads() {
		return fmt.Errorf("metalgemm: inkling2 prefill qkv post: SIMD groups of %d threads in threadgroups of at most %d", lanes, pipe.MaxThreads())
	}
	kvDim := p.KVHeads * p.HeadDim
	return d.Dispatch(pipe, [3]int{p.Heads + 2*p.KVHeads, T, 1}, [3]int{lanes, 1, 1},
		q.Span(T*p.Heads*p.HeadDim), r.Span(T*p.Heads*p.RelLen), k.Span(T*kvDim), v.Span(T*kvDim),
		qclip.Span(p.Heads*p.HeadDim), rclip.Span(p.Heads*p.RelLen), kcache.Span(cap*kvDim), vcache.Span(cap*kvDim),
		Ink2Const(ink2PQKVShape{
			heads: uint32(p.Heads), kvHeads: uint32(p.KVHeads), headDim: uint32(p.HeadDim), relLen: uint32(p.RelLen),
			cap: uint32(cap), lanes: uint32(lanes), global: uint32(btoi(p.Global)), pos0: uint32(pos0),
			qkClip: p.QKClip, relClip: p.RelClip, eps: p.Eps, alpha: p.Alpha, floorTok: p.Floor,
		}))
}

// PAttendSpan is the scores one (token, head) of a PAttend chunk needs: the
// most keys any of its queries sees.
func PAttendSpan(window, cap, pos0, T int) int {
	span := min(cap, pos0+T)
	if window > 0 {
		span = min(span, window)
	}
	return span
}

// PAttend encodes the attention of T queries at positions pos0+t over one KV
// slot into out ([T][heads][headDim]); q, r and sinks are [T] rows. Row t
// equals Attend at that position, over the keys PQKVPost appended for the
// whole chunk. scores is scratch of scoreFloats floats, at least
// heads*PAttendSpan: the chunk runs in as many token ranges as it takes to
// fit them.
func (d *Ink2Device) PAttend(q, r, sinks, relBias, kcache, vcache, scores Ink2Ref, scoreFloats int, out Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	const limit = math.MaxInt32
	if p.Heads < 1 || p.KVHeads < 1 || p.Heads%p.KVHeads != 0 || p.HeadDim < 1 || p.RelLen < 0 || extent < 0 || window < 0 ||
		cap < 1 || T < 1 || pos0 < 0 || pos0 > limit-T || p.Heads > limit/p.HeadDim/T || extent > limit || window > limit {
		return fmt.Errorf("%w: prefill attend of %d rows at %d, %d heads over %d kv heads of %d dims, extent %d, window %d, cap %d",
			ErrInk2Shape, T, pos0, p.Heads, p.KVHeads, p.HeadDim, extent, window, cap)
	}
	kvDim := p.KVHeads * p.HeadDim
	// The kernel indexes the caches and the bias table with 32-bit offsets.
	if cap > limit/p.Heads || cap > limit/kvDim ||
		(p.RelLen > 0 && (p.Heads > limit/p.RelLen/T || extent > limit/p.RelLen)) {
		return fmt.Errorf("%w: prefill attend operands exceed 32-bit offsets", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		extent = 0
	}
	if !q.Valid() || !sinks.Valid() || !kcache.Valid() || !vcache.Valid() || !scores.Valid() || !out.Valid() ||
		(extent > 0 && (!r.Valid() || !relBias.Valid())) {
		return fmt.Errorf("%w: prefill attend without an operand", ErrInk2Shape)
	}
	span := PAttendSpan(window, cap, pos0, T)
	perLaunch := min(T, scoreFloats/(p.Heads*span))
	if perLaunch < 1 {
		return fmt.Errorf("%w: prefill attend scores scratch of %d floats for %d heads of %d keys", ErrInk2Shape, scoreFloats, p.Heads, span)
	}
	pipe, err := d.PipelinePrecise(ink2PrefillAttendSource, "ink2p_attend")
	if err != nil {
		return err
	}
	width := pipe.ThreadWidth()
	if width < 1 {
		return fmt.Errorf("metalgemm: inkling2 prefill attend kernel reports no SIMD width")
	}
	threads := min(pipe.MaxThreads(), ink2AttendMaxThreads, ink2AttendMaxSIMD*width) / width * width
	if quads := (p.HeadDim + 3) / 4; quads > threads || quads > ink2AttendQTerms*width {
		return fmt.Errorf("%w: prefill attend head of %d dims on %d threads of width %d", ErrInk2Shape, p.HeadDim, threads, width)
	}
	rSpan, relSpan := Ink2Ref{}.Span(0), Ink2Ref{}.Span(0)
	if extent > 0 {
		rSpan, relSpan = r.Span(T*p.Heads*p.RelLen), relBias.Span(extent*p.RelLen)
	}
	shape := ink2PAttendShape{
		heads: uint32(p.Heads), perKV: uint32(p.Heads / p.KVHeads), headDim: uint32(p.HeadDim), relLen: uint32(p.RelLen),
		kvDim: uint32(kvDim), extent: uint32(extent), window: uint32(window), cap: uint32(cap),
		threads: uint32(threads), width: uint32(width), span: uint32(span), pos0: uint32(pos0), scale: p.Scale,
	}
	// Launches of one command buffer run in order, so the ranges share scores.
	for tok0 := 0; tok0 < T; tok0 += perLaunch {
		rows := min(perLaunch, T-tok0)
		shape.tok0 = uint32(tok0)
		if err := d.Dispatch(pipe, [3]int{p.Heads, rows, 1}, [3]int{threads, 1, 1},
			q.Span(T*p.Heads*p.HeadDim), rSpan, sinks.Span(T*p.Heads), relSpan,
			kcache.Span(cap*kvDim), vcache.Span(cap*kvDim), scores.Span(rows*p.Heads*span), out.Span(T*p.Heads*p.HeadDim),
			Ink2Const(shape)); err != nil {
			return err
		}
	}
	return nil
}

type ink2PFlashShape struct {
	heads, perKV, relLen, kvDim uint32
	extent, window, cap, tokens uint32
	pos0                        uint32
	scale                       float32
}

const (
	ink2FlashQueries = 16
	ink2FlashHeadDim = 128
	ink2FlashMaxRel  = 32
)

// PFlashAttend is PAttend as a tiled attention with an online softmax, which
// needs no scores scratch and sums in another order. ok is false (nothing
// encoded) for a shape it does not take: a head dim other than 128, more than
// 32 relative-bias terms, or a cache whose length is not a multiple of 8.
func (d *Ink2Device) PFlashAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) (bool, error) {
	const limit = math.MaxInt32
	if p.HeadDim != ink2FlashHeadDim || p.RelLen > ink2FlashMaxRel || cap%8 != 0 {
		return false, nil
	}
	if p.Heads < 1 || p.KVHeads < 1 || p.Heads%p.KVHeads != 0 || p.RelLen < 0 || extent < 0 || window < 0 ||
		cap < 1 || T < 1 || pos0 < 0 || pos0 > limit-T || p.Heads > limit/p.HeadDim/T || extent > limit || window > limit ||
		cap > limit/(p.KVHeads*p.HeadDim) || (p.RelLen > 0 && extent > limit/p.RelLen) {
		return false, fmt.Errorf("%w: flash attend of %d rows at %d, %d heads over %d kv heads of %d dims, extent %d, window %d, cap %d",
			ErrInk2Shape, T, pos0, p.Heads, p.KVHeads, p.HeadDim, extent, window, cap)
	}
	if p.RelLen == 0 {
		extent = 0
	}
	if !q.Valid() || !sinks.Valid() || !kcache.Valid() || !vcache.Valid() || !out.Valid() ||
		(extent > 0 && (!r.Valid() || !relBias.Valid())) {
		return false, fmt.Errorf("%w: flash attend without an operand", ErrInk2Shape)
	}
	pipe, err := d.PipelinePrecise(ink2PrefillFlashSource, "ink2p_flash")
	if err != nil {
		return false, err
	}
	if pipe.ThreadWidth() != 32 || pipe.MaxThreads() < 128 {
		return false, fmt.Errorf("metalgemm: inkling2 flash attend needs 32-wide SIMD groups of 128 threads")
	}
	rSpan, relSpan, relLen := Ink2Ref{}.Span(0), Ink2Ref{}.Span(0), 0
	if extent > 0 {
		rSpan, relSpan, relLen = r.Span(T*p.Heads*p.RelLen), relBias.Span(extent*p.RelLen), p.RelLen
	}
	kvDim := p.KVHeads * p.HeadDim
	return true, d.Dispatch(pipe, [3]int{p.Heads, (T + ink2FlashQueries - 1) / ink2FlashQueries, 1}, [3]int{128, 1, 1},
		q.Span(T*p.Heads*p.HeadDim), rSpan, sinks.Span(T*p.Heads), relSpan, kcache.Span(cap*kvDim), vcache.Span(cap*kvDim),
		out.Span(T*p.Heads*p.HeadDim), Ink2Const(ink2PFlashShape{
			heads: uint32(p.Heads), perKV: uint32(p.Heads / p.KVHeads), relLen: uint32(relLen), kvDim: uint32(kvDim),
			extent: uint32(extent), window: uint32(window), cap: uint32(cap), tokens: uint32(T), pos0: uint32(pos0), scale: p.Scale,
		}))
}

// PRoute encodes Route for each of T logits rows ([T][nexp]) into ids and wts
// ([T][k]).
func (d *Ink2Device) PRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	if !logits.Valid() || !ids.Valid() || !wts.Valid() || nexp < 1 || nexp > ink2OpsMaxExperts || k < 1 || k > ink2OpsMaxTopK ||
		k > nexp || T < 1 || T > math.MaxInt32/nexp {
		return fmt.Errorf("%w: prefill route of %d rows of %d experts to %d slots", ErrInk2Shape, T, nexp, k)
	}
	p, err := d.prefillKernel("ink2p_route", true)
	if err != nil {
		return err
	}
	return d.Dispatch(p, [3]int{T, 1, 1}, [3]int{ink2OpsRouteThreads, 1, 1},
		logits.Span(T*nexp), bias.Span(nexp), ids.Span(T*k), wts.Span(T*k),
		Ink2Const(ink2PRouteShape{experts: uint32(nexp), slots: uint32(k), biased: uint32(btoi(bias.Valid())), weightScale: scale}))
}

// PCombine encodes out[t] = sum over token t's k routed rows of down
// ([T*k][n]), in routing order, of wts[t][s] * down[t*k+s].
func (d *Ink2Device) PCombine(down, wts, out Ink2Ref, k, n, T int) error {
	if !down.Valid() || !wts.Valid() || !out.Valid() || k < 1 || n < 1 || T < 1 || T > math.MaxInt32/k/n {
		return fmt.Errorf("%w: prefill combine of %d rows of %d slots of width %d", ErrInk2Shape, T, k, n)
	}
	p, err := d.prefillKernel("ink2p_combine", false)
	if err != nil {
		return err
	}
	return d.DispatchThreads(p, [3]int{n, T, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		down.Span(T*k*n), wts.Span(T*k), out.Span(T*n), Ink2Const(ink2PCombineShape{uint32(k), uint32(n), uint32(T)}))
}

// PBucket encodes the stable sort of npairs routed ids by expert: counts
// ([nexp]), their exclusive scan offsets ([nexp+1]) and order ([npairs], the
// pair indices by expert), all int32.
func (d *Ink2Device) PBucket(ids, counts, offsets, order Ink2Ref, npairs, nexp int) error {
	if !ids.Valid() || !counts.Valid() || !offsets.Valid() || !order.Valid() || npairs < 1 || npairs > math.MaxInt32 ||
		nexp < 1 || nexp > ink2OpsMaxExperts {
		return fmt.Errorf("%w: bucket of %d pairs over %d experts", ErrInk2Shape, npairs, nexp)
	}
	scan, err := d.prefillKernel("ink2p_bucket_scan", false)
	if err != nil {
		return err
	}
	fill, err := d.prefillKernel("ink2p_bucket_fill", false)
	if err != nil {
		return err
	}
	threads := min(nexp, scan.MaxThreads())
	shape := Ink2Const(ink2PBucketShape{uint32(npairs), uint32(nexp), uint32(threads)})
	if err := d.Dispatch(scan, [3]int{1, 1, 1}, [3]int{threads, 1, 1}, ids.Span(npairs), counts.Span(nexp), offsets.Span(nexp+1), shape); err != nil {
		return err
	}
	return d.DispatchThreads(fill, [3]int{npairs, 1, 1}, [3]int{min(npairs, ink2OpsThreads, fill.MaxThreads()), 1, 1},
		ids.Span(npairs), offsets.Span(nexp+1), order.Span(npairs), shape)
}

// PExpertGrouped encodes y[i] = W[ids[i]] x[(i/xDiv)*xStride] (* scale[ids[i]])
// for npairs routed pairs, visiting them in the by-expert order PBucket left.
// Each pair equals ExpertMatVec of it.
func (d *Ink2Device) PExpertGrouped(w Ink2Weight, ids, order, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale Ink2Ref) error {
	if !ids.Valid() || !order.Valid() || !x.Valid() || !y.Valid() || npairs < 1 || xDiv < 1 || xStride < w.Cols ||
		npairs > math.MaxInt32/max(xStride, w.Rows, 1) {
		return fmt.Errorf("%w: grouped experts of %d pairs, input stride %d", ErrInk2Shape, npairs, xStride)
	}
	p, bytes, err := d.packedBatch(w, &ink2GroupedKernel, x, xStride)
	if err != nil {
		return err
	}
	rows := (npairs-1)/xDiv + 1
	// The scale table's length is the expert count, as in ExpertScale.
	args := append(w.args(x.Span((rows-1)*xStride+w.Cols), y.Span(npairs*w.Rows), w.Experts*bytes),
		ids.Span(npairs), order.Span(npairs), scale.Span(w.Experts),
		Ink2Const(ink2PackedGroup{uint32(npairs), uint32(xDiv), uint32(xStride), uint32(btoi(scale.Valid()))}))
	return d.Dispatch(p, [3]int{(w.Rows + 4*ink2PrefillRowsR - 1) / (4 * ink2PrefillRowsR), (npairs + ink2PrefillGroupTile - 1) / ink2PrefillGroupTile, 1}, [3]int{128, 1, 1}, args...)
}
