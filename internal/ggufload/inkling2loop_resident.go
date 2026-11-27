package ggufload

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2ResidentCtx is the global-slot KV capacity of a resident session; local
// slots keep a sliding-window ring. FAK_INKLING2_CTX raises it for agentic serving,
// paying device memory on every global slot, so the bench default stays small. A
// build carries at most one resident device family, so the bound is that family's.
var inkling2ResidentCtx = inkling2ResidentCtxFrom(os.LookupEnv)

// The resident context bounds, exported so admission can price a session before
// the driver exists. The requested ctx is fitted against device free memory when the
// session is built, and refused rather than clamped when it does not fit.
const (
	Inkling2ResidentCtxEnv     = "FAK_INKLING2_CTX"
	Inkling2ResidentCtxDefault = 4096
	Inkling2ResidentCtxMin     = 512
	// The ceiling used only when device free memory is unknown, where the global-slot
	// KV cannot be sized: a larger cache would spill into the allocator's managed-memory
	// fallback, where prefill faults. With free memory known, the fit alone bounds CUDA.
	Inkling2ResidentCtxMaxCUDA = 24576
	// Metal prefill scores tile under a fixed budget, a shape limit the memory fit does
	// not model, so a request above it is refused.
	Inkling2ResidentCtxMaxMetal = 32768
	// inkling2ResidentCtxParseMax only bounds the parsed integer so garbage cannot overflow.
	inkling2ResidentCtxParseMax = 1 << 20
	// Inkling2ResidentArenaFloats bounds the per-token scratch arena. Every
	// intermediate of a token takes the next arena slot, so a token's device
	// addresses repeat exactly on the next token and a captured graph stays valid.
	Inkling2ResidentArenaFloats = 16 << 20
)

const inkling2ArenaFloats = Inkling2ResidentArenaFloats

// Inkling2ResidentCtxMax is the largest resident context a session on the named
// backend takes; any backend but Metal gets the CUDA bound.
func Inkling2ResidentCtxMax(backend string) int {
	if backend == "metal" {
		return Inkling2ResidentCtxMaxMetal
	}
	return Inkling2ResidentCtxMaxCUDA
}

// inkling2ResidentCtxFrom parses FAK_INKLING2_CTX. An unset, unparseable or sub-floor value
// takes the default; a larger one is returned unclamped for the fit or the hard bound to refuse.
func inkling2ResidentCtxFrom(lookup func(string) (string, bool)) int {
	v, ok := lookup(Inkling2ResidentCtxEnv)
	if !ok {
		return Inkling2ResidentCtxDefault
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil || n < Inkling2ResidentCtxMin {
		return Inkling2ResidentCtxDefault
	}
	return min(n, inkling2ResidentCtxParseMax)
}

type ink2Ref = compute.Ink2Ref

// ink2Mat is one weight the device multiplies: a resident tensor (any MatMul dtype).
type ink2Mat struct {
	t          compute.Tensor
	rows, cols int
}

type ink2Experts struct {
	stack      compute.Tensor
	views      []compute.Tensor
	rows, padK int
	scale      ink2Ref
}

type ink2Layer struct {
	wq, wk, wv, wo, gate, rel, sink, kconv, vconv, router ink2Mat
	attnNorm, qNorm, kNorm, vNorm                         ink2Ref
	qclip, relBias, relClip                               ink2Ref
	kconvW, vconvW, attnPostConv, ffnPostConv             ink2Ref
	resQ                                                  ink2Ref
	resQRows                                              int
	ffnResQ, ffnNorm, probsBias                           ink2Ref
	gateUp, down                                          ink2Experts
}

type ink2Ctrl struct {
	msgQ         ink2Ref
	down         [4]ink2Ref
	gain, forget [4]ink2Ref
	rank         int
	// half holds down, gain and forget as bfloat16 when every value is exactly
	// one, which the decode step reads instead for the same bits.
	half *ink2CtrlHalf
}

type ink2CtrlHalf struct{ down, gain, forget [4]ink2Ref }

// ink2BF16Controls is a device that runs the controllers on bfloat16 weights.
type ink2BF16Controls interface {
	Ink2PackBF16(src compute.Ink2Ref, n int) (compute.Tensor, bool, error)
	Ink2ControlBF16(ctl compute.Ink2Controls, g, out compute.Ink2Ref, n, rank int, norms []compute.Ink2Ref, eps float32) (bool, error)
}

// inkling2Resident runs the inkling2_loop decode forward with every activation,
// cache and routing decision resident on a device; per token it uploads the
// embedding row and position and reads back one argmax index.
type inkling2Resident struct {
	hp     model.Inkling2LoopHParams
	w      *model.Inkling2LoopWeights
	be     compute.Backend
	dev    compute.Inkling2Device
	owned  []compute.Tensor
	layers []ink2Layer
	ctrls  []ink2Ctrl

	batchLookup func(string) (string, bool)

	tokNorm, outNorm, outResQ ink2Ref
	output                    ink2Mat

	// ctx is the global-slot cache capacity this session was built with, fixed at bind.
	ctx               int
	in, arena, scores ink2Ref
	argIdx            ink2Ref
	logits            ink2Ref // the last step's or prefill's softcapped logits
	stepLogits        ink2Ref // forward's logits: one arena offset, so graph replays rewrite it
	kc, vc, hist      []ink2Ref
	caps              []int
	off, arenaFloats  int
	pos               int
	err               error

	// kvHalf stores the global-slot K and V caches as f16 (FAK_INKLING2_KV=f16); halfDev
	// is the decode op set that reads them. The local rings always stay fp32.
	kvHalf  bool
	halfDev compute.Inkling2HalfKVDevice

	q8 map[int]ink2Ref // arena offset of an activation -> its Q8_0 codes this token

	graph       bool
	graphReady  bool
	hostRouting bool // an expert dtype has no indexed kernel: ids are read back per layer

	warm                               bool // one eager token has run since the last batched prefill
	batched                            bool
	prefillKeep, prefillScr, prefillIn ink2Ref
	prefillCap                         int
	// chunkMax is the widest prefill forward the local rings of this session hold and
	// chunkWanted the width a cohort on this binding may use. baseCaps are the ring
	// capacities at the base chunk, which every cohort state widens for itself.
	// prefillWidest is the widest prefill forward the session has run.
	chunkMax, chunkWanted int
	baseCaps              []int
	prefillWidest         int

	// speculative verify state, allocated by enableSpec
	specMax                            int
	convLog                            []ink2Ref
	verifyLogits, verifyIdx            ink2Ref
	verifyPos, verifyKeep              ink2Ref
	verifyWarm, verifyReady, keepReady bool
	// eager-warmed and captured verify graphs and captured commit graphs of the chunk
	// widths under specMax, by width
	tailWarm, tailReady, tailKeepReady [inkling2SpecMaxDraft + 1]bool
	draft                              ink2ContextDraft

	pipe *ink2Pipe // decode pipelining, nil when the device or table cannot run it

	// decode-window report of FAK_INK2_PROFILE on a Metal device, printed at
	// close; it counts plain steps only, so speculative tokens inflate it
	report       bool
	decN         int
	decT0, decT1 time.Time
	decS0, decS1 metalgemm.Ink2Stats
}

// ink2ContextDraft is a device draft whose context every target forward writes
// from the inputs of the tapped layer visits.
type ink2ContextDraft interface {
	// tapVisits are the visits whose input is a tap; the final-norm output is
	// always appended after them.
	tapVisits() []int
	commitTaps(p *ink2Prefill, taps []ink2Ref)
	// dropCaptures forgets any graph the draft captured over freed buffers.
	dropCaptures()
}

func (s *inkling2Resident) fail(err error) {
	if err != nil && s.err == nil {
		s.err = err
	}
}

// makeResident wires the session's buffers from s.owned[from:] on, plus extra, on
// Metal, so the first prefill's command buffer starts without paying for it.
func (s *inkling2Resident) makeResident(from int, extra []metalgemm.Ink2Buf) {
	if s.err != nil || s.be.Name() != "metal" {
		return
	}
	s.fail(compute.MetalInkling2MakeResident(append(compute.MetalInkling2Buffers(s.owned[from:]), extra...)))
}

func (s *inkling2Resident) alloc(n int) ink2Ref {
	t, err := s.dev.Ink2Alloc(n)
	if err != nil {
		s.fail(err)
		return ink2Ref{}
	}
	s.owned = append(s.owned, t)
	return compute.Ink2At(t, 0)
}

func (s *inkling2Resident) upload(v []float32) ink2Ref {
	if v == nil {
		return ink2Ref{}
	}
	r := s.alloc(len(v))
	if s.err == nil {
		s.fail(s.dev.Ink2Write(r, v))
	}
	return r
}

func (s *inkling2Resident) uploadF32Mat(rows, cols int, data []float32) ink2Mat {
	r := s.upload(data)
	if s.err != nil {
		return ink2Mat{}
	}
	t := r.T
	t.Shape = []int{rows, cols}
	return ink2Mat{t: t, rows: rows, cols: cols}
}

// mat makes a device weight of m: resident tensors are used as bound, host-side
// small dense matrices are widened to F32 and uploaded.
func (s *inkling2Resident) mat(m model.Inkling2LoopMatrix) ink2Mat {
	switch v := m.(type) {
	case nil:
		return ink2Mat{}
	case *inkling2DeviceMatrix:
		if v.padK != v.cols {
			s.fail(fmt.Errorf("inkling2 resident: padded dense weight [%d %d] pad %d", v.rows, v.cols, v.padK))
		}
		return ink2Mat{t: v.w, rows: v.rows, cols: v.cols}
	case *inkling2QuantMatrix:
		f := make([]float32, v.rows*v.cols)
		for r := 0; r < v.rows; r++ {
			if err := inkling2DequantRow(v.typ, v.raw[r*v.rowBytes:(r+1)*v.rowBytes], f[r*v.cols:(r+1)*v.cols]); err != nil {
				s.fail(err)
				return ink2Mat{}
			}
		}
		return s.uploadF32Mat(v.rows, v.cols, f)
	case *model.Inkling2LoopDenseMatrix:
		return s.uploadF32Mat(v.R, v.C, v.Data)
	}
	s.fail(fmt.Errorf("inkling2 resident: matrix %T is not device-bindable", m))
	return ink2Mat{}
}

func (s *inkling2Resident) experts(e model.Inkling2LoopExperts, scale []float32) ink2Experts {
	d, ok := e.(*inkling2DeviceExperts)
	if !ok {
		s.fail(fmt.Errorf("inkling2 resident: experts %T are not device-resident", e))
		return ink2Experts{}
	}
	return ink2Experts{stack: d.stack, views: d.w, rows: d.rows, padK: d.padK, scale: s.upload(scale)}
}

func concatRows(rows [][]float32) []float32 {
	var out []float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

func newInkling2Resident(m *Inkling2LoopModel, be compute.Backend) (*inkling2Resident, error) {
	ctx := inkling2ResidentCtx
	switch {
	case inkling2ResidentCtxPinned > 0:
		ctx = inkling2ResidentCtxPinned
	case inkling2ResidentCtxAutoSet(os.LookupEnv):
		ctx = inkling2ResidentCtxAuto
	}
	if limit := inkling2ResidentHardMax(be.Name()); limit > 0 && ctx > limit {
		return nil, fmt.Errorf("inkling2 resident: %d-token context exceeds the %d-token bound of backend %s", ctx, limit, be.Name())
	}
	return newInkling2ResidentSized(m, be, ctx, inkling2ArenaFloats)
}

// inkling2ResidentCtxPinned is a context a caller already admitted, which the next
// resident session builds exactly, instead of sizing it again.
var inkling2ResidentCtxPinned int

// Inkling2ResidentCtxPin fixes the global cache of the next resident session at ctx.
// The caller must have admitted that size against the memory it will use.
func Inkling2ResidentCtxPin(ctx int) { inkling2ResidentCtxPinned = ctx }

// Inkling2ResidentCtxAuto reports whether the resident session sizes its context from
// free device memory, which is the default when FAK_INKLING2_CTX is unset or blank.
func Inkling2ResidentCtxAuto(lookup func(string) (string, bool)) bool {
	return inkling2ResidentCtxAutoSet(lookup)
}

// Inkling2ResidentFitCtx is the largest resident global context hp fits in free bytes
// after the reserve the device budget keeps, or 0 when even the floor does not fit.
func Inkling2ResidentFitCtx(hp model.Inkling2LoopHParams, free int64) int {
	return inkling2FitResidentCtx(free, inkling2ResidentReserve(free), inkling2ResidentPerPosBytes(hp))
}

// inkling2ResidentCtxAuto asks newInkling2ResidentRing to size the global slots from
// the device memory left once the weights are bound.
const inkling2ResidentCtxAuto = -1

// inkling2ResidentCtxAutoSet reports whether FAK_INKLING2_CTX is unset or blank, the
// default that derives the context from free device memory.
func inkling2ResidentCtxAutoSet(lookup func(string) (string, bool)) bool {
	v, ok := lookup("FAK_INKLING2_CTX")
	return !ok || strings.TrimSpace(v) == ""
}

// inkling2ResidentPerPosBytes is the device cost of one global-slot cache position:
// its f32 K and V rows in every global slot, plus the attention score row, which
// grows with the global context.
func inkling2ResidentPerPosBytes(hp model.Inkling2LoopHParams) int64 {
	return inkling2ResidentPerPosBytesDtype(hp, false)
}

// inkling2ResidentPerPosBytesDtype is inkling2ResidentPerPosBytes with half selecting
// f16 for the global-slot K and V caches; the attention score row stays float32.
func inkling2ResidentPerPosBytesDtype(hp model.Inkling2LoopHParams, half bool) int64 {
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	var global int64
	for slot := 0; slot < slots; slot++ {
		if !hp.LocalLayer[inkling2SlotLayer(hp, slot)] {
			global++
		}
	}
	elemBytes := int64(4)
	if half {
		elemBytes = 2
	}
	kvDim := int64(hp.KVHeads * hp.HeadDim)
	return global*2*elemBytes*kvDim + 2*4*int64(hp.Heads)
}

// inkling2FitResidentCtx is the largest global context whose cache and score rows fit
// in free bytes after reserve, in 256-position steps, clamped to the device ceiling.
// It returns 0 when even the floor does not fit.
func inkling2FitResidentCtx(free, reserve, perPos int64) int {
	if perPos <= 0 || free-reserve < perPos*Inkling2ResidentCtxMin {
		return 0
	}
	n := int((free - reserve) / perPos)
	n = min(n/256*256, Inkling2ResidentCtxMax(inkling2ResidentBackend))
	return n
}

// inkling2AutoCtxCeiling is the largest derived context the default may take: 16384 is the
// all-VRAM context measured to load and decode on a 16 GiB RTX 5080.
const inkling2AutoCtxCeiling = 16384

// Inkling2AutoCtxCeiling is the single source the Metal admission in cmd/fak applies, so
// the two device paths size the default identically.
const Inkling2AutoCtxCeiling = inkling2AutoCtxCeiling

// inkling2ResidentHardMax is the largest explicit context a backend takes outright, before
// the memory fit. Zero means the fit alone bounds the request.
func inkling2ResidentHardMax(backend string) int {
	if backend == "metal" {
		return Inkling2ResidentCtxMaxMetal
	}
	return 0
}

// inkling2ResolveAutoCtx sizes the global cache from the device memory free now,
// after the weights bound. A backend that cannot report free memory keeps the
// fixed default rather than guess.
func inkling2ResolveAutoCtx(be compute.Backend, hp model.Inkling2LoopHParams, prefill int64, half bool) (int, error) {
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known || free == compute.FreeUnknown || free < 0 {
		fmt.Fprintf(os.Stderr, "inkling2 resident: device free memory unknown, context %d (default)\n", Inkling2ResidentCtxDefault)
		return Inkling2ResidentCtxDefault, nil
	}
	perPos := inkling2ResidentPerPosBytesDtype(hp, half)
	ctx := min(inkling2FitResidentCtx(free, inkling2ResidentReserve(free)+prefill, perPos), inkling2AutoCtxCeiling)
	if ctx == 0 {
		return 0, fmt.Errorf("inkling2 resident: %d free bytes after weights hold no %d-token global cache at %d bytes per position; set %s to a smaller context",
			free, Inkling2ResidentCtxMin, perPos, "FAK_INKLING2_CTX")
	}
	fmt.Fprintf(os.Stderr, "inkling2 resident: context %d (auto: %.2f GiB free after weights, %d bytes per position)\n",
		ctx, float64(free)/(1<<30), perPos)
	return ctx, nil
}

// prefillWidestBytes is the device memory of the prefill arenas at the widest chunk the
// session may use (chunkWanted, never narrower than the base chunk). ensurePrefillArenas
// allocates exactly these, so the global cache must leave them room; a narrower reserve
// lets a long prompt's wide prefill fail on the activation alloc after the session binds.
func (s *inkling2Resident) prefillWidestBytes() int64 {
	T := max(s.chunkWanted, inkling2PrefillChunkBase)
	keep, scr := s.prefillArenas(T)
	return 4 * int64(keep+scr+T*s.hp.Hidden)
}

// inkling2ResidentReserve is what the resident session keeps free of the global cache:
// the per-token arena and the same headroom share the device budget keeps.
func inkling2ResidentReserve(free int64) int64 {
	headroom := min(free*inkling2VRAMHeadroomPct/100, inkling2VRAMHeadroomCap)
	return headroom + 4*inkling2ArenaFloats
}

// newInkling2ResidentSized is newInkling2Resident with ctx global-slot cache positions
// and an arenaFloats per-token arena. ctx 0 allocates no slot caches, histories or
// scores (caps still hold the local ring sizes) for a caller that owns its own state.
func newInkling2ResidentSized(m *Inkling2LoopModel, be compute.Backend, ctx, arenaFloats int) (*inkling2Resident, error) {
	wanted := inkling2PrefillChunkWantedOn(be, os.LookupEnv)
	// a cohort binding (ctx 0) holds no per-sequence state: each cohort sizes its own
	ring := inkling2PrefillChunkBase
	if ctx != 0 && !inkling2ScalarWideHeld(wanted) {
		ring = wanted
	}
	s, err := newInkling2ResidentRing(m, be, ctx, arenaFloats, ring, wanted)
	if err != nil && ring > inkling2PrefillChunkBase && inkling2ChunkRefusal(err) {
		// the refused build freed everything it took, so the base build starts from
		// the memory a session that never opted in would have had
		inkling2ScalarWideRefused(err, ring)
		inkling2TrimIdle(be)
		s, err = newInkling2ResidentRing(m, be, ctx, arenaFloats, inkling2PrefillChunkBase, wanted)
	}
	return s, err
}

// inkling2ResidentKVBytes is the device memory newInkling2ResidentRing allocates for
// the slot K and V caches, the conv histories and the shared score row at ctx tokens.
// It walks the same caps the build uses, so the two cannot drift.
func inkling2ResidentKVBytes(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int) int64 {
	return inkling2ResidentKVBytesDtype(hp, ctx, batched, chunk, false)
}

// inkling2ResidentKVBytesDtype is inkling2ResidentKVBytes with half selecting f16 for the
// global-slot caches; the local rings, conv histories and scores stay float32.
func inkling2ResidentKVBytesDtype(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, half bool) int64 {
	if ctx <= 0 {
		return 0
	}
	kvDim := hp.KVHeads * hp.HeadDim
	channels := 2*kvDim + 2*hp.Hidden
	var bytes int64
	maxCap := hp.SlidingWindow
	for slot, c := range inkling2ResidentCaps(hp, ctx, batched, chunk) {
		maxCap = max(maxCap, c)
		elemBytes := int64(4)
		if half && !inkling2SlotIsLocal(hp, slot) {
			elemBytes = 2
		}
		bytes += 2*int64(c)*int64(kvDim)*elemBytes + int64(channels)*int64(hp.ConvKernel-1)*4
	}
	bytes += 2 * int64(hp.Heads) * int64(maxCap) * 4
	return bytes
}

// inkling2SlotIsLocal reports whether a cache slot is a sliding-window ring.
func inkling2SlotIsLocal(hp model.Inkling2LoopHParams, slot int) bool {
	return hp.LocalLayer[inkling2SlotLayer(hp, slot)]
}

// inkling2ResidentFixedBytes is the device memory every session allocates whatever its
// ctx: the input row and the per-token scratch arena and argmax index.
func inkling2ResidentFixedBytes(hp model.Inkling2LoopHParams, arenaFloats int) int64 {
	return 4 * int64(hp.Hidden+4+arenaFloats+4)
}

// inkling2ResidentOtherBytes is the non-KV device memory of a session: the fixed
// buffers plus, when wideChunk > 0, the wide prefill arenas and the chunk input.
func inkling2ResidentOtherBytes(hp model.Inkling2LoopHParams, arenaFloats, wideChunk, padK int) int64 {
	other := inkling2ResidentFixedBytes(hp, arenaFloats)
	if wideChunk > 0 {
		keep, scr := inkling2PrefillArenaFloats(hp, wideChunk, padK)
		other += 4 * int64(keep+scr+wideChunk*hp.Hidden)
	}
	return other
}

// inkling2ResidentTotalBytes is every device allocation a resident session makes at
// ctx tokens: the KV caches and the non-KV buffers of inkling2ResidentOtherBytes.
func inkling2ResidentTotalBytes(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk, arenaFloats, wideChunk, padK int) int64 {
	return inkling2ResidentKVBytes(hp, ctx, batched, chunk) + inkling2ResidentOtherBytes(hp, arenaFloats, wideChunk, padK)
}

// inkling2FitResidentKV returns the resident KV bytes at ctx after checking the KV plus
// other against the device free memory left at build time. The weights are already on
// the device, so free is what the session may take minus the weight headroom rule.
// Unknown free falls back to the per-backend Inkling2ResidentCtxMax ceiling.
func inkling2FitResidentKV(be compute.Backend, hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, other int64) (int64, error) {
	return inkling2FitResidentKVDtype(be, hp, ctx, batched, chunk, false, other)
}

// inkling2FitResidentKVDtype is inkling2FitResidentKV sized for half's global-slot caches.
func inkling2FitResidentKVDtype(be compute.Backend, hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, half bool, other int64) (int64, error) {
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known || free == compute.FreeUnknown || free < 0 {
		free = compute.FreeUnknown
	}
	return inkling2KVFitDtype(hp, ctx, batched, chunk, half, other, free)
}

// inkling2MaxResidentCtx is the largest ctx in [0, ceil] whose KV fits budget bytes,
// 0 when none does. KV is monotone in ctx, so bisection finds the exact bound.
func inkling2MaxResidentCtx(hp model.Inkling2LoopHParams, batched bool, chunk, ceil int, budget int64) int {
	return inkling2MaxResidentCtxDtype(hp, batched, chunk, false, ceil, budget)
}

func inkling2MaxResidentCtxDtype(hp model.Inkling2LoopHParams, batched bool, chunk int, half bool, ceil int, budget int64) int {
	lo, hi := 0, ceil
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if inkling2ResidentKVBytesDtype(hp, mid, batched, chunk, half) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo
}

// newInkling2ResidentRing builds the resident with local rings and prefill arenas for
// a forward of ring tokens; wanted is the chunk its cohorts may use.
func newInkling2ResidentRing(m *Inkling2LoopModel, be compute.Backend, ctx, arenaFloats, ring, wanted int) (*inkling2Resident, error) {
	dev, ok := be.(compute.Inkling2Device)
	if !ok {
		return nil, fmt.Errorf("inkling2 resident: backend %s has no inkling2 device ops", be.Name())
	}
	hp, w := m.HP, m.Weights
	if inkling2KVHalfErr != nil {
		return nil, inkling2KVHalfErr
	}
	s := &inkling2Resident{hp: hp, w: w, be: be, dev: dev, q8: map[int]ink2Ref{}, arenaFloats: arenaFloats, chunkMax: ring, chunkWanted: wanted}
	s.report = metalgemm.Ink2ReportRequested(os.Getenv(metalgemm.Ink2ProfileEnv))
	s.halfDev, _ = be.(compute.Inkling2HalfKVDevice)
	_, hasPrefill := be.(compute.Inkling2PrefillDevice)
	s.batched = hasPrefill && (inkling2BatchedPrefill() || inkling2SpecDepthFor(m) > 0)
	s.tokNorm, s.outNorm, s.outResQ = s.upload(w.TokNorm), s.upload(w.OutputNorm), s.upload(w.OutputResQuery)
	s.output = s.mat(w.Output)
	s.layers = make([]ink2Layer, hp.Layers)
	for i := range w.Layers {
		lw, l := &w.Layers[i], &s.layers[i]
		l.wq, l.wk, l.wv, l.wo = s.mat(lw.Wq), s.mat(lw.Wk), s.mat(lw.Wv), s.mat(lw.Wo)
		l.gate, l.rel, l.sink = s.mat(lw.Gate), s.mat(lw.Rel), s.mat(lw.SinkProj)
		l.kconv, l.vconv, l.router = s.mat(lw.KConvProj), s.mat(lw.VConvProj), s.mat(lw.Router)
		l.attnNorm, l.qNorm, l.kNorm, l.vNorm = s.upload(lw.AttnNorm), s.upload(lw.QInpNorm), s.upload(lw.KInpNorm), s.upload(lw.VInpNorm)
		l.qclip, l.relBias, l.relClip = s.upload(lw.QClip), s.upload(lw.RelBias), s.upload(lw.RelClip)
		l.kconvW, l.vconvW = s.upload(lw.KConv), s.upload(lw.VConv)
		l.attnPostConv, l.ffnPostConv = s.upload(lw.AttnPostConv), s.upload(lw.FFNPostConv)
		l.resQRows = len(lw.AttnResQuery)
		l.resQ = s.upload(concatRows(lw.AttnResQuery))
		l.ffnResQ, l.ffnNorm, l.probsBias = s.upload(lw.FFNResQuery), s.upload(lw.FFNNorm), s.upload(lw.ExpProbsBias)
		l.gateUp = s.experts(lw.GateUp, lw.GateUpScale)
		l.down = s.experts(lw.Down, lw.DownScale)
		if s.err != nil {
			s.close()
			return nil, fmt.Errorf("inkling2 resident: layer %d: %w", i, s.err)
		}
	}
	s.ctrls = make([]ink2Ctrl, len(w.Controllers))
	for i := range w.Controllers {
		cw, c := &w.Controllers[i], &s.ctrls[i]
		c.msgQ = s.upload(concatRows(cw.MsgQuery[:]))
		for j := 0; j < 4; j++ {
			c.down[j] = s.f32Ref(cw.Down[j])
			c.rank = cw.Down[j].Rows()
			c.gain[j] = s.f32Ref(cw.Gain[j])
			c.forget[j] = s.f32Ref(cw.Forget[j])
		}
		s.packCtrl(c, hp.Hidden)
	}

	halfKV := ctx != 0 && inkling2ResidentHalfKV(be)
	if ctx == inkling2ResidentCtxAuto {
		resolved, err := inkling2ResolveAutoCtx(be, hp, s.prefillWidestBytes(), halfKV)
		if err != nil {
			s.close()
			return nil, err
		}
		ctx = resolved
	}
	s.ctx = ctx
	s.kvHalf = ctx > 0 && halfKV
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	kvDim := hp.KVHeads * hp.HeadDim
	channels := 2*kvDim + 2*hp.Hidden
	for i := range s.layers {
		for _, e := range []ink2Experts{s.layers[i].gateUp, s.layers[i].down} {
			if !inkling2IndexedExpert(e.stack.Dtype) {
				s.hostRouting = true
			}
		}
	}
	// a ring is addressed by its capacity, so it cannot be widened once the session
	// holds positions: the wide chunk is decided, admitted and allocated here
	wide := s.chunkMax > inkling2PrefillChunkBase && ctx > 0 && s.batched && !s.hostRouting
	if wide && s.err != nil {
		s.close()
		return nil, s.err
	}
	if wide {
		if err := s.admitWideChunk(ctx); err != nil {
			inkling2ScalarWideRefused(err, s.chunkMax)
			wide = false
		}
	}
	if !wide {
		s.chunkMax = min(s.chunkMax, inkling2PrefillChunkBase)
	}
	kvBytes := int64(0)
	if ctx > 0 {
		wideChunk := 0
		if wide {
			wideChunk = s.chunkMax
		}
		other := inkling2ResidentOtherBytes(hp, arenaFloats, wideChunk, s.maxPadK())
		var err error
		if kvBytes, err = inkling2FitResidentKVDtype(be, hp, ctx, s.batched, s.chunkMax, s.kvHalf, other); err != nil {
			s.close()
			if wide {
				// the wide arenas may be what pushed the fit over; the base build may fit
				return nil, inkling2DeviceRefused{err}
			}
			return nil, err
		}
	}
	s.kc, s.vc, s.hist = make([]ink2Ref, slots), make([]ink2Ref, slots), make([]ink2Ref, slots)
	s.caps = inkling2ResidentCaps(hp, ctx, s.batched, s.chunkMax)
	s.baseCaps = inkling2ResidentCaps(hp, ctx, s.batched, inkling2PrefillChunkBase)
	maxCap := hp.SlidingWindow
	for slot, cap := range s.caps {
		maxCap = max(maxCap, cap)
		if ctx > 0 {
			kvFloats := inkling2BatchKVFloats(cap*kvDim, s.kvHalfSlot(slot))
			s.kc[slot], s.vc[slot] = s.alloc(kvFloats), s.alloc(kvFloats)
			s.hist[slot] = s.alloc(channels * (hp.ConvKernel - 1))
		}
	}
	if ctx > 0 {
		s.scores = s.alloc(2 * hp.Heads * maxCap)
	}
	if wide {
		// allocated once at the full width, so no later prompt regrows the arenas and
		// leaves a superseded pair behind
		s.ensurePrefillArenas(s.chunkMax)
	}
	s.in = s.alloc(hp.Hidden + 4)
	s.arena = s.alloc(arenaFloats)
	s.argIdx = s.alloc(4)
	// Metal has no graph capture, so its session never warms widths for one
	s.graph = inkling2DefaultOn(be, "FAK_INKLING2_GRAPH", os.LookupEnv) && !s.hostRouting && be.Name() != "metal"
	s.batched = s.batched && !s.hostRouting
	s.enablePipe(ctx)
	s.makeResident(0, nil)
	if s.err != nil {
		err := s.err
		s.close()
		if wide {
			// whatever the wide build could not allocate, the base build may still fit
			return nil, inkling2DeviceRefused{err}
		}
		return nil, err
	}
	noteInkling2KVBytes(m, ctx, kvBytes)
	return s, nil
}

func (m ink2Mat) ref() ink2Ref { return compute.Ink2At(m.t, 0) }

// packCtrl gives c its bfloat16 weights when the device runs them and every
// weight packs.
func (s *inkling2Resident) packCtrl(c *ink2Ctrl, n int) {
	pk, ok := s.dev.(ink2BF16Controls)
	if !ok || s.err != nil {
		return
	}
	var h ink2CtrlHalf
	var made []compute.Tensor
	pack := func(dst *ink2Ref, src ink2Ref, count int) bool {
		t, ok, err := pk.Ink2PackBF16(src, count)
		if err != nil {
			s.fail(err)
		}
		if !ok {
			return false
		}
		made = append(made, t)
		*dst = compute.Ink2At(t, 0)
		return true
	}
	for j := range 4 {
		if !pack(&h.down[j], c.down[j], c.rank*2*n) || !pack(&h.gain[j], c.gain[j], n*c.rank) || !pack(&h.forget[j], c.forget[j], n*c.rank) {
			for _, t := range made {
				s.be.Free(t)
			}
			return
		}
	}
	s.owned = append(s.owned, made...)
	c.half = &h
}

// f32Ref is a weight the controller kernel reads directly as dense F32 rows.
func (s *inkling2Resident) f32Ref(m model.Inkling2LoopMatrix) ink2Ref {
	d := s.mat(m)
	if s.err == nil && d.t.Dtype != compute.F32 {
		s.fail(fmt.Errorf("inkling2 resident: controller weight is %s, want F32", d.t.Dtype))
	}
	return d.ref()
}

func inkling2IndexedExpert(dt compute.Dtype) bool {
	switch dt {
	case compute.NVFP4, compute.IQ2_XS, compute.IQ2_S, compute.IQ3_XXS, compute.IQ3_S, compute.IQ4_NL, compute.Q3_K:
		return true
	}
	return false
}

// inkling2ResidentCaps is the cache capacity of every slot of a session: ctx for a
// global slot and the sliding window for a local one, plus chunk when the session
// prefills in batches so a chunk's appended keys never evict a key an earlier query
// of the same chunk still attends to.
func inkling2ResidentCaps(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int) []int {
	caps := make([]int, hp.Layers+(hp.Passes-1)*(hp.CoreEnd-hp.CoreStart))
	for slot := range caps {
		caps[slot] = ctx
		if hp.LocalLayer[inkling2SlotLayer(hp, slot)] {
			caps[slot] = hp.SlidingWindow
			if batched {
				caps[slot] += chunk
			}
		}
	}
	return caps
}

// inkling2SlotLayer maps a cache slot to its physical layer.
func inkling2SlotLayer(hp model.Inkling2LoopHParams, slot int) int {
	if slot < hp.Layers {
		return slot
	}
	core := hp.CoreEnd - hp.CoreStart
	return hp.CoreStart + (slot-hp.Layers)%core
}

func (s *inkling2Resident) close() {
	s.pipeDrain()
	if s.report && s.decN > 0 {
		s.report = false
		if p, ok := compute.MetalInkling2Profile(); ok {
			fmt.Fprint(os.Stderr, p.Format(s.decN, s.decT1.Sub(s.decT0), s.decS1.Sub(s.decS0)))
		}
	}
	if s.dev != nil {
		s.dev.Ink2GraphReset()
	}
	for _, t := range s.owned {
		s.be.Free(t)
	}
	s.owned = nil
}

func (s *inkling2Resident) tmp(n int) ink2Ref {
	n = (n + 63) &^ 63
	if s.off+n > s.arenaFloats {
		s.fail(fmt.Errorf("inkling2 resident: arena exhausted at %d floats", s.off+n))
		return s.arena
	}
	r := compute.Ink2At(s.arena.T, s.off)
	s.off += n
	return r
}

func at(r ink2Ref, off int) ink2Ref { return compute.Ink2At(r.T, r.Off+off) }

func (s *inkling2Resident) mv(m ink2Mat, x ink2Ref) ink2Ref {
	y := s.tmp(m.rows)
	s.mvInto(m, x, y)
	return y
}

func (s *inkling2Resident) mvInto(m ink2Mat, x, y ink2Ref) {
	if s.err != nil {
		return
	}
	if m.t.Dtype == compute.Q8_0 && x.T.Buf() == s.arena.T.Buf() {
		// arena slots are never reused within a token, so the offset names the value
		q, seen := s.q8[x.Off]
		if !seen {
			q = s.tmp(m.cols/4 + m.cols/32)
			s.q8[x.Off] = q
		}
		ok, err := s.dev.Ink2MatVecQ8(m.t, x, ink2Ref{}, q, !seen, y)
		if err != nil || ok {
			s.fail(err)
			return
		}
	}
	s.fail(s.dev.Ink2MatVec(m.t, x, y))
}

// mvGated is mv(m, x * sigmoid(gate)); on a Q8_0 weight the gate is applied while x
// is quantized, so x itself is left unscaled.
func (s *inkling2Resident) mvGated(m ink2Mat, x, gate ink2Ref, n int) ink2Ref {
	if s.err == nil && m.t.Dtype == compute.Q8_0 {
		y := s.tmp(m.rows)
		ok, err := s.dev.Ink2MatVecQ8(m.t, x, gate, s.tmp(m.cols/4+m.cols/32), true, y)
		if err != nil || ok {
			s.fail(err)
			return y
		}
	}
	if s.err == nil {
		s.fail(s.dev.Ink2SigmoidMul(x, gate, n))
	}
	return s.mv(m, x)
}

func (s *inkling2Resident) rms(x, w ink2Ref, n int, eps float32) ink2Ref {
	y := s.tmp(n)
	if s.err == nil {
		s.fail(s.dev.Ink2RMS(x, w, y, 1, n, eps))
	}
	return y
}

func (s *inkling2Resident) mix(states []ink2Ref, queries ink2Ref, nq int) ink2Ref {
	return s.mixNorm(states, queries, nq, nil)
}

// mixNorm is mix with each query row RMS-normalized by its weight in norms, in the
// same launch.
func (s *inkling2Resident) mixNorm(states []ink2Ref, queries ink2Ref, nq int, norms []ink2Ref) ink2Ref {
	out := s.tmp(nq * s.hp.Hidden)
	if s.err == nil {
		s.fail(s.dev.Ink2Mix(states, queries, nq, norms, s.hp.NormEps, out, s.hp.Hidden))
	}
	return out
}

// control runs controllers js of c on (curs[i], msgs[i]) in one batch and returns the
// outputs as consecutive rows, each RMS-normalized by norms[i].
func (s *inkling2Resident) control(curs, msgs []ink2Ref, c *ink2Ctrl, js []int, norms []ink2Ref) ink2Ref {
	h := s.hp.Hidden
	weights := func(down, gain, forget *[4]ink2Ref) compute.Ink2Controls {
		ctl := compute.Ink2Controls{Cur: curs, Msg: msgs}
		for _, j := range js {
			ctl.Down = append(ctl.Down, down[j])
			ctl.Gain = append(ctl.Gain, gain[j])
			ctl.Forget = append(ctl.Forget, forget[j])
		}
		return ctl
	}
	g := s.tmp(len(js) * c.rank)
	out := s.tmp(len(js) * h)
	if s.err != nil {
		return out
	}
	if c.half != nil {
		ok, err := s.dev.(ink2BF16Controls).Ink2ControlBF16(weights(&c.half.down, &c.half.gain, &c.half.forget), g, out, h, c.rank, norms, s.hp.NormEps)
		if err != nil || ok {
			s.fail(err)
			return out
		}
	}
	s.fail(s.dev.Ink2Control(weights(&c.down, &c.gain, &c.forget), g, out, h, c.rank, norms, s.hp.NormEps))
	return out
}

func (s *inkling2Resident) conv(slot int, x, weight, dyn, resid ink2Ref, n, heads, offset int) ink2Ref {
	kn := s.hp.ConvKernel
	norm := inkling2ConvNorm(kn, dyn)
	out := s.tmp(n)
	if s.err == nil {
		s.fail(s.dev.Ink2Conv(x, weight, dyn, at(s.hist[slot], offset*(kn-1)), out, resid, n, max(heads, 1), kn, norm))
	}
	return out
}

// inkling2ConvNorm is the output scale of a short conv of width kn, with or without
// dynamic taps.
func inkling2ConvNorm(kn int, dyn ink2Ref) float32 {
	if dyn.T.Buf() != nil {
		return float32(1 / math.Sqrt(float64(2*kn+1)))
	}
	return float32(1 / math.Sqrt(float64(kn+1)))
}

func with(blocks []ink2Ref, cur ink2Ref) []ink2Ref {
	return append(append([]ink2Ref(nil), blocks...), cur)
}

func (s *inkling2Resident) layer(slot, ip int, cur ink2Ref, blocks *[]ink2Ref, msgs []ink2Ref, c *ink2Ctrl) ink2Ref {
	hp := s.hp
	l := &s.layers[ip]
	local := hp.LocalLayer[ip]
	h := hp.Hidden
	kvDim := hp.KVHeads * hp.HeadDim
	inpSA := cur

	in := [3]ink2Ref{cur, cur, cur}
	normed := false
	if l.resQRows > 0 && msgs == nil {
		normed = true
		if local {
			m := s.mixNorm(with(*blocks, cur), l.resQ, 1, []ink2Ref{l.attnNorm})
			in = [3]ink2Ref{m, m, m}
		} else {
			m := s.mixNorm(with(*blocks, cur), l.resQ, 3, []ink2Ref{l.qNorm, l.kNorm, l.vNorm})
			in = [3]ink2Ref{m, at(m, h), at(m, 2*h)}
		}
	} else if l.resQRows > 0 {
		if local {
			m := s.mix(with(*blocks, cur), l.resQ, 1)
			in = [3]ink2Ref{m, m, m}
		} else {
			m := s.mix(with(*blocks, cur), l.resQ, 3)
			in = [3]ink2Ref{m, at(m, h), at(m, 2*h)}
		}
	}
	if msgs != nil {
		norms := []ink2Ref{l.qNorm, l.kNorm, l.vNorm}
		if local {
			norms = []ink2Ref{l.attnNorm, l.attnNorm, l.attnNorm}
		}
		o := s.control(in[:], msgs[:3], c, []int{0, 1, 2}, norms)
		in = [3]ink2Ref{o, at(o, h), at(o, 2*h)}
		normed = true
	}
	switch {
	case normed:
	case local:
		n0 := s.rms(in[0], l.attnNorm, h, hp.NormEps)
		in = [3]ink2Ref{n0, n0, n0}
	default:
		in[0] = s.rms(in[0], l.qNorm, h, hp.NormEps)
		in[1] = s.rms(in[1], l.kNorm, h, hp.NormEps)
		in[2] = s.rms(in[2], l.vNorm, h, hp.NormEps)
	}

	q := s.mv(l.wq, in[0])
	var kDyn, vDyn ink2Ref
	if !local {
		kDyn = s.mv(l.kconv, in[1])
		vDyn = s.mv(l.vconv, in[2])
	}
	k, v := s.mv(l.wk, in[1]), s.mv(l.wv, in[2])
	kn := hp.ConvKernel
	cv := &compute.Ink2KVConv{
		KW: l.kconvW, VW: l.vconvW, KDyn: kDyn, VDyn: vDyn, Kn: kn,
		KHist: s.hist[slot], VHist: at(s.hist[slot], kvDim*(kn-1)),
		KNorm: inkling2ConvNorm(kn, kDyn), VNorm: inkling2ConvNorm(kn, vDyn),
	}
	r := s.mv(l.rel, in[0])
	sinks := s.mv(l.sink, in[0])

	p := compute.Ink2AttnParams{
		Heads: hp.Heads, KVHeads: hp.KVHeads, HeadDim: hp.HeadDim, RelLen: hp.RelLength,
		QKClip: hp.QKClip, RelClip: hp.RelClip, Eps: hp.NormEps, Global: !local,
		Alpha: hp.LogScaleAlpha, Floor: hp.LogScaleFloor, Scale: float32(1 / math.Sqrt(float64(hp.HeadDim))),
	}
	meta := at(s.in, h)
	cap := s.caps[slot]
	qkvPost, attendOp := s.dev.Ink2QKVPost, s.dev.Ink2Attend
	if s.kvHalfSlot(slot) {
		qkvPost, attendOp = s.halfDev.Ink2QKVPostHalf, s.halfDev.Ink2AttendHalf
	}
	if s.err == nil {
		s.fail(qkvPost(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], meta, p, cap, cv))
	}
	extent, window := hp.RelExtent, 0
	if local {
		extent, window = hp.SlidingWindow, hp.SlidingWindow
	}
	attn := s.tmp(hp.Heads * hp.HeadDim)
	if s.err == nil {
		s.fail(attendOp(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.scores, attn, meta, p, extent, window, cap))
	}
	gate := s.mv(l.gate, in[0])
	o := s.mvGated(l.wo, attn, gate, hp.Heads*hp.HeadDim)
	if ip%hp.AttnResBlock == 0 {
		*blocks = append(*blocks, inpSA)
		cur = s.conv(slot, o, l.attnPostConv, ink2Ref{}, ink2Ref{}, h, 0, 2*kvDim)
	} else {
		cur = s.conv(slot, o, l.attnPostConv, ink2Ref{}, inpSA, h, 0, 2*kvDim)
	}
	ffnInp := cur

	var x ink2Ref
	if msgs != nil {
		x = s.mix(with(*blocks, cur), l.ffnResQ, 1)
		x = s.control([]ink2Ref{x}, msgs[3:4], c, []int{3}, []ink2Ref{l.ffnNorm})
	} else {
		x = s.mixNorm(with(*blocks, cur), l.ffnResQ, 1, []ink2Ref{l.ffnNorm})
	}
	return s.moeConv(l, x, slot, ffnInp, 2*kvDim+h)
}

// expertMatVec runs k routed slots through e into y (rows per slot), with the
// per-expert scale applied. situ fuses the SITU activation into the input staging.
func (s *inkling2Resident) expertMatVec(e *ink2Experts, ids, x ink2Ref, xStride int, y ink2Ref, rows int, situ *compute.Ink2Situ) {
	hp := s.hp
	if s.err != nil {
		return
	}
	ok, err := s.dev.Ink2ExpertMatVec(e.stack, rows, ids, x, xStride, y, hp.ExpertsUsed, e.scale, situ)
	if err != nil || ok {
		s.fail(err)
		return
	}
	if situ != nil {
		act := s.tmp(hp.ExpertsUsed * e.padK)
		s.fail(s.dev.Ink2Situ(x, act, hp.ExpertsUsed, situ.Width, e.padK, situ.Beta, situ.LinearBeta))
		x, xStride = act, e.padK
	}
	got, err := s.dev.Ink2ReadInts(ids, hp.ExpertsUsed)
	if err != nil {
		s.fail(err)
		return
	}
	for t, id := range got {
		if s.err == nil {
			s.fail(s.dev.Ink2MatVec(e.views[id], at(x, t*xStride), at(y, t*rows)))
		}
	}
	if e.scale.T.Buf() != nil && s.err == nil {
		s.fail(s.dev.Ink2ExpertScale(y, ids, e.scale, hp.ExpertsUsed, rows))
	}
}

func (s *inkling2Resident) moe(l *ink2Layer, x ink2Ref) ink2MoE {
	hp := s.hp
	k, width := hp.ExpertsUsed, hp.ExpertWidth
	logits := s.mv(l.router, x)
	ids, wts := s.tmp(k), s.tmp(k)
	if s.err == nil {
		s.fail(s.dev.Ink2Route(logits, l.probsBias, ids, wts, l.router.rows, k, hp.ExpertWeightsScale))
	}
	gu := s.tmp(k * 2 * width)
	s.expertMatVec(&l.gateUp, ids, x, 0, gu, 2*width, nil)
	// SITU runs once per routed value here; fused into the down GEMV staging it
	// would be recomputed by every block in float64
	padK := l.down.padK
	act := s.tmp(k * padK)
	if s.err == nil {
		s.fail(s.dev.Ink2Situ(gu, act, k, width, padK, float64(hp.SituBeta), float64(hp.SituLinearBeta)))
	}
	down := s.tmp(k * hp.Hidden)
	s.expertMatVec(&l.down, ids, act, padK, down, hp.Hidden, nil)
	return ink2MoE{down: down, wts: wts}
}

// ink2MoE is a MoE output before its combine: the routed slots' down rows and weights.
type ink2MoE struct{ down, wts ink2Ref }

// moeConv runs the FFN's MoE and its post conv with the combine folded into the conv.
func (s *inkling2Resident) moeConv(l *ink2Layer, x ink2Ref, slot int, resid ink2Ref, offset int) ink2Ref {
	hp := s.hp
	e := s.moe(l, x)
	kn := hp.ConvKernel
	out := s.tmp(hp.Hidden)
	if s.err == nil {
		s.fail(s.dev.Ink2CombineConv(e.down, e.wts, hp.ExpertsUsed, l.ffnPostConv, at(s.hist[slot], offset*(kn-1)), out, resid,
			hp.Hidden, kn, float32(1/math.Sqrt(float64(kn+1)))))
	}
	return out
}

// forward enqueues one token's forward from the uploaded input row through the
// argmax; it issues no host fence unless host routing is required.
func (s *inkling2Resident) forward() { s.forwardTo(s.argIdx, ink2Ref{}) }

// forwardTo is forward with the argmax written to arg and, when logits is set,
// the logits too instead of to the arena.
func (s *inkling2Resident) forwardTo(arg, logits ink2Ref) {
	hp := s.hp
	s.off = 0
	clear(s.q8)
	cur := s.rms(s.in, s.tokNorm, hp.Hidden, hp.NormEps)
	var blocks []ink2Ref
	for il := 0; il < hp.CoreStart; il++ {
		cur = s.layer(il, il, cur, &blocks, nil, nil)
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
			if pass == 0 {
				cur = s.layer(ip, ip, cur, &blocks, nil, nil)
				continue
			}
			slot := hp.Layers + (pass-1)*core + (ip - hp.CoreStart)
			c := &s.ctrls[ip-hp.CoreStart]
			m := s.mix(with(bank, bankOut), c.msgQ, 4)
			msgs := []ink2Ref{m, at(m, hp.Hidden), at(m, 2*hp.Hidden), at(m, 3*hp.Hidden)}
			cur = s.layer(slot, ip, cur, &blocks, msgs, c)
		}
	}
	for il := hp.CoreEnd; il < hp.Layers; il++ {
		cur = s.layer(il, il, cur, &blocks, nil, nil)
	}
	x := s.mixNorm(with(blocks, cur), s.outResQ, 1, []ink2Ref{s.outNorm})
	if logits.T.Buf() == nil {
		logits = s.tmp(s.output.rows)
	}
	s.mvInto(s.output, x, logits)
	s.stepLogits = logits
	if s.err == nil {
		s.fail(s.dev.Ink2SoftcapArgmax(logits, hp.VocabUnpadded, hp.FinalLogitSoftcap, arg))
	}
}

// step consumes one token at the next position and returns the greedy next token.
func (s *inkling2Resident) step(token int) (int, error) {
	if s.report && s.decN == 0 && s.warm {
		// the window opens one eager token in, so lazily sized scratch is excluded
		if err := compute.MetalInkling2ProfileReset(); err != nil {
			s.report = false
		} else {
			s.decS0, _ = compute.MetalInkling2Stats()
			s.decT0 = time.Now()
			s.decN = -1
		}
	}
	if s.pos >= s.ctx {
		return 0, fmt.Errorf("inkling2 resident: position %d exceeds the %d-token global cache", s.pos, s.ctx)
	}
	if next, ok, err := s.pipeStep(token); ok {
		return next, err
	}
	s.embedHost(token)
	switch {
	case s.graphReady:
		s.fail(s.dev.Ink2GraphLaunch())
	case s.graph && s.pos > 0 && s.warm:
		// capture after one eager token has grown every lazily sized GEMV scratch
		if err := s.dev.Ink2GraphBegin(); err != nil {
			s.graph = false
			s.forward()
			break
		}
		s.forward()
		if err := s.dev.Ink2GraphEnd(); err != nil || s.err != nil {
			s.fail(err)
			s.graph = false
			break
		}
		s.graphReady = true
		s.fail(s.dev.Ink2GraphLaunch())
	default:
		s.forward()
		s.warm = true
	}
	s.logits = s.stepLogits
	if s.err != nil {
		return 0, s.err
	}
	got, err := s.dev.Ink2ReadInts(s.argIdx, 1)
	if err != nil {
		return 0, err
	}
	return s.finishStep(int(got[0])), nil
}

// embedHost uploads the token's embedding row and the position.
func (s *inkling2Resident) embedHost(token int) {
	hp := s.hp
	row := make([]float32, hp.Hidden+4)
	s.w.TokEmbd.Row(token, row[:hp.Hidden])
	row[hp.Hidden] = math.Float32frombits(uint32(int32(s.pos)))
	s.fail(s.dev.Ink2Write(s.in, row))
}

// finishStep advances past the step that returned next.
func (s *inkling2Resident) finishStep(next int) int {
	s.pos++
	s.pipeExpect(next)
	if s.decN != 0 {
		s.decN = max(s.decN, 0) + 1
		s.decS1, _ = compute.MetalInkling2Stats()
		s.decT1 = time.Now()
	}
	if r, ok := s.be.(interface{ Recycle() }); ok {
		// token boundary for the backend's profile and transient bookkeeping
		r.Recycle()
	}
	return next
}

// lastLogits reads back the softcapped logits of the last step or prefill.
func (s *inkling2Resident) lastLogits() ([]float32, error) {
	return s.dev.Ink2Read(s.logits, s.hp.VocabUnpadded)
}

// inkling2ResidentGreedy is Inkling2LoopGreedyTimed on the resident device forward.
func inkling2ResidentGreedy(m *Inkling2LoopModel, be compute.Backend, prompt []int, n int) ([]int, Inkling2LoopGreedyStats, error) {
	return inkling2ResidentGreedyUntil(m, be, prompt, n, -1)
}

func inkling2ResidentGreedyUntil(m *Inkling2LoopModel, be compute.Backend, prompt []int, n, eos int) ([]int, Inkling2LoopGreedyStats, error) {
	s, err := newInkling2Resident(m, be)
	if err != nil {
		return nil, Inkling2LoopGreedyStats{}, err
	}
	defer s.close()
	// a CUDA stream capture is bound to the OS thread that opened it
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	out, st, err := inkling2GreedyStepsUntil(&inkling2ResidentSession{s: s}, prompt, n, eos)
	st.PrefillChunk = s.prefillWidest
	return out, st, err
}

func (s *inkling2Resident) localAt(slot int) bool {
	return s.hp.LocalLayer[inkling2SlotLayer(s.hp, slot)]
}

// kvHalfSlot reports whether slot's K and V caches are f16: the global slots of an f16
// session. Local rings stay fp32 whatever the session stores.
func (s *inkling2Resident) kvHalfSlot(slot int) bool {
	return s.kvHalf && !s.localAt(slot)
}
