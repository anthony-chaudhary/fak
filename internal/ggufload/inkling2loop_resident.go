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
	// Inkling2ResidentCtxFitMax is the upper bound for VRAM-fit-planned context when
	// device free memory is known and global KV uses a compressed dtype (f16/fp8/nvfp4).
	Inkling2ResidentCtxFitMax = 262144
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

	// kvHalf stores the global-slot K and V caches as f16 (FAK_INKLING2_KV=f16 or auto);
	// kvAutoHalf marks auto-selected f16 (unset FAK_INKLING2_KV) so enableSpec can safely
	// yield back to fp32. halfDev is the decode op set that reads them. kvFP8 stores them
	// as OCP E4M3FN codes with per-(token, kvHead) f32 scales (FAK_INKLING2_KV=fp8).
	// kvNVFP4 stores them as block-16 packed E2M1 codes with per-16-element E4M3FN scales
	// (FAK_INKLING2_KV=nvfp4). Both share ks/vs scale refs and kvOverflow. Local
	// sliding-window rings always stay fp32.
	kvHalf       bool
	kvAutoHalf   bool
	halfDev      compute.Inkling2HalfKVDevice
	kvFP8        bool
	fp8Dev       compute.Inkling2FP8KVDevice
	kvNVFP4      bool
	nvfp4Dev     compute.Inkling2NVFP4KVDevice
	kvAsym       bool
	asymDev      compute.Inkling2AsymKVDevice
	localKVDtype Inkling2KVDType
	ks, vs       []ink2Ref
	kvOverflow   ink2Ref

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
	return newInkling2ResidentWithSpec(m, be, false)
}

func newInkling2ResidentWithSpec(m *Inkling2LoopModel, be compute.Backend, forSpec bool) (*inkling2Resident, error) {
	ctx := inkling2ResidentCtx
	switch {
	case inkling2ResidentCtxPinned > 0:
		ctx = inkling2ResidentCtxPinned
	case inkling2ResidentCtxPinned == inkling2ResidentCtxAuto:
		ctx = inkling2ResidentCtxAuto
	case inkling2ResidentCtxAutoSet(os.LookupEnv):
		ctx = inkling2ResidentCtxAuto
	}
	if limit := inkling2ResidentHardMax(be.Name()); limit > 0 && ctx > limit {
		return nil, fmt.Errorf("inkling2 resident: %d-token context exceeds the %d-token bound of backend %s", ctx, limit, be.Name())
	}
	return newInkling2ResidentSizedFor(m, be, ctx, inkling2ArenaFloats, forSpec)
}

// Inkling2ResidentCtxAutoMode pins the next resident session to derive its global
// context from live post-weight free device memory even if FAK_INKLING2_CTX is set.
const Inkling2ResidentCtxAutoMode = inkling2ResidentCtxAuto

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
// When hp carries real SlidingWindow and Hidden geometry, local sliding-window rings,
// conv histories, and base-chunk prefill arenas are included so the derived context
// never OOMs on the first batched prefill.
func Inkling2ResidentFitCtx(hp model.Inkling2LoopHParams, free int64) int {
	if hp.SlidingWindow > 0 && hp.Hidden > 0 {
		keep, scr := inkling2PrefillArenaFloats(hp, inkling2PrefillChunkBase, 0)
		prefill := 4 * int64(keep+scr+inkling2PrefillChunkBase*hp.Hidden)
		return inkling2FitResidentCtxExact(hp, free, inkling2ResidentReserve(free)+prefill, true, inkling2PrefillChunkBase, false, Inkling2ResidentCtxMax(inkling2ResidentBackend))
	}
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

// inkling2LocalSWAKVDTypeFrom parses FAK_INKLING2_LOCAL_SWA_KV (f32|f16|fp8).
func inkling2LocalSWAKVDTypeFrom(lookup func(string) (string, bool)) (Inkling2KVDType, error) {
	if lookup == nil {
		return Inkling2KVF32, nil
	}
	v, ok := lookup("FAK_INKLING2_LOCAL_SWA_KV")
	if !ok {
		return Inkling2KVF32, nil
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "f32", "fp32":
		return Inkling2KVF32, nil
	case "f16", "fp16":
		return Inkling2KVF16, nil
	case "fp8", "e4m3", "e4m3fn":
		return Inkling2KVFP8, nil
	default:
		return Inkling2KVF32, fmt.Errorf("inkling2 resident: FAK_INKLING2_LOCAL_SWA_KV=%q is not f32, f16, or fp8", v)
	}
}

// inkling2LocalSWAKVDType resolves the effective local SWA ring storage dtype on be.
func inkling2LocalSWAKVDType(be compute.Backend, lookup func(string) (string, bool)) (Inkling2KVDType, error) {
	req, err := inkling2LocalSWAKVDTypeFrom(lookup)
	if err != nil || be == nil {
		return req, err
	}
	switch req {
	case Inkling2KVFP8:
		if inkling2BackendFP8KVErr(be) == nil {
			return Inkling2KVFP8, nil
		}
		if inkling2BackendHalfKVErr(be) == nil {
			return Inkling2KVF16, nil
		}
		return Inkling2KVF32, nil
	case Inkling2KVF16:
		if inkling2BackendHalfKVErr(be) == nil {
			return Inkling2KVF16, nil
		}
		return Inkling2KVF32, nil
	default:
		return Inkling2KVF32, nil
	}
}

// inkling2GraphEligible reports whether CUDA graph capture/replay is eligible for
// resident decode/verify on be with the given global KV dtype.
func inkling2GraphEligible(be compute.Backend, hostRouting bool, dtype Inkling2KVDType, lookup func(string) (string, bool)) bool {
	if be == nil || hostRouting || be.Name() == "metal" {
		return false
	}
	if !inkling2DefaultOn(be, "FAK_INKLING2_GRAPH", lookup) {
		return false
	}
	switch dtype {
	case Inkling2KVF32:
		return true
	case Inkling2KVF16:
		return inkling2BackendHalfKVErr(be) == nil
	case Inkling2KVFP8:
		return inkling2BackendFP8KVErr(be) == nil
	case Inkling2KVNVFP4:
		return inkling2BackendNVFP4KVErr(be) == nil
	case Inkling2KVAsym:
		return inkling2BackendAsymKVErr(be) == nil
	default:
		return false
	}
}

// inkling2LocalKVSlotBytes is the device bytes of one local SWA slot's K and V
// rings at cap positions for localDtype (f32, f16, or fp8 with f32 scales).
func inkling2LocalKVSlotBytes(hp model.Inkling2LoopHParams, cap int, localDtype Inkling2KVDType) int64 {
	kvDim := int64(hp.KVHeads * hp.HeadDim)
	switch localDtype {
	case Inkling2KVFP8:
		codeBytes := int64(compute.Ink2FP8KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)) * 4
		scaleBytes := int64(compute.Ink2FP8KVScaleFloats(cap, hp.KVHeads)) * 4
		return 2 * (codeBytes + scaleBytes)
	case Inkling2KVF16:
		return 2 * int64(cap) * kvDim * 2
	default:
		return 2 * int64(cap) * kvDim * 4
	}
}

// inkling2GlobalKVSlotBytes is the device bytes of one global slot's K and V caches
// (plus per-(token, kvHead) f32 scales for FP8, or per-16-element E4M3FN scales for
// NVFP4) at cap positions.
func inkling2GlobalKVSlotBytes(hp model.Inkling2LoopHParams, cap int, dtype Inkling2KVDType) int64 {
	kvDim := int64(hp.KVHeads * hp.HeadDim)
	switch dtype {
	case Inkling2KVAsym:
		kCodeBytes := int64(compute.Ink2FP8KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)) * 4
		vCodeBytes := int64(compute.Ink2NVFP4KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)) * 4
		scaleBytes := int64(compute.Ink2NVFP4KVScaleFloats(cap, hp.KVHeads, hp.HeadDim)) * 4
		return kCodeBytes + vCodeBytes + 2*scaleBytes
	case Inkling2KVNVFP4:
		codeBytes := int64(compute.Ink2NVFP4KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)) * 4
		scaleBytes := int64(compute.Ink2NVFP4KVScaleFloats(cap, hp.KVHeads, hp.HeadDim)) * 4
		return 2 * (codeBytes + scaleBytes)
	case Inkling2KVFP8:
		codeBytes := int64(compute.Ink2FP8KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)) * 4
		scaleBytes := int64(compute.Ink2FP8KVScaleFloats(cap, hp.KVHeads)) * 4
		return 2 * (codeBytes + scaleBytes)
	case Inkling2KVF16:
		return 2 * int64(cap) * kvDim * 2
	default:
		return 2 * int64(cap) * kvDim * 4
	}
}

// inkling2ResidentPerPosBytes is the device cost of one global-slot cache position:
// its f32 K and V rows in every global slot, plus the attention score row, which
// grows with the global context.
func inkling2ResidentPerPosBytes(hp model.Inkling2LoopHParams) int64 {
	return inkling2ResidentPerPosBytesFor(hp, Inkling2KVF32)
}

// inkling2ResidentPerPosBytesDtype is inkling2ResidentPerPosBytes with half selecting
// f16 for the global-slot K and V caches; the attention score row stays float32.
func inkling2ResidentPerPosBytesDtype(hp model.Inkling2LoopHParams, half bool) int64 {
	dtype := Inkling2KVF32
	if half {
		dtype = Inkling2KVF16
	}
	return inkling2ResidentPerPosBytesFor(hp, dtype)
}

// inkling2ResidentPerPosBytesFor is inkling2ResidentPerPosBytes sized for dtype's
// global-slot caches; the shared score row stays float32.
func inkling2ResidentPerPosBytesFor(hp model.Inkling2LoopHParams, dtype Inkling2KVDType) int64 {
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	var global int64
	for slot := 0; slot < slots; slot++ {
		if !hp.LocalLayer[inkling2SlotLayer(hp, slot)] {
			global++
		}
	}
	return global*inkling2GlobalKVSlotBytes(hp, 1, dtype) + 2*4*int64(hp.Heads)
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

// inkling2FitResidentCtxExact is the largest global context in 256-position steps
// whose full KV footprint (global slots at the active dtype, local sliding-window rings,
// conv histories, and score rows) fits in free minus reserve, clamped to ceiling.
// It returns 0 when even Inkling2ResidentCtxMin does not fit.
func inkling2FitResidentCtxExact(hp model.Inkling2LoopHParams, free, reserve int64, batched bool, chunk int, half bool, ceiling int) int {
	budget := free - reserve
	if budget <= 0 || ceiling < Inkling2ResidentCtxMin {
		return 0
	}
	n := inkling2MaxResidentCtxDtype(hp, batched, chunk, half, ceiling, budget)
	n = n / 256 * 256
	if n < Inkling2ResidentCtxMin {
		return 0
	}
	return n
}

// inkling2AutoCtxMax is the upper bound for auto-derived resident context on backend.
// FP32 keeps the 16384 ceiling; compressed global KV (f16/fp8/nvfp4) allows the fit
// planner to scale up to Inkling2ResidentCtxFitMax (262144), still bounded by the
// backend's hard max and exact free-VRAM budget.
func inkling2AutoCtxMax(backend string, half bool) int {
	ceil := inkling2AutoCtxCeiling
	if half {
		ceil = Inkling2ResidentCtxFitMax
	}
	if hard := inkling2ResidentHardMax(backend); hard > 0 && ceil > hard {
		ceil = hard
	}
	return ceil
}

// inkling2ResolveAutoCtx sizes the global cache from the device memory free now,
// after the weights bound. A backend that cannot report free memory keeps the
// fixed default rather than guess.
func inkling2ResolveAutoCtx(be compute.Backend, hp model.Inkling2LoopHParams, prefill int64, half bool) (int, error) {
	dtype := Inkling2KVF32
	if half {
		dtype = Inkling2KVF16
	}
	return inkling2ResolveAutoCtxFor(be, hp, false, 0, inkling2ResidentFixedBytes(hp, inkling2ArenaFloats)+prefill, dtype)
}

// inkling2ResidentKVBudget is the single shared VRAM budget helper used by both
// auto-context resolution (inkling2ResolveAutoCtxFor / inkling2AutoFitCtxFor) and
// session ring allocation (inkling2KVFitFor / newInkling2ResidentRingFor) so
// auto-ctx never drifts from ring allocation.
func inkling2ResidentKVBudget(free, other int64) (budget, left int64) {
	budget = free - inkling2DeviceHeadroom(free)
	left = budget - other
	return budget, left
}

// inkling2ResolveAutoCtxFor sizes the global cache from free device memory using the
// exact KV + other fit formula so the resolved auto context always passes
// inkling2FitResidentKVFor.
func inkling2ResolveAutoCtxFor(be compute.Backend, hp model.Inkling2LoopHParams, batched bool, chunk int, other int64, dtype Inkling2KVDType) (int, error) {
	return inkling2ResolveAutoCtxForLocal(be, hp, batched, chunk, other, dtype, Inkling2KVF32)
}

func inkling2ResolveAutoCtxForLocal(be compute.Backend, hp model.Inkling2LoopHParams, batched bool, chunk int, other int64, dtype, localDtype Inkling2KVDType) (int, error) {
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known || free == compute.FreeUnknown || free < 0 {
		fmt.Fprintf(os.Stderr, "inkling2 resident: device free memory unknown, context %d (default)\n", Inkling2ResidentCtxDefault)
		return Inkling2ResidentCtxDefault, nil
	}
	perPos := inkling2ResidentPerPosBytesFor(hp, dtype)
	_, left := inkling2ResidentKVBudget(free, other)
	ceil := inkling2AutoCtxMax(be.Name(), dtype != Inkling2KVF32)
	ctx := 0
	if left > 0 {
		ctx = inkling2MaxResidentCtxForLocal(hp, batched, chunk, dtype, localDtype, ceil, left)
		ctx = ctx / 256 * 256
	}
	if ctx < Inkling2ResidentCtxMin {
		return 0, fmt.Errorf("inkling2 resident: %d free bytes after weights hold no %d-token global cache at %d bytes per position; set %s to a smaller context",
			free, Inkling2ResidentCtxMin, perPos, "FAK_INKLING2_CTX")
	}
	fmt.Fprintf(os.Stderr, "inkling2 resident: context %d (auto: %.2f GiB free after weights, %d bytes per position, kv=%s, local=%s)\n",
		ctx, float64(free)/(1<<30), perPos, dtype, localDtype)
	return ctx, nil
}

// Inkling2ResolveServeCtx resolves the serving context window for m on be:
// an explicit requested > 0 is returned (clamped to the backend hard max if any);
// an explicit FAK_INKLING2_CTX returns that parsed value; otherwise free device
// memory is probed and fitted with inkling2FitResidentCtxExact (reserving headroom,
// per-token arena, local sliding-window rings, conv histories, and prefill arenas)
// at the active KV dtype (auto-FP16 when supported and spec decode is off).
func Inkling2ResolveServeCtx(m *Inkling2LoopModel, be compute.Backend, requested int, lookup func(string) (string, bool)) int {
	if requested > 0 {
		if be != nil {
			if hard := inkling2ResidentHardMax(be.Name()); hard > 0 && requested > hard {
				return hard
			}
		}
		return requested
	}
	if lookup == nil {
		lookup = os.LookupEnv
	}
	if !inkling2ResidentCtxAutoSet(lookup) {
		return inkling2ResidentCtxFrom(lookup)
	}
	if m == nil || be == nil {
		return Inkling2LoopResidentContext
	}
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known || free == compute.FreeUnknown || free <= 0 {
		return Inkling2LoopResidentContext
	}
	dtype, err := inkling2ResidentKVDTypeFor(m, be, lookup)
	if err != nil {
		dtype = Inkling2KVF32
	}
	fit := inkling2AutoFitCtxFor(m, be, free, lookup, dtype)
	if fit < Inkling2ResidentCtxMin {
		return Inkling2LoopResidentContext
	}
	return max(Inkling2LoopResidentContext, fit)
}

func inkling2AutoFitCtxFor(m *Inkling2LoopModel, be compute.Backend, free int64, lookup func(string) (string, bool), dtype Inkling2KVDType) int {
	if m == nil || be == nil || free <= 0 || free == compute.FreeUnknown {
		return 0
	}
	_, hasPrefill := be.(compute.Inkling2PrefillDevice)
	batched := hasPrefill && (inkling2BatchedPrefill() || inkling2SpecDepthFor(m) > 0)
	chunk := inkling2PrefillChunkWantedOn(be, lookup)
	prefillChunk := 0
	if batched && m.HP.Hidden > 0 {
		prefillChunk = max(chunk, inkling2PrefillChunkBase)
	}
	other := inkling2ResidentOtherBytes(m.HP, inkling2ArenaFloats, prefillChunk, 0)
	_, left := inkling2ResidentKVBudget(free, other)
	localDtype, err := inkling2LocalSWAKVDType(be, lookup)
	if err != nil {
		localDtype = Inkling2KVF32
	}
	ceil := inkling2AutoCtxMax(be.Name(), dtype != Inkling2KVF32)
	fit := 0
	if left > 0 {
		fit = inkling2MaxResidentCtxForLocal(m.HP, batched, prefillChunk, dtype, localDtype, ceil, left)
		fit = fit / 256 * 256
	}
	return fit
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
	return newInkling2ResidentSizedFor(m, be, ctx, arenaFloats, false)
}

func newInkling2ResidentSizedFor(m *Inkling2LoopModel, be compute.Backend, ctx, arenaFloats int, forSpec bool) (*inkling2Resident, error) {
	wanted := inkling2PrefillChunkWantedOn(be, os.LookupEnv)
	// a cohort binding (ctx 0) holds no per-sequence state: each cohort sizes its own
	ring := inkling2PrefillChunkBase
	if ctx != 0 && !inkling2ScalarWideHeld(wanted) {
		ring = wanted
	}
	s, err := newInkling2ResidentRingFor(m, be, ctx, arenaFloats, ring, wanted, forSpec)
	if err != nil && ring > inkling2PrefillChunkBase && inkling2ChunkRefusal(err) {
		// the refused build freed everything it took, so the base build starts from
		// the memory a session that never opted in would have had
		inkling2ScalarWideRefused(err, ring)
		inkling2TrimIdle(be)
		s, err = newInkling2ResidentRingFor(m, be, ctx, arenaFloats, inkling2PrefillChunkBase, wanted, forSpec)
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
	dtype := Inkling2KVF32
	if half {
		dtype = Inkling2KVF16
	}
	return inkling2ResidentKVBytesFor(hp, ctx, batched, chunk, dtype)
}

// inkling2ResidentKVBytesFor is inkling2ResidentKVBytes sized for dtype on the global
// slots (f32, f16, fp8, nvfp4, or fp8k_nvfp4v); the local SWA rings use fp32.
func inkling2ResidentKVBytesFor(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, dtype Inkling2KVDType) int64 {
	return inkling2ResidentKVBytesForLocal(hp, ctx, batched, chunk, dtype, Inkling2KVF32)
}

// inkling2ResidentKVBytesForLocal is inkling2ResidentKVBytesFor with localDtype
// sizing the local SWA rings (f32, f16, or fp8 with f32 scales).
func inkling2ResidentKVBytesForLocal(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, dtype, localDtype Inkling2KVDType) int64 {
	if ctx <= 0 {
		return 0
	}
	kvDim := hp.KVHeads * hp.HeadDim
	channels := 2*kvDim + 2*hp.Hidden
	var bytes int64
	maxCap := hp.SlidingWindow
	for slot, c := range inkling2ResidentCaps(hp, ctx, batched, chunk) {
		maxCap = max(maxCap, c)
		histBytes := int64(channels) * int64(hp.ConvKernel-1) * 4
		if inkling2SlotIsLocal(hp, slot) {
			bytes += inkling2LocalKVSlotBytes(hp, c, localDtype) + histBytes
		} else {
			bytes += inkling2GlobalKVSlotBytes(hp, c, dtype) + histBytes
		}
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
// buffers plus, when prefillChunk > 0, the prefill arenas (base or wide) and the
// chunk input.
func inkling2ResidentOtherBytes(hp model.Inkling2LoopHParams, arenaFloats, prefillChunk, padK int) int64 {
	other := inkling2ResidentFixedBytes(hp, arenaFloats)
	if prefillChunk > 0 {
		keep, scr := inkling2PrefillArenaFloats(hp, prefillChunk, padK)
		other += 4 * int64(keep+scr+prefillChunk*hp.Hidden)
	}
	return other
}

// inkling2ResidentTotalBytes is every device allocation a resident session makes at
// ctx tokens: the KV caches and the non-KV buffers of inkling2ResidentOtherBytes.
func inkling2ResidentTotalBytes(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk, arenaFloats, prefillChunk, padK int) int64 {
	return inkling2ResidentTotalBytesFor(hp, ctx, batched, chunk, Inkling2KVF32, arenaFloats, prefillChunk, padK)
}

// inkling2ResidentTotalBytesFor is inkling2ResidentTotalBytes sized for dtype's global KV.
func inkling2ResidentTotalBytesFor(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, dtype Inkling2KVDType, arenaFloats, prefillChunk, padK int) int64 {
	return inkling2ResidentKVBytesFor(hp, ctx, batched, chunk, dtype) + inkling2ResidentOtherBytes(hp, arenaFloats, prefillChunk, padK)
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
	dtype := Inkling2KVF32
	if half {
		dtype = Inkling2KVF16
	}
	return inkling2FitResidentKVFor(be, hp, ctx, batched, chunk, dtype, other)
}

// inkling2FitResidentKVFor is inkling2FitResidentKV sized for dtype's global-slot caches.
func inkling2FitResidentKVFor(be compute.Backend, hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, dtype Inkling2KVDType, other int64) (int64, error) {
	return inkling2FitResidentKVForLocal(be, hp, ctx, batched, chunk, dtype, Inkling2KVF32, other)
}

func inkling2FitResidentKVForLocal(be compute.Backend, hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, dtype, localDtype Inkling2KVDType, other int64) (int64, error) {
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known || free == compute.FreeUnknown || free < 0 {
		free = compute.FreeUnknown
	}
	return inkling2KVFitForLocal(hp, ctx, batched, chunk, dtype, localDtype, other, free)
}

// inkling2KVFitFor is inkling2KVFit sized for dtype's global-slot caches.
func inkling2KVFitFor(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, dtype Inkling2KVDType, other, free int64) (int64, error) {
	return inkling2KVFitForLocal(hp, ctx, batched, chunk, dtype, Inkling2KVF32, other, free)
}

func inkling2KVFitForLocal(hp model.Inkling2LoopHParams, ctx int, batched bool, chunk int, dtype, localDtype Inkling2KVDType, other, free int64) (int64, error) {
	kv := inkling2ResidentKVBytesForLocal(hp, ctx, batched, chunk, dtype, localDtype)
	need := kv + other
	if free == compute.FreeUnknown {
		ceiling := Inkling2ResidentCtxMax(inkling2ResidentBackend)
		if ctx <= ceiling {
			return kv, nil
		}
		return kv, &Inkling2KVOverBudgetError{Ctx: ctx, MaxCtx: ceiling, NeedBytes: need, FreeBytes: free}
	}
	budget, left := inkling2ResidentKVBudget(free, other)
	if need <= budget {
		return kv, nil
	}
	maxCtx := 0
	if left >= 0 {
		maxCtx = inkling2MaxResidentCtxForLocal(hp, batched, chunk, dtype, localDtype, ctx-1, left)
	}
	if maxCtx < Inkling2ResidentCtxMin {
		maxCtx = 0
	}
	return kv, &Inkling2KVOverBudgetError{Ctx: ctx, MaxCtx: maxCtx, NeedBytes: need, FreeBytes: free}
}

// inkling2MaxResidentCtx is the largest ctx in [0, ceil] whose KV fits budget bytes,
// 0 when none does. KV is monotone in ctx, so bisection finds the exact bound.
func inkling2MaxResidentCtx(hp model.Inkling2LoopHParams, batched bool, chunk, ceil int, budget int64) int {
	return inkling2MaxResidentCtxDtype(hp, batched, chunk, false, ceil, budget)
}

func inkling2MaxResidentCtxDtype(hp model.Inkling2LoopHParams, batched bool, chunk int, half bool, ceil int, budget int64) int {
	dtype := Inkling2KVF32
	if half {
		dtype = Inkling2KVF16
	}
	return inkling2MaxResidentCtxFor(hp, batched, chunk, dtype, ceil, budget)
}

func inkling2MaxResidentCtxFor(hp model.Inkling2LoopHParams, batched bool, chunk int, dtype Inkling2KVDType, ceil int, budget int64) int {
	return inkling2MaxResidentCtxForLocal(hp, batched, chunk, dtype, Inkling2KVF32, ceil, budget)
}

func inkling2MaxResidentCtxForLocal(hp model.Inkling2LoopHParams, batched bool, chunk int, dtype, localDtype Inkling2KVDType, ceil int, budget int64) int {
	lo, hi := 0, ceil
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if inkling2ResidentKVBytesForLocal(hp, mid, batched, chunk, dtype, localDtype) <= budget {
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
	return newInkling2ResidentRingFor(m, be, ctx, arenaFloats, ring, wanted, false)
}

func newInkling2ResidentRingFor(m *Inkling2LoopModel, be compute.Backend, ctx, arenaFloats, ring, wanted int, forSpec bool) (*inkling2Resident, error) {
	dev, ok := be.(compute.Inkling2Device)
	if !ok {
		return nil, fmt.Errorf("inkling2 resident: backend %s has no inkling2 device ops", be.Name())
	}
	hp, w := m.HP, m.Weights
	if _, kvErr := Inkling2KVDTypeFrom(os.LookupEnv); kvErr != nil {
		return nil, kvErr
	}
	localKVDtype, localKVErr := inkling2LocalSWAKVDType(be, os.LookupEnv)
	if localKVErr != nil {
		return nil, localKVErr
	}
	s := &inkling2Resident{hp: hp, w: w, be: be, dev: dev, localKVDtype: localKVDtype, q8: map[int]ink2Ref{}, arenaFloats: arenaFloats, chunkMax: ring, chunkWanted: wanted}
	s.report = metalgemm.Ink2ReportRequested(os.Getenv(metalgemm.Ink2ProfileEnv))
	_, hasPrefill := be.(compute.Inkling2PrefillDevice)
	s.batched = hasPrefill && (inkling2BatchedPrefill() || forSpec || inkling2SpecDepthFor(m) > 0)
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

	for i := range s.layers {
		for _, e := range []ink2Experts{s.layers[i].gateUp, s.layers[i].down} {
			if !inkling2IndexedExpert(e.stack.Dtype) {
				s.hostRouting = true
			}
		}
	}
	if ctx != 0 {
		dtype, autoKV, err := inkling2ResidentKVDTypeForMode(m, be, os.LookupEnv, forSpec)
		if err != nil {
			s.close()
			return nil, err
		}
		switch dtype {
		case Inkling2KVAsym:
			s.kvAsym = true
			s.asymDev, _ = be.(compute.Inkling2AsymKVDevice)
		case Inkling2KVNVFP4:
			s.kvNVFP4 = true
			s.nvfp4Dev, _ = be.(compute.Inkling2NVFP4KVDevice)
		case Inkling2KVFP8:
			s.kvFP8 = true
			s.kvAutoHalf = autoKV
			s.fp8Dev, _ = be.(compute.Inkling2FP8KVDevice)
		case Inkling2KVF16:
			s.kvHalf = true
			s.kvAutoHalf = autoKV
			s.halfDev, _ = be.(compute.Inkling2HalfKVDevice)
		}
		if s.localKVDtype == Inkling2KVFP8 && s.fp8Dev == nil {
			s.fp8Dev, _ = be.(compute.Inkling2FP8KVDevice)
		}
		if s.localKVDtype == Inkling2KVF16 && s.halfDev == nil {
			s.halfDev, _ = be.(compute.Inkling2HalfKVDevice)
		}
	}
	if ctx == inkling2ResidentCtxAuto {
		autoChunk := 0
		if s.batched && !s.hostRouting {
			autoChunk = max(s.chunkMax, inkling2PrefillChunkBase)
		}
		autoOther := inkling2ResidentOtherBytes(hp, arenaFloats, autoChunk, s.maxPadK())
		resolved, err := inkling2ResolveAutoCtxForLocal(be, hp, s.batched, autoChunk, autoOther, s.kvDType(), s.localKVDtype)
		if err != nil {
			s.close()
			return nil, err
		}
		ctx = resolved
	}
	s.ctx = ctx
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	kvDim := hp.KVHeads * hp.HeadDim
	channels := 2*kvDim + 2*hp.Hidden
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
		prefillChunk := 0
		if s.batched && !s.hostRouting {
			prefillChunk = s.chunkMax
		}
		other := inkling2ResidentOtherBytes(hp, arenaFloats, prefillChunk, s.maxPadK())
		var err error
		if kvBytes, err = inkling2FitResidentKVForLocal(be, hp, ctx, s.batched, s.chunkMax, s.kvDType(), s.localKVDtype, other); err != nil {
			s.close()
			if wide {
				// the wide arenas may be what pushed the fit over; the base build may fit
				return nil, inkling2DeviceRefused{err}
			}
			return nil, err
		}
	}
	s.kc, s.vc, s.hist = make([]ink2Ref, slots), make([]ink2Ref, slots), make([]ink2Ref, slots)
	s.ks, s.vs = make([]ink2Ref, slots), make([]ink2Ref, slots)
	s.caps = inkling2ResidentCaps(hp, ctx, s.batched, s.chunkMax)
	s.baseCaps = inkling2ResidentCaps(hp, ctx, s.batched, inkling2PrefillChunkBase)
	maxCap := hp.SlidingWindow
	for slot, cap := range s.caps {
		maxCap = max(maxCap, cap)
		if ctx > 0 {
			switch {
			case s.kvAsymSlot(slot):
				kCodeFloats := compute.Ink2FP8KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)
				vCodeFloats := compute.Ink2NVFP4KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)
				scaleFloats := compute.Ink2NVFP4KVScaleFloats(cap, hp.KVHeads, hp.HeadDim)
				s.kc[slot], s.vc[slot] = s.alloc(kCodeFloats), s.alloc(vCodeFloats)
				s.ks[slot], s.vs[slot] = s.alloc(scaleFloats), s.alloc(scaleFloats)
			case s.kvNVFP4Slot(slot):
				codeFloats := compute.Ink2NVFP4KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)
				scaleFloats := compute.Ink2NVFP4KVScaleFloats(cap, hp.KVHeads, hp.HeadDim)
				s.kc[slot], s.vc[slot] = s.alloc(codeFloats), s.alloc(codeFloats)
				s.ks[slot], s.vs[slot] = s.alloc(scaleFloats), s.alloc(scaleFloats)
			case s.kvFP8Slot(slot):
				codeFloats := compute.Ink2FP8KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)
				scaleFloats := compute.Ink2FP8KVScaleFloats(cap, hp.KVHeads)
				s.kc[slot], s.vc[slot] = s.alloc(codeFloats), s.alloc(codeFloats)
				s.ks[slot], s.vs[slot] = s.alloc(scaleFloats), s.alloc(scaleFloats)
			default:
				kvFloats := inkling2BatchKVFloats(cap*kvDim, s.kvHalfSlot(slot))
				s.kc[slot], s.vc[slot] = s.alloc(kvFloats), s.alloc(kvFloats)
			}
			s.hist[slot] = s.alloc(channels * (hp.ConvKernel - 1))
		}
	}
	if ctx > 0 {
		s.scores = s.alloc(2 * hp.Heads * maxCap)
		if s.kvFP8 || s.kvNVFP4 || s.kvAsym || s.localKVDtype == Inkling2KVFP8 {
			s.kvOverflow = s.alloc(1)
		}
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
	s.graph = inkling2GraphEligible(be, s.hostRouting, s.kvDType(), os.LookupEnv)
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

func (s *inkling2Resident) hasAnyCapture() bool {
	if s.graphReady || s.verifyReady || s.keepReady {
		return true
	}
	for i := range s.tailReady {
		if s.tailReady[i] || s.tailKeepReady[i] {
			return true
		}
	}
	if d, ok := s.draft.(*ink2DeviceDraft); ok && d != nil && d.ready {
		return true
	}
	return false
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
		dev := inkling2DeviceFor(s.dev)
		cur := dev.Owner()
		if cur == nil || inkling2SameOwner(cur, s) || s.hasAnyCapture() || s.warm || s.verifyWarm {
			dev.Reset()
		}
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
	if s.err == nil {
		switch {
		case s.kvAsymSlot(slot):
			s.fail(s.asymDev.Ink2QKVPostAsym(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.kvOverflow, meta, p, cap, cv))
		case s.kvNVFP4Slot(slot):
			s.fail(s.nvfp4Dev.Ink2QKVPostNVFP4(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.kvOverflow, meta, p, cap, cv))
		case s.kvFP8Slot(slot):
			s.fail(s.fp8Dev.Ink2QKVPostFP8(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.kvOverflow, meta, p, cap, cv))
		case s.kvHalfSlot(slot):
			s.fail(s.halfDev.Ink2QKVPostHalf(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], meta, p, cap, cv))
		default:
			s.fail(s.dev.Ink2QKVPost(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], meta, p, cap, cv))
		}
	}
	extent, window := hp.RelExtent, 0
	if local {
		extent, window = hp.SlidingWindow, hp.SlidingWindow
	}
	attn := s.tmp(hp.Heads * hp.HeadDim)
	if s.err == nil {
		switch {
		case s.kvAsymSlot(slot):
			s.fail(s.asymDev.Ink2AttendAsym(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.scores, attn, meta, p, extent, window, cap))
		case s.kvNVFP4Slot(slot):
			s.fail(s.nvfp4Dev.Ink2AttendNVFP4(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.scores, attn, meta, p, extent, window, cap))
		case s.kvFP8Slot(slot):
			s.fail(s.fp8Dev.Ink2AttendFP8(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot], s.scores, attn, meta, p, extent, window, cap))
		case s.kvHalfSlot(slot):
			s.fail(s.halfDev.Ink2AttendHalf(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.scores, attn, meta, p, extent, window, cap))
		default:
			s.fail(s.dev.Ink2Attend(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.scores, attn, meta, p, extent, window, cap))
		}
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
	if overflow, oerr := s.checkKVOverflow(); oerr != nil {
		return 0, oerr
	} else if overflow {
		if s.kvNVFP4 || s.kvAsym {
			return 0, ErrInkling2NVFP4ScaleOverflow
		}
		return 0, ErrInkling2FP8ScaleOverflow
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

// kvHalfSlot reports whether slot's K and V caches are f16: either global slots
// of an f16 session or local SWA rings when FAK_INKLING2_LOCAL_SWA_KV=f16.
func (s *inkling2Resident) kvHalfSlot(slot int) bool {
	if s.localAt(slot) {
		return s.localKVDtype == Inkling2KVF16
	}
	return s.kvHalf
}

// kvFP8Slot reports whether slot's K and V caches are OCP E4M3FN with per-(token, kvHead)
// f32 scales: either global slots of an fp8 session or local SWA rings when
// FAK_INKLING2_LOCAL_SWA_KV=fp8.
func (s *inkling2Resident) kvFP8Slot(slot int) bool {
	if s.localAt(slot) {
		return s.localKVDtype == Inkling2KVFP8
	}
	return s.kvFP8
}

// kvAsymSlot reports whether slot's K cache is block-16 FP8 and V cache is block-16 NVFP4
// with per-16-element E4M3FN scales: the global slots of an fp8k_nvfp4v session.
func (s *inkling2Resident) kvAsymSlot(slot int) bool {
	return s.kvAsym && !s.localAt(slot)
}

// kvNVFP4Slot reports whether slot's K and V caches are block-16 packed E2M1 with
// per-16-element E4M3FN scales: the global slots of an nvfp4 session. Local rings stay fp32.
func (s *inkling2Resident) kvNVFP4Slot(slot int) bool {
	return s.kvNVFP4 && !s.localAt(slot)
}

// kvDType returns the active global-slot KV dtype of the session.
func (s *inkling2Resident) kvDType() Inkling2KVDType {
	switch {
	case s.kvAsym:
		return Inkling2KVAsym
	case s.kvNVFP4:
		return Inkling2KVNVFP4
	case s.kvFP8:
		return Inkling2KVFP8
	case s.kvHalf:
		return Inkling2KVF16
	default:
		return Inkling2KVF32
	}
}

// checkFP8Overflow reads the device overflow flag when the session uses FP8 global KV.
func (s *inkling2Resident) checkFP8Overflow() (bool, error) {
	if !s.kvFP8 {
		return false, nil
	}
	return s.checkKVOverflow()
}

// checkKVOverflow reads the device overflow flag when the session uses FP8, NVFP4, or asymmetric KV.
func (s *inkling2Resident) checkKVOverflow() (bool, error) {
	if (!s.kvFP8 && !s.kvNVFP4 && !s.kvAsym && s.localKVDtype != Inkling2KVFP8) || s.kvOverflow.T.Buf() == nil {
		return false, nil
	}
	flag, err := s.dev.Ink2ReadInts(s.kvOverflow, 1)
	if err != nil {
		return false, err
	}
	return len(flag) > 0 && flag[0] != 0, nil
}

// fallbackNVFP4ToHigher releases the NVFP4 global-slot buffers and reallocates global
// slots as FP8 (when available) or FP16, clearing conv histories and resetting position
// to 0 so a cold prefill that hit NVFP4 scale overflow can re-run cleanly.
func (s *inkling2Resident) fallbackNVFP4ToHigher() error {
	useFP8 := inkling2BackendFP8KVErr(s.be) == nil
	if !useFP8 {
		if err := inkling2BackendHalfKVErr(s.be); err != nil {
			return fmt.Errorf("%w: no fp8 or f16 fallback available (%v)", ErrInkling2NVFP4ScaleOverflow, err)
		}
	}
	target := "f16"
	if useFP8 {
		target = "fp8"
	}
	inkling2KVNVFP4OverflowLog(target)
	s.dropCaptures()
	hp := s.hp
	kvDim := hp.KVHeads * hp.HeadDim
	for slot, cap := range s.caps {
		if !s.kvNVFP4Slot(slot) && !s.kvAsymSlot(slot) {
			continue
		}
		s.release(s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot])
		s.ks[slot], s.vs[slot] = ink2Ref{}, ink2Ref{}
		if useFP8 {
			codeFloats := compute.Ink2FP8KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)
			scaleFloats := compute.Ink2FP8KVScaleFloats(cap, hp.KVHeads)
			s.kc[slot], s.vc[slot] = s.alloc(codeFloats), s.alloc(codeFloats)
			s.ks[slot], s.vs[slot] = s.alloc(scaleFloats), s.alloc(scaleFloats)
		} else {
			kvFloats := inkling2BatchKVFloats(cap*kvDim, true)
			s.kc[slot], s.vc[slot] = s.alloc(kvFloats), s.alloc(kvFloats)
		}
	}
	s.release(s.kvOverflow)
	s.kvOverflow = ink2Ref{}
	s.kvNVFP4 = false
	s.nvfp4Dev = nil
	s.kvAsym = false
	s.asymDev = nil
	if useFP8 {
		s.kvFP8 = true
		s.fp8Dev, _ = s.be.(compute.Inkling2FP8KVDevice)
		s.kvOverflow = s.alloc(1)
	} else {
		s.kvHalf = true
		s.halfDev, _ = s.be.(compute.Inkling2HalfKVDevice)
	}
	zero := make([]float32, s.histFloats())
	for slot := range s.hist {
		s.fail(s.dev.Ink2Write(s.hist[slot], zero))
	}
	s.pos = 0
	return s.err
}

// fallbackFP8ToFP16 releases the FP8 global-slot buffers, allocates FP16 global-slot
// caches in their place, clears conv histories, and resets position to 0 so a cold
// prefill that hit FP8 scale overflow can re-run cleanly in FP16.
func (s *inkling2Resident) fallbackFP8ToFP16() error {
	if err := inkling2BackendHalfKVErr(s.be); err != nil {
		return fmt.Errorf("%w: no f16 fallback available (%v)", ErrInkling2FP8ScaleOverflow, err)
	}
	inkling2KVFP8OverflowLog()
	s.dropCaptures()
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	for slot, cap := range s.caps {
		if s.localAt(slot) || !s.kvFP8Slot(slot) {
			continue
		}
		s.release(s.kc[slot], s.vc[slot], s.ks[slot], s.vs[slot])
		s.ks[slot], s.vs[slot] = ink2Ref{}, ink2Ref{}
		kvFloats := inkling2BatchKVFloats(cap*kvDim, true)
		s.kc[slot], s.vc[slot] = s.alloc(kvFloats), s.alloc(kvFloats)
	}
	if s.localKVDtype != Inkling2KVFP8 {
		s.release(s.kvOverflow)
		s.kvOverflow = ink2Ref{}
	}
	s.kvFP8 = false
	if s.localKVDtype != Inkling2KVFP8 {
		s.fp8Dev = nil
	}
	s.kvHalf = true
	s.halfDev, _ = s.be.(compute.Inkling2HalfKVDevice)
	zero := make([]float32, s.histFloats())
	for slot := range s.hist {
		s.fail(s.dev.Ink2Write(s.hist[slot], zero))
	}
	s.pos = 0
	return s.err
}

// downgradeAutoHalfToFP32 reallocates auto-selected f16/fp8 global-slot KV caches to fp32
// at position 0 so speculative decode (whose verify/draft ops read fp32 caches) can
// run without failing when FAK_INKLING2_KV was left unset.
func (s *inkling2Resident) downgradeAutoHalfToFP32() error {
	if (!s.kvHalf && !s.kvFP8) || !s.kvAutoHalf {
		return nil
	}
	if s.pos != 0 {
		return fmt.Errorf("inkling2 resident: cannot switch auto-%s KV to fp32 at position %d", s.kvDType(), s.pos)
	}
	s.dropCaptures()
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	for slot, cap := range s.caps {
		if s.localAt(slot) || (!s.kvHalfSlot(slot) && !s.kvFP8Slot(slot)) {
			continue
		}
		if slot < len(s.kc) && slot < len(s.vc) {
			s.release(s.kc[slot], s.vc[slot])
			if s.kvFP8Slot(slot) && slot < len(s.ks) && slot < len(s.vs) {
				s.release(s.ks[slot], s.vs[slot])
				s.ks[slot], s.vs[slot] = ink2Ref{}, ink2Ref{}
			}
			kvFloats := inkling2BatchKVFloats(cap*kvDim, false)
			s.kc[slot], s.vc[slot] = s.alloc(kvFloats), s.alloc(kvFloats)
		}
	}
	if s.kvFP8 && s.localKVDtype != Inkling2KVFP8 && s.kvOverflow.T.Buf() != nil {
		s.release(s.kvOverflow)
		s.kvOverflow = ink2Ref{}
	}
	s.kvHalf = false
	s.kvFP8 = false
	s.kvAutoHalf = false
	if s.localKVDtype != Inkling2KVF16 {
		s.halfDev = nil
	}
	if s.localKVDtype != Inkling2KVFP8 {
		s.fp8Dev = nil
	}
	return s.err
}
