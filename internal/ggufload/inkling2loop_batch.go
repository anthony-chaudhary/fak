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
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2BatchCtx is the default global-slot KV capacity of each sequence of a
// batched decode; local slots keep their sliding-window ring.
const inkling2BatchCtx = 512

// inkling2Batch decodes C sequences in lockstep through one resident weight set,
// one row per sequence. Dense, mix, controller, routing and expert ops run the
// row-batched prefill kernels over the C rows, so each weight is streamed once per
// step for every sequence, and the looped core reuses that read on every pass. The
// ops that touch sequence state (k/v conv and cache append, attention, post convs,
// argmax) run against per-sequence rows of strided arenas: each slot owns one
// [C][cap][kvDim] k and v arena and one [C][histStride] conv history, all sequences
// sharing the slot's cap, so a sequence stride replaces any pointer table. With seq
// set those ops are one launch for all C rows; otherwise they loop per row.
type inkling2Batch struct {
	s      *inkling2Resident
	p      ink2Prefill
	C, ctx int
	pos    []int
	seq    compute.Inkling2BatchDevice
	mem    compute.Inkling2BatchMemory
	need   int64
	kvHalf bool
	// half is the cohort prefill's f16-cache append and attend, set when prefill runs on f16 caches
	half compute.Inkling2PrefillHalfKV

	kc, vc, hist []ink2Ref
	caps         []int
	scores       ink2Ref
	histStride   int
	scoreStride  int

	in, meta, argIdx ink2Ref

	graph, graphReady, warm bool

	// prefillT is the token width of a cohort prefill chunk; 0 sizes the state for
	// lockstep prompts only
	prefillT int
	// capT is the prefill chunk the state was built for and capPrefillT the widest
	// prefillT its arenas and rings hold; a cohort that wants more rebuilds the state
	capT, capPrefillT int
	// sharedMin is the shortest common prompt prefix a cohort prefill computes once and
	// copies into every row sharing it; 0 prefills every row's whole prompt
	sharedMin int
	// store is the cohorts' cross-request prefix store, nil when it is off
	store *inkling2PrefixStore

	// capC and capCtx are the shape the arenas were sized for; owned are the arenas
	owned        []compute.Tensor
	capC, capCtx int
	err          error
}

// inkling2BatchGraph reports whether the batched step is captured and replayed as a
// device graph (FAK_INKLING2_BATCH_GRAPH=0 runs every step eagerly).
func inkling2BatchGraph() bool { return os.Getenv("FAK_INKLING2_BATCH_GRAPH") != "0" }

// inkling2BatchSeqOps returns the backend's sequence-batched state ops, or nil to loop
// them per sequence (no such ops, or FAK_INKLING2_BATCH_SEQOPS=0 for an A/B).
func inkling2BatchSeqOps(be any, lookup func(string) (string, bool)) compute.Inkling2BatchDevice {
	if v, ok := lookup("FAK_INKLING2_BATCH_SEQOPS"); ok && v == "0" {
		return nil
	}
	d, _ := be.(compute.Inkling2BatchDevice)
	return d
}

// newInkling2Batch binds the weights and builds the state of one C x ctx cohort whose
// longest prompt is longest tokens.
func newInkling2Batch(m *Inkling2LoopModel, be compute.Backend, C, ctx, longest int) (*inkling2Batch, error) {
	s, err := newInkling2BatchResident(m, be)
	if err != nil {
		return nil, err
	}
	b, err := newInkling2BatchState(s, C, ctx, inkling2CohortChunk(s.chunkWanted, longest))
	if err != nil {
		s.close()
		inkling2TrimIdle(be)
		return nil, err
	}
	return b, nil
}

// newInkling2BatchResident binds the weights a batch forward reads. It draws scratch
// from the batch's own arenas, never the per-token one, and keeps every sequence's
// state in the batch's strided arenas, so it carries no cohort-sized state itself.
func newInkling2BatchResident(m *Inkling2LoopModel, be compute.Backend) (*inkling2Resident, error) {
	if _, ok := be.(compute.Inkling2PrefillDevice); !ok {
		return nil, fmt.Errorf("inkling2 batch: backend %s has no row-batched ops", be.Name())
	}
	s, err := newInkling2ResidentSized(m, be, 0, 64)
	if err != nil {
		return nil, err
	}
	if s.hostRouting {
		s.close()
		inkling2TrimIdle(be)
		return nil, fmt.Errorf("inkling2 batch: an expert dtype has no indexed kernel")
	}
	s.graph = false
	return s, nil
}

// newInkling2BatchState allocates the arenas of a C-sequence, ctx-position batch on
// the binding s and readies it for a cohort of that shape; begin re-readies it for
// any cohort no wider and no longer.
//
// want is the cohort prefill chunk the state is sized for. A wide want the device or
// the admission refuses is rebuilt at the base chunk, whose rings and arenas are the
// smaller ones; the returned state then reports capT below want.
func newInkling2BatchState(s *inkling2Resident, C, ctx, want int) (*inkling2Batch, error) {
	b, err := newInkling2BatchChunk(s, C, ctx, want)
	if err != nil && want > inkling2PrefillChunkBase && inkling2ChunkRefusal(err) {
		logInkling2PrefillFallback(err, want)
		return newInkling2BatchChunk(s, C, ctx, inkling2PrefillChunkBase)
	}
	return b, err
}

// newInkling2BatchChunk builds the state with a cohort prefill chunk of chunk tokens.
func newInkling2BatchChunk(s *inkling2Resident, C, ctx, chunk int) (*inkling2Batch, error) {
	sh, err := s.batchShapeChunk(C, ctx, chunk)
	if err != nil {
		return nil, err
	}
	be := s.be
	pd := be.(compute.Inkling2PrefillDevice)
	hp := s.hp
	kvHalf, keep, scr, caps, rowsT := sh.kvHalf, sh.keep, sh.scr, sh.caps, sh.rowsT
	b := &inkling2Batch{s: s, capC: C, capCtx: ctx, seq: sh.seq, kvHalf: kvHalf, mem: inkling2BatchMem(be),
		prefillT: sh.prefillT, need: sh.need, capT: chunk, capPrefillT: sh.prefillT, half: sh.half}
	b.p = ink2Prefill{s: s, pd: pd, T: C}
	if b.mem != nil && inkling2BatchQ8RowTile(os.LookupEnv) {
		b.p.rows = b.mem
	}
	if inkling2BatchAdmitOn(os.LookupEnv) {
		// idle pool blocks count as used to the driver; return them before probing
		inkling2TrimIdle(be)
		_, free, known := inkling2DeviceFree(be)
		admit := inkling2BatchAdmit
		if b.prefillT > inkling2PrefillChunkBase {
			admit = inkling2WideAdmit
			free, known = inkling2WideFree(be, free, known)
		}
		if err := admit(C, b.need, free, known); err != nil {
			return nil, err
		}
	}
	b.p.keepFloats, b.p.scrFloats = keep, scr
	b.p.keep, b.p.scr = b.alloc(keep), b.alloc(scr)
	// each argmax takes a 4-int slot: the index, then its 8-byte reduction key
	b.in, b.meta, b.argIdx = b.alloc(rowsT*hp.Hidden), b.alloc(C), b.alloc(4*C)

	slots := len(caps)
	kvDim := hp.KVHeads * hp.HeadDim
	b.histStride = inkling2BatchHistStride(hp)
	b.kc, b.vc, b.hist = make([]ink2Ref, slots), make([]ink2Ref, slots), make([]ink2Ref, slots)
	maxCap := 0
	for slot, cap := range caps {
		maxCap = max(maxCap, cap)
		kv := inkling2BatchKVFloats(C*cap*kvDim, kvHalf)
		b.kc[slot], b.vc[slot] = b.alloc(kv), b.alloc(kv)
		b.hist[slot] = b.alloc(C * b.histStride)
	}
	// float64 scores rows per head per sequence, so sequences never serialize on them
	b.scores = b.alloc(C * 2 * hp.Heads * maxCap)
	if b.err != nil {
		err := b.err
		b.release()
		return nil, err
	}
	if err := b.begin(C, ctx); err != nil {
		b.release()
		return nil, err
	}
	return b, nil
}

// alloc takes n zeroed floats of batch state, device-only when the backend can refuse
// rather than spill to managed memory.
func (b *inkling2Batch) alloc(n int) ink2Ref {
	if b.err != nil {
		return ink2Ref{}
	}
	var t compute.Tensor
	var err error
	if b.mem != nil {
		t, err = b.mem.Ink2AllocDeviceOnly(n)
	} else {
		t, err = b.s.dev.Ink2Alloc(n)
	}
	if err != nil {
		b.err = &Inkling2BatchMemoryError{Sequences: b.capC, Need: b.need, Free: compute.FreeUnknown, Alloc: err}
		return ink2Ref{}
	}
	b.owned = append(b.owned, t)
	return compute.Ink2At(t, 0)
}

// inkling2BatchShape is the buffer plan of a C-sequence, ctx-position batch, computed
// without touching the device so a cohort can be sized before it is admitted.
type inkling2BatchShape struct {
	seq             compute.Inkling2BatchDevice
	kvHalf          bool
	half            compute.Inkling2PrefillHalfKV
	prefillT, rowsT int
	keep, scr       int
	caps            []int
	need            int64
}

// batchShape is the plan of a cohort none of whose prompts passes the base chunk.
func (s *inkling2Resident) batchShape(C, ctx int) (inkling2BatchShape, error) {
	return s.batchShapeChunk(C, ctx, inkling2CohortChunk(s.chunkWanted, 0))
}

// batchShapeChunk is the plan at a cohort prefill chunk of chunk tokens (capped at the
// state's C*ctx tokens), its rings widened from the base capacities by that chunk alone.
func (s *inkling2Resident) batchShapeChunk(C, ctx, chunk int) (inkling2BatchShape, error) {
	if C < 1 {
		return inkling2BatchShape{}, fmt.Errorf("inkling2 batch: %d sequences", C)
	}
	limit := s.ctx
	if limit == 0 {
		// a cohort binding holds no global cache, so each cohort is bounded by the resident context
		limit = Inkling2LoopResidentContext
	}
	if ctx < 1 || ctx > limit {
		return inkling2BatchShape{}, fmt.Errorf("inkling2 batch: context %d outside 1..%d", ctx, limit)
	}
	be := s.be
	sh := inkling2BatchShape{seq: inkling2BatchSeqOps(be, os.LookupEnv)}
	kvHalf, err := inkling2BatchKVHalf(os.LookupEnv)
	if err != nil {
		return inkling2BatchShape{}, err
	}
	if kvHalf && sh.seq == nil {
		// the per-sequence fallback addresses cache rows by float offsets
		return inkling2BatchShape{}, fmt.Errorf("inkling2 batch: FAK_INKLING2_BATCH_KV=f16 needs the sequence-batched device ops")
	}
	sh.kvHalf = kvHalf
	hp := s.hp
	on, half, err := inkling2CohortPrefillKV(be, kvHalf, os.LookupEnv)
	if err != nil {
		return inkling2BatchShape{}, err
	}
	if on {
		sh.half = half
		sh.prefillT = min(chunk, C*ctx)
	}
	// the prefill chunks and the decode steps run one after the other on one arena pair
	sh.rowsT = max(C, sh.prefillT)
	sh.keep, sh.scr = s.prefillArenas(sh.rowsT)
	sh.scr += C*s.output.rows + 64
	// a narrower cohort begun later may take the few-row path, so size for the widest one that can
	if few := min(C, inkling2BatchFewRowsMost(os.LookupEnv)); few > 0 && inkling2BatchFewRows(be, few, os.LookupEnv) != nil {
		sh.scr += inkling2BatchFewRowsScratch(hp, few)
	}
	sh.caps = inkling2BatchCaps(hp, s.baseCaps, ctx, sh.prefillT)
	sh.need = inkling2BatchBytes(hp, sh.caps, C, sh.keep, sh.scr, kvHalf)
	return sh, nil
}

// inkling2BatchAdmitOn reports whether a batch is checked against free device memory
// before it allocates (FAK_INKLING2_BATCH_ADMIT=0 skips the check).
func inkling2BatchAdmitOn(lookup func(string) (string, bool)) bool {
	v, _ := lookup("FAK_INKLING2_BATCH_ADMIT")
	return v != "0"
}

// usePrefill sets the prefill chunk of the next cohort, at most the width the state
// holds, so a narrower cohort on a wide state forwards and addresses its rings exactly
// as it would on a state built for it.
func (b *inkling2Batch) usePrefill(want int) { b.prefillT = min(want, b.capPrefillT) }

// begin readies the arenas for a cohort of C sequences at ctx positions. Every
// stride is the cohort's own, so a narrower or shorter cohort addresses a prefix of
// each arena. Attention only reads positions the cohort has written, but a conv reads
// its history before the first write, so histories are cleared.
func (b *inkling2Batch) begin(C, ctx int) error {
	if C < 1 || C > b.capC || ctx < 1 || ctx > b.capCtx {
		return fmt.Errorf("inkling2 batch: cohort %dx%d outside the %dx%d state", C, ctx, b.capC, b.capCtx)
	}
	s := b.s
	hp := s.hp
	b.C, b.ctx, b.pos = C, ctx, make([]int, C)
	b.sharedMin = inkling2CohortSharedPrefixMin(os.LookupEnv)
	b.p.T = C
	b.p.keepOff, b.p.scrOff = 0, 0
	clear(b.p.q8)
	b.p.grouped = inkling2BatchGrouped(C, os.LookupEnv)
	b.p.vd, b.p.fewRows = nil, false
	if vd := inkling2BatchFewRows(s.be, C, os.LookupEnv); vd != nil {
		b.p.vd, b.p.fewRows = vd, true
	}
	b.caps = inkling2BatchCaps(hp, s.baseCaps, ctx, b.prefillT)
	maxCap := 0
	for _, c := range b.caps {
		maxCap = max(maxCap, c)
	}
	b.scoreStride = 2 * hp.Heads * maxCap
	// a captured step bakes the previous cohort's width and strides
	s.dev.Ink2GraphReset()
	b.graphReady, b.warm = false, false
	b.graph = inkling2BatchGraph() && inkling2DefaultOn(s.be, "FAK_INKLING2_GRAPH", os.LookupEnv)
	if n := C * b.histStride; n > 0 {
		zero := make([]float32, n)
		for _, h := range b.hist {
			if err := s.dev.Ink2Write(h, zero); err != nil {
				return err
			}
		}
	}
	return nil
}

// resetRow readies finished row i for a new sequence at position 0. Its KV needs no
// clearing: the row writes every position below its own before attention reads it.
// Its conv history is read before the first write, so it is cleared in every slot.
func (b *inkling2Batch) resetRow(row int) error {
	b.pos[row] = 0
	zero := make([]float32, b.histStride)
	for _, h := range b.hist {
		if err := b.s.dev.Ink2Write(at(h, row*b.histStride), zero); err != nil {
			return err
		}
	}
	return nil
}

// release frees the batch arenas and keeps the weight binding; the freed blocks go
// back to the driver so the next state's admission probe sees them free.
func (b *inkling2Batch) release() {
	for _, t := range b.owned {
		b.s.be.Free(t)
	}
	b.owned = nil
	b.err = nil
	inkling2TrimIdle(b.s.be)
}

func (b *inkling2Batch) close() {
	b.release()
	b.s.close()
	inkling2TrimIdle(b.s.be)
}

// inkling2BatchGroupedMin is the smallest batch whose routed experts run the grouped
// GEMM; below it the sorted decode GEMV is faster per step.
const inkling2BatchGroupedMin = 3

// inkling2BatchGrouped reports whether a C-sequence batch runs its experts grouped:
// C >= FAK_INKLING2_BATCH_GROUPED_MIN (default inkling2BatchGroupedMin), unless
// FAK_INKLING2_PREFILL_GROUPED=0 forces the sorted path at every C.
func inkling2BatchGrouped(C int, lookup func(string) (string, bool)) bool {
	if v, ok := lookup("FAK_INKLING2_PREFILL_GROUPED"); ok && v == "0" {
		return false
	}
	least := inkling2BatchGroupedMin
	if v, ok := lookup("FAK_INKLING2_BATCH_GROUPED_MIN"); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n >= 1 {
			least = n
		}
	}
	return C >= least
}

// inkling2BatchFewRowsMax is the largest batch that runs the few-row verify kernels
// by default; the kernels themselves stop at inkling2FewRowsLimit rows.
const (
	inkling2BatchFewRowsMax = 8
	inkling2FewRowsLimit    = 9
)

// inkling2BatchFewRows returns the backend's few-row verify device when a C-sequence
// batch runs on it: C <= FAK_INKLING2_BATCH_ROWS_MAX (default inkling2BatchFewRowsMax,
// 0 disables, capped at inkling2FewRowsLimit); nil otherwise.
func inkling2BatchFewRows(be any, C int, lookup func(string) (string, bool)) compute.Inkling2VerifyDevice {
	vd, ok := be.(compute.Inkling2VerifyDevice)
	if !ok {
		return nil
	}
	if C > inkling2BatchFewRowsMost(lookup) {
		return nil
	}
	return vd
}

// inkling2BatchFewRowsMost is the widest batch the few-row kernels take under
// FAK_INKLING2_BATCH_ROWS_MAX.
func inkling2BatchFewRowsMost(lookup func(string) (string, bool)) int {
	most := inkling2BatchFewRowsMax
	if v, ok := lookup("FAK_INKLING2_BATCH_ROWS_MAX"); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n >= 0 {
			most = n
		}
	}
	return min(most, inkling2FewRowsLimit)
}

// inkling2BatchFewRowsScratch is the per-visit scratch the few-row path takes past the
// prefill arenas: Q8_0 codes for each distinct matmul activation (the three attention
// inputs, the router input or head input, and the attention output) and the
// pairwise expert input repeated per routed slot.
func inkling2BatchFewRowsScratch(hp model.Inkling2LoopHParams, C int) int {
	pad := func(n int) int { return (n + 63) &^ 63 }
	codes := func(n int) int { return pad(C * (n/4 + n/32)) }
	qd := hp.Heads * hp.HeadDim
	return 4*codes(hp.Hidden) + codes(qd) + pad(C*hp.ExpertsUsed*hp.Hidden)
}

// inkling2BatchMem returns the backend's strict allocator and row-tile matmul seam,
// or nil for a backend without it.
func inkling2BatchMem(be any) compute.Inkling2BatchMemory {
	m, _ := be.(compute.Inkling2BatchMemory)
	return m
}

// inkling2BatchQ8RowTile reports whether the batch pins its 2..127-row Q8_0 matmuls
// to the row-tile panel GEMM (FAK_INKLING2_BATCH_Q8_ROWTILE=0 leaves them on the
// backend's default small-panel kernel).
func inkling2BatchQ8RowTile(lookup func(string) (string, bool)) bool {
	v, ok := lookup("FAK_INKLING2_BATCH_Q8_ROWTILE")
	return !ok || v != "0"
}

// inkling2BatchHeadroom is device memory kept free past the batch's own buffers for
// kernel scratch that grows lazily on the first step and for the step's graph.
const inkling2BatchHeadroom = 64 << 20

// Inkling2BatchMemoryError refuses a batch whose buffers would not fit in free device
// memory (FAK_INKLING2_BATCH_ADMIT=0 skips the check), or whose allocation the device
// refused (Alloc). Either way no row of the cohort has run, so its rows can be served
// by a narrower cohort or the serialized engine.
type Inkling2BatchMemoryError struct {
	Sequences  int
	Need, Free int64
	Alloc      error
}

func (e *Inkling2BatchMemoryError) Error() string {
	if e.Alloc != nil {
		return fmt.Sprintf("inkling2 batch: %d sequences: %v", e.Sequences, e.Alloc)
	}
	return fmt.Sprintf("inkling2 batch: %d sequences need %.1f MiB + %d MiB headroom, %.1f MiB device memory free",
		e.Sequences, float64(e.Need)/(1<<20), inkling2BatchHeadroom>>20, float64(e.Free)/(1<<20))
}

func (e *Inkling2BatchMemoryError) Unwrap() error { return e.Alloc }

// inkling2BatchAdmit refuses need bytes against a probed free; a device that cannot
// report free memory fails open.
func inkling2BatchAdmit(C int, need, free int64, known bool) error {
	if !known || free < 0 || need+inkling2BatchHeadroom <= free {
		return nil
	}
	return &Inkling2BatchMemoryError{Sequences: C, Need: need, Free: free}
}

// inkling2BatchKVHalf reports whether FAK_INKLING2_BATCH_KV stores the batch caches as
// f16 ("f16"/"fp16") rather than the default fp32 (unset, "f32"/"fp32").
func inkling2BatchKVHalf(lookup func(string) (string, bool)) (bool, error) {
	v, _ := lookup("FAK_INKLING2_BATCH_KV")
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "f16", "fp16":
		return true, nil
	case "", "f32", "fp32":
		return false, nil
	}
	return false, fmt.Errorf("inkling2 batch: FAK_INKLING2_BATCH_KV=%q is not f32 or f16", v)
}

// inkling2BatchKVFloats is the float allocation holding n cache elements.
func inkling2BatchKVFloats(n int, half bool) int {
	if half {
		return (n + 1) / 2
	}
	return n
}

// inkling2BatchStateBytes is the device memory of C sequences' caches and conv
// histories over slots of the given caps, the caches f16 when half.
func inkling2BatchStateBytes(hp model.Inkling2LoopHParams, caps []int, C int, half bool) int64 {
	kvDim := hp.KVHeads * hp.HeadDim
	hist := int64(inkling2BatchHistStride(hp))
	var n int64
	for _, c := range caps {
		n += 2*int64(inkling2BatchKVFloats(C*c*kvDim, half)) + hist*int64(C)
	}
	return 4 * n
}

// inkling2BatchBytes is every device buffer newInkling2Batch allocates past the
// resident: the state, the keep and scratch arenas of keep and scr floats, the scores
// rows, and the step's input, position and argmax rows.
func inkling2BatchBytes(hp model.Inkling2LoopHParams, caps []int, C, keep, scr int, half bool) int64 {
	maxCap := 0
	for _, c := range caps {
		maxCap = max(maxCap, c)
	}
	floats := int64(keep) + int64(scr) + int64(C)*int64(hp.Hidden+1+4+2*hp.Heads*maxCap)
	return inkling2BatchStateBytes(hp, caps, C, half) + 4*floats
}

func inkling2BatchHistStride(hp model.Inkling2LoopHParams) int {
	return (2*hp.KVHeads*hp.HeadDim + 2*hp.Hidden) * (hp.ConvKernel - 1)
}

// inkling2BatchTrimKeep is the largest idle allocator block a trim leaves cached, so
// the small per-step buffers stay pooled while the batch arenas go back to the device.
const inkling2BatchTrimKeep = 1 << 20

// inkling2TrimIdle returns idle cached device blocks to the driver, so the next batch
// and a free-memory probe see them free. A trim that fails leaves the blocks cached,
// which costs headroom but not correctness, so its panic is dropped.
func inkling2TrimIdle(be any) {
	t, ok := be.(interface{ TrimLarge(int) })
	if !ok {
		return
	}
	defer func() { _ = recover() }()
	t.TrimLarge(inkling2BatchTrimKeep)
}

func inkling2DeviceFree(be any) (total, free int64, known bool) {
	d, ok := be.(interface {
		DeviceMemory() (total, free int64, known bool)
	})
	if !ok {
		return 0, compute.FreeUnknown, false
	}
	return d.DeviceMemory()
}

// inkling2LocalRing reports whether a cache slot belongs to a sliding-window layer,
// whose cache is a ring of recent positions rather than the whole context.
func inkling2LocalRing(hp model.Inkling2LoopHParams, slot int) bool {
	return hp.LocalLayer[inkling2SlotLayer(hp, slot)]
}

// inkling2BatchCaps is the cache capacity of every slot: ctx for a global slot, the
// resident's ring size for a local one, widened to window+chunk so a prefill chunk of
// chunk tokens never evicts a key an earlier query of the same chunk attends to.
func inkling2BatchCaps(hp model.Inkling2LoopHParams, local []int, ctx, chunk int) []int {
	caps := make([]int, len(local))
	for slot := range caps {
		caps[slot] = ctx
		if inkling2LocalRing(hp, slot) {
			ring := local[slot]
			if chunk > 0 {
				ring = max(ring, hp.SlidingWindow+chunk)
			}
			// a batch never passes position ctx-1, so a ring past ctx is never addressed
			caps[slot] = min(ring, ctx)
		}
	}
	return caps
}

// StateBytes is the device memory of every sequence's caches and conv histories.
func (b *inkling2Batch) StateBytes() int64 {
	return inkling2BatchStateBytes(b.s.hp, b.caps, b.C, b.kvHalf)
}

// inkling2SeqRow is sequence i's row of a strided operand; an absent operand stays
// absent, as the batched kernels keep a null pointer null.
func inkling2SeqRow(r ink2Ref, stride, i int) ink2Ref {
	if r.T.Buf() == nil {
		return ink2Ref{}
	}
	return at(r, i*stride)
}

func (b *inkling2Batch) layer(slot, ip int, cur ink2Ref, blocks *[]ink2Ref, msgs []ink2Ref, c *ink2Ctrl) ink2Ref {
	p := &b.p
	s := b.s
	hp := s.hp
	l := &s.layers[ip]
	local := hp.LocalLayer[ip]
	h := hp.Hidden
	kvDim := hp.KVHeads * hp.HeadDim
	kn := hp.ConvKernel
	inpSA := cur

	in := [3]ink2Ref{cur, cur, cur}
	// the few-row path folds the input norms into the mix when no controller sits
	// between them, as a verify chunk does
	normed := false
	if l.resQRows > 0 {
		var norms []ink2Ref
		if p.fewRows && msgs == nil {
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
		if p.fewRows {
			o := p.controls(in[:], msgs[:3], c)
			in = [3]ink2Ref{o[0], o[1], o[2]}
		} else {
			for j := 0; j < 3; j++ {
				in[j] = p.control(in[j], msgs[j], c, j)
			}
		}
	}
	switch {
	case normed:
	case local && msgs != nil && p.fewRows:
		// controls left its three outputs contiguous
		o := p.rmsRows(in[0], l.attnNorm, 3*b.C, h, hp.NormEps)
		in = [3]ink2Ref{o, at(o, b.C*h), at(o, 2*b.C*h)}
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

	q := p.mm(l.wq, in[0])
	var kDyn, vDyn ink2Ref
	if !local {
		kDyn = p.mm(l.kconv, in[1])
		vDyn = p.mm(l.vconv, in[2])
	}
	k, v := p.mm(l.wk, in[1]), p.mm(l.wv, in[2])
	r := p.mm(l.rel, in[0])
	sinks := p.mm(l.sink, in[0])

	ap := compute.Ink2AttnParams{
		Heads: hp.Heads, KVHeads: hp.KVHeads, HeadDim: hp.HeadDim, RelLen: hp.RelLength,
		QKClip: hp.QKClip, RelClip: hp.RelClip, Eps: hp.NormEps, Global: !local,
		Alpha: hp.LogScaleAlpha, Floor: hp.LogScaleFloor, Scale: float32(1 / math.Sqrt(float64(hp.HeadDim))),
	}
	extent, window := hp.RelExtent, 0
	if local {
		extent, window = hp.SlidingWindow, hp.SlidingWindow
	}
	qd := hp.Heads * hp.HeadDim
	attn := p.tmp(b.C * qd)
	cap := b.caps[slot]
	cv := compute.Ink2KVConv{KW: l.kconvW, VW: l.vconvW, Kn: kn, KHist: b.hist[slot], VHist: at(b.hist[slot], kvDim*(kn-1))}
	if !local {
		cv.KDyn, cv.VDyn = kDyn, vDyn
	}
	cv.KNorm, cv.VNorm = inkling2ConvNorm(kn, cv.KDyn), inkling2ConvNorm(kn, cv.VDyn)
	st := compute.Ink2SeqStrides{Q: l.wq.rows, R: l.rel.rows, K: l.wk.rows, V: l.wv.rows, KDyn: l.kconv.rows, VDyn: l.vconv.rows,
		Sinks: l.sink.rows, KCache: cap * kvDim, VCache: cap * kvDim, Hist: b.histStride, Scores: b.scoreStride, Out: qd, Meta: 1}
	if b.seq != nil {
		if s.err == nil {
			s.fail(b.seq.Ink2BQKVPost(q, r, k, v, l.qclip, l.relClip, b.kc[slot], b.vc[slot], b.meta, ap, cap, &cv, st, b.kvHalf, b.C))
		}
		if s.err == nil {
			s.fail(b.seq.Ink2BAttend(q, r, sinks, l.relBias, b.kc[slot], b.vc[slot], b.scores, attn, b.meta, ap, extent, window, cap, st, b.kvHalf, b.C))
		}
	}
	for i := 0; b.seq == nil && i < b.C && s.err == nil; i++ {
		ci := cv
		ci.KDyn, ci.VDyn = inkling2SeqRow(cv.KDyn, st.KDyn, i), inkling2SeqRow(cv.VDyn, st.VDyn, i)
		ci.KHist, ci.VHist = inkling2SeqRow(cv.KHist, st.Hist, i), inkling2SeqRow(cv.VHist, st.Hist, i)
		qi, ri, meta := inkling2SeqRow(q, st.Q, i), inkling2SeqRow(r, st.R, i), inkling2SeqRow(b.meta, st.Meta, i)
		kci, vci := inkling2SeqRow(b.kc[slot], st.KCache, i), inkling2SeqRow(b.vc[slot], st.VCache, i)
		s.fail(s.dev.Ink2QKVPost(qi, ri, inkling2SeqRow(k, st.K, i), inkling2SeqRow(v, st.V, i), l.qclip, l.relClip, kci, vci, meta, ap, cap, &ci))
		if s.err == nil {
			s.fail(s.dev.Ink2Attend(qi, ri, inkling2SeqRow(sinks, st.Sinks, i), l.relBias, kci, vci, inkling2SeqRow(b.scores, st.Scores, i),
				inkling2SeqRow(attn, st.Out, i), meta, ap, extent, window, cap))
		}
	}
	gate := p.mm(l.gate, in[0])
	if s.err == nil {
		s.fail(s.dev.Ink2SigmoidMul(attn, gate, b.C*qd))
	}
	o := p.mm(l.wo, attn)
	var ffnInp ink2Ref
	if ip%hp.AttnResBlock == 0 {
		*blocks = append(*blocks, inpSA)
		ffnInp = b.conv(slot, o, l.attnPostConv, ink2Ref{}, 2*kvDim, false)
	} else {
		ffnInp = b.conv(slot, o, l.attnPostConv, inpSA, 2*kvDim, false)
	}

	var x ink2Ref
	if p.fewRows && msgs == nil {
		x = p.mixNorm(with(*blocks, ffnInp), l.ffnResQ, 1, []ink2Ref{l.ffnNorm})[0]
	} else {
		x = p.mix(with(*blocks, ffnInp), l.ffnResQ, 1)[0]
		if msgs != nil {
			x = p.control(x, msgs[3], c, 3)
		}
		x = p.rms(x, l.ffnNorm, h, hp.NormEps)
	}
	x = p.moe(l, x)
	return b.conv(slot, x, l.ffnPostConv, ffnInp, 2*kvDim+h, true)
}

// conv runs a hidden-width post conv on every row against its sequence's history at
// channel offset; keep holds the output past the layer visit.
func (b *inkling2Batch) conv(slot int, x, weight, resid ink2Ref, offset int, keep bool) ink2Ref {
	s := b.s
	h, kn := s.hp.Hidden, s.hp.ConvKernel
	norm := inkling2ConvNorm(kn, ink2Ref{})
	var out ink2Ref
	if keep {
		out = b.p.hold(b.C * h)
	} else {
		out = b.p.tmp(b.C * h)
	}
	hist := at(b.hist[slot], offset*(kn-1))
	st := compute.Ink2SeqStrides{X: h, Out: h, Resid: h, Hist: b.histStride}
	if b.seq != nil {
		if s.err == nil {
			s.fail(b.seq.Ink2BConv(x, weight, hist, out, resid, h, kn, norm, st, b.C))
		}
		return out
	}
	for i := 0; i < b.C && s.err == nil; i++ {
		s.fail(s.dev.Ink2Conv(inkling2SeqRow(x, st.X, i), weight, ink2Ref{}, inkling2SeqRow(hist, st.Hist, i), inkling2SeqRow(out, st.Out, i),
			inkling2SeqRow(resid, st.Resid, i), h, 1, kn, norm))
	}
	return out
}

// forward enqueues one lockstep step from the uploaded rows through every row's argmax.
func (b *inkling2Batch) forward() {
	p := &b.p
	s := b.s
	hp := s.hp
	h := hp.Hidden
	p.keepOff, p.scrOff = 0, 0
	clear(p.q8)
	cur := p.hold(b.C * h)
	if s.err == nil {
		s.fail(s.dev.Ink2RMS(b.in, s.tokNorm, cur, b.C, h, hp.NormEps))
	}
	// the few-row matmuls key cached Q8_0 codes by scratch offset, so they go with it
	visit := func() {
		p.scrOff = 0
		clear(p.q8)
	}
	var blocks []ink2Ref
	for il := 0; il < hp.CoreStart; il++ {
		visit()
		cur = b.layer(il, il, cur, &blocks, nil, nil)
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
				cur = b.layer(ip, ip, cur, &blocks, nil, nil)
				continue
			}
			slot := hp.Layers + (pass-1)*core + (ip - hp.CoreStart)
			c := &s.ctrls[ip-hp.CoreStart]
			msgs := p.mix(with(bank, bankOut), c.msgQ, 4)
			cur = b.layer(slot, ip, cur, &blocks, msgs, c)
		}
	}
	for il := hp.CoreEnd; il < hp.Layers; il++ {
		visit()
		cur = b.layer(il, il, cur, &blocks, nil, nil)
	}
	visit()
	var xn ink2Ref
	if p.fewRows {
		xn = p.mixNorm(with(blocks, cur), s.outResQ, 1, []ink2Ref{s.outNorm})[0]
	} else {
		x := p.mix(with(blocks, cur), s.outResQ, 1)[0]
		xn = p.rms(x, s.outNorm, h, hp.NormEps)
	}
	logits := p.mm(s.output, xn)
	if b.seq != nil {
		if s.err == nil {
			s.fail(b.seq.Ink2BSoftcapArgmax(logits, hp.VocabUnpadded, s.output.rows, hp.FinalLogitSoftcap, b.argIdx, b.C))
		}
		return
	}
	for i := 0; i < b.C && s.err == nil; i++ {
		s.fail(s.dev.Ink2SoftcapArgmax(at(logits, i*s.output.rows), hp.VocabUnpadded, hp.FinalLogitSoftcap, at(b.argIdx, 4*i)))
	}
}

// step feeds tokens[i] to sequence i at its next position and returns every
// sequence's greedy next token.
func (b *inkling2Batch) step(tokens []int) ([]int, error) {
	s := b.s
	hp := s.hp
	if len(tokens) != b.C {
		return nil, fmt.Errorf("inkling2 batch: %d tokens for %d sequences", len(tokens), b.C)
	}
	rows := make([]float32, b.C*hp.Hidden)
	meta := make([]float32, b.C)
	limit := b.ctx
	for i, t := range tokens {
		if b.pos[i] >= limit {
			return nil, fmt.Errorf("inkling2 batch: sequence %d position %d exceeds the %d-token cache", i, b.pos[i], limit)
		}
		s.w.TokEmbd.Row(t, rows[i*hp.Hidden:(i+1)*hp.Hidden])
		meta[i] = math.Float32frombits(uint32(int32(b.pos[i])))
	}
	s.fail(s.dev.Ink2Write(b.in, rows))
	s.fail(s.dev.Ink2Write(b.meta, meta))
	switch {
	case s.err != nil:
	case b.graphReady:
		s.fail(s.dev.Ink2GraphLaunch())
	case b.graph && b.warm:
		// capture after one eager step has grown every lazily sized kernel scratch
		if err := s.dev.Ink2GraphBegin(); err != nil {
			b.graph = false
			b.forward()
			break
		}
		b.forward()
		if err := s.dev.Ink2GraphEnd(); err != nil || s.err != nil {
			s.fail(err)
			b.graph = false
			break
		}
		b.graphReady = true
		s.fail(s.dev.Ink2GraphLaunch())
	default:
		b.forward()
		b.warm = true
	}
	if s.err != nil {
		return nil, s.err
	}
	got, err := s.dev.Ink2ReadInts(b.argIdx, 4*b.C)
	if err != nil {
		return nil, err
	}
	out := make([]int, b.C)
	for i := range out {
		out[i] = int(got[4*i])
		b.pos[i]++
	}
	if r, ok := s.be.(interface{ Recycle() }); ok {
		r.Recycle()
	}
	return out, nil
}

// Inkling2LoopBatchStats times a lockstep batched greedy decode. Prompt covers the
// steps in which some sequence still feeds a prompt token; Decode covers the
// DecodeSteps steps in which every sequence feeds a generated token, and
// DecodeTokens counts the kept tokens those steps produced. NeedBytes is the batch's
// admission estimate; DeviceTotal, FreeBefore and FreeAfter probe device memory around
// building the batch (FreeBefore precedes the resident upload; compute.FreeUnknown
// when not probeable).
type Inkling2LoopBatchStats struct {
	Sequences    int
	StateBytes   int64
	NeedBytes    int64
	DeviceTotal  int64
	FreeBefore   int64
	FreeAfter    int64
	Prompt       time.Duration
	Decode       time.Duration
	DecodeSteps  int
	DecodeTokens int
}

// AggregateTokensPerSecond is the generated tokens of every sequence per second of
// the decode phase.
func (st Inkling2LoopBatchStats) AggregateTokensPerSecond() float64 {
	if st.Decode <= 0 {
		return 0
	}
	return float64(st.DecodeTokens) / st.Decode.Seconds()
}

// inkling2BatchGreedy decodes n greedy tokens after each prompt. With cohort prefill
// every prompt is prefilled first and every step feeds generated tokens; otherwise
// every sequence advances in lockstep from position 0, one step per token: a sequence
// still inside its prompt feeds its next prompt token, and the others feed their last
// generated token.
func inkling2BatchGreedy(m *Inkling2LoopModel, be compute.Backend, prompts [][]int, n, ctx int) ([][]int, Inkling2LoopBatchStats, error) {
	st := Inkling2LoopBatchStats{Sequences: len(prompts)}
	longest := 0
	for i, pr := range prompts {
		if len(pr) == 0 {
			return nil, st, fmt.Errorf("inkling2 batch: sequence %d has an empty prompt", i)
		}
		longest = max(longest, len(pr))
	}
	st.DeviceTotal, st.FreeBefore, _ = inkling2DeviceFree(be)
	b, err := newInkling2Batch(m, be, len(prompts), ctx, longest)
	if err != nil {
		return nil, st, err
	}
	defer b.close()
	_, st.FreeAfter, _ = inkling2DeviceFree(be)
	st.StateBytes, st.NeedBytes = b.StateBytes(), b.need
	// a CUDA stream capture is bound to the OS thread that opened it
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	outs := make([][]int, len(prompts))
	tokens := make([]int, len(prompts))
	start := time.Now()
	// skip is the prompt positions already fed before the first step
	skip := 0
	if b.prefillT > 0 {
		first := make([]int, len(prompts))
		if err := b.prefill(prompts, nil, func(row, token int) { first[row] = token }); err != nil {
			return nil, st, err
		}
		for i := range outs {
			outs[i] = append(outs[i], first[i])
		}
		skip = longest
		st.Prompt = time.Since(start)
		start = time.Now()
	}
	for t := skip; ; t++ {
		done := true
		for i, pr := range prompts {
			if t < len(pr) && skip == 0 {
				tokens[i] = pr[t]
			} else {
				tokens[i] = outs[i][len(outs[i])-1]
			}
			if len(outs[i]) < n {
				done = false
			}
		}
		if done {
			break
		}
		if t == longest && skip == 0 {
			st.Prompt = time.Since(start)
			start = time.Now()
		}
		next, err := b.step(tokens)
		if err != nil {
			return nil, st, err
		}
		if t >= longest {
			st.DecodeSteps++
		}
		for i, pr := range prompts {
			if (skip > 0 || t >= len(pr)-1) && len(outs[i]) < n {
				outs[i] = append(outs[i], next[i])
				if t >= longest {
					st.DecodeTokens++
				}
			}
		}
	}
	if st.DecodeSteps > 0 {
		st.Decode = time.Since(start)
	} else if skip == 0 {
		st.Prompt = time.Since(start)
	}
	return outs, st, nil
}
