package model

import (
	"fmt"
	"io"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expert_checkpoint_tier.go — R5 of the activated-expert offload ladder (#5616, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md): put a CHECKPOINT tier under a ring miss, so the tier below
// the bounded device ring stops being "the whole model, resident in host RAM".
//
// The gap this closes. R0 bounded DEVICE residency; nothing bounded HOST residency. A GGUF MoE
// checkpoint stores each layer's routed experts as one fused [E, out, in] tensor, and the loader
// materializes the whole E-expert slab before splitting it, so a top-k route pays E*stride bytes of
// host residency for the k*stride it activates. On a checkpoint whose expert bulk exceeds host RAM
// that is the wall, and the ring cannot help: a ring miss had nowhere cheap to fall to, because the
// miss path had no way to read ONE expert.
//
// Both halves of the answer already existed and neither was wired. ggufExpertSource
// (gguf_expert_source.go) indexes {offset, [E,out,in], quant-block geometry} per fused tensor at
// construction — no payload IO — and reads exactly one expert's sub-range through a caller-supplied
// io.ReaderAt; its own witness proves those bytes are bit-identical to the resident slab slice and
// that a read moves exactly stride bytes. What it lacked was a CONSUMER on the decode path. This
// file is that consumer: it maps a canonical per-expert tensor name to (fused tensor, expert index,
// quant geometry), faults exactly that expert's stride on a miss, and hands back the very
// q4kTensor / kQuantTensor the resident path would have handed back — so the device / pinned-host /
// checkpoint ladder is a real three-rung ladder rather than a two-rung one.
//
// Where it sits. Session.expertSwiGLUHAL resolves a routed expert's three projections against
// m.q4kw / m.kqw and, only when BOTH are empty for that name, against this tier. So a fully-resident
// checkpoint never reaches it and nothing about the existing path moves; a checkpoint loaded without
// materializing its expert slabs finds every routed weight here instead. The R3 prefetch resolves
// through the same seam, which is what makes a checkpoint fault overlappable at all: the activated
// set is known at layer entry, so its reads are issued together rather than one GEMM apart.
//
// Host residency is BOUNDED, by the same policy the ring uses. The retained-host cache is a
// polymodel.Pool — the byte budget, the LRU victim and the all-on-nothing admit the ring already
// borrows — so `resident <= budget` holds here by construction too. The default budget is 0, which
// means STREAM-THROUGH: an expert is read, handed to the staging that uploads it, and dropped. That
// is the honest default, because on a device backend the host copy is dead the moment Upload
// returns, and a tier that retained it by default would re-introduce exactly the unbounded host
// residency this rung exists to remove. A positive budget makes the middle "pinned-host" rung real
// for a host that has RAM to spare.
//
// Concurrency. Two sessions sharing one Model share one tier. The bookkeeping is mutex-guarded, but
// the READ is issued outside the lock, because ggufExpertSource is safe for concurrent use whenever
// its ReaderAt is (os.File, bytes.Reader and stripeload.StripedReaderAt all are) and serializing
// every fault behind one lock would make the tier a throughput floor for the whole fleet. The cost
// is that two agents faulting the same cold expert at the same instant may both read it — a
// duplicated read of byte-identical bytes, never a corrupted one. Coalescing those into one read is
// precisely R7/#5618, which needs a shared ring to coalesce ONTO; this rung supplies the tier it
// coalesces at.
//
// A read failure is never swallowed silently: the failure count and the last error surface through
// ExpertCheckpointStats, and the fault itself reports the error to its caller. On the staging path
// there is deliberately no soft "weight absent" degradation, because over a streamed checkpoint the
// weight is resident nowhere else — a caller told "absent" would report a missing tensor and bury
// the real IO error underneath it. See staging's own note.
//
// The tier composes with the ring instead of shadowing it. Everything the weight HAL needs to STAGE
// a projection — the ring key, the dtype, the resident byte cost — comes from the index, so a weight
// the ring already holds is answered on a ring hit and no byte is read. Faulting eagerly would have
// been the quiet defeat of this whole ladder: every routed expert would pay checkpoint IO even at a
// ring budget large enough to hold the entire activated set.
//
// INVALIDATING assumption, inherited from the source and restated here because this file is what
// makes it load-bearing: experts are contiguous, equal-stride, unpadded segments of the fused
// tensor. newGGUFExpertSource refuses a description whose derived slab does not fit the source, and
// the per-expert geometry here is DERIVED from the quant kind rather than caller-supplied, so a
// caller cannot state a stride that disagrees with the representation the decode path will build.

// ExpertCheckpointQuant is the quantized representation a fused routed-expert slab is stored in.
// Only the k-quant forms the routed-expert weight HAL can stage resident are admitted: a tier that
// accepted a representation expertSwiGLUHAL cannot serve would fault bytes nothing could then use.
type ExpertCheckpointQuant int

const (
	// ExpertCheckpointQ4K is the Q4_K super-block form, ~0.56 B/weight — the expert majority of a
	// memory-lean GLM-5.2-class checkpoint.
	ExpertCheckpointQ4K ExpertCheckpointQuant = iota
	// ExpertCheckpointQ5K and ExpertCheckpointQ6K are the mixed-quant forms a UD-Q4_K_M artifact
	// uses for the projections it keeps at higher precision.
	ExpertCheckpointQ5K
	ExpertCheckpointQ6K
)

// String names the representation for a report or an error.
func (q ExpertCheckpointQuant) String() string {
	switch q {
	case ExpertCheckpointQ5K:
		return "Q5_K"
	case ExpertCheckpointQ6K:
		return "Q6_K"
	case ExpertCheckpointQ4K:
		return "Q4_K"
	}
	return fmt.Sprintf("ExpertCheckpointQuant(%d)", int(q))
}

// blockGeometry is the quant-block geometry of this representation: weights per block and bytes per
// block. It is DERIVED from the kind rather than accepted from the caller, so the stride the source
// reads and the stride the decode path expects cannot disagree.
func (q ExpertCheckpointQuant) blockGeometry() (weights, bytes int, ok bool) {
	switch q {
	case ExpertCheckpointQ4K:
		return qkK, q4kBlockBytes, true
	case ExpertCheckpointQ5K:
		return qkK, kindQ5K.blockBytes(), true
	case ExpertCheckpointQ6K:
		return qkK, kindQ6K.blockBytes(), true
	}
	return 0, 0, false
}

// FusedExpertTensor describes one fused [Experts, Rows, Cols] routed-expert tensor as it sits in a
// checkpoint: where it starts, how many experts it fuses, one expert's 2-D shape, and which layer
// and projection its per-expert segments belong to. Layer/Proj are what let the tier answer the
// canonical per-expert names the decode path asks for (`model.layers.L.mlp.experts.E.PROJ`) without
// the loader having to invent them.
type FusedExpertTensor struct {
	// Name is the checkpoint's own name for the fused tensor (e.g. `blk.3.ffn_gate_exps.weight`).
	// It is the source key and appears in every diagnostic, so a refusal names a real tensor.
	Name  string
	Layer int
	// Proj is the canonical projection suffix without `.weight` — "gate_proj", "up_proj" or
	// "down_proj".
	Proj    string
	Quant   ExpertCheckpointQuant
	Offset  int64
	Experts int
	// Rows/Cols are ONE expert's shape: [out, in], Cols being the reduction dimension.
	Rows, Cols int
}

// expertCheckpointEntry is one indexed per-expert projection: which shard holds it, which fused
// tensor and index inside it, the geometry needed to rebuild the resident tensor the weight HAL
// stages, and the byte stride a fault would move. The stride is indexed rather than derived at read
// time because it is what lets the tier answer "how big is this weight" WITHOUT reading it — the
// whole reason a ring hit over a streamed checkpoint costs no IO.
type expertCheckpointEntry struct {
	shard  int
	fused  string
	expert int
	quant  ExpertCheckpointQuant
	rows   int
	cols   int
	nblk   int
	stride int64
}

// halKey / dtype are the staging identity of this projection — the same dtype-prefixed key and
// compute dtype the resident path uses (expertWeight.halKey, weightHALQ4K / weightHALKQuant), so a
// checkpoint-served expert lands under exactly the ring entry a resident one would have.
func (e expertCheckpointEntry) halKey(name string) string {
	if e.quant == ExpertCheckpointQ4K {
		return "q4k:" + name
	}
	return "kquant-raw:" + name
}

func (e expertCheckpointEntry) dtype() compute.Dtype {
	switch e.quant {
	case ExpertCheckpointQ5K:
		return compute.Q5_K
	case ExpertCheckpointQ6K:
		return compute.Q6_K
	default:
		return compute.Q4_K
	}
}

// weight rebuilds the resident representation from one faulted expert's raw bytes — the SAME
// q4kTensor / kQuantTensor shape the resident loader path produces, so everything downstream (the
// staged builders, the ring key, the resident byte accounting) is unchanged by where the bytes
// came from.
func (e expertCheckpointEntry) weight(name string, raw []byte) expertWeight {
	switch e.quant {
	case ExpertCheckpointQ5K:
		return expertWeight{name: name, kq: &kQuantTensor{out: e.rows, in: e.cols, nblk: e.nblk, kind: kindQ5K, raw: raw}}
	case ExpertCheckpointQ6K:
		return expertWeight{name: name, kq: &kQuantTensor{out: e.rows, in: e.cols, nblk: e.nblk, kind: kindQ6K, raw: raw}}
	default:
		return expertWeight{name: name, q4: &q4kTensor{out: e.rows, in: e.cols, nblk: e.nblk, raw: raw}}
	}
}

// ExpertCheckpointTier is the checkpoint rung under the device ring: a per-expert range reader over
// one or more checkpoint shards, plus a bounded host-resident cache over what it faults. Its zero
// value is not usable — construct it with NewExpertCheckpointTier — and a nil *ExpertCheckpointTier
// is a valid "no tier", which is the default and which every method tolerates.
type ExpertCheckpointTier struct {
	mu     sync.Mutex
	shards []*ggufExpertSource
	index  map[string]expertCheckpointEntry

	// pool bounds the RETAINED host copies. A zero budget retains nothing (every Admit reports
	// ErrTooLarge and the faulted expert is simply handed to the caller and dropped), which is the
	// stream-through default.
	pool *polymodel.Pool
	host map[polymodel.ModelID]expertWeight
	peak int64

	reads     int
	hits      int
	evictions int
	failures  int
	bytesRead int64
	lastErr   error
}

// NewExpertCheckpointTier returns an empty tier whose retained host cache is bounded by hostBytes.
// hostBytes <= 0 is stream-through: bytes are faulted, staged and dropped, which is what keeps host
// residency for the expert bulk at zero on the device path this rung is for.
func NewExpertCheckpointTier(hostBytes int64) *ExpertCheckpointTier {
	if hostBytes < 0 {
		hostBytes = 0
	}
	return &ExpertCheckpointTier{
		index: map[string]expertCheckpointEntry{},
		pool:  polymodel.NewPool(hostBytes),
		host:  map[polymodel.ModelID]expertWeight{},
	}
}

// AddShard indexes one checkpoint shard's fused routed-expert tensors over r, whose readable extent
// is size bytes. It performs NO payload IO: every description is validated against the declared
// extent at construction, so a malformed layout is a load-time refusal rather than a decode-time
// read of misaligned bytes. A split checkpoint calls this once per shard file, because each shard's
// offsets are relative to its own file.
func (t *ExpertCheckpointTier) AddShard(r io.ReaderAt, size int64, fused []FusedExpertTensor) error {
	if t == nil {
		return fmt.Errorf("%w: nil tier", ErrGGUFExpertMetadata)
	}
	descs := make([]ggufExpertTensorDesc, 0, len(fused))
	for _, f := range fused {
		blockWeights, blockBytes, ok := f.Quant.blockGeometry()
		if !ok {
			return fmt.Errorf("%w: %s declares unstageable representation %s",
				ErrGGUFExpertMetadata, f.Name, f.Quant)
		}
		if f.Layer < 0 {
			return fmt.Errorf("%w: %s declares negative layer %d", ErrGGUFExpertMetadata, f.Name, f.Layer)
		}
		if f.Proj == "" {
			return fmt.Errorf("%w: %s declares no projection", ErrGGUFExpertMetadata, f.Name)
		}
		descs = append(descs, ggufExpertTensorDesc{
			Name: f.Name, Offset: f.Offset, Experts: f.Experts, Rows: f.Rows, Cols: f.Cols,
			BlockWeights: blockWeights, BlockBytes: blockBytes,
		})
	}
	src, err := newGGUFExpertSource(r, size, descs)
	if err != nil {
		return err
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	shard := len(t.shards)
	// Index every per-expert projection BEFORE publishing the shard, so a duplicate name aborts
	// with the tier exactly as it was found rather than half-indexed.
	staged := make(map[string]expertCheckpointEntry, len(fused)*8)
	for i, f := range fused {
		blockWeights, _, _ := f.Quant.blockGeometry()
		for e := 0; e < f.Experts; e++ {
			name := expertName(f.Layer, e, f.Proj+".weight")
			if _, dup := t.index[name]; dup {
				return fmt.Errorf("%w: %s already indexed when adding %s", ErrGGUFExpertMetadata, name, f.Name)
			}
			if _, dup := staged[name]; dup {
				return fmt.Errorf("%w: %s indexed twice by shard %d", ErrGGUFExpertMetadata, name, shard)
			}
			stride, _ := src.expertStride(descs[i].Name) // validated at construction
			staged[name] = expertCheckpointEntry{
				shard: shard, fused: descs[i].Name, expert: e, quant: f.Quant,
				rows: f.Rows, cols: f.Cols, nblk: f.Cols / blockWeights, stride: stride,
			}
		}
	}
	t.shards = append(t.shards, src)
	for name, entry := range staged {
		t.index[name] = entry
	}
	return nil
}

// checkpointStaging is how a checkpoint-served projection would be staged into the weight HAL and
// the routed-expert ring: the ring key, the host-source builder, the dtype and the resident byte
// cost. An expertWeight carries THIS instead of the expert's bytes, which is the whole point of the
// shape — every field is known from the index, so the bytes are read only when the ring actually
// misses.
type checkpointStaging struct {
	key   string
	mk    func() compute.Tensor
	dt    compute.Dtype
	bytes int64
}

// staging derives that descriptor from the INDEX alone: the key, the dtype and the byte cost are
// answered WITHOUT reading a byte. That is what makes the tier compose with the ring instead of
// defeating it — a projection the ring already holds is served on a hit and the fault inside mk
// never runs, so a resident activated set costs zero checkpoint IO no matter how often it is routed.
// ok=false means this tier simply does not carry the name, which is how a fully-resident checkpoint
// (and the nil tier) declines.
//
// mk PANICS on a read failure rather than returning a sentinel, matching requireTensorPresent's
// convention for an unavailable weight. There is deliberately no soft fallback: over a streamed
// checkpoint the weight is resident nowhere else, so a caller told "absent" would report a missing
// tensor and bury the real IO error. The failure is counted in the ledger either way.
func (t *ExpertCheckpointTier) staging(name string) (*checkpointStaging, bool) {
	if t == nil {
		return nil, false
	}
	t.mu.Lock()
	entry, indexed := t.index[name]
	t.mu.Unlock()
	if !indexed {
		return nil, false
	}
	return &checkpointStaging{
		key: entry.halKey(name),
		mk: func() compute.Tensor {
			w, err := t.fault(name)
			if err != nil {
				panic("model: checkpoint expert read " + name + ": " + err.Error())
			}
			if w.q4 != nil {
				return compute.NewQ4K(compute.Default(), []int{w.q4.out, w.q4.in}, w.q4.raw)
			}
			if w.kq.kind == kindQ6K {
				return compute.NewQ6K(compute.Default(), []int{w.kq.out, w.kq.in}, w.kq.raw)
			}
			return compute.NewQ5K(compute.Default(), []int{w.kq.out, w.kq.in}, w.kq.raw)
		},
		dt:    entry.dtype(),
		bytes: entry.stride,
	}, true
}

// fault answers one canonical per-expert tensor name by reading exactly that expert's stride — or
// by serving a retained host copy when the bounded host cache holds one. It never returns a
// partially-built weight.
func (t *ExpertCheckpointTier) fault(name string) (expertWeight, error) {
	if t == nil {
		return expertWeight{}, fmt.Errorf("%w: %s (no checkpoint tier)", ErrGGUFExpertNotFound, name)
	}
	id := polymodel.ModelID(name)

	t.mu.Lock()
	entry, indexed := t.index[name]
	if !indexed {
		t.mu.Unlock()
		return expertWeight{}, fmt.Errorf("%w: %s", ErrGGUFExpertNotFound, name)
	}
	if w, live := t.host[id]; live {
		t.pool.Touch(id)
		t.hits++
		t.mu.Unlock()
		return w, nil
	}
	src := t.shards[entry.shard]
	t.mu.Unlock()

	// Issued OUTSIDE the lock: see the file header on why a duplicated concurrent read of identical
	// bytes is the right trade against serializing every fleet fault behind one mutex.
	raw, err := src.readExpert(entry.fused, entry.expert)
	if err != nil {
		t.mu.Lock()
		t.failures, t.lastErr = t.failures+1, err
		t.mu.Unlock()
		return expertWeight{}, err
	}
	w := entry.weight(name, raw)

	t.mu.Lock()
	defer t.mu.Unlock()
	t.reads++
	t.bytesRead += int64(len(raw))
	if have, live := t.host[id]; live {
		// A concurrent fault landed first. Its bytes and ours are byte-identical, so serve the
		// retained copy and let this one go — but the read still happened and is still counted, or
		// the ledger would under-report the IO the workload actually issued.
		t.pool.Touch(id)
		return have, nil
	}
	evicted, admitErr := t.pool.Admit(polymodel.Model{ID: id, WeightBytes: int64(len(raw))})
	if admitErr != nil {
		return w, nil // stream-through (the default) or an expert larger than the whole host budget
	}
	for _, vid := range evicted {
		delete(t.host, vid)
		t.evictions++
	}
	t.host[id] = w
	if used := t.pool.Used(); used > t.peak {
		t.peak = used
	}
	return w, nil
}

// ExpertCheckpointStats is the checkpoint tier's ledger — the evidence for this rung's claim that
// bytes read per decode step scale with the ACTIVATED count k rather than with the expert count E.
// A hit rate alone cannot show that: the number that matters is BytesRead against the slab bytes a
// fully-resident load would have paid.
type ExpertCheckpointStats struct {
	Enabled bool `json:"enabled"`
	// Tensors is how many per-expert projections the tier indexes — E*3 per MoE layer indexed.
	Tensors int `json:"tensors"`
	// Reads is how many expert faults reached the checkpoint; BytesRead their total stride bytes.
	// Hits are faults served from the retained host cache, Evictions its page-outs.
	Reads     int   `json:"reads"`
	BytesRead int64 `json:"bytes_read"`
	Hits      int   `json:"hits"`
	Evictions int   `json:"evictions"`
	// BudgetBytes/ResidentBytes/PeakBytes are the HOST residency bound. Budget 0 is stream-through:
	// the expert bulk never accumulates in host RAM at all, which is the point of the rung.
	BudgetBytes   int64 `json:"budget_bytes"`
	ResidentBytes int64 `json:"resident_bytes"`
	PeakBytes     int64 `json:"peak_bytes"`
	ResidentCount int   `json:"resident_count"`
	// Failures counts reads that errored, and LastError names the most recent one. The demand path
	// raises an IO failure rather than swallowing it, but the R3 prefetch is best-effort and a
	// checkpoint whose backing file went away mid-run would otherwise be visible only as a stall;
	// these two are the standing record that the tier tried and could not read.
	Failures  int    `json:"failures"`
	LastError string `json:"last_error,omitempty"`
}

// Stats reports this tier's ledger. The nil tier reports the zero value (Enabled=false), which is
// the honest reading for a model whose experts are fully resident: there is no checkpoint IO to
// account because there is no checkpoint tier.
func (t *ExpertCheckpointTier) Stats() ExpertCheckpointStats {
	if t == nil {
		return ExpertCheckpointStats{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st := ExpertCheckpointStats{
		Enabled:       true,
		Tensors:       len(t.index),
		Reads:         t.reads,
		BytesRead:     t.bytesRead,
		Hits:          t.hits,
		Evictions:     t.evictions,
		BudgetBytes:   t.pool.Budget(),
		ResidentBytes: t.pool.Used(),
		PeakBytes:     t.peak,
		ResidentCount: len(t.host),
		Failures:      t.failures,
	}
	if t.lastErr != nil {
		st.LastError = t.lastErr.Error()
	}
	return st
}

// SetExpertCheckpoint attaches a checkpoint tier to this model, so a routed-expert weight absent
// from the resident stores is faulted per expert instead of being reported missing. It is the
// loader's hand-off: nothing else in the model mutates it, and a model that never had one keeps the
// fully-resident path byte-for-byte.
func (m *Model) SetExpertCheckpoint(t *ExpertCheckpointTier) {
	if m == nil {
		return
	}
	m.expertCheckpoint = t
}

// ExpertCheckpointStats reports this model's checkpoint-tier ledger (the zero value when it has
// none) — the operator-facing counterpart of Session.ExpertRing() one rung down.
func (m *Model) ExpertCheckpointStats() ExpertCheckpointStats {
	if m == nil {
		return ExpertCheckpointStats{}
	}
	return m.expertCheckpoint.Stats()
}
