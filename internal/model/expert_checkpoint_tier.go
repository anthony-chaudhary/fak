package model

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expert_checkpoint_tier.go â€” R5 of the activated-expert offload ladder (#5616, epic #5606,
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
// construction â€” no payload IO â€” and reads exactly one expert's sub-range through a caller-supplied
// io.ReaderAt; its own witness proves those bytes are bit-identical to the resident slab slice and
// that a read moves exactly stride bytes. What it lacked was a CONSUMER on the decode path. This
// file is that consumer: it maps a canonical per-expert tensor name to (fused tensor, expert index,
// quant geometry), faults exactly that expert's stride on a miss, and hands back the very
// q4kTensor / kQuantTensor the resident path would have handed back â€” so the device / pinned-host /
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
// polymodel.Pool â€” the byte budget, the LRU victim and the all-on-nothing admit the ring already
// borrows â€” so `resident <= budget` holds here by construction too. The default budget is 0, which
// means STREAM-THROUGH: an expert is read, handed to the staging that uploads it, and dropped. That
// is the honest default, because on a device backend the host copy is dead the moment Upload
// returns, and a tier that retained it by default would re-introduce exactly the unbounded host
// residency this rung exists to remove. A positive budget makes the middle "pinned-host" rung real
// for a host that has RAM to spare.
//
// Concurrency. Sessions sharing a Model share this tier. One canonical-key loading owner
// publishes a complete weight or error to all attached waiters; failed ownership permits retry.
// Source I/O stays outside the bookkeeping mutex so different experts read concurrently.
//
// A read failure is never swallowed silently: the failure count and the last error surface through
// ExpertCheckpointStats, and the fault itself reports the error to its caller. On the staging path
// there is deliberately no soft "weight absent" degradation, because over a streamed checkpoint the
// weight is resident nowhere else â€” a caller told "absent" would report a missing tensor and bury
// the real IO error underneath it. See staging's own note.
//
// The tier composes with the ring instead of shadowing it. Everything the weight HAL needs to STAGE
// a projection â€” the ring key, the dtype, the resident byte cost â€” comes from the index, so a weight
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
// Every member is a form a routed-expert slab can be STAGED from: a tier that accepted a
// representation no consumer could serve would fault bytes nothing could then use.
//
// Every admitted kind stages its GGUF super-blocks VERBATIM through an internal/compute host
// constructor: Q4_K/Q5_K/Q6_K/Q2_K always had one, and Q3_K gained one in fak#13149
// (compute.NewQ3K / compute.Q3_K). That last addition is what lets the published DeepSeek-V4.1
// Flash Q2_K artifact - Q2_K gate/up, Q3_K down - reach the streamed forward with no f32
// materialization of the expert bulk (fak#13144).
type ExpertCheckpointQuant int

const (
	// ExpertCheckpointQ4K is the Q4_K super-block form, ~0.56 B/weight â€” the expert majority of a
	// memory-lean GLM-5.2-class checkpoint.
	ExpertCheckpointQ4K ExpertCheckpointQuant = iota
	// ExpertCheckpointQ5K and ExpertCheckpointQ6K are the mixed-quant forms a UD-Q4_K_M artifact
	// uses for the projections it keeps at higher precision.
	ExpertCheckpointQ5K
	ExpertCheckpointQ6K
	// ExpertCheckpointQ2K is the Q2_K super-block form, the narrowest k-quant a routed-expert slate
	// is published in. It is appended AFTER Q6K so the existing constant values never shift; the
	// DeepSeek-V4.1 Flash Q2_K arm is its first producer.
	ExpertCheckpointQ2K
	// ExpertCheckpointQ3K is the Q3_K super-block form the published DeepSeek-V4.1 Q2_K slate uses
	// for its down projection. Appended AFTER Q2K so the existing constant values never shift, and
	// staged VERBATIM like its siblings now that internal/compute carries a Q3_K host tensor kind
	// (compute.NewQ3K, fak#13149). This overturns the fak#13122 settlement, which declined Q3_K
	// because staging().mk must return a compute.Tensor and no Q3_K kind existed; fak#13149 built
	// exactly that kind, so the settlement's premise no longer holds and the tier admits Q3_K the
	// same way it admits Q2_K (fak#13144).
	ExpertCheckpointQ3K
)

// String names the representation for a report or an error.
func (q ExpertCheckpointQuant) String() string {
	switch q {
	case ExpertCheckpointQ5K:
		return "Q5_K"
	case ExpertCheckpointQ6K:
		return "Q6_K"
	case ExpertCheckpointQ2K:
		return "Q2_K"
	case ExpertCheckpointQ3K:
		return "Q3_K"
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
	case ExpertCheckpointQ2K:
		return qkK, q2kBlockBytes, true
	case ExpertCheckpointQ3K:
		return qkK, q3kBlockBytes, true
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
	// Proj is the canonical projection suffix without `.weight` â€” "gate_proj", "up_proj" or
	// "down_proj".
	Proj string
	// Arch is the checkpoint's architecture (cfg.ModelType). It selects the
	// per-expert canonical spelling the index is keyed on: arch "deepseek41"
	// indexes the native V4.1 ffn.experts.<e>.{w1,w3,w2} leaves; an empty or any
	// other arch keeps the historical mlp.experts.<e>.<proj> spelling
	// byte-for-byte.
	Arch    string
	Quant   ExpertCheckpointQuant
	Offset  int64
	Experts int
	// Rows/Cols are ONE expert's shape: [out, in], Cols being the reduction dimension.
	Rows, Cols int
}

// expertCheckpointEntry is one indexed per-expert projection: which shard holds it, which fused
// tensor and index inside it, the geometry needed to rebuild the resident tensor the weight HAL
// stages, and the byte stride a fault would move. The stride is indexed rather than derived at read
// time because it is what lets the tier answer "how big is this weight" WITHOUT reading it â€” the
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
	group  string
	proj   string
}

// halKey / dtype are the staging identity of this projection â€” the same dtype-prefixed key and
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
	case ExpertCheckpointQ2K:
		return compute.Q2_K
	case ExpertCheckpointQ3K:
		return compute.Q3_K
	default:
		return compute.Q4_K
	}
}

// weight rebuilds the resident representation from one faulted expert's raw bytes â€” the SAME
// q4kTensor / kQuantTensor shape the resident loader path produces, so everything downstream (the
// staged builders, the ring key, the resident byte accounting) is unchanged by where the bytes
// came from.
func (e expertCheckpointEntry) weight(name string, raw []byte) expertWeight {
	switch e.quant {
	case ExpertCheckpointQ5K:
		return expertWeight{name: name, kq: &kQuantTensor{out: e.rows, in: e.cols, nblk: e.nblk, kind: kindQ5K, raw: raw}}
	case ExpertCheckpointQ6K:
		return expertWeight{name: name, kq: &kQuantTensor{out: e.rows, in: e.cols, nblk: e.nblk, kind: kindQ6K, raw: raw}}
	case ExpertCheckpointQ2K:
		return expertWeight{name: name, kq: &kQuantTensor{out: e.rows, in: e.cols, nblk: e.nblk, kind: kindQ2K, raw: raw}}
	case ExpertCheckpointQ3K:
		return expertWeight{name: name, kq: &kQuantTensor{out: e.rows, in: e.cols, nblk: e.nblk, kind: kindQ3K, raw: raw}}
	default:
		return expertWeight{name: name, q4: &q4kTensor{out: e.rows, in: e.cols, nblk: e.nblk, raw: raw}}
	}
}

// ExpertCheckpointTier is the checkpoint rung under the device ring: a per-expert range reader over
// one or more checkpoint shards, plus a bounded host-resident cache over what it faults. Its zero
// value is not usable â€” construct it with NewExpertCheckpointTier â€” and a nil *ExpertCheckpointTier
// is a valid "no tier", which is the default and which every method tolerates.
type ExpertCheckpointTier struct {
	mu               sync.Mutex
	shards           []*ggufExpertSource
	index            map[string]expertCheckpointEntry
	deviceRingBudget int64

	// overlays holds the optional per-rank sparse row overlays, keyed by the fused tensor they
	// band (#13030). It is nil until SetExpertSparseOverlay registers one, so an overlay-free tier
	// carries no extra field traffic and fault is byte-identical to the dense-stride path.
	overlays map[string]*ExpertSparseOverlay

	// pool bounds the RETAINED host copies. A zero budget retains nothing (every Admit reports
	// ErrTooLarge and the faulted expert is simply handed to the caller and dropped), which is the
	// stream-through default.
	pool      *polymodel.Pool
	host      map[polymodel.ModelID]expertWeight
	peak      int64
	loading   map[polymodel.ModelID]*expertCheckpointLoad
	coalesced int

	// prefetchJobs is the aggregate count of speculative reads this tier has in flight across
	// every simultaneous Prefetch call. fakExpertPrefetchWorkers bounds ONE call's readers; only
	// this counter bounds N concurrent calls. It is incremented under the mutex when a job is
	// admitted and decremented exactly once by that job's single deferred finalizer, so a
	// best-effort hint can never leak its share of the bound.
	prefetchJobs int

	// ready holds prefetched projections no demand fault has consumed yet. Each entry is
	// handed out once, and entries left from an older Prefetch generation are dropped.
	ready         map[polymodel.ModelID]expertCheckpointReady
	readyGen      uint64
	prefetched    int
	prefetchHits  int
	prefetchStale int
	readahead     ExpertCheckpointReadaheadStats

	reads     int
	hits      int
	evictions int
	failures  int
	bytesRead int64
	lastErr   error

	// overlayRows / overlayBytes are the sparse-overlay ledger (#13030): faults served from a
	// registered overlay's copied rows and the row bytes they moved. Zero with no overlay, which is
	// the default.
	overlayRows  int
	overlayBytes int64
}

// Loading ownership and complete-only publication adapt JigSawPT/llama.cpp's
// host-tier contract; the pinned source and MIT notice are in THIRD_PARTY_NOTICES.md.
type expertCheckpointLoad struct {
	done    chan struct{}
	weight  expertWeight
	err     error
	waiters int
}

type expertCheckpointReady struct {
	weight expertWeight
	gen    uint64
}

var ErrExpertCheckpointLoadAborted = errors.New("model: expert checkpoint load owner aborted")

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

var ErrExpertCheckpointDeviceBudget = errors.New("model: checkpoint expert device ring budget refused")

// ExpertCheckpointDeviceBudgetError identifies an undersized or sealed device budget.
type ExpertCheckpointDeviceBudgetError struct {
	Tensor string
	Bytes  int64
	Budget int64
}

func (e *ExpertCheckpointDeviceBudgetError) Error() string {
	return fmt.Sprintf("%v: tensor %s, bytes %d, budget %d", ErrExpertCheckpointDeviceBudget, e.Tensor, e.Bytes, e.Budget)
}

func (e *ExpertCheckpointDeviceBudgetError) Unwrap() error {
	return ErrExpertCheckpointDeviceBudget
}

// SetDeviceRingBudget declares a device ceiling independently of host retention. A positive
// ceiling must fit every complete indexed expert and seals the index and budget without payload IO.
func (t *ExpertCheckpointTier) SetDeviceRingBudget(bytes int64) error {
	if t == nil || bytes < 0 {
		return fmt.Errorf("%w: invalid device ring budget", ErrGGUFExpertMetadata)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.deviceRingBudget > 0 {
		if bytes == t.deviceRingBudget {
			return nil
		}
		return &ExpertCheckpointDeviceBudgetError{Tensor: "sealed index", Bytes: t.deviceRingBudget, Budget: bytes}
	}
	if bytes == 0 {
		return nil
	}
	tensor, required, err := t.maxExpertGroupBytesLocked()
	if err != nil {
		return err
	}
	if bytes < required {
		return &ExpertCheckpointDeviceBudgetError{Tensor: tensor, Bytes: required, Budget: bytes}
	}
	t.deviceRingBudget = bytes
	return nil
}

// DeviceRingBudget reports the declared device ceiling; zero preserves the existing session policy.
func (t *ExpertCheckpointTier) DeviceRingBudget() int64 {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.deviceRingBudget
}

func (t *ExpertCheckpointTier) maxExpertGroupBytesLocked() (string, int64, error) {
	type groupGeometry struct {
		bytes       int64
		projections uint8
	}
	groups := make(map[string]groupGeometry)
	for name, entry := range t.index {
		var projection uint8
		switch entry.proj {
		case "gate_proj":
			projection = 1
		case "up_proj":
			projection = 2
		case "down_proj":
			projection = 4
		default:
			return "", 0, fmt.Errorf("%w: %s has unknown expert projection", ErrGGUFExpertMetadata, name)
		}
		group := groups[entry.group]
		if entry.group == "" || entry.stride <= 0 || group.projections&projection != 0 || group.bytes > math.MaxInt64-entry.stride {
			return "", 0, fmt.Errorf("%w: %s has invalid expert group geometry", ErrGGUFExpertMetadata, name)
		}
		group.bytes += entry.stride
		group.projections |= projection
		groups[entry.group] = group
	}
	if len(groups) == 0 {
		return "", 0, fmt.Errorf("%w: empty expert index", ErrGGUFExpertMetadata)
	}
	var tensor string
	var required int64
	for name, group := range groups {
		if group.projections != 7 {
			return "", 0, fmt.Errorf("%w: %s has incomplete expert projections", ErrGGUFExpertMetadata, name)
		}
		if group.bytes > required || group.bytes == required && name < tensor {
			tensor, required = name, group.bytes
		}
	}
	return tensor, required, nil
}

// Has reports whether this tier indexes the named per-expert projection. It is an
// INDEX lookup only: it reads no payload and does not fault, so the V4.1 forward
// admission can confirm a streamed routed expert is servable without paying the
// checkpoint IO the tier exists to defer. A nil tier (the no-tier default) has no
// names and reports false.
func (t *ExpertCheckpointTier) Has(name string) bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	_, ok := t.index[name]
	t.mu.Unlock()
	return ok
}

// AddShard indexes one checkpoint shard's fused routed-expert tensors over r, whose readable extent
// is size bytes. It performs NO payload IO: every description is validated against the declared
// extent at construction, so a malformed layout is a load-time refusal rather than a decode-time
// read of misaligned bytes. A split checkpoint calls this once per shard file, because each shard's
// offsets are relative to its own file.
func (t *ExpertCheckpointTier) AddShard(r io.ReaderAt, size int64, fused []FusedExpertTensor) error {
	return t.AddShardData(r, size, nil, fused)
}

// AddShardData is AddShard plus an optional page-cache-visible mapped region backing r (#1302). When
// data is non-empty and exactly size bytes, the shard's per-expert projections are read through the
// page-cache-aware path (ggufExpertSource.enablePageCache / readExpertPageCache): a repeat read of an
// already-faulted expert is served zero-copy from the mapping and moves no device bytes. A nil/empty
// or mismatched data leaves page-cache accounting off. Eligible zero-retention Q2_K/Q3_K
// faults may still borrow through r's optional read-only range capability; an ordinary
// ReaderAt retains the historical copied path, including platforms without mmap support.
func (t *ExpertCheckpointTier) AddShardData(r io.ReaderAt, size int64, data []byte, fused []FusedExpertTensor) error {
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
	// Opt this shard's source into the page-cache-aware read path when its retained reader carries a
	// mapped region exactly matching the declared extent (#1302). enablePageCache refuses a partial or
	// over-long map, so a refusal here is the historical ReadAt path, not a silent mixed one.
	src.enablePageCache(data)

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.deviceRingBudget > 0 {
		return fmt.Errorf("%w: device ring budget seals expert index", ErrGGUFExpertMetadata)
	}
	shard := len(t.shards)
	// Index every per-expert projection BEFORE publishing the shard, so a duplicate name aborts
	// with the tier exactly as it was found rather than half-indexed.
	staged := make(map[string]expertCheckpointEntry, len(fused)*8)
	for i, f := range fused {
		blockWeights, _, _ := f.Quant.blockGeometry()
		for e := 0; e < f.Experts; e++ {
			name := expertNameArch(f.Arch, f.Layer, e, f.Proj+".weight")
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
				group: expertNameArch(f.Arch, f.Layer, e, "gate_proj.weight"), proj: f.Proj,
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
// shape â€” every field is known from the index, so the bytes are read only when the ring actually
// misses.
type checkpointStaging struct {
	key   string
	mk    func() compute.Tensor
	dt    compute.Dtype
	bytes int64
}

// staging derives that descriptor from the INDEX alone: the key, the dtype and the byte cost are
// answered WITHOUT reading a byte. That is what makes the tier compose with the ring instead of
// defeating it â€” a projection the ring already holds is served on a hit and the fault inside mk
// never runs, so a resident activated set costs zero checkpoint IO no matter how often it is routed.
// ok=false means this tier simply does not carry the name, which is how a fully-resident checkpoint
// (and the nil tier) declines.
//
// mk PANICS on a read failure rather than returning a sentinel, matching requireTensorPresent's
// convention for an unavailable weight. There is deliberately no soft fallback: over a streamed
// checkpoint the weight is resident nowhere else, so a caller told "absent" would report a missing
// tensor and bury the real IO error. The failure is counted in the ledger either way.
func (t *ExpertCheckpointTier) staging(name string) (*checkpointStaging, bool) {
	return t.stagingForBackend(name, nil)
}

// borrowedExpertUploadBackend explicitly admits immutable Q2_K/Q3_K source
// views. Every selected upload entry point (Upload, UploadClass, or UploadAsync)
// must consume/copy every source byte before returning, even if device completion
// is deferred, and must not mutate or retain the source. Returned tensors require
// independent ownership: freeing one cannot invalidate another upload of the same
// host range.
// Unknown backends, including host-pointer-interning backends, keep owned bytes.
type borrowedExpertUploadBackend interface {
	SupportsBorrowedExpertUploads() bool
}

func (t *ExpertCheckpointTier) stagingForBackend(name string, be compute.Backend) (*checkpointStaging, bool) {
	if t == nil {
		return nil, false
	}
	t.mu.Lock()
	entry, indexed := t.index[name]
	t.mu.Unlock()
	if !indexed {
		return nil, false
	}
	capability, capable := be.(borrowedExpertUploadBackend)
	borrowSafe := capable && capability.SupportsBorrowedExpertUploads()
	return &checkpointStaging{
		key: entry.halKey(name),
		mk: func() compute.Tensor {
			w, err := t.fault(name)
			if err != nil {
				panic("model: checkpoint expert read " + name + ": " + err.Error())
			}
			if w.borrowed && !borrowSafe {
				// Ownership is per consumer, after shared fault publication. Never
				// replace a waiter's kq/raw in place: another backend may use it.
				w = entry.weight(name, append([]byte(nil), w.kq.raw...))
			}
			if w.q4 != nil {
				return compute.NewQ4K(compute.Default(), []int{w.q4.out, w.q4.in}, w.q4.raw)
			}
			if w.kq.kind == kindQ6K {
				return compute.NewQ6K(compute.Default(), []int{w.kq.out, w.kq.in}, w.kq.raw)
			}
			if w.kq.kind == kindQ2K {
				return compute.NewQ2K(compute.Default(), []int{w.kq.out, w.kq.in}, w.kq.raw)
			}
			if w.kq.kind == kindQ3K {
				// Verbatim, like every other admitted k-quant: no f32 expansion of the expert bulk.
				return compute.NewQ3K(compute.Default(), []int{w.kq.out, w.kq.in}, w.kq.raw)
			}
			return compute.NewQ5K(compute.Default(), []int{w.kq.out, w.kq.in}, w.kq.raw)
		},
		dt:    entry.dtype(),
		bytes: entry.stride,
	}, true
}

// ErrExpertCheckpointBudget reports that a routed-expert fault could not be served within the
// tier's declared hostBytes residency budget. A POSITIVE budget is a real ceiling on the bytes the
// tier hands out and retains, so a projection that cannot be made resident within that ceiling is
// refused BY NAME rather than streamed through unbounded -- the fail-closed guard against the
// unbounded routed-expert working set that OOM-killed the physical strix3 V4.1 serve (fak#13280). A
// budget of 0 (the default) is stream-through and never produces this error.
var ErrExpertCheckpointBudget = errors.New("model: checkpoint expert exceeds the bounded host residency budget")

// ExpertCheckpointBudgetError is the typed fail-closed refusal: which projection, how large its
// faulted slab is, and which budget refused it. It wraps ErrExpertCheckpointBudget so errors.Is
// reaches the closed class while the operands name the real tensor for the operator.
type ExpertCheckpointBudgetError struct {
	Tensor string
	Bytes  int64
	Budget int64
}

func (e *ExpertCheckpointBudgetError) Error() string {
	return fmt.Sprintf("%v: tensor %s (slab %d bytes) cannot be made resident within budget %d bytes",
		ErrExpertCheckpointBudget, e.Tensor, e.Bytes, e.Budget)
}

// Unwrap exposes the sentinel so errors.Is(err, ErrExpertCheckpointBudget) holds for the typed
// refusal while the caller can still read the tensor and budget operands.
func (e *ExpertCheckpointBudgetError) Unwrap() error { return ErrExpertCheckpointBudget }

// admitBounded is the residency-ceiling guard shared by the overlay and dense fault paths, called
// with t.mu held. It is EXACTLY the historical admit with one addition: when Admit fails AND the
// tier declares a positive budget, the slab cannot be made resident within the ceiling at all
// (Admit has already evicted every cold unpinned resident it may), so the fault fails closed with
// the typed named budget refusal instead of handing out bytes past the bound. A zero budget is the
// stream-through default: Admit is still called exactly as before, its error is still swallowed, and
// the faulted slab is handed to the caller and dropped, byte-for-byte unchanged. It never mutates
// the tier on refusal.
func (t *ExpertCheckpointTier) admitBounded(id polymodel.ModelID, name string, w expertWeight, slab int64) (expertWeight, error) {
	evicted, admitErr := t.pool.Admit(polymodel.Model{ID: id, WeightBytes: slab})
	if admitErr != nil {
		if t.pool.Budget() > 0 {
			refusal := &ExpertCheckpointBudgetError{Tensor: name, Bytes: slab, Budget: t.pool.Budget()}
			t.failures, t.lastErr = t.failures+1, refusal
			return expertWeight{}, refusal
		}
		return w, nil // stream-through (the default): never refused, never retained
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

// fault answers one canonical per-expert tensor name by reading exactly that expert's stride â€” or
// by serving a retained host copy when the bounded host cache holds one. It never returns a
// partially-built weight.
func (t *ExpertCheckpointTier) fault(name string) (w expertWeight, err error) {
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
	if r, ok := t.ready[id]; ok {
		delete(t.ready, id)
		t.prefetchHits++
		t.mu.Unlock()
		return r.weight, nil
	}
	if load := t.loading[id]; load != nil {
		t.coalesced++
		load.waiters++
		t.mu.Unlock()
		<-load.done
		return load.weight, load.err
	}
	load := &expertCheckpointLoad{done: make(chan struct{})}
	if t.loading == nil {
		t.loading = make(map[polymodel.ModelID]*expertCheckpointLoad)
	}
	t.loading[id] = load
	src := t.shards[entry.shard]
	overlay := t.overlays[entry.fused] // nil unless a per-rank sparse overlay was registered
	t.mu.Unlock()
	completed := false
	defer func() {
		t.mu.Lock()
		defer t.mu.Unlock()
		if !completed {
			w, err = expertWeight{}, ErrExpertCheckpointLoadAborted
			t.failures, t.lastErr = t.failures+1, err
		}
		load.weight, load.err = w, err
		delete(t.loading, id)
		close(load.done)
	}()
	w, err = t.faultOwned(name, id, entry, src, overlay)
	completed = true
	return w, err
}

// Prefetch starts one concurrent read for each named projection that is indexed, not retained,
// not in flight and not already prefetched, and returns how many it started. A page-in at queue
// depth one leaves the NVMe mostly idle; reading a layer's misses together keeps several requests
// outstanding. A read that a demand fault joined is handed to it through the in-flight load; one
// that finishes first waits in ready for the next fault of that name. Ready entries from the
// previous generation are dropped, so at most two layers' prefetches are held.
//
// PINNED-BUFFER IDENTITY + DEPENDENCY ORDERING (vLLM offloader port, see THIRD_PARTY_NOTICES.md).
// vLLM's static buffer pool keys a staging buffer by the parameter's IDENTITY -- name, shape,
// stride, dtype -- and reuses it across layers, while its prefetch custom ops declare the
// prefetch/use edges as data dependencies so the compiler cannot reorder a wait ahead of the copy
// it guards. This port takes the two parts that survive the move to a synchronous ReaderAt ring
// fill:
//
//  1. IDENTITY. A miss is identified by its name, so the same name arriving twice in one layer's
//     pick list is one in-flight read, not two -- the existing t.loading guard already enforced
//     that inside this method's single lock hold, and it is now covered by a regression witness.
//     The read's FILE RANGE (src.expertReadSpan) is derived from the directory geometry alone,
//     moving no payload, and is what the dispatch order below is stated in.
//  2. ORDERING. Jobs are dispatched in ascending file-offset order, so the fill walks the shard
//     forward instead of seeking at random. The dependency edge itself was already here -- a
//     demand fault joins the in-flight load through t.loading and observes the completed bytes
//     through load.done -- and this port adds a bounded worker pool so a layer's misses cannot fan
//     out past a fixed concurrency.
//
// fakExpertPrefetchWorkers caps how many reads a single fill has in flight. It is deliberately NOT
// 1 (that is the queue-depth-one baseline P2 step 1 already beat) and not unbounded (a deep top-k
// over several shard readers would spawn one goroutine per miss).
const fakExpertPrefetchWorkers = 8

// fakExpertPrefetchJobs caps the AGGREGATE speculative reads in flight across every simultaneous
// Prefetch call. fakExpertPrefetchWorkers bounds one call's readers; without an aggregate bound N
// concurrent callers each get 8 readers and the tier's speculative read set is unbounded. The cap
// is on admitted jobs, so it also bounds the job slice and the goroutines a call may start: a call
// admits at most this many jobs and declines the rest rather than allocating one per input name.
const fakExpertPrefetchJobs = 32

// prefetchDeclined reports whether speculative work must be declined outright for this entry.
// Called with t.mu held. The issue's contract is that a hint is only started when its owned
// allocation is fully accounted for under the tier's EXISTING positive host allowance; a hint that
// cannot be charged is declined and demand serving is unaffected.
//
// A zero budget is NOT a decline: zero is the documented stream-through default, where the faulted
// slab is handed to the caller and dropped, so a prefetch of a zero-budget tier still starts reads
// (the existing dedupe/ordering/bounded-concurrency witnesses depend on that). What a zero budget
// cannot do is RESERVE, so a zero-budget prefetch is admitted only while the aggregate job bound
// has room, and it holds no reservation.
//
// The two unsupported-allocation paths are declined because their true allocation extent is not
// the projection stride: a page-cache cold read allocates a page-aligned buffer and returns a
// stride-length subslice (expert_pagecache.go), and a sparse overlay read has its own allocation
// path. Charging the stride for either would undercount the backing allocation, so this leaf
// declines them rather than silently assuming their extent.
func (t *ExpertCheckpointTier) prefetchDeclined(src *ggufExpertSource, overlay *ExpertSparseOverlay) bool {
	if overlay != nil {
		return true
	}
	if src == nil {
		return true
	}
	return src.PageCacheEnabled()
}

func (t *ExpertCheckpointTier) Prefetch(names []string) int {
	if t == nil || len(names) == 0 {
		return 0
	}
	type job struct {
		name    string
		id      polymodel.ModelID
		entry   expertCheckpointEntry
		src     *ggufExpertSource
		overlay *ExpertSparseOverlay
		load    *expertCheckpointLoad
		off     int64
	}
	t.mu.Lock()
	t.readyGen++
	for id, r := range t.ready {
		if r.gen+1 < t.readyGen {
			delete(t.ready, id)
			t.prefetchStale++
		}
	}
	remainingJobs := max(0, fakExpertPrefetchJobs-t.prefetchJobs)
	jobs := make([]job, 0, min(len(names), remainingJobs))
	for _, name := range names {
		id := polymodel.ModelID(name)
		entry, indexed := t.index[name]
		if !indexed {
			continue
		}
		if _, live := t.host[id]; live {
			continue
		}
		if _, ok := t.ready[id]; ok {
			continue
		}
		// The in-flight guard is also the identity dedup: a name repeated in this slice finds its
		// own load here and starts no second read.
		if t.loading[id] != nil {
			continue
		}
		src := t.shards[entry.shard]
		overlay := t.overlays[entry.fused]
		if t.prefetchDeclined(src, overlay) {
			continue
		}
		// Aggregate admission: bound the job collection itself, so a caller passing thousands of
		// names allocates and starts at most the remaining aggregate allowance. Best-effort and
		// nonblocking -- a declined hint is simply not started, and demand serving is untouched.
		if t.prefetchJobs+len(jobs) >= fakExpertPrefetchJobs {
			break
		}
		if t.loading == nil {
			t.loading = make(map[polymodel.ModelID]*expertCheckpointLoad)
		}
		load := &expertCheckpointLoad{done: make(chan struct{})}
		t.loading[id] = load
		var off int64
		if src != nil {
			off, _, _ = src.expertReadSpan(entry.fused, entry.expert)
		}
		jobs = append(jobs, job{name: name, id: id, entry: entry, src: src, overlay: overlay, load: load, off: off})
	}
	// Ascending file offset is the fill ORDER (dependency ordering). Offsets come from
	// src.expertReadSpan, which moves no payload; off stays 0 for a shard that could not be
	// resolved, which sorts to the front and cannot reorder a readable job's range.
	sort.SliceStable(jobs, func(i, j int) bool { return jobs[i].off < jobs[j].off })
	gen := t.readyGen
	t.prefetched += len(jobs)
	t.prefetchJobs += len(jobs)
	t.mu.Unlock()

	// Bounded worker pool over an ordered queue. Prefetch enqueues every job (in ascending-offset
	// order) into a buffer large enough to never block, then returns; a fixed pool of
	// fakExpertPrefetchWorkers goroutines drains it. So the cap is on reads IN FLIGHT, not reads
	// started — a layer's whole miss set is handed off without the caller waiting on IO (Prefetch's
	// historical non-blocking contract), while at most 8 readers touch the shard at once, and the
	// queue preserves the ascending order the sort above established.
	queue := make(chan job, len(jobs))
	workers := fakExpertPrefetchWorkers
	if workers > len(jobs) {
		workers = len(jobs)
	}
	// Goexit settles the current load in finishPrefetchJob but also terminates its
	// queue worker. Replace that reader slot after settlement so queued owners are
	// not stranded. Do not inspect len(queue): the producer may still be enqueueing.
	// A successor consumes the same finite queue and generation without admission.
	// Active reads remain bounded by workers; transient unwinding goroutines may
	// overlap successors, with at most workers+len(jobs) total goroutine starts.
	var drain func()
	drain = func() {
		drained := false
		defer func() {
			if !drained {
				go drain()
			}
		}()
		for j := range queue {
			t.finishPrefetchJob(j.name, j.id, j.entry, j.src, j.overlay, j.load, gen)
		}
		drained = true
	}
	for w := 0; w < workers; w++ {
		go drain()
	}
	for _, j := range jobs {
		queue <- j
	}
	close(queue)
	return len(jobs)
}

// finishPrefetchJob runs one prefetch read and publishes its handoff. It is the body the goroutine
// used to inline, lifted out so the bounded pool above reads as a pool rather than a spawn loop.
//
// Exactly-once completion: the deferred finalizer is the ONLY place that settles this job's
// ownership -- it removes that exact loading owner, releases the aggregate job counter, publishes
// the permitted result, and closes the completion notification. It runs on success, read error,
// panic, and Goexit alike, so a best-effort hint can never strand its loading entry, its waiters,
// or its share of the aggregate bound. A background panic is recovered into the existing
// aborted-load error class rather than terminating the process.
func (t *ExpertCheckpointTier) finishPrefetchJob(name string, id polymodel.ModelID, entry expertCheckpointEntry, src *ggufExpertSource, overlay *ExpertSparseOverlay, load *expertCheckpointLoad, gen uint64) {
	var (
		w         expertWeight
		err       error
		completed bool
	)
	defer func() {
		if r := recover(); r != nil {
			w, err, completed = expertWeight{}, ErrExpertCheckpointLoadAborted, false
		}
		t.mu.Lock()
		defer t.mu.Unlock()
		if !completed {
			err = ErrExpertCheckpointLoadAborted
			t.failures, t.lastErr = t.failures+1, err
		}
		load.weight, load.err = w, err
		delete(t.loading, id)
		if t.prefetchJobs > 0 {
			t.prefetchJobs--
		}
		if err == nil && load.waiters == 0 {
			if _, live := t.host[id]; !live {
				if gen+1 < t.readyGen {
					t.prefetchStale++
				} else {
					if t.ready == nil {
						t.ready = make(map[polymodel.ModelID]expertCheckpointReady)
					}
					t.ready[id] = expertCheckpointReady{weight: w, gen: gen}
				}
			}
		}
		close(load.done)
	}()
	w, err = t.faultOwned(name, id, entry, src, overlay)
	completed = true
}

func (t *ExpertCheckpointTier) faultOwned(name string, id polymodel.ModelID, entry expertCheckpointEntry, src *ggufExpertSource, overlay *ExpertSparseOverlay) (expertWeight, error) {

	// Sparse-overlay fast path (#13030): when this rank registered an overlay banding this fused
	// tensor AND the routed expert is bitmap-present, serve the copied rows and move no device
	// bytes. An expert outside the overlay (or no overlay at all) falls through to the dense stride
	// below, byte-for-byte, so the default-off path is exactly the historical one.
	var raw []byte
	if overlay != nil {
		if have, ok := overlay.row(name); ok {
			t.mu.Lock()
			t.overlayRows++
			t.overlayBytes += int64(len(have))
			t.mu.Unlock()
			w := entry.weight(name, have)
			// An overlay-served fault still populates the bounded host cache, so a repeat fault of
			// the same expert is a host hit and the overlay bytes are moved exactly once.
			t.mu.Lock()
			defer t.mu.Unlock()
			return t.admitBounded(id, name, w, int64(len(have)))
		}
	}

	// The narrow Q2_K/Q3_K stream-through path can borrow immutable checkpoint
	// bytes under the model/session lifetime. Staging separately requires an
	// explicit borrow-safe upload capability or makes a consumer-owned copy.
	// Retained caches, overlays (including absent
	// rows), and page-cache accounting keep their existing read contract. Source
	// I/O stays outside the bookkeeping lock; ownership/publication below is shared.
	var err error
	var borrowed bool
	if (entry.quant == ExpertCheckpointQ2K || entry.quant == ExpertCheckpointQ3K) && t.pool.Budget() == 0 && overlay == nil && !src.PageCacheEnabled() {
		_, borrowed = src.r.(ggufReadOnlyRange)
		raw, err = src.readExpertReadOnly(entry.fused, entry.expert)
	} else {
		raw, _, err = src.readExpertPageCache(entry.fused, entry.expert)
	}
	if err != nil {
		t.mu.Lock()
		t.failures, t.lastErr = t.failures+1, err
		t.mu.Unlock()
		return expertWeight{}, err
	}
	w := entry.weight(name, raw)
	w.borrowed = borrowed

	t.mu.Lock()
	defer t.mu.Unlock()
	t.reads++
	t.bytesRead += int64(len(raw))
	return t.admitBounded(id, name, w, int64(len(raw)))
}

// ExpertCheckpointStats is the checkpoint tier's ledger â€” the evidence for this rung's claim that
// bytes read per decode step scale with the ACTIVATED count k rather than with the expert count E.
// A hit rate alone cannot show that: the number that matters is BytesRead against the slab bytes a
// fully-resident load would have paid.
type ExpertCheckpointStats struct {
	Enabled bool `json:"enabled"`
	// Tensors is how many per-expert projections the tier indexes â€” E*3 per MoE layer indexed.
	Tensors int `json:"tensors"`
	// Reads is how many expert faults reached the checkpoint; BytesRead their total stride bytes.
	// Hits are faults served from the retained host cache, Evictions its page-outs.
	Reads     int   `json:"reads"`
	BytesRead int64 `json:"bytes_read"`
	Hits      int   `json:"hits"`
	// Coalesced counts callers that shared an in-flight fault, separately from retained hits.
	Coalesced int `json:"coalesced"`
	// Prefetched counts reads Prefetch started, PrefetchHits demand faults served from their
	// handoff, PrefetchStale handoffs dropped unconsumed.
	Prefetched    int `json:"prefetched"`
	PrefetchHits  int `json:"prefetch_hits"`
	PrefetchStale int `json:"prefetch_stale"`

	// Readahead records advisory outcomes separately from demand reads.
	Readahead ExpertCheckpointReadaheadStats `json:"readahead"`

	Evictions int `json:"evictions"`
	// OverlayRows/OverlayBytesRead are the per-rank sparse-overlay ledger (#13030): how many faults
	// were served from a registered overlay's copied rows, and the row bytes those faults moved.
	// A fault outside every registered overlay falls through to the dense stride and is counted in
	// Reads/BytesRead, so the two ledgers together account for every byte the tier moved.
	OverlayRows      int   `json:"overlay_rows"`
	OverlayBytesRead int64 `json:"overlay_bytes_read"`
	// OverlayTensors/PresentRows describe the registered overlays: how many fused tensors carry one,
	// and the total present rows across them. Both are zero with no overlay registered.
	OverlayTensors int `json:"overlay_tensors"`
	OverlayPresent int `json:"overlay_present"`
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
		Coalesced:     t.coalesced,
		Prefetched:    t.prefetched,
		PrefetchHits:  t.prefetchHits,
		PrefetchStale: t.prefetchStale,
		Readahead:     t.readahead,
		Evictions:     t.evictions,
		BudgetBytes:   t.pool.Budget(),
		ResidentBytes: t.pool.Used(),
		PeakBytes:     t.peak,
		ResidentCount: len(t.host),
		Failures:      t.failures,

		OverlayRows:      t.overlayRows,
		OverlayBytesRead: t.overlayBytes,
		OverlayTensors:   len(t.overlays),
	}
	for _, o := range t.overlays {
		st.OverlayPresent += o.PresentRows()
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

// ---- agent-startup expert cache warming (CW-15, #13343) ----------------------
//
// expert_warm_profile.go (CW-10, #13349) SELECTS a bounded warm set: given a recorded
// demand profile and the tier's index, ExpertWarmPlanFor returns canonical routed-expert
// names whose bytes fit a budget. This section EXECUTES that plan: it faults exactly the
// selected projections through the existing tier fault + bounded pool admission, so a
// later real v41ExpertF32Into demand for a retained projection is served from residency
// with zero source reads — the disk fault the prefill first-token path pays (fak#13294)
// is removed for the routed experts a workload is about to touch.
//
// Three invariants make the warm safe rather than a second, hidden loader:
//
//   - Bounded reuse only. The warm set is the SELECTED prefix of the plan, and a
//     projection is faulted only when it fits the pool's REMAINING budget without
//     eviction. A warm that would evict a resident to seat a speculative expert is a
//     pessimization (it drops demanded state for a guess), so the executor simply stops
//     instead. The warm never resizes the pool, never pins, and never raises a ceiling.
//
//   - Demand hot state is preserved. The fit check above means admission never needs to
//     evict; whatever the demand path already made resident stays resident. No permanent
//     pins are added and no ExpertRing is introduced.
//
//   - Attributed separately. Warm source reads land in the tier's own gate counters
//     (Reads/BytesRead through the same fault door), and the result reports the actual
//     retained/read/skipped counts. The caller's profile value is never mutated, so a
//     warm can never feed the recorded-demand histogram it was selected from.
//
// A zero or insufficient retention budget warms nothing and reports skipped, never
// "warm": issuing reads a zero-budget tier streams and drops is not residency.

// ExpertWarmResult is the typed receipt of one warm execution. Its counters are ACTUAL, read
// back from the tier and the pool after the run — never the plan's projections — so a caller
// can distinguish "selected N" from "retained N" and can never mistake a plan for a residency.
type ExpertWarmResult struct {
	// Reason is the closed verdict: the plan's own reason when selection declined, or
	// ExpertWarmReasonPlanned when at least one projection was retained.
	Reason ExpertWarmReason
	// Selected is how many projections the plan named; Retained how many were faulted and
	// kept. Retained <= Selected always.
	Selected int
	// Retained/RetainedBytes are the projections actually seated in the host cache and
	// their resident bytes. Skipped counts selected projections the bounded warm declined
	// (they did not fit the remaining budget without eviction).
	Retained      int
	RetainedBytes int64
	Skipped       int
	// Reads/ReadBytes are the source reads the warm issued, as the tier gate counted them
	// (a projection already resident costs none). Evicted is how many residents admission
	// retired; by construction the bounded warm performs no eviction, so this is 0 on a
	// successful warm and is carried only so a caller can detect an unexpected one.
	Reads     int
	ReadBytes int64
	Evicted   int
	// Failed counts selected projections whose fault errored (IO failure or a typed
	// budget refusal). A warm is best-effort: a failure seats nothing and never claims a
	// hit, so the demand path still faults the projection itself.
	Failed int
}

// Warmed reports whether the warm retained at least one projection's bytes. It is the single
// bit a caller branches on to decide whether a later demand can hit residency; a selected but
// unseated plan (insufficient budget, stale identity, IO failure) reports false.
func (r ExpertWarmResult) Warmed() bool { return r.Retained > 0 && r.RetainedBytes > 0 }

// WarmExpertProfile executes a bounded warm of the expert cache from a recorded demand
// profile. It selects the set with the tier's ExpertWarmPlanFor (CW-10, #13349) and faults
// exactly the selected projections that fit the tier's REMAINING residency budget without
// eviction, returning the ACTUAL retained/read/skipped counters.
//
// It is inert on the default path: an absent/empty/stale profile, an unindexed expert, a
// zero budget or a budget smaller than one projection all warm nothing and return the
// plan's own reason, so a model that never opts into warming keeps the historical path
// byte-for-byte. A nil model or a model with no checkpoint tier warms nothing and never
// panics.
//
// budgetBytes is the caller's ceiling on bytes this warm may retain. It is a second bound
// on top of the tier's own host budget, not a replacement for it: the warm never seats more
// than min(budgetBytes, remaining pool budget).
func (m *Model) WarmExpertProfile(profile ExpertWarmProfile, identity ExpertWarmProfileIdentity, budgetBytes int64) ExpertWarmResult {
	if m == nil || m.expertCheckpoint == nil {
		// Nothing to warm from. Reuse the closed vocabulary: a profile with no indexed
		// expert is exactly this shape, and it is honest for a tier-less model.
		return ExpertWarmResult{Reason: ExpertWarmReasonNoIndexedExpert}
	}
	t := m.expertCheckpoint

	plan := t.ExpertWarmPlanFor(profile, identity, budgetBytes)
	res := ExpertWarmResult{
		Reason:   plan.Reason,
		Selected: len(plan.Selected),
		Skipped:  plan.Skipped,
	}
	if plan.Empty() {
		// Selection declined (no profile, stale identity, zero budget, nothing fits, or
		// no indexed expert): warm nothing, never stream-and-drop.
		return res
	}

	before := t.Stats()
	// The running allowance starts at the caller's full budget, NOT budgetBytes minus the
	// plan total: the plan's own byte sum is the projection to be seated, and deducting it
	// up front would leave no room to seat any of it. Each seated projection then charges
	// its real stride against the allowance below.
	remaining := budgetBytes
	for _, sel := range plan.Selected {
		// The byte ceiling, checked on the actual stride the tier reported, never a
		// profile estimate. A projection that cannot fit the remaining allowance is
		// skipped, not overrun.
		if sel.Bytes <= 0 || sel.Bytes > remaining {
			res.Skipped++
			continue
		}
		if !t.warmFits(sel.Bytes) {
			// Seating this projection would require evicting a resident — i.e. demanded
			// hot state. A speculative warm must never displace demanded residency, so the
			// bounded warm stops here and reports the projections it declined.
			res.Skipped++
			continue
		}
		if _, err := t.fault(sel.Name); err != nil {
			res.Failed++
			continue
		}
		res.Retained++
		res.RetainedBytes += sel.Bytes
		remaining -= sel.Bytes
	}

	after := t.Stats()
	res.Reads = after.Reads - before.Reads
	res.ReadBytes = after.BytesRead - before.BytesRead
	res.Evicted = after.Evictions - before.Evictions
	if res.Retained == 0 {
		// Selection produced a plan but the bounded execution seated nothing: report the
		// plan's reason so the caller sees why no residency was established.
		if res.Reason == ExpertWarmReasonPlanned {
			res.Reason = ExpertWarmReasonNoFit
		}
		return res
	}
	res.Reason = ExpertWarmReasonPlanned
	return res
}

// warmFits reports whether a slab of slabBytes can be made resident within the pool's
// REMAINING budget without evicting any current resident — the fit-before-fault guard that
// keeps a speculative warm from displacing demanded hot state. It takes the tier lock only
// long enough to read the pool's used/budget, and performs no mutation, so a caller can
// decide to skip before paying the fault. A zero-budget (stream-through) pool never fits
// anything, which is the intended "never warm" default.
func (t *ExpertCheckpointTier) warmFits(slabBytes int64) bool {
	if t == nil || t.pool == nil || slabBytes <= 0 {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	budget := t.pool.Budget()
	if budget <= 0 {
		return false
	}
	return t.pool.Used()+slabBytes <= budget
}
