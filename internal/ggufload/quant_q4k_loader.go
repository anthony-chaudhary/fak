package ggufload

// quant_q4k_loader.go — the direct-q4 GGUF loader for the resident Q4_K path
// (QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P1). Mirrors WeightSource.QuantModelProfile but
// routes every eligible Q4_K matmul tensor straight into a resident q4kTensor (raw GGUF
// bytes, no dequantF32 Q4→f32, no f32→Q8 re-quant): the ~10× load win + the drop in
// resident footprint, streaming the q4_k_m bytes llama.cpp streams.
//
// Eligibility (model.ResidentQ4KEligible) is the correctness gate: only IDENTITY-
// normalized matmul weights (MLP gate/up/down, self_attn.v_proj/o_proj, lm_head, expert
// FFN) are held raw, because the GGUF's ggml-layout bytes are already the HF layout the
// forward expects for those. The normalize-sensitive weights (qwen35 linear_attn family +
// rotary/gated self_attn q/k/qkv) MUST stay on the proven dequant→normalize→Q8 path —
// storing their raw bytes would feed wrongly-laid-out weights to the forward and produce
// garbage. The Q6_K matmul minority (often attn_qkv / ffn_down / lm_head in a q4_k_m mix)
// also falls through to Q8, since the resident q4kTensor holds Q4_K blocks only.

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// LoadModelQ4K loads a GGUF checkpoint through the direct-resident-Q4_K path: eligible
// Q4_K matmul tensors are held raw (no round-trip), and everything else follows the
// standard quant-on-load path (Q8_0 for the remaining matmul weights, f32 for small
// tensors). Run the returned model through a Session with Q4K=true.
func LoadModelQ4K(path string) (*model.Model, error) {
	return LoadModelQ4KContext(context.Background(), path)
}

// LoadModelQ4KContext is LoadModelQ4K with cooperative cancellation. It does not return
// until admitted tensor work has drained and the checkpoint readers have been closed.
func LoadModelQ4KContext(ctx context.Context, path string) (*model.Model, error) {
	return LoadModelQ4KProfileContext(ctx, path, nil)
}

// ExpertShard names the routed expert band [Lo,Hi) a rank owns during an expert-parallel
// load. Dense, attention, router, embeddings, and shared-expert tensors remain replicated; this
// band filters only batched routed-expert tensors before they enter the resident store.
type ExpertShard struct {
	Lo int
	Hi int
}

// ExpertShardForRank derives the contiguous expert band owned by rank under the same tiling as
// model.ExpertParallelPlan. It is a loader-facing helper so serve code can use the planner and the
// resident admission filter from one source of truth.
func ExpertShardForRank(numExperts, ranks, rank int) (ExpertShard, error) {
	plan, err := model.ExpertParallelPlan(numExperts, ranks)
	if err != nil {
		return ExpertShard{}, err
	}
	if rank < 0 || rank >= len(plan.Shards) {
		return ExpertShard{}, fmt.Errorf("gguf: expert-parallel rank %d outside [0,%d)", rank, len(plan.Shards))
	}
	shard := plan.Shards[rank]
	return ExpertShard{Lo: shard.Lo, Hi: shard.Hi}, nil
}

type q4kLoadOptions struct {
	expertShardSet       bool
	expertShard          ExpertShard
	residentDenseKQuant  bool
	residentDenseQ2K     bool
	residentQ2KEmbedding bool
	residentQ4KEmbedding bool
	streamedExperts      bool
	streamedExpertBytes  int64
	streamedDenseQ4K     bool
}

// Q4KLoadOption configures the direct-resident-Q4_K GGUF load path.
type Q4KLoadOption func(*q4kLoadOptions)

// WithQ2KEmbeddingResident controls whether eligible Q2_K token embedding tables stay in
// raw Q2_K packed format for on-demand row gathering, skipping full F32 expansion.
func WithQ2KEmbeddingResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.residentQ2KEmbedding = enabled }
}

// WithQ4KEmbeddingResident controls whether an eligible Q4_K token embedding table stays
// in its source format for bounded row gathering. It is intentionally default-off.
func WithQ4KEmbeddingResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.residentQ4KEmbedding = enabled }
}

// WithDenseKQuantResident controls whether eligible dense Q5_K/Q6_K/IQ tensors stay in
// the raw k-quant store. Backends without dense k-quant kernels must disable this so those
// tensors follow the proven dequant-to-Q8 path instead of becoming unreachable at decode.
func WithDenseKQuantResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.residentDenseKQuant = enabled }
}

// WithDenseQ2KResident retains eligible dense Q2_K tensors even when blanket dense
// k-quant residency is disabled. This lets a backend with a Q2_K HAL path avoid the
// dequant-to-Q8 round trip without stranding unsupported IQ/Q3 formats.
func WithDenseQ2KResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.residentDenseQ2K = enabled }
}

// WithExpertShard keeps only routed experts in [lo,hi) when splitting batched MoE expert GGUF
// tensors. Use this for expert-parallel per-rank loads; omit it for the historical full load.
func WithExpertShard(lo, hi int) Q4KLoadOption {
	return func(o *q4kLoadOptions) {
		o.expertShardSet = true
		o.expertShard = ExpertShard{Lo: lo, Hi: hi}
	}
}

// WithStreamedExperts leaves the batched routed-expert slabs ON DISK and attaches an R5 checkpoint
// tier (#5616) over them instead of materializing E per-expert copies at load: a routed expert is
// then read, one stride at a time, when a router actually picks it. hostBytes is the tier's host
// retention budget; 0 — the value to pass unless you have measured a reason not to — is
// stream-through, where an expert is read, handed to the bounded device ring and dropped, so host
// residency for the expert bulk stays at zero and a checkpoint bigger than host RAM is servable.
//
// This option needs a checkpoint that stays OPEN for the life of the model (the tier reads through
// the WeightSource's own shard readers), so it is available only on the WeightSource-form entry
// points; LoadModelQ4KProfileOptions refuses it rather than returning a model whose experts became
// unreadable when it closed the source.
func WithStreamedExperts(hostBytes int64) Q4KLoadOption {
	return func(o *q4kLoadOptions) {
		o.streamedExperts = true
		o.streamedExpertBytes = hostBytes
	}
}

// WithStreamedDenseQ4K leaves eligible identity-layout dense Q4_K tensors on disk. The returned model retains checkpoint range descriptors and requires the WeightSource to stay open.
func WithStreamedDenseQ4K(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.streamedDenseQ4K = enabled }
}

// probeQ4KLoadOptions applies opts to the zero value without the config-dependent validation, for
// callers that must inspect a request BEFORE they have a parsed checkpoint to validate it against.
func probeQ4KLoadOptions(opts []Q4KLoadOption) q4kLoadOptions {
	var out q4kLoadOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&out)
		}
	}
	return out
}

func resolveQ4KLoadOptions(cfg model.Config, opts []Q4KLoadOption) (q4kLoadOptions, error) {
	out := q4kLoadOptions{residentDenseKQuant: true}
	for _, opt := range opts {
		if opt != nil {
			opt(&out)
		}
	}
	if out.residentQ2KEmbedding && out.residentQ4KEmbedding {
		return out, fmt.Errorf("gguf: Q2_K and Q4_K resident embedding modes are mutually exclusive")
	}
	if out.streamedExperts {
		if out.streamedExpertBytes < 0 {
			return out, fmt.Errorf("gguf: streamed-expert host budget %d is negative", out.streamedExpertBytes)
		}
		// A streamed tier serves EVERY expert the checkpoint carries; an expert-parallel band says
		// this rank must never touch the others. Honouring both would mean silently serving a rank
		// experts it was sharded out of, so refuse instead of picking one.
		if out.expertShardSet {
			return out, fmt.Errorf("gguf: streamed routed experts and an expert-parallel shard [%d,%d) cannot both be requested",
				out.expertShard.Lo, out.expertShard.Hi)
		}
	}
	if !out.expertShardSet {
		return out, nil
	}
	if cfg.NumExperts <= 0 {
		return out, fmt.Errorf("gguf: expert shard requested for non-MoE config (NumExperts=%d)", cfg.NumExperts)
	}
	if out.expertShard.Lo < 0 || out.expertShard.Hi <= out.expertShard.Lo || out.expertShard.Hi > cfg.NumExperts {
		return out, fmt.Errorf("gguf: expert shard [%d,%d) outside [0,%d)", out.expertShard.Lo, out.expertShard.Hi, cfg.NumExperts)
	}
	return out, nil
}

func (o q4kLoadOptions) keepExpert(expert int) bool {
	if !o.expertShardSet {
		return true
	}
	return expert >= o.expertShard.Lo && expert < o.expertShard.Hi
}

func (o q4kLoadOptions) keptExperts(total int) int {
	if !o.expertShardSet {
		return total
	}
	lo, hi := o.expertShard.Lo, o.expertShard.Hi
	if lo < 0 {
		lo = 0
	}
	if hi > total {
		hi = total
	}
	if hi <= lo {
		return 0
	}
	return hi - lo
}

// LoadModelQ4KProfile is LoadModelQ4K with an optional load profiler so the direct-resident-Q4_K
// path streams the same load-progress lines the lean-Q8 path does (a 466 GB GLM-5.2 resident load
// must not be silent). Nil profiler = no progress, byte-identical to the old LoadModelQ4K.
func LoadModelQ4KProfile(path string, p *LoadProfiler) (*model.Model, error) {
	return LoadModelQ4KProfileContext(context.Background(), path, p)
}

// LoadModelQ4KProfileContext is LoadModelQ4KProfile with cooperative cancellation.
func LoadModelQ4KProfileContext(ctx context.Context, path string, p *LoadProfiler) (*model.Model, error) {
	return LoadModelQ4KProfileOptionsContext(ctx, path, p)
}

// LoadModelQ4KProfileOptions is LoadModelQ4KProfile with explicit load options.
//
// It refuses WithStreamedExperts: this entry point closes the checkpoint before it returns
// (loadVia), and an R5 checkpoint tier reads through those very readers, so the model it handed
// back would fail on the first routed expert a router picked — a checkpoint whose expert bulk is
// resident nowhere else has no fallback to degrade to. Open the checkpoint yourself and keep it
// open for the model's life instead.
func LoadModelQ4KProfileOptions(path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return LoadModelQ4KProfileOptionsContext(context.Background(), path, p, opts...)
}

// LoadModelQ4KProfileOptionsContext is LoadModelQ4KProfileOptions with cooperative
// cancellation. Reader cleanup happens synchronously, exactly once, before it returns.
func LoadModelQ4KProfileOptionsContext(ctx context.Context, path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return loadModelQ4KProfileOptionsContext(ctx, path, p, OpenWeights, opts...)
}

// LoadQ4KModelContextWithProgress is an alias for LoadModelQ4KProfileOptionsContext.
func LoadQ4KModelContextWithProgress(ctx context.Context, path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return LoadModelQ4KProfileOptionsContext(ctx, path, p, opts...)
}

func loadModelQ4KProfileOptionsContext(ctx context.Context, path string, p *LoadProfiler, open func(string) (*WeightSource, error), opts ...Q4KLoadOption) (*model.Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if o := probeQ4KLoadOptions(opts); o.streamedExperts || o.streamedDenseQ4K {
		return nil, fmt.Errorf("gguf: streamed weights need a checkpoint that outlives the model; " +
			"LoadModelQ4KProfileOptions closes it on return — use OpenWeights + (*WeightSource).QuantModelQ4KProfileOptions " +
			"and close the source only after the model is done")
	}
	ws, err := open(path)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return ws.QuantModelQ4KProfileOptionsContext(ctx, p, opts...)
}

// LoadModelQ4KStreamedDense opens path and transfers the checkpoint lifetime to the returned model. CloseWeights must be called when serving stops.
func LoadModelQ4KStreamedDense(path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return LoadModelQ4KStreamedDenseContext(context.Background(), path, p, opts...)
}

// LoadModelQ4KStreamedDenseContext is LoadModelQ4KStreamedDense with cooperative cancellation.
// On cancellation it closes the checkpoint before returning.
func LoadModelQ4KStreamedDenseContext(ctx context.Context, path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	ws, err := OpenWeights(path)
	if err != nil {
		return nil, err
	}
	opts = append(opts, WithStreamedDenseQ4K(true))
	m, err := ws.QuantModelQ4KProfileOptionsContext(ctx, p, opts...)
	if err != nil {
		_ = ws.Close()
		return nil, err
	}
	m.SetWeightCloser(ws)
	return m, nil
}

// QuantModelQ4K is the WeightSource form of LoadModelQ4K: QuantModelProfile with the
// eligible-Q4_K branch pulled before the dequant, so those tensors never pay the f32
// round-trip.
func (s *WeightSource) QuantModelQ4K() (*model.Model, error) {
	return s.QuantModelQ4KContext(context.Background())
}

// QuantModelQ4KContext is QuantModelQ4K with cooperative cancellation.
func (s *WeightSource) QuantModelQ4KContext(ctx context.Context) (*model.Model, error) {
	return s.QuantModelQ4KProfileContext(ctx, nil)
}

// QuantModelQ4KProfile is QuantModelQ4K with optional progress reporting (p.SetTotal/Tick).
//
// The per-tensor work (read + dequant + normalize + expert split) runs on a bounded worker
// pool (gguf_parload.go); the builder mutations + the MLA KV-b merge + the profiler are
// applied SERIALLY in original tensor order by a single collector, so the built model is
// byte-identical to a serial load — only the CPU-bound dequant is parallelized. This is the
// S1 lever against the ~100-min single-core GLM-5.2 load
// (docs/notes/GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md): zero arithmetic change,
// every core busy. The collector also records the per-quant-type resident-vs-dequant
// breakdown (the S4 visibility) so the mixed-quant cost is legible without an external dump.
func (s *WeightSource) QuantModelQ4KProfile(p *LoadProfiler) (*model.Model, error) {
	return s.QuantModelQ4KProfileContext(context.Background(), p)
}

// QuantModelQ4KProfileContext is QuantModelQ4KProfile with cooperative cancellation.
func (s *WeightSource) QuantModelQ4KProfileContext(ctx context.Context, p *LoadProfiler) (*model.Model, error) {
	return s.QuantModelQ4KProfileOptionsContext(ctx, p)
}

// shapeAndBytesOrFail calls s.shapeAndBytes(info) and, on error, records it onto
// tw.err — the shared "err -> tw.err; bail" shape every shapeAndBytes call site
// in the per-tensor compute path below repeats. ok is false when the call
// failed (tw already carries the error; the caller should `return tw`
// immediately).
func (s *WeightSource) shapeAndBytesOrFail(info TensorInfo, tw *tensorWork) (shape []int, raw []byte, ok bool) {
	shape, raw, err := s.shapeAndBytes(info)
	if err != nil {
		tw.err = err
		return nil, nil, false
	}
	return shape, raw, true
}

func (s *WeightSource) validateResidentQ2KEmbedding(cfg model.Config) error {
	return s.validateResidentPackedEmbedding(cfg, TensorQ2_K, blockQ2KBytes)
}

func (s *WeightSource) validateResidentQ4KEmbedding(cfg model.Config) error {
	return s.validateResidentPackedEmbedding(cfg, TensorQ4_K, blockQ4KBytes)
}

func (s *WeightSource) validateResidentPackedEmbedding(cfg model.Config, wantType TensorType, blockBytes uint64) error {
	format := wantType.String()
	if !cfg.IsQwen35Hybrid() || cfg.IsMoE() {
		return fmt.Errorf("gguf: resident %s embedding requires a dense Qwen3.5-family hybrid model", format)
	}
	if cfg.TieWordEmbeddings {
		return fmt.Errorf("gguf: resident %s embedding requires untied word embeddings", format)
	}
	var (
		embInfo *TensorInfo
		outInfo *TensorInfo
	)
	for i := range s.File.Tensors {
		t := &s.File.Tensors[i]
		if t.Name == "token_embd.weight" {
			embInfo = t
		} else if t.Name == "output.weight" {
			outInfo = t
		}
	}
	if embInfo == nil {
		return fmt.Errorf("gguf: token_embd.weight tensor missing")
	}
	if outInfo == nil {
		return fmt.Errorf("gguf: resident %s embedding requires distinct output.weight", format)
	}
	if embInfo.FileOffset == outInfo.FileOffset && embInfo.Offset == outInfo.Offset {
		return fmt.Errorf("gguf: output.weight is aliased to token_embd.weight")
	}
	if embInfo.Type != wantType {
		return fmt.Errorf("gguf: token_embd.weight is %s, want %s", embInfo.Type, format)
	}
	shape, err := modelShapeFromGGUFDims(embInfo.Name, embInfo.Dims)
	if err != nil {
		return err
	}
	if len(shape) != 2 {
		return fmt.Errorf("gguf: token_embd.weight rank %d != 2", len(shape))
	}
	vocab, hidden := shape[0], shape[1]
	if cfg.HiddenSize != 0 && hidden != cfg.HiddenSize {
		return fmt.Errorf("gguf: token_embd.weight hidden %d != config %d", hidden, cfg.HiddenSize)
	}
	if cfg.VocabSize != 0 && vocab != cfg.VocabSize {
		return fmt.Errorf("gguf: token_embd.weight vocab %d != config %d", vocab, cfg.VocabSize)
	}
	if hidden%qkK != 0 {
		return fmt.Errorf("gguf: token_embd.weight hidden dimension %d is not divisible by %d", hidden, qkK)
	}
	wantPayload := uint64(vocab) * (uint64(hidden) / qkK) * blockBytes
	payloadBytes, err := tensorPayloadBytes(*embInfo)
	if err != nil {
		return err
	}
	if payloadBytes != wantPayload {
		return fmt.Errorf("gguf: token_embd.weight payload bytes %d != expected %d", payloadBytes, wantPayload)
	}
	return nil
}

// QuantModelQ4KProfileOptions is QuantModelQ4KProfile with explicit load options. The default
// option set is byte-compatible with QuantModelQ4KProfile; an expert shard only filters routed
// expert tensors after the GGUF batched expert split.
func (s *WeightSource) QuantModelQ4KProfileOptions(p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return s.QuantModelQ4KProfileOptionsContext(context.Background(), p, opts...)
}

// QuantModelQ4KProfileOptionsContext is QuantModelQ4KProfileOptions with cooperative cancellation.
func (s *WeightSource) QuantModelQ4KProfileOptionsContext(ctx context.Context, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	// Capture the default-off W3 decision once. Worker goroutines and GEMV loops never
	// re-read process environment, so a load has one immutable selection contract.
	w3Requested := model.W3MLPRequested()
	if w3Requested && (!cfg.IsQwen35Hybrid() || cfg.IsMoE()) {
		return nil, fmt.Errorf("gguf: FAK_W3_MLP requires a dense Qwen3.5-family hybrid model")
	}
	loadOpts, err := resolveQ4KLoadOptions(cfg, opts)
	if err != nil {
		return nil, err
	}
	if loadOpts.residentQ2KEmbedding {
		if err := s.validateResidentQ2KEmbedding(cfg); err != nil {
			return nil, err
		}
	}
	if loadOpts.residentQ4KEmbedding {
		if err := s.validateResidentQ4KEmbedding(cfg); err != nil {
			return nil, err
		}
	}
	// R5/#5616: under WithStreamedExperts the fused routed-expert slabs are described from the
	// tensor directory (no payload IO) and left on disk; `streamed` is the read-only set of GGUF
	// tensor names the tier took ownership of, which the per-tensor workers consult to skip
	// materializing them. Built BEFORE the load so a checkpoint the tier cannot serve is refused
	// up front rather than after paying for the whole load.
	var expertTier *model.ExpertCheckpointTier
	var streamed map[string]bool
	if loadOpts.streamedExperts {
		shards, err := s.FusedExpertTensors()
		if err != nil {
			return nil, err
		}
		expertTier, err = buildExpertCheckpointTier(shards, loadOpts.streamedExpertBytes)
		if err != nil {
			return nil, err
		}
		if expertTier == nil {
			return nil, fmt.Errorf("gguf: streamed routed experts requested, but this %s checkpoint carries no fused expert slab the tier can serve", cfg.ModelType)
		}
		streamed = make(map[string]bool)
		for _, sh := range shards {
			for _, f := range sh.Fused {
				streamed[f.Name] = true
			}
		}
	}

	builder := model.NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
	kvbHalf := map[int]glmKVBHalf{} // MLA KV-b 2->1 merge buffer (see QuantModelProfile)
	qwenMTPSeen := newQwen35MTPSeen(cfg)
	p.SetTotal(len(s.File.Tensors))

	// computeFn is the pure, concurrency-safe per-tensor work: it reads + dequantizes +
	// normalizes + splits, returning the builder mutations to apply. It touches no shared
	// state (TensorBytes copies; dequantF32 allocates fresh; the helpers are pure over the
	// read-only Config), so it is safe to run from many workers at once.
	computeFn := func(info TensorInfo, innerWorkers int) tensorWork {
		return s.computeQ4KTensorWork(info, cfg, w3Requested, streamed, loadOpts, innerWorkers)
	}

	// applyFn owns all shared mutable state (builder, KV-b merge buffer, profiler) and runs
	// on the single collector goroutine in original tensor order.
	applyFn := func(tw tensorWork) error {
		if err := applyQ4KTensorWork(tw, p, cfg, builder, kvbHalf, w3Requested); err != nil {
			return err
		}
		if tw.mtpMaterialized != "" {
			qwenMTPSeen[tw.mtpMaterialized] = true
		}
		return nil
	}

	if err := s.parallelQuantLoadContextBudget(ctx, computeFn, applyFn); err != nil {
		return nil, err
	}
	if err := glmKVBUnpaired(kvbHalf); err != nil {
		return nil, err
	}
	if err := validateQwen35MTPMaterialized(qwenMTPSeen); err != nil {
		return nil, err
	}
	if p != nil {
		p.EmitLoadPathSummary(p.Progress)
	}
	m, err := builder.Build()
	if err != nil {
		return nil, err
	}
	if expertTier != nil {
		m.SetExpertCheckpoint(expertTier)
		if p != nil && p.Progress != nil {
			st := expertTier.Stats()
			fmt.Fprintf(p.Progress, "experts: %d routed projections streamed from the checkpoint (host budget %d B)\n",
				st.Tensors, st.BudgetBytes)
		}
	}
	// #4974: reproduce the witnessed `numactl --interleave=all` weight placement in-process so the
	// CPU Q4_K decode path gets the multi-node bandwidth regime out of the box (no external wrapper).
	// Gated + no-op off linux/amd64, on a single-node/constrained host, or under FAK_NUMA_INTERLEAVE=off
	// (see Model.ApplyDecodeNUMAInterleave). Reported through the load profiler so the placement
	// decision that governed the run is visible on the load-path summary line.
	if lbl := m.ApplyDecodeNUMAInterleave(); p != nil && p.Progress != nil && lbl != "" {
		fmt.Fprintf(p.Progress, "numa: %s\n", lbl)
	}
	if lbl := m.ApplyNUMAWeightReplicas(""); p != nil && p.Progress != nil && lbl != "" {
		fmt.Fprintf(p.Progress, "numa_replicas: %s\n", lbl)
	}
	if w3Requested {
		if err := m.ValidateResidentW3MLP(); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// applyQ4KTensorWork is the collector-side apply step of QuantModelQ4KProfileOptions,
// extracted verbatim: it ticks the profiler and applies one tensorWork's pending builder
// mutations (KV-b half buffering + merge, resident raw-quant adds, f32 adds) in order. It
// owns all shared mutable state (builder, KV-b merge buffer, profiler) and must only run
// on the single collector goroutine in original tensor order.

func (s *WeightSource) lazyDenseQ4KTensorWork(info TensorInfo, canon string, tickBytes int64) tensorWork {
	tw := tensorWork{tickBytes: tickBytes}
	shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
	if err != nil {
		tw.err = err
		return tw
	}
	r, size, err := s.tensorReader(info)
	if err != nil {
		tw.err = err
		return tw
	}
	n, err := tensorPayloadBytes(info)
	if err != nil || n > uint64(math.MaxInt) {
		if err == nil {
			err = fmt.Errorf("gguf: tensor %s payload is too large", info.Name)
		}
		tw.err = err
		return tw
	}
	r = s.mappedQ4KReader(info, r, size, int(n))
	tw.pending = []pendingTensor{{lazyQ4K: true, name: canon, shape: shape, sourceInfo: info, lazyReader: r}}
	tw.acctType, tw.acctBytes, tw.acctTensors, tw.acctResident = info.Type.String(), tensorOnDiskBytes(info), 1, true
	return tw
}

func applyQ4KTensorWork(tw tensorWork, p *LoadProfiler, cfg model.Config, builder *model.QuantBuilder, kvbHalf map[int]glmKVBHalf, w3Requested bool) error {
	p.Tick(tw.tickBytes)
	p.recordLoadPath(tw.acctType, tw.acctExpert, tw.acctResident, tw.acctBytes, tw.acctTensors)
	for _, pt := range tw.pending {
		switch {
		case pt.canonicalMTPQ4K:
			if err := builder.AddCanonicalMTPQ4K(pt.name, pt.shape, pt.raw); err != nil {
				return err
			}
		case pt.canonicalMTPFCQ8:
			if err := builder.AddCanonicalMTPFCQ8(pt.name, pt.shape, pt.raw); err != nil {
				return err
			}
		case pt.q2kEmbed:
			embed, err := model.NewQ2KEmbedding(pt.raw, pt.shape[0], pt.shape[1])
			if err != nil {
				return err
			}
			if err := builder.SetQ2KEmbedding(embed); err != nil {
				return err
			}
		case pt.q4kEmbed:
			embed, err := model.NewQ4KEmbedding(pt.raw, pt.shape[0], pt.shape[1])
			if err != nil {
				return err
			}
			if err := builder.SetQ2KEmbedding(embed); err != nil {
				return err
			}
		case pt.lazyQ4K:
			n, err := tensorPayloadBytes(pt.sourceInfo)
			if err != nil {
				return err
			}
			src := model.LazyQ4KRange{Reader: pt.lazyReader, Offset: pt.sourceInfo.FileOffset, Bytes: int(n)}
			if mapped, ok := pt.lazyReader.(*mappedQ4KReaderAt); ok {
				src.MappedSpan = mapped.span
				src.MappedOffset = mapped.offset
			}
			if err := builder.AddLazyQ4K(pt.name, pt.shape, src); err != nil {
				return err
			}
		case pt.isKVBHalf:
			merged, ready, err := bufferGLMKVBHalf(kvbHalf, pt.layer, pt.half, pt.shape, pt.f32)
			if err != nil {
				return err
			}
			if ready {
				md, err := normalizeCanonicalTensorData(merged.Name, merged.Data, cfg)
				if err != nil {
					return err
				}
				if err := builder.AddF32Tensor(merged.Name, merged.Shape, md); err != nil {
					return err
				}
			}
		case pt.resident:
			switch pt.residentType {
			case TensorQ3_K:
				if err := builder.AddResidentQ3K(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ3_S:
				if err := builder.AddResidentIQ3S(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorQ6_K:
				if err := builder.AddResidentQ6K(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorQ5_K:
				if err := builder.AddResidentQ5K(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ3_XXS:
				add := builder.AddResidentIQ3XXS
				if w3Requested && model.ResidentW3MLPEligible(cfg, pt.name) {
					add = builder.AddResidentW3MLPIQ3XXS
				}
				if err := add(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ2_XXS:
				if err := builder.AddResidentIQ2XXS(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ2_XS:
				if err := builder.AddResidentIQ2XS(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ1_S:
				if err := builder.AddResidentIQ1S(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ2_S:
				if err := builder.AddResidentIQ2S(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ1_M:
				if err := builder.AddResidentIQ1M(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorIQ4_XS:
				if err := builder.AddResidentIQ4XS(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorQ8_0:
				if err := builder.AddResidentQ8_0(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorQ2_0:
				if err := builder.AddResidentQ2(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorQ4_0:
				// Must be an explicit arm: the default below is the Q4_K super-block wrapper, and
				// a 32-weight/18-byte Q4_0 payload handed to it fails the 256-weight/144-byte
				// geometry check. This switch is the single funnel every resident route lands in,
				// so naming Q4_0 here is what keeps the dense arm and the batched-expert arm from
				// half-applying the format.
				if err := builder.AddResidentQ4_0(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			case TensorQ2_K:
				if err := builder.AddResidentQ2K(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			default: // TensorQ4_K
				if err := builder.AddResidentQ4K(pt.name, pt.shape, pt.raw); err != nil {
					return err
				}
			}
		default:
			if err := builder.AddF32Tensor(pt.name, pt.shape, pt.f32); err != nil {
				return err
			}
		}
	}
	return nil
}

var qwen35MTPResidentTypes = map[string]TensorType{
	"mtp.fc.weight":                        TensorQ8_0,
	"mtp.layers.0.self_attn.q_proj.weight": TensorQ4_K,
	"mtp.layers.0.self_attn.k_proj.weight": TensorQ4_K,
	"mtp.layers.0.self_attn.v_proj.weight": TensorQ6_K,
	"mtp.layers.0.self_attn.o_proj.weight": TensorQ4_K,
	"mtp.layers.0.mlp.gate_proj.weight":    TensorQ4_K,
	"mtp.layers.0.mlp.up_proj.weight":      TensorQ4_K,
	"mtp.layers.0.mlp.down_proj.weight":    TensorQ6_K,
}

// reorderQwen35MTPQ4KRows performs the same rotary row permutation as the f32
// normalization path, but copies whole independently-quantized Q4_K rows. The
// operation is lossless: no block is decoded or requantized.
func reorderQwen35MTPQ4KRows(name string, raw []byte, shape []int, cfg model.Config) ([]byte, error) {
	if len(shape) != 2 || shape[1] <= 0 || shape[1]%qkK != 0 {
		return nil, fmt.Errorf("gguf: Qwen MTP tensor %s cannot reorder Q4_K shape %v", name, shape)
	}
	rowBytes := shape[1] / qkK * blockQ4KBytes
	if shape[0] > math.MaxInt/rowBytes || len(raw) != shape[0]*rowBytes {
		return nil, fmt.Errorf("gguf: Qwen MTP tensor %s has %d Q4_K bytes for shape %v", name, len(raw), shape)
	}
	dst := make([]byte, len(raw))
	copyRow := func(dstRow, srcRow int) {
		copy(dst[dstRow*rowBytes:(dstRow+1)*rowBytes], raw[srcRow*rowBytes:(srcRow+1)*rowBytes])
	}
	half := cfg.HeadDim / 2
	if cfg.HeadDim <= 0 || cfg.HeadDim%2 != 0 {
		return nil, fmt.Errorf("gguf: Qwen MTP tensor %s has invalid head_dim %d", name, cfg.HeadDim)
	}
	switch name {
	case "mtp.layers.0.self_attn.q_proj.weight":
		if shape[0] != cfg.NumHeads*2*cfg.HeadDim {
			return nil, fmt.Errorf("gguf: Qwen MTP tensor %s has shape %v incompatible with gated q heads", name, shape)
		}
		for h := 0; h < cfg.NumHeads; h++ {
			head := h * 2 * cfg.HeadDim
			for j := 0; j < half; j++ {
				for p := 0; p < 2; p++ {
					copyRow(head+p*half+j, head+j*2+p)
				}
			}
			for row := cfg.HeadDim; row < 2*cfg.HeadDim; row++ {
				copyRow(head+row, head+row)
			}
		}
	case "mtp.layers.0.self_attn.k_proj.weight":
		if shape[0] != cfg.NumKVHeads*cfg.HeadDim {
			return nil, fmt.Errorf("gguf: Qwen MTP tensor %s has shape %v incompatible with k heads", name, shape)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			for j := 0; j < half; j++ {
				for p := 0; p < 2; p++ {
					copyRow((h*2+p)*half+j, (h*half+j)*2+p)
				}
			}
		}
	default:
		return nil, fmt.Errorf("gguf: Qwen MTP tensor %s is not a reorderable q/k projection", name)
	}
	return dst, nil
}

func (s *WeightSource) computeQwen35MTPQ4KTensorWork(info TensorInfo, canon string, cfg model.Config, innerWorkers int, tickBytes int64) tensorWork {
	tw := tensorWork{tickBytes: tickBytes, mtpMaterialized: canon}
	if !model.RetainMTP {
		tw.err = fmt.Errorf("gguf: Qwen MTP tensor %s reached resident materialization without RetainMTP", info.Name)
		return tw
	}
	shape, raw, ok := s.shapeAndBytesOrFail(info, &tw)
	if !ok {
		return tw
	}
	if err := validateQwen35MTPShape(canon, shape, cfg); err != nil {
		tw.err = err
		return tw
	}
	wantType, matrix := qwen35MTPResidentTypes[canon]
	if matrix {
		if info.Type != wantType {
			tw.err = fmt.Errorf("gguf: Qwen MTP tensor %s has type %s, want %s", canon, info.Type, wantType)
			return tw
		}
		if strings.HasSuffix(canon, ".q_proj.weight") || strings.HasSuffix(canon, ".k_proj.weight") {
			var err error
			raw, err = reorderQwen35MTPQ4KRows(canon, raw, shape, cfg)
			if err != nil {
				tw.err = err
				return tw
			}
		}
		tw.pending = []pendingTensor{{
			resident:         true,
			residentType:     info.Type,
			canonicalMTPQ4K:  info.Type == TensorQ4_K && (strings.HasSuffix(canon, ".q_proj.weight") || strings.HasSuffix(canon, ".k_proj.weight")),
			canonicalMTPFCQ8: canon == "mtp.fc.weight",
			name:             canon,
			shape:            shape,
			raw:              raw,
		}}
		tw.acctType, tw.acctBytes, tw.acctTensors, tw.acctResident = info.Type.String(), tensorOnDiskBytes(info), 1, true
		return tw
	}
	if info.Type != TensorF32 {
		tw.err = fmt.Errorf("gguf: Qwen MTP tensor %s has type %s, want F32", canon, info.Type)
		return tw
	}
	data, err := dequantF32Limited(info, raw, innerWorkers)
	if err != nil {
		tw.err = err
		return tw
	}
	data, err = normalizeCanonicalTensorData(canon, data, cfg)
	if err != nil {
		tw.err = err
		return tw
	}
	tw.pending = []pendingTensor{{name: canon, shape: shape, f32: data}}
	tw.acctType, tw.acctBytes, tw.acctTensors = info.Type.String(), tensorOnDiskBytes(info), 1
	return tw
}

func (s *WeightSource) computeQ4KTensorWork(info TensorInfo, cfg model.Config, w3Requested bool, streamed map[string]bool, loadOpts q4kLoadOptions, innerWorkers int) tensorWork {
	tw := tensorWork{tickBytes: tensorOnDiskBytes(info)}
	if w3Requested && info.Type == TensorIQ3_XXS &&
		archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(info.Name) {
		tw.err = fmt.Errorf("gguf: FAK_W3_MLP refuses IQ3_XXS tensor %s outside dense MLP W3 band", info.Name)
		return tw
	}
	if canon, handled := qwen35MTPMaterializationName(info.Name, cfg); handled {
		if canon == "" {
			return tw
		}
		return s.computeQwen35MTPQ4KTensorWork(info, canon, cfg, innerWorkers, tw.tickBytes)
	}
	if archShipsMTPOrVisionSidecar(cfg.ModelType) && glmMoeDsaMTPOrVisionTensor(info.Name) {
		return tw
	}
	if archUsesMLAMoELayout(cfg.ModelType) {
		if layer, half, ok := glmMoeDsaSplitKVB(info.Name); ok {
			shape, data, err := s.dequantGGUFShapeF32Limited(info, innerWorkers)
			if err != nil {
				tw.err = err
				return tw
			}
			tw.pending = []pendingTensor{{isKVBHalf: true, layer: layer, half: half, shape: shape, f32: append([]float32(nil), data...)}}
			return tw
		}
	}
	if archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
		if layer, proj, ok := glmMoeDsaBatchedExpert(info.Name); ok {
			if streamed[info.Name] {
				return tw
			}
			shape, raw, okShape := s.shapeAndBytesOrFail(info, &tw)
			if !okShape {
				return tw
			}
			tw.acctType, tw.acctExpert, tw.acctBytes = info.Type.String(), true, tensorOnDiskBytes(info)
			if blockWeights, blockBytes, residentable := residentExpertBlockGeometry(info.Type); residentable {
				kqExperts, aligned, err := splitGLMMoeDsaExpertsRawQuant(layer, proj, shape, raw, blockWeights, blockBytes)
				if err != nil {
					tw.err = err
					return tw
				}
				if aligned && model.ResidentKQuantEligible(cfg, kqExperts[0].Name) {
					kept := loadOpts.keptExperts(len(kqExperts))
					tw.pending = make([]pendingTensor, 0, kept)
					for i, ex := range kqExperts {
						if !loadOpts.keepExpert(i) {
							continue
						}
						tw.pending = append(tw.pending, pendingTensor{resident: true, residentType: info.Type, name: ex.Name, shape: ex.Shape, raw: ex.Raw})
					}
					tw.acctResident, tw.acctTensors = true, kept
					if loadOpts.expertShardSet {
						if b, err := scaleExpertBandBytes(uint64(tw.acctBytes), kept, len(kqExperts)); err == nil {
							tw.acctBytes = int64(b)
						}
					}
					return tw
				}
			}
			data, err := dequantF32Limited(info, raw, innerWorkers)
			if err != nil {
				tw.err = err
				return tw
			}
			experts, err := splitGLMMoeDsaExperts(layer, proj, shape, data)
			if err != nil {
				tw.err = err
				return tw
			}
			kept := loadOpts.keptExperts(len(experts))
			tw.pending = make([]pendingTensor, 0, kept)
			for i, ex := range experts {
				if !loadOpts.keepExpert(i) {
					continue
				}
				tw.pending = append(tw.pending, pendingTensor{resident: false, name: ex.Name, shape: ex.Shape, f32: ex.Data})
			}
			tw.acctResident, tw.acctTensors = false, kept
			if loadOpts.expertShardSet {
				if b, err := scaleExpertBandBytes(uint64(tw.acctBytes), kept, len(experts)); err == nil {
					tw.acctBytes = int64(b)
				}
			}
			return tw
		}
	}
	canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
	if !ok {
		tw.err = fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
		return tw
	}
	if loadOpts.residentQ2KEmbedding && info.Name == "token_embd.weight" {
		shape, raw, ok := s.shapeAndBytesOrFail(info, &tw)
		if !ok {
			return tw
		}
		tw.acctType, tw.acctExpert, tw.acctBytes, tw.acctTensors, tw.acctResident = info.Type.String(), false, tensorOnDiskBytes(info), 1, true
		tw.pending = []pendingTensor{{q2kEmbed: true, name: canon, shape: shape, raw: raw}}
		return tw
	}
	if loadOpts.residentQ4KEmbedding && info.Name == "token_embd.weight" {
		shape, raw, ok := s.shapeAndBytesOrFail(info, &tw)
		if !ok {
			return tw
		}
		tw.acctType, tw.acctExpert, tw.acctBytes, tw.acctTensors, tw.acctResident = info.Type.String(), false, tensorOnDiskBytes(info), 1, true
		tw.pending = []pendingTensor{{q4kEmbed: true, name: canon, shape: shape, raw: raw}}
		return tw
	}
	if loadOpts.streamedDenseQ4K && info.Type == TensorQ4_K && model.ResidentQ4KEligible(cfg, canon) {
		return s.lazyDenseQ4KTensorWork(info, canon, tw.tickBytes)
	}
	shape, raw, ok := s.shapeAndBytesOrFail(info, &tw)
	if !ok {
		return tw
	}
	tw.acctType, tw.acctExpert, tw.acctBytes, tw.acctTensors = info.Type.String(), false, tensorOnDiskBytes(info), 1
	w3Eligible := info.Type == TensorIQ3_XXS && model.ResidentW3MLPEligible(cfg, canon)
	switch {
	case w3Eligible && w3Requested:
		tw.pending = []pendingTensor{{resident: true, residentType: info.Type, name: canon, shape: shape, raw: raw}}
		tw.acctResident = true
		return tw
	case info.Type == TensorIQ3_XXS && w3Requested:
		tw.err = fmt.Errorf("gguf: FAK_W3_MLP refuses IQ3_XXS tensor %s outside dense MLP W3 band", canon)
		return tw
	}
	if info.Type == TensorQ4_K && model.ResidentQ4KEligible(cfg, canon) {
		tw.pending = []pendingTensor{{resident: true, residentType: info.Type, name: canon, shape: shape, raw: raw}}
		tw.acctResident = true
		return tw
	}
	retainDenseKQuant := loadOpts.residentDenseKQuant || (loadOpts.residentDenseQ2K && info.Type == TensorQ2_K)
	if _, _, residentable := residentExpertBlockGeometry(info.Type); retainDenseKQuant && residentable &&
		info.Type != TensorQ4_K && !archUsesMLAMoELayout(cfg.ModelType) &&
		model.ResidentKQuantEligible(cfg, canon) {
		tw.pending = []pendingTensor{{resident: true, residentType: info.Type, name: canon, shape: shape, raw: raw}}
		tw.acctResident = true
		return tw
	}
	data, err := dequantF32Limited(info, raw, innerWorkers)
	if err != nil {
		tw.err = err
		return tw
	}
	data, err = normalizeCanonicalTensorData(canon, data, cfg)
	if err != nil {
		tw.err = err
		return tw
	}
	tw.pending = []pendingTensor{{resident: false, name: canon, shape: shape, f32: data}}
	return tw
}
