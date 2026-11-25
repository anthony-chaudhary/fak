package ggufload

// Q4KLoadOption configures the direct-resident-Q4_K GGUF load path.
type Q4KLoadOption func(*q4kLoadOptions)

// WithMTPRetention controls Qwen MTP head retention for this Q4K load. Omitting
// it captures the legacy model.RetainMTP default when the load resolves options.
// It does not change the process default or qualify retained-head memory admission.
func WithMTPRetention(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.setMTPRetention(enabled) }
}

func (o *q4kLoadOptions) setMTPRetention(enabled bool) {
	o.retainMTP = enabled
}

// WithNativeLinearAttnRows controls the Qwen3.5-family native-row residency (fak#13567): the
// GDN in_proj_qkv/z/a/b and self_attn q/k projections, whose GGUF-to-model transform is a pure
// output-row permutation, keep their Q4_K (q4kw) / Q6_K / Q5_K (kqw) bytes after a lossless raw row
// reorder instead of being requantized to Q8. The default (enabled) is automatic: it applies only
// where qwen35NativeRowResident admits a tensor, never to the exact 64-layer Qwen3.8 runtime whose
// Metal decode requires those weights as Q8. Pass false to force the historical Q8 route.
func WithNativeLinearAttnRows(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.nativeRowsOff = !enabled }
}

// WithQwen35NativeSSMOut controls the Qwen3.5-family GDN out_proj (GGUF ssm_out) native route:
// on a raw-i-quant device backend the weight keeps its GGUF bytes and tiled input-column order,
// and the forward tiles the GDN core before the projection, instead of deinterleaving the
// columns into a Q8_0 copy. The default (enabled) applies only where qwen35TiledOutProjResident
// admits a tensor; pass false (or set FAK_QWEN35_NATIVE_SSM_OUT=0) to force the Q8 route.
func WithQwen35NativeSSMOut(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.nativeSSMOutOff = !enabled }
}

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

// WithDenseQ6KResident retains eligible dense Q6_K tensors even when blanket dense
// k-quant residency is disabled. This lets a backend with a Q6_K HAL path avoid the
// dequant-to-Q8 round trip without stranding Q5_K/Q3_K/Q2_K/IQ formats on a dequant path.
func WithDenseQ6KResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.residentDenseQ6K = enabled }
}

// WithDenseQ5KResident retains eligible dense Q5_K tensors even when blanket dense
// k-quant residency is disabled, for a backend with a native Q5_K device matmul. Without it a
// dense Q5_K weight is re-quantized to native Q8 (1.125 B/weight vs Q5_K's 0.6875).
func WithDenseQ5KResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.residentDenseQ5K = enabled }
}

// WithDenseQ3KResident retains eligible dense Q3_K tensors even when blanket dense k-quant
// residency is disabled, for a backend with a qualified packed Q3_K matmul. Without it a mixed
// UD artifact's Q3_K FFN layers expand to native Q8 (1.125 B/weight vs 0.43).
func WithDenseQ3KResident(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.residentDenseQ3K = enabled }
}

// WithDenseIQResident retains eligible dense tensors of the given i-quant types even when
// blanket dense k-quant residency is disabled, for a backend with native kernels for exactly
// those formats (Vulkan iq*_matvec, ticket 05f). Other i-quants keep the dequant-to-Q8 path.
// Passing it also admits the qwen35 raw-row routes (see qwen35DeviceQKNativeRow), so only a
// backend that tiles the GDN core may pass it; a backend that merely advertises i-quant matmul
// dtypes must use WithDenseIQPacked instead.
func WithDenseIQResident(types ...TensorType) Q4KLoadOption {
	return func(o *q4kLoadOptions) {
		if o.residentDenseIQ == nil {
			o.residentDenseIQ = map[TensorType]bool{}
		}
		for _, t := range types {
			o.residentDenseIQ[t] = true
		}
	}
}

// WithDenseIQPacked retains eligible dense tensors of the given i-quant types packed, exactly
// as WithDenseIQResident does for identity-ordered weights, but does not admit the qwen35
// raw-i-quant row routes (native q/k and GDN in-proj rows, tiled GDN out_proj) that
// WithDenseIQResident enables: those need a backend that tiles the GDN core
// (compute.Qwen35GDNTiledOutBackend). Tensors those routes would take keep the Q8 path.
func WithDenseIQPacked(types ...TensorType) Q4KLoadOption {
	return func(o *q4kLoadOptions) {
		if o.residentDenseIQPack == nil {
			o.residentDenseIQPack = map[TensorType]bool{}
		}
		for _, t := range types {
			o.residentDenseIQPack[t] = true
		}
	}
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

// WithStreamedExpertDeviceRing declares the streamed tier's DEVICE ring ceiling (bytes > 0), so a
// faulted routed expert pages into a bounded device ring instead of an unbounded memo. It only takes
// effect with WithStreamedExperts; the tier refuses a ceiling below one expert group by name.
func WithStreamedExpertDeviceRing(bytes int64) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.streamedExpertDeviceRing = bytes }
}

// WithExpertPageCache controls whether the R5 checkpoint tier adopts a shard's mapped
// region for fused routed-expert faults (#1302): enabled, warm expert faults are served
// zero-copy from the mmap; disabled (the default), every fault reads through the shard's
// io.ReaderAt exactly as before. Mapping a shard stays the load-wide FAK_GGUF_MMAP storage
// decision; this option is the narrower expert-path behavior change, so it needs its own
// explicit opt-in. It replaces the retired FAK_EXPERT_PAGECACHE ambient read (#13605 leaf
// A-cfg-int4): the decision is resolved once per load request and passed explicitly to
// FusedExpertTensors and the checkpoint-tier construction instead of being cached in a
// process-global sync.Once.
func WithExpertPageCache(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.expertPageCache = enabled }
}

// WithStreamedDenseQ4K leaves eligible identity-layout dense Q4_K tensors on disk. The returned model retains checkpoint range descriptors and requires the WeightSource to stay open.
func WithStreamedDenseQ4K(enabled bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.streamedDenseQ4K = enabled }
}

// WithStreamedDenseQ4KWorkingSet is the BOUNDED form of WithStreamedDenseQ4K: it turns the
// read-defer into a bounded host working set. Eligible dense k-quant tensors stay on disk with
// range descriptors (as WithStreamedDenseQ4K does), but the load plan charges only hostBytes of
// resident dense working set instead of the FULL dense side, exactly as WithStreamedExperts does
// for the routed band. hostBytes <= 0 is stream-through: no dense host residency is charged and
// only the device dense side remains. A negative budget is refused by name rather than clamped,
// so a caller cannot ask for an unrepresentable policy.
//
// This is the dense sibling of the fak#13121 bounded routed-expert policy: it lets a device
// dense remainder larger than host RAM (the pinned V4.1 Q2_K side is 63.099 GiB against a
// 62.425 GiB Halo MemTotal) produce a plan that fits, so the fit guard judges the working set
// the loader actually retains rather than the full on-disk side. Like WithStreamedDenseQ4K it
// needs a checkpoint that outlives the model; it is available only on the WeightSource-form
// entry points.
func WithStreamedDenseQ4KWorkingSet(hostBytes int64) Q4KLoadOption {
	return func(o *q4kLoadOptions) {
		o.streamedDenseQ4K = true
		o.streamedDenseBounded = true
		o.streamedDenseBytes = hostBytes
	}
}

// WithStreamedDenseDeviceResident declares which eligible streamed-dense tensors the executing
// backend's kernel preflight requires to stay DEVICE-RESIDENT, by canonical name. The bounded
// streamed-dense policy otherwise folds every eligible dense k-quant into one bounded HOST working
// set - but a backend whose sequence-prefill kernel validates its operands for a live default-device
// allocation (the qwen35-hybrid CUDA/Vulkan paths) fails closed at warmup on any operand the HAL
// spilled to managed memory under VRAM pressure. Pinned tensors are excluded from EVERY mirror of
// the lazy route (the estimate fold, the loader dispatch, the host-load-peak mirrors), so they keep
// their device charge in both the plan and the load and the fit guard judges what the kernel
// actually needs. The predicate receives the canonical name AFTER the qwen35 source chain, exactly
// as the eligibility gates do; a nil or absent option pins nothing. It is orthogonal to the
// working-set bound: it shrinks the streamed set, it never changes the bound.
func WithStreamedDenseDeviceResident(pinned func(canon string) bool) Q4KLoadOption {
	return func(o *q4kLoadOptions) { o.streamedDensePinned = pinned }
}

// streamedDensePins is the nil-safe read of the device-residency pin for one canonical tensor.
func (o q4kLoadOptions) streamedDensePins(canon string) bool {
	return o.streamedDensePinned != nil && o.streamedDensePinned(canon)
}
