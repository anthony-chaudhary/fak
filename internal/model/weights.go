// Package model is the in-kernel inference core: a pure-Go forward pass over a
// single small open-source model (SmolLM2-135M / Qwen2.5-0.5B), with the KV cache
// as a first-class Go data structure the kernel OWNS. This is the deepest fusion
// the goal asks for — the model runs INSIDE the kernel address space, so the
// context-MMU, vDSO, and blob store stop being metaphors-over-HTTP and become real
// operations on real attention state.
//
// Correctness is not asserted; it is PROVEN. internal/model/export_oracle.py dumps,
// from HuggingFace transformers (the witness we did not author), the per-layer
// hidden states, logits, and greedy continuation for fixed token-id prompts; the
// oracle test reproduces every one of them to f32 tolerance. A bug in any rung is
// localized because the comparison is layer-by-layer, not just end-to-end.
//
// This file is the weights loader: it maps the flat f32 blob + manifest produced by
// export_oracle.py into zero-copy []float32 tensor views.
package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

// windowLo is the read-time SWA mask, expressed as a lower-bound START INDEX into the
// in-order key rows. pos[] holds each cached key's ABSOLUTE position (pos[j] == j when
// no eviction has happened; after an Evict it is the compacted contiguous run); qpos is
// the query's absolute position; W is the window (windowForLayer). It returns the first
// key index lo such that every key in [lo, nPos) is inside the window
// (pos[j] >= qpos-W+1), so callers iterate j from lo instead of 0.
//
// W < 0 (the default) returns 0 — the full-causal path, with NO change to which keys are
// visited — so the non-SWA reduction is byte-for-byte the pre-SWA loop. Because pos[] is
// monotonically non-decreasing, the visible keys are always a contiguous suffix; a window
// can only ever DROP the oldest keys, never reorder the survivors, so the in-order softmax
// + V accumulation over [lo, nPos) is the same arithmetic restricted to a sub-range.
func windowLo(pos []int, nPos, qpos, W int) int {
	if W < 0 {
		return 0
	}
	lower := qpos - W + 1
	if lower <= 0 {
		return 0
	}
	lo := 0
	for lo < nPos && pos[lo] < lower {
		lo++
	}
	return lo
}

// windowLoContig is windowLo for a CONTIGUOUS cache (pos[j] == j for every row), which
// is the invariant on every prefill path: a prior Evict renumbers pos[i]=i and prefill
// always appends at Cache.Len(), so the row index equals the absolute position. The
// window lower bound max(0, qpos-W+1) is then directly the start index, with no pos[]
// scan. W < 0 returns 0 (full causal, no change). nPos clamps the bound into range.
func windowLoContig(nPos, qpos, W int) int {
	if W < 0 {
		return 0
	}
	lo := qpos - W + 1
	if lo <= 0 {
		return 0
	}
	if lo > nPos {
		lo = nPos
	}
	return lo
}

// windowLoStep is windowLo for the single-token cached decode paths, where the query's
// own K row has ALREADY been appended to the cache (so there are nPos key rows) but its
// absolute position has NOT yet been appended to priorPos (priorPos covers only the
// nPos-1 earlier keys; the query, at key index nPos-1, sits at absolute position qpos).
// W < 0 returns 0 (full causal, no change). Since the query is always inside its own
// window, only the priorPos prefix can be dropped.
func windowLoStep(priorPos []int, nPos, qpos, W int) int {
	if W < 0 {
		return 0
	}
	lower := qpos - W + 1
	if lower <= 0 {
		return 0
	}
	lo := 0
	for lo < len(priorPos) && lo < nPos && priorPos[lo] < lower {
		lo++
	}
	return lo
}

func layerPrefix(layer int) string {
	return "model.layers." + itoa(layer) + "."
}

func layerName(layer int, suffix string) string {
	return layerPrefix(layer) + suffix
}

func (m *Model) attentionNorms(layer int) normWeights {
	preName := layerName(layer, "input_layernorm.weight")
	var pre, preBias []float32
	if m.has(preName) {
		pre = m.tensor(preName)
		preBias = m.tensorOptional(layerName(layer, "input_layernorm.bias"))
	}
	post := pre
	postBias := preBias
	if name := layerName(layer, "post_attention_layernorm.weight"); m.has(name) {
		post = m.tensor(name)
		postBias = m.tensorOptional(layerName(layer, "post_attention_layernorm.bias"))
	}
	if pre == nil {
		pre = post
		preBias = postBias
	}
	if pre == nil {
		pre = m.tensor(preName)
	}
	return normWeights{pre: pre, preBias: preBias, post: post, postBias: postBias}
}

func (m *Model) mlpNorms(layer int) normWeights {
	preName := layerName(layer, "post_attention_layernorm.weight")
	if name := layerName(layer, "pre_feedforward_layernorm.weight"); m.has(name) {
		preName = name
	}
	pre := m.tensor(preName)
	preBias := m.tensorOptional(strings.TrimSuffix(preName, ".weight") + ".bias")
	post := pre
	postBias := preBias
	if name := layerName(layer, "post_feedforward_layernorm.weight"); m.has(name) {
		post = m.tensor(name)
		postBias = m.tensorOptional(layerName(layer, "post_feedforward_layernorm.bias"))
	}
	return normWeights{pre: pre, preBias: preBias, post: post, postBias: postBias}
}

func (m *Model) parallelMLPNorms(layer int, shared normWeights) normWeights {
	if m.has(layerName(layer, "post_attention_layernorm.weight")) {
		return m.mlpNorms(layer)
	}
	return shared
}

type tensorMeta struct {
	Dtype  string `json:"dtype"`
	Shape  []int  `json:"shape"`
	Offset int    `json:"offset"`
	Nbytes int    `json:"nbytes"`
}

// NamedTensorF32 is a loader-neutral f32 tensor payload. Source-format leaves such as GGUF
// use this to build the same packed raw+manifest representation as the native fak export.
type NamedTensorF32 struct {
	Name  string
	Shape []int
	Data  []float32
}

// Model is a loaded checkpoint: the config, the tensor manifest, and the raw
// little-endian f32 bytes of every weight, kept in one buffer so a tensor view is
// a zero-copy reinterpretation, not a copy.
type Model struct {
	Cfg      Config
	manifest map[string]tensorMeta
	raw      []byte // all tensors, f32 LE, at the manifest offsets

	// q8w holds the optional Q8_0-quantized copy of the matmul weights, built once by
	// Quantize() or on-demand by q8(), and consumed only by the opt-in quantized forward
	// path (quant.go / quant_forward.go). nil unless quantization was requested; the f32
	// path never reads it.
	q8w       map[string]*q8Tensor
	q8layers  []q8Layer
	q8head    *q8Tensor
	quantized bool

	// q4w holds the optional resident int4 (Q4_0-style) copy of the matmul weights, built
	// once by QuantizeQ4() and consumed only by the opt-in int4 forward path (quant_q4.go).
	// It is the decode-bandwidth lever for the in-kernel Qwen3.6 engine: int4 streams
	// ~1.8× fewer bytes/token than Q8_0 (see QWEN36-NATIVE-PERF-PLAN-2026-06-19.md). nil
	// unless QuantizeQ4 ran; the f32/Q8 paths never read it.
	q4w    map[string]*q4Tensor
	q4head *q4Tensor // pinned at QuantizeQ4 time; headName() can shift once q8w is freed

	// q4kw holds the optional resident Q4_K (k-quant super-block) copy of the matmul
	// weights, built once by QuantizeQ4K() straight from the GGUF payload (no f32 round
	// trip) and consumed only by the opt-in resident-Q4_K forward path (quant_q4k.go). It
	// is the load+decode+memory lever for QWEN36-NATIVE-PERF-PLAN P1: raw Q4_K streams
	// 0.5625 B/weight (fewer than Q8_0's 1.125 or q4w's 0.625), needs no q8w co-residency,
	// and matches the llama.cpp q4_k_m artifact. nil unless QuantizeQ4K ran; the f32/Q8/Q4_0
	// paths never read it.
	q4kw    map[string]*q4kTensor
	q4khead *q4kTensor // pinned when lm_head is held raw in q4kw; headName() can't see q4kw
	// q4kResidency owns the once-only streamed upload evidence for this model. Keeping the state
	// on the model gives it the same lifetime without a global pointer-key retention map.
	q4kResidency *q4kResidencyState

	// numaInterleaveLabel caches the verdict of the last ApplyDecodeNUMAInterleave call
	// (#4974) so a later NUMAInterleaveLabel() — e.g. a decode-witness RESULT line — can
	// report the same placement decision without re-walking the resident weight regions.
	// "" ⇒ ApplyDecodeNUMAInterleave has not run on this model yet.
	numaInterleaveLabel string

	// kqw holds the optional resident Q5_K/Q6_K (k-quant super-block) copy of MoE EXPERT
	// matmul weights, built straight from the GGUF payload (no f32 round trip) for GLM-5.2's
	// mixed-quant UD-Q4_K_M experts and consumed on the host expert seam (residentMatRows ->
	// kQuantMatRows; quant_kquant.go). It is the load-time twin of q4kw for the non-Q4_K
	// experts. nil unless such experts loaded; the f32/Q8/Q4_K paths never read it.
	kqw map[string]*kQuantTensor

	// expertCheckpoint is the R5/#5616 tier BELOW the bounded device ring: the per-expert range
	// reader that serves a routed-expert weight ABSENT from q4kw/kqw by faulting exactly that
	// expert's stride out of the fused checkpoint slab. nil unless a loader attached one
	// (SetExpertCheckpoint), and then every routed expert resolves from the resident stores exactly
	// as before — this is an added tier, not a replacement.
	expertCheckpoint *ExpertCheckpointTier
	weightCloser     *weightCloserState

	// q2w holds the optional resident ternary Q2_0 copy of matmul weights, fed the raw
	// GGUF group-128 blocks straight from the loader and consumed by q2MatRows.
	q2w map[string]*q2Tensor
	// awqw holds the optional resident AWQ (Activation-aware Weight Quantization) 4-bit
	// copy of the matmul weights, populated by LoadAWQ straight from an AutoAWQ
	// safetensors export and consumed only by the opt-in AWQ path (awq.go). nil unless
	// an AWQ checkpoint was loaded; the f32/Q8/Q4 paths never read it.
	awqw map[string]*awqTensor

	// awqg holds the REAL AutoAWQ group-wise asymmetric 4-bit copy of the matmul
	// weights (per-group scales + 4-bit zeros), populated by LoadAWQ when the export
	// is a genuine qweight/qzeros/scales triple — the format real Llama-2/3 & Qwen2
	// AWQ checkpoints ship. Consumed only by the opt-in AWQ path (awq_group.go); nil
	// unless such a checkpoint was loaded. See awqw for the simplified symmetric stub.
	awqg map[string]*awqGroupTensor

	// gptqw holds optional resident GPTQ weight-only tensors loaded from AutoGPTQ /
	// GPTQModel qweight/qzeros/scales triples. It is consumed only by Session.GPTQ via
	// residentMatRows; the f32/Q8/Q4/Q4_K paths never read it.
	gptqw map[string]*gptqTensor

	// MLA holds the DeepSeek V2/V3 Multi-head Latent Attention projection geometry
	// when this model uses the MLA kvLayout (issue #25). It is nil for Llama/Qwen
	// models, which keep the default standard per-head kvLayout unchanged — so adding
	// this field does not touch the proven Llama path. modelLayout() consults it to
	// pick standardKVLayout (MLA==nil) vs mlaKVLayout (MLA!=nil).
	MLA *MLAConfig

	// Vision holds the retained CLIP/ViT image-tower weights when a VLM's vision
	// source is loaded (an mmproj GGUF or an inline model.visual.* safetensors set),
	// mirroring MLA as a dedicated sub-struct for a stack the text forward never
	// reads. It is nil for every text-only model — the unchanged default — so the
	// proven decoder path is byte-for-byte untouched. The vision encoder (#4030)
	// consumes it; only a load with RetainVision set (the --mmproj flag, #4032) ever
	// populates it. See vision.go.
	Vision *VisionTower
	// sourceDir anchors V4 lazy expert/index range reads to the admitted snapshot.
	sourceDir string

	// attnObs is the optional attention-mass witness (#852). nil by default — the
	// unobserved forward pass is byte-identical and allocation-identical. When set via
	// SetAttnObserver, the named attention seams emit a COPY of their post-softmax
	// weights at the softmax seam (emission only; the math is untouched). See
	// attn_observer.go.
	attnObs AttnObserver

	// expertRouteObs is the optional routed-expert touch witness (#4233). nil is the
	// production default, so ordinary routing performs no extra allocation or accounting.
	// The observer is diagnostic only and never changes a pick or eviction policy.
	expertRouteObs ExpertRouteObserver

	// routeObs is the optional per-token MoE expert-routing witness (#2623), the routing
	// analogue of attnObs (#852): when set via SetRouteObserver, route()/glmRoute() emit a
	// COPY of each token's top-k (expert, gate-weight) picks. nil by default — the
	// unobserved forward pass is byte-identical and allocation-identical (emission only,
	// the routing math is untouched). See route_observer.go. Unlike expertRouteObs (which
	// carries expert ids only, for residency replay), routeObs carries the gate weights the
	// per-span expert_hist descriptor needs. routePos is the position, within the sequence
	// chunk the current forward pass processes, of the token whose picks are being emitted;
	// mlpSeq stamps it before each per-token FFN apply and route() reads it. It is only
	// meaningful while routeObs is set (mlpSeq only writes it then), so the off path never
	// touches it.
	routeObs RouteObserver
	routePos int

	// lora is the optional set of active LoRA adapters applied dynamically at the
	// named-projection seam (#291). nil by default — residentMatRows is then
	// byte-identical and allocation-identical. When set via SetLoRA, each named
	// projection adds its active adapters' low-rank delta after the base matvec
	// (decode-time apply, no merged weight copy). See lora.go.
	lora *LoRASet

	// epRanks is the expert-parallel rank count for the routed MoE FFN: the number of
	// expert shards the per-token MoE delta is reduced across (expert_parallel.go). 0/1
	// keep the live forward on the monolith glmMoeFFN (the no-op default — nothing
	// changes for an existing serve); >1 routes routed-expert picks through glmMoeEPFFN,
	// which reduces the per-rank residual partials with one AllReduceSum. The reduction
	// runs through the Collective the serve wires: LocalCollective (single-box, bit-exact)
	// until the device NCCL CollectiveBackend lands, at which point the same plan reduces
	// experts resident across real GPUs. Set via SetExpertParallelRanks from the serve
	// flag; the EP arithmetic is bit-exact vs the monolith at ranks=1 (expert_parallel_test.go).
	epRanks int

	// epColl is the Collective the live decode EP path reduces the per-rank expert partials
	// through (glmMoeEPFFN -> expertParallelGLMMoEDelta). nil keeps the single-box, bit-exact
	// LocalCollective default (an existing serve, or a box with no device collective). The
	// serve sets it to a BackendCollective wrapping the device CollectiveBackend (NCCL) when
	// --expert-parallel N>1 runs on a multi-GPU box, so the decode reduction the serve REQUIRED
	// a device collective for (serve.go's Caps().Collective gate) actually flows across the
	// GPUs instead of being computed host-side. On cpu-ref a BackendCollective is byte-identical
	// to LocalCollective (collective_bridge_test.go), so wiring it changes no host-tested bytes;
	// on NCCL the same call issues a real cross-GPU all-reduce. Set via SetExpertParallelCollective.
	epColl Collective

	// epRank + epRankSet identify THIS process's expert-parallel rank in a SHARDED (multi-
	// process) EP serve, where each rank loaded ONLY its expert band (ggufload.WithExpertShard)
	// and so holds only plan.Shards[epRank]'s experts. When epRankSet, the live MoE forward
	// computes only this rank's band partial (expertParallelRankLocalGLMMoEDelta) and reduces it
	// across the process group through epColl (a distCommCollective) — the residency win #971
	// needs, since no single process holds the full expert set. When epRankSet is FALSE (the
	// default — the setter is never called), the forward keeps the single-process all-band path
	// (expertParallelPartials on a full model), so every existing serve and every bit-exact EP
	// test is byte-for-byte unchanged. epRank is an explicit-flag sentinel, NOT epRank==0: rank 0
	// is a valid sharded rank, so presence must be tracked by epRankSet, not by a zero value.
	// Set via SetExpertParallelRank from the serve's FAK_EP_RANK.
	epRank    int
	epRankSet bool

	// epCoord is rank 0's coordinated-decode driver in a SHARDED EP serve (#4835). When set,
	// every Prefill/Step on this model's sessions first broadcasts the forward it is about to
	// run to the follower ranks, so ONE request produces one tokenize/sample on rank 0 and N
	// local expert contributions — replacing the HTTP mirror that ran the whole request N
	// times. nil (the default, and every non-sharded serve) leaves the forward entry points
	// byte-identical: the hook reads this one nil field and returns. Set via
	// SetEPDecodeCoordinator; follower ranks run RunEPFollower instead and never set it.
	epCoord *EPDecodeCoordinator
}

// newModel assembles a Model from a built manifest + packed f32 blob, applying
// source-format tensor aliases and then the load-time fused-tensor split
// (Phi qkv_proj / gate_up_proj -> separate q/k/v / gate/up component views)
// before the model is handed to the forward pass. It is the single construction
// point every loader funnels through, so the split rule is applied uniformly and
// exactly once. On a Llama-shaped checkpoint with no aliases and no fused tensor,
// both steps are no-ops, so this path stays bit-identical for non-Phi models.
func newModel(cfg Config, man map[string]tensorMeta, raw []byte) (*Model, error) {
	if cfg.IsDeepSeekV4() {
		if err := AdmitDeepSeekV4Config(cfg); err != nil {
			return nil, err
		}
	}
	if err := materializeQwen35Tensors(cfg, man); err != nil {
		return nil, err
	}
	if err := validateQwen35MTPLoadArtifact(cfg, man, raw); err != nil {
		return nil, err
	}
	// Segregate the retained inline vision tower (model.visual.*) into Model.Vision
	// BEFORE any other pass sees it, so the decoder manifest below holds only text
	// weights. Only reached when RetainVision is set (the --mmproj flag, #4032);
	// materializeQwen35Tensors drops model.visual.* otherwise, so this is nil for a
	// default text load and the proven decoder path is unchanged.
	var vision *VisionTower
	if RetainVision {
		vt, err := extractQwen35VisionTower(man, raw)
		if err != nil {
			return nil, err
		}
		vision = vt
	}
	if err := materializeTensorAliases(cfg, man); err != nil {
		return nil, err
	}
	if err := materializeGPTNeoXTensors(cfg, man, &raw); err != nil {
		return nil, err
	}
	if err := materializeFalconTensors(cfg, man, &raw); err != nil {
		return nil, err
	}
	if err := materializeMPTTensors(cfg, man); err != nil {
		return nil, err
	}
	if err := materializeMixtralBlockSparseTensors(cfg, man); err != nil {
		return nil, err
	}
	// #934: refuse an unrecognized Gated-DeltaNet/SSM hybrid (fused attn_qkv +
	// per-layer linear_attn/ssm_* core, no self_attn.q_proj) with a typed,
	// named load-time error instead of letting the standard forward panic on a
	// missing self_attn.q_proj.weight mid-request. Must run before
	// splitFusedProjections so the operator gets the arch refusal rather than a
	// misleading fused-tensor row-count mismatch from the GDN in_proj.
	if err := refuseUnsupportedHybridArch(cfg, man); err != nil {
		return nil, err
	}
	// Refuse the two Falcon variants whose fused-qkv layout the contiguous split
	// below does not implement (Falcon-40B/180B new_decoder_architecture, and
	// Falcon-RW). Both carry the SAME total row count as the contiguous cut, so
	// splitFusedProjections cannot catch them and would load them silently wrong;
	// this must therefore run before it. See falcon_variant.go.
	if err := refuseUnsupportedFalconVariant(cfg, man); err != nil {
		return nil, err
	}
	if err := splitFusedProjections(cfg, man); err != nil {
		return nil, err
	}
	if err := materializeGPTOSSTensors(cfg, man, &raw); err != nil {
		return nil, err
	}
	if err := splitBatchedMoEExperts(cfg, man); err != nil {
		return nil, err
	}
	if err := materializeMiniMaxSharedExperts(cfg, man); err != nil {
		return nil, err
	}
	return &Model{Cfg: cfg, manifest: man, raw: raw, Vision: vision}, nil
}

// Load reads a directory produced by export_oracle.py (config.json, manifest.json,
// weights.f32).
func Load(dir string) (*Model, error) {
	var cfg Config
	if err := readJSON(filepath.Join(dir, "config.json"), &cfg); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	var man map[string]tensorMeta
	if err := readJSON(filepath.Join(dir, "manifest.json"), &man); err != nil {
		return nil, fmt.Errorf("manifest: %w", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "weights.f32"))
	if err != nil {
		return nil, fmt.Errorf("weights: %w", err)
	}
	// export_oracle.py writes HF tensors verbatim, so this is an HF-layout checkpoint and
	// takes the same rotary re-layout as the safetensors loaders (cohere_rotary.go).
	return newHFCheckpointModel(cfg, man, raw)
}

// NewFromF32Tensors packs decoded source-format tensors into the same little-endian f32
// raw+manifest layout that Load and LoadSafetensors produce.
func NewFromF32Tensors(cfg Config, tensors []NamedTensorF32) (*Model, error) {
	man := make(map[string]tensorMeta, len(tensors))
	var raw []byte
	off := 0
	for _, t := range tensors {
		if t.Name == "" {
			return nil, fmt.Errorf("model: empty tensor name")
		}
		if _, ok := man[t.Name]; ok {
			return nil, fmt.Errorf("model: duplicate tensor %s", t.Name)
		}
		elems, err := tensorShapeElems(t.Name, t.Shape)
		if err != nil {
			return nil, err
		}
		if elems != len(t.Data) {
			return nil, fmt.Errorf("model: tensor %s has %d values, shape wants %d", t.Name, len(t.Data), elems)
		}
		nbytes := len(t.Data) * 4
		if nbytes/4 != len(t.Data) || off > math.MaxInt-nbytes {
			return nil, fmt.Errorf("model: tensor %s byte size overflows int", t.Name)
		}
		start := len(raw)
		raw = append(raw, make([]byte, nbytes)...)
		for i, v := range t.Data {
			binary.LittleEndian.PutUint32(raw[start+i*4:], math.Float32bits(v))
		}
		shape := append([]int(nil), t.Shape...)
		man[t.Name] = tensorMeta{Dtype: "f32", Shape: shape, Offset: off, Nbytes: nbytes}
		off += nbytes
	}
	return newModel(cfg, man, raw)
}

func tensorShapeElems(name string, shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("model: tensor %s has no dimensions", name)
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			return 0, fmt.Errorf("model: tensor %s has invalid dimension %d", name, d)
		}
		if n > math.MaxInt/d {
			return 0, fmt.Errorf("model: tensor %s element count overflows int", name)
		}
		n *= d
	}
	return n, nil
}

// manifestTensor returns a zero-copy []float32 view of the named tensor inside raw, the
// blob its manifest indexes. The blob is little-endian f32 and amd64 is little-endian, so
// the bytes reinterpret directly. A missing name is a load-time contract break, not a
// runtime condition, so it panics; owner prefixes the message ("" for the decoder's own
// weights, "vision tower " for the encoder's) to keep the two crashes distinguishable.
func manifestTensor(manifest map[string]tensorMeta, raw []byte, owner, name string) []float32 {
	meta, ok := manifest[name]
	if !ok {
		panic("model: " + owner + "missing tensor " + name)
	}
	n := meta.Nbytes / 4
	return unsafe.Slice((*float32)(unsafe.Pointer(&raw[meta.Offset])), n)
}

// tensor returns a zero-copy []float32 view of a named weight.
func (m *Model) tensor(name string) []float32 {
	return manifestTensor(m.manifest, m.raw, "", name)
}

// has reports whether a tensor is present (e.g. q/k/v bias only exist on Qwen2).
func (m *Model) has(name string) bool {
	_, ok := m.manifest[name]
	return ok
}

// hasWeight reports whether a matmul weight is resident in ANY of the stores a
// quantized serve uses: the f32 manifest, the Q8_0 store (q8w), or the raw-resident
// Q4_K store (q4kw). m.has alone only sees the f32 manifest, so on a lean-Q8 or
// resident-Q4_K model (the cuda serve path) a router/dense-MLP weight that was
// quantized at load is invisible to it. ffnForLayer's dense-vs-MoE dispatch keys on
// the PRESENCE of a layer's router (mlp.gate.weight) vs its dense MLP
// (mlp.gate_proj.weight); keying that on m.has would mis-route every layer in a
// quantized model (the weights live in q8w/q4kw), sending a dense first-k layer
// down the MoE path whose router mul then panics in glmDsaWeightHAL. hasWeight is
// the residency-complete presence check that dispatch must use.
func (m *Model) hasWeight(name string) bool {
	if m.has(name) {
		return true
	}
	if m.q8w != nil {
		if _, ok := m.q8w[name]; ok {
			return true
		}
	}
	if m.q4kw != nil {
		if _, ok := m.q4kw[name]; ok {
			return true
		}
	}
	if m.kqw != nil {
		if _, ok := m.kqw[name]; ok {
			return true
		}
	}
	if m.gptqw != nil {
		if _, ok := m.gptqw[name]; ok {
			return true
		}
	}
	return false
}

func (m *Model) tensorOptional(name string) []float32 {
	if m.has(name) {
		return m.tensor(name)
	}
	return nil
}

func (m *Model) finalNorm(x []float32) []float32 {
	return normCfg(x, m.tensor("model.norm.weight"), m.tensorOptional("model.norm.bias"), float32(m.Cfg.RMSNormEps), m.Cfg)
}

// embedRows returns the [vocab, hidden] embedding matrix, which is also the tied
// LM-head matrix when TieWordEmbeddings.
func (m *Model) embedRows() []float32 { return m.tensor("model.embed_tokens.weight") }

// lmHead returns the [vocab, hidden] output projection. Tied -> the embedding.
func (m *Model) lmHead() []float32 {
	if m.has("lm_head.weight") {
		return m.tensor("lm_head.weight")
	}
	return m.embedRows()
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
