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
	// Quantize() and consumed only by the opt-in quantized forward path (quant.go /
	// quant_forward.go). nil unless quantization was requested; the f32 path never reads it.
	q8w      map[string]*q8Tensor
	q8layers []q8Layer
	q8head   *q8Tensor

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

	// attnObs is the optional attention-mass witness (#852). nil by default — the
	// unobserved forward pass is byte-identical and allocation-identical. When set via
	// SetAttnObserver, the named attention seams emit a COPY of their post-softmax
	// weights at the softmax seam (emission only; the math is untouched). See
	// attn_observer.go.
	attnObs AttnObserver
}

// newModel assembles a Model from a built manifest + packed f32 blob, applying
// source-format tensor aliases and then the load-time fused-tensor split
// (Phi qkv_proj / gate_up_proj -> separate q/k/v / gate/up component views)
// before the model is handed to the forward pass. It is the single construction
// point every loader funnels through, so the split rule is applied uniformly and
// exactly once. On a Llama-shaped checkpoint with no aliases and no fused tensor,
// both steps are no-ops, so this path stays bit-identical for non-Phi models.
func newModel(cfg Config, man map[string]tensorMeta, raw []byte) (*Model, error) {
	if err := materializeQwen35Tensors(cfg, man); err != nil {
		return nil, err
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
	return &Model{Cfg: cfg, manifest: man, raw: raw}, nil
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
	return newModel(cfg, man, raw)
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

// tensor returns a zero-copy []float32 view of a named weight. The blob is
// little-endian f32 and amd64 is little-endian, so the bytes reinterpret directly.
func (m *Model) tensor(name string) []float32 {
	meta, ok := m.manifest[name]
	if !ok {
		panic("model: missing tensor " + name)
	}
	n := meta.Nbytes / 4
	return unsafe.Slice((*float32)(unsafe.Pointer(&m.raw[meta.Offset])), n)
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
