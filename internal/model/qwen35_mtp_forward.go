package model

import (
	"fmt"
	"strings"
)

// Qwen35MTPForwardError reports a typed contract failure in the retained
// Qwen3.8 MTP path. It never permits substituting target-layer tensors or fake
// logits for a malformed or unsupported checkpoint.
type Qwen35MTPForwardError struct {
	Stage  string
	Tensor string
	Want   string
	Got    string
}

func (e *Qwen35MTPForwardError) Error() string {
	where := e.Stage
	if e.Tensor != "" {
		where += " tensor " + e.Tensor
	}
	return fmt.Sprintf("model: qwen3.8 MTP %s: got %s, want %s", where, e.Got, e.Want)
}

// Qwen35MTPForward is an isolated, stateful native draft head. Its decoder
// layer owns a separate KV cache; the target Session cache is never read or
// mutated. The retained checkpoint payload and shared target LM head remain
// immutable and are reused without copying.
type Qwen35MTPForward struct {
	target  *Model
	draft   *Session
	lastPos int
	closed  bool
}

// NewQwen35MTPForward binds the exact mtp.layers.0 namespace to the shared Qwen
// decoder-layer primitive and binds mtp.norm plus the target model's LM head.
// The current primitive is f32-only because the retained MTP tensors do not yet
// have resident Q8/Q4 stores; unsupported storage is refused explicitly.
func (m *Model) NewQwen35MTPForward() (*Qwen35MTPForward, error) {
	if m == nil {
		return nil, qwen35MTPStateError("model", "non-nil model", "nil")
	}
	if !m.holdModelWeights() {
		return nil, qwen35MTPStateError("model weights", "open checkpoint weights", "closing or closed")
	}
	held := true
	defer func() {
		if held {
			m.releaseWeightSession()
		}
	}()

	mode, err := qwen35MTPAdmission(m.Cfg, m.manifest, false)
	if err != nil {
		return nil, err
	}
	if !mode.Enabled {
		return nil, qwen35MTPStateError("model", "eligible one-layer shared-embedding Qwen3.8 MTP model", mode.Reason)
	}
	if err := m.validateQwen35MTPForwardTensors(); err != nil {
		return nil, err
	}

	cfg := m.Cfg
	cfg.NumLayers = 1
	// Qwen3.8's retained MTP decoder is a full-attention layer. The target may
	// be hybrid, but its layer_types indices describe target layers, not this
	// separately named draft layer.
	cfg.LayerTypes = []string{"full_attention"}

	aliases := make(map[string]tensorMeta, len(qwen35MTPDecoderAliases)+2)
	for dst, src := range qwen35MTPDecoderAliases {
		aliases[dst] = m.manifest[src]
	}
	aliases["model.norm.weight"] = m.manifest["mtp.norm.weight"]
	headName, err := m.qwen35MTPHeadName()
	if err != nil {
		return nil, err
	}
	aliases["lm_head.weight"] = m.manifest[headName]

	draftModel := &Model{Cfg: cfg, manifest: aliases, raw: m.raw}
	draft := &Session{M: draftModel, Cache: NewKVCache(cfg)}
	draft.initMixedQKV()
	held = false
	return &Qwen35MTPForward{target: m, draft: draft, lastPos: -1}, nil
}

// Close releases the target checkpoint lifetime held by this draft head.
func (f *Qwen35MTPForward) Close() {
	if f == nil || f.closed {
		return
	}
	f.closed = true
	if f.draft != nil {
		f.draft.Close()
	}
	if f.target != nil {
		f.target.releaseWeightSession()
	}
}

// Forward executes one native Qwen3.8 MTP draft position:
//
//  1. normalize prior target hidden and current token embedding separately;
//  2. concatenate [hidden, embedding] and apply mtp.fc;
//  3. execute the exact retained mtp.layers.0 decoder tensors;
//  4. apply mtp.norm and the shared target LM head.
func (f *Qwen35MTPForward) Forward(pos int, priorHidden, currentEmbedding []float32) ([]float32, error) {
	if f == nil || f.target == nil || f.draft == nil || f.draft.Cache == nil {
		return nil, qwen35MTPStateError("forward state", "initialized Qwen35MTPForward", "nil or incomplete")
	}
	if f.closed {
		return nil, qwen35MTPStateError("forward state", "open Qwen35MTPForward", "closed")
	}
	if pos < 0 {
		return nil, qwen35MTPStateError("position", "non-negative", fmt.Sprint(pos))
	}
	if pos <= f.lastPos {
		return nil, qwen35MTPStateError("position", fmt.Sprintf("greater than %d", f.lastPos), fmt.Sprint(pos))
	}

	x, err := f.target.Qwen35MTPFuse(priorHidden, currentEmbedding)
	if err != nil {
		return nil, err
	}
	cos, sin := ropeRowForLayer(f.draft.M.Cfg, 0, pos)
	x = f.draft.blockStep(0, pos, x, cos, sin, f32Kernel{f.draft.M})
	f.draft.Cache.appendPosition(pos, -1)
	f.lastPos = pos
	return f.draft.head(f.draft.M.finalNorm(x)), nil
}

// Qwen35MTPFuse implements the checkpoint-defined pre-layer path exactly:
// normalize the prior target hidden state and current token embedding
// independently, concatenate them in that order, then apply mtp.fc.
func (m *Model) Qwen35MTPFuse(priorHidden, currentEmbedding []float32) ([]float32, error) {
	if m == nil {
		return nil, qwen35MTPStateError("model", "non-nil model", "nil")
	}
	if _, err := qwen35MTPAdmission(m.Cfg, m.manifest, false); err != nil {
		return nil, err
	}
	if !m.Cfg.isQwen35TextFamily() || m.Cfg.NumMTPLayers() != 1 || m.Cfg.MTPUseDedicatedEmbeddings {
		return nil, qwen35MTPStateError("model", "eligible one-layer shared-embedding Qwen3.8 MTP model", "ineligible config")
	}

	h := m.Cfg.HiddenSize
	if h <= 0 {
		return nil, qwen35MTPStateError("hidden size", "positive", fmt.Sprint(h))
	}
	if len(priorHidden) != h {
		return nil, qwen35MTPStateError("prior hidden shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(priorHidden)))
	}
	if len(currentEmbedding) != h {
		return nil, qwen35MTPStateError("current embedding shape", fmt.Sprintf("[%d]", h), fmt.Sprintf("[%d]", len(currentEmbedding)))
	}
	if m.Cfg.RMSNormEps < 0 {
		return nil, qwen35MTPStateError("RMS norm epsilon", "non-negative", fmt.Sprint(m.Cfg.RMSNormEps))
	}

	hiddenNorm, err := m.qwen35MTPF32Tensor("mtp.pre_fc_norm_hidden.weight", []int{h})
	if err != nil {
		return nil, err
	}
	embeddingNorm, err := m.qwen35MTPF32Tensor("mtp.pre_fc_norm_embedding.weight", []int{h})
	if err != nil {
		return nil, err
	}
	fc, err := m.qwen35MTPF32Tensor("mtp.fc.weight", []int{h, 2 * h})
	if err != nil {
		return nil, err
	}

	eps := float32(m.Cfg.RMSNormEps)
	fusedInput := make([]float32, 0, 2*h)
	fusedInput = append(fusedInput, rmsnorm(priorHidden, hiddenNorm, eps)...)
	fusedInput = append(fusedInput, rmsnorm(currentEmbedding, embeddingNorm, eps)...)
	return parMatRows(fc, fusedInput, h, 2*h), nil
}

var qwen35MTPDecoderAliases = map[string]string{
	"model.layers.0.input_layernorm.weight":          "mtp.layers.0.input_layernorm.weight",
	"model.layers.0.post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
	"model.layers.0.self_attn.q_norm.weight":         "mtp.layers.0.self_attn.q_norm.weight",
	"model.layers.0.self_attn.k_norm.weight":         "mtp.layers.0.self_attn.k_norm.weight",
	"model.layers.0.self_attn.q_proj.weight":         "mtp.layers.0.self_attn.q_proj.weight",
	"model.layers.0.self_attn.k_proj.weight":         "mtp.layers.0.self_attn.k_proj.weight",
	"model.layers.0.self_attn.v_proj.weight":         "mtp.layers.0.self_attn.v_proj.weight",
	"model.layers.0.self_attn.o_proj.weight":         "mtp.layers.0.self_attn.o_proj.weight",
	"model.layers.0.mlp.gate_proj.weight":            "mtp.layers.0.mlp.gate_proj.weight",
	"model.layers.0.mlp.up_proj.weight":              "mtp.layers.0.mlp.up_proj.weight",
	"model.layers.0.mlp.down_proj.weight":            "mtp.layers.0.mlp.down_proj.weight",
}

func (m *Model) validateQwen35MTPForwardTensors() error {
	cfg := m.Cfg
	h, hd := cfg.HiddenSize, cfg.HeadDim
	if h <= 0 || hd <= 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.IntermediateSize <= 0 || cfg.VocabSize <= 0 {
		return qwen35MTPStateError("config dimensions", "positive hidden/head/attention/intermediate/vocab dimensions", fmt.Sprintf("hidden=%d head_dim=%d heads=%d kv_heads=%d intermediate=%d vocab=%d", h, hd, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize, cfg.VocabSize))
	}
	qOut := cfg.NumHeads * hd
	if cfg.AttnOutputGate {
		qOut *= 2
	}
	shapes := map[string][]int{
		"mtp.fc.weight":                                {h, 2 * h},
		"mtp.pre_fc_norm_embedding.weight":             {h},
		"mtp.pre_fc_norm_hidden.weight":                {h},
		"mtp.norm.weight":                              {h},
		"mtp.layers.0.input_layernorm.weight":          {h},
		"mtp.layers.0.post_attention_layernorm.weight": {h},
		"mtp.layers.0.self_attn.q_norm.weight":         {hd},
		"mtp.layers.0.self_attn.k_norm.weight":         {hd},
		"mtp.layers.0.self_attn.q_proj.weight":         {qOut, h},
		"mtp.layers.0.self_attn.k_proj.weight":         {cfg.NumKVHeads * hd, h},
		"mtp.layers.0.self_attn.v_proj.weight":         {cfg.NumKVHeads * hd, h},
		"mtp.layers.0.self_attn.o_proj.weight":         {h, cfg.NumHeads * hd},
		"mtp.layers.0.mlp.gate_proj.weight":            {cfg.IntermediateSize, h},
		"mtp.layers.0.mlp.up_proj.weight":              {cfg.IntermediateSize, h},
		"mtp.layers.0.mlp.down_proj.weight":            {h, cfg.IntermediateSize},
	}
	for _, name := range qwen35MTPRequiredTensors {
		if _, err := m.qwen35MTPF32Tensor(name, shapes[name]); err != nil {
			return err
		}
	}
	_, err := m.qwen35MTPHeadName()
	return err
}

func (m *Model) qwen35MTPHeadName() (string, error) {
	name := "lm_head.weight"
	if !m.has(name) {
		name = "model.embed_tokens.weight"
	}
	if _, err := m.qwen35MTPF32Tensor(name, []int{m.Cfg.VocabSize, m.Cfg.HiddenSize}); err != nil {
		return "", err
	}
	return name, nil
}

func (m *Model) qwen35MTPF32Tensor(name string, wantShape []int) ([]float32, error) {
	meta, ok := m.manifest[name]
	if !ok {
		return nil, &Qwen35MTPForwardError{Stage: "weight lookup", Tensor: name, Want: "present", Got: "missing"}
	}
	if !strings.EqualFold(meta.Dtype, "F32") {
		return nil, &Qwen35MTPForwardError{Stage: "weight dtype", Tensor: name, Want: "F32", Got: meta.Dtype}
	}
	if !sameIntShape(meta.Shape, wantShape) {
		return nil, &Qwen35MTPForwardError{Stage: "weight shape", Tensor: name, Want: fmt.Sprint(wantShape), Got: fmt.Sprint(meta.Shape)}
	}
	wantBytes := 4
	for _, dim := range wantShape {
		wantBytes *= dim
	}
	if meta.Nbytes != wantBytes || meta.Offset < 0 || meta.Offset+meta.Nbytes > len(m.raw) {
		return nil, &Qwen35MTPForwardError{Stage: "weight storage", Tensor: name, Want: fmt.Sprintf("%d bytes inside model payload", wantBytes), Got: fmt.Sprintf("offset=%d nbytes=%d payload=%d", meta.Offset, meta.Nbytes, len(m.raw))}
	}
	return m.tensor(name), nil
}

func qwen35MTPStateError(stage, want, got string) *Qwen35MTPForwardError {
	return &Qwen35MTPForwardError{Stage: stage, Want: want, Got: got}
}

func sameIntShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
