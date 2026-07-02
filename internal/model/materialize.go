package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// materializeTensorAliases applies explicit source-format aliases before any
// fused-tensor split. Each entry maps canonical-name -> source-name and creates a
// zero-copy canonical manifest row when the source exists. This lets a loader point
// a canonical fused tensor at a source-format fused name; splitFusedProjections can
// then carve the normal q/k/v component views without knowing the source name.
func materializeTensorAliases(cfg Config, man map[string]tensorMeta) error {
	if len(cfg.TensorAliases) == 0 {
		return nil
	}
	for canonical, source := range cfg.TensorAliases {
		if canonical == "" || source == "" {
			return fmt.Errorf("model: tensor_aliases contains empty canonical/source name")
		}
		if _, exists := man[canonical]; exists {
			continue
		}
		meta, ok := man[source]
		if !ok {
			return fmt.Errorf("model: tensor_aliases maps %s to missing source tensor %s", canonical, source)
		}
		man[canonical] = meta
	}
	return nil
}

func materializeGPTNeoXTensors(cfg Config, man map[string]tensorMeta, raw *[]byte) error {
	if !cfg.isGPTNeoX() && !manifestHasPrefix(man, "gpt_neox.") {
		return nil
	}
	aliasTensorIfPresent(man, "model.embed_tokens.weight", "gpt_neox.embed_in.weight")
	aliasTensorIfPresent(man, "lm_head.weight", "embed_out.weight")
	aliasTensorIfPresent(man, "model.norm.weight", "gpt_neox.final_layer_norm.weight")
	aliasTensorIfPresent(man, "model.norm.bias", "gpt_neox.final_layer_norm.bias")

	for l := 0; l < cfg.NumLayers; l++ {
		dst := layerPrefix(l)
		src := "gpt_neox.layers." + itoa(l) + "."
		aliasTensorIfPresent(man, dst+"input_layernorm.weight", src+"input_layernorm.weight")
		aliasTensorIfPresent(man, dst+"input_layernorm.bias", src+"input_layernorm.bias")
		aliasTensorIfPresent(man, dst+"post_attention_layernorm.weight", src+"post_attention_layernorm.weight")
		aliasTensorIfPresent(man, dst+"post_attention_layernorm.bias", src+"post_attention_layernorm.bias")
		aliasTensorIfPresent(man, dst+"self_attn.o_proj.weight", src+"attention.dense.weight")
		aliasTensorIfPresent(man, dst+"self_attn.o_proj.bias", src+"attention.dense.bias")
		aliasTensorIfPresent(man, dst+"mlp.gate_proj.weight", src+"mlp.dense_h_to_4h.weight")
		aliasTensorIfPresent(man, dst+"mlp.gate_proj.bias", src+"mlp.dense_h_to_4h.bias")
		aliasTensorIfPresent(man, dst+"mlp.down_proj.weight", src+"mlp.dense_4h_to_h.weight")
		aliasTensorIfPresent(man, dst+"mlp.down_proj.bias", src+"mlp.dense_4h_to_h.bias")
		if err := materializeGPTNeoXQKVWeight(cfg, l, man, raw); err != nil {
			return err
		}
		if err := materializeGPTNeoXQKVBias(cfg, l, man, raw); err != nil {
			return err
		}
	}
	return nil
}

func materializeFalconTensors(cfg Config, man map[string]tensorMeta, raw *[]byte) error {
	if !strings.Contains(cfg.archFamilyKey(), "falcon") && !manifestHasPrefix(man, "transformer.h.") {
		return nil
	}
	aliasTensorIfPresent(man, "model.embed_tokens.weight", "transformer.word_embeddings.weight")
	aliasTensorIfPresent(man, "model.norm.weight", "transformer.ln_f.weight")
	aliasTensorIfPresent(man, "model.norm.bias", "transformer.ln_f.bias")

	for l := 0; l < cfg.NumLayers; l++ {
		dst := layerPrefix(l)
		src := "transformer.h." + itoa(l) + "."
		aliasTensorIfPresent(man, dst+"input_layernorm.weight", src+"input_layernorm.weight")
		aliasTensorIfPresent(man, dst+"input_layernorm.bias", src+"input_layernorm.bias")
		aliasTensorIfPresent(man, dst+"self_attn.qkv_proj.weight", src+"self_attention.query_key_value.weight")
		aliasTensorIfPresent(man, dst+"self_attn.o_proj.weight", src+"self_attention.dense.weight")
		aliasTensorIfPresent(man, dst+"self_attn.o_proj.bias", src+"self_attention.dense.bias")
		aliasTensorIfPresent(man, dst+"mlp.gate_proj.weight", src+"mlp.dense_h_to_4h.weight")
		aliasTensorIfPresent(man, dst+"mlp.gate_proj.bias", src+"mlp.dense_h_to_4h.bias")
		aliasTensorIfPresent(man, dst+"mlp.down_proj.weight", src+"mlp.dense_4h_to_h.weight")
		aliasTensorIfPresent(man, dst+"mlp.down_proj.bias", src+"mlp.dense_4h_to_h.bias")
		if err := materializeContiguousQKVBias(cfg, dst, src+"self_attention.query_key_value.bias", man, raw); err != nil {
			return err
		}
	}
	return nil
}

func materializeMPTTensors(cfg Config, man map[string]tensorMeta) error {
	if !strings.Contains(cfg.archFamilyKey(), "mpt") && !manifestHasPrefix(man, "transformer.blocks.") {
		return nil
	}
	aliasTensorIfPresent(man, "model.embed_tokens.weight", "transformer.wte.weight")
	aliasTensorIfPresent(man, "model.norm.weight", "transformer.norm_f.weight")

	for l := 0; l < cfg.NumLayers; l++ {
		dst := layerPrefix(l)
		src := "transformer.blocks." + itoa(l) + "."
		aliasTensorIfPresent(man, dst+"input_layernorm.weight", src+"norm_1.weight")
		aliasTensorIfPresent(man, dst+"post_attention_layernorm.weight", src+"norm_2.weight")
		aliasTensorIfPresent(man, dst+"self_attn.qkv_proj.weight", src+"attn.Wqkv.weight")
		aliasTensorIfPresent(man, dst+"self_attn.o_proj.weight", src+"attn.out_proj.weight")
		aliasTensorIfPresent(man, dst+"mlp.gate_proj.weight", src+"ffn.up_proj.weight")
		aliasTensorIfPresent(man, dst+"mlp.down_proj.weight", src+"ffn.down_proj.weight")
	}
	return nil
}

func materializeMixtralBlockSparseTensors(cfg Config, man map[string]tensorMeta) error {
	if cfg.NumExperts <= 0 {
		return nil
	}
	if !strings.Contains(cfg.archFamilyKey(), "mixtral") && !manifestHasPrefix(man, "model.layers.0.block_sparse_moe.") {
		return nil
	}
	for l := 0; l < cfg.NumLayers; l++ {
		prefix := layerName(l, "block_sparse_moe.")
		aliasTensorIfPresent(man, routerName(l), prefix+"gate.weight")
		for e := 0; e < cfg.NumExperts; e++ {
			expertPrefix := prefix + "experts." + itoa(e) + "."
			aliasTensorIfPresent(man, expertName(l, e, "gate_proj.weight"), expertPrefix+"w1.weight")
			aliasTensorIfPresent(man, expertName(l, e, "down_proj.weight"), expertPrefix+"w2.weight")
			aliasTensorIfPresent(man, expertName(l, e, "up_proj.weight"), expertPrefix+"w3.weight")
		}
	}
	return nil
}

func materializeGPTOSSTensors(cfg Config, man map[string]tensorMeta, raw *[]byte) error {
	if !cfg.isGPTOSS() && !manifestHasPrefix(man, "model.layers.0.mlp.router.") {
		return nil
	}
	for l := 0; l < cfg.NumLayers; l++ {
		prefix := layerPrefix(l)
		aliasTensorIfPresent(man, prefix+"mlp.gate.weight", prefix+"mlp.router.weight")
		aliasTensorIfPresent(man, prefix+"mlp.gate.bias", prefix+"mlp.router.bias")
		if err := materializeGPTOSSExpertGateUp(cfg, l, man, raw); err != nil {
			return err
		}
		if err := materializeGPTOSSExpertDown(cfg, l, man, raw); err != nil {
			return err
		}
		if err := materializeGPTOSSExpertGateUpBias(cfg, l, man, raw); err != nil {
			return err
		}
		if err := materializeGPTOSSExpertDownBias(cfg, l, man, raw); err != nil {
			return err
		}
	}
	return nil
}

// materializeFusedExpertTensor is the shared driver for splitting one fused
// per-layer expert tensor into per-expert components: it resolves the first
// present alias among names, validates the fused tensor against wantShape via
// check (requireF32Shape for data-copy splits, requireMoESplitShape for
// zero-copy manifest splits), invokes perExpert for each of the E experts, and
// removes the fused source from the manifest. An absent source is a no-op.
func materializeFusedExpertTensor(man map[string]tensorMeta, E int, wantShape []int, check func(string, tensorMeta, []int) error, perExpert func(name string, meta tensorMeta, e int) error, names ...string) error {
	name, meta, ok := firstTensor(man, names...)
	if !ok {
		return nil
	}
	if err := check(name, meta, wantShape); err != nil {
		return err
	}
	for e := 0; e < E; e++ {
		if err := perExpert(name, meta, e); err != nil {
			return err
		}
	}
	delete(man, name)
	return nil
}

// metaF32Reader returns an accessor for flat f32 element k of a manifest tensor.
func metaF32Reader(raw *[]byte, meta tensorMeta) func(int) float32 {
	return func(k int) float32 { return readF32At(*raw, meta, k) }
}

// gptossGateUpExpert de-interleaves expert e's gate and up projections (each
// [I, H] row-major) out of a fused GPT-OSS gate_up tensor laid out [E, H, 2I]
// with gate/up interleaved on the last axis. at reads flat element k of the
// fused tensor.
func gptossGateUpExpert(e, I, H int, at func(int) float32) (gate, up []float32) {
	gate = make([]float32, I*H)
	up = make([]float32, I*H)
	for i := 0; i < I; i++ {
		for h := 0; h < H; h++ {
			src := ((e*H+h)*2*I + 2*i)
			gate[i*H+h] = at(src)
			up[i*H+h] = at(src + 1)
		}
	}
	return gate, up
}

// gptossGateUpBiasExpert de-interleaves expert e's gate and up bias vectors
// (each [I]) out of a fused GPT-OSS gate_up bias tensor laid out [E, 2I].
func gptossGateUpBiasExpert(e, I int, at func(int) float32) (gate, up []float32) {
	gate = make([]float32, I)
	up = make([]float32, I)
	for i := 0; i < I; i++ {
		src := e*2*I + 2*i
		gate[i] = at(src)
		up[i] = at(src + 1)
	}
	return gate, up
}

// gptossDownExpert transposes expert e's down projection ([H, I] row-major)
// out of a fused GPT-OSS down tensor laid out [E, I, H]. at reads flat element
// k of the fused tensor.
func gptossDownExpert(e, I, H int, at func(int) float32) []float32 {
	down := make([]float32, H*I)
	for h := 0; h < H; h++ {
		for i := 0; i < I; i++ {
			down[h*I+i] = at((e*I+i)*H + h)
		}
	}
	return down
}

// gptossGateUpDest returns expert e's gate/up destination tensor names with
// suffix kind ("weight" or "bias") and errors if either already exists in man;
// what names the components in the conflict message.
func gptossGateUpDest(man map[string]tensorMeta, name string, layer, e int, kind, what string) (gateName, upName string, err error) {
	gateName = expertName(layer, e, "gate_proj."+kind)
	upName = expertName(layer, e, "up_proj."+kind)
	if anyTensorPresent(man, gateName, upName) {
		return "", "", fmt.Errorf("model: cannot materialize %s: expert %d %s already exists", name, e, what)
	}
	return gateName, upName, nil
}

// gptossDownDest returns expert e's down destination tensor name with suffix
// kind and errors if it already exists in man.
func gptossDownDest(man map[string]tensorMeta, name string, layer, e int, kind, what string) (string, error) {
	downName := expertName(layer, e, "down_proj."+kind)
	if _, exists := man[downName]; exists {
		return "", fmt.Errorf("model: cannot materialize %s: expert %d %s already exists", name, e, what)
	}
	return downName, nil
}

// materializeGPTOSSGateUpPair splits one fused GPT-OSS gate/up tensor (weight
// or bias flavor) into per-expert gate/up components appended to the manifest.
func materializeGPTOSSGateUpPair(man map[string]tensorMeta, raw *[]byte, layer, E int, wantShape, partShape []int, kind, what string, extract func(meta tensorMeta, e int) (gate, up []float32), names ...string) error {
	return materializeFusedExpertTensor(man, E, wantShape, requireF32Shape,
		func(name string, meta tensorMeta, e int) error {
			gateName, upName, err := gptossGateUpDest(man, name, layer, e, kind, what)
			if err != nil {
				return err
			}
			gate, up := extract(meta, e)
			appendF32Tensor(man, raw, gateName, partShape, gate)
			appendF32Tensor(man, raw, upName, partShape, up)
			return nil
		}, names...)
}

// materializeGPTOSSDownSingle splits one fused GPT-OSS down tensor (weight or
// bias flavor) into per-expert down components appended to the manifest.
func materializeGPTOSSDownSingle(man map[string]tensorMeta, raw *[]byte, layer, E int, wantShape, partShape []int, kind, what string, extract func(meta tensorMeta, e int) []float32, names ...string) error {
	return materializeFusedExpertTensor(man, E, wantShape, requireF32Shape,
		func(name string, meta tensorMeta, e int) error {
			downName, err := gptossDownDest(man, name, layer, e, kind, what)
			if err != nil {
				return err
			}
			appendF32Tensor(man, raw, downName, partShape, extract(meta, e))
			return nil
		}, names...)
}

func materializeGPTOSSExpertGateUp(cfg Config, layer int, man map[string]tensorMeta, raw *[]byte) error {
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	return materializeGPTOSSGateUpPair(man, raw, layer, E, []int{E, H, 2 * I}, []int{I, H}, "weight", "gate/up component",
		func(meta tensorMeta, e int) (gate, up []float32) {
			return gptossGateUpExpert(e, I, H, metaF32Reader(raw, meta))
		},
		layerName(layer, "mlp.experts.gate_up_proj"),
		layerName(layer, "mlp.experts.gate_up_proj.weight"),
	)
}

func materializeGPTOSSExpertDown(cfg Config, layer int, man map[string]tensorMeta, raw *[]byte) error {
	E, I, H := cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize
	return materializeGPTOSSDownSingle(man, raw, layer, E, []int{E, I, H}, []int{H, I}, "weight", "down component",
		func(meta tensorMeta, e int) []float32 {
			return gptossDownExpert(e, I, H, metaF32Reader(raw, meta))
		},
		layerName(layer, "mlp.experts.down_proj"),
		layerName(layer, "mlp.experts.down_proj.weight"),
	)
}

func materializeGPTOSSExpertGateUpBias(cfg Config, layer int, man map[string]tensorMeta, raw *[]byte) error {
	E, I := cfg.NumExperts, cfg.IntermediateSize
	return materializeGPTOSSGateUpPair(man, raw, layer, E, []int{E, 2 * I}, []int{I}, "bias", "gate/up bias",
		func(meta tensorMeta, e int) (gate, up []float32) {
			return gptossGateUpBiasExpert(e, I, metaF32Reader(raw, meta))
		},
		layerName(layer, "mlp.experts.gate_up_proj_bias"),
		layerName(layer, "mlp.experts.gate_up_proj.bias"),
	)
}

func materializeGPTOSSExpertDownBias(cfg Config, layer int, man map[string]tensorMeta, raw *[]byte) error {
	E, H := cfg.NumExperts, cfg.HiddenSize
	return materializeGPTOSSDownSingle(man, raw, layer, E, []int{E, H}, []int{H}, "bias", "down bias",
		func(meta tensorMeta, e int) []float32 {
			down := make([]float32, H)
			for h := 0; h < H; h++ {
				down[h] = readF32At(*raw, meta, e*H+h)
			}
			return down
		},
		layerName(layer, "mlp.experts.down_proj_bias"),
		layerName(layer, "mlp.experts.down_proj.bias"),
	)
}

func materializeContiguousQKVBias(cfg Config, dstPrefix, srcName string, man map[string]tensorMeta, raw *[]byte) error {
	src, ok := man[srcName]
	if !ok {
		return nil
	}
	qName, kName, vName := dstPrefix+"self_attn.q_proj.bias", dstPrefix+"self_attn.k_proj.bias", dstPrefix+"self_attn.v_proj.bias"
	if skip, err := qkvDestStatus(man, srcName, "bias", qName, kName, vName); skip || err != nil {
		return err
	}
	qRows, kRows, vRows := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	if err := requireF32Shape(srcName, src, []int{qRows + kRows + vRows}); err != nil {
		return err
	}
	q, k, v := make([]float32, qRows), make([]float32, kRows), make([]float32, vRows)
	for i := range q {
		q[i] = readF32At(*raw, src, i)
	}
	for i := range k {
		k[i] = readF32At(*raw, src, qRows+i)
	}
	for i := range v {
		v[i] = readF32At(*raw, src, qRows+kRows+i)
	}
	appendF32Tensor(man, raw, qName, []int{qRows}, q)
	appendF32Tensor(man, raw, kName, []int{kRows}, k)
	appendF32Tensor(man, raw, vName, []int{vRows}, v)
	return nil
}

// qkvDestStatus reports whether a q/k/v materialization should proceed for the given
// destination tensor names. It returns skip=true when all three already exist (nothing to
// do), and a non-nil error when only some exist (a partial/conflicting manifest). `kind`
// names the components in the conflict message ("bias", "component"). Shared by the
// contiguous and GPT-NeoX q/k/v materializers, which differ only in the split that follows.
func qkvDestStatus(man map[string]tensorMeta, srcName, kind, qName, kName, vName string) (skip bool, err error) {
	if allTensorsPresent(man, qName, kName, vName) {
		return true, nil
	}
	if anyTensorPresent(man, qName, kName, vName) {
		return false, fmt.Errorf("model: cannot materialize %s: one or more q/k/v %s tensors already exist", srcName, kind)
	}
	return false, nil
}

func manifestHasPrefix(man map[string]tensorMeta, prefix string) bool {
	for name := range man {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func aliasTensorIfPresent(man map[string]tensorMeta, canonical, source string) {
	if _, exists := man[canonical]; exists {
		return
	}
	if meta, ok := man[source]; ok {
		man[canonical] = meta
	}
}

func materializeGPTNeoXQKVWeight(cfg Config, layer int, man map[string]tensorMeta, raw *[]byte) error {
	srcName := "gpt_neox.layers." + itoa(layer) + ".attention.query_key_value.weight"
	src, ok := man[srcName]
	if !ok {
		return nil
	}
	p := layerPrefix(layer)
	qName, kName, vName := p+"self_attn.q_proj.weight", p+"self_attn.k_proj.weight", p+"self_attn.v_proj.weight"
	if skip, err := qkvDestStatus(man, srcName, "component", qName, kName, vName); skip || err != nil {
		return err
	}
	if cfg.NumKVHeads != cfg.NumHeads {
		return fmt.Errorf("model: GPT-NeoX query_key_value split requires NumKVHeads==NumHeads, got %d/%d", cfg.NumKVHeads, cfg.NumHeads)
	}
	H, nH, hd := cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim
	if err := requireF32Shape(srcName, src, []int{3 * nH * hd, H}); err != nil {
		return err
	}
	q, k, v := make([]float32, nH*hd*H), make([]float32, nH*hd*H), make([]float32, nH*hd*H)
	for h := 0; h < nH; h++ {
		for d := 0; d < hd; d++ {
			dstRow := h*hd + d
			srcQ := h*3*hd + d
			srcK := h*3*hd + hd + d
			srcV := h*3*hd + 2*hd + d
			copyF32Row(q, dstRow, *raw, src, srcQ, H)
			copyF32Row(k, dstRow, *raw, src, srcK, H)
			copyF32Row(v, dstRow, *raw, src, srcV, H)
		}
	}
	appendF32Tensor(man, raw, qName, []int{nH * hd, H}, q)
	appendF32Tensor(man, raw, kName, []int{nH * hd, H}, k)
	appendF32Tensor(man, raw, vName, []int{nH * hd, H}, v)
	return nil
}

func materializeGPTNeoXQKVBias(cfg Config, layer int, man map[string]tensorMeta, raw *[]byte) error {
	srcName := "gpt_neox.layers." + itoa(layer) + ".attention.query_key_value.bias"
	src, ok := man[srcName]
	if !ok {
		return nil
	}
	p := layerPrefix(layer)
	qName, kName, vName := p+"self_attn.q_proj.bias", p+"self_attn.k_proj.bias", p+"self_attn.v_proj.bias"
	if skip, err := qkvDestStatus(man, srcName, "bias", qName, kName, vName); skip || err != nil {
		return err
	}
	if cfg.NumKVHeads != cfg.NumHeads {
		return fmt.Errorf("model: GPT-NeoX query_key_value bias split requires NumKVHeads==NumHeads, got %d/%d", cfg.NumKVHeads, cfg.NumHeads)
	}
	nH, hd := cfg.NumHeads, cfg.HeadDim
	if err := requireF32Shape(srcName, src, []int{3 * nH * hd}); err != nil {
		return err
	}
	q, k, v := make([]float32, nH*hd), make([]float32, nH*hd), make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		for d := 0; d < hd; d++ {
			dst := h*hd + d
			q[dst] = readF32At(*raw, src, h*3*hd+d)
			k[dst] = readF32At(*raw, src, h*3*hd+hd+d)
			v[dst] = readF32At(*raw, src, h*3*hd+2*hd+d)
		}
	}
	appendF32Tensor(man, raw, qName, []int{nH * hd}, q)
	appendF32Tensor(man, raw, kName, []int{nH * hd}, k)
	appendF32Tensor(man, raw, vName, []int{nH * hd}, v)
	return nil
}

func allTensorsPresent(man map[string]tensorMeta, names ...string) bool {
	for _, name := range names {
		if _, ok := man[name]; !ok {
			return false
		}
	}
	return true
}

func anyTensorPresent(man map[string]tensorMeta, names ...string) bool {
	for _, name := range names {
		if _, ok := man[name]; ok {
			return true
		}
	}
	return false
}

func requireF32Shape(name string, meta tensorMeta, want []int) error {
	if !strings.EqualFold(meta.Dtype, "f32") {
		return fmt.Errorf("model: tensor %s has dtype %s, want f32", name, meta.Dtype)
	}
	if len(meta.Shape) != len(want) {
		return fmt.Errorf("model: tensor %s has shape %v, want %v", name, meta.Shape, want)
	}
	elems := 1
	for i, d := range want {
		if meta.Shape[i] != d {
			return fmt.Errorf("model: tensor %s has shape %v, want %v", name, meta.Shape, want)
		}
		elems *= d
	}
	if meta.Nbytes != elems*4 {
		return fmt.Errorf("model: tensor %s has %d bytes, shape %v f32 implies %d", name, meta.Nbytes, meta.Shape, elems*4)
	}
	return nil
}

func copyF32Row(dst []float32, dstRow int, raw []byte, src tensorMeta, srcRow, cols int) {
	for c := 0; c < cols; c++ {
		dst[dstRow*cols+c] = readF32At(raw, src, srcRow*cols+c)
	}
}

func readF32At(raw []byte, meta tensorMeta, idx int) float32 {
	off := meta.Offset + idx*4
	return math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
}

func appendF32Tensor(man map[string]tensorMeta, raw *[]byte, name string, shape []int, data []float32) {
	offset := len(*raw)
	nbytes := len(data) * 4
	*raw = append(*raw, make([]byte, nbytes)...)
	for i, v := range data {
		binary.LittleEndian.PutUint32((*raw)[offset+i*4:], math.Float32bits(v))
	}
	man[name] = tensorMeta{Dtype: "f32", Shape: append([]int(nil), shape...), Offset: offset, Nbytes: nbytes}
}
