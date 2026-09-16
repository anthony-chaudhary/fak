package model

// v41_inventory.go Ã¢â‚¬â€ the exact, typed DeepSeek-V4.1-Flash text-tensor inventory
// (issue #12892, leaf of parent #12640; coordinates with #12803).
//
// The pinned config (testdata/deepseek_v41_flash_config.json) admits V4.1
// geometry, but nothing enumerated the checkpoint's text tensors: a consumer
// could not name a tensor reference without re-deriving the naming scheme and
// shape formulas by hand. This file is that enumeration. It is metadata only Ã¢â‚¬â€
// it reads no weights, allocates no payload, and does not admit or implement the
// native forward (ErrV41NativeUnsupported still fences every execution seam).
//
// Naming scheme. The on-disk V4.1 safetensors tensors use the checkpoint's own
// compact namespace, shared with V4-Flash's quant path:
//
//	layers.<N>.attn.wq_a.weight        dense FP8, F8_E4M3 + .scale F8_E8M0
//	layers.<N>.attn.wq_b.weight        dense FP8
//	layers.<N>.attn.wkv.weight         dense FP8 (fused MLA KV latent)
//	layers.<N>.attn.wo_a.weight        dense FP8 (grouped low-rank output A)
//	layers.<N>.attn.wo_b.weight        dense FP8 (grouped low-rank output B)
//	layers.<N>.ffn.gate.weight         router, dense FP8
//	layers.<N>.ffn.gate.e_score_correction_bias   F32
//	layers.<N>.ffn.shared_experts.w1.weight        dense FP8
//	layers.<N>.ffn.shared_experts.w2.weight        dense FP8
//	layers.<N>.ffn.shared_experts.w3.weight        dense FP8
//	layers.<N>.ffn.experts.<E>.w1.weight           packed MXFP4, I8 + .scale F8_E8M0
//	layers.<N>.ffn.experts.<E>.w2.weight           packed MXFP4
//	layers.<N>.ffn.experts.<E>.w3.weight           packed MXFP4
//	layers.<N>.attn_norm.weight        F32
//	layers.<N>.ffn_norm.weight         F32
//	model.embed_tokens.weight          BF16
//	model.norm.weight                  F32
//	lm_head.weight                     dense FP8 (untied; V4.1 ties are false)
//
// Every dense FP8 weight carries a sibling `<name-without-.weight>.scale` of
// dtype F8_E8M0 with shape [ceil(O/32), ceil(I/32)] (config weight_block_size
// [32,32]). Every routed expert weight carries its own `<...>.scale` sibling and
// packs two OCP E2M1 nibbles per byte with one E8M0 byte per 32 unpacked K
// values. The sibling name is exactly the weight name with its trailing
// ".weight" replaced by ".scale", matching the existing pair convention.
//
// Shapes are exact and derived only from admitted config fields, except the KV
// latent width, which the published V4.1 text config does not carry as a key and
// which is therefore pinned as v41KVLoraRank (a checkpoint constant).

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ErrV41Inventory reports malformed or inadmissible V4.1 inventory geometry. A
// construction that cannot produce an exact inventory fails closed before any
// tensor payload is considered.
var ErrV41Inventory = errors.New("model: invalid DeepSeek V4.1 tensor inventory")

// v41FP8BlockDim is the DeepSeek-V4.1 dense FP8 block edge: the checkpoint
// declares weight_block_size [32,32], so each E8M0 scale covers a 32x32 tile.
const v41FP8BlockDim = 32

// V41FP8BlockDim exposes the V4.1 dense FP8 block edge to the internal/model/v41
// leaf package without renaming the core constant.
const V41FP8BlockDim = v41FP8BlockDim

// v41KVLoraRank is the published DeepSeek-V4.1 KV latent width. The V4.1 text
// config does not carry a kv_lora_rank key, so this geometry is a checkpoint
// constant rather than a parsed field (mirrors the quant decode path).
const v41KVLoraRank = 512

// V41TensorClass is the closed class of a V4.1 text tensor.
type V41TensorClass string

const (
	V41ClassEmbedding      V41TensorClass = "embedding"
	V41ClassNorm           V41TensorClass = "norm"
	V41ClassAttentionDense V41TensorClass = "attention_dense"
	V41ClassRouter         V41TensorClass = "router"
	V41ClassRouterBias     V41TensorClass = "router_bias"
	V41ClassRoutedExpert   V41TensorClass = "routed_expert"
	V41ClassSharedExpert   V41TensorClass = "shared_expert"
	V41ClassHead           V41TensorClass = "head"
)

// V41TensorRef is one exact text-tensor reference. Shape describes the on-disk
// weight tensor; ScaleName/ScaleShape/ScaleDtype describe its scale sibling
// (empty when the family carries no scale, e.g. high-precision norms, or an F32
// router bias). PackedWeight marks a weight whose Shape is the on-disk *packed*
// shape (two E2M1 nibbles per byte); LogicalShape carries the unpacked [O,I]
// shape for every tensor.
type V41TensorRef struct {
	Name         string         `json:"name"`
	Class        V41TensorClass `json:"class"`
	Layer        int            `json:"layer,omitempty"`
	Dtype        string         `json:"dtype"`
	Shape        []int          `json:"shape"`
	LogicalShape []int          `json:"logical_shape"`
	PackedWeight bool           `json:"packed_weight,omitempty"`
	ScaleName    string         `json:"scale_name,omitempty"`
	ScaleDtype   string         `json:"scale_dtype,omitempty"`
	ScaleShape   []int          `json:"scale_shape,omitempty"`
}

// HasScale reports whether the tensor carries a scale sibling.
func (r V41TensorRef) HasScale() bool { return r.ScaleName != "" }

// V41Inventory is an immutable, name-addressable V4.1 text-tensor manifest. It
// contains no payload bytes.
type V41Inventory struct {
	byName map[string]V41TensorRef
	order  []V41TensorRef
	names  []string
	layers int
}

// Len returns the number of weight tensors (scale siblings are not counted).
func (inv *V41Inventory) Len() int {
	if inv == nil {
		return 0
	}
	return len(inv.order)
}

// Layers returns the number of decoder layers the inventory was built for.
func (inv *V41Inventory) Layers() int {
	if inv == nil {
		return 0
	}
	return inv.layers
}

// Entries returns the inventory in stable construction order. The returned slice
// is a copy.
func (inv *V41Inventory) Entries() []V41TensorRef {
	if inv == nil {
		return nil
	}
	return append([]V41TensorRef(nil), inv.order...)
}

// Names returns the weight tensor names in the same stable order. The returned
// slice is a copy.
func (inv *V41Inventory) Names() []string {
	if inv == nil {
		return nil
	}
	return append([]string(nil), inv.names...)
}

// Lookup returns the exact reference for a weight name or its scale sibling, or
// ok=false when the name is not part of this inventory.
func (inv *V41Inventory) Lookup(name string) (V41TensorRef, bool) {
	if inv == nil {
		return V41TensorRef{}, false
	}
	if ref, ok := inv.byName[name]; ok {
		return ref, true
	}
	for _, ref := range inv.order {
		if ref.ScaleName == name {
			return ref, true
		}
	}
	return V41TensorRef{}, false
}

// TotalWeightBytes returns the sum of only the weight payload bytes the
// inventory describes (scale bytes excluded), computed with checked arithmetic.
// It is a deterministic metadata figure, not a residency claim.
func (inv *V41Inventory) TotalWeightBytes() (int64, error) {
	if inv == nil {
		return 0, fmt.Errorf("%w: nil inventory", ErrV41Inventory)
	}
	var total int64
	for _, ref := range inv.order {
		n, ok := checkedShapeProduct(ref.Shape...)
		if !ok {
			return 0, fmt.Errorf("%w: %s shape %v overflows", ErrV41Inventory, ref.Name, ref.Shape)
		}
		total += int64(n)
	}
	return total, nil
}

// DeepSeekV41Inventory builds the exact text-tensor inventory for an admitted
// V4.1 config. It fails closed (ErrV41Inventory) unless the config is V4.1 with
// every geometry axis the inventory depends on positive. The result contains no
// payload and opens no file.
func DeepSeekV41Inventory(c Config) (*V41Inventory, error) {
	if !c.IsDeepSeekV41() || c.DeepSeekV41 == nil {
		return nil, fmt.Errorf("%w: config is not admitted DeepSeek V4.1", ErrV41Inventory)
	}
	if c.NumLayers <= 0 || c.HiddenSize <= 0 || c.NumHeads <= 0 || c.HeadDim <= 0 ||
		c.QLoraRank <= 0 || c.OLoraRank <= 0 || c.OGroups <= 0 ||
		c.MoEIntermediateSize <= 0 || c.NumExperts <= 0 || c.VocabSize <= 0 {
		return nil, fmt.Errorf("%w: incomplete geometry", ErrV41Inventory)
	}
	// The routed-expert packed layout halves the reduction dimension (two E2M1
	// nibbles per byte) and places one E8M0 scale byte per 32 unpacked K values.
	// A reduction axis that is not a multiple of both is not representable, so
	// refuse rather than silently truncate into a wrong packed/scale shape.
	if c.HiddenSize%2 != 0 || c.HiddenSize%v41FP8BlockDim != 0 {
		return nil, fmt.Errorf("%w: hidden size %d is not a multiple of %d", ErrV41Inventory, c.HiddenSize, v41FP8BlockDim)
	}
	if c.MoEIntermediateSize%2 != 0 || c.MoEIntermediateSize%v41FP8BlockDim != 0 {
		return nil, fmt.Errorf("%w: moe_intermediate_size %d is not a multiple of %d", ErrV41Inventory, c.MoEIntermediateSize, v41FP8BlockDim)
	}
	qHeadDim, ok := checkedShapeProduct(c.NumHeads, c.HeadDim)
	if !ok || qHeadDim <= 0 {
		return nil, fmt.Errorf("%w: num_heads*head_dim overflow", ErrV41Inventory)
	}
	oDim, ok := checkedShapeProduct(c.OLoraRank, c.OGroups)
	if !ok || oDim <= 0 {
		return nil, fmt.Errorf("%w: o_lora_rank*o_groups overflow", ErrV41Inventory)
	}

	inv := &V41Inventory{byName: map[string]V41TensorRef{}, layers: c.NumLayers}
	add := func(ref V41TensorRef) error {
		if ref.Name == "" {
			return fmt.Errorf("%w: empty tensor name", ErrV41Inventory)
		}
		if _, dup := inv.byName[ref.Name]; dup {
			return fmt.Errorf("%w: duplicate tensor %s", ErrV41Inventory, ref.Name)
		}
		inv.byName[ref.Name] = ref
		inv.order = append(inv.order, ref)
		return nil
	}
	dense := func(name string, class V41TensorClass, layer, o, i int) V41TensorRef {
		return V41TensorRef{
			Name:         name,
			Class:        class,
			Layer:        layer,
			Dtype:        "F8_E4M3",
			Shape:        []int{o, i},
			LogicalShape: []int{o, i},
			ScaleName:    strings.TrimSuffix(name, ".weight") + ".scale",
			ScaleDtype:   "F8_E8M0",
			ScaleShape:   []int{(o-1)/v41FP8BlockDim + 1, (i-1)/v41FP8BlockDim + 1},
		}
	}
	plain := func(name string, class V41TensorClass, dtype string, layer int, shape ...int) V41TensorRef {
		return V41TensorRef{
			Name:         name,
			Class:        class,
			Layer:        layer,
			Dtype:        dtype,
			Shape:        append([]int(nil), shape...),
			LogicalShape: append([]int(nil), shape...),
		}
	}

	if err := add(plain("model.embed_tokens.weight", V41ClassEmbedding, "BF16", 0, c.VocabSize, c.HiddenSize)); err != nil {
		return nil, err
	}
	for layer := 0; layer < c.NumLayers; layer++ {
		prefix := "layers." + strconv.Itoa(layer)
		for _, ref := range []V41TensorRef{
			plain(prefix+".attn_norm.weight", V41ClassNorm, "F32", layer, c.HiddenSize),
			plain(prefix+".ffn_norm.weight", V41ClassNorm, "F32", layer, c.HiddenSize),
			dense(prefix+".attn.wq_a.weight", V41ClassAttentionDense, layer, c.QLoraRank, c.HiddenSize),
			dense(prefix+".attn.wq_b.weight", V41ClassAttentionDense, layer, qHeadDim, c.QLoraRank),
			dense(prefix+".attn.wkv.weight", V41ClassAttentionDense, layer, v41KVLoraRank, c.HiddenSize),
			dense(prefix+".attn.wo_a.weight", V41ClassAttentionDense, layer, oDim, c.HiddenSize),
			dense(prefix+".attn.wo_b.weight", V41ClassAttentionDense, layer, c.HiddenSize, oDim),
			dense(prefix+".ffn.gate.weight", V41ClassRouter, layer, c.NumExperts, c.HiddenSize),
			plain(prefix+".ffn.gate.e_score_correction_bias", V41ClassRouterBias, "F32", layer, c.NumExperts),
			dense(prefix+".ffn.shared_experts.w1.weight", V41ClassSharedExpert, layer, c.MoEIntermediateSize, c.HiddenSize),
			dense(prefix+".ffn.shared_experts.w3.weight", V41ClassSharedExpert, layer, c.MoEIntermediateSize, c.HiddenSize),
			dense(prefix+".ffn.shared_experts.w2.weight", V41ClassSharedExpert, layer, c.HiddenSize, c.MoEIntermediateSize),
		} {
			if err := add(ref); err != nil {
				return nil, err
			}
		}
		for expert := 0; expert < c.NumExperts; expert++ {
			stem := prefix + ".ffn.experts." + strconv.Itoa(expert)
			for _, ref := range []V41TensorRef{
				packedExpert(stem+".w1", layer, c.MoEIntermediateSize, c.HiddenSize),
				packedExpert(stem+".w3", layer, c.MoEIntermediateSize, c.HiddenSize),
				packedExpert(stem+".w2", layer, c.HiddenSize, c.MoEIntermediateSize),
			} {
				if err := add(ref); err != nil {
					return nil, err
				}
			}
		}
	}
	if err := add(plain("model.norm.weight", V41ClassNorm, "F32", 0, c.HiddenSize)); err != nil {
		return nil, err
	}
	// V4.1 sets tie_word_embeddings=false, so the head is a distinct FP8 tensor.
	if err := add(dense("lm_head.weight", V41ClassHead, 0, c.VocabSize, c.HiddenSize)); err != nil {
		return nil, err
	}

	sort.SliceStable(inv.order, func(i, j int) bool {
		a, b := inv.order[i], inv.order[j]
		if a.Layer != b.Layer {
			return a.Layer < b.Layer
		}
		if a.Class != b.Class {
			return a.Class < b.Class
		}
		return a.Name < b.Name
	})
	inv.names = make([]string, 0, len(inv.order))
	for _, ref := range inv.order {
		inv.names = append(inv.names, ref.Name)
	}
	return inv, nil
}

// packedExpert builds a routed-expert MXFP4 reference for one projection. name
// is the weight stem without a dtype suffix (e.g.
// "layers.0.ffn.experts.0.w1"); o/k are the UNPACKED [O,K] logical shape. The
// on-disk weight halves K (two E2M1 nibbles per byte) and the F8_E8M0 scale
// sibling is [O, K/32], one E8M0 byte per 32 unpacked K values.
func packedExpert(name string, layer, o, k int) V41TensorRef {
	return V41TensorRef{
		Name:         name + ".weight",
		Class:        V41ClassRoutedExpert,
		Layer:        layer,
		Dtype:        "I8",
		Shape:        []int{o, k / 2},
		LogicalShape: []int{o, k},
		PackedWeight: true,
		ScaleName:    name + ".scale",
		ScaleDtype:   "F8_E8M0",
		ScaleShape:   []int{o, k / v41FP8BlockDim},
	}
}
