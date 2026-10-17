package model

import (
	"fmt"
	"math"
	"strings"
)

// quant_row_normalized.go — native K-quant residency for the Qwen3.5-family hybrid projections
// whose GGUF-to-model normalization is a pure OUTPUT-ROW permutation (fak#13567).
//
// ResidentQ4KEligible keeps every transformed tensor out of the raw store because storing a
// transformed tensor's GGUF bytes unchanged would feed mis-laid-out rows to the forward. But a
// K-quant row is independently quantized (each row is a run of whole 256-weight super-blocks),
// so a transform that only reorders OUTPUT rows can be applied to the raw bytes losslessly: the
// loader moves whole rows into canonical order and the dequantized result is bit-identical to
// dequantize-then-normalize. That keeps these projections at their native Q4_K/Q6_K width
// instead of requantizing them to Q8 (~1.125 B/weight against 0.5625 / 0.8203 B/weight).
//
// Column (input-dimension) transforms, e.g. linear_attn.out_proj, are NOT row-only and stay on
// the dequant -> normalize -> Q8 path.

// qwen35RowNormalizedSuffixes are the resolved projection names whose qwen35 normalization in
// ggufload.normalizeCanonicalTensorData only permutes output rows:
//   - linear_attn.in_proj_qkv: the V block's interleaved value-head rows are regrouped;
//   - linear_attn.in_proj_z: interleaved value-head rows (span = value head dim);
//   - linear_attn.in_proj_a / in_proj_b: interleaved value-head rows (span 1);
//   - self_attn.q_proj (gated or plain rotary) and self_attn.k_proj: the rotary unpermute.
var qwen35RowNormalizedSuffixes = [...]string{
	".linear_attn.in_proj_qkv.weight",
	".linear_attn.in_proj_z.weight",
	".linear_attn.in_proj_a.weight",
	".linear_attn.in_proj_b.weight",
	".self_attn.q_proj.weight",
	".self_attn.k_proj.weight",
}

// Qwen35RowNormalizedProjection reports whether canon (a canonical name, resolved here through
// the qwen35 source chain, so an already-resolved name is accepted too) is a Qwen3.5-family
// hybrid target-layer matmul weight whose GGUF normalization is a pure output-row permutation,
// and therefore may be held at native K-quant width after the loader reorders whole raw rows.
// MTP weights are excluded: the loader owns them through its dedicated MTP materializer.
func Qwen35RowNormalizedProjection(cfg Config, canon string) bool {
	if !cfg.IsQwen35Hybrid() {
		return false
	}
	name, keep := quantSourceTensorName(cfg, canon)
	if !keep || !isQuantWeight(name) || strings.HasPrefix(name, "mtp.") {
		return false
	}
	for _, suffix := range qwen35RowNormalizedSuffixes {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// Qwen38ExactMetalQ8Runtime reports whether cfg is the exact 64-layer Qwen3.8 hybrid whose
// no-copy Metal runtime (qwen38MetalQ8RuntimeNames: the fused decode mixer/block, resident GDN
// decode, and the exact-Q8 alias band) REQUIRES every linear_attn projection and full-attention
// q/k projection in the Q8 store. Native row residency must stay off for that topology.
func Qwen38ExactMetalQ8Runtime(cfg Config) bool {
	_, err := qwen38MetalQ8RuntimeNames(cfg)
	return err == nil && cfg.NumLayers == 64
}

// Qwen35MetalQ8DecodeRuntime reports whether cfg can take the resident Metal decode runtime,
// which serves every linear_attn and full-attention q/k projection from the Q8 store.
func Qwen35MetalQ8DecodeRuntime(cfg Config) bool {
	_, err := qwen38MetalQ8RuntimeNames(cfg)
	return err == nil
}

// AddCanonicalRowNormalizedQ4K stores a Qwen3.5-family row-permuted projection whose raw Q4_K
// rows the loader has ALREADY moved into canonical model order (fak#13567). Unlike
// AddResidentQ4K, whose callers gate on ResidentQ4KEligible (which rejects these names because
// their GGUF row order is not the model row order), this narrow entry point exists only for
// bytes that are canonical: it accepts nothing but Qwen35RowNormalizedProjection names and
// refuses the exact Qwen3.8 runtime whose Metal decode requires these weights as Q8.
func (b *QuantBuilder) AddCanonicalRowNormalizedQ4K(canon string, shape []int, raw []byte) error {
	name, err := b.canonicalRowNormalizedTarget(canon, shape, raw, q4kBlockBytes, "Q4_K", false)
	if err != nil {
		return err
	}
	if b.m.q4kw == nil {
		b.m.q4kw = map[string]*q4kTensor{}
	}
	b.m.q4kw[name] = quantizeQ4KFromRaw(raw, shape[0], shape[1])
	return nil
}

// AddCanonicalRowNormalizedQ6K is the Q6_K (kqw) twin of AddCanonicalRowNormalizedQ4K, for a
// q4_k_m checkpoint that ships the GDN QKV projection as Q6_K.
func (b *QuantBuilder) AddCanonicalRowNormalizedQ6K(canon string, shape []int, raw []byte) error {
	name, err := b.canonicalRowNormalizedTarget(canon, shape, raw, kindQ6K.blockBytes(), "Q6_K", false)
	if err != nil {
		return err
	}
	if b.m.kqw == nil {
		b.m.kqw = map[string]*kQuantTensor{}
	}
	b.m.kqw[name] = quantizeKQuantFromRaw(raw, shape[0], shape[1], kindQ6K)
	return nil
}

// AddCanonicalRowNormalizedQ5K is the Q5_K (kqw) twin of AddCanonicalRowNormalizedQ6K, for a
// q4_k_m checkpoint that ships the GDN in_proj_qkv / out_proj as Q5_K. The bytes are moved into a
// page-aligned owner so the Metal Q5_K graph kernels can alias them without a device copy.
func (b *QuantBuilder) AddCanonicalRowNormalizedQ5K(canon string, shape []int, raw []byte) error {
	name, err := b.canonicalRowNormalizedTarget(canon, shape, raw, kindQ5K.blockBytes(), "Q5_K", false)
	if err != nil {
		return err
	}
	if b.m.kqw == nil {
		b.m.kqw = map[string]*kQuantTensor{}
	}
	owned := makePageAlignedResidentBytes(len(raw))
	copy(owned, raw)
	b.m.kqw[name] = quantizeKQuantFromRaw(owned, shape[0], shape[1], kindQ5K)
	return nil
}

// Qwen35FullAttentionQKProjection reports whether canon is a row-normalized full-attention
// self_attn q_proj or k_proj (the subset a raw-i-quant device backend keeps native).
func Qwen35FullAttentionQKProjection(cfg Config, canon string) bool {
	if !Qwen35RowNormalizedProjection(cfg, canon) {
		return false
	}
	name, _ := quantSourceTensorName(cfg, canon)
	return strings.HasSuffix(name, ".self_attn.q_proj.weight") || strings.HasSuffix(name, ".self_attn.k_proj.weight")
}

// Qwen35GDNInProjQKVZProjection reports whether canon is a row-normalized GDN in_proj_qkv or
// in_proj_z (the large in-proj pair a raw-i-quant device backend keeps native; in_proj_a/b stay Q8).
func Qwen35GDNInProjQKVZProjection(cfg Config, canon string) bool {
	if !Qwen35RowNormalizedProjection(cfg, canon) {
		return false
	}
	name, _ := quantSourceTensorName(cfg, canon)
	return strings.HasSuffix(name, ".linear_attn.in_proj_qkv.weight") || strings.HasSuffix(name, ".linear_attn.in_proj_z.weight")
}

// AddCanonicalRowNormalizedDeviceQK stores a canonical-row full-attention q/k or GDN qkv/z projection in its
// native quant (quant is the GGUF type name: Q4_K, Q6_K, Q2_K or a raw i-quant) for a device
// backend with native kernels for it. It admits the exact Qwen3.8 topology: that topology's Q8
// requirement is the Metal runtime's, and the loader routes here only for raw-i-quant backends.
func (b *QuantBuilder) AddCanonicalRowNormalizedDeviceQK(canon string, shape []int, raw []byte, quant string) error {
	if b == nil || b.m == nil {
		return fmt.Errorf("model: nil QuantBuilder")
	}
	if !Qwen35FullAttentionQKProjection(b.m.Cfg, canon) && !Qwen35GDNInProjQKVZProjection(b.m.Cfg, canon) {
		return fmt.Errorf("model: device row-normalized tensor %q is not a Qwen3.5 full-attention q/k or GDN qkv/z projection", canon)
	}
	if quant == "Q4_K" {
		name, err := b.canonicalRowNormalizedTarget(canon, shape, raw, q4kBlockBytes, quant, true)
		if err != nil {
			return err
		}
		if b.m.q4kw == nil {
			b.m.q4kw = map[string]*q4kTensor{}
		}
		b.m.q4kw[name] = quantizeQ4KFromRaw(raw, shape[0], shape[1])
		return nil
	}
	desc, ok := LookupQuantDescriptorByName(quant)
	if !ok || desc.BlockWeights() != qkK {
		return fmt.Errorf("model: device q/k row-normalized tensor %s has unsupported quant %s", canon, quant)
	}
	name, err := b.canonicalRowNormalizedTarget(canon, shape, raw, desc.BlockBytes(), quant, true)
	if err != nil {
		return err
	}
	if b.m.kqw == nil {
		b.m.kqw = map[string]*kQuantTensor{}
	}
	b.m.kqw[name] = quantizeKQuantFromRaw(raw, shape[0], shape[1], desc.Kind())
	return nil
}

func (b *QuantBuilder) canonicalRowNormalizedTarget(canon string, shape []int, raw []byte, blockBytes int, label string, allowExactQwen38 bool) (string, error) {
	if b == nil || b.m == nil {
		return "", fmt.Errorf("model: nil QuantBuilder")
	}
	cfg := b.m.Cfg
	if !Qwen35RowNormalizedProjection(cfg, canon) {
		return "", fmt.Errorf("model: canonical row-normalized %s tensor %q is not a Qwen3.5 row-permuted projection", label, canon)
	}
	if !allowExactQwen38 && Qwen38ExactMetalQ8Runtime(cfg) {
		return "", fmt.Errorf("model: canonical row-normalized %s tensor %q refused: the exact Qwen3.8 Metal runtime requires it in the Q8 store", label, canon)
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%qkK != 0 {
		return "", fmt.Errorf("model: canonical row-normalized %s tensor %s has invalid shape %v", label, canon, shape)
	}
	rowBytes := shape[1] / qkK * blockBytes
	if shape[0] > math.MaxInt/rowBytes || len(raw) != shape[0]*rowBytes {
		return "", fmt.Errorf("model: canonical row-normalized %s tensor %s has %d payload bytes for shape %v", label, canon, len(raw), shape)
	}
	name, ok, err := b.residentQuantTarget(canon, shape)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("model: canonical row-normalized %s tensor %s is not a resident quant target", label, canon)
	}
	if b.m.q4kw[name] != nil || b.m.q8w[name] != nil || b.m.kqw[name] != nil || b.m.q2w[name] != nil {
		return "", fmt.Errorf("model: canonical row-normalized %s tensor %s already has a resident representation", label, name)
	}
	if _, exists := b.m.manifest[name]; exists {
		return "", fmt.Errorf("model: canonical row-normalized %s tensor %s already has a decoded representation", label, name)
	}
	return name, nil
}
