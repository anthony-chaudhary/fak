package ggufload

// Semantic transform contracts for canonical tensor mappings (#4744).
//
// Why shape/dtype parity alone cannot establish loader correctness: a GGUF
// tensor and fak's canonical tensor can agree byte-for-byte on shape and dtype
// while carrying DIFFERENT mathematical meanings. The root incident (#4273) was
// exactly this class of defect: GGUF blk.*.ssm_a stores the already-transformed
// negative decay coefficient -exp(A_log), while fak's canonical tensor is the
// pre-transform A_log. The loader preserved every byte, every dimension and the
// dtype — and still assigned the wrong semantics, so the forward applied exp a
// second time. Short prompts stayed plausible, which made the defect nearly
// invisible to output eyeballing. A shape-first manifest (#3251) or an
// independent oracle (#442/#474) can catch layout and end-to-end drift, but only
// an explicit SEMANTIC contract — "this external tensor is in domain X, the
// canonical tensor is in domain Y, and the loader must apply named transform T
// between them, with these validity ranges" — pins the meaning of the bytes.
//
// This file declares those contracts for the Qwen3.5/Qwen3.6 (qwen35 hybrid)
// family, whose GGUF exports carry the largest set of non-identity mappings in
// the loader (gguf_tensor_canonical.go). Each contract records the external and
// canonical names, the source and destination semantic domains, a NAMED
// transform identifier, provenance for why the transform exists, and whether it
// is lossless/invertible. The transform identifier is derivable from the tensor
// NAME alone (TransformIDForGGUFTensor), so a shape-first manifest can expose it
// without reading any weight payload.
//
// The paired test (gguf_transform_contract_test.go) behaviorally probes every
// tensor a qwen35 GGUF carries through the live loader path
// (normalizeCanonicalTensorData) and fails if a mapping transforms data without
// a declared contract here, or if a contract goes stale (declares a transform
// the loader no longer performs). Contracts carrying a value witness also kill
// the identity mutation: replacing the inverse export transform with identity
// fails the witness check at `go test` time, before any model generation.

import (
	"strconv"
	"strings"
)

// Named transform identifiers. A transform id names WHAT the loader does to
// move a tensor from its external (GGUF) domain to fak's canonical domain.
// Composite transforms join ids with "+" in application order.
const (
	// TransformInvertNegExpDecay recovers A_log from the exported negated
	// exponential decay coefficient: A = -exp(A_log)  =>  A_log = log(-A).
	// Source values must be finite and strictly negative.
	TransformInvertNegExpDecay = "invert-neg-exp-decay"
	// TransformGainMinusOne converts a full RMSNorm gain g to the residual
	// gain g-1 consumed by the NormGain1p forward, which computes (1+w)*x̂.
	TransformGainMinusOne = "gain-minus-one"
	// TransformValueHeadDeinterleave reorders GDN value-head blocks from the
	// exporter's repeat-major order (head r*nK+k) to the forward's head-major
	// order (head k*ratio+r).
	TransformValueHeadDeinterleave = "value-head-deinterleave"
	// TransformQKVValueRowsDeinterleave applies the value-head deinterleave to
	// only the value-row block of a fused [q;k;v]-rows tensor, leaving the
	// query/key rows untouched.
	TransformQKVValueRowsDeinterleave = "qkv-value-rows-deinterleave"
	// TransformValueColsDeinterleave applies the value-head deinterleave along
	// the COLUMN axis (out_proj consumes value heads as its input columns).
	TransformValueColsDeinterleave = "value-cols-deinterleave"
	// TransformRotaryUnpermute undoes the GGML interleaved rotary-pair row
	// permutation (row j*2+p) back to the HF rotate_half layout (row p*half+j)
	// that fak's forward applies.
	TransformRotaryUnpermute = "rotary-unpermute"
	// TransformStackedQRotaryUnpermute is the rotary unpermute for the qwen35
	// gated-attention q tensor, which stacks [query;gate] per head; only the
	// query half of each head's stack is rotary-permuted.
	TransformStackedQRotaryUnpermute = "stacked-q-rotary-unpermute"
)

// TensorTransformContract declares the semantic contract for one non-identity
// external->canonical tensor mapping.
type TensorTransformContract struct {
	// External is the GGUF tensor name with any "blk.<layer>." prefix removed
	// (model-global tensors carry their full name, e.g. "output_norm.weight").
	External string
	// Canonical is fak's canonical tensor name with any "model.layers.<n>."
	// prefix removed (model-global: full name, e.g. "model.norm.weight").
	Canonical string
	// Transform is the named transform identifier ("+"-joined if composite).
	Transform string
	// SourceDomain describes the mathematical meaning and validity range of
	// the values as stored in the GGUF.
	SourceDomain string
	// CanonicalDomain describes the meaning and range of the canonical values.
	CanonicalDomain string
	// Provenance records why the transform exists: which exporter or format
	// convention produced the source domain, and what the forward consumes.
	Provenance string
	// Lossless reports whether applying the transform preserves the values
	// bit-exactly (pure layout permutations are lossless; floating-point value
	// transforms generally are not).
	Lossless bool
	// Invertible reports whether the transform is mathematically invertible.
	Invertible bool
	// RejectsCanonicalDomain reports that the loader VALIDATES the source
	// domain and refuses values that are only plausible in the canonical
	// domain (e.g. non-negative "decay" values). When false the two domains
	// overlap numerically and only the value witness can detect a fixture
	// authored in the wrong domain.
	RejectsCanonicalDomain bool
	// HasValueSample marks a per-value transform witness: applying the
	// loader's transform to a tensor filled with SampleSource must yield
	// SampleCanonical in every element. Because SampleSource differs from
	// SampleCanonical, mutating the transform into identity fails the
	// witness. Pure layout transforms carry no value witness (constant fills
	// are layout-invariant); their non-identity is proven by sequence probes.
	HasValueSample  bool
	SampleSource    float32
	SampleCanonical float32
}

// Qwen35TransformContracts returns the semantic transform contracts for every
// non-identity external->canonical tensor mapping of the qwen35 hybrid family.
//
// Registry invariants, enforced by TestQwen35NonIdentityMappingsDeclareTransformContracts:
//   - every mapping the loader transforms non-identically appears here;
//   - every entry here is transformed non-identically by the live loader
//     (no stale contracts);
//   - every entry names its transform and carries provenance and domains.
//
// NOTE (audit finding, #4744): normalizeQwen35LinearTensor returns early when
// LinearNumValueHeads == LinearNumKeyHeads, skipping BOTH the deinterleave and
// the ssm_a decay inversion plus its finite-negative validation. All shipped
// Qwen3.5/Qwen3-Next checkpoints declare nV > nK, so the contracts below hold
// for every real artifact; a future equal-heads export would need that early
// return revisited before these contracts extend to it.
func Qwen35TransformContracts() []TensorTransformContract {
	const (
		gdnProvenance = "convert_hf_to_gguf.py exports Qwen3-Next GDN value heads " +
			"interleaved across key-head groups (repeat-major r*nK+k); fak's gated-delta-net " +
			"forward (qwen35.go) indexes value heads head-major (k*ratio+r), so the loader " +
			"deinterleaves (gguf_tensor_canonical.go)"
		rotaryProvenance = "convert_hf_to_gguf.py permutes llama-family q/k rows into the " +
			"GGML interleaved rotary-pair layout; fak's forward applies the HF rotate_half " +
			"convention (forward.go), so the loader unpermutes back"
		normProvenance = "Qwen3.5 GGUF stores the full RMSNorm gain; fak's qwen35 forward " +
			"is NormGain1p and computes (1+w)*x_hat, so the loader subtracts 1 " +
			"(gguf_tensor_canonical.go)"
		normSource    = "full RMSNorm gain g (typically near 1)"
		normCanonical = "residual RMSNorm gain g-1 consumed by the NormGain1p forward"
		gdnRowsSource = "GDN value-head row blocks in exporter repeat-major order (head r*nK+k)"
		gdnRowsCanon  = "GDN value-head row blocks in forward head-major order (head k*ratio+r)"
	)
	normContract := func(external, canonical string) TensorTransformContract {
		return TensorTransformContract{
			External: external, Canonical: canonical,
			Transform:    TransformGainMinusOne,
			SourceDomain: normSource, CanonicalDomain: normCanonical,
			Provenance: normProvenance,
			Lossless:   false, Invertible: true,
			HasValueSample: true, SampleSource: 1.5, SampleCanonical: 0.5,
		}
	}
	deinterleave := func(external, canonical string) TensorTransformContract {
		return TensorTransformContract{
			External: external, Canonical: canonical,
			Transform:    TransformValueHeadDeinterleave,
			SourceDomain: gdnRowsSource, CanonicalDomain: gdnRowsCanon,
			Provenance: gdnProvenance,
			Lossless:   true, Invertible: true,
		}
	}
	return []TensorTransformContract{
		{
			External: "ssm_a", Canonical: "linear_attn.A_log",
			Transform: TransformValueHeadDeinterleave + "+" + TransformInvertNegExpDecay,
			SourceDomain: "negated exponential decay coefficient -exp(A_log): " +
				"finite and strictly negative",
			CanonicalDomain: "raw gated-delta-net decay parameter A_log (finite real)",
			Provenance: "convert_hf_to_gguf.py Qwen3-Next export stores A = -A_log.float().exp(); " +
				"fak's canonical tensor is the pre-transform A_log so every runtime path computes " +
				"exp(-exp(A_log)*softplus(dt)) exactly once (root incident #4273, fixed 4f302a441); " +
				"the loader inverts via A_log = log(-A) after the value-head deinterleave",
			Lossless: false, Invertible: true,
			RejectsCanonicalDomain: true,
			HasValueSample:         true,
			SampleSource:           -1.6487213, // -exp(0.5)
			SampleCanonical:        0.5,
		},
		normContract("output_norm.weight", "model.norm.weight"),
		normContract("attn_norm.weight", "input_layernorm.weight"),
		normContract("ffn_norm.weight", "post_attention_layernorm.weight"),
		normContract("attn_q_norm.weight", "self_attn.q_norm.weight"),
		normContract("attn_k_norm.weight", "self_attn.k_norm.weight"),
		{
			External: "attn_q.weight", Canonical: "self_attn.q_proj.weight",
			Transform: TransformStackedQRotaryUnpermute,
			SourceDomain: "per-head [query;gate] row stacks with query rows in GGML " +
				"interleaved rotary-pair order (row j*2+p)",
			CanonicalDomain: "per-head [query;gate] row stacks with query rows in HF " +
				"rotate_half order (row p*half+j)",
			Provenance: rotaryProvenance + "; qwen35 gated attention stacks a sigmoid gate " +
				"under each head's query rows, and only the query half is permuted",
			Lossless: true, Invertible: true,
		},
		{
			External: "attn_k.weight", Canonical: "self_attn.k_proj.weight",
			Transform:    TransformRotaryUnpermute,
			SourceDomain: "key rows in GGML interleaved rotary-pair order (row j*2+p)",
			CanonicalDomain: "key rows in HF rotate_half order (row p*half+j) as fak's " +
				"forward rotates them",
			Provenance: rotaryProvenance,
			Lossless:   true, Invertible: true,
		},
		{
			External: "attn_qkv.weight", Canonical: "self_attn.qkv_proj.weight",
			Transform:    TransformQKVValueRowsDeinterleave,
			SourceDomain: "fused [q;k;v] rows with the value block in " + gdnRowsSource,
			CanonicalDomain: "fused [q;k;v] rows with the value block in " + gdnRowsCanon +
				"; query/key rows unchanged",
			Provenance: gdnProvenance,
			Lossless:   true, Invertible: true,
		},
		deinterleave("attn_gate.weight", "self_attn.q_gate_proj.weight"),
		deinterleave("ssm_alpha.weight", "linear_attn.in_proj_a.weight"),
		deinterleave("ssm_beta.weight", "linear_attn.in_proj_b.weight"),
		{
			External: "ssm_conv1d.weight", Canonical: "linear_attn.conv1d.weight",
			Transform:    TransformQKVValueRowsDeinterleave,
			SourceDomain: "fused [q;k;v] conv channels with the value block in " + gdnRowsSource,
			CanonicalDomain: "fused [q;k;v] conv channels with the value block in " + gdnRowsCanon +
				"; query/key channels unchanged",
			Provenance: gdnProvenance,
			Lossless:   true, Invertible: true,
		},
		deinterleave("ssm_dt.bias", "linear_attn.dt_bias"),
		{
			External: "ssm_out.weight", Canonical: "linear_attn.out_proj.weight",
			Transform:    TransformValueColsDeinterleave,
			SourceDomain: "output-projection COLUMNS grouped by value head in exporter repeat-major order",
			CanonicalDomain: "output-projection columns grouped by value head in forward " +
				"head-major order (out_proj consumes value heads as input columns)",
			Provenance: gdnProvenance,
			Lossless:   true, Invertible: true,
		},
	}
}

// Qwen35TransformContractForExternal returns the contract for an external
// tensor suffix (or model-global name), if one is declared.
func Qwen35TransformContractForExternal(external string) (TensorTransformContract, bool) {
	for _, c := range Qwen35TransformContracts() {
		if c.External == external {
			return c, true
		}
	}
	return TensorTransformContract{}, false
}

// TransformIDForGGUFTensor maps a raw GGUF tensor name to its declared
// semantic transform identifier. It consults only the NAME — never a weight
// payload — so the shape-first manifest (#3251) can expose the transform id
// for a tensor from the header alone. ok=false means the mapping is identity
// (or unknown): no semantic transform is declared for it.
func TransformIDForGGUFTensor(name string) (string, bool) {
	external := name
	if strings.HasPrefix(name, "blk.") {
		rest := strings.TrimPrefix(name, "blk.")
		if dot := strings.IndexByte(rest, '.'); dot > 0 {
			if _, err := strconv.Atoi(rest[:dot]); err == nil {
				external = rest[dot+1:]
			}
		}
	}
	c, ok := Qwen35TransformContractForExternal(external)
	if !ok {
		return "", false
	}
	return c.Transform, true
}
