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
// This file declares those contracts for EVERY non-identity mapping
// normalizeCanonicalTensorData performs (gguf_tensor_canonical.go), keyed by
// architecture because the same external name carries different semantics per
// family:
//
//   - the Qwen3.5/Qwen3.6 (qwen35 hybrid) family, which carries the largest set
//     of non-identity mappings — the SSM/GDN value-head deinterleaves, the
//     NormGain1p residual-gain shift, and the ssm_a decay inversion;
//   - the llama-family NORM-rope architectures, whose q/k projections are
//     rotary-unpermuted (LlamaRotaryTransformContracts);
//   - the NEOX-layout architectures (qwen3, gemma3, phi3, …), whose mappings are
//     all identity and therefore declare no contract at all.
//
// Each contract records the external and canonical names, the source and
// destination semantic domains, a NAMED transform identifier, provenance for why
// the transform exists, and whether it is lossless/invertible. The transform
// identifier is derivable from the tensor NAME plus the header architecture
// (TransformIDForGGUFTensor), so a shape-first manifest can expose it without
// reading any weight payload.
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
	// qkvValueDeinterleave declares a fused [q;k;v] tensor whose VALUE block alone is
	// deinterleaved while the query and key parts pass through untouched. The fused
	// weight rows and the fused conv1d channels take the SAME transform over the SAME
	// GDN row domains and differ only in what the fused axis is called, so the axis
	// nouns are parameters (`fused` names it in the domain clause, `plain` in the
	// unchanged clause) and every shipped contract string stays byte-identical.
	qkvValueDeinterleave := func(external, canonical, fused, plain string) TensorTransformContract {
		return TensorTransformContract{
			External: external, Canonical: canonical,
			Transform:    TransformQKVValueRowsDeinterleave,
			SourceDomain: "fused [q;k;v] " + fused + " with the value block in " + gdnRowsSource,
			CanonicalDomain: "fused [q;k;v] " + fused + " with the value block in " + gdnRowsCanon +
				"; query/key " + plain + " unchanged",
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
		qkvValueDeinterleave("attn_qkv.weight", "self_attn.qkv_proj.weight", "rows", "rows"),
		deinterleave("attn_gate.weight", "self_attn.q_gate_proj.weight"),
		deinterleave("ssm_alpha.weight", "linear_attn.in_proj_a.weight"),
		deinterleave("ssm_beta.weight", "linear_attn.in_proj_b.weight"),
		qkvValueDeinterleave("ssm_conv1d.weight", "linear_attn.conv1d.weight",
			"conv channels", "channels"),
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

// LlamaRotaryTransformContracts returns the semantic transform contracts for
// the non-qwen35 half of the loader audit (#4744 "then audit other non-identity
// loader mappings"). Outside the qwen35 hybrid path, normalizeCanonicalTensorData
// has exactly two non-identity branches — the q/k rotary unpermute — and it
// applies them to every architecture NOT on the ggufArchStoresHFRotaryLayout
// NEOX allow-list. Those are the mappings declared here.
//
// The same external name carries a DIFFERENT transform per family, which is
// precisely why the registry is arch-keyed: "attn_q.weight" is
// stacked-q-rotary-unpermute under qwen35, rotary-unpermute under the
// llama-family NORM-rope arches, and identity under the NEOX arches (qwen3,
// gemma3, phi3, …), which declare no contract at all.
func LlamaRotaryTransformContracts() []TensorTransformContract {
	const provenance = "convert_hf_to_gguf.py permutes llama-family (NORM-rope) q/k rows " +
		"into the GGML interleaved rotary-pair layout on export; fak's forward applies the HF " +
		"rotate_half convention (forward.go), so the loader unpermutes back via " +
		"unpermuteRotaryTensor (gguf_tensor_canonical.go). NEOX-layout arches are exported " +
		"unpermuted and are excluded by ggufArchStoresHFRotaryLayout — for them this mapping " +
		"is identity and carries no contract"
	rotary := func(external, canonical, rows string) TensorTransformContract {
		return TensorTransformContract{
			External: external, Canonical: canonical,
			Transform:       TransformRotaryUnpermute,
			SourceDomain:    rows + " rows in GGML interleaved rotary-pair order (row j*2+p)",
			CanonicalDomain: rows + " rows in HF rotate_half order (row p*half+j)",
			Provenance:      provenance,
			// A pure row permutation: every value survives bit-exactly and the
			// permutation is its own well-defined inverse.
			Lossless: true, Invertible: true,
		}
	}
	return []TensorTransformContract{
		rotary("attn_q.weight", "self_attn.q_proj.weight", "query"),
		rotary("attn_k.weight", "self_attn.k_proj.weight", "key"),
	}
}

// TensorTransformContractsForArch returns the semantic transform contracts that
// apply to a GGUF of architecture arch. arch is the header's
// general.architecture value (canonicalized), so this stays payload-free.
//
// A nil result means the loader maps every tensor of that architecture
// identically — the NEOX-layout arches, whose q/k weights are consumed exactly
// as stored.
func TensorTransformContractsForArch(arch string) []TensorTransformContract {
	switch canonicalGGUFArch(arch) {
	case "qwen35", "qwen35moe":
		return Qwen35TransformContracts()
	}
	if ggufArchStoresHFRotaryLayout(canonicalGGUFArch(arch)) {
		return nil
	}
	return LlamaRotaryTransformContracts()
}

// Qwen35TransformContractForExternal returns the contract for an external
// tensor suffix (or model-global name), if one is declared.
func Qwen35TransformContractForExternal(external string) (TensorTransformContract, bool) {
	return TransformContractForExternalArch(external, "qwen35")
}

// TransformContractForExternalArch returns the contract declared for an
// external tensor suffix (or model-global name) under architecture arch.
func TransformContractForExternalArch(external, arch string) (TensorTransformContract, bool) {
	for _, c := range TensorTransformContractsForArch(arch) {
		if c.External == external {
			return c, true
		}
	}
	return TensorTransformContract{}, false
}

// ExternalTensorSuffix strips a "blk.<layer>." prefix from a raw GGUF tensor
// name, leaving the per-layer suffix a contract is keyed on. Model-global
// tensors (e.g. "output_norm.weight") are returned unchanged.
func ExternalTensorSuffix(name string) string {
	if !strings.HasPrefix(name, "blk.") {
		return name
	}
	rest := strings.TrimPrefix(name, "blk.")
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return name
	}
	if _, err := strconv.Atoi(rest[:dot]); err != nil {
		return name
	}
	return rest[dot+1:]
}

// TransformIDForGGUFTensor maps a raw GGUF tensor name to its declared semantic
// transform identifier under architecture arch. It consults only the NAME and
// the header architecture — never a weight payload — so the shape-first
// manifest (#3251) can expose the transform id for a tensor from the GGUF
// header alone. ok=false means the mapping is identity (or unknown): no
// semantic transform is declared for it.
func TransformIDForGGUFTensor(name, arch string) (string, bool) {
	c, ok := TransformContractForExternalArch(ExternalTensorSuffix(name), arch)
	if !ok {
		return "", false
	}
	return c.Transform, true
}

// TensorTransformID reports the declared semantic transform identifier for one
// tensor of a PARSED GGUF HEADER. It is the accessor a shape-first manifest
// (#3251) calls: it reads general.architecture out of the already-decoded
// metadata and the tensor's name, and touches no part of the tensor data blob —
// Read/Open stop at the header, so a caller holding only a *File has, by
// construction, read no weights. ok=false means the mapping is identity (or the
// tensor is unknown to the canonicalizer): no semantic transform is declared.
func (f *File) TensorTransformID(name string) (string, bool) {
	arch, _ := f.String("general.architecture")
	return TransformIDForGGUFTensor(name, arch)
}

// TensorTransformIDs returns the declared transform identifier of every tensor
// in the header that carries one, keyed by the raw GGUF tensor name. Identity
// mappings are omitted, so an empty map means this checkpoint's architecture is
// consumed exactly as stored. Like TensorTransformID it is header-only: the
// shape-first manifest can publish "which tensors does this loader reinterpret,
// and under what named transform" for a multi-hundred-GB checkpoint at the cost
// of a header parse.
func (f *File) TensorTransformIDs() map[string]string {
	arch, _ := f.String("general.architecture")
	if len(TensorTransformContractsForArch(arch)) == 0 {
		return nil
	}
	out := make(map[string]string)
	for _, t := range f.Tensors {
		if id, ok := TransformIDForGGUFTensor(t.Name, arch); ok {
			out[t.Name] = id
		}
	}
	return out
}
