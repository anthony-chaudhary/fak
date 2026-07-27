package ggufload

// Tests for the semantic transform contracts (#4744). These prove, against the
// LIVE loader path (normalizeCanonicalTensorData), that:
//
//  1. every non-identity qwen35 mapping declares a named contract with
//     provenance (and no contract is stale) — the lint that rejects new
//     transform-bearing mappings lacking a semantic contract;
//  2. the ssm_a -> linear_attn.A_log mapping inverts -exp(A_log) back to A_log
//     from a realistic transformed source-domain fixture, and REJECTS a fixture
//     authored directly in the canonical domain (finite-negative validation) —
//     so writing canonical values into a GGUF fixture fails;
//  3. each value-transform witness kills the identity mutation: if the inverse
//     export transform is replaced with identity, the witness comparison fails
//     here, before any model generation;
//  4. the transform identifier is exposed from the tensor name alone, without
//     reading weight payloads (#3251 seam).

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// transformContractProbeConfig is a minimal qwen35 hybrid config exercising
// every transform branch: gated attention (AttnOutputGate), rotary unpermute
// (qwen35 is not an HF-rotary-layout arch), and the GDN value-head interleave
// (LinearNumValueHeads > LinearNumKeyHeads, as every shipped checkpoint).
func transformContractProbeConfig() model.Config {
	return model.Config{
		ModelType:           "qwen35",
		HiddenSize:          8,
		NumHeads:            2,
		NumKVHeads:          1,
		HeadDim:             4,
		AttnOutputGate:      true,
		LayerTypes:          []string{"linear_attention", "full_attention"},
		LinearConvKernelDim: 3,
		LinearKeyHeadDim:    4,
		LinearNumKeyHeads:   2,
		LinearValueHeadDim:  4,
		LinearNumValueHeads: 4,
	}
}

// qwen35TransformProbes enumerates every tensor name a qwen35 hybrid GGUF
// carries, with the element count implied by transformContractProbeConfig.
// A NEW blk.* suffix added to the canonical map must be added here too: the
// registry cross-check below refuses contracts for unprobed externals, so a
// contract cannot be declared without also wiring its behavioral probe.
func qwen35TransformProbes() map[string]int {
	const (
		h     = 8 // HiddenSize
		heads = 2 // NumHeads
		hd    = 4 // HeadDim
		kv    = 1 // NumKVHeads
		nK    = 2 // LinearNumKeyHeads
		kHd   = 4 // LinearKeyHeadDim
		nV    = 4 // LinearNumValueHeads
		vHd   = 4 // LinearValueHeadDim
		conv  = 3 // LinearConvKernelDim
		fused = 2*nK*kHd + nV*vHd
	)
	return map[string]int{
		"output_norm.weight":       h,
		"blk.0.attn_norm.weight":   h,
		"blk.0.ffn_norm.weight":    h,
		"blk.0.attn_q_norm.weight": hd,
		"blk.0.attn_k_norm.weight": hd,
		"blk.0.attn_q.weight":      heads * 2 * hd * h,
		"blk.0.attn_k.weight":      kv * hd * h,
		"blk.0.attn_qkv.weight":    fused * h,
		"blk.0.attn_gate.weight":   nV * vHd * h,
		"blk.0.ssm_a":              nV,
		"blk.0.ssm_alpha.weight":   nV * h,
		"blk.0.ssm_beta.weight":    nV * h,
		"blk.0.ssm_conv1d.weight":  fused * conv,
		"blk.0.ssm_dt.bias":        nV,
		"blk.0.ssm_norm.weight":    vHd,
		"blk.0.ssm_out.weight":     h * nV * vHd,
		"blk.0.ffn_gate.weight":    h * h,
		"blk.0.ffn_up.weight":      h * h,
		"blk.0.ffn_down.weight":    h * h,
	}
}

func transformProbeExternal(name string) string {
	if !strings.HasPrefix(name, "blk.0.") {
		return name
	}
	return strings.TrimPrefix(name, "blk.0.")
}

// transformProbeInput builds a distinguishable input in the tensor's SOURCE
// domain: a strictly increasing sequence, except ssm_a which must be finite
// strictly-negative (-exp of distinct A_log values).
func transformProbeInput(external string, n int) []float32 {
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i + 1)
	}
	if external == "ssm_a" {
		for i := range src {
			src[i] = -float32(math.Exp(float64(i+1) * 0.1))
		}
	}
	return src
}

// llamaRotaryProbeConfig is a minimal llama-family (NORM-rope) config: not a
// qwen35 hybrid (no linear_attention layer types) and not on the NEOX
// allow-list, so the loader applies the q/k rotary unpermute.
func llamaRotaryProbeConfig() model.Config {
	return model.Config{
		ModelType:  "llama",
		HiddenSize: 8,
		NumHeads:   2,
		NumKVHeads: 1,
		HeadDim:    4,
	}
}

// llamaRotaryProbes enumerates the per-layer tensors a llama-family GGUF
// carries through normalizeCanonicalTensorData, with the element counts implied
// by llamaRotaryProbeConfig.
func llamaRotaryProbes() map[string]int {
	const (
		h     = 8 // HiddenSize
		heads = 2 // NumHeads
		hd    = 4 // HeadDim
		kv    = 1 // NumKVHeads
	)
	return map[string]int{
		"output_norm.weight":       h,
		"blk.0.attn_norm.weight":   h,
		"blk.0.ffn_norm.weight":    h,
		"blk.0.attn_q.weight":      heads * hd * h,
		"blk.0.attn_k.weight":      kv * hd * h,
		"blk.0.attn_v.weight":      kv * hd * h,
		"blk.0.attn_output.weight": h * heads * hd,
		"blk.0.ffn_gate.weight":    h * h,
		"blk.0.ffn_up.weight":      h * h,
		"blk.0.ffn_down.weight":    h * h,
	}
}

// TestNonIdentityMappingsDeclareTransformContracts is the contract lint, run
// over every architecture family the loader transforms: a mapping the loader
// transforms without a declared semantic contract fails, and so does a stale
// contract for a mapping the loader no longer transforms. A NEW transform-
// bearing family must be added here, or its mappings fail the lint unchecked.
func TestNonIdentityMappingsDeclareTransformContracts(t *testing.T) {
	for _, fam := range []struct {
		arch   string
		cfg    model.Config
		probes map[string]int
	}{
		{"qwen35", transformContractProbeConfig(), qwen35TransformProbes()},
		{"llama", llamaRotaryProbeConfig(), llamaRotaryProbes()},
	} {
		t.Run(fam.arch, func(t *testing.T) {
			assertContractsCoverLoader(t, fam.arch, fam.cfg, fam.probes)
		})
	}
}

// assertContractsCoverLoader probes every tensor of one architecture family
// through the LIVE loader path and cross-checks it against the declared
// registry in both directions.
func assertContractsCoverLoader(t *testing.T, arch string, cfg model.Config, probes map[string]int) {
	t.Helper()
	for name, n := range probes {
		external := transformProbeExternal(name)
		canonical, ok := CanonicalTensorNameArch(name, arch)
		if !ok {
			t.Fatalf("probe %s: no canonical mapping", name)
		}
		src := transformProbeInput(external, n)
		in := append([]float32(nil), src...)
		out, err := normalizeCanonicalTensorData(canonical, in, cfg)
		if err != nil {
			t.Fatalf("probe %s (%s): normalize: %v", name, canonical, err)
		}
		changed := len(out) != len(src)
		if !changed {
			for i := range out {
				if out[i] != src[i] {
					changed = true
					break
				}
			}
		}
		contract, has := TransformContractForExternalArch(external, arch)
		if changed && !has {
			t.Errorf("arch %s: mapping %s -> %s transforms data but lacks a semantic transform contract (#4744): declare it in the registry TensorTransformContractsForArch(%q) returns", arch, name, canonical, arch)
			continue
		}
		if !changed && has {
			t.Errorf("stale contract: %s declares transform %q but the loader maps %s identically", external, contract.Transform, canonical)
			continue
		}
		if !has {
			continue
		}
		if contract.Transform == "" || contract.Transform == "identity" {
			t.Errorf("contract %s: non-identity mapping must carry a named transform, got %q", external, contract.Transform)
		}
		if contract.Provenance == "" {
			t.Errorf("contract %s: missing provenance", external)
		}
		if contract.SourceDomain == "" || contract.CanonicalDomain == "" {
			t.Errorf("contract %s: missing source/canonical domain", external)
		}
		wantCanonical := canonical
		wantCanonical = strings.TrimPrefix(wantCanonical, "model.layers.0.")
		if contract.Canonical != wantCanonical {
			t.Errorf("contract %s: canonical %q does not match live mapping %q", external, contract.Canonical, wantCanonical)
		}
	}
	// Reverse direction: every declared contract must be behaviorally probed
	// above, so a contract cannot bypass the lint by naming an unknown tensor.
	probed := map[string]bool{}
	for name := range probes {
		probed[transformProbeExternal(name)] = true
	}
	for _, c := range TensorTransformContractsForArch(arch) {
		if !probed[c.External] {
			t.Errorf("arch %s: contract %s -> %s declares an external tensor with no behavioral probe; add it to the family's probe map", arch, c.External, c.Canonical)
		}
	}
}

// TestNEOXLayoutArchesDeclareNoTransformContracts pins the third branch of the
// arch-keyed registry: architectures exported already in the HF rotate_half
// layout are consumed as stored, so declaring a transform for them would be a
// stale contract. This is the case that makes the registry arch-keyed rather
// than name-keyed — "attn_q.weight" is transform-bearing under llama and
// qwen35, and identity here.
func TestNEOXLayoutArchesDeclareNoTransformContracts(t *testing.T) {
	cfg := llamaRotaryProbeConfig()
	for _, arch := range []string{"qwen3", "qwen2", "gemma3", "phi3", "gptoss"} {
		if got := TensorTransformContractsForArch(arch); len(got) != 0 {
			t.Errorf("arch %s stores the HF rotary layout but declares %d transform contracts", arch, len(got))
		}
		cfg.ModelType = arch
		n := cfg.NumHeads * cfg.HeadDim * cfg.HiddenSize
		src := transformProbeInput("attn_q.weight", n)
		out, err := normalizeCanonicalTensorData("model.layers.0.self_attn.q_proj.weight", append([]float32(nil), src...), cfg)
		if err != nil {
			t.Fatalf("arch %s: normalize q_proj: %v", arch, err)
		}
		for i := range out {
			if out[i] != src[i] {
				t.Fatalf("arch %s: loader transformed q_proj at [%d] (%g -> %g) but declares no contract", arch, i, src[i], out[i])
			}
		}
	}
}

// TestQwen35SSMADecayContractInvertsAndValidates proves the acceptance row
// "ssm_a -> linear_attn.A_log is declared as -exp(A_log) -> A_log, with
// finite-negative source validation" against the live loader path.
func TestQwen35SSMADecayContractInvertsAndValidates(t *testing.T) {
	cfg := transformContractProbeConfig()
	contract, ok := Qwen35TransformContractForExternal("ssm_a")
	if !ok {
		t.Fatalf("no semantic transform contract declared for ssm_a (#4744)")
	}
	if !strings.Contains(contract.Transform, TransformInvertNegExpDecay) {
		t.Fatalf("ssm_a contract transform = %q, want it to include %q", contract.Transform, TransformInvertNegExpDecay)
	}
	if !contract.RejectsCanonicalDomain {
		t.Fatalf("ssm_a contract must declare RejectsCanonicalDomain: the loader validates finite-negative -exp(A_log) sources")
	}
	if !contract.Invertible {
		t.Fatalf("ssm_a contract: -exp(A_log) -> A_log is mathematically invertible; declare it")
	}
	if !strings.Contains(contract.SourceDomain, "-exp") {
		t.Fatalf("ssm_a contract source domain %q must describe the -exp(A_log) export domain", contract.SourceDomain)
	}

	// Realistic transformed source-domain fixture: canonical A_log values,
	// exported the way convert_hf_to_gguf.py does (-exp, value heads
	// interleaved repeat-major: source head r*nK+k holds canonical head
	// k*ratio+r).
	aLog := []float32{0.1, 0.4, 0.7, 1.0}
	const nK, nV = 2, 4
	ratio := nV / nK
	src := make([]float32, nV)
	for k := 0; k < nK; k++ {
		for r := 0; r < ratio; r++ {
			src[r*nK+k] = -float32(math.Exp(float64(aLog[k*ratio+r])))
		}
	}
	out, err := normalizeCanonicalTensorData("model.layers.0.linear_attn.A_log", append([]float32(nil), src...), cfg)
	if err != nil {
		t.Fatalf("normalize transformed source fixture: %v", err)
	}
	for i, want := range aLog {
		if math.Abs(float64(out[i]-want)) > 1e-5 {
			// An identity mutation of the inverse transform lands here: the
			// output stays a negative decay coefficient instead of A_log.
			t.Fatalf("A_log[%d] = %g, want %g: inverse export transform not applied (identity mutation?)", i, out[i], want)
		}
	}
	// Invertibility: re-exporting the canonical values (re-interleaving and
	// re-applying -exp) reproduces the source fixture.
	for i := range out {
		back := -float32(math.Exp(float64(out[i])))
		srcIdx := (i%ratio)*nK + i/ratio
		if math.Abs(float64(back-src[srcIdx])) > 1e-5*math.Abs(float64(src[srcIdx])) {
			t.Fatalf("round trip [%d]: -exp(%g) = %g, want %g", i, out[i], back, src[srcIdx])
		}
	}

	// A fixture authored directly in the CANONICAL domain (A_log written into
	// the GGUF, as a naive fixture would) must be REFUSED: A_log values are
	// non-negative here while the source domain is strictly negative.
	if _, err := normalizeCanonicalTensorData("model.layers.0.linear_attn.A_log", []float32{0, 0.25, 0.5, 0.75}, cfg); err == nil {
		t.Fatalf("canonical-domain fixture (raw A_log values) was accepted; want finite-negative source validation to refuse it")
	} else if !strings.Contains(err.Error(), model.NegativeDecayDomain) {
		// Assert the SHARED domain constant rather than a paraphrase of it:
		// the refusal, the contract, and the provenance artifact are supposed to
		// quote one string (#4746), and a substring match on hand-written prose
		// would keep passing while they silently drifted into three.
		t.Fatalf("canonical-domain fixture refused with %q, want the %q source-domain error", err, model.NegativeDecayDomain)
	}
	// Non-finite sources are refused too.
	if _, err := normalizeCanonicalTensorData("model.layers.0.linear_attn.A_log", []float32{float32(math.NaN()), -1, -1, -1}, cfg); err == nil {
		t.Fatalf("NaN source accepted; want finite-negative source validation to refuse it")
	}
}

// TestQwen35ValueTransformWitnessesKillIdentityMutation applies each
// contract's value witness through the LIVE loader path: a tensor filled with
// SampleSource must come out as SampleCanonical everywhere. Layout
// deinterleaves are constant-invariant, so the witness isolates the VALUE
// transform: replacing it with identity fails here, before model generation.
func TestQwen35ValueTransformWitnessesKillIdentityMutation(t *testing.T) {
	cfg := transformContractProbeConfig()
	probes := qwen35TransformProbes()
	sizeFor := map[string]int{}
	nameFor := map[string]string{}
	for name, n := range probes {
		ext := transformProbeExternal(name)
		sizeFor[ext] = n
		canonical, ok := CanonicalTensorNameArch(name, "qwen35")
		if !ok {
			t.Fatalf("probe %s: no canonical mapping", name)
		}
		nameFor[ext] = canonical
	}
	witnessed := 0
	for _, c := range Qwen35TransformContracts() {
		if !c.HasValueSample {
			continue
		}
		witnessed++
		if c.SampleSource == c.SampleCanonical {
			t.Errorf("contract %s: witness source equals canonical (%g); it cannot kill an identity mutation", c.External, c.SampleSource)
			continue
		}
		n, ok := sizeFor[c.External]
		if !ok {
			t.Errorf("contract %s: no probe size", c.External)
			continue
		}
		src := make([]float32, n)
		for i := range src {
			src[i] = c.SampleSource
		}
		out, err := normalizeCanonicalTensorData(nameFor[c.External], src, cfg)
		if err != nil {
			t.Errorf("contract %s: witness normalize: %v", c.External, err)
			continue
		}
		for i := range out {
			if math.Abs(float64(out[i]-c.SampleCanonical)) > 1e-5 {
				t.Errorf("contract %s: witness value[%d] = %g, want %g (transform %s not applied — identity mutation?)", c.External, i, out[i], c.SampleCanonical, c.Transform)
				break
			}
		}
	}
	if witnessed == 0 {
		t.Fatalf("no contract carries a value witness; the identity mutation would go undetected")
	}
}

// TestTransformIDExposedWithoutReadingPayloads proves the #3251 seam: the
// transform identifier is derived from the tensor NAME alone.
func TestTransformIDExposedWithoutReadingPayloads(t *testing.T) {
	id, ok := TransformIDForGGUFTensor("blk.17.ssm_a", "qwen35")
	if !ok || !strings.Contains(id, TransformInvertNegExpDecay) {
		t.Fatalf("blk.17.ssm_a transform id = %q, %v; want the declared %s contract from the name alone", id, ok, TransformInvertNegExpDecay)
	}
	if id, ok := TransformIDForGGUFTensor("output_norm.weight", "qwen35"); !ok || id != TransformGainMinusOne {
		t.Fatalf("output_norm.weight transform id = %q, %v; want %q", id, ok, TransformGainMinusOne)
	}
	// Identity mappings expose no transform id.
	if id, ok := TransformIDForGGUFTensor("blk.3.ffn_gate.weight", "qwen35"); ok {
		t.Fatalf("blk.3.ffn_gate.weight is an identity mapping but exposes transform id %q", id)
	}
	if id, ok := TransformIDForGGUFTensor("blk.3.ssm_norm.weight", "qwen35"); ok {
		t.Fatalf("blk.3.ssm_norm.weight is shape-validated identity but exposes transform id %q", id)
	}

	// The SAME tensor name resolves to a different transform per architecture,
	// so the manifest must read general.architecture from the header alongside
	// the name — still no weight payload.
	perArch := map[string]string{
		"qwen35": TransformStackedQRotaryUnpermute,
		"qwen36": TransformStackedQRotaryUnpermute, // canonicalized onto qwen35
		"llama":  TransformRotaryUnpermute,
	}
	for arch, want := range perArch {
		if id, ok := TransformIDForGGUFTensor("blk.0.attn_q.weight", arch); !ok || id != want {
			t.Errorf("arch %s: blk.0.attn_q.weight transform id = %q, %v; want %q", arch, id, ok, want)
		}
	}
	if id, ok := TransformIDForGGUFTensor("blk.0.attn_q.weight", "qwen3"); ok {
		t.Errorf("arch qwen3 stores the HF rotary layout; blk.0.attn_q.weight must expose no transform id, got %q", id)
	}
	if id, ok := TransformIDForGGUFTensor("blk.0.attn_k.weight", "llama"); !ok || id != TransformRotaryUnpermute {
		t.Errorf("arch llama: blk.0.attn_k.weight transform id = %q, %v; want %q", id, ok, TransformRotaryUnpermute)
	}
}

// writeTransformManifestGGUF builds a GGUF containing a header and a tensor
// directory and NOTHING ELSE — the file ends where the tensor data blob would
// begin. Any code path that reached for a weight payload would read past EOF.
func writeTransformManifestGGUF(arch string) []byte {
	var b bytes.Buffer
	writeMinimalHeader(&b, 4, 2)
	writeKVString(&b, "general.architecture", arch)
	writeKVUint32(&b, "general.alignment", 32)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{8, 4}, TensorF32, 0)
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{8, 8}, TensorF32, 64)
	writeTensorInfoForTest(&b, "blk.0.attn_k.weight", []uint64{8, 4}, TensorF32, 128)
	writeTensorInfoForTest(&b, "blk.0.ffn_gate.weight", []uint64{8, 8}, TensorF32, 192)
	return b.Bytes()
}

// TestTransformManifestFromHeaderOnlyGGUF is the payload-free witness for the
// #3251 manifest seam, made non-forgeable: the fixture GGUF carries no tensor
// data blob at all, so resolving a transform identifier from it PROVES no weight
// was consulted — a manifest that needed a payload could not answer here.
//
// It also pins the property that makes the seam worth having: the identical
// tensor directory resolves to different transforms under different
// architectures, which is exactly the semantic distinction shape and dtype
// cannot carry (both files below are byte-identical apart from the arch string).
func TestTransformManifestFromHeaderOnlyGGUF(t *testing.T) {
	for _, tc := range []struct {
		arch string
		want map[string]string
	}{
		{"qwen35", map[string]string{
			"blk.0.attn_q.weight": TransformStackedQRotaryUnpermute,
			"blk.0.attn_k.weight": TransformRotaryUnpermute,
		}},
		{"llama", map[string]string{
			"blk.0.attn_q.weight": TransformRotaryUnpermute,
			"blk.0.attn_k.weight": TransformRotaryUnpermute,
		}},
		// A NEOX-layout arch is consumed exactly as stored: no transform at all.
		{"qwen3", map[string]string{}},
	} {
		t.Run(tc.arch, func(t *testing.T) {
			raw := writeTransformManifestGGUF(tc.arch)
			f, err := Read(bytes.NewReader(raw))
			if err != nil {
				t.Fatalf("Read header-only gguf: %v", err)
			}
			// The fixture really is payload-free: the header parse consumed the
			// whole file, so the tensor blob is not present to be read.
			if f.TensorDataOffset < int64(len(raw)) {
				t.Fatalf("fixture is not payload-free: data offset %d < file size %d", f.TensorDataOffset, len(raw))
			}

			got := f.TensorTransformIDs()
			if len(got) != len(tc.want) {
				t.Fatalf("arch %s: transform manifest = %v, want %v", tc.arch, got, tc.want)
			}
			for name, want := range tc.want {
				if got[name] != want {
					t.Errorf("arch %s: %s transform = %q, want %q", tc.arch, name, got[name], want)
				}
			}
			// Identity tensors must never appear in the manifest.
			for _, name := range []string{"token_embd.weight", "blk.0.ffn_gate.weight"} {
				if id, ok := f.TensorTransformID(name); ok {
					t.Errorf("arch %s: identity tensor %s exposed transform %q", tc.arch, name, id)
				}
			}

			// The weight-free metadata export carries the same identifiers, so a
			// manifest consumer reads them without a bespoke call.
			exp := f.ExportMetadata()
			if len(exp.Tensors) != 4 {
				t.Fatalf("arch %s: export tensor count = %d, want 4", tc.arch, len(exp.Tensors))
			}
			for _, te := range exp.Tensors {
				if want := tc.want[te.Name]; te.Transform != want {
					t.Errorf("arch %s: export %s transform = %q, want %q", tc.arch, te.Name, te.Transform, want)
				}
			}
		})
	}
}
