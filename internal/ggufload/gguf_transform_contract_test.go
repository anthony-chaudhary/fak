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

// TestQwen35NonIdentityMappingsDeclareTransformContracts is the contract lint:
// a mapping the loader transforms without a declared semantic contract fails,
// and so does a stale contract for a mapping the loader no longer transforms.
func TestQwen35NonIdentityMappingsDeclareTransformContracts(t *testing.T) {
	cfg := transformContractProbeConfig()
	probes := qwen35TransformProbes()
	for name, n := range probes {
		external := transformProbeExternal(name)
		canonical, ok := CanonicalTensorNameArch(name, "qwen35")
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
		contract, has := Qwen35TransformContractForExternal(external)
		if changed && !has {
			t.Errorf("mapping %s -> %s transforms data but lacks a semantic transform contract (#4744): declare it in Qwen35TransformContracts", name, canonical)
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
	for _, c := range Qwen35TransformContracts() {
		if !probed[c.External] {
			t.Errorf("contract %s -> %s declares an external tensor with no behavioral probe; add it to qwen35TransformProbes", c.External, c.Canonical)
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
	} else if !strings.Contains(err.Error(), "finite negative") {
		t.Fatalf("canonical-domain fixture refused with %q, want the finite-negative source-domain error", err)
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
	id, ok := TransformIDForGGUFTensor("blk.17.ssm_a")
	if !ok || !strings.Contains(id, TransformInvertNegExpDecay) {
		t.Fatalf("blk.17.ssm_a transform id = %q, %v; want the declared %s contract from the name alone", id, ok, TransformInvertNegExpDecay)
	}
	if id, ok := TransformIDForGGUFTensor("output_norm.weight"); !ok || id != TransformGainMinusOne {
		t.Fatalf("output_norm.weight transform id = %q, %v; want %q", id, ok, TransformGainMinusOne)
	}
	// Identity mappings expose no transform id.
	if id, ok := TransformIDForGGUFTensor("blk.3.ffn_gate.weight"); ok {
		t.Fatalf("blk.3.ffn_gate.weight is an identity mapping but exposes transform id %q", id)
	}
	if id, ok := TransformIDForGGUFTensor("blk.3.ssm_norm.weight"); ok {
		t.Fatalf("blk.3.ssm_norm.weight is shape-validated identity but exposes transform id %q", id)
	}
}
