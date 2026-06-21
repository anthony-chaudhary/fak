package model

import "fmt"

// Phi-3/3.5/4 fused-tensor split (Stage 6, MODEL-ARCH-SEAM.md §4).
//
// Phi ships a SINGLE fused attention projection `self_attn.qkv_proj.weight`
// (rows = q ++ k ++ v) and a SINGLE fused MLP input projection
// `mlp.gate_up_proj.weight` (rows = gate ++ up) instead of the separate
// q/k/v and gate/up tensors the Llama-shaped forward pass reads. Every
// forward-pass site (forward.go, kv.go, prefill_batch.go, batch.go,
// quant_forward.go, profile.go) calls m.tensor("…q_proj.weight") etc., so a
// Phi checkpoint cannot load unless the fused tensor is cut into the
// canonical components at LOAD time.
//
// The cut is a pure CONTIGUOUS BYTE-RANGE slice on axis-0. A weight is
// row-major [out, in], so its `out` rows are contiguous in memory and a
// component is an unbroken byte range [rowStart*in*4, rowEnd*in*4). The
// component manifest entries point at sub-ranges of the SAME raw blob — zero
// arithmetic, zero copy — so the split tensors are byte-identical to a
// checkpoint that stored q/k/v/gate/up separately. The forward pass is
// untouched and the f32 bit-exact rungs (R2/R14) carry no new claim.
//
// A non-Phi (already-unfused) checkpoint has no fused tensor, so this is a
// no-op there and the Llama path stays bit-identical.

const (
	suffixQKVProj    = "self_attn.qkv_proj.weight"
	suffixGateUpProj = "mlp.gate_up_proj.weight"

	suffixQProj    = "self_attn.q_proj.weight"
	suffixKProj    = "self_attn.k_proj.weight"
	suffixVProj    = "self_attn.v_proj.weight"
	suffixGateProj = "mlp.gate_proj.weight"
	suffixUpProj   = "mlp.up_proj.weight"
)

// fusedPart names one component carved out of a fused tensor: its output-name
// suffix and how many axis-0 rows it owns.
type fusedPart struct {
	suffix string
	rows   int
}

// splitFusedProjections rewrites the manifest in place: wherever a layer
// carries a fused qkv_proj / gate_up_proj tensor, it is replaced by its
// component tensors, each a contiguous axis-0 byte-range view into the same
// raw blob. Returns an error if a fused tensor's row count does not equal the
// sum of the component row counts the config implies (a corrupt / mismatched
// checkpoint) — fail closed rather than emit a silently-wrong slice.
//
// Pure manifest surgery: `raw` is never touched, so the component entries are
// bit-identical reinterpretations of the fused bytes.
func splitFusedProjections(cfg Config, man map[string]tensorMeta) error {
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	I := cfg.IntermediateSize

	qkvParts := []fusedPart{
		{suffixQProj, nH * hd},
		{suffixKProj, nKV * hd},
		{suffixVProj, nKV * hd},
	}
	gateUpParts := []fusedPart{
		{suffixGateProj, I},
		{suffixUpProj, I},
	}

	for l := 0; l < cfg.NumLayers; l++ {
		pre := layerPrefix(l)
		if err := splitOneFused(man, pre, suffixQKVProj, qkvParts); err != nil {
			return err
		}
		if err := splitOneFused(man, pre, suffixGateUpProj, gateUpParts); err != nil {
			return err
		}
	}
	return nil
}

// splitOneFused carves the fused tensor `pre+fusedSuffix` (if present) into the
// given parts and deletes the fused entry. The parts are laid out in order
// along axis-0; each part's byte range is [cursor, cursor+rows*in*4).
func splitOneFused(man map[string]tensorMeta, pre, fusedSuffix string, parts []fusedPart) error {
	fusedName := pre + fusedSuffix
	meta, ok := man[fusedName]
	if !ok {
		return nil // not a fused checkpoint for this tensor — no-op
	}
	if len(meta.Shape) != 2 {
		return fmt.Errorf("model: fused tensor %s has shape %v, want 2-D [out,in]", fusedName, meta.Shape)
	}
	out, in := meta.Shape[0], meta.Shape[1]

	wantRows := 0
	for _, p := range parts {
		wantRows += p.rows
	}
	if out != wantRows {
		return fmt.Errorf("model: fused tensor %s has %d rows, config implies %d (%s)",
			fusedName, out, wantRows, partsDesc(parts))
	}
	// The fused blob must be exactly out*in f32 = out*in*4 bytes; guard against a
	// dtype/shape mismatch that would make the byte cut land off the row boundary.
	if meta.Nbytes != out*in*4 {
		return fmt.Errorf("model: fused tensor %s has %d bytes, shape [%d,%d] f32 implies %d",
			fusedName, meta.Nbytes, out, in, out*in*4)
	}

	rowBytes := in * 4
	cursor := meta.Offset
	for _, p := range parts {
		name := pre + p.suffix
		if _, exists := man[name]; exists {
			return fmt.Errorf("model: cannot split %s: component %s already present", fusedName, name)
		}
		nbytes := p.rows * rowBytes
		man[name] = tensorMeta{
			Dtype:  meta.Dtype,
			Shape:  []int{p.rows, in},
			Offset: cursor,
			Nbytes: nbytes,
		}
		cursor += nbytes
	}
	delete(man, fusedName)
	return nil
}

func partsDesc(parts []fusedPart) string {
	s := ""
	for i, p := range parts {
		if i > 0 {
			s += "+"
		}
		s += fmt.Sprintf("%d", p.rows)
	}
	return s
}
