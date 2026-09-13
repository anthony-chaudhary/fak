package model

import "fmt"

// metal_q8_budget.go — the host-independent budget predicate that decides whether the
// Q8-minority projections may be uploaded to the GPU during resident-Q4_K prefill. It is split
// out of metal_q4k_on.go (which is darwin+arm64+cgo only) so the arithmetic is testable on every
// platform; the Metal-aware caller metalQ8UploadAllowed (metal_q4k_on.go) reads the live device
// working-set budget and delegates the decision here.
//
// WHY THIS EXISTS (#1087 regression): unlike the Q4_K upload — a no-copy alias of the resident
// bytes on Apple unified memory — metalgemm.UploadQ8 always COPIES the Q8_0 codes/scales into a
// fresh device buffer, and the CPU q8Tensor is kept alive because decode still reads it via
// qMatRows. So the Q8 GPU copy is purely ADDITIVE. For a 27B q4_k_m model resident at ~23 GiB on a
// 36 GiB Mac, that copy pushes the working set past the jetsam ceiling and the serve is SIGKILLed
// at the first prefill turn. This predicate declines the upload when it would not fit, keeping the
// Q8 minority on the proven CPU qGemm8 path (the pre-#1087 behavior, which serves without OOM).

// metalQ8UploadFraction is the fraction of the device working-set budget the resident weights PLUS
// the projected Q8-minority GPU copy may occupy before the upload is declined. 0.90 leaves headroom
// for the prefill activation panels + KV growth on top of the (temporarily doubled) projection store.
const metalQ8UploadFraction = 0.90

// metalQ8AliasFraction is the headroom fraction for the EXACT Qwen3.8 no-copy alias path ONLY. Unlike
// the additive UploadQ8 copy gated by metalQ8UploadFraction, AliasQ8 adds no new device bytes — it
// exposes the model's already-resident, page-aligned Q8 owners to Metal in place. The residual risk
// is therefore not a doubled projection store but a model whose resident owner already saturates the
// working set and leaves no room for prefill activations + KV. 0.97 keeps that fail-closed headroom
// while admitting the common case (e.g. a 23 GiB owner on a 27 GiB budget = 0.852) that the
// additive-copy 0.90 model wrongly refused. The #1087 additive gate above is deliberately unchanged.
const metalQ8AliasFraction = 0.97

// metalQ6AliasFraction is the headroom fraction for the Q6_K no-copy alias path. It is the SAME
// judgement as metalQ8AliasFraction for the SAME reason: a page-aligned, Go-heap-owned Q6_K payload
// is aliased in place by UploadQ6KGoOwned, so publishing the Q6_K band adds NO new device bytes. The
// additive q6kUploadFits 0.90 model below remains the fallback for a genuinely copied upload; the
// alias is judged only against the activation/KV headroom a resident owner must leave. A 23 GiB
// resident model on a 36 GiB Mac (0.639) admits; an already-saturating owner still declines.
const metalQ6AliasFraction = 0.97

// q8UploadFits is the pure budget predicate: does the already-resident weight footprint plus the
// projected additive Q8 GPU copy fit under metalQ8UploadFraction of the device working-set budget?
// forceEnv is the raw FAK_METAL_Q8_UPLOAD value: "1"/"on"/"true" forces the upload on (a roomy box
// that wants #1087's Metal-Q8 prefill regardless of the estimate), "0"/"off"/"false" forces it off;
// anything else defers to the budget test. deviceTotal <= 0 (unknown device budget) is treated as
// "cannot prove it fits" and declines — the conservative default that avoids the OOM.
func q8UploadFits(residentBytes, q8Bytes, deviceTotal int64, forceEnv string) bool {
	switch forceEnv {
	case "1", "on", "ON", "true", "TRUE":
		return true
	case "0", "off", "OFF", "false", "FALSE":
		return false
	}
	if deviceTotal <= 0 {
		return false // device budget unknown — do not risk the additive Q8 copy
	}
	projected := residentBytes + q8Bytes
	return float64(projected) <= metalQ8UploadFraction*float64(deviceTotal)
}

// MetalQ8ResidencyUnavailableError is the fail-closed reason an exact promised-Metal model could
// not publish its complete no-copy Q8 band. Ordinary execution may stay CPU-safe; profile evidence
// must refuse rather than relabel that work as Metal.
type MetalQ8ResidencyUnavailableError struct{ Reason string }

func (e *MetalQ8ResidencyUnavailableError) Error() string {
	return "model: exact Metal Q8 residency unavailable: " + e.Reason
}

// q8AliasFits decides whether the EXACT no-copy Q8 alias band may be published. It is the
// fail-closed gate for the alias path alone: the alias adds no device bytes, so it is judged against
// metalQ8AliasFraction (headroom for activations/KV only), NOT the additive-copy
// metalQ8UploadFraction. Every decline carries the concrete operands so a silent
// metal_live_q8_weights=0 is never unexplained.
// q6kUploadFits is the additive-copy budget predicate for eagerly uploading the resident Q6_K
// (k-quant) projection store at load time. Unlike the Q8 no-copy alias, UploadQ6KGoOwned COPIES the
// raw blocks into a fresh device buffer, so it is judged with the SAME 0.90 headroom as the #1087
// additive Q8 copy and is deliberately NOT overridable by FAK_METAL_Q8_UPLOAD (that env governs the
// Q8 path only; letting it force an additive Q6_K copy would reopen the OOM it guards).
func q6kUploadFits(residentBytes, q6kBytes, deviceTotal int64) bool {
	if deviceTotal <= 0 || q6kBytes <= 0 {
		return false
	}
	return float64(residentBytes+q6kBytes) <= metalQ8UploadFraction*float64(deviceTotal)
}

func q8AliasFits(residentBytes, deviceTotal int64, override string) error {
	if override != "" {
		return &MetalQ8ResidencyUnavailableError{Reason: fmt.Sprintf(
			"FAK_METAL_Q8_UPLOAD=%q override is not admissible for no-copy evidence (unset it to let the alias publish on its own evidence)", override)}
	}
	if deviceTotal <= 0 {
		return &MetalQ8ResidencyUnavailableError{Reason: "device working-set budget is unknown (Metal device memory probe returned 0)"}
	}
	if residentBytes < 0 {
		return &MetalQ8ResidencyUnavailableError{Reason: fmt.Sprintf("resident footprint is negative (%d bytes); refusing no-copy evidence", residentBytes)}
	}
	if limit := metalQ8AliasFraction * float64(deviceTotal); float64(residentBytes) > limit {
		return &MetalQ8ResidencyUnavailableError{Reason: fmt.Sprintf(
			"model owner leaves insufficient activation/KV headroom: resident=%d bytes > %.2f*deviceTotal=%d bytes (%.2f%% of device)",
			residentBytes, metalQ8AliasFraction, int64(limit), 100*float64(residentBytes)/float64(deviceTotal))}
	}
	return nil
}

// MetalQ6ResidencyUnavailableError is the fail-closed reason the Q6_K band could not be published as
// no-copy GPU residency. Ordinary execution may stay CPU-safe; the reason is carried so a
// metal_live_q6_weights=0 is never silent.
type MetalQ6ResidencyUnavailableError struct{ Reason string }

func (e *MetalQ6ResidencyUnavailableError) Error() string {
	return "model: Q6_K Metal residency unavailable: " + e.Reason
}

// q6kAliasFits decides whether the Q6_K band may be published as a no-copy alias. Mirroring
// q8AliasFits, the alias adds no device bytes, so it is judged against metalQ6AliasFraction (headroom
// for activations/KV only), NOT the additive-copy metalQ8UploadFraction. Every decline carries the
// concrete operands so a silent metal_live_q6_weights=0 is explained. The additive q6kUploadFits
// gate is deliberately unchanged and remains the fallback for a copied (unaligned/borrowed) upload.
func q6kAliasFits(residentBytes, deviceTotal int64) error {
	if deviceTotal <= 0 {
		return &MetalQ6ResidencyUnavailableError{Reason: "device working-set budget is unknown (Metal device memory probe returned 0)"}
	}
	if residentBytes < 0 {
		return &MetalQ6ResidencyUnavailableError{Reason: fmt.Sprintf("resident footprint is negative (%d bytes); refusing no-copy evidence", residentBytes)}
	}
	if limit := metalQ6AliasFraction * float64(deviceTotal); float64(residentBytes) > limit {
		return &MetalQ6ResidencyUnavailableError{Reason: fmt.Sprintf(
			"model owner leaves insufficient activation/KV headroom: resident=%d bytes > %.2f*deviceTotal=%d bytes (%.2f%% of device)",
			residentBytes, metalQ6AliasFraction, int64(limit), 100*float64(residentBytes)/float64(deviceTotal))}
	}
	return nil
}

func qwen38MetalQ8RuntimeNames(cfg Config) ([]string, error) {
	if cfg.NumLayers != 64 || len(cfg.LayerTypes) != 64 || !cfg.IsQwen35Hybrid() {
		return nil, &MetalQ8ResidencyUnavailableError{Reason: "runtime is not the exact 64-layer Qwen3.8 hybrid"}
	}
	names := make([]string, 0, 272)
	linear, full := 0, 0
	for l, kind := range cfg.LayerTypes {
		lp := func(s string) string { return layerName(l, s) }
		switch kind {
		case "linear_attention":
			linear++
			for _, suffix := range []string{
				"linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
				"linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
				"linear_attn.out_proj.weight",
			} {
				names = append(names, lp(suffix))
			}
		case "full_attention":
			full++
			names = append(names, lp("self_attn.q_proj.weight"), lp("self_attn.k_proj.weight"))
		default:
			return nil, &MetalQ8ResidencyUnavailableError{Reason: fmt.Sprintf("layer %d has unexpected type %q", l, kind)}
		}
	}
	if linear != 48 || full != 16 || len(names) != 272 {
		return nil, &MetalQ8ResidencyUnavailableError{Reason: fmt.Sprintf("runtime Q8 topology is %d linear/%d full/%d projections, want 48/16/272", linear, full, len(names))}
	}
	return names, nil
}

// buildAllOrNothing creates every requested owner or releases successful predecessors in reverse.
// Publication is deliberately the caller's next step, after this transaction has fully succeeded.
func buildAllOrNothing[T any](names []string, build func(string) (T, error), release func(T)) ([]T, error) {
	items := make([]T, 0, len(names))
	for _, name := range names {
		item, err := build(name)
		if err != nil {
			for i := len(items) - 1; i >= 0; i-- {
				release(items[i])
			}
			return nil, err
		}
		items = append(items, item)
	}
	return items, nil
}
