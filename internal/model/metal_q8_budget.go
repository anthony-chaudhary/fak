package model

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
