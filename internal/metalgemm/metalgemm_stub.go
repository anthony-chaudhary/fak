//go:build !(darwin && cgo && fakmetal)

// Package metalgemm stub — the default (non-fakmetal) build. The Metal backend is not
// linked, so the whole package degrades to "unavailable" and callers fall back to the
// pure-Go CPU prefill path. This file is what keeps `go build ./...` cgo-free and portable.
package metalgemm

// Available always reports false without the Metal backend compiled in.
func Available() bool { return false }

// Compiled reports false: this binary was not built with `-tags fakmetal`.
func Compiled() bool { return false }

// Weight is an inert handle in the stub build.
type Weight struct{ Out, In int }

// Upload is a no-op returning nil when Metal is not compiled in.
func Upload(w []float32, out, in int) *Weight { return nil }

// MatMul is unreachable in the stub build (Upload never returns a usable handle).
func (w *Weight) MatMul(x []float32, P int, y []float32) {}

// Free is a no-op in the stub build.
func (w *Weight) Free() {}

// ID returns an invalid handle in the stub build.
func (w *Weight) ID() int { return -1 }

// UploadVec is a no-op returning -1 when Metal is not compiled in.
func UploadVec(v []float32) int { return -1 }

// FwdConfig is a no-op in the stub build.
func FwdConfig(nLayers, H, hd, nH, nKV, I int, eps, theta float32, attnBias bool) {}

// FwdLayer is a no-op in the stub build.
func FwdLayer(layer, q, k, v, o, gate, up, down, inNorm, postNorm, qb, kb, vb int) {}

// FwdFinalNorm is a no-op in the stub build.
func FwdFinalNorm(id int) {}

// Reset is a no-op in the stub build (no resident Metal state to tear down).
func Reset() {}

// Prefill is unavailable in the stub build; ok is always false so callers fall back.
func Prefill(X []float32, P, nLayers, w, H int) (lastPre, kraw, kpost, v []float32, ok bool) {
	return nil, nil, nil, nil, false
}
