//go:build !(darwin && arm64 && cgo)

package model

// q4kGemmGroupCPU reuses one activation quantization across same-width resident
// projections (notably gate/up). It does not group or replace the weight kernels:
// Profiling retains the per-target caller path so its MAC attribution is intact.
// Every target uses the same extract-once dispatcher as q4kGemmInto, including its
// architecture, GEMM-mode and affine-min reduction choices. The panel is read-only
// until all synchronous target calls finish; outputs remain separately owned.
func (s *Session) q4kGemmGroupCPU(names []string, X []float32, P int) [][]float32 {
	if s == nil || s.M == nil || s.M.prism != nil || qprofOn || !qgemmGroup || P <= 1 ||
		!q4kSDOTEnabled() || !q4kExtractOnceGemmEnabled() {
		return nil
	}
	in, members := 0, 0
	for _, name := range names {
		qt := s.M.q4kw[name]
		if qt == nil {
			continue
		}
		if qt.in <= 0 || qt.out <= 0 || qt.nblk <= 0 || (members != 0 && qt.in != in) {
			return nil
		}
		in = qt.in
		members++
	}
	if members < 2 || in != len(X)/P || len(X)%P != 0 {
		return nil
	}
	// Keep the ordinary raw-residency guard, before allocating shared scratch or
	// publishing any output. A missing CPU payload is not a numerical fallback.
	for _, name := range names {
		if qt := s.M.q4kw[name]; qt != nil {
			qt.requireRawCPU("prefill GEMM")
		}
	}
	panel := quantizeBatchPanel(X, P, in)
	out := make([][]float32, len(names))
	for i, name := range names {
		if qt := s.M.q4kw[name]; qt != nil {
			s.observeQ4KPrefillGEMM(Q4KPrefillGEMMCPU)
			out[i] = make([]float32, P*qt.out)
			q4kGemmExtractOnceInt8Into(qt, panel, out[i])
		}
	}
	return out
}
