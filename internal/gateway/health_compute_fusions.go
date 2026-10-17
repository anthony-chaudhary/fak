package gateway

import "github.com/anthony-chaudhary/fak/internal/compute"

// computeFusionsHealth is the /healthz compute_fusions block: every default-off
// fusion arm and every fused shader missing at init, so a node running the
// composed path is visible without scraping /metrics. nil when no backend that
// dispatches fused kernels is live.
func computeFusionsHealth() map[string]any {
	if compute.FusedCounts() == nil {
		return nil
	}
	missing := []map[string]string{}
	for _, w := range compute.FusedShaderWarnings() {
		missing = append(missing, map[string]string{"op": w.Op, "shader": w.Shader})
	}
	return map[string]any{
		"default_off":    compute.DefaultOffFusions(),
		"missing_shader": missing,
	}
}
