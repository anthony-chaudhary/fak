package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=1ms lane=default
func TestFusedCounterReachesMetricsAndHealthz(t *testing.T) {
	compute.DeclareFusedOps()
	var b strings.Builder
	writeComputeWorkMetrics(&b)
	for _, want := range []string{
		`fak_compute_fused_total{op="q4k_swiglu_matmul_add",mode="fused"} `,
		`fak_compute_fused_total{op="gdn_q8_in_proj",mode="composed"} `,
	} {
		if !strings.Contains(b.String(), want) {
			t.Errorf("/metrics missing %q", want)
		}
	}
	h := computeFusionsHealth()
	arms, ok := h["default_off"].([]compute.FusionArm)
	if !ok || len(arms) == 0 {
		t.Fatalf("/healthz compute_fusions.default_off=%#v", h["default_off"])
	}
	if _, ok := h["missing_shader"]; !ok {
		t.Fatal("/healthz compute_fusions has no missing_shader list")
	}
}
