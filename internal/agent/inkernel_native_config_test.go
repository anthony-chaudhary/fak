package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestExplicitQ4KSlabConfigReachesRequestSession(t *testing.T) {
	t.Setenv("FAK_Q4K_GATEUP_SLAB", "0")
	p := NewInKernelPlannerWithConfig(model.NewSynthetic(model.Config{}), nil, "native-session-config", true, nil, false, InKernelPlannerConfig{Q4KGateUpOutputSlab: true})
	s := p.m.NewSession()
	p.configureNativeSession(s)
	if !s.Q4K || !s.Q4KGateUpOutputSlab {
		t.Fatalf("request session did not receive explicit Q4_K config: Q4K=%t slab=%t", s.Q4K, s.Q4KGateUpOutputSlab)
	}
}
