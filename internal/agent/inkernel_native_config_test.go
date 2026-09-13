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

// TestExplicitKVPrecisionReachesRequestSession proves --kv-precision= q8_0 threads
// through InKernelPlannerConfig and is realized on the request session's kernel-owned
// cache, while the default leaves the session on the exact f32 tier.
func TestExplicitKVPrecisionReachesRequestSession(t *testing.T) {
	cfg := model.Config{NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 32, HiddenSize: 128, IntermediateSize: 256, RopeTheta: 10000}

	def := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "kv-precision-default", true, nil, false, InKernelPlannerConfig{})
	s := def.m.NewSession()
	def.configureNativeSession(s)
	if s.KVPrecision != model.KVPrecisionFP32 || s.Cache.Precision() != model.KVPrecisionFP32 {
		t.Fatalf("default session tier = (%s,%s), want f32", s.KVPrecision, s.Cache.Precision())
	}

	q8 := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "kv-precision-q8", true, nil, false, InKernelPlannerConfig{KVPrecision: model.KVPrecisionQ8_0})
	ss := q8.m.NewSession()
	q8.configureNativeSession(ss)
	if ss.KVPrecision != model.KVPrecisionQ8_0 || ss.Cache.Precision() != model.KVPrecisionQ8_0 {
		t.Fatalf("q8 session tier = (%s,%s), want q8_0", ss.KVPrecision, ss.Cache.Precision())
	}
}

// TestKVPrecisionRefusesUnsupportedArch proves the gate refuses q8 on an architecture
// whose forward writes c.K directly, rather than silently mixing f32 and packed rows.
func TestKVPrecisionRefusesUnsupportedArch(t *testing.T) {
	cfg := model.Config{NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 32, HiddenSize: 128, IntermediateSize: 256, ModelType: "deepseek2"}
	p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "kv-precision-refuse", true, nil, false, InKernelPlannerConfig{KVPrecision: model.KVPrecisionQ8_0})
	defer func() {
		if recover() == nil {
			t.Fatal("configureNativeSession must panic (refuse) q8 on an unsupported architecture")
		}
	}()
	p.configureNativeSession(p.m.NewSession())
}
