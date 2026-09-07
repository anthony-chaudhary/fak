package gateway

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelPlannerConfigReachesProductionPlanner(t *testing.T) {
	t.Setenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS", "4096")
	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "off")
	want := agent.InKernelPlannerConfig{
		CPUOffloadExperts:         true,
		QwenQ4KPrefillChunkTokens: 2048,
		Qwen35MetalGDNSequence:    true,
		Q4KGateUpOutputSlab:       true,
	}
	planner := newInKernelChatPlanner(Config{
		InKernelModel:     model.NewSynthetic(model.Config{}),
		InKernelPlanner:   want,
		CPUOffloadExperts: true,
	}, "native-config-reachability", t.Logf)
	got := planner.(*agent.InKernelPlanner).RuntimeConfig()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("production planner config = %+v, want %+v", got, want)
	}
}

func TestInKernelPromptShrinkLeversConfigReachability(t *testing.T) {
	cfg := Config{
		InKernelModel:        model.NewSynthetic(model.Config{}),
		CompactHistoryBudget: 32000,
		ElideStaleReads:      true,
		DeferColdTools:       true,
	}
	planner := newInKernelChatPlanner(cfg, "native-shrink-reachability", t.Logf)
	rc := planner.(*agent.InKernelPlanner).RuntimeConfig()
	if rc.CompactHistoryBudget != 32000 {
		t.Fatalf("CompactHistoryBudget = %d, want 32000", rc.CompactHistoryBudget)
	}
	if !rc.ElideStaleReads {
		t.Fatal("ElideStaleReads = false, want true")
	}
	if !rc.DeferColdTools {
		t.Fatal("DeferColdTools = false, want true")
	}
}
