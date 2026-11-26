package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestPromptPanelChunkTokensHonorsAdmittedHeadroom pins the session chunk to the admitted
// headroom: none keeps the transient-only chunk, and the Qwen2.5-3B demand the planner
// prices round-trips into a full-cap chunk. A backend without the panel seam prices nothing.
func TestPromptPanelChunkTokensHonorsAdmittedHeadroom(t *testing.T) {
	m := &Model{Cfg: Config{
		HiddenSize: 2048, IntermediateSize: 11008, VocabSize: 151936, NumLayers: 36,
		NumHeads: 16, NumKVHeads: 2, HeadDim: 128,
	}}
	s := &Session{M: m}
	base := compute.PromptPanelChunkTokens(m.promptPanelScratchConfig(), promptPanelMaxTokens)
	if got := s.promptPanelChunkTokens(); got != base || base >= 256 {
		t.Fatalf("no-headroom chunk = %d, want transient-only %d (< 256)", got, base)
	}
	demand := compute.PromptPanelHeadroomDemand(m.promptPanelScratchConfig(), promptPanelMaxTokens, 4096, 1<<30)
	s.PromptPanelHeadroomBytes = demand.Total()
	if got := s.promptPanelChunkTokens(); got != promptPanelMaxTokens {
		t.Fatalf("headroom chunk = %d, want %d", got, promptPanelMaxTokens)
	}
	if plan := m.PromptPanelHeadroomDemand(nil, 4096, 1<<30); plan != nil {
		t.Fatalf("seamless backend demand = %v, want nil", plan)
	}
}
