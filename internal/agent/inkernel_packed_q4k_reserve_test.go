package agent

import (
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestPackedQ4KRequestReserve(t *testing.T) {
	embed, err := model.NewQ4KEmbedding(make([]byte, 144), 1, 256)
	if err != nil {
		t.Fatalf("NewQ4KEmbedding: %v", err)
	}
	cfg := model.Config{
		ModelType: "qwen3_5_text", VocabSize: 1, HiddenSize: 256,
		NumLayers: 4, NumHeads: 2, NumKVHeads: 1, HeadDim: 8,
		LayerTypes: []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
	}
	m := &model.Model{Cfg: cfg, Q2KEmbedding: embed}
	planner := &InKernelPlanner{m: m, quant: true, q4k: true, metal: true}
	session := m.NewSession()
	planner.configureNativeSession(session)

	t.Run("cold reserves only attention token planes", func(t *testing.T) {
		const prompt, maxNew = 5, 2
		before := session.Cache.OwnedPayloadBytes()
		planner.preReservePackedQ4KRequest(session, prompt, maxNew)
		got := session.Cache.OwnedPayloadBytes() - before
		// One full-attention layer owns K/Kraw/V rows (3*HeadDim*f32);
		// pos and token lineage add one int and one uint32 per position.
		bytesPerToken := 3*8*4 + strconv.IntSize/8 + 4
		if want := int64((prompt + maxNew) * bytesPerToken); got != want {
			t.Fatalf("reserved payload delta = %d, want %d; recurrent layers must add no token planes", got, want)
		}
	})

	t.Run("remaining count", func(t *testing.T) {
		for _, tc := range []struct {
			name           string
			resident, want int
		}{
			{name: "cold", resident: 0, want: 9},
			{name: "reused", resident: 6, want: 3},
			{name: "exact prompt hit", resident: 7, want: 2},
			{name: "already planned", resident: 9, want: 0},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if got := requestReserveExtra(7, 2, tc.resident); got != tc.want {
					t.Fatalf("requestReserveExtra = %d, want %d", got, tc.want)
				}
			})
		}
	})

	t.Run("legacy and unsupported predicates", func(t *testing.T) {
		q2, err := model.NewQ2KEmbedding(make([]byte, 84), 1, 256)
		if err != nil {
			t.Fatalf("NewQ2KEmbedding: %v", err)
		}
		q2Model := &model.Model{Cfg: cfg, Q2KEmbedding: q2}
		moeCfg := cfg
		moeCfg.NumExperts = 2
		moeModel := &model.Model{Cfg: moeCfg, Q2KEmbedding: embed}
		cases := []struct {
			name   string
			p      *InKernelPlanner
			mutate func(*model.Session)
		}{
			{name: "nil F32 embedding", p: &InKernelPlanner{m: &model.Model{Cfg: cfg}, quant: true, q4k: true, metal: true}},
			{name: "Q2_K", p: &InKernelPlanner{m: q2Model, q4k: true, metal: true}},
			{name: "backend", p: &InKernelPlanner{m: m, q4k: true, metal: true, backend: compute.Default()}},
			{name: "MoE", p: &InKernelPlanner{m: moeModel, q4k: true, metal: true}},
			{name: "missing Metal", p: &InKernelPlanner{m: m, q4k: true}},
			{name: "conflicting F16", p: planner, mutate: func(s *model.Session) { s.F16 = true }},
			{name: "GPU offload", p: planner, mutate: func(s *model.Session) { s.GPULayers = 1 }},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				s := tc.p.m.NewSession()
				tc.p.configureNativeSession(s)
				if tc.mutate != nil {
					tc.mutate(s)
				}
				if packedQ4KRequestReserveSupported(tc.p, s) {
					t.Fatal("unsupported route admitted pre-reserve")
				}
			})
		}
	})
}
