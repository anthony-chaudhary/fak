//go:build vulkan && (windows || linux) && cgo

package model

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime integration est=5s lane=optin
// Refs #13668. The registered Vulkan backend satisfies the three slice-3 model
// seams, and each session-bound seam matches its host arm on the device:
// embedding rows and index rows exactly, the inverse rotation within the
// tail-RoPE bound (prefix bit-exact, tail 2e-6 abs+rel). The index-select
// module is optional; FAK_VULKAN_REQUIRE_V41_INDEX_SELECT=1 makes it required.
func TestV41GatherOpsVulkanSeams(t *testing.T) {
	be := compute.Pick("vulkan")
	if be.Name() != "vulkan" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device not registered")
		}
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	if _, ok := be.(v41TailRoPERowsBackend); !ok {
		t.Fatal("vulkan backend lacks the inverse-rotation rows seam")
	}
	if _, ok := be.(v41EmbeddingGatherBackend); !ok {
		t.Fatal("vulkan backend lacks the embedding gather seam")
	}
	selectBackend, ok := be.(v41IndexSelectBackend)
	if !ok {
		t.Fatal("vulkan backend lacks the index-select seam")
	}

	t.Run("inverse-output", func(t *testing.T) {
		// Layer 3 must be a configured decoder layer: the ordinary BF16
		// publication's closed site table refuses layers outside the model.
		s := &Session{M: &Model{Cfg: Config{NumLayers: 4}}, Backend: be}
		t.Cleanup(s.Close)
		ops := s.v41DeviceGatherOpsFunc()
		if ops.inverseOutput == nil {
			t.Fatal("inverse rotation seam was not bound")
		}
		cfg := Config{HeadDim: v41KVLoraRank, NumHeads: 4, QKRopeHeadDim: 64, RopeTheta: 10000}
		out := make([]float32, cfg.NumHeads*cfg.HeadDim)
		for i := range out {
			out[i] = float32((i%41)-20)*0.03125 + 0.001953125
		}
		want, err := v41AttentionOutputForProjection(cfg, 3, 11, append([]float32(nil), out...), cfg.NumHeads, cfg.HeadDim)
		if err != nil {
			t.Fatal(err)
		}
		got, err := v41RotatedOutputForProjection(cfg, 3, 11, out, cfg.NumHeads, cfg.HeadDim, ops)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if i%cfg.HeadDim < cfg.HeadDim-cfg.QKRopeHeadDim {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("prefix lane %d changed", i)
				}
				continue
			}
			// Both arms round to BF16 after rotation; allow one BF16 ulp of the
			// 2e-6 F32 tail bound crossing a rounding boundary.
			if d := math.Abs(float64(got[i]) - float64(want[i])); d > 2e-6+math.Abs(float64(want[i]))/128 {
				t.Fatalf("tail lane %d got %g want %g", i, got[i], want[i])
			}
		}
	})

	t.Run("index-select", func(t *testing.T) {
		if !selectBackend.V41IndexSelectAvailable() {
			if os.Getenv("FAK_VULKAN_REQUIRE_V41_INDEX_SELECT") == "1" {
				t.Fatal("required v41_index_select.spv not loaded")
			}
			t.Skip("optional V4.1 index-select module not loaded")
		}
		s := &Session{M: &Model{}, Backend: be}
		t.Cleanup(s.Close)
		sel := s.v41DeviceGatherOpsFunc().indexSelectFunc()
		scores := make([]float32, 777)
		for i := range scores {
			scores[i] = float32((i*37)%11-5) * 0.25
		}
		scores[3], scores[400] = float32(math.Inf(-1)), float32(math.Copysign(0, -1))
		for _, tc := range [][3]int{{64, 0, 0}, {512, 16, 8}, {5, 1, 4}} {
			want, err := v41IndexSelectHost(scores, tc[0], tc[1], tc[2])
			if err != nil {
				t.Fatal(err)
			}
			got, err := sel(0, scores, tc[0], tc[1], tc[2])
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("selection %v differs from host: err=%v", tc, err)
			}
		}
	})

	t.Run("q2k-embedding-gather", func(t *testing.T) {
		q2k, ok := be.(v41Q2KEmbeddingGatherBackend)
		if !ok {
			t.Fatal("vulkan backend lacks the Q2_K embedding gather seam")
		}
		if !q2k.V41Q2KEmbeddingGatherAvailable() {
			if os.Getenv("FAK_VULKAN_REQUIRE_V41_Q2K_EMBED_GATHER") == "1" {
				t.Fatal("required v41_q2k_embed_gather.spv not loaded")
			}
			t.Skip("optional V4.1 Q2_K embedding gather module not loaded")
		}
		const vocab, hidden = 11, 512
		embed, err := NewQ2KEmbedding(v41Q2KEmbeddingTestRaw(vocab, hidden, 7), vocab, hidden)
		if err != nil {
			t.Fatal(err)
		}
		m := &Model{Cfg: Config{HiddenSize: hidden, VocabSize: vocab}, Q2KEmbedding: embed}
		s := &Session{M: m, Backend: be}
		t.Cleanup(s.Close)
		ops := s.v41DeviceGatherOpsFunc()
		if ops.embedGather == nil {
			t.Fatal("Q2_K embedding gather seam was not bound")
		}
		ids := []int{10, 0, 4, 4, 9}
		want, err := m.v41EmbeddingPanel(ids)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.v41EmbeddingPanelOn(ids, ops)
		if err != nil {
			t.Fatal(err)
		}
		v41TailRoPECheckBits(t, "device Q2_K embedding panel", got, want)
	})

	t.Run("embedding-gather", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		s := v41DenseTestSession(t, m, be)
		ops := s.v41State().gatherOps
		if ops.embedGather == nil {
			t.Fatal("embedding gather seam was not bound for an F32 table")
		}
		ids := []int{2, 0, 2, 5}
		want, err := m.v41EmbeddingPanel(ids)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.v41EmbeddingPanelOn(ids, ops)
		if err != nil {
			t.Fatal(err)
		}
		v41TailRoPECheckBits(t, "device embedding panel", got, want)
	})
}
