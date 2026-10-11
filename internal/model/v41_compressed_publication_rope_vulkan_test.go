//go:build vulkan && (windows || linux) && cgo && !v41_indexer_witness && !v41_bf16_boundary_witness && !v41_logit_scale_witness && !v41_tail_rope_exact_witness

// Run serially from the repository root with the coupled V12/72 shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41CompressedPublicationRoPEVulkan$' -count=1
// Ordinary archive only; synthetic fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The compressed publication RoPE runs on the ordinary build's strict tail-RoPE
// twin by default (#13816). Rotated BF16 tails and untouched prefixes are
// bit-identical to the host arm, including subnormal, signed-zero and tie
// vectors, and the default session forward routes every completed group through
// the seam, matching the host-arm session and the host greedy tokens.
// fak-test:justify why=integration when=changed:internal/model/v41_compressed_publication_rope*
// fak-test:runtime slow est=20s lane=optin
func TestV41CompressedPublicationRoPEVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	exact, ok := be.(compute.V41TailRoPEExactBackend)
	if !ok || !exact.V41TailRoPEExactAvailable() || !exact.V41TailRoPERowsAvailable() {
		t.Fatal("ordinary Vulkan build did not admit the strict tail-RoPE twin on this device")
	}
	t.Run("seam", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		s := v41DenseTestSession(t, m, be)
		rotate := s.v41State().gatherOps.compressedPublicationRoPE
		if rotate == nil {
			t.Fatal("compressed publication RoPE seam was not bound")
		}
		cfg := m.Cfg
		rd, width, keyWidth := cfg.QKRopeHeadDim, cfg.HeadDim, cfg.IndexHeadDim
		if rd <= 0 || rd > width || rd > keyWidth {
			t.Fatalf("fixture rotary geometry rd=%d latent=%d key=%d", rd, width, keyWidth)
		}
		roles := m.v41AttentionRolesCached()
		tails, exactLanes, calls := 0, 0, 0
		for l := 0; l < cfg.NumLayers; l++ {
			plan, err := v41AttentionPlanFor(cfg, l, roles)
			if err != nil {
				t.Fatal(err)
			}
			if !v41OwnsCompressedRows(plan) {
				continue
			}
			for _, pos := range []int{0, 1, 7, 130, 4097} {
				for _, withKey := range []bool{true, false} {
					latent := make([]float32, width)
					for i := range latent {
						latent[i] = v41RoundBF16(float32(((i*13+pos*5+l)%41)-20)*0.03125 + 0.000244140625)
					}
					var key []float32
					if withKey {
						key = make([]float32, keyWidth)
						for i := range key {
							key[i] = v41RoundBF16(float32(((i*7+pos*3+l)%37)-18)*0.0625 + 0.00048828125)
						}
					}
					wantL, wantK := append([]float32(nil), latent...), append([]float32(nil), key...)
					if !withKey {
						wantK = nil
					}
					if err := m.v41CompressedPublicationRoPE(l, pos, wantL, wantK); err != nil {
						t.Fatal(err)
					}
					gotL, gotK := append([]float32(nil), latent...), append([]float32(nil), key...)
					if !withKey {
						gotK = nil
					}
					n := v41SeamDispatches(t, be, func() { err = m.v41CompressedPublicationRoPEWithDevice(l, pos, gotL, gotK, rotate) })
					if err != nil {
						t.Fatal(err)
					}
					if n == 0 {
						t.Fatalf("layer %d pos %d key=%t device publication issued no native dispatch", l, pos, withKey)
					}
					calls++
					v41TailRoPECheckBits(t, "latent publication", gotL, wantL)
					v41TailRoPECheckBits(t, "index publication", gotK, wantK)
					tails += rd
					exactLanes += rd
					if withKey {
						tails += rd
						exactLanes += rd
					}
				}
			}
		}
		if calls == 0 {
			t.Fatal("fixture has no compressed producer layer; the seam witness would be vacuous")
		}
		t.Logf("V41_COMPRESSED_PUBLICATION_ROPE_SEAM calls=%d tail_lanes=%d bit_exact=%d device=%t", calls, tails, exactLanes, true)
	})
	t.Run("bits", func(t *testing.T) {
		vectors := []struct {
			name string
			pos  int
			a, b uint32
			want [2]uint32
		}{
			{"negative-positive-zero", 0, 0x80000000, 0, [2]uint32{0x80000000, 0}},
			{"positive-negative-zero", 0, 0, 0x80000000, [2]uint32{0, 0}},
			{"both-negative-zero", 0, 0x80000000, 0x80000000, [2]uint32{0, 0x80000000}},
			{"minimum-subnormal", 0, 0x00010000, 0, [2]uint32{0x00010000, 0}},
			{"negative-minimum-subnormal", 0, 0x80010000, 0, [2]uint32{0x80010000, 0}},
			{"zero-subnormal-tie", 0, 0x00008000, 0, [2]uint32{0, 0}},
			{"odd-subnormal-tie", 0, 0x00018000, 0, [2]uint32{0x00020000, 0}},
			{"positive-subnormal", 0, 0x00400000, 0, [2]uint32{0x00400000, 0}},
			{"negative-subnormal", 0, 0x80400000, 0, [2]uint32{0x80400000, 0}},
			{"tie-even-positive", 0, 0x3f808000, 0, [2]uint32{0x3f800000, 0}},
			{"tie-odd-positive", 0, 0x3f818000, 0, [2]uint32{0x3f820000, 0}},
			{"tie-even-negative", 0, 0xbf808000, 0, [2]uint32{0xbf800000, 0}},
			{"tie-odd-negative", 0, 0xbf818000, 0, [2]uint32{0xbf820000, 0}},
			{"cancellation-rounding", 2, 0x41820000, 0xc0ee0000, [2]uint32{0x3a060000, 0x418f0000}},
		}
		lanes := 0
		for _, v := range vectors {
			for _, rows := range []int{1, 2} {
				m := &Model{Cfg: Config{HeadDim: 8, IndexHeadDim: 6, QKRopeHeadDim: 2, RopeTheta: 10000}}
				s := &Session{M: m, Backend: be}
				rotate := s.v41CompressedPublicationRoPEFunc()
				if rotate == nil {
					t.Fatal("strict publication seam did not bind on the device")
				}
				prefix := []float32{math.Float32frombits(0x80000000), math.Float32frombits(0x7fc01234), 1.001, -2.003, 3, 4}
				latent := append(append([]float32(nil), prefix...), math.Float32frombits(v.a), math.Float32frombits(v.b))
				var key []float32
				if rows == 2 {
					key = append(append([]float32(nil), prefix[:4]...), math.Float32frombits(v.b), math.Float32frombits(v.a))
				}
				cos, sin := v41RopeTableForLayer(m.Cfg, 0, v.pos)
				wantL := v41PublicationRoPEOracle(latent, 2, cos, sin)
				for i, u := range v.want {
					if math.Float32bits(wantL[6+i]) != u {
						t.Fatalf("%s literal oracle disagreement %08x != %08x", v.name, math.Float32bits(wantL[6+i]), u)
					}
				}
				var wantK []float32
				if rows == 2 {
					wantK = v41PublicationRoPEOracle(key, 2, cos, sin)
				}
				var err error
				n := v41SeamDispatches(t, be, func() { err = m.v41CompressedPublicationRoPEWithDevice(0, v.pos, latent, key, rotate) })
				if err != nil {
					t.Fatalf("%s rows=%d: %v", v.name, rows, err)
				}
				if n == 0 {
					t.Fatalf("%s rows=%d issued no native dispatch", v.name, rows)
				}
				v41TailRoPECheckBits(t, v.name+" latent", latent, wantL)
				v41TailRoPECheckBits(t, v.name+" index", key, wantK)
				s.Close()
				lanes += 2 * rows
			}
		}
		t.Logf("V41_COMPRESSED_PUBLICATION_ROPE_BITS vectors=%d rows=1,2 tail_lanes=%d bit_exact=%d", len(vectors), lanes, lanes)
	})
	t.Run("default-session", func(t *testing.T) {
		v41SeamGreedy(t, "compressed-publication-rope", func() *Model { return v41IndexerTestFixture(t) }, be, func(st *v41ForwardState) *int {
			calls := new(int)
			if st.gatherOps == nil || st.gatherOps.compressedPublicationRoPE == nil {
				t.Fatal("compressed publication RoPE seam was not bound")
			}
			inner := st.gatherOps.compressedPublicationRoPE
			st.gatherOps.compressedPublicationRoPE = func(l int, tails, cos, sin []float32, rows, rd int) ([]float32, error) {
				*calls++
				return inner(l, tails, cos, sin, rows, rd)
			}
			return calls
		}, func(st *v41ForwardState) {
			st.gatherOps.compressedPublicationRoPE = nil
		}, func(m *Model) int {
			// One call per completed group of each compressed producer layer over
			// the 3-token prompt plus the decoded tokens fed back by Step.
			positions := 3 + v41DeviceParitySteps - 1
			roles, want := m.v41AttentionRolesCached(), 0
			for l := 0; l < m.Cfg.NumLayers; l++ {
				plan, err := v41AttentionPlanFor(m.Cfg, l, roles)
				if err != nil {
					t.Fatal(err)
				}
				if v41OwnsCompressedRows(plan) {
					want += positions / plan.kvGroupSize(m.Cfg)
				}
			}
			if want == 0 {
				t.Fatal("fixture completes no compressed group; the session witness would be vacuous")
			}
			return want
		})
	})
}
