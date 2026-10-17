package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestQ2KEmbeddingRowGatherHALUnavailableOnCPURef(t *testing.T) {
	m := stepGreedyTestModel()
	s := m.NewBackendSession(compute.Pick("cpu-ref"))
	if s.q2kEmbeddingRowGatherHAL() != nil {
		t.Fatal("cpu-ref session selected a device Q2_K embedding gather")
	}
}

func TestQ2KEmbeddingRowGatherHALMatchesHostGatherRow(t *testing.T) {
	be := compute.Pick("vulkan")
	if be.Name() != "vulkan" {
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	const vocab, hidden = 41, 512
	m := NewSynthetic(Config{
		HiddenSize: hidden, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 128,
		IntermediateSize: 64, VocabSize: vocab, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1,
	})
	rng := rand.New(rand.NewSource(13668))
	blocks := vocab * hidden / qkK
	raw := make([]byte, blocks*q2kBlockBytes)
	halves := []uint16{0x3c00, 0xbc00, 0x2e66, 0xb555, 0x0001, 0x8000, 0x0000, 0x3555}
	for b := 0; b < blocks; b++ {
		blk := raw[b*q2kBlockBytes : (b+1)*q2kBlockBytes]
		rng.Read(blk[:80])
		binary.LittleEndian.PutUint16(blk[80:], halves[rng.Intn(len(halves))])
		binary.LittleEndian.PutUint16(blk[82:], halves[rng.Intn(len(halves))])
	}
	packed, err := NewQ2KEmbedding(raw, vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}
	m.Q2KEmbedding = packed
	s := m.NewBackendSession(be)
	s.ensureOpenBackendSession()
	gather := s.q2kEmbeddingRowGatherHAL()
	if gather == nil {
		t.Skip("vulkan Q2_K embedding gather module not loaded")
	}
	want := make([]float32, hidden)
	for _, id := range []int{0, 7, vocab - 1, 7} {
		row := gather(id)
		if len(row.Shape) != 1 || row.Shape[0] != hidden {
			t.Fatalf("id %d: row shape %v, want [%d]", id, row.Shape, hidden)
		}
		got := be.Read(row)
		be.Free(row)
		if err := packed.GatherRow(id, want, 1.0); err != nil {
			t.Fatal(err)
		}
		for d := range want {
			if math.Float32bits(got[d]) != math.Float32bits(want[d]) {
				t.Fatalf("id %d lane %d bits=%08x want host GatherRow %08x", id, d, math.Float32bits(got[d]), math.Float32bits(want[d]))
			}
		}
	}
}
