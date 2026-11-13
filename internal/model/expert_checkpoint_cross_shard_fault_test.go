package model

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_checkpoint_cross_shard_fault_test.go — byte-level FAULT/PREFETCH witness
// for the DS4.1 layer-16 cross-shard split (fak-private ds41-5090 DIVERGENCE.md).
//
// Complementary to, not a duplicate of, deepseek41_cross_shard_layer_test.go
// (ggufload, commit 475abfc5a1e): that guard pins DESCRIPTOR resolution — each
// fused slab groups under its own shard reader at its own offset, TensorBytes
// reads each tensor from its own file, and tier.Has indexes every leaf. It stops
// at the index. Every committed ExpertCheckpointTier test builds a SINGLE-shard
// tier, so nothing pins what happens when one tier holds TWO shard sources and a
// leaf's entry.shard routes the fault: AddShardData records entry.shard PER
// FUSED TENSOR, and fault / staging().mk / Prefetch must resolve through
// t.shards[entry.shard]. A regression that cached one shard handle per layer, or
// pinned entries to the first (or last) shard added, keeps tier.Has green and
// hands the down projection's source to the gate projection — exactly the
// mis-routing class the raider layer-16 signature implied. Mutation witness:
// forcing entry.shard to 0 in AddShardData turns this test red while leaving
// 475abfc5a1e's assertions green.
//
// Layout mirrors the published artifact's layer 16: down (Q3_K) in the earlier
// shard, gate/up (Q2_K) in the later one, with distinct per-tensor payloads so a
// reader or offset swap between the two shards cannot read back equal bytes.
// fak-test:runtime fast est=50ms lane=default
func TestExpertCheckpointTierCrossShardFaultReturnsOwningShardBytes(t *testing.T) {
	const E, H = 4, 256
	downStride := int64(H * (H / qkK) * q3kBlockBytes)
	gateStride := int64(H * (H / qkK) * q2kBlockBytes)

	// patterned fills a payload with a tensor-specific byte stream: salt keeps the
	// three slabs distinct even where they share an in-shard offset, and the
	// per-expert term keeps experts distinct within a slab.
	patterned := func(n, salt int) []byte {
		b := make([]byte, n)
		per := n / E
		for i := range b {
			b[i] = byte((salt*197 + (i/per)*131 + i*7 + 17) & 0xff)
		}
		return b
	}
	downSlab := patterned(int(downStride)*E, 1)
	gateSlab := patterned(int(gateStride)*E, 2)
	upSlab := patterned(int(gateStride)*E, 3)

	// Shard A ("00004"): the down slab only, at offset 0.
	// Shard B ("00005"): gate at offset 0, up after it — as in the real split.
	var shardB []byte
	shardB = append(shardB, gateSlab...)
	shardB = append(shardB, upSlab...)

	tier := NewExpertCheckpointTier(0) // stream-through: no host retention
	if err := tier.AddShard(bytes.NewReader(downSlab), int64(len(downSlab)), []FusedExpertTensor{{
		Name: "blk.0.ffn_down_exps.weight", Layer: 0, Proj: "down_proj", Arch: "deepseek41",
		Quant: ExpertCheckpointQ3K, Offset: 0, Experts: E, Rows: H, Cols: H,
	}}); err != nil {
		t.Fatalf("AddShard (down shard): %v", err)
	}
	if err := tier.AddShard(bytes.NewReader(shardB), int64(len(shardB)), []FusedExpertTensor{
		{Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj", Arch: "deepseek41",
			Quant: ExpertCheckpointQ2K, Offset: 0, Experts: E, Rows: H, Cols: H},
		{Name: "blk.0.ffn_up_exps.weight", Layer: 0, Proj: "up_proj", Arch: "deepseek41",
			Quant: ExpertCheckpointQ2K, Offset: int64(len(gateSlab)), Experts: E, Rows: H, Cols: H},
	}); err != nil {
		t.Fatalf("AddShard (gate/up shard): %v", err)
	}

	type want struct {
		leaf   string
		suffix string
		slab   []byte
		stride int64
		dt     compute.Dtype
	}
	wants := []want{
		{"w1", "gate_proj.weight", gateSlab, gateStride, compute.Q2_K},
		{"w3", "up_proj.weight", upSlab, gateStride, compute.Q2_K},
		{"w2", "down_proj.weight", downSlab, downStride, compute.Q3_K},
	}
	for e := 0; e < E; e++ {
		for _, w := range wants {
			name := expertNameArch("deepseek41", 0, e, w.suffix)
			ck, ok := tier.staging(name)
			if !ok {
				t.Fatalf("staging declined %s; the cross-shard index lost a leaf", name)
			}
			if ck.dt != w.dt {
				t.Fatalf("%s: staging dtype = %s, want %s", name, ck.dt, w.dt)
			}
			if ck.bytes != w.stride {
				t.Fatalf("%s: staging bytes = %d, want one expert stride %d", name, ck.bytes, w.stride)
			}
			hb, ok := ck.mk().Buf().(compute.HostBuffer)
			if !ok {
				t.Fatalf("%s: staged tensor is not host-addressable", name)
			}
			i8 := hb.I8()
			want := w.slab[e*int(w.stride) : (e+1)*int(w.stride)]
			if len(i8) != len(want) {
				t.Fatalf("%s: staged %d bytes, want %d", name, len(i8), len(want))
				continue
			}
			got := make([]byte, len(i8))
			for i, v := range i8 {
				got[i] = byte(v)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("%s: faulted bytes are not this expert's slice of the shard that owns %s", name, w.suffix)
			}
		}
	}
}

// TestExpertCheckpointTierCrossShardPrefetchHandsOffOwningShardBytes covers the
// speculative path over the same split: Prefetch resolves each hint through
// t.shards[entry.shard] and the settled handoff must carry the owning shard's
// bytes, so a layer-entry miss set spanning both shard files reads the right
// ranges when the demand faults consume them.
// fak-test:runtime fast est=50ms lane=default
func TestExpertCheckpointTierCrossShardPrefetchHandsOffOwningShardBytes(t *testing.T) {
	const E, H = 4, 256
	downStride := int64(H * (H / qkK) * q3kBlockBytes)
	gateStride := int64(H * (H / qkK) * q2kBlockBytes)
	patterned := func(n, salt int) []byte {
		b := make([]byte, n)
		per := n / E
		for i := range b {
			b[i] = byte((salt*197 + (i/per)*131 + i*7 + 17) & 0xff)
		}
		return b
	}
	downSlab := patterned(int(downStride)*E, 11)
	gateSlab := patterned(int(gateStride)*E, 12)
	upSlab := patterned(int(gateStride)*E, 13)

	var shardB []byte
	shardB = append(shardB, gateSlab...)
	shardB = append(shardB, upSlab...)

	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(bytes.NewReader(downSlab), int64(len(downSlab)), []FusedExpertTensor{{
		Name: "blk.0.ffn_down_exps.weight", Layer: 0, Proj: "down_proj", Arch: "deepseek41",
		Quant: ExpertCheckpointQ3K, Offset: 0, Experts: E, Rows: H, Cols: H,
	}}); err != nil {
		t.Fatalf("AddShard (down shard): %v", err)
	}
	if err := tier.AddShard(bytes.NewReader(shardB), int64(len(shardB)), []FusedExpertTensor{
		{Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj", Arch: "deepseek41",
			Quant: ExpertCheckpointQ2K, Offset: 0, Experts: E, Rows: H, Cols: H},
		{Name: "blk.0.ffn_up_exps.weight", Layer: 0, Proj: "up_proj", Arch: "deepseek41",
			Quant: ExpertCheckpointQ2K, Offset: int64(len(gateSlab)), Experts: E, Rows: H, Cols: H},
	}); err != nil {
		t.Fatalf("AddShard (gate/up shard): %v", err)
	}

	names := make([]string, 0, E*3)
	for e := 0; e < E; e++ {
		for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
			names = append(names, expertNameArch("deepseek41", 0, e, suffix))
		}
	}
	started, join := tier.prefetchJoined(names)
	if started != len(names) {
		t.Fatalf("Prefetch started %d reads, want %d (every indexed leaf)", started, len(names))
	}
	join()

	for e := 0; e < E; e++ {
		cases := []struct {
			suffix string
			slab   []byte
			stride int64
		}{
			{"gate_proj.weight", gateSlab, gateStride},
			{"up_proj.weight", upSlab, gateStride},
			{"down_proj.weight", downSlab, downStride},
		}
		for _, c := range cases {
			name := expertNameArch("deepseek41", 0, e, c.suffix)
			w, err := tier.fault(name)
			if err != nil {
				t.Fatalf("fault(%s): %v", name, err)
			}
			var raw []byte
			if w.q4 != nil {
				raw = w.q4.raw
			} else if w.kq != nil {
				raw = w.kq.raw
			}
			want := c.slab[e*int(c.stride) : (e+1)*int(c.stride)]
			if !bytes.Equal(raw, want) {
				t.Errorf("%s: prefetched handoff does not carry this expert's bytes from the owning shard", name)
			}
		}
	}
}
