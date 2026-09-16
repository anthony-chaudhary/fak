package model

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_checkpoint_q2k_test.go — the Q2_K staging witness for fak#13121.
//
// The DeepSeek-V4.1 Flash routed-expert slate is published Q2_K, which the checkpoint tier did not
// stage before (only Q4_K/Q5_K/Q6_K). This pins that a Q2_K fused slab is now indexable, that one
// expert's stride faults back as the SAME Q2_K kQuantTensor the resident path would carry, and that
// the staging descriptor reports the compute.Q2_K dtype with an 84-byte super-block stride.
func TestExpertCheckpointTierStagesQ2K(t *testing.T) {
	const H, E = 256, 4
	nblk := H / qkK
	stride := int64(H * nblk * q2kBlockBytes)

	// Fill one expert's stride with a deterministic pattern so a faulted expert is distinguishable
	// from its neighbours. Unlike the Q4_K fixture (which reuses the resident model's own bytes),
	// this test's comparable quantity is the raw stride bytes themselves.
	expertBytes := func(e int) []byte {
		raw := make([]byte, stride)
		for i := range raw {
			raw[i] = byte((e*31 + i*7) & 0xff)
		}
		return raw
	}
	var blob []byte
	for e := 0; e < E; e++ {
		blob = append(blob, expertBytes(e)...)
	}

	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
		Quant: ExpertCheckpointQ2K, Offset: 0, Experts: E, Rows: H, Cols: H,
	}}); err != nil {
		t.Fatalf("AddShard over a well-formed Q2_K slab: %v", err)
	}

	name := expertName(0, 2, "gate_proj.weight")
	ck, ok := tier.staging(name)
	if !ok {
		t.Fatalf("staging declined %s; the Q2_K slab was not indexed", name)
	}
	if ck.dt != compute.Q2_K {
		t.Fatalf("staging dtype = %s, want Q2_K", ck.dt)
	}
	if ck.bytes != stride {
		t.Fatalf("staging resident bytes = %d, want one expert stride %d", ck.bytes, stride)
	}
	if got := ck.key; got != "kquant-raw:"+name {
		t.Fatalf("staging key = %q, want the kquant-raw ring key for %s", got, name)
	}

	// The staged tensor must carry the faulted expert's own bytes, as a Q2_K host tensor.
	tensor := ck.mk()
	if tensor.Dtype != compute.Q2_K {
		t.Fatalf("staged tensor dtype = %s, want Q2_K", tensor.Dtype)
	}
	hb, ok := tensor.Buf().(compute.HostBuffer)
	if !ok {
		t.Fatal("staged Q2_K tensor is not host-addressable; the CPU reference backend was not used")
	}
	got := make([]byte, len(expertBytes(2)))
	i8 := hb.I8()
	if len(i8) != len(got) {
		t.Fatalf("staged Q2_K byte length = %d, want %d", len(i8), len(got))
	}
	for i, v := range i8 {
		got[i] = byte(v)
	}
	if !bytes.Equal(got, expertBytes(2)) {
		t.Fatal("staged Q2_K tensor did not carry expert 2's stride bytes")
	}

	st := tier.Stats()
	if st.Reads != 1 || st.BytesRead != stride {
		t.Fatalf("tier ledger reads=%d bytes=%d, want 1 read / %d bytes", st.Reads, st.BytesRead, stride)
	}
}
