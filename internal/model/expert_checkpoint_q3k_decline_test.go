package model

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_checkpoint_q3k_decline_test.go — the Q3_K ADMISSION witness for fak#13144.
//
// This file REPLACES the fak#13122 settlement witness that used to live here. That settlement
// declined Q3_K because staging().mk must return a compute.Tensor and no Q3_K compute kind existed.
// fak#13149 then built exactly that kind (compute.NewQ3K / compute.Q3_K, a verbatim host tensor),
// so the settlement's premise no longer holds and the tier admits Q3_K the same way it admits
// Q2_K — staged VERBATIM, with no f32 materialization of the expert bulk. The published
// DeepSeek-V4.1 Q2_K artifact stores its down projection as Q3_K, so admitting it is what lets the
// artifact reach the streamed forward instead of a refusal or a ~393 GiB eager-f32 OOM.
//
// The settled, load-bearing properties this pins:
//   - ExpertCheckpointQ3K is a real tier member with the Q3_K 256-weight/110-byte super-block;
//   - staging() admits a Q3_K slab and reports compute.Q3_K, keyed under kquant-raw:; and
//   - staging().mk stages the faulted expert's raw bytes VERBATIM as a compute.Q3_K host tensor.

// TestExpertCheckpointTierStagesQ3K is the Q3_K admission witness. A Q3_K fused slab is indexable,
// one expert faults back as a verbatim compute.Q3_K host tensor carrying its own raw bytes, and the
// staging descriptor reports the one-expert Q3_K stride as its resident byte cost.
func TestExpertCheckpointTierStagesQ3K(t *testing.T) {
	const H, E = 256, 4
	nblk := H / qkK
	stride := int64(H * nblk * q3kBlockBytes)

	expertBytes := func(e int) []byte {
		raw := make([]byte, stride)
		for i := range raw {
			raw[i] = byte((e*37 + i*11) & 0xff)
		}
		return raw
	}
	var blob []byte
	for e := 0; e < E; e++ {
		blob = append(blob, expertBytes(e)...)
	}

	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{
		Name: "blk.0.ffn_down_exps.weight", Layer: 0, Proj: "down_proj",
		Quant: ExpertCheckpointQ3K, Offset: 0, Experts: E, Rows: H, Cols: H,
	}}); err != nil {
		t.Fatalf("AddShard over a well-formed Q3_K slab: %v", err)
	}

	name := expertName(0, 2, "down_proj.weight")
	ck, ok := tier.staging(name)
	if !ok {
		t.Fatalf("staging declined %s; the Q3_K slab must be admitted", name)
	}
	if ck.dt != compute.Q3_K {
		t.Fatalf("staging dtype = %s, want Q3_K (the verbatim host tensor kind from fak#13149)", ck.dt)
	}
	if ck.bytes != stride {
		t.Fatalf("staging resident bytes = %d, want one expert stride %d", ck.bytes, stride)
	}
	if got := ck.key; got != "kquant-raw:"+name {
		t.Fatalf("staging key = %q, want the kquant-raw ring key for %s", got, name)
	}

	// The staged tensor must carry THIS expert's raw bytes verbatim, as a Q3_K host tensor.
	tensor := ck.mk()
	if tensor.Dtype != compute.Q3_K {
		t.Fatalf("staged tensor dtype = %s, want Q3_K", tensor.Dtype)
	}
	hb, ok := tensor.Buf().(compute.HostBuffer)
	if !ok {
		t.Fatal("staged Q3_K tensor is not host-addressable; the CPU reference backend was not used")
	}
	want := expertBytes(2)
	i8 := hb.I8()
	if len(i8) != len(want) {
		t.Fatalf("staged Q3_K byte length = %d, want %d", len(i8), len(want))
	}
	got := make([]byte, len(want))
	for i, v := range i8 {
		got[i] = byte(v)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("staged Q3_K tensor did not carry expert 2's stride bytes verbatim")
	}

	st := tier.Stats()
	if st.Reads != 1 || st.BytesRead != stride {
		t.Fatalf("tier ledger reads=%d bytes=%d, want 1 read / %d bytes", st.Reads, st.BytesRead, stride)
	}
}

// TestExpertCheckpointQ3KMemberGeometry pins the enum member added by the reversal: Q3_K reports a
// real 256-weight/110-byte super-block geometry and names itself, so a descriptor cannot be built at
// the wrong block size. It also pins that the enum VALUE appended did not shift its neighbours — the
// four older members keep the exact values a serialized descriptor would carry.
func TestExpertCheckpointQ3KMemberGeometry(t *testing.T) {
	if ExpertCheckpointQ4K != 0 || ExpertCheckpointQ5K != 1 || ExpertCheckpointQ6K != 2 || ExpertCheckpointQ2K != 3 {
		t.Fatalf("existing enum values shifted: Q4K=%d Q5K=%d Q6K=%d Q2K=%d, want 0/1/2/3",
			ExpertCheckpointQ4K, ExpertCheckpointQ5K, ExpertCheckpointQ6K, ExpertCheckpointQ2K)
	}
	if ExpertCheckpointQ3K != 4 {
		t.Fatalf("Q3_K enum value = %d, want the next iota after Q2_K (4)", ExpertCheckpointQ3K)
	}
	if got := ExpertCheckpointQ3K.String(); got != "Q3_K" {
		t.Fatalf("Q3_K String() = %q, want Q3_K", got)
	}
	w, b, ok := ExpertCheckpointQ3K.blockGeometry()
	if !ok {
		t.Fatal("Q3_K blockGeometry declined; an admitted member must have geometry")
	}
	if w != qkK || b != q3kBlockBytes {
		t.Fatalf("Q3_K geometry = %d weights / %d bytes, want %d / %d", w, b, qkK, q3kBlockBytes)
	}
}

// TestExpertCheckpointQ3KIsDistinctFromVerbatimKQuants pins that Q3_K now stages verbatim, exactly
// like Q2_K: each reports its OWN compute dtype (Q2_K / Q3_K), never a shared fall-through. This is
// the fak#13149 convergence — the settlement's "stage Q3_K as something else" tension is gone,
// because a verbatim Q3_K host tensor kind now exists.
func TestExpertCheckpointQ3KIsDistinctFromVerbatimKQuants(t *testing.T) {
	if _, _, ok := ExpertCheckpointQ2K.blockGeometry(); !ok {
		t.Fatal("Q2_K geometry must exist")
	}
	q2 := expertCheckpointEntry{quant: ExpertCheckpointQ2K}
	q3 := expertCheckpointEntry{quant: ExpertCheckpointQ3K}
	if q2.dtype() != compute.Q2_K {
		t.Fatalf("Q2_K staging dtype = %s, want Q2_K (verbatim)", q2.dtype())
	}
	if q3.dtype() != compute.Q3_K {
		t.Fatalf("Q3_K staging dtype = %s, want Q3_K (verbatim host tensor, fak#13149)", q3.dtype())
	}
	if q2.halKey("n") != q3.halKey("n") {
		t.Fatalf("Q2_K and Q3_K ring keys differ (%q vs %q); both non-Q4_K arms share the kquant-raw prefix",
			q2.halKey("n"), q3.halKey("n"))
	}
}
