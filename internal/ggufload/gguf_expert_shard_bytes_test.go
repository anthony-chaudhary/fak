package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// gguf_expert_shard_bytes_test.go — the REAL-BYTES sharded-load gate. The existing shard tests
// (TestQwen3MoEGGUFExpertShardLoadsOnlyResidentBand, TestGLMMoeDsaKQuantExpertShardLoadsOnlyResidentBand)
// prove WHICH expert tensors a WithExpertShard load admits, but on ZEROED Q4_K payloads — so they
// cannot tell whether the loader admitted the RIGHT band's bytes or merely the right COUNT of
// tensors. This gate closes that gap: it writes a DISTINCT per-expert byte pattern into each
// expert's raw Q4_K super-blocks, loads a mid-file band [2,4), and asserts the resident bytes of a
// band expert equal THAT expert's source pattern (not rank 0's) — the fidelity a sharded serve
// (#971) depends on, since a rank that silently loaded expert 0's bytes under expert 2's name would
// serve wrong tokens while still passing every count/presence check.

// qwen3MoEExpertGGUFDistinctQ4K writes the qwen3moe expert GGUF with each expert's Q4_K super-block
// region filled with a distinct, recoverable per-expert byte pattern (expert x -> bytes keyed on x),
// so a loaded band's resident bytes can be matched back to the exact source expert. Non-expert
// tensors keep the standard F32 fill.
func qwen3MoEExpertGGUFDistinctQ4K(E, I, H int) []byte {
	tensors := []qwen3MoETestTensor{
		{name: "blk.0.ffn_down_exps.weight", dims: []uint64{uint64(I), uint64(H), uint64(E)}, typ: TensorQ4_K},
		{name: "blk.0.ffn_gate_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: TensorQ4_K},
		{name: "blk.0.ffn_gate_inp.weight", dims: []uint64{uint64(H), uint64(E)}, typ: TensorF32, data: sequenceF32ForTest(300, E*H)},
		{name: "blk.0.ffn_up_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: TensorQ4_K},
	}
	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		off += qwen3MoEPayloadBytes(tt.typ, tt.dims)
		off = (off + 31) / 32 * 32
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 10)
	writeKVString(&b, "general.architecture", "qwen3moe")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "qwen3moe.embedding_length", uint32(H))
	writeKVUint32(&b, "qwen3moe.block_count", 1)
	writeKVUint32(&b, "qwen3moe.attention.head_count", 1)
	writeKVUint32(&b, "qwen3moe.feed_forward_length", uint32(maxForTest(I*2, I)))
	writeKVFloat32(&b, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32(&b, "qwen3moe.expert_count", uint32(E))
	writeKVUint32(&b, "qwen3moe.expert_used_count", 1)
	writeKVUint32(&b, "qwen3moe.expert_feed_forward_length", uint32(I))
	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, tt.typ, offsets[i])
	}
	padToAlignment(&b, 32)
	for _, tt := range tensors {
		switch tt.typ {
		case TensorF32:
			for _, v := range tt.data {
				writeF32ForTest(&b, v)
			}
		case TensorQ4_K:
			// The batched [.,.,E] blob is E contiguous per-expert regions. Fill expert x's region
			// with a byte pattern keyed on x so the loaded band's bytes are recoverable per-expert.
			total := qwen3MoEPayloadBytes(tt.typ, tt.dims)
			per := total / E
			payload := make([]byte, total)
			for x := 0; x < E; x++ {
				for i := 0; i < per; i++ {
					payload[x*per+i] = expertPatternByte(x, i)
				}
			}
			b.Write(payload)
		default:
			panic("unsupported qwen3moe test tensor type")
		}
		padToAlignment(&b, 32)
	}
	return b.Bytes()
}

// expertPatternByte is the distinct, recoverable per-expert byte at offset i: keyed on BOTH the
// expert index and the offset so expert 2's region can never be confused with expert 0's.
func expertPatternByte(expert, i int) byte {
	return byte((expert*131 + i*7 + 17) & 0xff)
}

func TestQwen3MoEGGUFExpertShardLoadsCorrectBandBytes(t *testing.T) {
	const E, I, H = 4, 256, 256
	path := filepath.Join(t.TempDir(), "qwen3moe_q4k_shard_bytes.gguf")
	if err := os.WriteFile(path, qwen3MoEExpertGGUFDistinctQ4K(E, I, H), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	// Rank 1 of 2 owns the mid-file band [2,4) — a band that does NOT start at expert 0, so a loader
	// that mistakenly kept the first experts' bytes would be caught by the byte check below.
	shard, err := ExpertShardForRank(E, 2, 1)
	if err != nil {
		t.Fatalf("ExpertShardForRank: %v", err)
	}
	if shard.Lo != 2 || shard.Hi != 4 {
		t.Fatalf("rank-1 shard = [%d,%d), want [2,4)", shard.Lo, shard.Hi)
	}
	m, err := LoadModelQ4KProfileOptions(path, nil, WithExpertShard(shard.Lo, shard.Hi))
	if err != nil {
		t.Fatalf("LoadModelQ4KProfileOptions: %v", err)
	}

	// Load the FULL model too, to recover each expert's expected resident bytes from the same
	// fixture through the same loader — the ground truth the sharded band must reproduce.
	fullPath := filepath.Join(t.TempDir(), "qwen3moe_q4k_full_bytes.gguf")
	if err := os.WriteFile(fullPath, qwen3MoEExpertGGUFDistinctQ4K(E, I, H), 0o644); err != nil {
		t.Fatalf("write full fixture: %v", err)
	}
	full, err := LoadModelQ4KProfile(fullPath, nil)
	if err != nil {
		t.Fatalf("LoadModelQ4KProfile (full): %v", err)
	}

	for x := 0; x < E; x++ {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			name := "model.layers.0.mlp.experts." + itoaForTest(x) + "." + proj + ".weight"
			inBand := x >= shard.Lo && x < shard.Hi
			shardRaw, ok := m.Q4KRaw(name)
			if ok != inBand {
				t.Fatalf("sharded Q4KRaw(%q) present=%v, want %v for band [%d,%d)", name, ok, inBand, shard.Lo, shard.Hi)
			}
			if !inBand {
				continue
			}
			// The band expert's resident bytes must equal the SAME expert's bytes from the full load
			// (not, say, expert 0's) — proving the shard admitted the correct band, byte-for-byte.
			wantRaw, ok := full.Q4KRaw(name)
			if !ok {
				t.Fatalf("full load missing Q4KRaw(%q)", name)
			}
			if !bytes.Equal(shardRaw, wantRaw) {
				t.Fatalf("sharded expert %q resident bytes != full-load bytes (len %d vs %d) — shard admitted the WRONG band", name, len(shardRaw), len(wantRaw))
			}
			// And the bytes must be the distinct pattern for THIS expert, not another's: check the
			// first block against expert x's source pattern to catch an off-by-one band mapping.
			if len(shardRaw) == 0 || shardRaw[0] != expertPatternByte(x, 0) {
				got := byte(0)
				if len(shardRaw) > 0 {
					got = shardRaw[0]
				}
				t.Fatalf("sharded expert %q first resident byte = %d, want expert-%d pattern %d", name, got, x, expertPatternByte(x, 0))
			}
		}
	}
	t.Logf("sharded load band [2,4): resident expert bytes match the correct source expert byte-for-byte (E=%d, real non-zero Q4_K payload)", E)
}
