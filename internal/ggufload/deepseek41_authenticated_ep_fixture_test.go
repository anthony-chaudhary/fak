package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// Extend the existing mixed-quant fixture at the file boundary, not by mutating
// the loaded Model. H=I=256, one layer and eight tokens keep it compact; the
// 512-wide Q/KV path and separate HC4 sublayers select the full native graph.
// This is synthetic software input, never a published checkpoint qualification.
func ds41AuthenticatedEPShards(t *testing.T) ([]ShardIdentity, string) {
	t.Helper()
	raw := deepseek41ExpertHALGGUF(t)
	f, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWeightSource(f, bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	meta := f.Metadata
	for key, value := range map[string]uint64{
		"attention.kv_lora_rank": 512, "attention.qk_nope_head_dim": 448,
		"attention.qk_rope_head_dim": 64, "attention.value_length_mla": 512,
	} {
		meta["deepseek41."+key] = Value{Type: TypeUint64, Value: value}
	}
	var tensors []splitTensor
	for _, info := range f.Tensors {
		data, _, err := ws.TensorBytes(info.Name)
		if err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, splitTensor{info.Name, info.Dims, info.Type, data})
	}
	// The new dense leaves use small deterministic values; norms/scales are one.
	var seed uint64 = 73
	replace := func(name string, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		values := make([]float32, n)
		for i := range values {
			seed = seed*6364136223846793005 + 1442695040888963407
			values[i] = (float32(seed>>40)/float32(1<<24)*2 - 1) * .02
			if strings.Contains(name, "norm") || strings.Contains(name, "scale") {
				values[i] = 1
			} else if strings.Contains(name, "base") {
				values[i] = 0
			}
		}
		next := splitTensor{name, dims, TensorF32, f32Payload(values...)}
		for i := range tensors {
			if tensors[i].name == name {
				tensors[i] = next
				return
			}
		}
		tensors = append(tensors, next)
	}
	replace("blk.0.attn_q_b.weight", ds41HALQLora, ds41HALHeads*512)
	replace("blk.0.attn_q_a_norm.weight", ds41HALQLora)
	replace("blk.0.attn_kv.weight", ds41HALHidden, 512)
	replace("blk.0.attn_kv_a_norm.weight", 512)
	replace("blk.0.attn_output_a.weight", ds41HALHeads*512, ds41HALOutRank)
	replace("blk.0.mhc_mixes.weight", 4*ds41HALHidden, 24)
	replace("blk.0.hc_ffn_fn.weight", 4*ds41HALHidden, 24)
	replace("blk.0.hc_ffn_base.weight", 24)
	replace("blk.0.hc_ffn_scale.weight", 3)
	var shards [2][]splitTensor
	for _, tensor := range tensors {
		if tensor.name == "blk.0.exp_probs_b.bias" {
			bias := make([]float32, 384)
			// Interleave owners in the global pick order. Rank pre-sums must
			// not replace the existing exact per-pick reduction.
			for i, expert := range []int{258, 130, 2, 257, 129, 1} {
				bias[expert] = float32(1000 - i*10)
			}
			tensor.data = f32Payload(bias...)
		}
		if tensor.name == "blk.0.ffn_up_exps.weight" {
			// Gate/up have distinct byte identities, so recorder assertions
			// can name each projection without private model-map access.
			for off := 0; off < len(tensor.data); off += blockQ2KBytes {
				tensor.data[off] ^= 1
			}
		}
		shard := 0
		if tensor.name == "blk.0.ffn_down_exps.weight" {
			shard = 1
		}
		shards[shard] = append(shards[shard], tensor)
	}
	dir := t.TempDir()
	var manifest []ShardIdentity
	for shard, tensors := range shards {
		meta["split.no"] = Value{Type: TypeUint32, Value: uint32(shard)}
		meta["split.count"] = Value{Type: TypeUint32, Value: uint32(2)}
		meta["split.tensors.count"] = Value{Type: TypeUint32, Value: uint32(len(shards[0]) + len(shards[1]))}
		keys := make([]string, 0, len(meta))
		for key := range meta {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		var b bytes.Buffer
		writeMinimalHeader(&b, uint64(len(tensors)), uint64(len(meta)))
		for _, key := range keys {
			if key == "tokenizer.ggml.tokens" {
				tokens, ok := f.StringArray(key)
				if !ok {
					t.Fatal("fixture tokenizer must contain string tokens")
				}
				writeKVStringArray(&b, key, tokens)
			} else {
				writeMetaValueForTest(t, &b, key, meta[key])
			}
		}
		offset := 0
		for _, tensor := range tensors {
			writeTensorInfoForTest(&b, tensor.name, tensor.dims, tensor.typ, uint64(offset))
			offset = (offset + len(tensor.data) + 31) / 32 * 32
		}
		padToAlignment(&b, 32)
		for _, tensor := range tensors {
			b.Write(tensor.data)
			padToAlignment(&b, 32)
		}
		path := filepath.Join(dir, fmt.Sprintf("authenticated-ep-%05d-of-00002.gguf", shard+1))
		if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		// The fixture producer supplies trust from its known bytes before any
		// consumer opens the files, never from the consumer's observed hashes.
		manifest = append(manifest, ShardIdentity{path, int64(b.Len()), fmt.Sprintf("%x", sha256.Sum256(b.Bytes()))})
	}
	encoded, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	return manifest, fmt.Sprintf("%x", sha256.Sum256(encoded))
}
