package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func writeV4Shard(t *testing.T, path string, tensors map[string][]float32) {
	t.Helper()
	h := make(map[string]any, len(tensors))
	data := make([]byte, 0)
	for name, vals := range tensors {
		start := len(data)
		for _, v := range vals {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			data = append(data, b[:]...)
		}
		h[name] = map[string]any{"dtype": "F32", "shape": []int{1, len(vals)}, "data_offsets": []int{start, len(data)}}
	}
	hb, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	blob := make([]byte, 8)
	binary.LittleEndian.PutUint64(blob, uint64(len(hb)))
	blob = append(blob, hb...)
	blob = append(blob, data...)
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestV4ShardedExpertSourceRoutesSelectedTensors(t *testing.T) {
	dir := t.TempDir()
	n := func(e int, w string) string {
		return "model.layers.0.ffn.experts." + string(rune('0'+e)) + "." + w + ".weight"
	}
	a1, a2, a3 := n(1, "w1"), n(1, "w2"), n(1, "w3")
	b1, b2, b3 := n(2, "w1"), n(2, "w2"), n(2, "w3")
	writeV4Shard(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{a1: {1}, a3: {3}, b2: {20}})
	writeV4Shard(t, filepath.Join(dir, "b.safetensors"), map[string][]float32{a2: {2}, b1: {10}, b3: {30}})
	wm := map[string]string{a1: "a.safetensors", a2: "b.safetensors", a3: "a.safetensors", b1: "b.safetensors", b2: "a.safetensors", b3: "b.safetensors", "model.embed_tokens.weight": "b.safetensors"}
	ib, _ := json.Marshal(map[string]any{"weight_map": wm})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}

	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.openCount != 0 || s.readCount != 0 {
		t.Fatalf("constructor performed shard IO: opens=%d reads=%d", s.openCount, s.readCount)
	}
	batch, err := s.readV4ExpertBatch(0, []int{2, 1}, 24)
	if err != nil {
		t.Fatal(err)
	}
	if batch.Plan.Bytes != 24 || len(batch.Tensors) != 6 || s.readCount != 6 {
		t.Fatalf("plan=%+v tensors=%d reads=%d", batch.Plan, len(batch.Tensors), s.readCount)
	}
	if len(s.open) > 1 || s.openCount < 2 {
		t.Fatalf("bounded handles violated: resident=%d cumulative=%d", len(s.open), s.openCount)
	}
	want := map[string]float32{a1: 1, a2: 2, a3: 3, b1: 10, b2: 20, b3: 30}
	for _, x := range batch.Tensors {
		if len(x.Bytes) != 4 || math.Float32frombits(binary.LittleEndian.Uint32(x.Bytes)) != want[x.Name] {
			t.Fatalf("bad tensor %s", x.Name)
		}
	}
}

func TestV4ShardedExpertStagerReadsAcrossShardBoundary(t *testing.T) {
	dir := t.TempDir()
	w1 := "model.layers.0.ffn.experts.1.w1.weight"
	w2 := "model.layers.0.ffn.experts.1.w2.weight"
	w3 := "model.layers.0.ffn.experts.1.w3.weight"
	writeV4Shard(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{w1: {1, 0}, w3: {3, 0}})
	writeV4Shard(t, filepath.Join(dir, "b.safetensors"), map[string][]float32{w2: {2, 0}})
	ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{w1: "a.safetensors", w2: "b.safetensors", w3: "a.safetensors"}})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	plan, err := s.planV4ExpertBatch(0, []int{1}, 24)
	if err != nil {
		t.Fatal(err)
	}
	be := compute.Default()
	stager, err := newV4ShardedExpertStager(s, newPagedRing(be, 16), plan, compute.F32, func(tensor v4ExpertTensor) (compute.Tensor, error) {
		vals := make([]float32, len(tensor.Bytes)/4)
		for i := range vals {
			vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Bytes[i*4:]))
		}
		return compute.NewF32(be, tensor.Shape, vals), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var input [4]byte
	binary.LittleEndian.PutUint32(input[:], math.Float32bits(2))
	got, err := stager.matMul(w1, be.Upload(compute.NewF32(be, []int{2}, []float32{math.Float32frombits(binary.LittleEndian.Uint32(input[:])), 0}), compute.F32))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != 2 {
		t.Fatalf("matmul=%v want [2]", got)
	}
	if stager.Stats().SourceReads != 1 || s.readCount != 1 {
		t.Fatalf("stager=%+v source reads=%d", stager.Stats(), s.readCount)
	}
}

func TestV4ShardedExpertSourceRejectsUnsafeIndexBeforeShardOpen(t *testing.T) {
	for _, shard := range []string{"../escape.safetensors", "missing.bin"} {
		t.Run(shard, func(t *testing.T) {
			dir := t.TempDir()
			ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{"model.layers.0.ffn.experts.1.w1.weight": shard}})
			os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600)
			if _, err := newV4ShardedExpertSource(dir, 1); err == nil {
				t.Fatal("expected refusal")
			}
		})
	}
}

func TestV4ShardedExpertSourceMissingIndexedTensorFailsBeforePayloadRead(t *testing.T) {
	dir := t.TempDir()
	name := "model.layers.0.ffn.experts.1.w1.weight"
	writeV4Shard(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{"model.layers.0.ffn.experts.2.w1.weight": {1}})
	ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{name: "a.safetensors"}})
	os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600)
	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err = s.readV4ExpertBatch(0, []int{1}, 4); err == nil {
		t.Fatal("expected missing tensor refusal")
	}
	if s.readCount != 0 {
		t.Fatalf("payload reads=%d", s.readCount)
	}
}
