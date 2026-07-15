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

func writeV4RawShard(t *testing.T, path string, tensors map[string]tinySTTensor) {
	t.Helper()
	h := map[string]any{}
	data := []byte{}
	for name, x := range tensors {
		start := len(data)
		data = append(data, x.data...)
		h[name] = map[string]any{"dtype": x.dtype, "shape": x.shape, "data_offsets": []int{start, len(data)}}
	}
	hb, _ := json.Marshal(h)
	blob := make([]byte, 8)
	binary.LittleEndian.PutUint64(blob, uint64(len(hb)))
	blob = append(blob, hb...)
	blob = append(blob, data...)
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestV4ShardedExpertQuantStagerCrossShardCompanion(t *testing.T) {
	dir := t.TempDir()
	weight := "layers.0.ffn.experts.1.w1.weight"
	scale := "layers.0.ffn.experts.1.w1.scale"
	codes := make([]byte, 3072*3584)
	codes[0] = 0x21
	scales := make([]byte, 3072*224)
	for i := range scales {
		scales[i] = 127
	}
	writeV4RawShard(t, filepath.Join(dir, "w.safetensors"), map[string]tinySTTensor{weight: {dtype: "I8", shape: []int{3072, 3584}, data: codes}})
	writeV4RawShard(t, filepath.Join(dir, "s.safetensors"), map[string]tinySTTensor{scale: {dtype: "F8_E8M0", shape: []int{3072, 224}, data: scales}})
	ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{weight: "w.safetensors", scale: "s.safetensors"}})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	plan, err := s.planV4ExpertBatch(0, []int{1}, int64(len(codes)+len(scales)))
	if err != nil {
		t.Fatal(err)
	}
	be := compute.Default()
	stager, err := newV4ShardedExpertQuantStager(s, newPagedRing(be, int64(3072*7168*4)), plan)
	if err != nil {
		t.Fatal(err)
	}
	xv := make([]float32, 7168)
	xv[0] = 1
	got, err := stager.matMul(weight, be.Upload(compute.NewF32(be, []int{7168}, xv), compute.F32))
	if err != nil {
		t.Fatal(err)
	}
	decoded, shape, err := decodeV4ExpertQuant(weight, scale, stEntry{Dtype: "I8", Shape: []int{3072, 3584}}, stEntry{Dtype: "F8_E8M0", Shape: []int{3072, 224}}, codes, scales)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Float32frombits(binary.LittleEndian.Uint32(decoded))
	if len(got) != 3072 || got[0] != want || shape[1] != 7168 {
		t.Fatalf("got0=%v len=%d want=%v shape=%v", got[0], len(got), want, shape)
	}
	stats := stager.Stats()
	if stats.SourceReads != 2 || stats.SourceBytes != int64(len(codes)+len(scales)) || s.readCount != 2 || len(s.open) > 1 {
		t.Fatalf("stats=%+v source_reads=%d handles=%d", stats, s.readCount, len(s.open))
	}
}

func TestV4ShardedExpertQuantStagerMissingScaleFailsBeforeRingAdmission(t *testing.T) {
	dir := t.TempDir()
	weight := "layers.0.ffn.experts.1.w1.weight"
	scale := "layers.0.ffn.experts.1.w1.scale"
	writeV4RawShard(t, filepath.Join(dir, "w.safetensors"), map[string]tinySTTensor{weight: {dtype: "I8", shape: []int{1, 32}, data: make([]byte, 16)}})
	ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{weight: "w.safetensors", scale: "missing.safetensors"}})
	os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600)
	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err = s.planV4ExpertBatch(0, []int{1}, 17); err == nil {
		t.Fatal("expected missing shard refusal")
	}
	if s.readCount != 0 {
		t.Fatalf("payload reads=%d", s.readCount)
	}
}
