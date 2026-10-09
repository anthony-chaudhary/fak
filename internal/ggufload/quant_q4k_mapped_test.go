package ggufload

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
	"unsafe"
)

const mappedFixtureRows = 8192

type mappedFixtureTensor struct {
	name  string
	canon string
	typ   TensorType
	dims  []uint64
	data  []byte
}

func mappedFixturePayload(seed byte, n int) []byte {
	payload := make([]byte, n)
	for i := range payload {
		payload[i] = byte((int(seed)*131 + i*7 + i>>9 + 17) & 0xff)
	}
	return payload
}

func writeMappedQ4KFixture(t *testing.T, path string) []mappedFixtureTensor {
	t.Helper()
	const align = 32
	tensors := []mappedFixtureTensor{
		{name: "blk.0.ffn_down.weight", canon: "model.layers.0.mlp.down_proj.weight", typ: TensorQ4_K, dims: []uint64{256, mappedFixtureRows}, data: mappedFixturePayload(0x41, mappedFixtureRows*blockQ4KBytes)},
		{name: "blk.0.ffn_up.weight", canon: "model.layers.0.mlp.up_proj.weight", typ: TensorQ4_K, dims: []uint64{256, mappedFixtureRows}, data: mappedFixturePayload(0x52, mappedFixtureRows*blockQ4KBytes)},
		{name: "blk.0.ffn_gate.weight", canon: "model.layers.0.mlp.gate_proj.weight", typ: TensorQ2_K, dims: []uint64{256, mappedFixtureRows}, data: mappedFixturePayload(0x63, mappedFixtureRows*blockQ2KBytes)},
		{name: "blk.0.attn_v.weight", typ: TensorF32, dims: []uint64{256, 1}, data: make([]byte, 256*4)},
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 9)
	writeKVString(&b, "general.architecture", "qwen2")
	writeKVString(&b, "general.name", "mapped-resident-fixture")
	writeKVUint32(&b, "general.alignment", align)
	writeKVUint32(&b, "qwen2.embedding_length", 256)
	writeKVUint32(&b, "qwen2.block_count", 0)
	writeKVUint32(&b, "qwen2.attention.head_count", 1)
	writeKVUint32(&b, "qwen2.attention.head_count_kv", 1)
	writeKVUint32(&b, "qwen2.feed_forward_length", 256)
	writeKVFloat32(&b, "qwen2.attention.layer_norm_rms_epsilon", 1e-6)
	var offset uint64
	for _, tensor := range tensors {
		writeTensorInfoForTest(&b, tensor.name, tensor.dims, tensor.typ, offset)
		offset += uint64(len(tensor.data))
		offset = (offset + align - 1) / align * align
	}
	padToAlignment(&b, align)
	for _, tensor := range tensors {
		b.Write(tensor.data)
		padToAlignment(&b, align)
	}
	b.Write(make([]byte, 64<<10))
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("write mapped fixture: %v", err)
	}
	return tensors
}

func mappedFixtureOptions() []Q4KLoadOption {
	return []Q4KLoadOption{WithDenseKQuantResident(false), WithDenseQ2KResident(true)}
}

func mappedFixtureRaw(t *testing.T, m interface {
	Q4KRaw(string) ([]byte, bool)
	KQuantRaw(string) ([]byte, bool)
}, tensor mappedFixtureTensor) []byte {
	t.Helper()
	if tensor.typ == TensorQ4_K {
		raw, ok := m.Q4KRaw(tensor.canon)
		if !ok {
			t.Fatalf("missing Q4_K tensor %s", tensor.canon)
		}
		return raw
	}
	raw, ok := m.KQuantRaw(tensor.canon)
	if !ok {
		t.Fatalf("missing k-quant tensor %s", tensor.canon)
	}
	return raw
}

func mappedSliceInside(inner, outer []byte) bool {
	if len(inner) == 0 || len(outer) == 0 {
		return false
	}
	outerStart := uintptr(unsafe.Pointer(&outer[0]))
	outerEnd := outerStart + uintptr(len(outer))
	innerStart := uintptr(unsafe.Pointer(&inner[0]))
	return innerStart >= outerStart && innerStart+uintptr(len(inner)) <= outerEnd
}

// fak-test:runtime fast est=300ms lane=default
func TestMappedResidentQ4KOwnsCheckpointUntilClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mapped.gguf")
	tensors := writeMappedQ4KFixture(t, path)
	owned, err := LoadModelQ4KProfileOptions(path, nil, mappedFixtureOptions()...)
	if err != nil {
		t.Fatalf("owned load: %v", err)
	}
	defer owned.CloseWeights()

	var source *WeightSource
	open := func(path string) (*WeightSource, error) {
		var err error
		source, err = OpenWeightsMapped(path)
		return source, err
	}
	mapped, err := loadModelQ4KMappedResidentContext(context.Background(), path, nil, open, mappedFixtureOptions()...)
	if err != nil {
		t.Fatalf("mapped load: %v", err)
	}
	for _, tensor := range tensors {
		if tensor.typ == TensorF32 {
			continue
		}
		ownedRaw := mappedFixtureRaw(t, owned, tensor)
		mappedRaw := mappedFixtureRaw(t, mapped, tensor)
		if !bytes.Equal(mappedRaw, tensor.data) || !bytes.Equal(mappedRaw, ownedRaw) {
			t.Fatalf("mapped bytes differ for %s", tensor.canon)
		}
		if source != nil && len(source.data) > 0 && !mappedSliceInside(mappedRaw, source.data) {
			t.Fatalf("%s is not backed by the mapped checkpoint", tensor.canon)
		}
	}
	if err := mapped.CloseWeights(); err != nil {
		t.Fatalf("CloseWeights: %v", err)
	}
	for _, tensor := range tensors {
		if tensor.typ == TensorQ4_K {
			if raw, _ := mapped.Q4KRaw(tensor.canon); len(raw) != 0 {
				t.Fatalf("%s remains reachable after CloseWeights", tensor.canon)
			}
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestMappedResidentQ4KRefusesSeparateLifetimeRoutesBeforeOpen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mapped.gguf")
	writeMappedQ4KFixture(t, path)
	for name, option := range map[string]Q4KLoadOption{
		"unbounded dense": WithStreamedDenseQ4K(true),
		"bounded dense":   WithStreamedDenseQ4KWorkingSet(1 << 20),
		"experts":         WithStreamedExperts(0),
		"shard":           WithExpertShard(0, 1),
	} {
		t.Run(name, func(t *testing.T) {
			opened := false
			open := func(path string) (*WeightSource, error) {
				opened = true
				return OpenWeightsMapped(path)
			}
			model, err := loadModelQ4KMappedResidentContext(context.Background(), path, nil, open, option)
			if model != nil {
				_ = model.CloseWeights()
			}
			if err == nil {
				t.Fatal("alternate lifetime route was admitted")
			}
			if opened {
				t.Fatal("refused option opened the checkpoint")
			}
		})
	}
}
