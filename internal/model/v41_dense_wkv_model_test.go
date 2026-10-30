package model

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type packedWKVModelBackend struct {
	compute.Backend
	qualified               bool
	uploads, calls, retired int
	rows                    []int
	fail                    bool
	freePanic               bool
}

var errPackedWKVModelSelected = errors.New("selected native fault")

type packedWKVModelWeight struct{ b *packedWKVModelBackend }

func (w *packedWKVModelWeight) Release() error            { w.b.retired++; return nil }
func (b *packedWKVModelBackend) Caps() compute.Caps       { return compute.Caps{DeviceMemory: true} }
func (b *packedWKVModelBackend) V41FP8WKVSupported() bool { return true }
func (b *packedWKVModelBackend) V41FP8WKVQualified() bool { return b.qualified }
func (b *packedWKVModelBackend) UploadV41FP8WKV(w, s []byte) (compute.V41FP8WKVWeights, error) {
	b.uploads++
	return &packedWKVModelWeight{b}, nil
}
func (b *packedWKVModelBackend) Upload(t compute.Tensor, dt compute.Dtype) compute.Tensor {
	return compute.NewF32(b, t.Shape, b.Backend.Read(t))
}
func (b *packedWKVModelBackend) V41FP8WKV(w compute.V41FP8WKVWeights, x compute.Tensor, rows int) (compute.Tensor, error) {
	b.calls++
	b.rows = append(b.rows, rows)
	if b.fail {
		return compute.Tensor{}, errPackedWKVModelSelected
	}
	in := b.Backend.Read(x)
	out := make([]float32, rows*512)
	for r := 0; r < rows; r++ {
		for c := 0; c < 512; c++ {
			out[r*512+c] = in[r*5120]
		}
	}
	y := compute.NewF32(b, []int{rows, 512}, out)
	y.Dtype = compute.BF16
	return y, nil
}
func (b *packedWKVModelBackend) Free(t compute.Tensor) {
	if b.freePanic {
		panic("secondary free fault")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41PackedWKVModelConsumer(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	sourceCfg := cfg
	metadata := *cfg.DeepSeekV41
	cfg.DeepSeekV41 = &metadata
	cfg.DeepSeekV41.NumericalProfile = &DeepSeekV41NumericalProfileRequest{Semantics: "unqualified-test"}
	m := &Model{Cfg: cfg}
	b := &packedWKVModelBackend{Backend: compute.Default()}
	if err := m.InstallV41DenseWKV(0, nil, V41DenseWKVPayloadBytes, b); err == nil {
		t.Fatal("unqualified install admitted")
	}
	b.qualified = true
	m.Cfg.DeepSeekV41.NumericalProfile.Semantics = " "
	if err := m.InstallV41DenseWKV(0, nil, V41DenseWKVPayloadBytes, b); err == nil {
		t.Fatal("empty named semantics admitted")
	}
	m.Cfg.DeepSeekV41.NumericalProfile.Semantics = "unqualified-test"
	// Fixture entries carry the exact on-disk names of the pinned checkpoint
	// shard, which is what InstallV41DenseWKV reads through the inventory.
	name, scale := v41DenseWKVNames(0)
	data := make([]byte, V41DenseWKVPayloadBytes)
	for i := 512 * 5120; i < len(data); i++ {
		data[i] = 127
	}
	p := filepath.Join(t.TempDir(), "wkv.safetensors")
	v41TestWriteEntries(t, p, map[string]stEntry{name: {Dtype: "F8_E4M3", Shape: []int{512, 5120}, DataOffsets: []int{0, 512 * 5120}}, scale: {Dtype: "F8_E8M0", Shape: []int{16, 160}, DataOffsets: []int{512 * 5120, len(data)}}}, data)
	shard, err := OpenV41PackedShard(p, sourceCfg)
	if err != nil {
		t.Fatal(err)
	}
	defer shard.Close()
	if err := m.InstallV41DenseWKV(0, shard, V41DenseWKVPayloadBytes, b); err != nil {
		t.Fatal(err)
	}
	if err := m.InstallV41DenseWKV(0, shard, V41DenseWKVPayloadBytes, b); err == nil {
		t.Fatal("duplicate replaced immutable pair")
	}
	if b.uploads != 0 {
		t.Fatal("install eagerly uploaded device weights")
	}
	if _, err := v41ForwardGeometry(cfg); err == nil {
		t.Fatal("named numerical refusal weakened")
	}
	if !m.holdModelWeights() {
		t.Fatal("weight hold failed")
	}
	s := &Session{M: m, Backend: b}
	project := s.v41DenseProjectionFunc()
	input := make([]float32, 257*5120)
	for r := 0; r < 257; r++ {
		input[r*5120] = float32(r + 1)
	}
	got, err := m.v41ProjPanelWithProjection(0, "attn.wkv.weight", input, 512, 5120, 257, project)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 257*512 || len(b.rows) != 3 || b.rows[0] != 128 || b.rows[1] != 128 || b.rows[2] != 128 || b.uploads != 1 {
		t.Fatalf("not persistent 128-row dispatch: rows=%v uploads=%d", b.rows, b.uploads)
	}
	for r := 0; r < 257; r++ {
		if got[r*512] != float32(r+1) {
			t.Fatalf("tail/order mismatch row %d", r)
		}
	}
	b.fail = true
	b.freePanic = true
	if _, err := m.v41ProjPanelWithProjection(0, "attn.wkv.weight", input[:5120], 512, 5120, 1, project); !errors.Is(err, errPackedWKVModelSelected) {
		t.Fatalf("selected fault replaced or fell back: %v", err)
	}
	b.freePanic = false
	if err := m.CloseWeights(); err == nil {
		t.Fatal("closed active session")
	}
	if b.retired != 0 {
		t.Fatal("premature retirement")
	}
	m.releaseWeightSession()
	if b.retired != 1 {
		t.Fatal("missing exactly-once model retirement")
	}
	if err := m.InstallV41DenseWKV(0, shard, V41DenseWKVPayloadBytes, b); err == nil {
		t.Fatal("closed owner resurrected")
	}
}
