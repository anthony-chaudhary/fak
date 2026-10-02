package model

import (
	"encoding/binary"
	"testing"
)

// TestPrismHadamardChangesResidentProjection checks the Bonsai-2 execution
// seam, rather than only parsing its GGUF declaration. For a four-lane block,
// H4(signs*x) of [1,2,3,4] with signs [1,-1,1,1] is [3,1,-4,2]. The packed
// ternary row [0.5,0,-0.5,0] must therefore yield 3.5, versus -1 without
// the declared activation transform.
func TestPrismHadamardChangesResidentProjection(t *testing.T) {
	const name = "model.layers.0.mlp.down_proj.weight"
	raw := make([]byte, q2G128BlockBytes)
	binary.LittleEndian.PutUint16(raw[:2], 0x3800) // f16 0.5
	for i := 2; i < len(raw); i++ {
		raw[i] = 0x55 // four zero-valued trits per byte
	}
	raw[2] = 0x46 // low-to-high trits 2,1,0,1 => 0.5,0,-0.5,0
	m := &Model{q2w: map[string]*q2Tensor{name: wrapQ2G128FromRaw(raw, 1, 128)}}
	if !m.hasWeight(name) {
		t.Fatal("resident PQ2-only projection must be visible to model weight routing")
	}
	x := make([]float32, 128)
	copy(x, []float32{1, 2, 3, 4})
	if got := m.residentMatRows(name, x, 1, 128)[0]; got != -1 {
		t.Fatalf("unrotated projection = %g, want -1", got)
	}

	signs := make([]int, 128)
	for i := range signs {
		signs[i] = 1
	}
	signs[1] = -1
	if err := m.SetPrismHadamard(PrismHadamardSpec{
		BlockSize:   4,
		SignWidths:  []int{128},
		SignValues:  signs,
		WeightNames: []string{name},
	}); err != nil {
		t.Fatalf("SetPrismHadamard: %v", err)
	}
	if got := m.residentMatRows(name, x, 1, 128)[0]; got != 3.5 {
		t.Fatalf("rotated projection = %g, want 3.5 from explicit signed H4", got)
	}
	if x[0] != 1 || x[1] != 2 || x[2] != 3 || x[3] != 4 {
		t.Fatalf("projection mutated caller activation: %v", x[:4])
	}
}

// The inverse embedding transform must run after bounded PQ2 row decoding.
// H4*[0.5,0,0,0] is [0.25,0.25,0.25,0.25]; the second lane then gets
// its declared negative sign. This also catches a no-op metadata attachment.
func TestPrismHadamardInvertsPackedPQ2EmbeddingRow(t *testing.T) {
	const weight = "model.layers.0.mlp.down_proj.weight"
	raw := make([]byte, q2G128BlockBytes)
	binary.LittleEndian.PutUint16(raw[:2], 0x3800) // f16 0.5
	for i := 2; i < len(raw); i++ {
		raw[i] = 0x55 // all zero trits
	}
	raw[2] = 0x56 // first four trits: 2,1,1,1 => [0.5,0,0,0]
	embed, err := NewPQ2Embedding(raw, 1, 128)
	if err != nil {
		t.Fatal(err)
	}
	b := NewQuantBuilder(Config{ModelType: "qwen35", HiddenSize: 128, IntermediateSize: 128, NumLayers: 1}, false)
	if err := b.SetQ2KEmbedding(embed); err != nil {
		t.Fatal(err)
	}
	if err := b.AddResidentQ2(weight, []int{1, 128}, raw); err != nil {
		t.Fatal(err)
	}
	m, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	row := make([]float32, 128)
	if err := m.Q2KEmbedding.GatherRow(0, row, 1); err != nil {
		t.Fatal(err)
	}
	if row[0] != 0.5 || row[1] != 0 || row[2] != 0 || row[3] != 0 {
		t.Fatalf("unrotated packed row = %v, want [0.5 0 0 0]", row[:4])
	}
	signs := make([]int, 128)
	for i := range signs {
		signs[i] = 1
	}
	signs[1] = -1
	if err := m.SetPrismHadamard(PrismHadamardSpec{
		BlockSize:    4,
		SignWidths:   []int{128},
		SignValues:   signs,
		WeightNames:  []string{weight},
		InverseNames: []string{"model.embed_tokens.weight"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := m.Q2KEmbedding.GatherRow(0, row, 1); err != nil {
		t.Fatal(err)
	}
	if row[0] != 0.25 || row[1] != -0.25 || row[2] != 0.25 || row[3] != 0.25 {
		t.Fatalf("inverse-transformed packed row = %v, want [0.25 -0.25 0.25 0.25]", row[:4])
	}
	if m.Q2KEmbedding.Bytes() != q2G128BlockBytes {
		t.Fatalf("packed embedding expanded: bytes=%d, want %d", m.Q2KEmbedding.Bytes(), q2G128BlockBytes)
	}
}
