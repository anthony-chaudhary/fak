package computebuild

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"testing"
)

// This inert word-stream is independent of the production parser, shader, and
// expected-contract constructor. It is not a valid executable shader and never
// establishes source correspondence, scalar arithmetic results, or GPU support.
func vulkanV5StructuralFixture() []byte {
	words := []uint32{
		0x07230203, 0x00010500, 0, 64, 0,
		2<<16 | 17, 1,
		2<<16 | 17, 4464,
		2<<16 | 17, 4466,
		2<<16 | 17, 4467,
		7<<16 | 10, 0x5f565053, 0x5f52484b, 0x616f6c66, 0x6f635f74, 0x6f72746e, 0x0000736c,
		5<<16 | 15, 5, 1, 0x6e69616d, 0,
		6<<16 | 16, 1, 17, 1, 1, 1,
		4<<16 | 16, 1, 4459, 32,
		4<<16 | 16, 1, 4461, 32,
		4<<16 | 16, 1, 4462, 32,
		3<<16 | 22, 2, 32,
		3<<16 | 71, 10, 42,
		3<<16 | 71, 11, 42,
		3<<16 | 71, 12, 42,
		3<<16 | 71, 13, 42,
		5<<16 | 133, 2, 10, 3, 4,
		5<<16 | 129, 2, 11, 5, 10,
		5<<16 | 133, 2, 12, 11, 6,
		5<<16 | 129, 2, 13, 7, 12,
	}
	return vulkanV5WordBytes(words)
}

func vulkanV5WordBytes(words []uint32) []byte {
	raw := make([]byte, 4*len(words))
	for i, word := range words {
		binary.LittleEndian.PutUint32(raw[i*4:], word)
	}
	return raw
}

// fak-test:runtime fast est=20ms lane=default
// Estimate is unmeasured; these checks execute only the static Go parser.
func TestVulkanV5IndexerScoreStructuralControls(t *testing.T) {
	raw := vulkanV5StructuralFixture()
	got, err := inspectVulkanIndexerScoreSPIRV(raw)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	want := VulkanIndexerScoreContract{
		PolicyID: "fak.v41-indexer-score-f32-structure.v1", ModuleSHA256: hex.EncodeToString(digest[:]),
		EntryPoint: "main", LocalSize: [3]uint32{1, 1, 1}, FloatWidth: 32,
		DenormPreserve: true, SignedZeroInfNanPreserve: true, RoundingModeRTE: true,
		NoContractionFAddCount: 2, NoContractionFMulCount: 2,
	}
	if got != want {
		t.Fatalf("structural proof = %+v; want %+v", got, want)
	}
	words := make([]uint32, len(raw)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(raw[i*4:])
	}
	// Locate an instruction independently by opcode and optional third operand.
	locate := func(w []uint32, op, third uint32) int {
		for p := 5; p < len(w); p += int(w[p] >> 16) {
			if w[p]&0xffff == op && (third == 0 || w[p+2] == third) {
				return p
			}
		}
		t.Fatalf("fixture opcode %d/%d absent", op, third)
		return 0
	}
	for _, tc := range []struct {
		name string
		edit func([]uint32) []uint32
	}{
		{"missing extension", func(w []uint32) []uint32 { p := locate(w, 10, 0); return append(w[:p], w[p+7:]...) }},
		{"missing denorm", func(w []uint32) []uint32 { p := locate(w, 16, 4459); return append(w[:p], w[p+4:]...) }},
		{"missing signed zero", func(w []uint32) []uint32 { p := locate(w, 16, 4461); return append(w[:p], w[p+4:]...) }},
		{"missing RTE", func(w []uint32) []uint32 { p := locate(w, 16, 4462); return append(w[:p], w[p+4:]...) }},
		{"wrong width", func(w []uint32) []uint32 { w[locate(w, 16, 4459)+3] = 16; return w }},
		{"wrong entry", func(w []uint32) []uint32 { w[locate(w, 16, 4462)+1] = 2; return w }},
		{"missing capability", func(w []uint32) []uint32 { return append(w[:7], w[9:]...) }},
		{"duplicate mode", func(w []uint32) []uint32 { return append(w, 4<<16|16, 1, 4462, 32) }},
		{"flush to zero", func(w []uint32) []uint32 { return append(w, 4<<16|16, 1, 4460, 32) }},
		{"round to zero", func(w []uint32) []uint32 { return append(w, 4<<16|16, 1, 4463, 32) }},
		{"local size", func(w []uint32) []uint32 { w[locate(w, 16, 17)+3] = 32; return w }},
		{"missing NoContraction", func(w []uint32) []uint32 { p := locate(w, 71, 42); return append(w[:p], w[p+3:]...) }},
		{"orphan NoContraction", func(w []uint32) []uint32 { w[locate(w, 71, 42)+1] = 14; return w }},
		{"double precision", func(w []uint32) []uint32 { w[locate(w, 22, 0)+2] = 64; return w }},
		{"float vector", func(w []uint32) []uint32 { return append(w, 4<<16|23, 30, 2, 2) }},
		{"Fma ext inst", func(w []uint32) []uint32 { return append(w, 8<<16|12, 2, 14, 20, 50, 3, 4, 5) }},
		{"FNegate", func(w []uint32) []uint32 { return append(w, 4<<16|127, 2, 14, 3) }},
		{"QuantizeToF16", func(w []uint32) []uint32 { return append(w, 4<<16|116, 2, 14, 3) }},
		{"specialized QuantizeToF16", func(w []uint32) []uint32 { return append(w, 5<<16|52, 2, 14, 116, 3) }},
		{"group FAdd", func(w []uint32) []uint32 { return append(w, 6<<16|265, 2, 14, 3, 0, 4) }},
		{"subgroup FMul", func(w []uint32) []uint32 { return append(w, 6<<16|352, 2, 14, 3, 0, 4) }},
		{"rounding decoration", func(w []uint32) []uint32 { return append(w, 4<<16|71, 10, 39, 0) }},
		{"fastmath decoration", func(w []uint32) []uint32 { return append(w, 4<<16|71, 10, 40, 1) }},
		{"relaxed precision", func(w []uint32) []uint32 { return append(w, 3<<16|71, 10, 0) }},
		{"group decoration", func(w []uint32) []uint32 { return append(w, 2<<16|73, 30) }},
		{"ID mode", func(w []uint32) []uint32 { return append(w, 6<<16|331, 1, 38, 2, 2, 2) }},
		{"malformed count", func(w []uint32) []uint32 { w[5] = 17; return w }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := tc.edit(append([]uint32(nil), words...))
			if _, err := inspectVulkanIndexerScoreSPIRV(vulkanV5WordBytes(changed)); err == nil {
				t.Fatal("invalid structural control admitted")
			}
		})
	}
	// Integer vectors/SNegate do not weaken binary32 arithmetic constraints.
	integer := append(append([]uint32(nil), words...), 4<<16|21, 30, 32, 1, 4<<16|23, 31, 30, 3, 4<<16|126, 30, 32, 33)
	if _, err := inspectVulkanIndexerScoreSPIRV(vulkanV5WordBytes(integer)); err != nil {
		t.Fatalf("integer control: %v", err)
	}
}
