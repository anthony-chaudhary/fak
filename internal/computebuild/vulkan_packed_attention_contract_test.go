package computebuild

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"testing"
)

// Inert UNIT-ONLY instruction fixture, following vulkanV5StructuralFixture.
// Not an executable or compiler-produced shader; never write it as a .spv,
// qualify a module, register a pipeline, or infer shader/device correctness.
func packedAttentionStructuralFixture() []uint32 {
	w := []uint32{0x07230203, 0x10500, 0, 256, 0}
	add := func(op uint32, a ...uint32) { w = append(w, uint32(len(a)+1)<<16|op); w = append(w, a...) }
	for _, cap := range []uint32{1, 4464, 4466, 4467} {
		add(17, cap)
	}
	add(10, 0x5f565053, 0x5f52484b, 0x616f6c66, 0x6f635f74, 0x6f72746e, 0x0000736c)
	add(11, 70, 0x4c534c47, 0x6474732e, 0x3035342e, 0)
	add(14, 0, 1)
	add(15, 5, 1, 0x6e69616d, 0)
	add(16, 1, 17, 128, 1, 1)
	for _, mode := range []uint32{4459, 4461, 4462} {
		add(16, 1, mode, 32)
	}
	add(22, 2, 32)
	add(21, 3, 32, 1)
	add(21, 4, 32, 0)
	add(29, 5, 2)
	add(29, 6, 4)
	add(71, 5, 6, 4)
	add(71, 6, 6, 4)
	add(30, 10, 3, 3, 3, 3, 3, 2, 4, 4)
	add(71, 10, 2)
	for m := uint32(0); m < 8; m++ {
		add(72, 10, m, 35, m*4)
	}
	add(32, 11, 9, 10)
	add(59, 11, 12, 9)
	for slot := uint32(0); slot < 5; slot++ {
		st, ptr, v := 20+3*slot, 21+3*slot, 22+3*slot
		array := uint32(5)
		if slot == 1 || slot == 4 {
			array = 6
		}
		add(30, st, array)
		add(71, st, 2)
		add(72, st, 0, 35, 0)
		if slot < 3 {
			add(72, st, 0, 24)
		}
		if slot == 4 {
			add(72, st, 0, 25)
		}
		add(32, ptr, 12, st)
		add(59, ptr, v, 12)
		add(71, v, 33, slot)
		add(71, v, 34, 0)
	}
	add(43, 4, 43, 2048)
	add(43, 4, 44, 128)
	add(28, 40, 2, 43)
	add(28, 41, 3, 44)
	add(28, 42, 2, 44)
	add(32, 45, 4, 40)
	add(32, 46, 4, 41)
	add(32, 47, 4, 42)
	add(32, 48, 4, 4)
	add(32, 49, 4, 2)
	add(32, 50, 4, 3)
	// Two 2048-element arrays, three 128-element arrays, four scalars = 17936.
	for i, ptr := range []uint32{45, 45, 46, 46, 47, 48, 49, 49, 50} {
		add(59, ptr, uint32(60+i), 4)
	}
	for i, op := range []uint32{129, 131, 133, 136} {
		id := uint32(80 + i)
		add(71, id, 42)
		add(op, 2, id, 90, 91)
	}
	add(12, 2, 84, 70, 27, 90)
	add(112, 2, 85, 91)
	return w
}

func packedAttentionFixtureBytes(w []uint32) []byte {
	b := make([]byte, 4*len(w))
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	return b
}

// fak-test:runtime fast est=20ms lane=default
// Estimate unmeasured. UNIT-ONLY parser regression, not compiler/hardware evidence.
func TestPackedAttentionStructuralABI(t *testing.T) {
	fixture := packedAttentionStructuralFixture()
	raw := packedAttentionFixtureBytes(fixture)
	got, err := inspectPackedAttentionSPIRV(raw)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	if got.ModuleSHA256 != hex.EncodeToString(digest[:]) || got.LocalSize != [3]uint32{128, 1, 1} || got.PushBytes != 32 || got.WorkgroupDeclaredBytes != 17936 {
		t.Fatalf("unexpected structural observation: %+v", got)
	}
	find := func(w []uint32, op, id uint32) int {
		for p := 5; p < len(w); p += int(w[p] >> 16) {
			if w[p]&0xffff == op && w[p+1] == id {
				return p
			}
		}
		t.Fatalf("fixture instruction %d/%d absent", op, id)
		return 0
	}
	for _, tc := range []struct {
		name   string
		mutate func([]uint32) []uint32
	}{
		{"F32 binding one", func(w []uint32) []uint32 { w[find(w, 30, 23)+2] = 5; return w }},
		{"old 24-byte push", func(w []uint32) []uint32 {
			p := find(w, 30, 10)
			w[p] = 8<<16 | 30
			return append(w[:p+8], w[p+10:]...)
		}},
		{"push metadata signed", func(w []uint32) []uint32 { w[find(w, 30, 10)+8] = 3; return w }},
		{"push scale wrong type", func(w []uint32) []uint32 { w[find(w, 30, 10)+7] = 4; return w }},
		{"wrong push offset", func(w []uint32) []uint32 { w[find(w, 72, 10)+4] = 4; return w }},
		{"extra descriptor", func(w []uint32) []uint32 { return append(w, 4<<16|59, 21, 100, 12) }},
		{"workgroup stride", func(w []uint32) []uint32 { return append(w, 4<<16|71, 40, 6, 8) }},
		{"local64", func(w []uint32) []uint32 { w[find(w, 16, 1)+3] = 64; return w }},
		{"wrong descriptor set", func(w []uint32) []uint32 { p := find(w, 71, 25); w[p+7] = 1; return w }},
		{"duplicate binding", func(w []uint32) []uint32 { w[find(w, 71, 25)+3] = 0; return w }},
		{"wrong array stride", func(w []uint32) []uint32 { w[find(w, 71, 6)+3] = 8; return w }},
		{"changed shared memory", func(w []uint32) []uint32 { w[find(w, 43, 4)+3] = 2047; return w }},
		{"missing NoContraction", func(w []uint32) []uint32 { p := find(w, 71, 80); return append(w[:p], w[p+3:]...) }},
		{"missing float controls", func(w []uint32) []uint32 { p := find(w, 16, 1) + 6; return append(w[:p], w[p+4:]...) }},
		{"unknown opcode", func(w []uint32) []uint32 { return append(w, 1<<16|65535) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := tc.mutate(append([]uint32(nil), fixture...))
			if _, err := inspectPackedAttentionSPIRV(packedAttentionFixtureBytes(changed)); err == nil {
				t.Fatal("mutated structural ABI accepted")
			}
		})
	}
}
