package model

import (
	"fmt"
	"math"
	"testing"
)

// This is the original two-pass prefill body, intentionally independent of the
// fused helper. It keeps the explicit f64 FMA expression as the arithmetic oracle.
func qwen35DecayReadSerialOracle(st, kn, mem []float32, g float32, kHd, vHd int) {
	for i := range st {
		st[i] *= g
	}
	for d := range mem {
		mem[d] = 0
	}
	for i := 0; i < kHd; i++ {
		ki := kn[i]
		base := i * vHd
		for d := 0; d < vHd; d++ {
			mem[d] = float32(math.FMA(float64(st[base+d]), float64(ki), float64(mem[d])))
		}
	}
}

func qwen35DecayReadSameBits(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s lengths: %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] bits: %08x != %08x", name, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestQwen35PrefillDecayReadOrderedBits(t *testing.T) {
	for _, shape := range [][2]int{{1, 1}, {3, 5}, {17, 9}, {128, 128}} {
		for _, exceptional := range []bool{false, true} {
			t.Run(fmt.Sprintf("%dx%d/exceptional=%t", shape[0], shape[1], exceptional), func(t *testing.T) {
				kHd, vHd := shape[0], shape[1]
				state := make([]float32, kHd*vHd)
				key := make([]float32, kHd)
				for i := range state {
					state[i] = float32((i*29)%97-48) / 31
				}
				for i := range key {
					key[i] = float32((i*11)%31-15) / 17
				}
				if exceptional {
					bits := []uint32{0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345}
					for i := range state {
						state[i] = math.Float32frombits(bits[i%len(bits)])
					}
				}
				keyBefore := append([]float32(nil), key...)
				for _, g := range []float32{0, math.Float32frombits(0x80000000), 1, 0.9375, math.SmallestNonzeroFloat32, float32(math.Inf(1)), math.Float32frombits(0x7fc34567)} {
					guarded := make([]float32, len(state)+2)
					guarded[0], guarded[len(guarded)-1] = 19, -23
					got := guarded[1 : len(guarded)-1 : len(guarded)-1]
					copy(got, state)
					want := append([]float32(nil), state...)
					mem, wantMem := make([]float32, vHd), make([]float32, vHd)
					for step := 0; step < 3; step++ {
						for d := range mem {
							mem[d], wantMem[d] = 99, 99
						}
						qwen35DecayReadSerialOracle(want, key, wantMem, g, kHd, vHd)
						if !qwen35PrefillDecayRead(got, key, mem, g, kHd, vHd) {
							t.Fatal("valid owned rows declined")
						}
						qwen35DecayReadSameBits(t, "state", got, want)
						qwen35DecayReadSameBits(t, "memory", mem, wantMem)
						qwen35DecayReadSameBits(t, "key", key, keyBefore)
						if guarded[0] != 19 || guarded[len(guarded)-1] != -23 {
							t.Fatal("state canary overwritten")
						}
					}
				}
			})
		}
	}
}

func TestQwen35PrefillDecayReadDeclinesWithoutWrites(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                      string
		kHd, vHd, state, key, mem int
	}{
		{"zero", 0, 2, 4, 2, 2},
		{"negative", 2, -1, 4, 2, 2},
		{"overflow", maxInt, 2, 4, 2, 2},
		{"short-state", 2, 2, 3, 2, 2},
		{"extra-state", 2, 2, 5, 2, 2},
		{"short-key", 2, 2, 4, 1, 2},
		{"short-memory", 2, 2, 4, 2, 1},
		{"extra-memory", 2, 2, 4, 2, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st, key, mem := make([]float32, tc.state), make([]float32, tc.key), make([]float32, tc.mem)
			for _, values := range [][]float32{st, key, mem} {
				for i := range values {
					values[i] = float32(i + 1)
				}
			}
			stBefore, keyBefore, memBefore := append([]float32(nil), st...), append([]float32(nil), key...), append([]float32(nil), mem...)
			if qwen35PrefillDecayRead(st, key, mem, 0.5, tc.kHd, tc.vHd) {
				t.Fatal("malformed geometry admitted")
			}
			qwen35DecayReadSameBits(t, "state", st, stBefore)
			qwen35DecayReadSameBits(t, "key", key, keyBefore)
			qwen35DecayReadSameBits(t, "memory", mem, memBefore)
		})
	}
}

// This witnesses consumer wiring to the existing opt-in operation, NOT scalar
// versus hardware numeric qualification. It is authored but has not been run.
func TestQwen35PrefillVectorizedConsumerMatchesExistingOperation(t *testing.T) {
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")
	if !HasVectorizedDeltaNetFor(128, 128) {
		t.Skip("requires the existing executable AVX-512 decode operation")
	}
	st, q, k, v := make([]float32, 128*128), make([]float32, 128), make([]float32, 128), make([]float32, 128)
	for i := range st {
		st[i] = float32(i%41-20) / 37
	}
	for i := range q {
		q[i], k[i], v[i] = float32(i%11-5)/19, float32(i%13-6)/17, float32(i%17-8)/23
	}
	wantState := append([]float32(nil), st...)
	mem, delta, wantMem, wantDelta := make([]float32, 128), make([]float32, 128), make([]float32, 128), make([]float32, 128)
	core := make([]float32, 2*256)
	for i := range core {
		core[i] = 7
	}
	wantOutput := append([]float32(nil), core[384:512]...)
	VectorizedHeadStep(wantState, q, k, v, 0.375, 0.9375, wantOutput, wantMem, wantDelta)
	if !qwen35PrefillVectorizedStep(st, q, k, v, core, mem, delta, 0.375, 0.9375, 1, 256, 1) {
		t.Fatal("qualified exact-shape consumer declined")
	}
	qwen35DecayReadSameBits(t, "vector state", st, wantState)
	qwen35DecayReadSameBits(t, "vector memory", mem, wantMem)
	qwen35DecayReadSameBits(t, "vector delta", delta, wantDelta)
	qwen35DecayReadSameBits(t, "vector output", core[384:512], wantOutput)
	for _, value := range core[:384] {
		if value != 7 {
			t.Fatal("unowned output row changed")
		}
	}
}

func TestQwen35PrefillVectorizedConsumerDeclinesBeforeMutation(t *testing.T) {
	st, q, k, v := make([]float32, 128*128), make([]float32, 128), make([]float32, 128), make([]float32, 128)
	mem, delta, core := make([]float32, 128), make([]float32, 128), make([]float32, 256)
	for _, values := range [][]float32{st, q, k, v, mem, delta, core} {
		for i := range values {
			values[i] = 3
		}
	}
	t.Setenv("FAK_VECTORIZED_DELTANET", "1")
	maxInt := int(^uint(0) >> 1)
	for _, geometry := range [][3]int{{-1, 256, 0}, {0, 256, -1}, {0, 127, 0}, {maxInt, 256, 0}, {0, 256, maxInt}, {0, 256, 2}, {1, 256, 0}} {
		if qwen35PrefillVectorizedStep(st, q, k, v, core, mem, delta, 0.5, 0.5, geometry[0], geometry[1], geometry[2]) {
			t.Fatal("invalid output geometry admitted")
		}
	}
	if qwen35PrefillVectorizedStep(st[:len(st)-1], q, k, v, core, mem, delta, 0.5, 0.5, 0, 256, 0) {
		t.Fatal("short state admitted")
	}
	for _, setting := range []string{"", "0", "true", "off"} {
		t.Setenv("FAK_VECTORIZED_DELTANET", setting)
		if qwen35PrefillVectorizedStep(st, q, k, v, core, mem, delta, 0.5, 0.5, 0, 256, 0) {
			t.Fatal("default or unselected prefill capability admitted")
		}
	}
	for _, values := range [][]float32{st, q, k, v, mem, delta, core} {
		for _, value := range values {
			if value != 3 {
				t.Fatal("decline changed a borrowed buffer")
			}
		}
	}
}
