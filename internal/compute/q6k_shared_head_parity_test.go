package compute

import (
	"encoding/binary"
	"math"
	"testing"
)

// This is a same-backend scalar/panel contract, not CPU/Vulkan bit identity.
// The reduced vocabulary keeps the real Flash hidden width without allocating
// the full checkpoint head. Survivor rows are distinct BF16 values stored as F32;
// head outputs remain F32, as in the model's final-norm/head boundary.
// fak-test:runtime fast est=500ms lane=default
func TestQ6KSharedHeadPanelBitIdenticalToScalar(t *testing.T) {
	t.Parallel()
	const out, in, maxRows = 129, 5120, 4
	be := cpu()
	w := NewQ6K(be, []int{out, in}, q6KSharedHeadFixture(out, in))
	x := q6KSharedHeadInput(maxRows, in)
	want := q6KSharedHeadScalar(t, be, w, x, out, in, maxRows)
	for rows := 1; rows <= maxRows; rows++ {
		panel := NewF32(be, []int{rows, in}, x[:rows*in])
		got := be.Read(be.BatchedMatMul(w, panel, rows))
		q6KSharedHeadCompare(t, got, want[:rows*out], rows, out)
	}
}

func q6KSharedHeadFixture(out, in int) []byte {
	const blockBytes = 210
	raw := make([]byte, out*(in/256)*blockBytes)
	for block := 0; block < len(raw)/blockBytes; block++ {
		b := raw[block*blockBytes : (block+1)*blockBytes]
		for i := 0; i < 192; i++ {
			b[i] = byte(block*29 + i*71 + 11)
		}
		for i := 0; i < 16; i++ {
			b[192+i] = byte(int8((block*7+i*13)%31 - 15))
		}
		// Non-power-of-two finite half scales make low-bit reduction drift
		// observable even though every input is exactly representable in BF16.
		binary.LittleEndian.PutUint16(b[208:], uint16(0x27d3+block%127))
	}
	return raw
}

func q6KSharedHeadInput(rows, in int) []float32 {
	x := make([]float32, rows*in)
	for row := 0; row < rows; row++ {
		for col := 0; col < in; col++ {
			h := uint32(col+row*in)*2654435761 + uint32(row+1)*40503
			h ^= h >> 15
			h *= 2246822519
			h ^= h >> 13
			value := float32(int32(h>>8)-(1<<23)) / float32(1<<23)
			bits := math.Float32bits(value)
			// Independent round-to-nearest-even finite F32 -> BF16 boundary.
			bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
			x[row*in+col] = math.Float32frombits(bits)
		}
	}
	return x
}

func q6KSharedHeadScalar(t *testing.T, be Backend, w Tensor, x []float32, out, in, rows int) []float32 {
	t.Helper()
	want := make([]float32, 0, rows*out)
	retainsF32 := false
	for row := 0; row < rows; row++ {
		input := be.Upload(NewF32(be, []int{in}, x[row*in:(row+1)*in]), F32)
		output := be.MatMul(w, input)
		values := be.Read(output)
		if len(values) != out {
			t.Fatalf("scalar row %d: length=%d, want %d", row, len(values), out)
		}
		for i, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("scalar row %d output %d is nonfinite", row, i)
			}
			retainsF32 = retainsF32 || math.Float32bits(value)&0xffff != 0
		}
		want = append(want, values...)
		be.Free(output)
		be.Free(input)
	}
	if !retainsF32 {
		t.Fatal("fixture failed to distinguish F32 head logits from rounded BF16")
	}
	return want
}

func q6KSharedHeadCompare(t *testing.T, got, want []float32, rows, out int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("rows=%d output length=%d, want %d", rows, len(got), len(want))
	}
	for i, value := range got {
		if math.Float32bits(value) != math.Float32bits(want[i]) {
			t.Fatalf("rows=%d row=%d output=%d: panel=%g (%#08x), scalar=%g (%#08x)",
				rows, i/out, i%out, value, math.Float32bits(value), want[i], math.Float32bits(want[i]))
		}
	}
}
