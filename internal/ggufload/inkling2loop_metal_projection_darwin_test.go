//go:build darwin && arm64 && cgo

package ggufload

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"os"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// fak-test:runtime integration est=2s lane=default
func TestInkling2MetalPackedProjections(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("requires a Metal device")
	}
	const rows, cols, experts = 7, 512, 2
	file, err := os.CreateTemp(t.TempDir(), "packed")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	size := 16*os.Getpagesize() - 4
	if err := file.Truncate(int64(size)); err != nil {
		t.Fatal(err)
	}
	data, err := syscall.Mmap(int(file.Fd()), 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	rng := rand.New(rand.NewSource(29))
	order := []int{0, 1, 0, 1, 0, 1, 0, 0}
	var jobs []metalgemm.PackedGGUFJob
	var want, tol [][]float64
	types := []TensorType{TensorF32, TensorF16, TensorBF16, TensorQ8_0, TensorQ3_K, TensorQ4_K, TensorQ5_K, TensorQ6_K, TensorIQ2_XS, TensorIQ2_S, TensorIQ3_XXS, TensorIQ3_S, TensorIQ4_NL, TensorNVFP4}
	offset := 32
	for _, typ := range types {
		stride, err := inkling2RowBytes(typ, cols)
		if err != nil {
			t.Fatal(err)
		}
		if typ == TensorNVFP4 {
			offset = len(data) - experts*rows*stride
		}
		buf := data[offset : offset+experts*rows*stride]
		rng.Read(buf)
		for r := 0; r < experts*rows; r++ {
			finiteProjectionBlocks(typ, buf[r*stride:(r+1)*stride], rng)
		}
		matrix, err := store.Matrix(offset, uint32(typ), rows, cols, experts)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		for _, e := range order {
			x, y := make([]float32, cols), make([]float32, rows)
			for i := range x {
				x[i] = rng.Float32() - .5
			}
			result, bound := make([]float64, rows), make([]float64, rows)
			w := make([]float32, cols)
			for r := 0; r < rows; r++ {
				start := (e*rows + r) * stride
				if err := inkling2DequantRow(typ, buf[start:start+stride], w); err != nil {
					t.Fatal(err)
				}
				for j, v := range w {
					p := float64(v) * float64(x[j])
					result[r] += p
					bound[r] += math.Abs(p)
				}
				bound[r] = 1e-6 + 1e-6*bound[r]
			}
			want = append(want, result)
			tol = append(tol, bound)
			jobs = append(jobs, metalgemm.PackedGGUFJob{Matrix: matrix, Expert: e, X: x, Dst: y})
		}
		offset = (offset + len(buf) + 31) &^ 31
	}
	if err := store.Run(jobs); err != nil {
		t.Fatal(err)
	}
	for i, j := range jobs {
		for r, y := range j.Dst {
			if !(math.Abs(float64(y)-want[i][r]) <= tol[i][r]) {
				t.Errorf("%s expert %d row %d: got %g want %g within %g", types[i/len(order)], j.Expert, r, y, want[i][r], tol[i][r])
				break
			}
		}
	}
	if _, err := store.Matrix(0, 999, 1, 1, 1); !errors.Is(err, metalgemm.ErrPackedGGUFType) {
		t.Fatalf("unsupported format: %v", err)
	}
	bad := jobs[0]
	bad.Expert = experts
	if err := store.Run([]metalgemm.PackedGGUFJob{bad}); !errors.Is(err, metalgemm.ErrPackedGGUFShape) {
		t.Fatalf("expert bounds: %v", err)
	}
	store.Close()
	if err := store.Run(jobs); !errors.Is(err, metalgemm.ErrPackedGGUFClosed) {
		t.Fatalf("closed store: %v", err)
	}
}

func finiteProjectionBlocks(typ TensorType, b []byte, rng *rand.Rand) {
	block, scale := 0, 0
	switch typ {
	case TensorF32:
		for i := 0; i < len(b); i += 4 {
			binary.LittleEndian.PutUint32(b[i:], math.Float32bits(rng.Float32()-.5))
		}
		return
	case TensorBF16:
		for i := 0; i < len(b); i += 2 {
			binary.LittleEndian.PutUint16(b[i:], uint16(math.Float32bits(rng.Float32()-.5)>>16))
		}
		return
	case TensorF16:
		block = 2
	case TensorQ8_0:
		block = 34
	case TensorQ3_K:
		block, scale = 110, 108
	case TensorQ4_K:
		block = 144
	case TensorQ5_K:
		block = 176
	case TensorQ6_K:
		block, scale = 210, 208
	case TensorIQ2_XS:
		block = 74
	case TensorIQ2_S:
		block = 82
	case TensorIQ3_XXS:
		block = 98
	case TensorIQ3_S:
		block = 110
	case TensorIQ4_NL:
		block = 18
	case TensorNVFP4:
		for i := 0; i < len(b); i += 36 {
			for j := 0; j < 4; j++ {
				b[i+j] &= 127
			}
		}
		return
	default:
		// Only the host-reference encodings have a block layout here; any other
		// type would leave block at zero and spin the loop below forever.
		panic("finiteProjectionBlocks: no fixture layout for " + typ.String())
	}
	for i := 0; i < len(b); i += block {
		binary.LittleEndian.PutUint16(b[i+scale:], uint16(0x3000+rng.Intn(1024)))
		if typ == TensorQ4_K || typ == TensorQ5_K {
			binary.LittleEndian.PutUint16(b[i+2:], 0x3000)
		}
	}
}
