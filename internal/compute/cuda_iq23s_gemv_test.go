package compute

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// cuda_iq23s_gemv_test.go — resident IQ2_S / IQ3_XXS / IQ3_S CUDA GEMV (cuda/iq23s_gemv.cuh).
//
// Runs in both builds: the C ABI/source contract, the device grids checked entry for entry
// against the Go oracle tables, and a lane-for-lane simulation of each kernel's decode
// against rawIQRowDot. The on-device witness is TestCUDAIQ2SIQ3XXSIQ3SGemvMatchesRef
// (-tags cuda).

// simIQ23SLaneRow mirrors k_iq2s_gemv / k_iq3xxs_gemv / k_iq3s_gemv for one row: per lane
// (ib32 = lane>>2, l = lane&3) the same byte reads, db, grid and sign decode, then the warp sum.
func simIQ23SLaneRow(dt Dtype, row []byte, x []float32) float32 {
	bb, _ := RawIQBlockBytes(dt)
	nsb := len(row) / bb
	var total float32
	for lane := 0; lane < 32; lane++ {
		ib32, l := lane>>2, lane&3
		var sum float32
		for sb := 0; sb < nsb; sb++ {
			blk := row[sb*bb:]
			d := f16At(blk)
			xs := x[sb*256+lane*8:]
			switch dt {
			case IQ2_S:
				idx := uint32(blk[2+4*ib32+l]) | (uint32(blk[66+ib32])<<uint(8-2*l))&0x300
				sc := blk[74+ib32]
				nib := sc & 0x0f
				if l >= 2 {
					nib = sc >> 4
				}
				db := d * (0.5 + float32(nib)) * 0.25
				sum += db * simIQDot8(iq2SGrid[idx], blk[34+4*ib32+l], xs)
			case IQ3_XXS:
				aux32 := uint32(binary.LittleEndian.Uint16(blk[66+4*ib32:])) | uint32(binary.LittleEndian.Uint16(blk[68+4*ib32:]))<<16
				db := d * (0.5 + float32(aux32>>28)) * 0.5
				signs := ksignsIQ2XS[(aux32>>uint(7*l))&127]
				grid := uint64(iq3xxsGrid[blk[2+8*ib32+2*l]]) | uint64(iq3xxsGrid[blk[2+8*ib32+2*l+1]])<<32
				sum += db * simIQDot8(grid, signs, xs)
			case IQ3_S:
				high := uint32(blk[66+ib32])
				idx1 := uint32(blk[2+8*ib32+2*l]) | (high<<uint(8-2*l))&0x100
				idx2 := uint32(blk[2+8*ib32+2*l+1]) | (high<<uint(7-2*l))&0x100
				nib := (blk[106+ib32/2] >> uint(4*(ib32&1))) & 0x0f
				db := d * float32(1+2*int(nib))
				grid := uint64(iq3SGrid[idx1]) | uint64(iq3SGrid[idx2])<<32
				sum += db * simIQDot8(grid, blk[74+4*ib32+l], xs)
			default:
				panic("simIQ23SLaneRow: unsupported dtype " + dt.String())
			}
		}
		total += sum
	}
	return total
}

// fak-test:runtime fast est=20ms lane=default
func TestIQ2SIQ3XXSIQ3SCUDASourceContract(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := findRepoRootForTest(t, wd)
	header, err := os.ReadFile(filepath.Join(root, "internal", "compute", "cuda_backend.h"))
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range []string{
		"void fcuda_iq2s_gemv(const uint8_t *dW, const float *dX, float *dY, int out, int in, int P);",
		"void fcuda_iq3xxs_gemv(const uint8_t *dW, const float *dX, float *dY, int out, int in, int P);",
		"void fcuda_iq3s_gemv(const uint8_t *dW, const float *dX, float *dY, int out, int in, int P);",
	} {
		if !strings.Contains(string(header), decl) {
			t.Errorf("cuda_backend.h missing %q", decl)
		}
	}
	src := cudaSourceClosure(t, filepath.Join(root, "internal", "compute", "cuda_kernels.cu"))
	for _, sym := range []string{"k_iq2s_gemv", "k_iq3xxs_gemv", "k_iq3s_gemv"} {
		if !strings.Contains(src, sym) {
			t.Errorf("CUDA source closure missing %q", sym)
		}
	}
	checkGrid := func(decl string, want []uint64) {
		t.Helper()
		lits := cudaIQTableLiterals(t, src, decl)
		if len(lits) != len(want) {
			t.Fatalf("%s has %d entries, want %d", decl, len(lits), len(want))
		}
		for i, l := range lits {
			v, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(l, "0x"), "ULL"), 16, 64)
			if err != nil || v != want[i] {
				t.Fatalf("%s[%d] = %s, Go 0x%x (%v)", decl, i, l, want[i], err)
			}
		}
	}
	widen := func(g []uint32) []uint64 {
		out := make([]uint64, len(g))
		for i, v := range g {
			out[i] = uint64(v)
		}
		return out
	}
	checkGrid("fcuda_iq2s_grid[1024] =", iq2SGrid[:])
	checkGrid("fcuda_iq3xxs_grid[256] =", widen(iq3xxsGrid[:]))
	checkGrid("fcuda_iq3s_grid[512] =", widen(iq3SGrid[:]))
}

// fak-test:runtime fast est=50ms lane=default
func TestIQ2SIQ3XXSIQ3SCUDALaneSimulationMatchesCPUOracle(t *testing.T) {
	const out, in = 7, 768
	for i, dt := range []Dtype{IQ2_S, IQ3_XXS, IQ3_S} {
		bb, _ := RawIQBlockBytes(dt)
		raw := iqGEMVTestRaw(out, in, bb, 0x23500+uint64(i))
		x := iqGEMVTestX(in, i)
		rowBytes := in / 256 * bb
		got, want := make([]float32, out), make([]float32, out)
		scratch := make([]float32, rawIQSuper)
		f := rawIQFormats[dt]
		for o := 0; o < out; o++ {
			row := raw[o*rowBytes : (o+1)*rowBytes]
			got[o] = simIQ23SLaneRow(dt, row, x)
			want[o] = rawIQRowDot(f, row, x, scratch)
		}
		if e := iqGEMVRelErr(got, want); e > 1e-5 {
			t.Fatalf("%s lane simulation rel err %g > 1e-5 (got %v want %v)", dt, e, got, want)
		}
	}
}
