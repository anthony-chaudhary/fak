package compute

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// cuda_iq_gemv_test.go — oss-index TICKET-01: resident IQ2_XXS / IQ2_XS CUDA GEMV
// (cuda/iq_gemv.cuh, ported from llama.cpp vecdotq.cuh @f7c01c52, MIT).
//
// Runs in both builds. Pure Go: the C ABI/source contract plus a lane-for-lane simulation
// of the kernel's decode (lane -> ib32=lane>>2, l=lane&3, the exact byte offsets the
// kernel reads) against the CPU oracles iq2xxsRowDot / rawIQRowDot. With -tags cuda and a
// registered device: on-device parity at small and DS41/Qwen3.8 shapes, and the
// BenchmarkCUDAIQ GB/s table with a k_q2k_gemv comparator at the same shape.

// iqGEMVTestRaw fills out rows of in/256 super-blocks of blockBytes each with xorshift
// bytes and a bounded f16 d at offset 0, so every decoded weight is finite.
func iqGEMVTestRaw(out, in, blockBytes int, seed uint64) []byte {
	raw := make([]byte, out*(in/256)*blockBytes)
	state := seed
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state)
	}
	dBits := [...]uint16{0x2c00, 0x3000, 0x3400, 0x3800} // 0.0625 .. 0.5
	for b, base := 0, 0; base < len(raw); b, base = b+1, base+blockBytes {
		binary.LittleEndian.PutUint16(raw[base:], dBits[b%len(dBits)])
	}
	return raw
}

func iqGEMVTestX(n int, token int) []float32 {
	x := make([]float32, n)
	for i := range x {
		phase := float64((token + 1) * (i + 3))
		x[i] = float32(math.Sin(phase*0.017) + 0.25*math.Cos(phase*0.031))
	}
	return x
}

// simIQ2XXSLaneRow mirrors k_iq2xxs_gemv for one row: per lane, per super-block, the same
// gidx/aux1/db/signs reads, then the warp sum.
func simIQ2XXSLaneRow(row []byte, x []float32) float32 {
	nsb := len(row) / iq2xxsSuperBlock
	var lanes [32]float32
	for lane := 0; lane < 32; lane++ {
		ib32, l := lane>>2, lane&3
		var sum float32
		for sb := 0; sb < nsb; sb++ {
			blk := row[sb*iq2xxsSuperBlock:]
			d := f16At(blk)
			gidx := blk[2+8*ib32+l]
			aux1 := uint32(binary.LittleEndian.Uint16(blk[2+2*(4*ib32+2):])) | uint32(binary.LittleEndian.Uint16(blk[2+2*(4*ib32+3):]))<<16
			db := d * (0.5 + float32(aux1>>28)) * 0.25
			signs := ksignsIQ2XS[(aux1>>uint(7*l))&127]
			sum += db * simIQDot8(iq2xxsGrid[gidx], signs, x[sb*256+lane*8:])
		}
		lanes[lane] = sum
	}
	var total float32
	for _, v := range lanes {
		total += v
	}
	return total
}

// simIQ2XSLaneRow mirrors k_iq2xs_gemv for one row.
func simIQ2XSLaneRow(row []byte, x []float32) float32 {
	nsb := len(row) / iq2xsSuperBlock
	var total float32
	for lane := 0; lane < 32; lane++ {
		ib32, l := lane>>2, lane&3
		for sb := 0; sb < nsb; sb++ {
			blk := row[sb*iq2xsSuperBlock:]
			d := f16At(blk)
			q := binary.LittleEndian.Uint16(blk[2+2*(4*ib32+l):])
			sc := blk[66+ib32]
			nib := sc & 0x0f
			if l >= 2 {
				nib = sc >> 4
			}
			db := d * (0.5 + float32(nib)) * 0.25
			total += db * simIQDot8(iq2XSGrid[q&511], ksignsIQ2XS[q>>9], x[sb*256+lane*8:])
		}
	}
	return total
}

func simIQDot8(grid uint64, signs uint8, xs []float32) float32 {
	var part float32
	for j := 0; j < 8; j++ {
		v := float32(byte(grid >> uint(8*j)))
		if signs&(1<<uint(j)) != 0 {
			v = -v
		}
		part += v * xs[j]
	}
	return part
}

func iqGEMVRelErr(got, want []float32) float64 {
	var maxAbs, maxErr float64
	for i := range want {
		maxAbs = math.Max(maxAbs, math.Abs(float64(want[i])))
		maxErr = math.Max(maxErr, math.Abs(float64(got[i]-want[i])))
	}
	if maxAbs == 0 {
		return maxErr
	}
	return maxErr / maxAbs
}

// fak-test:runtime fast est=10ms lane=default
func TestIQ2XXSCUDASourceContract(t *testing.T) {
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
		"void fcuda_iq2xxs_gemv(const uint8_t *dW, const float *dX, float *dY, int out, int in, int P);",
		"void fcuda_iq2xs_gemv(const uint8_t *dW, const float *dX, float *dY, int out, int in, int P);",
	} {
		if !strings.Contains(string(header), decl) {
			t.Errorf("cuda_backend.h missing %q", decl)
		}
	}
	src := cudaSourceClosure(t, filepath.Join(root, "internal", "compute", "cuda_kernels.cu"))
	for _, sym := range []string{"k_iq2xxs_gemv", "k_iq2xs_gemv", "fcuda_iq2xxs_grid[256]", "fcuda_iq2xs_grid[512]", "fcuda_ksigns_iq2xs[128]", "f7c01c5226cac1de1052c17920ff38cdefea0ae6"} {
		if !strings.Contains(src, sym) {
			t.Errorf("CUDA source closure missing %q", sym)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestIQ2XXSCUDALaneSimulationMatchesCPUOracle(t *testing.T) {
	const out, in = 7, 768
	raw := iqGEMVTestRaw(out, in, iq2xxsSuperBlock, 0x7c01c5)
	x := iqGEMVTestX(in, 0)
	rowBytes := in / 256 * iq2xxsSuperBlock
	got, want := make([]float32, out), make([]float32, out)
	scratch := make([]float32, iq2xxsSuper)
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		got[o] = simIQ2XXSLaneRow(row, x)
		want[o] = iq2xxsRowDot(row, x, scratch)
	}
	if e := iqGEMVRelErr(got, want); e > 1e-5 {
		t.Fatalf("IQ2_XXS lane simulation rel err %g > 1e-5 (got %v want %v)", e, got, want)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestIQ2XSCUDALaneSimulationMatchesCPUOracle(t *testing.T) {
	const out, in = 7, 768
	raw := iqGEMVTestRaw(out, in, iq2xsSuperBlock, 0x1146)
	x := iqGEMVTestX(in, 1)
	rowBytes := in / 256 * iq2xsSuperBlock
	got, want := make([]float32, out), make([]float32, out)
	scratch := make([]float32, rawIQSuper)
	f := rawIQFormats[IQ2_XS]
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		got[o] = simIQ2XSLaneRow(row, x)
		want[o] = rawIQRowDot(f, row, x, scratch)
	}
	if e := iqGEMVRelErr(got, want); e > 1e-5 {
		t.Fatalf("IQ2_XS lane simulation rel err %g > 1e-5 (got %v want %v)", e, got, want)
	}
}

func iqGEMVBlockBytes(dt Dtype) int {
	if dt == IQ2_XXS {
		return iq2xxsSuperBlock
	}
	return iq2xsSuperBlock
}

// iqGEMVDeviceParity uploads one raw IQ2 weight and compares device MatMul (P=1) and
// BatchedMatMul (P>1) against the CPU reference backend through the same NewRawIQ bytes.
func iqGEMVDeviceParity(t *testing.T, dt Dtype) {
	be, ok := Lookup("cuda")
	if !ok {
		t.Skip("cuda backend not registered (build without -tags cuda or no device)")
	}
	ref := Default()
	shapes := []struct{ out, in int }{{321, 512}, {2304, 5120}, {5120, 2304}}
	for _, sh := range shapes {
		raw := iqGEMVTestRaw(sh.out, sh.in, iqGEMVBlockBytes(dt), uint64(sh.out*7919+sh.in))
		devW := be.Upload(NewRawIQ(be, dt, []int{sh.out, sh.in}, raw), dt)
		refW := NewRawIQ(ref, dt, []int{sh.out, sh.in}, raw)
		for _, P := range []int{1, 3} {
			x := make([]float32, 0, P*sh.in)
			for tok := 0; tok < P; tok++ {
				x = append(x, iqGEMVTestX(sh.in, tok)...)
			}
			xShape := []int{P, sh.in}
			if P == 1 {
				xShape = []int{sh.in}
			}
			devX := be.Upload(NewF32(be, xShape, x), F32)
			var devY, refY Tensor
			if P == 1 {
				devY = be.MatMul(devW, devX)
				refY = ref.MatMul(refW, NewF32(ref, xShape, x))
			} else {
				devY = be.BatchedMatMul(devW, devX, P)
				refY = ref.BatchedMatMul(refW, NewF32(ref, xShape, x), P)
			}
			got, want := be.Read(devY), ref.Read(refY)
			be.Free(devY)
			be.Free(devX)
			if len(got) != len(want) {
				t.Fatalf("%s %dx%d P=%d: len %d != %d", dt, sh.out, sh.in, P, len(got), len(want))
			}
			if e := iqGEMVRelErr(got, want); e > 1e-4 {
				t.Errorf("%s %dx%d P=%d: rel err %g > 1e-4", dt, sh.out, sh.in, P, e)
			}
		}
		be.Free(devW)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestIQ2XXSCUDADeviceParity(t *testing.T) { iqGEMVDeviceParity(t, IQ2_XXS) }

// fak-test:runtime fast est=2s lane=default
func TestIQ2XSCUDADeviceParity(t *testing.T) { iqGEMVDeviceParity(t, IQ2_XS) }

// BenchmarkCUDAIQ reports weight-stream GB/s for the IQ2 GEMV at the DS41 routed-expert
// (2304x5120, 5120x2304) and Qwen3.8 GDN in_proj (10240x5120) shapes, with Q2_K through
// k_q2k_gemv at the same shape as the TICKET-01 comparator (target: IQ2_XXS >= 0.7x Q2_K).
// One Read after b.N launches drains the stream, so the timing is kernel-dominated.
func BenchmarkCUDAIQ(b *testing.B) {
	be, ok := Lookup("cuda")
	if !ok {
		b.Skip("cuda backend not registered")
	}
	shapes := []struct{ out, in int }{{2304, 5120}, {5120, 2304}, {10240, 5120}}
	dts := []struct {
		name  string
		dt    Dtype
		block int
	}{{"iq2_xxs", IQ2_XXS, iq2xxsSuperBlock}, {"iq2_xs", IQ2_XS, iq2xsSuperBlock}, {"q2_k", Q2_K, q2kSuperBlock}}
	for _, sh := range shapes {
		for _, d := range dts {
			b.Run(d.name+"/"+strconv.Itoa(sh.out)+"x"+strconv.Itoa(sh.in), func(b *testing.B) {
				raw := iqGEMVTestRaw(sh.out, sh.in, d.block, 0xbe7c4)
				var host Tensor
				if d.dt == Q2_K {
					host = NewQ2K(be, []int{sh.out, sh.in}, raw)
				} else {
					host = NewRawIQ(be, d.dt, []int{sh.out, sh.in}, raw)
				}
				devW := be.Upload(host, d.dt)
				defer be.Free(devW)
				devX := be.Upload(NewF32(be, []int{sh.in}, iqGEMVTestX(sh.in, 0)), F32)
				defer be.Free(devX)
				be.Free(be.MatMul(devW, devX)) // warm
				b.SetBytes(int64(len(raw)))
				b.ResetTimer()
				start := time.Now()
				var y Tensor
				for i := 0; i < b.N; i++ {
					if i > 0 {
						be.Free(y)
					}
					y = be.MatMul(devW, devX)
				}
				_ = be.Read(y)
				el := time.Since(start)
				b.StopTimer()
				be.Free(y)
				b.ReportMetric(float64(len(raw))*float64(b.N)/el.Seconds()/1e9, "weightGB/s")
			})
		}
	}
}
