//go:build cuda

package compute_test

import (
	"errors"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Set FAK_CUDA_GDN_REQUIRED=1 on a hardware acceptance node. In that mode a
// missing device/backend is a hard failure rather than a skip, so a skipped run
// can never be mistaken for #4725 closure evidence.
const cudaGDNRequiredEnv = "FAK_CUDA_GDN_REQUIRED"

type cudaGDNWitnessBackend interface {
	compute.Backend
	model.Qwen35GDNBackend
	HostXferBytes() uint64
	ResetHostXfer()
	Qwen35GDNOperationCount() uint64
	ResetQwen35GDNOperationCount()
	Recycle()
}

func cudaGDNBackend(t *testing.T) cudaGDNWitnessBackend {
	t.Helper()
	be := compute.Pick("cuda")
	if be == nil {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatalf("%s=1: real CUDA GDN fixture required, but backend cuda is not registered", cudaGDNRequiredEnv)
		}
		t.Skip("cuda backend not registered (set FAK_CUDA_GDN_REQUIRED=1 on the acceptance node to fail rather than skip)")
	}
	gdn, ok := be.(cudaGDNWitnessBackend)
	if !ok {
		t.Fatalf("registered cuda backend %T does not structurally implement model.Qwen35GDNBackend plus witness counters", be)
	}
	return gdn
}

type gdnFixtureGeometry struct {
	hidden, nK, nV, kHd, vHd, kernel int
	eps                              float32
}

func (g gdnFixtureGeometry) keyDim() int   { return g.nK * g.kHd }
func (g gdnFixtureGeometry) valueDim() int { return g.nV * g.vHd }
func (g gdnFixtureGeometry) convDim() int  { return 2*g.keyDim() + g.valueDim() }

type gdnFixtureData struct {
	x, inQKV, inZ, inB, inA         []float32
	convW, aLog, dtBias, norm       []float32
	outW, convState, recurrentState []float32
}

type gdnLCG uint64

func (r *gdnLCG) next(scale float32) float32 {
	*r = *r*6364136223846793005 + 1442695040888963407
	v := float32(uint32(*r>>32))/float32(uint64(1)<<32) - 0.5
	return v * scale
}

func gdnVector(r *gdnLCG, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = r.next(scale)
	}
	return out
}

func newGDNFixtureData(g gdnFixtureGeometry) *gdnFixtureData {
	r := gdnLCG(0x4725c0da)
	d := &gdnFixtureData{
		x:              gdnVector(&r, g.hidden, 1.0),
		inQKV:          gdnVector(&r, g.convDim()*g.hidden, 0.24),
		inZ:            gdnVector(&r, g.valueDim()*g.hidden, 0.20),
		inB:            gdnVector(&r, g.nV*g.hidden, 0.16),
		inA:            gdnVector(&r, g.nV*g.hidden, 0.12),
		convW:          gdnVector(&r, g.convDim()*g.kernel, 0.35),
		aLog:           gdnVector(&r, g.nV, 0.8),
		dtBias:         gdnVector(&r, g.nV, 0.6),
		norm:           gdnVector(&r, g.vHd, 0.16),
		outW:           gdnVector(&r, g.hidden*g.valueDim(), 0.22),
		convState:      gdnVector(&r, (g.kernel-1)*g.convDim(), 0.18),
		recurrentState: gdnVector(&r, g.nV*g.kHd*g.vHd, 0.10),
	}
	for i := range d.norm {
		d.norm[i] += 1
	}
	for i := range d.aLog {
		d.aLog[i] -= 0.7 // realistic positive A=exp(A_log), with a stable sub-unit decay.
	}
	return d
}

func gdnMatVec(w, x []float32, out, in int) []float32 {
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		var sum float32
		for i := 0; i < in; i++ {
			sum += w[o*in+i] * x[i]
		}
		y[o] = sum
	}
	return y
}

func gdnSilu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

func gdnSoftplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func gdnL2(dst, src []float32, scale float32) {
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	inv := float32(1 / math.Sqrt(float64(ss)+1e-6))
	for i, v := range src {
		dst[i] = v * inv * scale
	}
}

// referenceGDNDecode is intentionally independent of compute's CUDA implementation
// and of model.linearAttnStep. It directly spells out the documented CPU recurrence
// over host slices, including the pre-update decay and post-update readout order.
func referenceGDNDecode(g gdnFixtureGeometry, d *gdnFixtureData) (out, nextConv, nextRecurrent []float32) {
	keyDim, valueDim, convDim := g.keyDim(), g.valueDim(), g.convDim()
	mixed := gdnMatVec(d.inQKV, d.x, convDim, g.hidden)
	z := gdnMatVec(d.inZ, d.x, valueDim, g.hidden)
	b := gdnMatVec(d.inB, d.x, g.nV, g.hidden)
	a := gdnMatVec(d.inA, d.x, g.nV, g.hidden)
	nextConv = append([]float32(nil), d.convState...)
	convOut := make([]float32, convDim)
	for c := 0; c < convDim; c++ {
		var acc float32
		for j := 0; j < g.kernel-1; j++ {
			acc += d.convW[c*g.kernel+j] * nextConv[j*convDim+c]
		}
		acc += d.convW[c*g.kernel+g.kernel-1] * mixed[c]
		convOut[c] = gdnSilu(acc)
	}
	for j := 0; j < g.kernel-2; j++ {
		copy(nextConv[j*convDim:(j+1)*convDim], nextConv[(j+1)*convDim:(j+2)*convDim])
	}
	copy(nextConv[(g.kernel-2)*convDim:], mixed)

	qNorm, kNorm := make([]float32, keyDim), make([]float32, keyDim)
	qScale := float32(1 / math.Sqrt(float64(g.kHd)))
	for h := 0; h < g.nK; h++ {
		lo, hi := h*g.kHd, (h+1)*g.kHd
		gdnL2(qNorm[lo:hi], convOut[lo:hi], qScale)
		gdnL2(kNorm[lo:hi], convOut[keyDim+lo:keyDim+hi], 1)
	}

	nextRecurrent = append([]float32(nil), d.recurrentState...)
	core := make([]float32, valueDim)
	repeat := g.nV / g.nK
	for h := 0; h < g.nV; h++ {
		kh := h / repeat
		beta := float32(1 / (1 + math.Exp(float64(-b[h]))))
		aa := float32(math.Exp(float64(d.aLog[h])))
		dt := gdnSoftplus(a[h] + d.dtBias[h])
		decay := float32(math.Exp(float64(-aa * dt)))
		for stateIndex := h * g.kHd * g.vHd; stateIndex < (h+1)*g.kHd*g.vHd; stateIndex++ {
			nextRecurrent[stateIndex] *= decay
		}
		for vd := 0; vd < g.vHd; vd++ {
			var kvmem float32
			for i := 0; i < g.kHd; i++ {
				stateIndex := (h*g.kHd+i)*g.vHd + vd
				kvmem += nextRecurrent[stateIndex] * kNorm[kh*g.kHd+i]
			}
			v := convOut[2*keyDim+h*g.vHd+vd]
			delta := (v - kvmem) * beta
			var readout float32
			for i := 0; i < g.kHd; i++ {
				stateIndex := (h*g.kHd+i)*g.vHd + vd
				nextRecurrent[stateIndex] += kNorm[kh*g.kHd+i] * delta
				readout += nextRecurrent[stateIndex] * qNorm[kh*g.kHd+i]
			}
			core[h*g.vHd+vd] = readout
		}
		var ss float32
		for vd := 0; vd < g.vHd; vd++ {
			v := core[h*g.vHd+vd]
			ss += v * v
		}
		inv := float32(1 / math.Sqrt(float64(ss)/float64(g.vHd)+float64(g.eps)))
		for vd := 0; vd < g.vHd; vd++ {
			i := h*g.vHd + vd
			core[i] = d.norm[vd] * (core[i] * inv) * gdnSilu(z[i])
		}
	}
	return gdnMatVec(d.outW, core, g.hidden, valueDim), nextConv, nextRecurrent
}

func gdnCosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return ab / math.Sqrt(aa*bb)
}

func gdnMaxAbs(a, b []float32) float64 {
	var max float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > max {
			max = d
		}
	}
	return max
}

func uploadGDN(t *testing.T, be compute.Backend, shape []int, data []float32) compute.Tensor {
	t.Helper()
	resident := be.Upload(compute.NewF32(compute.Default(), shape, data), compute.F32)
	t.Cleanup(func() { be.Free(resident) })
	if _, host := be.Host(resident); host {
		t.Fatalf("uploaded tensor %v is host-addressable; expected CUDA residency", shape)
	}
	return resident
}

func TestCUDAQwen35GDNWholeOperationMatchesIndependentReference(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	if model.Qwen35GDNCUDAPath != compute.Qwen35GDNCUDAPath {
		t.Fatalf("model/compute Qwen35 GDN path constants diverged: model=%q compute=%q", model.Qwen35GDNCUDAPath, compute.Qwen35GDNCUDAPath)
	}
	if got := be.Qwen35GDNPath(); got != model.Qwen35GDNCUDAPath {
		t.Fatalf("Qwen35GDNPath = %q, want %q", got, model.Qwen35GDNCUDAPath)
	}

	g := gdnFixtureGeometry{hidden: 16, nK: 2, nV: 4, kHd: 4, vHd: 4, kernel: 3, eps: 1e-5}
	d := newGDNFixtureData(g)
	wantOut, wantConv, wantRecurrent := referenceGDNDecode(g, d)

	x := uploadGDN(t, be, []int{g.hidden}, d.x)
	inQKV := uploadGDN(t, be, []int{g.convDim(), g.hidden}, d.inQKV)
	inZ := uploadGDN(t, be, []int{g.valueDim(), g.hidden}, d.inZ)
	inB := uploadGDN(t, be, []int{g.nV, g.hidden}, d.inB)
	inA := uploadGDN(t, be, []int{g.nV, g.hidden}, d.inA)
	convW := uploadGDN(t, be, []int{g.convDim(), 1, g.kernel}, d.convW)
	aLog := uploadGDN(t, be, []int{g.nV}, d.aLog)
	dtBias := uploadGDN(t, be, []int{g.nV}, d.dtBias)
	norm := uploadGDN(t, be, []int{g.vHd}, d.norm)
	outW := uploadGDN(t, be, []int{g.hidden, g.valueDim()}, d.outW)
	convState := uploadGDN(t, be, []int{g.kernel - 1, g.convDim()}, d.convState)
	recurrentState := uploadGDN(t, be, []int{g.nV, g.kHd, g.vHd}, d.recurrentState)

	be.ResetHostXfer()
	be.ResetQwen35GDNOperationCount()
	xferBefore := be.HostXferBytes()
	opsBefore := be.Qwen35GDNOperationCount()
	gotDev, nextConvDev, nextRecurrentDev, err := be.Qwen35GDNDecode(
		x, inQKV, inZ, inB, inA, convW, aLog, dtBias, norm, outW,
		convState, recurrentState,
		g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps,
	)
	if err != nil {
		t.Fatalf("Qwen35GDNDecode real CUDA operation: %v", err)
	}
	if got := be.Qwen35GDNOperationCount() - opsBefore; got != 1 {
		t.Fatalf("whole-operation counter delta = %d, want 1", got)
	}
	xferInside := be.HostXferBytes() - xferBefore
	if xferInside != 0 {
		t.Fatalf("device->host bytes inside measured GDN operation = %d, want 0", xferInside)
	}
	for name, tensor := range map[string]compute.Tensor{
		"output": gotDev, "next_conv_state": nextConvDev, "next_recurrent_state": nextRecurrentDev,
	} {
		if _, host := be.Host(tensor); host {
			t.Fatalf("%s became host-addressable inside the whole operation", name)
		}
	}

	gotOut := be.Read(gotDev)
	gotConv := be.Read(nextConvDev)
	gotRecurrent := be.Read(nextRecurrentDev)
	if len(gotOut) != len(wantOut) || len(gotConv) != len(wantConv) || len(gotRecurrent) != len(wantRecurrent) {
		t.Fatalf("result lengths out/conv/recurrent = %d/%d/%d, want %d/%d/%d",
			len(gotOut), len(gotConv), len(gotRecurrent), len(wantOut), len(wantConv), len(wantRecurrent))
	}
	cosine := gdnCosine(wantOut, gotOut)
	if cosine < model.Qwen35GDNParityCosineMin {
		t.Fatalf("real CUDA whole-operation cosine %.9f < %.3f (max_abs=%.3e)", cosine, model.Qwen35GDNParityCosineMin, gdnMaxAbs(wantOut, gotOut))
	}
	stateCosine := gdnCosine(append(append([]float32(nil), wantConv...), wantRecurrent...), append(append([]float32(nil), gotConv...), gotRecurrent...))
	if stateCosine < model.Qwen35GDNParityCosineMin {
		t.Fatalf("updated device-state cosine %.9f < %.3f", stateCosine, model.Qwen35GDNParityCosineMin)
	}
	maxAbs := math.Max(gdnMaxAbs(wantOut, gotOut), math.Max(gdnMaxAbs(wantConv, gotConv), gdnMaxAbs(wantRecurrent, gotRecurrent)))
	wantReadBytes := uint64(len(gotOut)+len(gotConv)+len(gotRecurrent)) * 4
	readBytes := be.HostXferBytes() - xferBefore
	if readBytes != wantReadBytes {
		t.Fatalf("post-operation proof reads transferred %d bytes, want %d", readBytes, wantReadBytes)
	}
	t.Logf("Qwen3.5/3.6 GDN CUDA whole operation: path=%s cosine=%.9f state_cosine=%.9f max_abs=%.3e operations=%d d2h_inside=%d d2h_proof_reads=%d",
		be.Qwen35GDNPath(), cosine, stateCosine, maxAbs, be.Qwen35GDNOperationCount()-opsBefore, xferInside, readBytes)
}

func TestCUDAQwen35GDNInvalidGeometryFailsClosed(t *testing.T) {
	be := cudaGDNBackend(t)
	be.ResetQwen35GDNOperationCount()
	_, _, _, err := be.Qwen35GDNDecode(
		compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, compute.Tensor{},
		compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, compute.Tensor{},
		compute.Tensor{}, compute.Tensor{},
		2, 3, 4, 4, 3, 1e-5,
	)
	var geometry *compute.Qwen35GDNGeometryError
	if !errors.As(err, &geometry) {
		t.Fatalf("invalid head grouping error = %T %v, want *compute.Qwen35GDNGeometryError", err, err)
	}
	if got := be.Qwen35GDNOperationCount(); got != 0 {
		t.Fatalf("invalid geometry launched %d GDN operations, want 0", got)
	}
}
