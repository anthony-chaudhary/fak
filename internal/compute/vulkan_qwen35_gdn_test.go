//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"testing"
)

func TestVulkanQwen35GDNPreprojectedParityAndStateContinuity(t *testing.T) {
	be, ok := Pick("vulkan").(*vulkanBackend)
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	// A value head wider than one 64-lane workgroup proves the shader loops
	// lanes without racing the cross-column RMS reduction.
	tokens, nK, nV, kHd, vHd, kernel := 2, 1, 1, 2, 65, 3
	convDim := 2*nK*kHd + nV*vHd
	valueDim := nV * vHd
	mixed := make([]float32, tokens*convDim)
	z := make([]float32, tokens*valueDim)
	for i := range mixed {
		mixed[i] = float32((i%17)-8) * .025
	}
	for i := range z {
		z[i] = float32((i%11)-5) * .04
	}
	beta, alpha := []float32{.1, -.2}, []float32{.2, .3}
	convW := make([]float32, convDim*kernel)
	for c := 0; c < convDim; c++ {
		convW[c*kernel] = .1
		convW[c*kernel+1] = -.2
		convW[c*kernel+2] = .7
	}
	aLog, dtBias := []float32{-1}, []float32{.1}
	norm := make([]float32, valueDim)
	for i := range norm {
		norm[i] = .9 + float32(i%5)*.05
	}
	convStateHost := make([]float32, (kernel-1)*convDim)
	recStateHost := make([]float32, nV*kHd*vHd)
	upload := func(shape []int, data []float32, class MemoryClass, name string) Tensor {
		return be.UploadClass(NewF32(Default(), shape, data), F32, class, name)
	}
	m := upload([]int{tokens, convDim}, mixed, MemoryActivation, "gdn mixed")
	zt := upload([]int{tokens, valueDim}, z, MemoryActivation, "gdn z")
	bt := upload([]int{tokens, nV}, beta, MemoryActivation, "gdn beta")
	at := upload([]int{tokens, nV}, alpha, MemoryActivation, "gdn alpha")
	cw := upload([]int{convDim, kernel}, convW, MemoryWeights, "gdn conv")
	al := upload([]int{nV}, aLog, MemoryWeights, "gdn alog")
	dt := upload([]int{nV}, dtBias, MemoryWeights, "gdn dt")
	nw := upload([]int{nV, vHd}, norm, MemoryWeights, "gdn norm")
	cs := upload([]int{kernel - 1, convDim}, convStateHost, MemoryKVCache, "gdn conv state")
	rs := upload([]int{nV, kHd, vHd}, recStateHost, MemoryKVCache, "gdn recurrent state")
	oldC, oldR := cs.Buf(), rs.Buf()
	out, err := be.Qwen35GDNPreprojected(m, zt, bt, at, cw, al, dt, nw, cs, rs, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	if cs.Buf() != oldC || rs.Buf() != oldR {
		t.Fatal("persistent state identity changed")
	}
	got := be.Read(out)
	gotC := be.Read(cs)
	gotR := be.Read(rs)
	want, wantC, wantR := qwen35GDNPreprojectedOracle(mixed, z, beta, alpha, convW, aLog, dtBias, norm, convStateHost, recStateHost, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
	closeVec := func(name string, a, b []float32) {
		if len(a) != len(b) {
			t.Fatalf("%s length", name)
		}
		for i := range a {
			if math.Abs(float64(a[i]-b[i])) > 2e-4 {
				t.Fatalf("%s[%d]=%g want %g", name, i, a[i], b[i])
			}
		}
	}
	closeVec("output", got, want)
	closeVec("conv state", gotC, wantC)
	closeVec("recurrent state", gotR, wantR)
}

func qwen35GDNPreprojectedOracle(mixed, z, beta, alpha, convW, aLog, dtBias, norm, convState, recurrent []float32, tokens, nK, nV, kHd, vHd, kernel int, eps float32) ([]float32, []float32, []float32) {
	convDim := 2*nK*kHd + nV*vHd
	keyDim := nK * kHd
	hist := kernel - 1
	cs := append([]float32(nil), convState...)
	rs := append([]float32(nil), recurrent...)
	out := make([]float32, tokens*nV*vHd)
	sig := func(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
	soft := func(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) }
	for t := 0; t < tokens; t++ {
		co := make([]float32, convDim)
		for c := 0; c < convDim; c++ {
			s := mixed[t*convDim+c] * convW[c*kernel+hist]
			for k := 0; k < hist; k++ {
				s += cs[k*convDim+c] * convW[c*kernel+k]
			}
			for k := 0; k+1 < hist; k++ {
				cs[k*convDim+c] = cs[(k+1)*convDim+c]
			}
			if hist > 0 {
				cs[(hist-1)*convDim+c] = mixed[t*convDim+c]
			}
			co[c] = s * sig(s)
		}
		for h := 0; h < nV; h++ {
			kh := h / (nV / nK)
			q2, k2 := float32(0), float32(0)
			for i := 0; i < kHd; i++ {
				q := co[kh*kHd+i]
				k := co[keyDim+kh*kHd+i]
				q2 += q * q
				k2 += k * k
			}
			qi := 1 / float32(math.Sqrt(float64(q2+eps)))
			ki := 1 / float32(math.Sqrt(float64(k2+eps)))
			av := float32(math.Exp(float64(-float32(math.Exp(float64(aLog[h]))) * soft(alpha[t*nV+h]+dtBias[h]))))
			bv := sig(beta[t*nV+h])
			vals := make([]float32, vHd)
			for d := 0; d < vHd; d++ {
				v := co[2*keyDim+h*vHd+d]
				acc := float32(0)
				for i := 0; i < kHd; i++ {
					q := co[kh*kHd+i] * qi
					k := co[keyDim+kh*kHd+i] * ki
					si := (h*kHd+i)*vHd + d
					rs[si] = av*rs[si] + bv*k*v
					acc += q * rs[si]
				}
				vals[d] = acc * sig(z[t*nV*vHd+h*vHd+d])
			}
			ss := float32(0)
			for _, x := range vals {
				ss += x * x
			}
			inv := 1 / float32(math.Sqrt(float64(ss/float32(vHd)+eps)))
			for d, x := range vals {
				out[t*nV*vHd+h*vHd+d] = x * inv * norm[d]
			}
		}
	}
	return out, cs, rs
}
