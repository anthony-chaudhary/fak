package model

import (
	"fmt"
	"math"
	"testing"
)

// linearAttnStepAllocatingReference freezes the pre-reuse implementation at
// 7b08d6da99bcabf804faaaccb9fba2f654f6f82b. Keep its allocation and arithmetic
// independent of linearDecodeScratch so poison/reuse parity is non-circular.
func (s *Session) linearAttnStepAllocatingReference(l int, xn []float32, mat matKernel) []float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	nK, nV, kHd, vHd, keyDim, valDim, convDim := cfg.linearAttnDims()
	K := cfg.LinearConvKernelDim
	eps := float32(cfg.RMSNormEps)
	p := func(str string) string { return layerName(l, str) }
	if s.Cache.linear == nil {
		s.Cache.linear = newLinearAttnCache(cfg)
	}
	lst := s.Cache.linear.layer(cfg, l)
	if _, resident := mat.(sessionQ4KKernel); resident {
		if out, _, accepted, err := s.tryQwen35MetalDecodeMixer(l, xn); accepted {
			if err != nil {
				panic(err)
			}
			s.tapOp(l, "out", out)
			return out
		}
	}

	t := s.phaseStart()
	xp := mat.prep(xn)
	// The four GDN in_proj weights all read the same normed input xp — group them so the
	// resident-Q4_K Metal kernel issues the q4_k-resident members (qkv, z) in a single command
	// buffer (the per-token decode lever, #67); the tiny b/a fall back per-call.
	in4 := mulGroup(mat, []string{
		p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"),
		p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight"),
	}, xp, []int{convDim, valDim, nV, nV}, H)
	mixed, zAll, bvec, avec := in4[0], in4[1], in4[2], in4[3]
	s.phaseEnd("qwen35_linear_step_in_proj", t)
	conv := m.tensor(p("linear_attn.conv1d.weight")) // [convDim*K] depthwise
	aLog := m.tensor(p("linear_attn.A_log"))
	dtBias := m.tensor(p("linear_attn.dt_bias"))
	normW := m.tensor(p("linear_attn.norm.weight"))

	if core, used, err := s.tryQwen35MetalGDNDecode(l, mixed, zAll, bvec, avec, conv, aLog, dtBias, normW, eps); used {
		if err != nil {
			panic(err)
		}
		out := mat.mul(p("linear_attn.out_proj.weight"), mat.prep(core), H, valDim)
		s.tapOp(l, "out", out)
		return out
	}

	convOut := make([]float32, convDim)
	t = s.phaseStart()
	for c := 0; c < convDim; c++ {
		var acc float32
		cb := c * K
		for j := 0; j < K; j++ {
			var row []float32
			if j == K-1 {
				row = mixed
			} else {
				idx := len(lst.conv) + j - (K - 1)
				if idx < 0 {
					continue
				}
				row = lst.conv[idx]
			}
			acc += conv[cb+j] * row[c]
		}
		convOut[c] = silu(acc)
	}
	s.phaseEnd("qwen35_linear_step_conv", t)
	s.tapOp(l, "convOut", convOut)
	lst.pushConvRow(mixed, K-1)

	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK
	core := make([]float32, valDim)
	qNorm := make([]float32, keyDim)
	kNorm := make([]float32, keyDim)
	q := convOut[0:keyDim]
	k := convOut[keyDim : 2*keyDim]
	v := convOut[2*keyDim : 2*keyDim+valDim]
	t = s.phaseStart()
	for h := 0; h < nK; h++ {
		l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
		l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
		for i := h * kHd; i < (h+1)*kHd; i++ {
			qNorm[i] *= scale
		}
	}
	s.phaseEnd("qwen35_linear_step_qk_norm", t)
	s.tapOp(l, "qk_norm", qNorm)
	t = s.phaseStart()
	// Each value head's Gated-DeltaNet update is self-contained — it reads the shared (now
	// read-only) qNorm/kNorm and v, mutates only its OWN recurrent state lst.recurrent[h], and
	// writes the disjoint core[h*vHd:(h+1)*vHd]. The kvmem/delta scratch is the only shared
	// hazard, so each worker takes its own. The arithmetic per head is identical to the serial
	// loop, so the result is BIT-IDENTICAL regardless of worker count (pinned by the bit-exact
	// hybrid split-prefill/decode witness in qwen35_test.go) — this just spreads the nV
	// independent heads across cores, the per-token decode lever for the 75%-of-layers GDN scan.
	headStep := func(h int, kvmem, delta []float32) {
		kh := h / repeat
		qn := qNorm[kh*kHd : (kh+1)*kHd]
		kn := kNorm[kh*kHd : (kh+1)*kHd]
		vh := v[h*vHd : (h+1)*vHd]
		st := lst.recurrent[h]
		bt := sigmoidf(bvec[h])
		a := float32(math.Exp(float64(aLog[h])))
		dt := softplus(avec[h] + dtBias[h])
		g := float32(math.Exp(float64(-a * dt)))
		od := core[h*vHd : (h+1)*vHd]
		if HasVectorizedDeltaNetFor(kHd, vHd) {
			VectorizedHeadStep(st, qn, kn, vh, bt, g, od, kvmem, delta)
		} else {
			ScalarHeadStep(st, qn, kn, vh, bt, g, od, kvmem, delta)
		}
	}
	if dispatchWorkers <= 1 || nV*kHd*vHd < parThreshold {
		kvmem := make([]float32, vHd)
		delta := make([]float32, vHd)
		for h := 0; h < nV; h++ {
			headStep(h, kvmem, delta)
		}
	} else {
		parFor(nV, dispatchWorkers, func(lo, hi int) {
			kvmem := make([]float32, vHd) // per-worker scratch (the only cross-head hazard)
			delta := make([]float32, vHd)
			for h := lo; h < hi; h++ {
				headStep(h, kvmem, delta)
			}
		})
	}
	s.phaseEnd("qwen35_linear_step_recurrent", t)
	s.tapOp(l, "recurrent", core)
	t = s.phaseStart()
	for h := 0; h < nV; h++ {
		rmsNormGatedInPlace(core[h*vHd:(h+1)*vHd], normW, zAll[h*vHd:(h+1)*vHd], eps)
	}
	s.phaseEnd("qwen35_linear_step_gated_norm", t)
	s.tapOp(l, "gated_norm", core)
	t = s.phaseStart()
	out := mat.mul(p("linear_attn.out_proj.weight"), mat.prep(core), H, valDim)
	s.phaseEnd("qwen35_linear_step_out_proj", t)
	s.tapOp(l, "out", out)
	return out
}

func poisonLinearDecodeScratch(w *linearDecodeScratch) {
	for _, buf := range [][]float32{w.convOut, w.core, w.qNorm, w.kNorm, w.kvmem, w.delta} {
		for i := range buf {
			buf[i] = float32(math.NaN())
		}
	}
}

func assertLinearDecodeState(t *testing.T, got, want *Session) {
	t.Helper()
	for l := range want.Cache.linear.layers {
		g, w := got.Cache.linear.layers[l], want.Cache.linear.layers[l]
		if len(g.conv) != len(w.conv) || len(g.recurrent) != len(w.recurrent) {
			t.Fatalf("layer %d state shape differs", l)
		}
		for h := range w.recurrent {
			assertFloat32BitsEqual(t, "recurrent", g.recurrent[h], w.recurrent[h])
		}
		for i := range w.conv {
			assertFloat32BitsEqual(t, "convolution", g.conv[i], w.conv[i])
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestQwen35DecodeScratchReuse(t *testing.T) {
	for _, workers := range []int{1, 4} {
		for _, vectorized := range []string{"0", "1"} {
			t.Run(fmt.Sprintf("workers%d/vectorized%s", workers, vectorized), func(t *testing.T) {
				pinDecodeWorkers(t, workers)
				t.Setenv("FAK_VECTORIZED_DELTANET", vectorized)
				cfg := qwen35HybridTestCfg()
				cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim = 128, 128
				t.Logf("vectorized GDN selected: %t", HasVectorizedDeltaNetFor(128, 128))
				m := NewSynthetic(cfg)
				got, want := m.NewSession(), m.NewSession()
				other, otherWant := m.NewSession(), m.NewSession()
				input := make([]float32, cfg.HiddenSize)
				for step := 0; step < 5; step++ {
					for i := range input {
						input[i] = float32((i+step)%11-5) / 11
					}
					for l := 0; l < 3; l++ {
						poisonLinearDecodeScratch(&got.linearDecode)
						out := got.linearAttnStep(l, input, f32Kernel{m})
						ref := want.linearAttnStepAllocatingReference(l, input, f32Kernel{m})
						assertFloat32BitsEqual(t, "output", out, ref)
						assertLinearDecodeState(t, got, want)
						// An independent session must not share workspace, and later calls must
						// not mutate a previously returned projection output.
						other.linearAttnStep(l, input, f32Kernel{m})
						otherWant.linearAttnStepAllocatingReference(l, input, f32Kernel{m})
						assertLinearDecodeState(t, other, otherWant)
						poisonLinearDecodeScratch(&got.linearDecode)
						assertFloat32BitsEqual(t, "retained output", out, ref)
					}
				}
				clone := m.SessionFromPrefix(got.Cache)
				if len(clone.linearDecode.core) != 0 {
					t.Fatal("prefix clone retained transient scratch")
				}
				cloneWant := m.SessionFromPrefix(want.Cache)
				assertFloat32BitsEqual(t, "cloned prefix", clone.linearAttnStep(0, input, f32Kernel{m}), cloneWant.linearAttnStepAllocatingReference(0, input, f32Kernel{m}))
				assertLinearDecodeState(t, clone, cloneWant)
			})
		}
	}
	t.Run("geometry", func(t *testing.T) {
		var w linearDecodeScratch
		for _, dims := range [][3]int{{16, 32, 64}, {256, 512, 1024}, {8, 16, 32}} {
			poisonLinearDecodeScratch(&w)
			w.prepare(dims[0], dims[1], dims[2])
			if len(w.qNorm) != dims[0] || len(w.kNorm) != dims[0] || len(w.core) != dims[1] || len(w.kvmem) != dims[1] || len(w.delta) != dims[1] || len(w.convOut) != dims[2] {
				t.Fatal("workspace geometry mismatch")
			}
			for _, v := range w.core {
				if math.Float32bits(v) != 0 {
					t.Fatal("core was not cleared")
				}
			}
			if n := testing.AllocsPerRun(10, func() { w.prepare(dims[0], dims[1], dims[2]) }); n != 0 {
				t.Fatalf("warm prepare allocs = %g", n)
			}
		}
	})
	t.Run("whole-step-allocations", func(t *testing.T) {
		pinDecodeWorkers(t, 1)
		cfg := qwen35HybridTestCfg()
		cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim = 128, 128
		m := NewSynthetic(cfg)
		got, want := m.NewSession(), m.NewSession()
		input := make([]float32, cfg.HiddenSize)
		for i := 0; i < cfg.LinearConvKernelDim; i++ {
			got.linearAttnStep(0, input, f32Kernel{m})
			want.linearAttnStepAllocatingReference(0, input, f32Kernel{m})
		}
		reused := testing.AllocsPerRun(10, func() { got.linearAttnStep(0, input, f32Kernel{m}) })
		allocating := testing.AllocsPerRun(10, func() { want.linearAttnStepAllocatingReference(0, input, f32Kernel{m}) })
		if allocating-reused < 6 {
			t.Fatalf("whole step allocations: reference=%g reuse=%g; want at least six removed", allocating, reused)
		}
	})
}

// BenchmarkQwen38LinearDecodeScratch exercises the complete production mixer,
// including projections, at Qwen3.8 GDN head geometry with a synthetic H=32 model.
// It is a CPU component fixture, not a full-checkpoint throughput claim.
func BenchmarkQwen38LinearDecodeScratch(b *testing.B) {
	cfg := qwen35HybridTestCfg()
	cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim = 128, 128
	cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads = 16, 48
	m := NewSynthetic(cfg)
	input := make([]float32, cfg.HiddenSize)
	for i := range input {
		input[i] = float32(i%7-3) / 7
	}
	for _, reuse := range []bool{false, true} {
		b.Run(fmt.Sprintf("reuse=%t", reuse), func(b *testing.B) {
			s := m.NewSession()
			step := s.linearAttnStepAllocatingReference
			if reuse {
				step = s.linearAttnStep
			}
			for i := 0; i < cfg.LinearConvKernelDim; i++ {
				step(0, input, f32Kernel{m})
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				step(0, input, f32Kernel{m})
			}
		})
	}
}
