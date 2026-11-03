package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime medium est=15s lane=default
func TestQwen35GDNChunkedHostContract(t *testing.T) {
	t.Setenv("FAK_VECTORIZED_DELTANET", "")
	t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "")
	t.Run("state_readout_tail_and_refusal", func(t *testing.T) {
		in := qwen35GDNHostFixture(137, 2, 4, 7, 11)
		before := qwen35GDNHostCloneInput(in)
		want := qwen35GDNHostScalarOracle(in)
		got, ok := qwen35GDNHostCompute(in, 3)
		if !ok {
			t.Fatal("finite rectangular input declined")
		}
		qwen35GDNHostCheck(t, "core", want.core, got.core, 3e-5)
		for h := range want.state {
			qwen35GDNHostCheck(t, "state", want.state[h], got.state[h], 3e-5)
		}
		if !reflect.DeepEqual(in, before) {
			t.Fatal("compute mutated borrowed input")
		}
		serial, ok := qwen35GDNHostCompute(in, 1)
		if !ok || !reflect.DeepEqual(got, serial) {
			t.Fatal("worker partition changed output/state")
		}

		first := qwen35GDNHostSlice(in, 0, 64, in.state)
		a, ok := qwen35GDNHostCompute(first, 2)
		if !ok {
			t.Fatal("first complete chunk declined")
		}
		second := qwen35GDNHostSlice(in, 64, in.tokens, a.state)
		b, ok := qwen35GDNHostCompute(second, 4)
		if !ok {
			t.Fatal("resumed prefix with partial final chunk declined")
		}
		qwen35GDNHostCheck(t, "continuation", got.core[64*in.valueHeads*in.valueWidth:], b.core, 3e-5)
		for h := range b.state {
			qwen35GDNHostCheck(t, "continued state", got.state[h], b.state[h], 3e-5)
		}

		// Refusal after a late finite-input overflow must not leak earlier heads
		// or chunks. NaN, infinity, malformed geometry and oversized indices also
		// keep the original scalar path's ownership and exceptional behavior.
		cases := []qwen35GDNHostInput{
			qwen35GDNHostCloneInput(in), qwen35GDNHostCloneInput(in),
			qwen35GDNHostCloneInput(in), qwen35GDNHostCloneInput(in),
			qwen35GDNHostCloneInput(in), qwen35GDNHostCloneInput(in),
		}
		cases[0].q[len(in.q)-1] = float32(math.NaN())
		cases[1].state[3][0] = float32(math.Inf(1))
		cases[2].state[0] = cases[2].state[0][:1]
		cases[3].tokens = int(^uint(0) >> 1)
		cases[4].valueHeads = 3 // non-integral value/key grouping
		late := &cases[5]
		for i := len(late.q) - in.keyHeads*in.keyWidth; i < len(late.q); i++ {
			late.q[i] = math.MaxFloat32
		}
		for i := len(late.k) - in.keyHeads*in.keyWidth; i < len(late.k); i++ {
			late.k[i] = math.MaxFloat32
		}
		for i, bad := range cases {
			snapshot := qwen35GDNHostInputBits(bad)
			out, ok := qwen35GDNHostCompute(bad, 3)
			if ok || out.core != nil || out.state != nil {
				t.Fatalf("refusal %d published a candidate", i)
			}
			if !reflect.DeepEqual(snapshot, qwen35GDNHostInputBits(bad)) {
				t.Fatalf("refusal %d mutated input", i)
			}
		}
		got.core[0] = 99
		got.state[0][0] = 88
		if !reflect.DeepEqual(in, before) {
			t.Fatal("returned owners alias borrowed input")
		}
	})

	t.Run("ordinary_prefill_and_next_decode", func(t *testing.T) {
		for _, q4 := range []bool{false, true} {
			cfg := qwen35HybridTestCfg()
			if q4 {
				cfg = qwen35HybridQ4KTestCfg()
			}
			m := NewSynthetic(cfg)
			m.Quantize()
			if q4 {
				fillQ4KMajority(t, m, cfg)
			}
			oracle, candidate := m.NewSession(), m.NewSession()
			defer oracle.Close()
			defer candidate.Close()
			oracle.Quant, candidate.Quant = !q4, !q4
			oracle.Q4K, candidate.Q4K = q4, q4
			oracle.PhaseProfiler, candidate.PhaseProfiler = NewPhaseProfiler(), NewPhaseProfiler()
			tol := 3e-4
			if !qwen35GDNChunkedHostQualification {
				tol = 0
			}
			// Both panels enter the ordinary batched route, the second from an
			// existing convolution/recurrent prefix. Each has a partial chunk.
			for _, count := range []int{65, 67} {
				ids := make([]int, count)
				for i := range ids {
					ids[i] = (i*7 + 3 + oracle.Cache.Len()) % cfg.VocabSize
				}
				// Restrict only this invocation to the old recurrence. The candidate
				// still enters ordinary Session.Prefill, and every other stage/tag
				// is identical to the baseline's normal batched pipeline.
				var want []float32
				if q4 {
					hidden := oracle.prefillQwen35HybridQ4KHiddenWithGDNMode(ids, oracle.newQwen35PrefillNormWorkspace(len(ids)), qwen35GDNHostForceScalar)
					want = append([]float32(nil), oracle.headResident(hidden)...)
				} else {
					hidden := oracle.prefillQwen35HybridQHiddenWithGDNMode(ids, oracle.newQwen35PrefillNormWorkspace(len(ids)), qwen35GDNHostForceScalar)
					want = append([]float32(nil), oracle.headQ(hidden)...)
				}
				got := append([]float32(nil), candidate.Prefill(ids)...)
				qwen35GDNHostCheck(t, "ordinary logits", want, got, tol)
				qwen35GDNHostCheckCache(t, oracle, candidate, tol)
			}
			marker := candidate.PhaseProfiler.stat["qwen35_linear_chunked_host"]
			if qwen35GDNChunkedHostQualification {
				if marker == nil || marker.Calls != 6 {
					t.Fatalf("ordinary q4=%v selected calls = %+v, want 6", q4, marker)
				}
			} else if marker != nil {
				t.Fatal("default build selected chunkwise recurrence")
			}
			if oracle.PhaseProfiler.stat["qwen35_linear_chunked_host"] != nil {
				t.Fatal("typed scalar oracle selected chunkwise recurrence")
			}
			want := append([]float32(nil), oracle.Step(7)...)
			got := append([]float32(nil), candidate.Step(7)...)
			qwen35GDNHostCheck(t, "next decode logits", want, got, tol)
			qwen35GDNHostCheckCache(t, oracle, candidate, tol)
		}
	})

	t.Run("selection_provenance", func(t *testing.T) {
		s := NewSynthetic(qwen35HybridTestCfg()).NewSession()
		defer s.Close()
		if s.qwen35GDNHostSelected(true, false, 64) != qwen35GDNChunkedHostQualification {
			t.Fatal("typed default/qualification gate")
		}
		if s.qwen35GDNHostSelected(false, false, 64) || s.qwen35GDNHostSelected(true, true, 64) || s.qwen35GDNHostSelected(true, false, 63) {
			t.Fatal("callback, vectorized opt-in or short panel selected")
		}
		s.Metal = true
		if s.qwen35GDNHostSelected(true, false, 64) {
			t.Fatal("Metal selected host chunkwise route")
		}
		s.Metal = false
		s.MetalQ4K = true
		if s.qwen35GDNHostSelected(true, false, 64) {
			t.Fatal("Metal Q4K selected host chunkwise route")
		}
		s.MetalQ4K = false
		s.captureTargetHidden = true
		if s.qwen35GDNHostSelected(true, false, 64) {
			t.Fatal("captured path selected")
		}
	})
}

func qwen35GDNHostFixture(p, nk, nv, kd, vd int) qwen35GDNHostInput {
	keyDim, valDim := nk*kd, nv*vd
	in := qwen35GDNHostInput{
		tokens: p, keyHeads: nk, valueHeads: nv, keyWidth: kd, valueWidth: vd,
		q: make([]float32, p*keyDim), k: make([]float32, p*keyDim),
		conv: make([]float32, p*(2*keyDim+valDim)),
		a:    make([]float32, p*nv), b: make([]float32, p*nv),
		aExp: make([]float32, nv), dtBias: make([]float32, nv), state: make([][]float32, nv),
	}
	for i := range in.q {
		in.q[i] = float32(math.Sin(float64(i+3)*0.17)) / float32(kd)
		in.k[i] = float32(math.Cos(float64(i+5)*0.11)) / float32(kd)
	}
	for i := range in.conv {
		in.conv[i] = float32(math.Sin(float64(i+1)*0.07)) * 0.2
	}
	for i := range in.a {
		in.a[i] = float32(math.Cos(float64(i)*0.13)) - 1
		in.b[i] = float32(math.Sin(float64(i) * 0.19))
	}
	for h := 0; h < nv; h++ {
		in.aExp[h], in.dtBias[h] = 0.25+float32(h)*0.1, -1
		in.state[h] = make([]float32, kd*vd)
		for i := range in.state[h] {
			in.state[h][i] = float32(math.Sin(float64(i+h+1)*0.31)) * 0.05
		}
	}
	// Include exact rounded-decay reset and near-zero/unit beta in a tail.
	in.a[(p-2)*nv] = 10000
	in.b[(p-3)*nv] = -1000
	in.b[(p-1)*nv] = 1000
	return in
}

func qwen35GDNHostCloneInput(in qwen35GDNHostInput) qwen35GDNHostInput {
	in.q = append([]float32(nil), in.q...)
	in.k = append([]float32(nil), in.k...)
	in.conv = append([]float32(nil), in.conv...)
	in.a = append([]float32(nil), in.a...)
	in.b = append([]float32(nil), in.b...)
	in.aExp = append([]float32(nil), in.aExp...)
	in.dtBias = append([]float32(nil), in.dtBias...)
	state := make([][]float32, len(in.state))
	for h := range state {
		state[h] = append([]float32(nil), in.state[h]...)
	}
	in.state = state
	return in
}

func qwen35GDNHostInputBits(in qwen35GDNHostInput) [][]uint32 {
	xs := [][]float32{in.q, in.k, in.conv, in.a, in.b, in.aExp, in.dtBias}
	xs = append(xs, in.state...)
	out := make([][]uint32, len(xs))
	for i, x := range xs {
		out[i] = make([]uint32, len(x))
		for j, v := range x {
			out[i][j] = math.Float32bits(v)
		}
	}
	return out
}

func qwen35GDNHostSlice(in qwen35GDNHostInput, lo, hi int, state [][]float32) qwen35GDNHostInput {
	kd, vd := in.keyHeads*in.keyWidth, in.valueHeads*in.valueWidth
	in.tokens = hi - lo
	in.q = in.q[lo*kd : hi*kd]
	in.k = in.k[lo*kd : hi*kd]
	in.conv = in.conv[lo*(2*kd+vd) : hi*(2*kd+vd)]
	in.a = in.a[lo*in.valueHeads : hi*in.valueHeads]
	in.b = in.b[lo*in.valueHeads : hi*in.valueHeads]
	in.state = state
	return in
}

// This oracle deliberately retains the serial decode order, not the triangular
// equations being tested: decay, decayed-state key projection, delta, update, q.
func qwen35GDNHostScalarOracle(input qwen35GDNHostInput) qwen35GDNHostResult {
	in := qwen35GDNHostCloneInput(input)
	nv, kd, vd := in.valueHeads, in.keyWidth, in.valueWidth
	keyDim, valDim := in.keyHeads*kd, nv*vd
	convDim := 2*keyDim + valDim
	out := qwen35GDNHostResult{core: make([]float32, in.tokens*valDim), state: in.state}
	for h := 0; h < nv; h++ {
		st := out.state[h]
		kh := h / (nv / in.keyHeads)
		mem, delta := make([]float32, vd), make([]float32, vd)
		for t := 0; t < in.tokens; t++ {
			q := in.q[t*keyDim+kh*kd : t*keyDim+(kh+1)*kd]
			k := in.k[t*keyDim+kh*kd : t*keyDim+(kh+1)*kd]
			v := in.conv[t*convDim+2*keyDim+h*vd : t*convDim+2*keyDim+(h+1)*vd]
			beta := sigmoidf(in.b[t*nv+h])
			g := float32(math.Exp(float64(-in.aExp[h] * softplus(in.a[t*nv+h]+in.dtBias[h]))))
			for i := range st {
				st[i] *= g
			}
			for d := range mem {
				mem[d] = 0
			}
			for i := 0; i < kd; i++ {
				for d := 0; d < vd; d++ {
					mem[d] = gdnFMA32(st[i*vd+d], k[i], mem[d])
				}
			}
			for d := range delta {
				delta[d] = (v[d] - mem[d]) * beta
			}
			for i := 0; i < kd; i++ {
				for d := 0; d < vd; d++ {
					st[i*vd+d] = gdnFMA32(k[i], delta[d], st[i*vd+d])
					idx := t*valDim + h*vd + d
					out.core[idx] = gdnFMA32(st[i*vd+d], q[i], out.core[idx])
				}
			}
		}
	}
	return out
}

func qwen35GDNHostCheck(t *testing.T, label string, want, got []float32, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s lengths %d/%d", label, len(want), len(got))
	}
	for i, a := range want {
		b := got[i]
		if tol == 0 && math.Float32bits(a) != math.Float32bits(b) {
			t.Fatalf("%s[%d] default bits differ", label, i)
		}
		if math.IsNaN(float64(a)) || math.IsInf(float64(a), 0) || math.IsNaN(float64(b)) || math.IsInf(float64(b), 0) || math.Abs(float64(a)-float64(b)) > tol*(1+math.Abs(float64(a))) {
			t.Fatalf("%s[%d] got %g want %g tolerance %g", label, i, b, a, tol)
		}
	}
}

func qwen35GDNHostCheckCache(t *testing.T, a, b *Session, tol float64) {
	t.Helper()
	if a.Cache.Len() != b.Cache.Len() {
		t.Fatal("cache position mismatch")
	}
	for l := range a.Cache.K {
		qwen35GDNHostCheck(t, "K", a.Cache.K[l], b.Cache.K[l], tol)
		qwen35GDNHostCheck(t, "Kraw", a.Cache.Kraw[l], b.Cache.Kraw[l], tol)
		qwen35GDNHostCheck(t, "V", a.Cache.V[l], b.Cache.V[l], tol)
	}
	if a.Cache.linear == nil || b.Cache.linear == nil || len(a.Cache.linear.layers) != len(b.Cache.linear.layers) {
		t.Fatal("linear cache shape")
	}
	for l, as := range a.Cache.linear.layers {
		bs := b.Cache.linear.layers[l]
		if len(as.conv) != len(bs.conv) || len(as.recurrent) != len(bs.recurrent) {
			t.Fatal("linear state shape")
		}
		for i := range as.conv {
			qwen35GDNHostCheck(t, "conv", as.conv[i], bs.conv[i], tol)
		}
		for h := range as.recurrent {
			qwen35GDNHostCheck(t, "recurrent", as.recurrent[h], bs.recurrent[h], tol)
		}
	}
}
