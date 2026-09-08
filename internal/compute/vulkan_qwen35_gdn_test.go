//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
	"unsafe"
)

type qwen35GDNPreprojectedParityOracleEvent struct {
	Schema         string                                    `json:"schema"`
	Selector       string                                    `json:"selector"`
	TestName       string                                    `json:"test_name"`
	OracleKind     string                                    `json:"oracle_kind"`
	Engine         string                                    `json:"engine"`
	DeviceObserved bool                                      `json:"device_observed"`
	CaseCount      int                                       `json:"case_count"`
	Passed         bool                                      `json:"passed"`
	Observed       qwen35GDNPreprojectedParityOracleObserved `json:"observed"`
	Bounds         qwen35GDNPreprojectedParityOracleBounds   `json:"bounds"`
}

type qwen35GDNPreprojectedParityOracleObserved struct {
	MaxAbsDelta   float64 `json:"max_abs_delta"`
	StateIdentity bool    `json:"state_identity"`
	FiniteOutput  bool    `json:"finite_output"`
}

type qwen35GDNPreprojectedParityOracleBounds struct {
	MaxAbsDelta          float64 `json:"max_abs_delta"`
	RequireStateIdentity bool    `json:"require_state_identity"`
	RequireFinite        bool    `json:"require_finite"`
}

func formatQwen35GDNPreprojectedParityOracle(maxAbsDelta float64, stateIdentity, finiteOutput bool, caseCount int) ([]byte, error) {
	if caseCount <= 0 {
		caseCount = 4
	}
	passed := stateIdentity && finiteOutput && maxAbsDelta <= 2e-4
	event := qwen35GDNPreprojectedParityOracleEvent{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       "qwen35_gdn_preprojected",
		TestName:       "TestVulkanQwen35GDNPreprojectedParityAndStateContinuity",
		OracleKind:     "state_continuity",
		Engine:         "fak-native/vulkan",
		DeviceObserved: true,
		CaseCount:      caseCount,
		Passed:         passed,
		Observed: qwen35GDNPreprojectedParityOracleObserved{
			MaxAbsDelta:   maxAbsDelta,
			StateIdentity: stateIdentity,
			FiniteOutput:  finiteOutput,
		},
		Bounds: qwen35GDNPreprojectedParityOracleBounds{
			MaxAbsDelta:          2e-4,
			RequireStateIdentity: true,
			RequireFinite:        true,
		},
	}
	return json.Marshal(event)
}

func TestVulkanQwen35GDNPreprojectedParityAndStateContinuity(t *testing.T) {
	be := vk(t)

	var worstMaxAbsDelta float64
	allStateIdentity := true
	allFinite := true
	caseCount := 0

	testGDN := func(t *testing.T, tokens, nK, nV, kHd, vHd, kernel int, nonzeroRecState bool, inputScale float32) {
		t.Helper()
		convDim := 2*nK*kHd + nV*vHd
		valueDim := nV * vHd
		mixed := make([]float32, tokens*convDim)
		z := make([]float32, tokens*valueDim)
		for i := range mixed {
			mixed[i] = float32((i%17)-8) * inputScale
		}
		for i := range z {
			// Nontrivial gates covering negative, near-zero, and positive saturating regions
			z[i] = float32((i%23)-11) * .15
		}
		beta := make([]float32, tokens*nV)
		alpha := make([]float32, tokens*nV)
		for i := range beta {
			beta[i] = float32((i%13)-6) * .2
			alpha[i] = float32((i%11)-5) * .15
		}
		convW := make([]float32, convDim*kernel)
		for c := 0; c < convDim; c++ {
			convW[c*kernel] = .1
			convW[c*kernel+1] = -.2
			convW[c*kernel+2] = .7
		}
		aLog := make([]float32, nV)
		dtBias := make([]float32, nV)
		for h := 0; h < nV; h++ {
			aLog[h] = -1.0 + float32(h)*0.1
			dtBias[h] = 0.1 - float32(h)*0.05
		}
		norm := make([]float32, vHd)
		for i := range norm {
			norm[i] = .9 + float32(i%5)*.05
		}
		convStateHost := make([]float32, (kernel-1)*convDim)
		for i := range convStateHost {
			convStateHost[i] = float32((i%7)-3) * (inputScale * .8)
		}
		recStateHost := make([]float32, nV*kHd*vHd)
		if nonzeroRecState {
			for i := range recStateHost {
				recStateHost[i] = float32((i%19)-9) * .03
			}
		}
		upload := func(shape []int, data []float32, class MemoryClass, name string) Tensor {
			tensor := be.UploadClass(NewF32(Default(), shape, data), F32, class, name)
			t.Cleanup(func() { be.Free(tensor) })
			return tensor
		}
		m := upload([]int{tokens, convDim}, mixed, MemoryActivation, "gdn mixed")
		zt := upload([]int{tokens, valueDim}, z, MemoryActivation, "gdn z")
		bt := upload([]int{tokens, nV}, beta, MemoryActivation, "gdn beta")
		at := upload([]int{tokens, nV}, alpha, MemoryActivation, "gdn alpha")
		cw := upload([]int{convDim, kernel}, convW, MemoryWeights, "gdn conv")
		al := upload([]int{nV}, aLog, MemoryWeights, "gdn alog")
		dt := upload([]int{nV}, dtBias, MemoryWeights, "gdn dt")
		nw := upload([]int{vHd}, norm, MemoryWeights, "gdn norm")
		cs := upload([]int{kernel - 1, convDim}, convStateHost, MemoryKVCache, "gdn conv state")
		rs := upload([]int{nV, kHd, vHd}, recStateHost, MemoryKVCache, "gdn recurrent state")
		oldC, oldR := cs.Buf(), rs.Buf()
		out, err := be.Qwen35GDNPreprojected(m, zt, bt, at, cw, al, dt, nw, cs, rs, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { be.Free(out) })
		if cs.Buf() != oldC || rs.Buf() != oldR {
			allStateIdentity = false
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
				if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
					allFinite = false
					t.Fatalf("%s[%d] must be finite: got=%g want=%g", name, i, a[i], b[i])
				}
				delta := math.Abs(float64(a[i] - b[i]))
				if delta > worstMaxAbsDelta {
					worstMaxAbsDelta = delta
				}
				if delta > 2e-4 {
					t.Fatalf("%s[%d]=%g want %g", name, i, a[i], b[i])
				}
			}
		}
		closeVec("output", got, want)
		closeVec("conv state", gotC, wantC)
		closeVec("recurrent state", gotR, wantR)
		caseCount++
	}

	t.Run("ZeroStateTwoTokens", func(t *testing.T) {
		// A value head wider than one 64-lane workgroup proves the shader loops
		// lanes without racing the cross-column RMS reduction.
		testGDN(t, 2, 1, 1, 2, 65, 3, false, .025)
	})

	t.Run("NonzeroInitialState", func(t *testing.T) {
		// Nonzero initial recurrent state verifies state decay and delta accumulation
		// from existing memory rather than zeroed memory.
		testGDN(t, 2, 1, 1, 2, 65, 3, true, .025)
	})

	t.Run("MultiTokenMultiHeadSequence", func(t *testing.T) {
		// Multi-token sequence (tokens=5, nK=2, nV=4) with group repeat factor 2,
		// nonzero initial state, and nontrivial gating.
		testGDN(t, 5, 2, 4, 4, 65, 3, true, .025)
	})

	t.Run("SmallNormKeepsL2EpsilonSeparateFromRMS", func(t *testing.T) {
		testGDN(t, 3, 1, 2, 4, 65, 3, true, .00025)
	})

	oracleJSON, err := formatQwen35GDNPreprojectedParityOracle(worstMaxAbsDelta, allStateIdentity, allFinite, caseCount)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
}

func TestVulkanQwen35GDNPreprojectedParityOracleFormat(t *testing.T) {
	raw, err := formatQwen35GDNPreprojectedParityOracle(1.2e-4, true, true, 4)
	if err != nil {
		t.Fatalf("formatQwen35GDNPreprojectedParityOracle failed: %v", err)
	}
	var parsed struct {
		Schema         string `json:"schema"`
		Selector       string `json:"selector"`
		TestName       string `json:"test_name"`
		OracleKind     string `json:"oracle_kind"`
		Engine         string `json:"engine"`
		DeviceObserved bool   `json:"device_observed"`
		CaseCount      int    `json:"case_count"`
		Passed         bool   `json:"passed"`
		Observed       struct {
			MaxAbsDelta   float64 `json:"max_abs_delta"`
			StateIdentity bool    `json:"state_identity"`
			FiniteOutput  bool    `json:"finite_output"`
		} `json:"observed"`
		Bounds struct {
			MaxAbsDelta          float64 `json:"max_abs_delta"`
			RequireStateIdentity bool    `json:"require_state_identity"`
			RequireFinite        bool    `json:"require_finite"`
		} `json:"bounds"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("unmarshal formatted oracle failed: %v", err)
	}
	if parsed.Schema != "fak.strix.subkernel-parity/v1" {
		t.Errorf("schema = %q, want fak.strix.subkernel-parity/v1", parsed.Schema)
	}
	if parsed.Selector != "qwen35_gdn_preprojected" {
		t.Errorf("selector = %q, want qwen35_gdn_preprojected", parsed.Selector)
	}
	if parsed.TestName != "TestVulkanQwen35GDNPreprojectedParityAndStateContinuity" {
		t.Errorf("test_name = %q, want TestVulkanQwen35GDNPreprojectedParityAndStateContinuity", parsed.TestName)
	}
	if parsed.OracleKind != "state_continuity" {
		t.Errorf("oracle_kind = %q, want state_continuity", parsed.OracleKind)
	}
	if parsed.Engine != "fak-native/vulkan" {
		t.Errorf("engine = %q, want fak-native/vulkan", parsed.Engine)
	}
	if !parsed.DeviceObserved {
		t.Errorf("device_observed must be true")
	}
	if parsed.CaseCount != 4 {
		t.Errorf("case_count = %d, want 4", parsed.CaseCount)
	}
	if !parsed.Passed {
		t.Errorf("passed must be true")
	}
	if parsed.Observed.MaxAbsDelta != 1.2e-4 {
		t.Errorf("observed max_abs_delta = %g, want 1.2e-4", parsed.Observed.MaxAbsDelta)
	}
	if !parsed.Observed.StateIdentity {
		t.Errorf("observed state_identity must be true")
	}
	if !parsed.Observed.FiniteOutput {
		t.Errorf("observed finite_output must be true")
	}
	if parsed.Bounds.MaxAbsDelta != 2e-4 {
		t.Errorf("bounds max_abs_delta = %g, want 2e-4", parsed.Bounds.MaxAbsDelta)
	}
	if !parsed.Bounds.RequireStateIdentity {
		t.Errorf("bounds require_state_identity must be true")
	}
	if !parsed.Bounds.RequireFinite {
		t.Errorf("bounds require_finite must be true")
	}

	// Boundary failure: delta exceeds threshold
	failRaw, err := formatQwen35GDNPreprojectedParityOracle(3e-4, true, true, 4)
	if err != nil {
		t.Fatalf("format failed oracle failed: %v", err)
	}
	if err := json.Unmarshal(failRaw, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when max_abs_delta > 2e-4")
	}

	// Boundary failure: state identity broken
	failRaw2, err := formatQwen35GDNPreprojectedParityOracle(1.2e-4, false, true, 4)
	if err != nil {
		t.Fatalf("format failed oracle 2 failed: %v", err)
	}
	if err := json.Unmarshal(failRaw2, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle 2 failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when state_identity=false")
	}

	// Boundary failure: non-finite output
	failRaw3, err := formatQwen35GDNPreprojectedParityOracle(1.2e-4, true, false, 4)
	if err != nil {
		t.Fatalf("format failed oracle 3 failed: %v", err)
	}
	if err := json.Unmarshal(failRaw3, &parsed); err != nil {
		t.Fatalf("unmarshal fail oracle 3 failed: %v", err)
	}
	if parsed.Passed {
		t.Errorf("expected passed=false when finite_output=false")
	}
}

func qwen35GDNPreprojectedOracle(mixed, z, beta, alpha, convW, aLog, dtBias, norm, convState, recurrent []float32, tokens, nK, nV, kHd, vHd, kernel int, eps float32) ([]float32, []float32, []float32) {
	convDim := 2*nK*kHd + nV*vHd
	keyDim := nK * kHd
	hist := kernel - 1
	cs := append([]float32(nil), convState...)
	rs := append([]float32(nil), recurrent...)
	out := make([]float32, tokens*nV*vHd)
	sig := func(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
	silu := func(x float32) float32 { return x * sig(x) }
	soft := func(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) }
	qScale := 1 / float32(math.Sqrt(float64(kHd)))
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
			qi := (1 / float32(math.Sqrt(float64(q2)+1e-6))) * qScale
			ki := 1 / float32(math.Sqrt(float64(k2)+1e-6))
			av := float32(math.Exp(float64(-float32(math.Exp(float64(aLog[h]))) * soft(alpha[t*nV+h]+dtBias[h]))))
			bv := sig(beta[t*nV+h])
			vals := make([]float32, vHd)
			for d := 0; d < vHd; d++ {
				v := co[2*keyDim+h*vHd+d]
				kvmem := float32(0)
				for i := 0; i < kHd; i++ {
					k := co[keyDim+kh*kHd+i] * ki
					si := (h*kHd+i)*vHd + d
					rs[si] *= av
					kvmem += rs[si] * k
				}
				delta := (v - kvmem) * bv
				acc := float32(0)
				for i := 0; i < kHd; i++ {
					q := co[kh*kHd+i] * qi
					k := co[keyDim+kh*kHd+i] * ki
					si := (h*kHd+i)*vHd + d
					rs[si] += k * delta
					acc += q * rs[si]
				}
				vals[d] = acc
			}
			ss := float32(0)
			for _, x := range vals {
				ss += x * x
			}
			inv := 1 / float32(math.Sqrt(float64(ss/float32(vHd)+eps)))
			for d, x := range vals {
				zv := z[t*nV*vHd+h*vHd+d]
				out[t*nV*vHd+h*vHd+d] = norm[d] * (x * inv) * silu(zv)
			}
		}
	}
	return out, cs, rs
}

func BenchmarkVulkanQwen35GDNPreprojected(b *testing.B) {
	be, ok := Pick("vulkan").(*vulkanBackend)
	if !ok {
		b.Skip("Vulkan backend unavailable")
	}
	tokens, nK, nV, kHd, vHd, kernel := 1, 16, 48, 128, 128, 4
	convDim := 2*nK*kHd + nV*vHd
	valueDim := nV * vHd
	mixed := make([]float32, tokens*convDim)
	z := make([]float32, tokens*valueDim)
	beta, alpha := make([]float32, tokens*nV), make([]float32, tokens*nV)
	convW := make([]float32, convDim*kernel)
	aLog, dtBias := make([]float32, nV), make([]float32, nV)
	norm := make([]float32, vHd)
	convStateHost := make([]float32, (kernel-1)*convDim)
	recStateHost := make([]float32, nV*kHd*vHd)

	upload := func(shape []int, data []float32, class MemoryClass, name string) Tensor {
		return be.UploadClass(NewF32(Default(), shape, data), F32, class, name)
	}
	m := upload([]int{tokens, convDim}, mixed, MemoryActivation, "gdn mixed")
	defer be.Free(m)
	zt := upload([]int{tokens, valueDim}, z, MemoryActivation, "gdn z")
	defer be.Free(zt)
	bt := upload([]int{tokens, nV}, beta, MemoryActivation, "gdn beta")
	defer be.Free(bt)
	at := upload([]int{tokens, nV}, alpha, MemoryActivation, "gdn alpha")
	defer be.Free(at)
	cw := upload([]int{convDim, kernel}, convW, MemoryWeights, "gdn conv")
	defer be.Free(cw)
	al := upload([]int{nV}, aLog, MemoryWeights, "gdn alog")
	defer be.Free(al)
	dt := upload([]int{nV}, dtBias, MemoryWeights, "gdn dt")
	defer be.Free(dt)
	nw := upload([]int{vHd}, norm, MemoryWeights, "gdn norm")
	defer be.Free(nw)
	cs := upload([]int{kernel - 1, convDim}, convStateHost, MemoryKVCache, "gdn conv state")
	defer be.Free(cs)
	rs := upload([]int{nV, kHd, vHd}, recStateHost, MemoryKVCache, "gdn recurrent state")
	defer be.Free(rs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := be.Qwen35GDNPreprojected(m, zt, bt, at, cw, al, dt, nw, cs, rs, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
		if err != nil {
			b.Fatal(err)
		}
		be.Free(out)
	}
}

func TestVulkanQwen35GDNConvTiledChannelTransposeDirectFailClosed(t *testing.T) {
	v := &vulkanBackend{}

	dummyByte := byte(0)
	makeResident := func(shape []int) Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		return Tensor{
			Dtype:  F32,
			Layout: RowMajor,
			Shape:  shape,
			buf:    &vulkanBuf{ptr: unsafe.Pointer(&dummyByte), n: n * 4},
			be:     v,
		}
	}

	tokens := 4
	convDim := 64
	kernel := 4
	hist := kernel - 1

	// 1. Unaligned strides fail closed with *Qwen35GDNGeometryError
	t.Run("UnalignedStride", func(t *testing.T) {
		unalignedDim := 65 // 260 bytes (not multiple of 32)
		m := makeResident([]int{tokens, unalignedDim})
		w := makeResident([]int{unalignedDim, kernel})
		s := makeResident([]int{hist, unalignedDim})
		_, _, err := v.Qwen35GDNConvTiledChannelTranspose(m, w, s, tokens, unalignedDim, kernel)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		var geomErr *Qwen35GDNGeometryError
		if !errors.As(err, &geomErr) {
			t.Fatalf("expected *Qwen35GDNGeometryError, got %T: %v", err, err)
		}
		if !errors.Is(err, ErrVulkanInvalidGeometry) {
			t.Fatalf("expected ErrVulkanInvalidGeometry, got %v", err)
		}
	})

	// 2. Non-positive dimensions fail closed before execution with *Qwen35GDNGeometryError
	t.Run("ZeroOrNegativeDimensions", func(t *testing.T) {
		m := makeResident([]int{tokens, convDim})
		w := makeResident([]int{convDim, kernel})
		s := makeResident([]int{hist, convDim})
		dimCases := [][3]int{
			{0, convDim, kernel},
			{-1, convDim, kernel},
			{tokens, 0, kernel},
			{tokens, -64, kernel},
			{tokens, convDim, 0},
			{tokens, convDim, -1},
		}
		for _, dc := range dimCases {
			_, _, err := v.Qwen35GDNConvTiledChannelTranspose(m, w, s, dc[0], dc[1], dc[2])
			if err == nil {
				t.Fatalf("expected error for dims %v, got nil", dc)
			}
			var geomErr *Qwen35GDNGeometryError
			if !errors.As(err, &geomErr) {
				t.Fatalf("expected *Qwen35GDNGeometryError, got %T: %v", err, err)
			}
			if !errors.Is(err, ErrVulkanInvalidGeometry) {
				t.Fatalf("expected ErrVulkanInvalidGeometry, got %v", err)
			}
		}
	})

	// 3. Nil tensors or non-resident tensors fail closed with *Qwen35GDNResidencyError
	t.Run("NilOrNonResidentTensors", func(t *testing.T) {
		m := makeResident([]int{tokens, convDim})
		w := makeResident([]int{convDim, kernel})
		s := makeResident([]int{hist, convDim})

		c := cpu()
		hostM := NewF32(c, []int{tokens, convDim}, make([]float32, tokens*convDim))
		nilTensor := Tensor{}
		typedNilBuf := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tokens, convDim}, buf: (*vulkanBuf)(nil), be: v}
		nullPtrBuf := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tokens, convDim}, buf: &vulkanBuf{ptr: nil}, be: v}

		resCases := []struct {
			name    string
			m, w, s Tensor
		}{
			{"NilMixed", nilTensor, w, s},
			{"HostMixed", hostM, w, s},
			{"TypedNilBufMixed", typedNilBuf, w, s},
			{"NullPtrBufMixed", nullPtrBuf, w, s},
			{"NilWeights", m, nilTensor, s},
			{"NilState", m, w, nilTensor},
		}

		for _, rc := range resCases {
			t.Run(rc.name, func(t *testing.T) {
				_, _, err := v.Qwen35GDNConvTiledChannelTranspose(rc.m, rc.w, rc.s, tokens, convDim, kernel)
				if err == nil {
					t.Fatalf("expected error for %s, got nil", rc.name)
				}
				var resErr *Qwen35GDNResidencyError
				var geomErr *Qwen35GDNGeometryError
				if !errors.As(err, &resErr) && !errors.As(err, &geomErr) {
					t.Fatalf("expected *Qwen35GDNResidencyError or *Qwen35GDNGeometryError, got %T: %v", err, err)
				}
				if !errors.Is(err, ErrVulkanInvalidGeometry) {
					t.Fatalf("expected ErrVulkanInvalidGeometry, got %v", err)
				}
			})
		}
	})
}
