//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"runtime"
	"testing"
	"time"
)

// denseUngatedAttentionCPU is the ungated (dense Llama/Qwen2) reference: no qk-norm,
// rotate-half RoPE over the first `rotary` dims, causal softmax over prefix+panel, and
// the plain softmax-weighted V sum as the output (no sigmoid gate).
func denseUngatedAttentionCPU(q, k, v, cosv, sinv, prefixK, prefixV []float32, rows, base, nH, nKV, hd, rotary int, scale float32) (out, kpost []float32) {
	qpost := append([]float32(nil), q...)
	kpost = append([]float32(nil), k...)
	half := rotary / 2
	rot := func(x []float32, off, row int) {
		for dim := 0; dim < half; dim++ {
			a, b := x[off+dim], x[off+half+dim]
			c, s := cosv[row*half+dim], sinv[row*half+dim]
			x[off+dim], x[off+half+dim] = a*c-b*s, a*s+b*c
		}
	}
	for row := 0; row < rows; row++ {
		for head := 0; head < nH; head++ {
			rot(qpost, (row*nH+head)*hd, row)
		}
		for head := 0; head < nKV; head++ {
			rot(kpost, (row*nKV+head)*hd, row)
		}
	}
	allK := append(append([]float32(nil), prefixK...), kpost...)
	allV := append(append([]float32(nil), prefixV...), v...)
	out = make([]float32, rows*nH*hd)
	for row := 0; row < rows; row++ {
		for head := 0; head < nH; head++ {
			kvHead, upto := head/(nH/nKV), base+row+1
			scores := make([]float64, upto)
			maxScore := math.Inf(-1)
			for token := 0; token < upto; token++ {
				var dot float64
				for dim := 0; dim < hd; dim++ {
					dot += float64(qpost[(row*nH+head)*hd+dim]) * float64(allK[(token*nKV+kvHead)*hd+dim])
				}
				scores[token] = dot * float64(scale)
				maxScore = math.Max(maxScore, scores[token])
			}
			var denom float64
			for token := range scores {
				scores[token] = math.Exp(scores[token] - maxScore)
				denom += scores[token]
			}
			for dim := 0; dim < hd; dim++ {
				var sum float64
				for token := 0; token < upto; token++ {
					sum += scores[token] * float64(allV[(token*nKV+kvHead)*hd+dim])
				}
				out[(row*nH+head)*hd+dim] = float32(sum / denom)
			}
		}
	}
	return out, kpost
}

func denseTestBias(n int, seed int) []float32 {
	b := make([]float32, n)
	for i := range b {
		b[i] = float32((i*seed)%23-11) * 0.013
	}
	return b
}

func denseCosineMaxAbs(want, got []float32) (float64, float32) {
	var dot, nw, ng float64
	for i := range want {
		dot += float64(want[i]) * float64(got[i])
		nw += float64(want[i]) * float64(want[i])
		ng += float64(got[i]) * float64(got[i])
	}
	cos := 1.0
	if nw > 0 && ng > 0 {
		cos = dot / math.Sqrt(nw*ng)
	}
	return cos, maxAbsDiff(want, got)
}

// TestProjectionGraphUngatedAttentionParity drives FullAttention and
// FullAttentionDevice with gate == nil across every attention pipeline the default
// selector picks (MLX hd∈{64,128}, the generic qg_attn at hd=96, split-KV at P=1 with
// total>2048, and the online kernel at P>1 with total>4096), with q/k/v biases
// applied through AddBiasInPlace, and checks the output and the appended KV rows
// against the ungated CPU reference.
//
// fak-test:runtime integration est=2s lane=default
func TestProjectionGraphUngatedAttentionParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const input, nH, nKV = 256, 4, 2
	for _, tc := range []struct {
		name                 string
		rows, hd, rotary     int
		base                 int
		checkDeviceKVAppends bool
	}{
		{name: "p1_hd64_mlx", rows: 1, hd: 64, rotary: 64, base: 5, checkDeviceKVAppends: true},
		{name: "p4_hd64_mlx", rows: 4, hd: 64, rotary: 64, base: 3, checkDeviceKVAppends: true},
		{name: "p1_hd128_mlx", rows: 1, hd: 128, rotary: 128, base: 40},
		{name: "p1_hd96_generic", rows: 1, hd: 96, rotary: 64, base: 9},
		{name: "p1_hd128_split_kv", rows: 1, hd: 128, rotary: 128, base: 2500},
		{name: "p2_hd64_online", rows: 2, hd: 64, rotary: 64, base: 4100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows, hd, rotary, base := tc.rows, tc.hd, tc.rotary, tc.base
			qwidth, kvwidth := nH*hd, nKV*hd
			qW := UploadQ4K(q4kTestRaw(qwidth, input, uint64(4131+hd)), qwidth, input)
			kW := UploadQ4K(q4kTestRaw(kvwidth, input, uint64(4132+hd)), kvwidth, input)
			vW := UploadQ4K(q4kTestRaw(kvwidth, input, uint64(4133+hd)), kvwidth, input)
			if qW == nil || kW == nil || vW == nil {
				t.Fatal("ungated attention Q4_K upload")
			}
			qb, kb, vb := denseTestBias(qwidth, 7), denseTestBias(kvwidth, 5), denseTestBias(kvwidth, 3)
			// Unused when qkNorm=false, but the entry requires hd-length norm slices.
			ones := make([]float32, hd)
			for i := range ones {
				ones[i] = 1
			}
			prefixK, prefixV := make([]float32, base*kvwidth), make([]float32, base*kvwidth)
			for i := range prefixK {
				prefixK[i] = float32((i*7)%29-14) * 0.02
				prefixV[i] = float32((i*11)%31-15) * 0.017
			}
			cosv, sinv := make([]float32, rows*(rotary/2)), make([]float32, rows*(rotary/2))
			for row := 0; row < rows; row++ {
				for dim := 0; dim < rotary/2; dim++ {
					angle := float64(base+row) / math.Pow(10000, float64(2*dim)/float64(rotary))
					cosv[row*(rotary/2)+dim], sinv[row*(rotary/2)+dim] = float32(math.Cos(angle)), float32(math.Sin(angle))
				}
			}
			scale := float32(1 / math.Sqrt(float64(hd)))
			x := q4kTestVector(rows*input, int64(4140+rows+base))
			for i := range x {
				x[i] *= 0.05
			}
			// mode 0 attends a host prefix, 1 a full DeviceKV, 2 an attend-only DeviceKV.
			check := func(t *testing.T, label string, mode int) {
				device := mode != 0
				g, err := BeginProjectionGraph(x, nil, nil, rows, input)
				if err != nil {
					t.Fatal(err)
				}
				defer g.Free()
				q, err := g.EncodeQ4K(qW)
				if err != nil {
					t.Fatal(err)
				}
				k, err := g.EncodeQ4K(kW)
				if err != nil {
					t.Fatal(err)
				}
				v, err := g.EncodeQ4K(vW)
				if err != nil {
					t.Fatal(err)
				}
				for i, pair := range []struct {
					r *GraphResult
					b []float32
				}{{q, qb}, {k, kb}, {v, vb}} {
					if err := g.AddBiasInPlace(pair.r, pair.b); err != nil {
						t.Fatalf("bias %d: %v", i, err)
					}
				}
				var kv *DeviceKV
				var att Qwen35GraphAttentionResult
				if device {
					// Two layers; this panel writes layer 1 so the layer offset is exercised.
					if mode == 2 {
						kv = NewDeviceKVAttendOnly(2, base+rows, kvwidth)
					} else {
						kv = NewDeviceKV(2, base+rows, kvwidth)
					}
					if kv == nil {
						t.Fatal("NewDeviceKV")
					}
					defer kv.Close()
					if err := kv.UploadRegion(1, kv.LayerStride(), prefixK); err != nil {
						t.Fatal(err)
					}
					if err := kv.UploadRegion(2, kv.LayerStride(), prefixV); err != nil {
						t.Fatal(err)
					}
					att, err = g.FullAttentionDevice(q, k, v, nil, kv, 1, ones, ones, cosv, sinv, base, nH, nKV, hd, rotary, scale, 1e-6, false, false)
				} else {
					att, err = g.FullAttention(q, k, v, nil, ones, ones, cosv, sinv, prefixK, prefixV, base, nH, nKV, hd, rotary, scale, 1e-6, false, false)
				}
				if err != nil {
					t.Fatalf("%s ungated attention: %v", label, err)
				}
				outs, receipt, err := g.FinishRead(q, k, v, att.Output, att.KRaw, att.KPost, att.V)
				if err != nil {
					t.Fatal(err)
				}
				if !receipt.Committed || !receipt.CompletedWait || receipt.HostReadbacks != 1 {
					t.Fatalf("%s receipt %+v", label, receipt)
				}
				gq, gk, gv := outs[0], outs[1], outs[2]
				wantOut, wantKPost := denseUngatedAttentionCPU(gq, gk, gv, cosv, sinv, prefixK, prefixV, rows, base, nH, nKV, hd, rotary, scale)
				if cos, diff := denseCosineMaxAbs(wantOut, outs[3]); cos < 0.99999 || diff > 2e-4 {
					t.Fatalf("%s output cos=%.7f maxAbs=%.3g", label, cos, diff)
				}
				if diff := maxAbsDiff(gk, outs[4]); diff != 0 {
					t.Fatalf("%s KRaw must equal the biased k projection, maxAbs=%g", label, diff)
				}
				if cos, diff := denseCosineMaxAbs(wantKPost, outs[5]); cos < 0.999999 || diff > 1e-5 {
					t.Fatalf("%s KPost cos=%.7f maxAbs=%.3g", label, cos, diff)
				}
				if diff := maxAbsDiff(gv, outs[6]); diff != 0 {
					t.Fatalf("%s V must alias the biased v projection, maxAbs=%g", label, diff)
				}
				if device && tc.checkDeviceKVAppends {
					got := make([]float32, rows*kvwidth)
					if err := kv.DownloadRegion(1, kv.LayerStride()+base*kvwidth, got); err != nil {
						t.Fatal(err)
					}
					if diff := maxAbsDiff(outs[5], got); diff != 0 {
						t.Fatalf("%s device KPost append differs from the read-back row, maxAbs=%g", label, diff)
					}
					if mode == 2 {
						if err := kv.DownloadRegion(0, kv.LayerStride()+base*kvwidth, got); err == nil {
							t.Fatalf("%s attend-only pair served a KRaw download", label)
						}
					}
					// Layer 0's slice must be untouched by a layer-1 append.
					zero := make([]float32, kv.LayerStride())
					if err := kv.DownloadRegion(1, 0, zero); err != nil {
						t.Fatal(err)
					}
					for i, value := range zero {
						if value != 0 {
							t.Fatalf("%s layer-0 KPost slice written at %d", label, i)
						}
					}
				}
			}
			t.Run("host_prefix", func(t *testing.T) { check(t, "host", 0) })
			t.Run("device_kv", func(t *testing.T) { check(t, "device", 1) })
			t.Run("device_kv_attend_only", func(t *testing.T) { check(t, "attend-only", 2) })
		})
	}
}

// TestProjectionGraphAddBiasInPlaceParity checks the broadcast bias op against the host
// add on a single row and on a panel, its upload accounting, and its refusals.
//
// fak-test:runtime integration est=0.5s lane=default
func TestProjectionGraphAddBiasInPlaceParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const input, out = 256, 96
	w := UploadQ4K(q4kTestRaw(out, input, 9911), out, input)
	if w == nil {
		t.Fatal("bias Q4_K upload")
	}
	bias := denseTestBias(out, 13)
	for _, rows := range []int{1, 3} {
		x := q4kTestVector(rows*input, int64(9912+rows))
		plain, err := BeginProjectionGraph(x, nil, nil, rows, input)
		if err != nil {
			t.Fatal(err)
		}
		r, err := plain.EncodeQ4K(w)
		if err != nil {
			t.Fatal(err)
		}
		base, _, err := plain.FinishRead(r)
		plain.Free()
		if err != nil {
			t.Fatal(err)
		}
		g, err := BeginProjectionGraph(x, nil, nil, rows, input)
		if err != nil {
			t.Fatal(err)
		}
		rb, err := g.EncodeQ4K(w)
		if err != nil {
			t.Fatal(err)
		}
		if err := g.AddBiasInPlace(rb, bias[:out-1]); err == nil {
			t.Fatal("AddBiasInPlace accepted a bias narrower than the result")
		}
		if err := g.AddBiasInPlace(rb, bias); err != nil {
			t.Fatal(err)
		}
		got, receipt, err := g.FinishRead(rb)
		if err != nil {
			t.Fatal(err)
		}
		if want := uint64(rows*input*4 + out*4); receipt.HostUploadBytes != want {
			t.Fatalf("rows=%d HostUploadBytes=%d want %d", rows, receipt.HostUploadBytes, want)
		}
		for i := range got[0] {
			if want := base[0][i] + bias[i%out]; got[0][i] != want {
				t.Fatalf("rows=%d bias[%d]: got %g want %g", rows, i, got[0][i], want)
			}
		}
		if err := g.AddBiasInPlace(rb, bias); err == nil {
			t.Fatal("AddBiasInPlace accepted a committed graph")
		}
		g.Free()
	}
}

// TestProjectionGraphCommittedRefusesInPlaceOps pins the open-graph guard on the
// in-place residual and SwiGLU ops: once Finish commits the command buffer they must
// refuse instead of encoding onto a committed MTLCommandBuffer.
//
// fak-test:runtime integration est=0.5s lane=default
func TestProjectionGraphCommittedRefusesInPlaceOps(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const input, out = 256, 64
	w := UploadQ4K(q4kTestRaw(out, input, 7711), out, input)
	if w == nil {
		t.Fatal("Q4_K upload")
	}
	g, err := BeginProjectionGraph(q4kTestVector(input, 7712), nil, nil, 1, input)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	a, err := g.EncodeQ4K(w)
	if err != nil {
		t.Fatal(err)
	}
	b, err := g.EncodeQ4K(w)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.Finish(); err != nil {
		t.Fatal(err)
	}
	if err := g.AddInPlace(a, b); err == nil {
		t.Fatal("AddInPlace accepted a committed graph")
	}
	if err := g.SwiGLUInPlace(a, b); err == nil {
		t.Fatal("SwiGLUInPlace accepted a committed graph")
	}
}

// TestProjectionGraphDeviceKVRefusesSliceOverflow pins the layer-slice capacity bound:
// a panel whose rows [base, base+P) end past the slice must be refused rather than
// spilling into the next layer's row 0.
//
// fak-test:runtime integration est=0.5s lane=default
func TestProjectionGraphDeviceKVRefusesSliceOverflow(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const input, nH, nKV, hd, capacity = 256, 2, 1, 64, 4
	qwidth, kvwidth := nH*hd, nKV*hd
	qW := UploadQ4K(q4kTestRaw(qwidth, input, 6611), qwidth, input)
	kW := UploadQ4K(q4kTestRaw(kvwidth, input, 6612), kvwidth, input)
	if qW == nil || kW == nil {
		t.Fatal("Q4_K upload")
	}
	ones := make([]float32, hd)
	for i := range ones {
		ones[i] = 1
	}
	kv := NewDeviceKV(2, capacity, kvwidth)
	if kv == nil {
		t.Fatal("NewDeviceKV")
	}
	defer kv.Close()
	for _, tc := range []struct {
		rows, base int
		admit      bool
	}{
		{rows: 1, base: capacity - 1, admit: true},
		{rows: 1, base: capacity, admit: false},
		{rows: 2, base: capacity - 1, admit: false},
		{rows: 2, base: capacity - 2, admit: true},
	} {
		g, err := BeginProjectionGraph(q4kTestVector(tc.rows*input, int64(6613+tc.rows)), nil, nil, tc.rows, input)
		if err != nil {
			t.Fatal(err)
		}
		q, err := g.EncodeQ4K(qW)
		if err != nil {
			t.Fatal(err)
		}
		k, err := g.EncodeQ4K(kW)
		if err != nil {
			t.Fatal(err)
		}
		v, err := g.EncodeQ4K(kW)
		if err != nil {
			t.Fatal(err)
		}
		cosv, sinv := make([]float32, tc.rows*hd/2), make([]float32, tc.rows*hd/2)
		for i := range cosv {
			cosv[i] = 1
		}
		_, err = g.FullAttentionDevice(q, k, v, nil, kv, 0, ones, ones, cosv, sinv, tc.base, nH, nKV, hd, hd, 0.125, 1e-6, false, false)
		if tc.admit && err != nil {
			t.Fatalf("rows=%d base=%d: refused an in-slice panel: %v", tc.rows, tc.base, err)
		}
		if !tc.admit && err == nil {
			t.Fatalf("rows=%d base=%d: admitted a panel past the %d-row layer slice", tc.rows, tc.base, capacity)
		}
		g.Free()
	}
	if NewDeviceKV(1, math.MaxInt32/64+1, 64) != nil {
		t.Fatal("NewDeviceKV admitted a side wider than the native int element count")
	}
}

// TestDeviceKVResidentAccountingAndCleanup pins the device KV lifecycle the dense decode
// graph budgets against: every live pair counts toward DeviceKVResidentBytes (two sides
// for an attend-only pair, three for a full one), an attend-only pair refuses KRaw I/O,
// Close returns the bytes and is idempotent, and a pair dropped WITHOUT Close is freed
// by its runtime cleanup instead of pinning device memory for the process lifetime.
//
// fak-test:runtime integration est=0.5s lane=default
func TestDeviceKVResidentAccountingAndCleanup(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	const layers, tokens, width = 2, 8, 64
	side := int64(layers * tokens * width * 4)
	base := DeviceKVResidentBytes()
	full := NewDeviceKV(layers, tokens, width)
	attend := NewDeviceKVAttendOnly(layers, tokens, width)
	if full == nil || attend == nil {
		t.Fatal("NewDeviceKV")
	}
	if full.ResidentBytes() != 3*side || attend.ResidentBytes() != 2*side {
		t.Fatalf("resident bytes full=%d attend-only=%d, want %d and %d", full.ResidentBytes(), attend.ResidentBytes(), 3*side, 2*side)
	}
	if got := DeviceKVResidentBytes() - base; got != 5*side {
		t.Fatalf("process resident delta %d, want %d", got, 5*side)
	}
	if err := attend.UploadRegion(0, 0, make([]float32, width)); err == nil {
		t.Fatal("attend-only pair accepted a KRaw upload")
	}
	if err := attend.UploadRegion(1, 0, make([]float32, width)); err != nil {
		t.Fatalf("attend-only KPost upload: %v", err)
	}
	full.Close()
	full.Close()
	attend.Close()
	if got := DeviceKVResidentBytes(); got != base {
		t.Fatalf("resident after Close %d, want %d", got, base)
	}
	func() {
		if NewDeviceKVAttendOnly(layers, tokens, width) == nil {
			t.Fatal("NewDeviceKVAttendOnly")
		}
	}()
	deadline := time.Now().Add(5 * time.Second)
	for DeviceKVResidentBytes() != base {
		if time.Now().After(deadline) {
			t.Fatalf("dropped pair not reclaimed: resident %d, want %d", DeviceKVResidentBytes(), base)
		}
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
}
