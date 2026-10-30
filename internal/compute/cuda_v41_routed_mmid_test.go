//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"testing"
)

// cuda_v41_routed_mmid_test.go - device parity of the CUDA V4.1 routed-expert id
// and fused paths (cuda_v41_routed_mmid.cu) against a float64 host model of the
// same arithmetic: the q8_1 activation replay (quantize_q8_1, f16 scale), the
// dequantized Q2_K/Q3_K rows, the V4.1 clamp-SwiGLU with BF16 publication, and
// the checked BF16 down publication summed in routed order. The DS4.1 expert
// geometry is gate/up Q2_K [2304,5120] and down Q3_K [5120,2304]
// (fak-private ds41-5090 tensor-ledger, 384 experts, 6 used).

const (
	v41MMIDHidden = 5120
	v41MMIDInter  = 2304
	v41MMIDLimit  = float32(10)
)

func v41MMIDQ2K(out, in int, seed uint64) []byte {
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	state := seed | 1
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state)
	}
	dBits := [...]uint16{0x2c00, 0x3000, 0x2e00, 0x3200} // 0.0625 .. 0.1875
	mBits := [...]uint16{0x2400, 0x2800, 0x2600}         // 0.015625 .. 0.03125
	for b, base := 0, 0; base < len(raw); b, base = b+1, base+q2kSuperBlock {
		binary.LittleEndian.PutUint16(raw[base+80:], dBits[b%len(dBits)])
		binary.LittleEndian.PutUint16(raw[base+82:], mBits[b%len(mBits)])
	}
	return raw
}

func v41MMIDBF16(v float32) float32 {
	u := math.Float32bits(v)
	return math.Float32frombits((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000)
}

func v41MMIDNonfinite(v float32) bool { return math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) }

// v41MMIDQ8 returns x as the GEMV reads it: quantize_q8_1 codes times the f16 scale.
func v41MMIDQ8(x []float32) []float64 {
	qs, ds := q3kQ8_1(x, roundF16RNE)
	out := make([]float64, len(x))
	for i, q := range qs {
		out[i] = float64(q) * float64(ds[i/32])
	}
	return out
}

// v41MMIDRows dequantizes every [out,in] row of a Q2_K/Q3_K matrix.
func v41MMIDRows(dt Dtype, raw []byte, out, in int) [][]float32 {
	blockBytes, dq := q2kSuperBlock, q2kDequantSuperBlock
	if dt == Q3_K {
		blockBytes, dq = q3kSuperBlock, q3kDequantSuperBlock
	}
	rowBytes := (in / 256) * blockBytes
	rows := make([][]float32, out)
	for o := range rows {
		rows[o] = make([]float32, in)
		for b := 0; b < in/256; b++ {
			dq(rows[o][b*256:], raw[o*rowBytes+b*blockBytes:])
		}
	}
	return rows
}

func v41MMIDDot(row []float32, x []float64) float64 {
	var s float64
	for i, w := range row {
		s += float64(w) * x[i]
	}
	return s
}

func v41MMIDSwiGLU(gate, up, limit, weight float32) float32 {
	g, u := v41MMIDBF16(gate), v41MMIDBF16(up)
	if limit > 0 {
		g = float32(math.Min(float64(g), float64(limit)))
		u = float32(math.Max(float64(-limit), math.Min(float64(u), float64(limit))))
	}
	t := 1 + float32(math.Exp(float64(-g)))
	return v41MMIDBF16((g / t) * u * weight)
}

type v41MMIDLayer struct {
	gateRaw, upRaw, downRaw [][]byte
	gate, up, down          []Tensor // slot table order
	slots, order            []int
	weights                 []float32
	x                       []float32 // hidden or picks*hidden
}

func newV41MMIDLayer(t *testing.T, cb *cudaBackend, picks, hidden, inter int, perPickX bool, seed uint64) *v41MMIDLayer {
	t.Helper()
	l := &v41MMIDLayer{}
	for s := 0; s < picks; s++ {
		k := seed*131 + uint64(s)*17
		l.gateRaw = append(l.gateRaw, v41MMIDQ2K(inter, hidden, k+1))
		l.upRaw = append(l.upRaw, v41MMIDQ2K(inter, hidden, k+2))
		l.downRaw = append(l.downRaw, q3kTestRaw(hidden, inter, k+3))
		l.gate = append(l.gate, cb.Upload(NewQ2K(cb, []int{inter, hidden}, l.gateRaw[s]), Q2_K))
		l.up = append(l.up, cb.Upload(NewQ2K(cb, []int{inter, hidden}, l.upRaw[s]), Q2_K))
		l.down = append(l.down, cb.Upload(NewQ3K(cb, []int{hidden, inter}, l.downRaw[s]), Q3_K))
	}
	// Pick p binds slot (p*5+3)%picks, a permutation; the sum order reverses the picks.
	for p := 0; p < picks; p++ {
		l.slots = append(l.slots, (p*5+3)%picks)
		l.order = append(l.order, picks-1-p)
		l.weights = append(l.weights, 0.05+0.1*float32(p))
	}
	if picks%5 == 0 {
		for p := range l.slots {
			l.slots[p] = p
		}
	}
	n := hidden
	if perPickX {
		n = picks * hidden
	}
	l.x = q3kTestX(1, n, seed+7)
	for i := range l.x {
		l.x[i] *= 0.1
	}
	return l
}

func (l *v41MMIDLayer) free(cb *cudaBackend) {
	for _, ws := range [][]Tensor{l.gate, l.up, l.down} {
		for _, w := range ws {
			cb.Free(w)
		}
	}
}

// host returns the per-pick gate/up rows, the mid and the routed sum.
func (l *v41MMIDLayer) host(picks, hidden, inter int) (gate, up [][]float64, mid []float32, sum []float32) {
	mid = make([]float32, picks*inter)
	for p := 0; p < picks; p++ {
		xs := l.x
		if len(l.x) != hidden {
			xs = l.x[p*hidden : (p+1)*hidden]
		}
		xq := v41MMIDQ8(xs)
		s := l.slots[p]
		gRows, uRows := v41MMIDRows(Q2_K, l.gateRaw[s], inter, hidden), v41MMIDRows(Q2_K, l.upRaw[s], inter, hidden)
		g, u := make([]float64, inter), make([]float64, inter)
		for o := 0; o < inter; o++ {
			g[o], u[o] = v41MMIDDot(gRows[o], xq), v41MMIDDot(uRows[o], xq)
			mid[p*inter+o] = v41MMIDSwiGLU(float32(g[o]), float32(u[o]), v41MMIDLimit, l.weights[p])
		}
		gate, up = append(gate, g), append(up, u)
	}
	sum = make([]float32, hidden)
	down := make([][]float32, picks)
	for p := 0; p < picks; p++ {
		mq := v41MMIDQ8(mid[p*inter : (p+1)*inter])
		dRows := v41MMIDRows(Q3_K, l.downRaw[l.slots[p]], hidden, inter)
		down[p] = make([]float32, hidden)
		for o := range down[p] {
			down[p][o] = v41MMIDBF16(float32(v41MMIDDot(dRows[o], mq)))
		}
	}
	for _, p := range l.order {
		for o := range sum {
			sum[o] += down[p][o]
		}
	}
	return gate, up, mid, sum
}

func v41MMIDMaxAbs(v []float32) float64 {
	m := 0.0
	for _, x := range v {
		m = math.Max(m, math.Abs(float64(x)))
	}
	return m
}

// v41MMIDCheckMid allows one BF16 ULP where the device and host gate/up sums
// round to neighbouring BF16 values, on at most 1% of lanes.
func v41MMIDCheckMid(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", what, len(got), len(want))
	}
	off := 0
	for i := range want {
		if got[i] == want[i] {
			continue
		}
		d := math.Abs(float64(got[i] - want[i]))
		if v41MMIDNonfinite(got[i]) || d > math.Ldexp(math.Max(math.Abs(float64(want[i])), 1e-30), -7)*1.01+1e-6 {
			t.Fatalf("%s: lane %d = %g, want %g", what, i, got[i], want[i])
		}
		off++
	}
	if off*100 > len(want) {
		t.Fatalf("%s: %d of %d lanes differ by one BF16 ULP", what, off, len(want))
	}
}

func v41MMIDCheckClose(t *testing.T, what string, got, want []float32, rel float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", what, len(got), len(want))
	}
	scale := v41MMIDMaxAbs(want)
	var dot, gg, ww float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); v41MMIDNonfinite(got[i]) || d > rel*math.Max(scale, 1e-6) {
			t.Fatalf("%s: lane %d = %g, want %g (max |want| %g)", what, i, got[i], want[i], scale)
		}
		dot += float64(got[i]) * float64(want[i])
		gg += float64(got[i]) * float64(got[i])
		ww += float64(want[i]) * float64(want[i])
	}
	if cos := dot / math.Sqrt(gg*ww); !(cos >= 0.9999) {
		t.Fatalf("%s: cosine %.7f", what, cos)
	}
}

func v41MMIDVerdicts(t *testing.T, cb *cudaBackend, st V41DeferredStatus) []error {
	t.Helper()
	v, err := cb.cudaV41MMIDVerdicts(st)
	if err != nil {
		t.Fatalf("read deferred status: %v", err)
	}
	if len(v) != st.Records {
		t.Fatalf("%d verdicts for %d records", len(v), st.Records)
	}
	return v
}

// TestCUDAV41RoutedMMIDParity runs one DS4.1-shaped routed layer through the id
// path (five stages) and the fused path (two stages) and checks every stage
// against the host model, and that fused == id bit for bit.
//
// fak-test:runtime slow est=20s lane=cuda
func TestCUDAV41RoutedMMIDParity(t *testing.T) {
	cb := q3kCUDAOrFail(t)
	if !cb.V41RoutedIDAvailable() || !cb.V41RoutedFusedAvailable() {
		t.Fatal("routed mmid candidate compiled but native admission refused")
	}
	cases := []struct {
		hidden, inter, picks int
		perPickX             bool
	}{
		{512, 256, 3, false},
		{v41MMIDHidden, v41MMIDInter, 1, false},
		{v41MMIDHidden, v41MMIDInter, 6, false},
		{v41MMIDHidden, v41MMIDInter, 6, true},
		{v41MMIDHidden, v41MMIDInter, 8, false},
	}
	for ci, tc := range cases {
		t.Run(fmt.Sprintf("H%d_I%d_P%d_x%v", tc.hidden, tc.inter, tc.picks, tc.perPickX), func(t *testing.T) {
			H, I, P := tc.hidden, tc.inter, tc.picks
			l := newV41MMIDLayer(t, cb, P, H, I, tc.perPickX, uint64(0x41d0+ci))
			defer l.free(cb)
			wantGate, wantUp, wantMid, wantSum := l.host(P, H, I)

			ids, err := cb.V41RoutedIDs(l.slots, l.weights, l.order)
			if err != nil {
				t.Fatal(err)
			}
			defer cb.Free(ids)
			x := mkResident(cb, []int{len(l.x)}, l.x)
			defer cb.Free(x)

			// Id path.
			launches := cudaV41MMIDLaunchesForTest()
			g, err := cb.V41MatVecID(l.gate, ids, P, x)
			if err != nil {
				t.Fatal(err)
			}
			u, err := cb.V41MatVecID(l.up, ids, P, x)
			if err != nil {
				t.Fatal(err)
			}
			a, ast, err := cb.V41RoutedSwiGLUIDDeferred(g, u, ids, P, v41MMIDLimit)
			if err != nil {
				t.Fatal(err)
			}
			d, err := cb.V41MatVecID(l.down, ids, P, a)
			if err != nil {
				t.Fatal(err)
			}
			s, sst, err := cb.V41RoutedPublishSumDeferred(d, ids, P, V41BF16PublishRoutedDown)
			if err != nil {
				t.Fatal(err)
			}
			if n := cudaV41MMIDLaunchesForTest() - launches; n != 8 {
				t.Fatalf("id path enqueued %d kernels, want 8 (3 quantize, 3 GEMV, SwiGLU, publish-sum)", n)
			}
			for i, v := range append(v41MMIDVerdicts(t, cb, ast), v41MMIDVerdicts(t, cb, sst)...) {
				if v != nil {
					t.Fatalf("id path record %d: %v", i, v)
				}
			}
			// The q8_1 integer dot replays exactly; only F32 accumulation order
			// separates device and host, bounded by 3e-5 of the largest row.
			for _, c := range []struct {
				name string
				got  []float32
				want [][]float64
			}{{"gate", cb.Read(g), wantGate}, {"up", cb.Read(u), wantUp}} {
				scale := 1.0
				for _, rows := range c.want {
					for _, v := range rows {
						scale = math.Max(scale, math.Abs(v))
					}
				}
				for p := 0; p < P; p++ {
					for o := 0; o < I; o++ {
						if got := float64(c.got[p*I+o]); math.Abs(got-c.want[p][o]) > 3e-5*scale {
							t.Fatalf("%s pick %d row %d = %.9g, want %.9g (scale %g)", c.name, p, o, got, c.want[p][o], scale)
						}
					}
				}
			}
			idMid, idSum := cb.Read(a), cb.Read(s)
			v41MMIDCheckMid(t, "id mid", idMid, wantMid)
			v41MMIDCheckClose(t, "id sum", idSum, wantSum, 1e-2)

			// Fused path.
			launches = cudaV41MMIDLaunchesForTest()
			mid, mst, err := cb.V41RoutedGateUpSwiGLUDeferred(l.gate, l.up, ids, x, P, v41MMIDLimit)
			if err != nil {
				t.Fatal(err)
			}
			sum, dst, err := cb.V41RoutedDownSumDeferred(l.down, mid, ids, P, V41BF16PublishRoutedDown)
			if err != nil {
				t.Fatal(err)
			}
			if n := cudaV41MMIDLaunchesForTest() - launches; n != 4 {
				t.Fatalf("fused path enqueued %d kernels, want 4 (2 quantize, gate/up, down-sum)", n)
			}
			for i, v := range append(v41MMIDVerdicts(t, cb, mst), v41MMIDVerdicts(t, cb, dst)...) {
				if v != nil {
					t.Fatalf("fused record %d: %v", i, v)
				}
			}
			fusedMid, fusedSum := cb.Read(mid), cb.Read(sum)
			for i := range idMid {
				if math.Float32bits(fusedMid[i]) != math.Float32bits(idMid[i]) {
					t.Fatalf("fused mid lane %d = %g, id path %g", i, fusedMid[i], idMid[i])
				}
			}
			for i := range idSum {
				if math.Float32bits(fusedSum[i]) != math.Float32bits(idSum[i]) {
					t.Fatalf("fused sum lane %d = %g, id path %g", i, fusedSum[i], idSum[i])
				}
			}
			for _, o := range []Tensor{g, u, a, d, s, mid, sum} {
				cb.Free(o)
			}
			t.Logf("H%d I%d P%d: |sum| max %.4g, mid/sum fused == id bitwise", H, I, P, v41MMIDMaxAbs(wantSum))
		})
	}
}

// TestCUDAV41RoutedMMIDFaults poisons one pick's gate (block 0 d = +Inf) and
// another pick's down weight, and checks each fault lands in that pick's own
// record at lane 0 while the other records stay clean.
//
// fak-test:runtime slow est=3s lane=cuda
func TestCUDAV41RoutedMMIDFaults(t *testing.T) {
	cb := q3kCUDAOrFail(t)
	const H, I, P = 512, 256, 4
	l := newV41MMIDLayer(t, cb, P, H, I, false, 0xfa17)
	defer l.free(cb)
	gatePick, downPick := 1, 2
	poisoned := func(raw []byte, off int) []byte {
		c := append([]byte(nil), raw...)
		binary.LittleEndian.PutUint16(c[off:], 0x7c00)
		return c
	}
	badGate := cb.Upload(NewQ2K(cb, []int{I, H}, poisoned(l.gateRaw[l.slots[gatePick]], 80)), Q2_K)
	defer cb.Free(badGate)
	badDown := cb.Upload(NewQ3K(cb, []int{H, I}, poisoned(l.downRaw[l.slots[downPick]], q3kSuperBlock-2)), Q3_K)
	defer cb.Free(badDown)
	gates, downs := append([]Tensor(nil), l.gate...), append([]Tensor(nil), l.down...)
	gates[l.slots[gatePick]], downs[l.slots[downPick]] = badGate, badDown

	ids, err := cb.V41RoutedIDs(l.slots, l.weights, l.order)
	if err != nil {
		t.Fatal(err)
	}
	defer cb.Free(ids)
	x := mkResident(cb, []int{H}, l.x)
	defer cb.Free(x)
	mid, mst, err := cb.V41RoutedGateUpSwiGLUDeferred(gates, l.up, ids, x, P, v41MMIDLimit)
	if err != nil {
		t.Fatal(err)
	}
	defer cb.Free(mid)
	sum, dst, err := cb.V41RoutedDownSumDeferred(downs, mid, ids, P, V41BF16PublishRoutedDown)
	if err != nil {
		t.Fatal(err)
	}
	defer cb.Free(sum)
	for p, v := range v41MMIDVerdicts(t, cb, mst) {
		var fault *V41RoutedSwiGLUFault
		if p != gatePick {
			if v != nil {
				t.Fatalf("gate/up record %d: %v, want clean", p, v)
			}
			continue
		}
		if !errors.As(v, &fault) || fault.Stage != V41RoutedSwiGLUStageActivation || fault.Index != 0 {
			t.Fatalf("gate/up record %d: %v, want activation fault at lane 0", p, v)
		}
	}
	for p, v := range v41MMIDVerdicts(t, cb, dst) {
		var arith *V41BF16CheckedArithmeticError
		switch p {
		case gatePick:
			// Its mid carries NaN lanes the caller must not publish; the down
			// verdict of that pick is not asserted (quantize_q8_1's fmaxf
			// drops NaN from the block scale).
		case downPick:
			if !errors.As(v, &arith) || arith.Site != V41BF16PublishRoutedDown || arith.Index != 0 {
				t.Fatalf("down record %d: %v, want a checked BF16 publication fault at lane 0", p, v)
			}
		default:
			if v != nil {
				t.Fatalf("down record %d: %v, want clean", p, v)
			}
		}
	}
}

// TestCUDAV41RoutedMMIDRefusals checks the dtype advertisement and that an
// unserved dtype or a foreign activation is a BackendError before any launch.
//
// fak-test:runtime slow est=1s lane=cuda
func TestCUDAV41RoutedMMIDRefusals(t *testing.T) {
	cb := q3kCUDAOrFail(t)
	for _, dt := range []Dtype{Q2_K, Q3_K} {
		if !cb.SupportsV41RoutedIDWeight(dt) || !cb.SupportsV41RoutedFusedDown(dt) || !cb.SupportsV41RoutedFusedGateUp(dt, dt) {
			t.Fatalf("%v not advertised", dt)
		}
	}
	for _, dt := range []Dtype{F32, Q4_K, IQ2_XXS} {
		if cb.SupportsV41RoutedIDWeight(dt) || cb.SupportsV41RoutedFusedDown(dt) || cb.SupportsV41RoutedFusedGateUp(dt, dt) {
			t.Fatalf("%v advertised but not served", dt)
		}
	}
	if cb.SupportsV41RoutedFusedGateUp(Q2_K, Q3_K) {
		t.Fatal("mixed gate/up dtypes advertised")
	}
	const H, I = 256, 256
	w := cb.Upload(NewQ2K(cb, []int{I, H}, v41MMIDQ2K(I, H, 3)), Q2_K)
	defer cb.Free(w)
	ids, err := cb.V41RoutedIDs([]int{0}, []float32{1}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	defer cb.Free(ids)
	launches := cudaV41MMIDLaunchesForTest()
	host := NewF32(Default(), []int{H}, make([]float32, H))
	var be *BackendError
	if _, err := cb.V41MatVecID([]Tensor{w}, ids, 1, host); !errors.As(err, &be) {
		t.Fatalf("host activation: %v, want *BackendError", err)
	}
	wf := mkResident(cb, []int{I, H}, make([]float32, I*H))
	defer cb.Free(wf)
	x := mkResident(cb, []int{H}, make([]float32, H))
	defer cb.Free(x)
	if _, err := cb.V41MatVecID([]Tensor{wf}, ids, 1, x); !errors.As(err, &be) {
		t.Fatalf("F32 weights: %v, want *BackendError", err)
	}
	if _, err := cb.V41RoutedIDs([]int{1}, []float32{1}, []int{0}); !errors.As(err, &be) {
		t.Fatalf("out-of-table slot: %v, want *BackendError", err)
	}
	if n := cudaV41MMIDLaunchesForTest() - launches; n != 0 {
		t.Fatalf("refusals enqueued %d kernels", n)
	}
}
