package compute

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// TestMoeRouterGemmFp32MatchesOracle pins the batched multi-row router gate GEMM
// (MoeRouterGemmFp32, the CPU reference of moe_router_gemm.cu) against the
// existing one-row gate path (MatMul, the canonical decode gate projection):
//
//	(a) rows=1 equals the existing one-row gate path, bit for bit;
//	(b) rows>1 accumulation order matches the Go reference (per-row MatMul) for
//	    every row and expert;
//	(c) hidden/experts bound overflow is refused with the closed sentinel
//	    ErrMoeRouterGemmBounds (asserted via errors.Is, never error prose);
//	(d) the documented device-vs-oracle ULP parity gate
//	    (MoeRouterGemmWithinParity / MoeRouterGemmParityMaxULPs) admits
//	    bit-exact and boundary-inclusive drift, refuses one ULP past the bound,
//	    opposite signs, NaN/finite mixtures, and length mismatch.
//
// The fixture is deterministic (seeded) and mixed-sign so the FDot eight-
// accumulator tree is exercised on both polarities. No GPU or throughput claim:
// this is a software bit-identity witness on the CPU reference plus the closed
// contract of the numeric gate the retained hardware panel applies.
// fak-test:runtime fast est=1s
func TestMoeRouterGemmFp32MatchesOracle(t *testing.T) {
	be := &cpuBackend{}
	const hidden, experts = 256, 8

	// (a) rows=1 equals the existing one-row gate path.
	{
		w := make([]float32, experts*hidden)
		x := make([]float32, hidden)
		r := rand.New(rand.NewSource(1))
		for i := range w {
			w[i] = r.Float32()*2 - 1
		}
		for i := range x {
			x[i] = r.Float32()*2 - 1
		}
		got, err := MoeRouterGemmFp32(w, x, 1, hidden, experts)
		if err != nil {
			t.Fatalf("MoeRouterGemmFp32 rows=1: %v", err)
		}
		want := be.Read(be.MatMul(NewF32(be, []int{experts, hidden}, w), NewF32(be, []int{hidden}, x)))
		if len(got) != len(want) {
			t.Fatalf("rows=1 length = %d, want %d", len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("rows=1 expert=%d drift: got %v want %v", i, got[i], want[i])
			}
		}
	}

	// (b) rows>1 accumulation order vs the Go reference (per-row MatMul).
	{
		const rows = 4
		w := make([]float32, experts*hidden)
		x := make([]float32, rows*hidden)
		r := rand.New(rand.NewSource(2))
		for i := range w {
			w[i] = r.Float32()*2 - 1
		}
		for i := range x {
			x[i] = r.Float32()*2 - 1
		}
		got, err := MoeRouterGemmFp32(w, x, rows, hidden, experts)
		if err != nil {
			t.Fatalf("MoeRouterGemmFp32 rows=%d: %v", rows, err)
		}
		if len(got) != rows*experts {
			t.Fatalf("rows=%d length = %d, want %d", rows, len(got), rows*experts)
		}
		for row := 0; row < rows; row++ {
			want := be.Read(be.MatMul(NewF32(be, []int{experts, hidden}, w), NewF32(be, []int{hidden}, x[row*hidden:(row+1)*hidden])))
			for o := 0; o < experts; o++ {
				if math.Float32bits(got[row*experts+o]) != math.Float32bits(want[o]) {
					t.Fatalf("rows=%d row=%d expert=%d drift: got %v want %v", rows, row, o, got[row*experts+o], want[o])
				}
			}
		}
	}

	// (c) hidden/experts bound overflow refusal (closed sentinel, errors.Is).
	{
		// experts past the 512-class ceiling.
		wBig := make([]float32, (MoeRouterGemmMaxExperts+1)*hidden)
		xOne := make([]float32, hidden)
		if _, err := MoeRouterGemmFp32(wBig, xOne, 1, hidden, MoeRouterGemmMaxExperts+1); !errors.Is(err, ErrMoeRouterGemmBounds) {
			t.Fatalf("experts=%d: err=%v, want ErrMoeRouterGemmBounds", MoeRouterGemmMaxExperts+1, err)
		}
		// hidden past the ceiling.
		wHidden := make([]float32, experts*(MoeRouterGemmMaxHidden+1))
		if _, err := MoeRouterGemmFp32(wHidden, xOne, 1, MoeRouterGemmMaxHidden+1, experts); !errors.Is(err, ErrMoeRouterGemmBounds) {
			t.Fatalf("hidden=%d: err=%v, want ErrMoeRouterGemmBounds", MoeRouterGemmMaxHidden+1, err)
		}
		// rows past the M<=48 envelope.
		wOk := make([]float32, experts*hidden)
		xMany := make([]float32, (MoeRouterGemmMaxRows+1)*hidden)
		if _, err := MoeRouterGemmFp32(wOk, xMany, MoeRouterGemmMaxRows+1, hidden, experts); !errors.Is(err, ErrMoeRouterGemmBounds) {
			t.Fatalf("rows=%d: err=%v, want ErrMoeRouterGemmBounds", MoeRouterGemmMaxRows+1, err)
		}
		// the admitted boundary is accepted (512 experts, max hidden, max rows).
		wEdge := make([]float32, MoeRouterGemmMaxExperts*MoeRouterGemmMaxHidden)
		xEdge := make([]float32, MoeRouterGemmMaxRows*MoeRouterGemmMaxHidden)
		if _, err := MoeRouterGemmFp32(wEdge, xEdge, MoeRouterGemmMaxRows, MoeRouterGemmMaxHidden, MoeRouterGemmMaxExperts); err != nil {
			t.Fatalf("admitted boundary (rows=%d hidden=%d experts=%d): %v", MoeRouterGemmMaxRows, MoeRouterGemmMaxHidden, MoeRouterGemmMaxExperts, err)
		}
	}

	// (d) Documented device-vs-oracle ULP parity gate: closed contract.
	{
		base := float32(1.0) // positive normal; small bit shifts stay in-binade
		shift := func(f float32, n int) float32 {
			return math.Float32frombits(math.Float32bits(f) + uint32(n))
		}
		nan := float32(math.NaN())
		posInf := float32(math.Inf(1))

		cases := []struct {
			name      string
			got, want []float32
			ok        bool
			firstBad  int
		}{
			{"bit-exact", []float32{base, -base}, []float32{base, -base}, true, -1},
			{"boundary-inclusive +64 ULP", []float32{shift(base, 64)}, []float32{base}, true, -1},
			{"boundary-inclusive -64 ULP", []float32{shift(base, -64)}, []float32{base}, true, -1},
			{"one past bound +65 ULP", []float32{shift(base, 65)}, []float32{base}, false, 0},
			{"first bad index reported", []float32{base, shift(base, 65)}, []float32{base, base}, false, 1},
			{"opposite signs", []float32{-base}, []float32{base}, false, 0},
			{"both NaN", []float32{nan}, []float32{nan}, true, -1},
			{"NaN vs finite", []float32{nan}, []float32{base}, false, 0},
			{"finite vs NaN", []float32{base}, []float32{nan}, false, 0},
			{"matching infinities", []float32{posInf}, []float32{posInf}, true, -1},
			{"infinite vs finite", []float32{posInf}, []float32{base}, false, 0},
			{"signed zeros equal", []float32{0.0}, []float32{-0.0}, true, -1},
			{"smallest subnormal one ULP from zero", []float32{math.Float32frombits(1)}, []float32{-0.0}, true, -1},
			{"length mismatch", []float32{base}, []float32{}, false, -1},
		}
		for _, tc := range cases {
			ok, firstBad := MoeRouterGemmWithinParity(tc.got, tc.want)
			if ok != tc.ok || firstBad != tc.firstBad {
				t.Fatalf("%s: got (%v,%d), want (%v,%d)", tc.name, ok, firstBad, tc.ok, tc.firstBad)
			}
		}
	}
}
