//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

func TestProjectionGraphMixedQuantizedSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer ResetQ8()
	const P, in, out = 2, 256, 32
	xf := q4kTestVector(P*in, 9267)
	q4 := UploadQ4K(q4kTestRaw(out, in, 9267), out, in)
	if q4 == nil {
		t.Fatal("q4 upload")
	}
	// Q6 remains a valid zero matrix in this mixed packet; Q4/Q8 carry non-zero parity.
	q6 := UploadQ6K(make([]byte, out*(in/256)*210), out, in)
	if q6 == nil {
		t.Fatal("q6 upload")
	}
	q8codes := make([]int8, out*in)
	q8scales := make([]float32, out*(in/32))
	for i := range q8codes {
		q8codes[i] = int8(i%15 - 7)
	}
	for i := range q8scales {
		q8scales[i] = 0.02
	}
	q8 := UploadQ8(q8codes, q8scales, out, in)
	if q8 == nil {
		t.Fatal("q8 upload")
	}
	xq := make([]int8, P*in)
	xd := make([]float32, P*(in/32))
	for row := 0; row < P; row++ {
		for b := 0; b < in/32; b++ {
			xd[row*(in/32)+b] = 0.01
			for j := 0; j < 32; j++ {
				xq[row*in+b*32+j] = int8((row+b+j)%17 - 8)
			}
		}
	}

	want4 := make([]float32, P*out)
	q4.GEMM(xf, P, want4)
	want6 := make([]float32, P*out)
	q6.GEMM(xf, P, want6)
	want8 := make([]float32, P*out)
	q8.GEMM(xq, xd, P, want8)
	g, err := BeginProjectionGraph(xf, xq, xd, P, in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	r4, err := g.EncodeQ4K(q4)
	if err != nil {
		t.Fatal(err)
	}
	r6, err := g.EncodeQ6K(q6)
	if err != nil {
		t.Fatal(err)
	}
	r8, err := g.EncodeQ8(q8)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := g.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 3 || receipt.HostReadbacks != 0 {
		t.Fatalf("receipt=%+v", receipt)
	}
	for i, pair := range []struct {
		r    *GraphResult
		want []float32
	}{{r4, want4}, {r6, want6}, {r8, want8}} {
		got, err := g.Read(pair.r)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(pair.want) {
			t.Fatalf("result %d len=%d", i, len(got))
		}
		for j := range got {
			d := math.Abs(float64(got[j] - pair.want[j]))
			if d > 1e-5 {
				t.Fatalf("result %d[%d] got=%g want=%g delta=%g", i, j, got[j], pair.want[j], d)
			}
		}
	}
}
