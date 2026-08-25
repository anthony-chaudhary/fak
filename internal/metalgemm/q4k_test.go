//go:build darwin && arm64 && cgo

package metalgemm

import "testing"

func TestMixedQ4KQ8GEMVUsesOneCommandBuffer(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device available")
	}
	defer ResetQ4K()
	defer ResetQ8()

	const in = 256
	q4 := UploadQ4K(make([]byte, 64*(in/256)*144), 64, in)
	if q4 == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	q8ws := make([]*Q8Weight, 2)
	for i, out := range []int{64, 32} {
		codes := make([]int8, out*in)
		scales := make([]float32, out*(in/32))
		for j := range codes {
			codes[j] = int8(j%7) - 3
		}
		for j := range scales {
			scales[j] = 0.01
		}
		q8ws[i] = UploadQ8(codes, scales, out, in)
		if q8ws[i] == nil {
			t.Fatalf("UploadQ8[%d] returned nil", i)
		}
	}
	x := make([]float32, in)
	xq := make([]int8, in)
	xd := make([]float32, in/32)
	for i := range x {
		x[i] = float32(i%11-5) * 0.02
		xq[i] = int8(i%9) - 4
	}
	for i := range xd {
		xd[i] = 0.02
	}

	beforeControl := QuantNativeEvents()
	q4Control := make([]float32, q4.Out)
	q4.GEMV(x, q4Control)
	q8Control := GEMVGroupQ8(q8ws, xq, xd)
	control := QuantNativeEvents()

	beforeCandidate := control
	q4Candidate, q8Candidate, status := GEMVGroupMixedQ4KQ8([]*Q4KWeight{q4}, q8ws, x, xq, xd)
	if status != 1 {
		t.Fatalf("mixed status=%d", status)
	}
	candidate := QuantNativeEvents()

	if control.CommandBuffers-beforeControl.CommandBuffers != 2 ||
		candidate.CommandBuffers-beforeCandidate.CommandBuffers != 1 {
		t.Fatalf("command buffers control=%d candidate=%d",
			control.CommandBuffers-beforeControl.CommandBuffers,
			candidate.CommandBuffers-beforeCandidate.CommandBuffers)
	}
	for i := range q4Control {
		if q4Candidate[0][i] != q4Control[i] {
			t.Fatalf("q4[%d]=%g want %g", i, q4Candidate[0][i], q4Control[i])
		}
	}
	for w := range q8Control {
		for i := range q8Control[w] {
			if q8Candidate[w][i] != q8Control[w][i] {
				t.Fatalf("q8[%d][%d]=%g want %g", w, i, q8Candidate[w][i], q8Control[w][i])
			}
		}
	}
}
