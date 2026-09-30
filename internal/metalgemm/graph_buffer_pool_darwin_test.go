//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

// TestProjectionGraphBufferPoolAllocationAndAliasSafety exercises reuse on physical
// Metal, including two independently held result handles for the same buffer. Both
// terminal branches must remain exact after duplicate releases of consumed inputs.
// fak-test:runtime integration est=2s lane=default
func TestProjectionGraphBufferPoolAllocationAndAliasSafety(t *testing.T) {
	if !Available() || DeviceName() == "" {
		t.Skip("buffer allocation witness requires a physical Metal device")
	}
	defer ResetQ4K()
	const width = 256
	x := q4kTestVector(width, 1359501)
	for i := range x {
		x[i] *= 0.02
	}
	// Bound the block scales so fourteen unnormalized projections cannot overflow.
	// The random scale/nibble payload stays nonzero and distinct between branches.
	boundedRaw := func(seed uint64) []byte {
		raw := q4kTestRaw(width, width, seed)
		for base := 0; base < len(raw); base += q4kTestBlockBytes {
			raw[base], raw[base+1] = 0, 0x0c
			raw[base+2], raw[base+3] = 0, 0x08
		}
		return raw
	}
	first := UploadQ4K(boundedRaw(1359502), width, width)
	second := UploadQ4K(boundedRaw(1359503), width, width)
	if first == nil || second == nil {
		t.Fatal("Q4_K fixture upload failed")
	}
	owners, buffers := graphLiveOwnerCount(), graphLiveBufferCount()
	run := func(pooled bool) ([][]float32, GraphReceipt) {
		t.Helper()
		g, err := BeginProjectionGraph(x, nil, nil, 1, width)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Free()
		if pooled && !g.SetBufferPool(8) {
			t.Fatal("physical graph declined its buffer pool")
		}
		cur, err := g.Input(width)
		if err != nil {
			t.Fatal(err)
		}
		for stage := 0; stage < 12; stage++ {
			next, err := g.EncodeQ4KFrom(first, cur)
			if err != nil {
				t.Fatalf("stage %d: %v", stage, err)
			}
			if stage > 0 {
				alias := *cur
				g.Release(cur)
				g.Release(cur)
				g.Release(&alias)
			}
			cur = next
		}
		left, err := g.EncodeQ4KFrom(first, cur)
		if err != nil {
			t.Fatal(err)
		}
		right, err := g.EncodeQ4KFrom(second, cur)
		if err != nil {
			t.Fatal(err)
		}
		out, receipt, err := g.FinishRead(left, right)
		if err != nil {
			t.Fatal(err)
		}
		if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 14 ||
			receipt.IntermediateWaits != 0 || receipt.IntermediateReadbacks != 0 {
			t.Fatalf("pooled=%v incomplete graph execution: %+v", pooled, receipt)
		}
		if receipt.AllocatedBuffers <= 0 || receipt.RetainedBufferBytes == 0 {
			t.Fatalf("pooled=%v lacks native tracked-buffer evidence: %+v", pooled, receipt)
		}
		return out, receipt
	}
	plain, control := run(false)
	pooled, candidate := run(true)
	if len(plain) != 2 || len(pooled) != len(plain) {
		t.Fatalf("terminal branch count: plain=%d pooled=%d", len(plain), len(pooled))
	}
	distinctBranches := false
	for branch := range plain {
		if len(plain[branch]) != width || len(pooled[branch]) != width {
			t.Fatalf("branch %d shape: plain=%d pooled=%d", branch, len(plain[branch]), len(pooled[branch]))
		}
		nonzero := false
		for i, want := range plain[branch] {
			got := pooled[branch][i]
			if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) ||
				math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
				t.Fatalf("branch %d element %d non-finite: plain=%g pooled=%g", branch, i, want, got)
			}
			if math.Float32bits(want) != math.Float32bits(got) {
				t.Fatalf("released input alias changed branch %d element %d: plain=%g pooled=%g", branch, i, want, got)
			}
			nonzero = nonzero || want != 0
			if branch == 1 && math.Float32bits(want) != math.Float32bits(plain[0][i]) {
				distinctBranches = true
			}
		}
		if !nonzero {
			t.Fatalf("branch %d underflowed to all zero, making alias parity vacuous", branch)
		}
	}
	if !distinctBranches {
		t.Fatal("terminal branches are identical, making result-alias safety vacuous")
	}
	if candidate.AllocatedBuffers >= control.AllocatedBuffers || candidate.RetainedBufferBytes >= control.RetainedBufferBytes {
		t.Fatalf("pool did not reduce native tracked allocation: control=%+v candidate=%+v", control, candidate)
	}
	if gotOwners, gotBuffers := graphLiveOwnerCount(), graphLiveBufferCount(); gotOwners != owners || gotBuffers != buffers {
		t.Fatalf("graph teardown leaked: owners %d->%d buffers %d->%d", owners, gotOwners, buffers, gotBuffers)
	}
	t.Logf("physical tracked buffers %d->%d retained bytes %d->%d; 512 finite terminal elements bit-exact; cleanup owners=%d buffers=%d",
		control.AllocatedBuffers, candidate.AllocatedBuffers, control.RetainedBufferBytes, candidate.RetainedBufferBytes, owners, buffers)
}
