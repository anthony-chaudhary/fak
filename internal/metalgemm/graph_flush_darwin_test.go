//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

// TestProjectionGraphFlushSegmentsAndConstantCache holds a graph committed in
// several segments to the same graph committed once: pooled intermediates recycled
// across a segment boundary must stay exact, the receipt counts every committed
// command buffer, an abandoned flushed graph is released through the quarantine
// waiter, and a cached constant span that the host rewrites in place is re-read.
// fak-test:runtime integration est=2s lane=default
func TestProjectionGraphFlushSegmentsAndConstantCache(t *testing.T) {
	if !Available() || DeviceName() == "" {
		t.Skip("graph flush witness requires a physical Metal device")
	}
	defer ResetQ4K()
	const width = 256
	x := q4kTestVector(width, 7130001)
	for i := range x {
		x[i] *= 0.02
	}
	raw := q4kTestRaw(width, width, 7130002)
	for base := 0; base < len(raw); base += q4kTestBlockBytes {
		raw[base], raw[base+1] = 0, 0x0c
		raw[base+2], raw[base+3] = 0, 0x08
	}
	w := UploadQ4K(raw, width, width)
	if w == nil {
		t.Fatal("Q4_K fixture upload failed")
	}
	norm := make([]float32, width)
	for i := range norm {
		norm[i] = 0.5 + float32(i%7)*0.125
	}
	owners := graphLiveOwnerCount()
	run := func(flushAt map[int]bool, normWeight []float32) ([]float32, GraphReceipt) {
		t.Helper()
		g, err := BeginProjectionGraph(x, nil, nil, 1, width)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Free()
		g.SetBufferPool(8)
		cur, err := g.Input(width)
		if err != nil {
			t.Fatal(err)
		}
		for stage := 0; stage < 10; stage++ {
			if flushAt[stage] {
				if err := g.Flush(); err != nil {
					t.Fatalf("flush before stage %d: %v", stage, err)
				}
			}
			normed, err := g.RMSNorm(cur, normWeight, 1e-5, false)
			if err != nil {
				t.Fatal(err)
			}
			next, err := g.EncodeQ4KFrom(w, normed)
			if err != nil {
				t.Fatal(err)
			}
			g.Release(normed)
			if stage > 0 {
				g.Release(cur)
			}
			cur = next
		}
		out, receipt, err := g.FinishRead(cur)
		if err != nil {
			t.Fatal(err)
		}
		if !receipt.Committed || !receipt.CompletedWait || receipt.IntermediateWaits != 0 || receipt.HostReadbacks != 1 {
			t.Fatalf("receipt %+v", receipt)
		}
		return out[0], receipt
	}
	want, single := run(nil, norm)
	got, segmented := run(map[int]bool{2: true, 6: true}, norm)
	if single.CommandBuffers != 1 || segmented.CommandBuffers != 3 {
		t.Fatalf("command buffers single=%d segmented=%d, want 1 and 3", single.CommandBuffers, segmented.CommandBuffers)
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) || math.IsNaN(float64(want[i])) {
			t.Fatalf("segmented output[%d]=%g, single %g", i, got[i], want[i])
		}
	}

	// The constant cache keys on the host span; rewriting it in place must be seen.
	rewritten := append([]float32(nil), norm...)
	for i := range norm {
		norm[i] *= 1.5
		rewritten[i] *= 1.5
	}
	inPlace, _ := run(nil, norm)
	fresh, _ := run(nil, rewritten)
	same := true
	for i := range fresh {
		if math.Float32bits(inPlace[i]) != math.Float32bits(fresh[i]) {
			t.Fatalf("in-place rewritten constant output[%d]=%g, fresh copy %g", i, inPlace[i], fresh[i])
		}
		same = same && inPlace[i] == want[i]
	}
	if same {
		t.Fatal("rewritten constant span left the output unchanged")
	}

	// An abandoned graph whose first segment is committed is freed by the quarantine waiter.
	g, err := BeginProjectionGraph(x, nil, nil, 1, width)
	if err != nil {
		t.Fatal(err)
	}
	in, err := g.Input(width)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.EncodeQ4KFrom(w, in); err != nil {
		t.Fatal(err)
	}
	if err := g.Flush(); err != nil {
		t.Fatal(err)
	}
	g.Free()
	if g.quarantineDone == nil {
		t.Fatal("abandoned flushed graph was not quarantined")
	}
	<-g.quarantineDone
	if got := graphLiveOwnerCount(); got != owners {
		t.Fatalf("live graph owners=%d after abandon, want %d", got, owners)
	}
}
