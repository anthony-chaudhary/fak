//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"testing"
)

// The model's resident indexer key cache scores a [rows,headDim] view of a
// capacity-doubling row log. Spare rows are NaN-poisoned, so a read past the
// declared rows would change score bits. Each append, including every doubling,
// must match an exact-sized upload and the serial oracle bit for bit, while an
// undersized key buffer and oversized query/weights still refuse.
// fak-test:runtime integration est=1s lane=optin
func TestVulkanV41IndexerScoreSpareKeyCapacity(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41IndexerScore() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required device lacks the V4.1 indexer score: %s", v.V41IndexerScoreUnavailableReason())
		}
		t.Skip("V4.1 indexer score pipeline unavailable")
	}
	if !v.V41IndexerScoreSpareKeyCapacity() {
		t.Fatal("Vulkan does not advertise spare indexer key capacity")
	}
	const heads, dim, total = 64, 128, 70
	values := func(n, salt int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = decodeSplitValue(i, salt) * scale
		}
		return out
	}
	q, keys, weights := values(heads*dim, 3, 1), values(total*dim, 11, 1), values(heads, 17, 0.125)
	upload := func(shape []int, x []float32) Tensor {
		return v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "indexer spare key test")
	}
	qt, wt := upload([]int{heads, dim}, q), upload([]int{heads}, weights)
	defer v.Free(qt)
	defer v.Free(wt)
	poisoned := func(capacity int) Tensor {
		log, err := v.V41RowLogAlloc(capacity, dim)
		if err != nil {
			t.Fatal(err)
		}
		nan := make([]float32, capacity*dim)
		for i := range nan {
			nan[i] = float32(math.NaN())
		}
		src := upload([]int{capacity, dim}, nan)
		defer v.Free(src)
		if err := v.V41RowLogCopy(log, 0, src, 0, capacity); err != nil {
			t.Fatal(err)
		}
		return log
	}
	score := func(k Tensor, rows int) []float32 {
		t.Helper()
		v.BeginBatch()
		out, err := v.V41IndexerScore(qt, k, wt, rows, heads, dim)
		if err != nil {
			v.FlushBatch()
			t.Fatalf("rows=%d: %v", rows, err)
		}
		got := append([]float32(nil), v.Read(out)...)
		v.Free(out)
		v.FlushBatch()
		return got
	}
	var log Tensor
	rows, capacity, doublings, spare := 0, 0, 0, 0
	for _, n := range []int{1, 1, 2, 1, 3, 5, 1, 8, 13, 1, 34} {
		if rows+n > capacity {
			grown := max(2*capacity, rows+n)
			next := poisoned(grown)
			if rows > 0 {
				if err := v.V41RowLogCopy(next, 0, log, 0, rows); err != nil {
					t.Fatal(err)
				}
				doublings++
			}
			if capacity > 0 {
				v.Free(log)
			}
			log, capacity = next, grown
		}
		staged := upload([]int{n, dim}, keys[rows*dim:(rows+n)*dim])
		if err := v.V41RowLogCopy(log, rows, staged, 0, n); err != nil {
			t.Fatal(err)
		}
		v.Free(staged)
		rows += n
		view := log
		view.Shape = []int{rows, dim}
		exact := upload([]int{rows, dim}, keys[:rows*dim])
		got, want := score(view, rows), score(exact, rows)
		v.Free(exact)
		oracle := v41IndexerScoreSerialOracle(q, keys[:rows*dim], weights, rows, heads, dim)
		for i := range oracle {
			if g, w, o := math.Float32bits(got[i]), math.Float32bits(want[i]), math.Float32bits(oracle[i]); g != w || w != o {
				t.Fatalf("rows=%d capacity=%d score[%d] spare %08x exact %08x oracle %08x", rows, capacity, i, g, w, o)
			}
		}
		if capacity > rows {
			spare++
		}
	}
	defer v.Free(log)
	if rows != total || doublings < 6 || spare < 6 {
		t.Fatalf("spare-capacity witness vacuous: rows=%d doublings=%d spare scores=%d", rows, doublings, spare)
	}

	over := log
	over.Shape = []int{capacity + 1, dim}
	bigQ, bigW := upload([]int{heads + 1, dim}, values((heads+1)*dim, 5, 1)), upload([]int{heads + 1}, values(heads+1, 7, 1))
	defer v.Free(bigQ)
	defer v.Free(bigW)
	viewQ, viewW, view := bigQ, bigW, log
	viewQ.Shape, viewW.Shape, view.Shape = []int{heads, dim}, []int{heads}, []int{rows, dim}
	for name, call := range map[string]func() (Tensor, error){
		"keys past capacity": func() (Tensor, error) { return v.V41IndexerScore(qt, over, wt, capacity+1, heads, dim) },
		"spare query":        func() (Tensor, error) { return v.V41IndexerScore(viewQ, view, wt, rows, heads, dim) },
		"spare weights":      func() (Tensor, error) { return v.V41IndexerScore(qt, view, viewW, rows, heads, dim) },
	} {
		if out, err := call(); !errors.Is(err, ErrVulkanInvalidGeometry) || out.Buf() != nil {
			t.Fatalf("%s: err=%v, want ErrVulkanInvalidGeometry and no output", name, err)
		}
	}
	short := upload([]int{rows - 1, dim}, keys[:(rows-1)*dim])
	defer v.Free(short)
	outBuf := upload([]int{rows}, make([]float32, rows))
	defer v.Free(outBuf)
	native := func(qb, kb, wb *vulkanBuf) int {
		vulkanMu.Lock()
		defer vulkanMu.Unlock()
		return v41IndexerScoreStatusLocked(qb, kb, wb, outBuf.buf.(*vulkanBuf), rows, heads, dim)
	}
	if code := native(qt.buf.(*vulkanBuf), short.buf.(*vulkanBuf), wt.buf.(*vulkanBuf)); code != 2 {
		t.Fatalf("native accepted an undersized key buffer, status=%d", code)
	}
	if code := native(bigQ.buf.(*vulkanBuf), log.buf.(*vulkanBuf), wt.buf.(*vulkanBuf)); code != 2 {
		t.Fatalf("native accepted an oversized query buffer, status=%d", code)
	}
	if code := native(qt.buf.(*vulkanBuf), log.buf.(*vulkanBuf), bigW.buf.(*vulkanBuf)); code != 2 {
		t.Fatalf("native accepted an oversized weights buffer, status=%d", code)
	}
	if ok, err := v.V41IndexerScoreAdmission(); !ok || err != nil {
		t.Fatalf("refusals left the device unadmitted: ok=%v err=%v", ok, err)
	}
}
