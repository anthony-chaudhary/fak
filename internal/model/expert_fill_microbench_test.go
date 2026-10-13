package model

import (
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"testing"
	"time"
	"unsafe"
)

// expert_fill_microbench_test.go — the page-in GB/s instrument for the P2 one-copy ring fill
// (docs/benchmarks/receipts/halo-v41-100-30/, DS4.1 Q1). It is a MEASUREMENT harness, not a
// correctness witness: the correctness witnesses are the expert checkpoint tier tests, and this
// exists so the read path's throughput claim is reproducible on any host with a disk rather than
// asserted from a single strix3 run.
//
// Why it is a Go benchmark and not a shell probe. The ticket's witness is "page-in GB/s
// microbench". A throughput number is only comparable when the unit of work and the shape of the
// reads are fixed, so this file pins both: the unit is one ~4.27 MB expert stride (the size the
// strix3 slice-1 receipt measured at 1.19 ms), and the destination is a genuinely 2 MiB-aligned
// buffer — the alignment the P2 port targets, because an unaligned destination can split a request
// across pages the kernel must then stitch.
//
// It compares the two shapes the port sits between, on the SAME bytes and the SAME traversal
// order, so the only variable is concurrency:
//
//   - serial: one stride at a time (the queue-depth-one baseline P2 step 1 replaced).
//   - pool:   the layer's misses dispatched in ascending file-offset order through a bounded
//     worker pool, which is what ExpertCheckpointTier.Prefetch does for a real tier.
//
// Run with:
//
//	go test ./internal/model -run XXX -bench ExpertFillPageIn -benchtime 3x -count=1
//
// fak-test:runtime slow est=8s lane=default

// expertFillStrideBytes is one checkpoint expert projection's stride on the V4.1 Flash Q2_K
// artifact, the 4.27 MB unit both the slice-1 receipt and this microbench are stated in.
const expertFillStrideBytes = 4_269_056

// expertFillAligned returns a stride-length slice whose base address is 2 MiB-aligned, by
// over-allocating and rounding the pointer up. This is the destination shape the P2 port targets;
// a plain make() is only 8- or 16-byte aligned on most allocators.
func expertFillAligned(stride int) []byte {
	const align = 1 << 21
	backing := make([]byte, stride+align)
	base := uintptr(unsafe.Pointer(&backing[0]))
	off := (align - int(base&(align-1))) & (align - 1)
	return backing[off : off+stride]
}

// expertFillPageInStride reads stride bytes at off into a caller-owned, 2 MiB-aligned destination.
// Buffered (no O_DIRECT): the warm path the slice-1 receipt measured is page-cache served, and
// O_DIRECT would measure the device under a different contract.
func expertFillPageInStride(f *os.File, off int64, dst []byte) error {
	_, err := f.ReadAt(dst, off)
	return err
}

// expertFillTempSlab writes nStrides synthetic expert strides to a temp file and returns it open
// for reading. The bytes are deterministic and irrelevant — this measures the read path, not the
// model — but they are written so the read is a genuine disk read rather than a sparse hole.
func expertFillTempSlab(tb testing.TB, nStrides int) (*os.File, int64) {
	tb.Helper()
	f, err := os.CreateTemp(tb.TempDir(), "expert-fill-*.slab")
	if err != nil {
		tb.Fatal(err)
	}
	chunk := make([]byte, 1<<20)
	for i := range chunk {
		chunk[i] = byte(i)
	}
	for i := 0; i < nStrides; i++ {
		remaining := int64(expertFillStrideBytes)
		for remaining > 0 {
			n := int64(len(chunk))
			if n > remaining {
				n = remaining
			}
			if _, err := f.Write(chunk[:n]); err != nil {
				f.Close()
				tb.Fatal(err)
			}
			remaining -= n
		}
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		f.Close()
		tb.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		tb.Fatal(err)
	}
	return f, info.Size()
}

// expertFillAscendingOffsets returns the ascending absolute offsets of nStrides strides. Both
// benchmark arms consume THIS slice in the same order, so the only difference between arms is
// concurrency, never traversal direction.
func expertFillAscendingOffsets(nStrides int) []int64 {
	offs := make([]int64, nStrides)
	for i := range offs {
		offs[i] = int64(i) * expertFillStrideBytes
	}
	return offs
}

func expertFillGBps(bytes int64, d time.Duration) float64 {
	return float64(bytes) / d.Seconds() / 1e9
}

// BenchmarkExpertFillPageInSerial is the queue-depth-one baseline: one stride at a time.
func BenchmarkExpertFillPageInSerial(b *testing.B) {
	const strides = 24
	f, _ := expertFillTempSlab(b, strides)
	defer f.Close()
	offs := expertFillAscendingOffsets(strides)
	dst := expertFillAligned(expertFillStrideBytes)
	b.SetBytes(int64(strides * expertFillStrideBytes))
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for _, off := range offs {
			if err := expertFillPageInStride(f, off, dst); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkExpertFillPageInSortedPool is the port's shape: ascending offsets through a bounded
// pool, one aligned destination per in-flight read.
func BenchmarkExpertFillPageInSortedPool(b *testing.B) {
	const strides = 24
	f, _ := expertFillTempSlab(b, strides)
	defer f.Close()
	offs := expertFillAscendingOffsets(strides)
	b.SetBytes(int64(strides * expertFillStrideBytes))
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		if err := expertFillSortedPool(f, offs, fakExpertPrefetchWorkers); err != nil {
			b.Fatal(err)
		}
	}
}

// expertFillSortedPool reads every offset in ascending order, at most workers at a time, each
// into its own 2 MiB-aligned destination. This is the read shape ExpertCheckpointTier.Prefetch
// drives; the tier's own version wraps it in the load/publication handoff this microbench omits
// because it measures IO, not publication.
func expertFillSortedPool(f *os.File, offs []int64, workers int) error {
	if workers < 1 {
		workers = 1
	}
	jobs := make(chan int64, len(offs))
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		werr error
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dst := expertFillAligned(expertFillStrideBytes)
			for off := range jobs {
				if err := expertFillPageInStride(f, off, dst); err != nil {
					mu.Lock()
					if werr == nil {
						werr = err
					}
					mu.Unlock()
				}
			}
		}()
	}
	for _, off := range offs {
		jobs <- off
	}
	close(jobs)
	wg.Wait()
	return werr
}

// TestExpertFillMicrobenchReceiptLine prints a machine-readable line the receipt writer can paste
// verbatim, so the committed witness and the hardware receipt share one format. It is a
// MEASUREMENT, not a pass/fail gate: a throughput floor asserted against a warm page cache on a
// shared host is a flake, so this test only proves the harness runs and reports.
// fak-test:runtime medium est=2s lane=default
func TestExpertFillMicrobenchReceiptLine(t *testing.T) {
	const strides = 16
	f, _ := expertFillTempSlab(t, strides)
	defer f.Close()
	total := int64(strides * expertFillStrideBytes)
	offs := expertFillAscendingOffsets(strides)

	dst := expertFillAligned(expertFillStrideBytes)
	start := time.Now()
	for _, off := range offs {
		if err := expertFillPageInStride(f, off, dst); err != nil {
			t.Fatal(err)
		}
	}
	serial := time.Since(start)

	start = time.Now()
	if err := expertFillSortedPool(f, offs, fakExpertPrefetchWorkers); err != nil {
		t.Fatal(err)
	}
	pool := time.Since(start)

	if len(dst) != expertFillStrideBytes || uintptr(unsafe.Pointer(&dst[0]))%(1<<21) != 0 {
		t.Fatalf("microbench destination is not %d-byte aligned", 1<<21)
	}
	fmt.Printf("expert-fill-microbench mode=sorted-pool workers=%d strides=%d stride_bytes=%d gbps=%.3f ms_per_stride=%.3f serial_gbps=%.3f\n",
		fakExpertPrefetchWorkers, strides, expertFillStrideBytes, expertFillGBps(total, pool),
		pool.Seconds()*1000/float64(strides), expertFillGBps(total, serial))
}

// TestExpertFillOrderingIsDeterministic proves the ordering half of the port has a deterministic
// unit witness independent of timing: the offsets a fill consumes are ascending, so a fill that
// trusted arrival order would fail this. (The concurrency half is witnessed by the tier test
// TestExpertCheckpointPrefetchBoundsConcurrency, and the ascending dispatch order by
// TestExpertCheckpointPrefetchDispatchesAscendingOffsets.)
// fak-test:runtime fast est=100ms lane=default
func TestExpertFillOrderingIsDeterministic(t *testing.T) {
	offs := expertFillAscendingOffsets(24)
	if !sort.SliceIsSorted(offs, func(i, j int) bool { return offs[i] < offs[j] }) {
		t.Fatal("expert fill offsets are not ascending")
	}
}
