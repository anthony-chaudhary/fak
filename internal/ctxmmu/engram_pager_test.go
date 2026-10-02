package ctxmmu

import (
	"bytes"
	"context"
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// cb14BlockedIO joins both callers before reporting a blocked concurrent read.
func cb14BlockedIO(t *testing.T, blocked func() error, entered <-chan struct{}, release chan struct{}, reader func() error) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- blocked() }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		close(release)
		<-done
		t.Fatal("I/O callback never entered")
	}
	readDone := make(chan error, 1)
	go func() { readDone <- reader() }()
	var readErr error
	responsive := false
	select {
	case readErr = <-readDone:
		responsive = true
	case <-time.After(150 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("blocked operation: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked operation did not join")
	}
	if !responsive {
		select {
		case readErr = <-readDone:
		case <-time.After(time.Second):
			t.Fatal("reader did not join")
		}
	}
	if readErr != nil {
		t.Errorf("concurrent gather: %v", readErr)
	}
	if !responsive {
		t.Error("global pager lock held across blocked I/O")
	}
}

// fak-test:runtime slow est=1s lane=default
func TestCB14EngramPagerAdvisoryDoesNotBlockGather(t *testing.T) {
	for _, advice := range []int{MadvWillNeed, MadvDontNeed} {
		t.Run(string(rune('0'+advice)), func(t *testing.T) {
			entered := make(chan struct{})
			release := make(chan struct{})
			var armed atomic.Bool
			var once sync.Once
			hook := func(_ uintptr, _ uintptr, a int) error {
				if armed.Load() && a == advice {
					once.Do(func() { close(entered); <-release })
				}
				return nil
			}
			data := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
			pager, err := NewEngramPager(EngramPagerConfig{RowSizeBytes: 4, TotalRows: 4, PageSizeBytes: 8, MaxResidentBytes: 8, AnonymousData: data, MadviseSyscall: hook})
			if err != nil {
				t.Fatal(err)
			}
			defer pager.Close()
			if err := pager.GatherRows([]uint64{0}, make([]byte, 4)); err != nil {
				t.Fatal(err)
			}
			armed.Store(true)
			operation := func() error { return pager.PrefetchUbatchRows(context.Background(), []uint64{2}) }
			row := uint64(0)
			if advice == MadvDontNeed {
				row = 2
			}
			cb14BlockedIO(t, operation, entered, release, func() error {
				out := make([]byte, 4)
				err := pager.GatherRows([]uint64{row}, out)
				if err == nil && !bytes.Equal(out, data[row*4:row*4+4]) {
					t.Errorf("wrong concurrent row bytes %v", out)
				}
				_ = pager.Telemetry()
				return err
			})
		})
	}
}

func cb14FilePager(t *testing.T) (*EngramPager, *os.File, []byte) {
	t.Helper()
	data := make([]byte, 32)
	for i := range data {
		data[i] = byte(i)
	}
	path := filepath.Join(t.TempDir(), "table.bin")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewEngramPager(EngramPagerConfig{TablePath: path, RowSizeBytes: 4, TotalRows: 8, PageSizeBytes: 8, MaxResidentBytes: 32})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close() })
	f, err := os.OpenFile(path, os.O_RDWR, 0600)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })
	return p, f, data
}

// fak-test:runtime slow est=1s lane=default
func TestCB14EngramPagerReadFaultDoesNotBlockOtherPage(t *testing.T) {
	p, f, data := cb14FilePager(t)
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	p.fileReader = func(b []byte, off int64) (int, error) {
		if off == 0 {
			once.Do(func() { close(entered); <-release })
		}
		return f.ReadAt(b, off)
	}
	cb14BlockedIO(t, func() error { return p.GatherRows([]uint64{0}, make([]byte, 4)) }, entered, release, func() error {
		out := make([]byte, 4)
		err := p.GatherRows([]uint64{4}, out)
		if err == nil && !bytes.Equal(out, data[16:20]) {
			t.Errorf("wrong other-page bytes %v", out)
		}
		_ = p.Telemetry()
		return err
	})
}

// fak-test:runtime slow est=1s lane=default
func TestCB14EngramPagerWriteDoesNotBlockOtherPage(t *testing.T) {
	p, f, data := cb14FilePager(t)
	if err := p.GatherRows([]uint64{4}, make([]byte, 4)); err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	p.fileWriter = func(b []byte, off int64) (int, error) {
		if off == 0 {
			once.Do(func() { close(entered); <-release })
		}
		return f.WriteAt(b, off)
	}
	cb14BlockedIO(t, func() error { return p.WriteRow(0, []byte{90, 91, 92, 93}) }, entered, release, func() error {
		out := make([]byte, 4)
		err := p.GatherRows([]uint64{4}, out)
		if err == nil && !bytes.Equal(out, data[16:20]) {
			t.Errorf("wrong unaffected row %v", out)
		}
		_ = p.Telemetry()
		return err
	})
	out := make([]byte, 8)
	if err := p.GatherRows([]uint64{0, 1}, out); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, []byte{90, 91, 92, 93, 4, 5, 6, 7}) {
		t.Fatalf("write corrupted adjacent row: %v", out)
	}
}

// fak-test:runtime fast est=0.1s lane=unit
func TestCB14EngramPagerConcurrentSamePageRows(t *testing.T) {
	p, _, _ := cb14FilePager(t)
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for row := uint64(0); row < 2; row++ {
		wg.Add(1)
		go func(row uint64) {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				if err := p.WriteRow(row, bytes.Repeat([]byte{byte(row + 90)}, 4)); err != nil {
					errs <- err
					return
				}
			}
		}(row)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	out := make([]byte, 8)
	if err := p.GatherRows([]uint64{0, 1}, out); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, []byte{90, 90, 90, 90, 91, 91, 91, 91}) {
		t.Fatalf("same-page row writes lost: %v", out)
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_SequentialDistribution(t *testing.T) {
	ctx := context.Background()
	const totalRows = 50000
	const rowSize = DefaultRowSizeBytes         // 130 bytes
	const maxResident = DefaultMaxResidentBytes // 2 GiB

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:        totalRows,
		RowSizeBytes:     rowSize,
		MaxResidentBytes: maxResident,
	})
	if err != nil {
		t.Fatalf("failed to create engram pager: %v", err)
	}
	defer pager.Close()

	// Initial advisory must be MADV_RANDOM
	advisories := pager.Advisories()
	if len(advisories) == 0 || advisories[0].Advice != MadvRandom {
		t.Fatalf("expected first advisory to be MADV_RANDOM, got %+v", advisories)
	}

	// Sequential ubatch: rows 1000 to 6000 (5000 rows)
	const ubatchRows = 5000
	indices := make([]uint64, ubatchRows)
	for i := 0; i < ubatchRows; i++ {
		indices[i] = uint64(1000 + i)
	}

	// 1. Batched prefetch
	if err := pager.PrefetchUbatchRows(ctx, indices); err != nil {
		t.Fatalf("PrefetchUbatchRows failed: %v", err)
	}

	// 2. Gather rows
	out := make([]byte, ubatchRows*int(rowSize))
	if err := pager.GatherRows(indices, out); err != nil {
		t.Fatalf("GatherRows failed: %v", err)
	}

	// 3. Assert resident memory RSS strictly <= 2.0 GiB
	residentBytes := pager.ResidentBytes()
	if residentBytes > maxResident {
		t.Fatalf("resident memory RSS exceeded 2 GiB limit: %d > %d", residentBytes, maxResident)
	}
	if residentBytes == 0 {
		t.Fatalf("expected non-zero resident bytes after prefetch")
	}

	// 4. Assert read amplification ratio stays <= 1.2x
	amp := pager.ReadAmplificationRatio()
	t.Logf("Sequential read amplification: %.4fx (resident: %d bytes, gathered: %d bytes)",
		amp, residentBytes, pager.TotalGatheredBytes())

	if amp > 1.2 {
		t.Fatalf("expected read amplification <= 1.2x, got %.4fx", amp)
	}

	// 5. Compare with standard sequential readahead baseline
	telem := pager.Telemetry()
	t.Logf("Standard sequential readahead amplification: %.2fx vs batched: %.4fx",
		telem.StandardAmplification, telem.ReadAmplification)
	if telem.StandardAmplification < 100.0 {
		t.Fatalf("expected standard sequential baseline amplification > 100x, got %.2fx",
			telem.StandardAmplification)
	}

	// 6. Assert all rows hit cache since they were prefetched
	if telem.CacheMisses != 0 {
		t.Fatalf("expected 0 cache misses after prefetch, got %d", telem.CacheMisses)
	}
	if telem.CacheHits != ubatchRows {
		t.Fatalf("expected %d cache hits, got %d", ubatchRows, telem.CacheHits)
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_RandomDistribution(t *testing.T) {
	ctx := context.Background()
	const totalRows = 100000
	const rowSize = DefaultRowSizeBytes
	const maxResident = DefaultMaxResidentBytes

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:        totalRows,
		RowSizeBytes:     rowSize,
		MaxResidentBytes: maxResident,
	})
	if err != nil {
		t.Fatalf("failed to create engram pager: %v", err)
	}
	defer pager.Close()

	// Simulate 4 ubatches with Zipfian/clustered token access pattern
	rng := rand.New(rand.NewSource(42))
	vocabSubset := make([]uint64, 500)
	for i := range vocabSubset {
		vocabSubset[i] = uint64(rng.Intn(totalRows))
	}

	const rowsPerUbatch = 1024
	out := make([]byte, rowsPerUbatch*int(rowSize))

	for ubatch := 0; ubatch < 4; ubatch++ {
		indices := make([]uint64, rowsPerUbatch)
		for i := 0; i < rowsPerUbatch; i++ {
			// 80% of tokens come from hot vocabSubset, 20% random
			if rng.Float64() < 0.8 {
				indices[i] = vocabSubset[rng.Intn(len(vocabSubset))]
			} else {
				indices[i] = uint64(rng.Intn(totalRows))
			}
		}

		if err := pager.PrefetchUbatchRows(ctx, indices); err != nil {
			t.Fatalf("ubatch %d: PrefetchUbatchRows failed: %v", ubatch, err)
		}

		if err := pager.GatherRows(indices, out); err != nil {
			t.Fatalf("ubatch %d: GatherRows failed: %v", ubatch, err)
		}

		// Assert resident memory RSS stays strictly <= 2.0 GiB RSS
		res := pager.ResidentBytes()
		if res > maxResident {
			t.Fatalf("ubatch %d: resident memory %d > 2 GiB", ubatch, res)
		}
	}

	telem := pager.Telemetry()
	t.Logf("Random distribution telemetry: hits=%d, misses=%d, hit_ratio=%.2f%%, resident=%d bytes",
		telem.CacheHits, telem.CacheMisses, pager.CacheHitRatio()*100, telem.ResidentBytes)

	if telem.CacheHits == 0 {
		t.Fatalf("expected cache hits during clustered random gathers")
	}
	if telem.ResidentBytes > maxResident {
		t.Fatalf("resident memory exceeded 2 GiB: %d", telem.ResidentBytes)
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_MaxResidentBoundAndEviction(t *testing.T) {
	ctx := context.Background()
	const totalRows = 50000
	const rowSize = DefaultRowSizeBytes
	// Set small 256 KiB budget (64 pages) to trigger active eviction
	const smallResidentLimit = 256 * 1024

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:        totalRows,
		RowSizeBytes:     rowSize,
		MaxResidentBytes: smallResidentLimit,
	})
	if err != nil {
		t.Fatalf("failed to create engram pager: %v", err)
	}
	defer pager.Close()

	// Access 15,000 sequential rows (~1.95 MiB, ~488 pages), far exceeding 256 KiB
	const numRows = 15000
	indices := make([]uint64, numRows)
	for i := 0; i < numRows; i++ {
		indices[i] = uint64(i)
	}

	// Prefetch in chunks of 1000
	for chunkStart := 0; chunkStart < numRows; chunkStart += 1000 {
		chunkEnd := chunkStart + 1000
		if chunkEnd > numRows {
			chunkEnd = numRows
		}
		chunk := indices[chunkStart:chunkEnd]

		if err := pager.PrefetchUbatchRows(ctx, chunk); err != nil {
			t.Fatalf("prefetch chunk failed: %v", err)
		}

		// Assert resident memory strictly never exceeds the configured limit
		res := pager.ResidentBytes()
		if res > smallResidentLimit {
			t.Fatalf("resident bytes %d exceeded limit %d", res, smallResidentLimit)
		}
		if res > DefaultMaxResidentBytes {
			t.Fatalf("resident bytes %d exceeded 2 GiB limit", res)
		}
	}

	telem := pager.Telemetry()
	t.Logf("Eviction test: evicted_pages=%d, resident_pages=%d, resident_bytes=%d",
		telem.EvictedPages, telem.ResidentPages, telem.ResidentBytes)

	if telem.EvictedPages == 0 {
		t.Fatalf("expected page evictions when working set exceeded limit")
	}

	// Verify MADV_DONTNEED advisories were issued for evicted pages
	advisories := pager.Advisories()
	dontNeedCount := 0
	for _, adv := range advisories {
		if adv.Advice == MadvDontNeed {
			dontNeedCount++
		}
	}
	if dontNeedCount == 0 {
		t.Fatalf("expected MADV_DONTNEED advisories to be issued on eviction")
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_ReadAmplificationRatio(t *testing.T) {
	ctx := context.Background()
	const totalRows = 20000
	const rowSize = DefaultRowSizeBytes

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:    totalRows,
		RowSizeBytes: rowSize,
	})
	if err != nil {
		t.Fatalf("failed to create engram pager: %v", err)
	}
	defer pager.Close()

	// Prefetch 2,000 contiguous rows (~260 KB)
	indices := make([]uint64, 2000)
	for i := range indices {
		indices[i] = uint64(i + 500)
	}

	if err := pager.PrefetchUbatchRows(ctx, indices); err != nil {
		t.Fatalf("PrefetchUbatchRows failed: %v", err)
	}

	out := make([]byte, len(indices)*int(rowSize))
	if err := pager.GatherRows(indices, out); err != nil {
		t.Fatalf("GatherRows failed: %v", err)
	}

	ratio := pager.ReadAmplificationRatio()
	t.Logf("Read amplification ratio: %.4fx (NVMe read: %d bytes, gathered: %d bytes)",
		ratio, pager.NVMeReadBytes(), pager.TotalGatheredBytes())

	if ratio > 1.2 {
		t.Fatalf("read amplification ratio %.4fx exceeded 1.2x threshold", ratio)
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_ThroughputBenchmark(t *testing.T) {
	ctx := context.Background()
	const totalRows = 50000
	const rowSize = DefaultRowSizeBytes
	// 1 token = 16 embedding rows
	const rowsPerToken = 16
	const testTokens = 1000
	const totalGatherRows = testTokens * rowsPerToken // 16,000 rows

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:    totalRows,
		RowSizeBytes: rowSize,
	})
	if err != nil {
		t.Fatalf("failed to create engram pager: %v", err)
	}
	defer pager.Close()

	// Prepare row indices for testTokens
	indices := make([]uint64, totalGatherRows)
	for i := range indices {
		indices[i] = uint64(i % int(totalRows))
	}

	// Warmup/prefetch the rows
	if err := pager.PrefetchUbatchRows(ctx, indices); err != nil {
		t.Fatalf("PrefetchUbatchRows failed: %v", err)
	}

	out := make([]byte, totalGatherRows*int(rowSize))

	// Measure gather throughput
	start := time.Now()
	if err := pager.GatherRows(indices, out); err != nil {
		t.Fatalf("GatherRows failed: %v", err)
	}
	elapsed := time.Since(start)

	tokPerSec := float64(testTokens) / elapsed.Seconds()
	t.Logf("Gather throughput: %.2f tok/s (%d tokens, %d rows in %v)",
		tokPerSec, testTokens, totalGatherRows, elapsed)

	// Assert throughput exceeds 300 tok/s
	if tokPerSec <= 300.0 {
		t.Fatalf("throughput %.2f tok/s failed to meet > 300 tok/s threshold", tokPerSec)
	}
}

func BenchmarkEngramPager_GatherRate(b *testing.B) {
	ctx := context.Background()
	const totalRows = 50000
	const rowSize = DefaultRowSizeBytes
	const rowsPerToken = 16
	const tokens = 512
	const gatherRows = tokens * rowsPerToken

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:    totalRows,
		RowSizeBytes: rowSize,
	})
	if err != nil {
		b.Fatalf("create pager: %v", err)
	}
	defer pager.Close()

	indices := make([]uint64, gatherRows)
	for i := range indices {
		indices[i] = uint64(i % int(totalRows))
	}
	_ = pager.PrefetchUbatchRows(ctx, indices)

	out := make([]byte, gatherRows*int(rowSize))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := pager.GatherRows(indices, out); err != nil {
			b.Fatalf("gather: %v", err)
		}
	}
	b.StopTimer()

	totalTokens := float64(b.N * tokens)
	tokPerSec := totalTokens / b.Elapsed().Seconds()
	b.ReportMetric(tokPerSec, "tok/s")
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_FileBacked(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	tablePath := filepath.Join(tmpDir, "engram_table.bin")

	const totalRows = 5000
	const rowSize = DefaultRowSizeBytes
	totalBytes := totalRows * rowSize

	// Write synthetic test file
	data := make([]byte, totalBytes)
	for i := range data {
		data[i] = byte((i * 17) % 256)
	}
	if err := os.WriteFile(tablePath, data, 0644); err != nil {
		t.Fatalf("write table file: %v", err)
	}

	pager, err := NewEngramPager(EngramPagerConfig{
		TablePath:    tablePath,
		TotalRows:    totalRows,
		RowSizeBytes: rowSize,
	})
	if err != nil {
		t.Fatalf("create file-backed pager: %v", err)
	}
	defer pager.Close()

	// Prefetch and gather rows 100 to 500
	indices := make([]uint64, 400)
	for i := range indices {
		indices[i] = uint64(100 + i)
	}

	if err := pager.PrefetchUbatchRows(ctx, indices); err != nil {
		t.Fatalf("prefetch: %v", err)
	}

	out := make([]byte, len(indices)*int(rowSize))
	if err := pager.GatherRows(indices, out); err != nil {
		t.Fatalf("gather: %v", err)
	}

	// Verify data matches original file
	for i, idx := range indices {
		expected := data[idx*rowSize : (idx+1)*rowSize]
		actual := out[i*int(rowSize) : (i+1)*int(rowSize)]
		for b := 0; b < int(rowSize); b++ {
			if actual[b] != expected[b] {
				t.Fatalf("row %d byte %d mismatch: got %d, want %d", idx, b, actual[b], expected[b])
			}
		}
	}

	if amp := pager.ReadAmplificationRatio(); amp > 1.2 {
		t.Fatalf("expected read amp <= 1.2x on file-backed, got %.4fx", amp)
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_ConcurrencyRace(t *testing.T) {
	ctx := context.Background()
	const totalRows = 20000
	const rowSize = DefaultRowSizeBytes

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:    totalRows,
		RowSizeBytes: rowSize,
	})
	if err != nil {
		t.Fatalf("create pager: %v", err)
	}
	defer pager.Close()

	const numWorkers = 8
	const gathersPerWorker = 20
	const rowsPerGather = 64

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			out := make([]byte, rowsPerGather*int(rowSize))
			for g := 0; g < gathersPerWorker; g++ {
				indices := make([]uint64, rowsPerGather)
				offset := (workerID*100 + g*10) % int(totalRows-rowsPerGather)
				for i := range indices {
					indices[i] = uint64(offset + i)
				}

				if err := pager.PrefetchUbatchRows(ctx, indices); err != nil {
					t.Errorf("worker %d prefetch: %v", workerID, err)
					return
				}

				if err := pager.GatherRows(indices, out); err != nil {
					t.Errorf("worker %d gather: %v", workerID, err)
					return
				}
			}
		}()
	}

	wg.Wait()

	if res := pager.ResidentBytes(); res > DefaultMaxResidentBytes {
		t.Fatalf("resident bytes exceeded 2 GiB: %d", res)
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_EdgeCases(t *testing.T) {
	ctx := context.Background()

	// Zero total rows error
	_, err := NewEngramPager(EngramPagerConfig{TotalRows: 0})
	if !errors.Is(err, ErrTotalRowsZero) {
		t.Fatalf("expected ErrTotalRowsZero, got %v", err)
	}

	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:    1000,
		RowSizeBytes: 130,
	})
	if err != nil {
		t.Fatalf("create pager: %v", err)
	}
	defer pager.Close()

	// Row index out of bounds in prefetch
	err = pager.PrefetchUbatchRows(ctx, []uint64{1000})
	if !errors.Is(err, ErrRowOutOfBounds) {
		t.Fatalf("expected ErrRowOutOfBounds, got %v", err)
	}

	// Row index out of bounds in gather
	out := make([]byte, 130)
	err = pager.GatherRows([]uint64{1001}, out)
	if !errors.Is(err, ErrRowOutOfBounds) {
		t.Fatalf("expected ErrRowOutOfBounds, got %v", err)
	}

	// Small output buffer
	err = pager.GatherRows([]uint64{0}, make([]byte, 129))
	if !errors.Is(err, ErrOutputBufferSmall) {
		t.Fatalf("expected ErrOutputBufferSmall, got %v", err)
	}

	// Cancelled context
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel()
	err = pager.PrefetchUbatchRows(cancelCtx, []uint64{0})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	// Empty indices is a no-op
	if err := pager.PrefetchUbatchRows(ctx, nil); err != nil {
		t.Fatalf("expected nil for empty prefetch, got %v", err)
	}
	if err := pager.GatherRows(nil, nil); err != nil {
		t.Fatalf("expected nil for empty gather, got %v", err)
	}

	// Close pager
	if err := pager.Close(); err != nil {
		t.Fatalf("close pager: %v", err)
	}

	// Operations after close should return ErrPagerClosed
	if err := pager.PrefetchUbatchRows(ctx, []uint64{0}); !errors.Is(err, ErrPagerClosed) {
		t.Fatalf("expected ErrPagerClosed, got %v", err)
	}
	if err := pager.GatherRows([]uint64{0}, out); !errors.Is(err, ErrPagerClosed) {
		t.Fatalf("expected ErrPagerClosed, got %v", err)
	}
}

// fak-test:runtime slow est=2s lane=default
// Migrated existing software pager coverage; no physical NVMe measurement.
func TestEngramPager_WriteRow(t *testing.T) {
	pager, err := NewEngramPager(EngramPagerConfig{
		TotalRows:    100,
		RowSizeBytes: 130,
	})
	if err != nil {
		t.Fatalf("create pager: %v", err)
	}
	defer pager.Close()

	customRow := make([]byte, 130)
	for i := range customRow {
		customRow[i] = 0xAA
	}

	if err := pager.WriteRow(42, customRow); err != nil {
		t.Fatalf("WriteRow failed: %v", err)
	}

	out := make([]byte, 130)
	if err := pager.GatherRows([]uint64{42}, out); err != nil {
		t.Fatalf("GatherRows failed: %v", err)
	}

	for i, b := range out {
		if b != 0xAA {
			t.Fatalf("byte %d: expected 0xAA, got %x", i, b)
		}
	}
}
