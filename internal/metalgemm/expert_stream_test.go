package metalgemm

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

type concurrencyWitnessReader struct {
	data       []byte
	gate       chan struct{}
	active     atomic.Int32
	peak       atomic.Int32
	totalCalls atomic.Int64
}

func newConcurrencyWitnessReader(size int, gate chan struct{}) *concurrencyWitnessReader {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 251)
	}
	return &concurrencyWitnessReader{
		data: data,
		gate: gate,
	}
}

func (r *concurrencyWitnessReader) ReadAt(p []byte, off int64) (int, error) {
	cur := r.active.Add(1)
	for {
		peak := r.peak.Load()
		if cur <= peak || r.peak.CompareAndSwap(peak, cur) {
			break
		}
	}
	r.totalCalls.Add(1)

	if r.gate != nil {
		<-r.gate
	}
	r.active.Add(-1)

	if off >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// TestExpertStream_Concurrency verifies that exactly up to 32 parallel requests
// are handled concurrently across the QD32 worker lanes without deadlock or starvation.
func TestExpertStream_Concurrency(t *testing.T) {
	const (
		queueDepth = 32
		slotCount  = 64
		expertSize = 4096
		numExperts = 48
	)

	gate := make(chan struct{})
	reader := newConcurrencyWitnessReader(expertSize*numExperts, gate)

	cfg := StreamConfig{
		QueueDepth: queueDepth,
		SlotCount:  slotCount,
		SlotBytes:  expertSize,
		Reader:     reader,
	}
	q, err := NewExpertStreamQueue(cfg)
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	for i := 0; i < numExperts; i++ {
		q.RegisterLocation(i, int64(i*expertSize), expertSize)
	}

	// Dispatch 32 distinct expert requests concurrently.
	first32 := make([]ExpertRequest, 32)
	for i := 0; i < 32; i++ {
		first32[i] = ExpertRequest{ExpertID: i}
	}

	type batchResult struct {
		leases []*SlotLease
		err    error
	}
	doneCh := make(chan batchResult, 1)

	go func() {
		leases, err := q.StreamBatch(context.Background(), first32)
		doneCh <- batchResult{leases: leases, err: err}
	}()

	// Wait for all 32 worker lanes to become concurrently active in ReadAt.
	deadline := time.Now().Add(5 * time.Second)
	for reader.active.Load() < queueDepth && time.Now().Before(deadline) {
		time.Sleep(2 * time.Millisecond)
	}

	activeCount := reader.active.Load()
	if activeCount != queueDepth {
		t.Fatalf("expected exactly %d concurrent workers in ReadAt, got %d", queueDepth, activeCount)
	}

	// Dispatch additional requests while the first 32 are still held in gate.
	next16 := make([]ExpertRequest, 16)
	for i := 0; i < 16; i++ {
		next16[i] = ExpertRequest{ExpertID: 32 + i}
	}
	nextDoneCh := make(chan batchResult, 1)
	go func() {
		leases, err := q.StreamBatch(context.Background(), next16)
		nextDoneCh <- batchResult{leases: leases, err: err}
	}()

	// Brief sleep to let second batch queue up. Active workers must still be capped at 32.
	time.Sleep(20 * time.Millisecond)
	if activeAfterQueue := reader.active.Load(); activeAfterQueue > queueDepth {
		t.Fatalf("active workers exceeded queue depth %d: got %d", queueDepth, activeAfterQueue)
	}

	// Release all workers.
	close(gate)

	// Wait for first batch.
	res1 := <-doneCh
	if res1.err != nil {
		t.Fatalf("first batch failed: %v", res1.err)
	}
	if len(res1.leases) != 32 {
		t.Fatalf("expected 32 leases, got %d", len(res1.leases))
	}
	ReleaseLeases(res1.leases)

	// Wait for second batch.
	res2 := <-nextDoneCh
	if res2.err != nil {
		t.Fatalf("second batch failed: %v", res2.err)
	}
	if len(res2.leases) != 16 {
		t.Fatalf("expected 16 leases, got %d", len(res2.leases))
	}
	ReleaseLeases(res2.leases)

	// Verify peak queue depth witnessed.
	metrics := q.Metrics()
	if metrics.PeakQueueDepth != queueDepth {
		t.Errorf("peak queue depth = %d, want %d", metrics.PeakQueueDepth, queueDepth)
	}
	if q.ActiveQueueDepth() != 0 {
		t.Errorf("expected active queue depth 0 after completion, got %d", q.ActiveQueueDepth())
	}
	if metrics.TotalReads != 48 {
		t.Errorf("expected 48 total reads, got %d", metrics.TotalReads)
	}
}

// TestExpertStream_SlotRecycling verifies that slot eviction and reuse prevent
// unbounded memory growth, and active leases prevent premature eviction.
func TestExpertStream_SlotRecycling(t *testing.T) {
	const (
		slotCount  = 4
		queueDepth = 4
		expertSize = 1024
		numExperts = 16
	)

	payload := make([]byte, numExperts*expertSize)
	for i := range payload {
		payload[i] = byte((i*17 + 3) & 0xFF)
	}
	reader := bytes.NewReader(payload)

	q, err := NewExpertStreamQueue(StreamConfig{
		QueueDepth: queueDepth,
		SlotCount:  slotCount,
		SlotBytes:  expertSize,
		Reader:     reader,
	})
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	for i := 0; i < numExperts; i++ {
		q.RegisterLocation(i, int64(i*expertSize), expertSize)
	}

	ctx := context.Background()

	// Step 1: Fill the 4 slots with experts 0..3.
	batch1 := []ExpertRequest{
		{ExpertID: 0}, {ExpertID: 1}, {ExpertID: 2}, {ExpertID: 3},
	}
	leases1, err := q.StreamBatch(ctx, batch1)
	if err != nil {
		t.Fatalf("batch 1 failed: %v", err)
	}
	if q.ResidentCount() != slotCount {
		t.Fatalf("expected %d resident experts, got %d", slotCount, q.ResidentCount())
	}
	if q.ActiveLeases() != slotCount {
		t.Fatalf("expected %d active leases, got %d", slotCount, q.ActiveLeases())
	}
	if q.SlotEvictions() != 0 {
		t.Fatalf("expected 0 evictions on initial fill, got %d", q.SlotEvictions())
	}

	// Release all 4 leases. Slots remain resident as unleased cache entries.
	ReleaseLeases(leases1)
	if q.ActiveLeases() != 0 {
		t.Fatalf("expected 0 active leases after release, got %d", q.ActiveLeases())
	}

	// Step 2: Request experts 0 and 1 again. Must be cache hits.
	batch2 := []ExpertRequest{{ExpertID: 0}, {ExpertID: 1}}
	leases2, err := q.StreamBatch(ctx, batch2)
	if err != nil {
		t.Fatalf("batch 2 failed: %v", err)
	}
	if q.SlotHits() != 2 {
		t.Fatalf("expected 2 slot hits, got %d", q.SlotHits())
	}
	ReleaseLeases(leases2)

	// Step 3: Request 4 new experts (4..7). Must trigger 4 evictions of unleased slots.
	batch3 := []ExpertRequest{
		{ExpertID: 4}, {ExpertID: 5}, {ExpertID: 6}, {ExpertID: 7},
	}
	leases3, err := q.StreamBatch(ctx, batch3)
	if err != nil {
		t.Fatalf("batch 3 failed: %v", err)
	}
	if q.SlotEvictions() != 4 {
		t.Fatalf("expected 4 slot evictions, got %d", q.SlotEvictions())
	}
	if q.ResidentCount() != slotCount {
		t.Fatalf("resident count should remain bounded at %d, got %d", slotCount, q.ResidentCount())
	}
	ReleaseLeases(leases3)

	// Step 4: Active lease protection during eviction.
	// Hold lease on expert 4.
	lease4, err := q.StreamExpert(ctx, ExpertRequest{ExpertID: 4})
	if err != nil {
		t.Fatalf("failed to lease expert 4: %v", err)
	}
	defer lease4.Release()

	// Stream 3 new experts (8, 9, 10). Since slot count is 4 and 1 slot is leased,
	// exactly the 3 unleased slots (5, 6, 7) must be evicted.
	batch4 := []ExpertRequest{{ExpertID: 8}, {ExpertID: 9}, {ExpertID: 10}}
	leases4, err := q.StreamBatch(ctx, batch4)
	if err != nil {
		t.Fatalf("batch 4 failed: %v", err)
	}
	defer ReleaseLeases(leases4)

	// Verify expert 4 is STILL resident and its data is completely intact.
	expectedExp4 := payload[4*expertSize : 5*expertSize]
	if !bytes.Equal(lease4.Bytes(), expectedExp4) {
		t.Fatalf("leased expert 4 data was corrupted by concurrent evictions")
	}

	// Verify total evictions: 4 from step 3 + 3 from step 4 = 7.
	if q.SlotEvictions() != 7 {
		t.Fatalf("expected 7 total evictions, got %d", q.SlotEvictions())
	}
}

// TestExpertStream_DataIntegrity verifies that bytes read through the streaming
// queue match the reference payload with 100% fidelity across concurrency and cache hits.
func TestExpertStream_DataIntegrity(t *testing.T) {
	const (
		numExperts = 32
		chunkSize  = 64 * 1024 // 64 KB per expert
	)

	fullData := make([]byte, numExperts*chunkSize)
	for i := 0; i < numExperts; i++ {
		start := i * chunkSize
		for j := 0; j < chunkSize; j++ {
			fullData[start+j] = byte((i*101 + j*13) & 0xFF)
		}
	}

	reader := bytes.NewReader(fullData)
	q, err := NewExpertStreamQueue(StreamConfig{
		QueueDepth: 32,
		SlotCount:  32,
		SlotBytes:  chunkSize,
		Reader:     reader,
	})
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	for i := 0; i < numExperts; i++ {
		q.RegisterLocation(i, int64(i*chunkSize), chunkSize)
	}

	ctx := context.Background()

	// Run multiple rounds with randomized expert sets to test both hits and misses.
	rng := rand.New(rand.NewSource(42))
	for round := 0; round < 10; round++ {
		batchSize := 8 + rng.Intn(9) // 8 to 16 experts per batch
		reqs := make([]ExpertRequest, batchSize)
		for b := 0; b < batchSize; b++ {
			reqs[b] = ExpertRequest{ExpertID: rng.Intn(numExperts)}
		}

		leases, err := q.StreamBatch(ctx, reqs)
		if err != nil {
			t.Fatalf("round %d StreamBatch failed: %v", round, err)
		}

		for idx, lease := range leases {
			req := reqs[idx]
			if lease.ExpertID() != req.ExpertID {
				t.Fatalf("round %d idx %d: lease expertID %d != requested %d",
					round, idx, lease.ExpertID(), req.ExpertID)
			}

			expected := fullData[req.ExpertID*chunkSize : (req.ExpertID+1)*chunkSize]
			actual := lease.Bytes()
			if !bytes.Equal(actual, expected) {
				t.Fatalf("round %d idx %d (expert %d): data mismatch at byte comparison",
					round, idx, req.ExpertID)
			}
		}

		ReleaseLeases(leases)
	}

	metrics := q.Metrics()
	if metrics.BytesTransferred == 0 {
		t.Fatalf("expected non-zero bytes transferred")
	}
}

// TestExpertStream_DuplicateInBatch verifies that duplicate expert requests in the
// same batch share a single slot without redundant reads or lease conflicts.
func TestExpertStream_DuplicateInBatch(t *testing.T) {
	const (
		expertSize = 2048
	)
	payload := make([]byte, 8*expertSize)
	for i := range payload {
		payload[i] = byte(i % 127)
	}
	reader := bytes.NewReader(payload)

	q, err := NewExpertStreamQueue(StreamConfig{
		QueueDepth: 8,
		SlotCount:  8,
		SlotBytes:  expertSize,
		Reader:     reader,
	})
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	for i := 0; i < 8; i++ {
		q.RegisterLocation(i, int64(i*expertSize), expertSize)
	}

	// Request with duplicates: expert 3 appears three times, expert 5 appears twice.
	reqs := []ExpertRequest{
		{ExpertID: 3}, {ExpertID: 5}, {ExpertID: 3}, {ExpertID: 5}, {ExpertID: 3},
	}

	leases, err := q.StreamBatch(context.Background(), reqs)
	if err != nil {
		t.Fatalf("StreamBatch with duplicates failed: %v", err)
	}
	defer ReleaseLeases(leases)

	if len(leases) != 5 {
		t.Fatalf("expected 5 leases, got %d", len(leases))
	}

	// All expert 3 leases must point to the same physical slot.
	slot3 := leases[0].SlotID()
	if leases[2].SlotID() != slot3 || leases[4].SlotID() != slot3 {
		t.Errorf("expert 3 instances did not share slot: %d vs %d vs %d",
			slot3, leases[2].SlotID(), leases[4].SlotID())
	}

	// All expert 5 leases must point to the same physical slot.
	slot5 := leases[1].SlotID()
	if leases[3].SlotID() != slot5 {
		t.Errorf("expert 5 instances did not share slot: %d vs %d", slot5, leases[3].SlotID())
	}

	// Resident count should be exactly 2 distinct experts.
	if q.ResidentCount() != 2 {
		t.Errorf("expected 2 resident experts, got %d", q.ResidentCount())
	}
}

// TestExpertStream_ContextCancellation verifies that cancelling a context while
// waiting for slots or reads terminates promptly and cleans up state.
func TestExpertStream_ContextCancellation(t *testing.T) {
	gate := make(chan struct{})
	reader := newConcurrencyWitnessReader(1024*16, gate)

	q, err := NewExpertStreamQueue(StreamConfig{
		QueueDepth: 2,
		SlotCount:  2,
		SlotBytes:  1024,
		Reader:     reader,
	})
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	for i := 0; i < 4; i++ {
		q.RegisterLocation(i, int64(i*1024), 1024)
	}

	// Fill both slots with blocked reads.
	go func() {
		q.StreamBatch(context.Background(), []ExpertRequest{{ExpertID: 0}, {ExpertID: 1}})
	}()

	time.Sleep(10 * time.Millisecond)

	// Now try to stream expert 2 with a cancellable context.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err = q.StreamExpert(ctx, ExpertRequest{ExpertID: 2})
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context deadline error, got %v", err)
	}

	close(gate)
}

// TestExpertStream_ErrorConditions tests capacity validation and error reporting.
func TestExpertStream_ErrorConditions(t *testing.T) {
	reader := bytes.NewReader(make([]byte, 8192))
	q, err := NewExpertStreamQueue(StreamConfig{
		QueueDepth: 4,
		SlotCount:  4,
		SlotBytes:  2048,
		Reader:     reader,
	})
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()

	// 1. Batch exceeds pool capacity
	tooMany := []ExpertRequest{
		{ExpertID: 0, Size: 512},
		{ExpertID: 1, Size: 512},
		{ExpertID: 2, Size: 512},
		{ExpertID: 3, Size: 512},
		{ExpertID: 4, Size: 512},
	}
	_, err = q.StreamBatch(ctx, tooMany)
	if !errors.Is(err, ErrBatchExceedsPoolCapacity) {
		t.Errorf("expected ErrBatchExceedsPoolCapacity, got %v", err)
	}

	// 2. Expert size exceeds slot capacity
	_, err = q.StreamExpert(ctx, ExpertRequest{ExpertID: 0, Size: 4096})
	if !errors.Is(err, ErrSizeExceedsSlot) {
		t.Errorf("expected ErrSizeExceedsSlot, got %v", err)
	}

	// 3. Unregistered location
	_, err = q.StreamExpert(ctx, ExpertRequest{ExpertID: 99})
	if !errors.Is(err, ErrExpertNotFound) {
		t.Errorf("expected ErrExpertNotFound, got %v", err)
	}

	// 4. Closed queue rejects requests
	q.Close()
	_, err = q.StreamExpert(ctx, ExpertRequest{ExpertID: 0, Size: 512})
	if !errors.Is(err, ErrQueueClosed) {
		t.Errorf("expected ErrQueueClosed, got %v", err)
	}
}

// TestExpertStream_FilePreadDirect verifies streaming directly from a real filesystem
// file descriptor exercising the OS pread path.
func TestExpertStream_FilePreadDirect(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "expert_weights.bin")

	const (
		numExperts = 8
		expertSize = 128 * 1024 // 128 KB
	)
	fileData := make([]byte, numExperts*expertSize)
	for i := range fileData {
		fileData[i] = byte((i * 47) & 0xFF)
	}

	if err := os.WriteFile(filePath, fileData, 0o600); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	f, err := os.Open(filePath)
	if err != nil {
		t.Fatalf("failed to open test file: %v", err)
	}
	defer f.Close()

	q, err := NewExpertStreamQueue(StreamConfig{
		QueueDepth: 16,
		SlotCount:  16,
		SlotBytes:  expertSize,
		Reader:     f,
	})
	if err != nil {
		t.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	for i := 0; i < numExperts; i++ {
		q.RegisterLocation(i, int64(i*expertSize), expertSize)
	}

	leases, err := q.StreamExperts(context.Background(), []int{0, 2, 4, 6})
	if err != nil {
		t.Fatalf("StreamExperts failed: %v", err)
	}
	defer ReleaseLeases(leases)

	for _, l := range leases {
		expected := fileData[l.ExpertID()*expertSize : (l.ExpertID()+1)*expertSize]
		if !bytes.Equal(l.Bytes(), expected) {
			t.Errorf("file pread data mismatch for expert %d", l.ExpertID())
		}
	}
}

// BenchmarkExpertStream_QD32Throughput benchmarks the sustained throughput of the
// QD32 worker lanes reading simulated 4-bit expert records into the slot pool.
func BenchmarkExpertStream_QD32Throughput(b *testing.B) {
	// Qwen 3.8 4-bit expert record (gate/up/down tensors) is ~2.76 MB.
	// We benchmark 300 KB chunk and 2.76 MB full-record scenarios.
	b.Run("300KB_Chunks", func(b *testing.B) {
		benchmarkThroughput(b, 300*1024, 64, 32)
	})
	b.Run("2.76MB_FullExpertRecords", func(b *testing.B) {
		benchmarkThroughput(b, 2760*1024, 64, 32)
	})
}

func benchmarkThroughput(b *testing.B, expertBytes int, slotCount, queueDepth int) {
	const numDistinctExperts = 64
	totalPayload := int64(numDistinctExperts * expertBytes)

	// Fast in-memory reader
	payload := make([]byte, totalPayload)
	reader := bytes.NewReader(payload)

	q, err := NewExpertStreamQueue(StreamConfig{
		QueueDepth: queueDepth,
		SlotCount:  slotCount,
		SlotBytes:  expertBytes,
		Reader:     reader,
	})
	if err != nil {
		b.Fatalf("failed to create queue: %v", err)
	}
	defer q.Close()

	for i := 0; i < numDistinctExperts; i++ {
		q.RegisterLocation(i, int64(i*expertBytes), int64(expertBytes))
	}

	ctx := context.Background()
	const routedExpertsPerStep = 10 // typical Qwen 3.8 routed experts per token

	reqs := make([]ExpertRequest, routedExpertsPerStep)
	for i := 0; i < routedExpertsPerStep; i++ {
		reqs[i] = ExpertRequest{ExpertID: i}
	}

	stepBytes := int64(routedExpertsPerStep * expertBytes)
	b.SetBytes(stepBytes)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Rotate expert IDs to cycle through slot pool
		offset := (i * routedExpertsPerStep) % numDistinctExperts
		for j := 0; j < routedExpertsPerStep; j++ {
			reqs[j].ExpertID = (offset + j) % numDistinctExperts
		}

		leases, err := q.StreamBatch(ctx, reqs)
		if err != nil {
			b.Fatalf("step %d failed: %v", i, err)
		}
		ReleaseLeases(leases)
	}
	b.StopTimer()

	metrics := q.Metrics()
	b.ReportMetric(float64(metrics.BytesTransferred)/(1024*1024), "MB_transferred")
	b.ReportMetric(float64(metrics.SlotHits), "slot_hits")
	b.ReportMetric(float64(metrics.SlotEvictions), "slot_evictions")
}
