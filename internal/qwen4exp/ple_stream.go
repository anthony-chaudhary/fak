package qwen4exp

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// DefaultNgramSize is the standard 3-gram window for Qwen3.8-Flash-Next PLE.
	DefaultNgramSize = 3
	// DefaultEOSTokenID is the standard Qwen EOS padding token.
	DefaultEOSTokenID = 151643
	// PLERowsPerToken is the fixed number of embedding rows gathered per token (16 rows).
	PLERowsPerToken = 16
	// PLERowBytes is the fixed byte width of one PLE row (4KB = 4096 bytes).
	PLERowBytes = 4096
	// PLEBatchBytes is the fixed size of the pinned staging buffer (16 * 4096 = 64KB).
	PLEBatchBytes = PLERowsPerToken * PLERowBytes
	// MaxTargetLookupLatency is the target latency envelope for 16-row gather (0.5ms).
	MaxTargetLookupLatency = 500 * time.Microsecond
)

// ComputeNGramHash calculates a deterministic 64-bit hash for a token position and head.
// It is a pure function of input tokens, position, head [0..15], and EOS padding token.
func ComputeNGramHash(tokens []int, pos int, head int, eosToken int) uint64 {
	// Window of 3 tokens ending at pos: [w0, w1, w2]
	w2 := eosToken
	if pos >= 0 && pos < len(tokens) {
		w2 = tokens[pos]
	}
	w1 := eosToken
	if pos-1 >= 0 && pos-1 < len(tokens) {
		w1 = tokens[pos-1]
	}
	w0 := eosToken
	if pos-2 >= 0 && pos-2 < len(tokens) {
		w0 = tokens[pos-2]
	}

	// 64-bit FNV-1a prime mixing with SplitMix64 finalizer
	const (
		offset64 uint64 = 14695981039346656037
		prime64  uint64 = 1099511628211
	)
	h := offset64
	h ^= uint64(uint32(head))
	h *= prime64
	h ^= uint64(uint32(w0))
	h *= prime64
	h ^= uint64(uint32(w1))
	h *= prime64
	h ^= uint64(uint32(w2))
	h *= prime64

	// SplitMix64 mixing
	h ^= h >> 33
	h *= 0xff51afd7ed558ccd
	h ^= h >> 33
	h *= 0xc4ceb9fe1a85ec53
	h ^= h >> 33
	return h
}

// ComputeTokenPLERowIndices computes the 16 deterministic PLE table row indices
// for a given token position in tokens.
func ComputeTokenPLERowIndices(tokens []int, pos int, totalRows int64, eosToken int) [PLERowsPerToken]int64 {
	var rowIndices [PLERowsPerToken]int64
	if totalRows <= 0 {
		return rowIndices
	}
	for head := 0; head < PLERowsPerToken; head++ {
		h := ComputeNGramHash(tokens, pos, head, eosToken)
		rowIndices[head] = int64(h % uint64(totalRows))
	}
	return rowIndices
}

// PinnedStagingBuffer holds a fixed-shape 64KB pinned host memory buffer
// divided into 16 rows of 4KB. In accordance with zero-RAM-cache invariants,
// this buffer is allocated once per gatherer and reused across all lookups.
type PinnedStagingBuffer struct {
	mem      PinnedMemory
	rows     int
	rowBytes int
	raw      []byte
}

// AllocPinnedStagingBuffer allocates a 64KB page-aligned pinned staging buffer.
func AllocPinnedStagingBuffer() (*PinnedStagingBuffer, error) {
	mem, err := AllocPinnedHostMemory(PLEBatchBytes)
	if err != nil {
		return nil, fmt.Errorf("ple_stream: alloc pinned buffer: %w", err)
	}
	raw := mem.Bytes()
	if len(raw) < PLEBatchBytes {
		_ = mem.Free()
		return nil, fmt.Errorf("ple_stream: pinned buffer undersized (%d < %d)", len(raw), PLEBatchBytes)
	}
	return &PinnedStagingBuffer{
		mem:      mem,
		rows:     PLERowsPerToken,
		rowBytes: PLERowBytes,
		raw:      raw[:PLEBatchBytes],
	}, nil
}

// Row returns the 4KB slice corresponding to head index [0..15].
func (b *PinnedStagingBuffer) Row(head int) []byte {
	if head < 0 || head >= b.rows {
		panic(fmt.Sprintf("ple_stream: head %d out of bounds [0..%d)", head, b.rows))
	}
	start := head * b.rowBytes
	return b.raw[start : start+b.rowBytes]
}

// Bytes returns the full 64KB pinned buffer slice.
func (b *PinnedStagingBuffer) Bytes() []byte {
	return b.raw
}

// Rows returns the row capacity (16).
func (b *PinnedStagingBuffer) Rows() int {
	return b.rows
}

// RowBytes returns the bytes per row (4096).
func (b *PinnedStagingBuffer) RowBytes() int {
	return b.rowBytes
}

// Free releases the pinned host memory.
func (b *PinnedStagingBuffer) Free() error {
	if b.mem != nil {
		err := b.mem.Free()
		b.mem = nil
		b.raw = nil
		return err
	}
	return nil
}

// SyncMode determines how CUDA graph synchronization is handled.
type SyncMode string

const (
	// SyncModeStreamMemop synchronizes via cuStreamWaitValue64 / cuStreamWriteValue64.
	SyncModeStreamMemop SyncMode = "stream_memop"
	// SyncModeHostDispatch fails closed to host-side launch dispatch synchronization (blocking
	// kernel dispatch until D2H readback completion).
	SyncModeHostDispatch SyncMode = "host_dispatch"
)

// MemopSyncStats records observability telemetry for CUDA stream memop synchronization.
type MemopSyncStats struct {
	Mode              SyncMode `json:"mode"`
	DriverHasMemops   bool     `json:"driver_has_memops"`
	StreamWaitCalls   uint64   `json:"stream_wait_calls"`
	StreamWriteCalls  uint64   `json:"stream_write_calls"`
	HostDispatchWaits uint64   `json:"host_dispatch_waits"`
	LastSignalValue   uint64   `json:"last_signal_value"`
}

// CUDAMemopSync coordinates CUDA stream memop graph replay synchronization.
// When the driver lacks 64-bit stream memop capabilities, it fails closed to
// host-side dispatch barrier (host D2H readback completion before kernel dispatch).
type CUDAMemopSync struct {
	mode              SyncMode
	driverHasMemops   bool
	signalAddress     uintptr
	signalValue       uint64
	streamWaitCalls   uint64
	streamWriteCalls  uint64
	hostDispatchWaits uint64
	mu                sync.Mutex
}

// NewCUDAMemopSync constructs a CUDAMemopSync instance.
// If driverHasMemops is false, mode is forced to SyncModeHostDispatch (fail-closed).
func NewCUDAMemopSync(preferredMode SyncMode, driverHasMemops bool, signalAddress uintptr) *CUDAMemopSync {
	mode := preferredMode
	if !driverHasMemops || mode != SyncModeStreamMemop {
		mode = SyncModeHostDispatch
	}
	return &CUDAMemopSync{
		mode:            mode,
		driverHasMemops: driverHasMemops,
		signalAddress:   signalAddress,
	}
}

// Mode returns the active synchronization mode.
func (s *CUDAMemopSync) Mode() SyncMode {
	return s.mode
}

// DriverHasMemops reports whether the driver has 64-bit stream memop capability.
func (s *CUDAMemopSync) DriverHasMemops() bool {
	return s.driverHasMemops
}

// SignalAddress returns the device or host-pinned signal pointer.
func (s *CUDAMemopSync) SignalAddress() uintptr {
	return s.signalAddress
}

// SignalTransferComplete writes the sequence value to the memop signal address or opens
// the host dispatch barrier, signaling to the CUDA stream that the 16-row gather is complete.
func (s *CUDAMemopSync) SignalTransferComplete(seq uint64) {
	atomic.StoreUint64(&s.signalValue, seq)
	if s.mode == SyncModeStreamMemop {
		atomic.AddUint64(&s.streamWriteCalls, 1)
	}
}

// WaitStream emits a stream wait operation (cuStreamWaitValue64 in stream_memop mode)
// or blocks the host thread until signalValue >= seq in host_dispatch mode.
func (s *CUDAMemopSync) WaitStream(stream uintptr, seq uint64) error {
	if s.mode == SyncModeStreamMemop {
		atomic.AddUint64(&s.streamWaitCalls, 1)
		return nil
	}

	// Host-dispatch: host blocks until the transfer has completed before kernel dispatch
	atomic.AddUint64(&s.hostDispatchWaits, 1)
	for {
		if atomic.LoadUint64(&s.signalValue) >= seq {
			return nil
		}
		// Yield/pause briefly while waiting for host I/O completion
		time.Sleep(10 * time.Nanosecond)
	}
}

// Stats returns a snapshot of synchronization observability metrics.
func (s *CUDAMemopSync) Stats() MemopSyncStats {
	return MemopSyncStats{
		Mode:              s.mode,
		DriverHasMemops:   s.driverHasMemops,
		StreamWaitCalls:   atomic.LoadUint64(&s.streamWaitCalls),
		StreamWriteCalls:  atomic.LoadUint64(&s.streamWriteCalls),
		HostDispatchWaits: atomic.LoadUint64(&s.hostDispatchWaits),
		LastSignalValue:   atomic.LoadUint64(&s.signalValue),
	}
}

// ShardRowExtent maps a global PLE table row to its exact location in a Safetensors file.
type ShardRowExtent struct {
	ShardPath      string `json:"shard_path"`
	FileOffset     int64  `json:"file_offset"`
	ByteLength     int64  `json:"byte_length"`
	ShardRowIndex  int64  `json:"shard_row_index"`
	GlobalRowIndex int64  `json:"global_row_index"`
}

// PLEExtentIndex indexes row locations across one or more Safetensors shard files.
type PLEExtentIndex struct {
	mu         sync.RWMutex
	shards     []string
	rowExtents []ShardRowExtent
	totalRows  int64
}

// NewPLEExtentIndex creates an empty PLE extent index.
func NewPLEExtentIndex() *PLEExtentIndex {
	return &PLEExtentIndex{
		shards:     make([]string, 0, 8),
		rowExtents: make([]ShardRowExtent, 0, 1024),
	}
}

type safetensorsTensorHeader struct {
	Dtype       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// AddSafetensorsShard parses a Safetensors shard header, identifies the PLE embedding tensor,
// and registers extents for all contained rows.
func (idx *PLEExtentIndex) AddSafetensorsShard(path string, tensorName string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("ple_extent_index: open shard %s: %w", path, err)
	}
	defer f.Close()

	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return fmt.Errorf("ple_extent_index: read header len %s: %w", path, err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen == 0 || headerLen > 100<<20 {
		return fmt.Errorf("ple_extent_index: invalid header len %d in %s", headerLen, path)
	}

	headerJSON := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerJSON); err != nil {
		return fmt.Errorf("ple_extent_index: read header json %s: %w", path, err)
	}

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &parsed); err != nil {
		return fmt.Errorf("ple_extent_index: parse header json %s: %w", path, err)
	}

	rawTensor, ok := parsed[tensorName]
	if !ok {
		// Try finding tensor by suffix or prefix match
		for k, v := range parsed {
			if k == tensorName || k == "ngram_embedding" || (len(k) > 15 && k[:15] == "ngram_embedding") {
				rawTensor = v
				tensorName = k
				ok = true
				break
			}
		}
		if !ok {
			return fmt.Errorf("ple_extent_index: tensor %q not found in %s", tensorName, path)
		}
	}

	var th safetensorsTensorHeader
	if err := json.Unmarshal(rawTensor, &th); err != nil {
		return fmt.Errorf("ple_extent_index: parse tensor %q metadata: %w", tensorName, err)
	}

	if len(th.Shape) < 1 {
		return fmt.Errorf("ple_extent_index: tensor %q invalid shape: %v", tensorName, th.Shape)
	}
	if len(th.DataOffsets) != 2 || th.DataOffsets[1] <= th.DataOffsets[0] {
		return fmt.Errorf("ple_extent_index: tensor %q invalid data_offsets: %v", tensorName, th.DataOffsets)
	}

	numRows := th.Shape[0]
	tensorBytes := th.DataOffsets[1] - th.DataOffsets[0]
	rowBytes := tensorBytes / numRows
	if rowBytes != PLERowBytes {
		return fmt.Errorf("ple_extent_index: tensor %q row width %d != expected %d", tensorName, rowBytes, PLERowBytes)
	}

	dataBaseOffset := int64(8) + int64(headerLen) + th.DataOffsets[0]

	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.shards = append(idx.shards, path)
	startGlobal := idx.totalRows
	for r := int64(0); r < numRows; r++ {
		extent := ShardRowExtent{
			ShardPath:      path,
			FileOffset:     dataBaseOffset + r*PLERowBytes,
			ByteLength:     PLERowBytes,
			ShardRowIndex:  r,
			GlobalRowIndex: startGlobal + r,
		}
		idx.rowExtents = append(idx.rowExtents, extent)
	}
	idx.totalRows += numRows
	return nil
}

// LookupRow returns the shard row extent for rowIndex.
func (idx *PLEExtentIndex) LookupRow(rowIndex int64) (ShardRowExtent, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if rowIndex < 0 || rowIndex >= int64(len(idx.rowExtents)) {
		return ShardRowExtent{}, fmt.Errorf("ple_extent_index: row index %d out of bounds [0..%d)", rowIndex, len(idx.rowExtents))
	}
	return idx.rowExtents[rowIndex], nil
}

// TotalRows returns the total count of indexed rows across all shards.
func (idx *PLEExtentIndex) TotalRows() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.totalRows
}

type rowReadTask struct {
	head   int
	rowIdx int64
}

// PLERowGatherer implements the 16-row Direct I/O safetensors gather engine.
// It opens safetensors shard files without OS page caching (O_DIRECT / FILE_FLAG_NO_BUFFERING),
// batches and dedupes 16 row reads into pinned host staging memory, and synchronizes with
// the CUDA stream memop barrier.
type PLERowGatherer struct {
	mu            sync.Mutex
	handlesMu     sync.RWMutex
	extentIndex   *PLEExtentIndex
	stagingBuffer *PinnedStagingBuffer
	syncBarrier   *CUDAMemopSync
	handles       map[string]DirectIOHandle
	scratchBounce []byte

	// Persistent worker pool for zero-allocation parallel Direct I/O
	tasks        chan rowReadTask
	stopCh       chan struct{}
	activeWg     sync.WaitGroup
	workerWg     sync.WaitGroup
	workerErrors [PLERowsPerToken]error
}

// NewPLERowGatherer instantiates a PLERowGatherer.
func NewPLERowGatherer(extentIndex *PLEExtentIndex, syncBarrier *CUDAMemopSync) (*PLERowGatherer, error) {
	if extentIndex == nil {
		return nil, errors.New("ple_gatherer: extentIndex is required")
	}
	if syncBarrier == nil {
		return nil, errors.New("ple_gatherer: syncBarrier is required")
	}
	staging, err := AllocPinnedStagingBuffer()
	if err != nil {
		return nil, fmt.Errorf("ple_gatherer: staging buffer allocation: %w", err)
	}

	extentIndex.mu.RLock()
	shards := append([]string(nil), extentIndex.shards...)
	extentIndex.mu.RUnlock()

	handles := make(map[string]DirectIOHandle, len(shards))
	for _, shard := range shards {
		h, err := OpenDirectIO(shard)
		if err != nil {
			_ = staging.Free()
			for _, opened := range handles {
				_ = opened.Close()
			}
			return nil, fmt.Errorf("ple_gatherer: open shard direct io %s: %w", shard, err)
		}
		handles[shard] = h
	}

	// Allocate a 2-sector aligned bounce buffer for any unaligned sector edge reads
	bounce := make([]byte, SectorSize*2)

	g := &PLERowGatherer{
		extentIndex:   extentIndex,
		stagingBuffer: staging,
		syncBarrier:   syncBarrier,
		handles:       handles,
		scratchBounce: bounce,
		tasks:         make(chan rowReadTask, PLERowsPerToken),
		stopCh:        make(chan struct{}),
	}

	for w := 0; w < PLERowsPerToken; w++ {
		g.workerWg.Add(1)
		go g.worker()
	}

	return g, nil
}

func (g *PLERowGatherer) worker() {
	defer g.workerWg.Done()
	for {
		select {
		case <-g.stopCh:
			return
		case task := <-g.tasks:
			g.workerErrors[task.head] = g.readOneRow(task.head, task.rowIdx)
			g.activeWg.Done()
		}
	}
}

// StagingBuffer returns the pinned staging buffer.
func (g *PLERowGatherer) StagingBuffer() *PinnedStagingBuffer {
	return g.stagingBuffer
}

// Sync returns the active CUDA stream memop synchronizer.
func (g *PLERowGatherer) Sync() *CUDAMemopSync {
	return g.syncBarrier
}

func (g *PLERowGatherer) getHandle(path string) (DirectIOHandle, error) {
	g.handlesMu.RLock()
	h, ok := g.handles[path]
	g.handlesMu.RUnlock()
	if ok {
		return h, nil
	}

	g.handlesMu.Lock()
	defer g.handlesMu.Unlock()
	if h, ok := g.handles[path]; ok {
		return h, nil
	}
	h, err := OpenDirectIO(path)
	if err != nil {
		return nil, fmt.Errorf("ple_gatherer: open direct io %s: %w", path, err)
	}
	g.handles[path] = h
	return h, nil
}

// GatherRows gathers the 16 requested rows from disk into the pinned staging buffer.
// Repeated row indices within the 16 positions are deduped to eliminate redundant disk I/O.
// Unique row reads are executed with parallel Direct I/O to maximize NVMe queue depth,
// after which it signals the CUDA stream memop barrier with seq.
func (g *PLERowGatherer) GatherRows(rowIndices [PLERowsPerToken]int64, seq uint64) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Stack-allocated tracking for deduping without heap allocations
	var seenIndices [PLERowsPerToken]int64
	var seenHeads [PLERowsPerToken]int
	numSeen := 0

	var toRead [PLERowsPerToken]rowReadTask
	numToRead := 0
	var duplicates [PLERowsPerToken]struct{ head, sourceHead int }
	numDuplicates := 0

	for head := 0; head < PLERowsPerToken; head++ {
		rowIdx := rowIndices[head]
		foundHead := -1
		for i := 0; i < numSeen; i++ {
			if seenIndices[i] == rowIdx {
				foundHead = seenHeads[i]
				break
			}
		}
		if foundHead >= 0 {
			duplicates[numDuplicates] = struct{ head, sourceHead int }{head: head, sourceHead: foundHead}
			numDuplicates++
		} else {
			seenIndices[numSeen] = rowIdx
			seenHeads[numSeen] = head
			numSeen++
			toRead[numToRead] = rowReadTask{head: head, rowIdx: rowIdx}
			numToRead++
		}
	}

	// Execute unique row reads with persistent parallel Direct I/O workers
	if numToRead == 1 {
		op := toRead[0]
		if err := g.readOneRow(op.head, op.rowIdx); err != nil {
			return err
		}
	} else if numToRead > 1 {
		for i := 0; i < PLERowsPerToken; i++ {
			g.workerErrors[i] = nil
		}
		g.activeWg.Add(numToRead)
		for i := 0; i < numToRead; i++ {
			g.tasks <- toRead[i]
		}
		g.activeWg.Wait()
		for i := 0; i < numToRead; i++ {
			if err := g.workerErrors[toRead[i].head]; err != nil {
				return err
			}
		}
	}

	// Apply deduplicated copies
	for i := 0; i < numDuplicates; i++ {
		d := duplicates[i]
		copy(g.stagingBuffer.Row(d.head), g.stagingBuffer.Row(d.sourceHead))
	}

	// Signal transfer completion to the CUDA stream memop synchronizer
	g.syncBarrier.SignalTransferComplete(seq)
	return nil
}

func (g *PLERowGatherer) readOneRow(head int, rowIdx int64) error {
	extent, err := g.extentIndex.LookupRow(rowIdx)
	if err != nil {
		return fmt.Errorf("ple_gatherer: lookup row %d: %w", rowIdx, err)
	}

	h, err := g.getHandle(extent.ShardPath)
	if err != nil {
		return err
	}

	headDest := g.stagingBuffer.Row(head)

	// Direct I/O requires file offsets to be aligned to SectorSize (4096)
	if extent.FileOffset%SectorSize == 0 {
		// Perfectly aligned: zero-copy direct read into pinned host memory
		_, err = h.ReadAtAligned(headDest, extent.FileOffset)
		if err != nil {
			return fmt.Errorf("ple_gatherer: direct read head %d (row %d at %d): %w", head, rowIdx, extent.FileOffset, err)
		}
		return nil
	}

	// Unaligned safetensors header offset: align down, read sector, and copy row
	alignedOffset := AlignDown(extent.FileOffset, SectorSize)
	offsetWithinSector := int(extent.FileOffset - alignedOffset)
	neededLen := int(AlignUp(int64(offsetWithinSector+PLERowBytes), SectorSize))

	bounce := make([]byte, neededLen)
	_, err = h.ReadAtAligned(bounce, alignedOffset)
	if err != nil {
		return fmt.Errorf("ple_gatherer: unaligned direct read head %d: %w", head, err)
	}
	copy(headDest, bounce[offsetWithinSector:offsetWithinSector+PLERowBytes])
	return nil
}

// GatherToken extracts the 16 row indices from tokens at position pos and gathers them.
func (g *PLERowGatherer) GatherToken(tokens []int, pos int, seq uint64, eosToken int) error {
	totalRows := g.extentIndex.TotalRows()
	if totalRows == 0 {
		return errors.New("ple_gatherer: extent index has 0 rows")
	}
	indices := ComputeTokenPLERowIndices(tokens, pos, totalRows, eosToken)
	return g.GatherRows(indices, seq)
}

// VerifyAgainstGold compares each gathered row in the staging buffer against the
// corresponding 4KB slice in goldTable (an in-memory byte slice of totalRows * 4096 bytes).
func (g *PLERowGatherer) VerifyAgainstGold(goldTable []byte, rowIndices [PLERowsPerToken]int64) (bool, error) {
	for head := 0; head < PLERowsPerToken; head++ {
		rowIdx := rowIndices[head]
		start := rowIdx * PLERowBytes
		end := start + PLERowBytes
		if end > int64(len(goldTable)) {
			return false, fmt.Errorf("ple_gatherer: gold table too short for row %d (need %d, have %d)", rowIdx, end, len(goldTable))
		}
		got := g.stagingBuffer.Row(head)
		want := goldTable[start:end]
		if !bytes.Equal(got, want) {
			return false, fmt.Errorf("ple_gatherer: head %d (row %d) mismatch against gold tensor", head, rowIdx)
		}
	}
	return true, nil
}

// Close closes all open file handles, halts the worker pool, and frees the pinned staging buffer.
func (g *PLERowGatherer) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	select {
	case <-g.stopCh:
		// already closed
	default:
		close(g.stopCh)
		g.workerWg.Wait()
	}

	var firstErr error
	for path, h := range g.handles {
		if err := h.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("ple_gatherer: close %s: %w", path, err)
		}
	}
	g.handles = make(map[string]DirectIOHandle)

	if err := g.stagingBuffer.Free(); err != nil && firstErr == nil {
		firstErr = fmt.Errorf("ple_gatherer: free staging buffer: %w", err)
	}
	return firstErr
}

// WriteSyntheticSafetensorsShard generates a conforming Safetensors shard file
// containing numRows of 4KB rows. The JSON header is padded with spaces so that
// the binary tensor payload begins at an exact 4096-byte sector boundary, enabling
// optimal zero-bounce unbuffered Direct I/O.
func WriteSyntheticSafetensorsShard(path string, tensorName string, numRows int, rowData []byte) error {
	expectedBytes := numRows * PLERowBytes
	if len(rowData) < expectedBytes {
		return fmt.Errorf("synthetic_shard: rowData length %d < expected %d", len(rowData), expectedBytes)
	}

	headerMap := map[string]safetensorsTensorHeader{
		tensorName: {
			Dtype:       "BF16",
			Shape:       []int64{int64(numRows), int64(PLERowBytes / 2)},
			DataOffsets: []int64{0, int64(expectedBytes)},
		},
	}
	rawJSON, err := json.Marshal(headerMap)
	if err != nil {
		return fmt.Errorf("synthetic_shard: marshal header: %w", err)
	}

	// Pad header to ensure (8 + headerLen) is a multiple of SectorSize (4096)
	totalHeaderPadded := int(AlignUp(int64(8+len(rawJSON)), SectorSize))
	paddingNeeded := totalHeaderPadded - (8 + len(rawJSON))
	paddedJSON := make([]byte, len(rawJSON)+paddingNeeded)
	copy(paddedJSON, rawJSON)
	for i := len(rawJSON); i < len(paddedJSON); i++ {
		paddedJSON[i] = ' '
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("synthetic_shard: create %s: %w", path, err)
	}
	defer f.Close()

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(paddedJSON)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		return fmt.Errorf("synthetic_shard: write header len: %w", err)
	}
	if _, err := f.Write(paddedJSON); err != nil {
		return fmt.Errorf("synthetic_shard: write header json: %w", err)
	}
	if _, err := f.Write(rowData[:expectedBytes]); err != nil {
		return fmt.Errorf("synthetic_shard: write row data: %w", err)
	}
	return f.Sync()
}
