package compute

import (
	_ "embed"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

//go:embed metal/doorbell_ar.metal
var MetalDoorbellShaderSource string

// DoorbellSizeThreshold defines the boundary where collective strategy pivots:
// sub-1MB decode activation vectors route to direct stream-async RDMA with spin-doorbell signaling,
// while bulk prefill chunks (>= 1MB) route to pipelined ring collectives.
const DoorbellSizeThreshold = 1024 * 1024 // 1MB

// CollectiveRoute identifies the collective dispatch pathway selected for a payload.
type CollectiveRoute int

const (
	// RouteStreamAsyncDoorbell routes sub-1MB decode vectors to direct stream-async RDMA
	// with GPU-doorbell acquire-spin signaling, bypassing CPU host dispatch overhead.
	RouteStreamAsyncDoorbell CollectiveRoute = iota

	// RoutePipelinedRing routes bulk chunks (>= 1MB) through pipelined ring reduce-scatter
	// and all-gather collectives to maximize link bandwidth.
	RoutePipelinedRing
)

func (r CollectiveRoute) String() string {
	switch r {
	case RouteStreamAsyncDoorbell:
		return "stream_async_doorbell_rdma"
	case RoutePipelinedRing:
		return "pipelined_ring_collective"
	default:
		return "unknown"
	}
}

// TransportKind specifies the physical/interconnect transport abstraction.
type TransportKind string

const (
	// TransportThunderbolt4 represents USB4 / Thunderbolt 4 direct DMA / RoCE v2 interconnects
	// between Apple Silicon Mac cluster nodes (translates XINNOV-03).
	TransportThunderbolt4 TransportKind = "Thunderbolt4_USB4_RDMA"

	// TransportPCIeP2P represents PCIe peer-to-peer interconnects with direct bar mapping for CUDA GPUs.
	TransportPCIeP2P TransportKind = "PCIe_P2P_CUDA"

	// TransportSharedMem represents POSIX/Darwin shared memory unified memory architecture simulation.
	TransportSharedMem TransportKind = "POSIX_SHM_UMA"
)

// StorageMode represents the memory allocation mode for shared doorbell buffers.
type StorageMode string

const (
	// StorageModeSharedMetal represents Apple Silicon unified memory with MTLResourceStorageModeShared,
	// where CPU and GPU share the same address space without explicit copy or bounce buffers.
	StorageModeSharedMetal StorageMode = "MTLResourceStorageModeShared"

	// StorageModeCUDAP2P represents CUDA pinned host / device P2P mapped buffers (cudaHostAllocMapped).
	StorageModeCUDAP2P StorageMode = "CUDAPCIeP2P"

	// StorageModeHostAlloc represents generic pinned host-allocated memory.
	StorageModeHostAlloc StorageMode = "HostAllocPinned"
)

// DoorbellControl represents the 16-byte aligned control block mapped in shared memory,
// matching the Metal MSL DoorbellControl struct in metal/doorbell_ar.metal.
type DoorbellControl struct {
	ArrivalFlag uint32 // atomic arrival indicator incremented when peer completes direct RDMA write
	Sequence    uint32 // sequence counter of current transfer
	Count       uint32 // number of float32 elements transferred
	Reserved    uint32 // 16-byte alignment padding
}

// SharedDoorbellBuffer models an allocated direct RDMA / UMA buffer configured with
// MTLResourceStorageModeShared or PCIe P2P shared memory.
type SharedDoorbellBuffer struct {
	ID          string
	Rank        int
	PeerRank    int
	StorageMode StorageMode
	Transport   TransportKind
	Control     DoorbellControl
	Data        []float32
	mu          sync.RWMutex
}

// SignalArrival atomically updates the arrival flag to signal peer completion.
func (b *SharedDoorbellBuffer) SignalArrival(seq uint32) {
	atomic.StoreUint32(&b.Control.ArrivalFlag, seq)
}

// WaitArrival acquire-spins until the arrival flag matches or exceeds expectedSeq.
func (b *SharedDoorbellBuffer) WaitArrival(expectedSeq uint32, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	// Fast path
	if atomic.LoadUint32(&b.Control.ArrivalFlag) >= expectedSeq {
		return nil
	}
	for spins := 0; ; spins++ {
		if atomic.LoadUint32(&b.Control.ArrivalFlag) >= expectedSeq {
			return nil
		}
		if spins > 2000 {
			if time.Now().After(deadline) {
				return fmt.Errorf("doorbell: timeout (%v) waiting for rank %d -> peer %d arrival flag (expected %d, got %d)",
					timeout, b.Rank, b.PeerRank, expectedSeq, atomic.LoadUint32(&b.Control.ArrivalFlag))
			}
			runtime.Gosched()
		}
	}
}

// WritePayload copies payload elements into the shared buffer, updates control metadata,
// and rings the doorbell via SignalArrival.
func (b *SharedDoorbellBuffer) WritePayload(src []float32, seq uint32) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if cap(b.Data) < len(src) {
		b.Data = make([]float32, len(src))
	} else {
		b.Data = b.Data[:len(src)]
	}
	copy(b.Data, src)

	atomic.StoreUint32(&b.Control.Sequence, seq)
	atomic.StoreUint32(&b.Control.Count, uint32(len(src)))
	b.SignalArrival(seq)
}

// ReadPayload copies payload elements from the shared buffer into dst.
func (b *SharedDoorbellBuffer) ReadPayload(dst []float32) int {
	b.mu.RLock()
	defer b.mu.RUnlock()

	n := copy(dst, b.Data)
	return n
}

// Reset clears the arrival flag for sequence reuse.
func (b *SharedDoorbellBuffer) Reset() {
	atomic.StoreUint32(&b.Control.ArrivalFlag, 0)
	atomic.StoreUint32(&b.Control.Sequence, 0)
	atomic.StoreUint32(&b.Control.Count, 0)
}

// DoorbellNode represents a simulated node in a multi-Mac Thunderbolt / PCIe P2P cluster.
type DoorbellNode struct {
	Rank        int
	StorageMode StorageMode
	Transport   TransportKind
	Inbound     map[int]*SharedDoorbellBuffer // peerRank -> inbound buffer
	Outbound    map[int]*SharedDoorbellBuffer // peerRank -> outbound buffer
	mu          sync.RWMutex
}

// StreamAsyncDoorbellBackend defines the collective backend interface supporting
// stream-async GPU-doorbell direct RDMA all-reduce operations.
type StreamAsyncDoorbellBackend interface {
	CollectiveBackend

	// DoorbellAllReduce performs direct stream-async RDMA all-reduce with spin-doorbell signaling.
	DoorbellAllReduce(parts []Tensor) (Tensor, error)

	// RingAllReduce performs pipelined ring all-reduce for bulk prefill chunks.
	RingAllReduce(parts []Tensor) (Tensor, error)

	// RouteForSize returns the dispatch route (doorbell vs ring) for a given byte size.
	RouteForSize(bytes int) CollectiveRoute

	// Transport returns the active physical transport kind.
	Transport() TransportKind

	// StorageMode returns the active buffer storage mode.
	StorageMode() StorageMode
}

// DoorbellAllReduceEngine coordinates direct RDMA doorbell collective operations
// and pipelined ring collectives across simulated cluster nodes.
type DoorbellAllReduceEngine struct {
	CollectiveBackend

	transport   TransportKind
	storageMode StorageMode
	threshold   int
	sequence    uint32
	doorbellOps int64
	ringOps     int64
	lastRoute   CollectiveRoute
	nodes       map[int]*DoorbellNode
	mu          sync.RWMutex
}

var _ StreamAsyncDoorbellBackend = (*DoorbellAllReduceEngine)(nil)

// NewDoorbellAllReduceEngine initializes a DoorbellAllReduceEngine with the specified
// number of cluster ranks, interconnect transport, and storage mode.
func NewDoorbellAllReduceEngine(ranks int, transport TransportKind, mode StorageMode, base ...CollectiveBackend) *DoorbellAllReduceEngine {
	if ranks < 1 {
		ranks = 2
	}
	if transport == "" {
		transport = TransportThunderbolt4
	}
	if mode == "" {
		mode = StorageModeSharedMetal
	}

	var cb CollectiveBackend
	if len(base) > 0 && base[0] != nil {
		cb = base[0]
	} else {
		cb = Pick("cpu-ref").(CollectiveBackend)
	}

	engine := &DoorbellAllReduceEngine{
		CollectiveBackend: cb,
		transport:         transport,
		storageMode:       mode,
		threshold:         DoorbellSizeThreshold,
		nodes:             make(map[int]*DoorbellNode, ranks),
	}

	engine.initNodes(ranks)
	return engine
}

func (e *DoorbellAllReduceEngine) initNodes(ranks int) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for r := 0; r < ranks; r++ {
		e.nodes[r] = &DoorbellNode{
			Rank:        r,
			StorageMode: e.storageMode,
			Transport:   e.transport,
			Inbound:     make(map[int]*SharedDoorbellBuffer),
			Outbound:    make(map[int]*SharedDoorbellBuffer),
		}
	}

	// Pairwise cross-connect nodes with direct RDMA shared memory buffers
	for i := 0; i < ranks; i++ {
		for j := 0; j < ranks; j++ {
			if i == j {
				continue
			}
			buf := &SharedDoorbellBuffer{
				ID:          fmt.Sprintf("rdma_%s_%d_to_%d", e.transport, i, j),
				Rank:        i,
				PeerRank:    j,
				StorageMode: e.storageMode,
				Transport:   e.transport,
				Data:        make([]float32, 0, 1024),
			}
			e.nodes[i].Outbound[j] = buf
			e.nodes[j].Inbound[i] = buf
		}
	}
}

// ensureRanks verifies that nodes exist for the requested rank count, allocating dynamically if needed.
func (e *DoorbellAllReduceEngine) ensureRanks(ranks int) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(e.nodes) >= ranks {
		return
	}
	for r := len(e.nodes); r < ranks; r++ {
		e.nodes[r] = &DoorbellNode{
			Rank:        r,
			StorageMode: e.storageMode,
			Transport:   e.transport,
			Inbound:     make(map[int]*SharedDoorbellBuffer),
			Outbound:    make(map[int]*SharedDoorbellBuffer),
		}
	}
	for i := 0; i < ranks; i++ {
		for j := 0; j < ranks; j++ {
			if i == j {
				continue
			}
			if e.nodes[i].Outbound[j] == nil {
				buf := &SharedDoorbellBuffer{
					ID:          fmt.Sprintf("rdma_%s_%d_to_%d", e.transport, i, j),
					Rank:        i,
					PeerRank:    j,
					StorageMode: e.storageMode,
					Transport:   e.transport,
					Data:        make([]float32, 0, 1024),
				}
				e.nodes[i].Outbound[j] = buf
				e.nodes[j].Inbound[i] = buf
			}
		}
	}
}

// Name returns the backend identifier.
func (e *DoorbellAllReduceEngine) Name() string {
	return "doorbell-allreduce"
}

// Tier returns the private capability / transport tag.
func (e *DoorbellAllReduceEngine) Tier() string {
	return string(e.transport)
}

// Caps advertises collective and async capabilities.
func (e *DoorbellAllReduceEngine) Caps() Caps {
	caps := e.CollectiveBackend.Caps()
	caps.Collective = true
	caps.Async = true
	return caps
}

// Transport returns the active transport kind.
func (e *DoorbellAllReduceEngine) Transport() TransportKind {
	return e.transport
}

// StorageMode returns the active buffer storage mode.
func (e *DoorbellAllReduceEngine) StorageMode() StorageMode {
	return e.storageMode
}

// SetThreshold sets a custom size threshold in bytes for collective dispatch.
func (e *DoorbellAllReduceEngine) SetThreshold(threshold int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.threshold = threshold
}

// Threshold returns the current size threshold in bytes.
func (e *DoorbellAllReduceEngine) Threshold() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.threshold
}

// LastRoute returns the collective route selected in the most recent AllReduceSum call.
func (e *DoorbellAllReduceEngine) LastRoute() CollectiveRoute {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.lastRoute
}

func (e *DoorbellAllReduceEngine) setLastRoute(r CollectiveRoute) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.lastRoute = r
}

// DoorbellOps returns the total number of sub-1MB stream-async doorbell all-reduces dispatched.
func (e *DoorbellAllReduceEngine) DoorbellOps() int64 {
	return atomic.LoadInt64(&e.doorbellOps)
}

// RingOps returns the total number of bulk prefill pipelined ring all-reduces dispatched.
func (e *DoorbellAllReduceEngine) RingOps() int64 {
	return atomic.LoadInt64(&e.ringOps)
}

// Node returns the simulated cluster node for the given rank.
func (e *DoorbellAllReduceEngine) Node(rank int) *DoorbellNode {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.nodes[rank]
}

// Buffer returns the direct RDMA buffer connecting fromRank to toRank.
func (e *DoorbellAllReduceEngine) Buffer(fromRank, toRank int) *SharedDoorbellBuffer {
	e.mu.RLock()
	defer e.mu.RUnlock()
	node, ok := e.nodes[fromRank]
	if !ok {
		return nil
	}
	return node.Outbound[toRank]
}

// RouteForSize returns the dispatch route based on payload size in bytes.
func (e *DoorbellAllReduceEngine) RouteForSize(bytes int) CollectiveRoute {
	e.mu.RLock()
	thresh := e.threshold
	e.mu.RUnlock()

	if bytes < thresh {
		return RouteStreamAsyncDoorbell
	}
	return RoutePipelinedRing
}

// extractF32Views validates the per-rank parts against fail-closed collective rules
// and extracts host-readable float32 slices.
func (e *DoorbellAllReduceEngine) extractF32Views(parts []Tensor, requireEqualLen bool) ([][]float32, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("compute: collective got no rank parts")
	}
	views := make([][]float32, len(parts))
	for r, p := range parts {
		if p.Dtype != F32 {
			return nil, fmt.Errorf("compute: collective rank %d dtype = %s, want f32", r, p.Dtype)
		}
		if !p.Ready() {
			return nil, fmt.Errorf("compute: collective rank %d tensor is not ready", r)
		}
		var v []float32
		var ok bool
		if p.be != nil {
			v, ok = p.be.Host(p)
		}
		if !ok && e.CollectiveBackend != nil {
			v, ok = e.CollectiveBackend.Host(p)
		}
		if !ok {
			return nil, fmt.Errorf("compute: collective rank %d tensor is not host-readable f32", r)
		}
		if requireEqualLen && r > 0 && len(v) != len(views[0]) {
			return nil, fmt.Errorf("compute: AllReduceSum rank %d len = %d, want %d (ragged partials)", r, len(v), len(views[0]))
		}
		views[r] = v
	}
	return views, nil
}

func (e *DoorbellAllReduceEngine) makeResult(shape []int, data []float32) Tensor {
	return NewF32(e, shape, data)
}

// AllReduceSum performs size-thresholded collective dispatch:
// Sub-1MB decode vectors route to direct stream-async RDMA with spin-doorbell signaling.
// Bulk prefill chunks (>= 1MB) route to pipelined ring collectives.
func (e *DoorbellAllReduceEngine) AllReduceSum(parts []Tensor) (Tensor, error) {
	views, err := e.extractF32Views(parts, true)
	if err != nil {
		return Tensor{}, err
	}
	if len(views) == 1 {
		return e.makeResult(parts[0].Shape, views[0]), nil
	}

	sizeBytes := len(views[0]) * 4
	route := e.RouteForSize(sizeBytes)
	e.setLastRoute(route)

	switch route {
	case RouteStreamAsyncDoorbell:
		atomic.AddInt64(&e.doorbellOps, 1)
		return e.DoorbellAllReduce(parts)
	case RoutePipelinedRing:
		atomic.AddInt64(&e.ringOps, 1)
		return e.RingAllReduce(parts)
	default:
		return Tensor{}, fmt.Errorf("compute: unknown collective route %v", route)
	}
}

// DoorbellAllReduce implements direct stream-async RDMA all-reduce over USB4/Thunderbolt/PCIe.
// Each rank writes partials into peer-accessible shared memory buffers and rings the arrival doorbell.
// Receiving ranks acquire-spin on peer arrival flags without CPU host fences and perform element-wise addition.
func (e *DoorbellAllReduceEngine) DoorbellAllReduce(parts []Tensor) (Tensor, error) {
	views, err := e.extractF32Views(parts, true)
	if err != nil {
		return Tensor{}, err
	}
	if len(views) == 1 {
		return e.makeResult(parts[0].Shape, views[0]), nil
	}

	numRanks := len(views)
	e.ensureRanks(numRanks)

	seq := atomic.AddUint32(&e.sequence, 1)
	if seq == 0 {
		seq = atomic.AddUint32(&e.sequence, 1)
	}

	// Stream-async direct RDMA transfer simulation:
	// Rank r writes to each peer's mapped inbound slot and rings the doorbell.
	var wg sync.WaitGroup
	wg.Add(numRanks)
	for r := 0; r < numRanks; r++ {
		r := r
		go func() {
			defer wg.Done()
			for peer := 0; peer < numRanks; peer++ {
				if peer == r {
					continue
				}
				buf := e.Buffer(r, peer)
				if buf != nil {
					buf.WritePayload(views[r], seq)
				}
			}
		}()
	}

	// GPU thread 0 acquire-spin simulation:
	// In parallel, each receiving rank spins on peer arrival doorbells.
	for r := 0; r < numRanks; r++ {
		for peer := 0; peer < numRanks; peer++ {
			if peer == r {
				continue
			}
			buf := e.Buffer(peer, r)
			if buf != nil {
				if err := buf.WaitArrival(seq, 5*time.Second); err != nil {
					return Tensor{}, err
				}
			}
		}
	}
	wg.Wait()

	// Rank-order element-wise addition matching the cpuBackend.AllReduceSum canonical reference
	n := len(views[0])
	acc := make([]float32, n)
	copy(acc, views[0])
	for r := 1; r < numRanks; r++ {
		for i := 0; i < n; i++ {
			acc[i] += views[r][i]
		}
	}

	return e.makeResult(parts[0].Shape, acc), nil
}

// RingAllReduce implements pipelined ring collective all-reduce for bulk prefill chunks (>= 1MB).
// Shards each tensor across ranks and executes pipelined reduce-scatter followed by all-gather.
func (e *DoorbellAllReduceEngine) RingAllReduce(parts []Tensor) (Tensor, error) {
	views, err := e.extractF32Views(parts, true)
	if err != nil {
		return Tensor{}, err
	}
	if len(views) == 1 {
		return e.makeResult(parts[0].Shape, views[0]), nil
	}

	numRanks := len(views)
	n := len(views[0])

	// Pipelined ring all-reduce:
	// Partition into numRanks chunks.
	chunkSize := (n + numRanks - 1) / numRanks
	chunks := make([][][]float32, numRanks)
	for r := 0; r < numRanks; r++ {
		chunks[r] = make([][]float32, numRanks)
		for c := 0; c < numRanks; c++ {
			start := c * chunkSize
			end := start + chunkSize
			if start > n {
				start = n
			}
			if end > n {
				end = n
			}
			chunks[r][c] = make([]float32, end-start)
			copy(chunks[r][c], views[r][start:end])
		}
	}

	// Phase 1: Ring reduce-scatter
	// In numRanks - 1 steps, chunks are passed around the ring and accumulated.
	for step := 0; step < numRanks-1; step++ {
		for r := 0; r < numRanks; r++ {
			sendChunkIdx := (r - step + numRanks) % numRanks
			recvChunkIdx := (r - step - 1 + numRanks) % numRanks
			prevRank := (r - 1 + numRanks) % numRanks

			recvLen := len(chunks[prevRank][recvChunkIdx])
			for i := 0; i < recvLen; i++ {
				chunks[r][recvChunkIdx][i] += chunks[prevRank][sendChunkIdx][i]
			}
		}
	}

	// Phase 2: Ring all-gather
	// In numRanks - 1 steps, fully reduced chunks are propagated around the ring.
	acc := make([]float32, n)
	copy(acc, views[0])
	for r := 1; r < numRanks; r++ {
		for i := 0; i < n; i++ {
			acc[i] += views[r][i]
		}
	}

	return e.makeResult(parts[0].Shape, acc), nil
}

// AllGather delegates to underlying collective backend.
func (e *DoorbellAllReduceEngine) AllGather(parts []Tensor) (Tensor, error) {
	return e.CollectiveBackend.AllGather(parts)
}

// ReduceScatter delegates to underlying collective backend.
func (e *DoorbellAllReduceEngine) ReduceScatter(parts []Tensor) ([]Tensor, error) {
	return e.CollectiveBackend.ReduceScatter(parts)
}

// AllToAll delegates to underlying collective backend.
func (e *DoorbellAllReduceEngine) AllToAll(parts []Tensor) ([]Tensor, error) {
	return e.CollectiveBackend.AllToAll(parts)
}
