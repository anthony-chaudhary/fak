// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// StreamCopyChunkSize defines the 64-byte non-temporal vector chunk size matching Zen 5 CPU
// cache line width and RDNA 3.5 unified memory fabric access granularity on AMD Strix Halo.
const StreamCopyChunkSize = 64

// Stream copy error sentinels.
var (
	// ErrDestinationTooSmall indicates the destination buffer has fewer bytes than the source.
	ErrDestinationTooSmall = errors.New("strix/streamcopy: destination buffer smaller than source")

	// ErrNilBuffer indicates a nil buffer slice was passed for a non-zero copy operation.
	ErrNilBuffer = errors.New("strix/streamcopy: source or destination buffer is nil")
)

var (
	streamAVX512Mu            sync.RWMutex
	streamAVX512Override      *bool
	globalFenceCounter        uint64
	cachedAVX512Streaming     bool
	cachedAVX512StreamingInit bool
)

func evaluateAVX512Streaming() bool {
	if os.Getenv("FAK_AVX512_DISABLE") != "" || os.Getenv("FAK_AVX512_STREAMING_DISABLE") != "" {
		return false
	}
	if os.Getenv("FAK_AVX512_FORCE") != "" || os.Getenv("FAK_AVX512_STREAMING_FORCE") != "" {
		return true
	}
	return hardwareHasAVX512()
}

// ResetAVX512StreamingCapabilityForTesting invalidates the cached CPU capability so
// updated environment variables are re-read in tests.
func ResetAVX512StreamingCapabilityForTesting() {
	streamAVX512Mu.Lock()
	defer streamAVX512Mu.Unlock()
	cachedAVX512StreamingInit = false
}

// SetAVX512StreamingOverrideForTesting sets or clears a programmatic override for HasAVX512Streaming.
// Passing nil clears the override. Thread-safe across concurrent goroutines.
func SetAVX512StreamingOverrideForTesting(override *bool) {
	streamAVX512Mu.Lock()
	defer streamAVX512Mu.Unlock()
	cachedAVX512StreamingInit = false
	if override == nil {
		streamAVX512Override = nil
	} else {
		val := *override
		streamAVX512Override = &val
	}
}

// HasAVX512Streaming checks whether AVX-512 non-temporal streaming copy instructions
// (vmovntdqa streaming load and vmovntdq streaming store) are supported on the current CPU.
//
// Priority order:
//  1. Programmatic test override set via SetAVX512StreamingOverrideForTesting.
//  2. Environment variables FAK_AVX512_DISABLE or FAK_AVX512_STREAMING_DISABLE (returns false).
//  3. Environment variables FAK_AVX512_FORCE or FAK_AVX512_STREAMING_FORCE (returns true).
//  4. Physical hardware detection on amd64 via CPUID / cpu.X86.HasAVX512F.
func HasAVX512Streaming() bool {
	streamAVX512Mu.RLock()
	if streamAVX512Override != nil {
		val := *streamAVX512Override
		streamAVX512Mu.RUnlock()
		return val
	}
	if cachedAVX512StreamingInit {
		val := cachedAVX512Streaming
		streamAVX512Mu.RUnlock()
		return val
	}
	streamAVX512Mu.RUnlock()

	streamAVX512Mu.Lock()
	defer streamAVX512Mu.Unlock()
	if streamAVX512Override != nil {
		return *streamAVX512Override
	}
	if !cachedAVX512StreamingInit {
		cachedAVX512Streaming = evaluateAVX512Streaming()
		cachedAVX512StreamingInit = true
	}
	return cachedAVX512Streaming
}

// MemoryStoreFence executes an explicit store memory barrier (SFENCE on amd64).
// In AMD Strix Halo UMA, this ensures all non-temporal streaming writes to write-combining (WC)
// memory or host DRAM are globally visible before subsequent CPU reads or GPU AQL dispatch.
func MemoryStoreFence() {
	sfenceAsm()
	atomic.AddUint64(&globalFenceCounter, 1)
}

// StreamCopyTelemetry captures performance and operational diagnostics for a stream copy operation.
type StreamCopyTelemetry struct {
	BytesCopied      int64         `json:"bytes_copied"`
	ChunksProcessed  int64         `json:"chunks_processed"`
	NonTemporalLoops int64         `json:"non_temporal_loops"`
	HeadBytes        int64         `json:"head_bytes"`
	TailBytes        int64         `json:"tail_bytes"`
	Duration         time.Duration `json:"duration_ns"`
	ThroughputMBs    float64       `json:"throughput_mbs"`
	UsedAVX512       bool          `json:"used_avx512"`
	SourceWC         bool          `json:"source_wc"`
	DestWC           bool          `json:"dest_wc"`
}

type streamCopyConfig struct {
	sourceWC      bool
	destWC        bool
	forceFallback bool
	storeFence    bool
	telemetry     *StreamCopyTelemetry
}

// StreamCopyOption configures non-temporal stream copy execution.
type StreamCopyOption func(*streamCopyConfig)

// WithSourceWC declares that the source buffer resides in Write-Combining (WC) memory
// (e.g. GPU GTT allocation mapped uncached). When enabled, 64-byte streaming loads (vmovntdqa)
// are prioritized to coalesce memory reads and prevent read throughput collapse (~200 MB/s -> wire rate).
func WithSourceWC(wc bool) StreamCopyOption {
	return func(c *streamCopyConfig) {
		c.sourceWC = wc
	}
}

// SourceWC is an alias for WithSourceWC.
func SourceWC(wc bool) StreamCopyOption {
	return WithSourceWC(wc)
}

// WithDestWC declares that the destination buffer resides in Write-Combining (WC) memory
// (e.g. GPU framebuffer / VRAM / GTT staging area). When enabled, 64-byte streaming stores (vmovntdq)
// bypass CPU cache pollution and write directly to physical memory.
func WithDestWC(wc bool) StreamCopyOption {
	return func(c *streamCopyConfig) {
		c.destWC = wc
	}
}

// DestWC is an alias for WithDestWC.
func DestWC(wc bool) StreamCopyOption {
	return WithDestWC(wc)
}

// WithForceFallback forces execution of the optimized Go fallback loop even when AVX-512
// streaming instructions are supported on the host CPU.
func WithForceFallback(force bool) StreamCopyOption {
	return func(c *streamCopyConfig) {
		c.forceFallback = force
	}
}

// WithStoreFence controls whether an SFENCE store memory barrier is issued upon completion.
// Default is true.
func WithStoreFence(fence bool) StreamCopyOption {
	return func(c *streamCopyConfig) {
		c.storeFence = fence
	}
}

// WithTelemetry attaches a telemetry collector to receive performance counters.
func WithTelemetry(t *StreamCopyTelemetry) StreamCopyOption {
	return func(c *streamCopyConfig) {
		c.telemetry = t
	}
}

// StreamCopy performs a non-temporal streaming copy from src to dst.
// On AMD Strix Halo APUs, memory allocated on the GPU via KFD or hipMalloc is mapped into Zen 5
// CPU page tables as Write-Combining (WC) Uncached. Standard CPU memcpy loads from WC memory
// bypass CPU caches and issue uncoalesced 64-byte reads, causing read throughput to collapse to ~200 MB/s.
// StreamCopy utilizes 64-byte aligned non-temporal streaming instructions (vmovntdqa streaming load
// and vmovntdq streaming store) to coalesce reads and stores, bypass cache pollution, and restore
// full memory interconnect bandwidth.
//
// Unaligned boundaries are handled with standard byte copies for head and tail bytes.
// If AVX-512 streaming is not supported or fallback is requested, an optimized chunked Go fallback
// is executed.
func StreamCopy(dst, src []byte, opts ...StreamCopyOption) (int64, error) {
	n := len(src)
	if n == 0 {
		return 0, nil
	}
	if len(dst) < n {
		return 0, ErrDestinationTooSmall
	}

	cfg := streamCopyConfig{
		sourceWC:      true,
		destWC:        true,
		forceFallback: false,
		storeFence:    true,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	var start time.Time
	if cfg.telemetry != nil {
		start = time.Now()
	}

	dstAddr := uintptr(unsafe.Pointer(&dst[0]))
	srcAddr := uintptr(unsafe.Pointer(&src[0]))

	// Handle identical buffers
	if dstAddr == srcAddr {
		return int64(n), nil
	}

	// Check for destructive forward-copy overlap (dst inside src range after start)
	if dstAddr > srcAddr && dstAddr < srcAddr+uintptr(n) {
		copy(dst[:n], src[:n])
		if cfg.storeFence {
			MemoryStoreFence()
		}
		if cfg.telemetry != nil {
			duration := time.Since(start)
			*cfg.telemetry = StreamCopyTelemetry{
				BytesCopied:      int64(n),
				ChunksProcessed:  int64(n / StreamCopyChunkSize),
				NonTemporalLoops: 0,
				HeadBytes:        int64(n),
				TailBytes:        0,
				Duration:         duration,
				ThroughputMBs:    calcThroughput(int64(n), duration),
				UsedAVX512:       false,
				SourceWC:         cfg.sourceWC,
				DestWC:           cfg.destWC,
			}
		}
		return int64(n), nil
	}

	// AVX-512 execution requires both capability flag and physical hardware capability.
	useAVX512 := HasAVX512Streaming() && !cfg.forceFallback && canExecuteAVX512()

	var (
		headBytes        int
		chunks           int
		tailBytes        int
		nonTemporalLoops int64
	)

	dstOffset := int(dstAddr % StreamCopyChunkSize)
	srcOffset := int(srcAddr % StreamCopyChunkSize)

	if dstOffset == srcOffset {
		// Slices have identical 64-byte phase; align BOTH to 64 bytes via head bytes.
		if dstOffset != 0 {
			headBytes = StreamCopyChunkSize - dstOffset
		}

		if n < headBytes+StreamCopyChunkSize {
			copy(dst[:n], src[:n])
			headBytes = n
			chunks = 0
			tailBytes = 0
		} else {
			if headBytes > 0 {
				copy(dst[:headBytes], src[:headBytes])
			}
			rem := n - headBytes
			chunks = rem / StreamCopyChunkSize
			tailBytes = rem % StreamCopyChunkSize
			chunkBytes := chunks * StreamCopyChunkSize

			dstChunkPtr := unsafe.Pointer(&dst[headBytes])
			srcChunkPtr := unsafe.Pointer(&src[headBytes])

			if useAVX512 && chunks > 0 {
				streamCopyAVX512Kernel(dstChunkPtr, srcChunkPtr, int64(chunks))
				nonTemporalLoops = int64(chunks)
			} else if chunks > 0 {
				optimizedChunkCopy(dst[headBytes:headBytes+chunkBytes], src[headBytes:headBytes+chunkBytes], chunks)
			}

			if tailBytes > 0 {
				tailStart := headBytes + chunkBytes
				copy(dst[tailStart:n], src[tailStart:n])
			}
		}
	} else {
		// Differing 64-byte alignments.
		// If DestWC is enabled, align destination for vmovntdq streaming store.
		if cfg.destWC || (!cfg.sourceWC && !cfg.destWC) {
			if dstOffset != 0 {
				headBytes = StreamCopyChunkSize - dstOffset
			}

			if n < headBytes+StreamCopyChunkSize {
				copy(dst[:n], src[:n])
				headBytes = n
				chunks = 0
				tailBytes = 0
			} else {
				if headBytes > 0 {
					copy(dst[:headBytes], src[:headBytes])
				}
				rem := n - headBytes
				chunks = rem / StreamCopyChunkSize
				tailBytes = rem % StreamCopyChunkSize
				chunkBytes := chunks * StreamCopyChunkSize

				dstChunkPtr := unsafe.Pointer(&dst[headBytes])
				srcChunkPtr := unsafe.Pointer(&src[headBytes])

				if useAVX512 && chunks > 0 {
					streamCopyAVX512StoreOnly(dstChunkPtr, srcChunkPtr, int64(chunks))
					nonTemporalLoops = int64(chunks)
				} else if chunks > 0 {
					optimizedChunkCopy(dst[headBytes:headBytes+chunkBytes], src[headBytes:headBytes+chunkBytes], chunks)
				}

				if tailBytes > 0 {
					tailStart := headBytes + chunkBytes
					copy(dst[tailStart:n], src[tailStart:n])
				}
			}
		} else {
			// SourceWC enabled without DestWC: align source for vmovntdqa streaming load.
			if srcOffset != 0 {
				headBytes = StreamCopyChunkSize - srcOffset
			}

			if n < headBytes+StreamCopyChunkSize {
				copy(dst[:n], src[:n])
				headBytes = n
				chunks = 0
				tailBytes = 0
			} else {
				if headBytes > 0 {
					copy(dst[:headBytes], src[:headBytes])
				}
				rem := n - headBytes
				chunks = rem / StreamCopyChunkSize
				tailBytes = rem % StreamCopyChunkSize
				chunkBytes := chunks * StreamCopyChunkSize

				dstChunkPtr := unsafe.Pointer(&dst[headBytes])
				srcChunkPtr := unsafe.Pointer(&src[headBytes])

				if useAVX512 && chunks > 0 {
					streamCopyAVX512LoadOnly(dstChunkPtr, srcChunkPtr, int64(chunks))
					nonTemporalLoops = int64(chunks)
				} else if chunks > 0 {
					optimizedChunkCopy(dst[headBytes:headBytes+chunkBytes], src[headBytes:headBytes+chunkBytes], chunks)
				}

				if tailBytes > 0 {
					tailStart := headBytes + chunkBytes
					copy(dst[tailStart:n], src[tailStart:n])
				}
			}
		}
	}

	if cfg.storeFence {
		MemoryStoreFence()
	}

	if cfg.telemetry != nil {
		duration := time.Since(start)
		*cfg.telemetry = StreamCopyTelemetry{
			BytesCopied:      int64(n),
			ChunksProcessed:  int64(chunks),
			NonTemporalLoops: nonTemporalLoops,
			HeadBytes:        int64(headBytes),
			TailBytes:        int64(tailBytes),
			Duration:         duration,
			ThroughputMBs:    calcThroughput(int64(n), duration),
			UsedAVX512:       useAVX512,
			SourceWC:         cfg.sourceWC,
			DestWC:           cfg.destWC,
		}
	}

	return int64(n), nil
}

// optimizedChunkCopy executes 64-byte chunked copying in Go with 4x unrolling.
func optimizedChunkCopy(dst, src []byte, chunks int) {
	i := 0
	for ; i+3 < chunks; i += 4 {
		off := i * StreamCopyChunkSize
		copy(dst[off:off+256], src[off:off+256])
	}
	for ; i < chunks; i++ {
		off := i * StreamCopyChunkSize
		copy(dst[off:off+StreamCopyChunkSize], src[off:off+StreamCopyChunkSize])
	}
}

func calcThroughput(bytesCopied int64, d time.Duration) float64 {
	if bytesCopied <= 0 || d <= 0 {
		return 0.0
	}
	sec := d.Seconds()
	if sec <= 0 {
		return 0.0
	}
	return (float64(bytesCopied) / (1024.0 * 1024.0)) / sec
}

// -----------------------------------------------------------------------------
// Safe Integration Helpers for platform/strix/uma_mmu.go
// -----------------------------------------------------------------------------

// StreamCopyToUMA writes src into uma starting at offset using non-temporal streaming copy.
// The destination buffer is treated as Write-Combining (DestWC=true) and an atomic release
// fence is issued on completion to ensure RDNA 3.5 compute visibility.
func StreamCopyToUMA(uma *UMABuffer, offset int, src []byte, opts ...StreamCopyOption) (int64, error) {
	if uma == nil {
		return 0, ErrBufferClosed
	}
	return uma.StreamWrite(offset, src, opts...)
}

// StreamCopyFromUMA reads count bytes from uma starting at offset into dst using non-temporal streaming copy.
// The source buffer is treated as Write-Combining (SourceWC=true) to bypass CPU uncoalesced WC reads
// and restore read bandwidth from APU memory.
func StreamCopyFromUMA(dst []byte, uma *UMABuffer, offset int, count int, opts ...StreamCopyOption) (int64, error) {
	if uma == nil {
		return 0, ErrBufferClosed
	}
	return uma.StreamRead(dst, offset, count, opts...)
}

// StreamWrite writes src into the UMABuffer starting at offset using non-temporal streaming copy.
// Automatically applies WithDestWC(true) and issues MemoryFenceRelease() upon completion.
func (b *UMABuffer) StreamWrite(offset int, src []byte, opts ...StreamCopyOption) (int64, error) {
	if b == nil {
		return 0, ErrBufferClosed
	}
	b.mu.RLock()
	defer b.mu.RUnlock()

	if atomic.LoadUint32(&b.closed) != 0 {
		return 0, ErrBufferClosed
	}
	if offset < 0 || len(src) < 0 {
		return 0, ErrOutOfBounds
	}
	if offset+len(src) > b.size {
		return 0, fmt.Errorf("%w: offset %d + len %d > buffer size %d", ErrOutOfBounds, offset, len(src), b.size)
	}

	writeOpts := []StreamCopyOption{WithDestWC(true), WithStoreFence(true)}
	writeOpts = append(writeOpts, opts...)

	n, err := StreamCopy(b.slice[offset:offset+len(src)], src, writeOpts...)
	if err != nil {
		return n, err
	}

	b.MemoryFenceRelease()
	return n, nil
}

// StreamRead reads count bytes from the UMABuffer starting at offset into dst using non-temporal streaming copy.
// Automatically applies WithSourceWC(true) to avoid uncoalesced read collapse on write-combining APU memory.
func (b *UMABuffer) StreamRead(dst []byte, offset int, count int, opts ...StreamCopyOption) (int64, error) {
	if b == nil {
		return 0, ErrBufferClosed
	}
	b.mu.RLock()
	defer b.mu.RUnlock()

	if atomic.LoadUint32(&b.closed) != 0 {
		return 0, ErrBufferClosed
	}
	if offset < 0 || count < 0 {
		return 0, ErrOutOfBounds
	}
	if offset+count > b.size {
		return 0, fmt.Errorf("%w: offset %d + count %d > buffer size %d", ErrOutOfBounds, offset, count, b.size)
	}
	if len(dst) < count {
		return 0, ErrDestinationTooSmall
	}

	b.MemoryFenceAcquire()

	readOpts := []StreamCopyOption{WithSourceWC(true)}
	readOpts = append(readOpts, opts...)

	return StreamCopy(dst[:count], b.slice[offset:offset+count], readOpts...)
}
