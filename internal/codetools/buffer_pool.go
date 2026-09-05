package codetools

import (
	"errors"
	"io"
	"sync"
	"sync/atomic"
)

// buffer_pool.go — size-classed scratch buffer arenas for zero-subprocess tool execution.
//
// Classical code tools (Grep, Glob, Read, Edit) run entirely in-process rather than spawning
// external tools (rg, grep, find, cat, sed). This eliminates subprocess spawn overhead,
// prevents shell-escaping attacks, and avoids kernel context-switching latency.
//
// Size-classed sync.Pool arenas (64KB, 256KB, 1MB) provide bounded memory allocations under
// concurrent access and high buffer reuse hit rates.

const (
	// ArenaClass64K is the scratch arena class for small reads, chunking, and file headers.
	ArenaClass64K = 64 * 1024
	// ArenaClass256K is the scratch arena class for medium files.
	ArenaClass256K = 256 * 1024
	// ArenaClass1M is the scratch arena class for files up to the default MaxReadBytes limit.
	ArenaClass1M = 1024 * 1024
)

// BufferPoolMetrics tracks buffer pool usage, allocation counts, and subprocess prevention.
type BufferPoolMetrics struct {
	Acquires            uint64 `json:"acquires"`
	Releases            uint64 `json:"releases"`
	Allocations         uint64 `json:"allocations"`
	SubprocessesAvoided uint64 `json:"subprocesses_avoided"`
}

// HitRate returns the ratio of reused buffer pool acquisitions to total acquisitions.
func (m BufferPoolMetrics) HitRate() float64 {
	if m.Acquires == 0 {
		return 0.0
	}
	if m.Allocations >= m.Acquires {
		return 0.0
	}
	return float64(m.Acquires-m.Allocations) / float64(m.Acquires)
}

// BufferPool manages size-classed scratch buffer arenas.
type BufferPool struct {
	pool64K  sync.Pool
	pool256K sync.Pool
	pool1M   sync.Pool

	acquires            atomic.Uint64
	releases            atomic.Uint64
	allocations         atomic.Uint64
	subprocessesAvoided atomic.Uint64
}

// NewBufferPool initializes a dedicated buffer pool with size-classed arenas.
func NewBufferPool() *BufferPool {
	p := &BufferPool{}
	p.pool64K.New = func() any {
		p.allocations.Add(1)
		b := make([]byte, ArenaClass64K)
		return &b
	}
	p.pool256K.New = func() any {
		p.allocations.Add(1)
		b := make([]byte, ArenaClass256K)
		return &b
	}
	p.pool1M.New = func() any {
		p.allocations.Add(1)
		b := make([]byte, ArenaClass1M)
		return &b
	}
	return p
}

var globalPool = NewBufferPool()

// AcquireBuffer returns a pooled byte slice with capacity of at least minSize bytes.
// If minSize <= 0, ArenaClass64K is used as default.
func (p *BufferPool) AcquireBuffer(minSize int) []byte {
	p.acquires.Add(1)
	if minSize <= 0 {
		minSize = ArenaClass64K
	}
	if minSize <= ArenaClass64K {
		bp := p.pool64K.Get().(*[]byte)
		return (*bp)[:minSize]
	}
	if minSize <= ArenaClass256K {
		bp := p.pool256K.Get().(*[]byte)
		return (*bp)[:minSize]
	}
	if minSize <= ArenaClass1M {
		bp := p.pool1M.Get().(*[]byte)
		return (*bp)[:minSize]
	}
	// Oversize allocations fall through to direct make
	p.allocations.Add(1)
	return make([]byte, minSize)
}

// AcquireBufferPtr returns a pointer to a pooled byte slice with capacity >= minSize.
func (p *BufferPool) AcquireBufferPtr(minSize int) *[]byte {
	buf := p.AcquireBuffer(minSize)
	return &buf
}

// ReleaseBuffer returns a buffer to its size-classed arena.
// Accepts []byte, *[]byte, or *ByteBuffer.
func (p *BufferPool) ReleaseBuffer(buf any) {
	if buf == nil {
		return
	}
	switch b := buf.(type) {
	case []byte:
		p.releaseBytes(b)
	case *[]byte:
		if b != nil {
			p.releaseBytes(*b)
		}
	case *ByteBuffer:
		if b != nil {
			b.Release()
		}
	}
}

func (p *BufferPool) releaseBytes(b []byte) {
	c := cap(b)
	if c < ArenaClass64K {
		return
	}
	p.releases.Add(1)
	if c >= ArenaClass1M {
		b = b[:ArenaClass1M]
		p.pool1M.Put(&b)
	} else if c >= ArenaClass256K {
		b = b[:ArenaClass256K]
		p.pool256K.Put(&b)
	} else if c >= ArenaClass64K {
		b = b[:ArenaClass64K]
		p.pool64K.Put(&b)
	}
}

// Metrics reports the current atomic metrics snapshot.
func (p *BufferPool) Metrics() BufferPoolMetrics {
	return BufferPoolMetrics{
		Acquires:            p.acquires.Load(),
		Releases:            p.releases.Load(),
		Allocations:         p.allocations.Load(),
		SubprocessesAvoided: p.subprocessesAvoided.Load(),
	}
}

// ResetMetrics resets all metrics counters in this pool.
func (p *BufferPool) ResetMetrics() {
	p.acquires.Store(0)
	p.releases.Store(0)
	p.allocations.Store(0)
	p.subprocessesAvoided.Store(0)
}

// RecordSubprocessAvoided increments the count of external process spawns avoided.
func (p *BufferPool) RecordSubprocessAvoided() {
	p.subprocessesAvoided.Add(1)
}

// Package-level accessors delegating to globalPool:

// AcquireBuffer returns a pooled byte slice with capacity of at least minSize bytes.
func AcquireBuffer(minSize int) []byte {
	return globalPool.AcquireBuffer(minSize)
}

// AcquireBufferPtr returns a pointer to a pooled byte slice with capacity >= minSize.
func AcquireBufferPtr(minSize int) *[]byte {
	return globalPool.AcquireBufferPtr(minSize)
}

// ReleaseBuffer returns a buffer to its size-classed arena.
func ReleaseBuffer(buf any) {
	globalPool.ReleaseBuffer(buf)
}

// GetBufferPoolMetrics returns the current atomic metrics counters.
func GetBufferPoolMetrics() BufferPoolMetrics {
	return globalPool.Metrics()
}

// ResetBufferPoolMetrics resets the global buffer pool metrics.
func ResetBufferPoolMetrics() {
	globalPool.ResetMetrics()
}

// RecordSubprocessAvoided records an external subprocess execution avoided.
func RecordSubprocessAvoided() {
	globalPool.RecordSubprocessAvoided()
}

// ByteBuffer provides an io.Writer-compatible growable pooled buffer.
type ByteBuffer struct {
	pool *BufferPool
	buf  []byte
	len  int
}

// AcquireByteBuffer allocates a ByteBuffer wrapper backed by a pooled buffer.
func AcquireByteBuffer(minSize int) *ByteBuffer {
	return globalPool.AcquireByteBuffer(minSize)
}

// AcquireByteBuffer creates a ByteBuffer backed by this pool.
func (p *BufferPool) AcquireByteBuffer(minSize int) *ByteBuffer {
	b := p.AcquireBuffer(minSize)
	return &ByteBuffer{
		pool: p,
		buf:  b,
		len:  0,
	}
}

// Bytes returns the written portion of the buffer.
func (b *ByteBuffer) Bytes() []byte {
	if b == nil {
		return nil
	}
	return b.buf[:b.len]
}

// String returns the written content as a string.
func (b *ByteBuffer) String() string {
	if b == nil {
		return ""
	}
	return string(b.buf[:b.len])
}

// Len returns the number of bytes written.
func (b *ByteBuffer) Len() int {
	if b == nil {
		return 0
	}
	return b.len
}

// Cap returns the total capacity of the underlying scratch buffer.
func (b *ByteBuffer) Cap() int {
	if b == nil {
		return 0
	}
	return cap(b.buf)
}

// Reset clears the buffer length without freeing the underlying arena.
func (b *ByteBuffer) Reset() {
	if b != nil {
		b.len = 0
	}
}

func (b *ByteBuffer) grow(needed int) {
	if b.len+needed <= cap(b.buf) {
		return
	}
	newCap := cap(b.buf) * 2
	if newCap < b.len+needed {
		newCap = b.len + needed
	}
	p := b.pool
	if p == nil {
		p = globalPool
	}
	newBuf := p.AcquireBuffer(newCap)
	copy(newBuf, b.buf[:b.len])
	p.ReleaseBuffer(b.buf)
	b.buf = newBuf
}

// Write appends bytes to the buffer, growing through pooled arenas if needed.
func (b *ByteBuffer) Write(p []byte) (int, error) {
	if b == nil {
		return 0, errors.New("nil ByteBuffer")
	}
	b.grow(len(p))
	copy(b.buf[b.len:], p)
	b.len += len(p)
	return len(p), nil
}

// WriteString appends a string to the buffer.
func (b *ByteBuffer) WriteString(s string) (int, error) {
	if b == nil {
		return 0, errors.New("nil ByteBuffer")
	}
	b.grow(len(s))
	copy(b.buf[b.len:], s)
	b.len += len(s)
	return len(s), nil
}

// WriteByte appends a single byte.
func (b *ByteBuffer) WriteByte(c byte) error {
	if b == nil {
		return io.ErrShortWrite
	}
	b.grow(1)
	b.buf[b.len] = c
	b.len++
	return nil
}

// Release releases the underlying buffer back to the arena.
func (b *ByteBuffer) Release() {
	if b == nil || b.buf == nil {
		return
	}
	p := b.pool
	if p == nil {
		p = globalPool
	}
	p.ReleaseBuffer(b.buf)
	b.buf = nil
	b.len = 0
}

// ReleaseByteBuffer releases b back to its arena.
func ReleaseByteBuffer(b *ByteBuffer) {
	if b != nil {
		b.Release()
	}
}
