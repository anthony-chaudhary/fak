package xenginekv

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// makeBytes generates a deterministic byte slice of length n.
func makeBytes(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = seed + byte(i%251)
	}
	return b
}

// BenchmarkPut measures putting payloads into the co-resident arena across
// representative payload sizes: inline-equivalent tool arg (64B), structured
// tool result (1KB), and KV cache chunk (64KB).
func BenchmarkPut(b *testing.B) {
	sizes := []int{64, 1024, 64 * 1024}
	ctx := context.Background()
	const arenaSize = 32 << 20 // 32 MiB

	for _, size := range sizes {
		name := fmt.Sprintf("%dB", size)
		if size >= 1024 {
			name = fmt.Sprintf("%dKB", size/1024)
		}
		payload := makeBytes(size, 42)
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ReportAllocs()
			a := NewArena(arenaSize)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := a.Put(ctx, payload)
				if err != nil {
					b.StopTimer()
					a = NewArena(arenaSize)
					b.StartTimer()
					if _, err := a.Put(ctx, payload); err != nil {
						b.Fatalf("Put failed: %v", err)
					}
				}
			}
		})
	}
}

// BenchmarkResolve measures zero-copy view slice generation under RLock
// across representative payload sizes.
func BenchmarkResolve(b *testing.B) {
	ctx := context.Background()
	sizes := []int{64, 1024, 64 * 1024}
	for _, size := range sizes {
		name := fmt.Sprintf("%dB", size)
		if size >= 1024 {
			name = fmt.Sprintf("%dKB", size/1024)
		}
		payload := makeBytes(size, 88)
		b.Run(name, func(b *testing.B) {
			a := NewArena(size * 2)
			ref, err := a.Put(ctx, payload)
			if err != nil {
				b.Fatalf("Put: %v", err)
			}
			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				view, err := a.Resolve(ctx, ref)
				if err != nil || len(view) != size {
					b.Fatalf("Resolve: %v", err)
				}
			}
		})
	}
}

// BenchmarkResolve_Parallel measures concurrent zero-copy reads across goroutines
// simulating parallel worker / multi-arm replay access to shared KV.
func BenchmarkResolve_Parallel(b *testing.B) {
	ctx := context.Background()
	const size = 1024
	payload := makeBytes(size, 99)
	a := NewArena(size * 2)
	ref, err := a.Put(ctx, payload)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.SetBytes(int64(size))
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			view, err := a.Resolve(ctx, ref)
			if err != nil || len(view) != size {
				b.Fatalf("Resolve parallel: %v", err)
			}
		}
	})
}

// BenchmarkResolveLeased measures atomic zero-copy view resolution + transient
// reader lease acquisition and subsequent release.
func BenchmarkResolveLeased(b *testing.B) {
	ctx := context.Background()
	const size = 1024
	payload := makeBytes(size, 101)
	a := NewArena(size * 2)
	ref, err := a.Put(ctx, payload)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.SetBytes(int64(size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		view, release, err := a.ResolveLeased(ctx, ref)
		if err != nil || len(view) != size {
			b.Fatalf("ResolveLeased: %v", err)
		}
		release()
	}
}

// BenchmarkClone measures in-arena prefix replication for prompt/prefix reuse.
func BenchmarkClone(b *testing.B) {
	ctx := context.Background()
	const arenaSize = 32 << 20
	sizes := []int{64, 1024, 64 * 1024}

	for _, size := range sizes {
		name := fmt.Sprintf("%dB", size)
		if size >= 1024 {
			name = fmt.Sprintf("%dKB", size/1024)
		}
		prefix := makeBytes(size, 77)
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ReportAllocs()
			a := NewArena(arenaSize)
			src, err := a.Put(ctx, prefix)
			if err != nil {
				b.Fatalf("Put setup: %v", err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := a.Clone(src)
				if err != nil {
					b.StopTimer()
					a = NewArena(arenaSize)
					src, err = a.Put(ctx, prefix)
					if err != nil {
						b.Fatalf("Put after reset: %v", err)
					}
					b.StartTimer()
					if _, err := a.Clone(src); err != nil {
						b.Fatalf("Clone: %v", err)
					}
				}
			}
		})
	}
}

// BenchmarkEvict_Quarantine measures unconditional quarantine eviction: unmapping
// the span, physically zeroing its bytes, and deleting from the live map.
func BenchmarkEvict_Quarantine(b *testing.B) {
	ctx := context.Background()
	sizes := []int{64, 1024, 64 * 1024}
	for _, size := range sizes {
		name := fmt.Sprintf("%dB", size)
		if size >= 1024 {
			name = fmt.Sprintf("%dKB", size/1024)
		}
		payload := makeBytes(size, 109)
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ReportAllocs()
			const batch = 1024
			b.ResetTimer()
			for i := 0; i < b.N; {
				chunk := batch
				if b.N-i < chunk {
					chunk = b.N - i
				}
				b.StopTimer()
				a := NewArena(chunk * size)
				refs := make([]abi.Ref, chunk)
				for j := 0; j < chunk; j++ {
					var err error
					refs[j], err = a.Put(ctx, payload)
					if err != nil {
						b.Fatalf("Put setup: %v", err)
					}
				}
				b.StartTimer()
				for j := 0; j < chunk; j++ {
					if err := a.Evict(refs[j]); err != nil {
						b.Fatalf("Evict: %v", err)
					}
				}
				i += chunk
			}
		})
	}
}

// BenchmarkTryEvict_Capacity measures capacity-driven eviction gating: checking
// CanEvict (pins == 0 && readers == 0), clearing bytes, and deleting the span.
func BenchmarkTryEvict_Capacity(b *testing.B) {
	ctx := context.Background()
	const size = 1024
	payload := makeBytes(size, 111)
	b.SetBytes(int64(size))
	b.ReportAllocs()
	const batch = 1024
	b.ResetTimer()
	for i := 0; i < b.N; {
		chunk := batch
		if b.N-i < chunk {
			chunk = b.N - i
		}
		b.StopTimer()
		a := NewArena(chunk * size)
		refs := make([]abi.Ref, chunk)
		for j := 0; j < chunk; j++ {
			var err error
			refs[j], err = a.Put(ctx, payload)
			if err != nil {
				b.Fatalf("Put setup: %v", err)
			}
		}
		b.StartTimer()
		for j := 0; j < chunk; j++ {
			if !a.TryEvict(refs[j]) {
				b.Fatalf("TryEvict refused")
			}
		}
		i += chunk
	}
}

// BenchmarkPinUnpin measures persistent pin refcount lifecycle (Pin + Unpin)
// on a resident span.
func BenchmarkPinUnpin(b *testing.B) {
	ctx := context.Background()
	payload := makeBytes(256, 105)
	a := NewArena(1024)
	ref, err := a.Put(ctx, payload)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := a.Pin(ref); err != nil {
			b.Fatalf("Pin: %v", err)
		}
		if err := a.Unpin(ref); err != nil {
			b.Fatalf("Unpin: %v", err)
		}
	}
}

// BenchmarkAcquireReader measures transient reader lease lifecycle
// (AcquireReader + release).
func BenchmarkAcquireReader(b *testing.B) {
	ctx := context.Background()
	payload := makeBytes(256, 103)
	a := NewArena(1024)
	ref, err := a.Put(ctx, payload)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		release, ok := a.AcquireReader(ref)
		if !ok {
			b.Fatalf("AcquireReader failed")
		}
		release()
	}
}

// BenchmarkCanEvict measures the advisory two-axis evictability check under RLock.
func BenchmarkCanEvict(b *testing.B) {
	ctx := context.Background()
	payload := makeBytes(256, 107)
	a := NewArena(1024)
	ref, err := a.Put(ctx, payload)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !a.CanEvict(ref) {
			b.Fatalf("expected CanEvict true")
		}
	}
}

// BenchmarkPageOut_Region measures zero-movement context-MMU page-out for RefRegion.
func BenchmarkPageOut_Region(b *testing.B) {
	ctx := context.Background()
	a := NewArena(1024)
	ref, err := a.Put(ctx, makeBytes(128, 113))
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := a.PageOut(ctx, ref)
		if err != nil || out.Kind != abi.RefRegion {
			b.Fatalf("PageOut: %v", err)
		}
	}
}

// BenchmarkPageOut_Inline measures admitting an inline RefInline into the
// co-resident arena during page-out.
func BenchmarkPageOut_Inline(b *testing.B) {
	ctx := context.Background()
	inline := abi.Ref{
		Kind:   abi.RefInline,
		Inline: makeBytes(64, 115),
		Len:    64,
	}
	const arenaSize = 32 << 20
	a := NewArena(arenaSize)
	b.SetBytes(64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := a.PageOut(ctx, inline)
		if err != nil {
			b.StopTimer()
			a = NewArena(arenaSize)
			b.StartTimer()
			out, err = a.PageOut(ctx, inline)
			if err != nil {
				b.Fatalf("PageOut inline: %v", err)
			}
		}
		if out.Kind != abi.RefRegion {
			b.Fatalf("expected RefRegion, got %v", out.Kind)
		}
	}
}

// BenchmarkPageIn measures the zero-movement context-MMU page-in seam.
func BenchmarkPageIn(b *testing.B) {
	ctx := context.Background()
	a := NewArena(1024)
	ref, err := a.Put(ctx, makeBytes(128, 117))
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		in, err := a.PageIn(ctx, ref)
		if err != nil || in.Kind != abi.RefRegion {
			b.Fatalf("PageIn: %v", err)
		}
	}
}

// BenchmarkAttachArena measures attaching to an externally provided region buffer.
func BenchmarkAttachArena(b *testing.B) {
	buf := make([]byte, 1024*1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a := AttachArena(buf)
		if a == nil {
			b.Fatal("AttachArena returned nil")
		}
	}
}
