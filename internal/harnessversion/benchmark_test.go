package harnessversion

import (
	"fmt"
	"testing"
)

var (
	benchVersionSink string
	benchPinnedSink  bool
	benchActiveSink  []VersionDescriptor
)

// BenchmarkExplicitWireNegotiation measures wire version negotiation performance
// across header signals, path prefixes, header overrides, and invalid fallback.
func BenchmarkExplicitWireNegotiation(b *testing.B) {
	r := NewStickySessionRouter()
	_ = r.Register(VersionDescriptor{Version: "v1", Weight: 90, Active: true, Metadata: map[string]string{"role": "stable"}})
	_ = r.Register(VersionDescriptor{Version: "v2", Weight: 10, Active: true, Metadata: map[string]string{"role": "canary"}})

	b.Run("HeaderOnly", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVersionSink = r.Negotiate("v2", "")
		}
	})

	b.Run("PathPrefix", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVersionSink = r.Negotiate("", "/v2/execute")
		}
	})

	b.Run("HeaderPrecedence", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVersionSink = r.Negotiate("v1", "/v2/execute")
		}
	})

	b.Run("FallbackUnregistered", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchVersionSink = r.Negotiate("v999-unregistered", "")
		}
	})
}

// BenchmarkStickySessionPinning measures session lookup performance on already pinned sessions,
// testing sequential reads, multi-session lookups, and concurrent parallel access.
func BenchmarkStickySessionPinning(b *testing.B) {
	r := NewStickySessionRouter()
	_ = r.Register(VersionDescriptor{Version: "v1", Weight: 80, Active: true})
	_ = r.Register(VersionDescriptor{Version: "v2", Weight: 20, Active: true})

	const numSessions = 1000
	sessionIDs := make([]string, numSessions)
	for i := 0; i < numSessions; i++ {
		sid := fmt.Sprintf("session-%05d", i)
		sessionIDs[i] = sid
		_, _ = r.Route(sid, "", "")
	}

	b.Run("SequentialHit", func(b *testing.B) {
		sid := sessionIDs[0]
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v, p := r.Route(sid, "", "")
			benchVersionSink = v
			benchPinnedSink = p
		}
	})

	b.Run("MultiSessionHit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sid := sessionIDs[i%numSessions]
			v, p := r.Route(sid, "", "")
			benchVersionSink = v
			benchPinnedSink = p
		}
	})

	b.Run("GetPinnedVersion", func(b *testing.B) {
		sid := sessionIDs[0]
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v, p := r.GetPinnedVersion(sid)
			benchVersionSink = v
			benchPinnedSink = p
		}
	})

	b.Run("ParallelReadHit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			idx := 0
			for pb.Next() {
				sid := sessionIDs[idx%numSessions]
				idx++
				v, p := r.Route(sid, "", "")
				if !p || v == "" {
					b.Fatal("unexpected unpinned or empty version")
				}
			}
		})
	})
}

// BenchmarkCanarySplittingDistribution measures probabilistic traffic selection across active versions,
// comparing binary weighted distribution (90/10), multi-way split (5 versions), and stateless routing.
func BenchmarkCanarySplittingDistribution(b *testing.B) {
	b.Run("Weighted2Way", func(b *testing.B) {
		r := NewStickySessionRouter()
		_ = r.Register(VersionDescriptor{Version: "v1", Weight: 90, Active: true})
		_ = r.Register(VersionDescriptor{Version: "v2", Weight: 10, Active: true})

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v, p := r.Route("", "", "")
			benchVersionSink = v
			benchPinnedSink = p
		}
	})

	b.Run("Weighted5Way", func(b *testing.B) {
		r := NewStickySessionRouter()
		_ = r.Register(VersionDescriptor{Version: "v1", Weight: 50, Active: true})
		_ = r.Register(VersionDescriptor{Version: "v2", Weight: 25, Active: true})
		_ = r.Register(VersionDescriptor{Version: "v3", Weight: 15, Active: true})
		_ = r.Register(VersionDescriptor{Version: "v4", Weight: 7, Active: true})
		_ = r.Register(VersionDescriptor{Version: "v5", Weight: 3, Active: true})

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v, p := r.Route("", "", "")
			benchVersionSink = v
			benchPinnedSink = p
		}
	})

	b.Run("DeterministicRNG", func(b *testing.B) {
		r := NewStickySessionRouter()
		_ = r.Register(VersionDescriptor{Version: "v1", Weight: 80, Active: true})
		_ = r.Register(VersionDescriptor{Version: "v2", Weight: 20, Active: true})
		counter := 0
		r.SetRandFunc(func(n int) int {
			counter++
			return counter % n
		})

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v, p := r.Route("", "", "")
			benchVersionSink = v
			benchPinnedSink = p
		}
	})
}

// BenchmarkActiveVersions measures the defensive copying of active version descriptors.
func BenchmarkActiveVersions(b *testing.B) {
	r := NewStickySessionRouter()
	_ = r.Register(VersionDescriptor{Version: "v1", Weight: 70, Active: true, Metadata: map[string]string{"role": "stable", "tier": "foundation"}})
	_ = r.Register(VersionDescriptor{Version: "v2", Weight: 20, Active: true, Metadata: map[string]string{"role": "canary", "tier": "foundation"}})
	_ = r.Register(VersionDescriptor{Version: "v3", Weight: 10, Active: true, Metadata: map[string]string{"role": "experimental"}})
	_ = r.Register(VersionDescriptor{Version: "v0-legacy", Weight: 0, Active: false})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchActiveSink = r.ActiveVersions()
	}
}

// BenchmarkSessionLifecycle measures full session lifecycle: route and pin, followed by release.
func BenchmarkSessionLifecycle(b *testing.B) {
	r := NewStickySessionRouter()
	_ = r.Register(VersionDescriptor{Version: "v1", Weight: 90, Active: true})
	_ = r.Register(VersionDescriptor{Version: "v2", Weight: 10, Active: true})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sid := "bench-session"
		v, _ := r.Route(sid, "", "")
		benchVersionSink = v
		r.ReleaseSession(sid)
	}
}
