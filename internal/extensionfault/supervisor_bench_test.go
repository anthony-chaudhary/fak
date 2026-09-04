package extensionfault

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

// BenchmarkSupervisor measures the performance of Supervisor operations including
// specification initialization, state queries, and fast-path call rejections.
func BenchmarkSupervisor(b *testing.B) {
	b.Run("New", func(b *testing.B) {
		specs := make([]Spec, 8)
		for i := 0; i < 8; i++ {
			specs[i] = Spec{
				Name:           fmt.Sprintf("ext-%d", i),
				Command:        []string{"worker", "--id", fmt.Sprintf("%d", i)},
				StartupTimeout: time.Second,
				CallTimeout:    time.Second,
				MaxRestarts:    2,
			}
		}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			s, err := New(specs...)
			if err != nil {
				b.Fatalf("New failed: %v", err)
			}
			_ = s
		}
	})

	b.Run("Status", func(b *testing.B) {
		s, err := New(
			Spec{Name: "alpha", Command: []string{"worker"}, StartupTimeout: time.Second, CallTimeout: time.Second},
			Spec{Name: "beta", Command: []string{"worker"}, StartupTimeout: time.Second, CallTimeout: time.Second},
		)
		if err != nil {
			b.Fatalf("New failed: %v", err)
		}
		defer func() { _ = s.Close() }()

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			st, ok := s.Status("alpha")
			if !ok || st.Name != "alpha" {
				b.Fatalf("unexpected status: ok=%v, status=%+v", ok, st)
			}
		}
	})

	b.Run("CallUnavailable", func(b *testing.B) {
		s, err := New(
			Spec{Name: "alpha", Command: []string{"worker"}, StartupTimeout: time.Second, CallTimeout: time.Second},
		)
		if err != nil {
			b.Fatalf("New failed: %v", err)
		}
		defer func() { _ = s.Close() }()

		ctx := context.Background()
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, err := s.Call(ctx, "missing-ext", "payload")
			if err == nil {
				b.Fatal("expected error for missing extension")
			}
		}
	})

	b.Run("CallQuarantined", func(b *testing.B) {
		s, err := New(
			Spec{Name: "quarantined-ext", Command: []string{"worker"}, StartupTimeout: time.Second, CallTimeout: time.Second},
		)
		if err != nil {
			b.Fatalf("New failed: %v", err)
		}
		defer func() { _ = s.Close() }()

		// Artificially quarantine the extension to benchmark the rejection fast path.
		s.mu.Lock()
		ext := s.extensions["quarantined-ext"]
		ext.quarantined = true
		s.mu.Unlock()

		ctx := context.Background()
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, err := s.Call(ctx, "quarantined-ext", "payload")
			if err == nil {
				b.Fatal("expected error for quarantined extension")
			}
		}
	})
}

// BenchmarkReadFrame exercises JSON frame decoding throughput from buffered reader streams.
func BenchmarkReadFrame(b *testing.B) {
	rawFrame := []byte(`{"type":"result","payload":"ok:test-payload","pid":12345}` + "\n")
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := bufio.NewReader(bytes.NewReader(rawFrame))
		f, err := readFrame(ctx, r, 50*time.Millisecond)
		if err != nil || f.Type != "result" {
			b.Fatalf("readFrame failed: err=%v, frame=%+v", err, f)
		}
	}
}

// BenchmarkSupervisorStatusParallel benchmarks concurrent status access across goroutines.
func BenchmarkSupervisorStatusParallel(b *testing.B) {
	s, err := New(
		Spec{Name: "node-1", Command: []string{"node"}, StartupTimeout: time.Second, CallTimeout: time.Second},
		Spec{Name: "node-2", Command: []string{"node"}, StartupTimeout: time.Second, CallTimeout: time.Second},
		Spec{Name: "node-3", Command: []string{"node"}, StartupTimeout: time.Second, CallTimeout: time.Second},
	)
	if err != nil {
		b.Fatalf("New failed: %v", err)
	}
	defer func() { _ = s.Close() }()

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		names := []string{"node-1", "node-2", "node-3"}
		idx := 0
		for pb.Next() {
			name := names[idx%len(names)]
			st, ok := s.Status(name)
			if !ok || !strings.HasPrefix(st.Name, "node-") {
				b.Fatalf("unexpected status: ok=%v, status=%+v", ok, st)
			}
			idx++
		}
	})
}
