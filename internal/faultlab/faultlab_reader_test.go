package faultlab

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// Invariant: An empty underlying reader must gracefully handle all fault types without panic or hang.
func TestFaultReaderEmptyReader(t *testing.T) {
	ctx := context.Background()

	t.Run("EmptyReaderTruncation", func(t *testing.T) {
		fi := NewFaultInjector()
		_ = fi.RegisterRule(NewFaultRule("trunc_empty", Truncation, "empty/trunc"))

		r := strings.NewReader("")
		ir := fi.InterceptReader(ctx, "empty/trunc", r)

		buf := make([]byte, 16)
		n, err := ir.Read(buf)
		if n != 0 {
			t.Fatalf("expected 0 bytes read from empty reader, got %d", n)
		}
		if !errors.Is(err, io.EOF) {
			t.Fatalf("expected io.EOF, got %v", err)
		}
	})

	t.Run("EmptyReaderCorruptedJSON", func(t *testing.T) {
		fi := NewFaultInjector()
		_ = fi.RegisterRule(NewFaultRule("json_empty", CorruptedJSON, "empty/json"))

		r := bytes.NewReader(nil)
		ir := fi.InterceptReader(ctx, "empty/json", r)

		buf := make([]byte, 32)
		n, err := ir.Read(buf)
		if n != 0 {
			t.Fatalf("expected 0 bytes from empty json reader, got %d", n)
		}
		if !errors.Is(err, io.EOF) {
			t.Fatalf("expected io.EOF, got %v", err)
		}
	})

	t.Run("EmptyReaderImmediateErrors", func(t *testing.T) {
		tests := []struct {
			name      string
			faultType FaultType
			wantErr   error
		}{
			{"NetworkDrop", NetworkDrop, ErrNetworkDrop},
			{"HostReset", HostReset, ErrHostReset},
			{"MemoryPressure", MemoryPressure, ErrMemoryPressure},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				fi := NewFaultInjector()
				_ = fi.RegisterRule(NewFaultRule("r_"+tc.name, tc.faultType, "target/*"))

				r := strings.NewReader("")
				ir := fi.InterceptReader(ctx, "target/call", r)

				buf := make([]byte, 8)
				n, err := ir.Read(buf)
				if n != 0 {
					t.Fatalf("expected 0 bytes on immediate fault, got %d", n)
				}
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("expected error %v, got %v", tc.wantErr, err)
				}
			})
		}
	})

	t.Run("EmptyReaderPassthrough", func(t *testing.T) {
		fi := NewFaultInjector()
		r := strings.NewReader("")
		ir := fi.InterceptReader(ctx, "unmatched", r)

		buf := make([]byte, 8)
		n, err := ir.Read(buf)
		if n != 0 || !errors.Is(err, io.EOF) {
			t.Fatalf("expected 0, EOF for unmatched empty reader, got n=%d, err=%v", n, err)
		}
	})
}

// Invariant: Zero-byte buffers passed to Read must never cause out-of-bounds panics or corrupt state.
func TestFaultReaderZeroByteBuffer(t *testing.T) {
	ctx := context.Background()
	fi := NewFaultInjector()

	_ = fi.RegisterRule(NewFaultRule("r_trunc", Truncation, "stream/trunc"))
	r := strings.NewReader("some longer text payload")
	ir := fi.InterceptReader(ctx, "stream/trunc", r)

	zeroBuf := make([]byte, 0)
	n, err := ir.Read(zeroBuf)
	if n != 0 {
		t.Fatalf("expected 0 bytes for zero-length buffer, got %d", n)
	}
	if err != nil {
		t.Fatalf("expected nil error on 0-length buffer read before cutoff, got %v", err)
	}

	// Follow-up read with standard buffer should proceed normally
	validBuf := make([]byte, 4)
	n, err = ir.Read(validBuf)
	if err != nil && !errors.Is(err, io.EOF) {
		t.Fatalf("unexpected error on subsequent read: %v", err)
	}
	if n != 4 {
		t.Fatalf("expected 4 bytes read, got %d", n)
	}
}

// Invariant: Boundary conditions for byte limits and ratios must enforce precise truncation cutoffs.
func TestFaultReaderTruncationBoundaries(t *testing.T) {
	ctx := context.Background()

	t.Run("ExactMatchLength", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("exact", Truncation, "stream/exact")
		rule.TruncateBytes = 10
		_ = fi.RegisterRule(rule)

		data := "0123456789" // exactly 10 bytes
		ir := fi.InterceptReader(ctx, "stream/exact", strings.NewReader(data))

		readBytes, err := io.ReadAll(ir)
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
		if string(readBytes) != data {
			t.Fatalf("expected exact payload %q, got %q", data, string(readBytes))
		}
	})

	t.Run("LimitGreaterThanStream", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("overflow", Truncation, "stream/overflow")
		rule.TruncateBytes = 100
		_ = fi.RegisterRule(rule)

		data := "short"
		ir := fi.InterceptReader(ctx, "stream/overflow", strings.NewReader(data))

		readBytes, err := io.ReadAll(ir)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if string(readBytes) != data {
			t.Fatalf("expected %q, got %q", data, string(readBytes))
		}
	})

	t.Run("SingleByteBufferReads", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("single", Truncation, "stream/single")
		rule.TruncateBytes = 5
		_ = fi.RegisterRule(rule)

		data := "abcdefghijklmnop"
		ir := fi.InterceptReader(ctx, "stream/single", strings.NewReader(data))

		var collected []byte
		buf := make([]byte, 1)
		for {
			n, err := ir.Read(buf)
			if n > 0 {
				collected = append(collected, buf[:n]...)
			}
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("unexpected read error: %v", err)
			}
		}

		if string(collected) != "abcde" {
			t.Fatalf("expected 'abcde', got %q", string(collected))
		}
	})

	t.Run("RatioTruncationBoundary", func(t *testing.T) {
		fi := NewFaultInjector()
		rule := NewFaultRule("ratio", Truncation, "stream/ratio")
		rule.TruncateRatio = 0.5
		_ = fi.RegisterRule(rule)

		data := strings.Repeat("x", 40)
		ir := fi.InterceptReader(ctx, "stream/ratio", strings.NewReader(data))

		buf := make([]byte, 20)
		n, err := ir.Read(buf)
		if err != nil && !errors.Is(err, io.EOF) {
			t.Fatalf("read failed: %v", err)
		}
		// Based on len(p) = 20 * 0.5 = 10
		if n != 10 {
			t.Fatalf("expected 10 bytes truncated by ratio, got %d", n)
		}
	})
}

// Invariant: Multiple chunked reads of CorruptedJSON streams must corrupt read chunks.
func TestFaultReaderChunkedCorruptedJSON(t *testing.T) {
	ctx := context.Background()
	fi := NewFaultInjector()

	_ = fi.RegisterRule(NewFaultRule("r_chunk_json", CorruptedJSON, "stream/json_chunk"))

	rawJSON := `{"item1": "value1", "item2": [10, 20, 30], "item3": {"nested": true}}`
	ir := fi.InterceptReader(ctx, "stream/json_chunk", strings.NewReader(rawJSON))

	var output []byte
	buf := make([]byte, 24)
	for {
		n, err := ir.Read(buf)
		if n > 0 {
			output = append(output, buf[:n]...)
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("unexpected read error: %v", err)
		}
	}

	if json.Valid(output) {
		t.Fatalf("chunked read output should not be valid JSON: %q", string(output))
	}

	rep := fi.Report()
	if rep.HitsByType[CorruptedJSON] == 0 {
		t.Fatalf("expected CorruptedJSON hits to be recorded")
	}
}

type failingCloserReader struct {
	io.Reader
	closeErr error
}

func (f *failingCloserReader) Close() error {
	return f.closeErr
}

// Invariant: Close propagation must forward underlying errors when present, and remain a safe no-op otherwise.
func TestFaultReaderCloserPropagation(t *testing.T) {
	ctx := context.Background()
	fi := NewFaultInjector()

	expectedErr := errors.New("underlying close failed")
	failing := &failingCloserReader{
		Reader:   strings.NewReader("hello"),
		closeErr: expectedErr,
	}

	ir := fi.InterceptReader(ctx, "test/closer", failing)
	closer, ok := ir.(io.Closer)
	if !ok {
		t.Fatalf("expected intercepted reader to implement io.Closer")
	}

	if err := closer.Close(); !errors.Is(err, expectedErr) {
		t.Fatalf("expected closer to forward underlying error, got %v", err)
	}

	// Non-closer reader test
	plainReader := bytes.NewReader([]byte("non-closer"))
	irPlain := fi.InterceptReader(ctx, "test/plain", plainReader)
	if closerPlain, ok := irPlain.(io.Closer); ok {
		if err := closerPlain.Close(); err != nil {
			t.Fatalf("expected safe nil return from non-closer Close(), got %v", err)
		}
	}
}

// Invariant: Context deadlines during latency delays must fail-closed with context error.
func TestFaultReaderContextCancellation(t *testing.T) {
	fi := NewFaultInjector(
		WithSleep(func(ctx context.Context, d time.Duration) error {
			return ctx.Err()
		}),
	)

	rule := NewFaultRule("r_delay", LatencySpike, "stream/delay")
	rule.Delay = 100 * time.Millisecond
	_ = fi.RegisterRule(rule)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ir := fi.InterceptReader(ctx, "stream/delay", strings.NewReader("delayed"))
	buf := make([]byte, 16)
	_, err := ir.Read(buf)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

// Invariant: MaxHits quota must be respected across multiple reader stream instances.
func TestFaultReaderMaxHitsQuotaAcrossStreams(t *testing.T) {
	ctx := context.Background()
	fi := NewFaultInjector()

	rule := NewFaultRule("quota_rule", NetworkDrop, "stream/quota")
	rule.MaxHits = 3
	_ = fi.RegisterRule(rule)

	for i := 1; i <= 3; i++ {
		ir := fi.InterceptReader(ctx, "stream/quota", strings.NewReader("payload"))
		buf := make([]byte, 16)
		_, err := ir.Read(buf)
		if !errors.Is(err, ErrNetworkDrop) {
			t.Fatalf("stream %d: expected ErrNetworkDrop within quota, got %v", i, err)
		}
	}

	// 4th and 5th streams must bypass fault injection because quota is exhausted
	for i := 4; i <= 5; i++ {
		ir := fi.InterceptReader(ctx, "stream/quota", strings.NewReader("payload"))
		data, err := io.ReadAll(ir)
		if err != nil {
			t.Fatalf("stream %d: unexpected error after quota exhaustion: %v", i, err)
		}
		if string(data) != "payload" {
			t.Fatalf("stream %d: expected payload passthrough, got %q", i, string(data))
		}
	}
}

// Invariant: Concurrent reader operations across multiple streams must remain race-free and consistent.
func TestFaultReaderConcurrentStreams(t *testing.T) {
	fi := NewFaultInjector(WithMaxRecentHits(100))
	ctx := context.Background()

	_ = fi.RegisterRule(NewFaultRule("c_drop", NetworkDrop, "concurrent/drop/*"))
	ruleTrunc := NewFaultRule("c_trunc", Truncation, "concurrent/trunc/*")
	ruleTrunc.TruncateBytes = 8
	_ = fi.RegisterRule(ruleTrunc)
	_ = fi.RegisterRule(NewFaultRule("c_pass", LatencySpike, "concurrent/pass/*"))

	const numGoroutines = 30
	const iterations = 20

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				target := fmt.Sprintf("concurrent/%s/task_%d", map[int]string{
					0: "drop",
					1: "trunc",
					2: "pass",
				}[workerID%3], j)

				r := strings.NewReader("0123456789abcdefghijklmnopqrstuvwxyz")
				ir := fi.InterceptReader(ctx, target, r)

				buf := make([]byte, 64)
				n, err := ir.Read(buf)

				switch workerID % 3 {
				case 0:
					if !errors.Is(err, ErrNetworkDrop) {
						t.Errorf("worker %d: expected ErrNetworkDrop, got %v", workerID, err)
					}
				case 1:
					if err != nil && !errors.Is(err, io.EOF) {
						t.Errorf("worker %d: unexpected error: %v", workerID, err)
					}
					if n != 8 {
						t.Errorf("worker %d: expected 8 bytes, got %d", workerID, n)
					}
				case 2:
					if err != nil && !errors.Is(err, io.EOF) {
						t.Errorf("worker %d: unexpected error: %v", workerID, err)
					}
				}
			}
		}(i)
	}

	wg.Wait()

	rep := fi.Report()
	if rep.TotalInjections == 0 {
		t.Fatalf("expected non-zero total injections")
	}
}

// Invariant: InterceptReader with custom payload override provides exact replacement bytes.
func TestFaultReaderCustomPayload(t *testing.T) {
	ctx := context.Background()
	fi := NewFaultInjector()

	rule := NewFaultRule("custom_stream", CorruptedJSON, "stream/custom")
	rule.CustomPayload = []byte(`{"error": "injected_custom_failure"}`)
	_ = fi.RegisterRule(rule)

	res, err := fi.Inject(ctx, "stream/custom", []byte(`{"turn": 1}`))
	if !errors.Is(err, ErrCorruptedJSON) {
		t.Fatalf("expected ErrCorruptedJSON, got %v", err)
	}
	if string(res) != `{"error": "injected_custom_failure"}` {
		t.Fatalf("expected custom payload, got %q", string(res))
	}
}

// BenchmarkInterceptReader_Passthrough measures stream read overhead with no matched rules.
func BenchmarkInterceptReader_Passthrough(b *testing.B) {
	fi := NewFaultInjector()
	ctx := context.Background()
	payload := []byte(strings.Repeat("hello world data stream ", 32))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		r := bytes.NewReader(payload)
		ir := fi.InterceptReader(ctx, "unmatched_target", r)
		buf := make([]byte, 128)
		for {
			_, err := ir.Read(buf)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				b.Fatalf("read error: %v", err)
			}
		}
	}
}

// BenchmarkInterceptReader_Truncation measures performance of intercepted stream truncation.
func BenchmarkInterceptReader_Truncation(b *testing.B) {
	fi := NewFaultInjector()
	rule := NewFaultRule("bench_trunc", Truncation, "bench/trunc")
	rule.TruncateBytes = 64
	_ = fi.RegisterRule(rule)

	ctx := context.Background()
	payload := []byte(strings.Repeat("stream payload data chunk ", 32))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		r := bytes.NewReader(payload)
		ir := fi.InterceptReader(ctx, "bench/trunc", r)
		buf := make([]byte, 128)
		for {
			_, err := ir.Read(buf)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				b.Fatalf("read error: %v", err)
			}
		}
	}
}

// BenchmarkFaultInjector_Inject measures latency of in-memory payload disruption.
func BenchmarkFaultInjector_Inject(b *testing.B) {
	fi := NewFaultInjector()
	_ = fi.RegisterRule(NewFaultRule("bench_corrupt", CorruptedJSON, "bench/json"))

	ctx := context.Background()
	payload := []byte(`{"model": "qwen", "action": "tool_call", "args": {"key": "value"}}`)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = fi.Inject(ctx, "bench/json", payload)
	}
}
