package compute

import (
	"strings"
	"testing"
)

func TestSingleResourceCapExceeded(t *testing.T) {
	for _, tc := range []struct {
		name   string
		bytes  int
		cap    int64
		exceed bool
	}{
		{"unknown cap", 1 << 30, 0, false},
		{"under", 63, 64, false},
		{"equal", 64, 64, false},
		{"over", 65, 64, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := singleResourceCapExceeded(tc.bytes, tc.cap); got != tc.exceed {
				t.Fatalf("singleResourceCapExceeded(%d,%d)=%v want %v", tc.bytes, tc.cap, got, tc.exceed)
			}
		})
	}
}

func TestQ8RowChunksForCap(t *testing.T) {
	t.Run("unknown cap keeps single upload", func(t *testing.T) {
		chunks, chunked, ok := q8RowChunksForCap(5, 32, 32, 0)
		if !ok || chunked {
			t.Fatalf("q8RowChunksForCap unknown cap ok=%v chunked=%v, want ok=true chunked=false", ok, chunked)
		}
		if len(chunks) != 1 || chunks[0] != (q8RowChunk{start: 0, rows: 5}) {
			t.Fatalf("chunks=%+v, want one full chunk", chunks)
		}
	})

	t.Run("under cap keeps single upload", func(t *testing.T) {
		chunks, chunked, ok := q8RowChunksForCap(2, 32, 32, 96)
		if !ok || chunked {
			t.Fatalf("q8RowChunksForCap under cap ok=%v chunked=%v, want ok=true chunked=false", ok, chunked)
		}
		if len(chunks) != 1 || chunks[0] != (q8RowChunk{start: 0, rows: 2}) {
			t.Fatalf("chunks=%+v, want one full chunk", chunks)
		}
	})

	t.Run("over cap splits by output rows", func(t *testing.T) {
		chunks, chunked, ok := q8RowChunksForCap(5, 32, 32, 96)
		if !ok || !chunked {
			t.Fatalf("q8RowChunksForCap over cap ok=%v chunked=%v, want ok=true chunked=true", ok, chunked)
		}
		want := []q8RowChunk{{start: 0, rows: 3}, {start: 3, rows: 2}}
		if len(chunks) != len(want) {
			t.Fatalf("len(chunks)=%d want %d: %+v", len(chunks), len(want), chunks)
		}
		for i := range want {
			if chunks[i] != want[i] {
				t.Fatalf("chunk[%d]=%+v want %+v", i, chunks[i], want[i])
			}
			if codeBytes := chunks[i].rows * 32; codeBytes > 96 {
				t.Fatalf("chunk[%d] code bytes=%d exceeds cap", i, codeBytes)
			}
			if scaleBytes := chunks[i].rows * 4; scaleBytes > 96 {
				t.Fatalf("chunk[%d] scale bytes=%d exceeds cap", i, scaleBytes)
			}
		}
	})

	t.Run("single row over cap is impossible", func(t *testing.T) {
		chunks, chunked, ok := q8RowChunksForCap(2, 128, 32, 64)
		if ok || !chunked || len(chunks) != 0 {
			t.Fatalf("q8RowChunksForCap row over cap chunks=%+v chunked=%v ok=%v, want impossible chunked plan", chunks, chunked, ok)
		}
	})
}

func TestFormatVulkanResourceCapErrorNamesBufferAndCaps(t *testing.T) {
	got := formatVulkanResourceCapError("Q8_0 weight code buffer [8192,524288]", 4<<30, 2<<30, 2<<30, 3<<30)
	for _, want := range []string{
		"Q8_0 weight code buffer [8192,524288]",
		"4294967296 bytes",
		"2147483648 bytes",
		"maxStorageBufferRange=2147483648",
		"maxMemoryAllocationSize=3221225472",
		"split/chunk",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatted cap error missing %q:\n%s", want, got)
		}
	}
}
