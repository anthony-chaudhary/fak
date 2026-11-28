package ggufload

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func countOpenFDs(t *testing.T) int {
	t.Helper()
	if runtime.GOOS != "linux" {
		return -1
	}
	entries, err := os.ReadDir("/proc/self/fd")
	if err != nil {
		t.Fatalf("ReadDir(/proc/self/fd): %v", err)
	}
	return len(entries)
}

// fak-test:runtime fast est=20ms lane=default
func TestGGUFLoadNoFDLeak(t *testing.T) {
	dir := t.TempDir()
	validPath := filepath.Join(dir, "synthetic.gguf")
	writeMmapSingleFileFixture(t, validPath)

	badPath := filepath.Join(dir, "malformed.gguf")
	if err := os.WriteFile(badPath, []byte("NOT_GGUF_TRUNCATED"), 0o644); err != nil {
		t.Fatalf("write malformed fixture: %v", err)
	}

	missingShard1 := filepath.Join(dir, "missing-00001-of-00002.gguf")
	shard1Bytes := writeSplitShard(t, 1, 2, 2, true, []splitTensor{
		{name: "a.weight", dims: []uint64{8}, typ: TensorF32, data: mmapPatternPayload(0x44, 32)},
	})
	if err := os.WriteFile(missingShard1, shard1Bytes, 0o644); err != nil {
		t.Fatalf("write split shard 1 fixture: %v", err)
	}

	// Warm up once before capturing baseline FD count on Linux.
	if ws, err := OpenWeights(validPath); err != nil {
		t.Fatalf("warmup OpenWeights: %v", err)
	} else if err := ws.Close(); err != nil {
		t.Fatalf("warmup ws.Close: %v", err)
	}

	baselineFDs := countOpenFDs(t)

	const iterations = 64
	for i := 0; i < iterations; i++ {
		if _, err := Open(validPath); err != nil {
			t.Fatalf("iter %d Open(valid): %v", i, err)
		}
		if _, err := OpenHeaders(validPath); err != nil {
			t.Fatalf("iter %d OpenHeaders(valid): %v", i, err)
		}

		ws, err := OpenWeights(validPath)
		if err != nil {
			t.Fatalf("iter %d OpenWeights(valid): %v", i, err)
		}
		if err := ws.Close(); err != nil {
			t.Fatalf("iter %d OpenWeights.Close: %v", i, err)
		}

		wsMapped, err := OpenWeightsMapped(validPath)
		if err != nil {
			t.Fatalf("iter %d OpenWeightsMapped(valid): %v", i, err)
		}
		if err := wsMapped.Close(); err != nil {
			t.Fatalf("iter %d OpenWeightsMapped.Close: %v", i, err)
		}

		// Malformed / error open paths must also release all handles.
		if _, err := Open(badPath); err == nil {
			t.Fatalf("iter %d Open(bad): expected error", i)
		}
		if _, err := OpenHeaders(badPath); err == nil {
			t.Fatalf("iter %d OpenHeaders(bad): expected error", i)
		}
		if wsErr, err := OpenWeights(badPath); err == nil {
			_ = wsErr.Close()
			t.Fatalf("iter %d OpenWeights(bad): expected error", i)
		}
		if wsErr, err := OpenWeightsMapped(badPath); err == nil {
			_ = wsErr.Close()
			t.Fatalf("iter %d OpenWeightsMapped(bad): expected error", i)
		}
		if wsErr, err := OpenWeights(missingShard1); err == nil {
			_ = wsErr.Close()
			t.Fatalf("iter %d OpenWeights(missingShard1): expected error", i)
		}
	}

	if runtime.GOOS == "linux" {
		afterFDs := countOpenFDs(t)
		if afterFDs > baselineFDs {
			t.Fatalf("FD leak detected: baseline=%d after=%d", baselineFDs, afterFDs)
		}
	}

	// On Windows, os.Remove fails if any file handle is still open.
	for _, p := range []string{validPath, badPath, missingShard1} {
		if err := os.Remove(p); err != nil {
			t.Fatalf("os.Remove(%s) failed (leaked handle?): %v", p, err)
		}
	}
}
