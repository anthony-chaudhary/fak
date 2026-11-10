package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// fak-test:runtime fast est=1s
func TestMmapPrefaultNoFaultInRequest(t *testing.T) {
	const recSize = 1024
	const numRecs = 256 // 256 KiB total: at least one page on any supported page size
	const total = recSize * numRecs

	content := make([]byte, total)
	for i := range content {
		content[i] = byte(i % 251)
	}

	path := filepath.Join(t.TempDir(), "dataset.dat")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	t.Run("prefault_report_values", func(t *testing.T) {
		r, err := OpenMappedRegion(path, true)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer r.Close()

		rep := r.Report()
		if rep.Bytes != total {
			t.Errorf("Bytes=%d want %d", rep.Bytes, total)
		}
		if rep.PageSize != r.pageUnit() {
			t.Errorf("PageSize=%d want %d", rep.PageSize, r.pageUnit())
		}
		if want := (int64(total) + rep.PageSize - 1) / rep.PageSize; rep.Pages != want {
			t.Errorf("Pages=%d want %d (ceil(Bytes/PageSize))", rep.Pages, want)
		}
		if rep.Advised && !r.isUnixPlatform() {
			t.Errorf("Advised=true on !unix; the materializing path must not claim an advice")
		}
	})

	t.Run("slice_is_byte_exact", func(t *testing.T) {
		r, err := OpenMappedRegion(path, true)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer r.Close()

		got, err := r.Slice(0, int64(total))
		if err != nil {
			t.Fatalf("full Slice: %v", err)
		}
		if !bytes.Equal(got, content) {
			t.Fatalf("full Slice differs from source file (%d bytes)", total)
		}

		// One interior record, copied out.
		const rec = 137
		got, err = r.Slice(int64(rec*recSize), recSize)
		if err != nil {
			t.Fatalf("record Slice: %v", err)
		}
		if !bytes.Equal(got, content[rec*recSize:(rec+1)*recSize]) {
			t.Fatalf("record %d bytes differ", rec)
		}

		// Boundary: the final single byte.
		got, err = r.Slice(int64(total)-1, 1)
		if err != nil {
			t.Fatalf("tail Slice: %v", err)
		}
		if !bytes.Equal(got, content[total-1:]) {
			t.Fatalf("tail byte=%x want %x", got, content[total-1:])
		}

		// Zero-size slice is valid and empty.
		got, err = r.Slice(5, 0)
		if err != nil || len(got) != 0 {
			t.Fatalf("zero-size Slice: got=%d err=%v want empty,nil", len(got), err)
		}
	})

	t.Run("concurrent_readers_get_own_bytes", func(t *testing.T) {
		r, err := OpenMappedRegion(path, true)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer r.Close()

		const workers = 8
		const rounds = 50
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				for k := 0; k < rounds; k++ {
					rec := (w*rounds + k) % numRecs
					got, err := r.Slice(int64(rec*recSize), recSize)
					if err != nil {
						t.Errorf("worker %d round %d: Slice: %v", w, k, err)
						return
					}
					if !bytes.Equal(got, content[rec*recSize:(rec+1)*recSize]) {
						t.Errorf("worker %d round %d: record %d served foreign bytes", w, k, rec)
						return
					}
				}
			}(w)
		}
		wg.Wait()
	})

	t.Run("sentinels_and_lifecycle", func(t *testing.T) {
		r, err := OpenMappedRegion(path, false)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		if rep := r.Report(); rep != (PrefaultReport{}) {
			t.Errorf("no-prefault Report=%+v want zero value", rep)
		}

		for _, tc := range []struct{ off, size int64 }{
			{-1, 1},
			{int64(total), 1},
			{0, int64(total) + 1},
			{int64(total) - 1, 2},
		} {
			if _, err := r.Slice(tc.off, tc.size); !errors.Is(err, errOffsetOutOfRange) {
				t.Errorf("Slice(%d,%d): err=%v want errOffsetOutOfRange", tc.off, tc.size, err)
			}
		}

		if err := r.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		if err := r.Close(); err != nil {
			t.Fatalf("second close must be a no-op: %v", err)
		}
		if _, err := r.Slice(0, 1); !errors.Is(err, errClosedRegion) {
			t.Errorf("post-close Slice: err=%v want errClosedRegion", err)
		}
		if got := r.Len(); got != 0 {
			t.Errorf("Len after close=%d want 0", got)
		}
	})

	t.Run("empty_file_rejected", func(t *testing.T) {
		empty := filepath.Join(t.TempDir(), "empty.dat")
		if err := os.WriteFile(empty, nil, 0o644); err != nil {
			t.Fatalf("write empty fixture: %v", err)
		}
		if _, err := OpenMappedRegion(empty, true); !errors.Is(err, errEmptyRegion) {
			t.Errorf("open empty file: err=%v want errEmptyRegion", err)
		}
	})
}
