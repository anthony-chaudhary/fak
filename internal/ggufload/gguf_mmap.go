package ggufload

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_mmap.go — the opt-in mmap seam for GGUF shard readers (Tier-3 SSD offload
// foundation, #2726/#2722). Under FAK_GGUF_MMAP the retained per-shard reader becomes a
// read-only memory map served on demand by the page cache instead of an os.File +
// pread, so a later slice can alias tensor payloads straight out of the map and stream
// GLM-5.2's routed experts from SSD without materializing them on the heap. This slice
// deliberately changes NO read semantics: TensorBytes still copies into a fresh buffer
// (so `defer ws.Close()` stays sound), and every platform without an mmap impl —
// Windows included — degrades byte-identically to the os.Open + ReadAt path via the
// model.MmapOpen ok=false contract.

// mmapReaderAt adapts a memory-mapped byte region to io.ReaderAt so the existing
// TensorBytes bounds-check + ReadAt plumbing works unchanged over a map. Semantics
// mirror bytes.Reader.ReadAt: a negative offset errors, an offset at/past the end
// returns io.EOF, and a read that runs off the end returns the partial count with
// io.EOF — exactly what an os.File ReadAt of the same region would report.
type mmapReaderAt struct {
	data []byte
}

// ReadAt copies from the mapped region into p with bytes.Reader.ReadAt semantics
// (n < len(p) => io.EOF). The copy — not an alias — keeps the io.ReaderAt contract:
// callers own p and may retain it past the map's Closer.
func (m *mmapReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, fmt.Errorf("gguf: mmap ReadAt with negative offset %d", off)
	}
	if off >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n := copy(p, m.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// ggufMmapOnce/ggufMmapOn cache the FAK_GGUF_MMAP decision once per process (mirroring
// the W3MLPRequested opt-in idiom): loader goroutines never re-read process environment,
// so a load has one immutable reader-selection contract even across split shards opened
// at different times. Tests reset the pair directly (same package) to exercise both arms
// in one process.
var (
	ggufMmapOnce sync.Once
	ggufMmapOn   bool
)

// ggufMmapEnabled reports the opt-in mmap selection. Only the documented truthy
// spellings enable it; every other value (including unset) preserves the historical
// os.Open + ReadAt default.
func ggufMmapEnabled() bool {
	ggufMmapOnce.Do(func() {
		switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_GGUF_MMAP"))) {
		case "1", "on", "true":
			ggufMmapOn = true
		}
	})
	return ggufMmapOn
}

// openShardReader opens the RETAINED reader for one GGUF shard: the io.ReaderAt that
// WeightSource keeps for the checkpoint's lifetime and serves every TensorBytes from.
// Under FAK_GGUF_MMAP (and on a platform where model.MmapOpen succeeds) that reader is
// a read-only memory map — data is the mapped region, closer munmaps it — so tensor
// reads demand-page from SSD instead of issuing preads. Otherwise (gate off, or
// ok=false on e.g. Windows) it is the historical os.Open path: r is the *os.File,
// closer closes it, data is nil.
func openShardReader(path string) (r io.ReaderAt, size int64, closer io.Closer, data []byte, err error) {
	if ggufMmapEnabled() {
		mapped, mc, ok, merr := model.MmapOpen(path)
		if merr != nil {
			return nil, 0, nil, nil, fmt.Errorf("gguf: mmap %s: %w", path, merr)
		}
		if ok {
			return &mmapReaderAt{data: mapped}, int64(len(mapped)), mc, mapped, nil
		}
		// ok=false, nil err: this platform/file cannot map — fall through to os.Open.
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, nil, nil, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, 0, nil, nil, err
	}
	return f, st.Size(), f, nil, nil
}

// retainShardReader upgrades a parse-only open shard file to the retained per-shard
// reader. With the mmap gate off it retains f itself (the historical behaviour, no
// second open). With the gate on it opens the shard through openShardReader and closes
// the parse-only file — the header was already parsed from f, so only the retained
// reader changes. On error the parse-only file is closed here; the caller just
// propagates err.
func retainShardReader(path string, f *os.File, size int64) (io.ReaderAt, int64, io.Closer, []byte, error) {
	if !ggufMmapEnabled() {
		return f, size, f, nil, nil
	}
	r, rsize, closer, data, err := openShardReader(path)
	if err != nil {
		_ = f.Close()
		return nil, 0, nil, nil, err
	}
	_ = f.Close()
	return r, rsize, closer, data, nil
}
