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

// ReadOnlyRange borrows an exact, capacity-capped span of the read-only mapping.
// The caller must not modify it or retain it past the checkpoint owner's Close.
// The checkpoint file must remain immutable while mapped. ReadAt keeps its
// separate copy-and-own contract, including for TensorBytes callers.
//
// Prior-art: adapts llama_model_loader::load_data_range's mapped-address arm from
// ggml-org/llama.cpp src/llama-model-loader.cpp:1471-1491 at
// f2cc7282ce0042641df55e0e563b75e824e2b647 (MIT, Copyright (c) 2023-2026
// The ggml authors; license reproduced in THIRD_PARTY_NOTICES.md). Go uses
// subtraction-based bounds checks and a three-index slice instead of a pointer.
func (m *mmapReaderAt) ReadOnlyRange(off, size int64) ([]byte, error) {
	if m == nil || off < 0 || size < 0 || off > int64(len(m.data)) || size > int64(len(m.data))-off {
		return nil, fmt.Errorf("gguf: mmap range offset %d size %d outside mapping", off, size)
	}
	end := off + size // bounded above before addition
	return m.data[off:end:end], nil
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

// LoadMode selects how the retained shard reader materializes checkpoint bytes:
// none (the historical os.Open + ReadAt default), mmap (a read-only memory map served
// by the page cache), mlock (keep weights pinned in RAM — syscalls not yet implemented
// in this build), direct-io (bypass the page cache — syscalls not yet implemented in
// this build), or auto (pick mmap when the platform supports it, else none). The
// numeric values mirror llama.cpp's enum llama_load_mode (include/llama.h:206-211 at
// 83078fec0db82d6b5a00d9599062c38c39145755, MIT) so config spellings stay portable
// across the two codebases; upstream value 3 (mmap+mlock) is intentionally absent.
type LoadMode int32

const (
	LoadModeAuto     LoadMode = -1 // auto-detect based on platform capabilities
	LoadModeNone     LoadMode = 0  // no special load mode
	LoadModeMmap     LoadMode = 1  // memory-map the model
	LoadModeMlock    LoadMode = 2  // force the system to keep the model in RAM rather than swapping or compressing
	LoadModeDirectIO LoadMode = 4  // use direct I/O if available
)

// String renders the LoadMode as its config spelling ("auto", "none", "mmap",
// "mlock", "direct-io"), falling back to "LoadMode(n)" for an unrecognized value.
func (m LoadMode) String() string {
	switch m {
	case LoadModeAuto:
		return "auto"
	case LoadModeNone:
		return "none"
	case LoadModeMmap:
		return "mmap"
	case LoadModeMlock:
		return "mlock"
	case LoadModeDirectIO:
		return "direct-io"
	default:
		return fmt.Sprintf("LoadMode(%d)", m)
	}
}

// ResolveLoadMode resolves a requested LoadMode against the platform's actual
// capabilities into the load path this build will actually take, plus a
// human-readable reason when the resolution DEGRADED from the request ("" when
// the request was honored, or when AUTO made its policy pick). It never fails:
// an unsupported explicit request degrades to the plain-file-load path instead
// of erroring, mirroring llama_model_loader's constructor (src/llama-model-loader.cpp:559
// at 83078fec0db82d6b5a00d9599062c38c39145755, MIT), which derives use_mmap /
// use_direct_io from the requested mode and then downgrades use_mmap with a
// warning when mmap is unavailable on the platform.
//
// Rules (pinned by TestResolveLoadModeDegradesWithReason):
//   - Auto:     mmap when mmapSupported, else none — the policy pick carries no reason.
//   - None:     honored as-is.
//   - Mmap:     honored when mmapSupported; otherwise degrades to none with a reason.
//   - Mlock:    honored when mlockSupported; otherwise degrades to none with a reason
//     (upstream LLAMA_LOAD_MODE_MLOCK pins a normally-loaded buffer without mmap, so
//     the degraded path is the plain-file-load path, not the mapped one).
//   - DirectIO: always degrades to none with a reason — this build implements no
//     direct-I/O syscalls, so the request is never supported here.
//   - Unknown:  degrades to none with a reason naming the unrecognized value.
//
// Pure: no I/O, no env reads, no globals. Callers supply the capability flags
// (on Windows the mmap flag is false via the model.MmapOpen ok=false contract).
func ResolveLoadMode(requested LoadMode, mmapSupported, mlockSupported bool) (LoadMode, string) {
	switch requested {
	case LoadModeAuto:
		if mmapSupported {
			return LoadModeMmap, ""
		}
		return LoadModeNone, ""
	case LoadModeNone:
		return LoadModeNone, ""
	case LoadModeMmap:
		if mmapSupported {
			return LoadModeMmap, ""
		}
		return LoadModeNone, "mmap not supported on this platform; falling back to plain file reads"
	case LoadModeMlock:
		if mlockSupported {
			return LoadModeMlock, ""
		}
		return LoadModeNone, "mlock not supported on this platform; falling back to plain file reads"
	case LoadModeDirectIO:
		return LoadModeNone, "direct I/O not implemented in this build; falling back to plain file reads"
	default:
		return LoadModeNone, fmt.Sprintf("unknown load mode %s; falling back to plain file reads", requested.String())
	}
}

// ggufMmapRequest reports the LoadMode request implied by the FAK_GGUF_MMAP opt-in
// gate: LoadModeMmap when the gate is enabled, LoadModeNone otherwise. It delegates
// to ggufMmapEnabled so the gate keeps its exact historical semantics (same truthy
// spellings, same once-per-process capture); this mapping is what turns the boolean
// gate into a load-mode request that callers feed through ResolveLoadMode together
// with the platform's capability flags.
func ggufMmapRequest() LoadMode {
	if ggufMmapEnabled() {
		return LoadModeMmap
	}
	return LoadModeNone
}

// openShardReader opens the RETAINED reader for one GGUF shard: the io.ReaderAt that
// WeightSource keeps for the checkpoint's lifetime and serves every TensorBytes from.
// Under FAK_GGUF_MMAP (and on a platform where model.MmapOpen succeeds) that reader is
// a read-only memory map — data is the mapped region, closer munmaps it — so tensor
// reads demand-page from SSD instead of issuing preads. Otherwise (gate off, or
// ok=false on e.g. Windows) it is the historical os.Open path: r is the *os.File,
// closer closes it, data is nil.
//
// mmap is the caller's resolved reader selection: ggufMmapEnabled() for OpenWeights, or true
// for OpenWeightsMapped (the device-resident load that keeps weights in the page cache).
func openShardReader(path string, mmap bool) (r io.ReaderAt, size int64, closer io.Closer, data []byte, err error) {
	if mmap {
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
func retainShardReader(path string, f *os.File, size int64, mmap bool) (io.ReaderAt, int64, io.Closer, []byte, error) {
	if !mmap {
		return f, size, f, nil, nil
	}
	r, rsize, closer, data, err := openShardReader(path, true)
	if err != nil {
		_ = f.Close()
		return nil, 0, nil, nil, err
	}
	_ = f.Close()
	return r, rsize, closer, data, nil
}
