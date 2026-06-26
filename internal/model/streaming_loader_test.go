package model

import (
	"bytes"
	"errors"
	"io"
	"os"
	"testing"
)

// streaming_loader_test.go — the issue-#293 acceptance gate (A-009, "Model Streaming/Lazy
// Loading"): stream model weights on-demand rather than full upfront load. The three
// acceptance criteria and where each is witnessed:
//
//   - 50% memory reduction during initial load — the quant-on-load streaming loader
//     (LoadSafetensorsQuant{,Dir}) materializes the big matmul weights DIRECTLY to Q8_0
//     (~1.1 B/param) and drops their f32 copies (4 B/param), and the single-file mmap path
//     (openSafetensorsFileMmap) slices each tensor zero-copy so the whole file is never
//     heap-resident. Proven on synthetic fixtures by sharded_weightsource_test.go (#40) and
//     safetensors_mmap_test.go (#475); re-asserted below as the per-weight f32-drop, which is
//     a >2x footprint cut on the dominant weights.
//   - First-token latency unchanged — the streamed Model is BYTE-IDENTICAL to the eager
//     os.ReadFile load (same packed f32 raw + manifest, same Q8_0 codes/scales), so the
//     forward path it feeds is unchanged. Proven by safetensors_stream_test.go; the positive
//     control below re-asserts byte-identity for the quant-on-load path.
//   - Clean error on network failure — THIS file's load-bearing new witness. A streaming
//     weight source (a network-mounted checkpoint, an HTTP-range reader) can fail mid-stream
//     AFTER the header has parsed and the loader has committed to the tensor layout. We inject
//     exactly that fault at the io.ReaderAt seam the loader streams from, and prove the loader
//     returns a clean, wrapped error — no panic, no half-built Model handed back. A second
//     test drives the production public path (LoadSafetensorsQuant over a truncated file) with
//     no test seam, the on-disk analogue of a connection that drops mid-weight-stream.

// errStreamDropped stands in for a transport failure — a dropped connection or a network
// filesystem I/O error — surfacing through the io.ReaderAt the loader streams weights from.
var errStreamDropped = errors.New("model: weight stream dropped (simulated network failure)")

// dropAfterReaderAt serves ReadAt normally for the header region [0, dataBase) so the
// checkpoint parses, then fails every read that reaches into the data region — the byte-source
// analogue of a connection that drops after the headers, mid-weight-stream.
type dropAfterReaderAt struct {
	inner    io.ReaderAt
	dataBase int64
}

func (d dropAfterReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= d.dataBase {
		return 0, errStreamDropped
	}
	return d.inner.ReadAt(p, off)
}

// TestStreamingLoaderCleanErrorOnNetworkFailure injects a mid-stream byte-source failure into
// the real quant-on-load streaming loader and proves the failure is reported cleanly. The
// positive control over the same bytes pins the failure to the injected fault (not a bad
// fixture) and witnesses the byte-identity + f32-drop criteria along the way.
func TestStreamingLoaderCleanErrorOnNetworkFailure(t *testing.T) {
	path, cfg := writeSingleFileSafetensors(t)
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	_, dataBase, err := parseSafetensorsHeader(blob)
	if err != nil {
		t.Fatalf("parse header: %v", err)
	}

	// Positive control: a clean opener over the same bytes loads, and the streamed Model is
	// byte-identical to the eager os.ReadFile quant load — so the failure below is the injected
	// fault, not a malformed fixture (criterion: first-token latency unchanged == identical
	// model fed to the forward path).
	cleanOpen := func(string) (*safetensorsFile, error) {
		return newSafetensorsFile(bytes.NewReader(blob), int64(len(blob)), nil)
	}
	got, err := loadSafetensorsQuantFile(path, cfg, cleanOpen)
	if err != nil {
		t.Fatalf("clean streaming quant load: %v", err)
	}
	want, err := readFileLoadSafetensorsQuantForTest(path, cfg)
	if err != nil {
		t.Fatalf("readfile quant reference: %v", err)
	}
	assertModelRawEqual(t, want, got)
	assertQ8MapsEqual(t, want.q8w, got.q8w)

	// The memory-reduction criterion: every big matmul PROJECTION weight (isQuantWeight) had
	// its f32 DROPPED at load — it must not also survive in the resident f32 manifest — which
	// is the >2x footprint cut on the dominant weights. A tied embedding is the documented
	// exception: it lives in q8w (for the lm_head matmul) AND in the manifest (the f32 rows the
	// embedding lookup reads), so it is correctly excluded by isQuantWeight here.
	projDropped := 0
	for name := range got.q8w {
		if !isQuantWeight(name) {
			continue
		}
		if _, kept := got.manifest[name]; kept {
			t.Fatalf("quant-on-load kept the f32 copy of projection %s; the memory win requires dropping it", name)
		}
		projDropped++
	}
	if projDropped == 0 {
		t.Fatalf("no projection weight had its f32 dropped; nothing to reduce")
	}

	// Inject the network failure at the byte source: the header parses, then the first weight
	// stream read fails.
	flakyOpen := func(string) (*safetensorsFile, error) {
		r := dropAfterReaderAt{inner: bytes.NewReader(blob), dataBase: int64(dataBase)}
		return newSafetensorsFile(r, int64(len(blob)), nil)
	}
	m, err := loadSafetensorsQuantFile(path, cfg, flakyOpen)
	if err == nil {
		t.Fatalf("expected a clean error on a mid-stream weight read failure, got nil")
	}
	if !errors.Is(err, errStreamDropped) {
		t.Fatalf("loader must surface the underlying stream failure (wrapped), got: %v", err)
	}
	if m != nil {
		t.Fatalf("a failed streaming load must return no model, got a non-nil *Model")
	}
}

// TestStreamingLoaderCleanErrorOnTruncatedCheckpoint drives the production public loader
// (LoadSafetensorsQuant — no injected seam) over a checkpoint truncated INTO its data section:
// the header still parses so the loader commits to the tensor layout, but a weight read runs
// off the end of the byte source. That is what a dropped network stream looks like to the real
// ReadAt / mmap-bounds path, and it must error cleanly rather than panic or return a partial
// model.
func TestStreamingLoaderCleanErrorOnTruncatedCheckpoint(t *testing.T) {
	path, cfg := writeSingleFileSafetensors(t)
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	_, dataBase, err := parseSafetensorsHeader(blob)
	if err != nil {
		t.Fatalf("parse header: %v", err)
	}
	if int64(dataBase)+8 >= int64(len(blob)) {
		t.Fatalf("fixture data section too small to truncate meaningfully")
	}
	// Keep the full header + a sliver of data, then cut the rest: every tensor whose [start,end)
	// runs past the new size must be rejected by safetensorsDataBounds / a short ReadAt.
	if err := os.Truncate(path, int64(dataBase)+8); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	m, err := LoadSafetensorsQuant(path, cfg)
	if err == nil {
		t.Fatalf("expected a clean error loading a truncated checkpoint, got nil")
	}
	if m != nil {
		t.Fatalf("a failed load must return no model, got a non-nil *Model")
	}
}
