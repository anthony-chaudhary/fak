package model

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// safetensors_mmap_test.go — the issue-#475 acceptance gate. The single-file loaders now
// prefer a read-only memory map and slice each tensor's [start,end) zero-copy out of it, so a
// single-file checkpoint is never fully resident in the process heap (the single-file analogue
// of LoadSafetensorsQuantDir's per-shard free). These tests prove the load is BYTE-IDENTICAL
// to the historical os.ReadFile readers across the zero-copy data path AND the ReadAt fallback,
// on a synthetic in-test fixture so the claim is witnessed on whatever platform runs the suite.
//
// The zero-copy slicing branch (safetensorsFile.data != nil) is platform-independent — only
// the byte SOURCE differs (an OS map on darwin/linux/BSD, an os.ReadFile buffer in the
// data-backed test opener, both >=8-byte aligned and read-only-equivalent). So the data-backed
// leg run-witnesses the exact slicing logic on win32 (where production uses the ReadAt
// fallback), while the OS-mmap leg adds an end-to-end syscall-backed witness on unix.

// writeSingleFileSafetensors writes a one-layer dense checkpoint as a single little-endian
// model.safetensors (the format openSafetensorsFile parses), combining the big matmul weights
// the quant path consumes with the small f32 tensors it keeps resident.
func writeSingleFileSafetensors(t *testing.T) (path string, cfg Config) {
	t.Helper()
	const H, I, V = 64, 96, 32 // hidden / intermediate multiples of 32 (Q8_0 reduction dim), small vocab
	cfg = Config{
		HiddenSize: H, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: I, VocabSize: V, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true,
	}
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	tensors := []stTensor{
		{"model.layers.0.self_attn.q_proj.weight", []int{nH * hd, H}, rampF32(nH*hd*H, 1)},
		{"model.layers.0.self_attn.k_proj.weight", []int{nKV * hd, H}, rampF32(nKV*hd*H, 2)},
		{"model.layers.0.self_attn.v_proj.weight", []int{nKV * hd, H}, rampF32(nKV*hd*H, 3)},
		{"model.layers.0.self_attn.o_proj.weight", []int{H, nH * hd}, rampF32(H*nH*hd, 4)},
		{"model.layers.0.mlp.gate_proj.weight", []int{I, H}, rampF32(I*H, 5)},
		{"model.layers.0.mlp.up_proj.weight", []int{I, H}, rampF32(I*H, 6)},
		{"model.layers.0.mlp.down_proj.weight", []int{H, I}, rampF32(H*I, 7)},
		{"model.embed_tokens.weight", []int{V, H}, rampF32(V*H, 8)},
		{"model.layers.0.input_layernorm.weight", []int{H}, ones(H)},
		{"model.layers.0.post_attention_layernorm.weight", []int{H}, ones(H)},
		{"model.norm.weight", []int{H}, ones(H)},
	}
	path = filepath.Join(t.TempDir(), "model.safetensors")
	writeSafetensorsShard(t, path, tensors)
	return path, cfg
}

// openSafetensorsFileDataBackedForTest forces the zero-copy sf.data branch of tensorBytes on
// ANY platform by backing it with an os.ReadFile buffer instead of a real OS map. The byte
// source is the only thing that differs from a real mmap (alignment and read-only-ness are
// equivalent), so this run-witnesses the exact slicing logic the unix mmap path uses — even on
// win32, where the production opener uses the ReadAt fallback.
func openSafetensorsFileDataBackedForTest(path string) (*safetensorsFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return newSafetensorsFileMmap(data, closerFunc(func() error { return nil }))
}

// loadRegularViaOpener runs the streaming f32-resident decode through a specific opener,
// mirroring readFileLoadSafetensorsForTest's whole-file reference exactly (same tensor walk,
// same decode) but sourcing bytes through the safetensorsFile seam.
func loadRegularViaOpener(t *testing.T, path string, cfg Config, open func(string) (*safetensorsFile, error)) *Model {
	t.Helper()
	sf, err := open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer sf.Close()
	man := map[string]tensorMeta{}
	var raw []byte
	off := 0
	if err := appendSafetensorsFileInto(sf, man, &raw, &off, cfg); err != nil {
		t.Fatalf("appendSafetensorsFileInto: %v", err)
	}
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

func assertQuantModelsEqual(t *testing.T, label string, want, got *Model) {
	t.Helper()
	if len(got.q8w) != len(want.q8w) {
		t.Fatalf("%s: q8w count %d != %d", label, len(got.q8w), len(want.q8w))
	}
	for name, a := range want.q8w {
		b, ok := got.q8w[name]
		if !ok {
			t.Fatalf("%s: missing q8 tensor %s", label, name)
		}
		if a.out != b.out || a.in != b.in || a.nblk != b.nblk {
			t.Fatalf("%s: %s shape (%d,%d,%d) != (%d,%d,%d)", label, name, a.out, a.in, a.nblk, b.out, b.in, b.nblk)
		}
		if len(a.q) != len(b.q) {
			t.Fatalf("%s: %s code len %d != %d", label, name, len(a.q), len(b.q))
		}
		for i := range a.q {
			if a.q[i] != b.q[i] {
				t.Fatalf("%s: %s code[%d] %d != %d", label, name, i, a.q[i], b.q[i])
			}
		}
		if len(a.d) != len(b.d) {
			t.Fatalf("%s: %s scale len %d != %d", label, name, len(a.d), len(b.d))
		}
		for i := range a.d {
			if math.Float32bits(a.d[i]) != math.Float32bits(b.d[i]) {
				t.Fatalf("%s: %s scale[%d] %v != %v", label, name, i, a.d[i], b.d[i])
			}
		}
	}
	if !bytes.Equal(want.raw, got.raw) {
		t.Fatalf("%s: small-f32 raw bytes differ: %d vs %d", label, len(want.raw), len(got.raw))
	}
}

// safetensorsOpeners is the set of single-file byte-source openers to prove byte-identical.
// The data-backed (zero-copy) and ReadAt legs run on every platform; the OS-mmap leg is added
// only where a real map is available, so a host without mmap (win32) never silently proves
// fallback-vs-fallback as if it were the mmap path.
func safetensorsOpeners(t *testing.T, path string) map[string]func(string) (*safetensorsFile, error) {
	t.Helper()
	openers := map[string]func(string) (*safetensorsFile, error){
		"ReadAt fallback": openSafetensorsFileReadAt,
		"zero-copy data":  openSafetensorsFileDataBackedForTest,
	}
	if sf, err := openSafetensorsFileMmap(path); err != nil {
		t.Logf("real OS mmap unavailable on %s/%s (%v) — production uses the ReadAt fallback here; zero-copy slicing still witnessed via the data-backed leg",
			runtime.GOOS, runtime.GOARCH, err)
	} else {
		if sf.data == nil {
			t.Fatal("openSafetensorsFileMmap returned a file with nil mmap data")
		}
		_ = sf.Close()
		openers["OS mmap"] = openSafetensorsFileMmap
	}
	return openers
}

// TestLoadSafetensorsMmapMatchesReadFile is the bit-identity gate for #475: every byte-source
// opener decodes/quantizes byte-for-byte identically to the os.ReadFile reference loaders, for
// the regular f32-resident path AND the quant-on-load path.
func TestLoadSafetensorsMmapMatchesReadFile(t *testing.T) {
	path, cfg := writeSingleFileSafetensors(t)
	openers := safetensorsOpeners(t, path)

	// ---- regular f32-resident path ----
	refReg, err := readFileLoadSafetensorsForTest(path, cfg)
	if err != nil {
		t.Fatalf("readFileLoadSafetensorsForTest: %v", err)
	}
	for label, open := range openers {
		got := loadRegularViaOpener(t, path, cfg, open)
		if !bytes.Equal(refReg.raw, got.raw) {
			t.Fatalf("regular %s: raw bytes differ from os.ReadFile reference (%d vs %d)", label, len(refReg.raw), len(got.raw))
		}
		assertModelRawEqual(t, refReg, got) // manifest + raw
	}

	// ---- quant-on-load path (where the single-file mmap win is largest) ----
	refQuant, err := readFileLoadSafetensorsQuantForTest(path, cfg)
	if err != nil {
		t.Fatalf("readFileLoadSafetensorsQuantForTest: %v", err)
	}
	for label, open := range openers {
		gotQ, err := loadSafetensorsQuantFile(path, cfg, open)
		if err != nil {
			t.Fatalf("loadSafetensorsQuantFile (%s): %v", label, err)
		}
		assertQuantModelsEqual(t, "quant "+label+" vs os.ReadFile reference", refQuant, gotQ)
	}
}

// TestLoadSafetensorsMmapDefaultPath confirms the production single-file quant loader
// (LoadSafetensorsQuant -> openSafetensorsFile, which prefers mmap then falls back) yields the
// same Q8 store as the explicit ReadAt path — i.e. the default opener's mmap preference changes
// no bytes on any platform.
func TestLoadSafetensorsMmapDefaultPath(t *testing.T) {
	path, cfg := writeSingleFileSafetensors(t)

	def, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	readAt, err := loadSafetensorsQuantFile(path, cfg, openSafetensorsFileReadAt)
	if err != nil {
		t.Fatalf("loadSafetensorsQuantFile (ReadAt): %v", err)
	}
	assertQuantModelsEqual(t, "LoadSafetensorsQuant default vs ReadAt", readAt, def)
}

// TestLoadSafetensorsSingleFilePeakRSS is the measurement leg. The resident-peak win is only
// observable on a real ~14 GB single-file checkpoint, which is not in the tree; this test
// names the artifact and skips when absent (it does NOT block the gate). Point it at a
// single-file checkpoint dir (a model.safetensors + config.json) via FAK_SINGLEFILE_CHECKPOINT
// to measure. The reported numbers are a Go-heap proxy (runtime.MemStats); the authoritative
// resident-peak figure needs an OS RSS reading (e.g. /usr/bin/time -l on darwin) on a
// darwin/linux run where the OS map is live — it is UNWITNESSED on this win32 host.
func TestLoadSafetensorsSingleFilePeakRSS(t *testing.T) {
	dir := os.Getenv("FAK_SINGLEFILE_CHECKPOINT")
	if dir == "" {
		t.Skip("set FAK_SINGLEFILE_CHECKPOINT to a dir holding a single-file model.safetensors (~14GB) + config.json to measure single-file mmap peak RSS")
	}
	stPath := filepath.Join(dir, "model.safetensors")
	if !fileExists(stPath) {
		t.Skipf("no model.safetensors under %s — skipping single-file peak-RSS measurement", dir)
	}
	var cfg Config
	if err := readJSON(filepath.Join(dir, "config.json"), &cfg); err != nil {
		t.Skipf("no config.json under %s (%v) — skipping single-file peak-RSS measurement", dir, err)
	}

	measure := func(open func(string) (*safetensorsFile, error)) float64 {
		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)
		m, err := loadSafetensorsQuantFile(stPath, cfg, open)
		if err != nil {
			t.Fatalf("loadSafetensorsQuantFile: %v", err)
		}
		var after runtime.MemStats
		runtime.ReadMemStats(&after)
		runtime.KeepAlive(m)
		return float64(int64(after.TotalAlloc)-int64(before.TotalAlloc)) / (1 << 20)
	}

	readAtMB := measure(openSafetensorsFileReadAt)
	if sf, err := openSafetensorsFileMmap(stPath); err != nil {
		t.Logf("single-file %s: ReadAt total Go-heap alloc %.0f MB; OS mmap unavailable on %s (%v) — resident-peak win UNWITNESSED here",
			stPath, readAtMB, runtime.GOOS, err)
		return
	} else {
		_ = sf.Close()
	}
	mmapMB := measure(openSafetensorsFileMmap)
	t.Logf("single-file %s: total Go-heap alloc during load — ReadAt %.0f MB, OS mmap %.0f MB (heap proxy; OS RSS needs /usr/bin/time)",
		stPath, readAtMB, mmapMB)
}
