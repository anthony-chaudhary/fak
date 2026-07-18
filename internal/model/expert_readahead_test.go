package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// fusedExpertFixture builds a single fused [E, out, in] f32 expert tensor whose bytes are
// filled with their own file offset, so expert e's stride-byte sub-range is trivially
// verifiable, and returns the on-disk path plus the geometry readExpertSlice needs.
func fusedExpertFixture(t *testing.T, experts, out, in int) (path string, base, stride int64, data []byte) {
	t.Helper()
	stride = int64(out * in * 4) // f32
	total := int64(experts) * stride
	data = make([]byte, total)
	for i := range data {
		data[i] = byte(i)
	}
	name := "model.layers.0.mlp.experts.gate_up_proj"
	buf := tinySafetensorsBytes(t, map[string]tinySTTensor{
		name: {dtype: "F32", shape: []int{experts, out, in}, data: data},
	})
	path = filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write fused expert safetensors: %v", err)
	}
	// The fused tensor is the only tensor, so its data starts at dataBase (DataOffsets[0]==0).
	// Recover dataBase from a header parse of the raw bytes so base is derived, not assumed.
	hdr, dataBase, err := parseSafetensorsHeader(buf)
	if err != nil {
		t.Fatalf("parse header: %v", err)
	}
	var e stEntry
	if err := json.Unmarshal(hdr[name], &e); err != nil {
		t.Fatalf("decode entry: %v", err)
	}
	base = int64(dataBase) + int64(e.DataOffsets[0])
	return path, base, stride, data
}

// TestReadExpertSliceAvoidsReadAmplification is the #4359 compatibility witness: on the
// demand-paged (ReadAt) path a top-k route reads only the picked experts' k*stride bytes —
// byte-identical to the whole-tensor read then slice — instead of the whole E*stride layer.
func TestReadExpertSliceAvoidsReadAmplification(t *testing.T) {
	const experts, out, in = 8, 2, 1
	path, base, stride, data := fusedExpertFixture(t, experts, out, in)
	buf, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	rr := &v4ExpertSourceReaderAt{data: buf}
	sf, err := newSafetensorsFile(rr, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile: %v", err)
	}
	rr.dataBase = sf.dataBase
	if sf.data != nil {
		t.Fatalf("expected the ReadAt path (sf.data must be nil)")
	}

	// Baseline: reading the whole fused block moves the entire E*stride layer in one ReadAt.
	whole, err := sf.readExpertSlice(base, 0, 1, int64(experts)*stride)
	if err != nil {
		t.Fatalf("whole-block read: %v", err)
	}
	wholeBytes := rr.tensorBytes
	if int64(len(whole)) != int64(experts)*stride || wholeBytes != int64(experts)*stride {
		t.Fatalf("whole-block read moved %d bytes (len %d), want %d", wholeBytes, len(whole), int64(experts)*stride)
	}

	// A top-k route: read only the picked experts, each exactly stride bytes, byte-identical
	// to the corresponding slice of the whole block.
	rr.tensorReads, rr.tensorBytes = 0, 0
	picks := []int{1, 4, 6}
	for _, e := range picks {
		got, err := sf.readExpertSlice(base, e, experts, stride)
		if err != nil {
			t.Fatalf("readExpertSlice(e=%d): %v", e, err)
		}
		want := data[int64(e)*stride : int64(e+1)*stride]
		if string(got) != string(want) {
			t.Fatalf("expert %d bytes=%v, want %v", e, got, want)
		}
	}
	pickedBytes := rr.tensorBytes
	if want := int64(len(picks)) * stride; pickedBytes != want || rr.tensorReads != len(picks) {
		t.Fatalf("top-k route moved %d bytes over %d reads, want %d over %d", pickedBytes, rr.tensorReads, want, len(picks))
	}
	if pickedBytes >= wholeBytes {
		t.Fatalf("read amplification NOT avoided: top-k moved %d bytes, whole block %d", pickedBytes, wholeBytes)
	}
	t.Logf("read-amplification avoidance: top-%d/%d route moved %d/%d bytes (%.0f%% of the layer)",
		len(picks), experts, pickedBytes, wholeBytes, 100*float64(pickedBytes)/float64(wholeBytes))
}

// TestExpertSliceBoundsAndWillneed pins the bounds rejections and that the WILLNEED readahead
// is a safe no-op on the ReadAt path but actually fires on the mmap path where available.
func TestExpertSliceBoundsAndWillneed(t *testing.T) {
	const experts, out, in = 8, 2, 1
	path, base, stride, _ := fusedExpertFixture(t, experts, out, in)
	buf, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	sf, err := newSafetensorsFile(&v4ExpertSourceReaderAt{data: buf}, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile: %v", err)
	}

	// Out-of-range expert index and an overrunning stride are both refused.
	if _, err := sf.readExpertSlice(base, experts, experts, stride); err == nil {
		t.Fatal("readExpertSlice accepted an out-of-range expert index")
	}
	if _, err := sf.readExpertSlice(base, 0, 1, int64(experts)*stride+4); err == nil {
		t.Fatal("readExpertSlice accepted a slice that overruns the data region")
	}

	// ReadAt path: no mapped region to advise, so the hint is a no-op returning false.
	if sf.willneedExpertSlice(base, 0, experts, stride) {
		t.Fatal("willneedExpertSlice fired on the ReadAt path (sf.data is nil)")
	}

	// mmap path (unix hosts): the hint actually fires; the slice read is byte-identical and
	// zero-copy. On platforms without mmap (e.g. native Windows) openSafetensorsFileMmap
	// returns errMmapUnsupported and this leg is skipped — the ReadAt legs above still gate.
	msf, err := openSafetensorsFileMmap(path)
	if err != nil {
		t.Logf("mmap unsupported (%v); skipping the WILLNEED/zero-copy leg", err)
		return
	}
	defer msf.Close()
	if msf.data == nil {
		t.Fatal("mmap open returned no mapped bytes")
	}
	if !msf.willneedExpertSlice(base, 1, experts, stride) {
		t.Fatal("willneedExpertSlice did not fire on the mmap path")
	}
	got, err := msf.readExpertSlice(base, 1, experts, stride)
	if err != nil {
		t.Fatalf("mmap readExpertSlice: %v", err)
	}
	if int64(len(got)) != stride {
		t.Fatalf("mmap slice len=%d, want %d", len(got), stride)
	}
}
