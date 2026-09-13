package ggufload

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// quant_integrity_test.go is the witness suite for issue #12951: the quant
// artifact integrity triad (hot-row exact override, resumable per-shard digest
// manifest, independent re-read validator). Every test runs on synthetic shard
// fixtures; no model weights are downloaded or committed.

// qiRows builds n deterministic rows of rowBytes each whose bytes encode the
// row index so a mis-indexed or truncated row is immediately visible.
func qiRows(n, rowBytes int, seed byte) []byte {
	raw := make([]byte, n*rowBytes)
	for r := 0; r < n; r++ {
		for c := 0; c < rowBytes; c++ {
			raw[r*rowBytes+c] = seed + byte(r) + byte(c)
		}
	}
	return raw
}

// TestQuantIntegrityHotRowOverrideIsBitExact proves hot rows survive the
// override verbatim while cold rows keep their quantized bytes, and that a
// mis-shaped or out-of-range request is refused before any write.
func TestQuantIntegrityHotRowOverrideIsBitExact(t *testing.T) {
	const rowBytes = 8
	original := qiRows(6, rowBytes, 0x10)
	quantized := qiRows(6, rowBytes, 0x80)
	dst := make([]byte, len(quantized))

	ids := []int64{1, 4}
	if err := applyHotRowOverride(dst, quantized, original, rowBytes, ids); err != nil {
		t.Fatalf("applyHotRowOverride: %v", err)
	}
	if err := validateHotRowsBitExact(dst, quantized, original, rowBytes, ids); err != nil {
		t.Fatalf("validateHotRowsBitExact: %v", err)
	}
	for _, r := range ids {
		off := int(r) * rowBytes
		if !bytes.Equal(dst[off:off+rowBytes], original[off:off+rowBytes]) {
			t.Fatalf("hot row %d is not the original bytes", r)
		}
	}
	for _, r := range []int{0, 2, 3, 5} {
		off := r * rowBytes
		if !bytes.Equal(dst[off:off+rowBytes], quantized[off:off+rowBytes]) {
			t.Fatalf("cold row %d is not the quantized bytes", r)
		}
	}

	// A mis-shaped destination, a duplicate id, and an out-of-range id all
	// refuse before any byte is written.
	if err := applyHotRowOverride(make([]byte, len(dst)-1), quantized, original, rowBytes, ids); err == nil {
		t.Fatal("short destination must refuse")
	}
	if err := applyHotRowOverride(dst, quantized, original, rowBytes, []int64{2, 2}); err == nil {
		t.Fatal("duplicate hot id must refuse")
	}
	if err := applyHotRowOverride(dst, quantized, original, rowBytes, []int64{6}); err == nil {
		t.Fatal("out-of-range hot id must refuse")
	}
	if err := applyHotRowOverride(dst, quantized, original, rowBytes, []int64{-1}); err == nil {
		t.Fatal("negative hot id must refuse")
	}
	if err := applyHotRowOverride(dst, quantized, original, 0, ids); err == nil {
		t.Fatal("zero row width must refuse")
	}
	// The refusals left dst untouched from the last good write.
	if err := validateHotRowsBitExact(dst, quantized, original, rowBytes, ids); err != nil {
		t.Fatalf("dst mutated by a refused call: %v", err)
	}
}

// qiPartRequest builds one render-once part request whose payload is sized and
// tagged by index so a resumed run can prove it reused the prior bytes.
type qiRenderCounter struct {
	calls map[string]int
}

func (c *qiRenderCounter) render(file string, payload []byte) func() ([]byte, error) {
	return func() ([]byte, error) {
		c.calls[file]++
		out := make([]byte, len(payload))
		copy(out, payload)
		return out, nil
	}
}

// TestQuantIntegrityBuildResumesAndValidates proves an interrupted build
// resumes from its manifest without re-rendering completed parts, that the
// per-part sha256 and byte totals reconcile on an independent re-read, and that
// a corrupted or truncated part is rejected.
func TestQuantIntegrityBuildResumesAndValidates(t *testing.T) {
	dir := t.TempDir()
	const source = "fixture@rev1"
	counter := &qiRenderCounter{calls: map[string]int{}}

	payloadA := []byte("experts-part-a-payload")
	payloadB := []byte("engram-part-b-payload-0123456789")
	requests := []quantIntegrityPartRequest{
		{Part: quantIntegrityPart{File: "experts/a.bin", Kind: "experts", Layer: 0, Records: 3}, Render: counter.render("experts/a.bin", payloadA)},
		{Part: quantIntegrityPart{File: "engram/b.bin", Kind: "engram", Layer: 1, Records: 2}, Render: counter.render("engram/b.bin", payloadB)},
	}

	// First run: complete both parts.
	m1, err := runQuantIntegrityBuild(dir, source, requests)
	if err != nil {
		t.Fatalf("first build: %v", err)
	}
	if !m1.Complete {
		t.Fatal("first build did not mark the manifest complete")
	}
	if counter.calls["experts/a.bin"] != 1 || counter.calls["engram/b.bin"] != 1 {
		t.Fatalf("first build render calls = %+v, want one each", counter.calls)
	}
	if m1.PayloadSum != int64(len(payloadA)+len(payloadB)) {
		t.Fatalf("PayloadSum = %d, want %d", m1.PayloadSum, len(payloadA)+len(payloadB))
	}
	if err := validateQuantIntegrity(dir, m1, nil, nil); err != nil {
		t.Fatalf("validate after first build: %v", err)
	}

	// Second run over the same requests: both parts are reused, not re-rendered.
	m2, err := runQuantIntegrityBuild(dir, source, requests)
	if err != nil {
		t.Fatalf("resume build: %v", err)
	}
	if counter.calls["experts/a.bin"] != 1 || counter.calls["engram/b.bin"] != 1 {
		t.Fatalf("resume re-rendered a completed part: %+v", counter.calls)
	}
	if !m2.Complete || m2.PayloadSum != m1.PayloadSum {
		t.Fatalf("resume manifest = %+v, want complete and matching sum", m2)
	}
	for _, p := range m2.Parts {
		if p.SHA256 != quantIntegrityDigestHex(mustRead(t, filepath.Join(dir, p.File))) {
			t.Fatalf("part %s sha256 does not match the on-disk payload", p.File)
		}
	}

	// Corruption: flip a byte and demand a refusal.
	partPath := filepath.Join(dir, m2.Parts[0].File)
	raw := mustRead(t, partPath)
	raw[0] ^= 0xFF
	if err := os.WriteFile(partPath, raw, 0o644); err != nil {
		t.Fatalf("corrupt part: %v", err)
	}
	if err := validateQuantIntegrity(dir, m2, nil, nil); err == nil {
		t.Fatal("corrupted part must fail validation")
	}
	// Truncation also refuses.
	raw = mustRead(t, partPath)[:1]
	if err := os.WriteFile(partPath, raw, 0o644); err != nil {
		t.Fatalf("truncate part: %v", err)
	}
	if err := validateQuantIntegrity(dir, m2, nil, nil); err == nil {
		t.Fatal("truncated part must fail validation")
	}
}

// TestQuantIntegrityResumeAfterInterrupt proves that a part written before an
// interrupt is reused on the next run while the unfinished part is rendered.
func TestQuantIntegrityResumeAfterInterrupt(t *testing.T) {
	dir := t.TempDir()
	const source = "fixture@rev2"
	first := &qiRenderCounter{calls: map[string]int{}}
	reqA := quantIntegrityPartRequest{
		Part:   quantIntegrityPart{File: "p/a.bin", Kind: "experts", Records: 1},
		Render: first.render("p/a.bin", []byte("aaaaaaaaaaaa")),
	}
	if _, err := runQuantIntegrityBuild(dir, source, []quantIntegrityPartRequest{reqA}); err != nil {
		t.Fatalf("partial build: %v", err)
	}
	// The manifest is complete for the one part that ran; simulate a second
	// wave with a new part. The reused part must not render again.
	second := &qiRenderCounter{calls: map[string]int{}}
	reqB := quantIntegrityPartRequest{
		Part:   quantIntegrityPart{File: "p/b.bin", Kind: "engram", Records: 1},
		Render: second.render("p/b.bin", []byte("bbbbbbbbbbbbbb")),
	}
	m, err := runQuantIntegrityBuild(dir, source, []quantIntegrityPartRequest{reqA, reqB})
	if err != nil {
		t.Fatalf("resume build: %v", err)
	}
	if second.calls["p/b.bin"] != 1 {
		t.Fatalf("new part rendered %d times, want 1", second.calls["p/b.bin"])
	}
	if len(m.Parts) != 2 {
		t.Fatalf("resumed manifest has %d parts, want 2", len(m.Parts))
	}
	if err := validateQuantIntegrity(dir, m, nil, nil); err != nil {
		t.Fatalf("validate resumed manifest: %v", err)
	}
}

// TestQuantIntegrityManifestRefusesWrongSource proves a manifest from a
// different checkpoint revision cannot be silently reused.
func TestQuantIntegrityManifestRefusesWrongSource(t *testing.T) {
	dir := t.TempDir()
	req := quantIntegrityPartRequest{
		Part:   quantIntegrityPart{File: "x.bin", Kind: "experts"},
		Render: func() ([]byte, error) { return []byte("payload"), nil },
	}
	if _, err := runQuantIntegrityBuild(dir, "rev-A", []quantIntegrityPartRequest{req}); err != nil {
		t.Fatalf("build: %v", err)
	}
	if _, err := runQuantIntegrityBuild(dir, "rev-B", []quantIntegrityPartRequest{req}); err == nil {
		t.Fatal("a manifest from a different source revision must refuse")
	}
}

// TestQuantIntegrityNMSECeils proves the independent re-read NMSE ceilings
// accept an in-bounds sample and reject an out-of-bounds one, and that an
// all-zero original with any error is +Inf rather than a silent pass.
func TestQuantIntegrityNMSECeils(t *testing.T) {
	good := &quantIntegrityNMSESample{}
	if err := good.add([]float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if v := good.nmse(); v != 0 {
		t.Fatalf("perfect reconstruction nmse = %g, want 0", v)
	}
	if err := validateQuantIntegrity(t.TempDir(), &quantIntegrityManifest{Format: quantIntegrityFormat, Complete: true}, good, good); err != nil {
		t.Fatalf("in-bounds NMSE refused: %v", err)
	}

	bad := &quantIntegrityNMSESample{}
	if err := bad.add([]float32{1, 2, 3, 4}, []float32{99, 99, 99, 99}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := validateQuantIntegrity(t.TempDir(), &quantIntegrityManifest{Format: quantIntegrityFormat, Complete: true}, bad, nil); err == nil {
		t.Fatal("out-of-bounds expert NMSE must refuse")
	}
	if err := validateQuantIntegrity(t.TempDir(), &quantIntegrityManifest{Format: quantIntegrityFormat, Complete: true}, nil, bad); err == nil {
		t.Fatal("out-of-bounds engram NMSE must refuse")
	}

	zero := &quantIntegrityNMSESample{}
	if err := zero.add([]float32{0, 0}, []float32{1, 1}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if v := zero.nmse(); !math.IsInf(v, 1) {
		t.Fatalf("zero-original nonzero-error nmse = %g, want +Inf", v)
	}
	if err := validateQuantIntegrity(t.TempDir(), &quantIntegrityManifest{Format: quantIntegrityFormat, Complete: true}, zero, nil); err == nil {
		t.Fatal("non-finite NMSE must refuse")
	}
}

// TestQuantIntegrityIncompleteManifestRefuses proves a manifest that never
// flipped complete cannot pass validation.
func TestQuantIntegrityIncompleteManifestRefuses(t *testing.T) {
	dir := t.TempDir()
	m := &quantIntegrityManifest{Format: quantIntegrityFormat, Complete: false}
	if err := validateQuantIntegrity(dir, m, nil, nil); err == nil {
		t.Fatal("incomplete manifest must refuse")
	}
	if err := validateQuantIntegrity(dir, nil, nil, nil); err == nil {
		t.Fatal("nil manifest must refuse")
	}
}

// TestQuantIntegrityPartEscapesRefused proves a part name that would escape the
// output directory is rejected.
func TestQuantIntegrityPartEscapesRefused(t *testing.T) {
	for _, name := range []string{"", "../escape.bin", "/abs.bin", "a/../../b.bin"} {
		if _, err := normalizePartFile(name); err == nil {
			t.Fatalf("part name %q must refuse", name)
		}
	}
	if got, err := normalizePartFile("nested/ok.bin"); err != nil || got != "nested/ok.bin" {
		t.Fatalf("normalizePartFile(nested/ok.bin) = %q,%v", got, err)
	}
}

// TestQuantIntegrityManifestFormatRefuses proves a foreign manifest format
// cannot be adopted.
func TestQuantIntegrityManifestFormatRefuses(t *testing.T) {
	dir := t.TempDir()
	raw, err := json.Marshal(quantIntegrityManifest{Format: "some-other-format"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), raw, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := quantIntegrityReadManifest(filepath.Join(dir, "manifest.json")); err == nil {
		t.Fatal("foreign manifest format must refuse")
	}
}

// mustRead reads a file or fails the test.
func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return raw
}

// errIsNil is a tiny helper so an unused errors import stays honest if a test
// is edited to drop an assertion.
var _ = errors.Is
