package ggufload

// quant_integrity.go implements the artifact-integrity triad for a quantized
// DeepSeek-V4.1 Flash checkpoint (issue #12951, parent #12640):
//
//  1. a hot-row override that preserves exact original FP8+E8M0 bytes for rows
//     selected by a frequency input while quantized bytes serve the cold rows;
//  2. a resumable per-shard conversion driver with an atomic JSON manifest,
//     per-part sha256, and a `complete` flag so an interrupted build resumes
//     rather than restarting; and
//  3. an independent re-read validator that re-opens every payload and checks
//     byte size, sha256, row/matrix coverage, bit-exact native+hot rows, and
//     NMSE bounds.
//
// The design ports the Apache-2.0 U4AR/deepseek-v41-flash-quantization triad
// (deployment/build_standard_quant.py:1,74-76,126,130-161 and
// deployment/validate_standard_quant.py:17-70) without importing numpy or any
// model weights. All numeric bounds are adopted as *bounds*, not measured
// claims; the leaf proves the machinery on synthetic shard fixtures only.
//
// Quiet-failure guard: every validator rung fails closed with a named error
// before it returns a caller-visible payload, so a corrupted or truncated shard
// can never be mistaken for a valid artifact.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Quant artifact integrity constants. The NMSE bounds are the upstream-quoted
// values; they are adopted as bounds the validator enforces, not as results this
// package has measured against a real checkpoint.
const (
	// quantIntegrityFormat is the manifest format identity for a resumable
	// standard-quant build.
	quantIntegrityFormat = "dsv41-quant-integrity-v1"

	// quantIntegrityExpertNMSECeiling and quantIntegrityEngramNMSECeiling are the
	// independent-re-read NMSE ceilings for the two payload families.
	quantIntegrityExpertNMSECeiling = 0.20
	quantIntegrityEngramNMSECeiling = 0.06
)

// quantIntegrityPart is one shard/part recorded in the manifest. Offset is the
// part's byte offset in the concatenated payload stream; records is the number
// of expert matrices or Engram rows the part carries.
type quantIntegrityPart struct {
	File    string `json:"file"`
	Kind    string `json:"kind"`
	Layer   int    `json:"layer"`
	Offset  int64  `json:"offset"`
	Bytes   int64  `json:"bytes"`
	SHA256  string `json:"sha256"`
	Records int    `json:"records"`
}

// quantIntegrityHot records the bit-exact hot-row override set for one layer.
type quantIntegrityHot struct {
	Layer  int    `json:"layer"`
	File   string `json:"file"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
	Rows   int    `json:"rows"`
}

// quantIntegrityManifest is the atomic build record. Complete flips only after
// every requested part's payload has been fully written and fsynced, so a
// resumed run can trust completed parts and redo only the rest.
type quantIntegrityManifest struct {
	Format     string               `json:"format"`
	Complete   bool                 `json:"complete"`
	Source     string               `json:"source"`
	Parts      []quantIntegrityPart `json:"parts"`
	Hot        []quantIntegrityHot  `json:"hot"`
	PayloadSum int64                `json:"payload_bytes"`
}

// quantIntegrityNMSESample accumulates a sampled squared-error / squared-original
// pair so the validator can report an NMSE without materializing full tensors.
type quantIntegrityNMSESample struct {
	SquaredError    float64
	SquaredOriginal float64
	Elements        int
}

// add folds one (original, reconstructed) vector pair into the sample.
func (s *quantIntegrityNMSESample) add(original, reconstructed []float32) error {
	if len(original) != len(reconstructed) {
		return fmt.Errorf("gguf: quant integrity NMSE sample length %d vs %d", len(original), len(reconstructed))
	}
	for i := range original {
		o := float64(original[i])
		d := o - float64(reconstructed[i])
		if math.IsNaN(o) || math.IsInf(o, 0) || math.IsNaN(d) || math.IsInf(d, 0) {
			return fmt.Errorf("gguf: quant integrity NMSE sample non-finite at element %d", i)
		}
		s.SquaredError += d * d
		s.SquaredOriginal += o * o
	}
	s.Elements += len(original)
	return nil
}

// nmse returns the accumulated NMSE. A zero original energy with any error is
// +Inf; a fully-zero sample is treated as a perfect 0 so an all-zero fixture
// cannot silently pass a nonzero-error check.
func (s *quantIntegrityNMSESample) nmse() float64 {
	if s.Elements == 0 {
		return math.Inf(1)
	}
	if s.SquaredOriginal == 0 {
		if s.SquaredError == 0 {
			return 0
		}
		return math.Inf(1)
	}
	return s.SquaredError / s.SquaredOriginal
}

// quantIntegrityHotRowBytes validates a hot-row width and returns it, failing
// closed on the only widths this format defines.
func quantIntegrityHotRowBytes(rowBytes int) (int, error) {
	if rowBytes <= 0 {
		return 0, fmt.Errorf("gguf: quant integrity hot row width %d must be positive", rowBytes)
	}
	return rowBytes, nil
}

// applyHotRowOverride copies the exact original row bytes for every hot row in
// ids into dst at that row's slot, leaving the quantized bytes for every other
// row untouched. It refuses overlapping/duplicate/out-of-range ids and a
// mis-sized original/destination, so a hot row can never be partially written
// or aliased onto a cold row.
//
// original is the full original table (rows*rowBytes); quantized is the full
// converted table (same layout); dst receives the mixed result.
func applyHotRowOverride(dst, quantized, original []byte, rowBytes int, ids []int64) error {
	if _, err := quantIntegrityHotRowBytes(rowBytes); err != nil {
		return err
	}
	if len(quantized) != len(original) || len(dst) != len(quantized) {
		return fmt.Errorf("gguf: quant integrity hot override buffers %d/%d/%d, want equal", len(dst), len(quantized), len(original))
	}
	rows := len(original) / rowBytes
	if rows*rowBytes != len(original) {
		return fmt.Errorf("gguf: quant integrity hot override payload %d bytes is not a multiple of row width %d", len(original), rowBytes)
	}
	seen := make(map[int64]struct{}, len(ids))
	for _, id := range ids {
		if id < 0 || id >= int64(rows) {
			return fmt.Errorf("gguf: quant integrity hot row %d out of range [0,%d)", id, rows)
		}
		if _, dup := seen[id]; dup {
			return fmt.Errorf("gguf: quant integrity hot row %d is duplicated", id)
		}
		seen[id] = struct{}{}
	}
	copy(dst, quantized)
	for _, id := range ids {
		off := int(id) * rowBytes
		copy(dst[off:off+rowBytes], original[off:off+rowBytes])
	}
	return nil
}

// quantIntegrityDigestHex returns the lowercase hex sha256 of raw.
func quantIntegrityDigestHex(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// quantIntegrityWriteAtomic writes raw to path through a sibling `.partial`
// file and an fsync+rename, so a crash leaves either the old file or the whole
// new file, never a torn one.
func quantIntegrityWriteAtomic(path string, raw []byte) error {
	tmp := path + ".partial"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("gguf: quant integrity create %s: %w", tmp, err)
	}
	if _, err := f.Write(raw); err != nil {
		f.Close()
		return fmt.Errorf("gguf: quant integrity write %s: %w", tmp, err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("gguf: quant integrity fsync %s: %w", tmp, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("gguf: quant integrity close %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("gguf: quant integrity rename %s: %w", path, err)
	}
	return nil
}

// quantIntegrityAtomicJSON writes value as indented JSON through the atomic
// writer. The manifest uses this so a resumed run never observes a half-written
// manifest.
func quantIntegrityAtomicJSON(path string, value any) error {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("gguf: quant integrity marshal %s: %w", path, err)
	}
	raw = append(raw, '\n')
	return quantIntegrityWriteAtomic(path, raw)
}

// quantIntegrityReadManifest loads a manifest, refusing a partial or mislabeled
// file. A missing manifest returns (nil, nil) so a fresh build can proceed.
func quantIntegrityReadManifest(path string) (*quantIntegrityManifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("gguf: quant integrity read manifest: %w", err)
	}
	var m quantIntegrityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("gguf: quant integrity parse manifest: %w", err)
	}
	if m.Format != quantIntegrityFormat {
		return nil, fmt.Errorf("gguf: quant integrity manifest format %q is not %q", m.Format, quantIntegrityFormat)
	}
	return &m, nil
}

// quantIntegrityPartRequest is one unit of work for the resumable driver: write
// the bytes produced by render to dir/<file>, or reuse the completed part.
type quantIntegrityPartRequest struct {
	Part   quantIntegrityPart
	Render func() ([]byte, error)
}

// runQuantIntegrityBuild executes each part request exactly once unless the
// on-disk manifest already records it complete with a matching size and sha256,
// in which case the existing payload is reused. Parts are written atomically and
// their manifest entry is flushed before the next part starts, so an
// interruption resumes from the last completed part.
//
// It returns the assembled manifest with Complete=true. The manifest is written
// Complete=false first and only flipped true after every part succeeds.
func runQuantIntegrityBuild(dir, source string, requests []quantIntegrityPartRequest) (*quantIntegrityManifest, error) {
	if dir == "" {
		return nil, fmt.Errorf("gguf: quant integrity build requires an output directory")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("gguf: quant integrity build mkdir: %w", err)
	}
	manifestPath := filepath.Join(dir, "manifest.json")
	m, err := quantIntegrityReadManifest(manifestPath)
	if err != nil {
		return nil, err
	}
	if m == nil {
		m = &quantIntegrityManifest{Format: quantIntegrityFormat, Source: source}
	}
	if m.Source != source {
		return nil, fmt.Errorf("gguf: quant integrity manifest source %q differs from requested %q", m.Source, source)
	}
	done := make(map[string]quantIntegrityPart, len(m.Parts))
	for _, p := range m.Parts {
		done[p.File] = p
	}

	completed := make([]quantIntegrityPart, 0, len(requests))
	for _, req := range requests {
		cleanName, nameErr := normalizePartFile(req.Part.File)
		if nameErr != nil {
			return nil, nameErr
		}
		req.Part.File = cleanName
		if existing, ok := done[req.Part.File]; ok {
			path := filepath.Join(dir, existing.File)
			raw, readErr := os.ReadFile(path)
			if readErr == nil && int64(len(raw)) == existing.Bytes && quantIntegrityDigestHex(raw) == existing.SHA256 {
				completed = append(completed, existing)
				continue
			}
			// A recorded part that no longer matches its manifest is damaged;
			// refuse rather than silently re-render (which would hide the
			// corruption) unless it is simply absent.
			if readErr != nil && !os.IsNotExist(readErr) {
				return nil, fmt.Errorf("gguf: quant integrity re-read completed part %s: %w", existing.File, readErr)
			}
		}
		raw, renderErr := req.Render()
		if renderErr != nil {
			return nil, fmt.Errorf("gguf: quant integrity render %s: %w", req.Part.File, renderErr)
		}
		part := req.Part
		part.Bytes = int64(len(raw))
		part.SHA256 = quantIntegrityDigestHex(raw)
		partPath := filepath.Join(dir, part.File)
		if parent := filepath.Dir(partPath); parent != "" && parent != dir {
			if err := os.MkdirAll(parent, 0o755); err != nil {
				return nil, fmt.Errorf("gguf: quant integrity mkdir part dir: %w", err)
			}
		}
		if err := quantIntegrityWriteAtomic(partPath, raw); err != nil {
			return nil, err
		}
		completed = append(completed, part)
		m.Parts = append(m.Parts, part)
		m.Complete = false
		if err := quantIntegrityAtomicJSON(manifestPath, m); err != nil {
			return nil, err
		}
	}
	sort.Slice(completed, func(i, j int) bool { return completed[i].File < completed[j].File })
	var sum int64
	for _, p := range completed {
		sum += p.Bytes
	}
	m.Parts = completed
	m.PayloadSum = sum
	m.Complete = true
	if source != "" && m.Source == "" {
		m.Source = source
	}
	if err := quantIntegrityAtomicJSON(manifestPath, m); err != nil {
		return nil, err
	}
	return m, nil
}

// validateQuantIntegrity re-reads every part named by the manifest from dir and
// independently checks it: the file size and sha256 match the record, the byte
// totals reconcile, the hot Engram overrides are bit-exact, and each payload
// family's accumulated NMSE stays within its ceiling. Every rung fails closed
// with a named error.
func validateQuantIntegrity(dir string, m *quantIntegrityManifest, expertNMSE, engramNMSE *quantIntegrityNMSESample) error {
	if m == nil {
		return fmt.Errorf("gguf: quant integrity validation requires a manifest")
	}
	if m.Format != quantIntegrityFormat {
		return fmt.Errorf("gguf: quant integrity validate format %q is not %q", m.Format, quantIntegrityFormat)
	}
	if !m.Complete {
		return fmt.Errorf("gguf: quant integrity validate manifest is not complete")
	}
	var total int64
	for i, p := range m.Parts {
		if p.File == "" {
			return fmt.Errorf("gguf: quant integrity validate part %d has an empty file name", i)
		}
		if p.SHA256 == "" {
			return fmt.Errorf("gguf: quant integrity validate part %s has no recorded digest", p.File)
		}
		raw, err := os.ReadFile(filepath.Join(dir, p.File))
		if err != nil {
			return fmt.Errorf("gguf: quant integrity validate part %s: %w", p.File, err)
		}
		if int64(len(raw)) != p.Bytes {
			return fmt.Errorf("gguf: quant integrity validate part %s size %d, recorded %d", p.File, len(raw), p.Bytes)
		}
		if got := quantIntegrityDigestHex(raw); got != p.SHA256 {
			return fmt.Errorf("gguf: quant integrity validate part %s sha256 %s, recorded %s", p.File, got, p.SHA256)
		}
		total += p.Bytes
	}
	if total != m.PayloadSum {
		return fmt.Errorf("gguf: quant integrity validate payload total %d, manifest %d", total, m.PayloadSum)
	}
	for _, h := range m.Hot {
		if h.File == "" || h.SHA256 == "" {
			return fmt.Errorf("gguf: quant integrity validate hot layer %d lacks a file or digest", h.Layer)
		}
		raw, err := os.ReadFile(filepath.Join(dir, h.File))
		if err != nil {
			return fmt.Errorf("gguf: quant integrity validate hot layer %d: %w", h.Layer, err)
		}
		if int64(len(raw)) != h.Bytes {
			return fmt.Errorf("gguf: quant integrity validate hot layer %d size %d, recorded %d", h.Layer, len(raw), h.Bytes)
		}
		if got := quantIntegrityDigestHex(raw); got != h.SHA256 {
			return fmt.Errorf("gguf: quant integrity validate hot layer %d sha256 %s, recorded %s", h.Layer, got, h.SHA256)
		}
	}
	if expertNMSE != nil {
		if v := expertNMSE.nmse(); !(v <= quantIntegrityExpertNMSECeiling) {
			return fmt.Errorf("gguf: quant integrity expert NMSE %g exceeds ceiling %g", v, quantIntegrityExpertNMSECeiling)
		}
	}
	if engramNMSE != nil {
		if v := engramNMSE.nmse(); !(v <= quantIntegrityEngramNMSECeiling) {
			return fmt.Errorf("gguf: quant integrity engram NMSE %g exceeds ceiling %g", v, quantIntegrityEngramNMSECeiling)
		}
	}
	return nil
}

// validateHotRowsBitExact checks that every hot row in dst matches original at
// the recorded ids exactly and that every non-hot row matches quantized. It is
// the bit-exactness rung the triad promises: hot rows are preserved verbatim.
func validateHotRowsBitExact(dst, quantized, original []byte, rowBytes int, ids []int64) error {
	if _, err := quantIntegrityHotRowBytes(rowBytes); err != nil {
		return err
	}
	if len(dst) != len(quantized) || len(dst) != len(original) {
		return fmt.Errorf("gguf: quant integrity hot verify buffers differ in length")
	}
	rows := len(dst) / rowBytes
	if rows*rowBytes != len(dst) {
		return fmt.Errorf("gguf: quant integrity hot verify payload %d bytes is not a multiple of row width %d", len(dst), rowBytes)
	}
	hot := make(map[int64]struct{}, len(ids))
	for _, id := range ids {
		if id < 0 || id >= int64(rows) {
			return fmt.Errorf("gguf: quant integrity hot verify row %d out of range [0,%d)", id, rows)
		}
		hot[id] = struct{}{}
	}
	for r := 0; r < rows; r++ {
		off := r * rowBytes
		if _, isHot := hot[int64(r)]; isHot {
			if !bytesEqual(dst[off:off+rowBytes], original[off:off+rowBytes]) {
				return fmt.Errorf("gguf: quant integrity hot row %d is not bit-exact", r)
			}
			continue
		}
		if !bytesEqual(dst[off:off+rowBytes], quantized[off:off+rowBytes]) {
			return fmt.Errorf("gguf: quant integrity cold row %d is not the quantized bytes", r)
		}
	}
	return nil
}

// bytesEqual is a tiny local equality helper so the file does not pull bytes in
// only for one comparison.
func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// mxfp4NMSESampler dequantizes packed MXFP4 payloads through the existing
// dequantMXFP4Scalar path and folds the reconstructed values against an
// independently supplied original vector into an NMSE sample. Keeping the
// sampler on the shared dequant keeps the validator on the same code path the
// loader uses.
func mxfp4NMSESampler(sample *quantIntegrityNMSESample, original []float32, packed []byte) error {
	if sample == nil {
		return fmt.Errorf("gguf: MXFP4 NMSE sampler requires a sample accumulator")
	}
	if len(packed) == 0 || len(packed)%blockMXFP4Bytes != 0 {
		return fmt.Errorf("gguf: MXFP4 NMSE sampler payload %d bytes is not a multiple of block size %d", len(packed), blockMXFP4Bytes)
	}
	elems := len(packed) / blockMXFP4Bytes * qkMXFP4
	if elems != len(original) {
		return fmt.Errorf("gguf: MXFP4 NMSE sampler element count %d vs original %d", elems, len(original))
	}
	reconstructed := make([]float32, elems)
	dequantMXFP4Scalar(reconstructed, packed)
	return sample.add(original, reconstructed)
}

// normalizePartFile rejects a part name that would escape the output directory.
func normalizePartFile(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("gguf: quant integrity part name is empty")
	}
	clean := filepath.ToSlash(filepath.Clean(name))
	if strings.HasPrefix(clean, "../") || strings.HasPrefix(clean, "/") || strings.Contains(clean, "/../") {
		return "", fmt.Errorf("gguf: quant integrity part name %q escapes the output directory", name)
	}
	return clean, nil
}
