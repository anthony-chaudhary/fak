package ggufload

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41_engram_q2k_rows_test.go is the witness suite for issue #13008: a
// disk-backed model.V41EngramRowSource over the REAL published DeepSeek-V4.1-Flash
// Q2_K engram_embd.weight table (vcruz305/DeepSeek-V4.1-Flash-GGUF), with correct
// Q2_K row dequantization and a typed fail-closed negative.
//
// The fixture is a two-shard split checkpoint whose shard 2 carries a Q2_K
// (type 10) blk.1.engram_embd.weight of dims [qkK, N]. Each row is ONE 84-byte
// Q2_K super-block whose f16 super-scale d is set to a distinct per-row value, so
// a wrong row index or a wrong on-disk stride is immediately visible in the
// dequantized f32 output.
//
// Nothing here is a hardware witness: the row-read latency and page-amplification
// numbers on a real device are [HW-WITNESSED] only. This on-disk-tempfile run is
// [SW-VERIFIED].

// v41Q2KRows is the fixture row count for #13008. Three rows are enough to prove
// stride, non-contiguous access, and a repeat hit without a sprawling matrix.
const v41Q2KRows = 3

// v41Q2KBlockRow returns one 84-byte Q2_K super-block whose f16 super-scale d is
// (row+1) and whose q bytes come from the shared q2KFixtureBlock structure. The
// returned want is the independently-known dequantization: the fixture's own
// expected values scaled by (row+1). Because d multiplies every code and dmin is
// zero, this is exact up to float32 rounding.
func v41Q2KBlockRow(row int) ([]byte, []float32) {
	block, want := q2KFixtureBlock()
	d := float32(row + 1)
	dst := make([]byte, len(block))
	copy(dst, block)
	// d is the first f16 of the two trailing scale fields (after 16 scale bytes
	// and 64 q bytes = offset 80); dmin follows at 82 and stays 0.
	binary.LittleEndian.PutUint16(dst[qkK/16+qkK/4:], f32ToF16BitsForTestQ2K(d))
	scaled := make([]float32, len(want))
	for i, v := range want {
		scaled[i] = v * d
	}
	return dst, scaled
}

// f32ToF16BitsForTestQ2K encodes a small integer float32 as a half-precision bit
// pattern. The only values used are 1,2,3 (exactly representable in f16), so a
// direct exponent/mantissa construction is sufficient and deterministic.
func f32ToF16BitsForTestQ2K(v float32) uint16 {
	bits := math.Float32bits(v)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := uint16((bits >> 13) & 0x03ff)
	return sign | uint16(exp)<<10 | mant
}

// v41Q2KBuildShard writes the two-shard fixture with a Q2_K table of `rows`
// distinct super-blocks at blk.1.engram_embd.weight. A declared engram.rows entry
// is written when declared is true, so both the declared and dims-derived row
// paths are exercisable.
func v41Q2KBuildShard(t *testing.T, rows int, declared bool) (shard1, shard2 []byte) {
	t.Helper()
	const align = 32

	meta, p := v41PublishedArtifactMeta("deepseek41")
	if declared {
		meta[p+"engram.rows"] = Value{Type: TypeArray, Value: []Value{
			{Type: TypeUint64, Value: uint64(rows)},
			{Type: TypeUint64, Value: uint64(1000)},
		}}
	}

	written := 0
	for _, key := range v41PublishedMetaKeys {
		if _, ok := meta[key]; ok {
			written++
		}
	}

	var s1 bytes.Buffer
	writeMinimalHeader(&s1, 0, uint64(written+3))
	for _, key := range v41PublishedMetaKeys {
		v, ok := meta[key]
		if !ok {
			continue
		}
		writeMetaValueForTest(t, &s1, key, v)
	}
	writeKVUint32(&s1, "split.no", 1)
	writeKVUint32(&s1, "split.count", 2)
	writeKVUint32(&s1, "split.tensors.count", 1)
	padToAlignment(&s1, align)

	var infos bytes.Buffer
	writeTensorInfoForTest(&infos, "blk.1.engram_embd.weight",
		[]uint64{qkK, uint64(rows)}, TensorQ2_K, 0)

	var s2 bytes.Buffer
	writeMinimalHeader(&s2, 1, 4)
	writeKVUint32(&s2, "general.alignment", align)
	writeKVUint32(&s2, "split.no", 2)
	writeKVUint32(&s2, "split.count", 2)
	writeKVUint32(&s2, "split.tensors.count", 1)
	s2.Write(infos.Bytes())
	padToAlignment(&s2, align)
	for i := 0; i < rows; i++ {
		block, _ := v41Q2KBlockRow(i)
		dataStart := s2.Len()
		s2.Write(block)
		padToLen(&s2, dataStart+align)
	}
	return s1.Bytes(), s2.Bytes()
}

// v41Q2KBuildShardDeclaredRows writes the same two-shard fixture as
// v41Q2KBuildShard but takes the engram.rows array entries explicitly, so a
// malformed declared count (e.g. a non-positive entry) is exercisable. Layer_ids
// are [1,14], so index 0 is layer 1 and index 1 is layer 14.
func v41Q2KBuildShardDeclaredRows(t *testing.T, rows int, decl []int) (shard1, shard2 []byte) {
	t.Helper()
	const align = 32

	meta, p := v41PublishedArtifactMeta("deepseek41")
	entries := make([]Value, len(decl))
	for i, d := range decl {
		entries[i] = Value{Type: TypeUint64, Value: uint64(int64(d))}
	}
	meta[p+"engram.rows"] = Value{Type: TypeArray, Value: entries}

	written := 0
	for _, key := range v41PublishedMetaKeys {
		if _, ok := meta[key]; ok {
			written++
		}
	}

	var s1 bytes.Buffer
	writeMinimalHeader(&s1, 0, uint64(written+3))
	for _, key := range v41PublishedMetaKeys {
		v, ok := meta[key]
		if !ok {
			continue
		}
		writeMetaValueForTest(t, &s1, key, v)
	}
	writeKVUint32(&s1, "split.no", 1)
	writeKVUint32(&s1, "split.count", 2)
	writeKVUint32(&s1, "split.tensors.count", 1)
	padToAlignment(&s1, align)

	var infos bytes.Buffer
	writeTensorInfoForTest(&infos, "blk.1.engram_embd.weight",
		[]uint64{qkK, uint64(rows)}, TensorQ2_K, 0)

	var s2 bytes.Buffer
	writeMinimalHeader(&s2, 1, 4)
	writeKVUint32(&s2, "general.alignment", align)
	writeKVUint32(&s2, "split.no", 2)
	writeKVUint32(&s2, "split.count", 2)
	writeKVUint32(&s2, "split.tensors.count", 1)
	s2.Write(infos.Bytes())
	padToAlignment(&s2, align)
	for i := 0; i < rows; i++ {
		block, _ := v41Q2KBlockRow(i)
		dataStart := s2.Len()
		s2.Write(block)
		padToLen(&s2, dataStart+align)
	}
	return s1.Bytes(), s2.Bytes()
}

// v41Q2KOpenFixture writes and opens the two-shard Q2_K fixture.
func v41Q2KOpenFixture(t *testing.T, rows int, declared bool) *WeightSource {
	t.Helper()
	shard1, shard2 := v41Q2KBuildShard(t, rows, declared)
	dir := t.TempDir()
	p1 := filepath.Join(dir, "v41-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "v41-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	ws, err := OpenWeights(p1)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { ws.Close() })
	return ws
}

// f32BytesQ2KForTest serializes expected f32 values to the little-endian byte
// order ReadRows writes.
func f32BytesQ2KForTest(vals []float32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// TestV41EngramDiskRowSource is the named #13008 witness: a file-backed Q2_K
// Engram row source served through the bounded model row cache must return the
// CORRECT dequantized row for each index across a non-contiguous access order with
// a repeat, prove a cache hit, and report a finite page-amplification ratio.
func TestV41EngramDiskRowSource(t *testing.T) {
	ws := v41Q2KOpenFixture(t, v41Q2KRows, false)

	src, err := V41EngramQ2KOpen(ws, 1)
	if err != nil {
		t.Fatalf("V41EngramQ2KOpen: %v", err)
	}
	if got := src.RowBytes(); got != 1024 {
		t.Fatalf("RowBytes = %d, want 1024 (qkK*4 dequantized f32 row width)", got)
	}
	if got := src.TableRows(); got != v41Q2KRows {
		t.Fatalf("TableRows = %d, want %d (derived from dims[1])", got, v41Q2KRows)
	}

	// Bounded cache with a one-row prefetch: reading row 0 eagerly pulls row 1,
	// so the later row-1 access is the deterministic hit.
	cache, err := model.NewV41EngramRowCache(src, model.V41EngramRowCacheOptions{
		TableRows:    src.TableRows(),
		RowBytes:     src.RowBytes(),
		BudgetBytes:  int64(src.RowBytes()) * 4,
		PrefetchRows: 1,
	})
	if err != nil {
		t.Fatalf("NewV41EngramRowCache: %v", err)
	}

	// Non-contiguous order including a repeat. Expected bytes are the
	// independently-derived per-row dequantization, so a wrong index or a wrong
	// 84-byte stride fails loudly rather than just mismatching a length.
	for _, step := range []int{2, 0, 2, 1} {
		_, want := v41Q2KBlockRow(step)
		got, err := cache.Row(step)
		if err != nil {
			t.Fatalf("cache.Row(%d): %v", step, err)
		}
		if len(got) != src.RowBytes() {
			t.Fatalf("row %d: %d bytes, want the source's declared row width %d", step, len(got), src.RowBytes())
		}
		if !bytes.Equal(got, f32BytesQ2KForTest(want)) {
			t.Fatalf("row %d: dequantized bytes do not match the expected Q2_K super-block (row identity/stride defect)", step)
		}
	}

	stats := cache.Stats()
	// This single, stable, machine-greppable line is the #13008 [SW-VERIFIED]
	// page-amplification measurement witness: it captures the full row-retrieval
	// telemetry (source geometry + cache hit/miss/prefetch/bytes) so the measured
	// page_amplification can be quoted in the issue receipt. On-host and tempfile
	// only: Halo row-read latency on real device hardware stays [HW-WITNESSED].
	t.Logf("V41ENGRAM_Q2K_AMPLIFICATION rows=%d row_bytes=%d hits=%d misses=%d prefetches=%d reads_bytes=%d useful_bytes=%d page_amplification=%.4f",
		v41Q2KRows, src.RowBytes(),
		stats.Hits, stats.Misses, stats.Prefetches,
		stats.BytesRead, stats.UsefulBytes, stats.PageAmplification)

	if stats.Hits < 1 {
		t.Fatalf("stats.Hits = %d, want >= 1 (the repeated row 2 must be a cache hit)", stats.Hits)
	}
	// Page amplification is bytes read from the source divided by useful bytes
	// served to the caller; it is finite and >= 1 for any honest source (a source
	// can never read fewer bytes than it serves).
	if stats.PageAmplification <= 0 {
		t.Fatalf("PageAmplification = %v, want > 0", stats.PageAmplification)
	}
	if stats.PageAmplification < 1 {
		t.Fatalf("PageAmplification = %v, want >= 1 (bytes read / useful bytes)", stats.PageAmplification)
	}
}

// TestV41EngramQ2KAmplificationReflectsPrefetch proves the reported
// page_amplification is the REAL bytes-read/useful-bytes ratio, not a constant
// 1.0 by construction. A one-row prefetch over a >= 3-row table reads eagerly
// beyond what the caller asked for, so bytes read strictly exceeds useful bytes
// and the two are independently recomputed and compared here.
func TestV41EngramQ2KAmplificationReflectsPrefetch(t *testing.T) {
	ws := v41Q2KOpenFixture(t, v41Q2KRows, false)
	src, err := V41EngramQ2KOpen(ws, 1)
	if err != nil {
		t.Fatalf("V41EngramQ2KOpen: %v", err)
	}
	cache, err := model.NewV41EngramRowCache(src, model.V41EngramRowCacheOptions{
		TableRows:    src.TableRows(),
		RowBytes:     src.RowBytes(),
		BudgetBytes:  int64(src.RowBytes()) * 4,
		PrefetchRows: 1,
	})
	if err != nil {
		t.Fatalf("NewV41EngramRowCache: %v", err)
	}

	if _, err := cache.Row(0); err != nil {
		t.Fatalf("cache.Row(0): %v", err)
	}
	stats := cache.Stats()

	if stats.BytesRead <= 0 {
		t.Fatalf("stats.BytesRead = %d, want > 0", stats.BytesRead)
	}
	if stats.UsefulBytes <= 0 {
		t.Fatalf("stats.UsefulBytes = %d, want > 0", stats.UsefulBytes)
	}
	want := float64(stats.BytesRead) / float64(stats.UsefulBytes)
	if diff := stats.PageAmplification - want; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("PageAmplification = %v, want %v (BytesRead %d / UsefulBytes %d)",
			stats.PageAmplification, want, stats.BytesRead, stats.UsefulBytes)
	}
	if stats.Prefetches <= 0 {
		t.Fatalf("stats.Prefetches = %d, want > 0 (prefetch must have run)", stats.Prefetches)
	}
}

// TestV41EngramQ2KDeclaredRowCountAgrees proves the declared engram.rows path is
// used when present and agrees with the tensor dims.
func TestV41EngramQ2KDeclaredRowCountAgrees(t *testing.T) {
	ws := v41Q2KOpenFixture(t, v41Q2KRows, true)
	src, err := V41EngramQ2KOpen(ws, 1)
	if err != nil {
		t.Fatalf("V41EngramQ2KOpen (declared rows): %v", err)
	}
	if src.TableRows() != v41Q2KRows {
		t.Fatalf("TableRows = %d, want %d", src.TableRows(), v41Q2KRows)
	}
}

// TestV41EngramQ2KRejectsNonPositiveDeclaredRows proves that a declared
// engram.rows entry which is present but non-positive is a NAMED geometry
// refusal rather than a silent fallback to dims[1]. Layer_ids are [1,14], so the
// array [0, 1000] declares layer 1 with a zero row count.
func TestV41EngramQ2KRejectsNonPositiveDeclaredRows(t *testing.T) {
	shard1, shard2 := v41Q2KBuildShardDeclaredRows(t, v41Q2KRows, []int{0, 1000})
	dir := t.TempDir()
	p1 := filepath.Join(dir, "v41-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "v41-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	ws, err := OpenWeights(p1)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	src, err := V41EngramQ2KOpen(ws, 1)
	if err == nil {
		t.Fatal("V41EngramQ2KOpen accepted a non-positive declared row count; must fail closed")
	}
	if src != nil {
		t.Fatal("V41EngramQ2KOpen returned a non-nil source alongside the error")
	}
	var typed *V41EngramQ2KError
	if !errors.As(err, &typed) {
		t.Fatalf("error %v is not a *V41EngramQ2KError", err)
	}
	if typed.Kind != V41EngramQ2KGeometry {
		t.Fatalf("error kind = %q, want %q", typed.Kind, V41EngramQ2KGeometry)
	}
}

// TestV41EngramQ2KReadRowsBounds covers the range/geometry refusals that must fire
// before any payload read.
func TestV41EngramQ2KReadRowsBounds(t *testing.T) {
	ws := v41Q2KOpenFixture(t, v41Q2KRows, false)
	src, err := V41EngramQ2KOpen(ws, 1)
	if err != nil {
		t.Fatalf("V41EngramQ2KOpen: %v", err)
	}
	buf := make([]byte, 1024*4)

	if n, err := src.ReadRows(0, 0, nil); err != nil || n != 0 {
		t.Fatalf("count==0 = (%d,%v), want (0,nil)", n, err)
	}
	if _, err := src.ReadRows(-1, 1, buf); err == nil {
		t.Fatal("negative start accepted")
	}
	if _, err := src.ReadRows(0, -1, buf); err == nil {
		t.Fatal("negative count accepted")
	}
	if _, err := src.ReadRows(v41Q2KRows, 1, buf); err == nil {
		t.Fatal("start at table end accepted")
	}
	if _, err := src.ReadRows(0, v41Q2KRows+1, buf); err == nil {
		t.Fatal("count past table end accepted")
	}
	if _, err := src.ReadRows(0, 2, make([]byte, 1024)); err == nil {
		t.Fatal("short destination accepted")
	}
}

// TestV41EngramQ2KRefusesUnsupportedType is the typed negative: a tensor of an
// unsupported encoding must yield a *V41EngramQ2KError (kind tensor_type) and
// MUST NOT return zero-filled rows. It also checks the raw-I8 redirect message.
func TestV41EngramQ2KRefusesUnsupportedType(t *testing.T) {
	cases := []struct {
		name string
		typ  TensorType
	}{
		{"f16", TensorF16},
		{"q4_k", TensorQ4_K},
		{"raw_i8", v41EngramRawI8Type},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Rebuild the fixture with this tensor type at the Q2_K dims. The
			// builder writes 84-byte payloads; the factory must refuse on TYPE
			// before it ever reads them.
			shard1, shard2 := v41Q2KBuildShardWithType(t, v41Q2KRows, tc.typ)
			dir := t.TempDir()
			p1 := filepath.Join(dir, "v41-00001-of-00002.gguf")
			p2 := filepath.Join(dir, "v41-00002-of-00002.gguf")
			if err := os.WriteFile(p1, shard1, 0o644); err != nil {
				t.Fatalf("write shard 1: %v", err)
			}
			if err := os.WriteFile(p2, shard2, 0o644); err != nil {
				t.Fatalf("write shard 2: %v", err)
			}
			ws, err := OpenWeights(p1)
			if err != nil {
				t.Fatalf("OpenWeights: %v", err)
			}
			defer ws.Close()

			src, err := V41EngramQ2KOpen(ws, 1)
			if err == nil {
				t.Fatalf("V41EngramQ2KOpen accepted tensor type %s; must fail closed", tc.typ)
			}
			if src != nil {
				t.Fatal("V41EngramQ2KOpen returned a non-nil source alongside the error")
			}
			var typed *V41EngramQ2KError
			if !errors.As(err, &typed) {
				t.Fatalf("error %v is not a *V41EngramQ2KError", err)
			}
			if typed.Kind != V41EngramQ2KTensorType {
				t.Fatalf("error kind = %q, want %q", typed.Kind, V41EngramQ2KTensorType)
			}
			if typed.Type != tc.typ {
				t.Fatalf("error names type %s, want %s", typed.Type, tc.typ)
			}
			if tc.typ == v41EngramRawI8Type {
				if !strings.Contains(err.Error(), "V41EngramGGUFOpen") {
					t.Fatalf("raw-I8 redirect message %q does not name V41EngramGGUFOpen", err.Error())
				}
			}
		})
	}
}

// v41Q2KBuildShardWithType rewrites the Q2_K fixture's tensor directory entry at
// an unsupported encoding while keeping the 84-byte row payloads, so the type
// gate is the only thing under test.
func v41Q2KBuildShardWithType(t *testing.T, rows int, typ TensorType) (shard1, shard2 []byte) {
	t.Helper()
	const align = 32

	meta, _ := v41PublishedArtifactMeta("deepseek41")
	written := 0
	for _, key := range v41PublishedMetaKeys {
		if _, ok := meta[key]; ok {
			written++
		}
	}

	var s1 bytes.Buffer
	writeMinimalHeader(&s1, 0, uint64(written+3))
	for _, key := range v41PublishedMetaKeys {
		v, ok := meta[key]
		if !ok {
			continue
		}
		writeMetaValueForTest(t, &s1, key, v)
	}
	writeKVUint32(&s1, "split.no", 1)
	writeKVUint32(&s1, "split.count", 2)
	writeKVUint32(&s1, "split.tensors.count", 1)
	padToAlignment(&s1, align)

	var infos bytes.Buffer
	writeTensorInfoForTest(&infos, "blk.1.engram_embd.weight",
		[]uint64{qkK, uint64(rows)}, typ, 0)

	var s2 bytes.Buffer
	writeMinimalHeader(&s2, 1, 4)
	writeKVUint32(&s2, "general.alignment", align)
	writeKVUint32(&s2, "split.no", 2)
	writeKVUint32(&s2, "split.count", 2)
	writeKVUint32(&s2, "split.tensors.count", 1)
	s2.Write(infos.Bytes())
	padToAlignment(&s2, align)
	for i := 0; i < rows; i++ {
		block, _ := v41Q2KBlockRow(i)
		dataStart := s2.Len()
		s2.Write(block)
		padToLen(&s2, dataStart+align)
	}
	return s1.Bytes(), s2.Bytes()
}

// TestV41EngramQ2KShortReadRefused proves a truncated payload is a typed read
// refusal, never a silently shorter or zero-filled row.
func TestV41EngramQ2KShortReadRefused(t *testing.T) {
	ws := v41Q2KOpenFixture(t, v41Q2KRows, false)
	src, err := V41EngramQ2KOpen(ws, 1)
	if err != nil {
		t.Fatalf("V41EngramQ2KOpen: %v", err)
	}
	// Replace the reader with one that returns fewer bytes and no error: the
	// source must reject the nil-error short read.
	src.r = truncReaderAt{inner: src.r, keep: 10}
	_, err = src.ReadRows(0, 1, make([]byte, 1024))
	if err == nil {
		t.Fatal("short read accepted")
	}
	var typed *V41EngramQ2KError
	if !errors.As(err, &typed) || typed.Kind != V41EngramQ2KRead {
		t.Fatalf("error %v is not a *V41EngramQ2KError of kind %q", err, V41EngramQ2KRead)
	}
}

// truncReaderAt returns at most `keep` bytes per ReadAt with a nil error, to
// exercise the nil-error short-read guard.
type truncReaderAt struct {
	inner io.ReaderAt
	keep  int
}

func (t truncReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if len(p) > t.keep {
		p = p[:t.keep]
	}
	return t.inner.ReadAt(p, off)
}
