package model

// v41_engram_binding_test.go is the pub#13663 witness: the forward must consume
// the DEQUANTIZED-f32 Engram dialect (1024 bytes/row, the published vcruz Q2_K
// table) beside the packed 264-byte dialect, under the same verified artifact
// binding. It is the internal/model half split out of #13662; it makes no loader,
// fence, parity, or throughput claim.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// v41EngramF32RowsForPacked builds the dequantized-f32 view of packed rows using
// the SAME arithmetic production's dequantV41EngramRow applies (E4M3 weight times
// 2^(scale-127), bf16-rounded), serialized little-endian. A parity test can then
// assert the two dialects produce byte-identical forward results.
func v41EngramF32RowsForPacked(packed []byte, rows, dim int) []byte {
	out := make([]byte, rows*V41EngramF32RowBytes)
	for r := 0; r < rows; r++ {
		row := packed[r*V41EngramPackedRowBytes : (r+1)*V41EngramPackedRowBytes]
		for j := 0; j < dim; j++ {
			code := row[j]
			scale := row[256+j/32]
			value := v41BF16(fp8E4M3ToF32(code) * float32(math.Ldexp(1, int(scale)-127)))
			binary.LittleEndian.PutUint32(out[r*V41EngramF32RowBytes+j*4:], math.Float32bits(value))
		}
	}
	return out
}

// TestV41EngramF32BindingAdmitsOnlyMatchingArtifact covers the fail-closed half:
// an f32-row (1024 B/row) source is admitted only under a matching artifact
// binding, and a tampered byte, a wrong shard, a wrong offset, a wrong row count,
// and an unsupported width each refuse with a typed error before a single row is
// served.
// fak-test:runtime fast est=1s lane=default
func TestV41EngramF32BindingAdmitsOnlyMatchingArtifact(t *testing.T) {
	const rows, dim = 8, 256
	packed := v41EngramTestPackedRows(rows)
	f32 := v41EngramF32RowsForPacked(packed, rows, dim)
	// Every row is exactly one f32 table row: dim float32 values.
	if len(f32) != rows*V41EngramF32RowBytes {
		t.Fatalf("fixture f32 extent %d, want %d", len(f32), rows*V41EngramF32RowBytes)
	}

	binding := V41EngramArtifactBinding{
		Shard: "layer-1.engram.safetensors", Offset: 0,
		Size: int64(len(f32)), Rows: rows, RowBytes: V41EngramF32RowBytes,
		Digest: v41EngramTestDigest(f32),
	}
	shard := V41EngramShard{
		ID: binding.Shard, ExpectedID: binding.Shard,
		Offset: binding.Offset, Size: binding.Size, Rows: binding.Rows,
	}

	// The matching f32 shard is admitted and reports the bound width.
	src, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), shard, binding)
	if err != nil {
		t.Fatalf("matching f32 shard refused: %v", err)
	}
	if got := src.RowBytes(); got != V41EngramF32RowBytes {
		t.Fatalf("verified f32 source RowBytes() = %d, want %d", got, V41EngramF32RowBytes)
	}
	if gotBinding, ok := V41EngramBinding(src); !ok || gotBinding != binding {
		t.Fatalf("verified f32 source binding = %+v ok=%t, want %+v", gotBinding, ok, binding)
	}

	// A tampered byte is refused at construction, before any row is served.
	wrong := append([]byte(nil), f32...)
	wrong[0] ^= 0xff
	var typed *V41EngramStreamError
	if _, err := NewV41VerifiedEngramRowSource(bytes.NewReader(wrong), shard, binding); !errors.As(err, &typed) || typed.Kind != V41EngramStreamBindingMismatch {
		t.Fatalf("tampered f32 bytes: err=%v, want typed kind %q", err, V41EngramStreamBindingMismatch)
	}

	// A wrong shard name, a shifted offset, and a wrong row count each refuse.
	badName := shard
	badName.ID, badName.ExpectedID = "layer-2.engram.safetensors", "layer-2.engram.safetensors"
	if _, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), badName, binding); !errors.As(err, &typed) || typed.Kind != V41EngramStreamShardMismatch {
		t.Fatalf("wrong f32 shard name: err=%v, want typed kind %q", err, V41EngramStreamShardMismatch)
	}
	shifted := binding
	shifted.Offset = V41EngramF32RowBytes
	shifted.Size = int64(len(f32)) - shifted.Offset
	shifted.Rows = int(shifted.Size) / V41EngramF32RowBytes
	shifted.Digest = v41EngramTestDigest(f32[shifted.Offset : shifted.Offset+shifted.Size])
	shiftedShard := shard
	shiftedShard.Offset, shiftedShard.Size, shiftedShard.Rows = shifted.Offset, shifted.Size, shifted.Rows
	if _, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), shiftedShard, binding); !errors.As(err, &typed) {
		t.Fatalf("shifted f32 range under original binding: err=%v, want a typed refusal", err)
	}

	// An unsupported row width is refused by validate() before any IO.
	badWidth := binding
	badWidth.RowBytes = 100
	if _, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), shard, badWidth); err == nil {
		t.Fatal("unsupported f32 row width was admitted")
	}

	// The packed dialect is unchanged: the same bytes as a 264-packed binding are
	// still admitted and report 264.
	packedBinding := V41EngramArtifactBinding{
		Shard: "layer-1.engram.safetensors", Offset: 0,
		Size: int64(len(packed)), Rows: rows, RowBytes: V41EngramPackedRowBytes,
		Digest: v41EngramTestDigest(packed),
	}
	packedShard := V41EngramShard{
		ID: packedBinding.Shard, ExpectedID: packedBinding.Shard,
		Offset: packedBinding.Offset, Size: packedBinding.Size, Rows: packedBinding.Rows,
	}
	packedSrc, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(packed), packedShard, packedBinding)
	if err != nil {
		t.Fatalf("packed dialect regressed: %v", err)
	}
	if got := packedSrc.RowBytes(); got != V41EngramPackedRowBytes {
		t.Fatalf("packed source RowBytes() = %d, want %d", got, V41EngramPackedRowBytes)
	}
}

// TestV41EngramF32ForwardConsumeParity witnesses the forward-consume half: a
// model whose Engram stage is wired with a verified f32-row source consumes it and
// produces byte-identical logits to the same model wired with the packed-264
// source carrying the equivalent rows. The f32 fixture is the exact dequantization
// of the packed fixture, so parity is the contract; the packed path is untouched.
// fak-test:runtime fast est=1s lane=default
func TestV41EngramF32ForwardConsumeParity(t *testing.T) {
	ids := []int{1, 3, 5}

	// Packed model: the existing reduced fixture, wired with packed rows.
	mPacked, layout := v41ReducedEngramModel(t)
	want := mPacked.Forward(ids)
	if want == nil || len(want.Logits) != len(ids) {
		t.Fatalf("packed Forward returned %v positions", want)
	}

	// Same model, rewired with a verified f32-row source carrying the exact
	// dequantization of the packed rows.
	mF32, _ := v41ReducedEngramModel(t)
	rows := int(layout.Rows[0])
	dim := mF32.Cfg.DeepSeekV41.EngramHeadDim
	packed := v41EngramTestPackedRows(rows)
	f32 := v41EngramF32RowsForPacked(packed, rows, dim)

	binding := V41EngramArtifactBinding{
		Shard: "layer-1.engram.safetensors", Offset: 0,
		Size: int64(len(f32)), Rows: rows, RowBytes: V41EngramF32RowBytes,
		Digest: v41EngramTestDigest(f32),
	}
	verified, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), V41EngramShard{
		ID: binding.Shard, ExpectedID: binding.Shard,
		Offset: binding.Offset, Size: binding.Size, Rows: binding.Rows,
	}, binding)
	if err != nil {
		t.Fatalf("verified f32 source construction: %v", err)
	}

	// Rewire the same model onto the f32 dialect. The wire seam enforces one
	// dialect across sources and the cache width follows src.RowBytes().
	v41EngramStages.Delete(mF32)
	if err := mF32.wireV41Engram(layout, []V41EngramRowSource{verified}, int64(V41EngramF32RowBytes)*4); err != nil {
		t.Fatalf("wire f32 Engram stage: %v", err)
	}
	stage := mF32.v41EngramStageFor()
	if stage == nil || stage.rowBytes != V41EngramF32RowBytes {
		t.Fatalf("f32 stage rowBytes = %v, want %d", stage, V41EngramF32RowBytes)
	}

	got := mF32.Forward(ids)
	if got == nil || len(got.Logits) != len(ids) {
		t.Fatalf("f32 Forward returned %v positions", got)
	}
	for pos := range ids {
		if len(got.Logits[pos]) != len(want.Logits[pos]) {
			t.Fatalf("position %d logits width %d, want %d", pos, len(got.Logits[pos]), len(want.Logits[pos]))
		}
		for i := range got.Logits[pos] {
			if got.Logits[pos][i] != want.Logits[pos][i] {
				t.Fatalf("f32 logits differ from packed at position %d index %d: %v vs %v",
					pos, i, got.Logits[pos][i], want.Logits[pos][i])
			}
		}
	}
}
