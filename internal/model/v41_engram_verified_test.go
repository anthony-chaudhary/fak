package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"testing"
)

// v41EngramTestDigest is the content address of a byte slice in the same
// "sha256:<hex>" form the binding carries.
func v41EngramTestDigest(b []byte) string {
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// TestV41EngramVerifiedBindingRejectsWrongShard is the loader-bound negative
// fixture required before #12909 removes native admission refusal: a reader whose
// bytes do not match the pinned checkpoint-index shard's declared digest must be
// refused at construction, so a wrong shard/range can never reach Engram
// projection or Prefill/Step.
func TestV41EngramVerifiedBindingRejectsWrongShard(t *testing.T) {
	packed := streamTestPackedRows(8)
	wrong := streamTestPackedRows(8)
	wrong[0] ^= 0xff

	binding := V41EngramArtifactBinding{
		Shard:  "layer-1.safetensors",
		Offset: 0,
		Size:   int64(len(packed)),
		Rows:   8,
		Digest: v41EngramTestDigest(packed),
	}
	// The index-requested shard identity is what the loader intends to open; the
	// binding is what it observed. In the healthy case they agree.
	shard := V41EngramShard{
		ID: binding.Shard, ExpectedID: binding.Shard,
		Offset: binding.Offset, Size: binding.Size, Rows: binding.Rows,
	}

	// The matching bytes are admitted.
	if _, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(packed), shard, binding); err != nil {
		t.Fatalf("matching shard bytes were refused: %v", err)
	}

	// A reader whose bytes differ from the declared digest is refused BEFORE any
	// row is served, so a wrong shard cannot reach the Engram projection.
	_, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(wrong), shard, binding)
	var typed *V41EngramStreamError
	if !errors.As(err, &typed) || typed.Kind != V41EngramStreamBindingMismatch {
		t.Fatalf("wrong shard bytes: err=%v, want typed kind %q", err, V41EngramStreamBindingMismatch)
	}

	// A wrong declared extent (shard name or offset/length) is refused even when
	// the reader's bytes happen to hash correctly over a different range.
	badShard := shard
	badShard.ID, badShard.ExpectedID = "layer-2.safetensors", "layer-2.safetensors"
	if _, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(packed), badShard, binding); !errors.As(err, &typed) || typed.Kind != V41EngramStreamShardMismatch {
		t.Fatalf("wrong shard name: err=%v, want typed kind %q", err, V41EngramStreamShardMismatch)
	}
	badRows := shard
	badRows.Rows = 4
	badBinding := binding
	badBinding.Rows = 4
	badBinding.Size = int64(4) * V41EngramPackedRowBytes
	badBinding.Digest = v41EngramTestDigest(packed[:4*V41EngramPackedRowBytes])
	if _, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(packed), badRows, badBinding); !errors.As(err, &typed) {
		t.Fatalf("shard extent/rows disagreeing with binding: err=%v, want a typed refusal", err)
	}

	// A binding for a different offset over the same bytes is refused: a range
	// shifted within the shard must not be admitted under the wrong identity.
	shifted := binding
	shifted.Offset = V41EngramPackedRowBytes
	if shifted.Size+shifted.Offset > int64(len(packed)) {
		shifted.Size = int64(len(packed)) - shifted.Offset
		shifted.Rows = int(shifted.Size) / V41EngramPackedRowBytes
	}
	shiftedShard := shard
	shiftedShard.Offset, shiftedShard.Size, shiftedShard.Rows = shifted.Offset, shifted.Size, shifted.Rows
	shifted.Digest = v41EngramTestDigest(packed[shifted.Offset : shifted.Offset+shifted.Size])
	if _, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(packed), shiftedShard, binding); !errors.As(err, &typed) {
		t.Fatalf("shifted range under the original binding: err=%v, want a typed refusal", err)
	}
}

// TestV41EngramVerifiedSourceReachesForward proves a verified source is a
// drop-in for the unverified one on the ordinary Forward path: the same reduced
// V4.1 model whose Engram stage is wired with a digest-checked row source yields
// byte-identical logits to the unverified construction, and the verified source
// reports its binding while a plain one does not. It is the positive half of the
// loader-bound witness — the negative half is the wrong-shard fixture above.
func TestV41EngramVerifiedSourceReachesForward(t *testing.T) {
	mUnverified, layout := v41ReducedEngramModel(t)
	plain, _ := V41EngramBinding(nil)
	if plain != (V41EngramArtifactBinding{}) {
		t.Fatal("nil source reported a binding")
	}

	ids := []int{1, 3, 5}
	want := mUnverified.Forward(ids)
	if want == nil || len(want.Logits) != len(ids) {
		t.Fatalf("unverified Forward returned %v positions", want)
	}

	// Rebuild the identical model, but wire the stage with a source admitted only
	// after its bytes matched a loader binding over the synthetic shard.
	mVerified, layoutV := v41ReducedEngramModel(t)
	if layoutV.Rows[0] != layout.Rows[0] {
		t.Fatalf("fixture layout changed: %d vs %d", layoutV.Rows[0], layout.Rows[0])
	}
	packed := v41EngramTestPackedRows(int(layout.Rows[0]))
	binding := V41EngramArtifactBinding{
		Shard: "layer-1.safetensors", Offset: 0, Size: int64(len(packed)), Rows: int(layout.Rows[0]),
		Digest: v41EngramTestDigest(packed),
	}
	shard := V41EngramShard{
		ID: binding.Shard, ExpectedID: binding.Shard,
		Offset: binding.Offset, Size: binding.Size, Rows: binding.Rows,
	}
	verified, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(packed), shard, binding)
	if err != nil {
		t.Fatalf("verified source construction: %v", err)
	}
	gotBinding, ok := V41EngramBinding(verified)
	if !ok || gotBinding != binding {
		t.Fatalf("verified source binding = %+v ok=%t, want %+v", gotBinding, ok, binding)
	}
	if _, ok := V41EngramBinding(&v41EngramMemorySource{packed: packed, rows: int(layout.Rows[0])}); ok {
		t.Fatal("unverified source reported a binding")
	}

	v41EngramStages.Delete(mVerified)
	if err := mVerified.wireV41Engram(layoutV, []V41EngramRowSource{verified}, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatalf("wire verified Engram stage: %v", err)
	}
	if err := mVerified.v41ForwardAdmitted(); err != nil {
		t.Fatalf("admission with verified Engram stage = %v, want nil", err)
	}
	got := mVerified.Forward(ids)
	if got == nil || len(got.Logits) != len(ids) {
		t.Fatalf("verified Forward returned %v positions", got)
	}
	for tPos := range ids {
		for i := range got.Logits[tPos] {
			if got.Logits[tPos][i] != want.Logits[tPos][i] {
				t.Fatalf("verified logits differ from unverified at position %d index %d: %v vs %v",
					tPos, i, got.Logits[tPos][i], want.Logits[tPos][i])
			}
		}
	}
}
