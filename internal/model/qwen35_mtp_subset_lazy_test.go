package model

import (
	"math"
	"strings"
	"testing"
)

// qwen35_mtp_subset_lazy_test.go -- the #13218 RED->GREEN witness. kQuantMatRowsSubset is the
// subset (LM-head row-filter) twin of the CPU k-quant GEMV/GEMM entries hardened by the
// #13216 fix (ensureRawCPU, 7ce00cf1f). Its Q4_K sibling q4kMatRowsSubset already guards with
// qt.requireRawCPU before its parallel region; the k-quant subset projection did not, so a
// checkpoint-backed lazy k-quant LM head reached through it would slice a nil qt.raw and die
// with a bare `slice bounds out of range` -- or read zeros. These tests pin the materialize-or-
// refuse contract for the subset path.

// TestLazyKQuantSubsetProjectionMaterializesOnDemand drives a checkpoint-backed (lazy) k-quant
// tensor through kQuantMatRowsSubset and requires that it MATERIALIZES the bounded range and
// computes byte-identically to the resident twin. Before the fix this sliced a nil qt.raw.
func TestLazyKQuantSubsetProjectionMaterializesOnDemand(t *testing.T) {
	for _, kind := range []kQuantKind{kindQ2K, kindQ3K, kindQ5K, kindQ6K} {
		payload := kQuantLazyLazyPayload(kind, 4, 256)
		resident := &kQuantTensor{out: 4, in: 256, nblk: 1, kind: kind, raw: payload}
		reader := &chunkedProbeReaderAt{data: append([]byte(nil), payload...)}
		lazy := &kQuantTensor{
			out: 4, in: 256, nblk: 1, kind: kind,
			lazy: &LazyQ4KRange{Reader: reader, Bytes: len(payload)},
		}
		subset := []int{0, 2, 3}
		x := make([]float32, 256)
		for i := range x {
			x[i] = float32(i%13) * 0.25
		}
		want := kQuantMatRowsSubset(resident, x, subset)
		got := func() []float32 {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("%s: subset projection panicked on a materializable lazy tensor: %v", kind, r)
				}
			}()
			return kQuantMatRowsSubset(lazy, x, subset)
		}()
		if len(lazy.raw) == 0 {
			t.Fatalf("%s: subset projection did not memoize the materialized range into raw", kind)
		}
		if reader.reads == 0 {
			t.Fatalf("%s: subset projection produced a result without reading the checkpoint", kind)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: subset row %d = %v, want byte-identical resident result %v (silent-zeros guard)", kind, i, got[i], want[i])
			}
		}
	}
}

// TestLazyKQuantSubsetProjectionStillFailsClosedOnUnmaterializable is the retained fail-closed
// half of the #13202 contract for the subset path: a lazy tensor whose range genuinely cannot
// be read must still panic with a NAMED diagnostic (never a bare `slice bounds out of range`),
// and must not leave partial resident bytes behind.
func TestLazyKQuantSubsetProjectionStillFailsClosedOnUnmaterializable(t *testing.T) {
	for _, kind := range []kQuantKind{kindQ2K, kindQ6K} {
		qt := &kQuantTensor{
			out: 1, in: 256, nblk: 1, kind: kind,
			lazy: &LazyQ4KRange{Reader: failingReaderAt{}, Bytes: kind.blockBytes()},
		}
		msg := recoverContains(func() {
			kQuantMatRowsSubset(qt, make([]float32, 256), []int{0})
		})
		if msg == "" {
			t.Fatalf("%s: subset projection accepted a lazy tensor whose range cannot be read", kind)
		}
		if !strings.Contains(msg, "materialization failed") && !strings.Contains(msg, "#13216") {
			t.Fatalf("%s: fail-closed panic = %q, want a named materialization/guard diagnostic", kind, msg)
		}
		if strings.Contains(msg, "slice bounds out of range") {
			t.Fatalf("%s: guardrail should pre-empt the cryptic slice panic, got: %s", kind, msg)
		}
		if len(qt.raw) != 0 {
			t.Fatalf("%s: failed materialization left %d resident bytes, want none", kind, len(qt.raw))
		}
	}
}

// TestLazyKQuantSubsetProjectionResidentIsBitIdentical pins the no-op property the gold-plating
// boundary requires: the guard must not perturb a resident tensor's result.
func TestLazyKQuantSubsetProjectionResidentIsBitIdentical(t *testing.T) {
	kind := kindQ6K
	payload := kQuantLazyLazyPayload(kind, 3, 256)
	before := &kQuantTensor{out: 3, in: 256, nblk: 1, kind: kind, raw: append([]byte(nil), payload...)}
	after := &kQuantTensor{out: 3, in: 256, nblk: 1, kind: kind, raw: payload}
	subset := []int{0, 1, 2}
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i%7) * 0.5
	}
	want := kQuantMatRowsSubset(before, x, subset)
	got := kQuantMatRowsSubset(after, x, subset)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("resident subset row %d = %v, want %v", i, got[i], want[i])
		}
	}
}
