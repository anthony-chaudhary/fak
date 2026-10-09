//go:build v41_indexer_witness && vulkan && (windows || linux) && cgo

package compute

import (
	"crypto/sha256"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Deferred physical witness SOURCE, never a qualification receipt by itself.
// Requires a distinct test archive built with FAK_V41_INDEXER_SCORE_WITNESS,
// never the ordinary V5 production archive. Record its actual command/source
// identities independently; structural V5 receipts cannot attest that causality.
// Requires a separately reviewed 62-module/archive/binary provenance and explicit
// target/device identity. Bound: 15 tiny dispatches, H<=3, D<=3, rows<=2. Timing,
// native fault injection, model wiring and full selection parity remain TODO.
// Runtime estimate is unmeasured; authorization to execute is separate.
// fak-test:runtime integration est=5s lane=optin
func TestVulkanV41IndexerScorePhysical(t *testing.T) {
	if os.Getenv("FAK_V41_INDEXER_SCORE_WITNESS") != "1" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required-device run needs FAK_V41_INDEXER_SCORE_WITNESS=1")
		}
		t.Skip("deferred physical indexer witness is opt-in")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" || os.Getenv("FAK_VULKAN_EXPECT_DEVICE") == "" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("physical witness requires device, exact expected identity and dispatch profiling before startup")
	}
	v41IndexerScoreWitnessProvenance(t)
	v := vk(t)
	kind := v.VulkanPhysicalDeviceType()
	if kind != 1 && kind != 2 {
		t.Fatalf("physical integrated/discrete GPU required, got type=%d", kind)
	}
	if !v.v41IndexerScoreWitnessReady() {
		t.Fatalf("indexer unsupported: %s", v.V41IndexerScoreUnavailableReason())
	}
	if v.SupportsV41IndexerScore() || v.V41IndexerScoreUnavailableReason() == "" {
		t.Fatal("test archive must not enable production qualification")
	}
	identity, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if identity.Identity.Backend != "vulkan" || identity.Identity.Device != os.Getenv("FAK_VULKAN_EXPECT_DEVICE") || !identity.TransferCountersObserved || !identity.DeviceAllocationObserved {
		t.Fatalf("physical identity/counter requirement failed: %+v", identity)
	}
	if Default().Name() != "cpu-ref" {
		t.Fatal("fixture constructor must remain cpu-ref")
	}
	for _, tc := range v41IndexerScoreFixtures() {
		if !t.Run(tc.name, func(t *testing.T) {
			rows := len(tc.keys) / tc.dim
			upload := func(shape []int, values []float32) Tensor {
				x := v.UploadClass(NewF32(Default(), shape, values), F32, MemoryActivation, "indexer witness input")
				t.Cleanup(func() { v.Free(x) })
				return x
			}
			q, k, w := upload([]int{tc.heads, tc.dim}, tc.q), upload([]int{rows, tc.dim}, tc.keys), upload([]int{tc.heads}, tc.weights)
			before, err := v.BackendExecutionSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			v.BeginBatch()
			defer v.FlushBatch()
			out, err := v.v41IndexerScoreWitness(q, k, w, rows, tc.heads, tc.dim)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { v.Free(out) })
			if out.Buf() == nil || out.Buf() == q.Buf() || out.Buf() == k.Buf() || out.Buf() == w.Buf() || !out.buf.(*vulkanBuf).v41CheckedRead {
				t.Fatal("output is not independently owned with checked readback")
			}
			got := v.Read(out)
			v41IndexerScoreCompareBits(t, got, tc.want)
			v41IndexerScoreCompareBits(t, got, v41IndexerScoreIndependentOracle(tc.q, tc.keys, tc.weights, tc.heads, tc.dim))
			after, err := v.BackendExecutionSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			delta, err := BackendExecutionDelta(before, after)
			if err != nil {
				t.Fatal(err)
			}
			c := delta.Counters
			if c.ComputeDispatches != 1 || c.OtherDispatches != 1 || c.H2DCount != 0 || c.D2HCount != 1 || c.D2HBytes != uint64(rows*4) || c.D2DCopies != 0 || c.Fallbacks != 0 {
				t.Fatalf("incorrect isolated contraction accounting: %+v", delta)
			}
			v41IndexerScoreCompareBits(t, v.Read(q), tc.q)
			v41IndexerScoreCompareBits(t, v.Read(k), tc.keys)
			v41IndexerScoreCompareBits(t, v.Read(w), tc.weights)
			t.Logf("actual contraction counters=%+v", delta)
		}) {
			t.Fatal("physical witness stopped after first failure; no retry")
		}
	}
	// Empty geometry and malformed ownership refuse before dispatch/allocation.
	q := v.UploadClass(NewF32(Default(), []int{1, 1}, []float32{1}), F32, MemoryActivation, "empty indexer query")
	w := v.UploadClass(NewF32(Default(), []int{1}, []float32{1}), F32, MemoryActivation, "empty indexer weights")
	defer v.Free(q)
	defer v.Free(w)
	empty := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{0, 1}}
	before, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	out, err := v.v41IndexerScoreWitness(q, empty, w, 0, 1, 1)
	if err != nil || out.Numel() != 0 || out.Buf() == nil {
		t.Fatalf("zero rows output=%+v err=%v", out, err)
	}
	if len(v.Read(out)) != 0 {
		t.Fatal("empty output read was nonempty")
	}
	v.Free(out)
	bad := q
	bad.Shape = []int{1}
	if out, err = v.v41IndexerScoreWitness(bad, empty, w, 0, 1, 1); err == nil || out.Buf() != nil {
		t.Fatal("malformed shape published output")
	}
	if out, err = v.v41IndexerScoreWitness(NewF32(Default(), []int{1, 1}, []float32{1}), empty, w, 0, 1, 1); err == nil || out.Buf() != nil {
		t.Fatal("foreign owner published output")
	}
	// Genuine handles only: all extents match, leaving the output alias as the
	// sole native rejection reason. No synthetic pointers or device faults.
	code := func() int {
		vulkanMu.Lock()
		defer vulkanMu.Unlock()
		return v41IndexerScoreWitnessStatusLocked(q.buf.(*vulkanBuf), q.buf.(*vulkanBuf), w.buf.(*vulkanBuf), q.buf.(*vulkanBuf), 1, 1, 1)
	}()
	if code != 2 {
		t.Fatalf("native output alias accepted, status=%d", code)
	}
	after, err := v.BackendExecutionSnapshot()
	if err != nil || after.Counters != before.Counters || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
		t.Fatalf("empty/rejection controls changed counters or allocations: before=%+v after=%+v err=%v", before, after, err)
	}
	if !math.IsNaN(float64(v41IndexerScoreIndependentOracle([]float32{math.MaxFloat32}, []float32{2}, []float32{0}, 1, 1)[0])) {
		t.Fatal("independent oracle lost arithmetic NaN")
	}
	t.Logf("V41_INDEXER_SCORE_PHYSICAL_COMPLETE dispatches=%d identity=%+v", len(v41IndexerScoreFixtures()), identity.Identity)
	// TODO: separately authorized checked-submit/wait/readback failure injection,
	// DEVICE_LOST and quarantine ownership witnesses; exact selected indices and
	// tie/NaN/infinity policy at the model seam; actual dedicated device timestamps.
}

// These are recorded bytes, not a claim of trusted provenance by self-hash.
// The independently reviewed build record must bind these hashes to the pinned
// source and exact commands, including the witness-only define and Go build tag.
func v41IndexerScoreWitnessProvenance(t *testing.T) {
	t.Helper()
	archive := os.Getenv("FAK_V41_INDEXER_SCORE_WITNESS_ARCHIVE")
	record := os.Getenv("FAK_V41_INDEXER_SCORE_WITNESS_PROVENANCE")
	bundle := os.Getenv("FAK_VULKAN_SPIRV")
	if filepath.Base(archive) != "libfakvulkan_v41_indexer_witness.a" || record == "" || bundle == "" {
		t.Fatal("witness requires distinct archive, reviewed provenance record, and explicit module bundle paths")
	}
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct{ kind, path string }{
		{"witness-archive", archive}, {"reviewed-provenance", record},
		{"loaded-indexer-module", filepath.Join(bundle, "v41_indexer_score.spv")}, {"actual-test-binary", binary},
	} {
		f, err := os.Open(item.path)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.New()
		count, readErr := io.Copy(hash, f)
		closeErr := f.Close()
		if readErr != nil || closeErr != nil || count == 0 {
			t.Fatalf("invalid %s bytes=%d read=%v close=%v", item.kind, count, readErr, closeErr)
		}
		t.Logf("physical witness provenance kind=%s path=%q bytes=%d sha256=%s", item.kind, item.path, count, fmt.Sprintf("%x", hash.Sum(nil)))
	}
}
