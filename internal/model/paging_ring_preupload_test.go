package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type ringPreUploadBackend struct {
	compute.Backend
	uploads int
	frees   int
}

func (b *ringPreUploadBackend) Upload(src compute.Tensor, dtype compute.Dtype) compute.Tensor {
	b.uploads++
	return b.Backend.Upload(src, dtype)
}

func (b *ringPreUploadBackend) Free(t compute.Tensor) {
	b.frees++
	b.Backend.Free(t)
}

type ringPreUploadAsyncBackend struct {
	*ringPreUploadBackend
	fences []*asyncTestFence
}

func (b *ringPreUploadAsyncBackend) UploadAsync(src compute.Tensor, dtype compute.Dtype) (compute.Tensor, compute.Fence) {
	t := b.Upload(src, dtype)
	f := &asyncTestFence{}
	b.fences = append(b.fences, f)
	return t, f
}

// The ring must reject an oversized miss before building its source or issuing a
// doomed upload. This checks the actual lifecycle, not only the post-Free ledger.
// The caller's permanent/host fallback and successful-miss allocation order are
// deliberately outside this contract; this is not a session-wide memory bound.
// fak-test:runtime fast est=10ms lane=default
func TestPagedRingOversizedMissSkipsUpload(t *testing.T) {
	for _, tc := range []struct {
		name   string
		async  bool
		shared bool
	}{
		{name: "sync-private"},
		{name: "async-private", async: true},
		{name: "sync-shared", shared: true},
		{name: "async-shared", async: true, shared: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &ringPreUploadBackend{Backend: compute.Default()}
			ab := &ringPreUploadAsyncBackend{ringPreUploadBackend: rec}
			var be compute.Backend = rec
			if tc.async {
				be = ab
			}
			const budget = int64(8)
			r := newPagedRing(be, budget)
			var sh *SharedExpertRing
			if tc.shared {
				var err error
				sh, err = NewSharedExpertRing(SharedExpertRingConfig{Model: &Model{}, Backend: be, BudgetBytes: budget})
				if err != nil {
					t.Fatal(err)
				}
				r = sh.ring
				// Ring primitives are called under the shared owner's lock.
				sh.mu.Lock()
				defer sh.mu.Unlock()
			}
			defer r.freeAll()

			_, ok := r.stage("small", func() compute.Tensor {
				return compute.NewF32(compute.Default(), []int{1}, []float32{1})
			}, compute.F32, 4, false)
			if !ok || rec.uploads != 1 || rec.frees != 0 {
				t.Fatalf("ordinary-miss control: admitted=%v uploads=%d frees=%d", ok, rec.uploads, rec.frees)
			}
			warm, ok := r.stage("warm", func() compute.Tensor {
				return compute.NewF32(compute.Default(), []int{2}, []float32{1, 2})
			}, compute.F32, budget, false)
			if !ok || rec.uploads != 2 || rec.frees != 1 {
				t.Fatalf("exact-budget control: admitted=%v uploads=%d frees=%d", ok, rec.uploads, rec.frees)
			}
			if tc.async && (len(ab.fences) != rec.uploads || ab.fences[0].waits != 1 || ab.fences[1].waits != 1) {
				t.Fatal("async controls did not issue and settle exactly two transfers")
			}

			builds := 0
			mk := func() compute.Tensor {
				builds++
				return compute.NewF32(compute.Default(), []int{3}, []float32{3, 4, 5})
			}
			// Include the int64 ceiling: the guard must not add byte counts
			// and overflow while deciding a guaranteed oversized miss.
			oversizedBytes := int64(12)
			if tc.async {
				oversizedBytes = int64(^uint64(0) >> 1)
			}
			if _, ok := r.stage("oversized", mk, compute.F32, oversizedBytes, false); ok {
				t.Fatal("oversized miss was admitted")
			}
			if builds != 0 || rec.uploads != 2 || rec.frees != 1 {
				t.Fatalf("oversized miss built or staged a doomed weight: builds=%d uploads=%d frees=%d", builds, rec.uploads, rec.frees)
			}
			if len(r.pending) != 0 || (tc.async && len(ab.fences) != rec.uploads) {
				t.Fatal("oversized miss issued an async transfer")
			}
			if r.used() != budget || r.peakUsed() != budget || r.residentCount() != 1 || !r.isResident("warm") || r.isResident("oversized") {
				t.Fatal("oversized refusal changed resident weights or their footprint")
			}
			if r.lookups != 3 || r.pageIn != 2 || r.refused != 1 || r.hit != 0 || r.evict != 1 || r.pageInBytes != budget+4 {
				t.Fatalf("refusal accounting: lookups=%d pageIns=%d refusals=%d hits=%d evictions=%d pageInBytes=%d", r.lookups, r.pageIn, r.refused, r.hit, r.evict, r.pageInBytes)
			}
			if sh != nil && (sh.ledger.demands != 3 || sh.ledger.refusals != 1 || sh.ledger.pageInBytes != budget+4) {
				t.Fatal("oversized refusal was not reconciled in the shared ledger")
			}

			// Existing residents have immutable pool descriptors. A repeated key
			// remains a hit even if a caller supplies different byte metadata.
			hit, ok := r.stage("warm", mk, compute.F32, 12, false)
			if !ok || hit.Buf() != warm.Buf() || builds != 0 || rec.uploads != 2 || rec.frees != 1 || r.hit != 1 || r.refused != 1 {
				t.Fatal("oversized-miss guard changed the existing-resident hit path")
			}
			if r.lookups != r.hit+r.pageIn+r.refused {
				t.Fatal("lookup reconciliation failed")
			}
		})
	}
}
