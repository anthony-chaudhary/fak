package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type pinnedPreflightBackend struct {
	compute.Backend
	uploads int
	frees   int
	events  []string
}

func (b *pinnedPreflightBackend) Upload(src compute.Tensor, dtype compute.Dtype) compute.Tensor {
	b.uploads++
	b.events = append(b.events, "upload")
	return b.Backend.Upload(src, dtype)
}

func (b *pinnedPreflightBackend) Free(t compute.Tensor) {
	b.frees++
	b.events = append(b.events, "free")
	b.Backend.Free(t)
}

type pinnedPreflightAsyncBackend struct {
	*pinnedPreflightBackend
	fences []*asyncTestFence
}

func (b *pinnedPreflightAsyncBackend) UploadAsync(src compute.Tensor, dtype compute.Dtype) (compute.Tensor, compute.Fence) {
	t := b.Upload(src, dtype)
	f := &asyncTestFence{}
	b.fences = append(b.fences, f)
	return t, f
}

func pinnedPreflightSource(values ...float32) compute.Tensor {
	return compute.NewF32(compute.Default(), []int{len(values)}, values)
}

// Four bounded rows cover both victim policies, synchronous/delayed transfers,
// demand/hint refusal and private/shared accounting without a Cartesian suite.
// fak-test:runtime fast est=10ms lane=default
func TestPagedRingPinnedMissSkipsStaging(t *testing.T) {
	for _, tc := range []struct {
		name     string
		policy   ExpertRingEvictPolicy
		async    bool
		prefetch bool
		shared   bool
	}{
		{name: "lru-sync-demand"},
		{name: "value-sync-hint-shared", policy: ExpertRingEvictValueAware, prefetch: true, shared: true},
		{name: "lru-async-hint", async: true, prefetch: true},
		{name: "value-async-demand-shared", policy: ExpertRingEvictValueAware, async: true, shared: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &pinnedPreflightBackend{Backend: compute.Default()}
			ab := &pinnedPreflightAsyncBackend{pinnedPreflightBackend: rec}
			var be compute.Backend = rec
			if tc.async {
				be = ab
			}
			r := newPagedRing(be, 8)
			r.policy = tc.policy
			var sh *SharedExpertRing
			if tc.shared {
				var err error
				sh, err = NewSharedExpertRing(SharedExpertRingConfig{Model: &Model{}, Backend: be, BudgetBytes: 8, Evict: tc.policy})
				if err != nil {
					t.Fatal(err)
				}
				r = sh.ring
				sh.mu.Lock()
				defer sh.mu.Unlock()
			}
			defer r.freeAll()
			for _, name := range []string{"held", "cold"} {
				if _, ok := r.stageInFlight(name, func() compute.Tensor { return pinnedPreflightSource(1) }, compute.F32, 4, false); !ok {
					t.Fatalf("warmup %s refused", name)
				}
			}
			r.hold("held")
			if rec.uploads != 2 || rec.frees != 0 || r.used() != 8 {
				t.Fatal("warmup did not populate the ring")
			}
			if tc.async && len(ab.fences) != rec.uploads {
				t.Fatal("warmup did not create one fence per upload")
			}
			held, _ := r.pool.Get("held")
			cold, _ := r.pool.Get("cold")
			heldHandle := r.resident["held"].Buf()
			coldHandle := r.resident["cold"].Buf()
			pending := len(r.pending)
			pendingFences := make(map[string]compute.Fence, pending)
			for id, fence := range r.pending {
				pendingFences[string(id)] = fence
			}
			fences := append([]*asyncTestFence(nil), ab.fences...)
			events := append([]string(nil), rec.events...)
			builds := 0
			mk := func() compute.Tensor {
				builds++
				return pinnedPreflightSource(2, 3)
			}
			stage := r.stage
			if tc.prefetch {
				stage = r.stageInFlight
			}
			if _, ok := stage("next", mk, compute.F32, 8, false); ok {
				t.Fatal("pinned-no-room miss was admitted")
			}
			if builds != 0 || rec.uploads != 2 || rec.frees != 0 || !reflect.DeepEqual(events, rec.events) {
				t.Fatal("refused miss built, uploaded or freed storage")
			}
			if len(r.pending) != pending || len(ab.fences) != len(fences) {
				t.Fatal("refused miss changed pending transfers")
			}
			for id, fence := range r.pending {
				if pendingFences[string(id)] != fence {
					t.Fatal("refused miss replaced a resident's pending fence")
				}
			}
			for i, f := range fences {
				if ab.fences[i] != f || f.waits != 0 || f.landed {
					t.Fatal("refused miss waited or replaced a warmup fence")
				}
			}
			if got, ok := r.pool.Get("held"); !ok || got != held || !got.Pinned || r.resident["held"].Buf() != heldHandle {
				t.Fatal("refused miss changed the held resident")
			}
			if got, ok := r.pool.Get("cold"); !ok || got != cold || r.resident["cold"].Buf() != coldHandle {
				t.Fatal("refused miss discarded an unpinned resident")
			}
			if r.used() != 8 || r.peakUsed() != 8 || r.isResident("next") || r.lookups != 3 || r.pageIn != 2 || r.refused != 1 || r.evict != 0 || r.pageInBytes != 8 {
				t.Fatal("refusal changed residency or ring accounting")
			}
			wantDemands := int64(0)
			if !tc.prefetch {
				wantDemands = 1
			}
			if sh != nil && (sh.ledger.demands != wantDemands || sh.ledger.refusals != 1 || sh.ledger.pageInBytes != 8) {
				t.Fatal("demand/hint refusal did not reconcile in the shared ledger")
			}

			// Once the hold ends, the same request is feasible. Preserve upload
			// before victim cleanup, and settle all delayed transfers normally.
			r.release("held")
			if _, ok := r.stage("next", mk, compute.F32, 8, false); !ok {
				t.Fatal("released fitting control was refused")
			}
			if builds != 1 || rec.uploads != 3 || rec.frees != 2 || !r.isResident("next") || r.used() != 8 || len(r.pending) != 0 {
				t.Fatal("released fitting control did not use the ordinary lifecycle")
			}
			if !reflect.DeepEqual(rec.events[len(events):], []string{"upload", "free", "free"}) {
				t.Fatal("fitting upload/victim-cleanup order changed")
			}
			for _, f := range ab.fences {
				if f.waits != 1 || !f.landed {
					t.Fatal("fitting control failed to settle a transfer exactly once")
				}
			}
			if r.lookups != r.hit+r.pageIn+r.refused {
				t.Fatal("ring lookup reconciliation failed")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestPagedRingPinnedPreflightPreservesNestedAndDurableHolds(t *testing.T) {
	for _, tc := range []struct {
		name    string
		durable bool
	}{
		{name: "temporary"},
		{name: "durable", durable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := &pinnedPreflightBackend{Backend: compute.Default()}
			r := newPagedRing(rec, 8)
			defer r.freeAll()
			if _, ok := r.stage("held", func() compute.Tensor { return pinnedPreflightSource(1) }, compute.F32, 4, tc.durable); !ok {
				t.Fatal("held warmup refused")
			}
			if _, ok := r.stage("cold", func() compute.Tensor { return pinnedPreflightSource(1) }, compute.F32, 4, false); !ok {
				t.Fatal("cold warmup refused")
			}
			r.hold("held")
			r.hold("held")
			builds := 0
			mk := func() compute.Tensor {
				builds++
				return pinnedPreflightSource(2, 3)
			}
			refuse := func() {
				t.Helper()
				if _, ok := r.stage("next", mk, compute.F32, 8, false); ok || builds != 0 || rec.uploads != 2 || rec.frees != 0 || !r.isResident("cold") {
					t.Fatal("live pin did not refuse before staging and preserve both residents")
				}
			}
			refuse()
			r.release("held")
			refuse()
			r.release("held")
			if tc.durable {
				refuse()
				if descriptor, ok := r.pool.Get("held"); !ok || !descriptor.Pinned {
					t.Fatal("hold release cleared a durable pin")
				}
				return
			}
			if _, ok := r.stage("next", mk, compute.F32, 8, false); !ok || builds != 1 || rec.uploads != 3 || rec.frees != 2 {
				t.Fatal("final temporary release did not permit ordinary admission")
			}
		})
	}
}

// The early refusal avoids only the doomed ring upload. Existing HAL fallback
// remains permanent and unbounded, builds/uploads once, then memoizes the handle.
// fak-test:runtime fast est=10ms lane=default
func TestExpertHALPinnedPreflightRetainsPermanentFallback(t *testing.T) {
	rec := &pinnedPreflightBackend{Backend: compute.Default()}
	s := &Session{M: &Model{}, Backend: rec, ExpertRingBytes: 8, halW: map[string]compute.Tensor{}}
	defer s.Close()
	name := "model.layers.0.mlp.experts.2.gate_proj.weight"
	key := "f32:" + name
	r := s.routedExpertRing(name)
	if r == nil {
		t.Fatal("routed expert did not create a ring")
	}
	for _, warm := range []string{"held", "cold"} {
		if _, ok := r.stage(warm, func() compute.Tensor { return pinnedPreflightSource(1) }, compute.F32, 4, false); !ok {
			t.Fatal("ring warmup refused")
		}
	}
	r.hold("held")
	builds := 0
	mk := func() compute.Tensor {
		builds++
		return pinnedPreflightSource(2, 3)
	}
	w := s.weightHALStagedBounded(key, name, mk, compute.F32, 8)
	if builds != 1 || rec.uploads != 3 || rec.frees != 0 || r.refused != 1 || r.used() != 8 || r.evict != 0 || r.isResident(key) {
		t.Fatal("pinned refusal did not keep the one-upload permanent fallback")
	}
	if stored, ok := s.halW[key]; !ok || stored.Buf() != w.Buf() {
		t.Fatal("fallback handle was not memoized in halW")
	}
	lookups := r.lookups
	if again := s.weightHALStagedBounded(key, name, mk, compute.F32, 8); again.Buf() != w.Buf() || builds != 1 || rec.uploads != 3 || r.lookups != lookups {
		t.Fatal("permanent fallback memo hit restaged or consulted the ring")
	}
}
