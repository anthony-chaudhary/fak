package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// Only the storage lifecycle is exercised. The embedded backend is deliberately
// nil: no compute or allocation method may be used by this metadata-only fixture.
type arithmeticPolicyBackend struct {
	compute.Backend
	uploads int
	frees   int
	events  []string
}

func (b *arithmeticPolicyBackend) Upload(src compute.Tensor, _ compute.Dtype) compute.Tensor {
	b.uploads++
	b.events = append(b.events, "upload")
	return src
}

func (b *arithmeticPolicyBackend) Free(compute.Tensor) {
	b.frees++
	b.events = append(b.events, "free")
}

// Heat and recency deliberately disagree. A wrapping fit guard must not silently
// make the value-aware policy use Pool's LRU victim at a large byte budget.
// All sizes are metadata; the resident handles are fixed-count empty sentinels.
// fak-test:runtime fast est=10ms lane=default
func TestExpertRingAdmissionArithmeticPreservesPolicy(t *testing.T) {
	const maxBytes = int64(^uint64(0) >> 1)
	for _, tc := range []struct {
		name   string
		budget int64
		policy ExpertRingEvictPolicy
		victim string
	}{
		{name: "small-lru", budget: 8, victim: "older-hot"},
		{name: "max-lru", budget: maxBytes, victim: "older-hot"},
		{name: "small-value", budget: 8, policy: ExpertRingEvictValueAware, victim: "newer-cold"},
		{name: "max-value", budget: maxBytes, policy: ExpertRingEvictValueAware, victim: "newer-cold"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			be := &arithmeticPolicyBackend{}
			r := newPagedRing(be, tc.budget)
			r.policy = tc.policy
			defer r.freeAll()
			pin := polymodel.Model{ID: "pin", WeightBytes: tc.budget - 2, Pinned: true}
			for _, m := range []polymodel.Model{
				{ID: "older-hot", WeightBytes: 1}, pin,
				{ID: "newer-cold", WeightBytes: 1},
			} {
				if victims, err := r.pool.Admit(m); err != nil || len(victims) != 0 {
					t.Fatalf("seed Admit = (%v, %v)", victims, err)
				}
				r.resident[m.ID] = compute.Tensor{}
			}
			r.lastUse = map[polymodel.ModelID]uint64{"older-hot": 1, "pin": 2, "newer-cold": 3}
			r.clock = 3
			r.peak = tc.budget
			if tc.policy == ExpertRingEvictValueAware {
				r.heat = map[polymodel.ModelID]int{"older-hot": 10, "pin": 0, "newer-cold": 1}
			}
			builds := 0
			if _, ok := r.stage("next", func() compute.Tensor {
				builds++
				return compute.Tensor{}
			}, compute.F32, 1, false); !ok {
				t.Fatal("fitting metadata-only stage refused")
			}
			if r.isResident(tc.victim) || r.pool.Has(polymodel.ModelID(tc.victim)) {
				t.Fatalf("policy victim %q remained resident", tc.victim)
			}
			for _, name := range []string{"older-hot", "pin", "newer-cold", "next"} {
				if name != tc.victim && (!r.isResident(name) || !r.pool.Has(polymodel.ModelID(name))) {
					t.Fatalf("expected survivor %q is absent from ring or Pool", name)
				}
			}
			if got, ok := r.pool.Get("pin"); !ok || got != pin {
				t.Fatal("pinned resident changed")
			}
			if builds != 1 || be.uploads != 1 || be.frees != 1 || r.evict != 1 || r.pageIn != 1 || r.refused != 0 {
				t.Fatalf("lifecycle builds/uploads/frees/evictions/page-ins/refusals = %d/%d/%d/%d/%d/%d",
					builds, be.uploads, be.frees, r.evict, r.pageIn, r.refused)
			}
			if len(be.events) != 2 || be.events[0] != "upload" || be.events[1] != "free" {
				t.Fatalf("fitting lifecycle = %v, want [upload free]", be.events)
			}
			if r.used() != tc.budget || r.peakUsed() != tc.budget || r.residentCount() != 3 || r.pool.Len() != 3 {
				t.Fatal("ring and Pool did not retain the exact bounded resident set")
			}
			if tc.policy == ExpertRingEvictLRU && r.heat != nil {
				t.Fatal("LRU control allocated value-aware heat")
			}
		})
	}
}
