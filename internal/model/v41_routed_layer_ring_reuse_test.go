package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

type v41RoutedLayerReuseBackend struct {
	*v41RoutedLayerTestBackend
	declineQ3     bool
	weightUploads int
	failWeightAt  int
	uploadFault   error
}

func (b *v41RoutedLayerReuseBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.declineQ3 && dt == compute.Q3_K) && b.v41RoutedLayerTestBackend.SupportsDeviceWeightDtype(dt)
}

func (b *v41RoutedLayerReuseBackend) Upload(src compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt == compute.Q2_K || dt == compute.Q3_K {
		b.weightUploads++
		if b.weightUploads == b.failWeightAt {
			panic(b.uploadFault)
		}
	}
	return b.v41RoutedLayerTestBackend.Upload(src, dt)
}

// A full ring contains the later selected expert B, then newer unselected C.
// Staging picks [A, B] must evict C, retaining B's original three handles. The
// historical staging loop instead evicted B while loading A, then uploaded B
// again, despite the complete demand set fitting exactly in the same budget.
// This is a CPU recorder ownership/traffic witness, not a GPU speed claim.
// fak-test:runtime medium est=5s lane=default
func TestV41RoutedLayerKeepsResidentDemand(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	m := v41IncrementalExpertFixture(t, true, false)
	_, x := v41RoutedLayerFaultPicks(m)
	picks := []routePick{{expert: 0, weight: .25}, {expert: 1, weight: .5}}
	names := func(expert int) [3]string {
		stem := "ffn.experts." + itoa(expert)
		return [3]string{layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), layerName(0, stem+".w2.weight")}
	}
	var expertBytes int64
	for _, name := range names(0) {
		expertBytes += kQuantResidentBytes(m.kqw[name])
	}
	for _, shared := range []bool{false, true} {
		t.Run(map[bool]string{false: "session_ring", true: "shared_ring"}[shared], func(t *testing.T) {
			b := &v41RoutedLayerReuseBackend{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend()}
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			s.ExpertRingBytes = 2 * expertBytes
			if shared {
				sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: b, BudgetBytes: s.ExpertRingBytes})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := sh.Close(); err != nil {
						t.Error(err)
					}
				})
				if err := sh.Attach(s, "demand-reuse"); err != nil {
					t.Fatal(err)
				}
			}
			t.Cleanup(s.Close)
			// Warm B first, then C, releasing each projection. Neither is held
			// when the layer begins; plain LRU would choose B as the victim.
			for _, expert := range []int{1, 2} {
				for _, name := range names(expert) {
					_, release, ok := s.expertInputDeviceWeight(name)
					if !ok {
						t.Fatalf("warm %s declined", name)
					}
					release()
				}
			}
			r := s.expertRing
			before := s.ExpertRing()
			uploads := b.weightUploads
			if before.ResidentBytes != s.ExpertRingBytes || before.ResidentCount != 6 {
				t.Fatalf("fixture ring not full: %+v", before)
			}
			residentB := map[polymodel.ModelID]compute.Buffer{}
			for _, name := range names(1) {
				w, _ := s.resolveExpertWeight(name)
				key := polymodel.ModelID(w.halKey())
				residentB[key] = r.resident[key].Buf()
			}
			layer := s.v41RoutedLayerFunc()
			if layer == nil {
				t.Fatal("batched layer did not bind")
			}
			got, outcome, err := layer(0, picks, x)
			if err != nil || outcome != v41RoutedExpertHandled {
				t.Fatalf("layer outcome=%v error=%v", outcome, err)
			}
			after := s.ExpertRing()
			if delta := b.weightUploads - uploads; delta != 3 {
				t.Fatalf("backend Upload calls=%d, want 3 (unfixed loop makes 6)", delta)
			}
			if after.PageIns-before.PageIns != 3 || after.Hits-before.Hits != 3 || after.Evictions-before.Evictions != 3 || after.PageInBytes-before.PageInBytes != expertBytes {
				t.Fatalf("want only A uploaded and C evicted; before=%+v after=%+v", before, after)
			}
			if after.Lookups-before.Lookups != 6 || after.Refusals != before.Refusals || after.Prefetched != before.Prefetched || after.ResidentBytes != 2*expertBytes || after.PeakBytes > after.BudgetBytes {
				t.Fatalf("demand accounting or bound changed: before=%+v after=%+v", before, after)
			}
			for key, handle := range residentB {
				if r.resident[key].Buf() != handle {
					t.Fatalf("later selected resident %s was replaced", key)
				}
			}
			for _, name := range names(2) {
				w, _ := s.resolveExpertWeight(name)
				if r.isResident(w.halKey()) {
					t.Fatalf("unselected projection %s was retained instead of evicted", name)
				}
			}
			var expectedWeights []compute.Buffer
			for _, pick := range picks {
				for _, name := range names(pick.expert) {
					w, _ := s.resolveExpertWeight(name)
					expectedWeights = append(expectedWeights, r.resident[polymodel.ModelID(w.halKey())].Buf())
				}
			}
			if !reflect.DeepEqual(b.matmulWeights, expectedWeights) {
				t.Fatal("kernel weight order changed from pick-major gate/up/down")
			}
			checkReleased := func() {
				t.Helper()
				if len(r.holds) != 0 || len(r.heldPins) != 0 || s.ringDepth != 0 {
					t.Fatalf("leaked holds=%v temporary pins=%v shared depth=%d", r.holds, r.heldPins, s.ringDepth)
				}
			}
			checkReleased()
			// A fully warm replay has identical arithmetic and six real hits.
			want, outcome, err := layer(0, picks, x)
			if err != nil || outcome != v41RoutedExpertHandled || !reflect.DeepEqual(got, want) {
				t.Fatalf("cold/warm arithmetic differs: outcome=%v error=%v", outcome, err)
			}
			warm := s.ExpertRing()
			if warm.PageIns != after.PageIns || warm.Hits-after.Hits != 6 {
				t.Fatalf("warm replay uploaded weights: before=%+v after=%+v", after, warm)
			}
			checkReleased()
			// Repeated picks alias the same HAL keys. Extra temporary holds
			// must nest, without phantom uploads or a lingering ownership count.
			duplicates := []routePick{{expert: 1, weight: .25}, {expert: 1, weight: .5}}
			if _, outcome, err := layer(0, duplicates, x); outcome != v41RoutedExpertHandled || err != nil {
				t.Fatalf("duplicate key outcome=%v error=%v", outcome, err)
			}
			duplicated := s.ExpertRing()
			if duplicated.PageIns != warm.PageIns || duplicated.Hits-warm.Hits != 6 || b.weightUploads-uploads != 3 {
				t.Fatalf("duplicate keys changed residency traffic: before=%+v after=%+v", warm, duplicated)
			}
			checkReleased()
			// A late dtype decline must unwind the lookahead holds as well as
			// gate/up acquisition, without removing a pre-existing durable pin.
			w, _ := s.resolveExpertWeight(names(1)[0])
			pinnedKey := polymodel.ModelID(w.halKey())
			pinned, _ := r.pool.Get(pinnedKey)
			pinned.Pinned = true
			r.repin(pinnedKey, pinned)
			b.declineQ3 = true
			calls := b.calls
			if result, outcome, err := layer(0, picks, x); result != nil || outcome != v41RoutedExpertDeclined || err != nil || b.calls != calls {
				t.Fatalf("dtype decline dispatched: outcome=%v error=%v", outcome, err)
			}
			checkReleased()
			if pinned, ok := r.pool.Get(pinnedKey); !ok || !pinned.Pinned {
				t.Fatal("decline cleared a pre-existing pin")
			}
			// Remove only the test's durable pin, then fail the second new
			// upload. Gate acquisition and all lookahead holds must unwind
			// before selected-device failure closes/detaches the session.
			pinned.Pinned = false
			r.repin(pinnedKey, pinned)
			b.declineQ3 = false
			b.failWeightAt = b.weightUploads + 2
			b.uploadFault = &compute.BackendError{Backend: "test", Site: "Upload", Err: compute.ErrVulkanExecutionFailed}
			failurePicks := []routePick{{expert: 2, weight: .25}, {expert: 0, weight: .5}}
			result, outcome, err := layer(0, failurePicks, x)
			if result != nil || outcome != v41RoutedExpertError || !errors.Is(err, b.uploadFault) || !s.BackendSessionClosed() {
				t.Fatalf("selected upload failure outcome=%v error=%v closed=%t", outcome, err, s.BackendSessionClosed())
			}
			checkReleased()
		})
	}
}
