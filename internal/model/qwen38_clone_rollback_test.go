package model

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestQwen38MTP_ClonePreservesExactPrefix(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	target, serial := m.NewSession(), m.NewSession()
	t.Cleanup(target.Close)
	t.Cleanup(serial.Close)
	target.captureTargetHidden, serial.captureTargetHidden = true, true
	boundary := target.Prefill([]int{0, 1})
	serial.Prefill([]int{0, 1})
	// A snapshot also preserves the cache's non-append mutation history.
	target.Cache.mutGen, serial.Cache.mutGen = 7, 7

	if cloneFloat32WithReserve(nil, 0) != nil {
		t.Fatal("clone materialized an absent plane without a reservation")
	}
	if cloneFloat32WithReserve([]float32{}, 0) == nil {
		t.Fatal("clone erased an explicitly present empty plane")
	}
	if got := cloneFloat32WithReserve(nil, 4); got == nil || len(got) != 0 || cap(got) < 4 {
		t.Fatal("clone dropped an explicit capacity reservation")
	}
	for _, reserve := range []int{0, 4} {
		cloned := target.Cache.CloneWithReserve(reserve)
		// Compare against the actual, unnormalized prefix, never another clone.
		if !reflect.DeepEqual(cloned, target.Cache) {
			t.Fatalf("reserve %d changed raw prefix state", reserve)
		}
		for layer := range cloned.K {
			if len(cloned.K[layer]) != 0 {
				cloned.K[layer][0]++
				cloned.Kraw[layer][0]++
				cloned.V[layer][0]++
			}
			state := &cloned.linear.layers[layer]
			if len(state.recurrent) != 0 {
				state.recurrent[0][0]++
			}
			if len(state.conv) != 0 {
				state.conv[0][0]++
			}
		}
		assertQwen35MTPTargetStateEqual(t, target, serial)
	}

	snapshot, err := target.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	rollback, err := snapshot.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer rollback.Close()
	if !reflect.DeepEqual(rollback.Cache, target.Cache) {
		t.Fatal("independent rollback owner changed the raw prefix")
	}
	next := argmaxF32(boundary)
	target.Step(next)
	if err := rollback.Restore(target); err != nil {
		t.Fatal(err)
	}
	assertQwen35MTPTargetStateEqual(t, target, serial)
	if got, want := target.Step(next), serial.Step(next); !reflect.DeepEqual(got, want) {
		t.Fatal("restored prefix changed scalar continuation logits")
	}
	assertQwen35MTPTargetStateEqual(t, target, serial)
}
