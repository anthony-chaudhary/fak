package resourcelifecycle

import "testing"

func claim(kind, owner, compat string, mutable, share bool) Claim {
	return Claim{Kind: kind, Owner: owner, Isolation: owner, Lifetime: "session", Compatibility: compat, Mutable: mutable, Shareable: share, Bytes: 1024, Geometry: Geometry{Shape: []int{1, 2}, Alignment: 64}}
}
func TestUnifiedKindsReuseAndTeardown(t *testing.T) {
	m := New()
	kinds := []string{"model_weights", "attention_kv", "qwen_gdn_state", "tool_artifact"}
	refs := map[string]Ref{}
	for _, kind := range kinds {
		c := claim(kind, "session-a", kind+"/v1", kind != "model_weights", kind == "model_weights")
		a, err := m.Resolve(c, "host", "device")
		if err != nil {
			t.Fatal(err)
		}
		refs[kind] = a.Ref
		r, _ := m.Get(a.Ref)
		if r.TransferBytes != 1024 || r.PlannedLocality != "host" || r.ActualLocality != "device" {
			t.Fatal(r)
		}
	}
	again, err := m.Resolve(claim("model_weights", "session-a", "model_weights/v1", false, true), "device", "device")
	if err != nil {
		t.Fatal(err)
	}
	if again.Ref != refs["model_weights"] {
		t.Fatal("compatible immutable state not reused")
	}
	different, _ := m.Resolve(claim("model_weights", "session-a", "model_weights/v2", false, true), "device", "device")
	if different.Ref == again.Ref {
		t.Fatal("incompatible state reused")
	}
	released := m.Teardown("session-a")
	if len(released) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture creates five owned resources and requires every one to be released
		t.Fatalf("released=%d", len(released))
	}
	for _, ref := range refs {
		r, _ := m.Get(ref)
		if !r.Released {
			t.Fatalf("leaked %+v", r)
		}
	}
}
func TestResolvedHandleAndNetTrueObservation(t *testing.T) {
	m := New()
	a, _ := m.Resolve(claim("attention_kv", "s", "kv1", true, false), "host", "device")
	if err := m.Observe(Observation{Ref: a.Ref, Action: "recompute", RecomputeBytes: 128, Reason: "invalidated_lineage"}); err != nil {
		t.Fatal(err)
	}
	r, ok := m.Get(a.Ref)
	if !ok || r.RecomputeBytes != 128 || r.TransferBytes != 1024 {
		t.Fatal(r)
	}
}
