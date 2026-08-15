package categorybaseline

import "testing"

func TestEvaluateCompletedLayerRedirect(t *testing.T) {
	r := Normalize(Registry{Categories: []Category{{Name: "serving", Layers: []string{"medium-model", "l2-cache", "l3-cache"}, CompletedLayer: "medium-model", NextLayer: "l2-cache", Witness: "fak serve selfcheck"}}})
	if got := Evaluate(r, "serving", "medium-model", false); !got.Hold || got.NextLayer != "l2-cache" {
		t.Fatalf("completed layer = %+v", got)
	}
	if got := Evaluate(r, "serving", "l2-cache", false); got.Hold {
		t.Fatalf("next layer held: %+v", got)
	}
	if got := Evaluate(r, "serving", "medium-model", true); got.Hold {
		t.Fatalf("regression held: %+v", got)
	}
	if got := Evaluate(r, "", "", false); got.Hold {
		t.Fatalf("undeclared held: %+v", got)
	}
}
