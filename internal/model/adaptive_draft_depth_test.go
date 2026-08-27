package model

import "testing"

func TestAdaptiveDraftDepthHysteresisAndReceipt(t *testing.T) {
	c := AdaptiveDraftDepthController{MinDepth: 1, MaxDepth: 4, RaiseAcceptance: .8, LowerAcceptance: .3, HysteresisWindows: 2}
	samples := []AdaptiveDraftSample{{4, 4, 100, 200, 1, 2}, {4, 4, 110, 210, 1, 2}, {0, 3, 120, 220, 2, 3}, {0, 2, 130, 230, 2, 3}}
	r, e := EvaluateAdaptiveDraftDepth(c, 2, samples, "qwen3.8-exact", false)
	if e != nil {
		t.Fatal(e)
	}
	want := []int{1, 2, 2, 1}
	for i, v := range want {
		if r.Depths[i] != v {
			t.Fatalf("depths=%v", r.Depths)
		}
	}
	if r.Engine != "fak-native" || r.AcceptedTokens != 8 || r.ProposedTokens != 13 || r.VerifierBytes != 460 || r.VerifierFLOPs != 860 || r.FinalDepth != 1 {
		t.Fatalf("receipt=%+v", r)
	}
}
func TestAdaptiveDraftDepthBoundedAdversarialAndDFlashRejection(t *testing.T) {
	c := AdaptiveDraftDepthController{MinDepth: 1, MaxDepth: 2, RaiseAcceptance: .75, LowerAcceptance: .25, HysteresisWindows: 2}
	var s []AdaptiveDraftSample
	for i := 0; i < 20; i++ {
		if i%2 == 0 {
			s = append(s, AdaptiveDraftSample{Accepted: 1, Proposed: 1})
		} else {
			s = append(s, AdaptiveDraftSample{Accepted: 0, Proposed: 1})
		}
	}
	r, e := EvaluateAdaptiveDraftDepth(c, 2, s, "qwen3.8-exact", true)
	if e != nil {
		t.Fatal(e)
	}
	for _, d := range r.Depths {
		if d < 1 || d > 2 {
			t.Fatalf("unbounded %v", r.Depths)
		}
	}
	if r.Selector != "dflash2-algorithmic-candidate-rejected-no-trained-layer" {
		t.Fatal(r.Selector)
	}
}
func TestAdaptiveDraftDepthRejectsInvalid(t *testing.T) {
	c := AdaptiveDraftDepthController{MinDepth: 1, MaxDepth: 2, RaiseAcceptance: .8, LowerAcceptance: .2, HysteresisWindows: 2}
	if _, e := c.Observe(AdaptiveDraftSample{Accepted: 2, Proposed: 1}); e == nil {
		t.Fatal("accepted impossible sample")
	}
}
