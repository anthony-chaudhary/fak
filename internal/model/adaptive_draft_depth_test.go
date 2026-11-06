package model

import (
	"errors"
	"testing"
)

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

// specDrafter models the speculator serving path: it caches the fitted draft
// limit at construction and checks the controller generation before each
// draft, refreshing the cache only when a fit has advanced it.
type specDrafter struct {
	ctrl  *AdaptiveDraftDepthController
	gen   int
	limit int
}

func newSpecDrafter(ctrl *AdaptiveDraftDepthController) *specDrafter {
	return &specDrafter{ctrl: ctrl, gen: ctrl.Generation(), limit: ctrl.Limit()}
}

func (d *specDrafter) draftLimit() int {
	if g := d.ctrl.Generation(); g != d.gen {
		d.gen = g
		d.limit = d.ctrl.Limit()
	}
	return d.limit
}

// fak-test:runtime fast est=20ms lane=default
func TestAdaptiveDraftDepthSpecLimitsSync(t *testing.T) {
	c := AdaptiveDraftDepthController{MinDepth: 1, MaxDepth: 4, RaiseAcceptance: .8, LowerAcceptance: .3, HysteresisWindows: 2}
	d := newSpecDrafter(&c)
	if d.limit != 1 || d.gen != 0 {
		t.Fatalf("construction cache limit=%d gen=%d", d.limit, d.gen)
	}
	// In-band sample: no fit, no generation change - the drafter reuses its
	// cached construction-time limit.
	if _, e := c.Observe(AdaptiveDraftSample{Accepted: 1, Proposed: 2}); e != nil {
		t.Fatal(e)
	}
	if got := d.draftLimit(); got != 1 || d.gen != 0 {
		t.Fatalf("no-fit draft limit=%d gen=%d", got, d.gen)
	}
	// Two accepted windows fit the depth up; the generation must advance.
	for i := 0; i < 2; i++ {
		if _, e := c.Observe(AdaptiveDraftSample{Accepted: 4, Proposed: 4}); e != nil {
			t.Fatal(e)
		}
	}
	if c.Generation() != 1 || c.Limit() != 2 {
		t.Fatalf("post-fit generation=%d limit=%d", c.Generation(), c.Limit())
	}
	// Fit then draft: the drafter sees the new generation and drafts with the
	// fitted limit, not the stale construction-time one.
	if got := d.draftLimit(); got != 2 {
		t.Fatalf("fit-then-draft limit=%d", got)
	}
	// Two rejected windows fit the depth down; the next draft follows again.
	for i := 0; i < 2; i++ {
		if _, e := c.Observe(AdaptiveDraftSample{Accepted: 0, Proposed: 4}); e != nil {
			t.Fatal(e)
		}
	}
	if c.Generation() != 2 || c.Limit() != 1 {
		t.Fatalf("lower-fit generation=%d limit=%d", c.Generation(), c.Limit())
	}
	if got := d.draftLimit(); got != 1 {
		t.Fatalf("lower-fit draft limit=%d", got)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestAdaptiveDraftDepthModuleNaming(t *testing.T) {
	key, e := AdaptiveDraftModuleKey("qwen3.8-exact", 4)
	if e != nil {
		t.Fatal(e)
	}
	if key != "qwen3.8-exact_d4" {
		t.Fatalf("key=%q", key)
	}
	reg := NewAdaptiveDraftModuleRegistry()
	if e := reg.Compile("qwen3.8-exact", 4, "module-d4"); e != nil {
		t.Fatal(e)
	}
	got, ok := reg.Lookup("qwen3.8-exact", 4)
	if !ok || got != "module-d4" {
		t.Fatalf("same-length lookup=(%q,%v)", got, ok)
	}
	if _, ok := reg.Lookup("qwen3.8-exact", 5); ok {
		t.Fatal("length-mismatched lookup must miss")
	}
	if e := reg.Compile("qwen3.8-exact", 5, "module-d5"); e != nil {
		t.Fatal(e)
	}
	got, ok = reg.Lookup("qwen3.8-exact", 5)
	if !ok || got != "module-d5" {
		t.Fatalf("recompiled lookup=(%q,%v)", got, ok)
	}
	if _, ok := reg.Lookup("other-checkpoint", 4); ok {
		t.Fatal("cross-artifact lookup must miss")
	}
	if e := reg.Compile("", 4, "module"); !errors.Is(e, ErrInvalidAdaptiveDraftModule) {
		t.Fatalf("empty artifact err=%v", e)
	}
	if e := reg.Compile("qwen3.8-exact", 0, "module"); !errors.Is(e, ErrInvalidAdaptiveDraftModule) {
		t.Fatalf("zero draft length err=%v", e)
	}
	if e := reg.Compile("qwen3.8-exact", 4, ""); !errors.Is(e, ErrInvalidAdaptiveDraftModule) {
		t.Fatalf("empty module err=%v", e)
	}
}
