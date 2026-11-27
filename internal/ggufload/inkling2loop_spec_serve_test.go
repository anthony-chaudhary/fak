package ggufload

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInkling2ServeSpecPlanResolveAndErrors(t *testing.T) {
	unset := func(string) (string, bool) { return "", false }

	spec, err := OpenInkling2ServeSpec(nil, nil, unset)
	if err != nil {
		t.Fatalf("nil OpenInkling2ServeSpec err = %v", err)
	}
	if spec.Active() || spec.Label() != "spec=off" {
		t.Fatalf("nil spec = active:%v label:%q, want off", spec.Active(), spec.Label())
	}
	_ = spec.Close()

	m := &Inkling2LoopModel{}
	all := fakeVerifyAllKVBackend{}

	if _, err := OpenInkling2ServeSpec(m, all, func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_SPEC":
			return "4", true
		case "FAK_INKLING2_SPEC_DRAFT":
			return "bogus", true
		}
		return "", false
	}); err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_SPEC_DRAFT") {
		t.Fatalf("invalid draft kind err = %v", err)
	}

	if _, err := OpenInkling2ServeSpec(m, all, func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_SPEC":
			return "4", true
		case "FAK_INKLING2_SPEC_DRAFT":
			return "dspark", true
		}
		return "", false
	}); err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_MTP_DIR") {
		t.Fatalf("dspark without mtp dir err = %v", err)
	}

	if _, err := OpenInkling2ServeSpec(m, fakeNVFP4NoVerifyBackend{}, func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_SPEC":
			return "4", true
		case "FAK_INKLING2_KV":
			return "nvfp4", true
		}
		return "", false
	}); err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_SPEC=0") {
		t.Fatalf("unsupported nvfp4 verify err = %v", err)
	}

	spec, err = OpenInkling2ServeSpec(m, all, func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_SPEC":
			return "4", true
		case "FAK_INKLING2_KV":
			return "nvfp4", true
		}
		return "", false
	})
	if err != nil {
		t.Fatalf("valid nvfp4 OpenInkling2ServeSpec err = %v", err)
	}
	defer spec.Close()
	if !spec.Active() || spec.Depth() != 4 || spec.DraftKind() != inkling2DraftLookup || spec.Label() != "spec=lookup depth=4" {
		t.Fatalf("got active=%v depth=%d kind=%q label=%q", spec.Active(), spec.Depth(), spec.DraftKind(), spec.Label())
	}
	if err := spec.Close(); err != nil {
		t.Fatalf("first spec.Close() err = %v", err)
	}
	if err := spec.Close(); err != nil {
		t.Fatalf("second spec.Close() err = %v", err)
	}

	// Nil lookup must read ambient env without panicking, and nil spec / nil model must fail or report cleanly.
	t.Setenv("FAK_INKLING2_SPEC", "4")
	t.Setenv("FAK_INKLING2_KV", "nvfp4")
	spec, err = OpenInkling2ServeSpec(m, all, nil)
	if err != nil {
		t.Fatalf("nil-lookup OpenInkling2ServeSpec err = %v", err)
	}
	_ = spec.Close()
	if !spec.Active() || spec.Depth() != 4 || spec.DraftKind() != inkling2DraftLookup {
		t.Fatalf("nil-lookup spec = active:%v depth:%d kind:%q, want active depth=4 lookup", spec.Active(), spec.Depth(), spec.DraftKind())
	}
	var nilSpec *Inkling2ServeSpec
	if _, err := nilSpec.OpenSession(); err == nil {
		t.Fatal("nilSpec.OpenSession() succeeded, want error")
	}
	type fakeDFlashDev struct {
		fakeVerifyAllKVBackend
		compute.Inkling2DFlash2Device
	}
	if label := Inkling2SpecLabelFor(nil, fakeDFlashDev{}, nil); label != "spec=lookup depth=4" {
		t.Fatalf("Inkling2SpecLabelFor(nil, fakeDFlashDev, nil) = %q, want spec=lookup depth=4", label)
	}
}

func TestInkling2ResidentSpecCheckpointRestoresDraftRingsAndHistory(t *testing.T) {
	r, d := inkling2RewindFixture(t, false)
	s := r.s
	s.specMax = 5

	draftHP := model.Inkling2DSparkHParams{KVHeads: 1, HeadDim: 2}
	alloc := func(n int) ink2Ref {
		tt, err := d.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		return compute.Ink2At(tt, 0)
	}
	// Layer 0 mimics local window (cap=4, win=4), layer 1 mimics bounded 128k global window (cap=8, win=6).
	draft := &ink2DeviceDraft{
		s:  s,
		hp: draftHP,
		layers: []ink2DLayer{
			{kc: alloc(4 * 2), vc: alloc(4 * 2), cap: 4, window: 4},
			{kc: alloc(8 * 2), vc: alloc(8 * 2), cap: 8, window: 6},
		},
	}
	s.draft = draft

	for l, dl := range draft.layers {
		for i := range d.rows(dl.kc) {
			d.rows(dl.kc)[i] = float32(100*(l+1) + i + 1)
			d.rows(dl.vc)[i] = float32(500*(l+1) + i + 1)
		}
	}
	wantK0 := append([]float32(nil), d.rows(draft.layers[0].kc)...)
	wantV0 := append([]float32(nil), d.rows(draft.layers[0].vc)...)
	wantK1 := append([]float32(nil), d.rows(draft.layers[1].kc)...)
	wantV1 := append([]float32(nil), d.rows(draft.layers[1].vc)...)

	// pos=6, horizon=3 -> local draft (cap=4, win=4) saves [2,5), global draft (cap=8, win=6) saves [0,1).
	if err := r.Checkpoint(3); err != nil {
		t.Fatalf("Checkpoint on spec session failed: %v", err)
	}
	for _, p := range []int{2, 3, 4} {
		for j := 0; j < 2; j++ {
			d.rows(draft.layers[0].kc)[(p%4)*2+j] = -9
			d.rows(draft.layers[0].vc)[(p%4)*2+j] = -8
		}
	}
	for j := 0; j < 2; j++ {
		d.rows(draft.layers[1].kc)[(0%8)*2+j] = -7
		d.rows(draft.layers[1].vc)[(0%8)*2+j] = -6
	}
	s.pos = 9
	if err := r.Restore(); err != nil {
		t.Fatalf("Restore on spec session failed: %v", err)
	}
	if s.pos != 6 {
		t.Fatalf("pos = %d, want 6", s.pos)
	}
	if !reflect.DeepEqual(d.rows(draft.layers[0].kc), wantK0) || !reflect.DeepEqual(d.rows(draft.layers[0].vc), wantV0) {
		t.Fatalf("draft local ring mismatch after restore: gotK=%v wantK=%v", d.rows(draft.layers[0].kc), wantK0)
	}
	if !reflect.DeepEqual(d.rows(draft.layers[1].kc), wantK1) || !reflect.DeepEqual(d.rows(draft.layers[1].vc), wantV1) {
		t.Fatalf("draft global ring mismatch after restore: gotK=%v wantK=%v", d.rows(draft.layers[1].kc), wantK1)
	}
	if err := r.Reset(); err != nil {
		t.Fatalf("Reset on spec session failed: %v", err)
	}
	if s.pos != 0 {
		t.Fatalf("pos after Reset = %d, want 0", s.pos)
	}
}

type trimRecordingBackend struct {
	compute.Backend
	trims []int
}

func (b *trimRecordingBackend) TrimLarge(maxKeep int) {
	b.trims = append(b.trims, maxKeep)
}

func TestInkling2ResidentPrefillRegrowAndCloseTrimsIdlePool(t *testing.T) {
	r, _ := inkling2RewindFixture(t, false)
	tb := &trimRecordingBackend{Backend: r.s.be}
	r.s.be = tb

	r.s.ensurePrefillArenas(8)
	if r.s.err != nil {
		t.Fatalf("initial ensurePrefillArenas(8) err = %v", r.s.err)
	}
	if len(tb.trims) != 0 {
		t.Fatalf("initial alloc triggered unexpected trims %v", tb.trims)
	}
	r.s.ensurePrefillArenas(32)
	if r.s.err != nil {
		t.Fatalf("regrow ensurePrefillArenas(32) err = %v", r.s.err)
	}
	if len(tb.trims) != 1 || tb.trims[0] != inkling2BatchTrimKeep {
		t.Fatalf("regrow trims = %v, want [%d]", tb.trims, inkling2BatchTrimKeep)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close err = %v", err)
	}
	if len(tb.trims) != 2 || tb.trims[1] != inkling2BatchTrimKeep {
		t.Fatalf("post-Close trims = %v, want 2 trims at %d", tb.trims, inkling2BatchTrimKeep)
	}
	// Double Close on a closed or nil session must be a no-op.
	if err := r.Close(); err != nil {
		t.Fatalf("second Close err = %v", err)
	}
	var nilSess *inkling2ResidentSession
	if err := nilSess.Close(); err != nil {
		t.Fatalf("nil Close err = %v", err)
	}
}

type noZeroAllocDev struct {
	*fakeInk2Cohort
}

func (n noZeroAllocDev) Ink2Alloc(elems int) (compute.Tensor, error) {
	if elems <= 0 {
		return compute.Tensor{}, fmt.Errorf("unexpected zero-element Ink2Alloc(%d)", elems)
	}
	return n.fakeInk2Cohort.Ink2Alloc(elems)
}

func TestInkling2ResidentSpecCheckpointZeroRowsSkipsZeroAlloc(t *testing.T) {
	r, d := inkling2RewindFixture(t, false)
	nd := noZeroAllocDev{fakeInk2Cohort: d}
	r.s.be = nd
	r.s.dev = nd
	r.s.pos = 1
	r.s.specMax = 1
	tt, err := d.Ink2Alloc(8 * 2)
	if err != nil {
		t.Fatal(err)
	}
	r.s.draft = &ink2DeviceDraft{
		s:      r.s,
		hp:     model.Inkling2DSparkHParams{KVHeads: 1, HeadDim: 2},
		layers: []ink2DLayer{{kc: compute.Ink2At(tt, 0), vc: compute.Ink2At(tt, 0), cap: 8, window: 6}},
	}
	if err := r.Checkpoint(1); err != nil {
		t.Fatalf("Checkpoint(1) with 0 target/draft rows failed on zero-alloc guard: %v", err)
	}
}
