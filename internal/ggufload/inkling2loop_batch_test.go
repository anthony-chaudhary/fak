package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fakeInk2SeqOps struct{}

func (fakeInk2SeqOps) Ink2BQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv, st compute.Ink2SeqStrides, C int) error {
	return nil
}
func (fakeInk2SeqOps) Ink2BAttend(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int, st compute.Ink2SeqStrides, C int) error {
	return nil
}
func (fakeInk2SeqOps) Ink2BConv(x, w, hist, out, resid compute.Ink2Ref, n, kn int, norm float32, st compute.Ink2SeqStrides, C int) error {
	return nil
}
func (fakeInk2SeqOps) Ink2BSoftcapArgmax(logits compute.Ink2Ref, n, ld int, softcap float32, outIdx compute.Ink2Ref, C int) error {
	return nil
}

func TestInkling2BatchSeqOpsSelection(t *testing.T) {
	env := func(v string, set bool) func(string) (string, bool) {
		return func(string) (string, bool) { return v, set }
	}
	if inkling2BatchSeqOps(fakeInk2SeqOps{}, env("", false)) == nil {
		t.Fatal("a backend with the batched ops must use them by default")
	}
	if inkling2BatchSeqOps(fakeInk2SeqOps{}, env("1", true)) == nil {
		t.Fatal("FAK_INKLING2_BATCH_SEQOPS=1 must keep the batched ops")
	}
	if inkling2BatchSeqOps(fakeInk2SeqOps{}, env("0", true)) != nil {
		t.Fatal("FAK_INKLING2_BATCH_SEQOPS=0 must force the per-sequence loop")
	}
	if inkling2BatchSeqOps(struct{}{}, env("", false)) != nil {
		t.Fatal("a backend without the batched ops must loop")
	}
}

// The per-sequence fallback addresses row i exactly where the batched kernels do
// (stride*i), and keeps an absent operand absent as the kernels keep null null.
func TestInkling2SeqRow(t *testing.T) {
	if r := inkling2SeqRow(compute.Ink2Ref{}, 7, 3); r.T.Buf() != nil || r.Off != 0 {
		t.Fatalf("absent operand became %+v", r)
	}
	st := compute.Ink2SeqStrides{Q: 1, R: 2, K: 3, V: 4, KDyn: 5, VDyn: 6, Sinks: 7, KCache: 8, VCache: 9, Hist: 10,
		Scores: 12, Out: 13, Meta: 14, X: 15, Resid: 16}
	f := st.Fields()
	for i, want := range []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16} {
		if f[i] != want {
			t.Fatalf("stride field %d = %d, want %d (device launcher order)", i, f[i], want)
		}
	}
}

func TestInkling2BatchCaps(t *testing.T) {
	hp := model.Inkling2LoopHParams{Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 2,
		LocalLayer: []bool{true, false, true, false}}
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	local := make([]int, slots)
	for i := range local {
		local[i] = 160
	}
	got := inkling2BatchCaps(hp, local, 512)
	// slots 4, 5 revisit core layers 1, 2
	want := []int{160, 512, 160, 512, 512, 160}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("caps %v, want %v", got, want)
		}
	}
	// a ring longer than ctx is cut to ctx: positions past ctx-1 never occur
	for i, c := range inkling2BatchCaps(hp, local, 100) {
		if c != 100 {
			t.Fatalf("slot %d cap %d at ctx 100, want 100", i, c)
		}
	}
}
