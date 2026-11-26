package ggufload

import (
	"reflect"
	"testing"
)

type prefixRun struct {
	tokens   [][]int
	cached   []int
	embedded int
}

// prefixCohort runs one cohort on bt and counts the prompt and step tokens the
// cohort embedded, which is what it computed.
func prefixCohort(t *testing.T, bt *Inkling2LoopCohorts, d *fakeInk2Cohort, prompts [][]int, n int) prefixRun {
	t.Helper()
	got := make([][]int, len(prompts))
	var reqs []Inkling2LoopBatchRequest
	for i, p := range prompts {
		i := i
		reqs = append(reqs, Inkling2LoopBatchRequest{Prompt: p, MaxTokens: n, Emit: func(tok int) bool { got[i] = append(got[i], tok); return true }})
	}
	from := len(d.writeLog)
	res, err := bt.Generate(reqs)
	if err != nil {
		t.Fatal(err)
	}
	r := prefixRun{tokens: got}
	for _, x := range res {
		r.cached = append(r.cached, x.Restored)
	}
	h := bt.m.HP.Hidden
	for _, w := range d.writeLog[from:] {
		if bt.b != nil && w.buf == bt.b.in.T.Buf() {
			r.embedded += w.n / h
		}
	}
	return r
}

func prefixPrompt(n, base int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = base + i%7
	}
	return p
}

// The store serves a next turn its conversation's whole history, checkpoints a
// second conversation on a shared system prompt at its branch point so a third one
// restores it, shares one prefill between cohort peers on the same prefix, and runs
// a lone request as a one-row cohort; every row's tokens are those of the store
// off, and only the uncached tail is computed.
func TestInkling2CohortPrefixStoreReusesAcrossCohorts(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "0")
	t.Setenv("FAK_INKLING2_COHORT_SHARED_PREFIX", "0")
	sys := prefixPrompt(80, 10)
	a := append(append([]int(nil), sys...), 1, 2, 3)
	b := append(append([]int(nil), sys...), 4, 5)
	c := append(append([]int(nil), sys...), 6)
	other := prefixPrompt(90, 500)
	const n = 3
	plan := func(bt *Inkling2LoopCohorts, d *fakeInk2Cohort) []prefixRun {
		r1 := prefixCohort(t, bt, d, [][]int{a, other}, n)
		turn2 := append(append(append([]int(nil), a...), r1.tokens[0]...), 9)
		r2 := prefixCohort(t, bt, d, [][]int{turn2, b, other}, n)
		r3 := prefixCohort(t, bt, d, [][]int{c}, n)
		return []prefixRun{r1, r2, r3}
	}
	dOff := newFakeInk2Cohort()
	off := NewInkling2LoopCohorts(fakeInk2CohortModel(), dOff)
	t.Setenv("FAK_INKLING2_BATCH_PREFIX_CACHE_MIB", "0")
	// the store-off arm cannot run a lone request as a cohort, so its third cohort
	// pairs it with a throwaway row
	r1 := prefixCohort(t, off, dOff, [][]int{a, other}, n)
	turn2 := append(append(append([]int(nil), a...), r1.tokens[0]...), 9)
	r2 := prefixCohort(t, off, dOff, [][]int{turn2, b, other}, n)
	r3 := prefixCohort(t, off, dOff, [][]int{c, other}, n)
	ref := []prefixRun{r1, r2, {tokens: r3.tokens[:1]}}
	off.Close()

	dOn := newFakeInk2Cohort()
	on := NewInkling2LoopCohorts(fakeInk2CohortModel(), dOn)
	if ok, err := on.UsePrefixStore(func(k string) (string, bool) {
		if k == "FAK_INKLING2_BATCH_PREFIX_CACHE_MIB" {
			return "64", true
		}
		return "", false
	}); !ok || err != nil {
		t.Fatalf("store on: %v %v", ok, err)
	}
	defer on.Close()
	got := plan(on, dOn)
	for ci := range ref {
		if !reflect.DeepEqual(got[ci].tokens, ref[ci].tokens) {
			t.Fatalf("cohort %d: store on %v, off %v", ci, got[ci].tokens, ref[ci].tokens)
		}
	}
	// the next turn restores a's history (prompt and all but the last reply token);
	// b and a repeat of other are checkpointed at their branch points, and c restores
	// b's, the shared system prompt
	want := [][]int{{0, 0}, {len(a) + n - 1, 0, 0}, {len(sys)}}
	for ci := range want {
		if !reflect.DeepEqual(got[ci].cached, want[ci]) {
			t.Errorf("cohort %d cached %v, want %v", ci, got[ci].cached, want[ci])
		}
	}
	if s := on.PrefixStats(); s.Hits != 2 || s.Saves < 4 {
		t.Errorf("store stats %+v, want 2 hits and the finished rows plus b's branch point saved", s)
	}
	// cohort 2 embeds turn2's last token, b's whole prompt (its branch point is
	// computed once, then its tail) and other's last token, plus its steps
	if got[1].embedded >= r2.embedded {
		t.Errorf("cohort 2 embedded %d tokens with the store, %d without", got[1].embedded, r2.embedded)
	}
}

// Two cohort peers whose prompts share a long prefix with no entry yet compute it
// once: the first row prefills to the branch point and checkpoints it, the second
// restores that checkpoint, and both continue from there.
func TestInkling2CohortPrefixStoreSharesPeersBranch(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "0")
	t.Setenv("FAK_INKLING2_COHORT_SHARED_PREFIX", "0")
	sys := prefixPrompt(100, 10)
	p1 := append(append([]int(nil), sys...), 1)
	p2 := append(append([]int(nil), sys...), 2, 3)
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	bt.store = &inkling2PrefixStore{capBytes: 1 << 26, minTokens: inkling2PrefixMinDefault}
	defer bt.Close()
	r := prefixCohort(t, bt, d, [][]int{p1, p2}, 2)
	if !reflect.DeepEqual(r.cached, []int{0, len(sys)}) {
		t.Fatalf("cached %v, want the second peer restored at the %d-token branch point", r.cached, len(sys))
	}
	if want := [][]int{{2, 3}, {4, 5}}; !reflect.DeepEqual(r.tokens, want) {
		t.Fatalf("tokens %v, want %v", r.tokens, want)
	}
	if r.embedded != len(p1)+(len(p2)-len(sys))+2 {
		t.Fatalf("embedded %d tokens, want p1 whole, p2's tail and one step each", r.embedded)
	}
}
