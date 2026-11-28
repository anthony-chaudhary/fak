package radixkv

import (
	"math/rand"
	"testing"
)

func fpTokens(seed int64, n int) []int32 {
	r := rand.New(rand.NewSource(seed))
	t := make([]int32, n)
	for i := range t {
		t[i] = int32(r.Intn(1 << 20))
	}
	return t
}

// fpPermutedDocs builds a system prompt followed by a deterministic permutation of docs.
func fpPermutedDocs(sys []int32, docs [][]int32, perm []int) []int32 {
	out := append([]int32(nil), sys...)
	for _, p := range perm {
		out = append(out, docs[p]...)
	}
	return out
}

func TestFingerprintMatchAnyOffset(t *testing.T) {
	idx := NewFingerprintIndex(256)
	chunk := fpTokens(1, 256)
	if n, err := idx.Register(chunk, 1000); err != nil || n != 1 {
		t.Fatalf("register n=%d err=%v", n, err)
	}
	prompt := append(append(fpTokens(2, 37), chunk...), fpTokens(3, 100)...)
	got := idx.Match(prompt)
	if len(got) != 1 {
		t.Fatalf("want 1 candidate, got %+v", got)
	}
	c := got[0]
	if c.DstStart != 37 || c.DstEnd != 293 || c.SrcStart != 1000 || c.SrcEnd != 1256 || !c.Compatible {
		t.Fatalf("bad candidate %+v", c)
	}
}

func TestFingerprintCollisionRejected(t *testing.T) {
	idx := NewFingerprintIndex(4)
	a := []int32{1, 2, 3, 4}
	if _, err := idx.Register(a, 0); err != nil {
		t.Fatal(err)
	}
	// Find a different chunk landing in the same bucket but with a different full hash.
	ha := idx.hashAt(a)
	var b []int32
	for v := int32(5); ; v++ {
		c := []int32{v, 2, 3, 4}
		if h := idx.hashAt(c); h != ha && h&idx.mask == ha&idx.mask {
			b = c
			break
		}
	}
	if got := idx.Match(b); len(got) != 0 {
		t.Fatalf("bucket collision must be rejected, got %+v", got)
	}
	if got := idx.Match(a); len(got) != 1 {
		t.Fatalf("true match lost: %+v", got)
	}
}

func TestFingerprintEvict(t *testing.T) {
	idx := NewFingerprintIndex(8)
	toks := fpTokens(4, 24)
	idx.Register(toks, 0)
	if got := idx.Match(toks); len(got) != 1 || got[0].DstEnd != 24 {
		t.Fatalf("contiguous chunks should merge: %+v", got)
	}
	if n := idx.Evict(0); n != 3 || idx.Len() != 0 {
		t.Fatalf("evict n=%d len=%d", n, idx.Len())
	}
	if got := idx.Match(toks); len(got) != 0 {
		t.Fatalf("matched after evict: %+v", got)
	}
}

func TestFingerprintFeedsMaterializePlan(t *testing.T) {
	idx := NewFingerprintIndex(256)
	chunk := fpTokens(5, 256)
	idx.Register(chunk, 1000)
	prompt := append(append(fpTokens(6, 37), chunk...), fpTokens(7, 63)...)
	cands := idx.Match(prompt)
	plan, err := ComposeReusePlan(len(prompt), 0, cands, DefaultCostModel{TokenCost: 1})
	if err != nil {
		t.Fatal(err)
	}
	_ = plan
	for i := range cands {
		cands[i].Cost = 10
		cands[i].Action = ActionSelectiveRepair
	}
	plan, err = ComposeReusePlan(len(prompt), 0, cands, DefaultCostModel{TokenCost: 1})
	if err != nil {
		t.Fatal(err)
	}
	fence := sampleTargetFence()
	res, err := MaterializePlan(plan, fence, map[string]RACCompatibilityFence{cands[0].ID: fence})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, a := range res.Actions {
		if a.Mode == MatSelectiveRepair {
			found = true
			if a.ShiftTokens != -963 || !a.RequiresReRoPE {
				t.Fatalf("shift=%d rerope=%v", a.ShiftTokens, a.RequiresReRoPE)
			}
		}
	}
	if !found {
		t.Fatalf("no selective_repair span: %+v", res.Actions)
	}
}

func TestFingerprintPermutedDocsReusableFraction(t *testing.T) {
	const chunk, nDocs, docLen = 64, 8, 64*5 + 17
	sys := fpTokens(100, 128)
	docs := make([][]int32, nDocs)
	idx := NewFingerprintIndex(chunk)
	idx.Register(sys, 0)
	for d := range docs {
		docs[d] = fpTokens(int64(200+d), docLen)
		idx.Register(docs[d], 10000*(d+1))
	}
	perm := rand.New(rand.NewSource(9)).Perm(nDocs)
	prompt := fpPermutedDocs(sys, docs, perm)
	reused := 0
	for _, c := range idx.Match(prompt) {
		reused += c.DstEnd - c.DstStart
	}
	maxLoss := nDocs * (chunk - 1)
	if lost := len(prompt) - reused; lost < 0 || lost > maxLoss {
		t.Fatalf("lost %d tokens of %d, bound %d", lost, len(prompt), maxLoss)
	}
}

func BenchmarkFingerprintMatch(b *testing.B) {
	const n = 32 * 1024
	idx := NewFingerprintIndex(256)
	src := fpTokens(300, n)
	idx.Register(src, 0)
	prompt := append(fpTokens(301, 37), src[:n-37]...)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Match(prompt)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(len(prompt)), "ns/token")
}
