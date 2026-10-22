package agent

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"
)

// Invariants restated from DeepInfer at 774ce244; see third_party/deepinfer-sampling-invariants/NOTICE.

type truncInvRow struct {
	family string
	probs  []float64
	sum    float64
}

func truncInvAscSum(probs []float64) float64 {
	var s float64
	for _, p := range probs {
		s += p
	}
	return s
}

func truncInvPositive(probs []float64) int {
	n := 0
	for _, p := range probs {
		if p > 0 {
			n++
		}
	}
	return n
}

func truncInvHead(probs []float64) int {
	head := 0
	for i, p := range probs {
		if p > probs[head] {
			head = i
		}
	}
	return head
}

func truncInvRows(vocab int, seed int64) []truncInvRow {
	rng := rand.New(rand.NewSource(seed))
	families := []string{"randn3"}
	if vocab <= 32768 {
		families = append(families, "ties", "zeros")
	}
	rows := make([]truncInvRow, 0, len(families))
	logits := make([]float64, vocab)
	for _, family := range families {
		maxL := math.Inf(-1)
		for i := range logits {
			x := rng.NormFloat64() * 3
			if family == "ties" {
				x = math.Round(x)
			}
			logits[i] = x
			maxL = max(maxL, x)
		}
		probs := make([]float64, vocab)
		for i, x := range logits {
			probs[i] = math.Exp(x - maxL)
			if family == "zeros" && i%4 == 1 {
				probs[i] = 0
			}
		}
		rows = append(rows, truncInvRow{family: family, probs: probs, sum: truncInvAscSum(probs)})
	}
	return rows
}

// truncInvCheckSelection checks invariants (a) and (b) of one truncation output and
// returns the positive survivors in the stable descending order (prob desc, then
// lower index) the nucleus walk accumulates in.
func truncInvCheckSelection(t *testing.T, label string, in, out []float64, ret float64) []int {
	t.Helper()
	var survivors []int
	var survSum float64
	minSurv, maxPruned := math.Inf(1), math.Inf(-1)
	lastMinSurv, firstMaxPruned := -1, -1
	for i, p := range out {
		switch {
		case p > 0:
			if math.Float64bits(p) != math.Float64bits(in[i]) {
				t.Fatalf("%s: survivor %d = %v, input %v", label, i, p, in[i])
			}
			survivors = append(survivors, i)
			survSum += p
			if p <= minSurv {
				minSurv, lastMinSurv = p, i
			}
		case math.Float64bits(p) != 0:
			t.Fatalf("%s: pruned token %d = %v, want +0", label, i, p)
		case in[i] > maxPruned && in[i] > 0:
			maxPruned, firstMaxPruned = in[i], i
		}
	}
	if math.Float64bits(ret) != math.Float64bits(survSum) {
		t.Fatalf("%s: returned sum %v, ascending-index survivor sum %v", label, ret, survSum)
	}
	if firstMaxPruned >= 0 && lastMinSurv >= 0 {
		if maxPruned > minSurv || (maxPruned == minSurv && firstMaxPruned < lastMinSurv) {
			t.Fatalf("%s: pruned token %d (%v) outranks survivor %d (%v)", label, firstMaxPruned, maxPruned, lastMinSurv, minSurv)
		}
	}
	slices.SortFunc(survivors, func(a, b int) int {
		if c := cmp.Compare(out[b], out[a]); c != 0 {
			return c
		}
		return cmp.Compare(a, b)
	})
	return survivors
}

// fak-test:runtime fast est=400ms lane=default
func TestTruncationMinimalPrefixInvariants(t *testing.T) {
	vocabs := []int{2, 17, 1024, 32768, 248000}
	topPs := []float64{math.SmallestNonzeroFloat64, 0.25, 0.5, 0.75, 0.9, 0.95, 1}
	topKs := []int{0, 1, 20, 100}
	var w, probe samplerWorkspace
	var cells, checks, prefixHits, fullHits int
	for vi, vocab := range vocabs {
		rows := truncInvRows(vocab, int64(vi+1))
		scratch := make([]float64, vocab)
		for _, topK := range topKs {
			kEff := vocab
			if topK > 0 && topK < vocab {
				kEff = topK
			}
			s1s := make([][]float64, len(rows))
			m1s := make([]float64, len(rows))
			t1s := make([]int, len(rows))
			for ri, row := range rows {
				label := fmt.Sprintf("vocab=%d family=%s topK=%d top-k stage", vocab, row.family, topK)
				s1 := slices.Clone(row.probs)
				m1 := w.applyCPUSamplingTruncation(s1, row.sum, 1, topK)
				kept := truncInvCheckSelection(t, label, row.probs, s1, m1)
				if want := min(kEff, truncInvPositive(row.probs)); len(kept) != want {
					t.Fatalf("%s: %d positive survivors, want %d", label, len(kept), want)
				}
				s1s[ri], m1s[ri], t1s[ri] = s1, m1, len(kept)
			}
			for _, topP := range topPs {
				for ri, row := range rows {
					label := fmt.Sprintf("vocab=%d family=%s topK=%d topP=%v", vocab, row.family, topK, topP)
					s1, m1 := s1s[ri], m1s[ri]
					fin := scratch
					copy(fin, row.probs)
					mf := w.applyCPUSamplingTruncation(fin, row.sum, topP, topK)
					kept := truncInvCheckSelection(t, label, row.probs, fin, mf)
					n := len(kept)
					if n == 0 || fin[truncInvHead(row.probs)] <= 0 {
						t.Fatalf("%s: head token pruned", label)
					}
					if n > kEff {
						t.Fatalf("%s: %d survivors exceed k limit %d", label, n, kEff)
					}
					for _, i := range kept {
						if !(s1[i] > 0) {
							t.Fatalf("%s: survivor %d was pruned by the top-k stage", label, i)
						}
					}
					if !(topP > 0 && topP < 1) {
						for i := range fin {
							if math.Float64bits(fin[i]) != math.Float64bits(s1[i]) {
								t.Fatalf("%s: disabled top-p changed token %d", label, i)
							}
						}
						if math.Float64bits(mf) != math.Float64bits(m1) {
							t.Fatalf("%s: disabled top-p sum %v, top-k sum %v", label, mf, m1)
						}
					} else {
						probeRow := slices.Clone(s1)
						if _, ok := probe.nucleusTruncatePrefix(probeRow, topP*m1); ok {
							prefixHits++
						} else {
							fullHits++
						}
						// Exact float64, no eps: the boundary is >= on the target, and the
						// sums run in stable descending order because rounding depends on it.
						target := topP * m1
						var before float64
						for _, i := range kept[:n-1] {
							before += fin[i]
						}
						reached := before + fin[kept[n-1]]
						if n > 1 && !(before < target) {
							t.Fatalf("%s: kept %d tokens, but the first %d already reach %v >= target %v", label, n, n-1, before, target)
						}
						if !(reached >= target) && n != t1s[ri] {
							t.Fatalf("%s: stopped at %d tokens with mass %v < target %v before the top-k support %d", label, n, reached, target, t1s[ri])
						}
						if topP == math.SmallestNonzeroFloat64 && n != 1 {
							t.Fatalf("%s: narrowest nucleus kept %d tokens, want only the head", label, n)
						}
					}
					checks++
				}
				cells++
			}
		}
	}
	t.Logf("invariant-checked cells=%d rows=%d nucleus prefix=%d full=%d", cells, checks, prefixHits, fullHits)
	if cells < 140 {
		t.Fatalf("invariant-checked cells = %d, want >= 140", cells)
	}
	if prefixHits == 0 || fullHits == 0 {
		t.Fatalf("nucleus branches witnessed: prefix=%d full=%d, want both > 0", prefixHits, fullHits)
	}

	t.Run("disabled-bounds", func(t *testing.T) {
		for vi, vocab := range []int{2, 17, 1024} {
			for _, row := range truncInvRows(vocab, int64(100+vi)) {
				for _, topP := range []float64{0, -0.5, 1.5, math.NaN()} {
					for _, topK := range []int{0, -1, vocab, vocab + 7} {
						out := slices.Clone(row.probs)
						got := w.applyCPUSamplingTruncation(out, row.sum, topP, topK)
						if math.Float64bits(got) != math.Float64bits(row.sum) {
							t.Fatalf("vocab=%d family=%s topP=%v topK=%d: sum %v, want passed sum %v", vocab, row.family, topP, topK, got, row.sum)
						}
						for i := range out {
							if math.Float64bits(out[i]) != math.Float64bits(row.probs[i]) {
								t.Fatalf("vocab=%d family=%s topP=%v topK=%d: token %d changed", vocab, row.family, topP, topK, i)
							}
						}
					}
				}
			}
		}
	})
}

// truncInvDeepInferKept is DeepInfer's reference rule: stable top-p over the full
// normalized distribution, dropping a token once the mass strictly before it exceeds
// top_p, then stable top-k. topK<=0 is fak's disabled value, DeepInfer's -1.
func truncInvDeepInferKept(probs []float64, topP float64, topK int) ([]int, float64) {
	total := truncInvAscSum(probs)
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })
	var kept []int
	var cum float64
	for _, i := range order {
		q := probs[i] / total
		cum += q
		if !(cum-q > topP) && (topK <= 0 || len(kept) < topK) {
			kept = append(kept, i)
		}
	}
	slices.Sort(kept)
	var sum float64
	for _, i := range kept {
		sum += probs[i]
	}
	return kept, sum
}

func truncInvTotalVariation(probs []float64, a []int, aSum float64, b []int, bSum float64) float64 {
	var tv float64
	for i, p := range probs {
		var pa, pb float64
		if slices.Contains(a, i) {
			pa = p / aSum
		}
		if slices.Contains(b, i) {
			pb = p / bSum
		}
		tv += math.Abs(pa - pb)
	}
	return tv / 2
}

// fak-test:runtime fast est=5ms lane=default
func TestTruncationDeepInferDivergenceTable(t *testing.T) {
	rows := []struct {
		name     string
		probs    []float64
		topP     float64
		topK     int
		fakKept  []int
		fakSum   float64
		diKept   []int
		diSum    float64
		wantTV   float64
		diverges bool
		served   bool
	}{
		{"CD-02 top-p over post-top-k mass", []float64{0.5, 0.3, 0.2}, 0.6, 2, []int{0}, 0.5, []int{0, 1}, 0.8, 0.375, true, true},
		{"CD-03 tied boundary equality", []float64{0.5, 0.5}, 0.5, 0, []int{0}, 0.5, []int{0, 1}, 1, 0.5, true, true},
		// float32 log-probs cannot carry this exact dyadic boundary into the served path.
		{"CD-03 untied dyadic boundary", []float64{0.5, 0.375, 0.125}, 0.5, 0, []int{0}, 0.5, []int{0, 1}, 0.875, 0.4285714285714286, true, false},
		{"control", []float64{0.5, 0.3, 0.2}, 0.7, 0, []int{0, 1}, 0.8, []int{0, 1}, 0.8, 0, false, true},
	}
	divergent := 0
	for ri, row := range rows {
		sum := truncInvAscSum(row.probs)
		got := slices.Clone(row.probs)
		gotSum := applyCPUSamplingTruncation(got, sum, row.topP, row.topK)
		var fakKept []int
		for i, p := range got {
			if p > 0 {
				if math.Float64bits(p) != math.Float64bits(row.probs[i]) {
					t.Fatalf("%s: survivor %d = %v, input %v", row.name, i, p, row.probs[i])
				}
				fakKept = append(fakKept, i)
			}
		}
		if !slices.Equal(fakKept, row.fakKept) || math.Float64bits(gotSum) != math.Float64bits(row.fakSum) {
			t.Fatalf("%s: fak kept %v sum %v, want %v sum %v", row.name, fakKept, gotSum, row.fakKept, row.fakSum)
		}
		diKept, diSum := truncInvDeepInferKept(row.probs, row.topP, row.topK)
		if !slices.Equal(diKept, row.diKept) || math.Float64bits(diSum) != math.Float64bits(row.diSum) {
			t.Fatalf("%s: DeepInfer rule kept %v sum %v, want %v sum %v", row.name, diKept, diSum, row.diKept, row.diSum)
		}
		if slices.Equal(row.fakKept, row.diKept) == row.diverges {
			t.Fatalf("%s: fak %v vs DeepInfer %v, diverges=%v", row.name, row.fakKept, row.diKept, row.diverges)
		}
		if row.diverges {
			divergent++
		}
		tv := truncInvTotalVariation(row.probs, fakKept, gotSum, diKept, diSum)
		t.Logf("%s: probs=%v topP=%v topK=%d fak=%v DeepInfer=%v total-variation=%v", row.name, row.probs, row.topP, row.topK, fakKept, diKept, tv)
		if tv != row.wantTV {
			t.Fatalf("%s: total variation %v, want %v", row.name, tv, row.wantTV)
		}
		if !row.served {
			continue
		}
		logits := make([]float32, len(row.probs))
		for i, p := range row.probs {
			logits[i] = float32(math.Log(p))
		}
		rng := rand.New(rand.NewSource(int64(ri + 1)))
		for draw := 0; draw < 64; draw++ {
			if tok := sampleLogits(logits, 1, row.topP, row.topK, rng); !slices.Contains(row.fakKept, tok) {
				t.Fatalf("%s: served draw %d = token %d outside fak kept set %v", row.name, draw, tok, row.fakKept)
			}
		}
	}
	if divergent != 3 {
		t.Fatalf("divergent rows = %d, want 3", divergent)
	}
}
