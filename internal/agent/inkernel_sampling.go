package agent

import (
	"math"
	"math/rand"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// SampleLogits returns the next token id: argmax when temp<=0, else a
// temperature-scaled softmax draw.
func SampleLogits(logits []float32, temp float64, rng *rand.Rand) int {
	return sampleLogits(logits, temp, 0, 0, rng)
}

const clusterSamplingMinVocabulary = 1 << 16

type samplingTruncationPath uint8

const (
	samplingTruncationCPU samplingTruncationPath = iota
	samplingTruncationCluster
)

// samplingTruncationEnvelope records the capabilities that a future device-side
// sampler must prove before it may keep a very wide vocabulary resident and launch
// truncation across a thread-block cluster. Zero values deliberately select the CPU
// oracle/fallback used by the current host sampler.
type samplingTruncationEnvelope struct {
	clusterCapable bool
	deviceResident bool
}

// selectSamplingTruncationPath is the fail-closed dispatch seam for the
// cluster-launched top-k/top-p mechanism. It selects the cluster path only when a
// truncation is requested, the vocabulary is wide enough to justify partitioning,
// and the backend explicitly proves both cluster launch support and device
// residency. Selection is not execution: callers without a device implementation
// continue through applyCPUSamplingTruncation as the correctness oracle/fallback.
func selectSamplingTruncationPath(vocabSize int, topP float64, topK int, env samplingTruncationEnvelope) samplingTruncationPath {
	truncatesTopK := topK > 0 && topK < vocabSize
	truncatesTopP := topP > 0 && topP < 1
	if vocabSize >= clusterSamplingMinVocabulary && (truncatesTopK || truncatesTopP) && env.clusterCapable && env.deviceResident {
		return samplingTruncationCluster
	}
	return samplingTruncationCPU
}

// samplerWorkspace is the per-session scratch the in-kernel sampler reuses so a
// served token never allocates: the size-V probability row, the index permutation the
// truncation sorts consume, the concrete sort.Interface that sorts that permutation
// without the closure allocation sort.Slice pays, the top-k heap scratch and its key
// backup, and the float32 effective-logits row the bias/penalty paths build on. One
// workspace is owned per decode lane; it grows monotonically to the widest vocabulary
// it has seen and is then reused allocation-free for every subsequent token.
type samplerWorkspace struct {
	probs  []float64
	order  []int
	heap   []int
	vals   []float64
	eff    []float32
	sorter probSorter
}

func (w *samplerWorkspace) resize(vocab int) {
	if cap(w.probs) < vocab {
		w.probs = make([]float64, vocab)
	}
	w.probs = w.probs[:vocab]
}

func (w *samplerWorkspace) orderBuf(n int) []int {
	if cap(w.order) < n {
		w.order = make([]int, n)
	}
	return w.order[:n]
}

func (w *samplerWorkspace) heapBuf(k int) []int {
	if cap(w.heap) < k {
		w.heap = make([]int, k)
	}
	return w.heap[:k]
}

func (w *samplerWorkspace) valBuf(k int) []float64 {
	if cap(w.vals) < k {
		w.vals = make([]float64, k)
	}
	return w.vals[:k]
}

func (w *samplerWorkspace) effBuf(n int) []float32 {
	if cap(w.eff) < n {
		w.eff = make([]float32, n)
	}
	return w.eff[:n]
}

// probSorter is the concrete sort.Interface over an index permutation. Unlike the
// sort.Slice closure it replaces, constructing and sorting it allocates nothing once
// its probs/idx slices are wired to a workspace buffer. tieByIndex makes equal
// probability deterministic by keeping the lower token index. The nucleus path
// enables it because sort.Sort is not stable; relying on input order would leave the
// exact kept set unspecified when top-p cuts through a tie.
type probSorter struct {
	probs      []float64
	idx        []int
	tieByIndex bool
}

func (s *probSorter) Len() int { return len(s.idx) }
func (s *probSorter) Less(a, b int) bool {
	ia, ib := s.idx[a], s.idx[b]
	pa, pb := s.probs[ia], s.probs[ib]
	if pa != pb {
		return pa > pb
	}
	if s.tieByIndex {
		return ia < ib
	}
	return false
}
func (s *probSorter) Swap(a, b int) { s.idx[a], s.idx[b] = s.idx[b], s.idx[a] }

// applyCPUSamplingTruncation is the explicit host oracle/fallback. The current
// sampler always uses it because its probability row is host-resident; a future
// device caller may use selectSamplingTruncationPath before choosing a device
// implementation without weakening this fallback.
func applyCPUSamplingTruncation(probs []float64, sum, topP float64, topK int) float64 {
	return (&samplerWorkspace{}).applyCPUSamplingTruncation(probs, sum, topP, topK)
}

func (w *samplerWorkspace) applyCPUSamplingTruncation(probs []float64, sum, topP float64, topK int) float64 {
	// Host softmax materializes probs here, so the zero capability envelope must
	// fail closed to the existing CPU oracle/fallback.
	_ = selectSamplingTruncationPath(len(probs), topP, topK, samplingTruncationEnvelope{})
	if topK > 0 && topK < len(probs) {
		sum = w.topKTruncate(probs, sum, topK)
	}
	if topP > 0 && topP < 1 {
		sum = w.nucleusTruncate(probs, sum, topP)
	}
	return sum
}

// sampleLogits is the in-kernel sampler. topK then topP truncate the stochastic
// path, in that order (the standard top-k → top-p pipeline): top-k keeps only the k
// highest-probability tokens, then nucleus (top-p) keeps the smallest set whose
// cumulative mass reaches topP. The tail each step excludes is zeroed before the
// draw. A topK<=0 or topK>=len(logits) disables top-k; a topP<=0 or topP>=1 disables
// nucleus — with both off the draw is the full softmax, byte-for-byte the pre-seam
// behavior. The single most-probable token is always kept so neither cutoff can
// empty the candidate set. Both shape only the stochastic path: temp<=0 stays pure
// argmax (top-k/top-p never change the argmax winner).
//
// The package-level entry point allocates a one-shot workspace; a caller that samples
// every token (the served decode lane) holds a samplerWorkspace and calls
// sampleLogitsWS instead, reusing its buffers allocation-free.
func sampleLogits(logits []float32, temp, topP float64, topK int, rng *rand.Rand) int {
	return sampleLogitsWS(logits, temp, topP, topK, rng, &samplerWorkspace{})
}

// sampleLogitsWS is sampleLogits over a caller-owned reusable workspace. After the
// first call sizes the workspace to the vocabulary, repeated calls allocate nothing.
func sampleLogitsWS(logits []float32, temp, topP float64, topK int, rng *rand.Rand, w *samplerWorkspace) int {
	if temp <= 0 {
		best, bi := float32(-math.MaxFloat32), 0
		for i, x := range logits {
			if x > best {
				best, bi = x, i
			}
		}
		return bi
	}
	maxL := float32(-math.MaxFloat32)
	for _, x := range logits {
		if x > maxL {
			maxL = x
		}
	}
	w.resize(len(logits))
	probs := w.probs
	var sum float64
	for i, x := range logits {
		p := math.Exp(float64(x-maxL) / temp)
		probs[i] = p
		sum += p
	}
	sum = w.applyCPUSamplingTruncation(probs, sum, topP, topK)
	return sampleProbabilityRow(probs, sum, rng.Float64())
}

// sampleProbabilityRow inverts one probability row with a coin in [0,1). It uses
// the strict CDF boundary from DeepInfer's reference sampler: a zero-mass token is
// never selected, including when coin is exactly zero, and a coin exactly on a CDF
// boundary belongs to the next positive interval. The input row remains unchanged.
// Adapted from DeepInfer at 774ce244; see third_party/deepinfer-sampling/NOTICE.
func sampleProbabilityRow(probs []float64, sum, coin float64) int {
	target := coin * sum
	var cumulative float64
	for i, p := range probs {
		if p <= 0 {
			continue
		}
		cumulative += p
		if cumulative > target {
			return i
		}
	}
	// Fall back to the last token with nonzero mass (nucleus zeroed the tail).
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	// Preserve the historical last-index sentinel for an invalid row with no
	// positive mass. Sampling callers are responsible for supplying sum > 0.
	return len(probs) - 1
}

// sampleLogitsWithBias applies the OpenAI logit_bias map before the existing in-kernel
// sampler. The nil/empty map is a strict no-op, preserving the historical argmax /
// stochastic path byte-for-byte. Biases are clamped to the same [-100, 100] bound the
// native model constraint sink uses.
func sampleLogitsWithBias(logits []float32, temp, topP float64, topK int, bias model.LogitBias, rng *rand.Rand) int {
	return sampleLogitsWithBiasWS(logits, temp, topP, topK, bias, rng, &samplerWorkspace{})
}

func sampleLogitsWithBiasWS(logits []float32, temp, topP float64, topK int, bias model.LogitBias, rng *rand.Rand, w *samplerWorkspace) int {
	if len(bias) == 0 {
		return sampleLogitsWS(logits, temp, topP, topK, rng, w)
	}
	eff := w.effBuf(len(logits))
	copy(eff, logits)
	applyLogitBias(eff, bias)
	return sampleLogitsWS(eff, temp, topP, topK, rng, w)
}

// applyLogitBias adds the OpenAI logit_bias map into logits IN PLACE. Out-of-range
// token ids are skipped; biases are clamped to the same [-100, 100] bound the native
// model constraint sink uses. Factored out of sampleLogitsWithBias so
// sampleLogitsWithPenalty can share the exact same clamp/add behavior.
func applyLogitBias(logits []float32, bias model.LogitBias) {
	for tok, b := range bias {
		if tok < 0 || tok >= len(logits) {
			continue
		}
		if b > model.LogitBiasClamp {
			b = model.LogitBiasClamp
		} else if b < -model.LogitBiasClamp {
			b = -model.LogitBiasClamp
		}
		logits[tok] += float32(b)
	}
}

// sampleLogitsWithPenalty applies the OpenAI logit_bias map AND the OpenAI
// frequency/presence repetition penalties before the existing in-kernel sampler:
//
//	logit[t] -= frequency_penalty*count[t] + presence_penalty*(1 if count[t]>0 else 0)
//
// counts is indexed by token id and holds how many times each token has already been
// generated in this response (the caller's running per-token histogram); a nil/short
// counts slice treats every token as count 0. Both a zero frequencyPenalty AND a zero
// presencePenalty make this a byte-for-byte no-op versus sampleLogitsWithBias (the
// penalty term is skipped entirely, not just multiplied by zero), so an
// unset/zero-valued request reproduces the exact pre-penalty output.
func sampleLogitsWithPenalty(logits []float32, temp, topP float64, topK int, bias model.LogitBias, frequencyPenalty, presencePenalty float64, counts []int32, rng *rand.Rand) int {
	return sampleLogitsWithPenaltyWS(logits, temp, topP, topK, bias, frequencyPenalty, presencePenalty, counts, rng, &samplerWorkspace{})
}

func sampleLogitsWithPenaltyWS(logits []float32, temp, topP float64, topK int, bias model.LogitBias, frequencyPenalty, presencePenalty float64, counts []int32, rng *rand.Rand, w *samplerWorkspace) int {
	if frequencyPenalty == 0 && presencePenalty == 0 {
		return sampleLogitsWithBiasWS(logits, temp, topP, topK, bias, rng, w)
	}
	eff := w.effBuf(len(logits))
	copy(eff, logits)
	if len(bias) > 0 {
		applyLogitBias(eff, bias)
	}
	for tok, c := range counts {
		if tok >= len(eff) || c <= 0 {
			continue
		}
		penalty := frequencyPenalty*float64(c) + presencePenalty
		eff[tok] -= float32(penalty)
	}
	return sampleLogitsWS(eff, temp, topP, topK, rng, w)
}

// nucleusTruncate zeroes every probability outside the top-p nucleus in place and
// returns the surviving mass (the new normalization sum). The nucleus is the
// smallest set of highest-probability tokens whose cumulative mass reaches topP;
// the single most-probable token is always kept so the nucleus is never empty.
// probs is unsorted on entry and stays index-aligned to the caller's logits.
func nucleusTruncate(probs []float64, sum, topP float64) float64 {
	return (&samplerWorkspace{}).nucleusTruncate(probs, sum, topP)
}

func (w *samplerWorkspace) nucleusTruncate(probs []float64, sum, topP float64) float64 {
	if len(probs) == 0 {
		return 0
	}
	order := w.orderBuf(len(probs))
	for i := range order {
		order[i] = i
	}
	w.sorter.probs = probs
	w.sorter.idx = order
	// DeepInfer's reference uses a stable descending sort. Since order starts as
	// token-id order, lower token id wins equal-probability ties.
	w.sorter.tieByIndex = true
	sort.Sort(&w.sorter)
	target := topP * sum
	var cum float64
	cutoff := len(order)
	for rank, idx := range order {
		// Stop BEFORE adding this token once the nucleus already reached the target —
		// the kept set is the minimal prefix whose mass >= target. Rank 0 is always
		// kept (the head token) so the nucleus is never empty.
		if rank > 0 && cum >= target {
			cutoff = rank
			break
		}
		cum += probs[idx]
	}
	for _, idx := range order[cutoff:] {
		probs[idx] = 0
	}
	var newSum float64
	for _, p := range probs {
		newSum += p
	}
	return newSum
}

// descProbOrder returns the indices of probs ordered by the caller's less comparator, which
// ranks two ELEMENT indices (not positions in the returned slice). It is the shared
// highest-probability-first index permutation the correctness references sort on. The
// sampler hot path no longer calls it (it drives probSorter directly to stay
// allocation-free); it remains the golden oracle in the parity tests.
func descProbOrder(probs []float64, less func(i, j int) bool) []int {
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return less(order[a], order[b]) })
	return order
}

// maskKept zeroes every probability whose index is not in kept (in place) and returns the
// surviving mass (the new normalization sum) — the renormalization oracle for topKTruncate.
func maskKept(probs []float64, kept map[int]bool) float64 {
	var newSum float64
	for i := range probs {
		if kept[i] {
			newSum += probs[i]
		} else {
			probs[i] = 0
		}
	}
	return newSum
}

const topKSmallKThreshold = 64

// topKHeapLess returns true if heap index a is worse than heap index b.
// An element is worse if it has lower probability, or equal probability and higher index.
func topKHeapLess(heap []int, probs []float64, a, b int) bool {
	ia, ib := heap[a], heap[b]
	pa, pb := probs[ia], probs[ib]
	if pa != pb {
		return pa < pb
	}
	return ia > ib
}

func topKSiftDown(heap []int, probs []float64, root, n int) {
	for {
		child := 2*root + 1
		if child >= n {
			break
		}
		if right := child + 1; right < n && topKHeapLess(heap, probs, right, child) {
			child = right
		}
		if !topKHeapLess(heap, probs, child, root) {
			break
		}
		heap[root], heap[child] = heap[child], heap[root]
		root = child
	}
}

// topKTruncateSmallK implements sort-free bounded selection for k <= 64 via a
// bounded min-heap in O(V log k) time with zero heap allocations on the hot path.
func topKTruncateSmallK(probs []float64, k int) float64 {
	var heapBuf [topKSmallKThreshold]int
	heap := heapBuf[:k]
	for i := 0; i < k; i++ {
		heap[i] = i
	}
	for i := k/2 - 1; i >= 0; i-- {
		topKSiftDown(heap, probs, i, k)
	}

	rootProb := probs[heap[0]]
	for i := k; i < len(probs); i++ {
		p := probs[i]
		if p > rootProb {
			heap[0] = i
			topKSiftDown(heap, probs, 0, k)
			rootProb = probs[heap[0]]
		}
	}

	// Sort kept indices in ascending order (insertion sort over <= 64 elements,
	// zero allocations, ensures deterministic summation order matching maskKept).
	for i := 1; i < k; i++ {
		key := heap[i]
		j := i - 1
		for j >= 0 && heap[j] > key {
			heap[j+1] = heap[j]
			j--
		}
		heap[j+1] = key
	}

	var topVals [topKSmallKThreshold]float64
	var newSum float64
	for j := 0; j < k; j++ {
		idx := heap[j]
		val := probs[idx]
		topVals[j] = val
		newSum += val
	}
	clear(probs)
	for j := 0; j < k; j++ {
		probs[heap[j]] = topVals[j]
	}
	return newSum
}

// topKTruncate zeroes every probability outside the top-k highest-probability
// tokens in place and returns the surviving mass (the new normalization sum). Ties
// at the k-th rank are broken by index order (equal probability breaks by lower index),
// so the kept set is deterministic. probs is unsorted on entry and stays index-aligned
// to the caller's logits. The caller guarantees 0 < k < len(probs); k>=len(probs) is
// a no-op handled before the call so the full distribution stays byte-for-byte the
// pre-seam draw.
func topKTruncate(probs []float64, sum float64, k int) float64 {
	return (&samplerWorkspace{}).topKTruncate(probs, sum, k)
}

func (w *samplerWorkspace) topKTruncate(probs []float64, sum float64, k int) float64 {
	if k <= 0 || len(probs) == 0 {
		clear(probs)
		return 0
	}
	if k >= len(probs) {
		return sum
	}
	if k <= topKSmallKThreshold {
		return topKTruncateSmallK(probs, k)
	}
	return w.topKHeap(probs, k)
}

// topKHeap is the >64 generalization of topKTruncateSmallK over a workspace-backed
// bounded min-heap: O(V log k) time, no full sort and no map. It preserves the exact
// kept-set tie-break (lower index wins) and the ascending-index renormalization order
// of the small-k path and of the reference full-sort oracle.
func (w *samplerWorkspace) topKHeap(probs []float64, k int) float64 {
	heap := w.heapBuf(k)
	for i := 0; i < k; i++ {
		heap[i] = i
	}
	for i := k/2 - 1; i >= 0; i-- {
		topKSiftDown(heap, probs, i, k)
	}

	rootProb := probs[heap[0]]
	for i := k; i < len(probs); i++ {
		if p := probs[i]; p > rootProb {
			heap[0] = i
			topKSiftDown(heap, probs, 0, k)
			rootProb = probs[heap[0]]
		}
	}

	for i := 1; i < k; i++ {
		key := heap[i]
		j := i - 1
		for j >= 0 && heap[j] > key {
			heap[j+1] = heap[j]
			j--
		}
		heap[j+1] = key
	}

	vals := w.valBuf(k)
	var newSum float64
	for j := 0; j < k; j++ {
		idx := heap[j]
		vals[j] = probs[idx]
		newSum += vals[j]
	}
	clear(probs)
	for j := 0; j < k; j++ {
		probs[heap[j]] = vals[j]
	}
	return newSum
}
