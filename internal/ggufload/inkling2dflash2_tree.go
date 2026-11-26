package ggufload

import (
	"fmt"
	"math"
)

// Token trees for DFlash2 tree verify. Node 0 is the root, the token the
// verify feeds at row 0 (the chain loop's fed[0]); every other node is a
// draft candidate one position after its parent. Nodes are numbered in the
// order the builder takes them, so a parent always precedes its children and
// any prefix of the nodes is itself a tree.
const (
	ink2TreeMaxNodes = 32 // a node's ancestors fit one uint32 mask
	ink2TreeMaxWidth = 4
)

// The draft-step acceptance estimate sigma(slope*logit(q) + intercept[p]) of
// a candidate of draft probability q at proposal position p, with the prior
// fit's coefficients (incoai/splash@2afc7337, Apache-2.0: VerifyBudget kPrior,
// tree_draft_estimate). Positions past the table take its last intercept.
var (
	ink2TreeSlope      = 0.7
	ink2TreeIntercepts = []float64{0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45}
	// Stand-in draft probabilities by child rank when the selector reports no
	// candidate scores: rank 0 is the greedy chain's token.
	ink2TreeRankQ = [ink2TreeMaxWidth]float64{0.8, 0.1, 0.05, 0.03}
)

type ink2Tree struct {
	tokens []int32
	parent []int32 // -1 for the root
	depth  []int32 // position offset from the root
	anc    []uint32
	// est is the estimate the builder took node i at; 1 for the root.
	est []float64
}

func (t *ink2Tree) size() int { return len(t.tokens) }

// ancestorAt returns the node at depth d on node i's root path.
func (t *ink2Tree) ancestorAt(i, d int) int {
	for int(t.depth[i]) > d {
		i = int(t.parent[i])
	}
	return i
}

// expected is the estimated tokens a verify of the tree retains, its bonus
// token included.
func (t *ink2Tree) expected() float64 {
	sum := 1.0
	for _, e := range t.est[1:] {
		sum += e
	}
	return sum
}

// prefix returns the tree of its first n nodes.
func (t *ink2Tree) prefix(n int) *ink2Tree {
	n = min(max(n, 1), t.size())
	return &ink2Tree{tokens: t.tokens[:n], parent: t.parent[:n], depth: t.depth[:n], anc: t.anc[:n], est: t.est[:n]}
}

func (t *ink2Tree) add(token, parent int, est float64) int {
	i := len(t.tokens)
	t.tokens = append(t.tokens, int32(token))
	t.parent = append(t.parent, int32(parent))
	d, a := int32(0), uint32(1)<<i
	if parent >= 0 {
		d, a = t.depth[parent]+1, a|t.anc[parent]
	}
	t.depth = append(t.depth, d)
	t.anc = append(t.anc, a)
	t.est = append(t.est, est)
	return i
}

func ink2TreeEstimate(q float64, position int) float64 {
	q = min(max(q, 1e-6), 1-1e-6)
	b := ink2TreeIntercepts[min(position, len(ink2TreeIntercepts)-1)]
	return 1 / (1 + math.Exp(-(ink2TreeSlope*math.Log(q/(1-q)) + b)))
}

// buildInk2Tree grows a tree of at most budget nodes, root included, whose
// nodes have at most width children, best estimate first. path is the draft's
// greedy chain ([rows], row r the token at depth r+1) and cand its [rows][k]
// candidate ids (Ink2DFSelect's cand); -1 entries end a row or the chain.
// scores, when set, are [rows][k] candidate scores whose softmax gives each
// candidate's draft probability; otherwise a child's rank picks it. A node at
// depth d draws its children from row d: the chain's token first, then the
// other candidates by probability (the earlier candidate on a tie), so width
// 1 is exactly the chain root, path[0], path[1], ... The selector conditions
// a row on the chain's predecessor only, so a branch's deeper rows reuse the
// chain's candidates. Equal estimates go to the earlier push.
func buildInk2Tree(root int, path, cand []int, k int, scores []float32, width, budget int) (*ink2Tree, error) {
	rows := len(path)
	if k < 1 || len(cand) != rows*k || (scores != nil && len(scores) != rows*k) || width < 1 || width > ink2TreeMaxWidth ||
		budget < 1 || budget > ink2TreeMaxNodes {
		return nil, fmt.Errorf("inkling2 tree: %d rows of %d candidates (%d ids, %d scores), width %d, budget %d",
			rows, k, len(cand), len(scores), width, budget)
	}
	t := &ink2Tree{}
	t.add(root, -1, 1)
	type entry struct {
		bound float64
		token int
		node  int
	}
	var heap []entry
	// expand pushes node i's children from row d = depth(i), capping each
	// child's estimate by what its earlier siblings left of 1: the target
	// picks one token after a node.
	expand := func(i int) {
		d := int(t.depth[i])
		if d >= rows || path[d] < 0 {
			return
		}
		ids, q := ink2TreeChildren(path[d], cand[d*k:(d+1)*k], scoresRow(scores, d, k), width)
		left, bound := 1.0, t.est[i]
		for j, id := range ids {
			own := min(ink2TreeEstimate(q[j], d), left)
			left -= own
			bound = min(bound, t.est[i]*own)
			heap = append(heap, entry{bound, id, i})
		}
	}
	expand(0)
	for t.size() < budget && len(heap) > 0 {
		best := 0
		for j := 1; j < len(heap); j++ {
			if heap[j].bound > heap[best].bound {
				best = j
			}
		}
		e := heap[best]
		heap = append(heap[:best], heap[best+1:]...)
		expand(t.add(e.token, e.node, e.bound))
	}
	return t, nil
}

func scoresRow(scores []float32, d, k int) []float32 {
	if scores == nil {
		return nil
	}
	return scores[d*k : (d+1)*k]
}

// ink2TreeChildren orders a node's children: the chain token first, then the
// row's other distinct valid candidates, up to width, with their draft
// probabilities.
func ink2TreeChildren(chain int, cand []int, scores []float32, width int) ([]int, []float64) {
	var p []float64
	if scores != nil {
		p = make([]float64, len(cand))
		top := math.Inf(-1)
		for j, id := range cand {
			if id >= 0 && !math.IsNaN(float64(scores[j])) {
				top = max(top, float64(scores[j]))
			}
		}
		mass := 0.0
		for j, id := range cand {
			if id >= 0 && !math.IsNaN(float64(scores[j])) && !math.IsInf(top, -1) {
				p[j] = math.Exp(float64(scores[j]) - top)
				mass += p[j]
			}
		}
		for j := range p {
			if mass > 0 {
				p[j] /= mass
			}
		}
	}
	prob := func(id int) float64 {
		for j, c := range cand {
			if c == id {
				return p[j]
			}
		}
		return 0
	}
	ids := []int{chain}
	var q []float64
	if p != nil {
		q = append(q, prob(chain))
	}
	taken := func(id int) bool {
		for _, x := range ids {
			if x == id {
				return true
			}
		}
		return false
	}
	for len(ids) < width {
		pick := -1
		for j, id := range cand {
			if id < 0 || taken(id) {
				continue
			}
			if pick < 0 || (p != nil && p[j] > p[pick]) {
				pick = j
			}
		}
		if pick < 0 {
			break
		}
		ids = append(ids, cand[pick])
		if p != nil {
			q = append(q, p[pick])
		}
	}
	if p == nil {
		q = ink2TreeRankQ[:len(ids)]
	}
	return ids, q
}

// accept walks the tree from the root along the target's greedy argmax:
// argmax[i] is the target's next token after node i's path. It returns the
// retained nodes (root first) and the bonus token, the argmax after the last
// of them. A node's children carry distinct tokens, so the walk is unique.
func (t *ink2Tree) accept(argmax []int) (nodes []int, bonus int) {
	cur := 0
	nodes = append(nodes, 0)
	for {
		next := -1
		for c := cur + 1; c < t.size(); c++ {
			if int(t.parent[c]) == cur && int(t.tokens[c]) == argmax[cur] {
				next = c
				break
			}
		}
		if next < 0 {
			return nodes, argmax[cur]
		}
		nodes = append(nodes, next)
		cur = next
	}
}

// ink2TreeBestRows picks the node count r maximising retained tokens per
// millisecond, (expected tokens of the first r nodes) / costMs[r], from a fixed
// cost table (costMs[r] the verify step of r rows; index 0 unused). The tree's
// nodes are best-first, so its prefixes are the best trees of each size.
func ink2TreeBestRows(t *ink2Tree, costMs []float64) int {
	best, rows, expect := 0.0, 1, 1.0
	for r := 1; r <= t.size() && r < len(costMs); r++ {
		if r > 1 {
			expect += t.est[r-1]
		}
		if costMs[r] > 0 {
			if rate := expect / costMs[r]; rate > best {
				best, rows = rate, r
			}
		}
	}
	return rows
}
