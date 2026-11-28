// Adapted from sgl-project/sglang (Apache-2.0) at commit
// sgl-router/src/policies/scoring/prefix_cache.rs and
// sgl-router/src/state/kv_events/tree.rs. This is an adaptation, not a copy:
// storage tiers, data-parallel ranks, sharding, LRU node eviction, the fused
// weight and the min-share eligibility filter are left out.

package gateway

import (
	"sort"
	"strings"
	"sync"
)

// PrefixResidencyScorer is a shared block-hash trie fed by engine KV events
// (BlockStored, BlockRemoved, AllBlocksCleared). Each node records which workers
// hold the chain ending there, so one worker's eviction never hides another
// worker that still holds the same prefix. It only produces scores; it holds no
// routing authority and is not wired into admission. Safe for concurrent use.
type PrefixResidencyScorer struct {
	mu     sync.RWMutex
	root   *prsNode
	byHash map[string]map[*prsNode]struct{}
	nodes  int
	nextID uint64
}

type prsNode struct {
	id       uint64
	hash     string
	parent   *prsNode
	children map[string]*prsNode
	holders  map[string]struct{}
	pruned   bool
}

// PrefixResidencyScore is one candidate's contiguous matched depth and its share
// of the queried chain.
type PrefixResidencyScore struct {
	Worker        string
	MatchedBlocks int
	Score         float64
}

// NewPrefixResidencyScorer returns an empty scorer.
func NewPrefixResidencyScorer() *PrefixResidencyScorer {
	return &PrefixResidencyScorer{
		root:   newPRSNode(0, "", nil),
		byHash: make(map[string]map[*prsNode]struct{}),
	}
}

func newPRSNode(id uint64, hash string, parent *prsNode) *prsNode {
	return &prsNode{id: id, hash: hash, parent: parent, children: map[string]*prsNode{}, holders: map[string]struct{}{}}
}

// ApplyStored records that worker holds blocks, chained under parentHash ("" for
// a chain that starts at the root). A blank block hash ends the chain there.
func (s *PrefixResidencyScorer) ApplyStored(worker, parentHash string, blocks []string) {
	if worker == "" || len(blocks) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	cur := s.resolveParentLocked(worker, parentHash)
	for _, h := range blocks {
		if h == "" {
			return
		}
		child := cur.children[h]
		if child == nil {
			s.nextID++
			child = newPRSNode(s.nextID, h, cur)
			cur.children[h] = child
			set := s.byHash[h]
			if set == nil {
				set = map[*prsNode]struct{}{}
				s.byHash[h] = set
			}
			set[child] = struct{}{}
			s.nodes++
		}
		child.holders[worker] = struct{}{}
		cur = child
	}
}

// resolveParentLocked attaches a continuation only under a parent node the same
// worker holds. Hanging it under another worker's node would make its visibility
// depend on that other worker; a root-anchored chain is instead credited with the
// skipped prefix by the re-anchoring pass in depthsLocked.
func (s *PrefixResidencyScorer) resolveParentLocked(worker, parentHash string) *prsNode {
	if parentHash == "" {
		return s.root
	}
	var best *prsNode
	for n := range s.byHash[parentHash] {
		if _, ok := n.holders[worker]; ok && (best == nil || n.id < best.id) {
			best = n
		}
	}
	if best == nil {
		return s.root
	}
	return best
}

// ApplyRemoved drops worker from every node carrying any of blocks. Descendants
// are untouched, so a removed interior block stops that worker's contiguous
// depth at the hole while other holders keep theirs.
func (s *PrefixResidencyScorer) ApplyRemoved(worker string, blocks []string) {
	if worker == "" || len(blocks) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var targets []*prsNode
	for _, h := range blocks {
		for n := range s.byHash[h] {
			targets = append(targets, n)
		}
	}
	for _, n := range targets {
		if n.pruned {
			continue
		}
		delete(n.holders, worker)
		s.pruneLocked(n)
	}
}

// ApplyAllBlocksCleared forgets everything worker holds, leaving other workers'
// residency intact.
func (s *PrefixResidencyScorer) ApplyAllBlocksCleared(worker string) {
	if worker == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var held []*prsNode
	for _, set := range s.byHash {
		for n := range set {
			if _, ok := n.holders[worker]; ok {
				held = append(held, n)
			}
		}
	}
	for _, n := range held {
		delete(n.holders, worker)
	}
	for _, n := range held {
		if !n.pruned {
			s.pruneLocked(n)
		}
	}
}

// ApplyExternalBatch folds an ExternalKVCacheEventBatch. That shape carries no
// parent hash, so every stored run is treated as root-anchored. Returns the
// number of events applied; events with no usable block hash are skipped.
func (s *PrefixResidencyScorer) ApplyExternalBatch(batch ExternalKVCacheEventBatch) int {
	if batch.Worker == "" {
		return 0
	}
	applied := 0
	for _, ev := range batch.Events {
		switch externalKVEventKind(ev.Type) {
		case "stored":
			hs := trimBlockHashes(ev.BlockHashes)
			if len(hs) == 0 || hs[0] == "" {
				continue
			}
			s.ApplyStored(batch.Worker, "", hs)
		case "removed":
			hs := trimBlockHashes(ev.BlockHashes)
			if !anyBlockHash(hs) {
				continue
			}
			s.ApplyRemoved(batch.Worker, hs)
		case "cleared":
			s.ApplyAllBlocksCleared(batch.Worker)
		default:
			continue
		}
		applied++
	}
	return applied
}

func trimBlockHashes(in []string) []string {
	out := make([]string, len(in))
	for i, h := range in {
		out[i] = strings.TrimSpace(h)
	}
	return out
}

func anyBlockHash(hs []string) bool {
	for _, h := range hs {
		if h != "" {
			return true
		}
	}
	return false
}

func (s *PrefixResidencyScorer) pruneLocked(n *prsNode) {
	for n != s.root && len(n.holders) == 0 && len(n.children) == 0 {
		delete(n.parent.children, n.hash)
		if set := s.byHash[n.hash]; set != nil {
			delete(set, n)
			if len(set) == 0 {
				delete(s.byHash, n.hash)
			}
		}
		n.pruned = true
		s.nodes--
		n = n.parent
	}
}

// NodeCount reports the number of non-root trie nodes.
func (s *PrefixResidencyScorer) NodeCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.nodes
}

// Depths returns, for every worker holding any leading run of chain, how many
// leading blocks it holds contiguously. Workers holding nothing are absent.
func (s *PrefixResidencyScorer) Depths(chain []string) map[string]int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.depthsLocked(chain)
}

func (s *PrefixResidencyScorer) depthsLocked(chain []string) map[string]int {
	depths := map[string]int{}
	if len(chain) == 0 {
		return depths
	}
	full := s.descendLocked(chain, 0, depths)
	if full {
		return depths
	}
	for skip := 1; skip < len(chain); skip++ {
		if _, ok := s.root.children[chain[skip]]; ok {
			s.descendLocked(chain[skip:], skip, depths)
		}
	}
	return depths
}

// descendLocked walks chain from the root, raising each worker's depth to the
// last level it was present at contiguously from the first block, plus offset.
// It reports whether some worker held the whole chain.
func (s *PrefixResidencyScorer) descendLocked(chain []string, offset int, depths map[string]int) bool {
	record := func(w string, d int) {
		if d > depths[w] {
			depths[w] = d
		}
	}
	var alive []string
	cur := s.root
	reached := 0
	for _, h := range chain {
		next := cur.children[h]
		if next == nil {
			break
		}
		cur = next
		reached++
		if reached == 1 {
			for w := range next.holders {
				alive = append(alive, w)
			}
		} else {
			still := alive[:0]
			for _, w := range alive {
				if _, ok := next.holders[w]; ok {
					still = append(still, w)
				} else {
					record(w, offset+reached-1)
				}
			}
			alive = still
		}
		if len(alive) == 0 {
			break
		}
	}
	for _, w := range alive {
		record(w, offset+reached)
	}
	return len(alive) > 0 && reached == len(chain)
}

// Score returns one entry per candidate, in candidate order: matched depth and
// depth divided by chain length. An empty chain scores every candidate zero.
func (s *PrefixResidencyScorer) Score(chain []string, candidates []string) []PrefixResidencyScore {
	out := make([]PrefixResidencyScore, len(candidates))
	var depths map[string]int
	if len(chain) > 0 && len(candidates) > 0 {
		depths = s.Depths(chain)
	}
	for i, c := range candidates {
		d := depths[c]
		out[i] = PrefixResidencyScore{Worker: c, MatchedBlocks: d}
		if d > 0 {
			out[i].Score = float64(d) / float64(len(chain))
		}
	}
	return out
}

// RankPrefixResidencyScores returns a copy of scores ordered best first, ties
// broken by worker id.
func RankPrefixResidencyScores(scores []PrefixResidencyScore) []PrefixResidencyScore {
	out := append([]PrefixResidencyScore(nil), scores...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].MatchedBlocks != out[j].MatchedBlocks {
			return out[i].MatchedBlocks > out[j].MatchedBlocks
		}
		return out[i].Worker < out[j].Worker
	})
	return out
}

// Overlap implements ResidencyView.
func (s *PrefixResidencyScorer) Overlap(worker string, prefix []string) int {
	if worker == "" {
		return 0
	}
	return s.Depths(prefix)[worker]
}

// ResidentWorkers implements ResidencyView.
func (s *PrefixResidencyScorer) ResidentWorkers(prefix []string) []ResidentWorker {
	depths := s.Depths(prefix)
	if len(depths) == 0 {
		return nil
	}
	scores := make([]PrefixResidencyScore, 0, len(depths))
	for w, d := range depths {
		scores = append(scores, PrefixResidencyScore{Worker: w, MatchedBlocks: d})
	}
	ranked := RankPrefixResidencyScores(scores)
	out := make([]ResidentWorker, len(ranked))
	for i, r := range ranked {
		out[i] = ResidentWorker{Worker: r.Worker, Overlap: r.MatchedBlocks}
	}
	return out
}
