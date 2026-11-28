package radixkv

import "container/list"

// radix_refs.go — session-scoped radix refs (fak-private#3727): per-node coverage
// counts, last-holder release, a generation guard and a bounded closed-session
// tombstone. Borrowed from SGLang's session_ref_tracker (Apache-2.0).
//
// Each live session covers the root→leaf path of its latest registered tokens by
// incrementing node.holderRefs along it. A node with holderRefs>0 is immune to
// eviction (isNodePinnedOrImmuneAt). The counter is separate from the boolean
// `pinned` flag and Tier0 retention, so ReleaseSession never clears a PinPrefix pin
// and UnpinNode never clears a session's coverage.

// sessionTombstoneCap bounds the LRU of closed session ids that refuse late registers.
const sessionTombstoneCap = 8192

type sessionCover struct {
	ns     string
	tokens []int
}

type sessionState struct {
	gens    map[string]uint64       // live session id -> current generation
	covered map[string]sessionCover // live session id -> currently covered path
	nextGen uint64
	tombCap int
	tombs   map[string]*list.Element // closed session ids (value: id)
	tombLRU *list.List               // front = most recently closed
}

func (t *Tree) sessionsState() *sessionState {
	if t.sessions == nil {
		t.sessions = &sessionState{
			gens:    map[string]uint64{},
			covered: map[string]sessionCover{},
			tombCap: sessionTombstoneCap,
			tombs:   map[string]*list.Element{},
			tombLRU: list.New(),
		}
	}
	return t.sessions
}

// OpenSession starts (or restarts) session sid and returns its new generation.
// Reopening drops any closed-session tombstone and releases stale coverage, so
// registers carrying an older generation become no-ops.
func (t *Tree) OpenSession(sid string) uint64 {
	s := t.sessionsState()
	if _, live := s.gens[sid]; live {
		t.dropSessionCover(s, sid)
	}
	if e := s.tombs[sid]; e != nil {
		s.tombLRU.Remove(e)
		delete(s.tombs, sid)
	}
	s.nextGen++
	s.gens[sid] = s.nextGen
	return s.nextGen
}

// RegisterSessionLeaf makes session sid cover the cached prefix of tokens under ns.
// Any previously covered path is released first, so advancing a session never
// double-counts a shared ancestor. Stale-generation, unknown, or closed sessions are
// no-ops. It returns the number of tokens now covered (the cached prefix; nodes are
// never created here).
func (t *Tree) RegisterSessionLeaf(ns, sid string, gen uint64, tokens []int) int {
	s := t.sessionsState()
	if cur, ok := s.gens[sid]; !ok || cur != gen {
		return 0
	}
	t.dropSessionCover(s, sid)
	root := t.rootForRead(ns)
	if root == nil || len(tokens) == 0 {
		return 0
	}
	boundary, matched := t.boundaryFor(root, tokens)
	if matched == 0 {
		return 0
	}
	for p := boundary; p != nil && p.parent != nil; p = p.parent {
		p.holderRefs++
	}
	s.covered[sid] = sessionCover{ns: ns, tokens: append([]int(nil), tokens[:matched]...)}
	return matched
}

// ReleaseSession closes sid: it decrements every node its coverage touched (a node
// becomes evictable only when its last holder releases) and tombstones sid so a late
// register is refused. Releasing an unknown session still tombstones it.
func (t *Tree) ReleaseSession(sid string) {
	s := t.sessionsState()
	t.dropSessionCover(s, sid)
	delete(s.gens, sid)
	if e := s.tombs[sid]; e != nil {
		s.tombLRU.MoveToFront(e)
		return
	}
	s.tombs[sid] = s.tombLRU.PushFront(sid)
	for s.tombLRU.Len() > s.tombCap {
		old := s.tombLRU.Back()
		s.tombLRU.Remove(old)
		delete(s.tombs, old.Value.(string))
	}
}

// SessionClosed reports whether sid is in the closed-session tombstone.
func (t *Tree) SessionClosed(sid string) bool {
	if t.sessions == nil {
		return false
	}
	return t.sessions.tombs[sid] != nil
}

// SessionImmuneNodes counts nodes currently held by at least one session.
func (t *Tree) SessionImmuneNodes() int {
	n := 0
	t.forEachRoot(func(r *node) {
		stack := []*node{r}
		for len(stack) > 0 {
			x := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if x.holderRefs > 0 {
				n++
			}
			for _, c := range x.children {
				stack = append(stack, c)
			}
		}
	})
	return n
}

// dropSessionCover decrements the path sid covered, clamping at zero. Splits copy
// holderRefs to the new intermediate node, so a re-walk by tokens visits exactly the
// nodes that were incremented.
func (t *Tree) dropSessionCover(s *sessionState, sid string) {
	c, ok := s.covered[sid]
	if !ok {
		return
	}
	delete(s.covered, sid)
	n := t.rootForRead(c.ns)
	i := 0
	for n != nil && i < len(c.tokens) {
		ch := n.children[c.tokens[i]]
		if ch == nil || i+len(ch.key) > len(c.tokens) {
			return
		}
		for j, k := range ch.key {
			if c.tokens[i+j] != k {
				return
			}
		}
		if ch.holderRefs > 0 {
			ch.holderRefs--
		}
		n, i = ch, i+len(ch.key)
	}
}
