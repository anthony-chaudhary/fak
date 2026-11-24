package model

import "fmt"

// Inkling2TreeNode is one speculative token. Parent indexes an earlier node, or
// is -1 for a child of the committed prefix.
type Inkling2TreeNode struct {
	Token  int
	Parent int
}

// inkling2TreeState is the per-visit cache a node leaves behind: its ancestors'
// keys/values/positions plus its own, and its short-conv history (the last
// ConvKernel-1 inputs per channel), which is all a rollback has to restore.
type inkling2TreeState struct {
	keys, values [][][]float32
	positions    [][]int
	conv         [][]float32
	pos          int
	features     [][]float32
}

// Inkling2TreeVerify holds one verified token tree. Logits[i] are node i's
// unpadded logits, Features[i] its target feature taps.
type Inkling2TreeVerify struct {
	host     *Inkling2LoopHost
	base     inkling2TreeState
	states   []inkling2TreeState
	Logits   [][]float32
	Features [][][]float32
	done     bool
}

// snapshot caps every slice at its length, so a later append from any node
// reallocates instead of writing into storage a sibling also reads.
func (h *Inkling2LoopHost) snapshot() inkling2TreeState {
	s := inkling2TreeState{
		keys:      make([][][]float32, len(h.keys)),
		values:    make([][][]float32, len(h.values)),
		positions: make([][]int, len(h.positions)),
		conv:      make([][]float32, len(h.conv)),
		pos:       h.pos,
		features:  h.Features,
	}
	for i := range h.keys {
		s.keys[i] = h.keys[i][:len(h.keys[i]):len(h.keys[i])]
		s.values[i] = h.values[i][:len(h.values[i]):len(h.values[i])]
		s.positions[i] = h.positions[i][:len(h.positions[i]):len(h.positions[i])]
		s.conv[i] = append([]float32(nil), h.conv[i]...)
	}
	return s
}

// install makes s the host's live state; conv is copied because Step shifts it
// in place.
func (h *Inkling2LoopHost) install(s inkling2TreeState) {
	for i := range h.keys {
		h.keys[i], h.values[i], h.positions[i] = s.keys[i], s.values[i], s.positions[i]
		h.conv[i] = append(h.conv[i][:0], s.conv[i]...)
	}
	h.pos = s.pos
	h.Features = s.features
}

// VerifyTree runs every node of a token tree in one call, each at its depth's
// position over the committed prefix plus its ancestors only, and leaves the
// host exactly at the committed prefix. Node logits equal a serial decode of
// the node's root path bit for bit, because each node runs the same Step over
// the same cache contents.
func (h *Inkling2LoopHost) VerifyTree(nodes []Inkling2TreeNode) (*Inkling2TreeVerify, error) {
	for i, n := range nodes {
		if n.Parent < -1 || n.Parent >= i {
			return nil, fmt.Errorf("inkling2 tree: node %d parent %d must be -1 or an earlier node", i, n.Parent)
		}
	}
	v := &Inkling2TreeVerify{host: h, base: h.snapshot()}
	defer h.install(v.base)
	for _, n := range nodes {
		parent := v.base
		if n.Parent >= 0 {
			parent = v.states[n.Parent]
		}
		h.install(parent)
		v.Logits = append(v.Logits, append([]float32(nil), h.Step(n.Token)...))
		v.Features = append(v.Features, h.Features)
		v.states = append(v.states, h.snapshot())
	}
	return v, nil
}

// Commit keeps node's root path (or nothing for -1) and drops every other
// branch. The host must not have stepped since VerifyTree.
func (v *Inkling2TreeVerify) Commit(node int) error {
	if v.done {
		return fmt.Errorf("inkling2 tree: already committed")
	}
	if node < -1 || node >= len(v.states) {
		return fmt.Errorf("inkling2 tree: commit node %d of %d", node, len(v.states))
	}
	if v.host.pos != v.base.pos {
		return fmt.Errorf("inkling2 tree: host advanced from %d to %d since verify", v.base.pos, v.host.pos)
	}
	v.done = true
	if node >= 0 {
		v.host.install(v.states[node])
	}
	return nil
}
