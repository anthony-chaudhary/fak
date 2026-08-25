package model

import "sync"

type Qwen35DecodeGraphNode struct {
	Layer     int      `json:"layer"`
	Operation string   `json:"operation"`
	DependsOn []string `json:"depends_on,omitempty"`
	HostRead  bool     `json:"host_read"`
	Syncs     int      `json:"syncs"`
}

type Qwen35DecodeGraphTrace struct {
	Position     int                     `json:"position"`
	Nodes        []Qwen35DecodeGraphNode `json:"nodes"`
	NativeEvents Qwen35NativeEvents      `json:"native_events"`
	Aborted      bool                    `json:"aborted"`
}

type Qwen35NativeEvents struct {
	CommandBuffers uint64 `json:"command_buffers"`
	Commits        uint64 `json:"commits"`
	Waits          uint64 `json:"waits"`
}

func (e Qwen35NativeEvents) sub(before Qwen35NativeEvents) Qwen35NativeEvents {
	return Qwen35NativeEvents{
		CommandBuffers: e.CommandBuffers - before.CommandBuffers,
		Commits:        e.Commits - before.Commits,
		Waits:          e.Waits - before.Waits,
	}
}

type Qwen35MetalBatchMode uint8

const (
	Qwen35MetalBatchDisabled Qwen35MetalBatchMode = iota
	Qwen35MetalBatchControl
	Qwen35MetalBatchMixed
)

type qwen35ProjectionBatcher interface {
	Begin(*Session) bool
	MulGroup(*Session, int, []string, []float32, []int) ([][]float32, Qwen35NativeEvents, bool)
	Finish(*Session)
	Abort(*Session)
	Close(*Session)
}

type qwen35ProjectionBatchState struct {
	batcher qwen35ProjectionBatcher
	active  bool
}

type qwen35DecodeGraphRecorder struct {
	mu     sync.Mutex
	active *Qwen35DecodeGraphTrace
	last   Qwen35DecodeGraphTrace
}

// SetQwen35MetalBatchMode selects the explicit grouped-format control or the mixed Q8/Q4_K
// candidate for this session. Disabled is the default and leaves the production path behind one
// nil check. The candidate is fak-native Metal only; unsupported builds/devices return false and
// leave the session disabled.
func (s *Session) SetQwen35MetalBatchMode(mode Qwen35MetalBatchMode) bool {
	if s == nil {
		return false
	}
	s.closeQwen35ProjectionBatch()
	if mode == Qwen35MetalBatchDisabled {
		return true
	}
	b := newQwen35ProjectionBatcher(mode)
	if b == nil {
		return false
	}
	s.qwen35ProjectionBatch = &qwen35ProjectionBatchState{batcher: b}
	return true
}

func (s *Session) beginQwen35ProjectionBatch() func(bool) {
	st := s.qwen35ProjectionBatch
	if st == nil || st.batcher == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return func(bool) {}
	}
	if st.active {
		panic("model: qwen35 projection batch already active for session")
	}
	if !st.batcher.Begin(s) {
		return func(bool) {}
	}
	st.active = true
	return func(aborted bool) {
		if !st.active {
			return
		}
		st.active = false
		if aborted {
			st.batcher.Abort(s)
			return
		}
		st.batcher.Finish(s)
	}
}

func (s *Session) closeQwen35ProjectionBatch() {
	st := s.qwen35ProjectionBatch
	if st == nil {
		return
	}
	if st.active {
		st.active = false
		st.batcher.Abort(s)
	}
	st.batcher.Close(s)
	s.qwen35ProjectionBatch = nil
}

func (s *Session) qwen35MulGroup(l int, mat matKernel, xp any, names []string, outs []int, in int) [][]float32 {
	st := s.qwen35ProjectionBatch
	if st != nil && st.active {
		if xf, ok := xp.([]float32); ok {
			if out, events, handled := st.batcher.MulGroup(s, l, names, xf, outs); handled {
				s.recordQwen35NativeEvents(l, "full-attention", events)
				return out
			}
		}
	}
	out := make([][]float32, len(names))
	for i, name := range names {
		out[i] = mat.mul(name, xp, outs[i], in)
	}
	return out
}

func (s *Session) EnableQwen35DecodeGraphTrace() {
	if s.qwen35DecodeGraph == nil {
		s.qwen35DecodeGraph = &qwen35DecodeGraphRecorder{}
	}
}
func (s *Session) LastQwen35DecodeGraphTrace() Qwen35DecodeGraphTrace {
	r := s.qwen35DecodeGraph
	if r == nil {
		return Qwen35DecodeGraphTrace{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.last
	out.Nodes = append([]Qwen35DecodeGraphNode(nil), out.Nodes...)
	return out
}
func (s *Session) beginQwen35DecodeGraph(pos int) func(bool) {
	r := s.qwen35DecodeGraph
	if r == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return func(bool) {}
	}
	r.mu.Lock()
	r.active = &Qwen35DecodeGraphTrace{Position: pos}
	r.mu.Unlock()
	return func(aborted bool) {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.active == nil {
			return
		}
		r.active.Aborted = aborted
		r.last = *r.active
		r.last.Nodes = append([]Qwen35DecodeGraphNode(nil), r.active.Nodes...)
		r.active = nil
	}
}
func (s *Session) recordQwen35DecodeGraph(n Qwen35DecodeGraphNode) {
	r := s.qwen35DecodeGraph
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil {
		r.active.Nodes = append(r.active.Nodes, n)
	}
}
func (s *Session) recordQwen35LayerGraph(l int, linear bool) {
	kind := "full-attention"
	if linear {
		kind = "linear-attention"
	}
	syncs := 2
	mlpSyncs := 1
	if st := s.qwen35ProjectionBatch; st != nil && st.active {
		syncs = 0
		mlpSyncs = 0
	}
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: kind, DependsOn: []string{"layer-input"}, HostRead: true, Syncs: syncs})
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: "post-attention-residual-norm", DependsOn: []string{kind}, HostRead: true})
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: "mlp", DependsOn: []string{"post-attention-residual-norm"}, HostRead: true, Syncs: mlpSyncs})
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: "next-layer-handoff", DependsOn: []string{"mlp"}, HostRead: true})
}

func (s *Session) recordQwen35NativeEvents(layer int, operation string, events Qwen35NativeEvents) {
	r := s.qwen35DecodeGraph
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active == nil {
		return
	}
	r.active.NativeEvents.CommandBuffers += events.CommandBuffers
	r.active.NativeEvents.Commits += events.Commits
	r.active.NativeEvents.Waits += events.Waits
	for i := range r.active.Nodes {
		n := &r.active.Nodes[i]
		if n.Layer == layer && n.Operation == operation {
			n.Syncs += int(events.Waits)
			return
		}
	}
}
