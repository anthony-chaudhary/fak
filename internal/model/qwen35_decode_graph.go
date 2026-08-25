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
	Position int                     `json:"position"`
	Nodes    []Qwen35DecodeGraphNode `json:"nodes"`
	Aborted  bool                    `json:"aborted"`
}

type qwen35DecodeGraphRecorder struct {
	mu     sync.Mutex
	active *Qwen35DecodeGraphTrace
	last   Qwen35DecodeGraphTrace
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
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: kind, DependsOn: []string{"layer-input"}, HostRead: true, Syncs: 2})
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: "post-attention-residual-norm", DependsOn: []string{kind}, HostRead: true})
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: "mlp", DependsOn: []string{"post-attention-residual-norm"}, HostRead: true, Syncs: 1})
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Layer: l, Operation: "next-layer-handoff", DependsOn: []string{"mlp"}, HostRead: true})
}
