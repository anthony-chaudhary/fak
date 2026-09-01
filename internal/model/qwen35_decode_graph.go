package model

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

type Qwen35DecodeGraphNativeAttributionStatus string

const (
	Qwen35DecodeGraphNativeAttributionAvailable   Qwen35DecodeGraphNativeAttributionStatus = "available"
	Qwen35DecodeGraphNativeAttributionUnavailable Qwen35DecodeGraphNativeAttributionStatus = "unavailable"
)

type Qwen35DecodeGraphNativeAttributionReason string

const Qwen35DecodeGraphMetalEventSourceUnavailable Qwen35DecodeGraphNativeAttributionReason = "metal-event-source-unavailable:#8844"

type Qwen35DecodeGraphNode struct {
	Layer                   int                                      `json:"layer"`
	Operation               string                                   `json:"operation"`
	DependsOn               []string                                 `json:"depends_on,omitempty"`
	NativeAttributionStatus Qwen35DecodeGraphNativeAttributionStatus `json:"native_attribution_status"`
	UnavailableReason       Qwen35DecodeGraphNativeAttributionReason `json:"unavailable_reason,omitempty"`
	HostRead                *bool                                    `json:"host_read,omitempty"`
	Syncs                   *int                                     `json:"syncs,omitempty"`
	NativeEvents            []metalgemm.ExecutionEvent               `json:"native_events,omitempty"`
}

type Qwen35DecodeGraphTrace struct {
	Position int                     `json:"position"`
	Nodes    []Qwen35DecodeGraphNode `json:"nodes"`
	Aborted  bool                    `json:"aborted"`
}

type qwen35DecodeGraphRecorder struct {
	mu        sync.Mutex
	nextOwner uint64
	begins    uint64
	finishes  uint64
	active    *qwen35DecodeGraphActive
	last      Qwen35DecodeGraphTrace
}

type qwen35DecodeGraphActive struct {
	owner        uint64
	trace        Qwen35DecodeGraphTrace
	nativeEvents []metalgemm.ExecutionEvent
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
	return cloneQwen35DecodeGraphTrace(r.last)
}
func (s *Session) beginQwen35DecodeGraph(pos int) func(bool) {
	r := s.qwen35DecodeGraph
	if r == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return func(bool) {}
	}
	r.mu.Lock()
	if r.active != nil {
		r.mu.Unlock()
		return func(bool) {}
	}
	r.nextOwner++
	owner := r.nextOwner
	r.begins++
	r.active = &qwen35DecodeGraphActive{
		owner: owner,
		trace: Qwen35DecodeGraphTrace{Position: pos},
	}
	r.mu.Unlock()
	finished := false
	return func(aborted bool) {
		r.mu.Lock()
		defer r.mu.Unlock()
		if finished {
			return
		}
		finished = true
		if r.active == nil || r.active.owner != owner {
			return
		}
		r.active.trace.Aborted = aborted
		r.last = cloneQwen35DecodeGraphTrace(r.active.trace)
		r.active = nil
		r.finishes++
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
		r.active.trace.Nodes = append(r.active.trace.Nodes, n)
	}
}
func (s *Session) observeQwen35MetalExecutionSnapshot(snapshot metalgemm.ExecutionSnapshot) {
	if len(snapshot.Events) == 0 || s.qwen35DecodeGraph == nil {
		return
	}
	r := s.qwen35DecodeGraph
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil {
		r.active.nativeEvents = append(r.active.nativeEvents, snapshot.Events...)
	}
}

func (s *Session) recordQwen35LayerGraph(l int, linear bool) {
	kind := "full-attention"
	if linear {
		kind = "linear-attention"
	}
	s.recordQwen35ModelGraphNode(l, kind, "layer-input")
	s.recordQwen35ModelGraphNode(l, "post-attention-residual-norm", kind)
	// blockStep dispatches ordinary FFN primitives here; do not label the operation fused.
	s.recordQwen35ModelGraphNode(l, "mlp", "post-attention-residual-norm")
	s.recordQwen35ModelGraphNode(l, "next-layer-handoff", "mlp")
}

func (s *Session) recordQwen35ModelGraphNode(layer int, operation string, dependsOn ...string) {
	n := Qwen35DecodeGraphNode{
		Layer: layer, Operation: operation, DependsOn: dependsOn,
		NativeAttributionStatus: Qwen35DecodeGraphNativeAttributionUnavailable,
		UnavailableReason:       Qwen35DecodeGraphMetalEventSourceUnavailable,
	}
	r := s.qwen35DecodeGraph
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active == nil {
		return
	}
	if len(r.active.nativeEvents) != 0 {
		n.NativeAttributionStatus = Qwen35DecodeGraphNativeAttributionAvailable
		n.UnavailableReason = ""
		n.NativeEvents = append([]metalgemm.ExecutionEvent(nil), r.active.nativeEvents...)
		r.active.nativeEvents = nil
		hostRead, syncs := false, 0
		for _, event := range n.NativeEvents {
			hostRead = hostRead || event.HostReadback
			if event.CompletedWait {
				syncs++
			}
		}
		n.HostRead, n.Syncs = &hostRead, &syncs
	}
	r.active.trace.Nodes = append(r.active.trace.Nodes, n)
}

func cloneQwen35DecodeGraphTrace(in Qwen35DecodeGraphTrace) Qwen35DecodeGraphTrace {
	out := in
	out.Nodes = append([]Qwen35DecodeGraphNode(nil), in.Nodes...)
	for i := range out.Nodes {
		out.Nodes[i].DependsOn = append([]string(nil), in.Nodes[i].DependsOn...)
		out.Nodes[i].NativeEvents = append([]metalgemm.ExecutionEvent(nil), in.Nodes[i].NativeEvents...)
	}
	return out
}
