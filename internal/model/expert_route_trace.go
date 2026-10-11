package model

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
)

// ExpertRouteTraceSchema names the JSONL routed-expert selection trace: one
// ExpertRouteEvent per (token, layer) router decision, optionally preceded by a
// header line carrying the same schema plus the routed geometry.
const ExpertRouteTraceSchema = "fak.expert-route.trace/v1"

// ExpertRouteTraceEnv names the opt-in JSONL sink. When set to a path, every
// observed router decision in the process is appended to that file.
const ExpertRouteTraceEnv = "FAK_EXPERT_TRACE"

// ExpertRouteEvent is one router decision. Token is the absolute sequence
// position the router ran for (-1 when the caller does not know it). A
// full-history recompute re-routes earlier positions, so the same (Token,
// Layer) can legitimately appear more than once; consumers dedupe on it.
type ExpertRouteEvent struct {
	Layer   int    `json:"layer"`
	Token   int    `json:"token"`
	Phase   string `json:"phase"`
	Experts []int  `json:"experts"`
}

// ExpertRouteTraceHeader is the optional first JSONL line.
type ExpertRouteTraceHeader struct {
	Schema  string `json:"schema"`
	Layers  int    `json:"layers,omitempty"`
	Experts int    `json:"experts,omitempty"`
	TopK    int    `json:"top_k,omitempty"`
}

// ExpertRouteEventObserver receives a router decision. ev.Experts is a fresh
// copy owned by the observer.
type ExpertRouteEventObserver func(ev ExpertRouteEvent)

// SetExpertRouteEventObserver installs or clears the positional route seam. Like
// SetExpertRouteObserver it must not change concurrently with a forward pass.
func (m *Model) SetExpertRouteEventObserver(obs ExpertRouteEventObserver) {
	m.expertRouteEventObs = obs
}

type expertRouteTraceSink struct {
	mu   sync.Mutex
	f    *os.File
	err  error
	seen sync.Map
}

var (
	expertRouteSinkOnce sync.Once
	expertRouteSink     *expertRouteTraceSink
)

func processExpertRouteSink() *expertRouteTraceSink {
	expertRouteSinkOnce.Do(func() {
		path := strings.TrimSpace(os.Getenv(ExpertRouteTraceEnv))
		if path == "" {
			return
		}
		f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak: %s=%s: %v (expert route trace disabled)\n", ExpertRouteTraceEnv, path, err)
			return
		}
		expertRouteSink = &expertRouteTraceSink{f: f}
	})
	return expertRouteSink
}

func (s *expertRouteTraceSink) write(m *Model, ev ExpertRouteEvent) {
	if _, loaded := s.seen.LoadOrStore(m, true); !loaded {
		hdr := ExpertRouteTraceHeader{Schema: ExpertRouteTraceSchema, Layers: m.Cfg.NumLayers, Experts: m.Cfg.NumExperts, TopK: m.Cfg.NumExpertsPerTok}
		s.writeLine(hdr)
	}
	s.writeLine(ev)
}

func (s *expertRouteTraceSink) writeLine(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	b = append(b, '\n')
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return
	}
	if _, err := s.f.Write(b); err != nil {
		s.err = err
		fmt.Fprintf(os.Stderr, "fak: %s write: %v (expert route trace stopped)\n", ExpertRouteTraceEnv, err)
	}
}

// emitExpertRouteAt fans one router decision out to the id-only replay
// observer, the positional observer, and the opt-in JSONL sink. With none set
// it returns before allocating.
func (m *Model) emitExpertRouteAt(layer, token int, picks []routePick) {
	if m == nil {
		return
	}
	sink := processExpertRouteSink()
	if m.expertRouteObs == nil && m.expertRouteEventObs == nil && sink == nil {
		return
	}
	if m.expertRouteObs != nil {
		experts := make([]int, len(picks))
		for i, pick := range picks {
			experts[i] = pick.expert
		}
		m.expertRouteObs(layer, experts)
	}
	if m.expertRouteEventObs == nil && sink == nil {
		return
	}
	experts := make([]int, len(picks))
	for i, pick := range picks {
		experts[i] = pick.expert
	}
	ev := ExpertRouteEvent{Layer: layer, Token: token, Phase: m.v41ExpertFaultPhase().String(), Experts: experts}
	if sink != nil {
		sink.write(m, ev)
	}
	if m.expertRouteEventObs != nil {
		m.expertRouteEventObs(ev)
	}
}
