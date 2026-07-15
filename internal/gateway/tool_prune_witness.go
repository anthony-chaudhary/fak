package gateway

import (
	"strings"
)

// recordInboundPrunedToolDefinitions records the concrete tool names pruned from one
// served trace's advertised tools[]. It is observability-only: call-time adjudication still
// decides the proposed tool, this just makes a later "model named something we stopped
// advertising" event visible.
func (s *Server) recordInboundPrunedToolDefinitions(trace string, names []string) {
	if s == nil || trace == "" || len(names) == 0 {
		return
	}
	s.prunedToolDefsMu.Lock()
	if s.prunedToolDefs == nil {
		s.prunedToolDefs = map[string]map[string]struct{}{}
	}
	if len(s.prunedToolDefs) >= maxResetHealthSessions {
		for k := range s.prunedToolDefs {
			delete(s.prunedToolDefs, k)
			if s.notedPrunedToolProposals != nil {
				delete(s.notedPrunedToolProposals, k)
			}
			break
		}
	}
	set := s.prunedToolDefs[trace]
	if set == nil {
		set = map[string]struct{}{}
		s.prunedToolDefs[trace] = set
	}
	fresh := make([]string, 0, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if _, already := set[name]; already {
			continue
		}
		set[name] = struct{}{}
		fresh = append(fresh, name)
	}
	s.prunedToolDefsMu.Unlock()
	for _, name := range fresh {
		if j := activeJournal(); j != nil {
			j.AppendToolDefinitionPruned(trace, name)
		}
	}
}

func (s *Server) observePrunedToolProposal(trace, tool string) {
	tool = strings.TrimSpace(tool)
	if s == nil || trace == "" || tool == "" {
		return
	}
	fresh := false
	s.prunedToolDefsMu.Lock()
	if defs := s.prunedToolDefs[trace]; defs != nil {
		if _, pruned := defs[tool]; pruned {
			if s.notedPrunedToolProposals == nil {
				s.notedPrunedToolProposals = map[string]map[string]struct{}{}
			}
			seen := s.notedPrunedToolProposals[trace]
			if seen == nil {
				seen = map[string]struct{}{}
				s.notedPrunedToolProposals[trace] = seen
			}
			if _, already := seen[tool]; !already {
				seen[tool] = struct{}{}
				fresh = true
			}
		}
	}
	s.prunedToolDefsMu.Unlock()
	if !fresh {
		return
	}
	s.metrics.observeInboundPrunedToolProposal(1)
	if s.logf != nil {
		s.logf("gateway: pruned tool definition later proposed trace=%s tool=%s", trace, tool)
	}
}
