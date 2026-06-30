package gateway

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

func inboundSystemBody(t *testing.T) []byte {
	t.Helper()
	type obj map[string]any
	system := []obj{
		{"type": "text", "name": "core", "block": promptmmu.BlockSystem, "text": "resident spine"},
		{"type": "text", "name": "policy", "block": promptmmu.BlockSystem, "text": "resident policy",
			"cache_control": obj{"type": "ephemeral"}},
		{"type": "text", "name": "current_skill", "block": promptmmu.BlockSkills, "text": "fresh skill"},
		{"type": "text", "name": "old_skill", "block": promptmmu.BlockSkills, "text": "stale skill"},
		{"type": "text", "name": "old_skill", "block": promptmmu.BlockMemory, "text": "same name in memory"},
	}
	raw, err := json.Marshal(obj{
		"model": "claude-sonnet-4-6", "max_tokens": 1024, "stream": true,
		"system": system,
		"messages": []obj{
			{"role": "user", "content": []obj{{"type": "text", "text": "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

func anthropicServerWithSystemDrop(drop func(block, name string) bool) *Server {
	return &Server{
		planner:         &agent.HTTPPlanner{Provider: agent.ProviderAnthropic},
		systemBlockDrop: drop,
		logf:            func(string, ...any) {},
	}
}

func TestInboundSystemNilPredicateIsIdentity(t *testing.T) {
	raw := inboundSystemBody(t)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := anthropicServerWithSystemDrop(nil)
	if pruned := s.maybeCompactInboundSystem(req); pruned != nil {
		t.Fatalf("nil predicate must prune nothing, got %v", pruned)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("nil predicate must leave req.Raw unchanged")
	}
}

func TestInboundSystemNonPassthroughIsIdentity(t *testing.T) {
	raw := inboundSystemBody(t)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	s := &Server{
		planner: agent.NewMockPlanner("m"),
		systemBlockDrop: func(block, name string) bool {
			return block == promptmmu.BlockSkills && name == "old_skill"
		},
		logf: func(string, ...any) {},
	}
	if pruned := s.maybeCompactInboundSystem(req); pruned != nil {
		t.Fatalf("non-passthrough must prune nothing, got %v", pruned)
	}
	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("non-passthrough wire must leave req.Raw unchanged")
	}
}

func TestInboundSystemPrunesSelectedBlockKeepsPrefix(t *testing.T) {
	raw := inboundSystemBody(t)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)
	_, prefixEnd, _, ok := promptmmu.ArraySplicePoints(orig, "system")
	if !ok {
		t.Fatal("could not locate system[] cache breakpoint")
	}

	s := anthropicServerWithSystemDrop(func(block, name string) bool {
		return block == promptmmu.BlockSkills && name == "old_skill"
	})
	pruned := s.maybeCompactInboundSystem(req)
	if len(pruned) != 1 || pruned[0] != promptmmu.BlockSkills+":old_skill" {
		t.Fatalf("Pruned = %v, want [skills:old_skill]", pruned)
	}
	if bytes.Equal(req.Raw, orig) {
		t.Fatalf("expected a system-block prune, got identity")
	}
	if prefixEnd > len(req.Raw) || !bytes.Equal(orig[:prefixEnd], req.Raw[:prefixEnd]) {
		t.Fatalf("system cache prefix bytes changed")
	}
	out, err := agent.DecodeAnthropicMessagesRequest(req.Raw)
	if err != nil {
		t.Fatalf("pruned body failed to re-decode: %v", err)
	}
	if !bytes.Contains([]byte(out.System), []byte("same name in memory")) {
		t.Fatalf("memory block with the same name must survive; system=%q", out.System)
	}
	if bytes.Contains([]byte(out.System), []byte("stale skill")) {
		t.Fatalf("stale skill block should be gone; system=%q", out.System)
	}
}
