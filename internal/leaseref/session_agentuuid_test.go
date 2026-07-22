package leaseref

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestSessionDescriptorAgentUUIDRoundTrip proves the new optional AgentUUID field survives
// the real encode -> ref -> decode plumbing: a descriptor published with a Claude UUID reads
// back byte-identical, so a liveness reader can recover the UUID to join a wip checkpoint
// (#5343).
func TestSessionDescriptorAgentUUIDRoundTrip(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	in := SessionDescriptor{
		ID:        "agent-claude-10800-17b88dff",
		Host:      "h",
		PCBState:  "RUNNING",
		UpdatedAt: 1,
		TTLSecs:   1800,
		AgentUUID: "1e21323a-b92d-4b43-a495-1e0c1d46f3ef",
	}
	if _, err := s.PublishSession(ctx(), in); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	out, ok, err := s.GetSession(ctx(), in.ID)
	if err != nil || !ok {
		t.Fatalf("GetSession ok=%v err=%v", ok, err)
	}
	if out != in {
		t.Fatalf("round-trip descriptor = %+v, want %+v", out, in)
	}
	if out.AgentUUID != in.AgentUUID {
		t.Fatalf("AgentUUID not preserved: got %q, want %q", out.AgentUUID, in.AgentUUID)
	}
}

// TestSessionDescriptorLegacyBlobNoAgentUUID pins forward/backward compat on the fleet-visible
// ref blob: a LEGACY descriptor written by a prior binary (no agent_uuid key) unmarshals
// cleanly to an empty AgentUUID — no error, and no phantom key. It also asserts omitempty, so
// a descriptor with no UUID marshals to the byte-identical legacy shape an old reader expects.
func TestSessionDescriptorLegacyBlobNoAgentUUID(t *testing.T) {
	legacy := `{"id":"agent-claude-10800-17b88dff","host":"h","pcb_state":"RUNNING","updated_at":1,"ttl_seconds":1800}`
	var d SessionDescriptor
	if err := json.Unmarshal([]byte(legacy), &d); err != nil {
		t.Fatalf("legacy blob must unmarshal cleanly: %v", err)
	}
	if d.AgentUUID != "" {
		t.Fatalf("legacy blob AgentUUID = %q, want empty", d.AgentUUID)
	}
	// omitempty: an empty UUID must NOT appear on the wire, so an old reader still sees the
	// legacy shape and a new reader never manufactures an empty-string join key.
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "agent_uuid") {
		t.Fatalf("empty AgentUUID must be omitted from the wire form, got %s", b)
	}
}
