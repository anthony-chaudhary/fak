package cluster

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: direct ACK handling without Start, sockets or background work.
func TestGossipAckRestoresRouting(t *testing.T) {
	t.Parallel()
	for _, state := range []string{"removed", "suspect", "dead", "alive", "suspect_routed", "suspect_inactive"} {
		t.Run(state, func(t *testing.T) {
			ring := NewRing()
			g := NewGossip(GossipConfig{NodeID: "self", SuspectTimeout: time.Second}, ring)
			initial := state
			if state == "removed" || state == "suspect_routed" {
				initial = "alive"
			} else if state == "suspect_inactive" {
				initial = "suspect"
			}
			g.addMember(memberUpdate{NodeID: "peer", Addr: "peer:1234", State: initial})
			if state == "removed" {
				g.members["peer"].State = "suspect"
				g.members["peer"].SuspectAt = time.Now().Add(-3 * time.Second)
				g.checkSuspects()
				if g.members["peer"].State != "dead" || len(ring.Nodes()) != 0 {
					t.Fatal("death transition did not remove routing")
				}
			}
			var original *Node
			wantAddr := "peer:1234"
			if state == "alive" || state == "suspect_routed" || state == "suspect_inactive" {
				wantAddr = "retained:2345"
				original = &Node{ID: "peer", Addr: wantAddr, Alive: state != "suspect_inactive", Weight: 2}
				ring.AddNode(original)
			}
			if (original == nil || !original.Alive) && ring.GetPrimary([]byte("key")) != nil {
				t.Fatal("non-live fixture unexpectedly routes")
			}
			if state == "suspect_routed" {
				g.members["peer"].LastPing = time.Now().Add(-2 * time.Second)
				g.checkSuspects()
				if g.members["peer"].State != "suspect" || ring.GetPrimary([]byte("key")) != original {
					t.Fatal("suspicion unexpectedly changed the existing ring node")
				}
			}
			before := time.Now()
			g.handleMessage(gossipMessage{Type: "ack", SenderID: "peer"}, nil)
			member := g.members["peer"]
			if member.State != "alive" || member.LastPing.Before(before) {
				t.Fatal("ACK did not refresh live membership")
			}
			node := ring.GetPrimary([]byte("key"))
			if node == nil || node.ID != "peer" || node.Addr != wantAddr || !node.Alive || node.IsLocal || len(ring.Nodes()) != 1 {
				t.Fatalf("acknowledged peer is not routable: %+v", node)
			}
			if original != nil {
				if node.Weight != 2 || (original.Alive && node != original) {
					t.Fatal("ACK replaced existing routing metadata")
				}
				if state == "suspect_inactive" && (node == original || original.Alive) {
					t.Fatal("ACK mutated a previously returned inactive node")
				}
			}
			g.handleMessage(gossipMessage{Type: "ack", SenderID: "peer"}, nil)
			if len(ring.Nodes()) != 1 || ring.GetPrimary([]byte("key")) != node {
				t.Fatal("repeated ACK duplicated or replaced the live node")
			}
		})
	}
	t.Run("unknown ACK is ignored", func(t *testing.T) {
		ring := NewRing()
		g := NewGossip(GossipConfig{NodeID: "self"}, ring)
		g.handleMessage(gossipMessage{Type: "ack", SenderID: "unknown"}, nil)
		if len(g.members) != 0 || len(ring.Nodes()) != 0 {
			t.Fatal("unknown ACK admitted a member")
		}
	})
}
