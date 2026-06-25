package main

// steer_enqueue_test.go - proves the server-side half of #760: steerSession enqueues an
// operator steer onto the a2achan Session-locale bus (keyed by trace), deliverable to the
// running loop, and rides the fail-closed floor (the Shared cross-principal body is admitted;
// the receiver screens it on ingress). The floor's deny paths themselves are covered by
// internal/a2achan; this proves the cmd-side closure wires onto it correctly.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestSteerSessionEnqueuesToSessionBus(t *testing.T) {
	ctx := context.Background()
	const trace = "steer-enqueue-1"

	if err := steerSession(ctx, trace, "switch to plan B"); err != nil {
		t.Fatalf("steerSession returned error: %v", err)
	}

	// The running loop drains the Session-locale mailbox keyed by its trace.
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
	msg, v, ok := a2achan.Default.TryRecv(ctx, key, a2achan.CapA2ARecv)
	if !ok {
		t.Fatal("steer was not enqueued onto the session bus")
	}
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("received steer verdict = %v, want Allow (a clean operator body is admitted)", v.Kind)
	}
	if string(msg.Body.Inline) != "switch to plan B" {
		t.Fatalf("received steer body = %q, want 'switch to plan B'", msg.Body.Inline)
	}
	if msg.From != "operator" {
		t.Fatalf("steer From = %q, want 'operator'", msg.From)
	}
}

func TestSteerSessionRejectsEmptyTrace(t *testing.T) {
	if err := steerSession(context.Background(), "   ", "x"); err == nil {
		t.Fatal("steerSession accepted an empty trace id, want an error")
	}
}
