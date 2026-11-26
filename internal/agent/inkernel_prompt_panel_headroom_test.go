package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestPromptPanelHeadroomGrantRoundTrip pins the request-context grant: a positive grant
// reaches every configured session, an OOM retry revokes it for later configures, a reused
// session never keeps a previous request's grant, and no grant leaves the session at 0.
func TestPromptPanelHeadroomGrantRoundTrip(t *testing.T) {
	s := &model.Session{}
	applyPromptPanelHeadroom(context.Background(), s)
	if s.PromptPanelHeadroomBytes != 0 {
		t.Fatalf("no grant: headroom = %d, want 0", s.PromptPanelHeadroomBytes)
	}
	if ctx := withPromptPanelHeadroom(context.Background(), 0); promptPanelHeadroomBytes(ctx) != 0 {
		t.Fatal("zero grant attached a value")
	}

	ctx := withPromptPanelHeadroom(context.Background(), 1<<20)
	applyPromptPanelHeadroom(ctx, s)
	if s.PromptPanelHeadroomBytes != 1<<20 {
		t.Fatalf("granted headroom = %d, want %d", s.PromptPanelHeadroomBytes, 1<<20)
	}
	child, cancel := context.WithCancel(ctx)
	defer cancel()
	revokePromptPanelHeadroom(child)
	applyPromptPanelHeadroom(ctx, s)
	if s.PromptPanelHeadroomBytes != 0 {
		t.Fatalf("revoked headroom = %d, want 0", s.PromptPanelHeadroomBytes)
	}

	s.PromptPanelHeadroomBytes = 1 << 20
	applyPromptPanelHeadroom(context.Background(), s)
	if s.PromptPanelHeadroomBytes != 0 {
		t.Fatalf("reused session kept headroom %d, want 0", s.PromptPanelHeadroomBytes)
	}
	var nilPlanner *InKernelPlanner
	if got := nilPlanner.promptPanelHeadroomGrantBytes(4096, 16); got != 0 {
		t.Fatalf("nil planner grant = %d, want 0", got)
	}
}
