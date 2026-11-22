package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/macfit"
)

// fak-test:runtime fast est=50ms lane=default
func TestStartTurnkeyServerWiresToolAdjudicator(t *testing.T) {
	plan, err := macfit.ConfigureTurnkey(36 * macfit.GiB)
	if err != nil {
		t.Fatal(err)
	}
	server, err := startTurnkeyServer(context.Background(), plan, "127.0.0.1:0", true)
	if err != nil {
		t.Fatal(err)
	}
	if server.adjudicator == nil {
		t.Fatal("startTurnkeyServer left proposed tool calls without a serve-floor adjudicator")
	}
	if got, err := server.toolAdjudicator(); err != nil || got != server.adjudicator {
		t.Fatalf("toolAdjudicator() = %p, %v; want the server's own %p", got, err, server.adjudicator)
	}
	if err := server.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
}
