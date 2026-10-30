package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncFetchDepthPins(t *testing.T) {
	old := syncRouteReconciliation
	defer func() { syncRouteReconciliation = old }()
	base, target, after := strings.Repeat("a", 40), strings.Repeat("b", 40), strings.Repeat("c", 40)
	called := false
	syncRouteReconciliation = func(_ context.Context, o safesync.ReconcileOptions) (safesync.ReconcileAssessment, error) {
		called = true
		if !o.Fetch || o.FetchDepth != 30 || o.FetchBase != base || o.FetchTarget != target || o.FetchAfterBase != after {
			t.Fatalf("lost fetch pins: %+v", o)
		}
		return safesync.ReconcileAssessment{OK: true, Route: safesync.RouteNoop}, nil
	}
	var out, errb bytes.Buffer
	if code := runSync(&out, &errb, []string{"reconcile", "--fetch", "--fetch-depth", "30", "--fetch-base", base, "--fetch-target", target, "--fetch-after-base", after, "--json"}); code != syncExitOK || !called {
		t.Fatalf("code=%d called=%v stderr=%s", code, called, errb.String())
	}
	for _, args := range [][]string{
		{"check", "--fetch-after-base", after},
		{"check", "--fetch", "--fetch-after-base", after},
		{"reconcile", "--fetch-depth", "30", "--fetch-base", base, "--fetch-target", target},
		{"reconcile", "--fetch", "--fetch-depth", "30"},
		{"packet", "--fetch", "--fetch-depth", "30", "--fetch-base", base, "--fetch-target", target},
	} {
		called = false
		if code := runSync(&out, &errb, args); code != syncExitUsage || called {
			t.Fatalf("unsupported pins accepted: %q code=%d", args, code)
		}
	}
}
