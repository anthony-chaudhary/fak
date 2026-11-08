package gateway

// SessionLane (#fak-private #3707/#3708/#3709): the launch-time lane designation is
// stamped as trusted Meta["lane"] on every served tool call, so lane-aware floors
// (test immunity) resolve the acting lane the coordinator designated at spawn. The
// agent cannot set or spoof it — Meta is host-built, never wire input.

import (
	"context"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestNewWiresSessionLaneIntoCallMeta(t *testing.T) {
	// Metadata assembly needs no inference; select the package-registered mock explicitly.
	srv, err := New(Config{EngineID: "mock", SessionLane: "  test  "})
	if err != nil {
		t.Fatalf("New(Config{SessionLane}): %v", err)
	}
	t.Cleanup(srv.Close)
	if srv.sessionLane != "test" {
		t.Fatalf("New must trim the configured session lane, got %q", srv.sessionLane)
	}
	tc, err := srv.buildCall(context.Background(), "get_doc", `{"id":"x"}`, true, "", "trace-lane")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Meta["lane"] != "test" {
		t.Fatalf("session lane not threaded into call meta: %q", tc.Meta["lane"])
	}
	// A write-shaped call carries the same designation (the test-immunity rung
	// reads exactly this key).
	w, err := srv.buildCall(context.Background(), "write_file", `{"path":"x_test.go","content":"package x"}`, false, "", "trace-lane")
	if err != nil {
		t.Fatalf("buildCall(write): %v", err)
	}
	if w.Meta["lane"] != "test" {
		t.Fatalf("session lane missing on write-shaped call meta: %q", w.Meta["lane"])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBuildCallNoSessionLaneKeepsMetaUnset(t *testing.T) {
	srv, err := New(Config{EngineID: "mock"})
	if err != nil {
		t.Fatalf("New(Config{EngineID: mock}): %v", err)
	}
	t.Cleanup(srv.Close)
	if srv.sessionLane != "" {
		t.Fatalf("unset SessionLane must default to an unset session lane, got %q", srv.sessionLane)
	}
	tc, err := srv.buildCall(context.Background(), "get_doc", `{"id":"x"}`, true, "", "trace-nolane")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if _, ok := tc.Meta["lane"]; ok {
		t.Fatalf("empty SessionLane must not inject a lane meta key (historical behavior)")
	}
}

func TestSessionLaneConfigFieldShape(t *testing.T) {
	cfg := Config{SessionLane: "  test  "}
	if got := strings.TrimSpace(cfg.SessionLane); got != "test" {
		t.Fatalf("unexpected trim result: %q", got)
	}
}
