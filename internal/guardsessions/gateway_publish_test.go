package guardsessions

// gateway_publish_test.go — the READER-side half of #5400: a row that carries no
// gateway_url has TWO different causes, and PredatesGatewayPublish is what lets a consumer
// (`fak session status`) name the right one instead of guessing "recorded by an older fak?"
// at every row forever.

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestPredatesGatewayPublishSplitsLegacyRowsFromUnpublishedOnes(t *testing.T) {
	legacy := NewRow("trace-old", "claude", 1, "/w", "", "", GatewayPublishEpoch.Add(-24*time.Hour))
	if !legacy.PredatesGatewayPublish() {
		t.Fatalf("row started %s (before the epoch %s) must read as predating the publish",
			legacy.StartedAt, GatewayPublishEpoch.Format(time.RFC3339))
	}
	current := NewRow("trace-new", "claude", 1, "/w", "", "", GatewayPublishEpoch.Add(time.Hour))
	if current.PredatesGatewayPublish() {
		t.Fatalf("row started %s (after the epoch) must NOT be blamed on an older build", current.StartedAt)
	}
	// A row stamped at the epoch instant itself is a publishing build (the boundary is
	// exclusive), so it is not excused as legacy.
	at := NewRow("trace-at", "claude", 1, "/w", "", "", GatewayPublishEpoch)
	if at.PredatesGatewayPublish() {
		t.Fatal("a row started exactly at the epoch must not read as predating it")
	}
	// An unstamped/garbled start is legacy shape: it cannot be attributed to a live
	// session's failed publish, so it takes the version explanation, not the fault one.
	if !(Row{StartedAt: ""}).PredatesGatewayPublish() {
		t.Fatal("a row with no start time must read as predating the publish")
	}
	if !(Row{StartedAt: "not-a-time"}).PredatesGatewayPublish() {
		t.Fatal("a row with an unparseable start time must read as predating the publish")
	}
}

// A recorded row that never bound a gateway must stay a LEGAL shape on the wire: both
// fields absent (omitempty), not empty strings — that absence is what every consumer reads
// as "this session published nothing", and it must not become an error or a phantom key.
func TestRowWithoutGatewayOmitsBothFieldsOnTheWire(t *testing.T) {
	b, err := json.Marshal(NewRow("trace-plain", "claude", 7, "/w", "audit.jsonl", "n", time.Now()))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "gateway_url") || strings.Contains(string(b), "bearer") {
		t.Fatalf("unstamped row leaked a gateway key: %s", b)
	}
	stamped, err := json.Marshal(NewRow("trace-stamped", "claude", 7, "/w", "audit.jsonl", "n", time.Now()).
		WithGateway("http://127.0.0.1:5051", "fakread-abc"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(stamped), `"gateway_url":"http://127.0.0.1:5051"`) ||
		!strings.Contains(string(stamped), `"bearer":"fakread-abc"`) {
		t.Fatalf("stamped row did not publish both fields: %s", stamped)
	}
}

// The producer re-records the SAME handle once its gateway is serving. The fold must treat
// that as a supersede (one session, now reachable), never as a second session — otherwise
// every published guard would double-count in `fak session ls` / `fak cachevalue census`.
func TestRepublishedRowSupersedesTheLaunchRowUnderTheSameHandle(t *testing.T) {
	dir := t.TempDir()
	launch := NewRow("trace-pub", "claude", 42, "/w", "audit.jsonl", "n", time.Unix(1700000000, 0))
	if err := Record(dir, launch); err != nil {
		t.Fatal(err)
	}
	if err := Record(dir, launch.WithGateway("http://127.0.0.1:6060", "fakread-xyz")); err != nil {
		t.Fatal(err)
	}
	rows := Load(dir)
	if len(rows) != 1 {
		t.Fatalf("republish folded to %d rows, want 1 (same handle = one session)", len(rows))
	}
	if rows[0].Handle != launch.Handle {
		t.Fatalf("handle changed across the republish: %q -> %q", launch.Handle, rows[0].Handle)
	}
	if rows[0].GatewayURL != "http://127.0.0.1:6060" || rows[0].Bearer != "fakread-xyz" {
		t.Fatalf("folded row lost the published gateway: %+v", rows[0])
	}
	// Provenance from the launch row survives the supersede — the republish carries the
	// whole row, not a gateway-only patch.
	if rows[0].Agent != "claude" || rows[0].PID != 42 || rows[0].AuditPath != "audit.jsonl" {
		t.Fatalf("republished row dropped launch provenance: %+v", rows[0])
	}
}
