package promalert

import (
	"strings"
	"testing"
)

// a real Alertmanager v4 webhook body for the FakGatewayDown rule in
// tools/grafana/prometheus-alerts.yml — the exact shape Alertmanager POSTs.
const firingPayload = `{
  "version": "4",
  "groupKey": "{}:{alertname=\"FakGatewayDown\"}",
  "truncatedAlerts": 0,
  "status": "firing",
  "receiver": "fak-webhook",
  "groupLabels": {"alertname": "FakGatewayDown"},
  "commonLabels": {"alertname": "FakGatewayDown", "severity": "critical", "job": "fak_gateway"},
  "commonAnnotations": {"summary": "fak gateway metrics are not scrapeable"},
  "externalURL": "http://alertmanager:9093",
  "alerts": [
    {
      "status": "firing",
      "labels": {"alertname": "FakGatewayDown", "severity": "critical", "job": "fak_gateway", "instance": "host.docker.internal:8080"},
      "annotations": {"summary": "fak gateway metrics are not scrapeable", "description": "Prometheus cannot scrape fak serve at :8080/metrics for 2m.\nStart fak serve."},
      "startsAt": "2026-07-09T12:00:00.000Z",
      "endsAt": "0001-01-01T00:00:00Z",
      "generatorURL": "http://prometheus:9091/graph?g0.expr=up",
      "fingerprint": "abc123"
    }
  ]
}`

func TestParseAndRenderFiring(t *testing.T) {
	w, err := ParseBytes([]byte(firingPayload))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if w.Status != "firing" || len(w.Alerts) != 1 {
		t.Fatalf("unexpected webhook: status=%q alerts=%d", w.Status, len(w.Alerts))
	}
	out := Render(w, RenderOpts{})

	wantContains := []string{
		"🔴",                                         // critical severity emoji
		"FIRING",                                    // status word
		"1 alert(s)",                                // count
		`alertname="FakGatewayDown"`,                // group signature
		"FakGatewayDown (critical)",                 // per-alert header
		"fak gateway metrics are not scrapeable",    // summary
		"Prometheus cannot scrape fak serve",        // description (collapsed)
		"instance=host.docker.internal:8080",        // distinguishing label
		"since 2026-07-09 12:00:00Z",                // start time
		"↳ http://prometheus:9091/graph?g0.expr=up", // generator link
	}
	for _, w := range wantContains {
		if !strings.Contains(out, w) {
			t.Errorf("render missing %q\n--- got ---\n%s", w, out)
		}
	}
	// The multi-line description must be collapsed onto one line (no embedded newline
	// inside the description text) so the card stays compact.
	if strings.Contains(out, "for 2m.\nStart") {
		t.Errorf("description not collapsed:\n%s", out)
	}
}

func TestRenderResolved(t *testing.T) {
	payload := strings.NewReplacer(
		`"status": "firing"`, `"status": "resolved"`,
		`"endsAt": "0001-01-01T00:00:00Z"`, `"endsAt": "2026-07-09T12:05:00.000Z"`,
	).Replace(firingPayload)
	w, err := ParseBytes([]byte(payload))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	out := Render(w, RenderOpts{})
	if !strings.Contains(out, "✅") || !strings.Contains(out, "RESOLVED") {
		t.Errorf("resolved render missing check/word:\n%s", out)
	}
	if !strings.Contains(out, "resolved 2026-07-09 12:05:00Z") {
		t.Errorf("resolved render missing resolve time:\n%s", out)
	}
}

func TestParseRejectsEmptyAndNoAlerts(t *testing.T) {
	if _, err := ParseBytes([]byte("")); err == nil {
		t.Error("empty payload should error")
	}
	if _, err := ParseBytes([]byte(`{"status":"firing","alerts":[]}`)); err == nil {
		t.Error("no-alerts payload should error")
	}
}

func TestRenderCapsAlerts(t *testing.T) {
	// Build a group of 15 alerts; the render must cap and note the remainder.
	var alerts []string
	for i := 0; i < 15; i++ {
		alerts = append(alerts, `{"status":"firing","labels":{"alertname":"FleetBottleneckHigh","severity":"warning","id":"x`+string(rune('a'+i))+`"},"annotations":{"summary":"s"}}`)
	}
	payload := `{"version":"4","status":"firing","groupLabels":{"alertname":"FleetBottleneckHigh"},"alerts":[` + strings.Join(alerts, ",") + `]}`
	w, err := ParseBytes([]byte(payload))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	out := Render(w, RenderOpts{MaxAlerts: 10})
	if !strings.Contains(out, "15 alert(s)") {
		t.Errorf("header should carry full count:\n%s", out)
	}
	if !strings.Contains(out, "and 5 more alert(s)") {
		t.Errorf("render should note the 5 hidden alerts:\n%s", out)
	}
}
