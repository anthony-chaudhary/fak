package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

// amFiringPayload is a real Alertmanager v4 webhook body for the FakGatewayDown rule.
const amFiringPayload = `{
  "version": "4",
  "status": "firing",
  "receiver": "fak-webhook",
  "groupLabels": {"alertname": "FakGatewayDown"},
  "commonLabels": {"alertname": "FakGatewayDown", "severity": "critical", "job": "fak_gateway"},
  "commonAnnotations": {"summary": "fak gateway metrics are not scrapeable"},
  "externalURL": "http://alertmanager:9093",
  "alerts": [
    {"status": "firing",
     "labels": {"alertname": "FakGatewayDown", "severity": "critical", "job": "fak_gateway", "instance": "h:8080"},
     "annotations": {"summary": "fak gateway metrics are not scrapeable", "description": "down for 2m"},
     "startsAt": "2026-07-09T12:00:00Z",
     "generatorURL": "http://prometheus:9091/graph"}
  ]
}`

// TestSlackAlertOneShotEnqueuesAndPosts drives the file→render→enqueue→drain→post spine and
// witnesses the delivery both at the Slack wire (a real chat.postMessage reached it) and in
// the durable outbox fold (one posted row, zero dead).
func TestSlackAlertOneShotEnqueuesAndPosts(t *testing.T) {
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	f := filepath.Join(t.TempDir(), "firing.json")
	if err := os.WriteFile(f, []byte(amFiringPayload), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runSlackAlert(&out, &errb, []string{
		"--file", f, "--channel", "C_ALERTS",
		"--token", "xoxb-test", "--api-base", srv.URL + "/",
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if posts != 1 {
		t.Fatalf("posts=%d, want 1 (the alert card should have reached the Slack wire)", posts)
	}
	if !strings.Contains(out.String(), "enqueued 1 alert(s) durably") {
		t.Fatalf("missing enqueue line:\n%s", out.String())
	}

	ob, err := slackoutbox.Open(resolveOutboxDir())
	if err != nil {
		t.Fatal(err)
	}
	st, err := ob.Status(time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if st.Posted != 1 || st.Dead != 0 || st.Pending != 0 {
		t.Fatalf("outbox fold: posted=%d dead=%d pending=%d, want 1/0/0", st.Posted, st.Dead, st.Pending)
	}
}

// TestSlackAlertDryRunTouchesNothing confirms --dry-run renders but never writes a spool row.
func TestSlackAlertDryRunTouchesNothing(t *testing.T) {
	outboxTestDir(t)
	f := filepath.Join(t.TempDir(), "firing.json")
	if err := os.WriteFile(f, []byte(amFiringPayload), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	rc := runSlackAlert(&out, &errb, []string{"--dry-run", "--file", f, "--channel", "C_ALERTS"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "🔴 FIRING") {
		t.Fatalf("dry-run did not render the card:\n%s", out.String())
	}
	ob, _ := slackoutbox.Open(resolveOutboxDir())
	st, _ := ob.Status(time.Now())
	if st.Pending != 0 || st.Posted != 0 {
		t.Fatalf("dry-run wrote a row: pending=%d posted=%d", st.Pending, st.Posted)
	}
}

// TestSlackAlertServeHandler drives the exact HTTP handler Alertmanager POSTs to and confirms
// it answers 200 and lands one durable row.
func TestSlackAlertServeHandler(t *testing.T) {
	outboxTestDir(t)
	enqueued := make(chan struct{}, 1)
	mux := newAlertServeMux(&bytes.Buffer{}, "C_ALERTS", 0, func() { enqueued <- struct{}{} })
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// A malformed body is a client error, not a retry-forever 5xx.
	badResp, err := http.Post(ts.URL+"/alerts", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatal(err)
	}
	badResp.Body.Close()
	if badResp.StatusCode != http.StatusBadRequest {
		t.Fatalf("malformed POST status=%d, want 400", badResp.StatusCode)
	}

	resp, err := http.Post(ts.URL+"/alerts", "application/json", strings.NewReader(amFiringPayload))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST status=%d, want 200", resp.StatusCode)
	}
	select {
	case <-enqueued:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not enqueue")
	}

	ob, _ := slackoutbox.Open(resolveOutboxDir())
	st, _ := ob.Status(time.Now())
	if st.Pending != 1 {
		t.Fatalf("serve handler: pending=%d, want 1 durable row awaiting drain", st.Pending)
	}
}
