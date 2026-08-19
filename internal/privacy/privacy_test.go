package privacy

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestPrivacyPolicyLocalRedactedExportAndTelemetryOptOut(t *testing.T) {
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	p := Policy{Schema: PolicySchema, Log: SinkPolicy{Enabled: true}, Retain: RetentionPolicy{Enabled: true, MaxCount: 2, MaxAgeSeconds: 60}, Export: SinkPolicy{Enabled: true, RedactFields: []string{"secret", "token"}}, Telemetry: SinkPolicy{Enabled: false}}
	payload := json.RawMessage(`{"event":"turn","secret":"s3","nested":{"token":"t4"}}`)
	local, err := p.Evaluate(SinkLog, payload, now)
	if err != nil || !strings.Contains(string(local.Payload), "s3") {
		t.Fatalf("local=%s err=%v", local.Payload, err)
	}
	exported, err := p.Evaluate(SinkExport, payload, now)
	if err != nil || strings.Contains(string(exported.Payload), "s3") || strings.Contains(string(exported.Payload), "t4") || exported.Receipt.Action != ActionRedact || len(exported.Receipt.RedactedFields) != 2 {
		t.Fatalf("export=%+v err=%v", exported, err)
	}
	telemetry, err := p.Evaluate(SinkTelemetry, payload, now)
	if err != nil || telemetry.Receipt.Action != ActionDeny || len(telemetry.Payload) != 0 {
		t.Fatalf("telemetry=%+v", telemetry)
	}
}
func TestDefaultPrivacyPolicySendsNoTelemetry(t *testing.T) {
	d, err := DefaultPolicy().Evaluate(SinkTelemetry, json.RawMessage(`{"secret":"never"}`), time.Now())
	if err != nil || d.Receipt.Action != ActionDeny || len(d.Payload) != 0 {
		t.Fatalf("decision=%+v err=%v", d, err)
	}
}
func TestPrivacyRetentionCountAndExpiration(t *testing.T) {
	p := DefaultPolicy()
	p.Retain.MaxCount = 2
	p.Retain.MaxAgeSeconds = 10
	s, err := NewStore(p)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	for i := 0; i < 3; i++ {
		_, _ = s.Append(json.RawMessage(`{"x":1}`), now.Add(time.Duration(i)*time.Second))
	}
	if len(s.Rows(now.Add(2*time.Second))) != 2 {
		t.Fatal("count bound failed")
	}
	if len(s.Rows(now.Add(20*time.Second))) != 0 {
		t.Fatal("expiration failed")
	}
}
func TestPrivacyMalformedFailsClosed(t *testing.T) {
	bad := []string{`{}`, `{"schema":"wrong"}`, `{"schema":"fak-privacy-policy/1","unknown":true}`, `{"schema":"fak-privacy-policy/1","retain":{"enabled":true,"max_count":0,"max_age_seconds":0}}`}
	for _, raw := range bad {
		if _, err := ParsePolicy([]byte(raw)); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
}
