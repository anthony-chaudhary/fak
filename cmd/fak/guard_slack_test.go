package main

import (
	"strings"
	"testing"
	"time"
)

func TestGuardSessionAgentNameNormalizesLauncher(t *testing.T) {
	cases := map[string]string{
		`C:\Program Files\Claude\claude.exe`: "claude.exe",
		"/usr/local/bin/codex":               "codex",
		"opencode":                           "opencode",
		"":                                   "unknown",
	}
	for in, want := range cases {
		if got := guardSessionAgentName([]string{in}); got != want {
			t.Fatalf("guardSessionAgentName(%q) = %q, want %q", in, got, want)
		}
	}
	if got := guardSessionAgentName(nil); got != "unknown" {
		t.Fatalf("guardSessionAgentName(nil) = %q, want unknown", got)
	}
}

func TestGuardSessionThreadRowIsRootPost(t *testing.T) {
	row := guardSessionThreadRow(guardSessionThreadMeta{
		Channel:   "CCHAN",
		Nonce:     "nonce-1",
		TraceID:   "trace one\ntrace two",
		Agent:     "claude",
		Provider:  "anthropic",
		AuditPath: "audit.jsonl",
		StartedAt: time.Unix(1000, 0),
		PID:       123,
	})
	if row.Channel != "CCHAN" || row.Nonce != "nonce-1" || row.Source != guardSessionThreadSource {
		t.Fatalf("row identity = %+v", row)
	}
	if row.ThreadTS != "" || row.UpdateTS != "" {
		t.Fatalf("guard session row must be a root post, got thread_ts=%q update_ts=%q", row.ThreadTS, row.UpdateTS)
	}
	for _, want := range []string{
		"fak guard session started",
		"session_thread_id: nonce-1",
		"trace_id: trace one trace two",
		"agent: claude",
		"provider: anthropic",
		"started_utc: 1970-01-01T00:16:40Z",
		"audit: audit.jsonl",
	} {
		if !strings.Contains(row.Text, want) {
			t.Fatalf("row text missing %q:\n%s", want, row.Text)
		}
	}
}

func TestEnqueueGuardSessionThreadDefaultsToDedicatedChannel(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)

	row, err := enqueueGuardSessionThread("trace-a", "anthropic", []string{"claude"}, "audit.jsonl", time.Unix(2000, 0))
	if err != nil {
		t.Fatalf("enqueueGuardSessionThread: %v", err)
	}
	if row.Channel != guardSessionsChannelDefault {
		t.Fatalf("channel = %q, want %q", row.Channel, guardSessionsChannelDefault)
	}
	if row.ThreadTS != "" || row.UpdateTS != "" {
		t.Fatalf("enqueued guard session must create a root thread, got thread_ts=%q update_ts=%q", row.ThreadTS, row.UpdateTS)
	}

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 1 {
		t.Fatalf("spool rows = %d, want 1", len(snap.Rows))
	}
	got := snap.Rows[0]
	if got.Channel != guardSessionsChannelDefault || got.Source != guardSessionThreadSource || got.ThreadTS != "" {
		t.Fatalf("spooled row = %+v", got)
	}
	if !strings.Contains(got.Text, "trace_id: trace-a") || !strings.Contains(got.Text, "agent: claude") {
		t.Fatalf("spooled row text missing guard identity:\n%s", got.Text)
	}
}
