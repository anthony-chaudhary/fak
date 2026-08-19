package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentquery"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func TestAgentsUnionExactIdentityAndRestartHistory(t *testing.T) {
	live := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/fak/sessions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"count":2,"sessions":[{"trace_id":"shared","run":"running","time":{"elapsed_seconds":60}},{"trace_id":"live-long","run":"running","parent_trace":"shared","time":{"elapsed_seconds":300}}]}`))
	}))
	journal := filepath.Join(t.TempDir(), "sessions.jsonl")
	events := []sessionjournal.Event{
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "shared", TS: "2026-08-18T00:01:00Z", Host: "h1", Boot: "boot-a"},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "closed", TS: "2026-08-18T00:00:00Z", Host: "h2", Boot: "boot-a", Registration: &sessionjournal.RegistrationCarry{Lane: "cmd", TaskID: "group-a"}},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindClose, ID: "closed", TS: "2026-08-18T00:02:00Z", Reason: "done"},
	}
	var data bytes.Buffer
	enc := json.NewEncoder(&data)
	for _, e := range events {
		if err := enc.Encode(e); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(journal, data.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errout bytes.Buffer
	code := runAgents(&out, &errout, []string{"--addr", live.URL, "--journal", journal, "--source", "union", "--all", "--json", "--now", "1787011800"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var got agentquery.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Metadata.Schema != agentquery.Schema || got.Metadata.Deduplicated != 1 || len(got.Rows) != 3 {
		t.Fatalf("got=%+v", got)
	}
	if got.Rows[0].AgentID != "live-long" || got.Rows[0].ParentID == nil {
		t.Fatalf("ordering/lineage=%+v", got.Rows)
	}
	for _, r := range got.Rows {
		if r.AgentID == "shared" && r.Source != "live" {
			t.Fatalf("live did not win: %+v", r)
		}
	}
	live.Close()
	out.Reset()
	errout.Reset()
	code = runAgents(&out, &errout, []string{"--journal", journal, "--source", "history", "--all", "--json", "--now", "1787011800"})
	if code != 0 {
		t.Fatalf("restart code=%d stderr=%s", code, errout.String())
	}
	if !strings.Contains(out.String(), `"agent_id": "closed"`) || !strings.Contains(out.String(), `"source": "history"`) {
		t.Fatalf("restart history=%s", out.String())
	}
}

func TestAgentsDefaultActiveLongestFirst(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"count":3,"sessions":[{"trace_id":"short","run":"running","time":{"elapsed_seconds":5}},{"trace_id":"closed","run":"stopped","time":{"elapsed_seconds":500}},{"trace_id":"long","run":"running","time":{"elapsed_seconds":50}}]}`))
	}))
	defer srv.Close()
	var out, errout bytes.Buffer
	if code := runAgents(&out, &errout, []string{"--addr", srv.URL, "--now", "1787011800"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	text := out.String()
	if strings.Contains(text, "closed") || strings.Index(text, "long") > strings.Index(text, "short") {
		t.Fatalf("render=%s", text)
	}
}

func TestAgentsRejectsUnboundedLimit(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runAgents(&out, &errout, []string{"--limit", "10001"}); code != 2 || !strings.Contains(errout.String(), "between 1 and 10000") {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
}

func TestAgentsHistoryTornTailIsExplicitlyDegraded(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "sessions.jsonl")
	valid := `{"schema":"fak.sessionjournal.v1","kind":"open","id":"survivor","ts":"2026-08-18T00:00:00Z"}`
	if err := os.WriteFile(journal, []byte(valid+"\n"+`{"schema":"fak.sessionjournal.v1","kind":`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errout bytes.Buffer
	code := runAgents(&out, &errout, []string{"--journal", journal, "--source", "history", "--all", "--json", "--now", "1787011800"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var got agentquery.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Rows) != 1 || got.Rows[0].AgentID != "survivor" || got.Metadata.History == nil || got.Metadata.History.Status != "degraded" || got.Metadata.History.AcceptedRows != 1 || got.Metadata.History.MalformedRows != 1 {
		t.Fatalf("got=%+v", got)
	}
	if strings.Contains(out.String(), `"kind":`) {
		t.Fatalf("source content leaked into health output: %s", out.String())
	}

	out.Reset()
	errout.Reset()
	code = runAgents(&out, &errout, []string{"--journal", journal, "--source", "history", "--all", "--now", "1787011800"})
	if code != 0 || !strings.Contains(out.String(), "history status=degraded accepted=1 rejected=1") {
		t.Fatalf("code=%d stderr=%s out=%s", code, errout.String(), out.String())
	}
}

func TestAgentsExplicitUnreadableHistoryFailsClosed(t *testing.T) {
	var out, errout bytes.Buffer
	missing := filepath.Join(t.TempDir(), "private-history-name.jsonl")
	code := runAgents(&out, &errout, []string{"--journal", missing, "--source", "history", "--json"})
	if code != 1 || !strings.Contains(errout.String(), "history source not_found") || strings.Contains(errout.String(), "private-history-name") || out.Len() != 0 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), errout.String())
	}
}
