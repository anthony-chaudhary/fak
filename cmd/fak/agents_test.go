package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

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

func TestAgentsGroupedSevenDayLaneStateQuery(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "sessions.jsonl")
	events := []sessionjournal.Event{
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "cmd-a", TS: "2026-08-17T00:00:00Z", Registration: &sessionjournal.RegistrationCarry{Lane: "cmd"}},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindClose, ID: "cmd-a", TS: "2026-08-17T00:10:00Z", Reason: "done"},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "cmd-b", TS: "2026-08-16T00:00:00Z", Registration: &sessionjournal.RegistrationCarry{Lane: "cmd"}},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindClose, ID: "cmd-b", TS: "2026-08-16T00:20:00Z", Reason: "done"},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "docs-a", TS: "2026-08-15T00:00:00Z", Registration: &sessionjournal.RegistrationCarry{Lane: "docs"}},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "unknown", TS: "2026-08-14T00:00:00Z"},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "old", TS: "2026-08-10T23:59:59Z", Registration: &sessionjournal.RegistrationCarry{Lane: "old"}},
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
	args := []string{"--journal", journal, "--history", "7d", "--group-by", "lane,state", "--count", "--json", "--now", "1787011200"}
	var out, errout bytes.Buffer
	if code := runAgents(&out, &errout, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	var got agentquery.GroupResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Metadata.Schema != agentquery.GroupSchema || got.Metadata.MatchedRows != 4 || len(got.Rows) != 3 {
		t.Fatalf("got=%+v", got)
	}
	if got.Rows[0].Lane == nil || *got.Rows[0].Lane != "cmd" || got.Rows[0].Count != 2 || got.Rows[0].MaxElapsedMS == nil || *got.Rows[0].MaxElapsedMS != 1200000 {
		t.Fatalf("first=%+v", got.Rows[0])
	}
	if got.Rows[2].Lane != nil {
		t.Fatalf("unknown lane not last/null: %+v", got.Rows)
	}
	flagResult := got
	query := "SELECT lane, state, count(*) AS agents, max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC"
	out.Reset()
	errout.Reset()
	if code := runAgents(&out, &errout, []string{"--journal", journal, "--query", query, "--json", "--now", "1787011200"}); code != 0 {
		t.Fatalf("query code=%d stderr=%s", code, errout.String())
	}
	var queryResult agentquery.GroupResult
	if err := json.Unmarshal(out.Bytes(), &queryResult); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(flagResult, queryResult) {
		t.Fatalf("flag/query mismatch\nflag=%+v\nquery=%+v", flagResult, queryResult)
	}
	out.Reset()
	errout.Reset()
	args[7] = "false" // invalid positional-like count value must fail closed
	if code := runAgents(&out, &errout, args); code != 2 {
		t.Fatalf("invalid grouped contract code=%d out=%s err=%s", code, out.String(), errout.String())
	}
}

func TestParseAgentHistoryWindowDays(t *testing.T) {
	d, err := parseAgentHistoryWindow("7d")
	if err != nil || d != 7*24*time.Hour {
		t.Fatalf("d=%v err=%v", d, err)
	}
	if _, err := parseAgentHistoryWindow("0d"); err == nil {
		t.Fatal("expected invalid window")
	}
}

func TestAgentsQueryTextFailsClosed(t *testing.T) {
	cases := []string{"DELETE FROM agents", "SELECT * FROM agents", "SELECT lane,state,count(*) AS agents,max(cost) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC"}
	for _, q := range cases {
		var out, errout bytes.Buffer
		code := runAgents(&out, &errout, []string{"--query", q, "--json"})
		if code != 2 || !strings.Contains(errout.String(), "query rejected") || out.Len() != 0 {
			t.Errorf("q=%q code=%d stdout=%s stderr=%s", q, code, out.String(), errout.String())
		}
	}
}

func TestAgentsAsOfReconstructsPreAndPostClose(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "sessions.jsonl")
	events := []sessionjournal.Event{{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "epoch-a", TS: "2026-08-18T01:00:00Z", Boot: "boot-a"}, {Schema: sessionjournal.Schema, Kind: sessionjournal.KindBeat, ID: "epoch-a", TS: "2026-08-18T01:59:00Z"}, {Schema: sessionjournal.Schema, Kind: sessionjournal.KindClose, ID: "epoch-a", TS: "2026-08-18T02:00:00Z", Reason: "done"}}
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
	run := func(at string) agentquery.Result {
		var out, errout bytes.Buffer
		code := runAgents(&out, &errout, []string{"--journal", journal, "--source", "history", "--as-of", at, "--json", "--now", "1787036400"})
		if code != 0 {
			t.Fatalf("at=%s code=%d stderr=%s", at, code, errout.String())
		}
		var got agentquery.Result
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	before := run("2026-08-18T00:59:59Z")
	if len(before.Rows) != 0 {
		t.Fatalf("before=%+v", before)
	}
	atOpen := run("2026-08-18T01:00:00Z")
	if len(atOpen.Rows) != 1 || atOpen.Rows[0].Liveness != "LIVE" || atOpen.Metadata.AsOf == nil || *atOpen.Metadata.AsOf != "2026-08-18T01:00:00Z" {
		t.Fatalf("at open=%+v", atOpen)
	}
	preClose := run("2026-08-18T01:59:59Z")
	if preClose.Rows[0].Liveness != "LIVE" {
		t.Fatalf("pre close=%+v", preClose)
	}
	atClose := run("2026-08-18T02:00:00Z")
	if atClose.Rows[0].Liveness != "CLOSED" || atClose.Rows[0].EndedAt == nil {
		t.Fatalf("at close=%+v", atClose)
	}
}

func TestAgentsAsOfRejectsInvalidSourceAndFuture(t *testing.T) {
	cases := [][]string{{"--as-of", "bad", "--source", "history"}, {"--as-of", "2026-08-19T00:00:00Z", "--source", "history", "--now", "1787036400"}, {"--as-of", "2026-08-18T00:00:00Z", "--source", "live", "--now", "1787036400"}}
	for _, args := range cases {
		var out, errout bytes.Buffer
		if code := runAgents(&out, &errout, args); code != 2 || out.Len() != 0 {
			t.Errorf("args=%v code=%d out=%s err=%s", args, code, out.String(), errout.String())
		}
	}
}

func TestAgentsListFiltersPlanAndUnknownSort(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "sessions.jsonl")
	events := []sessionjournal.Event{{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "wanted", TS: "2026-08-17T00:00:00Z", Host: "h1", Account: "alice", Model: "m", Agent: "provider", Registration: &sessionjournal.RegistrationCarry{Lane: "cmd", TaskID: "group", RootRegistrationID: "root", ParentRegistrationID: "parent"}}, {Schema: sessionjournal.Schema, Kind: sessionjournal.KindClose, ID: "wanted", TS: "2026-08-17T01:00:00Z", Reason: "done"}, {Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "other", TS: "2026-08-16T00:00:00Z", Host: "h2"}}
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
	args := []string{"--journal", journal, "--source", "history", "--all", "--state", "closed", "--owner", "alice", "--host", "h1", "--lane", "cmd", "--group", "group", "--model", "m", "--provider", "provider", "--root", "root", "--parent", "parent", "--started-after", "2026-08-17T00:00:00Z", "--started-before", "2026-08-17T00:00:00Z", "--order-by", "identity_desc", "--json", "--now", "1787036400"}
	var out, errout bytes.Buffer
	if code := runAgents(&out, &errout, args); code != 0 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	var got agentquery.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Rows) != 1 || got.Rows[0].AgentID != "wanted" || got.Metadata.ListPlan == nil || got.Metadata.ListPlan.Owner != "alice" {
		t.Fatalf("got=%+v", got)
	}
	out.Reset()
	errout.Reset()
	if code := runAgents(&out, &errout, []string{"--order-by", "unknown"}); code != 2 || !strings.Contains(errout.String(), "unsupported order-by") {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errout.String())
	}
}

func TestAgentsRejectsGroupedListFlagMix(t *testing.T) {
	var out, errout bytes.Buffer
	code := runAgents(&out, &errout, []string{"--history", "7d", "--group-by", "lane,state", "--count", "--lane", "cmd"})
	if code != 2 || !strings.Contains(errout.String(), "cannot be combined") {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errout.String())
	}
}

func TestAgentsSchemaDescriptorJSON(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runAgents(&out, &errout, []string{"--schema", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	var got agentquery.Descriptor
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != agentquery.DescriptorSchema || got.RelationSchema != agentquery.Schema || len(got.Fields) == 0 || got.MaxRows != 10000 {
		t.Fatalf("got=%+v", got)
	}
	out.Reset()
	errout.Reset()
	if code := runAgents(&out, &errout, []string{"--schema"}); code != 2 || !strings.Contains(errout.String(), agentquery.DescriptorSchema) {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errout.String())
	}
}

func TestAgentsFullAggregatesFlagsAndQueryEquivalent(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "sessions.jsonl")
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	rows := []string{
		`{"schema":"fak.sessionjournal.v1","kind":"open","id":"a","ts":"2026-08-19T11:59:59.990Z","registration":{"registration_id":"a","lane":"cmd"}}`,
		`{"schema":"fak.sessionjournal.v1","kind":"close","id":"a","ts":"2026-08-19T12:00:00Z"}`,
		`{"schema":"fak.sessionjournal.v1","kind":"open","id":"b","ts":"2026-08-19T11:59:59.980Z","registration":{"registration_id":"b","lane":"cmd"}}`,
		`{"schema":"fak.sessionjournal.v1","kind":"close","id":"b","ts":"2026-08-19T12:00:00Z"}`,
	}
	if err := os.WriteFile(journal, []byte(strings.Join(rows, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	flags := []string{"--journal", journal, "--history", "168h", "--group-by", "lane,state", "--count", "--all-aggregates", "--now", strconv.FormatInt(now.Unix(), 10), "--json"}
	query := "SELECT lane,state,count(*) AS agents,min(elapsed_ms) AS min_elapsed_ms,max(elapsed_ms) AS max_elapsed_ms,sum(elapsed_ms) AS sum_elapsed_ms,avg(elapsed_ms) AS avg_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC"
	queryArgs := []string{"--journal", journal, "--query", query, "--now", strconv.FormatInt(now.Unix(), 10), "--json"}
	var flagsOut, queryOut, errout bytes.Buffer
	if code := runAgents(&flagsOut, &errout, flags); code != 0 {
		t.Fatalf("flags code=%d err=%s", code, errout.String())
	}
	errout.Reset()
	if code := runAgents(&queryOut, &errout, queryArgs); code != 0 {
		t.Fatalf("query code=%d err=%s", code, errout.String())
	}
	var a, b agentquery.GroupResult
	if err := json.Unmarshal(flagsOut.Bytes(), &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(queryOut.Bytes(), &b); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("flags=%s query=%s", flagsOut.String(), queryOut.String())
	}
	if len(a.Rows) != 1 || a.Rows[0].SumElapsedMS == nil || *a.Rows[0].SumElapsedMS != 30 || a.Rows[0].AvgElapsedMS == nil || *a.Rows[0].AvgElapsedMS != 15 {
		t.Fatalf("result=%+v", a)
	}
}
