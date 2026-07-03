package repoguard

import (
	"strings"
	"testing"
)

func TestLiveMonitorOutputTaskID(t *testing.T) {
	cases := []struct {
		path string
		want string
		ok   bool
	}{
		{`C:\Users\u\AppData\Local\Claude\tasks\mon-1.output`, "mon-1", true},
		{`tasks\mon_2.output`, "mon_2", true},
		{`/tmp/tasks/mon.3.output`, "mon.3", true},
		{`/tmp/tasks/mon.output.bak`, "", false},
		{`/tmp/other/mon.output`, "", false},
		{`/tmp/tasks/.output`, "", false},
	}
	for _, tc := range cases {
		got, ok := LiveMonitorOutputTaskID(tc.path)
		if got != tc.want || ok != tc.ok {
			t.Errorf("LiveMonitorOutputTaskID(%q) = %q,%v want %q,%v", tc.path, got, ok, tc.want, tc.ok)
		}
	}
}

func TestLiveMonitorTaskIDsFromJournal(t *testing.T) {
	journal := strings.Join([]string{
		`{"kind":"spawn","call_id":"mon-live","session":"s1","tool":"Monitor","at_unix_ms":1}`,
		`{"kind":"spawn","call_id":"bg:bg-live","session":"s1","tool":"Monitor[bg]","at_unix_ms":2}`,
		`{"kind":"spawn","call_id":"mon-other-session","session":"s2","tool":"Monitor","at_unix_ms":3}`,
		`{"kind":"spawn","call_id":"mon-done","session":"s1","tool":"Monitor","at_unix_ms":4}`,
		`{"kind":"exit","call_id":"mon-done","at_unix_ms":5,"status":"ok"}`,
	}, "\n")
	ids, err := LiveMonitorTaskIDsFromJournal(strings.NewReader(journal), "s1")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"mon-live", "bg-live"} {
		if !ids[want] {
			t.Fatalf("live ids = %v, missing %q", ids, want)
		}
	}
	for _, absent := range []string{"mon-other-session", "mon-done"} {
		if ids[absent] {
			t.Fatalf("live ids = %v, must not include %q", ids, absent)
		}
	}
}

func TestLiveMonitorTaskIDsFromJournalDropsEndedSession(t *testing.T) {
	journal := strings.Join([]string{
		`{"kind":"spawn","call_id":"mon-live","session":"s1","tool":"Monitor","at_unix_ms":1}`,
		`{"kind":"session_end","session":"s1","at_unix_ms":2}`,
	}, "\n")
	ids, err := LiveMonitorTaskIDsFromJournal(strings.NewReader(journal), "s1")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 0 {
		t.Fatalf("live ids after session_end = %v, want empty", ids)
	}
}

func TestEvaluateReadLiveMonitorOutputDenied(t *testing.T) {
	ids := map[string]bool{"mon-live": true}
	v := EvaluateWithLiveMonitorIDs("Read",
		map[string]any{"file_path": `C:\Users\u\AppData\Local\Claude\tasks\mon-live.output`},
		wsTest, safeTest, ids)
	if len(v) != 1 || v[0].Reason != ReasonLiveMonitorOutputRead {
		t.Fatalf("EvaluateWithLiveMonitorIDs(Read) = %+v, want one %s", v, ReasonLiveMonitorOutputRead)
	}
	reason := RenderReason(v)
	if !strings.Contains(reason, ReasonLiveMonitorOutputRead) ||
		!strings.Contains(reason, "live Monitor events are pushed") ||
		!strings.Contains(reason, "appears only after the stream ends") {
		t.Fatalf("RenderReason = %q, want typed live Monitor hint", reason)
	}
}

func TestEvaluateReadMonitorOutputAllowedWhenNotLive(t *testing.T) {
	v := EvaluateWithLiveMonitorIDs("Read",
		map[string]any{"file_path": `C:\Users\u\AppData\Local\Claude\tasks\mon-done.output`},
		wsTest, safeTest, map[string]bool{"other": true})
	if len(v) != 0 {
		t.Fatalf("non-live monitor output read got violations: %+v", v)
	}
}
