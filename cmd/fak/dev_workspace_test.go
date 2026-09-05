package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// The #3426 (epic #3256) contract for `fak dev workspace` — the first honest
// increment: it maps the local agentic-dev spine and folds the guard floor's
// durable decision journals into the allowed/blocked/quarantined/witnessed view.
// These tests pin the fold and the readout hermetically (no guard process): a
// temp repo with hand-written journal rows under .dispatch-runs/guard-audit is the
// exact on-disk shape `fak guard` leaves.

// writeGuardJournal writes JSONL rows (the ReadRows on-disk schema) to a
// per-process guard-audit journal under root, creating the dir as guard does.
func writeGuardJournal(t *testing.T, root, name string, rows []map[string]any) {
	t.Helper()
	dir := filepath.Join(root, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, r := range rows {
		line, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestClassifyRowFold(t *testing.T) {
	cases := []struct {
		kind, verdict, want string
	}{
		{"DECIDE", "ALLOW", "allowed"},
		{"DECIDE", "TRANSFORM", "allowed"},
		{"DECIDE", "WITNESS", "allowed"},
		{"DECIDE", "DENY", "blocked"},
		{"DECIDE", "QUARANTINE", "quarantined"},
		{"DENY", "", "blocked"},
		{"RESULT_DENY", "", "blocked"},
		{"QUARANTINE", "", "quarantined"},
		{"AGENT_SPAWN", "", ""},   // lifecycle, not an adjudication
		{"CAP_FAULT", "", ""},     // capability, not an adjudication
		{"VDSO_HIT", "ALLOW", ""}, // cache hit, not folded as a decision
	}
	for _, c := range cases {
		got := classifyRow(rowFor(c.kind, c.verdict))
		if got != c.want {
			t.Errorf("classifyRow(%s/%s) = %q, want %q", c.kind, c.verdict, got, c.want)
		}
	}
}

func rowFor(kind, verdict string) journal.Row { return journal.Row{Kind: kind, Verdict: verdict} }

func TestScanStreamViewFoldsAllJournals(t *testing.T) {
	root := t.TempDir()
	writeGuardJournal(t, root, "interactive-1-aaaa.jsonl", []map[string]any{
		{"seq": 1, "ts_unix_nano": 10, "kind": "DECIDE", "verdict": "ALLOW", "tool": "Read"},
		{"seq": 2, "ts_unix_nano": 20, "kind": "DENY", "tool": "Bash", "reason": "OFF_TRUNK"},
		{"seq": 3, "ts_unix_nano": 30, "kind": "AGENT_SPAWN", "tool": "agent-x"}, // not a decision
	})
	writeGuardJournal(t, root, "interactive-2-bbbb.jsonl", []map[string]any{
		{"seq": 1, "ts_unix_nano": 15, "kind": "QUARANTINE", "tool": "WebFetch"},
		{"seq": 2, "ts_unix_nano": 25, "kind": "DECIDE", "verdict": "DENY", "tool": "Write", "witness": "self-modify glob"},
	})
	st, err := scanStreamView(root, 5)
	if err != nil {
		t.Fatalf("scanStreamView: %v", err)
	}
	if st.Journals != 2 {
		t.Errorf("journals = %d, want 2", st.Journals)
	}
	if st.Rows != 5 {
		t.Errorf("rows = %d, want 5 (all rows incl. lifecycle)", st.Rows)
	}
	if st.Decisions != 4 {
		t.Errorf("decisions = %d, want 4 (lifecycle row excluded)", st.Decisions)
	}
	if st.Allowed != 1 || st.Blocked != 2 || st.Quarantined != 1 {
		t.Errorf("fold = allowed %d blocked %d quarantined %d, want 1/2/1", st.Allowed, st.Blocked, st.Quarantined)
	}
	if st.Witnessed != 1 {
		t.Errorf("witnessed = %d, want 1", st.Witnessed)
	}
	// The tail merges both chains by wall-clock (ts 10,15,20,25) and drops the
	// lifecycle row (ts 30), rendered oldest -> newest.
	if len(st.Last) != 4 {
		t.Fatalf("tail len = %d, want 4 adjudications", len(st.Last))
	}
	gotKinds := make([]string, len(st.Last))
	for i, r := range st.Last {
		gotKinds[i] = r.Kind
	}
	want := []string{"DECIDE", "QUARANTINE", "DENY", "DECIDE"}
	if strings.Join(gotKinds, ",") != strings.Join(want, ",") {
		t.Errorf("tail order = %v, want %v (wall-clock)", gotKinds, want)
	}
}

func TestScanStreamViewNoJournalsIsHonestZero(t *testing.T) {
	root := t.TempDir() // no .dispatch-runs at all
	st, err := scanStreamView(root, 5)
	if err != nil {
		t.Fatalf("scanStreamView on empty repo must not error: %v", err)
	}
	if st.Journals != 0 || st.Rows != 0 || st.Decisions != 0 {
		t.Errorf("empty repo must fold to zero, got %+v", st)
	}
	if !strings.HasSuffix(filepath.ToSlash(st.Dir), ".dispatch-runs/guard-audit") {
		t.Errorf("stream dir = %q, want the guard-audit path", st.Dir)
	}
}

func TestRunDevWorkspaceText(t *testing.T) {
	root := t.TempDir()
	writeGuardJournal(t, root, "interactive-1-aaaa.jsonl", []map[string]any{
		{"seq": 1, "ts_unix_nano": 10, "kind": "DECIDE", "verdict": "ALLOW", "tool": "Read"},
		{"seq": 2, "ts_unix_nano": 20, "kind": "DENY", "tool": "Bash", "reason": "OFF_TRUNK"},
	})
	var out, errb strings.Builder
	code := runDevWorkspace(&out, &errb, []string{"--repo", root})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr: %s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{
		"local agentic-dev workspace",
		"#3426",
		"fak guard -- claude", // the spine map is present
		"fak goal sync push",  // goal sync component present in spine map
		"decision stream",
		"allowed=1 blocked=1",
		"not yet wired", // honest about what remains
		"offline-capable --gguf",
		"promotion evidence", // gen/next: what moves this toward now
		"invalidated if:",    // ...and the assumption that retires it instead
	} {
		if !strings.Contains(s, want) {
			t.Errorf("text readout missing %q\n---\n%s", want, s)
		}
	}
}

func TestRunDevWorkspaceJSONSchema(t *testing.T) {
	root := t.TempDir()
	writeGuardJournal(t, root, "interactive-1-aaaa.jsonl", []map[string]any{
		{"seq": 1, "ts_unix_nano": 10, "kind": "DECIDE", "verdict": "DENY", "tool": "Write", "witness": "glob"},
	})
	var out, errb strings.Builder
	code := runDevWorkspace(&out, &errb, []string{"--repo", root, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr: %s", code, errb.String())
	}
	var rep devWorkspaceReport
	if err := json.Unmarshal([]byte(out.String()), &rep); err != nil {
		t.Fatalf("JSON did not round-trip: %v\n%s", err, out.String())
	}
	if rep.Issue != "#3426" || rep.Epic != "#3256" || !rep.Preview {
		t.Errorf("report envelope wrong: %+v", rep)
	}
	if len(rep.Spine) == 0 || len(rep.NotYet) == 0 {
		t.Errorf("report must carry the spine map and the not-yet-wired list: %+v", rep)
	}
	var foundGoalSync bool
	for _, c := range rep.Spine {
		if c.Name == "goal sync" && c.Command == "fak goal sync push" && c.Role == "sync durable goal specs and registry to fak-private" {
			foundGoalSync = true
			break
		}
	}
	if !foundGoalSync {
		t.Errorf("rep.Spine missing 'goal sync' component: %+v", rep.Spine)
	}
	// gen/next closure contract: a preview must name what promotes it AND the
	// assumption whose failure retires it, or it is an unfalsifiable placeholder.
	if len(rep.Promotion) == 0 || rep.InvalidatedIf == "" {
		t.Errorf("report must carry promotion evidence and an invalidating assumption: %+v", rep)
	}
	if rep.Stream.Blocked != 1 || rep.Stream.Witnessed != 1 {
		t.Errorf("stream fold wrong: %+v", rep.Stream)
	}
}

func TestRunDevWorkspaceBadLimit(t *testing.T) {
	var out, errb strings.Builder
	if code := runDevWorkspace(&out, &errb, []string{"--limit", "-1"}); code != 2 {
		t.Fatalf("bad --limit exit = %d, want 2", code)
	}
}

func TestDevWorkspace(t *testing.T) {
	spine := devWorkspaceSpine()
	var found bool
	for _, c := range spine {
		if c.Name == "goal sync" {
			found = true
			if c.Command != "fak goal sync push" {
				t.Errorf("goal sync command = %q, want %q", c.Command, "fak goal sync push")
			}
			if c.Role != "sync durable goal specs and registry to fak-private" {
				t.Errorf("goal sync role = %q, want %q", c.Role, "sync durable goal specs and registry to fak-private")
			}
		}
	}
	if !found {
		t.Errorf("devWorkspaceSpine() missing 'goal sync' component")
	}
}
