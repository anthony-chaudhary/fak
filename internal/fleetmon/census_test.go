package fleetmon

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// censusFixtureNow is the fixed clock the census fixtures are aged against.
var censusFixtureNow = time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)

// writeSessionFile writes a session transcript file at path (creating parents)
// and stamps its mtime, so the census's last-turn-age/liveness rung is
// deterministic regardless of when the test runs.
func writeSessionFile(t *testing.T, path string, mtime time.Time) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(`{"type":"user"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatalf("chtimes %s: %v", path, err)
	}
}

// findRow returns the first census row for an agent, optionally filtered by kind.
func findRow(rows []CensusRow, agent string, kind RowKind) (CensusRow, bool) {
	for _, r := range rows {
		if r.Agent == agent && r.Kind == kind {
			return r, true
		}
	}
	return CensusRow{}, false
}

// TestCensusCrossAgentGolden is the issue #2213 witness: on a fixture host with a
// live codex session AND a live claude session, the census emits one row each
// with correct agent attribution, and an agent with no discoverable namespace
// (the env-key openai-generic profile) yields a typed NO_NAMESPACE row — never
// silence.
func TestCensusCrossAgentGolden(t *testing.T) {
	home := t.TempDir()

	// A live claude session: ~/.claude/projects/<ns>/<uuid>.jsonl, fresh mtime.
	const claudeNS = "-c-work-fak"
	const claudeSID = "11111111-1111-4111-8111-111111111111"
	claudePath := filepath.Join(home, ".claude", "projects", claudeNS, claudeSID+".jsonl")
	writeSessionFile(t, claudePath, censusFixtureNow)

	// A live codex session: ~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl.
	const codexSID = "22222222-2222-4222-8222-222222222222"
	codexPath := filepath.Join(home, ".codex", "sessions", "2026", "07", "04",
		"rollout-2026-07-04T12-00-00-"+codexSID+".jsonl")
	writeSessionFile(t, codexPath, censusFixtureNow)

	rows := Census(home, censusFixtureNow)

	// Exactly one claude SESSION row, correctly attributed and byte-compatible
	// with discoverWorkers (session id = bare basename, namespace = project dir).
	claude, ok := findRow(rows, "claude", KindSession)
	if !ok {
		t.Fatalf("no claude SESSION row; got %+v", rows)
	}
	if claude.Session != claudeSID {
		t.Errorf("claude session id = %q, want %q (byte-compat with discoverWorkers)", claude.Session, claudeSID)
	}
	if claude.Namespace != claudeNS {
		t.Errorf("claude namespace = %q, want %q", claude.Namespace, claudeNS)
	}
	if claude.Path != claudePath {
		t.Errorf("claude path = %q, want %q", claude.Path, claudePath)
	}
	if claude.Liveness != LivenessLive {
		t.Errorf("claude liveness = %q, want LIVE (fresh mtime)", claude.Liveness)
	}
	if !claude.HasAge {
		t.Errorf("claude row should carry a last-turn age")
	}

	// Exactly one codex SESSION row: the uuid is pulled out of the rollout name.
	codex, ok := findRow(rows, "codex", KindSession)
	if !ok {
		t.Fatalf("no codex SESSION row; got %+v", rows)
	}
	if codex.Session != codexSID {
		t.Errorf("codex session id = %q, want %q (uuid from rollout filename)", codex.Session, codexSID)
	}
	if codex.Liveness != LivenessLive {
		t.Errorf("codex liveness = %q, want LIVE (fresh mtime)", codex.Liveness)
	}

	// The env-key harness (opencode/aider/hermes → openai-generic) has no config
	// home, so it yields a typed NO_NAMESPACE row, never silence.
	env, ok := findRow(rows, "openai-generic", KindNoNamespace)
	if !ok {
		t.Fatalf("no NO_NAMESPACE row for the env-key harness; got %+v", rows)
	}
	if env.Liveness != LivenessUnknown {
		t.Errorf("NO_NAMESPACE liveness = %q, want UNKNOWN", env.Liveness)
	}
	if env.Session != "" || env.Path != "" {
		t.Errorf("NO_NAMESPACE row should carry no session/path, got session=%q path=%q", env.Session, env.Path)
	}
	if env.Note == "" {
		t.Errorf("NO_NAMESPACE row should explain why the namespace is absent")
	}

	// The Gemini and pi harnesses have no config homes in this fixture, so like
	// the env-key harness they yield typed NO_NAMESPACE rows, never silence.
	gemini, ok := findRow(rows, "gemini", KindNoNamespace)
	if !ok {
		t.Fatalf("no NO_NAMESPACE row for the gemini harness; got %+v", rows)
	}
	if gemini.Liveness != LivenessUnknown || gemini.Note == "" {
		t.Errorf("gemini NO_NAMESPACE row = %+v, want UNKNOWN with explanation", gemini)
	}

	pi, ok := findRow(rows, "pi", KindNoNamespace)
	if !ok {
		t.Fatalf("no NO_NAMESPACE row for the pi harness; got %+v", rows)
	}
	if pi.Liveness != LivenessUnknown {
		t.Errorf("pi NO_NAMESPACE liveness = %q, want UNKNOWN", pi.Liveness)
	}
	if pi.Note == "" {
		t.Errorf("pi NO_NAMESPACE row should explain why the namespace is absent")
	}

	// No agent is silently dropped: every built-in profile contributes at least one
	// row, and each SESSION row's agent is exactly its profile Name.
	if got := len(rows); got != 5 {
		t.Fatalf("census row count = %d, want 5 (claude + codex + gemini + openai-generic + pi); rows=%+v", got, rows)
	}
	for _, r := range rows {
		if r.Agent == "" {
			t.Errorf("row with empty agent attribution: %+v", r)
		}
	}
}

// TestCensusNoAgentsNeverSilent asserts the "never silence" contract at the other
// extreme: a bare home with no harness config dirs at all still yields one typed
// NO_NAMESPACE row per built-in agent (nothing is silently omitted).
func TestCensusNoAgentsNeverSilent(t *testing.T) {
	home := t.TempDir() // empty: no .claude, no .codex

	rows := Census(home, censusFixtureNow)

	agents := []string{"claude", "codex", "gemini", "openai-generic", "pi"}
	for _, agent := range agents {
		r, ok := findRow(rows, agent, KindNoNamespace)
		if !ok {
			t.Fatalf("agent %q missing its NO_NAMESPACE row; got %+v", agent, rows)
		}
		if r.Liveness != LivenessUnknown {
			t.Errorf("agent %q NO_NAMESPACE liveness = %q, want UNKNOWN", agent, r.Liveness)
		}
	}
	if len(rows) != len(agents) {
		t.Errorf("bare home census = %d rows, want one NO_NAMESPACE row per %d agents", len(rows), len(agents))
	}
}

// TestCensusIdleAndMultiSession covers the recency rung (a stale transcript reads
// IDLE, not LIVE) and multiple sessions under one claude namespace.
func TestCensusIdleAndMultiSession(t *testing.T) {
	home := t.TempDir()
	const ns = "-c-work-fak"

	fresh := filepath.Join(home, ".claude", "projects", ns, "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa.jsonl")
	stale := filepath.Join(home, ".claude", "projects", ns, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb.jsonl")
	writeSessionFile(t, fresh, censusFixtureNow)
	writeSessionFile(t, stale, censusFixtureNow.Add(-(RecentWindow + time.Hour)))

	rows := Census(home, censusFixtureNow)

	var claudeRows []CensusRow
	for _, r := range rows {
		if r.Agent == "claude" && r.Kind == KindSession {
			claudeRows = append(claudeRows, r)
		}
	}
	if len(claudeRows) != 2 {
		t.Fatalf("want 2 claude sessions under one namespace, got %d: %+v", len(claudeRows), claudeRows)
	}
	byLive := map[Liveness]int{}
	for _, r := range claudeRows {
		byLive[r.Liveness]++
		if !strings.HasPrefix(r.Path, filepath.Join(home, ".claude")) {
			t.Errorf("unexpected path %q", r.Path)
		}
	}
	if byLive[LivenessLive] != 1 || byLive[LivenessIdle] != 1 {
		t.Errorf("recency split = %v, want one LIVE + one IDLE", byLive)
	}
}
