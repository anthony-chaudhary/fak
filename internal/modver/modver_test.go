package modver

import (
	"context"
	"strings"
	"testing"
)

func TestModuleOf(t *testing.T) {
	cases := []struct {
		path string
		name string
		kind string
		ok   bool
	}{
		{"internal/gateway/wire.go", "internal/gateway", "internal", true},
		{"internal/gateway/sub/deep.go", "internal/gateway", "internal", true},
		{"cmd/fak/main.go", "cmd/fak", "cmd", true},
		{"cmd/trychatdemo/page.html", "cmd/trychatdemo", "cmd", true},
		{"internal\\modver\\modver.go", "internal/modver", "internal", true},
		{".github/workflows/ci.yml", ".github/workflows/ci.yml", "workflow", true},
		{".github\\workflows\\release-cadence.yml", ".github/workflows/release-cadence.yml", "workflow", true},
		{".github/workflows/nested/x.yml", "", "", false},   // Actions ignores subdirs
		{".github/actions/setup/action.yml", "", "", false}, // not the workflows keyspace
		{"docs/notes/X.md", "", "", false},
		{"cmd/orphan.go", "", "", false}, // directly under a root: no module
		{"", "", "", false},
	}
	for _, c := range cases {
		name, kind, ok := moduleOf(c.path)
		if name != c.name || kind != c.kind || ok != c.ok {
			t.Errorf("moduleOf(%q) = (%q, %q, %v), want (%q, %q, %v)",
				c.path, name, kind, ok, c.name, c.kind, c.ok)
		}
	}
}

// fixture history, newest-first: three commits over two live modules plus one
// deleted module that must not ghost into the report.
const logFixture = "\x1e" + "aaa11111\t2026-07-02T10:00:00Z\n" +
	"internal/gateway/wire.go\n" +
	"internal/gateway/metrics.go\n" + // same module twice in one commit: counts once
	"cmd/fak/main.go\n" +
	"\x1e" + "bbb22222\t2026-07-01T09:00:00Z\n" +
	"internal/gateway/wire.go\n" +
	"internal/deleted/gone.go\n" +
	"\x1e" + "ccc33333\t2026-06-30T08:00:00Z\n" +
	"cmd/fak/main.go\n"

func liveFixture() map[string]bool {
	return map[string]bool{"internal/gateway": true, "cmd/fak": true}
}

func TestParseLog(t *testing.T) {
	mods := parseLog([]byte(logFixture), liveFixture())
	if len(mods) != 2 {
		t.Fatalf("got %d modules, want 2: %+v", len(mods), mods)
	}
	// sorted by name: cmd/fak, internal/gateway
	fak, gw := mods[0], mods[1]
	if fak.Name != "cmd/fak" || fak.Rev != 2 || fak.LastCommit != "aaa11111" {
		t.Errorf("cmd/fak = %+v, want rev 2 last aaa11111", fak)
	}
	if gw.Name != "internal/gateway" || gw.Rev != 2 || gw.LastCommit != "aaa11111" || gw.LastDate != "2026-07-02T10:00:00Z" {
		t.Errorf("internal/gateway = %+v, want rev 2 last aaa11111 @2026-07-02", gw)
	}
	if v := gw.Version(); v != "r2+gaaa11111" {
		t.Errorf("Version() = %q, want r2+gaaa11111", v)
	}
}

func TestSnapshotWithFakeRunner(t *testing.T) {
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("deadbee1\n"), nil
		case "ls-files":
			return []byte("internal/gateway/wire.go\x00cmd/fak/main.go\x00"), nil
		case "log":
			return []byte(logFixture), nil
		}
		t.Fatalf("unexpected git args: %v", args)
		return nil, nil
	}
	rep, err := Snapshot(context.Background(), t.TempDir(), run)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Head != "deadbee1" {
		t.Errorf("head = %q", rep.Head)
	}
	if len(rep.Modules) != 2 {
		t.Fatalf("got %d modules, want 2 (deleted module must be excluded): %+v", len(rep.Modules), rep.Modules)
	}
	for _, m := range rep.Modules {
		if m.Name == "internal/deleted" {
			t.Errorf("deleted module ghosted into the report")
		}
	}
}

// TestWorkflowKeyspace is the #2464 witness: a .github/workflows/<file> flows
// through Snapshot as a file-keyed "workflow" module and produces a ledger row,
// while a non-workflow .github file is excluded from the keyspace.
func TestWorkflowKeyspace(t *testing.T) {
	const wfLog = "\x1e" + "wf111111\t2026-07-04T12:00:00Z\n" +
		".github/workflows/ci.yml\n" +
		".github/workflows/ci.yml\n" + // same workflow twice in one commit: counts once
		".github/actions/setup/action.yml\n" + // not the workflows keyspace: excluded
		"\x1e" + "wf000000\t2026-07-03T09:00:00Z\n" +
		".github/workflows/ci.yml\n"
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("wfhead01\n"), nil
		case "ls-files":
			return []byte(".github/workflows/ci.yml\x00.github/actions/setup/action.yml\x00"), nil
		case "log":
			return []byte(wfLog), nil
		}
		t.Fatalf("unexpected git args: %v", args)
		return nil, nil
	}
	rep, err := Snapshot(context.Background(), t.TempDir(), run)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Modules) != 1 {
		t.Fatalf("got %d modules, want 1 (only the workflow file): %+v", len(rep.Modules), rep.Modules)
	}
	m := rep.Modules[0]
	if m.Name != ".github/workflows/ci.yml" || m.Kind != "workflow" || m.Rev != 2 {
		t.Fatalf("workflow module = %+v, want .github/workflows/ci.yml kind=workflow rev=2", m)
	}
	if v := m.Version(); v != "r2+gwf111111" {
		t.Errorf("Version() = %q, want r2+gwf111111", v)
	}
	// The workflow module must be emittable as a ledger row (empty prior ledger).
	rows := DeltaRows(rep, nil, "2026-07-04T12:00:00Z")
	if len(rows) != 1 || rows[0].Module != ".github/workflows/ci.yml" || rows[0].Kind != "workflow" {
		t.Fatalf("ledger rows = %+v, want one workflow row", rows)
	}
}

func TestJoinScores(t *testing.T) {
	rep := Report{Modules: []Module{{Name: "internal/gateway"}, {Name: "cmd/fak"}}}
	scores, err := LoadScores([]byte(`{"internal/gateway": 8.5, "internal/other": 1}`))
	if err != nil {
		t.Fatal(err)
	}
	if n := rep.JoinScores(scores); n != 1 {
		t.Fatalf("matched %d, want 1", n)
	}
	if rep.Modules[0].Score == nil || *rep.Modules[0].Score != 8.5 {
		t.Errorf("internal/gateway score = %v, want 8.5", rep.Modules[0].Score)
	}
	if rep.Modules[1].Score != nil {
		t.Errorf("cmd/fak score should be unset")
	}
	if _, err := LoadScores([]byte(`["not","a","map"]`)); err == nil {
		t.Errorf("LoadScores should reject a non-map")
	}
}

func TestDeltaRows(t *testing.T) {
	rep := Report{
		Head:       "deadbee1",
		AppVersion: "0.37.0",
		Modules: []Module{
			{Name: "cmd/fak", Kind: "cmd", Rev: 2, LastCommit: "aaa11111", LastDate: "2026-07-02T10:00:00Z"},
			{Name: "internal/gateway", Kind: "internal", Rev: 5, LastCommit: "aaa11111", LastDate: "2026-07-02T10:00:00Z"},
			{Name: "internal/modver", Kind: "internal", Rev: 1, LastCommit: "aaa11111", LastDate: "2026-07-02T10:00:00Z"},
		},
	}
	prev := strings.Join([]string{
		`{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":2,"version":"r2+gccc33333"}`,
		`this line is scar tissue and must be tolerated`,
		`{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":4,"version":"r4+gbbb22222"}`,
	}, "\n")
	rows := DeltaRows(rep, []byte(prev), "2026-07-03T00:00:00Z")
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2 (gateway moved, modver new, fak unchanged): %+v", len(rows), rows)
	}
	byMod := map[string]LedgerRow{}
	for _, r := range rows {
		byMod[r.Module] = r
		if r.Schema != Schema || r.TS != "2026-07-03T00:00:00Z" || r.Head != "deadbee1" {
			t.Errorf("row envelope wrong: %+v", r)
		}
	}
	if _, ok := byMod["cmd/fak"]; ok {
		t.Errorf("unchanged module stamped a row")
	}
	if r := byMod["internal/gateway"]; r.Rev != 5 || r.Version != "r5+gaaa11111" {
		t.Errorf("gateway row = %+v", r)
	}
	if _, ok := byMod["internal/modver"]; !ok {
		t.Errorf("new module missing a row")
	}

	lines, err := AppendLines(rows)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(lines), "\n"); got != 2 {
		t.Errorf("AppendLines produced %d lines, want 2", got)
	}

	// Stamping the appended ledger again must be a no-op: the delta converges.
	again := DeltaRows(rep, append([]byte(prev+"\n"), lines...), "2026-07-03T01:00:00Z")
	if len(again) != 0 {
		t.Errorf("second stamp not empty: %+v", again)
	}
}
