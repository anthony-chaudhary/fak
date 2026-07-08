package modver

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
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
		{"cmd/orphan.go", "", "", false},      // directly under a root: no module
		{"internal/orphan.go", "", "", false}, // directly under a root: no module
		// tools/ is a flat, family-keyed script keyspace.
		{"tools/account_probe.py", "tools/account_probe", "tools", true},
		{"tools/account_probe_test.py", "tools/account_probe", "tools", true}, // _test folds into the family
		{"tools/auto_push_on_lag.sh", "tools/auto_push_on_lag", "tools", true},
		{"tools\\account_probe.py", "tools/account_probe", "tools", true},          // backslash-normalized
		{"tools/agent_test_harness.py", "tools/agent_test_harness", "tools", true}, // only a trailing _test folds
		{"tools/bench_baseline.json", "", "", false},                               // data/fixture, not a script
		{"tools/FLEET.md", "", "", false},                                          // doc, not a script
		{"tools/_registry/state.py", "", "", false},                                // nested: registry, not the flat inventory
		{"tools/__pycache__/x.pyc", "", "", false},                                 // nested cache
		{"tools/.gitignore", "", "", false},                                        // bare dotfile
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

// TestToolsKeyspace is the #2459 witness: a top-level tools/ script flows through
// Snapshot as a family-keyed "tools" module — its _test sibling folds into the same
// family (tools/<name>), non-script fixtures and nested registry paths are excluded,
// and the module is emittable as a ledger row (the "live stamp showing tools rows").
func TestToolsKeyspace(t *testing.T) {
	const toolsLog = "\x1e" + "tl222222\t2026-07-06T12:00:00Z\n" +
		"tools/account_probe.py\n" +
		"tools/account_probe_test.py\n" + // _test folds into tools/account_probe: one module, counts once
		"tools/bench_baseline.json\n" + // fixture: excluded from the keyspace
		"tools/_registry/state.py\n" + // nested registry: excluded
		"\x1e" + "tl111111\t2026-07-05T09:00:00Z\n" +
		"tools/account_probe.py\n" +
		"tools/auto_push_on_lag.sh\n"
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("tlhead01\n"), nil
		case "ls-files":
			return []byte("tools/account_probe.py\x00tools/account_probe_test.py\x00" +
				"tools/auto_push_on_lag.sh\x00tools/bench_baseline.json\x00tools/_registry/state.py\x00"), nil
		case "log":
			return []byte(toolsLog), nil
		}
		t.Fatalf("unexpected git args: %v", args)
		return nil, nil
	}
	rep, err := Snapshot(context.Background(), t.TempDir(), run)
	if err != nil {
		t.Fatal(err)
	}
	// Two families survive: tools/account_probe and tools/auto_push_on_lag.
	// The fixture and the nested registry path are excluded.
	if len(rep.Modules) != 2 {
		t.Fatalf("got %d modules, want 2 (two script families): %+v", len(rep.Modules), rep.Modules)
	}
	probe := findModuleMV(t, rep, "tools/account_probe")
	if probe.Kind != "tools" || probe.Rev != 2 || probe.LastCommit != "tl222222" {
		t.Fatalf("tools/account_probe = %+v, want kind=tools rev=2 last=tl222222 (both commits touch the family; the _test sibling counts once)", probe)
	}
	if v := probe.Version(); v != "r2+gtl222222" {
		t.Errorf("Version() = %q, want r2+gtl222222", v)
	}
	push := findModuleMV(t, rep, "tools/auto_push_on_lag")
	if push.Kind != "tools" || push.Rev != 1 {
		t.Fatalf("tools/auto_push_on_lag = %+v, want kind=tools rev=1", push)
	}
	// The tools modules must be emittable as ledger rows (empty prior ledger).
	rows := DeltaRows(rep, nil, "2026-07-06T12:00:00Z")
	if len(rows) != 2 {
		t.Fatalf("ledger rows = %+v, want two tools rows", rows)
	}
	for _, r := range rows {
		if r.Kind != "tools" || !strings.HasPrefix(r.Module, "tools/") {
			t.Errorf("ledger row not a tools row: %+v", r)
		}
	}
}

// TestSnapshotHostPathParity is the #2478 witness: the same logical history must
// produce a byte-identical Report whether git emits POSIX paths joined by "\n"
// (Linux/WSL) or Windows-style backslash paths joined by "\r\n". The fleet runs
// the same repo natively and under WSL; moduleOf normalizes separators and
// parseLog/moduleOf trim the trailing "\r", but nothing witnessed that the
// end-to-end Snapshot output AGREES across host path styles until this test.
//
// Host caveat: git itself prints forward slashes on every platform, so a real
// `git ls-files`/`git log` never emits the backslash form. The test synthesizes
// the Windows-native style (backslashes + CRLF) directly against the Runner seam
// to witness the normalization contract a non-git or core.autocrlf source could
// still feed in — a stronger, host-agnostic parity check than a WSL-only run.
func TestSnapshotHostPathParity(t *testing.T) {
	type commit struct {
		sha, date string
		files     []string
	}
	// One logical history over two live modules (plus a deleted one that must not
	// ghost), expressed host-neutrally with "/" separators.
	history := []commit{
		{"aaa11111", "2026-07-02T10:00:00Z", []string{
			"internal/gateway/wire.go", "internal/gateway/metrics.go", "cmd/fak/main.go"}},
		{"bbb22222", "2026-07-01T09:00:00Z", []string{
			"internal/gateway/wire.go", "internal/deleted/gone.go"}},
		{"ccc33333", "2026-06-30T08:00:00Z", []string{"cmd/fak/main.go"}},
	}
	liveFiles := []string{"internal/gateway/wire.go", "cmd/fak/main.go"}

	// host renders the fixtures under a given separator + line ending, mimicking
	// what git prints on that platform, and returns a fake Runner over them.
	host := func(sep, eol string) Runner {
		render := func(p string) string { return strings.ReplaceAll(p, "/", sep) }
		var logB strings.Builder
		for _, c := range history {
			logB.WriteString("\x1e" + c.sha + "\t" + c.date + eol)
			for _, f := range c.files {
				logB.WriteString(render(f) + eol)
			}
		}
		var lsB strings.Builder
		for _, f := range liveFiles {
			lsB.WriteString(render(f) + "\x00") // ls-files -z is NUL-terminated, no EOL
		}
		return func(_ context.Context, _ string, args ...string) ([]byte, error) {
			switch args[0] {
			case "rev-parse":
				return []byte("deadbee1" + eol), nil
			case "ls-files":
				return []byte(lsB.String()), nil
			case "log":
				return []byte(logB.String()), nil
			}
			t.Fatalf("unexpected git args: %v", args)
			return nil, nil
		}
	}

	dir := t.TempDir() // same (repo-external) dir both ways: AppVersion is not the variable under test
	posix, err := Snapshot(context.Background(), dir, host("/", "\n"))
	if err != nil {
		t.Fatalf("posix snapshot: %v", err)
	}
	windows, err := Snapshot(context.Background(), dir, host("\\", "\r\n"))
	if err != nil {
		t.Fatalf("windows snapshot: %v", err)
	}

	// Guard against a trivial both-empty pass: the fixture must yield real work.
	if len(posix.Modules) != 2 {
		t.Fatalf("posix produced %d modules, want 2: %+v", len(posix.Modules), posix.Modules)
	}
	if posix.Head != "deadbee1" {
		t.Errorf("posix head = %q, want deadbee1 (trailing CRLF/whitespace not trimmed?)", posix.Head)
	}
	// sorted by name: cmd/fak, internal/gateway — assert the fields the ledger renders.
	if m := posix.Modules[1]; m.Name != "internal/gateway" || m.Rev != 2 ||
		m.LastCommit != "aaa11111" || m.LastDate != "2026-07-02T10:00:00Z" {
		t.Errorf("posix internal/gateway = %+v, want rev 2 last aaa11111 @2026-07-02", m)
	}
	if len(windows.Modules) != len(posix.Modules) {
		t.Fatalf("windows produced %d modules, want %d — backslash paths not normalized?",
			len(windows.Modules), len(posix.Modules))
	}

	if !reflect.DeepEqual(posix, windows) {
		t.Errorf("host path parity broken:\n posix   = %+v\n windows = %+v", posix, windows)
	}
}

// TestSnapshotPassesNoMerges is the #2475 witness at the invocation seam: the
// rev semantics are "distinct NON-MERGE commits touching the module", pinned by
// passing --no-merges to git log. Asserting the flag here makes the decision a
// tested fact independent of git's (configurable, --diff-merges) default merge-
// diff behavior, and guards against a well-meaning switch to --first-parent,
// which would undercount work that reaches the trunk through a merge.
func TestSnapshotPassesNoMerges(t *testing.T) {
	var logArgs []string
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("deadbee1\n"), nil
		case "ls-files":
			return []byte("internal/gateway/wire.go\x00"), nil
		case "log":
			logArgs = append([]string{}, args...)
			return []byte(logFixture), nil
		}
		t.Fatalf("unexpected git args: %v", args)
		return nil, nil
	}
	if _, err := Snapshot(context.Background(), t.TempDir(), run); err != nil {
		t.Fatal(err)
	}
	noMerges := false
	for _, a := range logArgs {
		switch a {
		case "--no-merges":
			noMerges = true
		case "--first-parent":
			t.Errorf("git log must NOT use --first-parent (it undercounts merged-in work): %v", logArgs)
		}
	}
	if !noMerges {
		t.Fatalf("git log missing --no-merges (rev must exclude merge commits): %v", logArgs)
	}
}

// TestSnapshotExcludesMergeCommits is the #2475 end-to-end witness — a fixture
// history WITH a real merge in it. rev counts distinct non-merge commits
// touching a module; the merged-in non-merge commits DO count (they are real
// authored work) but the merge commit that joins them does not, so the merge
// commit is never a module's last_commit and does not inflate rev.
func TestSnapshotExcludesMergeCommits(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	repo := modverGitRepo(t)
	commitFileMV(t, repo, "internal/foo/a.go", "package foo\n// c1\n", "c1") // internal/foo #1
	gitMV(t, repo, "checkout", "-q", "-b", "side")
	commitFileMV(t, repo, "internal/foo/b.go", "package foo\n// s1\n", "s1") // internal/foo #2 (on side)
	gitMV(t, repo, "checkout", "-q", "main")
	commitFileMV(t, repo, "internal/foo/a.go", "package foo\n// c2\n", "c2") // internal/foo #3
	// A real, no-fast-forward merge of the diverged side branch: creates a merge
	// commit joining c2 and s1. It touches internal/foo transitively but must not
	// count — it is the "in-place trunk merge" the rev must be stable across.
	gitMV(t, repo, "merge", "--no-ff", "-q", "-m", "merge side", "side")

	rep, err := Snapshot(context.Background(), repo, RealRunner)
	if err != nil {
		t.Fatal(err)
	}
	foo := findModuleMV(t, rep, "internal/foo")
	if foo.Rev != 3 {
		t.Fatalf("internal/foo rev = %d, want 3 (c1,s1,c2 — merge excluded): %+v", foo.Rev, foo)
	}
	mergeSHA := strings.TrimSpace(string(mustGitMV(t, repo, "rev-parse", "--short=8", "HEAD")))
	if foo.LastCommit == mergeSHA {
		t.Fatalf("merge commit %s leaked in as internal/foo last_commit — merge counted as a rev", mergeSHA)
	}
}

func modverGitRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	gitMV(t, repo, "init", "-q", "-b", "main")
	gitMV(t, repo, "config", "core.autocrlf", "false")
	gitMV(t, repo, "config", "user.name", "test")
	gitMV(t, repo, "config", "user.email", "test@example.com")
	writeMV(t, filepath.Join(repo, "README.md"), "base\n") // root file: belongs to no module
	gitMV(t, repo, "add", "README.md")
	gitMV(t, repo, "commit", "-q", "-m", "base")
	return repo
}

func commitFileMV(t *testing.T, repo, rel, body, msg string) {
	t.Helper()
	writeMV(t, filepath.Join(repo, filepath.FromSlash(rel)), body)
	gitMV(t, repo, "add", rel)
	gitMV(t, repo, "commit", "-q", "-m", msg)
}

func writeMV(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func gitMV(t *testing.T, repo string, args ...string) { mustGitMV(t, repo, args...) }

func mustGitMV(t *testing.T, repo string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = repo
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, repo, err, out)
	}
	return out
}

func findModuleMV(t *testing.T, rep Report, name string) Module {
	t.Helper()
	for _, m := range rep.Modules {
		if m.Name == name {
			return m
		}
	}
	t.Fatalf("module %q not in report: %+v", name, rep.Modules)
	return Module{}
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

func TestView(t *testing.T) {
	rep := Report{
		Head:       "deadbee1",
		AppVersion: "0.37.0",
		Modules: []Module{
			{Name: "cmd/fak", Kind: "cmd", Rev: 2, LastDate: "2026-07-02T10:00:00Z"},
			{Name: "internal/alpha", Kind: "internal", Rev: 9, LastDate: "2026-07-01T10:00:00Z"},
			{Name: "internal/beta", Kind: "internal", Rev: 5, LastDate: "2026-07-05T10:00:00Z"},
		},
	}

	// --only filters by name prefix and leaves the receiver untouched.
	got, err := rep.View("internal/", "name", 0)
	if err != nil {
		t.Fatal(err)
	}
	if names := moduleNames(got); !reflect.DeepEqual(names, []string{"internal/alpha", "internal/beta"}) {
		t.Errorf("only=internal/ names = %v, want [internal/alpha internal/beta]", names)
	}
	if len(rep.Modules) != 3 {
		t.Errorf("View mutated the receiver: %d modules left", len(rep.Modules))
	}

	// --sort rev is most-revised-first.
	got, err = rep.View("", "rev", 0)
	if err != nil {
		t.Fatal(err)
	}
	if names := moduleNames(got); !reflect.DeepEqual(names, []string{"internal/alpha", "internal/beta", "cmd/fak"}) {
		t.Errorf("sort=rev names = %v, want [internal/alpha internal/beta cmd/fak]", names)
	}

	// --sort date is most-recently-touched-first, --top truncates after sorting.
	got, err = rep.View("", "date", 2)
	if err != nil {
		t.Fatal(err)
	}
	if names := moduleNames(got); !reflect.DeepEqual(names, []string{"internal/beta", "cmd/fak"}) {
		t.Errorf("sort=date top=2 names = %v, want [internal/beta cmd/fak]", names)
	}

	// An unknown sort key fails loud rather than defaulting silently.
	if _, err := rep.View("", "bogus", 0); err == nil {
		t.Errorf("View should reject an unknown sort key")
	}
}

func moduleNames(rep Report) []string {
	names := make([]string, len(rep.Modules))
	for i, m := range rep.Modules {
		names[i] = m.Name
	}
	return names
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
