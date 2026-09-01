package modver

import (
	"context"
	"fmt"
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
		{"cmd/orphan.go", "", "", false},                    // directly under a root: no module
		{"internal/orphan.go", "", "", false},               // directly under a root: no module
		// docs/ is a hybrid prose keyspace (#2460): top-level pages are file-keyed,
		// sections are directory-keyed, and only .md prose counts.
		{"docs/notes/X.md", "docs/notes", "docs", true},
		{"docs/architecture.md", "docs/architecture.md", "docs", true}, // top-level page: file-keyed
		{"docs/fak/edge-quickstart.md", "docs/fak", "docs", true},      // section page: keys the section
		{"docs/adoption/deep/nested/x.md", "docs/adoption", "docs", true},
		{"docs\\fak\\concept-glossary.md", "docs/fak", "docs", true}, // backslash-normalized
		{"docs/nightrun/module-versions.jsonl", "", "", false},       // generated ledger: data, not prose
		{"docs/nightrun/README.md", "docs/nightrun", "docs", true},   // but prose beside it still counts
		{"docs/_config.yml", "", "", false},                          // site config: not prose
		{"docs/benchmark-methodology.witness.txt", "", "", false},    // witness artifact: not prose
		{"docs/adoption-visuals/chart.svg", "", "", false},           // image: not prose
		{"docs/README.md", "docs/README.md", "docs", true},           // top-level page
		{"docs", "", "", false},                                      // the root itself is no module
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
		// examples/ is a flat, file-keyed policy-manifest keyspace.
		{"examples/repo-guard-policy.json", "examples/repo-guard-policy.json", "policy", true},
		{"examples/customer-support-readonly-policy.json", "examples/customer-support-readonly-policy.json", "policy", true},
		{"examples\\dev-agent-policy.json", "examples/dev-agent-policy.json", "policy", true}, // backslash-normalized
		{"examples/README.md", "", "", false},                 // top-level non-JSON: excluded
		{"examples/mcp/.mcp.json", "", "", false},             // nested demo fixture: excluded
		{"examples/adjudication-demo/main.go", "", "", false}, // nested demo: excluded
		// .claude/skills/ is a directory-keyed agent-skill keyspace.
		{".claude/skills/commit-clean/SKILL.md", ".claude/skills/commit-clean", "skill", true},
		{".claude/skills/skill-overlap/skill_overlap.py", ".claude/skills/skill-overlap", "skill", true}, // helper script folds into the skill
		{".claude/skills/verify/refs/deep.md", ".claude/skills/verify", "skill", true},                   // deeper nesting still keys the skill dir
		{".claude\\skills\\commit-clean\\SKILL.md", ".claude/skills/commit-clean", "skill", true},        // backslash-normalized
		{".claude/skills/README.md", "", "", false},                                                      // directly under the skills root: no module
		{".claude/skills/.gitignore", "", "", false},                                                     // directly under the skills root: no module
		{".claude/settings.json", "", "", false},                                                         // not a skill definition
		{".claude/goal-prompts/ask-hard-questions.md", "", "", false},                                    // prompts, not the skills keyspace
		{".claude/memory/MEMORY.md", "", "", false},                                                      // memory mirror, not the skills keyspace
		{"internal/ctxknobs/testdata/repo/.claude/skills/x/SKILL.md", // a fixture skill under internal/ stays its Go leaf
			"internal/ctxknobs", "internal", true},
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

func TestSnapshotAtPinsLiveFilesAndHistoryToRef(t *testing.T) {
	var calls [][]string
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		calls = append(calls, append([]string(nil), args...))
		switch args[0] {
		case "rev-parse":
			return []byte("deadbeef\n"), nil
		case "ls-tree":
			return []byte("internal/issuepolicy/error_inventory.go\x00"), nil
		case "log":
			return []byte("\x1edeadbeef\t2026-09-01T00:00:00Z\ninternal/issuepolicy/error_inventory.go\n"), nil
		default:
			return nil, fmt.Errorf("unexpected git verb %q", args[0])
		}
	}
	rep, err := SnapshotAt(context.Background(), ".", run, "v1.2.3")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Head != "deadbeef" || len(rep.Modules) != 1 || rep.Modules[0].Name != "internal/issuepolicy" || rep.Modules[0].Version() != "r1+gdeadbeef" {
		t.Fatalf("unexpected historical report: %+v", rep)
	}
	if len(calls) != 3 || calls[0][0] != "rev-parse" || calls[1][0] != "ls-tree" || calls[2][0] != "log" {
		t.Fatalf("unexpected calls: %v", calls)
	}
	if got := strings.Join(calls[1], " "); !strings.Contains(got, "ls-tree -r -z --name-only v1.2.3 --") {
		t.Fatalf("live-file call not pinned to ref: %q", got)
	}
	if got := strings.Join(calls[2], " "); !strings.Contains(got, "--name-only v1.2.3 --") {
		t.Fatalf("history call not pinned to ref: %q", got)
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

// TestDocsKeyspace is the #2460 witness: docs/ prose flows through Snapshot as a
// "docs" module carrying a derived r<rev>+g<sha> version and is emittable as a
// ledger row (the "live stamp containing docs rows"), so docfreshrsi can read a
// doc's rev from the ledger instead of guessing at staleness. It pins the hybrid
// granularity — a top-level page is its own module, a section page keys its
// docs/<dir> at any depth — and the .md-only rule that keeps the ledger
// convergent: appending to docs/nightrun/module-versions.jsonl, the ledger this
// very package writes, must bump NO module, or every stamp would dirty the next.
func TestDocsKeyspace(t *testing.T) {
	const docsLog = "\x1e" + "dc222222\t2026-08-03T12:00:00Z\n" +
		"docs/architecture.md\n" +
		"docs/fak/edge-quickstart.md\n" +
		"docs/fak/concept-glossary.md\n" + // same section twice in one commit: counts once
		"docs/nightrun/module-versions.jsonl\n" + // the ledger itself: bumps nothing
		"docs/_config.yml\n" + // site config: not prose
		"\x1e" + "dc111111\t2026-08-02T09:00:00Z\n" +
		"docs/architecture.md\n" +
		"docs/adoption/deep/nested/playbook.md\n"
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("dchead01\n"), nil
		case "ls-files":
			return []byte("docs/architecture.md\x00docs/fak/edge-quickstart.md\x00" +
				"docs/fak/concept-glossary.md\x00docs/adoption/deep/nested/playbook.md\x00" +
				"docs/nightrun/module-versions.jsonl\x00docs/_config.yml\x00"), nil
		case "log":
			return []byte(docsLog), nil
		}
		t.Fatalf("unexpected git args: %v", args)
		return nil, nil
	}
	rep, err := Snapshot(context.Background(), t.TempDir(), run)
	if err != nil {
		t.Fatal(err)
	}
	// Three modules survive: the top-level page docs/architecture.md and the two
	// sections docs/fak + docs/adoption. The ledger and the site config are data.
	if len(rep.Modules) != 3 {
		t.Fatalf("got %d modules, want 3 (one page + two sections): %+v", len(rep.Modules), rep.Modules)
	}
	arch := findModuleMV(t, rep, "docs/architecture.md")
	if arch.Kind != "docs" || arch.Rev != 2 || arch.LastCommit != "dc222222" {
		t.Fatalf("docs/architecture.md = %+v, want kind=docs rev=2 last=dc222222 (both commits touch it)", arch)
	}
	if v := arch.Version(); v != "r2+gdc222222" {
		t.Errorf("Version() = %q, want r2+gdc222222", v)
	}
	// The section counts once for the commit, not once per page in it.
	fak := findModuleMV(t, rep, "docs/fak")
	if fak.Kind != "docs" || fak.Rev != 1 || fak.LastCommit != "dc222222" {
		t.Fatalf("docs/fak = %+v, want kind=docs rev=1 last=dc222222 (two pages, one commit)", fak)
	}
	// Nesting depth does not fragment the section key.
	adopt := findModuleMV(t, rep, "docs/adoption")
	if adopt.Kind != "docs" || adopt.Rev != 1 {
		t.Fatalf("docs/adoption = %+v, want kind=docs rev=1 (deep nesting still keys the section)", adopt)
	}
	// The ledger this package writes must never become a module: if it did, every
	// stamp would bump docs/nightrun and the next stamp could never report clean.
	for _, m := range rep.Modules {
		if m.Name == "docs/nightrun" {
			t.Errorf("docs/nightrun became a module from a .jsonl-only commit: %+v — the stamp would never converge", m)
		}
	}
	// The docs modules must be emittable as ledger rows (empty prior ledger).
	rows := DeltaRows(rep, nil, "2026-08-03T12:00:00Z")
	if len(rows) != 3 {
		t.Fatalf("ledger rows = %+v, want three docs rows", rows)
	}
	for _, r := range rows {
		if r.Kind != "docs" || !strings.HasPrefix(r.Module, "docs/") {
			t.Errorf("ledger row not a docs row: %+v", r)
		}
	}
	// The docfreshrsi integration seam: a corpus key resolves to the module key
	// whose rev the ledger carries, git-free.
	if got := ModulesForPaths([]string{"docs/fak/edge-quickstart.md", "docs/architecture.md",
		"docs/nightrun/module-versions.jsonl"}); len(got) != 2 ||
		got[0] != "docs/architecture.md" || got[1] != "docs/fak" {
		t.Errorf("ModulesForPaths = %v, want [docs/architecture.md docs/fak] (the ledger maps to no module)", got)
	}
}

// TestPolicyKeyspace is the #2462 witness: a top-level examples/<file>.json flows
// through Snapshot as a file-keyed "policy" module and produces a ledger row (the
// "live stamp with policy rows"), while a top-level non-JSON file and a nested
// demo/fixture path are excluded from the deployable-manifest keyspace.
func TestPolicyKeyspace(t *testing.T) {
	const polLog = "\x1e" + "pl222222\t2026-07-08T12:00:00Z\n" +
		"examples/customer-support-readonly-policy.json\n" +
		"examples/customer-support-readonly-policy.json\n" + // same manifest twice in one commit: counts once
		"examples/README.md\n" + // top-level non-JSON: excluded
		"examples/mcp/.mcp.json\n" + // nested demo fixture: excluded
		"\x1e" + "pl111111\t2026-07-07T09:00:00Z\n" +
		"examples/customer-support-readonly-policy.json\n" +
		"examples/repo-guard-policy.json\n"
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("plhead01\n"), nil
		case "ls-files":
			return []byte("examples/customer-support-readonly-policy.json\x00" +
				"examples/repo-guard-policy.json\x00examples/README.md\x00examples/mcp/.mcp.json\x00"), nil
		case "log":
			return []byte(polLog), nil
		}
		t.Fatalf("unexpected git args: %v", args)
		return nil, nil
	}
	rep, err := Snapshot(context.Background(), t.TempDir(), run)
	if err != nil {
		t.Fatal(err)
	}
	// Two manifests survive: the two top-level policy JSONs. The README and the
	// nested demo fixture are excluded from the keyspace.
	if len(rep.Modules) != 2 {
		t.Fatalf("got %d modules, want 2 (two policy manifests): %+v", len(rep.Modules), rep.Modules)
	}
	cs := findModuleMV(t, rep, "examples/customer-support-readonly-policy.json")
	if cs.Kind != "policy" || cs.Rev != 2 || cs.LastCommit != "pl222222" {
		t.Fatalf("customer-support policy = %+v, want kind=policy rev=2 last=pl222222 (both commits touch it)", cs)
	}
	if v := cs.Version(); v != "r2+gpl222222" {
		t.Errorf("Version() = %q, want r2+gpl222222", v)
	}
	guard := findModuleMV(t, rep, "examples/repo-guard-policy.json")
	if guard.Kind != "policy" || guard.Rev != 1 {
		t.Fatalf("repo-guard policy = %+v, want kind=policy rev=1", guard)
	}
	// The policy manifests must be emittable as ledger rows (empty prior ledger).
	rows := DeltaRows(rep, nil, "2026-07-08T12:00:00Z")
	if len(rows) != 2 {
		t.Fatalf("ledger rows = %+v, want two policy rows", rows)
	}
	for _, r := range rows {
		if r.Kind != "policy" || !strings.HasPrefix(r.Module, "examples/") {
			t.Errorf("ledger row not a policy row: %+v", r)
		}
	}
}

// TestSkillKeyspace is the #2461 witness: a .claude/skills/<name>/ directory flows
// through Snapshot as a directory-keyed "skill" module carrying a derived
// r<rev>+g<sha> version, and is emittable as a ledger row (the "live stamp with
// skill rows"). It pins the three exclusions that make the keyspace honest — a
// helper script folds into its skill rather than becoming a module of its own, a
// file sitting directly under .claude/skills/ is no module, and a non-skills
// .claude/ subtree is outside the keyspace — plus the kind="skill" tag that makes
// the skills-drift query one jq over the ledger.
func TestSkillKeyspace(t *testing.T) {
	// Only the newer commit touches commit-clean, so skill-overlap's last-touch
	// stays behind it: the two skills have to be separable by date for the
	// drift assertion below to mean anything.
	const skillLog = "\x1e" + "sk222222\t2026-07-26T12:00:00Z\n" +
		".claude/skills/commit-clean/SKILL.md\n" +
		".claude/skills/README.md\n" + // directly under the skills root: excluded
		".claude/settings.json\n" + // not a skill definition: excluded
		"\x1e" + "sk111111\t2026-07-25T09:00:00Z\n" +
		".claude/skills/commit-clean/SKILL.md\n" +
		".claude/skills/skill-overlap/SKILL.md\n" +
		".claude/skills/skill-overlap/skill_overlap.py\n" // helper folds into the skill: counts once
	run := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch args[0] {
		case "rev-parse":
			return []byte("skhead01\n"), nil
		case "ls-files":
			return []byte(".claude/skills/commit-clean/SKILL.md\x00" +
				".claude/skills/skill-overlap/SKILL.md\x00.claude/skills/skill-overlap/skill_overlap.py\x00" +
				".claude/skills/README.md\x00.claude/settings.json\x00"), nil
		case "log":
			return []byte(skillLog), nil
		}
		t.Fatalf("unexpected git args: %v", args)
		return nil, nil
	}
	rep, err := Snapshot(context.Background(), t.TempDir(), run)
	if err != nil {
		t.Fatal(err)
	}
	// Two skills survive: commit-clean and skill-overlap. The skills-root README
	// and the non-skills .claude/settings.json are excluded from the keyspace.
	if len(rep.Modules) != 2 {
		t.Fatalf("got %d modules, want 2 (two skill directories): %+v", len(rep.Modules), rep.Modules)
	}
	clean := findModuleMV(t, rep, ".claude/skills/commit-clean")
	if clean.Kind != "skill" || clean.Rev != 2 || clean.LastCommit != "sk222222" {
		t.Fatalf("commit-clean skill = %+v, want kind=skill rev=2 last=sk222222 (both commits touch it)", clean)
	}
	if v := clean.Version(); v != "r2+gsk222222" {
		t.Errorf("Version() = %q, want r2+gsk222222", v)
	}
	// The helper script must fold into its skill, not become a second module, and
	// must not double-count the rev of the commit that touched both its files.
	overlap := findModuleMV(t, rep, ".claude/skills/skill-overlap")
	if overlap.Kind != "skill" || overlap.Rev != 1 {
		t.Fatalf("skill-overlap = %+v, want kind=skill rev=1 (SKILL.md + helper are one module, one commit)", overlap)
	}
	// A skill untouched for longer must read as older: last-touch is a distinct
	// field from the derived rev, and it is what makes skill drift visible.
	if !(overlap.LastDate < clean.LastDate) {
		t.Errorf("last-touch dates do not separate the skills: overlap %q vs commit-clean %q",
			overlap.LastDate, clean.LastDate)
	}
	// The skills must be emittable as ledger rows (empty prior ledger), each
	// tagged kind=skill so a skills-drift query is one jq over the ledger.
	rows := DeltaRows(rep, nil, "2026-07-26T12:00:00Z")
	if len(rows) != 2 {
		t.Fatalf("ledger rows = %+v, want two skill rows", rows)
	}
	for _, r := range rows {
		if r.Kind != "skill" || !strings.HasPrefix(r.Module, ".claude/skills/") {
			t.Errorf("ledger row not a skill row: %+v", r)
		}
		if r.Version == "" || r.LastDate == "" {
			t.Errorf("skill row missing the rev/last-touch a drift query reads: %+v", r)
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
	// A flat score carries NO provenance — it must stay empty, never defaulted
	// to "witnessed", so a modeled score can never masquerade as a witnessed one.
	if p := rep.Modules[0].ScoreProvenance; p != "" {
		t.Errorf("flat-score provenance = %q, want empty", p)
	}
	if rep.Modules[1].Score != nil {
		t.Errorf("cmd/fak score should be unset")
	}
	if _, err := LoadScores([]byte(`["not","a","map"]`)); err == nil {
		t.Errorf("LoadScores should reject a non-map")
	}
}

// TestLoadScoresBothShapes is the #2498 witness: LoadScores accepts the flat
// {module: number} shape and the extended {module: {score, provenance}} shape,
// mixed freely, and a joined report carries provenance only where it was
// supplied — flat scores stay unlabeled.
func TestLoadScoresBothShapes(t *testing.T) {
	// Flat shape alone still decodes (back-compat).
	flat, err := LoadScores([]byte(`{"internal/gateway": 8.5}`))
	if err != nil {
		t.Fatalf("flat LoadScores: %v", err)
	}
	if got := flat["internal/gateway"]; got.Score != 8.5 || got.Provenance != "" {
		t.Errorf("flat entry = %+v, want {8.5 \"\"}", got)
	}

	// Extended shape, and a mix of both shapes in one file.
	mixed, err := LoadScores([]byte(`{
		"internal/gateway": {"score": 9, "provenance": "witnessed"},
		"internal/modver":  {"score": 4.2, "provenance": "modeled"},
		"cmd/fak":          {"score": 6, "provenance": "observed"},
		"internal/flat":    7.5
	}`))
	if err != nil {
		t.Fatalf("extended LoadScores: %v", err)
	}
	for name, want := range map[string]ScoreEntry{
		"internal/gateway": {Score: 9, Provenance: ProvenanceWitnessed},
		"internal/modver":  {Score: 4.2, Provenance: ProvenanceModeled},
		"cmd/fak":          {Score: 6, Provenance: ProvenanceObserved},
		"internal/flat":    {Score: 7.5, Provenance: ""},
	} {
		if got := mixed[name]; got != want {
			t.Errorf("%s entry = %+v, want %+v", name, got, want)
		}
	}

	// JoinScores propagates both score and provenance onto matching rows, and a
	// DeltaRows stamp carries the provenance label into the ledger row.
	rep := Report{Head: "deadbee1", Modules: []Module{
		{Name: "internal/modver", Kind: "internal", Rev: 3, LastCommit: "abc", LastDate: "2026-07-10T00:00:00Z"},
		{Name: "internal/flat", Kind: "internal", Rev: 1, LastCommit: "def", LastDate: "2026-07-09T00:00:00Z"},
	}}
	if n := rep.JoinScores(mixed); n != 2 {
		t.Fatalf("JoinScores matched %d, want 2", n)
	}
	if p := rep.Modules[0].ScoreProvenance; p != ProvenanceModeled {
		t.Errorf("internal/modver provenance = %q, want %q", p, ProvenanceModeled)
	}
	if p := rep.Modules[1].ScoreProvenance; p != "" {
		t.Errorf("internal/flat (flat score) provenance = %q, want empty", p)
	}

	rows := DeltaRows(rep, nil, "2026-07-11T00:00:00Z")
	byName := map[string]LedgerRow{}
	for _, r := range rows {
		byName[r.Module] = r
	}
	if r := byName["internal/modver"]; r.ScoreProvenance != ProvenanceModeled {
		t.Errorf("ledger row provenance = %q, want %q", r.ScoreProvenance, ProvenanceModeled)
	}
	if r := byName["internal/flat"]; r.ScoreProvenance != "" {
		t.Errorf("flat-score ledger row provenance = %q, want empty", r.ScoreProvenance)
	}
}

// TestLoadScoresRejectsUnknownProvenance locks the closed set: an unrecognized
// provenance label is refused rather than flowing into the ledger wearing the
// authority of a real one.
func TestLoadScoresRejectsUnknownProvenance(t *testing.T) {
	if _, err := LoadScores([]byte(`{"internal/gateway": {"score": 9, "provenance": "guessed"}}`)); err == nil {
		t.Errorf("LoadScores should reject an unknown provenance label")
	}
	// A malformed value (neither number nor {score, provenance}) is refused too.
	if _, err := LoadScores([]byte(`{"internal/gateway": "8.5"}`)); err == nil {
		t.Errorf("LoadScores should reject a string score value")
	}
}

// TestDeltaRowsProvenanceChange proves a provenance relabel is a real ledger
// movement even when the numeric score is unchanged: relabeling modeled->
// witnessed must append a fresh row so the corrected label is witnessed.
func TestDeltaRowsProvenanceChange(t *testing.T) {
	prev := []byte(`{"schema":"fak-module-versions/1","module":"internal/modver","kind":"internal","rev":3,"score":4.2,"score_provenance":"modeled"}` + "\n")
	s := 4.2
	rep := Report{Head: "deadbee1", Modules: []Module{
		{Name: "internal/modver", Kind: "internal", Rev: 3, LastCommit: "abc", LastDate: "2026-07-10T00:00:00Z", Score: &s, ScoreProvenance: ProvenanceWitnessed},
	}}
	rows := DeltaRows(rep, prev, "2026-07-11T00:00:00Z")
	if len(rows) != 1 {
		t.Fatalf("provenance relabel produced %d rows, want 1", len(rows))
	}
	if rows[0].ScoreProvenance != ProvenanceWitnessed {
		t.Errorf("relabeled row provenance = %q, want %q", rows[0].ScoreProvenance, ProvenanceWitnessed)
	}

	// No change at all (same rev, same score, same provenance) produces no row.
	prev2 := []byte(`{"schema":"fak-module-versions/1","module":"internal/modver","kind":"internal","rev":3,"score":4.2,"score_provenance":"witnessed"}` + "\n")
	if rows := DeltaRows(rep, prev2, "2026-07-11T00:00:00Z"); len(rows) != 0 {
		t.Errorf("unchanged module produced %d rows, want 0", len(rows))
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
