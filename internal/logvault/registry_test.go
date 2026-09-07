package logvault

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseSourcesTOMLArrayOfTables(t *testing.T) {
	toml := `
# Global comment
[[sources]]
id = "custom-audit"
root = ".custom-audit"
includes = ["*.jsonl", "*.log"]
excludes = ["tmp/", "*.tmp"]
max_bytes = 10485760
note = "custom audit directory" # inline comment

[[sources]]
id = "metrics"
root = "/var/log/metrics"
note = 'single-quoted note'
`
	srcs, err := ParseSourcesTOML([]byte(toml))
	if err != nil {
		t.Fatalf("ParseSourcesTOML failed: %v", err)
	}
	if len(srcs) != 2 {
		t.Fatalf("got %d sources, want 2", len(srcs))
	}

	s0 := srcs[0]
	if s0.ID != "custom-audit" {
		t.Errorf("s0.ID = %q, want custom-audit", s0.ID)
	}
	if s0.Root != ".custom-audit" {
		t.Errorf("s0.Root = %q, want .custom-audit", s0.Root)
	}
	if len(s0.Includes) != 2 || s0.Includes[0] != "*.jsonl" || s0.Includes[1] != "*.log" {
		t.Errorf("s0.Includes = %v, want [*.jsonl, *.log]", s0.Includes)
	}
	if len(s0.Excludes) != 2 || s0.Excludes[0] != "tmp/" || s0.Excludes[1] != "*.tmp" {
		t.Errorf("s0.Excludes = %v, want [tmp/, *.tmp]", s0.Excludes)
	}
	if s0.MaxBytes != 10485760 {
		t.Errorf("s0.MaxBytes = %d, want 10485760", s0.MaxBytes)
	}
	if s0.Note != "custom audit directory" {
		t.Errorf("s0.Note = %q, want 'custom audit directory'", s0.Note)
	}

	s1 := srcs[1]
	if s1.ID != "metrics" {
		t.Errorf("s1.ID = %q, want metrics", s1.ID)
	}
	if s1.Root != filepath.Clean("/var/log/metrics") {
		t.Errorf("s1.Root = %q, want %q", s1.Root, filepath.Clean("/var/log/metrics"))
	}
	if s1.Note != "single-quoted note" {
		t.Errorf("s1.Note = %q, want 'single-quoted note'", s1.Note)
	}
}

func TestParseSourcesTOMLTableOfTables(t *testing.T) {
	toml := `
[sources.app-logs]
root = "logs/app"
includes = ["*.jsonl"]
note = "application logs"

[sources.crash-dumps]
root = "crash"
excludes = ["core.*"]
note = "crash dumps"
`
	srcs, err := ParseSourcesTOML([]byte(toml))
	if err != nil {
		t.Fatalf("ParseSourcesTOML failed: %v", err)
	}
	if len(srcs) != 2 {
		t.Fatalf("got %d sources, want 2", len(srcs))
	}

	m := map[string]Source{}
	for _, s := range srcs {
		m[s.ID] = s
	}

	app, ok := m["app-logs"]
	if !ok {
		t.Fatal("missing app-logs source")
	}
	if app.Root != filepath.Clean("logs/app") || len(app.Includes) != 1 || app.Includes[0] != "*.jsonl" {
		t.Errorf("unexpected app-logs: %+v", app)
	}

	crash, ok := m["crash-dumps"]
	if !ok {
		t.Fatal("missing crash-dumps source")
	}
	if crash.Root != "crash" || len(crash.Excludes) != 1 || crash.Excludes[0] != "core.*" {
		t.Errorf("unexpected crash-dumps: %+v", crash)
	}
}

func TestLoadSourcesTOMLResolvesRelativeRoots(t *testing.T) {
	repoRoot := filepath.Join(t.TempDir(), "myrepo")
	toml := `
[[sources]]
id = "rel-source"
root = "sub/dir"
note = "relative root"

[[sources]]
id = "abs-source"
root = "/already/abs"
note = "absolute root"
`
	srcs, err := LoadSourcesTOML([]byte(toml), repoRoot)
	if err != nil {
		t.Fatalf("LoadSourcesTOML: %v", err)
	}
	if len(srcs) != 2 {
		t.Fatalf("got %d sources, want 2", len(srcs))
	}

	wantRel := filepath.Join(repoRoot, "sub", "dir")
	if srcs[0].Root != wantRel {
		t.Errorf("srcs[0].Root = %q, want %q", srcs[0].Root, wantRel)
	}
	if !filepath.IsAbs(srcs[1].Root) {
		t.Errorf("srcs[1].Root = %q, want absolute path", srcs[1].Root)
	}
}

func TestParseSourcesTOMLMultilineArray(t *testing.T) {
	toml := `
[[sources]]
id = "multiline"
root = "logs"
includes = [
    "*.jsonl",
    "*.log",
    "debug/*.txt",
]
note = "multiline test"
`
	srcs, err := ParseSourcesTOMLString(toml)
	if err != nil {
		t.Fatalf("ParseSourcesTOMLString: %v", err)
	}
	if len(srcs) != 1 {
		t.Fatalf("got %d sources, want 1", len(srcs))
	}
	if len(srcs[0].Includes) != 3 {
		t.Fatalf("got %d includes, want 3 (%v)", len(srcs[0].Includes), srcs[0].Includes)
	}
	if srcs[0].Includes[2] != "debug/*.txt" {
		t.Errorf("include[2] = %q, want debug/*.txt", srcs[0].Includes[2])
	}
}

func TestParseSourcesJSON(t *testing.T) {
	jsonData := `[
		{
			"id": "json-source",
			"root": "json/root",
			"includes": ["*.dat"],
			"note": "from json"
		}
	]`
	srcs, err := ParseSources([]byte(jsonData))
	if err != nil {
		t.Fatalf("ParseSources JSON: %v", err)
	}
	if len(srcs) != 1 || srcs[0].ID != "json-source" {
		t.Fatalf("unexpected parsed JSON sources: %+v", srcs)
	}
}

func TestMergeSources(t *testing.T) {
	base := []Source{
		{ID: "s1", Root: "r1", Note: "base s1"},
		{ID: "s2", Root: "r2", Note: "base s2"},
	}
	overlays := []Source{
		{ID: "s2", Root: "new-r2", Note: "override s2"},
		{ID: "s3", Root: "r3", Note: "new s3"},
	}

	merged := MergeSources(base, overlays)
	if len(merged) != 3 {
		t.Fatalf("merged len = %d, want 3", len(merged))
	}
	if merged[0].ID != "s1" || merged[0].Root != "r1" {
		t.Errorf("merged[0] = %+v, want s1 preserved", merged[0])
	}
	if merged[1].ID != "s2" || merged[1].Root != "new-r2" || merged[1].Note != "override s2" {
		t.Errorf("merged[1] = %+v, want s2 overridden", merged[1])
	}
	if merged[2].ID != "s3" || merged[2].Root != "r3" {
		t.Errorf("merged[2] = %+v, want s3 appended", merged[2])
	}

	// Prove base is unmodified
	if base[1].Root != "r2" {
		t.Errorf("base[1].Root was mutated to %q", base[1].Root)
	}
}

func TestLoadSourcesFallbackAndMerged(t *testing.T) {
	repoRoot := t.TempDir()
	home := t.TempDir()

	// 1. Empty config falls back to DefaultSources
	fallbackSrcs, err := LoadSourcesFallback(nil, repoRoot, home)
	if err != nil {
		t.Fatalf("LoadSourcesFallback: %v", err)
	}
	defaults := DefaultSources(repoRoot, home)
	if len(fallbackSrcs) != len(defaults) {
		t.Fatalf("empty fallback got %d sources, want %d default sources", len(fallbackSrcs), len(defaults))
	}

	// 2. Non-empty config with LoadSourcesFallback returns custom sources only
	customTOML := `
[[sources]]
id = "only-me"
root = "custom"
`
	onlyCustom, err := LoadSourcesFallback([]byte(customTOML), repoRoot, home)
	if err != nil {
		t.Fatalf("LoadSourcesFallback custom: %v", err)
	}
	if len(onlyCustom) != 1 || onlyCustom[0].ID != "only-me" {
		t.Fatalf("LoadSourcesFallback custom got %+v, want only [only-me]", onlyCustom)
	}

	// 3. LoadSourcesMerged merges custom over DefaultSources
	merged, err := LoadSourcesMerged([]byte(customTOML), repoRoot, home)
	if err != nil {
		t.Fatalf("LoadSourcesMerged: %v", err)
	}
	if len(merged) != len(defaults)+1 {
		t.Fatalf("LoadSourcesMerged got %d sources, want %d", len(merged), len(defaults)+1)
	}
	found := false
	for _, s := range merged {
		if s.ID == "only-me" {
			found = true
			wantRoot := filepath.Join(repoRoot, "custom")
			if s.Root != wantRoot {
				t.Errorf("merged custom root = %q, want %q", s.Root, wantRoot)
			}
		}
	}
	if !found {
		t.Fatal("custom source 'only-me' not found in merged sources")
	}
}

func TestEnvDrivenSourcesAutoRegister(t *testing.T) {
	repo := t.TempDir()

	auditPath := filepath.Join(repo, "audit", "guard.jsonl")
	loopPath := filepath.Join(repo, "fak-loops.jsonl")
	toolprocPath := filepath.Join(repo, "toolproc.log")
	slackDir := filepath.Join(repo, "slack-outbox")
	watchdogDir := filepath.Join(repo, "watchdog-autoheal")
	fleetRegDir := filepath.Join(repo, "custom-fleet-reg")
	blobDir := filepath.Join(repo, "blob-store")

	envMap := map[string]string{
		"FAK_AUDIT_JOURNAL":         auditPath,
		"FAK_LOOP_LEDGER":           loopPath,
		"FAK_TOOLPROC_JOURNAL":      toolprocPath,
		"FAK_SLACK_OUTBOX_DIR":      slackDir,
		"FAK_WATCHDOG_AUTOHEAL_DIR": watchdogDir,
		"FLEET_REG_DIR":             fleetRegDir,
		"FAK_BLOB_DIR":              blobDir,
	}

	lookup := func(k string) string {
		return envMap[k]
	}

	srcs := EnvSourcesFrom(lookup, repo)
	if len(srcs) != 7 {
		t.Fatalf("EnvSourcesFrom returned %d sources, want 7", len(srcs))
	}

	byID := map[string]Source{}
	for _, s := range srcs {
		byID[s.ID] = s
	}

	tests := []struct {
		id       string
		wantRoot string
		wantNote string
	}{
		{"audit-journal", auditPath, "env-registered audit journal"},
		{"loop-ledger", loopPath, "env-registered loop ledger"},
		{"toolproc-journal", toolprocPath, "env-registered toolproc journal"},
		{"slack-outbox", slackDir, "env-registered Slack outbox"},
		{"watchdog-autoheal", watchdogDir, "env-registered watchdog autoheal directory"},
		{"fleet-reg", fleetRegDir, "env-registered fleet registry"},
		{"blob-cas", blobDir, "env-registered blob CAS payload store"},
	}

	for _, tt := range tests {
		s, ok := byID[tt.id]
		if !ok {
			t.Errorf("missing env source %q", tt.id)
			continue
		}
		if s.Root != filepath.Clean(tt.wantRoot) {
			t.Errorf("source %q Root = %q, want %q", tt.id, s.Root, tt.wantRoot)
		}
		if s.Note != tt.wantNote {
			t.Errorf("source %q Note = %q, want %q", tt.id, s.Note, tt.wantNote)
		}
	}
}

func TestEnvSourcesInDefaultSources(t *testing.T) {
	tempBlob := t.TempDir()
	t.Setenv("FAK_BLOB_DIR", tempBlob)
	t.Setenv("FAK_AUDIT_JOURNAL", filepath.Join(t.TempDir(), "env-audit.jsonl"))

	srcs := DefaultSources(t.TempDir(), "")
	foundBlob := false
	foundAudit := false
	for _, s := range srcs {
		if s.ID == "blob-cas" {
			foundBlob = true
			if s.Root != filepath.Clean(tempBlob) {
				t.Errorf("blob-cas root = %q, want %q", s.Root, tempBlob)
			}
			if s.Note != "env-registered blob CAS payload store" {
				t.Errorf("blob-cas note = %q", s.Note)
			}
		}
		if s.ID == "audit-journal" {
			foundAudit = true
		}
	}
	if !foundBlob {
		t.Error("FAK_BLOB_DIR not auto-registered in DefaultSources")
	}
	if !foundAudit {
		t.Error("FAK_AUDIT_JOURNAL not auto-registered in DefaultSources")
	}
}

func TestEnvSourceCaptureSingleFileAndDir(t *testing.T) {
	// Prove that a single-file env source (FAK_AUDIT_JOURNAL) and directory env source (FAK_BLOB_DIR)
	// capture cleanly into the vault.
	repo := t.TempDir()
	auditFile := filepath.Join(repo, "audit.jsonl")
	writeFile(t, auditFile, "entry1\nentry2\n")

	blobDir := filepath.Join(repo, "blobs")
	writeFile(t, filepath.Join(blobDir, "b1"), "blob-content-1")

	v := &Vault{
		Dir: t.TempDir(),
		Sources: []Source{
			{ID: "audit-journal", Root: auditFile, Note: "single file"},
			{ID: "blob-cas", Root: blobDir, Note: "dir"},
		},
	}

	stats, err := v.Capture()
	if err != nil {
		t.Fatalf("Vault.Capture failed: %v", err)
	}
	if len(stats) != 2 {
		t.Fatalf("stats len = %d, want 2", len(stats))
	}
	if stats[0].Full != 1 || stats[0].Errors != 0 {
		t.Errorf("audit stats = %+v, want 1 full copy, 0 errors", stats[0])
	}
	if stats[1].Full != 1 || stats[1].Errors != 0 {
		t.Errorf("blob stats = %+v, want 1 full copy, 0 errors", stats[1])
	}

	// Verify mirrored contents in vault
	gotAudit := readFile(t, v.mirrorPath("audit-journal", "audit.jsonl"))
	if gotAudit != "entry1\nentry2\n" {
		t.Errorf("mirrored audit = %q, want entry1\\nentry2\\n", gotAudit)
	}
	gotBlob := readFile(t, v.mirrorPath("blob-cas", "b1"))
	if gotBlob != "blob-content-1" {
		t.Errorf("mirrored blob = %q, want blob-content-1", gotBlob)
	}
}

func TestDiscoverInstancesFindsUnregisteredItems(t *testing.T) {
	repo := t.TempDir()

	// 1. Registered instances under default sources
	writeFile(t, filepath.Join(repo, ".dos", "marker"), "marker")
	writeFile(t, filepath.Join(repo, "tools", ".dos", "marker"), "marker")
	writeFile(t, filepath.Join(repo, ".dispatch-runs", "guard-audit", "sess1.jsonl"), `{"ok":true}`+"\n")
	writeFile(t, filepath.Join(repo, "docs", "nightrun", "cache-value.jsonl"), `{"v":1}`+"\n")

	// 2. Unregistered instances (forked or stray)
	forkedDos := filepath.Join(repo, "sub", "worker", ".dos")
	if err := os.MkdirAll(forkedDos, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(forkedDos, "rogue"), "rogue")

	strayAudit := filepath.Join(repo, "stray", "guard-audit", "stray.jsonl")
	writeFile(t, strayAudit, `{"stray":true}`+"\n")

	strayNightrun := filepath.Join(repo, "other", "docs", "nightrun", "stray.jsonl")
	writeFile(t, strayNightrun, `{"nightrun":true}`+"\n")

	// 3. Ignored directory (.git)
	gitDos := filepath.Join(repo, ".git", ".dos")
	if err := os.MkdirAll(gitDos, 0o755); err != nil {
		t.Fatal(err)
	}

	registered := DefaultSources(repo, "")
	discovered, err := DiscoverInstances(repo, registered)
	if err != nil {
		t.Fatalf("DiscoverInstances: %v", err)
	}

	if len(discovered) != 3 {
		t.Fatalf("got %d discovered unregistered instances, want 3:\n%+v", len(discovered), discovered)
	}

	m := map[string]DiscoveredInstance{}
	for _, d := range discovered {
		m[d.RelPath] = d
		if d.Warning == "" {
			t.Errorf("instance at %s missing Warning", d.RelPath)
		}
	}

	wantForkedRel := filepath.ToSlash(filepath.Join("sub", "worker", ".dos"))
	if d, ok := m[wantForkedRel]; !ok {
		t.Errorf("missing unregistered instance for %s", wantForkedRel)
	} else if !d.IsDir {
		t.Errorf("forked .dos isDir = false, want true")
	}

	wantAuditRel := filepath.ToSlash(filepath.Join("stray", "guard-audit", "stray.jsonl"))
	if _, ok := m[wantAuditRel]; !ok {
		t.Errorf("missing unregistered instance for %s", wantAuditRel)
	}

	wantNightrunRel := filepath.ToSlash(filepath.Join("other", "docs", "nightrun", "stray.jsonl"))
	if _, ok := m[wantNightrunRel]; !ok {
		t.Errorf("missing unregistered instance for %s", wantNightrunRel)
	}

	// Verify DiscoverWarnings returns warning strings
	warnings, err := DiscoverWarnings(repo, registered)
	if err != nil {
		t.Fatalf("DiscoverWarnings: %v", err)
	}
	if len(warnings) != 3 {
		t.Fatalf("DiscoverWarnings got %d warnings, want 3", len(warnings))
	}
	foundWarning := false
	for _, w := range warnings {
		if strings.Contains(w, wantForkedRel) {
			foundWarning = true
		}
	}
	if !foundWarning {
		t.Errorf("warnings do not contain %s: %v", wantForkedRel, warnings)
	}

	// 4. Now register stray-audit declaratively and re-run: warning should disappear!
	registered = append(registered, Source{
		ID:   "stray-audit",
		Root: filepath.Join(repo, "stray", "guard-audit"),
	})
	registered = append(registered, Source{
		ID:   "forked-dos",
		Root: forkedDos,
	})
	registered = append(registered, Source{
		ID:   "stray-nightrun",
		Root: filepath.Join(repo, "other", "docs", "nightrun"),
	})

	afterRegistered, err := DiscoverInstances(repo, registered)
	if err != nil {
		t.Fatalf("DiscoverInstances after: %v", err)
	}
	if len(afterRegistered) != 0 {
		t.Errorf("after registering all instances, got %d warnings, want 0: %+v", len(afterRegistered), afterRegistered)
	}
}

func TestIsRegistered(t *testing.T) {
	repo := t.TempDir()
	dosDir := filepath.Join(repo, ".dos")
	writeFile(t, filepath.Join(dosDir, "file"), "x")

	srcs := []Source{
		{ID: "dos", Root: dosDir},
		{ID: "audits", Root: filepath.Join(repo, "audits"), Includes: []string{"*.jsonl"}},
	}

	if !IsRegistered(dosDir, true, repo, srcs) {
		t.Errorf("IsRegistered(%s) = false, want true", dosDir)
	}
	if !IsRegistered(filepath.Join(repo, "audits", "session.jsonl"), false, repo, srcs) {
		t.Error("audits/session.jsonl should be registered")
	}
	if IsRegistered(filepath.Join(repo, "audits", "session.txt"), false, repo, srcs) {
		t.Error("audits/session.txt should not be registered (not in includes)")
	}
	if IsRegistered(filepath.Join(repo, "unknown", ".dos"), true, repo, srcs) {
		t.Error("unknown/.dos should not be registered")
	}
}
