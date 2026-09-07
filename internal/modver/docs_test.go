package modver

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestDocModule verifies the mapping from various doc path formats to canonical
// module names in the docs keyspace.
func TestDocModule(t *testing.T) {
	tests := []struct {
		input string
		want  string
		ok    bool
	}{
		// Top-level markdown files
		{"docs/architecture.md", "docs/architecture.md", true},
		{"docs/README.md", "docs/README.md", true},
		{"docs/FAQ.md", "docs/FAQ.md", true},

		// Section pages (directory-keyed)
		{"docs/fak/edge-quickstart.md", "docs/fak", true},
		{"docs/fak/concept-glossary.md", "docs/fak", true},
		{"docs/adoption/deep/nested/playbook.md", "docs/adoption", true},
		{"docs/notes/X.md", "docs/notes", true},

		// Bare section directory or section path
		{"docs/fak", "docs/fak", true},
		{"docs/adoption", "docs/adoption", true},
		{"docs/adoption/deep", "docs/adoption", true},

		// Prefixed with docs:
		{"docs:architecture.md", "docs/architecture.md", true},
		{"docs:fak/edge-quickstart.md", "docs/fak", true},
		{"docs:fak", "docs/fak", true},
		{"docs:adoption/playbook.md", "docs/adoption", true},

		// Windows backslash paths
		{"docs\\fak\\edge-quickstart.md", "docs/fak", true},
		{"docs\\architecture.md", "docs/architecture.md", true},
		{"docs\\notes\\X.md", "docs/notes", true},

		// Doc-relative paths (without docs/ prefix)
		{"architecture.md", "docs/architecture.md", true},
		{"fak/edge-quickstart.md", "docs/fak", true},
		{"adoption/deep/playbook.md", "docs/adoption", true},
		{"fak", "docs/fak", true},

		// Non-markdown data files: excluded so ledgers and site config are not prose modules
		{"docs/nightrun/module-versions.jsonl", "", false},
		{"docs/_config.yml", "", false},
		{"docs/benchmark-methodology.witness.txt", "", false},
		{"docs/adoption-visuals/chart.svg", "", false},
		{"docs/data.json", "", false},
		{"docs/sub/data.csv", "", false},

		// Non-docs roots
		{"internal/modver/modver.go", "", false},
		{"cmd/fak/main.go", "", false},
		{".github/workflows/ci.yml", "", false},
		{"tools/probe.py", "", false},
		{"examples/policy.json", "", false},
		{".claude/skills/commit-clean/SKILL.md", "", false},

		// Degenerate inputs
		{"", "", false},
		{"docs", "", false},
		{"docs/", "", false},
		{"   ", "", false},
	}

	for _, tc := range tests {
		got, ok := DocModule(tc.input)
		if ok != tc.ok || got != tc.want {
			t.Errorf("DocModule(%q) = (%q, %v), want (%q, %v)", tc.input, got, ok, tc.want, tc.ok)
		}
	}
}

// TestReportDocRevAndLookup tests querying doc revisions and modules from Report.
func TestReportDocRevAndLookup(t *testing.T) {
	rep := Report{
		Head: "dchead01",
		Modules: []Module{
			{Name: "cmd/fak", Kind: "cmd", Rev: 10, LastCommit: "c111"},
			{Name: "docs/architecture.md", Kind: "docs", Rev: 5, LastCommit: "a555", LastDate: "2026-08-01T10:00:00Z"},
			{Name: "docs/fak", Kind: "docs", Rev: 12, LastCommit: "f222", LastDate: "2026-08-02T11:00:00Z"},
			{Name: "docs/adoption", Kind: "docs", Rev: 3, LastCommit: "d333", LastDate: "2026-08-03T12:00:00Z"},
			{Name: "internal/modver", Kind: "internal", Rev: 42, LastCommit: "m444"},
		},
	}

	// 1. Test rep.DocRev across different path styles
	queryTests := []struct {
		query   string
		wantRev int
		wantOK  bool
	}{
		{"docs/architecture.md", 5, true},
		{"architecture.md", 5, true},
		{"docs:architecture.md", 5, true},
		{"docs/fak/edge-quickstart.md", 12, true},
		{"docs/fak", 12, true},
		{"docs:fak", 12, true},
		{"fak/edge-quickstart.md", 12, true},
		{"docs\\fak\\concept-glossary.md", 12, true},
		{"docs/adoption/deep/nested/playbook.md", 3, true},
		{"docs/adoption", 3, true},
		{"docs/nonexistent.md", 0, false},
		{"docs/nightrun/module-versions.jsonl", 0, false},
		{"internal/modver", 0, false},
		{"cmd/fak", 0, false},
	}

	for _, tc := range queryTests {
		rev, ok := rep.DocRev(tc.query)
		if ok != tc.wantOK || rev != tc.wantRev {
			t.Errorf("rep.DocRev(%q) = (%d, %v), want (%d, %v)", tc.query, rev, ok, tc.wantRev, tc.wantOK)
		}
		// DocRevFromReport & LookupDocRev should yield identical results
		if r2, ok2 := DocRevFromReport(rep, tc.query); ok2 != ok || r2 != rev {
			t.Errorf("DocRevFromReport(%q) = (%d, %v), want (%d, %v)", tc.query, r2, ok2, rev, ok)
		}
		if r3, ok3 := LookupDocRev(rep, tc.query); ok3 != ok || r3 != rev {
			t.Errorf("LookupDocRev(%q) = (%d, %v), want (%d, %v)", tc.query, r3, ok3, rev, ok)
		}
	}

	// 2. Test rep.LookupDoc
	m, ok := rep.LookupDoc("docs/fak/edge-quickstart.md")
	if !ok || m.Name != "docs/fak" || m.Rev != 12 || m.LastCommit != "f222" {
		t.Fatalf("rep.LookupDoc(docs/fak/edge-quickstart.md) = %+v, ok=%v, want docs/fak rev=12", m, ok)
	}
	if _, ok := rep.LookupDoc("internal/modver"); ok {
		t.Errorf("rep.LookupDoc(internal/modver) returned ok=true, want false for non-doc")
	}

	// 3. Test rep.LookupModule (general module lookup)
	modM, ok := rep.LookupModule("internal/modver")
	if !ok || modM.Rev != 42 {
		t.Errorf("rep.LookupModule(internal/modver) = %+v, ok=%v, want rev=42", modM, ok)
	}

	// 4. Test rep.DocRevs
	docRevs := rep.DocRevs()
	if len(docRevs) != 3 {
		t.Fatalf("rep.DocRevs() len = %d, want 3: %v", len(docRevs), docRevs)
	}
	if docRevs["docs/architecture.md"] != 5 || docRevs["docs/fak"] != 12 || docRevs["docs/adoption"] != 3 {
		t.Errorf("rep.DocRevs() unexpected values: %v", docRevs)
	}

	// 5. Test DocRevsForPaths
	paths := []string{
		"docs/architecture.md",
		"docs/fak/edge-quickstart.md",
		"docs/fak/other.md",
		"docs/missing.md",
		"internal/modver",
	}
	revsForPaths := DocRevsForPaths(rep, paths)
	if len(revsForPaths) != 3 {
		t.Fatalf("DocRevsForPaths len = %d, want 3: %v", len(revsForPaths), revsForPaths)
	}
	if revsForPaths["docs/architecture.md"] != 5 {
		t.Errorf("revsForPaths[docs/architecture.md] = %d, want 5", revsForPaths["docs/architecture.md"])
	}
	if revsForPaths["docs/fak/edge-quickstart.md"] != 12 {
		t.Errorf("revsForPaths[docs/fak/edge-quickstart.md] = %d, want 12", revsForPaths["docs/fak/edge-quickstart.md"])
	}
	if revsForPaths["docs/fak/other.md"] != 12 {
		t.Errorf("revsForPaths[docs/fak/other.md] = %d, want 12", revsForPaths["docs/fak/other.md"])
	}
}

// TestLedgerDocRevAndLookup tests querying doc revisions and rows from append-only ledger bytes.
func TestLedgerDocRevAndLookup(t *testing.T) {
	const sampleLedger = `{"schema":"fak-module-versions/1","ts":"2026-08-01T00:00:00Z","head":"h01","module":"docs/architecture.md","kind":"docs","rev":1,"version":"r1+gh01"}
{"schema":"fak-module-versions/1","ts":"2026-08-01T00:00:00Z","head":"h01","module":"docs/fak","kind":"docs","rev":2,"version":"r2+gh01"}
{"schema":"fak-module-versions/1","ts":"2026-08-01T00:00:00Z","head":"h01","module":"internal/modver","kind":"internal","rev":10,"version":"r10+gh01"}
{"schema":"fak-module-versions/1","ts":"2026-08-02T00:00:00Z","head":"h02","module":"docs/architecture.md","kind":"docs","rev":3,"version":"r3+gh02"}
{"schema":"fak-module-versions/1","ts":"2026-08-02T00:00:00Z","head":"h02","module":"docs/fak","kind":"docs","rev":5,"version":"r5+gh02"}
{"schema":"fak-module-versions/1","ts":"2026-08-03T00:00:00Z","head":"h03","module":"docs/notes","kind":"docs","rev":1,"version":"r1+gh03"}
`
	ledgerBytes := []byte(sampleLedger)

	// 1. Test DocRevFromLedger (must return the latest rev for moved modules)
	queries := []struct {
		query   string
		wantRev int
		wantOK  bool
	}{
		{"docs/architecture.md", 3, true}, // grew from 1 to 3
		{"architecture.md", 3, true},
		{"docs:architecture.md", 3, true},
		{"docs/fak/edge-quickstart.md", 5, true}, // grew from 2 to 5
		{"docs/fak", 5, true},
		{"docs:fak", 5, true},
		{"fak/edge-quickstart.md", 5, true},
		{"docs/notes/X.md", 1, true},
		{"docs/notes", 1, true},
		{"docs/missing.md", 0, false},
		{"internal/modver", 0, false}, // not docs keyspace
	}

	for _, tc := range queries {
		rev, ok := DocRevFromLedger(ledgerBytes, tc.query)
		if ok != tc.wantOK || rev != tc.wantRev {
			t.Errorf("DocRevFromLedger(%q) = (%d, %v), want (%d, %v)", tc.query, rev, ok, tc.wantRev, tc.wantOK)
		}
		if r2, ok2 := LookupDocRevFromLedger(ledgerBytes, tc.query); ok2 != ok || r2 != rev {
			t.Errorf("LookupDocRevFromLedger(%q) = (%d, %v), want (%d, %v)", tc.query, r2, ok2, rev, ok)
		}
	}

	// 2. Test LookupDocFromLedger (returns latest row)
	row, ok := LookupDocFromLedger(ledgerBytes, "docs/architecture.md")
	if !ok || row.Rev != 3 || row.Head != "h02" || row.TS != "2026-08-02T00:00:00Z" {
		t.Fatalf("LookupDocFromLedger(docs/architecture.md) = %+v, ok=%v, want rev=3 head=h02", row, ok)
	}

	// 3. Test DocRevsFromLedger
	allDocs := DocRevsFromLedger(ledgerBytes)
	if len(allDocs) != 3 {
		t.Fatalf("DocRevsFromLedger len = %d, want 3: %v", len(allDocs), allDocs)
	}
	if allDocs["docs/architecture.md"] != 3 || allDocs["docs/fak"] != 5 || allDocs["docs/notes"] != 1 {
		t.Errorf("DocRevsFromLedger values mismatch: %v", allDocs)
	}

	// 4. Test DocRevsForPathsFromLedger
	batchPaths := []string{
		"docs/architecture.md",
		"docs/fak/edge-quickstart.md",
		"docs/notes/meeting.md",
		"docs/untracked.md",
	}
	batchRevs := DocRevsForPathsFromLedger(ledgerBytes, batchPaths)
	if len(batchRevs) != 3 {
		t.Fatalf("DocRevsForPathsFromLedger len = %d, want 3: %v", len(batchRevs), batchRevs)
	}
	if batchRevs["docs/architecture.md"] != 3 {
		t.Errorf("batchRevs[docs/architecture.md] = %d, want 3", batchRevs["docs/architecture.md"])
	}
	if batchRevs["docs/fak/edge-quickstart.md"] != 5 {
		t.Errorf("batchRevs[docs/fak/edge-quickstart.md] = %d, want 5", batchRevs["docs/fak/edge-quickstart.md"])
	}
	if batchRevs["docs/notes/meeting.md"] != 1 {
		t.Errorf("batchRevs[docs/notes/meeting.md] = %d, want 1", batchRevs["docs/notes/meeting.md"])
	}
}

// TestDocsLiveGitWorkflow exercises the entire lifecycle against a real git repository:
// creating doc files, checking Snapshot derives valid revs, checking DeltaRows emits
// ledger rows, updating docs, verifying rev progression, and querying revs at each stage.
func TestDocsLiveGitWorkflow(t *testing.T) {
	repo := modverGitRepo(t)

	// Commit 1: Initial docs and ledger
	commitFileMV(t, repo, "docs/architecture.md", "# Architecture\nInitial\n", "docs: initial architecture")
	commitFileMV(t, repo, "docs/fak/edge-quickstart.md", "# Quickstart\nInitial\n", "docs: initial quickstart")
	commitFileMV(t, repo, "docs/nightrun/module-versions.jsonl", "", "chore: initialize ledger")

	// Snapshot at Commit 1
	rep1, err := Snapshot(context.Background(), repo, nil)
	if err != nil {
		t.Fatalf("Snapshot 1: %v", err)
	}

	// Verify derived revisions
	if rev, ok := rep1.DocRev("docs/architecture.md"); !ok || rev != 1 {
		t.Errorf("rep1.DocRev(docs/architecture.md) = (%d, %v), want (1, true)", rev, ok)
	}
	if rev, ok := rep1.DocRev("docs/fak/edge-quickstart.md"); !ok || rev != 1 {
		t.Errorf("rep1.DocRev(docs/fak/edge-quickstart.md) = (%d, %v), want (1, true)", rev, ok)
	}
	if rev, ok := rep1.DocRev("docs/fak"); !ok || rev != 1 {
		t.Errorf("rep1.DocRev(docs/fak) = (%d, %v), want (1, true)", rev, ok)
	}
	if _, ok := rep1.DocRev("docs/nightrun/module-versions.jsonl"); ok {
		t.Errorf("rep1.DocRev(docs/nightrun/module-versions.jsonl) returned true, want false")
	}

	// Stamp ledger rows
	rows1 := DeltaRows(rep1, nil, "2026-08-01T12:00:00Z")
	if len(rows1) < 2 {
		t.Fatalf("rows1 len = %d, want at least 2 docs rows", len(rows1))
	}
	ledgerBytes, err := AppendLines(rows1)
	if err != nil {
		t.Fatalf("AppendLines: %v", err)
	}

	// Query from ledger
	if rev, ok := DocRevFromLedger(ledgerBytes, "docs/architecture.md"); !ok || rev != 1 {
		t.Errorf("DocRevFromLedger(docs/architecture.md) = (%d, %v), want (1, true)", rev, ok)
	}
	if rev, ok := DocRevFromLedger(ledgerBytes, "docs/fak/edge-quickstart.md"); !ok || rev != 1 {
		t.Errorf("DocRevFromLedger(docs/fak/edge-quickstart.md) = (%d, %v), want (1, true)", rev, ok)
	}

	// Commit 2: Update docs/architecture.md only
	commitFileMV(t, repo, "docs/architecture.md", "# Architecture\nUpdated\n", "docs: update architecture")

	// Snapshot at Commit 2
	rep2, err := Snapshot(context.Background(), repo, nil)
	if err != nil {
		t.Fatalf("Snapshot 2: %v", err)
	}

	// docs/architecture.md rev should now be 2; docs/fak should still be 1
	if rev, ok := rep2.DocRev("docs/architecture.md"); !ok || rev != 2 {
		t.Errorf("rep2.DocRev(docs/architecture.md) = (%d, %v), want (2, true)", rev, ok)
	}
	if rev, ok := rep2.DocRev("docs/fak/edge-quickstart.md"); !ok || rev != 1 {
		t.Errorf("rep2.DocRev(docs/fak/edge-quickstart.md) = (%d, %v), want (1, true)", rev, ok)
	}

	// Append delta rows to ledger
	rows2 := DeltaRows(rep2, ledgerBytes, "2026-08-02T12:00:00Z")
	// Only docs/architecture.md moved
	if len(rows2) != 1 || rows2[0].Module != "docs/architecture.md" || rows2[0].Rev != 2 {
		t.Fatalf("DeltaRows 2 = %+v, want only docs/architecture.md at rev 2", rows2)
	}
	appended, err := AppendLines(rows2)
	if err != nil {
		t.Fatalf("AppendLines 2: %v", err)
	}
	ledgerBytes = append(ledgerBytes, appended...)

	// Query from cumulative ledger
	if rev, ok := DocRevFromLedger(ledgerBytes, "docs/architecture.md"); !ok || rev != 2 {
		t.Errorf("DocRevFromLedger after delta = (%d, %v), want (2, true)", rev, ok)
	}
	if rev, ok := DocRevFromLedger(ledgerBytes, "architecture.md"); !ok || rev != 2 {
		t.Errorf("DocRevFromLedger relative = (%d, %v), want (2, true)", rev, ok)
	}
	if rev, ok := DocRevFromLedger(ledgerBytes, "docs/fak"); !ok || rev != 1 {
		t.Errorf("DocRevFromLedger docs/fak = (%d, %v), want (1, true)", rev, ok)
	}
}

// TestDocsRealRepoNightrunLedger reads the repository's real module-versions.jsonl
// if present and validates that DocRevFromLedger successfully queries real docs.
func TestDocsRealRepoNightrunLedger(t *testing.T) {
	ledgerPath := filepath.Join("..", "..", "docs", "nightrun", "module-versions.jsonl")
	data, err := os.ReadFile(ledgerPath)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("docs/nightrun/module-versions.jsonl not present")
		}
		t.Fatalf("read ledger: %v", err)
	}

	// Query several known docs that exist in the trunk repository
	knownDocs := []string{
		"docs/FAQ.md",
		"docs/cli-reference.md",
		"docs/HARDWARE-MATRIX.md",
		"docs/architecture.md",
		"docs/fak",
		"docs/adoption",
	}

	for _, doc := range knownDocs {
		rev, ok := DocRevFromLedger(data, doc)
		if !ok || rev <= 0 {
			t.Errorf("DocRevFromLedger(real, %q) = (%d, %v), want rev > 0 and ok=true", doc, rev, ok)
		}
	}

	docRevs := DocRevsFromLedger(data)
	if len(docRevs) < 50 {
		t.Errorf("DocRevsFromLedger(real) returned %d docs, want at least 50", len(docRevs))
	}
}

// TestDocsLedgerPrefixSupport verifies that if a ledger contains rows with either
// "docs/..." or "docs:..." module names, lookups succeed transparently.
func TestDocsLedgerPrefixSupport(t *testing.T) {
	const colonLedger = `{"schema":"fak-module-versions/1","ts":"2026-08-01T00:00:00Z","module":"docs:architecture.md","kind":"docs","rev":4,"version":"r4+gh01"}
{"schema":"fak-module-versions/1","ts":"2026-08-01T00:00:00Z","module":"docs:fak","kind":"docs","rev":8,"version":"r8+gh01"}
`
	colonBytes := []byte(colonLedger)

	if rev, ok := DocRevFromLedger(colonBytes, "docs/architecture.md"); !ok || rev != 4 {
		t.Errorf("DocRevFromLedger(colon, docs/architecture.md) = (%d, %v), want (4, true)", rev, ok)
	}
	if rev, ok := DocRevFromLedger(colonBytes, "docs:fak"); !ok || rev != 8 {
		t.Errorf("DocRevFromLedger(colon, docs:fak) = (%d, %v), want (8, true)", rev, ok)
	}
	if rev, ok := DocRevFromLedger(colonBytes, "docs/fak/edge-quickstart.md"); !ok || rev != 8 {
		t.Errorf("DocRevFromLedger(colon, docs/fak/edge-quickstart.md) = (%d, %v), want (8, true)", rev, ok)
	}
}
