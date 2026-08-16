package hooks

import (
	"os/exec"
	"strings"
	"testing"
)

// gate_brandconsistency_test.go — the BRAND_CONSISTENCY contract. the retired Python checker
// and its _test.py were retired in #6265 (the Go gate had already been the committed checker), so
// this file is now the SOLE owner of the behavior the oracle used to pin. It keeps every one of
// the oracle's assertions: (1) the EXACT golden vectors from check_brand_consistency_test.py —
// its must-flag / must-not-flag samples — replayed against the per-line decision, (2) the
// boundary cases the retired verdict-level differential used to backstop (window, sentence stop,
// word boundary, banner spacing, case), each expectation captured from the live Python before it
// was deleted, (3) the audit() file-walk itself — 1-based line numbers, stripped detail, exempt
// and non-scanned paths skipped, and (4) the live tracked tree is clean (the twin of the Python
// test_live_tree_is_clean).

// brandDrift are primary-descriptor DRIFT lines (fak declared to BE a retired descriptor) — the
// gate MUST flag each. Inherited verbatim from check_brand_consistency_test.py DRIFT.
var brandDrift = []string{
	"`fak` is an agent tool firewall: a single Go binary that sits between an agent and its tools",
	"fak — agent tool firewall",
	`fmt.Fprintf(os.Stderr, "fak - Agent Tool Firewall (Fused Agent Kernel, v%s)")`,
	"`fak` is the tool-call policy gateway for your fleet",
	"fak is a tool call policy gateway",
}

// brandAllowed are legitimate secondary uses (synonym lists, "also described as", the named
// asset) — the gate MUST NOT flag any. Inherited verbatim from check_brand_consistency_test.py
// ALLOWED.
var brandAllowed = []string{
	"`fak` is an **agent kernel** (also described as an *agent tool firewall*): an in-process gate",
	"<sub>Topics: agent kernel · agent tool firewall · AI agent security · prompt injection</sub>",
	"  - agent tool firewall",
	`"alternateName": ["the agent kernel", "agent tool firewall"],`,
	`aria-label="fak — the agent tool firewall: a ~44 second explainer reveal">`,
	"`fak` is the Fused Agent Kernel: a single Go binary that sits between an agent and its tools",
	"It is also described as an agent tool firewall.",
}

func TestBrandConsistency_DriftIsFlagged(t *testing.T) {
	for _, s := range brandDrift {
		if !brandLineViolates(s) {
			t.Errorf("should flag primary-descriptor drift: %q", s)
		}
	}
}

func TestBrandConsistency_AllowedNotFlagged(t *testing.T) {
	for _, s := range brandAllowed {
		if brandLineViolates(s) {
			t.Errorf("should NOT flag legitimate secondary use: %q", s)
		}
	}
}

// TestBrandConsistency_FileFilter pins the scan-scope predicate: exempt files/prefixes and
// non-text extensions are skipped; reader-facing text surfaces are scanned. Mirrors audit()'s
// EXEMPT_FILES / EXEMPT_PREFIXES / SCAN_EXT filter.
func TestBrandConsistency_FileFilter(t *testing.T) {
	cases := []struct {
		rel  string
		want bool
	}{
		{"README.md", true},
		{"docs/note.txt", true},
		{"cmd/fak/main.go", true},
		{"index.html", true},
		{"CITATION.cff", true},
		{"tools/gen_structured_data.py", false},                 // exempt file
		{"llms-full.txt", false},                                // exempt file (generated)
		{"internal/hooks/gate_brandconsistency.go", false},      // exempt: this gate's own source
		{"internal/hooks/gate_brandconsistency_test.go", false}, // exempt: this test's golden vectors
		{"visuals/poster.md", false},                            // exempt prefix
		{"tools/foo.py", false},                                 // not a scanned extension
		{"assets/logo.svg", false},                              // not a scanned extension
	}
	for _, c := range cases {
		if got := brandScanned(c.rel); got != c.want {
			t.Errorf("brandScanned(%q) = %v, want %v", c.rel, got, c.want)
		}
	}
}

// TestBrandConsistency_LiveTreeClean asserts the real tracked tree carries no primary-descriptor
// drift — the Go twin of the Python test_live_tree_is_clean. Skipped outside a git checkout.
func TestBrandConsistency_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateBrandConsistencyTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	if len(findings) != 0 {
		t.Fatalf("primary-descriptor drift on the tracked tree: %+v", findings)
	}
}

// TestBrandConsistency_PrimaryFormBoundaries pins the edges of the primary-descriptor decision
// that the retired Python differential used to backstop. Every want below was captured from the
// live the retired Python checker (PRIMARY_RE + ALLOW_MARKERS) before it was deleted in
// #6265, so these are the oracle's verdicts, not re-derived guesses. They fence the four ways
// the regex can drift: the 40-char lazy window between "fak" and the descriptor, the [^.\n]
// sentence stop, the \bfak\b word boundary, and the banner branch's mandatory spacing.
func TestBrandConsistency_PrimaryFormBoundaries(t *testing.T) {
	cases := []struct {
		line string
		want bool
	}{
		{"fak - agent tool firewall", true},                                      // ASCII-hyphen banner
		{"fak — the agent tool firewall", true},                                  // em-dash banner, optional article
		{"FAK IS AN AGENT TOOL FIREWALL", true},                                  // case-insensitive
		{"fak, the in-process kernel, is a tool call policy gateway", true},      // inside the 40-char window
		{"fak is an tool-call policy gateway", true},                             // hyphenated spelling
		{"   fak is an agent tool firewall   ", true},                            // leading/trailing space is irrelevant
		{"the fak binary is an agent tool firewall", true},                       // fak need not start the line
		{"fak: agent tool firewall", false},                                      // banner branch needs \s+ BEFORE the colon
		{"fak" + strings.Repeat(" x", 25) + " is an agent tool firewall", false}, // past the 40-char window
		{"fak. It is an agent tool firewall", false},                             // [^.\n]: a sentence stop breaks the claim
		{"fakery is an agent tool firewall", false},                              // \bfak\b: not the word "fak"
		{"`fak` is the Fused Agent Kernel and an agent kernel", false},           // the kept primary noun
		{"fak is an agent tool firewall card", false},                            // ALLOW_MARKERS: \bcard\b
	}
	for _, c := range cases {
		if got := brandLineViolates(c.line); got != c.want {
			t.Errorf("brandLineViolates(%q) = %v, want %v", c.line, got, c.want)
		}
	}
}

// TestBrandConsistency_TreeWalk pins audit()'s file walk now that the Python original is gone:
// in-scope files are scanned line by line, findings carry the path and a 1-BASED line number
// with the line stripped, and exempt files, exempt prefixes and unscanned extensions contribute
// nothing — even when they carry the very same drift line.
func TestBrandConsistency_TreeWalk(t *testing.T) {
	drift := "fak is an agent tool firewall"
	tree := treeFromFiles(map[string]string{
		"README.md":                    "intro\n   " + drift + "   \nouttro\n",
		"llms-full.txt":                drift + "\n", // exempt file
		"visuals/poster.md":            drift + "\n", // exempt prefix
		"tools/gen_structured_data.py": drift + "\n", // exempt file (and unscanned ext)
		"docs/design.png":              drift + "\n", // unscanned extension
	})
	findings, err := gateBrandConsistencyTree(tree)
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("want exactly the README drift flagged, got %+v", findings)
	}
	got := findings[0]
	if got.Gate != "BRAND_CONSISTENCY" || got.File != "README.md" || got.Line != 2 || got.Detail != drift {
		t.Errorf("finding = %+v, want gate BRAND_CONSISTENCY, file README.md, line 2, detail %q", got, drift)
	}
}
