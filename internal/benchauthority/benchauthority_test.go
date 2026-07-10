package benchauthority

import (
	"strings"
	"testing"
)

// TestMdCell asserts the markdown-cell sanitizer: newlines collapse to spaces,
// raw pipes are escaped, and the result is trimmed.
func TestMdCell(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"trims surrounding whitespace", "  hello  ", "hello"},
		{"escapes a pipe", "a|b", "a\\|b"},
		{"escapes multiple pipes", "a|b|c", "a\\|b\\|c"},
		{"newline becomes space", "line1\nline2", "line1 line2"},
		{"pipe and newline together", "a|b\nc", "a\\|b c"},
		{"leading/trailing newline trimmed", "\nx\n", "x"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := mdCell(tc.in); got != tc.want {
				t.Fatalf("mdCell(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestOrNone asserts empty/whitespace-only strings render as "none" while any
// string with real content is returned verbatim (including surrounding spaces).
func TestOrNone(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"empty is none", "", "none"},
		{"spaces only is none", "   ", "none"},
		{"tab only is none", "\t", "none"},
		{"real content unchanged", "claude", "claude"},
		{"content with spaces preserved verbatim", " x ", " x "},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := orNone(tc.in); got != tc.want {
				t.Fatalf("orNone(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestStatusBadge asserts the badge shows the bare status token, and appends the
// finer provenance in parentheses only when Provenance is set.
func TestStatusBadge(t *testing.T) {
	cases := []struct {
		name string
		in   Claim
		want string
	}{
		{"bare status", Claim{Status: Measured}, "`MEASURED`"},
		{"verified bare", Claim{Status: Verified}, "`VERIFIED`"},
		{"with provenance", Claim{Status: Verified, Provenance: "WITNESSED"}, "`VERIFIED` (WITNESSED)"},
		{"gated bare", Claim{Status: Gated}, "`GATED`"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := statusBadge(tc.in); got != tc.want {
				t.Fatalf("statusBadge(%+v) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestRenderCard asserts a fully-populated claim renders its anchor, headline,
// provenance meta line, artifact/reproduce/dossier links, and fence bullets.
func TestRenderCard(t *testing.T) {
	c := Claim{
		ID:         "ctx-horizon",
		Title:      "Context horizon",
		Headline:   "60.3x",
		Status:     Measured,
		Provenance: "OBSERVED",
		Model:      "claude",
		Baseline:   "no-cache",
		Commit:     "abc123",
		Artifact:   "artifacts/ctx.json",
		Reproduce:  "fak cachevalue report",
		Bench:      "cachevalue",
		Section:    "deep-dossier",
		Fences:     []string{"fence one", "fence two"},
	}
	var b strings.Builder
	renderCard(&b, 5, c)
	got := b.String()

	wantSubs := []string{
		`#### 5. Context horizon <a id="ledger-ctx-horizon"></a>`,
		"**60.3x** — `MEASURED` (OBSERVED)",
		"Model claude · vs no-cache · Commit `abc123` · Bench `cachevalue`",
		"- Artifact: [`artifacts/ctx.json`](artifacts/ctx.json)",
		"- Reproduce: `fak cachevalue report`",
		"- Full dossier: [deep-dossier](#deep-dossier)",
		"Fences:",
		"- fence one",
		"- fence two",
	}
	for _, sub := range wantSubs {
		if !strings.Contains(got, sub) {
			t.Errorf("renderCard output missing %q\n--- full output ---\n%s", sub, got)
		}
	}
}

// TestRenderCardRetractedAndEmptyFields asserts empty provenance fields fall back
// to "none", the Bench segment is omitted when empty, and a Retracted claim shows
// its replacement notice.
func TestRenderCardRetractedAndEmptyFields(t *testing.T) {
	c := Claim{
		ID:          "old-num",
		Title:       "Old claim",
		Headline:    "9x",
		Status:      Retracted,
		Artifact:    "artifacts/old.json",
		Replacement: "new-num",
	}
	var b strings.Builder
	renderCard(&b, 1, c)
	got := b.String()

	if !strings.Contains(got, "Model none · vs none · Commit `none`") {
		t.Errorf("expected empty fields to render as none; got:\n%s", got)
	}
	if strings.Contains(got, "Bench `") {
		t.Errorf("empty Bench should be omitted; got:\n%s", got)
	}
	if strings.Contains(got, "- Reproduce:") {
		t.Errorf("empty Reproduce should be omitted; got:\n%s", got)
	}
	if !strings.Contains(got, "⚠️ Retracted — superseded by: new-num") {
		t.Errorf("expected retraction notice; got:\n%s", got)
	}
}

// TestBlockDeterministicAndShape asserts Block() is byte-stable across calls and
// emits the scannable table header plus the dossier section heading.
func TestBlockDeterministicAndShape(t *testing.T) {
	a := Block()
	b := Block()
	if a != b {
		t.Fatalf("Block() is not deterministic:\nfirst:\n%s\nsecond:\n%s", a, b)
	}
	wantSubs := []string{
		"| # | Claim | Headline | Status | Model | Baseline |",
		"|---|---|---|---|---|---|",
		"### Claim dossiers",
	}
	for _, sub := range wantSubs {
		if !strings.Contains(a, sub) {
			t.Errorf("Block() missing %q\n--- output ---\n%s", sub, a)
		}
	}
	if !strings.HasSuffix(a, "\n") {
		t.Errorf("Block() should end with a single newline")
	}
}

// TestSpliceErrors asserts Splice refuses (does not guess) when the markers are
// missing or out of order.
func TestSpliceErrors(t *testing.T) {
	cases := []struct {
		name string
		doc  string
	}{
		{"no begin", "prefix\n" + End + "\nsuffix"},
		{"no end", "prefix\n" + Begin + "\nsuffix"},
		{"end before begin", "x " + End + " middle " + Begin + " y"},
		{"neither marker", "just prose with no markers"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := Splice(tc.doc)
			if err == nil {
				t.Fatalf("Splice(%q) expected an error, got nil (out=%q)", tc.doc, out)
			}
			if out != "" {
				t.Fatalf("Splice error case should return empty string, got %q", out)
			}
		})
	}
}

// TestSpliceExtractRoundTrip asserts Splice preserves the hand-authored prose
// outside the markers, and that Extract of a spliced doc equals Block() — the
// exact invariant the freshness gate relies on.
func TestSpliceExtractRoundTrip(t *testing.T) {
	prefix := "# Title\n\nHand-authored intro paragraph.\n\n"
	suffix := "\n\nHand-authored trailing paragraph.\n"
	doc := prefix + Begin + "\n\nstale content that must be replaced\n" + End + suffix

	out, err := Splice(doc)
	if err != nil {
		t.Fatalf("Splice returned unexpected error: %v", err)
	}
	if !strings.HasPrefix(out, prefix) {
		t.Errorf("Splice dropped the leading hand-authored prose:\n%s", out)
	}
	if !strings.HasSuffix(out, suffix) {
		t.Errorf("Splice dropped the trailing hand-authored prose:\n%s", out)
	}
	if strings.Contains(out, "stale content that must be replaced") {
		t.Errorf("Splice failed to replace the old block:\n%s", out)
	}
	if !strings.Contains(out, Begin) || !strings.Contains(out, End) {
		t.Errorf("Splice must keep both markers:\n%s", out)
	}

	inner, ok := Extract(out)
	if !ok {
		t.Fatalf("Extract failed to find markers in spliced doc:\n%s", out)
	}
	if inner != Block() {
		t.Errorf("Extract(Splice(doc)) != Block()\n--- extract ---\n%q\n--- block ---\n%q", inner, Block())
	}
}

// TestExtractNoMarkers asserts Extract reports absence rather than guessing.
func TestExtractNoMarkers(t *testing.T) {
	cases := []struct {
		name string
		doc  string
	}{
		{"empty doc", ""},
		{"no markers", "plain prose"},
		{"begin only", "x " + Begin + " y"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := Extract(tc.doc)
			if ok {
				t.Fatalf("Extract(%q) reported ok=true, want false (got=%q)", tc.doc, got)
			}
			if got != "" {
				t.Fatalf("Extract miss should return empty string, got %q", got)
			}
		})
	}
}

// TestLedgerReturnsIndependentCopy asserts Ledger() hands back a defensive copy:
// its length matches the registry and mutating the returned slice cannot corrupt
// the shared source of truth.
func TestLedgerReturnsIndependentCopy(t *testing.T) {
	first := Ledger()
	if len(first) != len(registry) {
		t.Fatalf("Ledger() len = %d, want %d", len(first), len(registry))
	}
	// Mutating the returned slice's backing array must not leak into registry.
	first = append(first, Claim{ID: "intruder", Title: "should not persist"})
	second := Ledger()
	if len(second) != len(registry) {
		t.Fatalf("Ledger() leaked a mutation: second call len = %d, want %d", len(second), len(registry))
	}
	for _, c := range second {
		if c.ID == "intruder" {
			t.Fatalf("Ledger() returned the shared backing array; mutation leaked")
		}
	}
}
