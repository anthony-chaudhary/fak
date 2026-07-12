package marketing

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// enterprise_positioning_test.go — the witness that binds the shipped enterprise
// positioning page (docs/enterprise-positioning.md, issue #3281, epic #3256,
// workstream C) to the honesty contract its own acceptance names. The page is a
// public, quotable surface: its whole value is that every external statistic is
// provenance-labeled and every fak claim is fenced [SHIPPED]/[TICKETED #NNNN], with
// no shipped-vs-research blur. Nothing enforced that until this test — a docs-only
// edit that un-fenced a claim, dropped an [EXTERNAL] label, thinned the gap→surface
// map below six rows, or led with the deferred EU AI Act deadline would have shipped
// silently. This is the marketing lane's mechanization of the issue's "passing
// `fak claim-check`" acceptance: the honesty rule the standard states, asserted
// against the artifact so it cannot drift back into an overclaim.
//
// It grades the SHAPE of the page's honesty (labels present, fences present, pain
// before the EU clock), not the truth of any single number — the [SHIPPED] fences
// each cite an in-repo witness (a CLAIMS.md row / a test) that owns that rung.

// numberedGapRowRE matches a numbered row of the gap→surface map table
// ("| 1 | ... |"), the one-to-one mapping the acceptance counts.
var numberedGapRowRE = regexp.MustCompile(`(?m)^\|\s*\d+\s*\|.*\|`)

// readEnterprisePositioningPage returns the page text, locating it from this test
// file's own path (the safecommit/reasons_doc_test.go repo-root idiom) so the test
// is runnable from the package dir with no CWD assumption.
func readEnterprisePositioningPage(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed; cannot locate the repo root")
	}
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..")
	pagePath := filepath.Join(repoRoot, "docs", "enterprise-positioning.md")
	b, err := os.ReadFile(pagePath)
	if err != nil {
		t.Fatalf("read enterprise positioning page at %s: %v\n"+
			"issue #3281 acceptance requires this page to ship", pagePath, err)
	}
	if len(b) < 1000 {
		t.Fatalf("enterprise positioning page is only %d bytes; a stub, not the shipped page", len(b))
	}
	return string(b)
}

// TestEnterprisePositioningHonorsIssue3281HonestyContract asserts the shipped page
// answers, clause by clause, the acceptance of issue #3281. Each subtest is one
// acceptance clause, so a red names exactly which honesty rule drifted.
func TestEnterprisePositioningHonorsIssue3281HonestyContract(t *testing.T) {
	page := readEnterprisePositioningPage(t)
	lower := strings.ToLower(page)

	// Clause: the page states, in its own words, the two governing honesty rules —
	// every external statistic is provenance-labeled, and every fak claim is fenced.
	// Binding the rule text itself means the contract cannot be quietly deleted from
	// the page while the labels are stripped underneath it.
	t.Run("declares the two honesty rules", func(t *testing.T) {
		if !strings.Contains(page, "Every external statistic is provenance-labeled") {
			t.Error("page does not state the provenance-labeling rule (#3281: every external stat provenance-labeled)")
		}
		if !strings.Contains(page, "Every fak claim is fenced") {
			t.Error("page does not state the fencing rule (#3281: every fak claim fenced [SHIPPED]/[TICKETED])")
		}
	})

	// Clause: every external statistic is provenance-labeled. The page tags each
	// external figure `[EXTERNAL ...]`; a healthy page carries several. A drop toward
	// zero means market stats are appearing unlabeled — the overclaim the rule forbids.
	t.Run("external stats carry [EXTERNAL] provenance labels", func(t *testing.T) {
		n := strings.Count(page, "[EXTERNAL")
		if n < 6 {
			t.Errorf("only %d [EXTERNAL] provenance labels; the 'why now' triggers + supply-chain "+
				"stat should each be labeled (want >= 6) — an unlabeled market stat is an overclaim", n)
		}
		// The named sources the issue's "why now" list calls out must still be present
		// and attributed, not laundered into unsourced assertions.
		for _, src := range []string{"Gartner", "a16z"} {
			if !strings.Contains(page, src) {
				t.Errorf("named external source %q missing; the 'why now' triggers lost their attribution", src)
			}
		}
	})

	// Clause: every fak claim is fenced — [SHIPPED] (with an in-repo witness) or
	// [TICKETED #NNNN] (named, planned, not built). Both fence kinds must be present,
	// because the page's honesty is precisely that it does NOT blur the two.
	t.Run("fak claims are fenced SHIPPED or TICKETED", func(t *testing.T) {
		if !strings.Contains(page, "[SHIPPED]") {
			t.Error("no [SHIPPED] fence; a page with only ticketed surfaces or bare claims fails the fencing rule")
		}
		if !strings.Contains(page, "[TICKETED #") {
			t.Error("no [TICKETED #NNNN] fence; an all-shipped page hides the ticketed surfaces (#3273/#3274/#3279) — a shipped-vs-research blur")
		}
	})

	// Clause: the page maps at least six named market gaps to specific
	// shipped-or-ticketed fak surfaces, and NO row is a bare "yes" — each numbered
	// gap row carries a fence token.
	t.Run("maps >= 6 gaps, each row fenced", func(t *testing.T) {
		rows := numberedGapRowRE.FindAllString(page, -1)
		if len(rows) < 6 {
			t.Fatalf("gap->surface map has %d numbered rows; acceptance requires >= 6 named gaps mapped to surfaces", len(rows))
		}
		for _, row := range rows {
			if !(strings.Contains(row, "[SHIPPED]") ||
				strings.Contains(row, "[TICKETED") ||
				strings.Contains(row, "Partial")) {
				t.Errorf("gap row is unfenced (a bare yes): %s", strings.TrimSpace(row))
			}
		}
	})

	// Clause: lead with operational pain (cost, breach, pilot-unblocking), NOT the
	// deferred EU AI Act deadline. Enforced two ways: operational-pain framing appears
	// before the first EU-AI-Act mention, AND the page explicitly demotes the EU
	// deadline as slipped — the honesty fence the issue says to lead away from.
	t.Run("leads with operational pain, not the EU deadline", func(t *testing.T) {
		euIdx := strings.Index(lower, "eu ai act")
		if euIdx < 0 {
			// The page need not mention the EU Act at all; if it does, ordering matters.
			return
		}
		painIdx := -1
		for _, anchor := range []string{"runaway agent cost", "runaway-cost", "cost", "breach", "pilot"} {
			if i := strings.Index(lower, anchor); i >= 0 && (painIdx < 0 || i < painIdx) {
				painIdx = i
			}
		}
		if painIdx < 0 || painIdx >= euIdx {
			t.Error("EU AI Act is mentioned before any operational-pain framing; #3281 says lead with cost/breach/pilot pain, not the deadline")
		}
		if !strings.Contains(lower, "slipped to 2027") {
			t.Error("page does not demote the EU AI Act deadline as slipped; the 'no compliance countdown' honesty fence is the one to lead away from")
		}
	})

	// Clause: the page names its verifier and its provenance home — the `claim-check`
	// honesty grader it says grades it, and the epic (#3256) whose research brief every
	// [EXTERNAL] figure is drawn from. Both keep the page tied to a re-derivable source.
	t.Run("cites its grader and its epic provenance", func(t *testing.T) {
		if !strings.Contains(page, "claim-check") {
			t.Error("page does not name the `fak claim-check` honesty grader from its own acceptance")
		}
		if !strings.Contains(page, "#3256") {
			t.Error("page does not cite epic #3256, the provenance home of every [EXTERNAL] figure")
		}
	})
}
