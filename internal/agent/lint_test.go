package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toollint"
)

// LintFacts assembles the FULL static surface: the kernel view (pure calculate,
// static list_all_airports) plus the catalog's hints, advertised schema, and
// grammar. This regression-guards the agent's own configured surface — it must lint
// to EXACTLY the advertised-but-unenforced (TL004) findings and nothing louder, so a
// future change that, say, marks a write tool readOnly (TL001) or drops the
// get_user schema (a new TL004) shows up here as a diff.
func TestAgentSurfaceLintsToKnownFindings(t *testing.T) {
	Configure()
	facts := LintFacts()
	rep := toollint.Lint(facts)

	if rep.Errors() != 0 || rep.Warnings() != 0 {
		t.Fatalf("agent surface should have no error/warn findings; got %d error %d warn: %+v",
			rep.Errors(), rep.Warnings(), rep.Findings)
	}

	// Exactly book_flight, fetch_policy, search_direct_flight are advertised-but-unenforced.
	gotTL004 := map[string]bool{}
	for _, f := range rep.Findings {
		if f.Code == toollint.AdvertisedUnenforced {
			gotTL004[f.Tool] = true
		}
	}
	want := []string{"book_flight", "fetch_policy", "search_direct_flight"}
	if len(gotTL004) != len(want) {
		t.Fatalf("TL004 set = %v, want %v (all findings: %+v)", gotTL004, want, rep.Findings)
	}
	for _, w := range want {
		if !gotTL004[w] {
			t.Fatalf("expected TL004 on %s; got set %v", w, gotTL004)
		}
	}
}

// The tools with an enforced contract (get_user via schema, convert_currency via
// grammar, calculate via being pure) must NOT be flagged as unenforced — the rule's
// exemptions are real, not luck.
func TestAgentEnforcedToolsNotFlagged(t *testing.T) {
	Configure()
	rep := toollint.Lint(LintFacts())
	exempt := map[string]bool{"get_user_details": true, "convert_currency": true, "calculate": true}
	for _, f := range rep.Findings {
		if exempt[f.Tool] {
			t.Fatalf("%s is enforced (schema/grammar/pure) but was flagged: %s %s", f.Tool, f.Code, f.Message)
		}
	}
}

// The guarded PolicyDenied population must NOT pad the surface with a denied tool
// that is on no fast path: delete_account is policy-denied but not pure/static/in the
// catalog, so it must stay out of the facts and the live surface must have no TL008.
func TestAgentDeniedUnregisteredToolNotInSurface(t *testing.T) {
	Configure()
	facts := LintFacts()
	for _, f := range facts {
		if f.Name == "delete_account" {
			t.Fatalf("delete_account (denied but on no fast path) must not pad the lint surface: %+v", f)
		}
	}
	for _, fnd := range toollint.Lint(facts).Findings {
		if fnd.Code == toollint.DenyFastPathBypass {
			t.Fatalf("live agent surface should have no TL008 (no denied tool is on a fast path): %+v", fnd)
		}
	}
}

// LintFacts must attach declared hints from the classifier: a read-only catalog tool
// carries readOnly+idempotent; the write tool carries destructive. Without these the
// hint-shaped rules would be blind.
func TestLintFactsCarryClassifierHints(t *testing.T) {
	Configure()
	by := map[string]toollint.ToolFacts{}
	for _, f := range LintFacts() {
		by[f.Name] = f
	}
	search, ok := by["search_direct_flight"]
	if !ok || !search.Hints.DeclaredReadOnly || !search.Hints.ReadOnly {
		t.Fatalf("search_direct_flight should carry a declared readOnly hint: %+v", search.Hints)
	}
	book, ok := by["book_flight"]
	if !ok || !book.Hints.DeclaredDestructive || !book.Hints.Destructive {
		t.Fatalf("book_flight should carry a declared destructive hint: %+v", book.Hints)
	}
}
