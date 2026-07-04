package devindex

import (
	"strings"
	"testing"
)

// TestVerbCatalogStyleCleanAtHead is the gate itself, run in make ci: the LIVE verb
// catalog carries no style debt outside the frozen baseline. A new uncataloged verb,
// an over-width synopsis, a sentence-case lead, a trailing period, or unbalanced
// notation makes this red — which is the point (#2249): the quality bar stops being
// advisory. Retire a finding by curating the verb (verbManifest) or fixing the
// synopsis, never by widening the baseline.
func TestVerbCatalogStyleCleanAtHead(t *testing.T) {
	c, err := Load(FindRoot("."))
	if err != nil {
		t.Skipf("no repo root (%v); skipping live verb-style gate", err)
	}
	viol := CheckVerbCatalogStyle(c.Verbs())
	if len(viol) == 0 {
		return
	}
	for _, v := range viol {
		t.Errorf("verb %q: %s — %s (curate the verb or fix its synopsis; do NOT add to verbStyleBaseline)", v.Verb, v.Kind, v.Detail)
	}
}

// TestVerbStyleBaselineShrinksOnly enforces the shrinks-only ratchet: every frozen
// baseline entry must still name a REAL current violation. A stale entry (a verb
// that has since been curated or removed) is refused, so the baseline can only get
// smaller as debt is retired — it can never be padded to silence a fresh finding.
func TestVerbStyleBaselineShrinksOnly(t *testing.T) {
	c, err := Load(FindRoot("."))
	if err != nil {
		t.Skipf("no repo root (%v); skipping baseline ratchet check", err)
	}
	live := map[string]bool{}
	for _, v := range rawVerbCatalogStyle(c.Verbs()) {
		live[v.key()] = true
	}
	for key := range verbStyleBaseline {
		if !live[key] {
			verb, kind, _ := strings.Cut(key, "\t")
			t.Errorf("stale baseline entry %q/%s no longer violates — delete it from verbStyleBaseline (shrinks-only)", verb, kind)
		}
	}
}

// TestVerbStyleFlagsSyntheticDefects proves the checker actually fires: a scratch
// catalog with one clean verb and one verb per defect class yields exactly the
// expected findings (baseline-agnostic, via verbStyleViolations).
func TestVerbStyleFlagsSyntheticDefects(t *testing.T) {
	cases := []struct {
		verb Verb
		want VerbStyleKind // "" => expect no violation
	}{
		{Verb{Name: "clean", Synopsis: "run a live agent turn through the kernel (offline or against a provider)"}, ""},
		{Verb{Name: "sym", Synopsis: "= fak ps --watch (the live process-table top mode)"}, ""},            // symbol lead OK
		{Verb{Name: "proper", Synopsis: "AILuminate safety-benchmark runner (describe/eval/compare)"}, ""}, // acronym lead OK
		{Verb{Name: "arrow", Synopsis: "fold orient->plan->act->verify->ship->learn into one index"}, ""},  // -> is not notation
		{Verb{Name: "placeholder", Synopsis: "not yet cataloged — `fak placeholder -h` for usage"}, VerbStyleUncataloged},
		{Verb{Name: "empty", Synopsis: ""}, VerbStyleUncataloged},
		{Verb{Name: "wide", Synopsis: strings.Repeat("x", VerbSynopsisMaxRunes+1)}, VerbStyleSynopsisWidth},
		{Verb{Name: "lead", Synopsis: "Run a live agent turn through the kernel"}, VerbStyleSynopsisLead},
		{Verb{Name: "period", Synopsis: "run a live agent turn through the kernel."}, VerbStyleTrailingPeriod},
		{Verb{Name: "paren", Synopsis: "run a live agent turn (offline or against a provider"}, VerbStyleNotation},
		{Verb{Name: "bracket", Synopsis: "run a live agent turn [offline"}, VerbStyleNotation},
	}
	for _, tc := range cases {
		got := verbStyleViolations(tc.verb)
		if tc.want == "" {
			if len(got) != 0 {
				t.Errorf("verb %q: expected clean, got %+v", tc.verb.Name, got)
			}
			continue
		}
		found := false
		for _, v := range got {
			if v.Kind == tc.want {
				found = true
			}
		}
		if !found {
			t.Errorf("verb %q: expected a %s finding, got %+v", tc.verb.Name, tc.want, got)
		}
	}
}

// TestVerbStyleGateGrandfathersButRedsOnNew proves the two halves of the ratchet in
// one place: a baselined uncataloged verb is suppressed, but a NEW uncataloged verb
// (not in the baseline) reds the gate.
func TestVerbStyleGateGrandfathersButRedsOnNew(t *testing.T) {
	// Pick any real baseline entry to stand in for a grandfathered gap.
	var grandfathered string
	for key := range verbStyleBaseline {
		grandfathered, _, _ = strings.Cut(key, "\t")
		break
	}
	if grandfathered == "" {
		t.Skip("empty baseline — nothing to assert about grandfathering")
	}
	fallback := "not yet cataloged — `fak x -h` for usage"
	verbs := []Verb{
		{Name: grandfathered, Synopsis: fallback},        // baselined -> suppressed
		{Name: "brand-new-verb-xyz", Synopsis: fallback}, // NOT baselined -> red
	}
	viol := CheckVerbCatalogStyle(verbs)
	if len(viol) != 1 || viol[0].Verb != "brand-new-verb-xyz" || viol[0].Kind != VerbStyleUncataloged {
		t.Fatalf("gate = %+v, want exactly the new uncataloged verb flagged (grandfathered %q suppressed)", viol, grandfathered)
	}
}
