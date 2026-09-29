package main

// Regression gate for three Python->Go migrations that deleted the Python tool but
// never dispatched the Go verb, leaving the handler an orphan and every caller
// (CI, make targets, skills) failing with `fak: unknown verb ...`:
//
//	dogfood-coverage        -> cmdDogfoodCoverage       (.github/workflows/dogfood-coverage.yml)
//	check-cache-headlines   -> cmdCheckCacheHeadlines   (make cache-headline-lint)
//	cachedoc-numbers-audit  -> cmdCachedocNumbersAudit  (refresh-cachedoc-numbers skill)
//
// It reads the dispatch switch through devindex's shared scanner (the same
// authority the verb catalog and freshness gate use) and the committed generated
// index, so it never executes a handler (they os.Exit and touch the tree).

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestOrphanMigrationVerbsAreDispatched(t *testing.T) {
	want := map[string]string{
		"dogfood-coverage":       "cmdDogfoodCoverage",
		"check-cache-headlines":  "cmdCheckCacheHeadlines",
		"cachedoc-numbers-audit": "cmdCachedocNumbersAudit",
	}
	root := devindex.FindRoot(".")
	src, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	dispatched := map[string]bool{}
	for _, tok := range devindex.DispatchVerbs(src) {
		dispatched[tok] = true
	}
	generated := map[string]generatedVerb{}
	for _, v := range generatedVerbIndex {
		generated[v.Name] = v
	}
	for verb, handler := range want {
		if !dispatched[verb] {
			t.Errorf("cmd/fak/main.go does not dispatch %q; `fak %s` exits `unknown verb`", verb, verb)
		}
		arm := regexp.MustCompile(`case "` + regexp.QuoteMeta(verb) + `":\s*\n\s*` + handler + `\(args\)`)
		if !arm.Match(src) {
			t.Errorf("dispatch arm for %q does not route to %s(args)", verb, handler)
		}
		g, ok := generated[verb]
		if !ok {
			t.Errorf("generated verb index lacks %q; run `fak-dev index verbs --write-usage`", verb)
			continue
		}
		if g.Tier != string(devindex.TierDev) {
			t.Errorf("verb %q tier = %q, want %q", verb, g.Tier, devindex.TierDev)
		}
		if tier, ok := devindex.TierOf(verb); !ok || tier != devindex.TierDev {
			t.Errorf("devindex.TierOf(%q) = (%q, %v), want (%q, true)", verb, tier, ok, devindex.TierDev)
		}
	}
}
