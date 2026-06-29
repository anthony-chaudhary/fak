package loopmap

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// repoRoot walks up from the package dir (go test CWD) until it finds go.mod, so the
// witnesses can read cmd/fak/main.go (the real verb registry) and the committed doc.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repo root (go.mod) above the package dir")
		}
		dir = parent
	}
}

// TestScriptedScenario is the DoD witness: a COLD mid-tier agent, given only the map,
// reaches for the correct verb at each loop stage. Each row is a free-text "what now?"
// an agent would actually type; Ask must return the documented verb. Witnessed, not
// asserted — the map either routes these or the gate reds.
func TestScriptedScenario(t *testing.T) {
	scenarios := []struct {
		situation string
		wantVerb  string
		wantStage string
	}{
		{"I am about to claim the work is done", "dos verify", "verify"},
		{"about to fan out several agents in parallel", "dos arbitrate", "plan"},
		{"I am about to trust a recalled memory", "fak recall", "orient"},
		{"the tree is green, time to ship", "fak commit", "ship"},
		{"this tool call looks malformed", "fak guard", "act"},
		{"the session is over, capture the outcome", "fak sessions", "learn"},
	}
	for _, s := range scenarios {
		got, ok := Ask(s.situation)
		if !ok {
			t.Errorf("Ask(%q): no match; want %q", s.situation, s.wantVerb)
			continue
		}
		if got.Verb != s.wantVerb {
			t.Errorf("Ask(%q) = %q (stage %s); want %q (stage %s)",
				s.situation, got.Verb, got.Stage, s.wantVerb, s.wantStage)
		}
		if got.Stage != s.wantStage {
			t.Errorf("Ask(%q) routed to stage %q; want %q", s.situation, got.Stage, s.wantStage)
		}
	}
}

// TestAskMissReturnsFalse pins the honest negative: a situation the map does not cover
// returns false, never a confident wrong guess.
func TestAskMissReturnsFalse(t *testing.T) {
	if e, ok := Ask("what is the airspeed velocity of an unladen swallow"); ok {
		t.Errorf("Ask(unrelated) matched %q; want no match", e.Verb)
	}
}

// TestEveryStageCovered requires at least one map row per canonical loop stage, so the
// affordance answers "what now?" at every point in the loop, not just the easy ones.
func TestEveryStageCovered(t *testing.T) {
	for _, st := range Stages() {
		if len(ForStage(st)) == 0 {
			t.Errorf("loop stage %q has no map entry — the affordance is blind at that stage", st)
		}
	}
}

// TestNoFakVerbDrift is the "generated from the real verb registry" witness: every
// `fak <verb>` the map names must be a real case in cmd/fak/main.go. Rename or remove a
// verb and this reds — the map cannot silently rot away from the binary's tool surface.
func TestNoFakVerbDrift(t *testing.T) {
	main, err := os.ReadFile(filepath.Join(repoRoot(t), "cmd", "fak", "main.go"))
	if err != nil {
		t.Fatalf("read cmd/fak/main.go: %v", err)
	}
	registry := map[string]struct{}{}
	for _, m := range regexp.MustCompile(`case "([a-z0-9-]+)"`).FindAllStringSubmatch(string(main), -1) {
		registry[m[1]] = struct{}{}
	}
	if len(registry) == 0 {
		t.Fatal("parsed zero verbs from cmd/fak/main.go — the registry regex drifted")
	}
	for _, v := range FakVerbs() {
		if _, ok := registry[v]; !ok {
			t.Errorf("map names `fak %s` but it is not a real verb in cmd/fak/main.go (drift)", v)
		}
	}
}

// TestDocListsEveryEntry keeps the committed doc snapshot honest against the data: the
// human-facing map (docs/fak/loop-tool-map.md) must mention every stage and every verb,
// so the doc and the queryable data cannot diverge.
func TestDocListsEveryEntry(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join(repoRoot(t), "docs", "fak", "loop-tool-map.md"))
	if err != nil {
		t.Fatalf("read docs/fak/loop-tool-map.md: %v", err)
	}
	body := string(doc)
	for _, e := range Map() {
		if !strings.Contains(body, e.Verb) {
			t.Errorf("doc is missing verb %q (stage %s) — regenerate docs/fak/loop-tool-map.md", e.Verb, e.Stage)
		}
		if !strings.Contains(strings.ToLower(body), e.Stage) {
			t.Errorf("doc is missing stage %q", e.Stage)
		}
	}
}
