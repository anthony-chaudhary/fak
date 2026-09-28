package devindex

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestVerbBinaryMatchesRealDispatch is the #13093 invariant: every verb Verbs()
// emits must be labeled with a binary whose dispatch switch actually routes it. A
// verb tagged `fak-dev` that cmd/fak-dev/main.go never cases is exactly the phantom
// capability card the issue names — `fak-dev search` for a verb only cmd/fak routes.
// The check reads the same dispatch scans production uses (mainDispatchVerbs /
// devDispatchVerbs), so the label and the proof can never disagree.
func TestVerbBinaryMatchesRealDispatch(t *testing.T) {
	root := repoRoot(t)
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load(%q): %v", root, err)
	}
	mainB, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak/main.go (%v)", err)
	}
	devB, err := os.ReadFile(filepath.Join(root, "cmd", "fak-dev", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak-dev/main.go (%v)", err)
	}
	mainSet := tokenSet(mainDispatchVerbs(mainB))
	devSet := tokenSet(devDispatchVerbs(devB))

	for _, v := range c.Verbs() {
		switch v.Binary {
		case BinaryFak:
			if !mainSet[v.Name] {
				t.Errorf("verb %q tagged %s but cmd/fak does not dispatch it", v.Name, v.Binary)
			}
		case BinaryFakDev:
			if !devSet[v.Name] {
				t.Errorf("verb %q tagged %s but cmd/fak-dev does not dispatch it (#13093 phantom card)", v.Name, v.Binary)
			}
		default:
			t.Errorf("verb %q has no binary label (want %s or %s)", v.Name, BinaryFak, BinaryFakDev)
		}
	}
}

// TestEveryVerbCanonicalSpellingIsDispatched is the tight form of the #13093 invariant,
// and the one the capability card actually depends on: a card advertises
// `<Binary> <Name>` using the verb's CANONICAL spelling, so it is the canonical spelling
// — not any alias — that must appear in the named binary's switch. Checking a looser
// "any spelling" set would let a card advertise `fak-dev <name>` for a verb cmd/fak-dev
// only routes under a different alias, which exits 2 (the exact phantom class). This
// asserts the card's advertised argv[1] token is real.
func TestEveryVerbCanonicalSpellingIsDispatched(t *testing.T) {
	root := repoRoot(t)
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load(%q): %v", root, err)
	}
	mainB, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak/main.go (%v)", err)
	}
	devB, err := os.ReadFile(filepath.Join(root, "cmd", "fak-dev", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak-dev/main.go (%v)", err)
	}
	mainSet := tokenSet(mainDispatchVerbs(mainB))
	devSet := tokenSet(devDispatchVerbs(devB))

	for _, v := range c.Verbs() {
		set := mainSet
		if v.Binary == BinaryFakDev {
			set = devSet
		}
		if !set[strings.ToLower(v.Name)] {
			t.Errorf("card advertises `%s %s` but %s cannot route %q (aliases: %v)",
				v.Binary, v.Name, v.Binary, v.Name, v.Aliases)
		}
	}
}

// TestSearchVerbIsTaggedFak pins the concrete defect from #13093: `search` is a
// cmd/fak verb (a fleet session-store search), never a fak-dev one.
func TestSearchVerbIsTaggedFak(t *testing.T) {
	root := repoRoot(t)
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load(%q): %v", root, err)
	}
	for _, v := range c.Verbs() {
		if v.Name != "search" {
			continue
		}
		if v.Binary != BinaryFak {
			t.Fatalf("search tagged %q, want %q (cmd/fak-dev cannot route it)", v.Binary, BinaryFak)
		}
		if v.Synopsis == "" || containsAny(v.Synopsis, "repository text corpus") {
			t.Fatalf("search synopsis still describes a repo text corpus: %q", v.Synopsis)
		}
		return
	}
	t.Fatal("search missing from the derived catalog")
}

// TestFallbackVerbBinaryMatchesRealDispatch is the no-switch half of the #13093
// invariant. A catalog rooted where neither cmd/fak/main.go nor cmd/fak-dev/main.go is
// readable — `go run ../fak/cmd/fak-dev feature query` from the companion checkout, or
// an installed binary outside a repo — takes the curated-overlay path, and every card
// it emits must still name a binary whose switch routes the verb. The old fallback read
// the concept tier (TierDev -> fak-dev), but TierDev is the `fak dev <verb>` help
// namespace, not the fak-dev artifact, so ~330 cmd/fak-only verbs (claude,
// claude-mac-fak, code-debt, c, ...) were advertised as `fak-dev <verb>`, which exits 2.
// A curated entry neither switch dispatches is a curation-drift question, not a label
// one, and is skipped here.
func TestFallbackVerbBinaryMatchesRealDispatch(t *testing.T) {
	root := repoRoot(t)
	mainB, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak/main.go (%v)", err)
	}
	devB, err := os.ReadFile(filepath.Join(root, "cmd", "fak-dev", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak-dev/main.go (%v)", err)
	}
	mainSet := tokenSet(mainDispatchVerbs(mainB))
	devSet := tokenSet(devDispatchVerbs(devB))

	fallback := &Catalog{Root: t.TempDir()} // no readable dispatch switch: the overlay path
	if len(fallback.liveDispatchTokens()) != 0 || len(fallback.liveDevDispatchTokens()) != 0 {
		t.Fatal("temp-dir catalog unexpectedly read a dispatch switch; the fallback path is not exercised")
	}
	checked := 0
	for _, v := range fallback.Verbs() {
		n := strings.ToLower(v.Name)
		if !mainSet[n] && !devSet[n] {
			continue
		}
		checked++
		set, other := mainSet, devSet
		if v.Binary == BinaryFakDev {
			set, other = devSet, mainSet
		}
		if !set[n] {
			t.Errorf("fallback card advertises `%s %s` but that binary cannot route it (other binary routes it: %v)",
				v.Binary, v.Name, other[n])
		}
	}
	if checked == 0 {
		t.Fatal("fallback catalog emitted no dispatched verbs; nothing was checked")
	}
}

// TestClaudeVerbsAreTaggedFak pins the concrete defect: `claude` and `claude-mac-fak`
// are dispatched by cmd/fak (`fak claude --help` works; `fak-dev claude` exits 2), so a
// card built from either the live catalog or the no-switch fallback must say `fak`.
func TestClaudeVerbsAreTaggedFak(t *testing.T) {
	for _, name := range []string{"claude", "claude-mac-fak"} {
		if got := binaryForVerb(name, nil); got != BinaryFak {
			t.Errorf("fallback binaryForVerb(%s) = %q, want %q", name, got, BinaryFak)
		}
	}
	root := repoRoot(t)
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load(%q): %v", root, err)
	}
	for _, v := range c.Verbs() {
		if (v.Name == "claude" || v.Name == "claude-mac-fak") && v.Binary != BinaryFak {
			t.Errorf("live catalog tags %s as %q, want %q", v.Name, v.Binary, BinaryFak)
		}
	}
}

// TestBinaryForVerbFallback pins the no-repo fallback: with no readable dev switch the
// label falls to the compiled-in fak-dev inventory (devhandoff.IsCommand -> fak-dev,
// else fak) rather than going empty — never to the concept tier.
func TestBinaryForVerbFallback(t *testing.T) {
	if got := binaryForVerb("index", nil); got != BinaryFakDev {
		t.Errorf("binaryForVerb(index, no dev tokens) = %q, want %q", got, BinaryFakDev)
	}
	if got := binaryForVerb("serve", nil); got != BinaryFak {
		t.Errorf("binaryForVerb(serve, no dev tokens) = %q, want %q", got, BinaryFak)
	}
	// A TierDev verb that only cmd/fak dispatches is still `fak`.
	if got := binaryForVerb("code-debt", nil); got != BinaryFak {
		t.Errorf("binaryForVerb(code-debt, no dev tokens) = %q, want %q", got, BinaryFak)
	}
	if got := binaryForVerb("search", []string{"index", "buildcheck"}); got != BinaryFak {
		t.Errorf("binaryForVerb(search, dev set) = %q, want %q", got, BinaryFak)
	}
	if got := binaryForVerb("index", []string{"index"}); got != BinaryFakDev {
		t.Errorf("binaryForVerb(index, dev set) = %q, want %q", got, BinaryFakDev)
	}
}

// repoRoot walks up from the test's working directory to the checkout root (the dir
// holding go.mod + dos.toml), so the test reads the real dispatch switches.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("no checkout root above the test working directory")
		}
		dir = parent
	}
}

func tokenSet(tokens []string) map[string]bool {
	out := make(map[string]bool, len(tokens))
	for _, t := range tokens {
		out[t] = true
	}
	return out
}

// spellingIn returns the first spelling of v present in set, else the canonical name —
// the dispatch scan keys on the exact alias spelling a `case` line carries.
func spellingIn(v Verb, set map[string]bool) string {
	for _, sp := range v.Spellings() {
		if set[sp] {
			return sp
		}
	}
	return v.Name
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if len(sub) > 0 && len(s) >= len(sub) {
			for i := 0; i+len(sub) <= len(s); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}
