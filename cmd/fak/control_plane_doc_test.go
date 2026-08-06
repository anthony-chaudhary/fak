package main

// control_plane_doc_test.go — the doc-side gate for the out-of-band operator control
// plane (#2768, epic #2753). docs/operator-control-plane.md is the ONE page that names
// the whole plane; this test is what keeps it from becoming a stale copy of the code.
//
// Three things are pinned:
//
//  1. Completeness + fidelity — every op in sessionctl.Vocabulary() (the closed registry,
//     #2754) appears on the page WITH its declared boundary, witness shape, and every one
//     of its closed refusal tokens. Register a new op and this test names the page as the
//     next edit, exactly as the loop-side #2766 witness table does for behavior.
//  2. Reachability — the page is linked from the three front doors the issue names
//     (llms.txt, INDEX.md, docs/cli-reference.md), so a reader or agent can land on it
//     from the doc map instead of knowing the filename.
//  3. Front-door binding — the control verbs carry the page as their curated devindex
//     doc link, so `fak help session|signal|ps` prints "see also: <the page>".
//
// The `steering` name collision is checked too: the page must keep naming the three
// verbs that share the "steer" root, because documenting the collision (rather than
// renaming a shipped verb) IS this pass's done-condition.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// controlPlaneDocPath is the repo-relative doctrine page, in slash form — the same
// string the devindex Doc links and the front-door indexes carry.
const controlPlaneDocPath = "docs/operator-control-plane.md"

// readRepoFile reads a repo-relative file for these gates. It SKIPS only when the
// checkout itself is not readable (repoRoot() cannot resolve, e.g. a bare-binary test
// run) — never when the file is merely absent, which is the failure this gate exists to
// catch. AGENTS.md is the always-present anchor that tells the two cases apart.
func readRepoFile(t *testing.T, rel string) string {
	t.Helper()
	root := repoRoot()
	if _, err := os.ReadFile(filepath.Join(root, "AGENTS.md")); err != nil {
		t.Skipf("repo checkout not readable from repoRoot()=%q: %v", root, err)
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	return string(b)
}

// TestControlPlaneDocNamesEveryControlOp is the completeness half: the doctrine page
// must name every registered op with the four fixed properties the spine declares. A
// bare mention is not enough — the boundary, the witness shape, and each closed refusal
// token have to be on the page, so a reader gets the contract and not just the verb.
func TestControlPlaneDocNamesEveryControlOp(t *testing.T) {
	doc := readRepoFile(t, controlPlaneDocPath)

	vocab := sessionctl.Vocabulary()
	if len(vocab) == 0 {
		t.Fatal("sessionctl.Vocabulary() is empty; the doc gate would be vacuous")
	}

	for _, spec := range vocab {
		op := string(spec.Op)
		if !strings.Contains(doc, "`"+op+"`") {
			t.Errorf("%s does not name control op `%s`", controlPlaneDocPath, op)
			continue
		}
		if b := string(spec.Boundary); !strings.Contains(doc, b) {
			t.Errorf("op %q: %s is missing its declared boundary %q", op, controlPlaneDocPath, b)
		}
		if w := string(spec.Witness); !strings.Contains(doc, w) {
			t.Errorf("op %q: %s is missing its witness shape %q", op, controlPlaneDocPath, w)
		}
		if len(spec.RefusalReasons) == 0 {
			t.Errorf("op %q has no refusal reasons in the spine", op)
		}
		for _, reason := range spec.RefusalReasons {
			if !strings.Contains(doc, "`"+reason+"`") {
				t.Errorf("op %q: %s is missing closed refusal token `%s`", op, controlPlaneDocPath, reason)
			}
		}
	}
}

// TestControlPlaneDocIsReachableFromTheDocMap is the discoverability half: the page is
// only a front door if the front doors point at it. These are the three surfaces #2768
// names — the agent doc map, the exhaustive repo map, and the CLI reference.
func TestControlPlaneDocIsReachableFromTheDocMap(t *testing.T) {
	for _, index := range []string{"llms.txt", "INDEX.md", "docs/cli-reference.md"} {
		if body := readRepoFile(t, index); !strings.Contains(body, "operator-control-plane.md") {
			t.Errorf("%s does not link %s; the plane is not reachable from the doc map", index, controlPlaneDocPath)
		}
	}
}

// TestControlPlaneFrontDoorVerbsPointAtTheDoc binds the CLI half: each control verb
// carries the doctrine page as its curated doc link, which is what `fak help <verb>`
// renders as "see also:". VerbByName reads the curated manifest directly, so this holds
// with no repo loaded.
func TestControlPlaneFrontDoorVerbsPointAtTheDoc(t *testing.T) {
	cat := &devindex.Catalog{}
	for _, verb := range []string{"session", "signal", "ps", "top"} {
		v, ok := cat.VerbByName(verb)
		if !ok {
			t.Errorf("verb %q has no curated devindex entry", verb)
			continue
		}
		if v.Doc != controlPlaneDocPath {
			t.Errorf("verb %q doc link = %q, want %q (so `fak help %s` names the plane)", verb, v.Doc, controlPlaneDocPath, verb)
		}
	}
}

// TestControlPlaneDocResolvesTheSteeringCollision pins this pass's done-condition. #2768
// says: document the collision, do NOT rename a shipped verb. So the page must keep
// naming all three "steer"-rooted verbs and separate them from the one real control op.
func TestControlPlaneDocResolvesTheSteeringCollision(t *testing.T) {
	doc := readRepoFile(t, controlPlaneDocPath)
	for _, verb := range []string{"`fak steering`", "`fak steer`", "`fak signal <id> steer`", "`fak trajctl`"} {
		if !strings.Contains(doc, verb) {
			t.Errorf("%s does not name %s; the name collision is left unresolved", controlPlaneDocPath, verb)
		}
	}
}
