package architest

// reason_vocabulary_test.go — bind the closed refusal vocabulary to the code that emits it
// (#5608, epic #5601).
//
// dos.toml's [reasons.*] table is one of the strongest things in the repo: a refusal names a
// reason from a fixed set instead of free text, so a consumer can route on it. That only holds
// if the table is TOTAL — every declared reason reachable by some code path. A declared reason
// nothing emits is worse than an undeclared one, because the table reads as complete: an
// operator or an agent sees the row, believes the refusal exists, and waits for a code that can
// never arrive.
//
// One row had drifted that way (ISSUEFANOUT_CONTRACT_REFUSED, declared and documented with no
// emitter). The row is the smaller half of the problem. The larger half is that nothing would
// have caught it, so the next reason declared without a caller drifts identically and silently.
// This is the same binding failclosed_ledger_test.go does for gates and architest_test.go does
// for tier rows, pointed at the reason table.

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

var reasonHeaderRE = regexp.MustCompile(`(?m)^\[reasons\.([A-Za-z0-9_]+)\]`)

// declaredReasons returns every reason token dos.toml declares.
func declaredReasons(t *testing.T, root string) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	var out []string
	seen := map[string]bool{}
	for _, m := range reasonHeaderRE.FindAllStringSubmatch(string(raw), -1) {
		if seen[m[1]] {
			t.Errorf("dos.toml declares [reasons.%s] more than once", m[1])
			continue
		}
		seen[m[1]] = true
		out = append(out, m[1])
	}
	if len(out) == 0 {
		t.Fatal("dos.toml declares no [reasons.*] rows; the vocabulary scan would pass vacuously")
	}
	sort.Strings(out)
	return out
}

// emitterCorpus reads every non-test Go and Python source under the trees that can emit a
// refusal. Tests are excluded deliberately: a reason mentioned ONLY by a test that asserts on it
// is not reachable in production, which is exactly the drift being caught.
func emitterCorpus(t *testing.T, root string) map[string]string {
	t.Helper()
	corpus := map[string]string{}
	for _, tree := range []string{"cmd", "internal", "tools"} {
		base := filepath.Join(root, tree)
		err := filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil // an unreadable corner must not fail the binding closed
			}
			if d.IsDir() {
				switch d.Name() {
				case "testdata", "__pycache__", ".bin", "node_modules":
					return filepath.SkipDir
				}
				return nil
			}
			name := d.Name()
			if strings.HasSuffix(name, "_test.go") || strings.HasSuffix(name, "_test.py") {
				return nil
			}
			if !strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, ".py") {
				return nil
			}
			b, rerr := os.ReadFile(path)
			if rerr != nil {
				return nil
			}
			rel, _ := filepath.Rel(root, path)
			corpus[filepath.ToSlash(rel)] = string(b)
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", base, err)
		}
	}
	if len(corpus) == 0 {
		t.Fatal("emitter corpus is empty; the scan would report every reason unemitted")
	}
	return corpus
}

// unemittedReasonBaseline grandfathers reason rows that have no production emitter TODAY, the
// same way internal/pythongate/baseline.go grandfathers the pre-existing Python tools. The gate
// is a ratchet, not a big-bang cleanup: a NEW reason declared without a caller fails, while
// known debt stays named and visible instead of either reddening the trunk or hiding.
//
// Each entry must say why it is here and what would retire it. Shrinking this map is the work;
// growing it needs a reason as good as these.
var unemittedReasonBaseline = map[string]string{
	// Its declared floor (AGENTS.md) is an architest RULE — TestHotPathHasNoExec — not a
	// production code path, so the token is raised by a test failing rather than by a running
	// binary. The scan below excludes tests on purpose (a reason only a test mentions is not
	// reachable in production), which makes this row a true negative of a different kind rather
	// than the drift #5608 is about. Retire it by having the architest rule emit the token in
	// its failure message, so the code and the vocabulary agree on the name.
	"OUT_OF_DIRECTION": "floor is the architest rule TestHotPathHasNoExec, which does not name the token in its failure",
}

// Every declared reason must be reachable from production code. This is the direction that
// caught the real drift (#5608), and the one a new reason row trips the moment it is added
// without a caller.
func TestEveryDeclaredReasonHasAnEmitter(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	reasons := declaredReasons(t, root)
	corpus := emitterCorpus(t, root)

	emitted := map[string]bool{}
	for _, reason := range reasons {
		for _, body := range corpus {
			if strings.Contains(body, reason) {
				emitted[reason] = true
				break
			}
		}
		if emitted[reason] {
			continue
		}
		if _, grandfathered := unemittedReasonBaseline[reason]; grandfathered {
			continue
		}
		t.Errorf("dos.toml declares [reasons.%s] but no non-test source under cmd/, internal/ or tools/ contains the token: "+
			"the vocabulary promises a refusal no code path can produce, so a consumer routing on reason codes waits for a code that never arrives. "+
			"Either emit it from the floor that owns it, or retire the row together with its AGENTS.md documentation.", reason)
	}

	// The baseline is a ratchet, so it must not rot in the other direction either: a row that
	// HAS gained an emitter has to leave the map, or the map slowly becomes a list of things
	// nobody rechecks.
	for reason := range unemittedReasonBaseline {
		if emitted[reason] {
			t.Errorf("reason %s now has a production emitter — drop it from unemittedReasonBaseline so the ratchet tightens", reason)
		}
	}
}

// The specific row this issue was filed for, asserted by name so the regression is legible
// without re-reading git history: it must be emittable from the floor AGENTS.md names.
func TestIssuefanoutContractRefusedReasonIsEmittedByItsOwnFloor(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	const reason = "ISSUEFANOUT_CONTRACT_REFUSED"

	raw, err := os.ReadFile(filepath.Join(root, "internal", "issuefanout", "issuefanout.go"))
	if err != nil {
		t.Fatalf("read internal/issuefanout/issuefanout.go: %v", err)
	}
	if !strings.Contains(string(raw), reason) {
		t.Fatalf("internal/issuefanout does not carry %s, the reason AGENTS.md names it as the floor for — "+
			"the refusal would classify correctly and still name no code", reason)
	}
}
