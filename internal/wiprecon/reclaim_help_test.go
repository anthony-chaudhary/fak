package wiprecon

// reclaim_help_test.go — the HELP/QUEUE AGREEMENT witness (#6010). `fak wip reconcile
// --reclaim` prints, per row, the exact argv that advances it (AdoptArgv, reclaim.go);
// the verb's own help text in cmd/fak/wip.go used to steer the operator to `fak wip land
// <session>` instead. `wip land` is a real verb, so the stale steer misdirected rather
// than erroring: an operator following it lands the delta with NO adoption receipt —
// exactly the two-successors-one-checkpoint race the adoption seam (#5998) closed.
//
// The pin lives here, beside the argv it pins, and DERIVES the expected commands from
// AdoptArgv rather than restating them: a hand-copied expectation is the same drift the
// help text just suffered. It reads the shipped source because the steer is prose in a
// raw string literal, and cmd/fak has no runnable test binary at trunk.

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// wipUsageSource returns cmd/fak/wip.go's bytes, walking up from this test file to the
// module root. A missing file is a hard failure: this witness silently passing because it
// could not find what it guards is the failure mode it exists to prevent.
func wipUsageSource(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Skip("cannot resolve caller path")
	}
	// self = <root>/internal/wiprecon/reclaim_help_test.go → up three to the root.
	root := filepath.Dir(filepath.Dir(filepath.Dir(self)))
	b, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "wip.go"))
	if err != nil {
		t.Fatalf("read cmd/fak/wip.go: %v", err)
	}
	return string(b)
}

// reclaimHelpParagraph slices the `--reclaim` paragraph out of `fak wip`'s usage text.
// The markers are the paragraph's own first words and those of the next paragraph; if
// either moves, the test FAILS rather than degrading into a whole-file scan that would
// pass on any `wip land` mention elsewhere in the usage (there are several, and they are
// correct — they describe `wip land` itself).
func reclaimHelpParagraph(t *testing.T, src string) string {
	t.Helper()
	const start, end = "With --reclaim,", "With --file-ticket,"
	i := strings.Index(src, start)
	if i < 0 {
		t.Fatalf("cmd/fak/wip.go: no %q paragraph in the wip usage text; re-aim this pin", start)
	}
	j := strings.Index(src[i:], end)
	if j < 0 {
		t.Fatalf("cmd/fak/wip.go: %q paragraph is not followed by %q; re-aim this pin", start, end)
	}
	return src[i : i+j]
}

// TestReclaimHelpSteersToTheAdoptArgvTheQueuePrints: the --reclaim help must name the
// SAME command the queue prints under it — the adopt argv for an unclaimed row, and the
// resume argv for one this session already holds (a held row's argv is nil, and the help
// says so in prose). Two instructions for one row is worse than none, because the wrong
// one still works.
func TestReclaimHelpSteersToTheAdoptArgvTheQueuePrints(t *testing.T) {
	para := reclaimHelpParagraph(t, wipUsageSource(t))
	const placeholder = "<session>"
	for _, tc := range []struct {
		name string
		row  ReclaimRow
	}{
		{"unclaimed row", ReclaimRow{Session: placeholder}},
		{"row this session holds", ReclaimRow{Session: placeholder, AdoptedBy: "me", AdoptedMine: true}},
	} {
		argv := AdoptArgv(tc.row)
		if len(argv) == 0 {
			t.Fatalf("%s: AdoptArgv returned no command; this pin assumes it is actionable", tc.name)
		}
		want := "fak " + strings.Join(argv, " ")
		if !strings.Contains(para, want) {
			t.Errorf("the --reclaim help does not name %q for a %s; the queue prints it per row, so the help must agree:\n%s", want, tc.name, para)
		}
	}
}

// TestReclaimHelpNeverSteersToWipLand: the specific stale steer. `wip land` commits the
// delta without an adoption receipt, so naming it here tells an operator to take the one
// action the recovery queue exists to make unnecessary.
func TestReclaimHelpNeverSteersToWipLand(t *testing.T) {
	para := reclaimHelpParagraph(t, wipUsageSource(t))
	if strings.Contains(para, "wip land") {
		t.Errorf("the --reclaim help still steers to 'wip land' — landing a reclaimed delta without an adoption receipt is the double-recovery race (#5998):\n%s", para)
	}
}
