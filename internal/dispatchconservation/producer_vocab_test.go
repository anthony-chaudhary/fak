package dispatchconservation

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// The #5866 regression guard, now that a PRODUCER stamps the refined classes.
//
// #5866 was this exact shape: tools/issue_resolve_dispatch.py emitted classes the Go
// vocabulary had never heard of, and ClassifyUnit's
// `if !noCommitReasons[reason] { reason = "unknown" }` folded them straight back into
// the bucket they were invented to escape — so restart_exhausted vanished from every
// report while the sidecars carried it.
//
// #5867 derived clean_exit_no_commit / died_before_epilogue / guard_child_spawn_failed
// HERE, from the log tail, and added their vocabulary rows — but the sweep still
// stamped "unknown", so the refinement existed only inside this package's own report.
// #5870 moves the typing to the WRITER, which puts those three strings on the
// producer's side of the same fold for the first time. This file pins the two halves
// together: the literals the Python writer stamps are read out of its source and
// required to survive ClassifyUnit unfolded. Adding a FOURTH class to the writer
// without its vocabulary row here fails this test rather than silently re-creating
// #5866.
var pythonProducerReasonConsts = []string{
	"NO_COMMIT_CLEAN_EXIT",
	"NO_COMMIT_DIED_BEFORE_EPILOGUE",
	"NO_COMMIT_GUARD_SPAWN_FAILED",
}

// pythonConst extracts a module-level `NAME = "value"` string literal from the sweep.
// A missing constant is a hard failure, not a skip: this test's whole job is to notice
// that the writer's vocabulary moved.
func pythonConst(t *testing.T, name string) string {
	t.Helper()
	src := filepath.Join("..", "..", "tools", "issue_resolve_dispatch.py")
	raw, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read %s: %v", src, err)
	}
	re := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(name) + ` = "([^"]*)"`)
	m := re.FindSubmatch(raw)
	if m == nil {
		t.Fatalf("%s does not define %s — the producer vocabulary this package folds "+
			"against has moved; re-check noCommitReasons before deleting this pin", src, name)
	}
	return string(m[1])
}

func TestPythonProducerNoCommitClassesAreNotFoldedToUnknown(t *testing.T) {
	for _, constName := range pythonProducerReasonConsts {
		reason := pythonConst(t, constName)
		t.Run(reason, func(t *testing.T) {
			dir := t.TempDir()
			log := filepath.Join(dir, "resolve-4242-20260807-120000.log")
			// A body-bearing log, so an unwitnessed unit could never be read as
			// header-only/spawn-failed and skip the witness branch entirely.
			if err := os.WriteFile(log, []byte(
				"# fak-spawn 20260807-120000 issue=4242 lane=dispatch backend=claude\n"+
					"fak-turn trace=abc ok\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			w := `{"claim":"CLAIM_NO_COMMIT","issue":4242,"reason":"` + reason + `"}`
			if err := os.WriteFile(log[:len(log)-4]+".witness", []byte(w), 0o644); err != nil {
				t.Fatal(err)
			}
			u := ClassifyUnit(log, AliveProbe{Scanned: true})
			if u.Outcome != OutcomeNoCommit {
				t.Fatalf("outcome = %q, want %q", u.Outcome, OutcomeNoCommit)
			}
			if u.Reason != reason {
				t.Fatalf("reason = %q, want %q — a class the Python sweep now STAMPS "+
					"was folded away (#5866 all over again); add it to noCommitReasons",
					u.Reason, reason)
			}
		})
	}
}

// The write-time and read-time refinements must key on the SAME evidence, or a sidecar
// stamped by the sweep and a sidecar refined by this package would name the same run
// differently — the seam #5870 exists to close. Pin the marker literals across the
// language boundary; over the 1278 retained stamped-`unknown` sidecars the two agreed
// 1278/1278, and that only stays true while these strings match.
func TestRefinementMarkersMatchThePythonWriter(t *testing.T) {
	for _, tc := range []struct{ pyName, goVal string }{
		{"_GUARD_EPILOGUE_MARKER", guardEpilogueMarker},
		{"_GUARD_CHILD_SPAWN_FAILED_MARKER", guardChildSpawnFailedMarker},
	} {
		if got := pythonConst(t, tc.pyName); got != tc.goVal {
			t.Errorf("%s = %q, Go marker = %q — the two refinements would disagree",
				tc.pyName, got, tc.goVal)
		}
	}
}

// The reader's refinement must keep working for the 1278 sidecars already on disk with
// a bare "unknown": #5870 types NEW sidecars, it does not (and must not) rewrite
// history. Both paths must land on the same class for the same underlying run.
func TestHistoricUnknownStillRefinesToTheSameClassAsTheWriter(t *testing.T) {
	section := "── guard · audit ──────────────────────────────────────\n"
	for _, tc := range []struct{ name, tail, want string }{
		{"clean exit", section + "  refused                    0\n", ReasonCleanExitNoCommit},
		{"died mid-turn", "fak-turn trace=e3f9 in-flight saved=12k tok\n", ReasonDiedBeforeEpilogue},
		{"spawn failed", section + "fak guard: could not run \"claude\": exec: not found\n",
			ReasonGuardChildSpawnFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			log := filepath.Join(dir, "resolve-4242-20260807-120000.log")
			body := "# fak-spawn 20260807-120000 issue=4242 lane=dispatch backend=claude\n" + tc.tail
			if err := os.WriteFile(log, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			// A HISTORIC sidecar: reason "unknown", the way the sweep stamped it before
			// #5870. The reader must still name it.
			if err := os.WriteFile(log[:len(log)-4]+".witness",
				[]byte(`{"claim":"CLAIM_NO_COMMIT","issue":4242,"reason":"unknown"}`), 0o644); err != nil {
				t.Fatal(err)
			}
			historic := ClassifyUnit(log, AliveProbe{Scanned: true}).Reason
			if historic != tc.want {
				t.Fatalf("read-time refinement of a historic sidecar = %q, want %q",
					historic, tc.want)
			}
			// The SAME run, stamped by the post-#5870 writer: identical result, and it
			// arrives without the reader having to re-read the log at all.
			if err := os.WriteFile(log[:len(log)-4]+".witness",
				[]byte(`{"claim":"CLAIM_NO_COMMIT","issue":4242,"reason":"`+tc.want+`"}`), 0o644); err != nil {
				t.Fatal(err)
			}
			if modern := ClassifyUnit(log, AliveProbe{Scanned: true}).Reason; modern != historic {
				t.Fatalf("write-time %q != read-time %q for the same run", modern, historic)
			}
		})
	}
}
