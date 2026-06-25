package rulesynth

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// guarded is the harness tree the test near-misses reach.
var guarded = []string{"internal/adjudicator/", "internal/abi/", "dos.toml"}

// TestDetectFlagsUnrecognizedWriteAsNearMiss: a write reaching a guarded tree by a verb
// the floor does NOT recognize (`php -r 'file_put_contents(...)'`; `ruby -e` was the
// prior allele, now closed in interpreterEvalFlags) slips the floor, so Detect captures
// it as a near-miss.
func TestDetectFlagsUnrecognizedWriteAsNearMiss(t *testing.T) {
	c := Call{Tool: "Bash", Arg: "command",
		Command: `php -r 'file_put_contents("internal/adjudicator/decide.go", $x);'`}
	nm, ok := Detect(c, guarded)
	if !ok {
		t.Fatalf("expected a near-miss for an unrecognized write verb into a guarded tree")
	}
	if nm.GuardedGlob != "internal/adjudicator/" {
		t.Fatalf("guarded glob = %q, want internal/adjudicator/", nm.GuardedGlob)
	}
}

// TestDetectIgnoresRecognizedWrite: a write the floor ALREADY catches (`sed -i` is in
// shellWriteVerbs) is not a near-miss — Detect must not re-mine a covered route.
func TestDetectIgnoresRecognizedWrite(t *testing.T) {
	c := Call{Tool: "Bash", Arg: "command", Command: `sed -i s/a/b/ internal/adjudicator/decide.go`}
	if _, ok := Detect(c, guarded); ok {
		t.Fatalf("a recognized write verb (sed -i) must not be captured as a near-miss")
	}
}

// TestDetectIgnoresUnguardedCommand: a write by an unrecognized verb that touches NO
// guarded tree is out of scope — there is nothing to guard.
func TestDetectIgnoresUnguardedCommand(t *testing.T) {
	c := Call{Tool: "Bash", Arg: "command", Command: `php -r 'file_put_contents("/tmp/out.txt", $x);'`}
	if _, ok := Detect(c, guarded); ok {
		t.Fatalf("a write that names no guarded tree must not be a near-miss")
	}
}

// TestProposeClustersByVerb: two near-misses of the same verb collapse to ONE candidate
// rule whose support counts both; a distinct verb yields a separate candidate.
func TestProposeClustersByVerb(t *testing.T) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/adjudicator/x.go","")'`}, GuardedGlob: "internal/adjudicator/"},
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/abi/y.go","")'`}, GuardedGlob: "internal/abi/"},
		{Call: Call{Tool: "Bash", Arg: "command", Command: `sponge internal/adjudicator/z.go`}, GuardedGlob: "internal/adjudicator/"},
	}
	cands := Propose(corpus)
	if len(cands) != 2 {
		t.Fatalf("want 2 candidates (ruby -e, sponge), got %d", len(cands))
	}
	// Sorted by verb: "ruby -e" before "sponge".
	if cands[0].Verb != "ruby -e" || cands[0].Support != 2 {
		t.Fatalf("first candidate = %q support=%d, want ruby -e support=2", cands[0].Verb, cands[0].Support)
	}
	if len(cands[0].Globs) != 2 {
		t.Fatalf("ruby -e candidate should carry both guarded globs, got %v", cands[0].Globs)
	}
	if !cands[0].SelfModify {
		t.Fatalf("a rule guarding internal/adjudicator must be flagged SelfModify (require-witness)")
	}
}

// TestValidateKeepsCatchingRuleNoRegression: the honesty gate KEEPs a rule that newly
// catches its near-misses (model-free replay through the real floor) while denying ZERO
// benign calls — and the keep-bit is shipgate's, not the package's say-so.
func TestValidateKeepsCatchingRuleNoRegression(t *testing.T) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/adjudicator/x.go","")'`}, GuardedGlob: "internal/adjudicator/"},
	}
	benign := []Call{
		{Tool: "Bash", Arg: "command", Command: `ruby app.rb`},                        // verb, unguarded path
		{Tool: "Bash", Arg: "command", Command: `cat internal/adjudicator/decide.go`}, // guarded path, but a read
	}
	cand := Propose(corpus)[0]
	v, err := Validate(cand, corpus, benign)
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if v.Caught != 1 {
		t.Fatalf("Caught = %d, want 1 (the near-miss is now denied)", v.Caught)
	}
	if v.Regressed != 0 {
		t.Fatalf("Regressed = %d, want 0 (benign calls stay allowed)", v.Regressed)
	}
	if v.Decision != shipgate.KEEP || !v.Kept {
		t.Fatalf("a catching, non-regressing rule must KEEP; got %v kept=%v", v.Decision, v.Kept)
	}
}

// TestValidateRevertsRegressingRule: a rule that DENIES a benign call is REVERTed — the
// suite-green bit refuses the regression even though the rule catches the near-miss.
func TestValidateRevertsRegressingRule(t *testing.T) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/adjudicator/x.go","")'`}, GuardedGlob: "internal/adjudicator/"},
	}
	cand := Propose(corpus)[0]
	// A benign call that the synthesized rule wrongly matches: it names the verb AND a
	// guarded tree but is a read — exactly the regression the keep-bit must catch.
	benign := []Call{
		{Tool: "Bash", Arg: "command", Command: `ruby -e 'puts File.read("internal/adjudicator/decide.go")'`},
	}
	v, err := Validate(cand, corpus, benign)
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if v.Regressed == 0 {
		t.Fatalf("expected the over-broad rule to regress the benign read")
	}
	if v.Decision != shipgate.REVERT || v.Kept {
		t.Fatalf("a regressing rule must REVERT; got %v kept=%v", v.Decision, v.Kept)
	}
}

// TestManifestDiffIsReviewableNotLive: the candidate's only output is a reviewable
// manifest fragment carrying its one rule — proof the loop lands a diff, never a live
// mutation. The fragment must re-compile through the real manifest loader.
func TestManifestDiffIsReviewableNotLive(t *testing.T) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `sponge internal/adjudicator/x.go`}, GuardedGlob: "internal/adjudicator/"},
	}
	cand := Propose(corpus)[0]
	m := cand.ManifestDiff()
	if len(m.ArgRules) != 1 {
		t.Fatalf("manifest diff must carry exactly the one synthesized rule, got %d", len(m.ArgRules))
	}
	if m.ArgRules[0].Reason != abi.ReasonName(abi.ReasonSelfModify) {
		t.Fatalf("synthesized rule reason = %q, want SELF_MODIFY", m.ArgRules[0].Reason)
	}
	if _, err := m.ToPolicy(); err != nil {
		t.Fatalf("the synthesized manifest diff must load as a real policy: %v", err)
	}
}
