package claimcheck

import (
	"strings"
	"testing"
)

// TestFixtureGradesAsExpected is the issue #1171 acceptance witness: the built-in
// corpus of honest and strawman claims each grades to its labeled verdict. A drift
// in the grader (a question that stops failing on silence, a strawman that slips to
// net-true) reds here. The same corpus backs `fak claim-check --self-test`, so the
// CLI and this test assert the exact same behavior.
func TestFixtureGradesAsExpected(t *testing.T) {
	for _, fc := range Fixture() {
		got := Grade(fc.Claim)
		if got.Verdict != fc.Expect {
			t.Errorf("%s: got verdict %q, want %q\n%s", fc.Name, got.Verdict, fc.Expect, got.String())
		}
	}
}

// TestRunFixtureAllPass is the same assertion through the shared RunFixture core the
// CLI `--self-test` mode calls, so a regression is caught on both paths.
func TestRunFixtureAllPass(t *testing.T) {
	cases, passed := RunFixture()
	if len(cases) == 0 {
		t.Fatal("fixture is empty — there is no acceptance corpus to grade")
	}
	if passed != len(cases) {
		for _, c := range cases {
			if !c.OK {
				t.Errorf("%s: got %q, want %q", c.Name, c.Got, c.Expect)
			}
		}
	}
}

// TestFixtureCoversAllThreeVerdicts proves the corpus is not degenerate: it exercises
// each of net-true, strawman, and not-yet at least once (a fixture of only passes
// would witness nothing about the failure paths).
func TestFixtureCoversAllThreeVerdicts(t *testing.T) {
	seen := map[Verdict]bool{}
	for _, fc := range Fixture() {
		seen[fc.Expect] = true
	}
	for _, v := range []Verdict{NetTrue, Strawman, NotYet} {
		if !seen[v] {
			t.Errorf("fixture never exercises the %q verdict", v)
		}
	}
}

// TestStrawmanWinsOverNotYet pins the standard's ordering rule: a number measured
// against the naive floor is a strawman REGARDLESS of how many other questions it
// fails — the wrong baseline is noise no matter what. Here every other question is
// also unanswered, yet the verdict is Strawman, not NotYet.
func TestStrawmanWinsOverNotYet(t *testing.T) {
	c := Claim{
		Statement: "60.3× faster",
		Baseline:  Baseline{Kind: BaselineStrawman, Description: "naive re-send-everything"},
		// every other question left unanswered:
		Net:        false,
		Scope:      "",
		Provenance: ProvNone,
		Witness:    "",
		Realized:   Realized{},
	}
	got := Grade(c)
	if got.Verdict != Strawman {
		t.Fatalf("a strawman baseline must short-circuit to Strawman, got %q", got.Verdict)
	}
	// The other gaps are still surfaced so the author sees the whole picture.
	if len(got.Missing) == 0 {
		t.Error("expected the other unanswered questions to be surfaced in Missing")
	}
}

// TestEveryQuestionGatesNotYet proves each of the six questions is load-bearing: a
// claim that answers all six except one grades NotYet, and the one it dropped is the
// only entry named in Missing. Silence on any single question is never a pass.
func TestEveryQuestionGatesNotYet(t *testing.T) {
	// A claim that answers all six against the real baseline (the honest control).
	full := func() Claim {
		return Claim{
			Statement:  "x is 2× vs the tuned alternative",
			Baseline:   Baseline{Kind: BaselineReal, Description: "tuned alternative"},
			Net:        true,
			Scope:      "the stated scope",
			Provenance: Witnessed,
			Witness:    "go test ./... -run TestX",
			Realized:   Realized{OnByDefault: true},
		}
	}
	if v := Grade(full()).Verdict; v != NetTrue {
		t.Fatalf("the full control claim must grade NetTrue, got %q", v)
	}

	drops := []struct {
		name string
		mut  func(*Claim)
		want string // the question name expected in Missing
	}{
		{"net", func(c *Claim) { c.Net = false }, "net"},
		{"scope", func(c *Claim) { c.Scope = "" }, "scope"},
		{"provenance", func(c *Claim) { c.Provenance = ProvNone }, "provenance"},
		{"witness", func(c *Claim) { c.Witness = "" }, "witness"},
		{"realized", func(c *Claim) { c.Realized = Realized{} }, "realized"},
		{"baseline-none", func(c *Claim) { c.Baseline = Baseline{Kind: BaselineNone} }, "baseline"},
	}
	for _, d := range drops {
		c := full()
		d.mut(&c)
		got := Grade(c)
		if got.Verdict != NotYet {
			t.Errorf("dropping %s: want NotYet, got %q", d.name, got.Verdict)
			continue
		}
		if len(got.Missing) != 1 || got.Missing[0] != d.want {
			t.Errorf("dropping %s: want Missing==[%q], got %v", d.name, d.want, got.Missing)
		}
	}
}

// TestInvalidProvenanceLabelFails proves Q4 rejects a non-empty but bogus provenance
// label (a typo is not a pass), distinct from the empty-label case.
func TestInvalidProvenanceLabelFails(t *testing.T) {
	c := Claim{
		Statement:  "y is 3× vs tuned",
		Baseline:   Baseline{Kind: BaselineReal},
		Net:        true,
		Scope:      "scope",
		Provenance: Provenance("PROVEN"), // not one of the four closed labels
		Witness:    "a test",
		Realized:   Realized{OnByDefault: true},
	}
	got := Grade(c)
	if got.Verdict != NotYet {
		t.Fatalf("a bogus provenance label must fail Q4 ⇒ NotYet, got %q", got.Verdict)
	}
	if len(got.Missing) != 1 || got.Missing[0] != "provenance" {
		t.Errorf("want Missing==[provenance], got %v", got.Missing)
	}
}

// TestHonestlyGatedRealizedPasses proves Q6 accepts an off-by-default gain WHEN it
// carries a stated reason (an honest gate), and rejects it when the reason is empty
// (a seam).
func TestHonestlyGatedRealizedPasses(t *testing.T) {
	base := Claim{
		Statement:  "z is 1.2× vs tuned",
		Baseline:   Baseline{Kind: BaselineReal},
		Net:        true,
		Scope:      "scope",
		Provenance: Witnessed,
		Witness:    "a test",
	}
	gated := base
	gated.Realized = Realized{OnByDefault: false, GateReason: "Mac-gated until on-device tok/s clears the bar"}
	if v := Grade(gated).Verdict; v != NetTrue {
		t.Errorf("an off-by-default gain WITH a stated reason must pass Q6 ⇒ NetTrue, got %q", v)
	}
	seam := base
	seam.Realized = Realized{OnByDefault: false, GateReason: ""}
	if v := Grade(seam).Verdict; v != NotYet {
		t.Errorf("an off-by-default gain with NO reason is a seam ⇒ NotYet, got %q", v)
	}
}

// TestResultStringRenders proves the operator-readable block names the verdict and
// every question (it is part of the CLI's output contract).
func TestResultStringRenders(t *testing.T) {
	got := Grade(Fixture()[0].Claim).String()
	if !strings.Contains(got, "claim-check:") {
		t.Errorf("rendered block missing the header, got:\n%s", got)
	}
	for _, name := range []string{"baseline", "net", "scope", "provenance", "witness", "realized"} {
		if !strings.Contains(got, name) {
			t.Errorf("rendered block missing question %q, got:\n%s", name, got)
		}
	}
}
