package claimcheck

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// claimsPath is the real honesty ledger, from this package's directory.
var claimsPath = filepath.Join("..", "..", "CLAIMS.md")

// TestCLAIMSLedgerIsClean is the `make claims-lint` gate itself (#6218): the real
// CLAIMS.md carries a machine-readable exposure state on every capability line,
// no gated claim is missing its stated reason, and the default-off count is
// EMITTED here rather than grepped out of prose. It supersedes the awk one-liner
// that could only count the three honesty tags.
func TestCLAIMSLedgerIsClean(t *testing.T) {
	rep, err := LintLedgerFile(claimsPath)
	if err != nil {
		t.Fatalf("read CLAIMS.md: %v", err)
	}
	t.Log(strings.TrimRight(rep.String(), "\n"))
	if !rep.OK() {
		t.Fatalf("CLAIMS.md has %d exposure/tag violations (above)", len(rep.Violations))
	}
	if rep.Capability == 0 {
		t.Fatal("CLAIMS.md parsed to zero capability lines")
	}
}

// TestCLAIMSLedgerExposureIsTotal proves the state is total over the ledger — every
// capability line lands in exactly one of default-on / gated / parked, and the
// [SHIPPED] rows split cleanly into the two Q6 answers. This is the invariant that
// makes "what fraction of claimed capabilities are on for a default operator" a
// division instead of a reading exercise.
func TestCLAIMSLedgerExposureIsTotal(t *testing.T) {
	rep, err := LintLedgerFile(claimsPath)
	if err != nil {
		t.Fatalf("read CLAIMS.md: %v", err)
	}
	if got, want := rep.DefaultOn+rep.Gated+rep.Parked, rep.Capability; got != want {
		t.Errorf("exposure states cover %d lines, ledger has %d", got, want)
	}
	if got, want := rep.DefaultOn+rep.Gated, rep.Shipped; got != want {
		t.Errorf("default-on+gated = %d, [SHIPPED] lines = %d", got, want)
	}
	if got, want := rep.Parked, rep.Simulated+rep.Stub; got != want {
		t.Errorf("parked = %d, [SIMULATED]+[STUB] = %d", got, want)
	}
	if rep.Gated == 0 {
		t.Error("no gated claim is declared — the backfill of the default-off lines is missing")
	}
	// Every gated line states a reason, and every declared line is one of the two.
	for _, l := range rep.Lines {
		if l.Exposure == ExposureGated && strings.TrimSpace(l.Realized.GateReason) == "" {
			t.Errorf("CLAIMS.md:%d is gated with no stated reason", l.N)
		}
		if l.Exposure == ExposureGated && !l.Declared {
			t.Errorf("CLAIMS.md:%d is gated but carries no explicit marker", l.N)
		}
	}
}

// TestCLAIMSLedgerBackfillCoversProseDisclosures pins the backfill: every line
// whose prose discloses gating now ALSO declares its exposure, so the disclosure
// is countable instead of merely readable. (The rule is enforced by the lint; this
// asserts the corpus it was backfilled over is non-trivial, so a future revert to
// prose-only disclosure cannot pass by emptying the set.)
func TestCLAIMSLedgerBackfillCoversProseDisclosures(t *testing.T) {
	data, err := os.ReadFile(claimsPath)
	if err != nil {
		t.Fatalf("read CLAIMS.md: %v", err)
	}
	rep := LintLedger(string(data))
	var disclosed int
	for _, l := range rep.Lines {
		if l.Tag != "[SHIPPED]" {
			continue
		}
		body, _, _ := splitExposure(l.Text)
		if proseGate(body) == "" {
			continue
		}
		disclosed++
		if !l.Declared {
			t.Errorf("CLAIMS.md:%d discloses gating in prose but declares no exposure", l.N)
		}
	}
	// The managed index intentionally keeps claim prose on linked detail pages.
	// Preserve the old non-trivial corpus assertion only while CLAIMS.md itself
	// remains the prose-bearing ledger; the index still has total exposure checks
	// above, and each managed page is linted by the document-set pipeline.
	if !strings.Contains(string(data), "<!-- fak:document-set -->") && disclosed < 16 {
		t.Errorf("only %d [SHIPPED] lines disclose gating in prose; the census found 16", disclosed)
	}
}

// TestLedgerRules is the fixture witness the issue names: a line claiming a gated
// capability with NO stated reason fails, and the same line with a reason passes.
// Every other rule in the closed vocabulary is pinned beside it, each against a
// near-miss that must stay clean (no false positive).
func TestLedgerRules(t *testing.T) {
	const gatedNoReason = "- [SHIPPED] A gated capability with no stated reason. [exposure: gated]"
	const gatedWithReason = "- [SHIPPED] A gated capability with a stated reason. [exposure: gated — FAK_THING=1 arms it; default OFF so an existing floor is byte-for-byte unchanged]"

	cases := []struct {
		name string
		line string
		want string // "" = must lint clean
	}{
		{"gated with no reason fails", gatedNoReason, RuleExposureNoReason},
		{"gated with a reason passes", gatedWithReason, ""},
		{"blank reason is no reason", "- [SHIPPED] Gated. [exposure: gated —   ]", RuleExposureNoReason},
		{"default-on passes", "- [SHIPPED] An on-by-default capability. [exposure: default-on]", ""},
		{"no marker asserts default-on", "- [SHIPPED] An on-by-default capability.", ""},
		{"prose gate without a marker fails", "- [SHIPPED] A capability that is off by default until you set FAK_THING.", RuleExposureUndeclared},
		{"prose gate with a marker passes", "- [SHIPPED] A capability that is off by default until you set FAK_THING. [exposure: gated — FAK_THING arms it]", ""},
		{"opt-in prose without a marker fails", "- [SHIPPED] An opt-in capability behind `WithThing`.", RuleExposureUndeclared},
		{"unknown exposure value fails", "- [SHIPPED] A capability. [exposure: someday]", RuleExposureUnknown},
		{"two markers fail", "- [SHIPPED] A capability. [exposure: default-on] [exposure: gated — r]", RuleExposureDuplicate},
		{"a marker mid-line fails", "- [SHIPPED] A capability [exposure: default-on] with prose after it.", RuleExposurePlacement},
		{"no tag fails", "- [SHIPPING] A capability.", RuleTag},
		{"two tags fail", "- [SHIPPED] [STUB] A capability.", RuleTag},
		{"a parked line needs no marker", "- [STUB] Plumbing present, behavior deferred; opt-in when it lands.", ""},
		{"a marker on a parked line fails", "- [SIMULATED] Modeled with stand-in data. [exposure: gated — no GPU on the box]", RuleExposureOnParked},
		{"provider prose is not a gate", "- [SHIPPED] Cache-savings attribution is owner/mechanism split by default.", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := LintLedger(tc.line + "\n")
			if rep.Capability != 1 {
				t.Fatalf("parsed %d capability lines, want 1", rep.Capability)
			}
			if tc.want == "" {
				if !rep.OK() {
					t.Fatalf("want clean, got:\n%s", rep.String())
				}
				return
			}
			if rep.OK() {
				t.Fatalf("want violation %s, got clean", tc.want)
			}
			var got []string
			for _, v := range rep.Violations {
				got = append(got, v.Rule)
				if v.N != 1 {
					t.Errorf("violation reports line %d, want 1", v.N)
				}
			}
			var found bool
			for _, g := range got {
				if g == tc.want {
					found = true
				}
			}
			if !found {
				t.Fatalf("want violation %s, got %v", tc.want, got)
			}
		})
	}

	// The report names the failing rule and the offending line, so a red is
	// actionable without re-reading the ledger.
	rep := LintLedger(gatedNoReason + "\n")
	if s := rep.String(); !strings.Contains(s, RuleExposureNoReason) || !strings.Contains(s, "no stated reason") {
		t.Errorf("report does not name the rule and the line:\n%s", s)
	}
}

// TestLedgerReusesQ6 proves the exposure lint is the SAME rule as the grader's Q6,
// not a parallel vocabulary: a parsed line's Realized value grades exactly as
// gradeRealized says it should.
func TestLedgerReusesQ6(t *testing.T) {
	src := strings.Join([]string{
		"- [SHIPPED] On for everyone. [exposure: default-on]",
		"- [SHIPPED] Off, honestly. [exposure: gated — needs a GPU the build box lacks]",
	}, "\n")
	rep := LintLedger(src)
	if !rep.OK() {
		t.Fatalf("fixture should lint clean:\n%s", rep.String())
	}
	if len(rep.Lines) != 2 {
		t.Fatalf("parsed %d lines, want 2", len(rep.Lines))
	}
	if !rep.Lines[0].Realized.OnByDefault {
		t.Error("default-on did not map to Realized{OnByDefault:true}")
	}
	if got, want := rep.Lines[1].Realized.GateReason, "needs a GPU the build box lacks"; got != want {
		t.Errorf("gate reason = %q, want %q", got, want)
	}
	for _, l := range rep.Lines {
		if q := gradeRealized(l.Realized); !q.Pass {
			t.Errorf("CLAIMS.md-shaped line %d fails Q6: %s", l.N, q.Detail)
		}
	}
	// And the same Realized value, dropped into a full claim, grades net-true —
	// the ledger and the grader agree on what an honest gate is.
	c := Claim{
		Statement:  "off, honestly",
		Baseline:   Baseline{Kind: BaselineReal, Description: "tuned alternative"},
		Net:        true,
		Scope:      "CPU build box",
		Provenance: Witnessed,
		Witness:    "go test ./internal/claimcheck",
		Realized:   rep.Lines[1].Realized,
	}
	if got := Grade(c).Verdict; got != NetTrue {
		t.Errorf("grade = %q, want %q", got, NetTrue)
	}
}

// TestLedgerEmptyIsNotAPass keeps the awk lint's `c==0` exit: a ledger that parsed
// to nothing is a broken read, never a green gate.
func TestLedgerEmptyIsNotAPass(t *testing.T) {
	rep := LintLedger("# CLAIMS.md\n\nno capability lines here\n")
	if rep.OK() {
		t.Fatal("an empty ledger linted clean")
	}
	if rep.Violations[0].Rule != RuleEmpty {
		t.Errorf("rule = %q, want %q", rep.Violations[0].Rule, RuleEmpty)
	}
}
