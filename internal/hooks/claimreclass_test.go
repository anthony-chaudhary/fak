package hooks

import (
	"strings"
	"testing"
)

// claimreclass_test.go — the two witnesses #5434 asks for.
//
//  1. the WEDGE case (a code-effect subject over a prose-only diff, already landed on a shared
//     trunk where amending is forbidden) now has a reachable cure that rewrites nothing, and it is
//     driven THROUGH the same gate decision the push seam takes; and
//  2. the ANTI-LAUNDERING case: a genuinely mis-described commit is still refused, so the cure did
//     not degenerate into a blanket pass. Every laundering shape is enumerated, and the structural
//     invariant ("no accepted reclassification ever names a code-effect type") is asserted over
//     all of them at once.

// wedgeReview is the claim-honesty review output for the shape this issue is about: one commit in
// the push range whose `fix(...)` subject claims a code effect while its diff touches only prose.
// The CLEARED band is present on purpose — a cleared row must never be read as curable.
const wedgeReview = `CLAIM REVIEW  origin/main..HEAD  (34 commits)
RESIDUAL — your 100% (1)  [a CLAIM the kernel could not witness]
  3e2404d4d  subject-only   fix(alpha): separate the noise prune failure from a real readback regression
             |- code-effect claim but the diff touches no SOURCE file
                (only: docs/alpha-notes.md) — the claim rests on the subject text
CLEARED (33)
  bbbbbb12  diff-witnessed  feat(alpha): add the retry path (fak alpha)
`

// landedResidual is what git recorded for that commit: the subject is immutable (it is ~30 deep in
// a shared trunk) and the diff is prose-only.
var landedResidual = CommitFacts{
	SHA:     "3e2404d4d1111111111111111111111111111111",
	Subject: "fix(alpha): separate the noise prune failure from a real readback regression",
	Paths:   []string{"docs/alpha-notes.md"},
}

func lookupLanded(facts ...CommitFacts) func(string) (CommitFacts, bool) {
	return func(id string) (CommitFacts, bool) {
		for _, f := range facts {
			if strings.HasPrefix(strings.ToLower(f.SHA), strings.ToLower(id)) {
				return f, true
			}
		}
		return CommitFacts{}, false
	}
}

// TestClaimReclass_wedgedRangeHasAReachableNonRewriteCure is witness (1). With no ledger the range
// stays refused — that is today's wedge, whose only exit is the override. Appending one
// forward-only record that DEMOTES the landed claim to the one its own diff witnesses clears the
// same gate, with the landed subject left byte-for-byte untouched.
func TestClaimReclass_wedgedRangeHasAReachableNonRewriteCure(t *testing.T) {
	residuals := ParseResidualCommits(wedgeReview)
	if len(residuals) != 1 || residuals[0] != "3e2404d4d" {
		t.Fatalf("residual band parse = %v, want [3e2404d4d] (a CLEARED row must not be read as curable)", residuals)
	}
	lookup := lookupLanded(landedResidual)

	// RED — the wedge as it stands today: nothing in the ledger, so the range stays refused and
	// the operator's only exit is the override this issue is about.
	before := ClearResiduals(residuals, nil, lookup)
	if before.OK {
		t.Fatalf("empty ledger cleared the range; the gate must stay refused with no cure on file")
	}
	if len(before.Uncured) != 1 || before.Uncured[0] != "3e2404d4d" {
		t.Fatalf("uncured = %v, want [3e2404d4d]", before.Uncured)
	}

	// The cure the refusal hands back, verbatim.
	tmpl := ReclassTemplate(landedResidual)
	if !strings.Contains(tmpl, "type: docs") || !strings.Contains(tmpl, "witness: docs/alpha-notes.md") {
		t.Fatalf("template does not derive the demoted type + witness from the real diff:\n%s", tmpl)
	}

	// GREEN — one appended record, no history rewritten.
	ledger := `# forward-only claim reclassification (#5434)
commit: 3e2404d4d
type: docs
witness: docs/alpha-notes.md
reason: the diff edits only a note; the landed subject typed a prose edit as a code effect
`
	records, problems := ParseReclassRecords(ledger)
	if len(problems) != 0 {
		t.Fatalf("ledger parse problems = %v, want none", problems)
	}
	after := ClearResiduals(residuals, records, lookup)
	if !after.OK {
		t.Fatalf("verified forward-only record did not clear the range: %+v", after.Verdicts)
	}
	if len(after.Cleared) != 1 || after.Cleared[0] != "3e2404d4d" {
		t.Fatalf("cleared = %v, want [3e2404d4d]", after.Cleared)
	}
	// The cure is a RETRACTION, not a pass: the accepted verdict demotes the claim and never
	// certifies the code effect the landed subject asserted.
	if got := after.Verdicts[0]; !got.Accepted || got.Type != "docs" || !strings.Contains(got.Reason, "demoted") {
		t.Fatalf("accepted verdict = %+v, want an accepted DEMOTION to docs", got)
	}
	// Nothing in the cure path touches the landed commit.
	if landedResidual.Subject != "fix(alpha): separate the noise prune failure from a real readback regression" {
		t.Fatalf("the landed subject was mutated; the cure must be forward-only")
	}
}

// launderAttempt is one way an operator could try to turn the cure into a blanket pass.
type launderAttempt struct {
	name   string
	ledger string
	facts  CommitFacts
	want   string // a substring the refusal reason must carry
}

// misdescribed is the commit the anti-laundering witness protects: its subject claims a feature,
// its diff is prose only. It is a TRUE positive — exactly the kind the override was spent on.
var misdescribed = CommitFacts{
	SHA:     "3e2404d4d1111111111111111111111111111111",
	Subject: "feat(alpha): add the retry loop (fak alpha)",
	Paths:   []string{"docs/alpha-notes.md"},
}

// sourceBacked is a commit whose diff DOES carry program source, used to prove a demotion cannot
// be used to understate a commit that actually changed behavior.
var sourceBacked = CommitFacts{
	SHA:     "3e2404d4d1111111111111111111111111111111",
	Subject: "feat(alpha): add the retry loop (fak alpha)",
	Paths:   []string{"internal/alpha/loop.go", "docs/alpha-notes.md"},
}

// TestClaimReclass_cureCannotLaunderAMisdescribedCommit is witness (2), the one that decides
// whether this fix is a fix at all. Every shape below is an attempt to make the gate accept a
// claim the diff does not support; every one must leave the range refused.
func TestClaimReclass_cureCannotLaunderAMisdescribedCommit(t *testing.T) {
	residuals := ParseResidualCommits(wedgeReview)
	if len(residuals) != 1 {
		t.Fatalf("fixture parse = %v", residuals)
	}
	attempts := []launderAttempt{
		{
			name:   "restate the same code-effect claim",
			ledger: "commit: 3e2404d4d\ntype: feat\nwitness: docs/alpha-notes.md\nreason: no really it is a feature\n",
			facts:  misdescribed,
			want:   "itself a code-effect claim",
		},
		{
			name:   "sidestep into another code-effect type",
			ledger: "commit: 3e2404d4d\ntype: perf\nwitness: docs/alpha-notes.md\nreason: it is a speedup honest\n",
			facts:  misdescribed,
			want:   "itself a code-effect claim",
		},
		{
			name:   "sidestep into refactor",
			ledger: "commit: 3e2404d4d\ntype: refactor\nwitness: docs/alpha-notes.md\nreason: restructuring\n",
			facts:  misdescribed,
			want:   "itself a code-effect claim",
		},
		{
			name:   "demote to a type the diff also cannot witness",
			ledger: "commit: 3e2404d4d\ntype: test\nwitness: docs/alpha-notes.md\nreason: it is really a test drop\n",
			facts:  misdescribed,
			want:   "no test or CI witness file",
		},
		{
			name:   "cite a witness the commit never touched",
			ledger: "commit: 3e2404d4d\ntype: docs\nwitness: internal/alpha/loop.go\nreason: the note explains the loop\n",
			facts:  misdescribed,
			want:   "is not in commit",
		},
		{
			name:   "demote a source-carrying commit to prose",
			ledger: "commit: 3e2404d4d\ntype: docs\nwitness: docs/alpha-notes.md\nreason: mostly a note\n",
			facts:  sourceBacked,
			want:   "also touches program source",
		},
		{
			name:   "record with no rationale",
			ledger: "commit: 3e2404d4d\ntype: docs\nwitness: docs/alpha-notes.md\n",
			facts:  misdescribed,
			want:   "no `reason:`",
		},
		{
			name:   "record citing nothing at all",
			ledger: "commit: 3e2404d4d\ntype: docs\nreason: it is a note\n",
			facts:  misdescribed,
			want:   "cites no `witness:` path",
		},
		{
			name:   "record for a different commit",
			ledger: "commit: 9999999\ntype: docs\nwitness: docs/alpha-notes.md\nreason: a note elsewhere\n",
			facts:  misdescribed,
			want:   "no reclassification record names this commit",
		},
		{
			name:   "an id too short to bind",
			ledger: "commit: 3e24\ntype: docs\nwitness: docs/alpha-notes.md\nreason: a note\n",
			facts:  misdescribed,
			want:   "no reclassification record names this commit",
		},
	}

	for _, a := range attempts {
		records, _ := ParseReclassRecords(a.ledger)
		res := ClearResiduals(residuals, records, lookupLanded(a.facts))
		if res.OK {
			t.Errorf("%s: the gate CLEARED a claim the diff does not support — the cure launders", a.name)
			continue
		}
		joined := ""
		for _, v := range res.Verdicts {
			joined += v.Reason + "\n"
			// The structural invariant, asserted on every accepted verdict anywhere in the table.
			if v.Accepted && reclassCodeEffectTypes[v.Type] {
				t.Errorf("%s: accepted a reclassification into code-effect type %q", a.name, v.Type)
			}
		}
		if !strings.Contains(joined, a.want) {
			t.Errorf("%s: refusal reasons %q do not explain the refusal (want %q)", a.name, joined, a.want)
		}
	}
}

// TestClaimReclass_unreadableReviewClearsNothing pins the fail-CLOSED direction of the relaxation:
// a review output the parser cannot read yields no residual list, and no residual list can ever
// clear a push. A rendering change upstream degrades to "the gate stays blocked".
func TestClaimReclass_unreadableReviewClearsNothing(t *testing.T) {
	for _, out := range []string{"", "some unexpected rendering with no residual band\n", "CLEARED (3)\n  aaaaaaa1  diff-witnessed  docs(alpha): update the note\n"} {
		if ids := ParseResidualCommits(out); len(ids) != 0 {
			t.Fatalf("ParseResidualCommits(%q) = %v, want none", out, ids)
		}
	}
	res := ClearResiduals(nil, []Reclass{{Commit: "3e2404d4d", Type: "docs", Witness: []string{"docs/alpha-notes.md"}, Reason: "x"}}, lookupLanded(landedResidual))
	if res.OK {
		t.Fatalf("an empty residual list cleared a push; the relaxation must be fail-closed")
	}
}

// TestClaimReclass_unresolvableCommitClearsNothing pins the other fail-closed edge: a record for an
// id git cannot resolve verifies against nothing.
func TestClaimReclass_unresolvableCommitClearsNothing(t *testing.T) {
	records, _ := ParseReclassRecords("commit: 3e2404d4d\ntype: docs\nwitness: docs/alpha-notes.md\nreason: a note\n")
	res := ClearResiduals([]string{"3e2404d4d"}, records, func(string) (CommitFacts, bool) { return CommitFacts{}, false })
	if res.OK {
		t.Fatalf("cleared a residual git could not resolve")
	}
	if len(res.Verdicts) != 1 || !strings.Contains(res.Verdicts[0].Reason, "could not resolve") {
		t.Fatalf("verdicts = %+v, want an unresolvable-commit refusal", res.Verdicts)
	}
}

// TestClaimReclass_ledgerProblemsAreReported — a malformed record must be surfaced, not dropped:
// a record the parser silently discards is a cure the operator believes they wrote.
func TestClaimReclass_ledgerProblemsAreReported(t *testing.T) {
	_, problems := ParseReclassRecords("type: docs\nthis is not a key value line\ncommit: 3e2404d4d\nbogus: x\n")
	if len(problems) != 3 {
		t.Fatalf("problems = %v, want 3 (orphan key, non-kv line, unknown key)", problems)
	}
	for _, p := range problems {
		if !strings.HasPrefix(p, ReclassFile+":") {
			t.Fatalf("problem %q is not located in the ledger", p)
		}
	}
}

// TestLintCommitMessage_codeEffectDocOnlyAdvisory is the commit-time half: the same shape is named
// BEFORE the commit exists, where retyping the subject is still legal, with the demoted type
// spelled out. It stays ADVISORY (a Note, never an Issue) so the shared pre-commit refusal set is
// not widened for concurrent lanes.
func TestLintCommitMessage_codeEffectDocOnlyAdvisory(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("fix(notes): correct the offline-check wording (fak docs)", []string{"docs/alpha-notes.md"}, root)
	if !hasNoteContaining(r, "touches no program source") {
		t.Fatalf("no code-effect/prose-only advisory in notes: %v (issues: %v)", r.Notes, r.Issues)
	}
	if !hasNoteContaining(r, "`docs(...)`") {
		t.Fatalf("advisory does not name the demoted type: %v", r.Notes)
	}
	if hasIssueContaining(r, "touches no program source") {
		t.Fatalf("the advisory became BLOCKING; that widens the shared pre-commit refusal set: %v", r.Issues)
	}

	// The negative: the same subject over real source earns nothing.
	ok := LintCommitMessage("fix(gateway): correct the retry bound (fak gateway)", []string{"internal/gateway/server.go"}, root)
	if hasNoteContaining(ok, "touches no program source") {
		t.Fatalf("advisory fired on a source-backed fix: %v", ok.Notes)
	}
	// And a correctly typed prose commit earns nothing either.
	docsOK := LintCommitMessage("docs(notes): correct the offline-check wording (fak docs)", []string{"docs/alpha-notes.md"}, root)
	if hasNoteContaining(docsOK, "touches no program source") {
		t.Fatalf("advisory fired on a correctly typed docs commit: %v", docsOK.Notes)
	}
}
