package rsiloop

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestSkillQuarantineAdmitAndState is the #2839 acceptance witness: an agent-authored
// skill that trips a HARDLINE, slop, or duplicate signal is QUARANTINED before it
// loads — the decision is a durable, structured, per-decision-reversible journal row
// (the same discipline as the curator) — while a terse-but-legitimate skill is
// admitted with no journal row. The two named confusion risks are covered directly:
// a terse skill that names a real check is NOT slop, and duplicate detection uses a
// HIGH floor so shared boilerplate alone does not flag.
func TestSkillQuarantineAdmitAndState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quarantine.jsonl")
	l, err := OpenQuarantineLedger(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}

	// A terse-but-legitimate skill: short body, but its lines name a real check and
	// its frontmatter clears the HARDLINE. It must ADMIT (no quarantine, no row).
	good := ParseQuarantineCandidate(
		"---\nname: run-ci\ndescription: run the CI gate and read the exit code\n---\n"+
			"Run `make ci`; exit 0 means the gate is green.\n", nil)
	admitted, _, _, err := l.Admit(good, SlopVerdict{Rejected: false, Score: 100}, nil)
	if err != nil || !admitted {
		t.Fatalf("terse-legit skill: admitted=%v err=%v, want admitted", admitted, err)
	}
	if _, held := l.IsQuarantined("run-ci"); held {
		t.Fatalf("terse-legit skill was quarantined")
	}
	if len(l.Rows()) != 0 {
		t.Fatalf("admitting a clean skill leaked a journal row: %+v", l.Rows())
	}

	// A slop skill: the scorecard rejected the body. It must be QUARANTINED with a
	// structured slop reason naming the tripped signals.
	slopCand := ParseQuarantineCandidate(
		"---\nname: dump-skill\ndescription: notes\n---\nsome pasted output\n", nil)
	admitted, slopSeq, reason, err := l.Admit(slopCand,
		SlopVerdict{Rejected: true, Score: 25, Signals: []string{"verbatim_dump", "missing_verification"}}, nil)
	if err != nil {
		t.Fatalf("admit slop skill: %v", err)
	}
	if admitted {
		t.Fatalf("slop skill was admitted, want quarantined")
	}
	if reason.Kind != QReasonSlop || len(reason.Signals) != 2 {
		t.Fatalf("slop reason = %+v, want slop with 2 signals", reason)
	}
	if r, held := l.IsQuarantined("dump-skill"); !held || r.Kind != QReasonSlop {
		t.Fatalf("dump-skill quarantine state = %+v held=%v, want slop", r, held)
	}

	// A HARDLINE skill: description > 60 chars. Quarantined with a hardline reason.
	longDesc := "this description is deliberately far too long to be a one-line trigger phrase"
	if len(longDesc) <= HardlineDescriptionMaxLen {
		t.Fatalf("test fixture bug: longDesc is not over the %d-char limit", HardlineDescriptionMaxLen)
	}
	hardCand := ParseQuarantineCandidate(
		"---\nname: verbose\ndescription: "+longDesc+"\n---\nRun `make ci`; verify exit 0.\n", nil)
	admitted, _, reason, err = l.Admit(hardCand, SlopVerdict{Rejected: false, Score: 100}, nil)
	if err != nil {
		t.Fatalf("admit hardline skill: %v", err)
	}
	if admitted || reason.Kind != QReasonHardline {
		t.Fatalf("hardline skill: admitted=%v reason=%+v, want quarantined/hardline", admitted, reason)
	}
	if !hasViolation(reason.Hardline, HardlineDescriptionTooLong) {
		t.Fatalf("hardline reason = %+v, want description_too_long", reason.Hardline)
	}

	// A duplicate skill: body near-identical to an existing library skill. The corpus
	// also holds an UNRELATED skill sharing only a common header line — that shared
	// boilerplate alone must NOT flag (the over-flag confusion risk).
	corpus := map[string]string{
		"existing-a": "step one: build\nstep two: test\nstep three: ship\n",
		"unrelated":  "step one: build\ntotally different body here\nand another distinct line\n",
	}
	dupCand := ParseQuarantineCandidate(
		"---\nname: copycat\ndescription: a copy\n---\nstep one: build\nstep two: test\nstep three: ship\n", nil)
	admitted, _, reason, err = l.Admit(dupCand, SlopVerdict{Rejected: false, Score: 100}, corpus)
	if err != nil {
		t.Fatalf("admit duplicate skill: %v", err)
	}
	if admitted || reason.Kind != QReasonDuplicate || reason.DuplicateOf != "existing-a" {
		t.Fatalf("duplicate skill: admitted=%v reason=%+v, want quarantined/duplicate-of-existing-a", admitted, reason)
	}

	// A skill sharing only the one boilerplate line with the corpus must ADMIT.
	notDup := ParseQuarantineCandidate(
		"---\nname: fresh\ndescription: genuinely new\n---\nstep one: build\na wholly new procedure line\none more unique line here\n", nil)
	if admitted, _, _, err := l.Admit(notDup, SlopVerdict{Score: 100}, corpus); err != nil || !admitted {
		t.Fatalf("boilerplate-only overlap: admitted=%v err=%v, want admitted", admitted, err)
	}

	// Per-decision release: re-admit ONLY the slop skill. Its siblings stay held.
	if err := l.Release(slopSeq); err != nil {
		t.Fatalf("release slop skill: %v", err)
	}
	if _, held := l.IsQuarantined("dump-skill"); held {
		t.Fatalf("dump-skill should be re-admitted after release")
	}
	if _, held := l.IsQuarantined("verbose"); !held {
		t.Fatalf("sibling 'verbose' was released by dump-skill's release")
	}
	if _, held := l.IsQuarantined("copycat"); !held {
		t.Fatalf("sibling 'copycat' was released by dump-skill's release")
	}

	// Release is refused when it cannot be scoped to one live quarantine.
	if err := l.Release(slopSeq); err == nil {
		t.Fatalf("double-release of seq %d should be refused", slopSeq)
	}
	if err := l.Release(9999); err == nil {
		t.Fatalf("release of unknown seq should be refused")
	}

	// "From the journal alone": reopen from disk and the folded state is identical.
	reopened, err := OpenQuarantineLedger(path)
	if err != nil {
		t.Fatalf("reopen ledger: %v", err)
	}
	if _, held := reopened.IsQuarantined("dump-skill"); held {
		t.Fatalf("reopened: dump-skill release did not survive on disk")
	}
	if r, held := reopened.IsQuarantined("verbose"); !held || r.Kind != QReasonHardline {
		t.Fatalf("reopened: verbose state = %+v held=%v, want hardline", r, held)
	}
	held := map[string]QuarantineReasonKind{}
	for _, e := range reopened.Log() {
		if e.Quarantined {
			held[e.Skill] = e.Reason.Kind
		}
	}
	if len(held) != 2 || held["verbose"] != QReasonHardline || held["copycat"] != QReasonDuplicate {
		t.Fatalf("reopened held-set = %+v, want {verbose:hardline, copycat:duplicate}", held)
	}
}

// TestSkillQuarantineRefusesUnstructuredReason guards that a quarantine without a
// valid structured reason never enters the journal — every held skill is answerable.
func TestSkillQuarantineRefusesUnstructuredReason(t *testing.T) {
	l, err := OpenQuarantineLedger("")
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	if _, err := l.Quarantine("skill-x", QuarantineReason{}); err == nil {
		t.Fatalf("quarantine with empty reason should be refused")
	}
	if _, err := l.Quarantine("skill-x", QuarantineReason{Kind: QReasonSlop}); err == nil {
		t.Fatalf("slop reason without signals should be refused")
	}
	if _, err := l.Quarantine("skill-x", QuarantineReason{Kind: QReasonDuplicate}); err == nil {
		t.Fatalf("duplicate reason without DuplicateOf should be refused")
	}
	if _, err := l.Quarantine("", QuarantineReason{Kind: QReasonHardline, Hardline: []HardlineViolation{HardlineNameMissing}}); err == nil {
		t.Fatalf("quarantine with empty skill should be refused")
	}
	if len(l.Rows()) != 0 {
		t.Fatalf("refused decisions leaked into the journal: %+v", l.Rows())
	}
}

// TestHardlineRungs asserts each HARDLINE structural rung fires on its own defect and
// a well-formed candidate clears — the structural half of the gate, enforced the same
// way for agent-authored skills as the human standard documents.
func TestHardlineRungs(t *testing.T) {
	clean := QuarantineCandidate{Name: "ok", Description: "a concise trigger line", Body: "Run `make ci`."}
	if v := Hardline(clean); len(v) != 0 {
		t.Fatalf("clean candidate has violations: %+v", v)
	}

	cases := []struct {
		name string
		cand QuarantineCandidate
		want HardlineViolation
	}{
		{"name", QuarantineCandidate{Description: "fine"}, HardlineNameMissing},
		{"desc-missing", QuarantineCandidate{Name: "x"}, HardlineDescriptionMissing},
		{"desc-long", QuarantineCandidate{Name: "x", Description: strings.Repeat("y", HardlineDescriptionMaxLen+1)}, HardlineDescriptionTooLong},
		{"marketing", QuarantineCandidate{Name: "x", Description: "a seamless revolutionary tool"}, HardlineDescriptionMarketing},
		{"unshipped", QuarantineCandidate{Name: "x", Description: "fine", Body: "run `python scripts/setup.py` first"}, HardlineUnshippedScript},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !hasViolation(Hardline(tc.cand), tc.want) {
				t.Fatalf("Hardline(%+v) = %+v, want %s", tc.cand, Hardline(tc.cand), tc.want)
			}
		})
	}

	// A shipped helper script clears the unshipped rung.
	shipped := QuarantineCandidate{Name: "x", Description: "fine",
		Body: "run `python scripts/setup.py` first", ShippedScripts: []string{"scripts/setup.py"}}
	if hasViolation(Hardline(shipped), HardlineUnshippedScript) {
		t.Fatalf("shipped helper script still flagged unshipped: %+v", Hardline(shipped))
	}
	// A bare tool name (not a relative script) must not trip the unshipped rung.
	tool := QuarantineCandidate{Name: "x", Description: "fine", Body: "run `go test ./...` and `make ci`"}
	if hasViolation(Hardline(tool), HardlineUnshippedScript) {
		t.Fatalf("bare tool invocation flagged as an unshipped script: %+v", Hardline(tool))
	}
}

// TestParseQuarantineCandidate asserts the frontmatter/body split the gate relies on.
func TestParseQuarantineCandidate(t *testing.T) {
	c := ParseQuarantineCandidate("---\nname: foo\ndescription: bar baz\n---\nline one\nline two\n", nil)
	if c.Name != "foo" || c.Description != "bar baz" {
		t.Fatalf("frontmatter = name=%q desc=%q, want foo/bar baz", c.Name, c.Description)
	}
	if strings.TrimSpace(c.Body) != "line one\nline two" {
		t.Fatalf("body = %q, want the two content lines", c.Body)
	}
	// No frontmatter → whole text is the body, empty name/description (which the
	// HARDLINE then flags as missing).
	c2 := ParseQuarantineCandidate("just a body, no frontmatter\n", nil)
	if c2.Name != "" || !strings.Contains(c2.Body, "just a body") {
		t.Fatalf("no-frontmatter parse = %+v", c2)
	}
}

func hasViolation(vs []HardlineViolation, want HardlineViolation) bool {
	for _, v := range vs {
		if v == want {
			return true
		}
	}
	return false
}
