package headlesslint

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestScanLeftoversBothArms is #3670's done-condition, verbatim: a run that ends
// with "two more things worth doing" prose and filed ZERO issues is refused; the
// SAME summary once those follow-ups were filed as open gh issues passes clean.
func TestScanLeftoversBothArms(t *testing.T) {
	summary := "Shipped the retry fix, tests pass, committed abc1234, pushed.\n" +
		"There are two more things worth doing: exponential backoff and a docs pass."

	// Arm 1 — narrated leftovers, zero issues filed -> refused.
	unfiled := ScanLeftovers(summary, 0, false)
	if unfiled.Verdict != LeftoversUnfiled || !unfiled.Refused() {
		t.Fatalf("arm1 (narrated + 0 filed): want %s/refused, got %s/refused=%v %+v",
			LeftoversUnfiled, unfiled.Verdict, unfiled.Refused(), unfiled.Hits)
	}
	if unfiled.Narrated == 0 {
		t.Fatalf("arm1: expected at least one narrated leftover, got 0")
	}
	if unfiled.Resolve == "" {
		t.Errorf("arm1: a refused report should carry the remediation string")
	}

	// Arm 2 — the same summary, but the two follow-ups were filed as gh issues.
	filed := ScanLeftovers(summary, 2, false)
	if filed.Verdict != LeftoversClean || filed.Refused() {
		t.Fatalf("arm2 (narrated + 2 filed): want %s/not-refused, got %s/refused=%v",
			LeftoversClean, filed.Verdict, filed.Refused())
	}
}

// TestScanLeftoversOperatorEscape: the "genuinely nothing left" escape forces clean
// even when leftovers are narrated and nothing was filed.
func TestScanLeftoversOperatorEscape(t *testing.T) {
	rep := ScanLeftovers("A couple more things are still out of scope and left to do.", 0, true)
	if rep.Verdict != LeftoversClean || rep.Refused() {
		t.Fatalf("operator escape: want %s/not-refused, got %s/refused=%v", LeftoversClean, rep.Verdict, rep.Refused())
	}
	if !rep.Overridden {
		t.Errorf("operator escape: report should record Overridden=true")
	}
}

// TestScanLeftoversCleanSummary: a summary that only reports completed work carries
// no leftover narration, so it is clean even with zero issues filed.
func TestScanLeftoversCleanSummary(t *testing.T) {
	rep := ScanLeftovers("Implemented the parser, committed abc123, tests pass, pushed.", 0, false)
	if rep.Verdict != LeftoversClean || rep.Narrated != 0 {
		t.Fatalf("clean summary: want clean/0 narrated, got %s/%d %+v", rep.Verdict, rep.Narrated, rep.Hits)
	}
}

// TestScanLeftoversTicketedLineNotNarration: a leftover line that itself cites a
// filed ticket is honest scoping, not bare narration, so it does not flag even with
// the issues-filed count at zero (the per-line ticket cross-check).
func TestScanLeftoversTicketedLineNotNarration(t *testing.T) {
	rep := ScanLeftovers("Out of scope for this change: exponential backoff, filed #4001.", 0, false)
	if rep.Verdict != LeftoversClean {
		t.Fatalf("ticketed leftover: want clean, got %s %+v", rep.Verdict, rep.Hits)
	}
}

// leftoversNarration is the summary every evidence arm below folds — it narrates two
// deferred follow-ups and cites no ticket, so the verdict turns purely on what the
// issues-filed evidence says.
const leftoversNarration = "Shipped the retry fix, tests pass.\n" +
	"There are two more things worth doing: exponential backoff and a docs pass."

// TestScanLeftoversEvidenceWitnessedZeroRefuses: a count read END TO END off a run's
// transcript is a real zero — the record was there and showed no filing — so the #3670
// refusal still fires on evidence, not only on a self-report.
func TestScanLeftoversEvidenceWitnessedZeroRefuses(t *testing.T) {
	rep := ScanLeftoversEvidence(leftoversNarration, WitnessedIssuesFiled(0), false)
	if !rep.Refused() || rep.Verdict != LeftoversUnfiled {
		t.Fatalf("witnessed zero: want %s/refused, got %s/refused=%v", LeftoversUnfiled, rep.Verdict, rep.Refused())
	}
	count, known := rep.FiledCount()
	if !known || count != 0 || rep.IssuesFiledSource != IssuesFiledFromTranscript {
		t.Fatalf("witnessed zero: want count=0 known=true source=%s, got %d/%v/%s",
			IssuesFiledFromTranscript, count, known, rep.IssuesFiledSource)
	}
	if rep.Undecided() {
		t.Errorf("a witnessed zero is decided, not unknown")
	}
}

// TestScanLeftoversEvidenceUnknownIsNotZero is the distinction #5425 turns on: with no
// evidence to read, the fold must say "cannot tell" — not clean (nothing proved the
// leftovers were filed) and not refused (nothing proved they were not). The wire format
// has to carry that too: an unknown count must NOT serialize as `"issues_filed": 0`,
// or every downstream reader silently gets the confident zero back.
func TestScanLeftoversEvidenceUnknownIsNotZero(t *testing.T) {
	rep := ScanLeftoversEvidence(leftoversNarration, UnknownIssuesFiled(), false)
	if rep.Verdict != LeftoversFilingUnknown || !rep.Undecided() {
		t.Fatalf("no evidence: want %s/undecided, got %s/undecided=%v", LeftoversFilingUnknown, rep.Verdict, rep.Undecided())
	}
	if rep.Refused() {
		t.Errorf("no evidence must not refuse: that would assert a zero the fold does not have")
	}
	if _, known := rep.FiledCount(); known {
		t.Errorf("no evidence: FiledCount must report known=false, got %+v", rep)
	}
	if rep.Narrated == 0 || rep.Resolve == "" {
		t.Errorf("no evidence: want the narration count and a remediation, got %+v", rep)
	}
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), `"issues_filed"`) {
		t.Errorf("unknown count must be ABSENT from the JSON, not rendered as 0: %s", b)
	}
	if !strings.Contains(string(b), `"issues_filed_known":false`) {
		t.Errorf("JSON should state the count is not known: %s", b)
	}
}

// TestScanLeftoversEvidenceTailZeroIsUnknown: a count taken over a bounded TAIL of a
// transcript is a lower bound. Positive settles the question; zero does not, because the
// filing may sit before the window — so a tail zero resolves to unknown rather than to a
// refusal. Under-count over over-count, and never a confident zero.
func TestScanLeftoversEvidenceTailZeroIsUnknown(t *testing.T) {
	zero := ScanLeftoversEvidence(leftoversNarration, WitnessedIssuesFiledTail(0), false)
	if zero.Verdict != LeftoversFilingUnknown {
		t.Fatalf("tail zero: want %s, got %s", LeftoversFilingUnknown, zero.Verdict)
	}
	two := ScanLeftoversEvidence(leftoversNarration, WitnessedIssuesFiledTail(2), false)
	if two.Verdict != LeftoversClean {
		t.Fatalf("tail lower bound of 2: want %s, got %s", LeftoversClean, two.Verdict)
	}
	if count, known := two.FiledCount(); !known || count != 2 {
		t.Errorf("tail lower bound of 2: want 2/known, got %d/%v", count, known)
	}
}

// TestScanLeftoversEvidenceOutranksTheClaim is the ticket in one assertion: a run that
// asserts it filed three issues while its transcript evidences none is refused, and the
// report keeps BOTH numbers so the gap between claim and record is measurable.
func TestScanLeftoversEvidenceOutranksTheClaim(t *testing.T) {
	rep := ScanLeftoversEvidence(leftoversNarration, WitnessedIssuesFiled(0).Supersedes(3), false)
	if !rep.Refused() {
		t.Fatalf("claim of 3 against evidence of 0: want refused, got %s", rep.Verdict)
	}
	count, known := rep.FiledCount()
	if !known || count != 0 {
		t.Fatalf("the evidence count must survive the claim: got %d/%v", count, known)
	}
	if rep.IssuesFiledClaimed == nil || *rep.IssuesFiledClaimed != 3 {
		t.Fatalf("the superseded claim must stay visible, got %v", rep.IssuesFiledClaimed)
	}
	if rep.IssuesFiledSource != IssuesFiledFromTranscript {
		t.Errorf("source = %q, want %q", rep.IssuesFiledSource, IssuesFiledFromTranscript)
	}
}

// TestScanLeftoversTagsTheSelfReport: the legacy ScanLeftovers signature still works for
// callers with no transcript, but its report now says the number was ASSERTED, so a
// reader can tell a claimed count from a witnessed one.
func TestScanLeftoversTagsTheSelfReport(t *testing.T) {
	rep := ScanLeftovers(leftoversNarration, 2, false)
	if rep.Verdict != LeftoversClean {
		t.Fatalf("asserted 2 filed: want clean, got %s", rep.Verdict)
	}
	if rep.IssuesFiledSource != IssuesFiledAsserted {
		t.Errorf("source = %q, want %q — a self-report must not read as evidence", rep.IssuesFiledSource, IssuesFiledAsserted)
	}
}

// TestLeftoversDoctrineBindsAgentsMd couples code to doctrine: the fold quotes the
// AGENTS.md spine-first rule verbatim, and this asserts AGENTS.md still carries that
// exact line. If the doctrine text moves, this reds — forcing the constant and the
// rule to stay in lockstep rather than drifting silently.
func TestLeftoversDoctrineBindsAgentsMd(t *testing.T) {
	path := filepath.Join("..", "..", "AGENTS.md")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if !strings.Contains(string(b), Doctrine) {
		t.Fatalf("AGENTS.md must carry the doctrine line %q that ScanLeftovers binds to (code↔doctrine coupling broke)", Doctrine)
	}
}
