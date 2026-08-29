package headlesslint

// leftovers.go — the RUN-LEVEL dual of Scan.
//
// Scan is a per-line sensor: it types each operator-directed note in a text and,
// for DeferredWork, suppresses a line that already cites a ticket. What it cannot
// see is the RUN: a final summary that narrates "there are two more things worth
// doing" while the run filed zero gh issues is a doctrine breach even though each
// line, read alone, is just prose. AGENTS.md makes the rule explicit —
//
//	At run end, dedupe and file every real leftover as an open issue; otherwise say nothing remains.
//
// — and #3670 asks for the enforcement that was missing: detect a run whose final
// summary narrates deferred / out-of-scope follow-ups AND cross-check whether the
// run filed (or resolved) any gh issue during its lifetime; refuse when leftovers
// are narrated but zero issues were filed, with an operator-overridable escape for
// "genuinely nothing left".
//
// ScanLeftovers is that fold: (final summary text, issues-filed count, override) in,
// a typed LeftoversReport out. Pure and stdlib-only — any layer (the Stop-hook guard
// fold, `fak headless-lint --leftovers`, a loop gate) can bind an agent's final turn
// to the doctrine through the same taxonomy.
//
// #5425 fixes the hole in the cross-check itself: the count used to arrive as a bare
// integer the AUDITED run handed over, so an agent could satisfy the doctrine by
// asserting a number about its own behaviour. ScanLeftoversEvidence takes an
// IssuesFiledEvidence instead — a count PLUS where it came from — so a witnessed
// count supersedes a claimed one, and "no evidence" stays UNKNOWN rather than
// collapsing into a confident zero.

import (
	"regexp"
	"strings"
)

// Doctrine is the AGENTS.md spine-first rule this fold enforces, quoted verbatim so
// code and doctrine stay coupled — TestLeftoversDoctrineBindsAgentsMd asserts
// AGENTS.md still carries this exact line, so if the doctrine text moves the fold's
// binding reds instead of silently drifting.
const Doctrine = "At run end, dedupe and file every real leftover as an open issue; otherwise say nothing remains."

// LeftoversSchema is the versioned envelope tag for a LeftoversReport.
const LeftoversSchema = "fak-leftovers-fold/1"

// Leftovers verdicts — the closed top-level judgment of ScanLeftovers.
const (
	// LeftoversClean: no unfiled leftovers. Either nothing was narrated, every
	// narrated follow-up already cited a filed ticket, at least one gh issue was
	// filed during the run, or the operator escape was set.
	LeftoversClean = "clean"
	// LeftoversUnfiled: the final summary narrates deferred / out-of-scope work but
	// the run filed zero gh issues and no escape was given — the breach this refuses.
	// Reaching this verdict requires a KNOWN zero: either a transcript that was read
	// end to end and showed no issue-creating tool call, or a run that itself claimed
	// nothing was filed.
	LeftoversUnfiled = "leftovers_unfiled"
	// LeftoversFilingUnknown: the final summary narrates deferred work and the fold
	// could not establish how many issues the run filed — no transcript, an unreadable
	// one, or only a bounded tail that happened to contain no filing. That is NOT the
	// same fact as a witnessed zero, so it gets its own verdict instead of being
	// silently reported as "0 filed" (#5425). Not refused: the honest answer to "did
	// this run file its leftovers?" is "cannot say", and refusing on it would be
	// exactly the confident-zero collapse this verdict exists to prevent.
	LeftoversFilingUnknown = "leftovers_filing_unknown"
)

// ---- issue-filing evidence (#5425) ------------------------------------------

// Issue-filing count provenance — the closed set of places an IssuesFiled count can
// come from. The distinction is the whole point: a count derived from the run's own
// tool-use record is a witness, while a count the audited run hands over is a claim
// about its own behaviour.
const (
	// IssuesFiledFromTranscript: counted from tool_use INPUTS over the run's whole
	// transcript. Authoritative — a zero here is a witnessed zero.
	IssuesFiledFromTranscript = "transcript"
	// IssuesFiledFromTranscriptTail: counted over a BOUNDED tail of the transcript, so
	// the count is a lower bound. Decisive when positive (a filing that is visible
	// happened); inconclusive at zero (the filing may sit before the window), which is
	// why a tail zero resolves to unknown rather than to a refusal.
	IssuesFiledFromTranscriptTail = "transcript-tail"
	// IssuesFiledAsserted: handed over by the run being audited — the deprecated
	// `--issues-filed N` self-report. Retained only as the fallback for callers with no
	// transcript to offer; any evidence supersedes it, and the report always discloses
	// that this is where the number came from.
	IssuesFiledAsserted = "asserted"
	// IssuesFiledNoEvidence: there was nothing to read. The count is unknown, not zero.
	IssuesFiledNoEvidence = "none"
)

// IssuesFiledEvidence is how many issues a run filed AND how that number was obtained.
// Known is the field that keeps "unknown" from collapsing into "zero": an absent or
// unreadable transcript leaves Count at 0 with Known=false, which the fold treats as
// "cannot say", never as "filed nothing". Claimed, when set, is the self-asserted
// count this evidence superseded — kept for disclosure so a soak can measure how often
// a run's claim runs ahead of its record.
type IssuesFiledEvidence struct {
	Count   int    `json:"count"`
	Known   bool   `json:"known"`
	Source  string `json:"source"`
	Claimed *int   `json:"claimed,omitempty"`
}

// WitnessedIssuesFiled is a count read off a run's whole transcript. Zero is a real,
// usable zero here: the record was read end to end and showed no issue being filed.
func WitnessedIssuesFiled(n int) IssuesFiledEvidence {
	if n < 0 {
		n = 0
	}
	return IssuesFiledEvidence{Count: n, Known: true, Source: IssuesFiledFromTranscript}
}

// WitnessedIssuesFiledTail is a count read off a BOUNDED tail of a transcript, so it is
// a lower bound. A positive lower bound still settles the doctrine question (at least
// one issue was filed), but a zero cannot: the filing may simply predate the window, so
// it resolves to unknown. Under-count over over-count, and no confident zero.
func WitnessedIssuesFiledTail(n int) IssuesFiledEvidence {
	if n <= 0 {
		return IssuesFiledEvidence{Source: IssuesFiledFromTranscriptTail}
	}
	return IssuesFiledEvidence{Count: n, Known: true, Source: IssuesFiledFromTranscriptTail}
}

// UnknownIssuesFiled is the reading when there was no evidence to read at all.
func UnknownIssuesFiled() IssuesFiledEvidence {
	return IssuesFiledEvidence{Source: IssuesFiledNoEvidence}
}

// ClaimedIssuesFiled wraps a self-asserted count — the legacy `--issues-filed N` path.
// It is Known (the fold can act on it) but its Source marks it as unwitnessed, which is
// what lets a reader tell a proven count from a claimed one at a glance.
func ClaimedIssuesFiled(n int) IssuesFiledEvidence {
	if n < 0 {
		n = 0
	}
	return IssuesFiledEvidence{Count: n, Known: true, Source: IssuesFiledAsserted}
}

// Supersedes records that this evidence displaced a self-asserted count of n. It never
// changes the evidence itself — the claim is carried only so the report can say what
// was claimed alongside what was witnessed.
func (e IssuesFiledEvidence) Supersedes(n int) IssuesFiledEvidence {
	e.Claimed = &n
	return e
}

// filed reports whether the evidence positively establishes that the run filed at least
// one issue. Anything short of that — an unknown reading, or a known zero — is not a
// filing, so the doctrine's cross-check stays deny-by-default.
func (e IssuesFiledEvidence) filed() bool { return e.Known && e.Count > 0 }

// LeftoversHit is one line of the final summary that narrates a leftover.
type LeftoversHit struct {
	Line    int    `json:"line"`
	Match   string `json:"match"`
	Excerpt string `json:"excerpt"`
}

// LeftoversReport is the fold over one final summary plus the run's issue-filing
// evidence. Verdict is LeftoversUnfiled iff leftovers were narrated, the filing count
// is KNOWN to be zero, and no operator escape was set; LeftoversFilingUnknown when
// leftovers were narrated but the count could not be established; otherwise clean.
//
// IssuesFiled is a POINTER on purpose: nil is "unknown", and it serializes as an
// absent field rather than as `"issues_filed": 0`. A JSON reader therefore cannot
// mistake "we could not tell" for "the run filed nothing" — the exact collapse #5425
// exists to prevent. IssuesFiledSource names the provenance, and IssuesFiledClaimed
// carries any self-asserted count the evidence superseded.
type LeftoversReport struct {
	Schema             string         `json:"schema"`
	Verdict            string         `json:"verdict"`
	Doctrine           string         `json:"doctrine"`
	Narrated           int            `json:"narrated"`
	IssuesFiled        *int           `json:"issues_filed,omitempty"`
	IssuesFiledKnown   bool           `json:"issues_filed_known"`
	IssuesFiledSource  string         `json:"issues_filed_source,omitempty"`
	IssuesFiledClaimed *int           `json:"issues_filed_claimed,omitempty"`
	Overridden         bool           `json:"overridden"`
	Hits               []LeftoversHit `json:"hits,omitempty"`
	Resolve            string         `json:"resolve,omitempty"`
}

// Refused reports whether this run narrated leftovers it did not file — the arm a
// Stop-hook / guard gate blocks (or nudges) on. A clean report is never refused, and
// neither is an undecided one: refusing on missing evidence would assert a zero the
// fold does not have.
func (r LeftoversReport) Refused() bool { return r.Verdict == LeftoversUnfiled }

// Undecided reports the third arm: leftovers were narrated but the filing count could
// not be established. Callers that must branch on "unknown" read this rather than
// inspecting a count that does not exist.
func (r LeftoversReport) Undecided() bool { return r.Verdict == LeftoversFilingUnknown }

// FiledCount is the accessor for the count and whether it is known at all. It exists so
// no caller has to dereference a pointer that may legitimately be nil, and so "unknown"
// is impossible to read as zero by accident.
func (r LeftoversReport) FiledCount() (int, bool) {
	if r.IssuesFiled == nil {
		return 0, false
	}
	return *r.IssuesFiled, r.IssuesFiledKnown
}

// leftoverRes is the ordered detection table for deferred / out-of-scope narration:
// the "there are two more things worth doing", "out of scope", "follow-up", "left to
// do", "TODO" prose an agent lists at the end instead of filing. A line that already
// cites a ticket (hasTicketRef) is scoping, not narration, and is skipped before this
// runs — so only a BARE punt counts.
var leftoverRes = []*regexp.Regexp{
	re(`\b(a couple|a few|two|three|four|several|some|another|more) (more )?things?\b`),
	re(`\bthings? (worth|left|still|to do|remaining|we|you)\b`),
	re(`\bworth (doing|adding|fixing|handling|filing)\b`),
	re(`\bout[ -]of[ -]scope\b`),
	re(`\bfollow[ -]?ups?\b`),
	re(`\bleft ?overs?\b`),
	re(`\bleft to do\b`),
	re(`\bstill (to do|left|remaining|need|needs|outstanding)\b`),
	re(`\b(remaining|outstanding) (work|item|items|task|tasks|follow|piece|pieces)\b`),
	re(`\bnext steps?\b`),
	re(`\btodos?\b`),
	re(`\bfix ?me\b`),
	re(`\b(defer|deferring|deferred|punt|punted|punting)\b`),
	re(`\bnot (yet )?(done|addressed|handled|implemented|covered|wired)\b`),
	re(`\bcould also\b`),
	re(`\bwe (can|could|should|might) (also|still|later|revisit|follow)\b`),
	re(`\b(can|could|will) be (done|added|addressed|handled|fixed|implemented) later\b`),
}

// ScanLeftovers folds a final summary and a SELF-ASSERTED issue-filing count into a
// LeftoversReport. It is the #3670 signature, kept for callers that have no transcript
// to offer; the resulting report is tagged IssuesFiledAsserted so a reader can see the
// number was claimed rather than witnessed. Prefer ScanLeftoversEvidence.
//
// Deprecated as the primary path (#5425): a count the audited run supplies about its
// own behaviour is precisely the claim this substrate exists to refuse.
func ScanLeftovers(summary string, issuesFiled int, override bool) LeftoversReport {
	return ScanLeftoversEvidence(summary, ClaimedIssuesFiled(issuesFiled), override)
}

// ScanLeftoversEvidence folds a final summary and the run's issue-filing EVIDENCE into
// a LeftoversReport. The three arms:
//   - narrated leftovers + a known-zero filing count + !override -> LeftoversUnfiled
//     (refused): #3670's done-condition, now reachable only on a count that is known.
//   - the same summary once the follow-ups were filed (a known positive count), or a
//     line that itself cites a ticket, or an operator escape -> LeftoversClean.
//   - narrated leftovers + no usable evidence -> LeftoversFilingUnknown: not clean
//     (nothing proved the leftovers were filed) and not refused (nothing proved they
//     were not). Absence of evidence is not a zero.
//
// filed is the run-lifetime evidence for gh issues the run created — the cross-check the
// doctrine turns on. override is the operator escape for "genuinely nothing left": it
// forces clean even when leftovers were narrated.
func ScanLeftoversEvidence(summary string, filed IssuesFiledEvidence, override bool) LeftoversReport {
	rep := LeftoversReport{
		Schema:             LeftoversSchema,
		Verdict:            LeftoversClean,
		Doctrine:           Doctrine,
		IssuesFiledKnown:   filed.Known,
		IssuesFiledSource:  filed.Source,
		IssuesFiledClaimed: filed.Claimed,
		Overridden:         override,
	}
	if filed.Known {
		count := filed.Count
		rep.IssuesFiled = &count
	}
	for i, raw := range splitLines(summary) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		low := strings.ToLower(line)
		if hasTicketRef(low) {
			// A leftover that cites a filed ticket ("out of scope, tracked in #4001")
			// is scoping, not bare narration — the honest end-of-run shape.
			continue
		}
		if m := firstMatch(leftoverRes, low, line); m != "" {
			rep.Hits = append(rep.Hits, LeftoversHit{
				Line:    i + 1,
				Match:   clip(m, 120),
				Excerpt: clip(line, 200),
			})
		}
	}
	rep.Narrated = len(rep.Hits)
	if rep.Narrated == 0 || override || filed.filed() {
		return rep
	}
	if !filed.Known {
		// Narrated leftovers, and nothing to check them against. Say so.
		rep.Verdict = LeftoversFilingUnknown
		rep.Resolve = "re-run the fold with the run's transcript so the issues-filed count is counted from tool-use evidence; until then this run's filing count is unknown, which is not the same as zero"
		return rep
	}
	rep.Verdict = LeftoversUnfiled
	rep.Resolve = "file each narrated leftover as an open gh issue (dedupe → done-condition → leak-check → label), then report the issue numbers; pass the operator escape only if there is genuinely nothing left"
	return rep
}
