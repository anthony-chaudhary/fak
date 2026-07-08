// Package sensecheck is the "does this actually make sense?" side-car.
//
// It is a common-sense smell battery. The rest of the kernel proves things
// with HARD rungs — dos verify witnesses a ship, commit-audit witnesses a
// diff-vs-claim, go test witnesses behaviour. Those rungs answer "is this
// TRUE?". sensecheck answers a softer, human question the hard rungs never
// ask: "wait, does this even make sense?" — the thing a person mutters when
// a report says a cache hit-rate of 142%, a commit says "fixed" over a
// visible `exit status 1`, or a test asserts `1 == 1`.
//
// It exists because a claim can be TECHNICALLY CORRECT yet CONCEPTUALLY
// WRONG: every token verifiable, every number real, and the whole still
// incoherent. A hard witness passes such a claim (nothing it checks is
// false); a human reading it goes "hm, that can't be right". This leaf is
// the mechanised form of that "hm".
//
// It is deliberately ADVISORY. Every smell carries a could_be_ok_if escape
// hatch, because a heuristic that raises a question is honest only if it
// admits when the answer is "actually, fine". sensecheck never blocks, never
// closes an issue, never claims truth — it points at a spot and asks a
// person (or a harder rung) to look. See Report.Note, which travels with
// every result.
//
// The leaf is a PURE fold: a Subject of already-ingested text segments in,
// a Report out. It reads no files and parses no transcripts (that impure
// ingestion lives in cmd/fak/sensecheck.go), so the battery itself is
// deterministic and unit-testable offline. Same Subject in ⇒ same Report.
package sensecheck

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Schema is the versioned envelope tag every Report carries.
const Schema = "fak.sensecheck.v1"

// Severity ranks how loudly common sense objects to a smell. The ladder is
// deliberately three-wide: most heuristics are soft (a Note worth a glance),
// a few are firm (a Smell that does not add up), and a handful are near-
// certain incoherence (a Reek). The name of the finding unit is "smell"
// (Fowler's code-smell lineage): a hint to look, not a proof of a bug.
type Severity int

const (
	// SevNote is a mild "hm, worth a glance" — easily legitimate.
	SevNote Severity = iota
	// SevSmell is a firm "wait, that doesn't add up".
	SevSmell
	// SevReek is a near-certain "this is almost surely incoherent".
	SevReek
)

// String renders a Severity as its stable lowercase token.
func (s Severity) String() string {
	switch s {
	case SevNote:
		return "note"
	case SevSmell:
		return "smell"
	case SevReek:
		return "reek"
	default:
		return "unknown"
	}
}

// ParseSeverity maps a token back to a Severity (for a --fail-on gate).
// ok is false for an unrecognised token.
func ParseSeverity(tok string) (Severity, bool) {
	switch strings.ToLower(strings.TrimSpace(tok)) {
	case "note":
		return SevNote, true
	case "smell":
		return SevSmell, true
	case "reek":
		return SevReek, true
	default:
		return SevNote, false
	}
}

// Verdict is the closed roll-up of a Report.
type Verdict string

const (
	// VerdictClean means the battery ran and raised nothing.
	VerdictClean Verdict = "CLEAN"
	// VerdictSmells means at least one smell was raised.
	VerdictSmells Verdict = "SMELLS_FOUND"
	// VerdictAbstain means there was nothing to read (an empty subject) —
	// an honest "no answer", never a silent CLEAN. Mirrors dos verify's
	// source="none".
	VerdictAbstain Verdict = "ABSTAIN"
)

// Segment is one labelled slice of the subject the battery reads. The label
// is provenance only ("commit-subject", "diff", "turn 12 (assistant)") so a
// smell can point at WHERE in the subject it fired.
type Segment struct {
	Label string `json:"label"`
	Text  string `json:"text"`
}

// Subject is the already-ingested thing being sense-checked. Kind/Ref are
// provenance; Segments are the text the pure battery folds over.
type Subject struct {
	Kind     string    `json:"kind"` // "commit" | "log" | "session" | "text"
	Ref      string    `json:"ref"`  // sha, path, session id — provenance only
	Segments []Segment `json:"segments"`
}

// Smell is one raised common-sense question. Every Smell carries CouldBeOK:
// a heuristic that cannot say "actually, fine" is a false-positive machine.
type Smell struct {
	Detector  string `json:"detector"`       // stable id, e.g. "success-over-error"
	Severity  string `json:"severity"`       // note | smell | reek
	Segment   string `json:"segment"`        // the Segment.Label it fired in
	Evidence  string `json:"evidence"`       // the bounded excerpt that tripped it
	Why       string `json:"why"`            // the one-line common-sense objection
	CouldBeOK string `json:"could_be_ok_if"` // the escape hatch
}

// SubjectRef is the provenance echo in a Report (kind+ref, never the text).
type SubjectRef struct {
	Kind string `json:"kind"`
	Ref  string `json:"ref"`
}

// Report is the fold result. Note is ALWAYS present — the advisory fence
// travels with every answer so a reader can never take a smell for a verdict.
type Report struct {
	Schema  string         `json:"schema"`
	Subject SubjectRef     `json:"subject"`
	Verdict Verdict        `json:"verdict"`
	Smells  []Smell        `json:"smells"`
	Counts  map[string]int `json:"counts"` // by severity token
	Note    string         `json:"note"`
}

// fence is the honest advisory boundary carried by every Report.
const fence = "sensecheck is ADVISORY. It raises common-sense coherence questions from " +
	"deterministic heuristics; it does not witness truth. A smell means 'look here', not " +
	"'this is wrong' — each carries a could_be_ok_if that may well apply. Confirm with a " +
	"human read or a hard rung (fak dos verify / commit-audit / go test)."

// Check runs the whole battery over a Subject and folds one Report. It is a
// pure function: no I/O, no clock, deterministic in the Subject.
func Check(s Subject) Report {
	r := Report{
		Schema:  Schema,
		Subject: SubjectRef{Kind: s.Kind, Ref: s.Ref},
		Counts:  map[string]int{},
		Note:    fence,
	}

	// Nothing to read ⇒ abstain, never a silent CLEAN.
	total := 0
	for _, seg := range s.Segments {
		total += len(strings.TrimSpace(seg.Text))
	}
	if total == 0 {
		r.Verdict = VerdictAbstain
		return r
	}

	// Per-segment detectors: coherence smells local to one slice of text.
	for _, seg := range s.Segments {
		for _, d := range perSegment {
			r.Smells = append(r.Smells, d(seg)...)
		}
	}
	// Subject-level detectors: contradictions that only appear when two
	// claims from different segments are read together.
	whole := wholeSegment(s)
	for _, d := range perSubject {
		r.Smells = append(r.Smells, d(whole)...)
	}

	// Stable order: severity desc, then detector, then segment — so the
	// same Subject always renders the same Report.
	sort.SliceStable(r.Smells, func(i, j int) bool {
		si, sj := sevRank(r.Smells[i].Severity), sevRank(r.Smells[j].Severity)
		if si != sj {
			return si > sj
		}
		if r.Smells[i].Detector != r.Smells[j].Detector {
			return r.Smells[i].Detector < r.Smells[j].Detector
		}
		return r.Smells[i].Segment < r.Smells[j].Segment
	})

	for _, sm := range r.Smells {
		r.Counts[sm.Severity]++
	}
	if len(r.Smells) == 0 {
		r.Verdict = VerdictClean
	} else {
		r.Verdict = VerdictSmells
	}
	return r
}

// wholeSegment concatenates every segment into one labelled slice for the
// subject-level detectors, keeping segment labels inline so evidence stays
// locatable.
func wholeSegment(s Subject) Segment {
	var b strings.Builder
	for _, seg := range s.Segments {
		b.WriteString(seg.Text)
		b.WriteByte('\n')
	}
	return Segment{Label: "subject", Text: b.String()}
}

func sevRank(tok string) int {
	sv, _ := ParseSeverity(tok)
	return int(sv)
}

// ---- the battery ----------------------------------------------------------

// detector reads one Segment and returns any smells it raises.
type detector func(Segment) []Smell

var perSegment = []detector{
	detectSuccessOverError,
	detectVacuousGuard,
	detectImpossibleMagnitude,
	detectTautology,
	detectScopeInflation,
}

var perSubject = []detector{
	detectPlaceholderShipped,
	detectContradictionPair,
}

// clip bounds an evidence excerpt so a Report never carries a whole file.
func clip(s string) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	const max = 120
	if len(s) > max {
		return s[:max] + "…"
	}
	return s
}

func smell(det string, sev Severity, seg, evidence, why, couldBeOK string) Smell {
	return Smell{
		Detector:  det,
		Severity:  sev.String(),
		Segment:   seg,
		Evidence:  clip(evidence),
		Why:       why,
		CouldBeOK: couldBeOK,
	}
}

// --- 1. success-over-error -------------------------------------------------
// A success claim sitting next to an OBSERVED failure. The flagship
// "technically correct but conceptually wrong": "done" is a true statement
// of intent while `exit status 1` is a true statement of fact, and the two
// together are incoherent. Guards hard against the false positive ("fixed
// the error") by requiring the failure to be an OBSERVED signal (a colonised
// `Error:`, a non-zero exit, a panic/traceback, a FAIL marker, ✗/❌), not the
// bare noun "error".
var (
	reSuccessClaim = regexp.MustCompile(`(?i)\b(all tests? pass(ed|ing)?|tests? pass(ed|ing)?|` +
		`done|complete[d]?|shipped|it works now|works now|success(ful|fully)?|` +
		`verified|green|passing|✓|✅)\b`)
	reObservedFail = regexp.MustCompile(`(?i)(exit status [1-9]|exit code [1-9]|` +
		`\bpanic:|traceback \(most recent call last\)|--- fail|\bfail\b:|` +
		`\berror:|\bassertion (failed|error)|non-zero exit|✗|❌)`)
)

func detectSuccessOverError(seg Segment) []Smell {
	claim := reSuccessClaim.FindString(seg.Text)
	fail := reObservedFail.FindString(seg.Text)
	if claim == "" || fail == "" {
		return nil
	}
	return []Smell{smell("success-over-error", SevReek, seg.Label,
		"claims '"+strings.TrimSpace(claim)+"' near observed '"+strings.TrimSpace(fail)+"'",
		"a success is narrated over an observed failure signal in the same place — one of the two cannot be right",
		"the failure text is being QUOTED as the thing just fixed, or belongs to an earlier attempt a later line resolved")}
}

// --- 2. vacuous-guard ------------------------------------------------------
// A knob set to a value that nullifies its own purpose: a zero backoff
// between retries (indistinguishable from no backoff), zero retries on a
// retry knob. A zero timeout/ttl is left NOTE-level because "0 = unbounded"
// is a real API convention (the escape hatch), where a zero backoff almost
// never is.
var (
	reZeroBackoff = regexp.MustCompile(`(?i)\b(backoff|retry[_ ]?delay|retry[_ ]?interval)\b\s*[:=(]?\s*0\b`)
	reZeroRetry   = regexp.MustCompile(`(?i)\b(retries|max[_ ]?retries|max[_ ]?attempts)\b\s*[:=]\s*0\b`)
	reZeroTimeout = regexp.MustCompile(`(?i)\b(timeout|deadline|ttl)\b\s*[:=(]?\s*0\b`)
)

func detectVacuousGuard(seg Segment) []Smell {
	var out []Smell
	if m := reZeroBackoff.FindString(seg.Text); m != "" {
		out = append(out, smell("vacuous-guard", SevSmell, seg.Label, m,
			"a zero backoff between retries is indistinguishable from no backoff at all — the guard does nothing",
			"the retry loop deliberately hammers with no delay (a fast local poll) and that is intended"))
	}
	if m := reZeroRetry.FindString(seg.Text); m != "" {
		out = append(out, smell("vacuous-guard", SevSmell, seg.Label, m,
			"a retry knob set to 0 means the code never retries — the retry logic around it is dead",
			"0 is this API's sentinel for 'use the default retry count', not 'never retry'"))
	}
	if m := reZeroTimeout.FindString(seg.Text); m != "" {
		out = append(out, smell("vacuous-guard", SevNote, seg.Label, m,
			"a zero timeout/ttl can mean 'wait forever' or 'expire instantly' — either is a common footgun worth a glance",
			"0 is this API's documented sentinel for 'no timeout / never expire' and that is intended"))
	}
	return out
}

// --- 3. impossible-magnitude -----------------------------------------------
// A number that cannot mean what it is next to: a BOUNDED rate over 100%
// (a hit-rate/coverage/accuracy is a share of a whole and cannot exceed it),
// a negative count/duration. Requires a bounded-metric noun nearby so "300%
// faster" (a legitimate unbounded growth) is left alone.
var (
	reOverHundredPct = regexp.MustCompile(`(?i)\b(\d{3,}(?:\.\d+)?)\s*%[^.\n]{0,24}?` +
		`(hit[- ]?rate|coverage|accuracy|success ?rate|pass ?rate|utili[sz]ation|probability|confidence)`)
	reOverHundredPctPre = regexp.MustCompile(`(?i)(hit[- ]?rate|coverage|accuracy|success ?rate|` +
		`pass ?rate|utili[sz]ation|probability|confidence)[^.\n]{0,24}?\b(\d{3,}(?:\.\d+)?)\s*%`)
	reNegCount = regexp.MustCompile(`(?i)-\d+\s*(ms|seconds?|minutes?|hours?|bytes|items|rows|files|tests|calls)\b`)
)

func detectImpossibleMagnitude(seg Segment) []Smell {
	var out []Smell
	over := func(pctStr, whole string) {
		if v, err := strconv.ParseFloat(pctStr, 64); err == nil && v > 100 {
			out = append(out, smell("impossible-magnitude", SevReek, seg.Label, whole,
				"a bounded rate (a share of a whole) is reported above 100% — that is not a possible value for this metric",
				"the metric is legitimately amortized/over-subscribed (e.g. cache warmed across runs) and the denominator is stated"))
		}
	}
	if m := reOverHundredPct.FindStringSubmatch(seg.Text); m != nil {
		over(m[1], m[0])
	} else if m := reOverHundredPctPre.FindStringSubmatch(seg.Text); m != nil {
		over(m[2], m[0])
	}
	if m := reNegCount.FindString(seg.Text); m != "" {
		out = append(out, smell("impossible-magnitude", SevSmell, seg.Label, m,
			"a negative count or duration was reported — a quantity of things or elapsed time cannot be below zero",
			"the value is a signed DELTA (a decrease) and the minus sign is meaningful, not an absolute count"))
	}
	return out
}

// --- 4. tautology ----------------------------------------------------------
// A condition that cannot fail, so it verifies nothing: `x == x`, `1 == 1`,
// `if true`. RE2 has no backreferences, so the self-compare is found by a
// light per-line scan splitting on `==` and comparing the trimmed sides.
// `x != x` is EXCLUDED — it is the canonical NaN check, not a tautology.
var (
	reIfTrue      = regexp.MustCompile(`(?i)\bif\s+true\s*[{:]`)
	reIdentOrNum  = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.]*$|^-?\d+(?:\.\d+)?$|^(?:true|false)$`)
	reAssertTrue1 = regexp.MustCompile(`(?i)\bassert(?:_?true|equal|equals|eq)?\s*\(\s*(true|1)\s*(?:,\s*(true|1)\s*)?\)`)
)

func detectTautology(seg Segment) []Smell {
	var out []Smell
	if m := reIfTrue.FindString(seg.Text); m != "" {
		out = append(out, smell("tautology", SevSmell, seg.Label, m,
			"an `if true` branch always runs — the condition guards nothing",
			"it is a deliberate, commented placeholder for a condition wired in a later change"))
	}
	if m := reAssertTrue1.FindString(seg.Text); m != "" {
		out = append(out, smell("tautology", SevSmell, seg.Label, m,
			"an assertion of a constant truth (assert(true)/assert(1==1)) can never fail — the test proves nothing",
			"it is a scaffold assertion explicitly marked TODO to be filled in"))
	}
	for _, line := range strings.Split(seg.Text, "\n") {
		if !strings.Contains(line, "==") || strings.Contains(line, "!=") {
			continue
		}
		// Only look at the first `==` on the line; compare trimmed sides.
		i := strings.Index(line, "==")
		left := strings.TrimSpace(line[:i])
		rest := line[i+2:]
		// stop the right side at a trailing comment/brace/operator boundary
		right := strings.TrimSpace(rest)
		if j := strings.IndexAny(right, "{}()&|,;/"); j >= 0 {
			right = strings.TrimSpace(right[:j])
		}
		// left may carry a leading keyword like "if "/"assert "; take last token
		if fields := strings.Fields(left); len(fields) > 0 {
			left = fields[len(fields)-1]
		}
		if left == "" || left != right {
			continue
		}
		if !reIdentOrNum.MatchString(left) {
			continue
		}
		out = append(out, smell("tautology", SevSmell, seg.Label, strings.TrimSpace(line),
			"a value is compared for equality with itself (`"+left+" == "+left+"`) — always true, so it checks nothing",
			"it is an intentional identity/NaN diagnostic, or the two sides differ by a subtlety this scan cannot see"))
		break // one per segment is enough signal
	}
	return out
}

// --- 5. scope-inflation ----------------------------------------------------
// A universal claim ("all users", "every case", "always") sitting next to
// singular evidence ("one test", "the example"). Soft (NOTE) — universals
// are often separately proven; this just asks whether the breadth matches
// the evidence.
var (
	reUniversal = regexp.MustCompile(`(?i)\b(all users|every (case|user|file|time)|always works|100% of \w+|handles any \w+|never fails)\b`)
	reSingular  = regexp.MustCompile(`(?i)\b(one test|a single test|the example|this one case|a lone \w+|tested it once)\b`)
)

func detectScopeInflation(seg Segment) []Smell {
	u := reUniversal.FindString(seg.Text)
	s := reSingular.FindString(seg.Text)
	if u == "" || s == "" {
		return nil
	}
	return []Smell{smell("scope-inflation", SevNote, seg.Label,
		"'"+strings.TrimSpace(u)+"' backed by '"+strings.TrimSpace(s)+"'",
		"a universal claim rests on a single example in the same breath — the breadth may outrun the evidence",
		"the universal is proven elsewhere and the singular is only an illustration")}
}

// --- 6. placeholder-shipped (subject-level) --------------------------------
// A done/shipped claim over an artifact that still carries a placeholder
// (TODO/FIXME/lorem ipsum/example.com/REPLACE_ME). A placeholder alone is
// normal in-progress code; the smell is claiming it DONE.
var (
	reDoneClaim   = regexp.MustCompile(`(?i)\b(done|complete[d]?|shipped|finished|ready to (merge|ship)|all set|fully implemented)\b`)
	rePlaceholder = regexp.MustCompile(`(?i)\b(TODO|FIXME|XXX|HACK|changeme|replace[_ ]?me|placeholder|lorem ipsum|TBD)\b|example\.com|<your[- ]?\w+>`)
)

func detectPlaceholderShipped(seg Segment) []Smell {
	done := reDoneClaim.FindString(seg.Text)
	ph := rePlaceholder.FindString(seg.Text)
	if done == "" || ph == "" {
		return nil
	}
	return []Smell{smell("placeholder-shipped", SevSmell, seg.Label,
		"claims '"+strings.TrimSpace(done)+"' but still contains '"+strings.TrimSpace(ph)+"'",
		"the work is called done while a placeholder/TODO marker is still present — 'finished' and 'placeholder' do not co-exist",
		"the marker is a tracked follow-up explicitly declared out of THIS unit's scope")}
}

// --- 7. contradiction-pair (subject-level) ---------------------------------
// Two claims that cannot both hold across the whole subject: shipping while
// nothing changed; tests passing while tests are skipped/disabled.
var (
	reNoChange     = regexp.MustCompile(`(?i)\b(no changes|0 files? changed|nothing to commit|empty (diff|commit)|no diff)\b`)
	reDidShip      = regexp.MustCompile(`(?i)\b(fixed|implemented|shipped|added|changed|refactored|done)\b`)
	reTestsPass    = regexp.MustCompile(`(?i)\b(all tests? pass(ed|ing)?|tests? pass(ed|ing)?|suite is green|100% pass)\b`)
	reTestsSkipped = regexp.MustCompile(`(?i)\b(t\.skip|skipped|\.skip\(|disabled the test|commented out the (test|assert)|xfail|@skip)\b`)
)

func detectContradictionPair(seg Segment) []Smell {
	var out []Smell
	if reNoChange.FindString(seg.Text) != "" && reDidShip.FindString(seg.Text) != "" {
		out = append(out, smell("contradiction-pair", SevReek, seg.Label,
			"'"+strings.TrimSpace(reNoChange.FindString(seg.Text))+"' with '"+strings.TrimSpace(reDidShip.FindString(seg.Text))+"'",
			"the work claims a concrete change AND that nothing changed — both cannot be true of the same artifact",
			"the 'no changes' refers to a DIFFERENT artifact (a clean sub-tree) than the thing that was changed"))
	}
	if reTestsPass.FindString(seg.Text) != "" && reTestsSkipped.FindString(seg.Text) != "" {
		out = append(out, smell("contradiction-pair", SevSmell, seg.Label,
			"'"+strings.TrimSpace(reTestsPass.FindString(seg.Text))+"' with '"+strings.TrimSpace(reTestsSkipped.FindString(seg.Text))+"'",
			"tests are reported passing while tests are also skipped/disabled — a green suite that skipped the relevant test proves less than it says",
			"the skip is for an unrelated, separately-tracked test and the passing claim is about the rest of the suite"))
	}
	return out
}

// ---- ingestion helpers (pure) ---------------------------------------------

// TextSubject wraps raw text as a single-segment "text" subject.
func TextSubject(ref, raw string) Subject {
	return Subject{Kind: "text", Ref: ref, Segments: []Segment{{Label: "text", Text: raw}}}
}

// CommitSubject builds a "commit" subject from a message and a unified diff,
// kept as separate segments so a smell can say whether it fired in the CLAIM
// (message) or the CODE (diff).
func CommitSubject(ref, message, diff string) Subject {
	segs := []Segment{{Label: "commit-message", Text: message}}
	if strings.TrimSpace(diff) != "" {
		segs = append(segs, Segment{Label: "diff", Text: diff})
	}
	return Subject{Kind: "commit", Ref: ref, Segments: segs}
}

// LogSegments splits raw log/report text into paragraph segments (blank-line
// separated), so a "log" subject reads as coherent chunks rather than one
// blob. Empty paragraphs are dropped. The result labels each chunk by its
// 1-based paragraph index.
func LogSegments(raw string) []Segment {
	var segs []Segment
	n := 0
	for _, para := range strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n\n") {
		if strings.TrimSpace(para) == "" {
			continue
		}
		n++
		segs = append(segs, Segment{Label: fmt.Sprintf("para %d", n), Text: para})
	}
	return segs
}

// ---- render ---------------------------------------------------------------

// Render produces the human-readable report.
func Render(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "sensecheck %s  [%s %s]\n", r.Verdict, r.Subject.Kind, r.Subject.Ref)
	switch r.Verdict {
	case VerdictAbstain:
		b.WriteString("  (nothing to read — no answer)\n")
	case VerdictClean:
		b.WriteString("  no common-sense smells raised\n")
	default:
		fmt.Fprintf(&b, "  %d smell(s): %d reek, %d smell, %d note\n",
			len(r.Smells), r.Counts["reek"], r.Counts["smell"], r.Counts["note"])
		for _, sm := range r.Smells {
			fmt.Fprintf(&b, "\n  [%s] %s  (in %s)\n", strings.ToUpper(sm.Severity), sm.Detector, sm.Segment)
			fmt.Fprintf(&b, "    evidence : %s\n", sm.Evidence)
			fmt.Fprintf(&b, "    why      : %s\n", sm.Why)
			fmt.Fprintf(&b, "    ok if    : %s\n", sm.CouldBeOK)
		}
	}
	fmt.Fprintf(&b, "\n%s\n", r.Note)
	return b.String()
}

// MaxSeverity returns the highest severity present in a Report, and whether
// any smell was raised at all — the input to a --fail-on gate.
func MaxSeverity(r Report) (Severity, bool) {
	if len(r.Smells) == 0 {
		return SevNote, false
	}
	max := SevNote
	for _, sm := range r.Smells {
		if sv, ok := ParseSeverity(sm.Severity); ok && sv > max {
			max = sv
		}
	}
	return max, true
}
