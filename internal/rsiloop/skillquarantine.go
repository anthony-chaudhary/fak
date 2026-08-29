package rsiloop

// skillquarantine.go is the PRE-LOAD quality gate for agent-authored skills
// (#2839, part of #2834 Track A) — the fak answer to Hermes governing agent-created
// skills by STALENESS alone.
//
// Hermes' curator (modeled here by curator.go) auto-archives an agent-created skill
// only once it goes idle. A freshly-authored low-quality ("slop") skill therefore
// LOADS on every subsequent session — paying context cost each turn — until it
// happens to go stale, and the human skill HARDLINE (description <= 60 chars, no
// marketing voice, ship the helper scripts you invoke) is never applied to the
// agent-authored path at all.
//
// This gate closes that window at CREATION time. It combines three signals into one
// admission decision and, on failure, records a durable QUARANTINE row so the skill
// is held OUT of the library before it ever loads (a pre-admission hold, distinct
// from the curator's post-admission archive):
//
//   - HARDLINE — the structural authoring standard, computed here from frontmatter
//     and body (Hardline below). Pure text; no scoring model needed.
//   - SLOP — the body-quality verdict from the shipped slop scorecard
//     (tools/skill_slop_scorecard.py, #2911): verbatim_dump / vacuous_body /
//     marketing / one_off_narrative / missing_verification. The scorecard stays the
//     SCORER; this gate consumes its verdict and owns the STATE, so the two are not
//     re-implemented against each other. The scorecard's structural body checks are
//     also where the HARDLINE's "modern section order" aspect is enforced (a
//     transcript-shaped or heading-only body trips verbatim_dump / vacuous_body), so
//     Hardline below deliberately does NOT invent a bespoke H2-ordering rule — a
//     wrong ordering variant is exactly the "looser variant" #2839's confusion risk
//     warns against.
//   - DUPLICATE — a near-identical body to an existing library skill (the
//     "duplicated body vs an existing skill" heuristic), detected here over a
//     supplied corpus with a HIGH similarity floor so a genuinely new skill that
//     merely shares boilerplate is not over-flagged.
//
// The quarantine STATE reuses curator.go's ledger discipline exactly: every
// quarantine is one append-only journal row with a STRUCTURED, closed-vocabulary
// reason; a RELEASE is one more row that names a single prior quarantine by its Seq;
// and the governing state is FOLDED from the journal so a release is per-decision
// and survives a restart. It is NOT the ctxmmu skill-body screen: that quarantines a
// body for SECURITY (injection / secrets); this quarantines a skill for QUALITY.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	frontmattercodec "github.com/anthony-chaudhary/fak/internal/frontmatter"
)

// HardlineDescriptionMaxLen is the documented skill-authoring ceiling on a
// description: a one-line trigger, not a paragraph. 60 characters is the HARDLINE
// number the agent-authored path must meet the same way a human-reviewed one does.
const HardlineDescriptionMaxLen = 60

// HardlineViolation is the CLOSED set of structural authoring-standard defects the
// gate checks directly from frontmatter and body. Each is a HARD defect (a true
// standard breach, never a style nit), so any one is sufficient to quarantine.
type HardlineViolation string

const (
	// HardlineNameMissing — the frontmatter declares no name; the skill cannot be
	// addressed or curated.
	HardlineNameMissing HardlineViolation = "name_missing"
	// HardlineDescriptionMissing — the frontmatter declares no description; nothing
	// tells the loader when to surface the skill.
	HardlineDescriptionMissing HardlineViolation = "description_missing"
	// HardlineDescriptionTooLong — the description exceeds HardlineDescriptionMaxLen;
	// a paragraph masquerading as a trigger line.
	HardlineDescriptionTooLong HardlineViolation = "description_too_long"
	// HardlineDescriptionMarketing — the description uses sales voice ("seamless",
	// "revolutionary", …); a skill describes, it does not sell.
	HardlineDescriptionMarketing HardlineViolation = "description_marketing"
	// HardlineUnshippedScript — the body tells the reader to run a relative helper
	// script that was NOT shipped with the skill ("ship helper scripts"). An
	// instruction to run a file the library does not contain is unrunnable.
	HardlineUnshippedScript HardlineViolation = "unshipped_helper_script"
)

// marketingRE mirrors the vocabulary tools/skill_slop_scorecard.py flags, so the
// HARDLINE's marketing check and the slop scorecard's agree on what "sales voice"
// means rather than drifting apart.
var marketingRE = regexp.MustCompile(`(?i)\b(seamless(?:ly)?|revolutionary|cutting[- ]edge|best[- ]in[- ]class|world[- ]class|state[- ]of[- ]the[- ]art|game[- ]chang\w*|effortless(?:ly)?|blazing(?:ly)?[- ]fast|supercharge\w*|magical(?:ly)?|enterprise[- ]grade|next[- ]generation|turnkey|frictionless)\b`)

// scriptRefRE matches an instruction to run a RELATIVE helper script: an interpreter
// (python/bash/sh/zsh) or a "./" prefix followed by a path ending in .sh or .py. It
// deliberately ignores absolute paths and bare tool names (e.g. "go test") so only a
// genuine shipped-helper reference is considered.
var scriptRefRE = regexp.MustCompile(`(?:\b(?:python3?|bash|sh|zsh)\s+|\./)([\w./-]+\.(?:sh|py))\b`)

// QuarantineCandidate is the minimal projection of an agent-authored SKILL.md the gate
// reads: its resolved name and description (frontmatter) plus the body (post-
// frontmatter) and the set of helper files actually shipped alongside it. Build one
// with ParseQuarantineCandidate, or populate it directly in a test.
type QuarantineCandidate struct {
	// Name is the frontmatter name (resolved; the caller may fall back to the
	// directory name before constructing the candidate).
	Name string
	// Description is the frontmatter description, verbatim.
	Description string
	// Body is the SKILL.md content with the leading frontmatter block removed.
	Body string
	// ShippedScripts is the set of helper-script paths shipped WITH the skill
	// (skill-relative, e.g. "scripts/setup.sh"). A body reference to a relative
	// script absent from this set is an unshipped-helper HARDLINE violation.
	ShippedScripts []string
}

// ParseQuarantineCandidate splits a raw SKILL.md into the fields the gate reads: the
// leading "---"-delimited YAML frontmatter (flat name/description scalars only) and
// the remaining body. It is the same deliberately small, dependency-free shape the
// capindex resolver uses; anything it does not recognize is ignored.
func ParseQuarantineCandidate(rawSkillMd string, shippedScripts []string) QuarantineCandidate {
	c := QuarantineCandidate{ShippedScripts: shippedScripts}
	body := rawSkillMd
	if strings.HasPrefix(rawSkillMd, "---") {
		sc := bufio.NewScanner(strings.NewReader(rawSkillMd))
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		started, inBlock, consumed := false, false, 0
		for sc.Scan() {
			line := sc.Text()
			consumed += len(line) + 1
			if strings.TrimSpace(line) == "---" {
				if !started {
					started, inBlock = true, true
					continue
				}
				break // closing delimiter — body is whatever follows
			}
			if !inBlock {
				continue
			}
			key, val, ok := strings.Cut(line, ":")
			if !ok {
				continue
			}
			val, _ = frontmattercodec.DecodeScalar(val)
			switch strings.TrimSpace(key) {
			case "name":
				c.Name = val
			case "description":
				c.Description = val
			}
		}
		if consumed > len(rawSkillMd) {
			consumed = len(rawSkillMd) // closing delimiter was the last line, no body
		}
		body = rawSkillMd[consumed:]
	}
	c.Body = body
	return c
}

// Hardline returns the HARD structural authoring-standard violations of a candidate,
// in a stable order. An empty slice means the candidate clears the HARDLINE. It is
// pure — no scoring model, no I/O — so the same candidate grades identically every
// time.
func Hardline(c QuarantineCandidate) []HardlineViolation {
	var v []HardlineViolation
	if strings.TrimSpace(c.Name) == "" {
		v = append(v, HardlineNameMissing)
	}
	desc := strings.TrimSpace(c.Description)
	switch {
	case desc == "":
		v = append(v, HardlineDescriptionMissing)
	default:
		if len(desc) > HardlineDescriptionMaxLen {
			v = append(v, HardlineDescriptionTooLong)
		}
		if marketingRE.MatchString(desc) {
			v = append(v, HardlineDescriptionMarketing)
		}
	}
	if unshippedScripts(c) {
		v = append(v, HardlineUnshippedScript)
	}
	return v
}

// unshippedScripts reports whether the body invokes a relative helper script that is
// not in the candidate's ShippedScripts set. A reference is matched by basename so
// "scripts/x.sh" in the body is satisfied by a shipped "scripts/x.sh" regardless of
// how the caller spelled the shipped path.
func unshippedScripts(c QuarantineCandidate) bool {
	shipped := map[string]bool{}
	for _, s := range c.ShippedScripts {
		shipped[baseName(s)] = true
	}
	for _, m := range scriptRefRE.FindAllStringSubmatch(c.Body, -1) {
		if !shipped[baseName(m[1])] {
			return true
		}
	}
	return false
}

func baseName(p string) string {
	p = strings.TrimPrefix(p, "./")
	if i := strings.LastIndexAny(p, "/\\"); i >= 0 {
		p = p[i+1:]
	}
	return p
}

// SlopVerdict is the body-quality result the gate consumes from the slop scorecard
// (tools/skill_slop_scorecard.py, #2911). The scorecard is the SCORER; this gate
// only reads its decision — Rejected plus the tripped signal names and score — so
// the two are not re-implemented against each other.
type SlopVerdict struct {
	// Rejected is the scorecard's HARD verdict: at least one slop signal tripped.
	Rejected bool
	// Score is the scorecard's 0-100 score (100 = clean); recorded for forensics.
	Score float64
	// Signals names the tripped slop signals (e.g. "verbatim_dump", "marketing").
	Signals []string
}

// QuarantineReasonKind is the CLOSED, structured vocabulary for WHY a skill was held
// out of the library. The tokens stay distinct on purpose: an operator asking "why
// is this skill quarantined?" gets a token they can act on — a slop body, a HARDLINE
// breach, or a duplicate of an existing skill — never a generic reason.
type QuarantineReasonKind string

const (
	// QReasonSlop — the slop scorecard rejected the body (QuarantineReason.Signals /
	// SlopScore carry which signals tripped and the score).
	QReasonSlop QuarantineReasonKind = "slop"
	// QReasonHardline — one or more HARDLINE structural checks failed
	// (QuarantineReason.Hardline carries the violation tokens).
	QReasonHardline QuarantineReasonKind = "hardline"
	// QReasonDuplicate — the body is near-identical to an existing skill
	// (QuarantineReason.DuplicateOf names it).
	QReasonDuplicate QuarantineReasonKind = "duplicate"
)

// QuarantineReason is the structured reason attached to a quarantine decision.
// Exactly one Kind is set and the field that Kind requires is populated; Valid
// enforces that so a journaled quarantine is always answerable.
type QuarantineReason struct {
	Kind QuarantineReasonKind `json:"kind"`
	// SlopScore / Signals are set iff Kind == QReasonSlop.
	SlopScore float64  `json:"slop_score,omitempty"`
	Signals   []string `json:"signals,omitempty"`
	// Hardline is set iff Kind == QReasonHardline (the tripped violation tokens).
	Hardline []HardlineViolation `json:"hardline,omitempty"`
	// DuplicateOf is set iff Kind == QReasonDuplicate (the existing skill it copies).
	DuplicateOf string `json:"duplicate_of,omitempty"`
}

// Valid reports whether the reason names a known Kind AND carries the field that
// Kind requires — the check that keeps a generic/empty reason out of the journal.
func (r QuarantineReason) Valid() bool {
	switch r.Kind {
	case QReasonSlop:
		return len(r.Signals) > 0
	case QReasonHardline:
		return len(r.Hardline) > 0
	case QReasonDuplicate:
		return r.DuplicateOf != ""
	default:
		return false
	}
}

// String renders the reason as a short, human-readable phrase for the read-path.
func (r QuarantineReason) String() string {
	switch r.Kind {
	case QReasonSlop:
		return "slop (score " + strconv.FormatFloat(r.SlopScore, 'g', -1, 64) +
			": " + strings.Join(r.Signals, ", ") + ")"
	case QReasonHardline:
		parts := make([]string, len(r.Hardline))
		for i, h := range r.Hardline {
			parts[i] = string(h)
		}
		return "hardline (" + strings.Join(parts, ", ") + ")"
	case QReasonDuplicate:
		return "duplicate of skill " + r.DuplicateOf
	default:
		return "unknown reason"
	}
}

// QuarantineAction is the CLOSED set of actions a quarantine row can carry: quarantine
// holds a skill out of the library; release is the undo row naming one prior
// quarantine by its Seq.
type QuarantineAction string

const (
	// QActQuarantine holds a skill out of the library (it is now "quarantined").
	QActQuarantine QuarantineAction = "quarantine"
	// QActRelease is the undo row: it names one prior quarantine Seq and carries no
	// reason of its own.
	QActRelease QuarantineAction = "release"
)

// QuarantineRow is one append-only decision record. Seq is a monotonic per-ledger id
// so a release can name exactly one prior quarantine. For a quarantine row Reason is
// set and Releases is 0; for a release row Reason is empty and Releases is the Seq it
// undoes.
type QuarantineRow struct {
	Seq      int              `json:"seq"`
	Action   QuarantineAction `json:"action"`
	Skill    string           `json:"skill"`
	Reason   QuarantineReason `json:"reason,omitzero"`
	Releases int              `json:"releases,omitempty"`
}

// QuarantineEntry is the folded, current status of one skill the read-path returns:
// whether it is presently quarantined, and if so the governing reason and its Seq.
type QuarantineEntry struct {
	Skill        string
	Quarantined  bool
	Reason       QuarantineReason
	GoverningSeq int
}

// QuarantineLedger is the append-only journal of pre-load quarantine decisions plus
// the folded read-path over it — the per-decision, structured-reason state that holds
// a slop / HARDLINE-failing / duplicate skill out of the library before it loads. It
// mirrors CuratorLedger's discipline: the governing state is FOLDED from the journal,
// so a release is per-decision and free, and Log/Why answer from the rows alone.
type QuarantineLedger struct {
	path string
	rows []QuarantineRow
}

// OpenQuarantineLedger opens (or creates) the JSONL ledger at path and loads any
// existing rows so Seq assignment and the fold continue across process restarts. A
// path of "" keeps the ledger purely in-memory. The load is CORRUPTION-TOLERANT: a
// torn final line (an O_APPEND process killed mid-write) is skipped, not fatal.
func OpenQuarantineLedger(path string) (*QuarantineLedger, error) {
	l := &QuarantineLedger{path: path}
	if path == "" {
		return l, nil
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return l, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r QuarantineRow
		if err := json.Unmarshal(line, &r); err != nil {
			continue
		}
		l.rows = append(l.rows, r)
	}
	return l, nil
}

// Quarantine records a quarantine decision for skill with a structured reason,
// returning the assigned Seq. The reason must be Valid — an unstructured or empty
// reason is refused so every held skill stays answerable.
func (l *QuarantineLedger) Quarantine(skill string, reason QuarantineReason) (int, error) {
	if skill == "" {
		return 0, fmt.Errorf("quarantine: needs a skill name")
	}
	if !reason.Valid() {
		return 0, fmt.Errorf("quarantine: %q needs a valid structured reason, got %+v", skill, reason)
	}
	seq := len(l.rows) + 1
	return seq, l.append(QuarantineRow{Seq: seq, Action: QActQuarantine, Skill: skill, Reason: reason})
}

// Release undoes exactly ONE prior quarantine by its Seq. The fold recomputes each
// skill's status from the live (un-released) rows, so releasing seq N re-admits only
// N's skill and never touches a sibling. Releasing an unknown seq, a release row, or
// an already-released quarantine is refused.
func (l *QuarantineLedger) Release(seq int) error {
	target, ok := l.rowBySeq(seq)
	if !ok {
		return fmt.Errorf("quarantine: cannot release unknown decision seq %d", seq)
	}
	if target.Action == QActRelease {
		return fmt.Errorf("quarantine: seq %d is itself a release, not a quarantine", seq)
	}
	if l.isReleased(seq) {
		return fmt.Errorf("quarantine: decision seq %d is already released", seq)
	}
	next := len(l.rows) + 1
	return l.append(QuarantineRow{Seq: next, Action: QActRelease, Skill: target.Skill, Releases: seq})
}

// IsQuarantined answers "is this skill held out of the library, and why?" from the
// journal alone: it returns the governing reason and true if the skill is currently
// quarantined, or a zero reason and false if it is admittable (never quarantined, or
// its quarantine was released). This is the predicate a loader consults BEFORE load.
func (l *QuarantineLedger) IsQuarantined(skill string) (QuarantineReason, bool) {
	for _, e := range l.Log() {
		if e.Skill == skill {
			return e.Reason, e.Quarantined
		}
	}
	return QuarantineReason{}, false
}

// Log folds the journal into the current status of every skill it has touched,
// ordered by first appearance — the "quarantine log" read-path.
func (l *QuarantineLedger) Log() []QuarantineEntry {
	released := l.releasedSet()
	governing := map[string]QuarantineRow{}
	var order []string
	seen := map[string]bool{}
	for _, r := range l.rows {
		if r.Action == QActRelease {
			continue
		}
		if !seen[r.Skill] {
			seen[r.Skill] = true
			order = append(order, r.Skill)
		}
		if released[r.Seq] {
			delete(governing, r.Skill)
			continue
		}
		governing[r.Skill] = r
	}
	entries := make([]QuarantineEntry, 0, len(order))
	for _, skill := range order {
		if g, held := governing[skill]; held {
			entries = append(entries, QuarantineEntry{
				Skill: skill, Quarantined: true, Reason: g.Reason, GoverningSeq: g.Seq,
			})
			continue
		}
		entries = append(entries, QuarantineEntry{Skill: skill})
	}
	return entries
}

// Rows returns a copy of the append-only journal for inspection/telemetry.
func (l *QuarantineLedger) Rows() []QuarantineRow {
	return append([]QuarantineRow(nil), l.rows...)
}

func (l *QuarantineLedger) rowBySeq(seq int) (QuarantineRow, bool) {
	for _, r := range l.rows {
		if r.Seq == seq {
			return r, true
		}
	}
	return QuarantineRow{}, false
}

func (l *QuarantineLedger) releasedSet() map[int]bool {
	released := map[int]bool{}
	for _, r := range l.rows {
		if r.Action == QActRelease && r.Releases != 0 {
			released[r.Releases] = true
		}
	}
	return released
}

func (l *QuarantineLedger) isReleased(seq int) bool { return l.releasedSet()[seq] }

func (l *QuarantineLedger) append(r QuarantineRow) error {
	return appendLedgerRow(l.path, &l.rows, r)
}

// DuplicateThreshold is the line-set Jaccard similarity at/above which a candidate
// body is treated as a duplicate of an existing skill. It is deliberately HIGH so a
// genuinely new skill that merely shares boilerplate (a common header, a license
// line) is not flagged — the confusion risk #2839 names for the "duplicated body"
// heuristic is over-flagging on a shared-corpus comparison.
const DuplicateThreshold = 0.9

// DuplicateOf returns the name of the corpus skill whose body is near-identical to
// the candidate's (line-set Jaccard >= DuplicateThreshold), or "" if none is. corpus
// maps existing skill name -> body. A candidate is never compared against itself
// (matched by name).
func DuplicateOf(c QuarantineCandidate, corpus map[string]string) string {
	cand := lineSet(c.Body)
	if len(cand) == 0 {
		return ""
	}
	best, bestName := 0.0, ""
	for name, body := range corpus {
		if name == c.Name {
			continue
		}
		if j := jaccard(cand, lineSet(body)); j > best {
			best, bestName = j, name
		}
	}
	if best >= DuplicateThreshold {
		return bestName
	}
	return ""
}

// lineSet is the set of non-blank, trimmed body lines — the comparison unit for
// duplicate detection (order-insensitive so a reordered copy is still caught).
func lineSet(body string) map[string]bool {
	set := map[string]bool{}
	for _, ln := range strings.Split(body, "\n") {
		if s := strings.TrimSpace(ln); s != "" {
			set[s] = true
		}
	}
	return set
}

func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 0
	}
	inter := 0
	for k := range a {
		if b[k] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// Admit is the pre-load admission gate: it runs the HARDLINE check and duplicate
// detection on the candidate, folds in the supplied slop verdict, and — if ANY of
// the three fails — records a durable quarantine row (so the skill is held out of the
// library before it loads) and returns admitted=false with the governing reason and
// its Seq. A candidate that clears all three is admitted with no journal row written.
//
// Precedence when more than one signal fires is fixed and documented: HARDLINE
// (structural) first, then SLOP (body quality), then DUPLICATE — so the recorded
// reason names the most fundamental defect. slop is the scorecard's verdict; corpus
// (name -> body of existing skills) may be nil to skip duplicate detection.
func (l *QuarantineLedger) Admit(c QuarantineCandidate, slop SlopVerdict, corpus map[string]string) (admitted bool, seq int, reason QuarantineReason, err error) {
	if hv := Hardline(c); len(hv) > 0 {
		reason = QuarantineReason{Kind: QReasonHardline, Hardline: hv}
	} else if slop.Rejected {
		reason = QuarantineReason{Kind: QReasonSlop, SlopScore: slop.Score, Signals: slop.Signals}
	} else if dup := DuplicateOf(c, corpus); dup != "" {
		reason = QuarantineReason{Kind: QReasonDuplicate, DuplicateOf: dup}
	} else {
		return true, 0, QuarantineReason{}, nil
	}
	seq, err = l.Quarantine(c.Name, reason)
	if err != nil {
		return false, 0, reason, err
	}
	return false, seq, reason, nil
}
