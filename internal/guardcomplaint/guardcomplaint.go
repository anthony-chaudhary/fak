package guardcomplaint

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

// Schema is the stable schema tag stamped on the machine-readable result.
const Schema = "fak.guard-complaint.v1"

// Label is the gh label attached to a newly-filed complaint so the appeal channel
// is filterable apart from the kernel's own RSI findings (guardroute).
const Label = "guard-complaint"

// markerRE matches the HTML-comment dedup marker stamped into every complaint body.
// It carries the stable key so a re-filed complaint about the SAME class updates one
// issue in place (and escalates its occurrence count) instead of opening duplicates.
var markerRE = regexp.MustCompile(`<!--\s*fak-guard-complaint-key:\s*([^>\s]+)\s*-->`)

// occurrencesRE reads the `occurrences` field back out of an existing complaint body so
// an update can increment it. A recurring false-positive is a stronger signal than a
// one-off — the same threshold logic guardrsi applies to recurring denials, but here the
// agent (not the journal fold) is the one asserting the refusal was wrong.
var occurrencesRE = regexp.MustCompile("(?m)^- occurrences: `?(\\d+)`?")

// Kinds is the closed set of complaint categories. The kind is the agent's claim about
// WHAT is wrong with the guard decision, which the journal evidence cannot settle on its
// own: a correct DENY and a false-positive DENY are byte-identical in the journal.
var Kinds = map[string]string{
	"false-positive": "a legitimate tool call the capability floor refused (the dominant case the journal cannot self-detect)",
	"over-broad":     "a gate that refuses more than its stated danger class — collateral denials",
	"latency":        "a guard or gate slow enough to hurt the loop (maps to GATE_LATENCY_REGRESSION)",
	"confusing":      "a refusal whose reason/message did not tell the agent how to recover",
	"other":          "a guard behavior the agent judges wrong that fits no other kind",
}

// DefaultKind is used when the agent names none.
const DefaultKind = "false-positive"

// Domains is the closed set of complaint domains (#5191). "guard" is the original
// capability-floor appeal channel (a wrong `fak guard` decision); "workflow" carries
// non-guard agentic-dev friction — shared-tree clobbers, tool timeouts, lane collisions —
// through the SAME deduping gh channel so operator-visible loop friction files a tracked
// ticket instead of a private journal note. The two domains share the create/update/
// occurrence plumbing and differ only in their kind vocabulary, title/body framing, and label.
var Domains = map[string]string{
	"guard":    "an appeal against a `fak guard` capability-floor decision (the original channel)",
	"workflow": "non-guard agentic-dev friction (shared-tree clobber, tool timeout, lane collision)",
}

// DefaultDomain defaults an unspecified domain to the original guard-appeal channel, so every
// pre-#5191 caller and every existing dedup key stays byte-identical.
const DefaultDomain = "guard"

// WorkflowKinds is the closed kind vocabulary for the workflow domain — the non-guard
// counterpart of Kinds. These are loop-level friction classes an agent hits that no guard
// refused, so they cannot ride the guard-reason/journal-witness shape.
var WorkflowKinds = map[string]string{
	"shared-tree-clobber": "a peer's concurrent edit or stage on the shared trunk overwrote or raced this agent's work",
	"tool-timeout":        "a tool or command that timed out or hung long enough to stall the loop",
	"lane-collision":      "two workers contending on the same files or lease so one had to back off or redo work",
	"other":               "agentic-dev friction worth tracking that fits no other workflow kind",
}

// DefaultWorkflowKind is used when a workflow complaint names no kind.
const DefaultWorkflowKind = "other"

// WorkflowLabel is the gh label for workflow-domain friction complaints, filterable apart
// from the guard-appeal channel (Label) while sharing its deduping plumbing.
const WorkflowLabel = "workflow-complaint"

// KindsFor returns the closed kind vocabulary for a domain: the workflow set for "workflow",
// else the guard Kinds (the default and every unknown domain, which NormalizeDomain rejects
// before it reaches here).
func KindsFor(domain string) map[string]string {
	if domain == "workflow" {
		return WorkflowKinds
	}
	return Kinds
}

// LabelFor returns the gh label for a domain: WorkflowLabel for "workflow", else the guard Label.
func LabelFor(domain string) string {
	if domain == "workflow" {
		return WorkflowLabel
	}
	return Label
}

func defaultKindFor(domain string) string {
	if domain == "workflow" {
		return DefaultWorkflowKind
	}
	return DefaultKind
}

// NormalizeDomain lowercases/validates a domain, defaulting an empty one to guard. It returns
// the canonical domain and an error naming the closed set when the domain is unknown.
func NormalizeDomain(domain string) (string, error) {
	d := strings.ToLower(strings.TrimSpace(domain))
	if d == "" {
		return DefaultDomain, nil
	}
	if _, ok := Domains[d]; !ok {
		return "", fmt.Errorf("unknown complaint domain %q (want one of: %s)", domain, strings.Join(sortedKeys(Domains), ", "))
	}
	return d, nil
}

// Evidence is the witnessed half of a complaint: a real adjudicated verdict pulled from
// the decision journal (or supplied manually). The agent's rationale is a self-report;
// this is the non-forgeable record that the refusal it is appealing actually happened.
type Evidence struct {
	Source      string `json:"source"` // "journal" | "manual" | "none"
	JournalPath string `json:"journal_path,omitempty"`
	Seq         uint64 `json:"seq,omitempty"`
	// CallSeq is the kernel's per-call submission id (journal `call_seq`), i.e. the
	// identity of the REFUSED CALL — as opposed to Seq, which is the journal ROW
	// ordinal and restarts at 1 in every file. Together with TraceID (the session)
	// it names one denial exactly, which is what makes an appeal auditable back to
	// the command that was refused rather than to whichever similar row was newest.
	CallSeq    uint64 `json:"call_seq,omitempty"`
	TSUnixNano int64  `json:"ts_unix_nano,omitempty"`
	Verdict    string `json:"verdict,omitempty"`
	Tool       string `json:"tool,omitempty"`
	Reason     string `json:"reason,omitempty"`
	By         string `json:"by,omitempty"`
	TraceID    string `json:"trace_id,omitempty"`
	ArgsDigest string `json:"args_digest,omitempty"`
}

// Complaint is one agent-authored complaint. In the guard domain (the default) it is an appeal
// against a `fak guard` decision; in the workflow domain (#5191) it is a non-guard dev-friction
// report. The fields are shared; Reason/Tool/Evidence are guard-shaped and simply stay empty for
// most workflow friction.
type Complaint struct {
	Domain    string    `json:"domain,omitempty"` // "guard" (default) | "workflow"
	Kind      string    `json:"kind"`
	Reason    string    `json:"reason,omitempty"` // the guard reason token being appealed (e.g. FILE_ADMISSION)
	Tool      string    `json:"tool,omitempty"`   // the refused tool (e.g. Bash, Write)
	Summary   string    `json:"summary"`          // one-line headline
	Rationale string    `json:"rationale"`        // why the agent judges the guard wrong
	Evidence  *Evidence `json:"evidence,omitempty"`
}

// domain resolves the complaint's domain, defaulting an empty one to guard so a zero-value
// Complaint and every pre-#5191 caller stays on the original channel with byte-identical output.
func (c Complaint) domain() string {
	if d := strings.ToLower(strings.TrimSpace(c.Domain)); d != "" {
		return d
	}
	return DefaultDomain
}

// PlanRow is one create/update decision for a complaint. Occurrences is the escalating
// count carried in the issue body (1 on first file, +1 per re-file of the same key).
type PlanRow struct {
	Action      string `json:"action"`
	Key         string `json:"key"`
	Number      *int   `json:"number"`
	State       string `json:"state"`
	Title       string `json:"title"`
	Body        string `json:"-"`
	Domain      string `json:"domain,omitempty"`
	Kind        string `json:"kind"`
	Reason      string `json:"reason,omitempty"`
	Tool        string `json:"tool,omitempty"`
	Occurrences int    `json:"occurrences"`
}

// Result is the machine-readable plan/result fold (mirrors dogfoodissues.Result).
type Result struct {
	Schema  string                  `json:"schema"`
	Mode    string                  `json:"mode"`
	Planned []PlanRow               `json:"planned"`
	Synced  []dogfoodissues.SyncRow `json:"synced"`
}

// NormalizeKindFor lowercases/validates a kind against a domain's closed set, defaulting an
// empty one to that domain's default kind. The error names the domain's own vocabulary so a
// workflow complaint is not told to use a guard kind and vice-versa.
func NormalizeKindFor(domain, kind string) (string, error) {
	kinds := KindsFor(domain)
	k := strings.ToLower(strings.TrimSpace(kind))
	if k == "" {
		return defaultKindFor(domain), nil
	}
	if _, ok := kinds[k]; !ok {
		// The guard domain keeps its historical "unknown complaint kind" wording (unqualified);
		// a non-guard domain names itself so a workflow complaint is not told to use a guard kind.
		label := "complaint kind"
		if domain != DefaultDomain {
			label = domain + " complaint kind"
		}
		return "", fmt.Errorf("unknown %s %q (want one of: %s)", label, kind, strings.Join(sortedKeys(kinds), ", "))
	}
	return k, nil
}

// NormalizeKind is the guard-domain kind normalizer, preserved byte-for-byte for every
// pre-#5191 caller (it delegates to NormalizeKindFor on the default guard domain).
func NormalizeKind(kind string) (string, error) {
	return NormalizeKindFor(DefaultDomain, kind)
}

// sortedKeys returns a map's keys sorted — a stable, deterministic ordering for the closed-set
// error messages (Kinds, WorkflowKinds, Domains).
func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

var slugStripRE = regexp.MustCompile(`[^a-z0-9]+`)

// slug normalizes free text into a stable, bounded key segment so trivial wording drift
// does not split one complaint into two issues.
func slug(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = slugStripRE.ReplaceAllString(s, "-")
	s = strings.Trim(s, "-")
	if len(s) > 60 {
		s = strings.Trim(s[:60], "-")
	}
	if s == "" {
		s = "unspecified"
	}
	return s
}

// Key derives the stable dedup key for a complaint: kind + appealed-reason + refused-tool
// + a slug of the summary. The same recurring false-positive folds onto one issue; two
// genuinely different appeals (different summary) stay apart.
func (c Complaint) Key() string {
	reason := strings.TrimSpace(c.Reason)
	if reason == "" {
		reason = "none"
	}
	tool := strings.TrimSpace(c.Tool)
	if tool == "" {
		tool = "any"
	}
	// The domain prefixes the key so a guard and a workflow complaint that happen to share a
	// summary never fold onto one issue. The guard prefix is left as the historical literal
	// "guard-complaint" so every pre-#5191 key is byte-identical.
	prefix := "guard-complaint"
	if c.domain() == "workflow" {
		prefix = "workflow-complaint"
	}
	return strings.Join([]string{
		prefix,
		slug(c.Kind),
		slug(reason),
		slug(tool),
		slug(c.Summary),
	}, "/")
}

// oneLine collapses every whitespace run — including CR/LF line breaks — to a
// single space. The summary is documented as a one-line headline, but nothing
// upstream enforces it: a summary pasted with an embedded newline would
// otherwise flow verbatim into the gh issue title and break the one-row-per-
// plan Render output.
func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// Title renders the issue title — clear and self-describing, with the appealed reason and
// tool inline so the tracker shows the class at a glance. Line breaks in the
// summary are collapsed: issue titles and Render rows are single-line surfaces.
func (c Complaint) Title() string {
	var b strings.Builder
	if c.domain() == "workflow" {
		b.WriteString("workflow friction [")
	} else {
		b.WriteString("guard complaint [")
	}
	b.WriteString(c.Kind)
	b.WriteString("]")
	scope := []string{}
	if r := strings.TrimSpace(c.Reason); r != "" {
		scope = append(scope, r)
	}
	if t := strings.TrimSpace(c.Tool); t != "" {
		scope = append(scope, "tool="+t)
	}
	if len(scope) > 0 {
		b.WriteString(" ")
		b.WriteString(strings.Join(scope, " "))
	}
	if s := oneLine(c.Summary); s != "" {
		b.WriteString(" — ")
		b.WriteString(s)
	}
	return b.String()
}

// Body renders the stable, marker-stamped issue body for a complaint at a given occurrence
// count. The marker (line 1) drives dedup; the rest is the structured appeal plus the
// journal witness, so a maintainer reads the agent's argument AND the non-forgeable record
// of the refusal it is appealing.
func (c Complaint) Body(occurrences int) string {
	if occurrences < 1 {
		occurrences = 1
	}
	if c.domain() == "workflow" {
		return c.workflowBody(occurrences)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-guard-complaint-key: %s -->\n", c.Key())
	b.WriteString("# Guard complaint (agent appeal)\n\n")
	b.WriteString("An agent governed by `fak guard` judged a guard decision wrong and filed this appeal. ")
	b.WriteString("This is the **subjective** channel: a false-positive refusal is byte-identical to a correct one in ")
	b.WriteString("the decision journal, so `fak guard-verdict-rsi` cannot surface it — only the agent that made the call ")
	b.WriteString("knows it was legitimate.\n\n")
	fmt.Fprintf(&b, "- kind: `%s` — %s\n", c.Kind, Kinds[c.Kind])
	if r := strings.TrimSpace(c.Reason); r != "" {
		fmt.Fprintf(&b, "- appealed reason: `%s`\n", r)
	}
	if t := strings.TrimSpace(c.Tool); t != "" {
		fmt.Fprintf(&b, "- refused tool: `%s`\n", t)
	}
	fmt.Fprintf(&b, "- occurrences: `%d`\n", occurrences)
	fmt.Fprintf(&b, "- stable key: `%s`\n\n", c.Key())

	b.WriteString("## Why the agent thinks the guard is wrong\n\n")
	rationale := strings.TrimSpace(c.Rationale)
	if rationale == "" {
		rationale = "_(none given)_"
	}
	b.WriteString(rationale)
	b.WriteString("\n\n")

	b.WriteString("## Evidence\n\n")
	b.WriteString(c.evidenceBlock())
	b.WriteString("\n")

	b.WriteString("---\n")
	b.WriteString("Filed by `fak complain`. Re-running it for the same class updates THIS issue in place and bumps ")
	b.WriteString("the occurrence count rather than opening a duplicate. A confirmed false positive is a floor bug to fix; ")
	b.WriteString("a rejected appeal is closed with the reason the refusal was correct.\n")
	return b.String()
}

// workflowBody renders the issue body for a workflow-domain (#5191) friction complaint. It
// reuses the SAME dedup marker tag (fak-guard-complaint-key) so MarkerKey/occurrence folding
// work unchanged; only the framing differs — this is non-guard loop friction, so there is no
// capability-floor decision to appeal and (usually) no journal witness.
func (c Complaint) workflowBody(occurrences int) string {
	if occurrences < 1 {
		occurrences = 1
	}
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-guard-complaint-key: %s -->\n", c.Key())
	b.WriteString("# Workflow friction (agent report)\n\n")
	b.WriteString("An agent hit agentic-dev friction that no `fak guard` decision caused — a shared-tree ")
	b.WriteString("clobber, a tool timeout, a lane collision, or similar loop friction. It is filed through ")
	b.WriteString("the same deduping complaint channel so recurring friction reads as the tracked signal it ")
	b.WriteString("is instead of a private journal note.\n\n")
	fmt.Fprintf(&b, "- domain: `workflow`\n")
	fmt.Fprintf(&b, "- kind: `%s` — %s\n", c.Kind, WorkflowKinds[c.Kind])
	if t := strings.TrimSpace(c.Tool); t != "" {
		fmt.Fprintf(&b, "- tool: `%s`\n", t)
	}
	if r := strings.TrimSpace(c.Reason); r != "" {
		fmt.Fprintf(&b, "- context: `%s`\n", r)
	}
	fmt.Fprintf(&b, "- occurrences: `%d`\n", occurrences)
	fmt.Fprintf(&b, "- stable key: `%s`\n\n", c.Key())

	b.WriteString("## What happened\n\n")
	rationale := strings.TrimSpace(c.Rationale)
	if rationale == "" {
		rationale = "_(none given)_"
	}
	b.WriteString(rationale)
	b.WriteString("\n\n")

	if c.Evidence != nil && c.Evidence.Source != "" && c.Evidence.Source != "none" {
		b.WriteString("## Evidence\n\n")
		b.WriteString(c.evidenceBlock())
		b.WriteString("\n")
	}

	b.WriteString("---\n")
	b.WriteString("Filed by `fak complain --domain workflow`. Re-running it for the same class updates THIS ")
	b.WriteString("issue in place and bumps the occurrence count rather than opening a duplicate. Recurring ")
	b.WriteString("workflow friction is a loop-ergonomics bug to fix.\n")
	return b.String()
}

func (c Complaint) evidenceBlock() string {
	e := c.Evidence
	if e == nil || e.Source == "" || e.Source == "none" {
		return "_No journal verdict attached — this appeal rests on the agent's rationale alone. " +
			"Re-file with `--from-journal` after the refusal to attach the witnessed verdict._\n"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Witnessed verdict (source: `%s`):\n\n", e.Source)
	if e.Verdict != "" {
		fmt.Fprintf(&b, "- verdict: `%s`\n", e.Verdict)
	}
	if e.Reason != "" {
		fmt.Fprintf(&b, "- reason: `%s`\n", e.Reason)
	}
	if e.Tool != "" {
		fmt.Fprintf(&b, "- tool: `%s`\n", e.Tool)
	}
	if e.By != "" {
		fmt.Fprintf(&b, "- decided by: `%s`\n", e.By)
	}
	if e.TraceID != "" {
		fmt.Fprintf(&b, "- trace id: `%s`\n", e.TraceID)
	}
	if e.ArgsDigest != "" {
		fmt.Fprintf(&b, "- args digest: `%s`\n", e.ArgsDigest)
	}
	if e.CallSeq != 0 {
		fmt.Fprintf(&b, "- call id: `%d`\n", e.CallSeq)
	}
	if e.Seq != 0 {
		fmt.Fprintf(&b, "- journal seq: `%d`\n", e.Seq)
	}
	if e.CallSeq != 0 && e.TraceID != "" {
		fmt.Fprintf(&b, "- reselect this exact denial: `--from-journal --trace-id %s --call-seq %d`\n",
			e.TraceID, e.CallSeq)
	}
	if e.JournalPath != "" {
		fmt.Fprintf(&b, "- journal: `%s` (verify with `fak audit verify`)\n", e.JournalPath)
	}
	return b.String()
}

// MarkerKey extracts the stable key from a body's marker comment, or "" when absent.
func MarkerKey(body string) string {
	m := markerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// OccurrencesOf reads the occurrence count back out of an existing issue body, or 0 when
// absent/unparseable (so a malformed body restarts the count at 1 on the next file). A
// negative count reads as 0 too: the field is a tally, and a body claiming -3 is corrupt,
// not evidence of anything.
//
// Exported because the `- occurrences:` line is shared MARKER GRAMMAR, not this package's
// private field: every producer on the internal/dogfoodissues gh plumbing writes it, and
// `fak knownbad correlate` carried a byte-identical private reader over a byte-identical
// private regex. One writer of the line deserves one reader of it.
func OccurrencesOf(body string) int {
	m := occurrencesRE.FindStringSubmatch(body)
	if m == nil {
		return 0
	}
	n, err := strconv.Atoi(m[1])
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// BuildPlan decides create vs update for a complaint against the existing issues (matched
// by marker key) and computes the escalated occurrence count.
func BuildPlan(c Complaint, existing []dogfoodissues.Issue) PlanRow {
	key := c.Key()
	row := PlanRow{
		Action:      "create",
		Key:         key,
		Title:       c.Title(),
		Domain:      c.domain(),
		Kind:        c.Kind,
		Reason:      c.Reason,
		Tool:        c.Tool,
		Occurrences: 1,
	}
	for _, issue := range existing {
		if MarkerKey(issue.Body) != key {
			continue
		}
		row.Action = "update"
		n := issue.Number
		row.Number = &n
		row.State = issue.State
		row.Occurrences = OccurrencesOf(issue.Body) + 1
		break
	}
	row.Body = c.Body(row.Occurrences)
	return row
}

// Sync materializes a planned complaint through the existing gh plumbing in
// internal/dogfoodissues (one create or one edit). runner defaults to the real gh CLI when
// nil; labels are added on create.
func Sync(row PlanRow, repo string, labels []string, runner dogfoodissues.Runner) dogfoodissues.SyncRow {
	ddRow := dogfoodissues.PlanRow{
		Action: row.Action,
		Key:    row.Key,
		Number: row.Number,
		State:  row.State,
		Title:  row.Title,
		Body:   dogfoodSyncBody(row),
	}
	out := dogfoodissues.Sync([]dogfoodissues.PlanRow{ddRow}, repo, labels, runner)
	if len(out) == 1 {
		return out[0]
	}
	return dogfoodissues.SyncRow{Key: row.Key, Action: row.Action}
}

func dogfoodSyncBody(row PlanRow) string {
	if dogfoodissues.MarkerKey(row.Body) == row.Key {
		return row.Body
	}
	return fmt.Sprintf("<!-- fak-dogfood-action-key: %s -->\n%s", row.Key, row.Body)
}

// FetchExisting queries gh for the existing issues to classify create vs update. It is a
// thin pass-through to the dogfoodissues fetcher so the appeal channel and the dogfood
// backlog share one gh issue-list path.
func FetchExisting(repo string, limit int) ([]dogfoodissues.Issue, error) {
	return dogfoodissues.FetchExistingIssues(repo, limit)
}

// DenialSelector identifies the journal denial a complaint is actually appealing.
// Reason and Tool are useful coarse filters, but they are not an identity: a busy guard
// session routinely records several denials with the same reason/tool pair. Seq, CallSeq,
// TraceID, and ArgsDigest are exact selectors that bind the complaint to the refused call
// rather than whichever similar denial happened most recently (#3830).
//
// TraceID (the session) and CallSeq (the kernel's per-call submission id) are the pair an
// appeal can actually quote back, because the refusal itself names them; they select one
// denial deterministically without scanning ambiguous rows (#5213). ArgsDigest identifies
// the call CONTENT, so it collapses a genuinely re-issued identical call into one match,
// and Seq addresses a journal ROW, which is only unique inside a single file.
type DenialSelector struct {
	Reason     string
	Tool       string
	Seq        uint64
	CallSeq    uint64
	TraceID    string
	ArgsDigest string
}

// DenialSelection is the honest result of selecting complaint evidence. Evidence is set
// only when exactly one row matches. Multiple matches are Ambiguous and deliberately carry
// no Evidence: attaching the newest of several plausible rows would make an unrelated
// denial look like a witness for the complaint (#3830).
type DenialSelection struct {
	Evidence  *Evidence
	Matches   int
	Ambiguous bool
}

// SelectDenial selects exactly one witnessed denial. A coarse reason/tool lookup remains
// convenient when it yields one row, while a busy journal must be disambiguated by seq,
// trace id, or args digest. No match and ambiguity are both fail-honest: neither fabricates
// or guesses a witness.
func SelectDenial(paths []string, selector DenialSelector) DenialSelection {
	matches := matchingDenials(paths, selector)
	result := DenialSelection{Matches: len(matches)}
	switch len(matches) {
	case 1:
		result.Evidence = matches[0]
	case 0:
		// Honest no-witness result.
	default:
		result.Ambiguous = true
	}
	return result
}

// LatestDenial scans the guard decision journals for the most recent DENY/QUARANTINE row,
// optionally filtered to a reason token and/or tool, and returns it as witnessed Evidence.
// paths is the set of journal files (use guardrsi.JournalPaths to discover them). It returns
// nil when no matching denial is present — an honest "no witness" rather than a fabricated one.
//
// Deprecated for complaint filing: use SelectDenial, which refuses an ambiguous reason/tool
// match instead of silently binding the appeal to the newest plausible row. LatestDenial is
// retained for callers that explicitly want a recency query rather than evidence identity.
func LatestDenial(paths []string, reasonFilter, toolFilter string) *Evidence {
	matches := matchingDenials(paths, DenialSelector{Reason: reasonFilter, Tool: toolFilter})
	var best *Evidence
	for _, cand := range matches {
		if best == nil || moreRecent(cand, best) {
			best = cand
		}
	}
	return best
}

func matchingDenials(paths []string, selector DenialSelector) []*Evidence {
	reasonFilter := strings.ToUpper(strings.TrimSpace(selector.Reason))
	toolFilter := strings.TrimSpace(selector.Tool)
	traceFilter := strings.TrimSpace(selector.TraceID)
	digestFilter := strings.TrimSpace(selector.ArgsDigest)
	matches := []*Evidence{}
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var row struct {
				Seq        uint64 `json:"seq"`
				CallSeq    uint64 `json:"call_seq"`
				TSUnixNano int64  `json:"ts_unix_nano"`
				Kind       string `json:"kind"`
				Tool       string `json:"tool"`
				TraceID    string `json:"trace_id"`
				Verdict    string `json:"verdict"`
				Reason     string `json:"reason"`
				By         string `json:"by"`
				ArgsDigest string `json:"args_digest"`
			}
			if json.Unmarshal([]byte(line), &row) != nil {
				continue
			}
			verdict := strings.ToUpper(strings.TrimSpace(row.Verdict))
			if verdict == "" {
				switch strings.ToUpper(strings.TrimSpace(row.Kind)) {
				case "DENY", "RESULT_DENY":
					verdict = "DENY"
				case "QUARANTINE":
					verdict = "QUARANTINE"
				}
			}
			if verdict != "DENY" && verdict != "QUARANTINE" {
				continue
			}
			if reasonFilter != "" && strings.ToUpper(strings.TrimSpace(row.Reason)) != reasonFilter {
				continue
			}
			if toolFilter != "" && row.Tool != toolFilter {
				continue
			}
			if selector.Seq != 0 && row.Seq != selector.Seq {
				continue
			}
			// A zero CallSeq means "unselected", matching the Seq convention above. A row
			// that carries no call id has CallSeq 0 and therefore never satisfies a
			// non-zero selector — fail-closed, so an unidentifiable row can not be
			// mistaken for the call being appealed.
			if selector.CallSeq != 0 && row.CallSeq != selector.CallSeq {
				continue
			}
			if traceFilter != "" && strings.TrimSpace(row.TraceID) != traceFilter {
				continue
			}
			if digestFilter != "" && strings.TrimSpace(row.ArgsDigest) != digestFilter {
				continue
			}
			cand := &Evidence{
				Source:      "journal",
				JournalPath: path,
				Seq:         row.Seq,
				CallSeq:     row.CallSeq,
				TSUnixNano:  row.TSUnixNano,
				Verdict:     verdict,
				Tool:        row.Tool,
				Reason:      strings.TrimSpace(row.Reason),
				By:          row.By,
				TraceID:     row.TraceID,
				ArgsDigest:  row.ArgsDigest,
			}
			matches = append(matches, cand)
		}
	}
	return matches
}

// moreRecent orders two evidence rows: by wall-clock timestamp, then by sequence as a
// tie-break (journals from concurrent sessions can share a timestamp).
func moreRecent(a, b *Evidence) bool {
	if a.TSUnixNano != b.TSUnixNano {
		return a.TSUnixNano > b.TSUnixNano
	}
	return a.Seq > b.Seq
}

// DiscoverJournals returns the guard decision journals under root (or the single explicit
// path), reusing the same discovery guardrsi and the audit tooling use.
func DiscoverJournals(root, explicit string) []string {
	return guardrsi.JournalPaths(root, explicit)
}

// Render produces the human-readable summary of a plan/result.
func Render(r Result) string {
	lines := []string{
		fmt.Sprintf("guard-complaint: %s", r.Mode),
	}
	for _, row := range r.Planned {
		target := "new issue"
		if row.Number != nil {
			target = "#" + strconv.Itoa(*row.Number)
		}
		lines = append(lines, fmt.Sprintf("  [%s] %s: %s (occurrences=%d)",
			row.Action, target, row.Title, row.Occurrences))
	}
	for _, s := range r.Synced {
		status := "ok"
		if !s.OK {
			status = "FAILED: " + strings.TrimSpace(s.Stderr)
		}
		lines = append(lines, fmt.Sprintf("  synced [%s] %s -> %s", s.Action, s.Key, status))
	}
	if r.Mode == "dry-run" {
		lines = append(lines, "  dry-run: pass --live to create/update the issue with gh")
	}
	return strings.Join(lines, "\n")
}
