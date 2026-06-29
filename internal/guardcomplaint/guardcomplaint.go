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

// Evidence is the witnessed half of a complaint: a real adjudicated verdict pulled from
// the decision journal (or supplied manually). The agent's rationale is a self-report;
// this is the non-forgeable record that the refusal it is appealing actually happened.
type Evidence struct {
	Source     string `json:"source"`                // "journal" | "manual" | "none"
	JournalPath string `json:"journal_path,omitempty"`
	Seq        uint64 `json:"seq,omitempty"`
	TSUnixNano int64  `json:"ts_unix_nano,omitempty"`
	Verdict    string `json:"verdict,omitempty"`
	Tool       string `json:"tool,omitempty"`
	Reason     string `json:"reason,omitempty"`
	By         string `json:"by,omitempty"`
	TraceID    string `json:"trace_id,omitempty"`
	ArgsDigest string `json:"args_digest,omitempty"`
}

// Complaint is one agent-authored appeal against a guard decision.
type Complaint struct {
	Kind      string    `json:"kind"`
	Reason    string    `json:"reason,omitempty"` // the guard reason token being appealed (e.g. FILE_ADMISSION)
	Tool      string    `json:"tool,omitempty"`   // the refused tool (e.g. Bash, Write)
	Summary   string    `json:"summary"`          // one-line headline
	Rationale string    `json:"rationale"`        // why the agent judges the guard wrong
	Evidence  *Evidence `json:"evidence,omitempty"`
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

// NormalizeKind lowercases/validates a kind, defaulting an empty one. It returns the
// canonical kind and an error naming the closed set when the kind is unknown.
func NormalizeKind(kind string) (string, error) {
	k := strings.ToLower(strings.TrimSpace(kind))
	if k == "" {
		return DefaultKind, nil
	}
	if _, ok := Kinds[k]; !ok {
		return "", fmt.Errorf("unknown complaint kind %q (want one of: %s)", kind, strings.Join(sortedKinds(), ", "))
	}
	return k, nil
}

func sortedKinds() []string {
	out := make([]string, 0, len(Kinds))
	for k := range Kinds {
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
	return strings.Join([]string{
		"guard-complaint",
		slug(c.Kind),
		slug(reason),
		slug(tool),
		slug(c.Summary),
	}, "/")
}

// Title renders the issue title — clear and self-describing, with the appealed reason and
// tool inline so the tracker shows the class at a glance.
func (c Complaint) Title() string {
	var b strings.Builder
	b.WriteString("guard complaint [")
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
	if s := strings.TrimSpace(c.Summary); s != "" {
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
	if e.Seq != 0 {
		fmt.Fprintf(&b, "- journal seq: `%d`\n", e.Seq)
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

// occurrencesOf reads the occurrence count back out of an existing complaint body, or 0
// when absent/unparseable (so a malformed body restarts the count at 1 on the next file).
func occurrencesOf(body string) int {
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
		row.Occurrences = occurrencesOf(issue.Body) + 1
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
		Body:   row.Body,
	}
	out := dogfoodissues.Sync([]dogfoodissues.PlanRow{ddRow}, repo, labels, runner)
	if len(out) == 1 {
		return out[0]
	}
	return dogfoodissues.SyncRow{Key: row.Key, Action: row.Action}
}

// FetchExisting queries gh for the existing issues to classify create vs update. It is a
// thin pass-through to the dogfoodissues fetcher so the appeal channel and the dogfood
// backlog share one gh issue-list path.
func FetchExisting(repo string, limit int) ([]dogfoodissues.Issue, error) {
	return dogfoodissues.FetchExistingIssues(repo, limit)
}

// LatestDenial scans the guard decision journals for the most recent DENY/QUARANTINE row,
// optionally filtered to a reason token and/or tool, and returns it as witnessed Evidence.
// paths is the set of journal files (use guardrsi.JournalPaths to discover them). It returns
// nil when no matching denial is present — an honest "no witness" rather than a fabricated one.
func LatestDenial(paths []string, reasonFilter, toolFilter string) *Evidence {
	reasonFilter = strings.ToUpper(strings.TrimSpace(reasonFilter))
	toolFilter = strings.TrimSpace(toolFilter)
	var best *Evidence
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
			cand := &Evidence{
				Source:      "journal",
				JournalPath: path,
				Seq:         row.Seq,
				TSUnixNano:  row.TSUnixNano,
				Verdict:     verdict,
				Tool:        row.Tool,
				Reason:      strings.TrimSpace(row.Reason),
				By:          row.By,
				TraceID:     row.TraceID,
				ArgsDigest:  row.ArgsDigest,
			}
			if best == nil || moreRecent(cand, best) {
				best = cand
			}
		}
	}
	return best
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
