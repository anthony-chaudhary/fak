// Package mlpscore grades epic #3256's first-lovable-cut contract from
// committed, machine-checkable witness manifests.
package mlpscore

import (
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

const (
	// Schema is the stable JSON contract emitted by `fak mlp-score --json`.
	Schema = "fak-mlp-score/1"
	// WitnessSchema is the contract each committed criterion witness must use.
	WitnessSchema = "fak-mlp-witness/1"
	// WitnessDir is the canonical home for MLP witness manifests.
	WitnessDir = "docs/mlp/witnesses"

	GradeWitnessed = "witnessed"
	GradeNotYet    = "not-yet"
)

// Snapshot is the committed-tree view used by Grade. The CLI supplies a HEAD
// snapshot, so an untracked note or proof artifact cannot make the score move.
type Snapshot interface {
	ReadFile(rel string) ([]byte, error)
	Exists(rel string) bool
}

type criterionSpec struct {
	key            string
	workstream     string
	title          string
	ownerIssues    []int
	requiredClaims []string
}

var criteria = []criterionSpec{
	{
		key:         "both_runtimes_one_command",
		workstream:  "B1",
		title:       "one command starts both runtimes and a POST completes a governed session",
		ownerIssues: []int{3420, 3258},
		requiredClaims: []string{
			"both_runtimes_ready",
			"governed_post_completed",
		},
	},
	{
		key:         "sdk_drives_offline",
		workstream:  "B3",
		title:       "a Python or TypeScript SDK drives the complete path offline",
		ownerIssues: []int{3261},
		requiredClaims: []string{
			"sdk_completed_loop",
			"offline_no_external_service",
		},
	},
	{
		key:         "audit_journal_and_cost_cap",
		workstream:  "C1/C3",
		title:       "the audit journal and per-session cost cap are enforced",
		ownerIssues: []int{3275, 3273},
		requiredClaims: []string{
			"audit_journal_enforced",
			"session_cost_cap_enforced",
		},
	},
	{
		key:         "init_agent_emits_governed_agent",
		workstream:  "D2",
		title:       "fak init agent emits a running governed agent",
		ownerIssues: []int{3283},
		requiredClaims: []string{
			"scaffolded_agent_completed",
		},
	},
	{
		key:         "time_to_first_governed_agent",
		workstream:  "D5",
		title:       "time-to-first-governed-agent is under the 10 minute target",
		ownerIssues: []int{3286},
		requiredClaims: []string{
			"time_to_first_under_10m",
		},
	},
}

// WitnessManifest is the committed evidence contract for one criterion. Each
// required claim must link to a committed test or captured-run artifact.
type WitnessManifest struct {
	Schema    string         `json:"schema"`
	Criterion string         `json:"criterion"`
	Claims    []WitnessClaim `json:"claims"`
}

// WitnessClaim binds one acceptance claim to a reproducible proof artifact.
type WitnessClaim struct {
	Key     string `json:"key"`
	Kind    string `json:"kind"` // test | captured-run
	Path    string `json:"path"`
	Command string `json:"command"`
}

// Criterion is one stable row in the scorecard JSON and markdown rollup.
type Criterion struct {
	Key            string         `json:"key"`
	Workstream     string         `json:"workstream"`
	Title          string         `json:"title"`
	OwnerIssues    []int          `json:"owner_issues"`
	Grade          string         `json:"grade"`
	WitnessRef     string         `json:"witness_ref"`
	RequiredClaims []string       `json:"required_claims"`
	Evidence       []WitnessClaim `json:"evidence"`
	Missing        []string       `json:"missing,omitempty"`
}

// Score is the stable MLP scorecard envelope.
type Score struct {
	trendreport.Envelope
	Criteria   []Criterion `json:"criteria"`
	Witnessed  int         `json:"witnessed"`
	Total      int         `json:"total"`
	Debt       int         `json:"mlp_debt"`
	Lovable    bool        `json:"lovable"`
	MLPVerdict string      `json:"mlp_verdict"` // lovable | not-yet
}

type FoldOpts = trendreport.Opts

// Grade folds the declared criteria against one committed snapshot. A manifest
// passes only when its schema and criterion match, every required claim appears
// exactly once, and every claim links to a committed proof artifact. A nil
// snapshot fails closed with every criterion not-yet.
func Grade(snapshot Snapshot, opts FoldOpts) Score {
	if snapshot == nil {
		snapshot = emptySnapshot{}
	}
	s := Score{Envelope: trendreport.Stamp(Schema, opts)}
	for _, spec := range criteria {
		row := gradeOne(spec, snapshot)
		if row.Grade == GradeWitnessed {
			s.Witnessed++
		}
		s.Criteria = append(s.Criteria, row)
	}
	s.Total = len(s.Criteria)
	s.Debt = s.Total - s.Witnessed
	s.Lovable = s.Total > 0 && s.Debt == 0
	if s.Lovable {
		s.MLPVerdict = "lovable"
		s.OK, s.Verdict, s.Finding = true, trendreport.VerdictOK, "mlp_lovable"
		s.Reason = fmt.Sprintf("MLP first lovable cut: all %d criteria have committed witnesses", s.Total)
		s.NextAction = "keep every linked witness reproducible as the milestone evolves"
		return s
	}

	s.MLPVerdict = "not-yet"
	s.OK, s.Verdict, s.Finding = false, "ACTION", "mlp_not_yet"
	s.Reason = fmt.Sprintf("MLP first lovable cut: %d of %d criteria witnessed; %d remain", s.Witnessed, s.Total, s.Debt)
	s.NextAction = "land the next missing witness manifest and its linked test or captured run"
	return s
}

type emptySnapshot struct{}

func (emptySnapshot) ReadFile(string) ([]byte, error) {
	return nil, fmt.Errorf("committed snapshot unavailable")
}

func (emptySnapshot) Exists(string) bool { return false }

func gradeOne(spec criterionSpec, snapshot Snapshot) Criterion {
	witnessRef := WitnessDir + "/" + spec.key + ".json"
	row := Criterion{
		Key:            spec.key,
		Workstream:     spec.workstream,
		Title:          spec.title,
		OwnerIssues:    append([]int(nil), spec.ownerIssues...),
		Grade:          GradeNotYet,
		WitnessRef:     witnessRef,
		RequiredClaims: append([]string(nil), spec.requiredClaims...),
		Evidence:       []WitnessClaim{},
	}

	raw, err := snapshot.ReadFile(witnessRef)
	if err != nil {
		row.Missing = []string{"witness manifest is not committed"}
		return row
	}

	var manifest WitnessManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		row.Missing = []string{"witness manifest is invalid JSON: " + err.Error()}
		return row
	}
	row.Evidence = append(row.Evidence, manifest.Claims...)
	row.Missing = validateManifest(spec, witnessRef, manifest, snapshot)
	if len(row.Missing) == 0 {
		row.Grade = GradeWitnessed
		row.Missing = nil
	}
	return row
}

func validateManifest(spec criterionSpec, witnessRef string, manifest WitnessManifest, snapshot Snapshot) []string {
	var missing []string
	if manifest.Schema != WitnessSchema {
		missing = append(missing, fmt.Sprintf("schema %q, want %q", manifest.Schema, WitnessSchema))
	}
	if manifest.Criterion != spec.key {
		missing = append(missing, fmt.Sprintf("criterion %q, want %q", manifest.Criterion, spec.key))
	}

	byKey := make(map[string][]WitnessClaim, len(manifest.Claims))
	for _, claim := range manifest.Claims {
		byKey[claim.Key] = append(byKey[claim.Key], claim)
	}
	for _, key := range spec.requiredClaims {
		claims := byKey[key]
		switch len(claims) {
		case 0:
			missing = append(missing, "missing claim: "+key)
		case 1:
			missing = append(missing, validateClaim(claims[0], witnessRef, snapshot)...)
		default:
			missing = append(missing, "duplicate claim: "+key)
		}
	}
	return missing
}

func validateClaim(claim WitnessClaim, witnessRef string, snapshot Snapshot) []string {
	var missing []string
	if claim.Kind != "test" && claim.Kind != "captured-run" {
		missing = append(missing, fmt.Sprintf("claim %s has unsupported kind %q", claim.Key, claim.Kind))
	}
	clean, ok := cleanRepoPath(claim.Path)
	if !ok || clean == witnessRef {
		missing = append(missing, fmt.Sprintf("claim %s has invalid proof path %q", claim.Key, claim.Path))
	} else if !snapshot.Exists(clean) {
		missing = append(missing, fmt.Sprintf("claim %s proof is not committed: %s", claim.Key, clean))
	}
	if strings.TrimSpace(claim.Command) == "" {
		missing = append(missing, "claim "+claim.Key+" has no reproduction command")
	}
	return missing
}

func cleanRepoPath(p string) (string, bool) {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	if p == "" || strings.HasPrefix(p, "/") || strings.Contains(p, ":") {
		return "", false
	}
	clean := path.Clean(p)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", false
	}
	return clean, clean == p
}

// Render returns the compact terminal view.
func Render(s Score) string {
	verdict := "NOT YET"
	if s.Lovable {
		verdict = "LOVABLE"
	}
	lines := []string{
		fmt.Sprintf("MLP scorecard - first lovable cut (epic #3256, milestone #17)  @%s  %s", s.Commit, s.Date),
		"",
		fmt.Sprintf("  verdict: %s - %d/%d criteria witnessed", verdict, s.Witnessed, s.Total),
		"",
	}
	for _, c := range s.Criteria {
		mark := "."
		if c.Grade == GradeWitnessed {
			mark = "+"
		}
		lines = append(lines, fmt.Sprintf("  %s [%s] %s", mark, c.Workstream, c.Title))
		lines = append(lines, fmt.Sprintf("      %s  witness: %s", c.Grade, c.WitnessRef))
		for _, gap := range c.Missing {
			lines = append(lines, "      missing: "+gap)
		}
	}
	lines = append(lines, "", "  -> "+s.NextAction)
	return strings.Join(lines, "\n")
}

// RenderMarkdown returns the deterministic milestone-ready rollup.
func RenderMarkdown(s Score) string {
	verdict := "**NOT YET**"
	if s.Lovable {
		verdict = "**LOVABLE**"
	}
	var b strings.Builder
	b.WriteString("### MLP scorecard - first lovable cut (epic #3256, milestone #17)\n\n")
	fmt.Fprintf(&b, "Verdict: %s - %d/%d criteria witnessed (`@%s` %s)\n\n", verdict, s.Witnessed, s.Total, shortCommit(s.Commit), s.Date)
	b.WriteString("| Criterion | Workstream | Grade | Witness | Owners |\n")
	b.WriteString("|---|---|---|---|---|\n")
	for _, c := range s.Criteria {
		fmt.Fprintf(&b, "| %s | %s | %s | [%s](%s) | %s |\n",
			escapePipes(c.Title), c.Workstream, c.Grade, c.WitnessRef, c.WitnessRef, ownerLinks(c.OwnerIssues))
	}
	return b.String()
}

func ownerLinks(issues []int) string {
	links := make([]string, 0, len(issues))
	for _, issue := range issues {
		links = append(links, fmt.Sprintf("[#%d](https://github.com/anthony-chaudhary/fak/issues/%d)", issue, issue))
	}
	return strings.Join(links, ", ")
}

func shortCommit(commit string) string {
	if len(commit) > 12 {
		return commit[:12]
	}
	return commit
}

func escapePipes(s string) string { return strings.ReplaceAll(s, "|", "\\|") }

// Workstreams returns the declared criterion order for contract tests and
// milestone projections.
func Workstreams() []string {
	out := make([]string, 0, len(criteria))
	for _, spec := range criteria {
		out = append(out, spec.workstream)
	}
	return out
}

// WitnessRefs returns the canonical manifest paths in stable order.
func WitnessRefs() []string {
	out := make([]string, 0, len(criteria))
	for _, spec := range criteria {
		out = append(out, WitnessDir+"/"+spec.key+".json")
	}
	sort.Strings(out)
	return out
}
