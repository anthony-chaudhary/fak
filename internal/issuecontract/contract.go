// Package issuecontract reviews machine-created GitHub issue candidates before
// they enter the dispatch loop.
//
// The contract is deliberately about the working spine, not polish: a generated
// issue is dispatchable only when it names the useful path that becomes more
// true, what is in and out of scope, how done will be witnessed, and where the
// work routes. Missing detail becomes triage debt instead of an unscoped worker
// prompt.
//
// Closure binding: this package plus cmd/fak/issue_contract.go's `fak issue
// contract --file candidate.json` CLI shell satisfy #1459's ask in full --
// review of dedupe key, parent context, current/in-scope/out-of-scope, done
// condition, witness, routing, and closure binding, returning a
// dispatchable/triage_only/refused verdict with closed-vocabulary reasons,
// covered by contract_test.go and cmd/fak/issue_contract_test.go. The work
// shipped incrementally citing narrower issues (#1727, #1755, #1761, and
// others); history on origin/main cannot be rewritten, so this comment
// restates the #1459 closure binding explicitly for the grep-based referee.
package issuecontract

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	Schema               = "fak.issue-candidate.v1"
	ReviewSchema         = "fak.issue-candidate-review.v1"
	TemplateRepairSchema = "fak.issue-template-repair.v1"

	Dispatchable = "dispatchable"
	TriageOnly   = "triage_only"
	Refused      = "refused"

	ReasonScopeIncomplete    = "ISSUE_SCOPE_INCOMPLETE"
	ReasonUnrouted           = "ISSUE_UNROUTED"
	ReasonPrivateBoundary    = "ISSUE_PRIVATE_BOUNDARY"
	ReasonLiveUnarmored      = "ISSUE_LIVE_UNARMORED"
	ReasonNotDispatchLeaf    = "ISSUE_NOT_DISPATCH_LEAF"
	ReasonOversizedSteps     = "ISSUE_OVERSIZED_EXPECTED_STEPS"
	ReasonNoiseIncomplete    = "ISSUE_NOISE_CONTROL_INCOMPLETE"
	ReasonAgentIncomplete    = "ISSUE_AGENT_CONTEXT_INCOMPLETE"
	ReasonUnexpandedTemplate = "ISSUE_UNEXPANDED_TEMPLATE"
)

const MaxDispatchExpectedSteps = 8

var keyRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$`)
var markdownHeadingRE = regexp.MustCompile(`^#{1,6}\s+(.+?)\s*$`)
var codeSpanRE = regexp.MustCompile("`([^`]+)`")
var issueReferenceRE = regexp.MustCompile(`#([1-9][0-9]*)`)
var issueMarkerKeyRE = regexp.MustCompile(`<!--\s*fak-[A-Za-z0-9_-]+-key:\s*([^>\s]+)\s*-->`)
var unexpandedIssueTemplateRE = regexp.MustCompile(`(?m)(\$\(@\{|System\.Collections|System\.Management\.Automation|\$\(System\.|\bSource:\s*\$source\b)`)
var unexpandedIssueTemplateMarkerRE = regexp.MustCompile(`(?m)\$\(@\{[^)\r\n]*\}\.[A-Za-z0-9_]+\)|\$\((?:System\.Collections|System\.Management\.Automation)[^)\r\n]*\)|^\s*(?:[-*]\s*)?Source:\s*\$source[^\r\n]*`)

// IssueLabel is the subset of a GitHub label row used by IssueDraft.
type IssueLabel struct {
	Name string `json:"name"`
}

// IssueDraft is the subset of a GitHub issue row needed to audit manual or
// already-filed issues against the same spine-first contract used by generated
// candidates. It matches `gh issue list --json number,title,body,labels`.
type IssueDraft struct {
	Number int          `json:"number,omitempty"`
	Title  string       `json:"title"`
	Body   string       `json:"body"`
	Labels []IssueLabel `json:"labels,omitempty"`
	URL    string       `json:"url,omitempty"`
}

// TemplateRepairPlan is a dry-run repair row for an already-filed issue whose
// generated metadata header still contains unexpanded batch-filer tokens.
type TemplateRepairPlan struct {
	Schema                   string   `json:"schema"`
	IssueNumber              int      `json:"issue_number,omitempty"`
	Title                    string   `json:"title,omitempty"`
	URL                      string   `json:"url,omitempty"`
	Key                      string   `json:"key,omitempty"`
	DetectedMarker           string   `json:"detected_marker"`
	DetectedMarkers          []string `json:"detected_markers,omitempty"`
	ProposedNormalizedHeader string   `json:"proposed_normalized_header"`
	DryRunOnly               bool     `json:"dry_run_only"`
}

// DependencyRef is one parsed issue-body dependency marker.
type DependencyRef struct {
	Relation string `json:"relation"`
	Issue    int    `json:"issue"`
	Blocking bool   `json:"blocking"`
	Raw      string `json:"raw,omitempty"`
}

// Candidate is the pure input shape a producer can review before rendering or
// syncing a public GitHub issue.
type Candidate struct {
	Schema          string          `json:"schema,omitempty"`
	IssueNumber     int             `json:"issue_number,omitempty"`
	Key             string          `json:"key"`
	Title           string          `json:"title"`
	Generation      string          `json:"generation,omitempty"`
	ParentRef       string          `json:"parent_ref,omitempty"`
	CurrentState    string          `json:"current_state,omitempty"`
	WhyNow          string          `json:"why_now,omitempty"`
	WorkingSpine    string          `json:"working_spine,omitempty"`
	PriorityContext string          `json:"priority_context,omitempty"`
	WorkUnit        string          `json:"work_unit,omitempty"`
	ExpectedSteps   int             `json:"expected_steps,omitempty"`
	Assumptions     []string        `json:"assumptions,omitempty"`
	ConfusionRisks  []string        `json:"confusion_risks,omitempty"`
	Coordination    []string        `json:"coordination,omitempty"`
	Trigger         string          `json:"trigger,omitempty"`
	BatchPolicy     string          `json:"batch_policy,omitempty"`
	InScope         string          `json:"in_scope,omitempty"`
	OutOfScope      string          `json:"out_of_scope,omitempty"`
	DoneCondition   string          `json:"done_condition,omitempty"`
	Witness         string          `json:"witness,omitempty"`
	AcceptanceGate  string          `json:"acceptance_gate,omitempty"`
	Lane            string          `json:"lane,omitempty"`
	Paths           []string        `json:"paths,omitempty"`
	Dependencies    []DependencyRef `json:"dependencies,omitempty"`
	Labels          []string        `json:"labels,omitempty"`
	Priority        string          `json:"priority,omitempty"`
	BoundaryNotes   []string        `json:"boundary_notes,omitempty"`
	Private         bool            `json:"private,omitempty"`
	ClosureBinding  string          `json:"closure_binding,omitempty"`
}

// Options carries context the issue body alone cannot prove, such as whether a
// live scheduled producer has already armed marker dedupe and a cap.
type Options struct {
	Live          bool
	DedupeChecked bool
	DedupeCap     int
}

// Score explains the spine-first readiness score. The four axes are intentionally
// coarse and equal-weighted so a caller cannot bury a missing witness under a long
// issue body.
type Score struct {
	Spine   int `json:"spine"`
	Scope   int `json:"scope"`
	Witness int `json:"witness"`
	Route   int `json:"route"`
	Total   int `json:"total"`
}

// SpinePriority scores whether a scoped issue moves the current working spine
// before polish. It is intentionally separate from Score: a low-priority issue
// can still be perfectly dispatchable, and a high-priority issue can still need
// scoping before a worker should pick it up.
type SpinePriority struct {
	WorkingPath int `json:"working_path"`
	CurrentNeed int `json:"current_need"`
	Unblocks    int `json:"unblocks"`
	NotPolish   int `json:"not_polish"`
	Total       int `json:"total"`
}

// AgentContext scores whether an issue carries the extra context a worker needs
// at scale: work-unit shape, assumption/confusion control, coordination hints,
// and trigger/batch policy so high-volume issue generation does not turn into
// spam. It is advisory; dispatchability still comes from the stricter scope /
// route / witness contract.
type AgentContext struct {
	Shape        int `json:"shape"`
	Assumptions  int `json:"assumptions"`
	Coordination int `json:"coordination"`
	NoiseControl int `json:"noise_control"`
	Total        int `json:"total"`
}

// GenerationFit is an advisory grooming score. It checks whether generation
// labels match the issue body, proof, and time-horizon cues. It is intentionally
// not part of dispatchability: a label mismatch needs operator review, not a
// silent refusal of otherwise scoped work.
type GenerationFit struct {
	Stream        string   `json:"stream,omitempty"`
	LabelStream   string   `json:"label_stream,omitempty"`
	BodyStream    string   `json:"body_stream,omitempty"`
	Label         int      `json:"label"`
	Body          int      `json:"body"`
	Horizon       int      `json:"horizon"`
	Evidence      int      `json:"evidence"`
	Orthogonality int      `json:"orthogonality"`
	Total         int      `json:"total"`
	Flags         []string `json:"flags,omitempty"`
	NextAction    string   `json:"next_action,omitempty"`
}

// Review is the closed-vocabulary verdict over a Candidate.
type Review struct {
	Schema          string          `json:"schema"`
	OK              bool            `json:"ok"`
	Verdict         string          `json:"verdict"`
	Dispatchability string          `json:"dispatchability"`
	Reasons         []string        `json:"reasons,omitempty"`
	MissingFields   []string        `json:"missing_fields,omitempty"`
	MissingSections []string        `json:"missing_sections,omitempty"`
	IssueNumber     int             `json:"issue_number,omitempty"`
	Key             string          `json:"key,omitempty"`
	Lane            string          `json:"lane,omitempty"`
	Paths           []string        `json:"paths,omitempty"`
	Dependencies    []DependencyRef `json:"dependencies,omitempty"`
	WorkUnit        string          `json:"work_unit,omitempty"`
	ExpectedSteps   int             `json:"expected_steps,omitempty"`
	Assumptions     []string        `json:"assumptions,omitempty"`
	ConfusionRisks  []string        `json:"confusion_risks,omitempty"`
	Coordination    []string        `json:"coordination,omitempty"`
	Trigger         string          `json:"trigger,omitempty"`
	BatchPolicy     string          `json:"batch_policy,omitempty"`
	Score           Score           `json:"score"`
	SpinePriority   SpinePriority   `json:"spine_priority"`
	AgentContext    AgentContext    `json:"agent_context"`
	GenerationFit   GenerationFit   `json:"generation_fit"`
}

// ReviewCandidate grades c. OK means the candidate is safe to sync as a
// dispatchable public issue; non-OK reviews still preserve enough detail to render
// a triage-only row or refuse a live sync.
func ReviewCandidate(c Candidate, opt Options) Review {
	c = normalize(c)

	scopeMissing := missingScopeFields(c)
	missing := append([]string(nil), scopeMissing...)
	agentMissing := []string{}
	noiseMissing := []string{}
	if opt.Live {
		agentMissing = missingAgentContextFields(c)
		noiseMissing = missingNoiseControlFields(c)
		missing = append(missing, agentMissing...)
		missing = append(missing, noiseMissing...)
	}
	reasons := reasonSet{}
	if len(scopeMissing) > 0 {
		reasons.add(ReasonScopeIncomplete)
	}
	routeOK := c.Lane != "" || len(c.Paths) > 0
	if !routeOK {
		reasons.add(ReasonUnrouted)
	}
	private := c.Private || containsPrivateBoundary(c)
	if private {
		reasons.add(ReasonPrivateBoundary)
	}
	if opt.Live && (!opt.DedupeChecked || opt.DedupeCap <= 0) {
		reasons.add(ReasonLiveUnarmored)
	}
	if opt.Live && len(noiseMissing) > 0 {
		reasons.add(ReasonNoiseIncomplete)
	}
	if opt.Live && len(agentMissing) > 0 {
		reasons.add(ReasonAgentIncomplete)
	}
	if !isDispatchLeaf(c) {
		reasons.add(ReasonNotDispatchLeaf)
	}
	if c.ExpectedSteps > MaxDispatchExpectedSteps {
		reasons.add(ReasonOversizedSteps)
	}

	score := score(c, routeOK)
	spinePriority := spinePriority(c)
	agentContext := agentContext(c)
	generationFit := generationFit(c)
	out := Review{
		Schema:         ReviewSchema,
		IssueNumber:    c.IssueNumber,
		Key:            c.Key,
		Lane:           c.Lane,
		Paths:          append([]string(nil), c.Paths...),
		Dependencies:   append([]DependencyRef(nil), c.Dependencies...),
		WorkUnit:       c.WorkUnit,
		ExpectedSteps:  c.ExpectedSteps,
		Assumptions:    append([]string(nil), c.Assumptions...),
		ConfusionRisks: append([]string(nil), c.ConfusionRisks...),
		Coordination:   append([]string(nil), c.Coordination...),
		Trigger:        c.Trigger,
		BatchPolicy:    c.BatchPolicy,
		Reasons:        reasons.list(),
		MissingFields:  missing,
		Score:          score,
		SpinePriority:  spinePriority,
		AgentContext:   agentContext,
		GenerationFit:  generationFit,
	}
	out.OK = len(out.Reasons) == 0
	switch {
	case out.OK:
		out.Verdict = "ready"
		out.Dispatchability = Dispatchable
	case private || reasons.has(ReasonLiveUnarmored) || reasons.has(ReasonNoiseIncomplete) || reasons.has(ReasonAgentIncomplete):
		out.Verdict = "refused"
		out.Dispatchability = Refused
	default:
		out.Verdict = "needs_scope"
		out.Dispatchability = TriageOnly
	}
	return out
}

// ReviewIssueDraft reviews a GitHub issue row whose body uses the standard
// issue-contract sections. This is the bridge from "all open GitHub issues" to
// the same dispatchability language used by generated issue producers.
func ReviewIssueDraft(d IssueDraft, opt Options) Review {
	candidate := CandidateFromIssueDraft(d)
	review := ReviewCandidate(candidate, opt)
	if missingSections := missingRequiredIssueSections(d.Body, candidate); len(missingSections) > 0 {
		review.OK = false
		review.MissingSections = missingSections
		review.MissingFields = appendUnique(review.MissingFields, missingSections...)
		addReviewReason(&review, ReasonScopeIncomplete)
		if review.Dispatchability == Dispatchable {
			review.Verdict = "needs_scope"
			review.Dispatchability = TriageOnly
		}
	}
	if HasUnexpandedTemplate(d.Body) {
		review.OK = false
		review.Verdict = "refused"
		review.Dispatchability = Refused
		addReviewReason(&review, ReasonUnexpandedTemplate)
	}
	return review
}

// HasUnexpandedTemplate reports whether an already-rendered issue body still
// contains PowerShell template tokens from a failed batch filer. Such a body is
// not safe to dispatch: routing metadata may be corrupt even when later
// acceptance sections are intact.
func HasUnexpandedTemplate(body string) bool {
	return unexpandedIssueTemplateRE.MatchString(body)
}

// UnexpandedTemplateMarkers returns the literal corrupt tokens that make a
// generated issue unsafe to dispatch. The markers are ordered by first
// appearance so the first row is the cheapest human triage cue.
func UnexpandedTemplateMarkers(body string) []string {
	matches := unexpandedIssueTemplateMarkerRE.FindAllString(body, -1)
	seen := map[string]bool{}
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		match = strings.TrimSpace(match)
		if match == "" || seen[match] {
			continue
		}
		seen[match] = true
		out = append(out, match)
	}
	return out
}

// BuildTemplateRepairPlan returns a dry-run repair plan for a corrupt generated
// issue body. It does not mutate GitHub; callers decide whether to render the
// plan as JSON, text, or an operator-reviewed edit.
func BuildTemplateRepairPlan(d IssueDraft) (TemplateRepairPlan, bool) {
	markers := UnexpandedTemplateMarkers(d.Body)
	if len(markers) == 0 {
		return TemplateRepairPlan{}, false
	}
	candidate := CandidateFromIssueDraft(d)
	return TemplateRepairPlan{
		Schema:                   TemplateRepairSchema,
		IssueNumber:              d.Number,
		Title:                    strings.TrimSpace(d.Title),
		URL:                      strings.TrimSpace(d.URL),
		Key:                      candidate.Key,
		DetectedMarker:           markers[0],
		DetectedMarkers:          markers,
		ProposedNormalizedHeader: proposedTemplateRepairHeader(d, candidate),
		DryRunOnly:               true,
	}, true
}

func proposedTemplateRepairHeader(d IssueDraft, c Candidate) string {
	labelStream, _ := generationStreamFromLabels(c.Labels)
	stream := firstNonEmpty(labelStream, generationStreamFromCandidate(c), generationStreamFromText(d.Body))
	if stream != "" {
		return proposedGenerationHeader(d, c, stream)
	}
	if isManagedContextIssue(d, c) {
		return proposedManagedContextHeader(d, c)
	}
	return proposedGenericIssueHeader(d, c)
}

func proposedGenerationHeader(d IssueDraft, c Candidate, stream string) string {
	parent := firstNonEmpty(cleanIssueHeaderValue(issueHeaderField(d.Body, "Parent")), c.ParentRef)
	lines := []string{
		"## Generation stream",
		"- Generation: " + stream,
		"- Stream rule: " + generationStreamRule(stream),
		"- Milestone: " + generationMilestone(stream),
	}
	if parent != "" {
		lines = append(lines, "- Parent: "+parent)
	}
	return strings.Join(lines, "\n")
}

func proposedManagedContextHeader(d IssueDraft, c Candidate) string {
	key := issueDraftMarkerKey(d.Body)
	track := firstNonEmpty(managedContextTrackFromKey(key), cleanIssueHeaderValue(issueHeaderField(d.Body, "Track")), "needs-track")
	parent := firstNonEmpty(cleanIssueHeaderValue(issueHeaderField(d.Body, "Parent")), c.ParentRef, "https://github.com/anthony-chaudhary/fak/issues/1570")
	lines := []string{}
	if key != "" {
		lines = append(lines, "<!-- fak-managed-context-key: "+key+" -->")
	}
	title := strings.TrimSpace(d.Title)
	if title == "" {
		title = "managed-context: repair generated issue metadata"
	}
	lines = append(lines,
		"# "+title,
		"",
		"- Program: managed-context",
		"- Track: "+track,
		"- Parent: "+parent,
	)
	return strings.Join(lines, "\n")
}

func proposedGenericIssueHeader(d IssueDraft, c Candidate) string {
	title := strings.TrimSpace(d.Title)
	if title == "" {
		title = "generated issue metadata repair"
	}
	lines := []string{
		"## Issue metadata",
		"- Title: " + title,
	}
	if c.Key != "" {
		lines = append(lines, "- Key: "+c.Key)
	}
	if parent := firstNonEmpty(cleanIssueHeaderValue(issueHeaderField(d.Body, "Parent")), c.ParentRef); parent != "" {
		lines = append(lines, "- Parent: "+parent)
	}
	return strings.Join(lines, "\n")
}

func issueHeaderField(body, name string) string {
	want := strings.ToLower(strings.TrimSpace(name))
	for _, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line := trimListPrefix(raw)
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(strings.Trim(key, "`*_ ")))
		if key == want {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func cleanIssueHeaderValue(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	if strings.Contains(value, "$(") || strings.Contains(value, "$(@{") ||
		strings.Contains(lower, "system.") || strings.Contains(lower, "$source") {
		return ""
	}
	return value
}

func generationStreamRule(stream string) string {
	switch stream {
	case "gen/now":
		return "Now gen: current product/runtime work with direct default-path evidence."
	case "gen/next":
		return "Next gen: near-term foundation that should become agent-runnable after gates, handoffs, and operator visibility exist."
	case "gen/second-next":
		return "Second-next gen: architecture and option work that can promote after next-gen gates prove out."
	case "gen/future":
		return "Future gen: long-horizon research, standards, narratives, and option-value work."
	default:
		return "Generation stream should be re-rendered from labels and issue scope."
	}
}

func generationMilestone(stream string) string {
	switch stream {
	case "gen/now":
		return "Generation G0 - Now"
	case "gen/next":
		return "Generation G1 - Next Gen"
	case "gen/second-next":
		return "Generation G2 - Second Next"
	case "gen/future":
		return "Generation G3 - Future"
	default:
		return "Generation milestone needs operator review"
	}
}

func isManagedContextIssue(d IssueDraft, c Candidate) bool {
	text := strings.ToLower(strings.Join([]string{
		d.Title,
		d.Body,
		c.Key,
		strings.Join(c.Labels, "\n"),
	}, "\n"))
	return strings.Contains(text, "managed-context")
}

func managedContextTrackFromKey(key string) string {
	key = strings.TrimSpace(key)
	const prefix = "managed-context/"
	if !strings.HasPrefix(key, prefix) {
		return ""
	}
	rest := strings.TrimPrefix(key, prefix)
	track, _, _ := strings.Cut(rest, "/")
	return strings.TrimSpace(track)
}

func hasReason(reasons []string, want string) bool {
	for _, reason := range reasons {
		if reason == want {
			return true
		}
	}
	return false
}

func addReviewReason(review *Review, reason string) {
	if hasReason(review.Reasons, reason) {
		return
	}
	review.Reasons = append(review.Reasons, reason)
	sort.Strings(review.Reasons)
}

// CandidateFromIssueDraft parses the standard issue-contract sections from an
// already-filed GitHub issue row. It is shared by audit surfaces and worker
// prompt renderers so every cycle interprets issue bodies the same way.
func CandidateFromIssueDraft(d IssueDraft) Candidate {
	sections := markdownSections(d.Body)
	section := func(names ...string) string {
		for _, name := range names {
			if s := strings.TrimSpace(sections[normalizeHeading(name)]); s != "" {
				return s
			}
		}
		return ""
	}
	doneWitness := section("Done condition / witness")
	return Candidate{
		Schema:          Schema,
		IssueNumber:     d.Number,
		Key:             issueDraftKey(d),
		Title:           d.Title,
		Generation:      issueDraftGeneration(d, section("Generation stream", "Generation")),
		ParentRef:       section("Parent context", "Parent ref", "Parent issue", "Source"),
		CurrentState:    section("Current state"),
		WhyNow:          section("Why this is next", "Why now"),
		WorkingSpine:    section("Working spine"),
		PriorityContext: section("Priority context", "Spine priority", "Importance"),
		WorkUnit:        section("Work unit", "Work-unit shape", "Issue shape"),
		ExpectedSteps:   parseExpectedSteps(section("Expected steps", "Step budget")),
		Assumptions:     issueDraftAgentNotes(section("Assumptions")),
		ConfusionRisks:  issueDraftAgentNotes(section("Confusion risks", "Known confusion", "Unknowns")),
		Coordination:    issueDraftAgentNotes(section("Coordination", "Coordination notes", "Handoff notes")),
		Trigger:         agentSectionValue(section("Trigger", "Creation trigger")),
		BatchPolicy:     agentSectionValue(section("Batch policy", "Noise control", "Spam control")),
		InScope:         section("In scope"),
		OutOfScope:      section("Out of scope"),
		DoneCondition:   firstNonEmpty(section("Done condition"), prefixedSectionValue(doneWitness, "Done condition")),
		Witness:         firstNonEmpty(section("Witness"), prefixedSectionValue(doneWitness, "Witness")),
		AcceptanceGate:  section("Acceptance gate"),
		Lane:            section("Lane"),
		Paths:           issueDraftPaths(section("Likely files", "Path hints", "Paths", "Files")),
		Dependencies:    ParseIssueDependencies(section("Dependencies", "Dependency markers")),
		Labels:          issueDraftLabels(d.Labels),
		BoundaryNotes:   issueDraftNotes(section("Boundary notes", "Risk / boundary notes")),
		ClosureBinding:  section("Closure binding"),
	}
}

func prefixedSectionValue(section, prefix string) string {
	prefix = strings.ToLower(strings.TrimSpace(prefix)) + ":"
	for _, line := range strings.Split(section, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToLower(line), prefix) {
			return strings.TrimSpace(strings.Trim(line[len(prefix):], "` "))
		}
	}
	return ""
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func issueDraftKey(d IssueDraft) string {
	if key := issueDraftMarkerKey(d.Body); key != "" {
		return key
	}
	if d.Number > 0 {
		return "issue/" + strconv.Itoa(d.Number)
	}
	slug := slugKeyPart(d.Title)
	if slug == "" {
		slug = "unknown"
	}
	return "manual/" + slug
}

func issueDraftMarkerKey(body string) string {
	m := issueMarkerKeyRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func markdownSections(body string) map[string]string {
	out := map[string]string{}
	var current string
	var buf []string
	flush := func() {
		if current == "" {
			return
		}
		out[current] = strings.TrimSpace(strings.Join(buf, "\n"))
	}
	for _, raw := range strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n") {
		line := strings.TrimSpace(raw)
		if m := markdownHeadingRE.FindStringSubmatch(line); m != nil {
			flush()
			current = normalizeHeading(m[1])
			buf = nil
			continue
		}
		if current != "" {
			buf = append(buf, raw)
		}
	}
	flush()
	return out
}

func normalizeHeading(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "`*_:# ")
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

func issueDraftLabels(labels []IssueLabel) []string {
	out := make([]string, 0, len(labels))
	for _, label := range labels {
		out = append(out, label.Name)
	}
	return compact(out)
}

func issueDraftNotes(section string) []string {
	var out []string
	for _, line := range strings.Split(section, "\n") {
		line = trimListPrefix(line)
		if line != "" && !strings.EqualFold(line, "none") {
			out = append(out, line)
		}
	}
	return compact(out)
}

func issueDraftAgentNotes(section string) []string {
	notes := issueDraftNotes(section)
	out := notes[:0]
	for _, note := range notes {
		if agentSectionValue(note) != "" {
			out = append(out, note)
		}
	}
	return compact(out)
}

func agentSectionValue(s string) string {
	s = strings.TrimSpace(s)
	switch strings.ToLower(s) {
	case "", "not specified.", "not specified", "none named.", "none named", "no special coordination beyond the lane lease.":
		return ""
	default:
		return s
	}
}

func issueDraftPaths(section string) []string {
	var out []string
	for _, m := range codeSpanRE.FindAllStringSubmatch(section, -1) {
		if path := cleanPathHint(m[1]); path != "" {
			out = append(out, path)
		}
	}
	for _, line := range strings.Split(section, "\n") {
		line = trimListPrefix(line)
		if path := cleanPathHint(line); path != "" {
			out = append(out, path)
		}
	}
	return compact(out)
}

// ParseIssueDependencies parses issue-body dependency markers from a
// Dependencies section. Blocking relations hold one side of the edge until the
// named issue is witnessed; related-only references stay advisory.
func ParseIssueDependencies(section string) []DependencyRef {
	var out []DependencyRef
	seen := map[string]bool{}
	for _, line := range strings.Split(section, "\n") {
		raw := trimListPrefix(line)
		if raw == "" {
			continue
		}
		key, rest, ok := strings.Cut(raw, ":")
		if !ok {
			continue
		}
		relation, blocking, ok := dependencyRelation(key)
		if !ok {
			continue
		}
		for _, m := range issueReferenceRE.FindAllStringSubmatch(rest, -1) {
			issue, err := strconv.Atoi(m[1])
			if err != nil || issue <= 0 {
				continue
			}
			seenKey := relation + "#" + strconv.Itoa(issue)
			if seen[seenKey] {
				continue
			}
			seen[seenKey] = true
			out = append(out, DependencyRef{
				Relation: relation,
				Issue:    issue,
				Blocking: blocking,
				Raw:      raw,
			})
		}
	}
	return out
}

func dependencyRelation(raw string) (relation string, blocking bool, ok bool) {
	key := strings.ToLower(strings.TrimSpace(strings.Trim(raw, "`*_ ")))
	key = strings.ReplaceAll(key, "_", "-")
	key = strings.Join(strings.Fields(key), "-")
	switch key {
	case "after", "depends", "depends-on", "requires", "prerequisite", "blocked-by":
		return "after", true, true
	case "blocks":
		return "blocks", true, true
	case "related", "related-only", "see-also":
		return "related", false, true
	default:
		return "", false, false
	}
}

func parseExpectedSteps(section string) int {
	for _, tok := range strings.Fields(strings.TrimSpace(section)) {
		tok = strings.Trim(tok, "`.,;:()[]")
		if n, err := strconv.Atoi(tok); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

func trimListPrefix(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "- ") || strings.HasPrefix(s, "* ") {
		return strings.TrimSpace(s[2:])
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i > 0 && i < len(s) && (s[i] == '.' || s[i] == ')') {
		return strings.TrimSpace(s[i+1:])
	}
	return s
}

func cleanPathHint(s string) string {
	s = strings.TrimSpace(strings.Trim(s, "`"))
	if s == "" || strings.ContainsAny(s, " \t\r\n") {
		return ""
	}
	s = strings.ReplaceAll(s, "\\", "/")
	if !strings.Contains(s, "/") {
		return ""
	}
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case strings.ContainsRune("._-/*", r):
		default:
			return ""
		}
	}
	return s
}

func slugKeyPart(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		default:
			if !lastDash {
				b.WriteByte('-')
				lastDash = true
			}
		}
		if b.Len() >= 80 {
			break
		}
	}
	return strings.Trim(b.String(), "-")
}

func normalize(c Candidate) Candidate {
	c.Schema = strings.TrimSpace(c.Schema)
	c.Key = strings.TrimSpace(c.Key)
	c.Title = strings.TrimSpace(c.Title)
	c.Generation = strings.TrimSpace(c.Generation)
	c.ParentRef = strings.TrimSpace(c.ParentRef)
	c.CurrentState = strings.TrimSpace(c.CurrentState)
	c.WhyNow = strings.TrimSpace(c.WhyNow)
	c.WorkingSpine = strings.TrimSpace(c.WorkingSpine)
	c.PriorityContext = strings.TrimSpace(c.PriorityContext)
	c.WorkUnit = strings.TrimSpace(c.WorkUnit)
	c.Assumptions = compact(c.Assumptions)
	c.ConfusionRisks = compact(c.ConfusionRisks)
	c.Coordination = compact(c.Coordination)
	c.Trigger = strings.TrimSpace(c.Trigger)
	c.BatchPolicy = strings.TrimSpace(c.BatchPolicy)
	c.InScope = strings.TrimSpace(c.InScope)
	c.OutOfScope = strings.TrimSpace(c.OutOfScope)
	c.DoneCondition = strings.TrimSpace(c.DoneCondition)
	c.Witness = strings.TrimSpace(c.Witness)
	c.AcceptanceGate = strings.TrimSpace(c.AcceptanceGate)
	c.Lane = strings.TrimSpace(c.Lane)
	c.Priority = strings.TrimSpace(c.Priority)
	c.ClosureBinding = strings.TrimSpace(c.ClosureBinding)
	c.Paths = compact(c.Paths)
	c.Dependencies = normalizeDependencies(c.Dependencies)
	c.Labels = compact(c.Labels)
	c.BoundaryNotes = compact(c.BoundaryNotes)
	return c
}

func missingRequiredIssueSections(body string, c Candidate) []string {
	sections := markdownSections(body)
	hasSection := func(names ...string) bool {
		for _, name := range names {
			if strings.TrimSpace(sections[normalizeHeading(name)]) != "" {
				return true
			}
		}
		return false
	}
	hasScope := hasSection("Scope") || (hasSection("In scope") && hasSection("Out of scope"))
	checks := []struct {
		field string
		ok    bool
	}{
		{"current_state", hasSection("Current state") && c.CurrentState != ""},
		{"scope", hasScope},
		{"done_condition", c.DoneCondition != "" && hasSection("Done condition", "Done condition / witness")},
		{"witness", c.Witness != "" && hasSection("Witness", "Done condition / witness")},
		{"likely_files", len(c.Paths) > 0 && hasSection("Likely files", "Path hints", "Paths", "Files")},
	}
	var missing []string
	for _, check := range checks {
		if !check.ok {
			missing = append(missing, check.field)
		}
	}
	return missing
}

func normalizeDependencies(in []DependencyRef) []DependencyRef {
	seen := map[string]bool{}
	out := make([]DependencyRef, 0, len(in))
	for _, dep := range in {
		relation, blocking, ok := dependencyRelation(dep.Relation)
		if !ok || dep.Issue <= 0 {
			continue
		}
		seenKey := relation + "#" + strconv.Itoa(dep.Issue)
		if seen[seenKey] {
			continue
		}
		seen[seenKey] = true
		out = append(out, DependencyRef{
			Relation: relation,
			Issue:    dep.Issue,
			Blocking: blocking,
			Raw:      strings.TrimSpace(dep.Raw),
		})
	}
	return out
}

func appendUnique(out []string, items ...string) []string {
	seen := map[string]bool{}
	for _, item := range out {
		seen[item] = true
	}
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" || seen[item] {
			continue
		}
		out = append(out, item)
		seen[item] = true
	}
	return out
}

func missingScopeFields(c Candidate) []string {
	var missing []string
	add := func(field, value string) {
		if value == "" {
			missing = append(missing, field)
		}
	}
	if c.Key == "" || !keyRE.MatchString(c.Key) {
		missing = append(missing, "key")
	}
	add("title", c.Title)
	add("parent_ref", c.ParentRef)
	add("current_state", c.CurrentState)
	add("why_now", c.WhyNow)
	add("working_spine", c.WorkingSpine)
	add("in_scope", c.InScope)
	add("out_of_scope", c.OutOfScope)
	add("done_condition", c.DoneCondition)
	add("witness", c.Witness)
	add("acceptance_gate", c.AcceptanceGate)
	add("closure_binding", c.ClosureBinding)
	return missing
}

func missingAgentContextFields(c Candidate) []string {
	var missing []string
	if c.WorkUnit == "" {
		missing = append(missing, "work_unit")
	}
	if c.ExpectedSteps <= 0 {
		missing = append(missing, "expected_steps")
	}
	if len(c.Assumptions) == 0 {
		missing = append(missing, "assumptions")
	}
	if len(c.ConfusionRisks) == 0 {
		missing = append(missing, "confusion_risks")
	}
	if len(c.Coordination) == 0 {
		missing = append(missing, "coordination")
	}
	return missing
}

func missingNoiseControlFields(c Candidate) []string {
	var missing []string
	if agentSectionValue(c.Trigger) == "" {
		missing = append(missing, "trigger")
	}
	if policy := agentSectionValue(c.BatchPolicy); policy == "" || !batchPolicyControlsSpam(policy) {
		missing = append(missing, "batch_policy")
	}
	return missing
}

func batchPolicyControlsSpam(policy string) bool {
	text := strings.ToLower(policy)
	return hasAny(text,
		"one issue per", "per ", "key", "class", "bucket", "group",
		"batch", "wave", "cap", "capped", "at most", "limit", "threshold",
		"marker", "dedupe", "dedup", "duplicate", "duplicates", "existing",
		"update", "updates", "rerun", "reruns", "re-run",
	)
}

func score(c Candidate, routeOK bool) Score {
	s := Score{}
	s.Spine = points(25,
		c.ParentRef != "",
		c.CurrentState != "",
		c.WhyNow != "",
		c.WorkingSpine != "",
	)
	s.Scope = points(25,
		c.InScope != "",
		c.OutOfScope != "",
		c.DoneCondition != "",
	)
	s.Witness = points(25,
		c.Witness != "",
		c.AcceptanceGate != "",
		c.ClosureBinding != "",
	)
	if routeOK {
		s.Route = 25
	}
	s.Total = s.Spine + s.Scope + s.Witness + s.Route
	return s
}

func spinePriority(c Candidate) SpinePriority {
	text := strings.ToLower(strings.Join([]string{
		c.PriorityContext,
		c.ParentRef,
		c.CurrentState,
		c.WhyNow,
		c.WorkingSpine,
		c.InScope,
		c.OutOfScope,
		c.DoneCondition,
		c.Witness,
		c.AcceptanceGate,
	}, "\n"))
	p := SpinePriority{}
	if hasAny(text,
		"working path", "end-to-end", "working spine", "dispatch", "issue-dispatch",
		"guard", "stop hook", "dos", "witness", "close arm", "serve", "release",
		"user path", "user-facing", "dogfood",
	) {
		p.WorkingPath = 25
	}
	if hasAny(text,
		"current blocker", "currently fails", "cannot", "blocked", "red",
		"regression", "missing", "weak point", "now", "next weak point",
		"current state", "failing gate",
	) {
		p.CurrentNeed = 25
	}
	if hasAny(text,
		"unblock", "unblocks", "blocks", "without this", "enables", "required for",
		"before", "load-bearing", "gate", "dependency", "prerequisite",
	) {
		p.Unblocks = 25
	}
	if hasAny(text,
		"not polish", "before polish", "not optimize", "do not optimize",
		"no optimization", "smallest", "leaf", "out of scope", "not a refactor",
		"not refactor", "not gold", "gold plating",
	) {
		p.NotPolish = 25
	}
	p.Total = p.WorkingPath + p.CurrentNeed + p.Unblocks + p.NotPolish
	return p
}

func agentContext(c Candidate) AgentContext {
	a := AgentContext{}
	shapeKnown := c.WorkUnit != "" && (isDispatchWorkUnit(c.WorkUnit) || isNonDispatchWorkUnit(c.WorkUnit))
	a.Shape = points(25, shapeKnown, c.ExpectedSteps > 0)
	a.Assumptions = points(25, len(c.Assumptions) > 0, len(c.ConfusionRisks) > 0)
	a.Coordination = points(25, len(c.Coordination) > 0)
	a.NoiseControl = points(25, c.Trigger != "", c.BatchPolicy != "")
	a.Total = a.Shape + a.Assumptions + a.Coordination + a.NoiseControl
	return a
}

func generationFit(c Candidate) GenerationFit {
	labelStream, labelFlags := generationStreamFromLabels(c.Labels)
	bodyStream := generationStreamFromCandidate(c)
	stream := firstNonEmpty(labelStream, bodyStream)
	flags := append([]string(nil), labelFlags...)
	if labelStream != "" && bodyStream != "" && labelStream != bodyStream {
		flags = append(flags, "generation_body_mismatch")
	}
	if labelStream == "" && bodyStream != "" {
		flags = append(flags, "generation_label_missing")
	}

	text := generationCandidateText(c)
	lower := strings.ToLower(text)
	// Conflation flags fire only when the anti-pattern is ADVOCATED. Correct
	// grooming guidance names the same phrase to forbid it ("do not create a
	// branch per generation"), so a nearby prohibition cue must suppress the
	// flag — otherwise the cleanest issue trips it.
	if advocates(lower, "branch per generation", "feature branch per generation", "generation branch") {
		flags = append(flags, "generation_branch_conflation")
	}
	if advocates(lower, "future is lower priority", "future as lower priority", "future lower priority") {
		flags = append(flags, "generation_priority_conflation")
	}
	if advocates(lower, "gen/next flag", "gen/future flag", "generation label enables", "generation label decides exposure") {
		flags = append(flags, "generation_runtime_gate_conflation")
	}

	g := GenerationFit{
		Stream:      stream,
		LabelStream: labelStream,
		BodyStream:  bodyStream,
		Flags:       flags,
	}
	if labelStream != "" && !containsString(g.Flags, "generation_label_multiple") && !containsString(g.Flags, "generation_parent_label_missing") {
		g.Label = 20
	}
	if bodyStream != "" && (labelStream == "" || bodyStream == labelStream) {
		g.Body = 20
	}
	if stream != "" && generationHorizonMatches(stream, lower) {
		g.Horizon = 20
	} else if stream != "" {
		g.Flags = append(g.Flags, "generation_horizon_cue_missing")
	}
	if generationEvidenceNamed(c, lower) {
		g.Evidence = 20
	} else if stream != "" {
		g.Flags = append(g.Flags, "generation_promotion_evidence_missing")
	}
	if generationOrthogonalityNamed(lower) {
		g.Orthogonality = 20
	} else if stream != "" {
		g.Flags = append(g.Flags, "generation_orthogonality_missing")
	}
	g.Total = g.Label + g.Body + g.Horizon + g.Evidence + g.Orthogonality
	if len(g.Flags) > 0 {
		g.Flags = compact(g.Flags)
		g.NextAction = "review generation label, body horizon, promotion evidence, demotion evidence, and priority/trunk/runtime-gate separation"
	}
	return g
}

func issueDraftGeneration(d IssueDraft, generationSection string) string {
	if s := generationStreamFromText(generationSection); s != "" {
		return s
	}
	if s := generationStreamFromText(d.Title); s != "" {
		return s
	}
	return ""
}

func generationStreamFromCandidate(c Candidate) string {
	if s := generationStreamFromText(c.Generation); s != "" {
		return s
	}
	return generationStreamFromText(c.Title)
}

func generationStreamFromLabels(labels []string) (string, []string) {
	var streams []string
	hasGenerationLabel := false
	for _, label := range labels {
		label = strings.ToLower(strings.TrimSpace(label))
		if label == "generation" {
			hasGenerationLabel = true
		}
		if isGenerationStream(label) {
			streams = append(streams, label)
		}
	}
	streams = compact(streams)
	var flags []string
	if len(streams) > 1 {
		flags = append(flags, "generation_label_multiple")
	}
	if len(streams) == 1 && !hasGenerationLabel {
		flags = append(flags, "generation_parent_label_missing")
	}
	if len(streams) == 0 && hasGenerationLabel {
		flags = append(flags, "generation_stream_label_missing")
	}
	if len(streams) != 1 {
		return "", flags
	}
	return streams[0], flags
}

func generationStreamFromText(text string) string {
	text = strings.ToLower(text)
	checks := []struct {
		stream  string
		needles []string
	}{
		{"gen/second-next", []string{"gen/second-next", "gen=second-next", "generation(second-next)", "generation: second-next", "generation: gen/second-next"}},
		{"gen/future", []string{"gen/future", "gen=future", "generation(future)", "generation: future", "generation: gen/future"}},
		{"gen/next", []string{"gen/next", "gen=next", "generation(next)", "generation: next", "generation: gen/next"}},
		{"gen/now", []string{"gen/now", "gen=now", "generation(now)", "generation: now", "generation: gen/now"}},
	}
	for _, check := range checks {
		if hasAny(text, check.needles...) {
			return check.stream
		}
	}
	return ""
}

func isGenerationStream(label string) bool {
	switch label {
	case "gen/now", "gen/next", "gen/second-next", "gen/future":
		return true
	default:
		return false
	}
}

func generationCandidateText(c Candidate) string {
	parts := []string{
		c.Generation, c.Title, c.ParentRef, c.CurrentState, c.WhyNow, c.WorkingSpine,
		c.PriorityContext, c.WorkUnit, c.Trigger, c.BatchPolicy, c.InScope, c.OutOfScope,
		c.DoneCondition, c.Witness, c.AcceptanceGate, c.ClosureBinding,
	}
	parts = append(parts, c.Assumptions...)
	parts = append(parts, c.ConfusionRisks...)
	parts = append(parts, c.Coordination...)
	parts = append(parts, c.BoundaryNotes...)
	parts = append(parts, c.Labels...)
	return strings.Join(parts, "\n")
}

func generationHorizonMatches(stream, text string) bool {
	switch stream {
	case "gen/now":
		return hasAny(text, "current product", "current path", "immediate", "now", "today", "default path", "direct witness")
	case "gen/next":
		return hasAny(text, "next gen", "next-generation", "near-term", "foundation", "gate", "handoff", "dogfood", "operator visibility", "agent-runnable", "runnable soon")
	case "gen/second-next":
		return hasAny(text, "second-next", "architecture", "compatibility", "simulation", "dependency", "option", "adapter")
	case "gen/future":
		return hasAny(text, "future", "research", "long-horizon", "market", "standards", "narrative", "option value")
	default:
		return false
	}
}

func generationEvidenceNamed(c Candidate, text string) bool {
	hasPromotion := hasAny(text, "promotion", "promote", "readiness", "dogfood", "default-on", "move toward now")
	hasDemotion := hasAny(text, "demotion", "demote", "retirement", "retire", "park", "parking")
	hasInvalidating := hasAny(text, "invalidating assumption", "assumption could fail", "if this assumption fails", "recheck")
	hasWitness := strings.TrimSpace(c.Witness) != "" || hasAny(text, "witness", "captured command", "focused test", "readout")
	return hasPromotion && hasDemotion && hasInvalidating && hasWitness
}

func generationOrthogonalityNamed(text string) bool {
	hasPriority := strings.Contains(text, "priority")
	hasTrunk := hasAny(text, "shared trunk", "trunk", "main")
	hasRuntimeGate := hasAny(text, "runtime feature gate", "feature gate", "runtime gate", "exposure gate", "default-off", "default on")
	return hasPriority && hasTrunk && hasRuntimeGate
}

func isDispatchLeaf(c Candidate) bool {
	if isNonDispatchWorkUnit(c.WorkUnit) {
		return false
	}
	for _, label := range c.Labels {
		label = strings.ToLower(strings.TrimSpace(label))
		switch label {
		case "epic", "research", "idea-scout", "needs-triage", "needs-scope", "triage-only", "triage_only":
			return false
		}
	}
	return true
}

func isDispatchWorkUnit(unit string) bool {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "leaf", "step", "patch", "task", "work-unit", "work_unit", "worker-ready":
		return true
	default:
		return false
	}
}

func isNonDispatchWorkUnit(unit string) bool {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "epic", "program", "research", "idea", "triage", "triage-only", "triage_only", "decompose", "umbrella":
		return true
	default:
		return false
	}
}

func hasAny(text string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

// generationProhibitionCues are the phrases that, when they precede an
// anti-pattern needle, mean the text is FORBIDDING it (correct grooming), not
// advocating it.
var generationProhibitionCues = []string{
	"do not", "don't", "never", "avoid", "without", "no longer",
	"instead of", "rather than", "remain orthogonal", "stay orthogonal",
	"keep orthogonal", "must not", "should not", "not create", "not a",
}

// advocates reports whether any needle appears in text in an ADVOCATING context
// — present, and not immediately preceded by a prohibition cue within a short
// window. It is the negation-guarded form of hasAny used by the conflation
// detectors so that guidance forbidding an anti-pattern does not trip its flag.
func advocates(text string, needles ...string) bool {
	for _, needle := range needles {
		from := 0
		for {
			idx := strings.Index(text[from:], needle)
			if idx < 0 {
				break
			}
			pos := from + idx
			start := pos - 48
			if start < 0 {
				start = 0
			}
			if !hasAny(text[start:pos], generationProhibitionCues...) {
				return true
			}
			from = pos + len(needle)
		}
	}
	return false
}

func containsString(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

func points(max int, checks ...bool) int {
	if len(checks) == 0 {
		return 0
	}
	hit := 0
	for _, ok := range checks {
		if ok {
			hit++
		}
	}
	return max * hit / len(checks)
}

func containsPrivateBoundary(c Candidate) bool {
	fields := []string{
		c.Title, c.ParentRef, c.CurrentState, c.WhyNow, c.WorkingSpine,
		c.PriorityContext, c.InScope, c.OutOfScope, c.DoneCondition, c.Witness, c.AcceptanceGate,
	}
	fields = append(fields, c.Paths...)
	fields = append(fields, c.BoundaryNotes...)
	text := strings.ToLower(strings.Join(fields, "\n"))
	for _, needle := range []string{
		"fak-private",
		"slack control",
		"slack-control",
		"gpu-server reservation",
		"operator-only evidence",
		"credential dump",
		"secret key",
		"api key",
		"private token",
	} {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

func compact(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

type reasonSet map[string]bool

func (s reasonSet) add(reason string) { s[reason] = true }
func (s reasonSet) has(reason string) bool {
	return s[reason]
}
func (s reasonSet) list() []string {
	out := make([]string, 0, len(s))
	for reason := range s {
		out = append(out, reason)
	}
	sort.Strings(out)
	return out
}
