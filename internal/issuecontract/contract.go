// Package issuecontract reviews machine-created GitHub issue candidates before
// they enter the dispatch loop.
//
// The contract is deliberately about the working spine, not polish: a generated
// issue is dispatchable only when it names the useful path that becomes more
// true, what is in and out of scope, how done will be witnessed, and where the
// work routes. Missing detail becomes triage debt instead of an unscoped worker
// prompt.
package issuecontract

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	Schema       = "fak.issue-candidate.v1"
	ReviewSchema = "fak.issue-candidate-review.v1"

	Dispatchable = "dispatchable"
	TriageOnly   = "triage_only"
	Refused      = "refused"

	ReasonScopeIncomplete = "ISSUE_SCOPE_INCOMPLETE"
	ReasonUnrouted        = "ISSUE_UNROUTED"
	ReasonPrivateBoundary = "ISSUE_PRIVATE_BOUNDARY"
	ReasonLiveUnarmored   = "ISSUE_LIVE_UNARMORED"
	ReasonNotDispatchLeaf = "ISSUE_NOT_DISPATCH_LEAF"
	ReasonOversizedSteps  = "ISSUE_OVERSIZED_EXPECTED_STEPS"
	ReasonNoiseIncomplete = "ISSUE_NOISE_CONTROL_INCOMPLETE"
	ReasonAgentIncomplete = "ISSUE_AGENT_CONTEXT_INCOMPLETE"
)

const MaxDispatchExpectedSteps = 8

var keyRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$`)
var markdownHeadingRE = regexp.MustCompile(`^#{1,6}\s+(.+?)\s*$`)
var codeSpanRE = regexp.MustCompile("`([^`]+)`")

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

// Candidate is the pure input shape a producer can review before rendering or
// syncing a public GitHub issue.
type Candidate struct {
	Schema          string   `json:"schema,omitempty"`
	Key             string   `json:"key"`
	Title           string   `json:"title"`
	ParentRef       string   `json:"parent_ref,omitempty"`
	CurrentState    string   `json:"current_state,omitempty"`
	WhyNow          string   `json:"why_now,omitempty"`
	WorkingSpine    string   `json:"working_spine,omitempty"`
	PriorityContext string   `json:"priority_context,omitempty"`
	WorkUnit        string   `json:"work_unit,omitempty"`
	ExpectedSteps   int      `json:"expected_steps,omitempty"`
	Assumptions     []string `json:"assumptions,omitempty"`
	ConfusionRisks  []string `json:"confusion_risks,omitempty"`
	Coordination    []string `json:"coordination,omitempty"`
	Trigger         string   `json:"trigger,omitempty"`
	BatchPolicy     string   `json:"batch_policy,omitempty"`
	InScope         string   `json:"in_scope,omitempty"`
	OutOfScope      string   `json:"out_of_scope,omitempty"`
	DoneCondition   string   `json:"done_condition,omitempty"`
	Witness         string   `json:"witness,omitempty"`
	AcceptanceGate  string   `json:"acceptance_gate,omitempty"`
	Lane            string   `json:"lane,omitempty"`
	Paths           []string `json:"paths,omitempty"`
	Labels          []string `json:"labels,omitempty"`
	Priority        string   `json:"priority,omitempty"`
	BoundaryNotes   []string `json:"boundary_notes,omitempty"`
	Private         bool     `json:"private,omitempty"`
	ClosureBinding  string   `json:"closure_binding,omitempty"`
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

// Review is the closed-vocabulary verdict over a Candidate.
type Review struct {
	Schema          string        `json:"schema"`
	OK              bool          `json:"ok"`
	Verdict         string        `json:"verdict"`
	Dispatchability string        `json:"dispatchability"`
	Reasons         []string      `json:"reasons,omitempty"`
	MissingFields   []string      `json:"missing_fields,omitempty"`
	Key             string        `json:"key,omitempty"`
	Lane            string        `json:"lane,omitempty"`
	Paths           []string      `json:"paths,omitempty"`
	Score           Score         `json:"score"`
	SpinePriority   SpinePriority `json:"spine_priority"`
	AgentContext    AgentContext  `json:"agent_context"`
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
	out := Review{
		Schema:        ReviewSchema,
		Key:           c.Key,
		Lane:          c.Lane,
		Paths:         append([]string(nil), c.Paths...),
		Reasons:       reasons.list(),
		MissingFields: missing,
		Score:         score,
		SpinePriority: spinePriority,
		AgentContext:  agentContext,
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
	return ReviewCandidate(CandidateFromIssueDraft(d), opt)
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
		Key:             issueDraftKey(d),
		Title:           d.Title,
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
		Paths:           issueDraftPaths(section("Path hints", "Paths", "Files")),
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
	if d.Number > 0 {
		return "issue/" + strconv.Itoa(d.Number)
	}
	slug := slugKeyPart(d.Title)
	if slug == "" {
		slug = "unknown"
	}
	return "manual/" + slug
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
	c.Labels = compact(c.Labels)
	c.BoundaryNotes = compact(c.BoundaryNotes)
	return c
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
	if c.Trigger == "" {
		missing = append(missing, "trigger")
	}
	if c.BatchPolicy == "" {
		missing = append(missing, "batch_policy")
	}
	return missing
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
