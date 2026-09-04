// Package fleetsearch joins the durable lifecycle, child-registration, and
// tool-process stores into one read-only operational session search.
package fleetsearch

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// Schema is the stable machine-output contract emitted by fak search.
const Schema = "fak-fleet-search/1"

const (
	// DefaultLimit defines the fallback ceiling for returned search hits when unspecified.
	DefaultLimit = 20
	// MaxLimit defines the upper bound for search results allowed in a single query.
	MaxLimit = 100
)

// Verdict expresses the terminal match assessment derived across all queried stores.
type Verdict string

const (
	// VerdictSoleMatch indicates exactly one operational session satisfied all query predicates.
	VerdictSoleMatch Verdict = "SOLE_MATCH"
	// VerdictMatches indicates multiple operational sessions satisfied the query criteria.
	VerdictMatches Verdict = "MATCHES"
	// VerdictNoMatch indicates zero operational sessions satisfied the query criteria.
	VerdictNoMatch Verdict = "NO_MATCH"
	// VerdictNoContentTerms indicates the query lacked plain-language terms for text matching.
	VerdictNoContentTerms Verdict = "NO_CONTENT_TERMS"
	// VerdictPartialCoverage indicates at least one requested store was skipped, unreadable, or degraded.
	VerdictPartialCoverage Verdict = "PARTIAL_COVERAGE"
)

// Store identifies an operational session data source ingested by fleet search.
type Store string

const (
	// StoreLifecycle selects the durable lifecycle session journal as a query source.
	StoreLifecycle Store = "lifecycle"
	// StoreRegistration selects the child worker registration store as a query source.
	StoreRegistration Store = "registration"
	// StoreToolProcess selects the background tool execution tracking store as a query source.
	StoreToolProcess Store = "tool_process"
)

var storeOrder = []Store{StoreLifecycle, StoreRegistration, StoreToolProcess}

// CoverageStatus describes whether an ingested store was completely read or degraded.
type CoverageStatus string

const (
	// CoverageComplete indicates all records from the store were successfully parsed.
	CoverageComplete CoverageStatus = "COMPLETE"
	// CoverageIncomplete indicates store records were partially read or had format anomalies.
	CoverageIncomplete CoverageStatus = "PARTIAL"
	// CoverageUnavailable indicates the store file was missing or could not be accessed.
	CoverageUnavailable CoverageStatus = "UNAVAILABLE"
	// CoverageSkipped indicates the store was intentionally bypassed by query configuration.
	CoverageSkipped CoverageStatus = "SKIPPED"
)

// Liveness classifies the execution state of an operational session.
type Liveness string

const (
	// LivenessActive designates sessions actively heartbeating or running child processes.
	LivenessActive Liveness = "ACTIVE"
	// LivenessStale designates sessions whose heartbeat or progress exceeded the threshold.
	LivenessStale Liveness = "STALE"
	// LivenessCrashed designates sessions terminated by unhandled errors or killed processes.
	LivenessCrashed Liveness = "CRASHED"
	// LivenessCompleted designates sessions that reached normal terminal exit.
	LivenessCompleted Liveness = "COMPLETED"
	// LivenessUnknown designates sessions with insufficient lifecycle telemetry to determine state.
	LivenessUnknown Liveness = "UNKNOWN"
)

// Query holds parsed content terms, store filters, liveness constraints, and result limits.
type Query struct {
	Raw      string     `json:"raw"`
	Terms    []string   `json:"terms"`
	Liveness []Liveness `json:"liveness,omitempty"`
	Stores   []Store    `json:"stores,omitempty"`
	Limit    int        `json:"limit"`
}

// Coverage records the read status, path, row count, and diagnostics for one store.
type Coverage struct {
	Store   Store          `json:"store"`
	Status  CoverageStatus `json:"status"`
	Path    string         `json:"path,omitempty"`
	Records int            `json:"records"`
	Detail  string         `json:"detail,omitempty"`
}

// Evidence links an individual matched session attribute back to its originating store.
type Evidence struct {
	Store   Store  `json:"store"`
	Kind    string `json:"kind"`
	ID      string `json:"id,omitempty"`
	Summary string `json:"summary"`
}

// Hit summarizes a matched operational session with scores, identifiers, and store evidence.
type Hit struct {
	SessionID    string     `json:"session_id"`
	Liveness     Liveness   `json:"liveness"`
	Score        int        `json:"score"`
	MatchedTerms []string   `json:"matched_terms"`
	LastSeen     *time.Time `json:"last_seen,omitempty"`
	Identifiers  []string   `json:"identifiers"`
	Objectives   []string   `json:"objectives"`
	Scope        []string   `json:"scope"`
	Tools        []string   `json:"tools"`
	Evidence     []Evidence `json:"evidence"`
}

// Report encapsulates the complete search output including query, hits, verdict, and coverage.
type Report struct {
	Schema       string     `json:"schema"`
	Query        Query      `json:"query"`
	Verdict      Verdict    `json:"verdict"`
	TotalMatches int        `json:"total_matches"`
	Returned     int        `json:"returned"`
	Hits         []Hit      `json:"hits"`
	Coverage     []Coverage `json:"coverage"`
}

// Input is the pure search boundary. Store loading lives in ingest.go; callers
// can drive the parser/ranker with already-read records and explicit coverage.
type Input struct {
	Query         Query
	Lifecycle     []sessionjournal.Event
	Registrations []sessionregistry.Record
	ToolProcesses []toolproc.Event
	Coverage      []Coverage
	Now           time.Time
	BootTime      time.Time
	StaleAfter    time.Duration
}

// ParseQuery separates plain-language terms from the explicit is:, store:, and
// limit: facets. Quoted phrases stay one term.
func ParseQuery(raw string, defaultLimit int) (Query, error) {
	if defaultLimit == 0 {
		defaultLimit = DefaultLimit
	}
	if defaultLimit < 1 || defaultLimit > MaxLimit {
		return Query{}, fmt.Errorf("limit must be between 1 and %d", MaxLimit)
	}
	words, err := splitQuery(raw)
	if err != nil {
		return Query{}, err
	}
	q := Query{Raw: strings.TrimSpace(raw), Limit: defaultLimit, Terms: []string{}}
	seenTerms := map[string]bool{}
	seenLive := map[Liveness]bool{}
	seenStores := map[Store]bool{}
	for _, word := range words {
		lower := strings.ToLower(strings.TrimSpace(word))
		switch {
		case strings.HasPrefix(lower, "is:"):
			value := strings.TrimPrefix(lower, "is:")
			live, ok := parseLiveness(value)
			if !ok {
				return Query{}, fmt.Errorf("unknown is: facet %q (want active, stale, crashed, completed, or unknown)", value)
			}
			if !seenLive[live] {
				q.Liveness = append(q.Liveness, live)
				seenLive[live] = true
			}
		case strings.HasPrefix(lower, "store:"):
			value := strings.TrimPrefix(lower, "store:")
			store, ok := parseStore(value)
			if !ok {
				return Query{}, fmt.Errorf("unknown store: facet %q (want lifecycle, registration, or tool-process)", value)
			}
			if !seenStores[store] {
				q.Stores = append(q.Stores, store)
				seenStores[store] = true
			}
		case strings.HasPrefix(lower, "limit:"):
			value := strings.TrimPrefix(lower, "limit:")
			limit, err := strconv.Atoi(value)
			if err != nil || limit < 1 || limit > MaxLimit {
				return Query{}, fmt.Errorf("limit facet must be between 1 and %d", MaxLimit)
			}
			q.Limit = limit
		default:
			if lower != "" && !seenTerms[lower] {
				q.Terms = append(q.Terms, lower)
				seenTerms[lower] = true
			}
		}
	}
	return q, nil
}

func splitQuery(raw string) ([]string, error) {
	var words []string
	var b strings.Builder
	var quote rune
	escaped := false
	flush := func() {
		if b.Len() > 0 {
			words = append(words, b.String())
			b.Reset()
		}
	}
	for _, r := range raw {
		if escaped {
			b.WriteRune(r)
			escaped = false
			continue
		}
		if r == '\\' && quote != 0 {
			escaped = true
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			} else {
				b.WriteRune(r)
			}
			continue
		}
		if r == '\'' || r == '"' {
			quote = r
			continue
		}
		if unicode.IsSpace(r) {
			flush()
			continue
		}
		b.WriteRune(r)
	}
	if escaped || quote != 0 {
		return nil, fmt.Errorf("unterminated quoted search term")
	}
	flush()
	return words, nil
}

func parseLiveness(value string) (Liveness, bool) {
	switch strings.ReplaceAll(value, "-", "_") {
	case "active", "live", "running":
		return LivenessActive, true
	case "stale", "stalled":
		return LivenessStale, true
	case "crashed", "failed", "lost", "killed":
		return LivenessCrashed, true
	case "completed", "complete", "closed", "done", "terminal":
		return LivenessCompleted, true
	case "unknown":
		return LivenessUnknown, true
	default:
		return "", false
	}
}

func parseStore(value string) (Store, bool) {
	switch strings.ReplaceAll(value, "-", "_") {
	case "lifecycle", "journal", "session_journal":
		return StoreLifecycle, true
	case "registration", "registrations", "child_registration", "registry":
		return StoreRegistration, true
	case "tool", "tools", "tool_process", "tool_processes", "toolproc":
		return StoreToolProcess, true
	default:
		return "", false
	}
}

type candidateID struct {
	value    string
	priority int
}

type searchItem struct {
	aliases      []string
	ids          []candidateID
	identity     []string
	objectives   []string
	scope        []string
	tools        []string
	lifecycle    []string
	registration []string
	toolText     []string
	evidence     []Evidence
	signals      []Liveness
	lastSeen     time.Time
}

type joined struct {
	ids          []candidateID
	identifiers  map[string]bool
	objectives   map[string]bool
	scope        map[string]bool
	tools        map[string]bool
	lifecycle    map[string]bool
	registration map[string]bool
	toolText     map[string]bool
	stores       map[Store]bool
	evidence     []Evidence
	signals      []Liveness
	lastSeen     time.Time
}

// Search folds and joins the supplied store rows, applies facets, ranks content
// terms, and derives a certainty-aware verdict from the declared coverage.
func Search(in Input) (Report, error) {
	if in.Query.Limit == 0 {
		in.Query.Limit = DefaultLimit
	}
	if in.Query.Limit < 1 || in.Query.Limit > MaxLimit {
		return Report{}, fmt.Errorf("limit must be between 1 and %d", MaxLimit)
	}
	if in.Now.IsZero() {
		return Report{}, fmt.Errorf("observation time is required")
	}
	in.Now = in.Now.UTC()
	coverage := normalizeEvidenceCompleteness(in.Coverage)
	report := Report{Schema: Schema, Query: in.Query, Hits: []Hit{}, Coverage: coverage}
	if len(in.Query.Terms) == 0 {
		report.Verdict = VerdictNoContentTerms
		return report, nil
	}
	items, err := itemsFromInput(in)
	if err != nil {
		return Report{}, err
	}
	groups := joinItems(items)
	for _, group := range groups {
		hit, ok := rankGroup(group, in.Query)
		if ok {
			report.Hits = append(report.Hits, hit)
		}
	}
	sort.Slice(report.Hits, func(i, j int) bool {
		if report.Hits[i].Score != report.Hits[j].Score {
			return report.Hits[i].Score > report.Hits[j].Score
		}
		return report.Hits[i].SessionID < report.Hits[j].SessionID
	})
	report.TotalMatches = len(report.Hits)
	if len(report.Hits) > in.Query.Limit {
		report.Hits = report.Hits[:in.Query.Limit]
	}
	report.Returned = len(report.Hits)
	switch {
	case incompleteEvidence(coverage):
		report.Verdict = VerdictPartialCoverage
	case report.TotalMatches == 0:
		report.Verdict = VerdictNoMatch
	case report.TotalMatches == 1:
		report.Verdict = VerdictSoleMatch
	default:
		report.Verdict = VerdictMatches
	}
	return report, nil
}

func itemsFromInput(in Input) ([]searchItem, error) {
	var items []searchItem
	folded := sessionjournal.FoldEvents(in.Lifecycle)
	classified := sessionjournal.Classify(folded, sessionjournal.ClassifyConfig{
		Now: in.Now, BootTime: in.BootTime, StaleAfter: in.StaleAfter,
	})
	for _, row := range classified {
		item := searchItem{
			aliases: []string{row.ID}, ids: []candidateID{{row.ID, 1}},
			identity:  []string{row.ID, row.Boot, strconv.Itoa(row.PID), strconv.Itoa(row.ParentPID), row.Host, row.Model, row.Agent, row.Account, row.StartSHA, row.Gateway},
			scope:     append([]string{row.CWD}, row.Argv...),
			lifecycle: []string{string(row.Status), row.Reason, row.CloseReason},
			evidence:  []Evidence{{Store: StoreLifecycle, Kind: "session", ID: row.ID, Summary: lifecycleSummary(row)}},
			signals:   []Liveness{livenessFromLifecycle(row.Status)}, lastSeen: row.LastSeen,
		}
		if row.Registration != nil {
			addRegistrationCarry(&item, *row.Registration)
		}
		items = append(items, cleanItem(item))
	}
	for _, row := range in.Registrations {
		item := searchItem{
			aliases:      []string{row.RegistrationID, row.AttemptID, row.Identity.SessionID, row.Identity.ThreadID},
			ids:          []candidateID{{row.Identity.SessionID, 0}, {row.Identity.ThreadID, 2}, {row.RegistrationID, 3}, {row.AttemptID, 4}},
			identity:     []string{row.RegistrationID, row.ParentRegistrationID, row.ParentAttemptID, row.RootRegistrationID, row.AttemptID, row.ResumeOfAttemptID, row.Identity.Runtime, row.Identity.SessionID, row.Identity.ThreadID, strconv.Itoa(row.Identity.PID), row.Identity.HostID, row.LeaseID},
			objectives:   []string{row.RootOutcome, row.RootIssue, row.TaskID, row.GoalID, row.Reason, row.WitnessRef, row.LaunchKind, row.Lane},
			scope:        append([]string(nil), row.Scope...),
			registration: []string{string(row.State)},
			evidence:     []Evidence{{Store: StoreRegistration, Kind: "child_registration", ID: row.RegistrationID, Summary: registrationSummary(row)}},
			signals:      []Liveness{livenessFromRegistration(row.State)}, lastSeen: latestTime(row.TerminalAt, row.HeartbeatAt, row.StartedAt, row.CreatedAt),
		}
		items = append(items, cleanItem(item))
	}
	if len(in.ToolProcesses) > 0 {
		table, err := toolproc.Fold(in.ToolProcesses, in.Now.UnixMilli(), toolproc.Config{})
		if err != nil {
			return nil, err
		}
		for _, row := range table.Procs {
			joinedRecord := strings.TrimSpace(row.Session)
			if joinedRecord == "" {
				joinedRecord = "unattributed:" + row.CallID
			}
			lastMS := row.StartMS
			if row.LastPulseMS > lastMS {
				lastMS = row.LastPulseMS
			}
			if row.EndMS > lastMS {
				lastMS = row.EndMS
			}
			item := searchItem{
				aliases: []string{joinedRecord}, ids: []candidateID{{joinedRecord, 0}},
				identity: []string{row.Session, row.CallID, row.ParentCallID}, tools: []string{row.Tool},
				toolText: []string{string(row.State), string(row.Liveness), row.ExitStatus, row.KillReason, row.Coverage, findingsText(row.Findings)},
				evidence: []Evidence{{Store: StoreToolProcess, Kind: "tool_process", ID: row.CallID, Summary: toolSummary(row)}},
				signals:  []Liveness{livenessFromTool(row)},
			}
			if lastMS > 0 {
				item.lastSeen = time.UnixMilli(lastMS).UTC()
			}
			items = append(items, cleanItem(item))
		}
	}
	return items, nil
}

func addRegistrationCarry(item *searchItem, row sessionjournal.RegistrationCarry) {
	item.aliases = append(item.aliases, row.RegistrationID, row.AttemptID, row.SessionID, row.ThreadID)
	item.ids = append(item.ids, candidateID{row.SessionID, 0}, candidateID{row.ThreadID, 2}, candidateID{row.RegistrationID, 3}, candidateID{row.AttemptID, 4})
	item.identity = append(item.identity, row.RegistrationID, row.ParentRegistrationID, row.ParentAttemptID, row.RootRegistrationID, row.AttemptID, row.ResumeOfAttemptID, row.Runtime, row.SessionID, row.ThreadID, strconv.Itoa(row.PID), row.HostID, row.LeaseID)
	item.objectives = append(item.objectives, row.RootOutcome, row.RootIssue, row.TaskID, row.GoalID, row.Reason, row.WitnessRef, row.LaunchKind, row.Lane)
	item.scope = append(item.scope, row.Scope...)
	item.registration = append(item.registration, row.State)
}

func cleanItem(item searchItem) searchItem {
	item.aliases = compactStrings(item.aliases)
	item.identity = compactStrings(item.identity)
	item.objectives = compactStrings(item.objectives)
	item.scope = compactStrings(item.scope)
	item.tools = compactStrings(item.tools)
	item.lifecycle = compactStrings(item.lifecycle)
	item.registration = compactStrings(item.registration)
	item.toolText = compactStrings(item.toolText)
	var ids []candidateID
	for _, id := range item.ids {
		id.value = strings.TrimSpace(id.value)
		if id.value != "" {
			ids = append(ids, id)
		}
	}
	item.ids = ids
	return item
}

type disjointSet struct{ parent []int }

func newDisjointSet(n int) *disjointSet {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	return &disjointSet{parent: p}
}
func (d *disjointSet) find(x int) int {
	if d.parent[x] != x {
		d.parent[x] = d.find(d.parent[x])
	}
	return d.parent[x]
}
func (d *disjointSet) union(a, b int) {
	a, b = d.find(a), d.find(b)
	if a != b {
		if a > b {
			a, b = b, a
		}
		d.parent[b] = a
	}
}

func joinItems(items []searchItem) []*joined {
	ds := newDisjointSet(len(items))
	seen := map[string]int{}
	for i, item := range items {
		for _, alias := range item.aliases {
			key := strings.ToLower(alias)
			if prior, ok := seen[key]; ok {
				ds.union(i, prior)
			} else {
				seen[key] = i
			}
		}
	}
	byRoot := map[int]*joined{}
	for i, item := range items {
		root := ds.find(i)
		group := byRoot[root]
		if group == nil {
			group = &joined{
				identifiers: map[string]bool{}, objectives: map[string]bool{}, scope: map[string]bool{}, tools: map[string]bool{},
				lifecycle: map[string]bool{}, registration: map[string]bool{}, toolText: map[string]bool{}, stores: map[Store]bool{},
			}
			byRoot[root] = group
		}
		group.ids = append(group.ids, item.ids...)
		addSet(group.identifiers, append(item.aliases, item.identity...)...)
		addSet(group.objectives, item.objectives...)
		addSet(group.scope, item.scope...)
		addSet(group.tools, item.tools...)
		addSet(group.lifecycle, item.lifecycle...)
		addSet(group.registration, item.registration...)
		addSet(group.toolText, item.toolText...)
		group.evidence = append(group.evidence, item.evidence...)
		for _, ev := range item.evidence {
			group.stores[ev.Store] = true
		}
		group.signals = append(group.signals, item.signals...)
		if item.lastSeen.After(group.lastSeen) {
			group.lastSeen = item.lastSeen
		}
	}
	roots := make([]int, 0, len(byRoot))
	for root := range byRoot {
		roots = append(roots, root)
	}
	sort.Ints(roots)
	out := make([]*joined, 0, len(roots))
	for _, root := range roots {
		out = append(out, byRoot[root])
	}
	return out
}

func rankGroup(group *joined, q Query) (Hit, bool) {
	live := foldLiveness(group.signals)
	if len(q.Liveness) > 0 && !hasLiveness(q.Liveness, live) {
		return Hit{}, false
	}
	for _, store := range q.Stores {
		if !group.stores[store] {
			return Hit{}, false
		}
	}

	identities := sortedKeys(group.identifiers)
	objectives := sortedKeys(group.objectives)
	scope := sortedKeys(group.scope)
	tools := sortedKeys(group.tools)
	lifecycle := sortedKeys(group.lifecycle)
	registration := sortedKeys(group.registration)
	toolText := sortedKeys(group.toolText)
	all := strings.ToLower(strings.Join(append(append(append(append(append(append([]string{}, identities...), objectives...), scope...), tools...), lifecycle...), append(registration, toolText...)...), " "))
	score := 0
	matched := make([]string, 0, len(q.Terms))
	for _, term := range q.Terms {
		term = strings.ToLower(term)
		if !strings.Contains(all, term) {
			return Hit{}, false
		}
		matched = append(matched, term)
		score += fieldScore(term, identities, objectives, scope, tools, lifecycle, registration, toolText)
	}
	score += len(group.stores)
	if score == 0 {
		return Hit{}, false
	}

	sort.Slice(group.evidence, func(i, j int) bool {
		if storeRank(group.evidence[i].Store) != storeRank(group.evidence[j].Store) {
			return storeRank(group.evidence[i].Store) < storeRank(group.evidence[j].Store)
		}
		if group.evidence[i].Kind != group.evidence[j].Kind {
			return group.evidence[i].Kind < group.evidence[j].Kind
		}
		return group.evidence[i].ID < group.evidence[j].ID
	})
	hit := Hit{
		SessionID: chooseRecordID(group.ids, identities), Liveness: live, Score: score, MatchedTerms: matched,
		Identifiers: identities, Objectives: objectives, Scope: scope, Tools: tools, Evidence: group.evidence,
	}
	if !group.lastSeen.IsZero() {
		last := group.lastSeen.UTC()
		hit.LastSeen = &last
	}
	return hit, true
}

func fieldScore(term string, fields ...[]string) int {
	weights := []int{30, 20, 15, 12, 8, 10, 6}
	best := 0
	for i, values := range fields {
		for _, value := range values {
			lower := strings.ToLower(value)
			if lower == term {
				if weights[i]+20 > best {
					best = weights[i] + 20
				}
			} else if strings.Contains(lower, term) && weights[i] > best {
				best = weights[i]
			}
		}
	}
	return best
}

func chooseRecordID(ids []candidateID, fallback []string) string {
	best := candidateID{priority: 1 << 30}
	for _, id := range ids {
		if id.value == "" {
			continue
		}
		if id.priority < best.priority || id.priority == best.priority && id.value < best.value {
			best = id
		}
	}
	if best.value != "" {
		return best.value
	}
	if len(fallback) > 0 {
		return fallback[0]
	}
	return "unknown"
}

func foldLiveness(signals []Liveness) Liveness {
	seen := map[Liveness]bool{}
	for _, signal := range signals {
		seen[signal] = true
	}
	for _, candidate := range []Liveness{LivenessActive, LivenessStale, LivenessCrashed, LivenessCompleted, LivenessUnknown} {
		if seen[candidate] {
			return candidate
		}
	}
	return LivenessUnknown
}

func livenessFromLifecycle(status sessionjournal.Status) Liveness {
	switch status {
	case sessionjournal.StatusLive:
		return LivenessActive
	case sessionjournal.StatusStale:
		return LivenessStale
	case sessionjournal.StatusCrashed:
		return LivenessCrashed
	case sessionjournal.StatusClosed:
		return LivenessCompleted
	default:
		return LivenessUnknown
	}
}

func livenessFromRegistration(state sessionregistry.State) Liveness {
	switch state {
	case sessionregistry.StateRegistered, sessionregistry.StateActive:
		return LivenessActive
	case sessionregistry.StateFailed, sessionregistry.StateLost:
		return LivenessCrashed
	case sessionregistry.StateCompleted, sessionregistry.StateCancelled, sessionregistry.StateReaped:
		return LivenessCompleted
	default:
		return LivenessUnknown
	}
}

func livenessFromTool(row toolproc.Proc) Liveness {
	switch row.State {
	case toolproc.StateRunning:
		if row.Liveness == toolproc.LivenessStalled {
			return LivenessStale
		}
		return LivenessActive
	case toolproc.StateKilled:
		return LivenessCrashed
	case toolproc.StateDone:
		return LivenessCompleted
	default:
		return LivenessUnknown
	}
}

func lifecycleSummary(row sessionjournal.Classified) string {
	parts := []string{"status=" + string(row.Status)}
	for _, pair := range [][2]string{{"cwd", row.CWD}, {"agent", row.Agent}, {"model", row.Model}, {"reason", row.Reason}} {
		if strings.TrimSpace(pair[1]) != "" {
			parts = append(parts, pair[0]+"="+pair[1])
		}
	}
	return strings.Join(parts, " ")
}

func registrationSummary(row sessionregistry.Record) string {
	parts := []string{"state=" + string(row.State)}
	for _, pair := range [][2]string{{"task", row.TaskID}, {"goal", row.GoalID}, {"lane", row.Lane}, {"runtime", row.Identity.Runtime}} {
		if strings.TrimSpace(pair[1]) != "" {
			parts = append(parts, pair[0]+"="+pair[1])
		}
	}
	return strings.Join(parts, " ")
}

func toolSummary(row toolproc.Proc) string {
	parts := []string{"tool=" + row.Tool, "state=" + string(row.State), "liveness=" + string(row.Liveness)}
	if row.KillReason != "" {
		parts = append(parts, "reason="+row.KillReason)
	}
	return strings.Join(parts, " ")
}

func findingsText(findings []toolproc.Finding) string {
	parts := make([]string, 0, len(findings)*3)
	for _, finding := range findings {
		parts = append(parts, finding.Reason, string(finding.Advice), finding.Detail)
	}
	return strings.Join(parts, " ")
}

func normalizeEvidenceCompleteness(in []Coverage) []Coverage {
	byStore := map[Store]Coverage{}
	for _, row := range in {
		store, ok := parseStore(string(row.Store))
		if !ok {
			continue
		}
		row.Store = store
		byStore[store] = row
	}
	out := make([]Coverage, 0, len(storeOrder))
	for _, store := range storeOrder {
		row, ok := byStore[store]
		if !ok {
			row = Coverage{Store: store, Status: CoverageUnavailable, Detail: "coverage not reported"}
		}
		if row.Status == "" {
			row.Status = CoverageUnavailable
			row.Detail = "coverage status missing"
		}
		out = append(out, row)
	}
	return out
}

func incompleteEvidence(rows []Coverage) bool {
	for _, row := range rows {
		if row.Status != CoverageComplete {
			return true
		}
	}
	return false
}

func compactStrings(values []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || value == "0" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func addSet(set map[string]bool, values ...string) {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && value != "0" {
			set[value] = true
		}
	}
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func latestTime(values ...time.Time) time.Time {
	var latest time.Time
	for _, value := range values {
		if value.After(latest) {
			latest = value.UTC()
		}
	}
	return latest
}

func hasLiveness(values []Liveness, want Liveness) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func storeRank(store Store) int {
	for i, candidate := range storeOrder {
		if store == candidate {
			return i
		}
	}
	return len(storeOrder)
}
