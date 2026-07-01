package dispatchtick

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

const RouterSchema = "fleet-issue-lane-router/1"

const BlockedByHumanLabel = "blocked-by-human"
const MaxDispatchExpectedSteps = 8

var TriageOnlyLabels = map[string]bool{
	"guard-complaint": true,
	"idea-scout":      true,
	"needs-triage":    true,
	"needs-scope":     true,
	"research":        true,
	"triage-only":     true,
	"triage_only":     true,
}

var (
	scopeRE      = regexp.MustCompile(`\b(\w+)\(([^)]+)\)`)
	barePrefixRE = regexp.MustCompile(`^([A-Za-z][\w-]*):\s`)
	epicTitleRE  = regexp.MustCompile(`(?i)^\s*epic\b\s*[\(:]`)
	pathRE       = regexp.MustCompile(`(?:(?:fak/)?(?:internal|cmd|experiments|tools|docs|visuals)|\.(?:github|claude))/[A-Za-z0-9_./-]+`)
)

var ExclusiveRouterLanes = map[string]bool{
	"abi":     true,
	"release": true,
	"global":  true,
}

var ScopeAlias = map[string]string{
	"cuda":             "compute",
	"gpu":              "compute",
	"vulkan":           "compute",
	"metal":            "compute",
	"serve":            "gateway",
	"anthropic":        "gateway",
	"inkernel":         "engine",
	"qwen35":           "model",
	"qwen36":           "model",
	"loader":           "ggufload",
	"swebench":         "experiments",
	"demo":             "experiments",
	"simpledemo":       "experiments",
	"fanbench":         "bench",
	"terminal-bench":   "bench",
	"testing":          "ci",
	"simd":             "model",
	"rehydrate":        "sessionimage",
	"devex":            "devindex",
	"readme":           "docs",
	"getting-started":  "docs",
	"fak":              "docs",
	"adopt":            "docs",
	"licensing":        "docs",
	"dashboard":        "metrics",
	"observability":    "metrics",
	"dos":              "tools",
	"control-pane":     "tools",
	"rsi":              "tools",
	"dispatch":         "tools",
	"scrub":            "tools",
	"ops":              "tools",
	"grafana":          "tools",
	"support-maturity": "tools",
	"cachevalue":       "tools",
	"tooling":          "tools",
	"mobile":           "examples",
	"edge":             "examples",
	"install":          "cmd",
	"adjudication":     "adjudicator",
}

var LabelAlias = map[string]string{
	"gpu":             "compute",
	"compute":         "compute",
	"performance":     "compute",
	"docs":            "docs",
	"documentation":   "docs",
	"model":           "model",
	"model-arch":      "model",
	"loader":          "ggufload",
	"security":        "policy",
	"trust-floor":     "policy",
	"build":           "ci",
	"testing":         "ci",
	"rsi":             "tools",
	"dispatch":        "tools",
	"agentic-serving": "gateway",
}

var KeywordAlias = map[string]string{
	"promptmmu":     "promptmmu",
	"cuda":          "compute",
	"a100":          "compute",
	"gpu":           "compute",
	"benchmark":     "bench",
	"dashboard":     "metrics",
	"observability": "metrics",
	"telemetry":     "metrics",
	"tooling":       "tools",
	"backlog":       "tools",
}

var ConfidenceRank = map[string]int{
	"path-confirmed": 5,
	"exact-scope":    4,
	"alias":          3,
	"label":          2,
	"keyword":        1,
	"none":           0,
}

type LaneTaxonomy struct {
	Concurrent []string
	Trees      map[string][]string
}

type IssueLabel struct {
	Name string `json:"name"`
}

type Issue struct {
	Number int          `json:"number"`
	Title  string       `json:"title"`
	Body   string       `json:"body"`
	Labels []IssueLabel `json:"labels"`
}

type IssueRoute struct {
	Number         int      `json:"number"`
	Title          string   `json:"title"`
	Lane           string   `json:"lane"`
	Confidence     string   `json:"confidence"`
	Signal         string   `json:"signal"`
	SignalConflict bool     `json:"signal_conflict"`
	Paths          []string `json:"paths,omitempty"`
	WorkUnit       string   `json:"work_unit,omitempty"`
	ExpectedSteps  int      `json:"expected_steps,omitempty"`
	Trigger        string   `json:"trigger,omitempty"`
	BatchPolicy    string   `json:"batch_policy,omitempty"`
	UnroutedReason string   `json:"unrouted_reason,omitempty"`
	// Generation is the issue's classified scheduling bucket (gen/now, gen/next,
	// gen/second-next, gen/future), omitted when the issue carries none of the four
	// labels. Per docs/generation-loop-scheduling.md, this is a scheduling hint
	// surfaced for a consumer to read, not a queue silo or a priority override.
	Generation string `json:"generation,omitempty"`
}

type SkippedIssue struct {
	Number        int    `json:"number"`
	Title         string `json:"title"`
	Reason        string `json:"reason,omitempty"`
	NextAction    string `json:"next_action,omitempty"`
	WorkUnit      string `json:"work_unit,omitempty"`
	ExpectedSteps int    `json:"expected_steps,omitempty"`
}

type RouterRepairQueue struct {
	Kind             string         `json:"kind"`
	Count            int            `json:"count"`
	StepBudget       int            `json:"step_budget"`
	ChildIssueBudget int            `json:"child_issue_budget,omitempty"`
	NextAction       string         `json:"next_action"`
	ByReason         map[string]int `json:"by_reason,omitempty"`
	Issues           []int          `json:"issues,omitempty"`
}

type RouterCoverage struct {
	Complete      bool     `json:"complete"`
	Truncated     bool     `json:"truncated"`
	Injected      bool     `json:"injected,omitempty"`
	IssuesFetched int      `json:"issues_fetched"`
	IssueLimit    int      `json:"issue_limit"`
	Notes         []string `json:"notes"`
}

type RouterCounts struct {
	Open                int            `json:"open"`
	Routed              int            `json:"routed"`
	Unrouted            int            `json:"unrouted"`
	UnroutedFrac        float64        `json:"unrouted_frac"`
	RoutedStepBudget    int            `json:"routed_step_budget,omitempty"`
	ByConfidence        map[string]int `json:"by_confidence"`
	SkippedHumanBlocked int            `json:"skipped_human_blocked"`
	SkippedByReason     map[string]int `json:"skipped_by_reason,omitempty"`
}

type RouterLaneGroup struct {
	Tree       []string        `json:"tree"`
	Count      int             `json:"count"`
	StepBudget int             `json:"step_budget,omitempty"`
	Issues     []int           `json:"issues"`
	SubLanes   []RouterSubLane `json:"sub_lanes,omitempty"`
	WorkUnits  map[int]string  `json:"work_units,omitempty"`
	IssueSteps map[int]int     `json:"issue_steps,omitempty"`
	// Priority maps an issue number to its dispatch-priority weight for the
	// issues that carry a priority/* label (unlabeled issues are omitted and
	// resolve to PriorityWeightDefault). It is how the picker orders the lane's
	// candidates priority-first (#1395) without re-deriving weights from labels.
	Priority map[int]int `json:"priority,omitempty"`
	// Generation maps an issue number to its classified generation bucket for the
	// issues that carry a gen/* label (unclassified issues are omitted). It is how
	// a generation-aware picker restricts a lane's candidates to the admitted
	// horizon without re-deriving the bucket from labels.
	Generation map[int]string `json:"generation,omitempty"`
}

type RouterSubLane struct {
	Prefix     string `json:"prefix"`
	Count      int    `json:"count"`
	StepBudget int    `json:"step_budget,omitempty"`
	Issues     []int  `json:"issues"`
}

type RouterPayload struct {
	Schema              string                     `json:"schema"`
	OK                  bool                       `json:"ok"`
	Verdict             string                     `json:"verdict"`
	Finding             string                     `json:"finding"`
	Reason              string                     `json:"reason"`
	NextAction          string                     `json:"next_action"`
	Workspace           string                     `json:"workspace"`
	Coverage            RouterCoverage             `json:"coverage"`
	Counts              RouterCounts               `json:"counts"`
	Lanes               map[string]RouterLaneGroup `json:"lanes"`
	Issues              []IssueRoute               `json:"issues"`
	RepairQueues        []RouterRepairQueue        `json:"repair_queues,omitempty"`
	SkippedHumanBlocked []SkippedIssue             `json:"skipped_human_blocked"`
}

type RouterInput struct {
	Workspace        string
	Taxonomy         LaneTaxonomy
	Issues           []Issue
	IssueLimit       int
	MaxUnroutedFrac  float64
	FetchError       string
	Injected         bool
	ScopeAlias       map[string]string
	LabelAlias       map[string]string
	KeywordAlias     map[string]string
	BlockedLabelName string
}

func RouteIssues(in RouterInput) RouterPayload {
	issueLimit := in.IssueLimit
	if issueLimit <= 0 {
		issueLimit = 1000
	}
	maxUnrouted := in.MaxUnroutedFrac
	if maxUnrouted == 0 {
		maxUnrouted = 0.25
	}
	blockedLabel := strings.TrimSpace(in.BlockedLabelName)
	if blockedLabel == "" {
		blockedLabel = BlockedByHumanLabel
	}
	coverage := ComputeRouterCoverage(len(in.Issues), issueLimit)
	if in.Injected {
		coverage = RouterCoverage{
			Complete:      true,
			Truncated:     false,
			Injected:      true,
			IssuesFetched: len(in.Issues),
			IssueLimit:    issueLimit,
			Notes:         []string{"issues injected via --issues or a named view; coverage reflects the provided slice, not a full gh fetch"},
		}
	}

	blocked := []Issue{}
	routable := []Issue{}
	for _, issue := range in.Issues {
		if !IsDispatchable(issue, blockedLabel) {
			blocked = append(blocked, issue)
			continue
		}
		routable = append(routable, issue)
	}

	fetchError := strings.TrimSpace(in.FetchError)
	if fetchError == "" && len(in.Taxonomy.Concurrent) == 0 {
		fetchError = "dos doctor returned no lanes - run from the repo root"
	} else if fetchError == "" && len(in.Issues) == 0 && !in.Injected {
		fetchError = "gh returned no open issues (auth/network?)"
	}

	routes := make([]IssueRoute, 0, len(routable))
	priority := map[int]int{}
	for _, issue := range routable {
		routes = append(routes, RouteIssue(issue, in.Taxonomy, RouteOptions{
			ScopeAlias:   in.ScopeAlias,
			LabelAlias:   in.LabelAlias,
			KeywordAlias: in.KeywordAlias,
		}))
		if w := PriorityWeight(labelNames(issue)); w != PriorityWeightDefault {
			priority[issue.Number] = w
		}
	}
	return BuildRouterPayload(RouterPayloadInput{
		Workspace:        in.Workspace,
		Routes:           routes,
		Trees:            in.Taxonomy.Trees,
		Priority:         priority,
		MaxUnroutedFrac:  maxUnrouted,
		FetchError:       fetchError,
		Coverage:         coverage,
		SkippedBlocked:   blocked,
		BlockedLabelName: blockedLabel,
	})
}

type RouteOptions struct {
	ScopeAlias   map[string]string
	LabelAlias   map[string]string
	KeywordAlias map[string]string
}

func RouteIssue(issue Issue, taxonomy LaneTaxonomy, opts RouteOptions) IssueRoute {
	scopeAlias := opts.ScopeAlias
	if scopeAlias == nil {
		scopeAlias = ScopeAlias
	}
	labelAlias := opts.LabelAlias
	if labelAlias == nil {
		labelAlias = LabelAlias
	}
	keywordAlias := opts.KeywordAlias
	if keywordAlias == nil {
		keywordAlias = KeywordAlias
	}

	title := issue.Title
	body := issue.Body
	laneSet := map[string]bool{}
	for _, lane := range taxonomy.Concurrent {
		laneSet[lane] = true
	}

	paths := ExtractIssueRepoPaths(title, body)
	pathLanes := []string{}
	seenPathLane := map[string]bool{}
	for _, p := range paths {
		for _, lane := range PathMatchesLane(p, taxonomy.Trees) {
			if laneSet[lane] && !seenPathLane[lane] {
				seenPathLane[lane] = true
				pathLanes = append(pathLanes, lane)
			}
		}
	}
	var pathLane string
	pathAmbiguous := false
	if len(pathLanes) == 1 {
		pathLane = pathLanes[0]
	} else if len(pathLanes) > 1 {
		pathAmbiguous = true
	}

	scope := scopeToken(title)
	typ := typeToken(title)
	scopeLane := ""
	scopeConf := ""
	switch {
	case scope != "" && laneSet[scope] && !ExclusiveRouterLanes[scope]:
		scopeLane, scopeConf = scope, "exact-scope"
	case scope != "" && scopeAlias[scope] != "" && laneSet[scopeAlias[scope]]:
		scopeLane, scopeConf = scopeAlias[scope], "alias"
	case typ != "" && scopeAlias[typ] != "" && laneSet[scopeAlias[typ]]:
		scopeLane, scopeConf = scopeAlias[typ], "alias"
	}

	labelLane := ""
	for _, label := range labelNames(issue) {
		if lane := labelAlias[label]; lane != "" && laneSet[lane] {
			labelLane = lane
			break
		}
	}

	keywordLane := ""
	keyword := ""
	searchable := title + "\n" + body
	keys := sortedKeys(keywordAlias)
	for _, key := range keys {
		lane := keywordAlias[key]
		if laneSet[lane] && HasKeyword(searchable, key) {
			keyword, keywordLane = key, lane
			break
		}
	}

	if ExclusiveRouterLanes[scope] {
		return route(issue, "", "none", "exclusive-scope:"+scope, false, paths,
			fmt.Sprintf("exclusive-lane scope '%s'; operator-gated", scope))
	}
	if pathLane != "" {
		weaker := firstNonEmptyString(scopeLane, labelLane)
		conflict := weaker != "" && weaker != pathLane
		signal := "path:" + pathLane
		if conflict {
			signal += " (overrode " + weaker + ")"
		}
		return route(issue, pathLane, "path-confirmed", signal, conflict, paths, "")
	}
	if pathAmbiguous {
		prefer := ""
		if contains(pathLanes, scopeLane) {
			prefer = scopeLane
		} else if contains(pathLanes, labelLane) {
			prefer = labelLane
		}
		sort.Strings(pathLanes)
		pick := firstNonEmptyString(prefer, pathLanes[0])
		return route(issue, pick, "path-confirmed", "path-ambiguous:"+strings.Join(pathLanes, "|"), true, paths, "")
	}
	if scopeLane != "" {
		token := scope
		if !(laneSet[scope] || scopeAlias[scope] != "") {
			token = typ
		}
		return route(issue, scopeLane, scopeConf, "scope:"+token+"->"+scopeLane, false, nil, "")
	}
	if labelLane != "" {
		return route(issue, labelLane, "label", "label->"+labelLane, false, nil, "")
	}
	if keywordLane != "" {
		return route(issue, keywordLane, "keyword", "keyword:"+keyword+"->"+keywordLane, false, nil, "")
	}
	reason := "no scope/path/label signal"
	if scope != "" {
		reason = "no scope, no repo-path, no aliasable label"
	}
	return route(issue, "", "none", "unrouted", false, nil, reason)
}

func ExtractRepoPaths(text string) []string {
	matches := pathRE.FindAllStringIndex(text, -1)
	out := make([]string, 0, len(matches))
	for _, m := range matches {
		path := text[m[0]:m[1]]
		var prev rune
		if m[0] > 0 {
			prev = []rune(text[:m[0]])[len([]rune(text[:m[0]]))-1]
		}
		if strings.HasPrefix(path, ".github") || strings.HasPrefix(path, ".claude") {
			if prev != 0 && (isWord(prev) || prev == '.') {
				continue
			}
		} else if prev != 0 && isWord(prev) {
			continue
		}
		out = append(out, path)
	}
	return out
}

func ExtractIssueRepoPaths(title, body string) []string {
	if hints, ok := issuePathHintSection(body); ok {
		return ExtractRepoPaths(hints)
	}
	return ExtractRepoPaths(title + "\n" + body)
}

func issuePathHintSection(body string) (string, bool) {
	sections := promptMarkdownSections(body)
	for _, name := range []string{"path hints", "paths", "file scope", "file scopes", "files"} {
		key := normalizePromptHeading(name)
		value, ok := sections[key]
		if ok {
			return value, true
		}
	}
	return "", false
}

func PathMatchesLane(path string, trees map[string][]string) []string {
	p := strings.ReplaceAll(path, "\\", "/")
	if strings.HasPrefix(p, "./") {
		p = p[2:]
	}
	if strings.HasPrefix(p, "fak/") {
		p = strings.TrimPrefix(p, "fak/")
	}
	lanes := sortedKeysSliceMap(trees)
	hits := []string{}
	for _, lane := range lanes {
		for _, glob := range trees[lane] {
			if prefix, ok := strings.CutSuffix(strings.ReplaceAll(glob, "\\", "/"), "/**"); ok && p == prefix {
				hits = append(hits, lane)
				break
			}
			re := globToRegexp(glob)
			if re.MatchString(p) {
				hits = append(hits, lane)
				break
			}
		}
	}
	return hits
}

func ComputeRouterCoverage(issuesFetched, issueLimit int) RouterCoverage {
	if issueLimit <= 0 {
		issueLimit = 1000
	}
	truncated := issuesFetched >= issueLimit
	notes := []string{}
	if truncated {
		notes = append(notes, fmt.Sprintf("gh fetch returned %d open issue(s) = the --issue-limit cap; older open issues may be unrouted - raise --issue-limit", issuesFetched))
	}
	return RouterCoverage{
		Complete:      !truncated,
		Truncated:     truncated,
		IssuesFetched: issuesFetched,
		IssueLimit:    issueLimit,
		Notes:         notes,
	}
}

type RouterPayloadInput struct {
	Workspace string
	Routes    []IssueRoute
	Trees     map[string][]string
	// Priority maps an issue number to its dispatch-priority weight for issues
	// that carry a priority/* label (unlabeled issues are omitted). It drives the
	// priority-first ordering of each lane group's Issues (#1395).
	Priority         map[int]int
	MaxUnroutedFrac  float64
	FetchError       string
	Coverage         RouterCoverage
	SkippedBlocked   []Issue
	BlockedLabelName string
}

func BuildRouterPayload(in RouterPayloadInput) RouterPayload {
	maxUnrouted := in.MaxUnroutedFrac
	if maxUnrouted == 0 {
		maxUnrouted = 0.25
	}
	byConf := map[string]int{}
	for conf := range ConfidenceRank {
		byConf[conf] = 0
	}
	lanes := map[string]RouterLaneGroup{}
	laneRoutes := map[string][]IssueRoute{}
	routedStepBudget := 0
	for _, r := range in.Routes {
		byConf[r.Confidence] = byConf[r.Confidence] + 1
		if r.Lane == "" {
			continue
		}
		laneRoutes[r.Lane] = append(laneRoutes[r.Lane], r)
		grp := lanes[r.Lane]
		grp.Tree = append([]string(nil), in.Trees[r.Lane]...)
		grp.Count++
		stepBudget := routeStepBudget(r)
		grp.StepBudget += stepBudget
		routedStepBudget += stepBudget
		grp.Issues = append(grp.Issues, r.Number)
		if r.WorkUnit != "" {
			if grp.WorkUnits == nil {
				grp.WorkUnits = map[int]string{}
			}
			grp.WorkUnits[r.Number] = r.WorkUnit
		}
		if r.ExpectedSteps > 0 {
			if grp.IssueSteps == nil {
				grp.IssueSteps = map[int]int{}
			}
			grp.IssueSteps[r.Number] = r.ExpectedSteps
		}
		if w := laneIssueWeight(in.Priority, r.Number); w != PriorityWeightDefault {
			if grp.Priority == nil {
				grp.Priority = map[int]int{}
			}
			grp.Priority[r.Number] = w
		}
		if r.Generation != "" {
			if grp.Generation == nil {
				grp.Generation = map[int]string{}
			}
			grp.Generation[r.Number] = r.Generation
		}
		lanes[r.Lane] = grp
	}
	// Order each lane's candidates priority-first (#1395): the heaviest priority/P*
	// label wins, oldest-first within a tier, so an old priority/P1 surfaces ahead
	// of newer unlabeled noise. The picker (cmd/fak) re-applies this with the
	// caller's recency tiebreak; ordering here keeps the payload itself honest for
	// any consumer that reads lanes[*].issues directly.
	for lane, grp := range lanes {
		cands := make([]LaneCandidate, len(grp.Issues))
		for i, n := range grp.Issues {
			cands[i] = LaneCandidate{Number: n, Weight: laneIssueWeight(grp.Priority, n)}
		}
		grp.Issues = OrderLaneCandidates(cands, false)
		grp.SubLanes = buildRouterSubLanes(laneRoutes[lane])
		lanes[lane] = grp
	}
	total := len(in.Routes)
	unrouted := byConf["none"]
	routed := total - unrouted
	frac := 0.0
	if total > 0 {
		frac = float64(unrouted) / float64(total)
		frac = float64(int(frac*10000+0.5)) / 10000
	}

	skipped := make([]SkippedIssue, 0, len(in.SkippedBlocked))
	skippedByReason := map[string]int{}
	for _, issue := range in.SkippedBlocked {
		sk := classifySkippedIssue(issue, in.BlockedLabelName)
		skipped = append(skipped, sk)
		if sk.Reason != "" {
			skippedByReason[sk.Reason]++
		}
	}
	sort.Slice(skipped, func(i, j int) bool { return skipped[i].Number > skipped[j].Number })

	coverage := in.Coverage
	if coverage.IssueLimit == 0 && coverage.IssuesFetched == 0 && len(coverage.Notes) == 0 {
		coverage = RouterCoverage{Complete: true, Notes: []string{}}
	}

	ok, verdict, finding, reason, next := true, "OK", "routed", "", "dos-dispatch workers fold lanes[<their lane>].issues into the dispositions sidecar"
	switch {
	case strings.TrimSpace(in.FetchError) != "":
		ok, verdict, finding = false, "FETCH_ERROR", "fetch_error"
		reason = strings.TrimSpace(in.FetchError)
		next = "fix the gh/dos read-back error, then re-run the lane router"
	case !coverage.Complete:
		ok, verdict, finding = false, "ACTION", "incomplete_coverage"
		reason = fmt.Sprintf("routed %d/%d fetched, but the open-issue fetch was truncated, so some open issues were never routed - %s",
			routed, total, strings.Join(coverage.Notes, "; "))
		next = "re-run with a higher --issue-limit so every open issue is routed"
	case total > 0 && frac > maxUnrouted:
		ok, verdict, finding = false, "ACTION", "high_unrouted"
		reason = fmt.Sprintf("%d/%d open issues UNROUTED (frac=%.4g > %.4g)", unrouted, total, frac, maxUnrouted)
		next = "operator: add scopes/labels or extend SCOPE_ALIAS so workers can target these"
	default:
		skippedNote := ""
		if len(skipped) > 0 {
			skippedNote = fmt.Sprintf("; %d skipped", len(skipped))
		}
		reason = fmt.Sprintf("%d/%d open issues routed to %d lane(s); %d UNROUTED%s", routed, total, len(lanes), unrouted, skippedNote)
	}

	issues := append([]IssueRoute(nil), in.Routes...)
	sort.Slice(issues, func(i, j int) bool { return routeSortLess(issues[i], issues[j]) })
	repairQueues := routerRepairQueues(issues, skipped)

	return RouterPayload{
		Schema:              RouterSchema,
		OK:                  ok,
		Verdict:             verdict,
		Finding:             finding,
		Reason:              reason,
		NextAction:          next,
		Workspace:           in.Workspace,
		Coverage:            coverage,
		Counts:              RouterCounts{Open: total, Routed: routed, Unrouted: unrouted, UnroutedFrac: frac, RoutedStepBudget: routedStepBudget, ByConfidence: byConf, SkippedHumanBlocked: len(skipped), SkippedByReason: skippedByReason},
		Lanes:               lanes,
		Issues:              issues,
		RepairQueues:        repairQueues,
		SkippedHumanBlocked: skipped,
	}
}

func buildRouterSubLanes(routes []IssueRoute) []RouterSubLane {
	if len(routes) < 2 {
		return nil
	}
	groups := map[string]*RouterSubLane{}
	for _, route := range routes {
		prefix := routeSubLanePrefix(route.Paths)
		if prefix == "" {
			continue
		}
		grp := groups[prefix]
		if grp == nil {
			grp = &RouterSubLane{Prefix: prefix}
			groups[prefix] = grp
		}
		grp.Count++
		grp.StepBudget += routeStepBudget(route)
		grp.Issues = append(grp.Issues, route.Number)
	}
	if len(groups) < 2 {
		return nil
	}
	out := make([]RouterSubLane, 0, len(groups))
	for _, grp := range groups {
		sort.Slice(grp.Issues, func(i, j int) bool { return grp.Issues[i] > grp.Issues[j] })
		out = append(out, *grp)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Prefix < out[j].Prefix
	})
	return out
}

func routeSubLanePrefix(paths []string) string {
	prefix := ""
	for _, path := range paths {
		next := pathOwnershipPrefix(path)
		if next == "" {
			continue
		}
		if prefix == "" {
			prefix = next
			continue
		}
		if prefix != next {
			return ""
		}
	}
	return prefix
}

func pathOwnershipPrefix(path string) string {
	parts := strings.Split(strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		return ""
	}
	switch parts[0] {
	case "internal", "cmd", "tools", "docs", "examples", "experiments", "visuals":
		if len(parts) >= 2 {
			return parts[0] + "/" + parts[1]
		}
		return parts[0]
	case ".github", ".claude":
		if len(parts) >= 2 {
			return parts[0] + "/" + parts[1]
		}
		return parts[0]
	default:
		if len(parts) >= 2 {
			return parts[0] + "/" + parts[1]
		}
		return parts[0]
	}
}

func routerRepairQueues(routes []IssueRoute, skipped []SkippedIssue) []RouterRepairQueue {
	queues := map[string]*RouterRepairQueue{}
	add := func(kind string, issue int, stepBudget int, childIssueBudget int, reason string) {
		if stepBudget <= 0 {
			stepBudget = 1
		}
		queue := queues[kind]
		if queue == nil {
			queue = &RouterRepairQueue{
				Kind:       kind,
				NextAction: routerRepairAction(kind),
				ByReason:   map[string]int{},
			}
			queues[kind] = queue
		}
		queue.Count++
		queue.StepBudget += stepBudget
		queue.ChildIssueBudget += childIssueBudget
		if reason != "" {
			queue.ByReason[reason]++
		}
		if issue > 0 && len(queue.Issues) < 12 {
			queue.Issues = append(queue.Issues, issue)
		}
	}
	for _, route := range routes {
		if route.Lane == "" {
			add("route", route.Number, routeStepBudget(route), 0, "ISSUE_UNROUTED")
			continue
		}
		add("dispatch", route.Number, routeStepBudget(route), 0, "")
	}
	for _, skippedIssue := range skipped {
		kind := routerRepairKind(skippedIssue.Reason)
		add(kind, skippedIssue.Number, skippedIssueStepBudget(skippedIssue), skippedIssueChildIssueBudget(skippedIssue, kind), skippedIssue.Reason)
	}
	out := make([]RouterRepairQueue, 0, len(queues))
	for _, queue := range queues {
		if len(queue.ByReason) == 0 {
			queue.ByReason = nil
		}
		out = append(out, *queue)
	}
	sort.Slice(out, func(i, j int) bool {
		ri, rj := routerRepairRank(out[i].Kind), routerRepairRank(out[j].Kind)
		if ri != rj {
			return ri < rj
		}
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Kind < out[j].Kind
	})
	return out
}

func skippedIssueStepBudget(issue SkippedIssue) int {
	if issue.ExpectedSteps > 0 {
		return issue.ExpectedSteps
	}
	return 1
}

func skippedIssueChildIssueBudget(issue SkippedIssue, kind string) int {
	if kind != "split" {
		return 0
	}
	if issue.ExpectedSteps <= 0 {
		return 1
	}
	return (issue.ExpectedSteps + MaxDispatchExpectedSteps - 1) / MaxDispatchExpectedSteps
}

func routerRepairKind(reason string) string {
	switch reason {
	case "BLOCKED_BY_HUMAN":
		return "human"
	case "ISSUE_NOT_DISPATCH_LEAF", "ISSUE_OVERSIZED_EXPECTED_STEPS":
		return "split"
	case "ISSUE_SCOPE_INCOMPLETE", "ISSUE_TRIAGE_ONLY":
		return "scope"
	case "ISSUE_UNROUTED":
		return "route"
	case "ISSUE_LIVE_UNARMORED", "ISSUE_NOISE_CONTROL_INCOMPLETE", "ISSUE_AGENT_CONTEXT_INCOMPLETE":
		return "noise"
	case "ISSUE_PRIVATE_BOUNDARY":
		return "private"
	default:
		return "other"
	}
}

func routerRepairRank(kind string) int {
	switch kind {
	case "dispatch":
		return 0
	case "split":
		return 1
	case "scope":
		return 2
	case "route":
		return 3
	case "noise":
		return 4
	case "private":
		return 5
	case "human":
		return 6
	default:
		return 9
	}
}

func routerRepairAction(kind string) string {
	switch kind {
	case "dispatch":
		return "launch scoped leaf issues through their routed lanes"
	case "split":
		return fmt.Sprintf("decompose non-leaves or oversized rows into child issues with <= %d expected steps", MaxDispatchExpectedSteps)
	case "scope":
		return "add worker-ready scope, done condition, witness, and agent context before dispatch"
	case "route":
		return "add lane/path hints or extend routing aliases so each issue maps to one lane"
	case "noise":
		return "add trigger, batch policy, agent context, and live dedupe/cap evidence"
	case "private":
		return "remove private/operator-only evidence or move the work to the private companion repo"
	case "human":
		return "wait for the human blocker to clear before worker dispatch"
	default:
		return "inspect the skipped reason before dispatch"
	}
}

func IsBlockedByHuman(issue Issue, label string) bool {
	if strings.TrimSpace(label) == "" {
		label = BlockedByHumanLabel
	}
	for _, name := range labelNames(issue) {
		if name == label {
			return true
		}
	}
	return false
}

func IsEpic(issue Issue) bool {
	for _, name := range labelNames(issue) {
		if name == "epic" {
			return true
		}
	}
	return epicTitleRE.MatchString(issue.Title)
}

func IsTriageOnly(issue Issue) bool {
	if triageOnlyLabel(issue) != "" {
		return true
	}
	if nonDispatchWorkUnit(issueWorkUnit(issue)) {
		return true
	}
	if oversizedWorkUnit(issue) {
		return true
	}
	if bodyTriageOnly(issue) {
		return true
	}
	return false
}

func classifySkippedIssue(issue Issue, blockedLabel string) SkippedIssue {
	workUnit := issueWorkUnit(issue)
	expectedSteps := issueExpectedSteps(issue)
	review := issueContractReview(issue)
	reason := "ISSUE_NOT_DISPATCHABLE"
	next := "add dispatch scope or remove the skip condition before sending this issue to a worker"
	triageLabel := triageOnlyLabel(issue)
	switch {
	case IsBlockedByHuman(issue, blockedLabel):
		reason = "BLOCKED_BY_HUMAN"
		next = "wait for the human blocker to clear before worker dispatch"
	case IsEpic(issue):
		reason = "ISSUE_NOT_DISPATCH_LEAF"
		next = "decompose the epic into path-scoped leaf issues before dispatch"
	case triageLabel == "needs-scope":
		reason = "ISSUE_SCOPE_INCOMPLETE"
		next = "add working spine, path hints, done condition, witness, and work-unit metadata"
	case triageLabel != "":
		reason = "ISSUE_TRIAGE_ONLY"
		next = "triage or scope the issue into one or more worker-ready leaves"
	case nonDispatchWorkUnit(workUnit):
		reason = "ISSUE_NOT_DISPATCH_LEAF"
		next = "split the non-leaf work unit into worker-ready leaf issues"
	case expectedSteps > MaxDispatchExpectedSteps:
		reason = "ISSUE_OVERSIZED_EXPECTED_STEPS"
		next = fmt.Sprintf("split into child issues with <= %d expected steps each", MaxDispatchExpectedSteps)
	case bodyTriageOnly(issue):
		reason = "ISSUE_TRIAGE_ONLY"
		next = "triage or scope the issue into one or more worker-ready leaves"
	case !review.OK:
		reason = firstIssueContractReason(review)
		next = issueContractNextAction(reason)
	}
	return SkippedIssue{
		Number:        issue.Number,
		Title:         truncateRunes(issue.Title, 80),
		Reason:        reason,
		NextAction:    next,
		WorkUnit:      workUnit,
		ExpectedSteps: expectedSteps,
	}
}

func triageOnlyLabel(issue Issue) string {
	for _, name := range labelNames(issue) {
		if TriageOnlyLabels[name] {
			return name
		}
	}
	return ""
}

func bodyTriageOnly(issue Issue) bool {
	text := strings.ToLower(issue.Title + "\n" + issue.Body)
	return strings.Contains(text, "dispatchability") && strings.Contains(text, "triage_only")
}

func issueWorkUnit(issue Issue) string {
	sections := promptMarkdownSections(issue.Body)
	value := firstPromptSection(sections, "work unit", "work-unit shape", "issue shape")
	return promptBriefValue(value)
}

func issueBriefField(issue Issue, names ...string) string {
	sections := promptMarkdownSections(issue.Body)
	return promptBriefValue(firstPromptSection(sections, names...))
}

func nonDispatchWorkUnit(unit string) bool {
	unit = strings.ToLower(strings.TrimSpace(unit))
	unit = strings.Trim(unit, "`*_:. ")
	switch unit {
	case "epic", "program", "research", "idea", "triage", "triage-only", "triage_only", "decompose", "umbrella":
		return true
	default:
		return false
	}
}

func oversizedWorkUnit(issue Issue) bool {
	return issueExpectedSteps(issue) > MaxDispatchExpectedSteps
}

func issueExpectedSteps(issue Issue) int {
	sections := promptMarkdownSections(issue.Body)
	value := firstPromptSection(sections, "expected steps", "step budget")
	return parseIssueStepCount(value)
}

func parseIssueStepCount(section string) int {
	for _, tok := range strings.Fields(strings.TrimSpace(section)) {
		tok = strings.Trim(tok, "`.,;:()[]")
		if n, err := strconv.Atoi(tok); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

func routeStepBudget(r IssueRoute) int {
	if r.ExpectedSteps > 0 {
		return r.ExpectedSteps
	}
	return 1
}

func IsDispatchable(issue Issue, blockedLabel string) bool {
	return !IsBlockedByHuman(issue, blockedLabel) &&
		!IsEpic(issue) &&
		!IsTriageOnly(issue) &&
		issueContractReview(issue).OK
}

func issueContractReview(issue Issue) issuecontract.Review {
	labels := make([]issuecontract.IssueLabel, 0, len(issue.Labels))
	for _, label := range issue.Labels {
		labels = append(labels, issuecontract.IssueLabel{Name: label.Name})
	}
	return issuecontract.ReviewIssueDraft(issuecontract.IssueDraft{
		Number: issue.Number,
		Title:  issue.Title,
		Body:   issue.Body,
		Labels: labels,
	}, issuecontract.Options{})
}

func firstIssueContractReason(review issuecontract.Review) string {
	if len(review.Reasons) == 0 {
		return "ISSUE_NOT_DISPATCHABLE"
	}
	return review.Reasons[0]
}

func issueContractNextAction(reason string) string {
	switch reason {
	case issuecontract.ReasonScopeIncomplete:
		return "add in-scope, out-of-scope, done condition, witness, and acceptance gate before dispatch"
	case issuecontract.ReasonUnrouted:
		return "add a lane or path hints section that maps to a dispatch lane"
	case issuecontract.ReasonPrivateBoundary:
		return "remove or redact private/operator-only evidence before public worker dispatch"
	case issuecontract.ReasonNotDispatchLeaf:
		return "split the non-leaf work unit into worker-ready leaf issues"
	case issuecontract.ReasonOversizedSteps:
		return fmt.Sprintf("split into child issues with <= %d expected steps each", MaxDispatchExpectedSteps)
	default:
		return "scope the issue until the shared issue contract marks it dispatchable"
	}
}

func route(issue Issue, lane, confidence, signal string, conflict bool, paths []string, unroutedReason string) IssueRoute {
	return IssueRoute{
		Number:         issue.Number,
		Title:          truncateRunes(issue.Title, 80),
		Lane:           lane,
		Confidence:     confidence,
		Signal:         signal,
		SignalConflict: conflict,
		Paths:          normalizeRepoPaths(paths),
		WorkUnit:       issueWorkUnit(issue),
		ExpectedSteps:  issueExpectedSteps(issue),
		Trigger:        issueBriefField(issue, "trigger", "creation trigger"),
		BatchPolicy:    issueBriefField(issue, "batch policy", "noise control", "spam control"),
		UnroutedReason: unroutedReason,
		Generation:     generationField(issue),
	}
}

// generationField reports issue's classified generation bucket, or "" when the issue
// carries none of the gen/now, gen/next, gen/second-next, gen/future labels -- so the
// json tag's omitempty keeps an ordinary, unlabeled issue's route payload unchanged.
func generationField(issue Issue) string {
	bucket := GenerationBucket(labelNames(issue))
	if bucket == GenUnclassified {
		return ""
	}
	return bucket
}

func normalizeRepoPaths(paths []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, p := range paths {
		p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
		p = strings.TrimPrefix(p, "./")
		p = strings.TrimPrefix(p, "fak/")
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

func globToRegexp(glob string) *regexp.Regexp {
	g := strings.ReplaceAll(glob, "\\", "/")
	var b strings.Builder
	b.WriteByte('^')
	for i := 0; i < len(g); {
		switch {
		case strings.HasPrefix(g[i:], "**/"):
			b.WriteString("(?:.*/)?")
			i += 3
		case strings.HasPrefix(g[i:], "**"):
			b.WriteString(".*")
			i += 2
		case g[i] == '*':
			b.WriteString("[^/]*")
			i++
		default:
			b.WriteString(regexp.QuoteMeta(g[i : i+1]))
			i++
		}
	}
	b.WriteByte('$')
	return regexp.MustCompile(b.String())
}

func scopeToken(title string) string {
	if m := scopeRE.FindStringSubmatch(title); m != nil {
		return strings.ToLower(strings.TrimSpace(m[2]))
	}
	if m := barePrefixRE.FindStringSubmatch(title); m != nil {
		return strings.ToLower(strings.TrimSpace(m[1]))
	}
	return ""
}

func typeToken(title string) string {
	if m := scopeRE.FindStringSubmatch(title); m != nil {
		return strings.ToLower(strings.TrimSpace(m[1]))
	}
	return ""
}

func labelNames(issue Issue) []string {
	set := map[string]bool{}
	for _, label := range issue.Labels {
		name := strings.TrimSpace(label.Name)
		if name != "" {
			set[name] = true
		}
	}
	out := make([]string, 0, len(set))
	for name := range set {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func HasKeyword(text, keyword string) bool {
	keyword = strings.ToLower(strings.TrimSpace(keyword))
	if keyword == "" {
		return false
	}
	lower := strings.ToLower(text)
	start := 0
	for {
		idx := strings.Index(lower[start:], keyword)
		if idx < 0 {
			return false
		}
		pos := start + idx
		beforeOK := pos == 0 || !isKeywordRune(rune(lower[pos-1]))
		after := pos + len(keyword)
		afterOK := after >= len(lower) || !isKeywordRune(rune(lower[after]))
		if beforeOK && afterOK {
			return true
		}
		start = pos + len(keyword)
	}
}

func routeSortLess(a, b IssueRoute) bool {
	aUnrouted := a.Lane == ""
	bUnrouted := b.Lane == ""
	if aUnrouted != bUnrouted {
		return aUnrouted
	}
	if ConfidenceRank[a.Confidence] != ConfidenceRank[b.Confidence] {
		return ConfidenceRank[a.Confidence] > ConfidenceRank[b.Confidence]
	}
	return a.Number > b.Number
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedKeysSliceMap(m map[string][]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want && v != "" {
			return true
		}
	}
	return false
}

func firstNonEmptyString(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func truncateRunes(s string, n int) string {
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n])
}

func isWord(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}

func isKeywordRune(r rune) bool {
	return r == '_' || r == '-' || unicode.IsLetter(r) || unicode.IsDigit(r)
}
