package dispatchtick

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

const RouterSchema = "fleet-issue-lane-router/1"

const BlockedByHumanLabel = "blocked-by-human"

var (
	scopeRE      = regexp.MustCompile(`\b(\w+)\(([^)]+)\)`)
	barePrefixRE = regexp.MustCompile(`^([A-Za-z][\w-]*):\s`)
	epicTitleRE  = regexp.MustCompile(`(?i)^\s*epic\b\s*[\(:]`)
	pathRE       = regexp.MustCompile(`(?:fak/(?:internal|cmd|experiments)|tools|docs|visuals|\.(?:github|claude))/[A-Za-z0-9_./-]+`)
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
	"documentation":   "docs",
	"model":           "model",
	"model-arch":      "model",
	"loader":          "ggufload",
	"security":        "policy",
	"trust-floor":     "policy",
	"build":           "ci",
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
	Number         int    `json:"number"`
	Title          string `json:"title"`
	Lane           string `json:"lane"`
	Confidence     string `json:"confidence"`
	Signal         string `json:"signal"`
	SignalConflict bool   `json:"signal_conflict"`
	UnroutedReason string `json:"unrouted_reason,omitempty"`
}

type SkippedIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
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
	ByConfidence        map[string]int `json:"by_confidence"`
	SkippedHumanBlocked int            `json:"skipped_human_blocked"`
}

type RouterLaneGroup struct {
	Tree   []string `json:"tree"`
	Count  int      `json:"count"`
	Issues []int    `json:"issues"`
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
	for _, issue := range routable {
		routes = append(routes, RouteIssue(issue, in.Taxonomy, RouteOptions{
			ScopeAlias:   in.ScopeAlias,
			LabelAlias:   in.LabelAlias,
			KeywordAlias: in.KeywordAlias,
		}))
	}
	return BuildRouterPayload(RouterPayloadInput{
		Workspace:        in.Workspace,
		Routes:           routes,
		Trees:            in.Taxonomy.Trees,
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

	pathLanes := []string{}
	seenPathLane := map[string]bool{}
	for _, p := range ExtractRepoPaths(title + "\n" + body) {
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
		return route(issue, "", "none", "exclusive-scope:"+scope, false,
			fmt.Sprintf("exclusive-lane scope '%s'; operator-gated", scope))
	}
	if pathLane != "" {
		weaker := firstNonEmptyString(scopeLane, labelLane)
		conflict := weaker != "" && weaker != pathLane
		signal := "path:" + pathLane
		if conflict {
			signal += " (overrode " + weaker + ")"
		}
		return route(issue, pathLane, "path-confirmed", signal, conflict, "")
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
		return route(issue, pick, "path-confirmed", "path-ambiguous:"+strings.Join(pathLanes, "|"), true, "")
	}
	if scopeLane != "" {
		token := scope
		if !(laneSet[scope] || scopeAlias[scope] != "") {
			token = typ
		}
		return route(issue, scopeLane, scopeConf, "scope:"+token+"->"+scopeLane, false, "")
	}
	if labelLane != "" {
		return route(issue, labelLane, "label", "label->"+labelLane, false, "")
	}
	if keywordLane != "" {
		return route(issue, keywordLane, "keyword", "keyword:"+keyword+"->"+keywordLane, false, "")
	}
	reason := "no scope/path/label signal"
	if scope != "" {
		reason = "no scope, no repo-path, no aliasable label"
	}
	return route(issue, "", "none", "unrouted", false, reason)
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
	Workspace        string
	Routes           []IssueRoute
	Trees            map[string][]string
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
	for _, r := range in.Routes {
		byConf[r.Confidence] = byConf[r.Confidence] + 1
		if r.Lane == "" {
			continue
		}
		grp := lanes[r.Lane]
		grp.Tree = append([]string(nil), in.Trees[r.Lane]...)
		grp.Count++
		grp.Issues = append(grp.Issues, r.Number)
		lanes[r.Lane] = grp
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
	for _, issue := range in.SkippedBlocked {
		skipped = append(skipped, SkippedIssue{Number: issue.Number, Title: truncateRunes(issue.Title, 80)})
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
		blockedNote := ""
		if len(skipped) > 0 {
			blockedNote = fmt.Sprintf("; %d human-blocked skipped", len(skipped))
		}
		reason = fmt.Sprintf("%d/%d open issues routed to %d lane(s); %d UNROUTED%s", routed, total, len(lanes), unrouted, blockedNote)
	}

	issues := append([]IssueRoute(nil), in.Routes...)
	sort.Slice(issues, func(i, j int) bool { return routeSortLess(issues[i], issues[j]) })

	return RouterPayload{
		Schema:              RouterSchema,
		OK:                  ok,
		Verdict:             verdict,
		Finding:             finding,
		Reason:              reason,
		NextAction:          next,
		Workspace:           in.Workspace,
		Coverage:            coverage,
		Counts:              RouterCounts{Open: total, Routed: routed, Unrouted: unrouted, UnroutedFrac: frac, ByConfidence: byConf, SkippedHumanBlocked: len(skipped)},
		Lanes:               lanes,
		Issues:              issues,
		SkippedHumanBlocked: skipped,
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

func IsDispatchable(issue Issue, blockedLabel string) bool {
	return !IsBlockedByHuman(issue, blockedLabel) && !IsEpic(issue)
}

func route(issue Issue, lane, confidence, signal string, conflict bool, unroutedReason string) IssueRoute {
	return IssueRoute{
		Number:         issue.Number,
		Title:          truncateRunes(issue.Title, 80),
		Lane:           lane,
		Confidence:     confidence,
		Signal:         signal,
		SignalConflict: conflict,
		UnroutedReason: unroutedReason,
	}
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
