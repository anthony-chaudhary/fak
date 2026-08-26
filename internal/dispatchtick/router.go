package dispatchtick

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/maputil"
)

const RouterSchema = "fleet-issue-lane-router/1"

const BlockedByHumanLabel = "blocked-by-human"
const MaxDispatchExpectedSteps = 8
const ReasonDuplicateRisk = "ISSUE_DUPLICATE_RISK"

type dispatchContractReview struct {
	OK      bool
	Reasons []string
}

const (
	reasonScopeIncomplete = "ISSUE_SCOPE_INCOMPLETE"
	reasonUnrouted        = "ISSUE_UNROUTED"
	reasonPrivateBoundary = "ISSUE_PRIVATE_BOUNDARY"
	reasonNotDispatchLeaf = "ISSUE_NOT_DISPATCH_LEAF"
	reasonOversizedSteps  = "ISSUE_OVERSIZED_EXPECTED_STEPS"
)

var dispatchPrivateCues = []string{
	"fak-private", "private control bridge", "docs/private-comms-channel.md",
	"oauth token", "api key", "secret key", "private key", "bearer token",
}

func dispatchPrivateBoundary(body string) bool {
	body = strings.ToLower(body)
	for _, cue := range dispatchPrivateCues {
		if strings.Contains(body, cue) {
			return true
		}
	}
	return false
}

// ReasonBlockedByHuman is the skip reason a SkippedIssue carries when it was held back from
// dispatch because it wears the blocked-by-human label — the subset a human must unblock. It
// is exported so a consumer (e.g. `fak dispatch skipped`) can select exactly the human-blocked
// rows out of the router's full skipped set without re-deriving the string.
const ReasonBlockedByHuman = "BLOCKED_BY_HUMAN"

// ReasonHumanBlockUnverified marks an issue that carries the blocked-by-human
// label but does NOT witness the block: no structured blocker section naming a
// genuine external/human-only action. The bare label is the difficulty dodge the
// BlockedByHumanLabel comment warns against, so it is not parked in the true
// human-blocked bucket -- it is routed to a fresh-context decision ("decide"
// repair queue) that confirms a real blocker or drops the label. This hardening
// keeps the witnessed BLOCKED_BY_HUMAN skip count near zero.
const ReasonHumanBlockUnverified = "ISSUE_HUMAN_BLOCK_UNVERIFIED"

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
	"cuda":   "compute",
	"gpu":    "compute",
	"vulkan": "compute",
	"metal":  "compute",
	// "kvbm" is the Dynamo-KVBM milestone metaphor, not a lane: no internal/kvbm/
	// directory exists, and the KV eviction/tiering primitives live in
	// internal/compute (kvcost.go, kvresidency.go, kvprecision.go). Aliasing keeps
	// a feat(kvbm) title from falling through to a phantom subsystem tag; R2
	// wiring issues that name internal/radixkv or internal/modelengine files still
	// route there via the stronger path rung (#3415).
	"kvbm":             "compute",
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
	"path-confirmed": 6,
	"exact-scope":    5,
	"alias":          4,
	"label":          3,
	"keyword":        2,
	// lane-hint is the weakest positive rung: a handoff author's declared lane read
	// from the issue body's "## Lane" section, honored only when scope, path, label,
	// and keyword all fail to route (see RouteIssue / #1854).
	"lane-hint": 1,
	"none":      0,
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
	// BlockedBy is the issue-number prerequisites this issue declares in its body via a
	// "depends-on:/blocked-by: #N" marker (CandidateBlockedBy), as bare-numeric IDs. omitempty
	// keeps an edge-free issue's route payload byte-identical to before the field existed. A live
	// dependency soft-hold (holdOpenPrereqForRoute) reads this to keep a leaf out of the pick
	// while a prerequisite it names is still an open candidate this tick.
	BlockedBy []string `json:"blocked_by,omitempty"`
	// Category/Layer are explicit category-baseline coordinates from matching issue-body
	// sections. They are scheduling data only; absent metadata keeps legacy work unchanged.
	Category string `json:"category,omitempty"`
	Layer    string `json:"layer,omitempty"`
	// Body is the issue's full markdown body, carried through the route so the picker can
	// build a worker prompt from the already-fetched row instead of a second `gh issue
	// view` on the hot dispatch path (#4167). omitempty keeps a body-less issue's route
	// payload byte-identical to before the field existed.
	Body string `json:"body,omitempty"`
	// Labels is the issue's label-name set (sorted, deduped via labelNames), carried
	// alongside Body so the cached prompt path has the labels the second fetch supplied
	// (#4167). omitempty keeps an unlabeled issue's route payload unchanged.
	Labels []string `json:"labels,omitempty"`
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

type UnroutableBacklogRow struct {
	Number     int    `json:"number"`
	Title      string `json:"title,omitempty"`
	Bucket     string `json:"bucket"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
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
	View                string                     `json:"view,omitempty"`
	ViewQuery           string                     `json:"view_query,omitempty"`
	ViewDigest          string                     `json:"view_digest,omitempty"`
	ViewFallback        bool                       `json:"view_fallback,omitempty"`
	ViewFallbackReason  string                     `json:"view_fallback_reason,omitempty"`
	Coverage            RouterCoverage             `json:"coverage"`
	Counts              RouterCounts               `json:"counts"`
	Lanes               map[string]RouterLaneGroup `json:"lanes"`
	Issues              []IssueRoute               `json:"issues"`
	RepairQueues        []RouterRepairQueue        `json:"repair_queues,omitempty"`
	UnroutableBacklog   []UnroutableBacklogRow     `json:"unroutable_backlog,omitempty"`
	SkippedHumanBlocked []SkippedIssue             `json:"skipped_human_blocked"`
	// NewlyUnblocked names open issues whose declared prerequisites were held on the prior
	// routing pass and are absent now. The command shell derives this one-pass transition
	// from durable state; the pure router only carries it to ordering and operator output.
	NewlyUnblocked []int `json:"newly_unblocked,omitempty"`
	// PrereqHeldCount is the number of dependency-held units observed before the command
	// shell removed them from routable lanes. It keeps durable queue consumers from bypassing
	// dependency reconciliation while any transition may still be pending.
	PrereqHeldCount int `json:"prereq_held_count,omitempty"`
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
	// DuplicateRiskCache optionally memoizes the O(n^2) duplicate-risk scan
	// across calls, keyed on a content hash of the routable backlog (#4171).
	// Nil (the default for every existing caller) means always recompute --
	// identical behavior to before the field existed.
	DuplicateRiskCache *DuplicateRiskCache
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
	duplicateRisk := in.DuplicateRiskCache.Risk(routable)
	duplicateSkipped := []Issue{}
	dispatchable := []Issue{}
	for _, issue := range routable {
		if duplicateRisk[issue.Number] {
			duplicateSkipped = append(duplicateSkipped, issue)
			continue
		}
		dispatchable = append(dispatchable, issue)
	}

	fetchError := strings.TrimSpace(in.FetchError)
	if fetchError == "" && len(in.Taxonomy.Concurrent) == 0 {
		fetchError = "dos doctor returned no lanes - run from the repo root"
	} else if fetchError == "" && len(in.Issues) == 0 && !in.Injected {
		fetchError = "gh returned no open issues (auth/network?)"
	}

	routes := make([]IssueRoute, 0, len(dispatchable))
	priority := map[int]int{}
	for _, issue := range dispatchable {
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
		SkippedDuplicate: duplicateSkipped,
		BlockedLabelName: blockedLabel,
	})
}

type RouteOptions struct {
	ScopeAlias   map[string]string
	LabelAlias   map[string]string
	KeywordAlias map[string]string
}

// RouteIssue picks the dos.toml lane an issue belongs to, then applies the
// routing-time trust-critical mis-route hold (#3122). The ladder itself is
// routeIssueByLadder; this wrapper is the single seam every caller already goes
// through, so a mis-route is held ONCE, where the lane is first decided, rather
// than re-discovered by each downstream consumer.
func RouteIssue(issue Issue, taxonomy LaneTaxonomy, opts RouteOptions) IssueRoute {
	return holdTrustCriticalMisroute(routeIssueByLadder(issue, taxonomy, opts), issue, taxonomy)
}

// holdTrustCriticalMisroute is the ROUTING-time arm of the self-modify hold (#3122):
// an issue whose TEXT targets fak's trust-critical witness machinery, but which the
// ladder aimed at a lane that is NOT that machinery, is held here instead of being
// handed to a guarded worker that can never ship it.
//
// It is the ROOT-CAUSE twin of the pick-time SelfModifyHoldForPick, which stays in
// place as defense-in-depth (#3122's acceptance says not to remove it). Without this
// rung the mis-routed issue sits at a SHIPPABLE lane's front: the picker re-reads its
// text and re-refuses it every tick, forever, while `fak dispatch route --json`
// reports it as routed-and-ready. Holding at routing time makes the picker a cheap
// fast-path and makes the mis-route visible to an operator.
//
// The live mis-routes this catches are the ones the lane tree HIDES: an issue naming
// internal/abi/** (whose lane is exclusive, so the path rung yields no concurrent
// lane) falling through to a tools/docs scope alias, and a multi-path issue whose
// ambiguity tie-break prefers the shippable lane over the trust-critical one.
//
// Two deliberate non-holds keep the rung from over-firing:
//
//   - A lane whose OWN tree is trust-critical (a correctly-routed policy / kernel /
//     architest issue) is left routed: the lane tree already reveals the hazard, so
//     the pick-time lane-tree arm holds it with the better witness and the lane
//     attribution operators rely on survives.
//   - An already-unrouted row (an exclusive-scope hold or a plain no-signal miss) is
//     returned untouched, so this rung never overwrites a stronger verdict.
//
// This mirrors tools/issue_lane_router.py's _hold_trust_critical_misroute, so the two
// routers hold the same issues for the same reason.
func holdTrustCriticalMisroute(routed IssueRoute, issue Issue, taxonomy LaneTaxonomy) IssueRoute {
	if routed.Lane == "" {
		return routed
	}
	if correctlyRouted, _ := SelfModifyHold(true, taxonomy.Trees[routed.Lane]); correctlyRouted {
		return routed
	}
	targets, tree := IssueTextTargetsTrustCritical(issue.Title + "\n" + issue.Body)
	if !targets {
		return routed
	}
	return route(issue, "", "none",
		TrustCriticalMisrouteSignalPrefix+tree+" (held from "+routed.Lane+")",
		false, routed.Paths,
		fmt.Sprintf("issue text targets '%s' (fak's own witness machinery, which a guarded "+
			"worker may never ship) but the routing ladder chose the shippable lane '%s'; "+
			"held before spawn", tree, routed.Lane))
}

// TrustCriticalMisrouteSignalPrefix marks a route held by holdTrustCriticalMisroute.
// The payload builder keys the operator-facing unroutable bucket off this prefix, so
// the hold is reported as its own class rather than folded into the generic
// "no lane signal" bucket, whose next-action ("add a scope/label/path hint") would be
// actively wrong advice for an issue that routed fine and was held on purpose.
const TrustCriticalMisrouteSignalPrefix = "trust-critical-text:"

// routeIssueByLadder picks the dos.toml lane an issue belongs to via a confidence ladder,
// strongest signal first: repo paths named in the body (path-confirmed) > a
// Conventional-Commit title scope that is or aliases a lane (exact-scope / alias) >
// an aliasable label > an explicit keyword > the "## Lane" section a `fak task
// handoff` issue body carries (lane-hint, the weakest rung). The lane-hint rung
// consumes internal/taskmgr.HandoffNextStep.Lane, which the handoff renders into the
// issue body: a handoff author's declared lane is honored only as a last resort,
// after every independently re-derived signal has failed, so a mislabeled Lane never
// overrides a real scope/path (the cross-check #1854 asked for) yet a plain-English
// handoff step with a correct Lane no longer routes UNROUTED.
func routeIssueByLadder(issue Issue, taxonomy LaneTaxonomy, opts RouteOptions) IssueRoute {
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

	// Weakest rung: a lane named in the body's "## Lane" section (a `fak task handoff`
	// step's declared Lane). Only a real, non-exclusive concurrent lane is accepted;
	// an exclusive lane stays operator-gated and a garbage/fallback value falls through
	// to UNROUTED.
	laneHintLane := ""
	if hint := issueLaneHint(body); hint != "" && laneSet[hint] && !ExclusiveRouterLanes[hint] {
		laneHintLane = hint
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
	if laneHintLane != "" {
		return route(issue, laneHintLane, "lane-hint", "lane-hint:"+laneHintLane, false, nil, "")
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

// handoffLaneNotSpecified is the fallback text internal/taskmgr.HandoffIssueBody writes
// into the "## Lane" section when a handoff step named no lane; it must never route.
const handoffLaneNotSpecified = "Not specified by this handoff."

// issueLaneHint returns the bare lane token named in the body's "## Lane" section (the
// lane a `fak task handoff` step declared via HandoffNextStep.Lane), or "" when the
// section is absent or holds the "not specified" fallback. The rendered section is a
// bare token, so the first whitespace-delimited word (lowercased, backtick/emphasis
// stripped) is the hint; the caller still validates it against the live lane set.
func issueLaneHint(body string) string {
	value := firstPromptSection(promptMarkdownSections(body), "lane")
	if value == "" || strings.EqualFold(value, handoffLaneNotSpecified) {
		return ""
	}
	fields := strings.Fields(value)
	if len(fields) == 0 {
		return ""
	}
	return strings.ToLower(strings.Trim(fields[0], "`*_"))
}

func PathMatchesLane(path string, trees map[string][]string) []string {
	p := strings.ReplaceAll(path, "\\", "/")
	if strings.HasPrefix(p, "./") {
		p = p[2:]
	}
	if strings.HasPrefix(p, "fak/") {
		p = strings.TrimPrefix(p, "fak/")
	}
	lanes := sortedKeys(trees)
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

func DuplicateRiskIssueNumbers(issues []Issue) map[int]bool {
	out := map[int]bool{}
	for i := 0; i < len(issues); i++ {
		for j := i + 1; j < len(issues); j++ {
			if duplicateRiskPair(issues[i], issues[j]) {
				out[issues[i].Number] = true
				out[issues[j].Number] = true
			}
		}
	}
	return out
}

func duplicateRiskPair(a, b Issue) bool {
	prefix := duplicateTitlePrefix(a.Title)
	if prefix == "" || prefix != duplicateTitlePrefix(b.Title) {
		return false
	}
	if !issuePathsOverlap(a, b) {
		return false
	}
	return issueBodySimilarity(a.Body, b.Body) >= 0.35
}

func duplicateTitlePrefix(title string) string {
	scope := normalizeDuplicateToken(scopeToken(title))
	if idx := strings.Index(title, ":"); idx > 0 {
		subject := duplicateSubjectPrefix(title[idx+1:], 2)
		if scope != "" && subject != "" {
			return scope + "/" + subject
		}
		if subject != "" {
			return normalizeDuplicateToken(title[:idx]) + "/" + subject
		}
		return normalizeDuplicateToken(title[:idx])
	}
	return scope
}

func normalizeDuplicateToken(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		case r == '-' || r == '_' || r == ' ' || r == '/':
			if !lastDash && b.Len() > 0 {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

func duplicateSubjectPrefix(s string, n int) string {
	words := []string{}
	var b strings.Builder
	flush := func() {
		if b.Len() >= 3 {
			words = append(words, b.String())
		}
		b.Reset()
	}
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			flush()
		}
		if len(words) >= n {
			break
		}
	}
	if len(words) < n {
		flush()
	}
	if len(words) == 0 {
		return ""
	}
	if len(words) > n {
		words = words[:n]
	}
	return strings.Join(words, "-")
}

func issuePathsOverlap(a, b Issue) bool {
	left := normalizeRepoPaths(ExtractIssueRepoPaths(a.Title, a.Body))
	right := normalizeRepoPaths(ExtractIssueRepoPaths(b.Title, b.Body))
	for _, lp := range left {
		for _, rp := range right {
			if duplicatePathOverlaps(lp, rp) {
				return true
			}
		}
	}
	return false
}

func duplicatePathOverlaps(a, b string) bool {
	a = strings.Trim(strings.TrimSuffix(strings.ReplaceAll(a, "\\", "/"), "/**"), "/")
	b = strings.Trim(strings.TrimSuffix(strings.ReplaceAll(b, "\\", "/"), "/**"), "/")
	if a == "" || b == "" {
		return false
	}
	if a == b {
		return true
	}
	if duplicatePathIsDir(a) && strings.HasPrefix(b, a+"/") {
		return true
	}
	if duplicatePathIsDir(b) && strings.HasPrefix(a, b+"/") {
		return true
	}
	return false
}

func duplicatePathIsDir(path string) bool {
	base := path
	if idx := strings.LastIndex(base, "/"); idx >= 0 {
		base = base[idx+1:]
	}
	return !strings.Contains(base, ".")
}

func issueBodySimilarity(a, b string) float64 {
	left := duplicateBodyTokens(a)
	right := duplicateBodyTokens(b)
	if len(left) == 0 || len(right) == 0 {
		return 0
	}
	intersection := 0
	for tok := range left {
		if right[tok] {
			intersection++
		}
	}
	union := len(left) + len(right) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func duplicateBodyTokens(body string) map[string]bool {
	out := map[string]bool{}
	var b strings.Builder
	flush := func() {
		if b.Len() < 3 {
			b.Reset()
			return
		}
		out[b.String()] = true
		b.Reset()
	}
	for _, r := range strings.ToLower(body) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			flush()
		}
	}
	flush()
	return out
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
	if multiDoneConditionIssue(issue) {
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
	review := reviewDispatchContract(issue)
	reason := "ISSUE_NOT_DISPATCHABLE"
	next := "add dispatch scope or remove the skip condition before sending this issue to a worker"
	triageLabel := triageOnlyLabel(issue)
	switch {
	case IsBlockedByHuman(issue, blockedLabel):
		if humanBlockVerified(issue) {
			// Witnessed: the block names a genuine external/human-only action. This
			// is the hardened, narrow bucket -- the only issues a worker truly cannot
			// close, kept visible for a human.
			reason = ReasonBlockedByHuman
			next = "wait for the named external/human blocker to clear before worker dispatch"
		} else {
			// Bare or agent-clearable label: do NOT park it in the human-blocked
			// bucket. Hand it to a fresh-context decision (meta issue) that confirms a
			// real blocker or drops the label so a worker can pick it up.
			reason = ReasonHumanBlockUnverified
			next = "spawn a fresh-context decision (meta issue): confirm a genuine external/human blocker in a '## Human blocker' section, or drop the blocked-by-human label so a worker can dispatch it"
		}
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
	case multiDoneConditionIssue(issue):
		reason = "ISSUE_NOT_DISPATCH_LEAF"
		next = "split the multiple done conditions into worker-ready leaf issues"
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

// humanBlockerSectionHeadings name the issue-body sections whose content, when it
// names a genuine external/human-only action, WITNESSES a blocked-by-human claim.
// The evidence must live in a dedicated section: a witness scattered anywhere in a
// long body is exactly the unverifiable dodge this hardening rejects.
var humanBlockerSectionHeadings = []string{
	"human blocker", "human blockers", "human dependency", "human-owned action",
	"external blocker", "external dependency", "blocked by", "blocked-by",
	"waiting on", "waiting for", "external action", "operator action",
}

// humanGatedActionCues name actions no agent can perform inside the repo -- only a
// human or an external party can complete them. A blocked-by-human claim is
// witnessed only when its blocker section names one of these; anything an agent
// could land itself (edit a file, run a gate, open a PR) is not a human block.
var humanGatedActionCues = []string{
	// legal / brand / contract
	"trademark", "legal", "licens", "patent", "contract", "nda",
	"sign-off", "signoff", "signature", "counsel", "attorney", "filing", "file a ",
	// money / procurement
	"payment", "purchase", "invoice", "billing", "procure", "budget",
	"credit card", "wire transfer", "reimburse", "subscription",
	// credentials / access an agent cannot mint
	"credential", "secret", "api key", "api-key", "access token", "oauth app",
	"account creation", "provision access", "grant access", "permission grant",
	"2fa", "mfa", "sso", "registrar", "dns record", "domain registration",
	// physical / infra owned by a human
	"physical", "hardware", "datacenter", "data center", "on-site", "on site",
	"ship the device", "power cycle",
	// third-party / vendor / upstream / leadership human decision
	"vendor", "third party", "third-party", "upstream maintainer", "upstream fix",
	"support ticket", "sla response", "external review", "human review",
	"leadership", "policy sign-off", "executive", "stakeholder decision",
	"government", "regulatory", "compliance approval",
}

// humanBlockVerified reports whether an issue carrying the blocked-by-human label
// also WITNESSES the block: a dedicated blocker section that names a genuine
// external/human-only action. A bare label, or a "blocker" an agent could clear
// itself, is not verified -- the caller routes it to a fresh-context decision
// instead of parking it in the human-blocked bucket forever.
func humanBlockVerified(issue Issue) bool {
	section, ok := humanBlockerSection(issue.Body)
	if !ok {
		return false
	}
	return namesHumanGatedAction(section)
}

func humanBlockerSection(body string) (string, bool) {
	sections := promptMarkdownSections(body)
	for _, name := range humanBlockerSectionHeadings {
		if value := strings.TrimSpace(sections[normalizePromptHeading(name)]); value != "" && !promptPlaceholder(value) {
			return value, true
		}
	}
	return "", false
}

func namesHumanGatedAction(text string) bool {
	lower := strings.ToLower(text)
	for _, cue := range humanGatedActionCues {
		if strings.Contains(lower, cue) {
			return true
		}
	}
	return false
}

func classifyDuplicateRiskIssue(issue Issue) SkippedIssue {
	return SkippedIssue{
		Number:        issue.Number,
		Title:         truncateRunes(issue.Title, 80),
		Reason:        ReasonDuplicateRisk,
		NextAction:    "dedupe this duplicate-risk issue before spawning a worker",
		WorkUnit:      issueWorkUnit(issue),
		ExpectedSteps: issueExpectedSteps(issue),
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

func multiDoneConditionIssue(issue Issue) bool {
	sections := promptMarkdownSections(issue.Body)
	return doneConditionItemCount(firstPromptSection(sections, "done condition", "done conditions")) > 1
}

func doneConditionItemCount(section string) int {
	listItems := 0
	plainLines := 0
	for _, raw := range strings.Split(section, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || promptPlaceholder(line) {
			continue
		}
		if isMarkdownListItem(line) {
			listItems++
			continue
		}
		plainLines++
	}
	if listItems > 0 {
		return listItems
	}
	if plainLines > 0 {
		return 1
	}
	return 0
}

func isMarkdownListItem(line string) bool {
	if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
		return true
	}
	i := 0
	for i < len(line) && line[i] >= '0' && line[i] <= '9' {
		i++
	}
	return i > 0 && i+1 < len(line) && (line[i] == '.' || line[i] == ')') && line[i+1] == ' '
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
		reviewDispatchContract(issue).OK
}

func reviewDispatchContract(issue Issue) dispatchContractReview {
	sections := promptMarkdownSections(issue.Body)
	has := func(names ...string) bool {
		return strings.TrimSpace(firstPromptSection(sections, names...)) != ""
	}
	hasScope := has("scope") ||
		(has("core through-line", "in scope") && has("gold-plating boundary", "out of scope"))
	missing := !has("current state") ||
		!hasScope ||
		!has("done condition", "done condition / witness") ||
		!has("witness", "done condition / witness") ||
		!has("likely files", "path hints", "paths", "files") ||
		!has("parent context", "parent ref") ||
		!has("why this is next", "why now") ||
		!has("working spine") ||
		!has("acceptance gate") ||
		!has("closure binding")
	if missing {
		return dispatchContractReview{Reasons: []string{reasonScopeIncomplete}}
	}
	if !has("lane") && !has("likely files", "path hints", "paths", "files") {
		return dispatchContractReview{Reasons: []string{reasonUnrouted}}
	}
	if dispatchPrivateBoundary(issue.Body) {
		return dispatchContractReview{Reasons: []string{reasonPrivateBoundary}}
	}
	return dispatchContractReview{OK: true}
}
func firstIssueContractReason(review dispatchContractReview) string {
	if len(review.Reasons) == 0 {
		return "ISSUE_NOT_DISPATCHABLE"
	}
	return review.Reasons[0]
}

func issueContractNextAction(reason string) string {
	switch reason {
	case reasonScopeIncomplete:
		return "add current state, scope, done condition, witness, likely files, and acceptance gate before dispatch"
	case reasonUnrouted:
		return "add a lane or path hints section that maps to a dispatch lane"
	case reasonPrivateBoundary:
		return "remove or redact private/operator-only evidence before public worker dispatch"
	case reasonNotDispatchLeaf:
		return "split the non-leaf work unit into worker-ready leaf issues"
	case reasonOversizedSteps:
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
		BlockedBy:      CandidateBlockedBy(issue.Body),
		Category:       issueBriefField(issue, "category"),
		Layer:          issueBriefField(issue, "layer", "capability layer"),
		Body:           issue.Body,
		Labels:         labelNames(issue),
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
	out := maputil.SortedKeys(set)
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

// sortedKeys returns the keys of m in ascending order, for any map value type —
// one generic helper the string-map and slice-map callers share.
func sortedKeys[V any](m map[string]V) []string {
	out := maputil.SortedKeys(m)
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
