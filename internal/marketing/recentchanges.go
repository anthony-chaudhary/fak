// recentchanges.go — the human-readable recent-changes front door (#6040).
//
// The AEO feeds (aeo.go) answer "what shipped recently?" for machines: a bounded JSON/plain
// list of witnessed ships. A human asking the same question needs something else — the same
// stream grouped into themes they can scan, every line carrying the evidence behind it, and
// ONE stated boundary so a stale page announces itself instead of quietly rotting.
//
// This file is that fold. It reuses the SAME witness rung as every other marketing artifact
// (hooks.StampOf via CollectShips, then the CLAIMS.md honesty gate), so the page can never
// claim more than the referee can bind: a ship whose leaf is still [SIMULATED]/[STUB] in
// CLAIMS.md renders as landed-but-not-claimed instead of as a shipped capability, and a
// commit that only touched research/plan surfaces renders as a plan. No sentence about what
// changed is hand-maintained — the only curated facts are the theme titles and their durable
// "why it matters" lines, which describe SUBSYSTEMS (stable) rather than commits (not).
package marketing

import (
	"fmt"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// RecentChangesPath is the one page this fold writes, relative to the repo root. It lives
// under docs/ (Jekyll publishes it; docs/releases/ is excluded from the published site) so
// answer engines and search crawlers can actually reach it.
const RecentChangesPath = "docs/whats-new.md"

// RecentChangesVerb is the committed refresh path, named on the page itself so a reader who
// finds it stale knows the one command that fixes it.
const RecentChangesVerb = "fak marketing whats-new"

// Scope labels. Every rendered item carries exactly one — the page never lists a change
// without saying what kind of claim it is.
const (
	RecentScopeShipped   = "shipped"                     // stamped ship, honesty gate clean
	RecentScopeResearch  = "research/plan"               // touched only research/plan/notes surfaces
	RecentScopeUnclaimed = "landed, not claimed shipped" // CLAIMS.md still tags the leaf [SIMULATED]/[STUB]
)

// recentTheme groups subsystems (hooks.StampOf leaves) a reader thinks about together. The
// leaf lists are the curated part of this page and the only place a human edits when a new
// subsystem appears; the ITEMS under each theme are always derived from git.
//
// Matching is first-match-wins in declaration order, so a leaf may appear in exactly one
// theme (TestRecentThemesAssignEachLeafOnce binds that). Anything unmapped falls into the
// trailing catch-all, which is rendered honestly as "other maintained subsystems" rather
// than silently dropped.
type recentTheme struct {
	Title   string   // the H3 a reader scans — natural language, not a leaf name
	Matters string   // why this subsystem group matters to a user; durable across commits
	Leaves  []string // hooks.StampOf leaves folded into this theme
}

var recentThemes = []recentTheme{
	{
		Title:   "Micro-agents and micro-context",
		Matters: "These subsystems decide what a model actually sees: which context is resident, what is paged out, and how a small agent is handed only the slice it needs. Work here changes token cost and answer quality on the same hardware.",
		Leaves: []string{
			"microagent", "microcontext", "microcontextdemo", "ctxplan", "ctxmmu", "ctxresidency",
			"context pack", "headroom", "memq", "recall", "memoryread", "memvaluescore", "sessionimage",
			"promptmmu", "trajectory",
		},
	},
	{
		Title:   "Context filtering and tool integration",
		Matters: "The gateway, capability index, and tool-process seams are where another agent or client plugs into fak, and where results are screened before they reach the model. Work here changes what integrations are possible and what a tool call is allowed to return.",
		Leaves: []string{
			"gateway", "capindex", "capindexgw", "toolproc", "tool gate", "wirescreen", "egressfloor",
			"integrations", "mcp", "a2a", "skills", "promptlint", "screen",
		},
	},
	{
		Title:   "Fleet, dispatch, and account routing",
		Matters: "These subsystems run many sessions across many machines and accounts: which worker picks up which unit of work, which seat or endpoint serves it, and how a stalled or leaked lease is recovered. Work here changes throughput and operational safety of a fleet.",
		Leaves: []string{
			"dispatchtick", "dispatch", "fleetaccounts", "accounts", "modelroute", "leaseref", "dos",
			"workerworktree", "gitbroker", "launchshim", "dgxbridge", "session", "sessionaudit",
			"stallscan", "loopmgr", "lifebridge", "runaway",
		},
	},
	{
		Title:   "Model runtime, kernels, and caching",
		Matters: "The loader, compute kernels, and cache tiers are the hot path: they decide how fast a token is produced and how much of a prior context is reused instead of recomputed. Work here shows up as latency and cost, not as new surface.",
		Leaves: []string{
			"model", "modelaccept", "modelscore", "modver", "ggufload", "compute", "cachemeta",
			"cachevaluereport", "cachevalueledger", "bench", "nativebench", "livecodebench",
			"tokenizer", "storedrv", "metrics",
		},
	},
	{
		Title:   "Policy, adjudication, and audit",
		Matters: "Policy and adjudication are the boundary that blocks a poisoned result or a destructive operation, and the audit trail is what proves the boundary held. Work here changes what an agent is allowed to do and what evidence you keep afterwards.",
		Leaves: []string{
			"policy", "adjudicator", "guard", "journal", "quality", "qaprocessscore",
			"mutationefficacy", "architest", "hooks", "usagelog", "provenance",
		},
	},
	{
		Title:   "Documentation and discoverability",
		Matters: "These changes maintain the pages, indexes, and machine-readable feeds a reader or answer engine uses to find the right authority. Work here changes what is findable, not what the kernel does.",
		Leaves: []string{
			"docs", "devindex", "marketing", "seoaeoscore", "ideascout", "glossary",
		},
	},
	{
		Title:   "Command line and developer workflow",
		Matters: "The `fak` verbs and the repository's own tooling are how both humans and automated contributors drive everything above. Work here changes the commands you run, not the behavior they govern.",
		Leaves: []string{
			"cmd", "tools", "gitdaily", "wipref", "agent", "shipgate", "worktype",
		},
	},
}

// recentCatchAll is the honest destination for a stamped ship whose leaf no theme claims —
// listed as its own group instead of being dropped, so the page's counts always reconcile
// with `git log`.
var recentCatchAll = recentTheme{
	Title:   "Other maintained subsystems",
	Matters: "These are witnessed changes in subsystems no theme above claims yet. They are listed rather than dropped so the totals on this page reconcile with git history; the theme table in `internal/marketing/recentchanges.go` is where a leaf graduates into a named theme.",
}

// researchPrefixes are the repository surfaces that hold research, plans, and dated notes.
// A ship whose every touched path is under one of them is a plan, not a shipped capability.
var researchPrefixes = []string{
	"docs/research/", "docs/notes/", "docs/planning/", "docs/plans/", "experiments/", ".claude/",
}

// typeRank orders conventional-commit types by how much a READER cares: a feature or a fix
// is the news, a perf change is close behind, and test/chore/docs churn is supporting work.
// It only decides which items are SHOWN when a theme has more changes than the page budget;
// every change is still counted.
var typeRank = map[string]int{
	"feat": 0, "fix": 1, "perf": 2, "revert": 3, "refactor": 4, "docs": 5, "build": 6,
	"ci": 7, "style": 8, "test": 9, "chore": 10,
}

var (
	reConventionalType    = regexp.MustCompile(`^([a-z]+)(?:\([^)]*\))?!?:`)
	reLeafTrailer         = regexp.MustCompile(`\s*\((?:fak|dos)\s+[A-Za-z0-9._/-]+\)\s*$`)
	reIssueSuffix         = regexp.MustCompile(`\s*\(?#\d+\)?`)
	reDanglingIssueJoiner = regexp.MustCompile(`(?i)\s+(?:for|close(?:s|d)?|fix(?:es|ed)?|refs?)\s*$`)
	reRecentAnchor        = regexp.MustCompile(`(?m)^<!--\s*fak:recent-changes\s+(.*?)\s*-->\s*$`)
	reAnchorField         = regexp.MustCompile(`([a-z-]+)=(\S+)`)
)

// RecentItem is one rendered line: a witnessed commit plus the scope it may be claimed at.
type RecentItem struct {
	SHA     string    `json:"sha"`
	Date    time.Time `json:"date"`
	Subject string    `json:"subject"` // trailers/issue-refs stripped; rendered as prose
	Leaf    string    `json:"leaf"`
	Type    string    `json:"type"`  // conventional-commit type ("feat", "fix", …); "" if none
	Issue   int       `json:"issue"` // 0 when the subject names none
	Scope   string    `json:"scope"` // one of the RecentScope* labels
}

// RecentGroup is one theme with its counted and shown changes.
type RecentGroup struct {
	Title      string       `json:"title"`
	Matters    string       `json:"matters"`
	Leaves     []string     `json:"leaves"` // subsystems that actually changed in this window
	Items      []RecentItem `json:"items"`  // the shown slice (ranked), never the full set
	Total      int          `json:"total"`
	Features   int          `json:"features"`
	Fixes      int          `json:"fixes"`
	Research   int          `json:"research"`
	Unclaimed  int          `json:"unclaimed"`
	Supporting int          `json:"supporting"` // everything that is not a feat/fix
}

// RecentChangesPage is the whole page as data — Markdown() is the only renderer, so the
// same fold backs the page, the --json evidence, and the freshness check.
type RecentChangesPage struct {
	AnchorSHA       string        `json:"anchor_sha"`       // the commit this page is current through
	AnchorDate      time.Time     `json:"anchor_date"`      // that commit's author date (never wall-clock)
	RangeSpec       string        `json:"range_spec"`       // machine range, e.g. "abc123..def456"
	RangeLabel      string        `json:"range_label"`      // human label, e.g. "the 7 days ending 2026-08-09"
	Version         string        `json:"version"`          // module/app version at the anchor
	GeneratorModule string        `json:"generator_module"` // derived internal/marketing@rev at the anchor
	Ships           int           `json:"ships"`            // stamped ships in the range (all scopes)
	Commits         int           `json:"commits"`          // non-merge commits scanned
	PerTheme        int           `json:"per_theme"`        // shown-items budget per theme
	Days            int           `json:"days"`             // window in days; 0 for an explicit range
	Groups          []RecentGroup `json:"groups"`
}

// RecentChangesOptions are the caller-supplied boundaries: everything else is derived from
// git. Now is deliberately absent — the page is stamped with the ANCHOR COMMIT's date so
// two runs over the same range produce identical bytes (that is what makes drift checkable).
type RecentChangesOptions struct {
	AnchorSHA       string
	AnchorDate      time.Time
	RangeSpec       string
	RangeLabel      string
	Version         string
	GeneratorModule string
	PerTheme        int // 0 → defaultPerTheme
	Days            int // the --days window, recorded so a check can replay the same label
}

const defaultPerTheme = 4

// BuildRecentChanges folds a gathered range into the page. col.Ships are the honesty-gate
// survivors; col.Excluded are stamped ships whose leaf/issue CLAIMS.md still tags
// [SIMULATED]/[STUB] — they are kept but DOWNGRADED to RecentScopeUnclaimed, never silently
// promoted and never silently dropped (dropping them would make the page disagree with git).
func BuildRecentChanges(col Collected, opt RecentChangesOptions) RecentChangesPage {
	perTheme := opt.PerTheme
	if perTheme <= 0 {
		perTheme = defaultPerTheme
	}
	items := make([]RecentItem, 0, len(col.Ships)+len(col.Excluded))
	for _, s := range col.Ships {
		items = append(items, recentItemOf(s, scopeOfShip(s, false)))
	}
	for _, e := range col.Excluded {
		items = append(items, recentItemOf(e.Ship, scopeOfShip(e.Ship, true)))
	}
	page := RecentChangesPage{
		AnchorSHA:       opt.AnchorSHA,
		AnchorDate:      opt.AnchorDate,
		RangeSpec:       opt.RangeSpec,
		RangeLabel:      opt.RangeLabel,
		Version:         opt.Version,
		GeneratorModule: opt.GeneratorModule,
		Ships:           len(items),
		Commits:         col.Activity.Commits,
		PerTheme:        perTheme,
		Days:            opt.Days,
		Groups:          groupRecentItems(items, perTheme),
	}
	return page
}

// recentItemOf projects one Ship onto a rendered line.
func recentItemOf(s Ship, scope string) RecentItem {
	issue := 0
	if refs := subjectIssueRefs(s.Subject); len(refs) > 0 {
		issue = refs[0]
	}
	return RecentItem{
		SHA:     s.SHA,
		Date:    s.Date,
		Subject: cleanRecentSubject(s.Subject),
		Leaf:    s.Leaf,
		Type:    conventionalType(s.Subject),
		Issue:   issue,
		Scope:   scope,
	}
}

// scopeOfShip grades one ship. The research test is path-based (what the commit actually
// touched), so a `feat(...)` subject over a research-only tree is still a plan.
func scopeOfShip(s Ship, honestyExcluded bool) string {
	if honestyExcluded {
		return RecentScopeUnclaimed
	}
	if researchOnly(s.Paths) {
		return RecentScopeResearch
	}
	return RecentScopeShipped
}

// researchOnly reports whether every touched path is a research/plan surface. An unknown
// path set (no paths recorded) is NOT research — the stamp is the stronger witness.
func researchOnly(paths []string) bool {
	if len(paths) == 0 {
		return false
	}
	for _, p := range paths {
		p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
		if p == "" {
			continue
		}
		hit := false
		for _, prefix := range researchPrefixes {
			if strings.HasPrefix(p, prefix) {
				hit = true
				break
			}
		}
		if !hit {
			return false
		}
	}
	return true
}

// conventionalType pulls the conventional-commit type off a subject ("feat(gateway): …" →
// "feat"); "" when the subject does not carry one.
func conventionalType(subject string) string {
	if m := reConventionalType.FindStringSubmatch(strings.TrimSpace(subject)); m != nil {
		return m[1]
	}
	return ""
}

// cleanRecentSubject strips the machine trailers a reader does not need inline: the
// "(fak <leaf>)" stamp (the leaf is rendered as its own field) and the trailing issue ref
// (rendered as a link). The conventional-commit prefix is KEPT — "fix(gateway):" tells a
// reader what kind of change it is at a glance.
func cleanRecentSubject(subject string) string {
	s := reLeafTrailer.ReplaceAllString(strings.TrimSpace(subject), "")
	s = strings.TrimSpace(reIssueSuffix.ReplaceAllString(s, ""))
	s = strings.TrimSpace(reDanglingIssueJoiner.ReplaceAllString(s, ""))
	return strings.TrimSpace(s)
}

// themeIndexOf returns the index of the theme claiming leaf, or -1 for the catch-all.
func themeIndexOf(leaf string) int {
	leaf = strings.ToLower(strings.TrimSpace(leaf))
	for i, t := range recentThemes {
		for _, l := range t.Leaves {
			if l == leaf {
				return i
			}
		}
	}
	return -1
}

// groupRecentItems buckets items by theme, counts every one of them, and keeps only the
// top perTheme for rendering (most user-relevant type first, then newest). Empty themes are
// omitted — a page that lists a theme with nothing under it is noise.
func groupRecentItems(items []RecentItem, perTheme int) []RecentGroup {
	buckets := make([][]RecentItem, len(recentThemes)+1)
	for _, it := range items {
		idx := themeIndexOf(it.Leaf)
		if idx < 0 {
			idx = len(recentThemes)
		}
		buckets[idx] = append(buckets[idx], it)
	}
	var out []RecentGroup
	for i, bucket := range buckets {
		if len(bucket) == 0 {
			continue
		}
		theme := recentCatchAll
		if i < len(recentThemes) {
			theme = recentThemes[i]
		}
		out = append(out, buildRecentGroup(theme, bucket, perTheme))
	}
	return out
}

func buildRecentGroup(theme recentTheme, bucket []RecentItem, perTheme int) RecentGroup {
	g := RecentGroup{Title: theme.Title, Matters: theme.Matters, Total: len(bucket)}
	leaves := map[string]bool{}
	for _, it := range bucket {
		leaves[it.Leaf] = true
		switch {
		case it.Type == "feat":
			g.Features++
		case it.Type == "fix":
			g.Fixes++
		default:
			g.Supporting++
		}
		switch it.Scope {
		case RecentScopeResearch:
			g.Research++
		case RecentScopeUnclaimed:
			g.Unclaimed++
		}
	}
	for l := range leaves {
		g.Leaves = append(g.Leaves, l)
	}
	sort.Strings(g.Leaves)
	sorted := append([]RecentItem(nil), bucket...)
	sort.SliceStable(sorted, func(i, j int) bool {
		ri, rj := itemRank(sorted[i]), itemRank(sorted[j])
		if ri != rj {
			return ri < rj
		}
		if !sorted[i].Date.Equal(sorted[j].Date) {
			return sorted[i].Date.After(sorted[j].Date)
		}
		return sorted[i].SHA < sorted[j].SHA
	})
	if len(sorted) > perTheme {
		sorted = selectRecentItems(sorted, perTheme)
	}
	g.Items = sorted
	return g
}

// selectRecentItems keeps the high-signal feature/fix ranking while reserving evidence for
// every non-shipped scope present in the group. Without the reservation, a busy theme's
// shipped features crowd every plan and honesty hold out of the visible slice, leaving the
// page to claim that it distinguishes scopes without showing a reader even one example.
func selectRecentItems(ranked []RecentItem, limit int) []RecentItem {
	if limit <= 0 || len(ranked) <= limit {
		return append([]RecentItem(nil), ranked...)
	}
	selected := append([]RecentItem(nil), ranked[:limit]...)
	for _, scope := range []string{RecentScopeResearch, RecentScopeUnclaimed} {
		if hasRecentScope(selected, scope) {
			continue
		}
		candidate := -1
		for i, it := range ranked {
			if it.Scope == scope {
				candidate = i
				break
			}
		}
		if candidate < 0 {
			continue
		}
		replace := len(selected) - 1
		for replace >= 0 {
			existingScope := selected[replace].Scope
			if existingScope != scope && (existingScope == RecentScopeShipped || countRecentScope(selected, existingScope) > 1) {
				break
			}
			replace--
		}
		if replace < 0 {
			break
		}
		selected[replace] = ranked[candidate]
	}
	sort.SliceStable(selected, func(i, j int) bool {
		ri, rj := itemRank(selected[i]), itemRank(selected[j])
		if ri != rj {
			return ri < rj
		}
		if !selected[i].Date.Equal(selected[j].Date) {
			return selected[i].Date.After(selected[j].Date)
		}
		return selected[i].SHA < selected[j].SHA
	})
	return selected
}

func countRecentScope(items []RecentItem, scope string) int {
	n := 0
	for _, it := range items {
		if it.Scope == scope {
			n++
		}
	}
	return n
}

func hasRecentScope(items []RecentItem, scope string) bool {
	for _, it := range items {
		if it.Scope == scope {
			return true
		}
	}
	return false
}

// itemRank is the show-order key: user-facing types first, and within a type a shipped
// change outranks a plan (a reader scanning a theme wants the landed news first).
func itemRank(it RecentItem) int {
	r, ok := typeRank[it.Type]
	if !ok {
		r = len(typeRank)
	}
	r *= 4
	switch it.Scope {
	case RecentScopeUnclaimed:
		r += 1
	case RecentScopeResearch:
		r += 2
	}
	return r
}

// ---------------------------------------------------------------------------
// Freshness anchor: the machine-checkable boundary the page carries.
// ---------------------------------------------------------------------------

// RecentChangesAnchor is the parsed freshness stamp of a committed page. It is what makes
// staleness DETECTABLE rather than a judgement call: the sha the page is current through,
// the exact range it folded, and the counts it rendered.
type RecentChangesAnchor struct {
	SHA             string    `json:"sha"`
	Date            time.Time `json:"date"`
	RangeSpec       string    `json:"range_spec"`
	Ships           int       `json:"ships"`
	Commits         int       `json:"commits"`
	Version         string    `json:"version"`
	GeneratorModule string    `json:"generator_module"`
	PerTheme        int       `json:"per_theme"`
	Days            int       `json:"days"` // 0 when the page was folded from an explicit --range
}

// Options rebuilds the option set that produced a committed page. The anchor carries every
// input the fold takes, so a checker can replay the exact render — that is what turns "does
// this page still match the repository?" into a byte comparison instead of an opinion.
func (a RecentChangesAnchor) Options() RecentChangesOptions {
	return RecentChangesOptions{
		AnchorSHA:       a.SHA,
		AnchorDate:      a.Date,
		RangeSpec:       a.RangeSpec,
		RangeLabel:      RecentRangeLabel(a.Days, a.RangeSpec, a.Date),
		Version:         a.Version,
		GeneratorModule: a.GeneratorModule,
		PerTheme:        a.PerTheme,
		Days:            a.Days,
	}
}

// RecentRangeLabel is the ONE place a window is put into words, so the label a page carries
// is a pure function of the anchor it records (and therefore replayable).
func RecentRangeLabel(days int, rangeSpec string, anchorDate time.Time) string {
	if days > 0 {
		unit := "days"
		if days == 1 {
			unit = "day"
		}
		return fmt.Sprintf("the %d %s of repository history ending %s", days, unit, anchorDate.Format("2006-01-02"))
	}
	return fmt.Sprintf("the revision range `%s`", rangeSpec)
}

// AnchorComment renders the one-line HTML comment the page carries. Keys are space-free so
// a single regex round-trips it; the human-readable copy lives in the page's freshness table.
func (p RecentChangesPage) AnchorComment() string {
	version := strings.TrimSpace(p.Version)
	if version == "" {
		version = "unknown"
	}
	generatorModule := strings.TrimSpace(p.GeneratorModule)
	if generatorModule == "" {
		generatorModule = "unknown"
	}
	return fmt.Sprintf("<!-- fak:recent-changes anchor=%s date=%s range=%s ships=%d commits=%d version=%s generator-module=%s per-theme=%d days=%d generator=%s -->",
		p.AnchorSHA, p.AnchorDate.Format("2006-01-02"), p.RangeSpec, p.Ships, p.Commits,
		version, generatorModule, p.PerTheme, p.Days, strings.ReplaceAll(RecentChangesVerb, " ", "-"))
}

// ParseRecentChangesAnchor reads the anchor back out of a rendered page. ok is false when
// the page carries no anchor at all (a hand-written replacement, or a truncated file) —
// which the check path reports as a defect, not as "fresh".
func ParseRecentChangesAnchor(md string) (RecentChangesAnchor, bool) {
	m := reRecentAnchor.FindStringSubmatch(md)
	if m == nil {
		return RecentChangesAnchor{}, false
	}
	var a RecentChangesAnchor
	for _, f := range reAnchorField.FindAllStringSubmatch(m[1], -1) {
		switch f[1] {
		case "anchor":
			a.SHA = f[2]
		case "date":
			if t, err := time.Parse("2006-01-02", f[2]); err == nil {
				a.Date = t
			}
		case "range":
			a.RangeSpec = f[2]
		case "ships":
			a.Ships, _ = strconv.Atoi(f[2])
		case "commits":
			a.Commits, _ = strconv.Atoi(f[2])
		case "version":
			if f[2] != "unknown" {
				a.Version = f[2]
			}
		case "generator-module":
			if f[2] != "unknown" {
				a.GeneratorModule = f[2]
			}
		case "per-theme":
			a.PerTheme, _ = strconv.Atoi(f[2])
		case "days":
			a.Days, _ = strconv.Atoi(f[2])
		}
	}
	return a, a.SHA != ""
}

// ---------------------------------------------------------------------------
// Git seams. Kept here beside CollectShips so every git call in this subsystem is in one
// package (and carries the no-popup window configuration on Windows).
// ---------------------------------------------------------------------------

// RecentAnchorCommit resolves HEAD at root to (sha, author date) — the boundary the page is
// stamped with. Using the COMMIT date, not the clock, keeps a regeneration byte-identical.
func RecentAnchorCommit(root string) (string, time.Time, error) {
	out, err := gitOut(root, "log", "-1", "--date=iso-strict", "--format=%H%x1f%ad")
	if err != nil {
		return "", time.Time{}, err
	}
	f := strings.Split(strings.TrimSpace(out), "\x1f")
	if len(f) < 2 {
		return "", time.Time{}, fmt.Errorf("unexpected git log output %q", out)
	}
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(f[1]))
	if err != nil {
		return strings.TrimSpace(f[0]), time.Time{}, nil
	}
	return strings.TrimSpace(f[0]), t, nil
}

// RecentWindowStart resolves the first commit at or before `days` before headDate — the
// start of the rendered window. Returning a SHA (not a relative date) is what lets the
// anchor pin an exact, replayable range.
func RecentWindowStart(root string, headDate time.Time, days int) (string, error) {
	before := headDate.AddDate(0, 0, -days).Format(time.RFC3339)
	out, err := gitOut(root, "rev-list", "-1", "--before="+before, "HEAD")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}

// RecentCommitsBetween counts non-merge commits in a range — the "commits behind" figure the
// freshness check reports.
func RecentCommitsBetween(root, revRange string) (int, error) {
	out, err := gitOut(root, "rev-list", "--no-merges", "--count", revRange)
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(out))
}

// RecentRepositoryVersion reads the release marker from the repository being folded. It
// deliberately does not use the running binary's version: an installed fak may refresh a
// newer checkout, and borrowing the binary's identity would stamp the wrong release onto
// the page. appversion.FromDir keeps the repository-boundary rule in one authority.
func RecentRepositoryVersion(root string) (string, bool) {
	return appversion.FromDir(root)
}

// RecentModuleVersion derives one module's identity at anchor using the same public
// r<rev>+g<shortsha> contract as `fak version modules`: non-merge commits touching the
// module, newest first. Pinning the query to anchor makes the value replayable after HEAD
// advances, so the generated page can cite its generator module without becoming stale as
// a side effect of committing the page itself.
func RecentModuleVersion(root, modulePath, anchor string) (string, error) {
	modulePath = strings.TrimSpace(strings.ReplaceAll(modulePath, "\\", "/"))
	anchor = strings.TrimSpace(anchor)
	if modulePath == "" || anchor == "" {
		return "", fmt.Errorf("module path and anchor are required")
	}
	out, err := gitOut(root, "log", "--no-merges", "--format=%h", anchor, "--", modulePath)
	if err != nil {
		return "", err
	}
	return recentModuleVersionFromLog(modulePath, anchor, out)
}

func recentModuleVersionFromLog(modulePath, anchor, out string) (string, error) {
	var commits []string
	for _, line := range strings.Split(out, "\n") {
		if sha := strings.TrimSpace(line); sha != "" {
			commits = append(commits, sha)
		}
	}
	if len(commits) == 0 {
		return "", fmt.Errorf("module %q has no committed history at %s", modulePath, anchor)
	}
	return fmt.Sprintf("%s@r%d+g%s", modulePath, len(commits), commits[0]), nil
}

func gitOut(root string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	b, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(b), nil
}
