package fleetaccounts

import (
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// classify_task / route_account / resolve_account port: the tier-aware account picker
// the dispatch front doors call. Read-only over an annotated roster.

var hardTaskHintRE = regexp.MustCompile(`(?i)\b(` +
	`implement|fix|debug|refactor|review|test|ship|complete|build|edit|write|` +
	`modify|patch|investigate|research|search|browse|audit|design|architect|` +
	`merge|rebase|deploy|security|production|goal|plan|best\s+effort` +
	`)\b`)

var lightTaskPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^(hi|hello|hey|ping|pong|thanks|thank you)[.!?\s]*$`),
	regexp.MustCompile(`(?i)^say\s+[\w .,'"-]{1,40}$`),
	regexp.MustCompile(`(?i)^reply\s+with\s+(exactly\s+)?[\w .,'"-]{1,50}$`),
	regexp.MustCompile(`(?i)^(what('| i)?s\s+)?(the\s+)?(time|date)(\s+now|\s+today)?[?]?$`),
	regexp.MustCompile(`(?i)^(pwd|whoami|date)$`),
}

var gardeningWorkKinds = map[string]bool{
	"gardening": true, "garden": true, "maintenance": true, "maint": true,
	"cleanup": true, "chore": true, "triage": true,
}
var engineeringWorkKinds = map[string]bool{
	"engineering": true, "eng": true, "dev": true, "feature": true, "implementation": true,
}

// TaskClass is the v1 routing classification of a request.
type TaskClass struct {
	Class          string  `json:"class"`
	Confidence     float64 `json:"confidence"`
	Reason         string  `json:"reason"`
	TargetTier     int     `json:"target_tier"`
	LightThreshold float64 `json:"light_threshold"`
}

// ClassifyTask classifies a request for v1 model routing.
func ClassifyTask(taskText, taskClass string, pol Policy) TaskClass {
	threshold := pol.Routing.LightConfidence
	if threshold == 0 {
		threshold = 0.999
	}
	requested := strings.ToLower(taskClass)
	if requested == "" {
		requested = "auto"
	}
	switch {
	case apexRequested(requested):
		// Apex (Fable 5) is the ONLY tier a task-text heuristic can never reach: it is
		// selected exclusively by this explicit operator request, so the restricted
		// model is used only in the clear cases it is asked for — see apextier.go.
		return TaskClass{"apex", 1.0,
			"operator explicitly requested the apex tier (Fable 5); tier 0 is never inferred",
			TierApex, threshold}
	case in(requested, "light", "easy", "tier2", "t2", "2"):
		return TaskClass{"light", 1.0, "operator requested light/tier2", TierLight, threshold}
	case in(requested, "hard", "default", "tier1", "t1", "1"):
		return TaskClass{"hard", 1.0, "operator requested hard/tier1", TierFrontier, threshold}
	case in(requested, "tier3", "t3", "3"):
		return TaskClass{"tier3", 1.0, "operator requested tier3", TierOther, threshold}
	}
	if gardeningWorkKinds[requested] {
		return TaskClass{"gardening", 1.0,
			"operator stated work_kind=" + requested + " (maintenance -> tier2)", 2, threshold}
	}
	if engineeringWorkKinds[requested] {
		return TaskClass{"engineering", 1.0,
			"operator stated work_kind=" + requested + " (engineering -> tier1)", 1, threshold}
	}
	// Task-text inference never targets apex: its ceiling is the frontier tier, so an
	// unclassified request can escalate at most to tier 1, never to the restricted
	// Fable-5 tier 0.
	text := strings.TrimSpace(wsRun.ReplaceAllString(taskText, " "))
	if text == "" {
		return TaskClass{"hard", 0.5, "no task text; defaulting to max-quality tier", TierFrontier, threshold}
	}
	if len(text) <= 80 && !hardTaskHintRE.MatchString(text) {
		for _, p := range lightTaskPatterns {
			if p.MatchString(text) {
				return TaskClass{"light", threshold,
					"short trivial prompt matched v1 light-task allowlist", TierLight, threshold}
			}
		}
	}
	return TaskClass{"hard", 1.0 - threshold,
		"not a high-confidence trivial prompt; defaulting to max-quality tier", TierFrontier, threshold}
}

func in(s string, opts ...string) bool {
	for _, o := range opts {
		if s == o {
			return true
		}
	}
	return false
}

// dispatchTaskClass applies the work-kind override shared by one-account and wave
// resolution before either delegates to the canonical task classifier.
func dispatchTaskClass(taskClass, workKind string, strictTier bool) (string, bool) {
	wk := strings.ToLower(strings.TrimSpace(workKind))
	if gardeningWorkKinds[wk] || engineeringWorkKinds[wk] {
		taskClass, strictTier = wk, false
	}
	return taskClass, strictTier
}

// routingTierOrder returns the shared target/fallback ladder and whether policy or the
// caller enabled hard-tier fallback. Apex remains explicit-only because no non-apex
// ladder can contain TierApex.
func routingTierOrder(target int, strictTier, allowTierFallback bool, pol Policy) ([]int, bool) {
	fallbackPolicy := strings.ToLower(pol.Routing.HardTier1Fallback)
	effectiveAllow := allowTierFallback || in(fallbackPolicy, "allow", "fallback", "tier2", "t2")
	tiers := []int{target}
	switch {
	case target == TierApex:
		if !strictTier {
			tiers = append(tiers, TierFrontier)
		}
	case target == TierLight && !strictTier:
		tiers = append(tiers, TierFrontier)
	case effectiveAllow:
		tiers = append(tiers, TierLight)
	}
	return tiers, effectiveAllow
}

func routeRank(r Account) (int, int, int, float64, string, string) {
	return -derefInt(r.RouteWeight),
		derefInt(r.LiveSessions),
		derefInt(r.ActiveSessions),
		derefFloat64(r.RoutingCostPerMTok),
		r.Product,
		strmatch.FirstNonEmpty(r.Tag, r.Account)
}

// BlockedAccount is the public projection of a blocked target account.
type BlockedAccount struct {
	Tag         string  `json:"tag"`
	Account     string  `json:"account"`
	Product     string  `json:"product"`
	ModelTier   *int    `json:"model_tier"`
	Model       *string `json:"model"`
	Reason      string  `json:"reason"`
	LoginStatus *string `json:"login_status,omitempty"`
	CanServe    *bool   `json:"can_serve,omitempty"`
}

func publicBlocked(r Account) BlockedAccount {
	reason := derefStr(r.BlockReason)
	if reason == "" && accountLoginBlocked(r) {
		reason = accountLoginBlockReason(r)
	}
	if reason == "" {
		reason = r.Reason
	}
	if reason == "" {
		reason = "blocked"
	}
	return BlockedAccount{
		Tag: r.Tag, Account: r.Account, Product: r.Product,
		ModelTier: r.ModelTier, Model: r.Model, Reason: reason,
		LoginStatus: r.LoginStatus, CanServe: r.CanServe,
	}
}

// RouteResult is the route_account return shape.
type RouteResult struct {
	OK                    bool             `json:"ok"`
	Reason                string           `json:"reason"`
	Task                  TaskClass        `json:"task"`
	TargetTier            int              `json:"target_tier"`
	SelectedTier          *int             `json:"selected_tier,omitempty"`
	FallbackUsed          bool             `json:"fallback_used"`
	Account               *Account         `json:"account"`
	BlockedTargetAccounts []BlockedAccount `json:"blocked_target_accounts"`
}

func tierOf(r Account) int {
	if r.ModelTier == nil {
		return 3
	}
	return *r.ModelTier
}

// atSessionCap reports whether r has no free session slot left: its registry-live
// session count has reached the per-account session budget. RouteAccount admits
// single launches through this gate so a route-weighted seat that is already full
// spills to the next candidate instead of accreting every new launch until the
// whole account limit-walls (AllocateWave enforces the same budget per pool).
func atSessionCap(r Account) bool {
	limit := AccountSessionCap(r)
	return limit > 0 && derefInt(r.LiveSessions) >= limit
}

// atCapBlocked projects an available-but-full seat into the blocked-account list
// so a route decision names the seats it refused to overfill.
func atCapBlocked(r Account) BlockedAccount {
	b := publicBlocked(r)
	b.Reason = "at session cap (" + itoa(derefInt(r.LiveSessions)) + " live >= cap " + itoa(AccountSessionCap(r)) + ")"
	return b
}

// blockedAtTiers lists the target-tier accounts a route turned away: login/availability
// blocks first, then available seats set aside only for being at their session cap.
func blockedAtTiers(workers []Account, tiers map[int]bool) []BlockedAccount {
	blocked := []BlockedAccount{}
	for _, r := range workers {
		if !tiers[tierOf(r)] {
			continue
		}
		if !accountCanBeOffered(r) {
			blocked = append(blocked, publicBlocked(r))
		} else if atSessionCap(r) {
			blocked = append(blocked, atCapBlocked(r))
		}
	}
	return blocked
}

// routableAndAvailable filters a roster to the routable worker accounts matching wantedProduct
// (empty matches any product), then to the subset that can be offered right now. Shared by
// RouteAccount and AllocateWave, which each derive wantedProduct before calling.
func routableAndAvailable(rows []Account, wantedProduct string) (workers, available []Account) {
	for _, r := range rows {
		if RoutableWorker(r) && (wantedProduct == "" || strings.ToLower(r.Product) == wantedProduct) {
			workers = append(workers, r)
		}
	}
	for _, r := range workers {
		if accountCanBeOffered(r) {
			available = append(available, r)
		}
	}
	return workers, available
}

// RouteAccount chooses an account by task difficulty and model tier over an annotated roster.
func RouteAccount(rows []Account, taskText, taskClass string, allowTierFallback, strictTier bool,
	product string, pol Policy) RouteResult {
	task := ClassifyTask(taskText, taskClass, pol)
	wantedProduct := strings.ToLower(product)
	workers, available := routableAndAvailable(rows, wantedProduct)
	if len(workers) == 0 {
		reason := "no worker accounts"
		if wantedProduct != "" {
			reason = "no worker accounts match product filter"
		}
		return RouteResult{OK: false, Reason: reason, Task: task, TargetTier: task.TargetTier,
			BlockedTargetAccounts: []BlockedAccount{}}
	}
	target := task.TargetTier
	// The fallback ladder. Crucially, no non-apex target ever lists tier 0, so a tier-0
	// (Fable 5) account is unreachable except when apex was explicitly targeted — the
	// account-side half of the apex restriction (see apextier.go).
	tierOrder, effectiveAllow := routingTierOrder(target, strictTier, allowTierFallback, pol)

	atCap := 0
	for _, tier := range tierOrder {
		var candidates []Account
		for _, r := range available {
			if tierOf(r) != tier {
				continue
			}
			if atSessionCap(r) {
				atCap++
				continue
			}
			candidates = append(candidates, r)
		}
		if len(candidates) > 0 {
			sort.SliceStable(candidates, func(i, j int) bool {
				return rankLess(candidates[i], candidates[j])
			})
			chosen := candidates[0]
			reason := "selected target tier"
			if tier != target {
				reason = "selected fallback tier"
			}
			st := tier
			return RouteResult{OK: true, Reason: reason, Task: task, TargetTier: target,
				SelectedTier: &st, FallbackUsed: tier != target, Account: &chosen,
				BlockedTargetAccounts: blockedAtTiers(workers, map[int]bool{target: true})}
		}
	}

	tierSet := map[int]bool{}
	for _, t := range tierOrder {
		tierSet[t] = true
	}
	blocked := blockedAtTiers(workers, tierSet)
	fallbackNote := ""
	if target == 1 && !effectiveAllow {
		fallbackNote = " (tier-1 fallback disabled)"
	} else if strictTier {
		fallbackNote = " (exact tier requested)"
	}
	anyTarget := false
	for _, r := range workers {
		if tierOf(r) == target {
			anyTarget = true
			break
		}
	}
	if !anyTarget {
		fallbackNote = " (no matching worker tier)"
	}
	reason := "no available tier " + itoa(target) + " account"
	if atCap > 0 {
		reason += " under the session cap (" + itoa(atCap) + " at cap)"
	}
	return RouteResult{OK: false,
		Reason: reason + fallbackNote,
		Task:   task, TargetTier: target, FallbackUsed: false, Account: nil,
		BlockedTargetAccounts: blocked}
}

func rankLess(a, b Account) bool {
	aw, al, aa, ac, ap, at := routeRank(a)
	bw, bl, ba, bc, bp, bt := routeRank(b)
	if aw != bw {
		return aw < bw
	}
	if al != bl {
		return al < bl
	}
	if aa != ba {
		return aa < ba
	}
	if ac != bc {
		return ac < bc
	}
	if ap != bp {
		return ap < bp
	}
	return at < bt
}
