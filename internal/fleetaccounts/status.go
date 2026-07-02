package fleetaccounts

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

// Registry is the live session registry (sessions.json) the watchdog produces. The
// static roster keeps working when no registry exists, so a missing/malformed file
// yields a zero Registry. Only the fields the passive runtime-status fold consults are
// modeled; unknown keys are ignored.
type Registry struct {
	GeneratedUTC string         `json:"generated_utc"`
	Throttle     map[string]any `json:"throttle"`
	Auth         map[string]any `json:"auth"`
	Sessions     []Session      `json:"sessions"`
}

// Session is one registry session row.
type Session struct {
	Account        string  `json:"account"`
	Project        string  `json:"project"`
	Disp           string  `json:"disp"`
	Action         string  `json:"action"`
	AgeMin         float64 `json:"age_min"`
	SeenUTC        string  `json:"seen_utc"`
	Last           string  `json:"last"`
	Reason         string  `json:"reason"`
	ProbeStatus    string  `json:"probe_status"`
	ThrottleReset  string  `json:"throttle_reset"`
	ThrottleWeekly string  `json:"throttle_weekly"`
	hasAge         bool
}

// LoadRegistry reads sessions.json best-effort: missing/malformed yields an empty Registry.
func LoadRegistry(path string) Registry {
	var reg Registry
	data, err := os.ReadFile(path)
	if err != nil {
		return Registry{}
	}
	if err := json.Unmarshal(data, &reg); err != nil {
		return Registry{}
	}
	// distinguish "age_min absent" from "age_min == 0" by a second raw parse pass.
	var raw struct {
		Sessions []map[string]any `json:"sessions"`
	}
	if json.Unmarshal(data, &raw) == nil {
		for i := range reg.Sessions {
			if i < len(raw.Sessions) {
				_, has := raw.Sessions[i]["age_min"]
				reg.Sessions[i].hasAge = has
			}
		}
	}
	return reg
}

func registryAgeMin(reg Registry) *float64 {
	if reg.GeneratedUTC == "" {
		return nil
	}
	ts := parseUTC(reg.GeneratedUTC)
	if ts == nil {
		return nil
	}
	v := math.Round(time.Since(*ts).Seconds()/60.0*10) / 10
	return &v
}

func parseUTC(raw string) *time.Time {
	if raw == "" {
		return nil
	}
	s := strings.Replace(raw, "Z", "+00:00", 1)
	for _, layout := range []string{
		"2006-01-02T15:04:05.999999-07:00",
		"2006-01-02T15:04:05-07:00",
		"2006-01-02T15:04:05.999999",
		"2006-01-02T15:04:05",
	} {
		if t, err := time.Parse(layout, s); err == nil {
			tu := t.UTC()
			return &tu
		}
	}
	return nil
}

func sessionAge(s Session) (float64, bool) {
	if !s.hasAge {
		return 0, false
	}
	return s.AgeMin, true
}

func rowSeenUTC(s Session, generatedUTC string) *time.Time {
	if seen := parseUTC(s.SeenUTC); seen != nil {
		return seen
	}
	age, ok := sessionAge(s)
	gen := parseUTC(generatedUTC)
	if !ok || gen == nil {
		return nil
	}
	t := gen.Add(-time.Duration(age * float64(time.Minute)))
	return &t
}

// dailyResetWindow is the slack allowed before declaring a passed bare reset time expired.
const dailyResetWindow = 6 * time.Hour

var (
	parenTail = regexp.MustCompile(`\s*\([^)]*$`)
	parenAll  = regexp.MustCompile(`\s*\([^)]*\)`)
	wsRun     = regexp.MustCompile(`\s+`)
)

// resetIsFuture is the best-effort future-check for Claude's reset strings. Returns a
// pointer: true for a still-future reset, false for an expired parsed reset, nil for an
// unknown format. It reads the shared resetTime core (soonness.go) — which owns the format
// handling and the daily-reset tomorrow-rollover — so a time-only reset that rolled forward
// into today's slack reads as future here exactly as before. Mirrors
// fleet_accounts._reset_is_future (UTC anchoring; LA zone handled only via the carried
// "America/Los_Angeles" hint, which the Python also keys off textually).
func resetIsFuture(reset string, now time.Time) *bool {
	cand, ok := resetTime(reset, now)
	if !ok {
		return nil
	}
	r := cand.After(now)
	return &r
}

func throttleIsActive(info map[string]any) bool {
	reset := resetText(info)
	state := resetIsFuture(reset, time.Now().UTC())
	if state != nil && !*state {
		return false
	}
	return true
}

// weeklyThrottleIsActive mirrors fleet_accounts._weekly_throttle_is_active: a carried
// throttle holds through a fresh OK probe only while its WEEKLY window is still open —
// no weekly text means no weekly cap, a weekly reset provably in the past means the cap
// expired, and anything else (unparseable/future) defers to the throttle's own liveness.
func weeklyThrottleIsActive(info map[string]any) bool {
	if info == nil {
		return false
	}
	weekly := asString(info["weekly"])
	if weekly == "" {
		return false
	}
	if state := resetIsFuture(weekly, time.Now().UTC()); state != nil && !*state {
		return false
	}
	return throttleIsActive(info)
}

// applyThrottleStatus stamps the carried-throttle block onto st, mirroring
// fleet_accounts._apply_throttle_status. Python stamps reset/weekly straight from
// thr.get(...): absent -> None (null), present (even "") -> the value; mirror that
// presence, not emptiness.
func applyThrottleStatus(st RuntimeStatus, thr map[string]any) RuntimeStatus {
	resetVal, hasReset := thr["reset"]
	weeklyVal, hasWeekly := thr["weekly"]
	reset := asString(resetVal)
	weekly := asString(weeklyVal)
	reason := "usage limit"
	if reset != "" {
		reason = "usage limit; resets " + reset
	}
	if weekly != "" {
		reason += "; weekly " + weekly
	}
	st.Available, st.Blocked = false, true
	st.BlockKind, st.hasBlockKind = "usage", true
	st.BlockReason = reason
	st.Reset, st.hasReset = reset, hasReset
	st.Weekly, st.hasWeekly = weekly, hasWeekly
	st.Throttled = true
	return st
}

// shouldConsultProbeLedger mirrors fleet_accounts._should_consult_probe_ledger for the
// passed-registry case (the only shape the Go fold has: Annotate always receives a
// loaded Registry): consult the ledger exactly when FLEET_REG_DIR names the registry
// dir the prober writes under, so the Go reader and the Python writer agree on the dir
// and callers without a prober keep the pure passive fold.
func shouldConsultProbeLedger() bool {
	return strings.TrimSpace(os.Getenv("FLEET_REG_DIR")) != ""
}

func resetText(info map[string]any) string {
	if info == nil {
		return ""
	}
	if r, ok := info["reset"]; ok {
		return asString(r)
	}
	return ""
}

func normalizeThrottle(throttle map[string]any) map[string]map[string]any {
	out := map[string]map[string]any{}
	for account, info := range throttle {
		if m, ok := info.(map[string]any); ok {
			out[account] = m
		} else {
			out[account] = map[string]any{"reset": info}
		}
	}
	return out
}

// RuntimeStatus is the live availability fold for one account. Field names match the
// status dict in fleet_accounts.runtime_status.
type RuntimeStatus struct {
	Available           bool
	Blocked             bool
	BlockKind           string // "" -> null
	BlockReason         string
	Reset               string // "" -> null
	Weekly              string // "" -> null
	Throttled           bool
	ActiveSessions      int
	LiveSessions        int
	AuthBlockedSessions int
	StatusSource        string
	RegistryAgeMin      *float64
	hasBlockKind        bool
	hasReset            bool
	hasWeekly           bool
}

// computeRuntimeStatus folds the passive registry signals (sessions/throttle) into one
// account's availability. Synthetic _probe session rows already present in sessions.json
// are honored first (the watchdog-folded path); absent those, a fresh active-probe
// verdict from the probe LEDGER (account_probe writes probe_ledger.jsonl, never
// sessions.json) overrides a carried block — consulted exactly when FLEET_REG_DIR names
// the prober's registry dir (see shouldConsultProbeLedger), so callers without a prober
// keep the pure passive fold.
func computeRuntimeStatus(account string, reg Registry) RuntimeStatus {
	throttleMap := normalizeThrottle(reg.Throttle)
	authMap := reg.Auth
	generatedUTC := reg.GeneratedUTC

	var acct []Session
	for _, s := range reg.Sessions {
		if s.Account == account {
			acct = append(acct, s)
		}
	}

	var probeRows []Session
	for _, s := range acct {
		if s.Project == "_probe" {
			probeRows = append(probeRows, s)
		}
	}
	freshProbeOK := false
	var freshProbeBlock *Session
	for i := range probeRows {
		s := probeRows[i]
		ps := strings.ToUpper(s.ProbeStatus)
		if ps == "OK" || (s.Disp == "LIVE" && s.ProbeStatus == "") {
			freshProbeOK = true
		}
		if ps != "" && ps != "OK" && freshProbeBlock == nil {
			freshProbeBlock = &probeRows[i]
		}
	}

	active, live := 0, 0
	for _, s := range acct {
		if s.Disp != "DONE" && s.Disp != "USER_CLOSED" {
			active++
		}
		if s.Disp == "LIVE" {
			live++
		}
	}

	var authBlocked []Session
	for _, s := range acct {
		if s.Action == "BLOCKED_AUTH" || s.Disp == "INFRA_AUTH" {
			authBlocked = append(authBlocked, s)
		}
	}
	latestAuthAge, haveAuthAge := minAge(authBlocked)
	var successRows []Session
	for _, s := range acct {
		if s.Disp == "LIVE" || s.Disp == "DONE" {
			successRows = append(successRows, s)
		}
	}
	latestSuccessAge, haveSuccessAge := minAge(successRows)

	sessionAuthCurrent := len(authBlocked) > 0 &&
		(!haveSuccessAge || !haveAuthAge || latestSuccessAge > latestAuthAge)

	var latestSuccessSeen *time.Time
	for _, s := range successRows {
		if seen := rowSeenUTC(s, generatedUTC); seen != nil {
			if latestSuccessSeen == nil || seen.After(*latestSuccessSeen) {
				latestSuccessSeen = seen
			}
		}
	}
	var authInfo map[string]any
	if ai, ok := authMap[account].(map[string]any); ok {
		authInfo = ai
	} else if _, ok := authMap[account]; ok {
		authInfo = map[string]any{}
	}
	var authSeen *time.Time
	if authInfo != nil {
		authSeen = parseUTC(asString(authInfo["seen_utc"]))
	}
	knownAuthCurrent := authInfo != nil &&
		(latestSuccessSeen == nil || authSeen == nil || !latestSuccessSeen.After(*authSeen))
	authCurrent := sessionAuthCurrent || knownAuthCurrent

	st := RuntimeStatus{
		Available:           true,
		Throttled:           false,
		ActiveSessions:      active,
		LiveSessions:        live,
		AuthBlockedSessions: len(authBlocked),
		StatusSource:        "none",
	}
	if !registryEmpty(reg) {
		st.StatusSource = "registry"
		st.RegistryAgeMin = registryAgeMin(reg)
	}

	if freshProbeOK {
		st.StatusSource = "probe"
		return st
	}
	if freshProbeBlock != nil {
		kind := map[string]string{"AUTH": "auth", "ACCESS": "access", "CREDIT": "credit", "LIMIT": "usage"}[strings.ToUpper(freshProbeBlock.ProbeStatus)]
		if kind == "" {
			kind = "auth"
		}
		reason := freshProbeBlock.Reason
		if reason == "" {
			reason = freshProbeBlock.Last
		}
		if reason == "" {
			reason = "blocked"
		}
		st.Available, st.Blocked = false, true
		st.BlockKind, st.hasBlockKind = kind, true
		st.BlockReason = reason
		st.Reset, st.hasReset = freshProbeBlock.ThrottleReset, true
		st.Weekly, st.hasWeekly = freshProbeBlock.ThrottleWeekly, true
		st.Throttled = kind == "usage"
		st.StatusSource = "probe"
		return st
	}

	// No synthetic _probe session row -> consult the active-probe LEDGER directly.
	// account_probe writes its verdict only there, so a fresh manual/watchdog probe
	// would otherwise be invisible here and the carried throttle below would win
	// (probe says OK, roster still "resets 11pm"). A ledger verdict within
	// ProbeLedgerFreshMin is the same authoritative fresh probe. Mirrors
	// fleet_accounts.runtime_status's probe-ledger rung.
	thr, hasThr := throttleMap[account]
	if shouldConsultProbeLedger() {
		if led := FreshProbeFromLedger(account, "", time.Now().UTC(), 0); led != nil {
			if led.Available {
				// A fresh OK must not reopen a seat whose WEEKLY cap is still
				// active: the weekly window outlives any single probe. (Python also
				// clears this hold when the throttle's stamped identity provably
				// differs from the seat's current identity; this port assumes the
				// identity matches — the fail-closed reading — leaving the
				// identity-match refinement as the named follow-on.)
				if hasThr && weeklyThrottleIsActive(thr) {
					return applyThrottleStatus(st, thr)
				}
				st.StatusSource = "probe-ledger"
				return st
			}
			kind := led.BlockKind
			if kind == "" {
				kind = "auth"
			}
			reason := led.BlockReason
			if reason == "" {
				reason = "blocked"
			}
			st.Available, st.Blocked = false, true
			st.BlockKind, st.hasBlockKind = kind, true
			st.BlockReason = reason
			st.Reset, st.hasReset = led.Reset, true
			st.Weekly, st.hasWeekly = led.Weekly, true
			st.Throttled = kind == "usage"
			st.StatusSource = "probe-ledger"
			return st
		}
	}

	if hasThr && throttleIsActive(thr) {
		return applyThrottleStatus(st, thr)
	}

	if authCurrent {
		var lastParts []string
		for _, s := range authBlocked {
			v := s.Last
			if v == "" {
				v = s.Reason
			}
			lastParts = append(lastParts, v)
		}
		last := strings.Join(lastParts, " ")
		var kind, reason string
		if knownAuthCurrent && !sessionAuthCurrent {
			kind = asString(authInfo["block_kind"])
			if kind == "" {
				kind = "auth"
			}
			reason = asString(authInfo["block_reason"])
			if reason == "" {
				reason = authBlockReason("")
			}
		} else {
			kind = authBlockKind(last)
			reason = authBlockReason(last)
		}
		st.Available, st.Blocked = false, true
		st.BlockKind, st.hasBlockKind = kind, true
		st.BlockReason = reason
	}
	return st
}

func registryEmpty(reg Registry) bool {
	return reg.GeneratedUTC == "" && len(reg.Throttle) == 0 &&
		len(reg.Auth) == 0 && len(reg.Sessions) == 0
}

func minAge(rows []Session) (float64, bool) {
	have := false
	var m float64
	for _, s := range rows {
		if age, ok := sessionAge(s); ok {
			if !have || age < m {
				m, have = age, true
			}
		}
	}
	return m, have
}

// Annotate attaches live availability fields to discovered rows. Worker rows get the
// runtime-status fold; non-worker rows get the static "not offered" shape. The result is
// sorted by (product, kind != worker, !available, tag) to match annotate_accounts.
func Annotate(rows []Account, reg Registry) []Account {
	out := make([]Account, len(rows))
	copy(out, rows)
	for i := range out {
		r := &out[i]
		if r.Kind == KindWorker {
			st := computeRuntimeStatus(r.Account, reg)
			applyStatus(r, st)
			applyLoginGate(r)
		} else {
			r.Available = boolp(false)
			r.Blocked = boolp(false)
			r.BlockKind = nil // null
			br := r.Reason
			r.BlockReason = strp(br)
			r.Reset = nil
			r.Weekly = nil
			r.Throttled = boolp(false)
			r.ActiveSessions = intp(0)
			r.LiveSessions = intp(0)
			r.AuthBlockedSessions = intp(0)
			r.StatusSource = strp("static")
			r.RegistryAgeMin = nil
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Product != out[j].Product {
			return out[i].Product < out[j].Product
		}
		wi, wj := out[i].Kind != KindWorker, out[j].Kind != KindWorker
		if wi != wj {
			return !wi && wj
		}
		ai, aj := derefBool(out[i].Available), derefBool(out[j].Available)
		if ai != aj {
			return ai && !aj
		}
		return out[i].Tag < out[j].Tag
	})
	return out
}

func applyStatus(r *Account, st RuntimeStatus) {
	r.Available = boolp(st.Available)
	r.Blocked = boolp(st.Blocked)
	if st.hasBlockKind {
		r.BlockKind = strp(st.BlockKind)
	} else {
		r.BlockKind = nil
	}
	r.BlockReason = strp(st.BlockReason)
	if st.hasReset {
		r.Reset = strp(st.Reset)
	} else {
		r.Reset = nil
	}
	if st.hasWeekly {
		r.Weekly = strp(st.Weekly)
	} else {
		r.Weekly = nil
	}
	r.Throttled = boolp(st.Throttled)
	r.ActiveSessions = intp(st.ActiveSessions)
	r.LiveSessions = intp(st.LiveSessions)
	r.AuthBlockedSessions = intp(st.AuthBlockedSessions)
	r.StatusSource = strp(st.StatusSource)
	r.RegistryAgeMin = st.RegistryAgeMin
}

func applyLoginGate(r *Account) {
	if r.Product != "claude" || r.Kind != KindWorker {
		return
	}
	if !accountLoginBlocked(*r) {
		return
	}
	r.Available = boolp(false)
	r.Blocked = boolp(true)
	r.BlockKind = strp("auth")
	r.BlockReason = strp(accountLoginBlockReason(*r))
	r.Throttled = boolp(false)
}

func accountCanBeOffered(r Account) bool {
	return derefBool(r.Available) && !accountLoginBlocked(r)
}

func accountLoginBlocked(r Account) bool {
	if r.Product != "claude" || r.Kind != KindWorker {
		return false
	}
	if r.CanServe != nil && !*r.CanServe {
		return true
	}
	st := configaccounts.LoginStatus(derefStr(r.LoginStatus))
	return st != "" && st != configaccounts.LoginReady
}

func accountLoginBlockReason(r Account) string {
	st := configaccounts.LoginStatus(derefStr(r.LoginStatus))
	if st != "" && st != configaccounts.LoginReady {
		reason, _ := configaccounts.LoginReasonAction(st, configaccounts.Home{Name: r.Tag, Dir: r.Dir})
		if reason != "" {
			return reason
		}
		return "account login status is " + string(st)
	}
	return "account login cannot serve"
}

// AnnotatedRoster is the canonical "give me the live accounts" call: discover + annotate.
func AnnotatedRoster(home, configHome string, pol Policy, reg Registry) []Account {
	return Annotate(Discover(home, configHome, pol), reg)
}

// Available returns worker accounts safe to offer right now (routable + available),
// excluding duplicate-identity dirs.
func Available(rows []Account) []Account {
	var out []Account
	for _, r := range rows {
		if RoutableWorker(r) && accountCanBeOffered(r) {
			out = append(out, r)
		}
	}
	return out
}
