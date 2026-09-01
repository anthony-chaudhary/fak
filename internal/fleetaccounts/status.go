package fleetaccounts

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

// Registry is the live session registry (sessions.json) the watchdog produces. The
// static roster keeps working when no registry exists, so a missing/malformed file
// yields a zero Registry. Only the fields the passive runtime-status fold consults are
// modeled; unknown keys are ignored.
type Registry struct {
	GeneratedUTC string           `json:"generated_utc"`
	Throttle     map[string]any   `json:"throttle"`
	Auth         map[string]any   `json:"auth"`
	Accounts     []map[string]any `json:"accounts"`
	Sessions     []Session        `json:"sessions"`
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
	// raw is the row's unmodeled JSON, kept by LoadRegistry so the identity fold can read
	// any account_uuid/login_email the watchdog stamps on a session row (see
	// sessionIdentity). A Session built directly (tests) leaves this nil — no identity.
	raw map[string]any
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
				reg.Sessions[i].raw = raw.Sessions[i]
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

// weeklyThrottleIsActive is the view for the weekly leg's independent liveness (the rung
// that holds a seat closed through a fresh OK probe). Mirrors
// fleet_accounts._weekly_throttle_is_active: no weekly text means no weekly cap, a weekly
// reset provably in the past means the cap expired, and anything else defers to the
// throttle's own liveness.
func weeklyThrottleIsActive(info map[string]any) bool {
	return DisambiguateCap(info, CapObservation{}, time.Now().UTC(), DefaultCapPolicy()).WeeklyActive
}

// applyThrottleStatus stamps the carried-throttle block onto st, reading the disambiguated
// CapState. Mirrors fleet_accounts._apply_throttle_status: reset/weekly are carried straight
// from the throttle map (absent -> None/null, present even "" -> the value; CapState.HasReset
// /HasWeekly preserve that presence, not emptiness), and the block reason is the core's
// composed "usage limit; resets X; weekly Y".
func applyThrottleStatus(st RuntimeStatus, thr map[string]any) RuntimeStatus {
	cs := DisambiguateCap(thr, CapObservation{}, time.Now().UTC(), DefaultCapPolicy())
	st.Available, st.Blocked = false, true
	st.BlockKind, st.hasBlockKind = "usage", true
	st.BlockReason = cs.BlockReason
	st.Reset, st.hasReset = cs.Reset, cs.HasReset
	st.Weekly, st.hasWeekly = cs.Weekly, cs.HasWeekly
	st.Throttled = true
	return st
}

// markUsageSoon carries a still-active DAILY cap onto st as advisory when a fresh OK probe
// reopens the seat over it. Mirrors fleet_accounts._mark_usage_soon: the reopen is correct
// for availability (a healthy seat must not sit behind a stale/near-expired cap — the day24
// incident), but dropping the cap entirely hid a seat sitting at its daily limit and about to
// roll over, which showed as a plain "serving" row with no usage. st.Available/st.Throttled
// are left untouched (the seat stays offered); only a still-future daily reset is surfaced so
// the roster can render "serving, cap resets X". A weekly-active cap never reaches here (it
// holds the seat closed upstream); an absent/expired cap contributes nothing, so a normal
// serving row gains no field and its byte-parity JSON is unchanged.
func markUsageSoon(st *RuntimeStatus, thr map[string]any) {
	if thr == nil {
		return
	}
	cs := DisambiguateCap(thr, CapObservation{}, time.Now().UTC(), DefaultCapPolicy())
	if !cs.Active || cs.WeeklyActive {
		return
	}
	if cs.Reset != "" {
		st.UsageSoonReset, st.hasUsageSoon = cs.Reset, true
	}
}

// shouldConsultProbeLedger reports whether the probe-ledger rung may run at all: the
// registry dir this process would actually READ must be able to derive a block verdict,
// i.e. it must carry the probe_ledger.jsonl account_probe appends to. That is exactly
// accountprobe.ResolveRegDir().BlocksDerivable(), so the rung is gated on the same
// resolution FreshProbeFromLedger/deriveCapObservation then perform with an empty regDir.
//
// This used to gate on FLEET_REG_DIR merely being SET, on the reasoning that the env var
// is what makes the Go reader and the Python writer agree on a dir. That reasoning does
// not survive #5390: naming a dir is not the same as the dir holding a ledger, and the
// resolver now knows the difference. Derivability is the wider and more honest test:
//
//   - FLEET_REG_DIR names a ledger-bearing dir -> consulted, exactly as before.
//   - FLEET_REG_DIR unset, but a discovered registry (typically the per-user Fleet dir the
//     prober writes under) carries the ledger -> consulted NOW, where before the rung never
//     ran and a fresh probe verdict was invisible to the roster no matter how correct the
//     resolver was. This is the #5439 finding.
//   - FLEET_REG_DIR names a ledger-LESS dir -> NOT consulted, where before the rung ran
//     over a ledger that was not there. Every read returned nothing, so the fold fell
//     through to the carried block anyway; refusing to run says so instead of pretending
//     to have looked.
//   - Nothing anywhere -> not consulted, exactly as before, so a fresh checkout and CI keep
//     the pure passive fold.
//
// The price is that the answer is a function of the filesystem rather than of one env var:
// a caller (or a test) that wants a definite verdict must arrange the dirs, not just the
// variable. Repeated calls over an unchanged filesystem are stable — ResolveRegDir consults
// no clock — so this does not make the fold flap.
func shouldConsultProbeLedger() bool {
	return accountprobe.ResolveRegDir().BlocksDerivable()
}

// seatProbeUnmeasured reports whether a BUSY probe ledger — one carrying rows for at least
// one account — has no current evidence about THIS seat: no row at all, or a newest row past
// accountprobe.SeatCoverageMaxAgeMin (or undatable). It is the per-seat half of the question
// shouldConsultProbeLedger asks per-registry, and the whole of #5391: on the host that filed
// it the ledger was present, derivable and busy — opencode-* rows current to the minute —
// while several claude seats' newest rows were 8-9 days old, so a registry-level "blocks are
// derivable here" was true and told the fold nothing whatever about those seats.
//
// The "busy" precondition (CoverageReport.Sufficient) is deliberate and is the reason this
// does not re-open #5439's boundary. A ledger that has recorded NOTHING is already described
// by the registry-level judgement, and a seat-level downgrade there would only restate it; a
// ledger that has recorded rows for OTHER accounts is the case where the registry-level
// answer is affirmatively misleading, because the prober demonstrably ran and demonstrably
// skipped this seat. Only the second case is what #5391 observed, so only the second case
// moves. now is injected by the caller for determinism.
func seatProbeUnmeasured(account string, now time.Time) bool {
	rep := accountprobe.GradeSeats([]string{account}, "", now)
	if !rep.Sufficient || len(rep.Seats) != 1 {
		return false
	}
	return !rep.Seats[0].Health.Measured()
}

// markUnknownHealth downgrades the fold's status_source from the confident "registry" to
// "registry-unknown" when an UNBLOCKED seat is being published on probe evidence that does
// not exist. Two disjoint absences reach the same verdict:
//
//   - blocksDerivable false — the registry itself cannot derive a block at all (no probe
//     ledger beside its sessions.json — see accountprobe.RegChoice.BlocksDerivable, whose doc
//     states the obligation this discharges: a caller that would otherwise publish "no seats
//     blocked" must publish nothing instead).
//   - seatUnmeasured true — the registry CAN derive blocks and its ledger is busy, but that
//     ledger holds no current row for this particular seat (see seatProbeUnmeasured). #5391:
//     "never probed" and "probed OK" must not both read as a proven-free seat just because
//     the prober is healthy for some other account class.
//
// The second is the narrower and later of the two, and it is what keeps the first from being
// read as a sufficient test. A registry-wide grade cannot see a per-class coverage hole, and
// a per-class hole is what routes workers at a seat that answers 403 to everything.
//
// Unknown-health is a THIRD state, and deliberately not a block. Converting absence into
// blocked would strand every seat on a host whose prober has not run — and worse, it is
// self-sealing: the roster is what routes the work that runs the probe, so a block imposed
// for want of a probe forbids the very probe that would clear it. That deadlock is the
// failure this repo already paid for once, so the seat stays offered (Available is
// untouched) and only the CLAIM is weakened. A consumer that cannot tolerate an unproven
// seat now has a name to switch on; one that does not care keeps today's behavior, since
// every existing status_source consumer treats an unrecognized value exactly as it treats
// "registry".
//
// A blocked seat keeps "registry": "blocked" is a positive derivation from the registry's
// own throttle/auth rows, not a statement about probe evidence, so its provenance is not in
// doubt. An empty registry keeps "none", which already says "nothing was consulted".
func markUnknownHealth(st RuntimeStatus, blocksDerivable, seatUnmeasured bool) RuntimeStatus {
	if st.Blocked || st.StatusSource != "registry" {
		return st
	}
	if blocksDerivable && !seatUnmeasured {
		return st
	}
	st.StatusSource = "registry-unknown"
	return st
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
	UsageSoonReset      string // advisory: a still-active daily cap a fresh probe reopened over
	hasBlockKind        bool
	hasReset            bool
	hasWeekly           bool
	hasUsageSoon        bool
}

// computeRuntimeStatus folds the passive registry signals (sessions/throttle) into one
// account's availability. Synthetic _probe session rows already present in sessions.json
// are honored first (the watchdog-folded path); absent those, a fresh active-probe
// verdict from the probe LEDGER (account_probe writes probe_ledger.jsonl, never
// sessions.json) overrides a carried block — consulted exactly when the registry this
// process resolves to can derive a block at all (see shouldConsultProbeLedger), so callers
// without a prober keep the pure passive fold. When it cannot, an unblocked seat is
// published as unknown-health rather than as a proven-free one (see markUnknownHealth).
// dir is the account's config home (the Account.Dir the caller already holds). It feeds the
// identity fold that decides whether a carried weekly cap still belongs to the seat's
// CURRENT login before a fresh OK is allowed to reopen it (see
// throttleMatchesCurrentIdentity); "" simply skips the current-config identity candidate.
// scanProbeRows folds an account's _probe sessions into the three signals
// computeRuntimeStatus needs: whether any probe is fresh-OK (an OK ProbeStatus, or a
// LIVE session with no explicit status), the first non-OK probe session (the block
// witness), and the first fresh-OK probe row itself — its stamped identity is the
// probe-identity candidate the weekly-cap hold consults, mirroring the Python's
// `probe_identity = _account_identity_from(row)` over the first OK/LIVE probe row.
func scanProbeRows(acct []Session) (freshProbeOK bool, freshProbeBlock, freshProbeOKRow *Session) {
	var probeRows []Session
	for _, s := range acct {
		if s.Project == "_probe" {
			probeRows = append(probeRows, s)
		}
	}
	for i := range probeRows {
		s := probeRows[i]
		ps := strings.ToUpper(s.ProbeStatus)
		if ps == "OK" || (s.Disp == "LIVE" && s.ProbeStatus == "") {
			freshProbeOK = true
			if freshProbeOKRow == nil {
				freshProbeOKRow = &probeRows[i]
			}
		}
		if ps != "" && ps != "OK" && freshProbeBlock == nil {
			freshProbeBlock = &probeRows[i]
		}
	}
	return freshProbeOK, freshProbeBlock, freshProbeOKRow
}

// countActiveLive tallies an account's sessions: active = not terminal (DONE/USER_CLOSED),
// live = currently LIVE. Split out of computeRuntimeStatus as a self-contained counting phase.
func countActiveLive(acct []Session) (active, live int) {
	for _, s := range acct {
		if s.Disp != "DONE" && s.Disp != "USER_CLOSED" {
			active++
		}
		if s.Disp == "LIVE" {
			live++
		}
	}
	return active, live
}

type runtimeEvidence struct {
	acct               []Session
	freshProbeOK       bool
	freshProbeBlock    *Session
	freshProbeOKRow    *Session
	active             int
	live               int
	authBlocked        []Session
	sessionAuthCurrent bool
	authInfo           map[string]any
	knownAuthCurrent   bool
	authCurrent        bool
}

func collectRuntimeEvidence(account string, reg Registry) runtimeEvidence {
	var evidence runtimeEvidence
	for _, session := range reg.Sessions {
		if session.Account == account {
			evidence.acct = append(evidence.acct, session)
		}
	}
	evidence.freshProbeOK, evidence.freshProbeBlock, evidence.freshProbeOKRow = scanProbeRows(evidence.acct)
	evidence.active, evidence.live = countActiveLive(evidence.acct)
	for _, session := range evidence.acct {
		if session.Action == "BLOCKED_AUTH" || session.Disp == "INFRA_AUTH" {
			evidence.authBlocked = append(evidence.authBlocked, session)
		}
	}
	latestAuthAge, haveAuthAge := minAge(evidence.authBlocked)
	var successRows []Session
	for _, session := range evidence.acct {
		if session.Disp == "LIVE" || session.Disp == "DONE" {
			successRows = append(successRows, session)
		}
	}
	latestSuccessAge, haveSuccessAge := minAge(successRows)
	evidence.sessionAuthCurrent = len(evidence.authBlocked) > 0 &&
		(!haveSuccessAge || !haveAuthAge || latestSuccessAge > latestAuthAge)
	var latestSuccessSeen *time.Time
	for _, session := range successRows {
		if seen := rowSeenUTC(session, reg.GeneratedUTC); seen != nil {
			if latestSuccessSeen == nil || seen.After(*latestSuccessSeen) {
				latestSuccessSeen = seen
			}
		}
	}
	if authInfo, ok := reg.Auth[account].(map[string]any); ok {
		evidence.authInfo = authInfo
	} else if _, ok := reg.Auth[account]; ok {
		evidence.authInfo = map[string]any{}
	}
	var authSeen *time.Time
	if evidence.authInfo != nil {
		authSeen = parseUTC(asString(evidence.authInfo["seen_utc"]))
	}
	evidence.knownAuthCurrent = evidence.authInfo != nil &&
		(latestSuccessSeen == nil || authSeen == nil || !latestSuccessSeen.After(*authSeen))
	evidence.authCurrent = evidence.sessionAuthCurrent || evidence.knownAuthCurrent
	return evidence
}

func computeRuntimeStatus(account, dir string, reg Registry) RuntimeStatus {
	throttleMap := normalizeThrottle(reg.Throttle)
	evidence := collectRuntimeEvidence(account, reg)
	acct := evidence.acct
	freshProbeOK, freshProbeBlock, freshProbeOKRow := evidence.freshProbeOK, evidence.freshProbeBlock, evidence.freshProbeOKRow
	authBlocked := evidence.authBlocked
	sessionAuthCurrent, knownAuthCurrent, authCurrent := evidence.sessionAuthCurrent, evidence.knownAuthCurrent, evidence.authCurrent
	authInfo := evidence.authInfo

	st := RuntimeStatus{
		Available:           true,
		Throttled:           false,
		ActiveSessions:      evidence.active,
		LiveSessions:        evidence.live,
		AuthBlockedSessions: len(authBlocked),
		StatusSource:        "none",
	}
	if !registryEmpty(reg) {
		st.StatusSource = "registry"
		st.RegistryAgeMin = registryAgeMin(reg)
	}

	thr, hasThr := throttleMap[account]
	if freshProbeOK {
		// A fresh synthetic-probe OK clears a carried block, but — exactly as in the
		// probe-ledger rung below and in the Python's fresh_probe_ok branch — it must
		// NOT reopen a seat whose WEEKLY cap is still active for the seat's CURRENT
		// login: the weekly window outlives any single probe. The hold is cleared only
		// on a proven identity mismatch (a usage limit stamped for a DIFFERENT account
		// the dir was logged into before a re-login); throttleMatchesCurrentIdentity
		// holds fail-closed when identity is unknown. The probe row's own stamped
		// identity is the leading candidate, mirroring the Python's probe_identity. This
		// rung has no cap_obs (the ledger-derived observation is folded only below), so
		// the weekly check is the legacy single-shot weeklyThrottleIsActive — matching
		// the Python's _weekly_throttle_is_active here.
		if hasThr {
			var probeIdentity map[string]string
			if freshProbeOKRow != nil {
				probeIdentity = sessionIdentity(*freshProbeOKRow)
			}
			if weeklyThrottleIsActive(thr) &&
				throttleMatchesCurrentIdentity(account, dir, thr, reg, acct, probeIdentity) {
				return applyThrottleStatus(st, thr)
			}
			markUsageSoon(&st, thr)
		}
		st.StatusSource = "probe"
		return st
	}
	if freshProbeBlock != nil {
		kind := map[string]string{"AUTH": "auth", "ACCESS": "access", "CREDIT": "credit", "LIMIT": "usage"}[strings.ToUpper(freshProbeBlock.ProbeStatus)]
		if kind == "" {
			kind = "auth"
		}
		reason := probeBlockReason(freshProbeBlock.Reason, freshProbeBlock.Last)
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
	now := time.Now().UTC()
	// The cap-disambiguation cycles (aging + probe-override) fold a CapObservation drawn
	// from the SAME probe ledger the fresh-probe rung reads, so gate it on the same switch:
	// with the ledger unconsulted (or no carried throttle) capObs is zero and
	// DisambiguateCap stays on its legacy single-shot path at both seams below.
	consultLedger := shouldConsultProbeLedger()
	var capObs CapObservation
	if consultLedger && hasThr {
		capObs = deriveCapObservation(account, "")
	}
	if consultLedger {
		if led := FreshProbeFromLedger(account, "", now, 0); led != nil {
			if led.Available {
				// A fresh OK must not reopen a seat whose WEEKLY cap is still
				// active: the weekly window outlives any single probe. But the hold
				// only stands while the cap still belongs to the seat's CURRENT
				// login — a usage limit stamped for a DIFFERENT account the dir was
				// logged into before a re-login must NOT keep the reborn seat closed.
				// throttleMatchesCurrentIdentity clears the hold on a proven identity
				// mismatch and, fail-closed, holds when identity is unknown (today's
				// recorders stamp none, so this preserves the conservative hold). The
				// probe-identity candidate is nil here: the account-probe ledger
				// stamps no identity, so — as in the Python on today's ledgers — it
				// contributes nothing and current-config identity decides.
				//
				// A RUN of fresh OKs can also overturn the hold: DisambiguateCap folds
				// the ledger-derived observation so that ≥2 consecutive OKs past a
				// passed daily reset clear a stale/unparseable weekly the seat has
				// demonstrably outgrown. With no such streak the observation is inert
				// and cs.WeeklyActive equals the legacy weeklyThrottleIsActive(thr).
				if hasThr {
					cs := DisambiguateCap(thr, capObs, now, DefaultCapPolicy())
					if cs.WeeklyActive &&
						throttleMatchesCurrentIdentity(account, dir, thr, reg, acct, nil) {
						return applyThrottleStatus(st, thr)
					}
					markUsageSoon(&st, thr)
				}
				st.StatusSource = "probe-ledger"
				return st
			}
			kind := led.BlockKind
			if kind == "" {
				kind = "auth"
			}
			reason := probeBlockReason(led.BlockReason)
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

	// Carried-throttle fallback (no fresh probe verdict). The aging valve lives here: a
	// seat blocked past WeeklyMaxAge with a stale/unparseable weekly and no live daily leg
	// has outlived any real weekly window, so DisambiguateCap releases it via the derived
	// episode start. Absent that (no ledger history, or a young/parseable-future cap),
	// cs.Active equals the legacy throttleIsActive(thr).
	if hasThr && DisambiguateCap(thr, capObs, now, DefaultCapPolicy()).Active {
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
	// Nothing above found a block. If the registry could not have derived one — or if the
	// busy ledger it derives them from has no current row for THIS seat — say so rather than
	// publishing a seat as proven-free on evidence that was never available. The seat grade
	// is read only when it can still change the answer, so a blocked seat (whose provenance
	// is not in doubt) and a ledger-less registry (already answered) pay no extra ledger read.
	seatUnmeasured := false
	if consultLedger && !st.Blocked && st.StatusSource == "registry" {
		seatUnmeasured = seatProbeUnmeasured(account, now)
	}
	return markUnknownHealth(st, consultLedger, seatUnmeasured)
}

// probeBlockReason preserves the first recorder-provided reason and supplies the
// stable fallback used by both synthetic-session and ledger probe verdicts.
func probeBlockReason(reasons ...string) string {
	for _, reason := range reasons {
		if reason != "" {
			return reason
		}
	}
	return "blocked"
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
	return AnnotateWithProbes(rows, reg, "")
}

// AnnotateWithProbes applies the runtime fold plus the latest active-probe verdicts from
// regDir. A fresh probe is authoritative because it exercised the same provider seam the
// worker will use; stale entries are ignored so recovered seats become eligible again.
func AnnotateWithProbes(rows []Account, reg Registry, regDir string) []Account {
	out := make([]Account, len(rows))
	copy(out, rows)
	for i := range out {
		r := &out[i]
		if r.Kind == KindWorker {
			st := computeRuntimeStatus(r.Account, r.Dir, reg)
			applyStatus(r, st)
			applyFreshProbeLoginOverride(r)
			applyLoginGate(r)
			applyCredExpiryGate(r, time.Now().UTC())
			applyLatestProbeVerdict(r, regDir, time.Now().UTC())
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
	reconcileIdentityPeerAvailability(out)
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

func reconcileIdentityPeerAvailability(rows []Account) {
	byUUID := map[string][]int{}
	for i := range rows {
		r := rows[i]
		if !configHomeLoginProduct(r.Product) || r.Kind != KindWorker {
			continue
		}
		uuid := derefStr(r.AccountUUID)
		if uuid == "" {
			continue
		}
		byUUID[r.Product+"\x00"+uuid] = append(byUUID[r.Product+"\x00"+uuid], i)
	}
	for _, group := range byUUID {
		if len(group) < 2 {
			continue
		}
		canon := -1
		for _, i := range group {
			if derefStr(rows[i].IdentityRole) == "canonical" {
				canon = i
				break
			}
		}
		if canon < 0 || accountCanBeOffered(rows[canon]) || accountLoginBlocked(rows[canon]) {
			continue
		}
		if kind := strings.ToLower(derefStr(rows[canon].BlockKind)); kind != "" && kind != "auth" {
			continue
		}
		peer := -1
		for _, i := range group {
			if i == canon || !accountCanBeOffered(rows[i]) {
				continue
			}
			if peer < 0 || identityPeerStatusLess(rows[i], rows[peer]) {
				peer = i
			}
		}
		if peer < 0 {
			continue
		}
		applyIdentityPeerStatus(&rows[canon], rows[peer])
	}
}

func identityPeerStatusLess(a, b Account) bool {
	if derefInt(a.LiveSessions) != derefInt(b.LiveSessions) {
		return derefInt(a.LiveSessions) > derefInt(b.LiveSessions)
	}
	if derefInt(a.ActiveSessions) != derefInt(b.ActiveSessions) {
		return derefInt(a.ActiveSessions) > derefInt(b.ActiveSessions)
	}
	return a.Tag < b.Tag
}

func applyIdentityPeerStatus(dst *Account, peer Account) {
	dst.Available = boolp(true)
	dst.Blocked = boolp(false)
	dst.BlockKind = nil
	dst.BlockReason = strp("")
	dst.Reset = nil
	dst.Weekly = nil
	dst.Throttled = boolp(false)
	dst.StatusSource = strp("identity-peer")
	dst.ActiveSessions = intp(fleetMaxInt(derefInt(dst.ActiveSessions), derefInt(peer.ActiveSessions)))
	dst.LiveSessions = intp(fleetMaxInt(derefInt(dst.LiveSessions), derefInt(peer.LiveSessions)))
	dst.AuthBlockedSessions = intp(derefInt(dst.AuthBlockedSessions) + derefInt(peer.AuthBlockedSessions))
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
	if st.hasUsageSoon {
		r.UsageSoonReset = strp(st.UsageSoonReset)
	} else {
		r.UsageSoonReset = nil
	}
}

func applyLoginGate(r *Account) {
	if !configHomeLoginProduct(r.Product) || r.Kind != KindWorker {
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

func applyFreshProbeLoginOverride(r *Account) {
	if r.Product != "claude" || r.Kind != KindWorker {
		return
	}
	switch derefStr(r.StatusSource) {
	case "probe", "probe-ledger":
	default:
		return
	}
	if ReadCredExpiry(r.Dir).HasExpiry {
		r.LoginStatus = strp(string(configaccounts.LoginReady))
		r.CanServe = boolp(true)
	}
}

func fleetMaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func accountCanBeOffered(r Account) bool {
	return derefBool(r.Available) && !accountLoginBlocked(r)
}

func accountLoginBlocked(r Account) bool {
	if !configHomeLoginProduct(r.Product) || r.Kind != KindWorker {
		return false
	}
	if r.CanServe != nil && !*r.CanServe {
		return true
	}
	st := configaccounts.LoginStatus(derefStr(r.LoginStatus))
	return st != "" && st != configaccounts.LoginReady
}

// configHomeLoginProduct reports the products whose picker rows are backed by an isolated
// config home with a real credential reader. Those rows must pass LoginStatus before routing;
// env/config-only products (OpenCode today) keep their existing runtime-status behavior.
func configHomeLoginProduct(product string) bool {
	switch strings.ToLower(strings.TrimSpace(product)) {
	case "claude", "codex":
		return true
	default:
		return false
	}
}

func accountLoginBlockReason(r Account) string {
	if r.RootState == RootAbsent || r.RootState == RootMalformed {
		if reason := strings.TrimSpace(r.Reason); reason != "" {
			return reason
		}
	}
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

const activeProbeFreshness = 15 * time.Minute

func applyLatestProbeVerdict(r *Account, regDir string, now time.Time) {
	entry, ok := accountprobe.LastProbeByAccount(regDir)[r.Account]
	if !ok {
		return
	}
	observed, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(entry.TS))
	if err != nil || now.Sub(observed) < 0 || now.Sub(observed) > activeProbeFreshness {
		return
	}
	status := strings.ToUpper(strings.TrimSpace(entry.Status))
	if status == "" || status == "OK" || status == "NEEDS_PROBE" {
		return
	}
	kind := strings.ToLower(strings.TrimSpace(entry.Status))
	if kind == "" {
		kind = strings.ToLower(status)
	}
	reason := strings.TrimSpace(entry.BlockReason)
	if reason == "" {
		reason = "fresh guarded probe: " + status
	}
	r.Available = boolp(false)
	r.Blocked = boolp(true)
	r.BlockKind = strp(kind)
	r.BlockReason = strp(reason)
	r.StatusSource = strp("fresh_probe")
	if kind == "auth" {
		r.LoginStatus = strp("needs_login")
		r.CanServe = boolp(false)
	}
}

// AnnotatedRoster is the canonical "give me the live accounts" call: discover + annotate.
func AnnotatedRoster(home, configHome string, pol Policy, reg Registry) []Account {
	return AnnotateWithProbes(Discover(home, configHome, pol), reg, "")
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
