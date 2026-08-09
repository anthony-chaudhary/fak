package accounts

// orgwall.go — the DURABLE upstream ORG-AUTH-WALL record (#4998).
//
// Account ADMISSION is structural and local (a config dir exists, a cached profile
// identity, a credential file with live tokens); account USABILITY is behavioral and
// UPSTREAM. A seat can pass every local check and still be unable to serve a single
// token, because its ORGANIZATION has OAuth/subscription inference disabled upstream.
// That denial arrives as a typed HTTP 403 on the first real turn, and no re-login can
// clear it: every login mints another token for the SAME walled org.
//
// Before this file that evidence lived only in the guard process's in-memory `walled`
// set (cmd/fak/guard_account_failover.go). When the process exited the diagnosis died
// with it, and once Claude blanked the unusable tokens the same seat degraded to a
// generic needs_login — so `fak accounts status` reported needs_login and
// `fak accounts doctor` prescribed `/login`, the one repair the original terminal error
// had ALREADY proven futile. A weaker cause displaced a stronger, known one.
//
// This file closes that loop with three pieces and NO new store:
//
//  1. ClassifySeatHealth — the pure probe classifier: one (status, body, headers, now)
//     in, one of ready | usage_limited | needs_login | org_auth_wall out. The
//     usage-cap fences run FIRST because a Claude 403 body says nearly the same words
//     for a standing org wall and for a self-recovering usage/overage cap; only the
//     anthropic-ratelimit-unified-* headers (or a reset named in the text) tell them
//     apart. Misfiling a cap as a wall would durably exclude an account that recovers
//     on its own, so the fences are load-bearing, not decoration.
//
//  2. RecordOrgAuthWall / ClearOrgAuthWall / ObserveSeatHealth — durable, timestamped
//     evidence keyed by CANONICAL ACCOUNT identity (Identity.AccountKey) in the
//     fleet-shared CooldownStore. No token material and no raw upstream body is ever
//     persisted: the stored reason is the classified label. Reusing the cooldown store
//     is what makes the wall survive a new process and drop the account from
//     launch/dispatch everywhere the pool ALREADY honors a cooldown (LoginReportAt,
//     rotation, the dispatch preflight) — durability and exclusion with no new wiring.
//
//  3. The EXPIRY / REPROBE semantics. A usage cap self-recovers, so its window
//     elapsing IS the recovery. An org wall does not: a window elapsing is not evidence
//     an admin re-enabled the organization. So the org-auth-wall signal never lapses on
//     the timer (see activeCooldownSignals in cooldown.go); its deadline instead marks
//     when a REPROBE is DUE (ReprobeDue), and only a WITNESSED healthy observation
//     clears the wall (ObserveSeatHealth with SeatHealthReady, or an explicit operator
//     `fak accounts cooldown --clear`). An upstream administrative repair therefore
//     returns the seat to service on EVIDENCE, never on hope.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
)

// SeatHealth is the closed vocabulary for a seat's UPSTREAM usability — what a live
// round-trip on this credential actually proves, as opposed to what the local registry
// can see. LoginStatus answers "is this seat set up?"; SeatHealth answers "does the
// upstream still serve it?", and only the second can see an organization wall.
type SeatHealth string

const (
	// SeatHealthReady — the upstream accepted a real inference round-trip on this
	// credential. The strongest possible witness, and the only one that clears a wall.
	SeatHealthReady SeatHealth = "ready"
	// SeatHealthUsageLimited — a self-recovering rolling-window usage/overage cap. The
	// credential and the organization are both fine; the account returns at its reset.
	SeatHealthUsageLimited SeatHealth = "usage_limited"
	// SeatHealthNeedsLogin — the credential itself is expired/absent/rejected. A fresh
	// `/login` under this CLAUDE_CONFIG_DIR genuinely repairs it.
	SeatHealthNeedsLogin SeatHealth = "needs_login"
	// SeatHealthOrgAuthWall — the credential authenticates but its ORGANIZATION has
	// OAuth/subscription access disabled upstream. Re-login is futile (a new token for
	// the same walled org); the repair is a different account, API-key billing, or an
	// org admin re-enabling access.
	SeatHealthOrgAuthWall SeatHealth = "org_auth_wall"
	// SeatHealthUnknown — the response matched no known signature. Deliberately NOT
	// folded into a neighbouring state: guessing here is how a cap becomes a wall.
	SeatHealthUnknown SeatHealth = "unknown"
)

// orgAuthWallRE matches the ORGANIZATION-scoped OAuth/subscription-disable signature in
// an upstream denial body. It unions the two live phrasings the fleet has witnessed:
// the API-side wording internal/agent.orgOAuthDisabled reads off the wire ("OAuth
// authentication is currently not allowed for this organization"), and the CLI-banner
// wording internal/fleetaccounts.accessWallRE reads off `claude -p` stdout ("your
// organization has disabled Claude subscription access", "use an Anthropic API key
// instead", "ask your admin to enable access").
//
// WHY A THIRD READER. Those two siblings cannot be imported here: internal/agent is
// tier 4 (this leaf is tier 1, so the edge is an upward ARCH_LAYER_VIOLATION) and
// fleetaccounts' predicate is unexported. Rather than widen either boundary for one
// regexp, the signature is restated here with the sibling call sites named, and
// TestOrgAuthWallSignatureCoversWitnessedBodies pins the exact witnessed strings so
// drift between the three readers fails a test instead of silently mis-triaging a seat.
var orgAuthWallRE = regexp.MustCompile(`(?i)not allowed for this organization|` +
	`oauth authentication is currently not|` +
	`organization has disabled|` +
	`claude subscription access .*disabled|` +
	`use an anthropic api key instead|` +
	`ask your admin to enable access`)

// usageCapTextRE matches a self-recovering USAGE/OVERAGE cap in a denial's text. It is
// the TEXT half of the cap-vs-wall fence, for the probe paths that never see response
// headers (a `claude -p` banner). Mirrors internal/fleetaccounts.usageCapRE; the reset
// half is delegated to this package's own ResolveReset so there is one reset parser.
var usageCapTextRE = regexp.MustCompile(`(?i)session limit|weekly limit|usage limit|usage cap|overage|/usage-credits`)

// needsLoginTextRE matches a CREDENTIAL-scoped failure: the token is expired, revoked,
// or absent. This is the one class `/login` actually repairs, so it must stay separable
// from the org wall rather than collapsing into it.
var needsLoginTextRE = regexp.MustCompile(`(?i)oauth session expired|could not be refreshed|` +
	`invalid[_ ]?(bearer[_ ]?)?token|authentication_error|please run /login|expired`)

// ClassifySeatHealth folds ONE upstream probe result into the closed SeatHealth
// vocabulary. It is pure (status + body + headers + now in, one state out) so it is
// testable against the exact bodies observed on the wire.
//
// ORDER IS THE WHOLE POINT. A 2xx is the only positive witness. A 401 is the credential
// (the `/login` case) unless a relay body identifies an upstream-provider block. For the
// 403/402 family the two usage-cap fences run
// BEFORE the org-wall signature, because the SAME prose covers a standing org wall and
// a rolling-window cap the account clears on its own:
//
//	fence 1 (headers): anthropic-ratelimit-unified-* / -overage-status show a rejection,
//	                   read through the accountobs leaf that already owns that taxonomy
//	                   for the live retry and post-mortem classifiers;
//	fence 2 (text):    the banner names a usage/overage cap, or names a reset instant
//	                   (ResolveReset) — a reset means it comes back by itself.
//
// Only a denial that clears BOTH fences and carries the organization signature is
// filed as org_auth_wall. Everything unrecognized stays SeatHealthUnknown; an unknown
// never writes a durable wall.
func upstreamProviderBlock(text string) bool {
	text = strings.ToLower(text)
	return strings.Contains(text, "blocked by upstream provider") ||
		strings.Contains(text, "request blocked by upstream")
}

func ClassifySeatHealth(status int, body []byte, h http.Header, now time.Time) SeatHealth {
	text := string(body)
	switch {
	case status >= 200 && status < 300:
		return SeatHealthReady
	case status == http.StatusTooManyRequests:
		return SeatHealthUsageLimited
	case status == http.StatusUnauthorized:
		if upstreamProviderBlock(text) {
			return SeatHealthUnknown // a relay borrowed the upstream vendor's status
		}
		return SeatHealthNeedsLogin
	case status == http.StatusForbidden, status == http.StatusPaymentRequired:
		if accountobs.UsageOverageRejection(h).Rejected {
			return SeatHealthUsageLimited // fence 1: the provider's own rejection headers
		}
		if usageCapTextRE.MatchString(text) || !ResolveReset(text, now).IsZero() {
			return SeatHealthUsageLimited // fence 2: a named cap or a named reset
		}
		if orgAuthWallRE.MatchString(text) {
			return SeatHealthOrgAuthWall
		}
		if needsLoginTextRE.MatchString(text) {
			return SeatHealthNeedsLogin
		}
		return SeatHealthUnknown
	default:
		return SeatHealthUnknown
	}
}

// ClassifySeatHealthText classifies a probe result that carries no HTTP status — the
// `claude -p` banner path, where only stdout/stderr text is available. It runs the same
// fences in the same order as the 403 arm of ClassifySeatHealth, so a banner and a wire
// body cannot be triaged differently. Text with no recognized signature is Unknown.
func ClassifySeatHealthText(text string, now time.Time) SeatHealth {
	switch {
	case usageCapTextRE.MatchString(text) || !ResolveReset(text, now).IsZero():
		return SeatHealthUsageLimited
	case orgAuthWallRE.MatchString(text):
		return SeatHealthOrgAuthWall
	case needsLoginTextRE.MatchString(text):
		return SeatHealthNeedsLogin
	default:
		return SeatHealthUnknown
	}
}

// CooldownOrgAuthWall is the durable cooldown KIND (and signal name) for an upstream
// organization auth wall. It is deliberately a cooldown signal rather than a new store:
// every consumer that already skips a cooled account — the LoginReportAt overlay,
// rotation, the dispatch preflight — then excludes a walled account across processes
// with no extra wiring. Unlike the usage/rate kinds this signal NEVER lapses on its own
// timer (see activeCooldownSignals); its deadline means "reprobe is due", not
// "re-admit".
const CooldownOrgAuthWall CooldownKind = "org-auth-wall"

// OrgAuthWallReprobeWindow is how long a witnessed org wall stands before a REPROBE is
// due. It is not an expiry: the seat stays excluded past it. It only tells an operator
// (and a future scheduled canary) that enough time has passed for an upstream
// administrative repair to have plausibly happened, so spending one bounded round-trip
// to re-check is worth it. Sized so a wall witnessed at the end of one working day is
// reprobe-due by the start of the next.
const OrgAuthWallReprobeWindow = 12 * time.Hour

// OrgAuthWalled reports whether this entry carries an upstream organization auth wall.
// It reads the SIGNAL set first so a mixed entry (an account that is both walled and
// usage-capped) is still recognized as walled after the cap's signal expires; Kind is
// consulted only for a legacy v1 row that predates the signal map.
func (e CooldownEntry) OrgAuthWalled() bool {
	if _, ok := e.Signals[string(CooldownOrgAuthWall)]; ok {
		return true
	}
	return len(e.Signals) == 0 && e.Kind == CooldownOrgAuthWall
}

// OrgAuthWallReprobeAt returns the instant at which re-probing this entry's wall is
// due, or the zero time when the entry carries no wall.
func (e CooldownEntry) OrgAuthWallReprobeAt() time.Time {
	if !e.OrgAuthWalled() {
		return time.Time{}
	}
	if at, ok := e.Signals[string(CooldownOrgAuthWall)]; ok && !at.IsZero() {
		return at.UTC()
	}
	return e.ResetAt.UTC()
}

// ReprobeDue reports whether a walled entry is old enough that a fresh bounded canary
// is worth spending. It never re-admits the seat by itself — only a witnessed healthy
// observation does that.
func (e CooldownEntry) ReprobeDue(now time.Time) bool {
	at := e.OrgAuthWallReprobeAt()
	return !at.IsZero() && !now.Before(at)
}

// RecordOrgAuthWall persists a typed, timestamped org-auth-wall observation for a
// CANONICAL account key. reason is a CLASSIFIED label (the caller's remedy enum, the
// SeatHealth value, …) — never a raw upstream body and never token material; an empty
// reason falls back to the SeatHealthOrgAuthWall label. Recording again later only
// pushes the reprobe deadline out, so a re-witnessed wall never shortens itself.
// Returns the stored entry and whether anything was recorded (an empty account key is
// a no-op, so a seat with no derivable identity can never wall the empty bucket).
func (s *CooldownStore) RecordOrgAuthWall(account, reason string, at time.Time) (CooldownEntry, bool) {
	account = strings.TrimSpace(account)
	if account == "" {
		return CooldownEntry{}, false
	}
	if strings.TrimSpace(reason) == "" {
		reason = string(SeatHealthOrgAuthWall)
	}
	e, _ := s.UpdateOverload(account, string(CooldownOrgAuthWall), CooldownOrgAuthWall,
		true, reason, at, at.Add(OrgAuthWallReprobeWindow))
	return e, true
}

// ClearOrgAuthWall drops the org-auth-wall signal for account and reports whether one
// was present. This is the WITNESSED re-admission path: the caller must have proof the
// organization now serves (a successful canary), because the wall has no timer that
// clears it. Any sibling usage/rate signal on the same account is left alone, so
// clearing a wall never re-admits a still-capped account.
func (s *CooldownStore) ClearOrgAuthWall(account string, at time.Time) bool {
	account = strings.TrimSpace(account)
	if account == "" {
		return false
	}
	e, ok := s.entries[account]
	if !ok || !e.OrgAuthWalled() {
		return false
	}
	s.UpdateOverload(account, string(CooldownOrgAuthWall), CooldownOrgAuthWall, false, "", at, time.Time{})
	return true
}

// OrgAuthWall returns the active org-auth-wall entry for account at now, and whether
// one exists. A seat with only a usage cooldown reports false, so a caller can tell the
// futile-`/login` case apart from the wait-for-reset case.
func (s *CooldownStore) OrgAuthWall(account string, now time.Time) (CooldownEntry, bool) {
	e, ok := s.CooledDown(account, now)
	if !ok || !e.OrgAuthWalled() {
		return CooldownEntry{}, false
	}
	return e, true
}

// ObserveSeatHealth folds ONE classified canary result for a canonical account key into
// the durable store — the single write seam, so no caller has to remember which health
// states write and which clear:
//
//	ready         → CLEARS the wall (the witnessed upstream repair; the only clear)
//	org_auth_wall → RECORDS/extends the wall
//	anything else → leaves the wall untouched. A usage cap is not evidence the org was
//	                repaired, and neither is a fresh needs_login — that is exactly the
//	                degradation (#4998) where the weaker cause displaced the stronger.
//
// changed reports whether the stored state moved; the caller Saves on true.
func (s *CooldownStore) ObserveSeatHealth(account string, health SeatHealth, at time.Time) (changed bool) {
	account = strings.TrimSpace(account)
	if account == "" {
		return false
	}
	switch health {
	case SeatHealthReady:
		return s.ClearOrgAuthWall(account, at)
	case SeatHealthOrgAuthWall:
		_, walledBefore := s.OrgAuthWall(account, at)
		s.RecordOrgAuthWall(account, string(SeatHealthOrgAuthWall), at)
		return !walledBefore
	default:
		return false
	}
}

// LoginOrgAuthWall means the seat's upstream ORGANIZATION has OAuth/subscription access
// disabled: the config home may look perfect (or may have been blanked to needs_login
// afterwards), but no credential minted for this organization can serve. Like
// LoginCooledDown it is an OVERLAY, never returned by the pure LoginStatus() fold — but
// unlike a cooldown it OUTRANKS needs_login, because the whole defect in #4998 was a
// weaker, wrong repair (`/login`) displacing a stronger, witnessed one. CanServe false.
const LoginOrgAuthWall LoginStatus = "org_auth_wall"

// orgAuthWallReasonAction renders the seat-facing reason and the ACCOUNT-LEVEL repair
// for a durable org wall. It deliberately never names `/login`: the witnessed 403
// already established that re-login mints another token for the same walled org.
func orgAuthWallReasonAction(e CooldownEntry, now time.Time) (string, string) {
	reason := "upstream organization has OAuth/subscription access disabled"
	if e.Reason != "" && e.Reason != string(SeatHealthOrgAuthWall) {
		reason = fmt.Sprintf("%s (%s)", reason, e.Reason)
	}
	if !e.CooledAt.IsZero() {
		reason = fmt.Sprintf("%s — witnessed %s", reason, e.CooledAt.UTC().Format(time.RFC3339))
	}
	if e.ReprobeDue(now) {
		reason += "; reprobe due"
	} else if at := e.OrgAuthWallReprobeAt(); !at.IsZero() {
		reason = fmt.Sprintf("%s; reprobe due %s", reason, at.Format(time.RFC3339))
	}
	action := "re-login cannot clear an organization wall — switch to a seat on a permitted " +
		"organization, bill this seat to an Anthropic API key, or ask an org admin to re-enable " +
		"Claude subscription access; once the upstream is repaired, `fak accounts cooldown --clear " +
		"<account>` returns the seat to the pool"
	return reason, action
}

// DefaultCanaryURL is the messages endpoint the seat canary calls. Overridable in
// CanarySeat for tests.
const DefaultCanaryURL = "https://api.anthropic.com/v1/messages"

// DefaultCanaryModel is the cheapest model the canary asks for. The canary only needs
// the upstream's ADMISSION decision, so it requests a single token.
const DefaultCanaryModel = "claude-haiku-4-5-20251001"

// anthropicVersion is the API version header the messages endpoint requires.
const anthropicVersion = "2023-06-01"

// canaryTimeout bounds one canary round-trip. "Bounded" is the whole contract: a probe
// that can hang is a probe no enrollment or doctor path will ever be allowed to call.
const canaryTimeout = 20 * time.Second

// CanarySeat runs ONE bounded live inference round-trip on token and classifies the
// result — the real probe seam behind an enrollment/doctor usability check. It returns
// the typed SeatHealth plus a SHORT classified detail (the HTTP status only, never the
// upstream body), so a caller can log or persist the verdict without ever carrying
// provider prose or credential material across the boundary. A transport failure
// returns SeatHealthUnknown and the error: fak must never file a wall it cannot prove.
//
// url defaults to DefaultCanaryURL, model to DefaultCanaryModel, client to a
// canaryTimeout-bounded client.
func CanarySeat(client *http.Client, url, token, model string, now time.Time) (SeatHealth, string, error) {
	token = strings.TrimSpace(token)
	if token == "" {
		return SeatHealthNeedsLogin, "no credential", nil
	}
	if url == "" {
		url = DefaultCanaryURL
	}
	if model == "" {
		model = DefaultCanaryModel
	}
	if client == nil {
		client = &http.Client{Timeout: canaryTimeout}
	}
	payload, err := json.Marshal(map[string]any{
		"model":      model,
		"max_tokens": 1,
		"messages":   []map[string]string{{"role": "user", "content": "ping"}},
	})
	if err != nil {
		return SeatHealthUnknown, "", fmt.Errorf("accounts: canary: %w", err)
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return SeatHealthUnknown, "", fmt.Errorf("accounts: canary: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("anthropic-version", anthropicVersion)
	req.Header.Set("anthropic-beta", oauthBeta)
	resp, err := client.Do(req)
	if err != nil {
		return SeatHealthUnknown, "", fmt.Errorf("accounts: canary %s: %w", url, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	return ClassifySeatHealth(resp.StatusCode, body, resp.Header, now),
		fmt.Sprintf("http %d", resp.StatusCode), nil
}
